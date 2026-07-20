// Fused SiLU-and-mul + dynamic per-token FP8 (e4m3) quant -- the down_proj input path.
//
// Stock runs two ops: `silu_and_mul` (reads gate|up [T,2I] bf16, writes g [T,I] bf16) then
// `dynamic_per_token_scaled_fp8_quant` (reads g [T,I] bf16, writes fp8 [T,I] + scale). This
// kernel does both in ONE pass: each thread holds its slice of g in registers across the amax
// reduction, so the wide gate|up tensor is read exactly once and the [T,I] bf16 intermediate
// (a full write + read) never touches HBM.
//
// Numerics matched to the stock unfused path element-for-element:
//   silu = scalar_t( f32(gate) / (1 + expf(-f32(gate))) )   -- sigmoid in fp32, NARROW to bf16
//   g    = silu * up                                        -- scalar_t * scalar_t -> scalar_t
//   gf   = f32(g);  amax = max_i |gf|                        -- per-token, fp32
//   s    = max(min(amax, scale_ub) / 448, 1/(448*512))
//   out  = fp8(clamp(gf * (1/s), -448, 448))                -- reciprocal-multiply, clamp then
//                                                              hw convert (byte-parity retired).
// The two narrowings (silu, and g = silu*up) are load-bearing: stock forms both in bf16.
//
// New op (not an override): vllm_cuda::silu_and_mul_dynamic_per_token_quant. Wired into the
// compiled graph by a fusion pattern (vtl/patches/silu_mul_quant.py) that rewrites
// `silu_and_mul -> dynamic_per_token_scaled_fp8_quant` to this single node. If the pattern does
// not fire, nothing calls this and down_proj keeps stock silu + our fast quant -- no crash.
//   silu_and_mul_dynamic_per_token_quant(Tensor! result, Tensor! scale, Tensor input,
//                                        Tensor? scale_ub) -> ()
//   input = [num_tokens, 2*I]  (gate = [:, :I], up = [:, I:]);  result = [num_tokens, I] fp8.

#include <torch/all.h>

#include "fp8_common.cuh"

namespace vtl {

namespace {

// silu(gate) narrowed to scalar_t, times up (scalar_t), returned as fp32. Exactly stock's
// silu_and_mul: sigmoid in fp32, both the silu and the product formed in scalar_t.
template <typename scalar_t>
__device__ __forceinline__ float silu_mul_one(scalar_t gate, scalar_t up) {
  float const g = static_cast<float>(gate);
  scalar_t const silu = static_cast<scalar_t>(g / (1.0f + expf(-g)));
  return static_cast<float>(silu * up);
}

// One block per token; each thread owns ITEMS kVec-wide slices of the I-wide output, strided by
// blockDim so consecutive threads hit consecutive addresses (coalesced) per item. It reads a
// 16-byte gate chunk + a 16-byte up chunk for each item, forms g, and holds them in registers
// across the single amax reduction -- gate|up read once. ITEMS=1 is the old single-slice kernel;
// ITEMS>=2 keeps a wide I (LFM2 down_proj I=12288 -> ITEMS=3, 512 threads) on this fast path
// under the 512-thread launch bound, instead of the 1024-thread block it used before.
template <typename scalar_t, int ITEMS>
__global__ void __launch_bounds__(kFastMaxThreads) silu_mul_quant_kernel(
    c10::Float8_e4m3fn* __restrict__ out,  // [num_tokens, I]
    float* __restrict__ scales,            // [num_tokens]
    scalar_t const* __restrict__ input,    // [num_tokens, 2*I]  (gate | up)
    float const* __restrict__ scale_ub,    // scalar or nullptr
    int const I, int64_t const in_row_stride) {
  constexpr int kVec = VecTraits<scalar_t>::kVec;

  int64_t const token = blockIdx.x;
  int const B = blockDim.x;
  int64_t const gate_row = token * in_row_stride;
  int64_t const out_row = token * static_cast<int64_t>(I);

  float g[ITEMS][kVec];  // narrowed silu*up, kept in fp32 for the reduction and multiply
  float amax = 0.0f;
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    int const idx = (threadIdx.x + k * B) * kVec;
    if (idx < I) {
      VecIn<scalar_t, kVec> vg =
          *reinterpret_cast<VecIn<scalar_t, kVec> const*>(input + gate_row + idx);
      VecIn<scalar_t, kVec> vu =
          *reinterpret_cast<VecIn<scalar_t, kVec> const*>(input + gate_row + I + idx);
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        g[k][j] = silu_mul_one(vg.v[j], vu.v[j]);
        amax = fmaxf(amax, fabsf(g[k][j]));
      }
    } else {
#pragma unroll
      for (int j = 0; j < kVec; ++j) g[k][j] = 0.0f;
    }
  }

  __shared__ float smem_max[32];
  amax = block_reduce(amax, MaxOp{}, 0.0f, smem_max);
  if (scale_ub != nullptr) amax = fminf(amax, *scale_ub);
  float const scale = fmaxf(amax / kFp8Max, kMinScale);
  if (threadIdx.x == 0) scales[token] = scale;
  // One divide per token, then a multiply per element (byte-parity retired; see file header).
  float const inv_scale = 1.0f / scale;

#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    int const idx = (threadIdx.x + k * B) * kVec;
    if (idx < I) {
      VecOut<kVec> vout;
#pragma unroll
      for (int j = 0; j < kVec; j += 2) {
        vout.pair[j >> 1] = floats_to_fp8x2(g[k][j] * inv_scale, g[k][j + 1] * inv_scale);
      }
      *reinterpret_cast<VecOut<kVec>*>(out + out_row + idx) = vout;
    }
  }
}

// Correctness fallback: I not a multiple of kVec, too wide for the coarsened block, or misaligned.
// Two passes over gate|up (re-reads it), so slower than stock -- but our target (I=12288) always
// takes the fast path (ITEMS=3); this only guards odd shapes.
template <typename scalar_t>
__global__ void silu_mul_quant_generic_kernel(
    c10::Float8_e4m3fn* __restrict__ out, float* __restrict__ scales,
    scalar_t const* __restrict__ input, float const* __restrict__ scale_ub,
    int const I, int64_t const in_row_stride) {
  int64_t const token = blockIdx.x;
  int64_t const gate_base = token * in_row_stride;
  int64_t const out_base = token * static_cast<int64_t>(I);

  __shared__ float smem_max[32];
  float amax = 0.0f;
  for (int i = threadIdx.x; i < I; i += blockDim.x) {
    amax = fmaxf(amax, fabsf(silu_mul_one(input[gate_base + i], input[gate_base + I + i])));
  }
  amax = block_reduce(amax, MaxOp{}, 0.0f, smem_max);
  if (scale_ub != nullptr) amax = fminf(amax, *scale_ub);
  float const scale = fmaxf(amax / kFp8Max, kMinScale);
  if (threadIdx.x == 0) scales[token] = scale;
  float const inv_scale = 1.0f / scale;

  for (int i = threadIdx.x; i < I; i += blockDim.x) {
    out[out_base + i] =
        float_to_fp8(silu_mul_one(input[gate_base + i], input[gate_base + I + i]) * inv_scale);
  }
}

template <typename scalar_t>
void launch(torch::Tensor const& out, torch::Tensor const& input,
            torch::Tensor const& scales, std::optional<torch::Tensor> const& scale_ub,
            cudaStream_t stream) {
  constexpr int kVec = VecTraits<scalar_t>::kVec;

  int64_t const two_i = input.size(-1);
  TORCH_CHECK(two_i % 2 == 0, "vtl: silu_and_mul input last dim must be even");
  int const I = static_cast<int>(two_i / 2);
  int64_t const in_row_stride = input.dim() == 2 ? input.stride(0) : two_i;
  int64_t const num_tokens = input.numel() / two_i;
  if (num_tokens == 0) return;

  auto* out_p = reinterpret_cast<c10::Float8_e4m3fn*>(out.data_ptr());
  auto* scales_p = scales.data_ptr<float>();
  auto const* in_p = input.const_data_ptr<scalar_t>();
  auto const* ub_p = scale_ub.has_value() ? scale_ub->const_data_ptr<float>() : nullptr;

  // I % kVec == 0 makes both the gate slice and the up half (offset I) 16-byte aligned, and
  // guarantees a thread's kVec block is wholly in-range (no partial tail).
  int const nvec = I % kVec == 0 ? I / kVec : 0;
  int const items = nvec > 0 ? coarsen_items(nvec) : 0;
  bool const fast =
      items > 0 && in_row_stride % kVec == 0 && aligned16(in_p) && aligned16(out_p);

  dim3 const grid(num_tokens);
  if (fast) {
    int const nthreads = (nvec + items - 1) / items;
    dim3 const block((nthreads + 31) / 32 * 32);
#define VTL_LAUNCH_SMQ(IT)                                                  \
  silu_mul_quant_kernel<scalar_t, IT><<<grid, block, 0, stream>>>(          \
      out_p, scales_p, in_p, ub_p, I, in_row_stride)
    switch (items) {
      case 1: VTL_LAUNCH_SMQ(1); break;
      case 2: VTL_LAUNCH_SMQ(2); break;
      case 3: VTL_LAUNCH_SMQ(3); break;
      default: VTL_LAUNCH_SMQ(4); break;  // coarsen_items caps at kMaxItems=4
    }
#undef VTL_LAUNCH_SMQ
  } else {
    dim3 const block(std::min((I + 31) / 32 * 32, 1024));
    silu_mul_quant_generic_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out_p, scales_p, in_p, ub_p, I, in_row_stride);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (kernel_sync_enabled()) {
    cudaError_t const err = cudaStreamSynchronize(stream);
    TORCH_CHECK(err == cudaSuccess, "vtl silu_and_mul_quant faulted: ",
                cudaGetErrorString(err), " | path=", (fast ? "fast" : "generic"),
                " items=", items, " dtype=", input.scalar_type(), " num_tokens=", num_tokens,
                " I=", I, " stride=", in_row_stride, " kVec=", kVec,
                " aligned16(in,out)=", aligned16(in_p), aligned16(out_p));
  }
}

}  // namespace

// result [T, I] fp8, scale [T, 1] fp32, input [T, 2I] (gate | up), scale_ub scalar or None.
void silu_and_mul_dynamic_per_token_quant(torch::Tensor& result, torch::Tensor& scale,
                                          torch::Tensor const& input,
                                          std::optional<torch::Tensor> const& scale_ub) {
  TORCH_CHECK(result.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "vtl: silu_and_mul_dynamic_per_token_quant supports fp8_e4m3 output only, got ",
              result.scalar_type());
  TORCH_CHECK(result.is_contiguous(), "vtl: result must be contiguous");
  TORCH_CHECK(input.dim() >= 2, "vtl: input must be at least 2-D");
  TORCH_CHECK(input.stride(-1) == 1, "vtl: input must be contiguous in the last dim");
  TORCH_CHECK(input.dim() == 2 || input.is_contiguous(),
              "vtl: input with rank > 2 must be contiguous");
  TORCH_CHECK(result.size(-1) * 2 == input.size(-1),
              "vtl: result last dim must be half of input last dim");
  TORCH_CHECK(scale.scalar_type() == at::ScalarType::Float, "vtl: scale must be fp32");

  c10::cuda::OptionalCUDAGuard const device_guard(input.device());
  cudaStream_t const stream = c10::cuda::getCurrentCUDAStream();

  switch (input.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch<c10::BFloat16>(result, input, scale, scale_ub, stream);
      break;
    case at::ScalarType::Half:
      launch<c10::Half>(result, input, scale, scale_ub, stream);
      break;
    case at::ScalarType::Float:
      launch<float>(result, input, scale, scale_ub, stream);
      break;
    default:
      TORCH_CHECK(false, "vtl: unsupported input dtype ", input.scalar_type());
  }
}

}  // namespace vtl
