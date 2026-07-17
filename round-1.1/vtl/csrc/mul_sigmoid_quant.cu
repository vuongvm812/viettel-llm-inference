// Fused gate multiply (attn_output * sigmoid(gate)) + dynamic per-token FP8 (e4m3) quant --
// the o_proj input path on the full-attention + MTP layers of the served qwen3_5 model.
//
// Stock runs three ops on that path: `sigmoid` (gate [T,H] bf16 -> [T,H] bf16), `mul`
// (attn_output * gate -> [T,H] bf16), then `dynamic_per_token_scaled_fp8_quant` (reads the
// [T,H] bf16 product, writes fp8 [T,H] + scale). The gate multiply is emitted DECOMPOSED
// (aten.sigmoid + aten.mul.Tensor -- there is no fused activation op here, unlike silu_and_mul),
// so vLLM leaves it unfused. This kernel does all three in ONE pass: each thread holds its slice
// of the product in registers across the amax reduction, so attn_output and gate are each read
// exactly once and the [T,H] bf16 intermediate (a full write + read) never touches HBM.
//
// Numerics matched to the stock decomposed path element-for-element:
//   sig = scalar_t( 1 / (1 + expf(-f32(gate))) )   -- sigmoid in fp32, NARROW to bf16
//         (matches torch.sigmoid on a bf16 tensor: compute in fp32, round to bf16)
//   g   = f32( attn * sig )                         -- scalar_t * scalar_t -> scalar_t, then f32
//   gf  = g;  amax = max_i |gf|                     -- per-token, fp32
//   s   = max(min(amax, scale_ub) / 448, 1/(448*512))
//   out = fp8(clamp(gf / s, -448, 448))             -- divide, clamp then hw convert.
// The two narrowings (sig, and g = attn*sig) are load-bearing: stock forms both in bf16.
//
// New op (not an override): vllm_cuda::mul_sigmoid_dynamic_per_token_quant. Wired into the
// compiled graph by a fusion pattern (vtl/patches/mul_sigmoid_quant.py) that rewrites
// `mul(input, sigmoid(gate)) -> dynamic_per_token_scaled_fp8_quant` to this single node. If the
// pattern does not fire, nothing calls this and o_proj keeps stock sigmoid+mul + our fast quant
// -- no crash.
//   mul_sigmoid_dynamic_per_token_quant(Tensor! result, Tensor! scale, Tensor input,
//                                       Tensor gate, Tensor? scale_ub) -> ()
//   input = attn_output [T, H];  gate = [T, H];  result = [T, H] fp8;  scale = [T, 1] fp32.
// Unlike silu_and_mul (one [T,2I] tensor split in half), the two operands are SEPARATE tensors
// and the output is the full width H (not halved).

#include <torch/all.h>

#include "fp8_common.cuh"

namespace vtl {

namespace {

// attn * sigmoid(gate), both narrowed to scalar_t, returned as fp32. Exactly stock's decomposed
// `attn_output * torch.sigmoid(gate)`: sigmoid in fp32 narrowed to scalar_t, product in scalar_t.
template <typename scalar_t>
__device__ __forceinline__ float mul_sigmoid_one(scalar_t x, scalar_t gate) {
  float const g = static_cast<float>(gate);
  scalar_t const sig = static_cast<scalar_t>(1.0f / (1.0f + expf(-g)));
  return static_cast<float>(x * sig);
}

// One block per token; one kVec-wide slice of the H-wide output per thread. blockDim =
// round_up_warp(H / kVec) <= 1024, so a thread reads one 16-byte attn chunk + one 16-byte gate
// chunk, forms the product, and holds it in registers across the single amax reduction.
template <typename scalar_t>
__global__ void __launch_bounds__(kFastMaxThreads) mul_sigmoid_quant_kernel(
    c10::Float8_e4m3fn* __restrict__ out,  // [num_tokens, H]
    float* __restrict__ scales,            // [num_tokens]
    scalar_t const* __restrict__ input,    // [num_tokens, H]  (attn_output)
    scalar_t const* __restrict__ gate,     // [num_tokens, H]
    float const* __restrict__ scale_ub,    // scalar or nullptr
    int const H, int64_t const in_row_stride, int64_t const gate_row_stride) {
  constexpr int kVec = VecTraits<scalar_t>::kVec;

  int64_t const token = blockIdx.x;
  int const idx = threadIdx.x * kVec;
  bool const active = idx < H;

  int64_t const in_off = token * in_row_stride + idx;
  int64_t const gate_off = token * gate_row_stride + idx;
  int64_t const out_off = token * static_cast<int64_t>(H) + idx;

  float g[kVec];  // attn*sigmoid(gate), kept in fp32 for the reduction and divide
  if (active) {
    VecIn<scalar_t, kVec> vx =
        *reinterpret_cast<VecIn<scalar_t, kVec> const*>(input + in_off);
    VecIn<scalar_t, kVec> vg =
        *reinterpret_cast<VecIn<scalar_t, kVec> const*>(gate + gate_off);
#pragma unroll
    for (int j = 0; j < kVec; ++j) g[j] = mul_sigmoid_one(vx.v[j], vg.v[j]);
  } else {
#pragma unroll
    for (int j = 0; j < kVec; ++j) g[j] = 0.0f;
  }

  __shared__ float smem_max[32];
  float amax = 0.0f;
#pragma unroll
  for (int j = 0; j < kVec; ++j) amax = fmaxf(amax, fabsf(g[j]));
  amax = block_reduce(amax, MaxOp{}, 0.0f, smem_max);
  if (scale_ub != nullptr) amax = fminf(amax, *scale_ub);
  float const scale = fmaxf(amax / kFp8Max, kMinScale);
  if (threadIdx.x == 0) scales[token] = scale;

  if (active) {
    VecOut<kVec> vout;
#pragma unroll
    for (int j = 0; j < kVec; j += 2) {
      vout.pair[j >> 1] = floats_to_fp8x2(g[j] / scale, g[j + 1] / scale);
    }
    *reinterpret_cast<VecOut<kVec>*>(out + out_off) = vout;
  }
}

// Correctness fallback: H not a multiple of kVec, too wide for a single block, or misaligned.
// Two passes over attn|gate (re-reads them), so slower than stock -- but our target (H=2048)
// always takes the fast path; this only guards odd shapes.
template <typename scalar_t>
__global__ void mul_sigmoid_quant_generic_kernel(
    c10::Float8_e4m3fn* __restrict__ out, float* __restrict__ scales,
    scalar_t const* __restrict__ input, scalar_t const* __restrict__ gate,
    float const* __restrict__ scale_ub, int const H, int64_t const in_row_stride,
    int64_t const gate_row_stride) {
  int64_t const token = blockIdx.x;
  int64_t const in_base = token * in_row_stride;
  int64_t const gate_base = token * gate_row_stride;
  int64_t const out_base = token * static_cast<int64_t>(H);

  __shared__ float smem_max[32];
  float amax = 0.0f;
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    amax = fmaxf(amax, fabsf(mul_sigmoid_one(input[in_base + i], gate[gate_base + i])));
  }
  amax = block_reduce(amax, MaxOp{}, 0.0f, smem_max);
  if (scale_ub != nullptr) amax = fminf(amax, *scale_ub);
  float const scale = fmaxf(amax / kFp8Max, kMinScale);
  if (threadIdx.x == 0) scales[token] = scale;

  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    out[out_base + i] =
        float_to_fp8(mul_sigmoid_one(input[in_base + i], gate[gate_base + i]) / scale);
  }
}

template <typename scalar_t>
void launch(torch::Tensor const& out, torch::Tensor const& input, torch::Tensor const& gate,
            torch::Tensor const& scales, std::optional<torch::Tensor> const& scale_ub,
            cudaStream_t stream) {
  constexpr int kVec = VecTraits<scalar_t>::kVec;

  int const H = static_cast<int>(input.size(-1));
  int64_t const in_row_stride = input.dim() == 2 ? input.stride(0) : H;
  int64_t const gate_row_stride = gate.dim() == 2 ? gate.stride(0) : H;
  int64_t const num_tokens = input.numel() / H;
  if (num_tokens == 0) return;

  auto* out_p = reinterpret_cast<c10::Float8_e4m3fn*>(out.data_ptr());
  auto* scales_p = scales.data_ptr<float>();
  auto const* in_p = input.const_data_ptr<scalar_t>();
  auto const* gate_p = gate.const_data_ptr<scalar_t>();
  auto const* ub_p = scale_ub.has_value() ? scale_ub->const_data_ptr<float>() : nullptr;

  int const nthreads = (H + kVec - 1) / kVec;
  // <= kFastMaxThreads (512): the o_proj input is H=2048 (256 threads at bf16), so 512 covers it
  // and lets the sm_90 allocator size registers for a <=512-thread block, keeping more blocks
  // resident. Wider H falls to the generic path (matches rms_norm_quant / dynamic_per_token_quant).
  bool const fast = H % kVec == 0 && nthreads <= kFastMaxThreads && in_row_stride % kVec == 0 &&
                    gate_row_stride % kVec == 0 && aligned16(in_p) && aligned16(gate_p) &&
                    aligned16(out_p);

  dim3 const grid(num_tokens);
  if (fast) {
    dim3 const block((nthreads + 31) / 32 * 32);
    mul_sigmoid_quant_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out_p, scales_p, in_p, gate_p, ub_p, H, in_row_stride, gate_row_stride);
  } else {
    dim3 const block(std::min((H + 31) / 32 * 32, 1024));
    mul_sigmoid_quant_generic_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out_p, scales_p, in_p, gate_p, ub_p, H, in_row_stride, gate_row_stride);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (kernel_sync_enabled()) {
    cudaError_t const err = cudaStreamSynchronize(stream);
    TORCH_CHECK(err == cudaSuccess, "vtl mul_sigmoid_quant faulted: ",
                cudaGetErrorString(err), " | path=", (fast ? "fast" : "generic"),
                " dtype=", input.scalar_type(), " num_tokens=", num_tokens, " H=", H,
                " in_stride=", in_row_stride, " gate_stride=", gate_row_stride,
                " kVec=", kVec, " aligned16(in,gate,out)=", aligned16(in_p),
                aligned16(gate_p), aligned16(out_p));
  }
}

}  // namespace

// result [T, H] fp8, scale [T, 1] fp32, input=attn_output [T, H], gate [T, H], scale_ub scalar/None.
void mul_sigmoid_dynamic_per_token_quant(torch::Tensor& result, torch::Tensor& scale,
                                         torch::Tensor const& input,
                                         torch::Tensor const& gate,
                                         std::optional<torch::Tensor> const& scale_ub) {
  TORCH_CHECK(result.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "vtl: mul_sigmoid_dynamic_per_token_quant supports fp8_e4m3 output only, got ",
              result.scalar_type());
  TORCH_CHECK(result.is_contiguous(), "vtl: result must be contiguous");
  TORCH_CHECK(input.dim() >= 2 && gate.dim() >= 2, "vtl: input/gate must be at least 2-D");
  TORCH_CHECK(input.stride(-1) == 1 && gate.stride(-1) == 1,
              "vtl: input/gate must be contiguous in the last dim");
  TORCH_CHECK(input.dim() == 2 || input.is_contiguous(),
              "vtl: input with rank > 2 must be contiguous");
  TORCH_CHECK(gate.dim() == 2 || gate.is_contiguous(),
              "vtl: gate with rank > 2 must be contiguous");
  TORCH_CHECK(input.sizes() == gate.sizes(), "vtl: input and gate must have equal shape");
  TORCH_CHECK(result.sizes() == input.sizes(),
              "vtl: result must match input shape [T, H] (fused sizes it so; guards direct calls)");
  TORCH_CHECK(input.scalar_type() == gate.scalar_type(), "vtl: input/gate dtype mismatch");
  TORCH_CHECK(scale.scalar_type() == at::ScalarType::Float, "vtl: scale must be fp32");
  TORCH_CHECK(scale.numel() == input.numel() / input.size(-1),
              "vtl: scale must have one entry per token");

  c10::cuda::OptionalCUDAGuard const device_guard(input.device());
  cudaStream_t const stream = c10::cuda::getCurrentCUDAStream();

  switch (input.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch<c10::BFloat16>(result, input, gate, scale, scale_ub, stream);
      break;
    case at::ScalarType::Half:
      launch<c10::Half>(result, input, gate, scale, scale_ub, stream);
      break;
    case at::ScalarType::Float:
      launch<float>(result, input, gate, scale, scale_ub, stream);
      break;
    default:
      TORCH_CHECK(false, "vtl: unsupported input dtype ", input.scalar_type());
  }
}

}  // namespace vtl
