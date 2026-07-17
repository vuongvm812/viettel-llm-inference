// Per-head gated RMSNorm for the GDN (linear-attention) output path -- the `linear_attn.norm`
// (RMSNormGated, weight [head_v_dim]=[128], fp32) applied per head to the gated-delta core
// output before out_proj, on all 18 GDN layers of the served qwen3_5 model.
//
// SEMANTICS -- matched to vLLM's RMSNormGated.forward_static element-for-element, for the
// config qwen3_5 actually uses: norm_before_gate=True, group_size=None (one group = head_v_dim),
// activation in {silu, sigmoid}. So the gate is applied AFTER the norm (NOT before):
//   xf   = f32(x)                                                 (per (token,head) row, len D)
//   rms  = rsqrt(mean_D(xf^2) + eps)
//   out  = scalar_t( xf * rms * weight * act(f32(z)) )            weight fp32; narrow once at end
//   act(z) = silu(z) = z*sigmoid(z)   if gate_is_silu, else sigmoid(z)
// !!! The PRIOR version of this kernel implemented norm_before_gate=False (out = norm(x*silu(z)))
// and its Python shim only fired when norm_before_gate was False -- so on qwen3_5 (which is True)
// it never ran. This file fixes that: correct gate order + a `gate_is_silu` flag.
//
// Two ops (own ops; wired in Python behind VTL_ENABLE_GDN_KERNELS):
//   gated_rmsnorm(Tensor! result, Tensor input, Tensor gate, Tensor weight, float epsilon,
//                 bool gate_is_silu) -> ()
//     input/gate/result = [num_rows, D] (num_rows = num_tokens*num_v_heads); weight = [D] fp32.
//
//   gated_rmsnorm_dynamic_per_token_quant(Tensor! result, Tensor! scale, Tensor input,
//       Tensor gate, Tensor weight, float epsilon, int num_heads, bool gate_is_silu,
//       Tensor? scale_ub) -> ()
//     The FUSED norm+quant: computes the per-head norm above, then a PER-TOKEN dynamic fp8
//     quant over the flattened heads (the out_proj input), in one pass -- so the bf16 norm
//     output never round-trips HBM. input/gate = [num_tokens*num_heads, D]; weight = [D] fp32;
//     result = [num_tokens, num_heads*D] fp8_e4m3; scale = [num_tokens, 1] fp32. This mirrors
//     the stock `RMSNormGated -> reshape(-1, H*D) -> dynamic_per_token_scaled_fp8_quant` chain
//     (there is no stock CUDA fusion of it -- only a ROCm/AITER pass exists).
//
// H200 tuning (vs the old block-per-row + block_reduce + __syncthreads design):
//   - warp-per-row: one row (or one head) per warp, so the reduction is a pure __shfl and there
//     is NO __syncthreads and NO shared memory on the fast path;
//   - thread coarsening + 16B vectorized loads/stores (kVec elems/lane);
//   - many rows/heads packed per block for occupancy.

#include <torch/all.h>

#include "fp8_common.cuh"

namespace vtl {

namespace {

// act(z): silu (z*sigmoid(z)) or sigmoid(z), in fp32 -- matches forward_static (z.float()).
__device__ __forceinline__ float gate_act(float z, bool is_silu) {
  float const s = 1.0f / (1.0f + expf(-z));
  return is_silu ? z * s : s;
}

// Intra-warp reductions (full 32-lane mask; inactive lanes carry the identity).
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// -------- standalone gated RMSNorm (bf16/fp16/fp32 out) ---------------------------------------

// Fast path: one warp per row, D <= 32*kVec and D % kVec == 0, each lane owns one 16B chunk.
template <typename scalar_t, int WARPS>
__global__ void gated_rmsnorm_warp_kernel(scalar_t* __restrict__ out,
                                          scalar_t const* __restrict__ input,
                                          scalar_t const* __restrict__ gate,
                                          float const* __restrict__ weight, float const eps,
                                          bool const is_silu, int const D,
                                          int64_t const num_rows) {
  constexpr int kVec = VecTraits<scalar_t>::kVec;
  int const warp = threadIdx.x >> 5;
  int const lane = threadIdx.x & 31;
  int64_t const row = static_cast<int64_t>(blockIdx.x) * WARPS + warp;
  if (row >= num_rows) return;  // uniform across the warp: row depends only on warp id

  int const idx = lane * kVec;
  bool const active = idx < D;
  int64_t const off = row * static_cast<int64_t>(D) + idx;

  float xf[kVec], zf[kVec];
  if (active) {
    VecIn<scalar_t, kVec> vx = *reinterpret_cast<VecIn<scalar_t, kVec> const*>(input + off);
    VecIn<scalar_t, kVec> vz = *reinterpret_cast<VecIn<scalar_t, kVec> const*>(gate + off);
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      xf[j] = static_cast<float>(vx.v[j]);
      zf[j] = static_cast<float>(vz.v[j]);
    }
  } else {
#pragma unroll
    for (int j = 0; j < kVec; ++j) xf[j] = zf[j] = 0.0f;
  }

  float ss = 0.0f;
#pragma unroll
  for (int j = 0; j < kVec; ++j) ss += xf[j] * xf[j];
  ss = warp_reduce_sum(ss);
  float const rms = rsqrtf(ss / D + eps);

  if (active) {
    VecIn<scalar_t, kVec> vo;
#pragma unroll
    for (int j = 0; j < kVec; ++j)
      vo.v[j] = static_cast<scalar_t>(xf[j] * rms * weight[idx + j] * gate_act(zf[j], is_silu));
    *reinterpret_cast<VecIn<scalar_t, kVec>*>(out + off) = vo;
  }
}

// Generic fallback: one block per row, strided, for D that the fast path cannot vectorize.
template <typename scalar_t>
__global__ void gated_rmsnorm_generic_kernel(scalar_t* __restrict__ out,
                                             scalar_t const* __restrict__ input,
                                             scalar_t const* __restrict__ gate,
                                             float const* __restrict__ weight, float const eps,
                                             bool const is_silu, int const D) {
  int64_t const base = static_cast<int64_t>(blockIdx.x) * D;
  __shared__ float smem[32];

  float ss = 0.0f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float const x = static_cast<float>(input[base + i]);
    ss += x * x;
  }
  ss = block_reduce(ss, AddOp{}, 0.0f, smem);
  float const rms = rsqrtf(ss / D + eps);

  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float const x = static_cast<float>(input[base + i]);
    float const z = static_cast<float>(gate[base + i]);
    out[base + i] = static_cast<scalar_t>(x * rms * weight[i] * gate_act(z, is_silu));
  }
}

// -------- fused gated RMSNorm + per-token fp8 quant -------------------------------------------

// Fast path: one BLOCK per token, one WARP per head. Each warp norms its head (pure __shfl,
// no sync), keeps the normed row in registers, then a single block-wide max over the H warps
// gives the per-token amax -> scale -> fp8 write. blockDim = H*32 (<= 1024).
template <typename scalar_t>
__global__ void gated_rmsnorm_quant_warp_kernel(c10::Float8_e4m3fn* __restrict__ out,  // [N, H*D]
                                                float* __restrict__ scales,            // [N]
                                                scalar_t const* __restrict__ input,  // [N*H, D]
                                                scalar_t const* __restrict__ gate,   // [N*H, D]
                                                float const* __restrict__ weight,    // [D]
                                                float const* __restrict__ scale_ub,  // scalar/null
                                                float const eps, bool const is_silu, int const H,
                                                int const D) {
  constexpr int kVec = VecTraits<scalar_t>::kVec;
  int const warp = threadIdx.x >> 5;  // head within the token
  int const lane = threadIdx.x & 31;
  int64_t const token = blockIdx.x;
  int const idx = lane * kVec;
  bool const active = idx < D;
  int64_t const in_off = (token * H + warp) * static_cast<int64_t>(D) + idx;

  float nrm[kVec];  // normed value (fp32), kept in registers across the block reduction
  {
    float xf[kVec], zf[kVec];
    if (active) {
      VecIn<scalar_t, kVec> vx = *reinterpret_cast<VecIn<scalar_t, kVec> const*>(input + in_off);
      VecIn<scalar_t, kVec> vz = *reinterpret_cast<VecIn<scalar_t, kVec> const*>(gate + in_off);
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        xf[j] = static_cast<float>(vx.v[j]);
        zf[j] = static_cast<float>(vz.v[j]);
      }
    } else {
#pragma unroll
      for (int j = 0; j < kVec; ++j) xf[j] = zf[j] = 0.0f;
    }
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < kVec; ++j) ss += xf[j] * xf[j];
    ss = warp_reduce_sum(ss);
    float const rms = rsqrtf(ss / D + eps);
#pragma unroll
    for (int j = 0; j < kVec; ++j)
      nrm[j] = active ? xf[j] * rms * weight[idx + j] * gate_act(zf[j], is_silu) : 0.0f;
  }

  float amax = 0.0f;
#pragma unroll
  for (int j = 0; j < kVec; ++j) amax = fmaxf(amax, fabsf(nrm[j]));
  __shared__ float smem[32];
  amax = block_reduce(amax, MaxOp{}, 0.0f, smem);  // over all H warps of this token
  if (scale_ub != nullptr) amax = fminf(amax, *scale_ub);
  float const scale = fmaxf(amax / kFp8Max, kMinScale);
  if (threadIdx.x == 0) scales[token] = scale;

  if (active) {
    int64_t const out_off =
        token * static_cast<int64_t>(H) * D + warp * static_cast<int64_t>(D) + idx;
    VecOut<kVec> vout;
#pragma unroll
    for (int j = 0; j < kVec; j += 2)
      vout.pair[j >> 1] = floats_to_fp8x2(nrm[j] / scale, nrm[j + 1] / scale);
    *reinterpret_cast<VecOut<kVec>*>(out + out_off) = vout;
  }
}

// Generic fused fallback: one block per token, loops heads. Two passes over x|z (re-reads), so
// slower than the fast path -- only for shapes the fast path rejects (D not vectorizable, or
// H*32 > 1024). dynamic shared: [H] rms + [32] reduce scratch.
template <typename scalar_t>
__global__ void gated_rmsnorm_quant_generic_kernel(c10::Float8_e4m3fn* __restrict__ out,
                                                   float* __restrict__ scales,
                                                   scalar_t const* __restrict__ input,
                                                   scalar_t const* __restrict__ gate,
                                                   float const* __restrict__ weight,
                                                   float const* __restrict__ scale_ub,
                                                   float const eps, bool const is_silu,
                                                   int const H, int const D) {
  int64_t const token = blockIdx.x;
  extern __shared__ float sh[];
  float* rms_sh = sh;      // [H]
  float* red = sh + H;     // [32]

  float my_amax = 0.0f;
  for (int h = 0; h < H; ++h) {
    int64_t const base = (token * H + h) * static_cast<int64_t>(D);
    float ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
      float const x = static_cast<float>(input[base + i]);
      ss += x * x;
    }
    ss = block_reduce(ss, AddOp{}, 0.0f, red);
    float const rms = rsqrtf(ss / D + eps);
    if (threadIdx.x == 0) rms_sh[h] = rms;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
      float const x = static_cast<float>(input[base + i]);
      float const z = static_cast<float>(gate[base + i]);
      my_amax = fmaxf(my_amax, fabsf(x * rms * weight[i] * gate_act(z, is_silu)));
    }
    __syncthreads();  // rms_sh[h] visible + red free before the next head
  }

  float amax = block_reduce(my_amax, MaxOp{}, 0.0f, red);
  if (scale_ub != nullptr) amax = fminf(amax, *scale_ub);
  float const scale = fmaxf(amax / kFp8Max, kMinScale);
  if (threadIdx.x == 0) scales[token] = scale;

  for (int h = 0; h < H; ++h) {
    int64_t const base = (token * H + h) * static_cast<int64_t>(D);
    int64_t const obase = token * static_cast<int64_t>(H) * D + h * static_cast<int64_t>(D);
    float const rms = rms_sh[h];
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
      float const x = static_cast<float>(input[base + i]);
      float const z = static_cast<float>(gate[base + i]);
      out[obase + i] = float_to_fp8(x * rms * weight[i] * gate_act(z, is_silu) / scale);
    }
  }
}

// -------- launchers ---------------------------------------------------------------------------

constexpr int kQuantWarpsPerBlock = 8;  // standalone: rows packed per block for occupancy

template <typename scalar_t>
void launch_norm(torch::Tensor const& out, torch::Tensor const& input, torch::Tensor const& gate,
                 torch::Tensor const& weight, double epsilon, bool is_silu, cudaStream_t stream) {
  constexpr int kVec = VecTraits<scalar_t>::kVec;
  int const D = input.size(-1);
  int64_t const num_rows = input.numel() / D;
  if (num_rows == 0) return;

  auto* out_p = out.data_ptr<scalar_t>();
  auto const* in_p = input.const_data_ptr<scalar_t>();
  auto const* gate_p = gate.const_data_ptr<scalar_t>();
  auto const* w_p = weight.const_data_ptr<float>();
  float const eps = static_cast<float>(epsilon);

  bool const fast = D % kVec == 0 && (D / kVec) <= 32 && aligned16(in_p) && aligned16(gate_p) &&
                    aligned16(out_p);
  if (fast) {
    constexpr int W = kQuantWarpsPerBlock;
    dim3 const grid((num_rows + W - 1) / W);
    dim3 const block(W * 32);
    gated_rmsnorm_warp_kernel<scalar_t, W>
        <<<grid, block, 0, stream>>>(out_p, in_p, gate_p, w_p, eps, is_silu, D, num_rows);
  } else {
    dim3 const grid(num_rows);
    dim3 const block(std::min((D + 31) / 32 * 32, 1024));
    gated_rmsnorm_generic_kernel<scalar_t>
        <<<grid, block, 0, stream>>>(out_p, in_p, gate_p, w_p, eps, is_silu, D);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (kernel_sync_enabled()) {
    cudaError_t const e = cudaStreamSynchronize(stream);
    TORCH_CHECK(e == cudaSuccess, "vtl gated_rmsnorm faulted: ", cudaGetErrorString(e),
                " dtype=", input.scalar_type(), " num_rows=", num_rows, " D=", D);
  }
}

template <typename scalar_t>
void launch_norm_quant(torch::Tensor const& out, torch::Tensor const& scale,
                       torch::Tensor const& input, torch::Tensor const& gate,
                       torch::Tensor const& weight, double epsilon, int H, bool is_silu,
                       std::optional<torch::Tensor> const& scale_ub, cudaStream_t stream) {
  constexpr int kVec = VecTraits<scalar_t>::kVec;
  int const D = input.size(-1);
  int64_t const num_rows = input.numel() / D;
  int64_t const num_tokens = num_rows / H;
  if (num_tokens == 0) return;

  auto* out_p = reinterpret_cast<c10::Float8_e4m3fn*>(out.data_ptr());
  auto* scale_p = scale.data_ptr<float>();
  auto const* in_p = input.const_data_ptr<scalar_t>();
  auto const* gate_p = gate.const_data_ptr<scalar_t>();
  auto const* w_p = weight.const_data_ptr<float>();
  auto const* ub_p = scale_ub.has_value() ? scale_ub->const_data_ptr<float>() : nullptr;
  float const eps = static_cast<float>(epsilon);

  bool const fast = D % kVec == 0 && (D / kVec) <= 32 && H * 32 <= 1024 && aligned16(in_p) &&
                    aligned16(gate_p) && aligned16(out_p);
  dim3 const grid(num_tokens);
  if (fast) {
    dim3 const block(H * 32);
    gated_rmsnorm_quant_warp_kernel<scalar_t>
        <<<grid, block, 0, stream>>>(out_p, scale_p, in_p, gate_p, w_p, ub_p, eps, is_silu, H, D);
  } else {
    dim3 const block(std::min((D + 31) / 32 * 32, 1024));
    size_t const shmem = (static_cast<size_t>(H) + 32) * sizeof(float);
    gated_rmsnorm_quant_generic_kernel<scalar_t>
        <<<grid, block, shmem, stream>>>(out_p, scale_p, in_p, gate_p, w_p, ub_p, eps, is_silu, H,
                                         D);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (kernel_sync_enabled()) {
    cudaError_t const e = cudaStreamSynchronize(stream);
    TORCH_CHECK(e == cudaSuccess, "vtl gated_rmsnorm_quant faulted: ", cudaGetErrorString(e),
                " path=", (fast ? "fast" : "generic"), " dtype=", input.scalar_type(),
                " num_tokens=", num_tokens, " H=", H, " D=", D);
  }
}

}  // namespace

void gated_rmsnorm(torch::Tensor& result, torch::Tensor const& input, torch::Tensor const& gate,
                   torch::Tensor const& weight, double epsilon, bool gate_is_silu) {
  TORCH_CHECK(input.is_contiguous() && gate.is_contiguous() && result.is_contiguous(),
              "vtl: gated_rmsnorm requires contiguous input/gate/result");
  TORCH_CHECK(input.sizes() == gate.sizes() && input.sizes() == result.sizes(),
              "vtl: gated_rmsnorm input/gate/result must have equal shape");
  TORCH_CHECK(weight.scalar_type() == at::ScalarType::Float, "vtl: gated_rmsnorm weight fp32");
  TORCH_CHECK(weight.numel() == input.size(-1), "vtl: gated_rmsnorm weight must be [D]");
  TORCH_CHECK(input.scalar_type() == result.scalar_type() &&
                  input.scalar_type() == gate.scalar_type(),
              "vtl: gated_rmsnorm input/gate/result dtype mismatch");

  c10::cuda::OptionalCUDAGuard const device_guard(input.device());
  cudaStream_t const stream = c10::cuda::getCurrentCUDAStream();

  switch (input.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_norm<c10::BFloat16>(result, input, gate, weight, epsilon, gate_is_silu, stream);
      break;
    case at::ScalarType::Half:
      launch_norm<c10::Half>(result, input, gate, weight, epsilon, gate_is_silu, stream);
      break;
    case at::ScalarType::Float:
      launch_norm<float>(result, input, gate, weight, epsilon, gate_is_silu, stream);
      break;
    default:
      TORCH_CHECK(false, "vtl: unsupported input dtype ", input.scalar_type());
  }
}

void gated_rmsnorm_dynamic_per_token_quant(torch::Tensor& result, torch::Tensor& scale,
                                           torch::Tensor const& input, torch::Tensor const& gate,
                                           torch::Tensor const& weight, double epsilon,
                                           int64_t num_heads, bool gate_is_silu,
                                           std::optional<torch::Tensor> const& scale_ub) {
  TORCH_CHECK(result.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "vtl: gated_rmsnorm_quant supports fp8_e4m3 output only, got ", result.scalar_type());
  TORCH_CHECK(input.is_contiguous() && gate.is_contiguous() && result.is_contiguous(),
              "vtl: gated_rmsnorm_quant requires contiguous input/gate/result");
  TORCH_CHECK(input.sizes() == gate.sizes(), "vtl: gated_rmsnorm_quant input/gate shape mismatch");
  TORCH_CHECK(input.dim() == 2, "vtl: gated_rmsnorm_quant expects input [num_tokens*H, D]");
  TORCH_CHECK(weight.scalar_type() == at::ScalarType::Float, "vtl: gated_rmsnorm_quant weight fp32");
  int const D = static_cast<int>(input.size(-1));
  int const H = static_cast<int>(num_heads);
  TORCH_CHECK(weight.numel() == D, "vtl: gated_rmsnorm_quant weight must be [D]");
  TORCH_CHECK(H >= 1 && input.size(0) % H == 0,
              "vtl: gated_rmsnorm_quant rows must be a multiple of num_heads");
  int64_t const num_tokens = input.size(0) / H;
  TORCH_CHECK(result.numel() == num_tokens * H * D,
              "vtl: gated_rmsnorm_quant result must be [num_tokens, H*D]");
  TORCH_CHECK(scale.scalar_type() == at::ScalarType::Float, "vtl: gated_rmsnorm_quant scale fp32");
  TORCH_CHECK(scale.numel() == num_tokens, "vtl: gated_rmsnorm_quant scale must be [num_tokens]");
  TORCH_CHECK(input.scalar_type() == gate.scalar_type(), "vtl: gated_rmsnorm_quant dtype mismatch");

  c10::cuda::OptionalCUDAGuard const device_guard(input.device());
  cudaStream_t const stream = c10::cuda::getCurrentCUDAStream();

  switch (input.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_norm_quant<c10::BFloat16>(result, scale, input, gate, weight, epsilon, H,
                                       gate_is_silu, scale_ub, stream);
      break;
    case at::ScalarType::Half:
      launch_norm_quant<c10::Half>(result, scale, input, gate, weight, epsilon, H, gate_is_silu,
                                   scale_ub, stream);
      break;
    case at::ScalarType::Float:
      launch_norm_quant<float>(result, scale, input, gate, weight, epsilon, H, gate_is_silu,
                               scale_ub, stream);
      break;
    default:
      TORCH_CHECK(false, "vtl: unsupported input dtype ", input.scalar_type());
  }
}

}  // namespace vtl
