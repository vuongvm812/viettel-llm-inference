// Standalone dynamic per-token FP8 (e4m3) quant -- overrides vLLM's
// `_C::dynamic_per_token_scaled_fp8_quant`.
//
// On the served qwen3_5 model this op quantises the activation feeding `o_proj` (after the
// full-attention `attn_output * sigmoid(gate)` multiply, on the 6 full-attn + 1 MTP layers)
// and -- when the Phase-2 SiLU-mul fusion is off -- the activation feeding `down_proj`. It
// also runs at load time on the per-output-channel weight quant (vtl_fp8). It is NOT preceded
// by a norm, so unlike rms_norm_quant.cu there is no residual/weight/rms here: just the fp8
// tail (per-token amax -> scale -> divide-and-convert).
//
// Numerics matched to stock element-for-element (common.cu
// dynamic_per_token_scaled_fp8_quant_kernel + scaled_fp8_conversion<false>):
//   amax = max_d |f32(input)|                 (per token, accumulated in fp32)
//   t    = scale_ub ? min(amax, *scale_ub) : amax
//   s    = max(t / 448, 1/(448*512))          (per token, written to scales[token])
//   out  = fp8(clamp(f32(input) / s, -448, 448))   -- DIVIDE (is_scale_inverted=false),
//                                                     clamp THEN hw convert (SATFINITE).
// amax and the divide are fp32; the only narrowing is the fp8 output.
//
// Schema (must match _C exactly, incl. the mutable-alias / optional markers):
//   dynamic_per_token_scaled_fp8_quant(Tensor! result, Tensor input, Tensor! scale,
//                                      Tensor? scale_ub) -> ()

#include <torch/all.h>

#include "fp8_common.cuh"

namespace vtl {

namespace {

// Fast path. hidden % kVec == 0 and hidden/kVec <= kFastMaxThreads, so a thread's slice is one
// 16-byte load held in registers across the single amax reduction -- input read exactly once.
template <typename scalar_t>
__global__ void __launch_bounds__(kFastMaxThreads) dynamic_per_token_quant_kernel(
    c10::Float8_e4m3fn* __restrict__ out,  // [num_tokens, hidden]
    float* __restrict__ scales,            // [num_tokens]
    scalar_t const* __restrict__ input,    // [num_tokens, hidden]
    float const* __restrict__ scale_ub,    // scalar or nullptr
    int const hidden_size, int64_t const input_stride) {
  constexpr int kVec = VecTraits<scalar_t>::kVec;

  int64_t const token = blockIdx.x;
  int const idx = threadIdx.x * kVec;
  bool const active = idx < hidden_size;

  int64_t const in_off = token * input_stride + idx;
  int64_t const row_off = token * static_cast<int64_t>(hidden_size) + idx;

  float x[kVec];
  if (active) {
    VecIn<scalar_t, kVec> vin =
        *reinterpret_cast<VecIn<scalar_t, kVec> const*>(input + in_off);
#pragma unroll
    for (int j = 0; j < kVec; ++j) x[j] = static_cast<float>(vin.v[j]);
  } else {
#pragma unroll
    for (int j = 0; j < kVec; ++j) x[j] = 0.0f;
  }

  __shared__ float smem_max[32];
  float amax = 0.0f;
#pragma unroll
  for (int j = 0; j < kVec; ++j) amax = fmaxf(amax, fabsf(x[j]));
  amax = block_reduce(amax, MaxOp{}, 0.0f, smem_max);
  if (scale_ub != nullptr) amax = fminf(amax, *scale_ub);
  float const scale = fmaxf(amax / kFp8Max, kMinScale);
  if (threadIdx.x == 0) scales[token] = scale;

  if (active) {
    VecOut<kVec> vout;
#pragma unroll
    for (int j = 0; j < kVec; j += 2) {
      // Divide, not reciprocal-multiply: stock's is_scale_inverted=false path, for exact
      // match with FBGemm. --use_fast_math is off so this stays a real divide.
      vout.pair[j >> 1] = floats_to_fp8x2(x[j] / scale, x[j + 1] / scale);
    }
    *reinterpret_cast<VecOut<kVec>*>(out + row_off) = vout;
  }
}

// Correctness fallback for shapes the fast path rejects (hidden not a multiple of the vector
// width, too wide to hold in registers, or misaligned). Scalar, two passes -- a direct port of
// stock. Handles the down_proj-quant-half (hidden=6144) when Phase-2 fusion is off, and the
// load-time weight quant.
template <typename scalar_t>
__global__ void dynamic_per_token_quant_generic_kernel(
    c10::Float8_e4m3fn* __restrict__ out, float* __restrict__ scales,
    scalar_t const* __restrict__ input, float const* __restrict__ scale_ub,
    int const hidden_size, int64_t const input_stride) {
  int64_t const token = blockIdx.x;
  int64_t const in_base = token * input_stride;
  int64_t const row_base = token * static_cast<int64_t>(hidden_size);

  __shared__ float smem_max[32];

  float amax = 0.0f;
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    amax = fmaxf(amax, fabsf(static_cast<float>(input[in_base + i])));
  }
  amax = block_reduce(amax, MaxOp{}, 0.0f, smem_max);
  if (scale_ub != nullptr) amax = fminf(amax, *scale_ub);
  float const scale = fmaxf(amax / kFp8Max, kMinScale);
  if (threadIdx.x == 0) scales[token] = scale;

  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    out[row_base + i] = float_to_fp8(static_cast<float>(input[in_base + i]) / scale);
  }
}

template <typename scalar_t>
void launch(torch::Tensor const& out, torch::Tensor const& input,
            torch::Tensor const& scales, std::optional<torch::Tensor> const& scale_ub,
            cudaStream_t stream) {
  constexpr int kVec = VecTraits<scalar_t>::kVec;

  int const hidden_size = input.size(-1);
  int64_t const input_stride = input.dim() == 2 ? input.stride(0) : hidden_size;
  int64_t const num_tokens = input.numel() / hidden_size;
  if (num_tokens == 0) return;

  auto* out_p = reinterpret_cast<c10::Float8_e4m3fn*>(out.data_ptr());
  auto* scales_p = scales.data_ptr<float>();
  auto const* in_p = input.const_data_ptr<scalar_t>();
  auto const* ub_p = scale_ub.has_value() ? scale_ub->const_data_ptr<float>() : nullptr;

  int const nthreads = (hidden_size + kVec - 1) / kVec;
  bool const fast = hidden_size % kVec == 0 && nthreads <= kFastMaxThreads &&
                    input_stride % kVec == 0 && aligned16(in_p) && aligned16(out_p);

  dim3 const grid(num_tokens);
  if (fast) {
    dim3 const block((nthreads + 31) / 32 * 32);
    dynamic_per_token_quant_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out_p, scales_p, in_p, ub_p, hidden_size, input_stride);
  } else {
    dim3 const block(std::min((hidden_size + 31) / 32 * 32, 1024));
    dynamic_per_token_quant_generic_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out_p, scales_p, in_p, ub_p, hidden_size, input_stride);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (kernel_sync_enabled()) {
    cudaError_t const err = cudaStreamSynchronize(stream);
    TORCH_CHECK(err == cudaSuccess, "vtl dynamic_per_token_quant faulted: ",
                cudaGetErrorString(err), " | path=", (fast ? "fast" : "generic"),
                " dtype=", input.scalar_type(), " num_tokens=", num_tokens,
                " hidden=", hidden_size, " stride=", input_stride, " kVec=", kVec,
                " aligned16(in,out)=", aligned16(in_p), aligned16(out_p));
  }
}

}  // namespace

// Same mutation set as `_C::dynamic_per_token_scaled_fp8_quant`: (Tensor! result, Tensor input,
// Tensor! scale, Tensor? scale_ub) -> (). Optionals taken by const reference to avoid an
// intrusive-refcount atomic RMW per call.
void dynamic_per_token_scaled_fp8_quant(torch::Tensor& result, torch::Tensor const& input,
                                        torch::Tensor& scale,
                                        std::optional<torch::Tensor> const& scale_ub) {
  TORCH_CHECK(result.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "vtl: dynamic_per_token_scaled_fp8_quant supports fp8_e4m3 output only, got ",
              result.scalar_type());
  TORCH_CHECK(result.is_contiguous(), "vtl: result must be contiguous");
  TORCH_CHECK(input.dim() >= 2, "vtl: input must be at least 2-D");
  TORCH_CHECK(input.stride(-1) == 1, "vtl: input must be contiguous in the last dim");
  TORCH_CHECK(input.dim() == 2 || input.is_contiguous(),
              "vtl: input with rank > 2 must be contiguous");
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
