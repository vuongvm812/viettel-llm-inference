// Fused elementwise-multiply + dynamic per-token FP8 (e4m3) quant -- the LFM2 short-conv gate.
//
// The short-conv decode epilogue is `y = C * Bx` (bf16 elementwise), whose result feeds the fp8
// `out_proj`. Stock runs the multiply (materialising a bf16 [T,D] intermediate in HBM) and then
// out_proj's own `dynamic_per_token_scaled_fp8_quant`. This kernel fuses both: it reads C and Bx
// once, forms the product in registers, and writes the fp8 output + per-token scale directly --
// the bf16 `y` never touches HBM and one kernel launch per conv layer is saved.
//
// Numerics match the stock `y = C * Bx` then quant, element-for-element:
//   p  = f32( scalar_t(C) * scalar_t(Bx) )        -- bf16 multiply then narrow, exactly as stock
//   amax = max_i |p|                              -- per token, fp32
//   s  = max(min(amax, scale_ub) / 448, 1/(448*512))
//   out = fp8(clamp(p * (1/s), -448, 448))        -- reciprocal-multiply, clamp then hw convert.
//
// New op (not an override); called DIRECTLY from the patched ShortConv.forward_cuda decode path
// (there is no FX pattern -- the whole short_conv runs inside an opaque custom op). Env-gated OFF
// by default (VTL_ENABLE_MUL_QUANT); if unregistered or disabled, short_conv keeps the stock
// `y = C*Bx` + normal out_proj path.
//   mul_dynamic_per_token_quant(Tensor! result, Tensor! scale, Tensor a, Tensor b,
//                               Tensor? scale_ub) -> ()
//   a, b = [num_tokens, D] (last-dim contiguous, row strides may differ); result = [T,D] fp8.

#include <torch/all.h>

#include "fp8_common.cuh"

namespace vtl {

namespace {

// bf16 multiply then narrow, returned as fp32 -- matches stock's bf16 `y = C * Bx`.
template <typename scalar_t>
__device__ __forceinline__ float mul_one(scalar_t a, scalar_t b) {
  return static_cast<float>(a * b);
}

// Fast path. D % kVec == 0 and D/kVec <= kFastMaxThreads (D is conv_dim=2048 -> 256 threads at
// bf16), so a thread holds one 16-byte slice of each input in registers across the single amax
// reduction -- a and b read exactly once. Row strides are independent: C is a chunk view (row
// stride 3*dim), Bx is contiguous (row stride dim); both are last-dim contiguous so the 16-byte
// vector load is valid when each row start is 16-byte aligned (checked at launch).
template <typename scalar_t>
__global__ void __launch_bounds__(kFastMaxThreads) mul_quant_kernel(
    c10::Float8_e4m3fn* __restrict__ out,  // [num_tokens, D]
    float* __restrict__ scales,            // [num_tokens]
    scalar_t const* __restrict__ a,        // [num_tokens, D]
    scalar_t const* __restrict__ b,        // [num_tokens, D]
    float const* __restrict__ scale_ub,    // scalar or nullptr
    int const D, int64_t const a_stride, int64_t const b_stride) {
  constexpr int kVec = VecTraits<scalar_t>::kVec;

  int64_t const token = blockIdx.x;
  int const idx = threadIdx.x * kVec;
  bool const active = idx < D;

  int64_t const a_off = token * a_stride + idx;
  int64_t const b_off = token * b_stride + idx;
  int64_t const out_off = token * static_cast<int64_t>(D) + idx;

  float p[kVec];
  if (active) {
    VecIn<scalar_t, kVec> va = *reinterpret_cast<VecIn<scalar_t, kVec> const*>(a + a_off);
    VecIn<scalar_t, kVec> vb = *reinterpret_cast<VecIn<scalar_t, kVec> const*>(b + b_off);
#pragma unroll
    for (int j = 0; j < kVec; ++j) p[j] = mul_one(va.v[j], vb.v[j]);
  } else {
#pragma unroll
    for (int j = 0; j < kVec; ++j) p[j] = 0.0f;
  }

  __shared__ float smem_max[32];
  float amax = 0.0f;
#pragma unroll
  for (int j = 0; j < kVec; ++j) amax = fmaxf(amax, fabsf(p[j]));
  amax = block_reduce(amax, MaxOp{}, 0.0f, smem_max);
  if (scale_ub != nullptr) amax = fminf(amax, *scale_ub);
  float const scale = fmaxf(amax / kFp8Max, kMinScale);
  if (threadIdx.x == 0) scales[token] = scale;
  float const inv_scale = 1.0f / scale;

  if (active) {
    VecOut<kVec> vout;
#pragma unroll
    for (int j = 0; j < kVec; j += 2) {
      vout.pair[j >> 1] = floats_to_fp8x2(p[j] * inv_scale, p[j + 1] * inv_scale);
    }
    *reinterpret_cast<VecOut<kVec>*>(out + out_off) = vout;
  }
}

// Correctness fallback for shapes the fast path rejects (D not a multiple of the vector width, or
// misaligned rows). Scalar, two passes. Not on the LFM2 hot path (D=2048 always takes fast).
template <typename scalar_t>
__global__ void mul_quant_generic_kernel(
    c10::Float8_e4m3fn* __restrict__ out, float* __restrict__ scales,
    scalar_t const* __restrict__ a, scalar_t const* __restrict__ b,
    float const* __restrict__ scale_ub, int const D, int64_t const a_stride,
    int64_t const b_stride) {
  int64_t const token = blockIdx.x;
  int64_t const a_base = token * a_stride;
  int64_t const b_base = token * b_stride;
  int64_t const out_base = token * static_cast<int64_t>(D);

  __shared__ float smem_max[32];
  float amax = 0.0f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    amax = fmaxf(amax, fabsf(mul_one(a[a_base + i], b[b_base + i])));
  }
  amax = block_reduce(amax, MaxOp{}, 0.0f, smem_max);
  if (scale_ub != nullptr) amax = fminf(amax, *scale_ub);
  float const scale = fmaxf(amax / kFp8Max, kMinScale);
  if (threadIdx.x == 0) scales[token] = scale;
  float const inv_scale = 1.0f / scale;

  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    out[out_base + i] = float_to_fp8(mul_one(a[a_base + i], b[b_base + i]) * inv_scale);
  }
}

template <typename scalar_t>
void launch(torch::Tensor const& out, torch::Tensor const& a, torch::Tensor const& b,
            torch::Tensor const& scales, std::optional<torch::Tensor> const& scale_ub,
            cudaStream_t stream) {
  constexpr int kVec = VecTraits<scalar_t>::kVec;

  int const D = a.size(-1);
  int64_t const a_stride = a.dim() == 2 ? a.stride(0) : D;
  int64_t const b_stride = b.dim() == 2 ? b.stride(0) : D;
  int64_t const num_tokens = a.numel() / D;
  if (num_tokens == 0) return;

  auto* out_p = reinterpret_cast<c10::Float8_e4m3fn*>(out.data_ptr());
  auto* scales_p = scales.data_ptr<float>();
  auto const* a_p = a.const_data_ptr<scalar_t>();
  auto const* b_p = b.const_data_ptr<scalar_t>();
  auto const* ub_p = scale_ub.has_value() ? scale_ub->const_data_ptr<float>() : nullptr;

  int const nthreads = (D + kVec - 1) / kVec;
  bool const fast = D % kVec == 0 && nthreads <= kFastMaxThreads && a_stride % kVec == 0 &&
                    b_stride % kVec == 0 && aligned16(a_p) && aligned16(b_p) && aligned16(out_p);

  dim3 const grid(num_tokens);
  if (fast) {
    dim3 const block((nthreads + 31) / 32 * 32);
    mul_quant_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out_p, scales_p, a_p, b_p, ub_p, D, a_stride, b_stride);
  } else {
    dim3 const block(std::min((D + 31) / 32 * 32, 1024));
    mul_quant_generic_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out_p, scales_p, a_p, b_p, ub_p, D, a_stride, b_stride);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (kernel_sync_enabled()) {
    cudaError_t const err = cudaStreamSynchronize(stream);
    TORCH_CHECK(err == cudaSuccess, "vtl mul_quant faulted: ", cudaGetErrorString(err),
                " | path=", (fast ? "fast" : "generic"), " dtype=", a.scalar_type(),
                " num_tokens=", num_tokens, " D=", D, " a_stride=", a_stride,
                " b_stride=", b_stride, " kVec=", kVec);
  }
}

}  // namespace

// (Tensor! result, Tensor! scale, Tensor a, Tensor b, Tensor? scale_ub) -> (). Optionals by const
// reference to avoid an intrusive-refcount atomic RMW per call. a and b must be the same dtype and
// shape [T, D] with a contiguous last dim; row strides may differ.
void mul_dynamic_per_token_quant(torch::Tensor& result, torch::Tensor& scale,
                                 torch::Tensor const& a, torch::Tensor const& b,
                                 std::optional<torch::Tensor> const& scale_ub) {
  TORCH_CHECK(result.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "vtl: mul_dynamic_per_token_quant supports fp8_e4m3 output only, got ",
              result.scalar_type());
  TORCH_CHECK(result.is_contiguous(), "vtl: result must be contiguous");
  TORCH_CHECK(a.dim() >= 2 && b.dim() >= 2, "vtl: a and b must be at least 2-D");
  TORCH_CHECK(a.sizes() == b.sizes(), "vtl: a and b must have the same shape");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "vtl: a and b must have the same dtype");
  TORCH_CHECK(a.stride(-1) == 1 && b.stride(-1) == 1, "vtl: a, b must be contiguous in last dim");
  TORCH_CHECK(scale.scalar_type() == at::ScalarType::Float, "vtl: scale must be fp32");

  c10::cuda::OptionalCUDAGuard const device_guard(a.device());
  cudaStream_t const stream = c10::cuda::getCurrentCUDAStream();

  switch (a.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch<c10::BFloat16>(result, a, b, scale, scale_ub, stream);
      break;
    case at::ScalarType::Half:
      launch<c10::Half>(result, a, b, scale, scale_ub, stream);
      break;
    case at::ScalarType::Float:
      launch<float>(result, a, b, scale, scale_ub, stream);
      break;
    default:
      TORCH_CHECK(false, "vtl: unsupported input dtype ", a.scalar_type());
  }
}

}  // namespace vtl
