// Single-pass fused residual-add + RMSNorm + dynamic per-token FP8 quant.
//
// Replaces vLLM's `_C::rms_norm_dynamic_per_token_quant` (see
// csrc/libtorch_stable/quantization/fused_kernels/), which the RMSNormQuantFusionPass
// emits for every `input_layernorm -> qkv_proj` and `post_attention_layernorm ->
// gate_up_proj` in Qwen2. Stock does three passes over input+residual
// (compute_rms -> compute_dynamic_per_token_scales -> norm_and_quant), loads 8 bytes
// at a time (`vec4_t<bf16>`), and launches min(hidden_size, 1024) threads -- at
// hidden_size=2048 that is 1024 threads of which only 512 ever enter the vectorized
// loop.
//
// This kernel makes ONE pass: each thread holds its 8 elements in registers across
// both block reductions, so input and residual are read exactly once. 16-byte loads,
// 256 threads all active at hidden_size=2048.
//
// Numerics are matched to stock element-for-element:
//   x    = f32(input) + f32(residual)            (residual := scalar_t(x))
//   rms  = rsqrtf(sum(x^2) / hidden_size + eps)
//   y    = f32(scalar_t(x * rms) * weight)
//   s    = max(min(max|y|, scale_ub) / 448, 1/(448*512))
//   out  = fp8(clamp(y / s, -448, 448))
//
// `y` rounds TWICE, and both roundings are load-bearing. c10::BFloat16's homogeneous
// `operator*(BFloat16, BFloat16)` computes in float but returns BFloat16, so stock's
// `static_cast<scalar_t>(x * rms) * weight[i]` (layernorm_utils.cuh:104, 161, 208, 363,
// 449, 555) narrows the weight product too. Keeping that multiply in fp32 would be more
// accurate, but it shifts amax -- and hence the per-token scale -- by up to ~6e-5
// relative, so it is not the same kernel. Hold `weight` as scalar_t, not float.

// Deliberately NOT <ATen/cuda/CUDAContext.h>: it drags in cusparse.h, which in the
// vllm-openai image lives only under the pip nvidia/cu13 tree, off the include path.
// The stream and the device guard both live in c10/cuda.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Half.h>
#include <cuda_fp8.h>
#include <cstdlib>
#include <torch/all.h>

namespace vtl {

namespace {

constexpr float kFp8Max = 448.0f;                      // quant_type_max_v<Float8_e4m3fn>
constexpr float kMinScale = 1.0f / (kFp8Max * 512.0f); // min_scaling_factor<fp8>::val()

// Fast-path block cap. Qwen2 (hidden=2048) launches 256 threads at bf16/fp16 (kVec=8) and
// 512 at fp32 (kVec=4), so 512 covers every shape the fast path currently takes; wider hidden
// falls to the generic kernel exactly as before. Given as an explicit __launch_bounds__ so the
// SM90 register allocator sizes for a <=512-thread block instead of the default 1024 worst
// case, keeping more blocks resident to hide HBM3e latency in the low-occupancy decode regime.
constexpr int kFastMaxThreads = 512;

// 16-byte loads: the widest a single thread can issue.
template <typename scalar_t>
struct VecTraits;
template <>
struct VecTraits<c10::BFloat16> {
  static constexpr int kVec = 8;
};
template <>
struct VecTraits<c10::Half> {
  static constexpr int kVec = 8;
};
template <>
struct VecTraits<float> {
  static constexpr int kVec = 4;
};

template <typename scalar_t, int N>
struct alignas(16) VecIn {
  scalar_t v[N];
};

// N fp8 codes, written only as N/2 pairs by the hardware two-at-a-time converter.
// One aligned N-byte store per thread.
template <int N>
struct alignas(N) VecOut {
  static_assert(N % 2 == 0, "fp8 pair conversion needs an even vector width");
  __nv_fp8x2_storage_t pair[N / 2];
};

// Clamp first, exactly as stock does, THEN convert. Not interchangeable with letting
// __NV_SATFINITE do the clamping: satfinite maps NaN to 0x7f, whereas fminf(NaN, 448)
// returns 448, so stock turns NaN into 0x7e. Measured, not assumed.
__device__ __forceinline__ float clamp_fp8(float x) {
  return fmaxf(-kFp8Max, fminf(x, kFp8Max));
}

// Hopper converts two floats per instruction. c10's static_cast is ~10 ALU ops of
// software bit-twiddling per element; at 8 elements/thread that is the difference
// between a free epilogue and a measurable one. Low byte is the first float (probed).
__device__ __forceinline__ __nv_fp8x2_storage_t floats_to_fp8x2(float a, float b) {
  float2 v;
  v.x = clamp_fp8(a);
  v.y = clamp_fp8(b);
  return __nv_cvt_float2_to_fp8x2(v, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ c10::Float8_e4m3fn float_to_fp8(float x) {
  __nv_fp8_storage_t const bits =
      __nv_cvt_float_to_fp8(clamp_fp8(x), __NV_SATFINITE, __NV_E4M3);
  return c10::Float8_e4m3fn(bits, c10::Float8_e4m3fn::from_bits());
}

struct AddOp {
  __device__ __forceinline__ float operator()(float a, float b) const { return a + b; }
};
struct MaxOp {
  __device__ __forceinline__ float operator()(float a, float b) const {
    return fmaxf(a, b);
  }
};

// Reduce across the block and broadcast to every thread. `op` is a functor resolved at
// compile time and inlined -- no indirect call. Barriers, not atomics: a block-wide
// atomicMax would serialise 256 lanes through one address and make the result depend on
// arrival order. `smem` is caller-owned so two reductions can run back to back without a
// barrier between them just to protect a shared buffer.
template <typename Op>
__device__ __forceinline__ float block_reduce(float v, Op op, float ident, float* smem) {
  int const lane = threadIdx.x & 31;
  int const wid = threadIdx.x >> 5;
  int const nwarps = (blockDim.x + 31) >> 5;

#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v = op(v, __shfl_xor_sync(0xffffffffu, v, off));
  }
  if (lane == 0) smem[wid] = v;
  __syncthreads();

  if (wid == 0) {
    v = (lane < nwarps) ? smem[lane] : ident;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      v = op(v, __shfl_xor_sync(0xffffffffu, v, off));
    }
    if (lane == 0) smem[0] = v;
  }
  __syncthreads();
  return smem[0];
}

// Fast path. Requires hidden_size % kVec == 0 and hidden_size / kVec <= 1024, so a
// thread's slice fits in registers and no element is visited twice.
template <typename scalar_t, bool kHasResidual>
__global__ void __launch_bounds__(kFastMaxThreads) fused_rms_norm_quant_kernel(
    c10::Float8_e4m3fn* __restrict__ out,   // [num_tokens, hidden_size]
    float* __restrict__ scales,             // [num_tokens]
    scalar_t const* __restrict__ input,     // [num_tokens, hidden_size]
    scalar_t const* __restrict__ weight,    // [hidden_size]
    scalar_t* __restrict__ residual,        // [num_tokens, hidden_size] or nullptr
    float const* __restrict__ scale_ub,     // scalar or nullptr
    float const epsilon, int const hidden_size, int64_t const input_stride) {
  constexpr int kVec = VecTraits<scalar_t>::kVec;

  int64_t const token = blockIdx.x;
  int const idx = threadIdx.x * kVec;
  bool const active = idx < hidden_size;

  int64_t const in_off = token * input_stride + idx;
  int64_t const row_off = token * static_cast<int64_t>(hidden_size) + idx;

  float x[kVec];
  // scalar_t, not float: stock's weight multiply happens in scalar_t and narrows.
  scalar_t w[kVec];

  if (active) {
    VecIn<scalar_t, kVec> vin = *reinterpret_cast<VecIn<scalar_t, kVec> const*>(input + in_off);
    VecIn<scalar_t, kVec> vw = *reinterpret_cast<VecIn<scalar_t, kVec> const*>(weight + idx);
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      x[j] = static_cast<float>(vin.v[j]);
      w[j] = vw.v[j];
    }
    if constexpr (kHasResidual) {
      VecIn<scalar_t, kVec>* rp = reinterpret_cast<VecIn<scalar_t, kVec>*>(residual + row_off);
      VecIn<scalar_t, kVec> vr = *rp;
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        x[j] += static_cast<float>(vr.v[j]);
      }
      // Nothing re-reads residual, so it can be finalised here in pass one.
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        vr.v[j] = static_cast<scalar_t>(x[j]);
      }
      *rp = vr;
    }
  } else {
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      x[j] = 0.0f;
      w[j] = static_cast<scalar_t>(0.0f);
    }
  }

  // Two buffers, so the second reduction does not need a barrier to wait for every thread
  // to finish reading the first one's result.
  __shared__ float smem_sum[32];
  __shared__ float smem_max[32];

  float ss = 0.0f;
#pragma unroll
  for (int j = 0; j < kVec; ++j) {
    ss += x[j] * x[j];
  }
  ss = block_reduce(ss, AddOp{}, 0.0f, smem_sum);
  // Not ss * (1/hidden_size): that is only exact for power-of-two widths, and stock
  // divides. Once per thread, against ~24 bytes of HBM -- it is not the bottleneck.
  float const rms = rsqrtf(ss / hidden_size + epsilon);

  // x[] is dead after this; reuse it for the normalised value rather than adding y[].
  // scalar_t * scalar_t -- narrows, exactly as stock does.
  float amax = 0.0f;
#pragma unroll
  for (int j = 0; j < kVec; ++j) {
    x[j] = static_cast<float>(static_cast<scalar_t>(x[j] * rms) * w[j]);
    amax = fmaxf(amax, fabsf(x[j]));
  }

  amax = block_reduce(amax, MaxOp{}, 0.0f, smem_max);
  if (scale_ub != nullptr) amax = fminf(amax, *scale_ub);
  float const scale = fmaxf(amax / kFp8Max, kMinScale);
  if (threadIdx.x == 0) scales[token] = scale;

  if (active) {
    VecOut<kVec> vout;
#pragma unroll
    for (int j = 0; j < kVec; j += 2) {
      // Stock divides by the scale rather than multiplying by its reciprocal
      // ("Do not invert token_scale for exact match with FBGemm"). Match it.
      vout.pair[j >> 1] = floats_to_fp8x2(x[j] / scale, x[j + 1] / scale);
    }
    *reinterpret_cast<VecOut<kVec>*>(out + row_off) = vout;
  }
}

// Correctness fallback for shapes the fast path rejects (hidden_size not a multiple of
// the vector width, or too large to hold in registers). Scalar, three passes, a direct
// port of stock. Never runs on Qwen2 (hidden_size=2048, bf16).
template <typename scalar_t, bool kHasResidual>
__global__ void fused_rms_norm_quant_generic_kernel(
    c10::Float8_e4m3fn* __restrict__ out, float* __restrict__ scales,
    scalar_t const* __restrict__ input, scalar_t const* __restrict__ weight,
    scalar_t* __restrict__ residual, float const* __restrict__ scale_ub,
    float const epsilon, int const hidden_size, int64_t const input_stride) {
  int64_t const token = blockIdx.x;
  int64_t const in_base = token * input_stride;
  int64_t const row_base = token * static_cast<int64_t>(hidden_size);

  __shared__ float smem_sum[32];
  __shared__ float smem_max[32];

  float ss = 0.0f;
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float x = static_cast<float>(input[in_base + i]);
    if constexpr (kHasResidual) x += static_cast<float>(residual[row_base + i]);
    ss += x * x;
  }
  ss = block_reduce(ss, AddOp{}, 0.0f, smem_sum);
  float const rms = rsqrtf(ss / hidden_size + epsilon);

  float amax = 0.0f;
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float x = static_cast<float>(input[in_base + i]);
    if constexpr (kHasResidual) x += static_cast<float>(residual[row_base + i]);
    float const y = static_cast<float>(static_cast<scalar_t>(x * rms) * weight[i]);
    amax = fmaxf(amax, fabsf(y));
  }
  amax = block_reduce(amax, MaxOp{}, 0.0f, smem_max);
  if (scale_ub != nullptr) amax = fminf(amax, *scale_ub);
  float const scale = fmaxf(amax / kFp8Max, kMinScale);
  if (threadIdx.x == 0) scales[token] = scale;

  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float x = static_cast<float>(input[in_base + i]);
    if constexpr (kHasResidual) {
      x += static_cast<float>(residual[row_base + i]);
      residual[row_base + i] = static_cast<scalar_t>(x);
    }
    float const y = static_cast<float>(static_cast<scalar_t>(x * rms) * weight[i]);
    out[row_base + i] = float_to_fp8(y / scale);
  }
}

bool aligned16(void const* p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; }

// Diagnostic, off unless VTL_KERNEL_SYNC is set (read once). When on, launch() synchronises
// right after the kernel and, on a CUDA error, throws with the exact shape and path -- so a
// fault is attributed to its own launch instead of surfacing at some later sync in another
// test. Zero cost when off (one predicted branch per call).
bool kernel_sync_enabled() {
  static bool const v = [] {
    char const* e = std::getenv("VTL_KERNEL_SYNC");
    return e != nullptr && (e[0] == '1' || e[0] == 't' || e[0] == 'y');
  }();
  return v;
}

template <typename scalar_t>
void launch(torch::Tensor const& out, torch::Tensor const& input,
            torch::Tensor const& weight, torch::Tensor const& scales, double epsilon,
            std::optional<torch::Tensor> const& scale_ub,
            std::optional<torch::Tensor> const& residual, cudaStream_t stream) {
  constexpr int kVec = VecTraits<scalar_t>::kVec;

  int const hidden_size = input.size(-1);
  // Stock calls view({-1, hidden_size}).stride(0), which heap-allocates a TensorImpl on
  // every one of the 72 calls per forward. The row stride is stride(-2) for a 2-D tensor
  // and hidden_size for a contiguous higher-rank one; the caller checks which.
  int64_t const input_stride = input.dim() == 2 ? input.stride(0) : hidden_size;
  int64_t const num_tokens = input.numel() / hidden_size;
  if (num_tokens == 0) return;

  auto* out_p = reinterpret_cast<c10::Float8_e4m3fn*>(out.data_ptr());
  auto* scales_p = scales.data_ptr<float>();
  auto const* in_p = input.const_data_ptr<scalar_t>();
  auto const* w_p = weight.const_data_ptr<scalar_t>();
  auto* res_p = residual.has_value() ? residual->data_ptr<scalar_t>() : nullptr;
  auto const* ub_p = scale_ub.has_value() ? scale_ub->const_data_ptr<float>() : nullptr;

  int const nthreads = (hidden_size + kVec - 1) / kVec;
  // out_p feeds an aligned VecOut store, so it must be gated too -- torch's allocator returns
  // 256-byte-aligned storage today, but demote to the generic path rather than assume it.
  bool const fast = hidden_size % kVec == 0 && nthreads <= kFastMaxThreads &&
                    input_stride % kVec == 0 && aligned16(in_p) && aligned16(w_p) &&
                    aligned16(out_p) && (res_p == nullptr || aligned16(res_p));

  dim3 const grid(num_tokens);
#define VTL_DISPATCH_RESIDUAL(KERNEL, BLOCK)                                          \
  do {                                                                                \
    if (res_p != nullptr) {                                                           \
      KERNEL<scalar_t, true><<<grid, (BLOCK), 0, stream>>>(                           \
          out_p, scales_p, in_p, w_p, res_p, ub_p, static_cast<float>(epsilon),       \
          hidden_size, input_stride);                                                 \
    } else {                                                                          \
      KERNEL<scalar_t, false><<<grid, (BLOCK), 0, stream>>>(                          \
          out_p, scales_p, in_p, w_p, res_p, ub_p, static_cast<float>(epsilon),       \
          hidden_size, input_stride);                                                 \
    }                                                                                 \
  } while (0)

  // Both block sizes are rounded up to a whole number of warps. block_reduce shuffles
  // with a full 0xffffffff mask, which is only valid if every named lane exists; threads
  // past hidden_size simply contribute the reduction identity.
  if (fast) {
    dim3 const block((nthreads + 31) / 32 * 32);
    VTL_DISPATCH_RESIDUAL(fused_rms_norm_quant_kernel, block);
  } else {
    dim3 const block(std::min((hidden_size + 31) / 32 * 32, 1024));
    VTL_DISPATCH_RESIDUAL(fused_rms_norm_quant_generic_kernel, block);
  }
#undef VTL_DISPATCH_RESIDUAL
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (kernel_sync_enabled()) {
    cudaError_t const err = cudaStreamSynchronize(stream);
    TORCH_CHECK(
        err == cudaSuccess, "vtl kernel faulted: ", cudaGetErrorString(err),
        " | path=", (fast ? "fast" : "generic"), " dtype=", input.scalar_type(),
        " num_tokens=", num_tokens, " hidden=", hidden_size, " stride=", input_stride,
        " kVec=", kVec, " residual=", (res_p != nullptr),
        " aligned16(in,out,w,res)=", aligned16(in_p), aligned16(out_p), aligned16(w_p),
        (res_p ? aligned16(res_p) : true));
  }
}

}  // namespace

// Signature and mutation set must match `_C::rms_norm_dynamic_per_token_quant` exactly:
//   (Tensor! result, Tensor input, Tensor weight, Tensor! scale, float epsilon,
//    Tensor? scale_ub, Tensor!? residual) -> ()
// Taken by const reference, not by value: a std::optional<Tensor> copy bumps the
// intrusive refcount, and that is an atomic RMW on a line other threads share. Four of
// them per call, 72 calls per forward, for arguments we only read.
void rms_norm_dynamic_per_token_quant(torch::Tensor& result, torch::Tensor const& input,
                                      torch::Tensor const& weight, torch::Tensor& scale,
                                      double epsilon,
                                      std::optional<torch::Tensor> const& scale_ub,
                                      std::optional<torch::Tensor> const& residual) {
  // ponytail: fp8-e4m3 only. Stock also accepts int8 out, but nothing reaches this op
  // with int8 (QUANT_OPS maps only fp8 keys, and _custom_ops' wrapper has no callers).
  // A loud check beats a second quant path we cannot verify. Add int8 if it ever fires.
  TORCH_CHECK(result.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "vtl: rms_norm_dynamic_per_token_quant supports fp8_e4m3 output only, got ",
              result.scalar_type());
  TORCH_CHECK(result.is_contiguous(), "vtl: result must be contiguous");
  TORCH_CHECK(input.dim() >= 2, "vtl: input must be at least 2-D");
  TORCH_CHECK(input.stride(-1) == 1, "vtl: input must be contiguous in the last dim");
  // launch() derives the row stride without allocating a view, which is only sound for a
  // 2-D tensor or a fully contiguous higher-rank one.
  TORCH_CHECK(input.dim() == 2 || input.is_contiguous(),
              "vtl: input with rank > 2 must be contiguous");
  TORCH_CHECK(weight.scalar_type() == input.scalar_type(), "vtl: weight dtype mismatch");
  TORCH_CHECK(weight.is_contiguous(), "vtl: weight must be contiguous");
  TORCH_CHECK(scale.scalar_type() == at::ScalarType::Float, "vtl: scale must be fp32");
  if (residual.has_value()) {
    TORCH_CHECK(residual->scalar_type() == input.scalar_type(),
                "vtl: residual dtype mismatch");
    TORCH_CHECK(residual->is_contiguous(), "vtl: residual must be contiguous");
  }

  c10::cuda::OptionalCUDAGuard const device_guard(input.device());
  cudaStream_t const stream = c10::cuda::getCurrentCUDAStream();

  switch (input.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch<c10::BFloat16>(result, input, weight, scale, epsilon, scale_ub, residual,
                            stream);
      break;
    case at::ScalarType::Half:
      launch<c10::Half>(result, input, weight, scale, epsilon, scale_ub, residual,
                        stream);
      break;
    case at::ScalarType::Float:
      launch<float>(result, input, weight, scale, epsilon, scale_ub, residual, stream);
      break;
    default:
      TORCH_CHECK(false, "vtl: unsupported input dtype ", input.scalar_type());
  }
}

}  // namespace vtl
