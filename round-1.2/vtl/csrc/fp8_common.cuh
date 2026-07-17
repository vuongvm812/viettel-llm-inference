// Shared device/host helpers for vtl's fp8-e4m3 quant kernels.
//
// Extracted verbatim from rms_norm_quant.cu so the RMSNorm-quant kernel, the standalone
// per-token quant (attn out_proj / down_proj-quant-half), and the fused SiLU-mul quant all
// share ONE copy of: the fp8 constants, the 16-byte vector types, the
// clamp-then-hw-convert epilogue, the warp-shuffle block reduction, and the launch guards.
// Numerics are load-bearing and match stock vLLM element-for-element -- do not "simplify"
// the clamp order, the divide-vs-reciprocal choice, or the bf16 narrowing at the call sites.
#pragma once

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
#include <cstdint>

namespace vtl {

constexpr float kFp8Max = 448.0f;                      // quant_type_max_v<Float8_e4m3fn>
constexpr float kMinScale = 1.0f / (kFp8Max * 512.0f); // min_scaling_factor<fp8>::val()

// Fast-path block cap. hidden=2048 launches 256 threads at bf16/fp16 (kVec=8) and 512 at
// fp32 (kVec=4), so 512 covers every fast-path shape; wider hidden falls to the generic /
// chunked kernels. Given as an explicit __launch_bounds__ so the SM90 register allocator
// sizes for a <=512-thread block instead of the default 1024 worst case, keeping more
// blocks resident to hide HBM3e latency in the low-occupancy decode regime.
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
// atomicMax would serialise every lane through one address and make the result depend on
// arrival order. `smem` is caller-owned so two reductions can run back to back without a
// barrier between them just to protect a shared buffer. `smem` must hold >= nwarps floats.
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

inline bool aligned16(void const* p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; }

// Diagnostic, off unless VTL_KERNEL_SYNC is set (read once). When on, a launch() synchronises
// right after the kernel and, on a CUDA error, throws with the exact shape and path -- so a
// fault is attributed to its own launch instead of surfacing at some later sync in another
// test. Zero cost when off (one predicted branch per call).
inline bool kernel_sync_enabled() {
  static bool const v = [] {
    char const* e = std::getenv("VTL_KERNEL_SYNC");
    return e != nullptr && (e[0] == '1' || e[0] == 't' || e[0] == 'y');
  }();
  return v;
}

}  // namespace vtl
