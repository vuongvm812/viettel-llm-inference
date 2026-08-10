// NVRTC variant of the fused residual-add + RMSNorm + dynamic per-token FP8 quant kernel.
//
// THIS IS THE TEMPLATE. Copy it when adding a kernel you want to iterate on without a
// 20-minute image rebuild. Three rules make a file NVRTC-compilable, and the AOT kernel in
// vtl/csrc/rms_norm_quant.cu breaks all three (which is why this is a port, not a copy):
//
//   1. NO torch/ATen headers. NVRTC sees only CUDA's own headers, so there is no
//      torch::Tensor, no c10::BFloat16, no AT_DISPATCH. Entry points take raw pointers
//      and the caller passes .data_ptr().
//   2. NO host code. Only __global__/__device__ functions. Launch config is computed on
//      the Python side, where the shapes already live.
//   3. Shapes arrive as -D macros, not arguments. That is the entire point: HIDDEN is a
//      compile-time constant here, so the loop below fully unrolls, HIDDEN/kVec folds to a
//      literal, and the compiler allocates exactly the registers this shape needs. The AOT
//      build cannot do any of that, because at AOT time the model is unknown.
//
// Required defines (vtl/patches/rms_norm_quant.py supplies them from the loaded config):
//   HIDDEN        hidden size; MUST be a multiple of 8 for this fast path
//   THREADS       block width; HIDDEN/8 must be <= THREADS and divide evenly
//   HAS_RESIDUAL  1 to fuse the residual add (and write it back), 0 to skip
//
// NUMERICS ARE MATCHED TO THE AOT KERNEL, INCLUDING ITS DOUBLE ROUNDING:
//   x    = f32(input) + f32(residual)            (residual := bf16(x))
//   rms  = rsqrtf(sum(x^2) / HIDDEN + eps)
//   y    = f32(bf16(x * rms) * bf16(weight))     <- BOTH narrowings are load-bearing
//   s    = max(max|y| / 448, 1/(448*512))
//   out  = fp8(clamp(y / s, -448, 448))
// Keeping `y` in fp32 would be more accurate and would shift the per-token scale by ~6e-5
// relative -- a different kernel, not a faster one. Do not "fix" it.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#ifndef HIDDEN
#error "NVRTC: -DHIDDEN=<hidden_size> is required"
#endif
#ifndef THREADS
#define THREADS 256
#endif
#ifndef HAS_RESIDUAL
#define HAS_RESIDUAL 0
#endif

#define VEC 8                  // 16-byte loads: 8 x bf16
#define NVEC (HIDDEN / VEC)    // total vectors per token == number of ACTIVE threads

static_assert(HIDDEN % VEC == 0, "HIDDEN must be a multiple of 8 for the vectorized path");
static_assert(NVEC <= THREADS, "one vector per thread: HIDDEN/8 must be <= THREADS");

// constexpr, not __constant__: a __constant__ needs a host-side symbol copy to be defined,
// which an NVRTC module has no way to do. These fold into the SASS as immediates anyway.
__device__ constexpr float kFp8Max = 448.0f;
__device__ constexpr float kMinScale = 1.0f / (448.0f * 512.0f);

// One 16-byte load.
struct __align__(16) Vec8 { __nv_bfloat16 v[VEC]; };

// Standard block reduction through shared memory. THREADS is a compile-time constant, so
// the warp count and the final loop bound are literals.
template <bool kIsMax>
__device__ __forceinline__ float block_reduce(float val, float* smem) {
    // Literal 32, not the built-in `warpSize`: warpSize is a runtime value, so the loop
    // would not unroll and the shuffle offsets would not be immediates.
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_down_sync(0xffffffffu, val, off);
        val = kIsMax ? fmaxf(val, other) : val + other;
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    // This barrier is what makes the function safe to call TWICE on the same smem. Without
    // it a thread that has already returned from the first reduction can race ahead and
    // overwrite smem[warp] while a slower thread is still reading the first result.
    __syncthreads();
    if (lane == 0) smem[warp] = val;
    __syncthreads();

    constexpr int kWarps = (THREADS + 31) / 32;
    // 0 is the identity for both a sum and an abs-max, so inactive lanes need no special case.
    val = (threadIdx.x < kWarps) ? smem[threadIdx.x] : 0.0f;
    if (warp == 0) {
#pragma unroll
        for (int off = kWarps / 2; off > 0; off >>= 1) {
            float other = __shfl_down_sync(0xffffffffu, val, off);
            val = kIsMax ? fmaxf(val, other) : val + other;
        }
    }
    if (threadIdx.x == 0) smem[0] = val;
    __syncthreads();
    const float result = smem[0];   // read before any caller can re-enter and clobber it
    return result;
}

// One token per block. Each thread holds its VEC elements in registers across BOTH
// reductions, so input and residual are read exactly once.
extern "C" __global__ void __launch_bounds__(THREADS) rms_norm_quant(
    __nv_fp8_e4m3* __restrict__ out,      // [num_tokens, HIDDEN]
    float* __restrict__ scales,           // [num_tokens]
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ residual, // [num_tokens, HIDDEN], in/out; ignored if !HAS_RESIDUAL
    const float epsilon) {
    __shared__ float smem[(THREADS + 31) / 32];

    const long long token = blockIdx.x;
    const long long base = token * (long long)HIDDEN;
    const int tid = threadIdx.x;

    float x[VEC];
    const bool active = tid < NVEC;

    // --- pass 1: load once, keep in registers, reduce sum(x^2) ---
    float sumsq = 0.0f;
    if (active) {
        const Vec8 vin = reinterpret_cast<const Vec8*>(input + base)[tid];
#if HAS_RESIDUAL
        Vec8 vres = reinterpret_cast<Vec8*>(residual + base)[tid];
#endif
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
            x[i] = __bfloat162float(vin.v[i]);
#if HAS_RESIDUAL
            x[i] += __bfloat162float(vres.v[i]);
            vres.v[i] = __float2bfloat16(x[i]);   // residual := bf16(x), as stock does
#endif
            sumsq += x[i] * x[i];
        }
#if HAS_RESIDUAL
        reinterpret_cast<Vec8*>(residual + base)[tid] = vres;
#endif
    }
    sumsq = block_reduce<false>(sumsq, smem);
    const float rms = rsqrtf(sumsq / (float)HIDDEN + epsilon);

    // --- pass 2: normalize + weight (double-rounded, see header), reduce max|y| ---
    float y[VEC];
    float amax = 0.0f;
    if (active) {
        const Vec8 vw = reinterpret_cast<const Vec8*>(weight)[tid];
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
            // bf16(x*rms) * bf16(weight), the product itself narrowed to bf16 -- this is
            // what c10::BFloat16's homogeneous operator* does in the AOT kernel.
            const __nv_bfloat16 xn = __float2bfloat16(x[i] * rms);
            y[i] = __bfloat162float(__hmul(xn, vw.v[i]));
            amax = fmaxf(amax, fabsf(y[i]));
        }
    }
    amax = block_reduce<true>(amax, smem);

    const float scale = fmaxf(amax / kFp8Max, kMinScale);
    if (tid == 0) scales[token] = scale;

    // --- pass 3: quantize. One true divide, then reciprocal-multiply (not fast-math). ---
    const float inv = 1.0f / scale;
    if (active) {
        __nv_fp8_e4m3 o[VEC];
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
            o[i] = __nv_fp8_e4m3(fminf(fmaxf(y[i] * inv, -kFp8Max), kFp8Max));
        }
        // 8 fp8 values = 8 bytes = one uint2 store.
        reinterpret_cast<uint2*>(out + base)[tid] = *reinterpret_cast<uint2*>(&o[0]);
    }
}
