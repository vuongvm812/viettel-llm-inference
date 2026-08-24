// NVRTC re-specialization of vLLM's STOCK group-128 fp8 block-quant CUDA kernels:
//
//   entry `rms_norm_block_quant`  == _C::rms_norm_per_block_quant
//        (csrc/libtorch_stable/quantization/fused_kernels/
//         fused_layernorm_dynamic_per_token_quant.cu :: rms_norm_per_block_quant_kernel)
//   entry `group_quant_fp8`      == _C::per_token_group_fp8_quant (non-ue8m0, fp8 path)
//        (csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu
//         :: per_token_group_quant_8bit_kernel)
//
// Same three NVRTC rules as vtl/kernels/rms_norm_quant.cu (the template): no torch
// headers, no host code, shapes as -D macros. The edge over stock is compile-time
// HIDDEN/GROUP: the per-token loop is gone (one vec8 per thread, exactly), group offsets
// fold to immediates, and 16-byte loads replace stock's 8-byte vec4 path. The ALGORITHM
// and the ROUNDING are stock's, not vtl's per-token kernel's -- the two differ:
//
// NUMERICS (matched to the STOCK ops, deliberate double rounding included):
//   rms entry:
//     x    = f32(input) + f32(residual)          (residual := bf16(x), written back)
//     rms  = rsqrtf(sum(x^2) / HIDDEN + eps)
//     y    = f32(bf16(x * rms) * bf16(weight))   <- both narrowings load-bearing
//     s_g  = max(min(amax_g, scale_ub) / 448, 1/(448*512))   per GROUP g of 128, not per token
//     out  = fp8( fmax(-448, fmin(y / s_g, 448)) )
//   NOTE the quant is a TRUE DIVISION y / s. Stock does not invert the scale for fp8
//   ("exact match with FBGemm"); vtl's per-token kernel multiplies by 1/s. Do not "fix".
//   group entry (no norm, raw activation quant):
//     s_g  = max(amax_g, eps) / 448              (eps=1e-10 seeds the absmax, stock-style)
//     out  = fp8( fmin(fmax(x / s_g, fp8_min), fp8_max) )
//   The fp32 sum(x^2) reduction ORDER differs from stock's cub::BlockReduce; every other
//   op is order-independent (max, per-element rounds). The bf16 narrowing after the rms
//   multiply absorbs the last-ulp rsqrt input difference in practice; the GPU parity test
//   (bench/test_nvrtc_block_quant.py) holds this to bit-equality against a torch oracle,
//   same doctrine as bench/test_nvrtc.py.
//
// Required defines (vtl/patches/nvrtc_block_quant.py supplies them from the config):
//   HIDDEN        hidden size (rms entry only); multiple of GROUP
//   GROUP         quant group size; this port requires 128 (= 16 lanes x 8 elems)
//   THREADS       rms entry block width; MUST equal HIDDEN/8 (one vec8 per thread)
//   HAS_RESIDUAL  1 to fuse the residual add (and write it back), 0 to skip
//   GQ_THREADS    group entry block width (default 256 -> 16 groups per block)
//
// Scale layouts (runtime args, matching stock exactly):
//   rms entry, transposed==0:  scales[token * (HIDDEN/GROUP) + g]
//   rms entry, transposed==1:  scales[g * scale_rows + token],
//                              scale_rows = ceil(gridDim.x / outer_stride) * outer_stride
//   group entry, row-major:    scales[gid]
//   group entry, col-major:    scales[(gid % snum_rows) * sstride + gid / snum_rows]

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#ifndef HIDDEN
#error "NVRTC: -DHIDDEN=<hidden_size> is required"
#endif
#ifndef GROUP
#error "NVRTC: -DGROUP=<group_size> is required"
#endif
#ifndef THREADS
#define THREADS (HIDDEN / 8)
#endif
#ifndef HAS_RESIDUAL
#define HAS_RESIDUAL 0
#endif
#ifndef GQ_THREADS
#define GQ_THREADS 256
#endif

#define VEC 8                        // 16-byte loads: 8 x bf16
#define NVEC (HIDDEN / VEC)          // vectors per token
#define LANES_PER_GROUP (GROUP / VEC)
#define NGROUPS (HIDDEN / GROUP)

static_assert(GROUP == 128, "this port fixes GROUP=128 (16 lanes x vec8); stock also does 64");
static_assert(HIDDEN % GROUP == 0, "HIDDEN must be a multiple of GROUP");
static_assert(THREADS == NVEC, "one vec8 per thread: THREADS must equal HIDDEN/8");
static_assert(THREADS % 32 == 0 && THREADS <= 1024, "THREADS must be whole warps, <= 1024");
static_assert(LANES_PER_GROUP == 16, "16 lanes per group is what the shuffle masks assume");
static_assert(GQ_THREADS % 32 == 0 && GQ_THREADS <= 1024, "GQ_THREADS must be whole warps");

__device__ constexpr float kFp8Max = 448.0f;                  // quant_type_max_v<fp8e4m3>
__device__ constexpr float kMinScale = 1.0f / (448.0f * 512.0f);  // min_scaling_factor

struct __align__(16) Vec8 { __nv_bfloat16 v[VEC]; };

// Block-wide fp32 sum. Unlike the rms_norm_quant.cu template this handles a NON-power-of-2
// warp count (HIDDEN=3072 -> THREADS=384 -> 12 warps): the cross-warp tree starts at the
// next power of two and missing slots read 0, the identity.
__device__ __forceinline__ float block_sum(float val, float* smem) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, off);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __syncthreads();  // safe re-entry if a caller ever reduces twice on the same smem
    if (lane == 0) smem[warp] = val;
    __syncthreads();
    constexpr int kWarps = THREADS / 32;
    constexpr int kPow2 = (kWarps <= 1) ? 1 : (kWarps <= 2) ? 2 : (kWarps <= 4) ? 4
                        : (kWarps <= 8) ? 8 : (kWarps <= 16) ? 16 : 32;
    val = (threadIdx.x < kWarps) ? smem[threadIdx.x] : 0.0f;
    if (warp == 0) {
#pragma unroll
        for (int off = kPow2 / 2; off > 0; off >>= 1)
            val += __shfl_down_sync(0xffffffffu, val, off);
    }
    if (threadIdx.x == 0) smem[0] = val;
    __syncthreads();
    return smem[0];
}

// Abs-max across the 16 lanes of one group (a half-warp). The xor ladder leaves the
// result in EVERY lane, so no broadcast is needed before the quant loop.
__device__ __forceinline__ float halfwarp_amax(float v) {
    const unsigned mask = 0xffffu << (threadIdx.x & 16u);
#pragma unroll
    for (int off = 8; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(mask, v, off));
    return v;
}

// ============================ entry 1: fused rms-norm + group quant =====================
// One token per block; thread t owns vec8 #t (elements 8t..8t+7); threads 16g..16g+15 own
// group g. Grid: (num_tokens, 1, 1). Block: (THREADS, 1, 1).
extern "C" __global__ void __launch_bounds__(THREADS) rms_norm_block_quant(
    __nv_fp8_e4m3* __restrict__ out,       // [num_tokens, HIDDEN], contiguous (stock checks)
    float* __restrict__ scales,            // see layout note in the header
    const __nv_bfloat16* __restrict__ input,   // [num_tokens, HIDDEN], row stride in_stride
    const __nv_bfloat16* __restrict__ weight,  // [HIDDEN]
    const float* __restrict__ scale_ub,    // optional; 0 if absent
    __nv_bfloat16* __restrict__ residual,  // [num_tokens, HIDDEN] in/out; ignored unless HAS_RESIDUAL
    const float epsilon,
    const long long in_stride,             // input.view(-1, HIDDEN).stride(0), elements
    const int transposed,                  // is_scale_transposed
    const long long outer_stride) {        // scales.stride(1); only read when transposed
    __shared__ float smem[THREADS / 32];

    const long long token = blockIdx.x;
    const long long in_base = token * in_stride;
    const long long base = token * (long long)HIDDEN;   // out/residual are contiguous
    const int tid = threadIdx.x;

    // --- pass 1: load once, keep fp32 x in registers, reduce sum(x^2) ---
    float x[VEC];
    float sumsq = 0.0f;
    {
        const Vec8 vin = reinterpret_cast<const Vec8*>(input + in_base)[tid];
#if HAS_RESIDUAL
        Vec8 vres = reinterpret_cast<Vec8*>(residual + base)[tid];
#endif
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
            x[i] = __bfloat162float(vin.v[i]);
#if HAS_RESIDUAL
            x[i] += __bfloat162float(vres.v[i]);
            vres.v[i] = __float2bfloat16(x[i]);   // residual := bf16(x), exactly as stock
#endif
            sumsq += x[i] * x[i];
        }
#if HAS_RESIDUAL
        reinterpret_cast<Vec8*>(residual + base)[tid] = vres;
#endif
    }
    sumsq = block_sum(sumsq, smem);
    const float rms = rsqrtf(sumsq / (float)HIDDEN + epsilon);

    // --- pass 2: y = f32(bf16(x*rms) * w) (stock's double rounding), group abs-max ---
    float y[VEC];
    float amax = 0.0f;
    {
        const Vec8 vw = reinterpret_cast<const Vec8*>(weight)[tid];
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
            const __nv_bfloat16 xn = __float2bfloat16(x[i] * rms);
            y[i] = __bfloat162float(__hmul(xn, vw.v[i]));   // bf16*bf16, product narrowed
            amax = fmaxf(amax, fabsf(y[i]));
        }
    }
    amax = halfwarp_amax(amax);   // every lane of the group now holds the group amax

    // --- scale: min(amax, ub)/448, floored -- identical in every lane, stored by one ---
    if (scale_ub != nullptr) amax = fminf(amax, *scale_ub);
    const float scale = fmaxf(amax / kFp8Max, kMinScale);
    const int g = tid / LANES_PER_GROUP;
    if ((tid & (LANES_PER_GROUP - 1)) == 0) {
        if (transposed) {
            // scale_rows = ceil(gridDim.x / outer_stride) * outer_stride -- stock verbatim.
            const long long rows =
                (gridDim.x + outer_stride - 1) / outer_stride * outer_stride;
            scales[(long long)g * rows + token] = scale;
        } else {
            scales[token * NGROUPS + g] = scale;
        }
    }

    // --- pass 3: quantize. TRUE division by the group scale, as stock does for fp8. ---
    __nv_fp8_e4m3 o[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
        o[i] = __nv_fp8_e4m3(fmaxf(-kFp8Max, fminf(y[i] / scale, kFp8Max)));
    }
    reinterpret_cast<uint2*>(out + base)[tid] = *reinterpret_cast<uint2*>(&o[0]);
}

// ============================ entry 2: plain per-group quant ============================
// Groups are counted over the FLATTENED input (stock: num_groups = numel/group_size), so
// this entry does not depend on HIDDEN and serves every activation shape in the model.
// One group per half-warp; GQ_THREADS/16 groups per block.
// Grid: (ceil(num_groups / (GQ_THREADS/16)), 1, 1). Block: (GQ_THREADS, 1, 1).
extern "C" __global__ void __launch_bounds__(GQ_THREADS) group_quant_fp8(
    __nv_fp8_e4m3* __restrict__ out_q,   // same flat layout as input, contiguous
    float* __restrict__ out_s,           // row- or col-major, see header
    const __nv_bfloat16* __restrict__ input,  // contiguous (stock checks)
    const float eps,                     // seeds the absmax (stock: local_absmax = eps)
    const float fp8_min,
    const float fp8_max,
    const long long num_groups,
    const int col_major,                 // output_s.stride(0) < output_s.stride(1) in stock
    const int snum_rows,                 // output_s.size(1)
    const int sstride) {                 // output_s.stride(1)
    const long long gid =
        (long long)blockIdx.x * (GQ_THREADS / LANES_PER_GROUP) + (threadIdx.x >> 4);
    if (gid >= num_groups) return;
    const int lane = threadIdx.x & (LANES_PER_GROUP - 1);
    const long long elem_base = gid * GROUP;

    const Vec8 vin =
        reinterpret_cast<const Vec8*>(input + elem_base)[lane];
    float xv[VEC];
    float amax = eps;   // NOT the rms entry's floor: stock seeds the max with eps=1e-10
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
        xv[i] = __bfloat162float(vin.v[i]);
        amax = fmaxf(amax, fabsf(xv[i]));
    }
    amax = halfwarp_amax(amax);

    const float y_s = amax / fp8_max;   // no lower clamp beyond the eps seed -- stock
    if (lane == 0) {
        if (col_major) {
            // float scales -> num_elems_per_pack==1 in stock's indexing.
            const long long row = gid / snum_rows;
            const long long col = gid % snum_rows;
            out_s[col * (long long)sstride + row] = y_s;
        } else {
            out_s[gid] = y_s;
        }
    }

    __nv_fp8_e4m3 o[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
        o[i] = __nv_fp8_e4m3(fminf(fmaxf(xv[i] / y_s, fp8_min), fp8_max));
    }
    reinterpret_cast<uint2*>(out_q + elem_base)[lane] = *reinterpret_cast<uint2*>(&o[0]);
}
