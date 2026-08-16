// NVRTC grouped-GEMV MoE decode kernel: W4A8 (int4 group-128 weights, fp8-e4m3 activations)
// with a -DWEIGHT_FP8=1 variant that reads the checkpoint's fp8 block-quant expert weights
// directly (the A/B arm to run before the quality gate clears the int4 requant).
//
// WHY A GEMV. Decode M is tiny (trace peak ~6 seqs, MTP adds a few tokens) and top-8-of-256
// routing makes expert collisions across tokens rare, so the grouped-GEMM machinery of the
// stock fused_moe path buys nothing: the whole job is streaming 8 x (w13 + w2) expert slices
// per token from HBM exactly once and doing ~one FMA per loaded value. The design goal is
// HBM-bandwidth-bound streaming: 16-byte loads of packed weights, one pass, fp32 accumulation
// (NO fast-math -- the accumulation order is part of the numerics contract, see nvrtc.py).
//
// DESIGN: FOUR LAUNCHES, ATOMIC-FREE. One block per (token, expert-slot, output-group):
//
//   1. moe_x_quant     x bf16 [M,K]           -> xq fp8 [M,K] + x_scale f32 [M,K/128]
//                      (dynamic group-128 symmetric, same recipe as the fp8-block act path)
//   2. moe_gemv_up     per (token,slot): h[n] = silu(w1[n].xq) * (w3[n].xq), then group-128
//                      fp8 requant of h. One block per (token,slot, group of 128 outputs) --
//                      the requant group IS the block, so the absmax is one block reduction.
//   3. moe_gemv_down   per (token,slot): staging[m,slot,k] = w2[k].h, fp32, UNWEIGHTED, into
//                      a private [M,TOPK,K] staging row -- no two blocks write the same
//                      element, so no atomics and the result is deterministic.
//   4. moe_finalize    out[m,k] = sum_slot staging[m,slot,k] * topk_weight[m,slot], cast bf16.
//
// Staging + a second pass was chosen over atomicAdd on an fp32 output because it is
// deterministic (a flake and a regression stay distinguishable), the staging tensor is tiny
// (M=16: 16*8*3072*4 B = 1.5 MB, resident in L2 between launches 3 and 4), and the extra
// launch is noise under CUDA-graph replay. cp.async is deliberately NOT used: each weight
// byte is touched exactly once, so there is nothing to pipeline against that the hardware's
// own miss-under-miss does not already cover; __ldg/__restrict__ mark the streams read-only.
//
// A fifth entry, moe_int4_to_fp8, expands the int4-packed experts back into fp8 (against the
// ORIGINAL [128,128] block scales) so that prefill / big-M batches could fall through to the
// stock fused_moe kernel after the int4 arm has freed the checkpoint's fp8 expert weights.
// RESERVED: vtl/patches/moe_decode_gemv.py does NOT call it today (with the fp8 originals
// freed, big-M runs the GEMV in token chunks instead -- correct, and slow, which is why
// VTL_MOE_GEMV_FREE_FP8 defaults to 0). It is kept here because it is the only shape of a
// real prefill answer for the int4 arm, and bench/test_moe_decode.py checks its numerics so
// it cannot rot between now and being wired up.
//
// WEIGHT LAYOUTS (defined here, packed by vtl/patches/moe_decode_gemv.py -- NOT the CUTLASS
// tile-interleaved layout quant_w4a8 uses for dense layers; a GEMV wants plain rows):
//   int4 arm:  w13_q  int32 [E, 2N, K/8]   word j holds elements k=8j..8j+7, nibble i = k=8j+i,
//                                          two's-complement signed int4 in [-8, 7]
//              w13_gs fp32  [E, 2N, K/128] RTN group scale (absmax/7), grouped along K
//              w2_q   int32 [E, K, N/8]    same packing, reduction dim = N
//              w2_gs  fp32  [E, K, N/128]
//   fp8 arm:   w13    fp8   [E, 2N, K]     the checkpoint tensors, untouched
//              w13_bs fp32  [E, 2N/128, K/128]  the checkpoint's [128,128] block scales
//              w2     fp8   [E, K, N]
//              w2_bs  fp32  [E, K/128, N/128]
// In BOTH arms the scale along the reduction dim changes every 128 elements, so the inner
// dot product is identical; only the per-row scale pointer and the 16-byte fragment decode
// differ, selected by WEIGHT_FP8 at compile time.
//
// Required defines (vtl/patches/moe_decode_gemv.py supplies them from the loaded model):
//   K        hidden size (reduction dim of w13, output dim of w2); % 128 == 0
//   N        moe intermediate size (output dim of w13 halves, reduction dim of w2); % 128 == 0
//   TOPK     experts per token
//   GROUP    quant group size along the reduction dim; must be 128
//   THREADS  block width, power of two, default 256
//   WEIGHT_FP8  0 = int4-packed weights (default), 1 = checkpoint fp8 + block scales
//
// NVRTC rules (HANDOFF.md 4.2): no torch headers, no host code, extern "C" entries.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#ifndef K
#error "NVRTC: -DK=<hidden_size> is required"
#endif
#ifndef N
#error "NVRTC: -DN=<moe_intermediate_size> is required"
#endif
#ifndef TOPK
#error "NVRTC: -DTOPK=<experts_per_token> is required"
#endif
#ifndef GROUP
#define GROUP 128
#endif
#ifndef THREADS
#define THREADS 256
#endif
#ifndef WEIGHT_FP8
#define WEIGHT_FP8 0
#endif

#define KGROUPS (K / GROUP)
#define NGROUPS (N / GROUP)
#define WARPS (THREADS / 32)

static_assert(GROUP == 128, "the [128,128] block-scale indexing assumes GROUP == 128");
static_assert(K % GROUP == 0, "K must be a multiple of GROUP");
static_assert(N % GROUP == 0, "N must be a multiple of GROUP");
static_assert(K % 32 == 0 && N % 32 == 0, "32-element chunks must tile the reduction dims");
static_assert((THREADS & (THREADS - 1)) == 0, "THREADS must be a power of two");
static_assert(THREADS >= 32 && THREADS <= 1024, "THREADS out of range");
static_assert(GROUP % WARPS == 0, "each warp must get a whole number of output rows");
// One requant group per block, carried by the first GROUP threads: block_absmax reduces over
// THREADS lanes, so a block narrower than the group would silently drop the tail.
static_assert(THREADS >= GROUP, "THREADS must be >= GROUP (one requant group per block)");

__device__ constexpr float kFp8Max = 448.0f;
// Same floor as rms_norm_quant.cu: an all-zero group must quantize with a finite scale.
__device__ constexpr float kMinScale = 1.0f / (448.0f * 512.0f);

// 16-byte fragments. 16 fp8 bytes = 16 elements; one uint4 = 32 packed int4 elements.
// A UNION, not a struct: `__ldg` is only declared for the built-in vector types, so the load
// has to name uint4 and the fp8 view has to alias it. (`__ldg(const FP8x16*)` does not exist
// and is a compile error, not a slow path.)
union __align__(16) Frag16 {
    uint4 b;
    __nv_fp8_e4m3 v[16];
};

// 8 fp8 bytes: the store granularity of moe_int4_to_fp8. Same reason for the union -- an
// `__nv_fp8_e4m3 x[8]` local is only 1-byte aligned, so casting it to uint2* is a misaligned
// store on the stack.
union __align__(8) Frag8 {
    uint2 b;
    __nv_fp8_e4m3 v[8];
};

__device__ __forceinline__ uint4 ldg16(const void* __restrict__ p) {
    return __ldg(reinterpret_cast<const uint4*>(p));
}

// ---------------------------------------------------------------------------------------
// Block-wide absmax over THREADS lanes (inactive lanes pass 0, the identity for absmax).
// smem must hold WARPS floats. Result is broadcast to every thread.
// ---------------------------------------------------------------------------------------
__device__ __forceinline__ float block_absmax(float v, float* smem) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __syncthreads();  // safe against back-to-back calls on the same smem
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    v = (threadIdx.x < WARPS) ? smem[threadIdx.x] : 0.0f;
    if (warp == 0) {
#pragma unroll
        for (int off = WARPS / 2; off > 0; off >>= 1)
            v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
        if (threadIdx.x == 0) smem[0] = v;
    }
    __syncthreads();
    const float result = smem[0];
    return result;
}

// ---------------------------------------------------------------------------------------
// One warp computes one dot product of length LEN (compile-time), reduction dim strided in
// 32-element chunks across the 32 lanes. Both operands are dequantized inline:
//   weight element  = q * w_scale_row[k / GROUP]   (q: signed int4 nibble, or fp8 byte)
//   activation      = a * act_scale[k / GROUP]     (a: fp8 byte)
// The two scales are folded into one multiply per 32-element chunk (a chunk never straddles
// a group boundary: GROUP=128 is a multiple of 32). Accumulation is fp32 throughout.
// Returns the full dot product, broadcast to every lane of the warp.
// ---------------------------------------------------------------------------------------
template <int LEN>
__device__ __forceinline__ float warp_dot(
    const unsigned char* __restrict__ w_row,   // LEN/2 bytes (int4) or LEN bytes (fp8)
    const float* __restrict__ w_scale_row,     // one float per GROUP elements
    const __nv_fp8_e4m3* __restrict__ act,     // LEN fp8 activations (16B aligned)
    const float* __restrict__ act_scale) {     // one float per GROUP elements
    constexpr int kChunks = LEN / 32;
    const int lane = threadIdx.x & 31;
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < (kChunks + 31) / 32; ++i) {
        const int c = i * 32 + lane;
        if (kChunks % 32 == 0 || c < kChunks) {
            const int k0 = c * 32;
            Frag16 a0, a1;
            a0.b = ldg16(act + k0);
            a1.b = ldg16(act + k0 + 16);
            float partial = 0.0f;
#if WEIGHT_FP8
            Frag16 w0, w1;
            w0.b = ldg16(w_row + (long long)k0);
            w1.b = ldg16(w_row + (long long)k0 + 16);
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                partial += float(w0.v[j]) * float(a0.v[j]);
                partial += float(w1.v[j]) * float(a1.v[j]);
            }
#else
            const uint4 wbits = ldg16(w_row + (long long)k0 / 2);
            const unsigned int words[4] = {wbits.x, wbits.y, wbits.z, wbits.w};
#pragma unroll
            for (int w = 0; w < 4; ++w) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    // Sign-extend nibble j of word w: elements k0 + 8*w + j.
                    const int q = ((int)(words[w] << (28 - 4 * j))) >> 28;
                    const __nv_fp8_e4m3 a =
                        (w < 2) ? a0.v[8 * w + j] : a1.v[8 * (w - 2) + j];
                    partial += (float)q * float(a);
                }
            }
#endif
            acc += partial * w_scale_row[k0 / GROUP] * act_scale[k0 / GROUP];
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    return __shfl_sync(0xffffffffu, acc, 0);
}

// Per-row scale pointers: the ONLY place the two weight formats diverge outside warp_dot.
// int4: one fp32 group scale per (row, 128 K-elements). fp8: the [128,128] block scale row.
__device__ __forceinline__ const float* w_scale_row_ptr(
    const float* __restrict__ scales, long long expert, long long rows_per_expert,
    long long row, long long len_groups) {
#if WEIGHT_FP8
    return scales + (expert * (rows_per_expert / GROUP) + row / GROUP) * len_groups;
#else
    return scales + (expert * rows_per_expert + row) * len_groups;
#endif
}

__device__ __forceinline__ long long w_row_bytes(long long len) {
#if WEIGHT_FP8
    return len;       // one byte per element
#else
    return len / 2;   // one nibble per element
#endif
}

// ---------------------------------------------------------------------------------------
// 1. Activation quant: bf16 [M, K] -> fp8 [M, K] + fp32 group scales [M, K/GROUP].
//    grid = (m * KGROUPS), block = THREADS; the first GROUP threads carry the data.
// ---------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(THREADS) moe_x_quant(
    __nv_fp8_e4m3* __restrict__ xq,        // [m, K]
    float* __restrict__ x_scale,           // [m, KGROUPS]
    const __nv_bfloat16* __restrict__ x) { // [m, K]
    __shared__ float smem[WARPS];
    const long long tok = blockIdx.x / KGROUPS;
    const int g = blockIdx.x % KGROUPS;
    const int t = threadIdx.x;
    const long long base = tok * K + (long long)g * GROUP;

    float v = 0.0f;
    if (t < GROUP) v = __bfloat162float(x[base + t]);
    const float amax = block_absmax(fabsf(v), smem);
    const float scale = fmaxf(amax / kFp8Max, kMinScale);
    if (t == 0) x_scale[tok * KGROUPS + g] = scale;
    if (t < GROUP) {
        const float q = fminf(fmaxf(v / scale, -kFp8Max), kFp8Max);
        xq[base + t] = __nv_fp8_e4m3(q);
    }
}

// ---------------------------------------------------------------------------------------
// 2. w13 GEMV + silu(gate)*up + group-128 fp8 requant of the intermediate.
//    grid = (m * TOPK, NGROUPS), block = THREADS. Block (ts, ng) produces intermediate
//    elements [ng*GROUP, (ng+1)*GROUP) of (token, slot) ts -- one requant group per block.
//    w13 row n is the gate, row N + n the up (vLLM's stacked [w1; w3] convention).
// ---------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(THREADS) moe_gemv_up(
    __nv_fp8_e4m3* __restrict__ h,           // [m*TOPK, N]
    float* __restrict__ h_scale,             // [m*TOPK, NGROUPS]
    const __nv_fp8_e4m3* __restrict__ xq,    // [m, K]
    const float* __restrict__ x_scale,       // [m, KGROUPS]
    const unsigned char* __restrict__ w13,   // int4-packed or fp8, see header
    const float* __restrict__ w13_scale,
    const int* __restrict__ topk_ids,        // [m, TOPK]
    const int num_experts) {
    __shared__ float smem_a[GROUP];
    __shared__ float smem_red[WARPS];
    const long long ts = blockIdx.x;         // token * TOPK + slot
    const long long tok = ts / TOPK;
    const int ng = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int e = topk_ids[ts];
    const bool expert_ok = (e >= 0) && (e < num_experts);

    const __nv_fp8_e4m3* xrow = xq + tok * K;
    const float* xs = x_scale + tok * KGROUPS;
    const long long row_bytes = w_row_bytes(K);
    const unsigned char* wbase = w13 + (long long)e * (2LL * N) * row_bytes;

#pragma unroll
    for (int i = 0; i < GROUP / WARPS; ++i) {
        const int n_local = i * WARPS + warp;
        const long long n = (long long)ng * GROUP + n_local;
        float a = 0.0f;
        if (expert_ok) {
            const float g = warp_dot<K>(
                wbase + n * row_bytes,
                w_scale_row_ptr(w13_scale, e, 2LL * N, n, KGROUPS),
                xrow, xs);
            const float u = warp_dot<K>(
                wbase + (N + n) * row_bytes,
                w_scale_row_ptr(w13_scale, e, 2LL * N, N + n, KGROUPS),
                xrow, xs);
            a = (g / (1.0f + expf(-g))) * u;  // silu(gate) * up
        }
        if (lane == 0) smem_a[n_local] = a;
    }
    __syncthreads();

    // Requant the GROUP intermediates this block just produced.
    const int t = threadIdx.x;
    float v = (t < GROUP) ? smem_a[t] : 0.0f;
    const float amax = block_absmax(fabsf(v), smem_red);
    const float scale = fmaxf(amax / kFp8Max, kMinScale);
    if (t == 0) h_scale[ts * NGROUPS + ng] = scale;
    if (t < GROUP) {
        const float q = fminf(fmaxf(v / scale, -kFp8Max), kFp8Max);
        h[ts * N + (long long)ng * GROUP + t] = __nv_fp8_e4m3(q);
    }
}

// ---------------------------------------------------------------------------------------
// 3. w2 GEMV into the private fp32 staging row (unweighted -- finalize applies the routing
//    weight). grid = (m * TOPK, KGROUPS), block = THREADS. No two blocks share an output
//    element, so this is atomic-free by construction.
// ---------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(THREADS) moe_gemv_down(
    float* __restrict__ staging,             // [m, TOPK, K] == [m*TOPK, K]
    const __nv_fp8_e4m3* __restrict__ h,     // [m*TOPK, N]
    const float* __restrict__ h_scale,       // [m*TOPK, NGROUPS]
    const unsigned char* __restrict__ w2,    // int4-packed or fp8, see header
    const float* __restrict__ w2_scale,
    const int* __restrict__ topk_ids,        // [m, TOPK]
    const int num_experts) {
    const long long ts = blockIdx.x;
    const int kg = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int e = topk_ids[ts];
    const bool expert_ok = (e >= 0) && (e < num_experts);

    const __nv_fp8_e4m3* hrow = h + ts * N;
    const float* hs = h_scale + ts * NGROUPS;
    const long long row_bytes = w_row_bytes(N);
    const unsigned char* wbase = w2 + (long long)e * (long long)K * row_bytes;

#pragma unroll
    for (int i = 0; i < GROUP / WARPS; ++i) {
        const int k_local = i * WARPS + warp;
        const long long k = (long long)kg * GROUP + k_local;
        float acc = 0.0f;
        if (expert_ok) {
            acc = warp_dot<N>(
                wbase + k * row_bytes,
                w_scale_row_ptr(w2_scale, e, K, k, NGROUPS),
                hrow, hs);
        }
        if (lane == 0) staging[ts * K + k] = acc;
    }
}

// ---------------------------------------------------------------------------------------
// 4. Reduce the TOPK staging rows with the routing weights, cast to bf16.
//    grid = (m), block = THREADS.
// ---------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(THREADS) moe_finalize(
    __nv_bfloat16* __restrict__ out,          // [m, K]
    const float* __restrict__ staging,        // [m, TOPK, K]
    const float* __restrict__ topk_weights) { // [m, TOPK]
    const long long tok = blockIdx.x;
    float w[TOPK];
#pragma unroll
    for (int s = 0; s < TOPK; ++s) w[s] = topk_weights[tok * TOPK + s];
    for (int k = threadIdx.x; k < K; k += THREADS) {
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < TOPK; ++s)
            acc += staging[(tok * TOPK + s) * K + k] * w[s];
        out[tok * K + k] = __float2bfloat16(acc);
    }
}

// ---------------------------------------------------------------------------------------
// 5. int4 -> fp8 re-expansion against the ORIGINAL [128,128] block scales, so prefill /
//    big-M batches can run the stock fused_moe kernel after the fp8 originals were freed.
//    dst[row][k] = fp8( q[row][k] * group_scale[row][k/128] / block_scale[row/128][k/128] )
//    i.e. dst * block_scale reproduces the int4-dequantized weight (one extra fp8 rounding).
//    grid = (num_experts * rows_per_expert), block = THREADS. Dims are runtime arguments --
//    this entry serves both w13 (rows=2N, len=K) and w2 (rows=K, len=N), and it is not on
//    the per-step decode path, so compile-time specialization buys nothing here.
// ---------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(THREADS) moe_int4_to_fp8(
    __nv_fp8_e4m3* __restrict__ dst,         // [E, rows_per_expert, rowlen]
    const unsigned int* __restrict__ q,      // [E, rows_per_expert, rowlen/8]
    const float* __restrict__ gscale,        // [E, rows_per_expert, rowlen/GROUP]
    const float* __restrict__ bscale,        // [E, rows_per_expert/128, rowlen/128]
    const int rows_per_expert,
    const int rowlen) {
    const long long row = blockIdx.x;
    const long long e = row / rows_per_expert;
    const long long r = row % rows_per_expert;
    const int words = rowlen / 8;
    const unsigned int* qrow = q + row * words;
    const float* gs = gscale + row * (rowlen / GROUP);
    const float* bs =
        bscale + (e * (rows_per_expert / 128) + r / 128) * (long long)(rowlen / 128);
    __nv_fp8_e4m3* drow = dst + row * rowlen;

    for (int j = threadIdx.x; j < words; j += THREADS) {
        const unsigned int wbits = qrow[j];
        const int k0 = j * 8;
        const float ratio = gs[k0 / GROUP] / bs[k0 / 128];
        Frag8 outv;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int qv = ((int)(wbits << (28 - 4 * i))) >> 28;
            const float f = fminf(fmaxf((float)qv * ratio, -kFp8Max), kFp8Max);
            outv.v[i] = __nv_fp8_e4m3(f);
        }
        // 8 fp8 bytes = one uint2 store; rowlen % 8 == 0 keeps it aligned.
        reinterpret_cast<uint2*>(drow)[j] = outv.b;
    }
}
