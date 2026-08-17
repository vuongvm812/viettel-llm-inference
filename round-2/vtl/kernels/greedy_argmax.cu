// Fused greedy argmax over the vocab: [rows, VOCAB] bf16 logits -> [rows] int64 token ids.
//
// WHO CALLS THIS. Two seams, one kernel:
//   * the forked V2 sampler (vtl/vllm_patches/v0.25.0/v2_greedy_sampler.patch) resolves
//     ``torch.ops.vllm_cuda.greedy_argmax_i64`` once on the first provably-greedy step and
//     calls it instead of ``logits.argmax(dim=-1)``;
//   * the N-step decode burst (vtl/patches/nstep_decode.py) calls the OUT variant through
//     its ``_ARGMAX`` indirection, so the launch lands inside the captured burst graphs.
// Both fall back to ``torch.argmax`` when this kernel did not compile, so nothing here is
// load-bearing for correctness of the boot -- only for the ~40 us/step it saves.
//
// Same three NVRTC rules as vtl/kernels/rms_norm_quant.cu (the template): no torch headers,
// no host code, shapes as -D macros. The edge over ``torch.argmax`` is not the reduction --
// torch's is already one fused pass -- it is that VOCAB is a compile-time constant, so the
// row loop's trip count, the vector count and every offset fold to immediates, the launch
// is one cuLaunchKernel with no TensorIterator setup, and the result lands DIRECTLY in the
// caller's persistent int64 buffer (no allocation, no `out=` copy).
//
// SEMANTICS ARE torch.argmax's, EXACTLY. This kernel's output is bit-compared against
// torch's at boot -- nstep_decode's `_graph_matches_eager` demotes the whole burst on a
// single differing token -- so "close enough" is a demotion, not a rounding error:
//
//   * NaN is the MAXIMUM. Any NaN outranks every finite value and +inf.
//   * TIES GO TO THE LOWEST INDEX. Exact-equal maxima, several -inf in a fully masked row,
//     and several NaN all resolve to the first occurrence.
//   * The comparison happens in fp32 on the widened bf16. bf16 -> fp32 is exact, so this
//     is the same order relation as comparing the bf16s.
// The reduction is therefore over (value, index) PAIRS with a combine that is commutative
// and associative under those rules -- which is what makes the warp/block tree order-free.
//
// Required defines (vtl/patches/greedy_argmax.py supplies them from the loaded config):
//   VOCAB     vocabulary size, i.e. logits.shape[1]. No default: guessing it would index
//             off the end of the row. 248320 for the shipped Qwen3.5 checkpoint.
//   THREADS   block width, one block per row. Default 512; must be a multiple of 32.
//
// LAYOUT ASSUMPTION: ``logits`` is contiguous [rows, VOCAB] and 16-byte aligned. The Python
// side re-checks both on every call and falls back to torch.argmax if either fails, because
// a misaligned 16-byte load is a fault, not a slow path.

#include <cuda_bf16.h>

#ifndef VOCAB
#error "NVRTC: -DVOCAB=<vocab_size> is required"
#endif
#ifndef THREADS
#define THREADS 512
#endif

#define VEC 8                  // 16-byte loads: 8 x bf16
#define NVEC (VOCAB / VEC)     // whole vectors per row
#define VEC_PATH ((VOCAB % VEC) == 0)

static_assert(VOCAB > 0, "VOCAB must be positive");
static_assert(THREADS % 32 == 0, "THREADS must be a whole number of warps");
static_assert(THREADS >= 32 && THREADS <= 1024, "THREADS must be a legal block width");

// One 16-byte load, same shape as rms_norm_quant.cu's.
struct __align__(16) Vec8 { __nv_bfloat16 v[VEC]; };

// Identity of the reduction: strictly worse than every real candidate. -inf loses on value
// to anything finite, ties with a real -inf and then loses on index (any real index is
// below 0x7fffffff), and loses to NaN. Written as bit patterns because NVRTC's INFINITY /
// INT_MAX come from headers this translation unit deliberately does not pull in.
__device__ constexpr int kNoIndex = 0x7fffffff;

__device__ __forceinline__ float neg_inf() { return __int_as_float((int)0xff800000); }

// `v != v` rather than isnan(): no <cmath> here, and this compiles to a single set.p
// without depending on which isnan overload NVRTC's -default-device mode picks.
__device__ __forceinline__ bool is_nan(float v) { return v != v; }

// Does candidate (v, i) BEAT incumbent (bv, bi)? This one predicate is the whole
// torch.argmax contract; every reduction step below is just this applied pairwise.
__device__ __forceinline__ bool beats(float v, int i, float bv, int bi) {
    const bool vn = is_nan(v);
    const bool bn = is_nan(bv);
    if (vn != bn) return vn;      // NaN is the maximum
    if (vn) return i < bi;        // NaN vs NaN: a tie, so the lower index holds
    if (v != bv) return v > bv;   // strictly greater wins
    return i < bi;                // exact tie: the lower index holds
}

// Warp-level tree over (value, index) pairs. Literal 32/16: `warpSize` is a runtime value,
// which would keep the loop rolled and the shuffle offsets out of the instruction.
__device__ __forceinline__ void warp_reduce(float& bv, int& bi) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, off);
        const int oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (beats(ov, oi, bv, bi)) { bv = ov; bi = oi; }
    }
}

// One row per block. Lanes stride the row so the loads coalesce, carry one (value, index)
// pair in registers for the whole row, and only then reduce -- the row is read exactly once.
extern "C" __global__ void __launch_bounds__(THREADS) greedy_argmax_i64(
    const __nv_bfloat16* __restrict__ logits,   // [rows, VOCAB], contiguous
    long long* __restrict__ out,                // [rows]
    const int rows) {
    constexpr int kWarps = (THREADS + 31) / 32;
    __shared__ float smem_v[kWarps];
    __shared__ int smem_i[kWarps];

    const int row = blockIdx.x;
    if (row >= rows) return;   // lets a caller launch a fixed grid for a smaller batch

    const long long base = (long long)row * (long long)VOCAB;
    const int tid = threadIdx.x;

    float bv = neg_inf();
    int bi = kNoIndex;

#if VEC_PATH
    // VOCAB % 8 == 0, so EVERY row base is 16-byte aligned and the whole row is vectors.
    const Vec8* __restrict__ rowv = reinterpret_cast<const Vec8*>(logits + base);
    for (int v = tid; v < NVEC; v += THREADS) {
        const Vec8 x = rowv[v];
        const int i0 = v * VEC;
#pragma unroll
        for (int k = 0; k < VEC; ++k) {
            const float val = __bfloat162float(x.v[k]);
            if (beats(val, i0 + k, bv, bi)) { bv = val; bi = i0 + k; }
        }
    }
#else
    // Guarded tail path for a VOCAB the vector path cannot serve. It is not just the
    // leftover elements: with VOCAB % 8 != 0 the row base itself is only 2-byte aligned on
    // odd rows, so there is no aligned bulk to split off. Scalar for the whole row is the
    // correct answer, and it is still one pass -- other checkpoints work, they just do not
    // get the 16-byte loads.
    const __nv_bfloat16* __restrict__ row_p = logits + base;
    for (int i = tid; i < VOCAB; i += THREADS) {
        const float val = __bfloat162float(row_p[i]);
        if (beats(val, i, bv, bi)) { bv = val; bi = i; }
    }
#endif

    warp_reduce(bv, bi);

    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) { smem_v[warp] = bv; smem_i[warp] = bi; }
    __syncthreads();

    // Full 5-step warp tree over kWarps entries padded with the identity, rather than a
    // log2(kWarps) tree: kWarps is a power of two for every THREADS this ships with, but a
    // partial tree would silently drop the last warp for any THREADS that is not, and the
    // extra shuffles cost nothing next to the row scan.
    if (warp == 0) {
        bv = (tid < kWarps) ? smem_v[tid] : neg_inf();
        bi = (tid < kWarps) ? smem_i[tid] : kNoIndex;
        warp_reduce(bv, bi);
        if (tid == 0) out[row] = (long long)bi;
    }
}
