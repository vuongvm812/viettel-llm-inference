// Fused greedy argmax over the vocab: [rows, VOCAB] bf16 logits -> [rows] int64 token ids.
//
// WHO CALLS THIS. Two seams, three launches:
//   * the forked V2 sampler (vtl/vllm_patches/v0.25.0/v2_greedy_sampler.patch) resolves
//     ``torch.ops.vllm_cuda.greedy_argmax_i64`` once on the first provably-greedy step and
//     calls it instead of ``logits.argmax(dim=-1)``;
//   * the N-step decode burst (vtl/patches/nstep_decode.py) calls the OUT variant through
//     its ``_ARGMAX`` indirection, so the launches land inside the captured burst graphs.
// Both fall back to ``torch.argmax`` when this kernel did not compile OR when the boot
// parity gate in vtl/patches/greedy_argmax.py caught a disagreement, so nothing here is
// load-bearing for correctness of the boot.
//
// WHAT IT IS WORTH, honestly. torch.argmax is already a single fused multi-CTA reduction at
// the memory roofline; at rows <= 8 over a 248320-wide bf16 row the reduction itself is
// ~2 us of pure bandwidth and nothing here beats that. What this kernel removes is the
// envelope around it: VOCAB is a compile-time constant, the launch is three cuLaunchKernels
// with no TensorIterator build / dtype dispatch / reduction-config heuristic, and the result
// lands DIRECTLY in the caller's persistent int64 buffer (no allocation, no `out=` copy).
// That is a few microseconds of HOST dispatch per step in eager mode and approximately
// nothing inside a replayed CUDA graph -- where the value is instead that the whole token
// pick is capturable at all. Measure with bench/test_greedy_argmax.py's VTL_BENCH=1 arm
// before believing any number in this comment.
//
// TWO TOLERATED NON-GOALS, stated so nobody "fixes" them:
//   1. BEATING torch on raw bandwidth. The split-row grid below exists to MATCH torch's
//      multi-CTA occupancy (one block per row used <= 8 of 132 SMs and was ~5x slower);
//      the remaining edge is the launch envelope, not the arithmetic.
//   2. Serving anything outside the compiled envelope -- another vocab, fp16/fp32, a
//      non-contiguous or misaligned view, a batch wider than the persistent workspace.
//      Every one of those is refused on the PYTHON side and degrades to torch.argmax; this
//      translation unit assumes its envelope holds.
//
// Same three NVRTC rules as vtl/kernels/rms_norm_quant.cu (the template): no torch headers,
// no host code, shapes as -D macros.
//
// SEMANTICS ARE torch.argmax's, EXACTLY. A one-shot BOOT PARITY GATE in
// vtl/patches/greedy_argmax.py bit-compares this kernel's output against torch.argmax on
// adversarial rows at the real VOCAB before either op is registered, and
// bench/test_greedy_argmax.py holds the same contract in CI. (nstep_decode's
// ``_graph_matches_eager`` does NOT check this: once ``_ARGMAX`` is rebound it compares
// fused-against-fused, so it is a pointer-staleness smoke test for graph capture and
// nothing more.) The contract:
//
//   * NaN is the MAXIMUM. Any NaN, of any payload and either sign, outranks every finite
//     value and +inf.
//   * TIES GO TO THE LOWEST INDEX. Exact-equal maxima, several -inf in a fully masked row,
//     several NaN, and a -0.0/+0.0 pair (IEEE says they are EQUAL) all resolve to the first
//     occurrence.
//   * The comparison happens in fp32 on the widened bf16. bf16 -> fp32 is exact, so this
//     is the same order relation as comparing the bf16s.
//
// THE PACKED KEY, and why max-by-key IS torch's winner. torch's reduction combines with
// ``GreaterOrNan(a, b, ia, ib)``: NaN a beats non-NaN b; NaN vs NaN and a == b both go to
// the lower index; otherwise a > b. We instead map each (value, index) to ONE unsigned
// 64-bit key and take the plain integer maximum, which is trivially commutative and
// associative -- that is what makes a split-row atomicMax tree legal at all. The map:
//
//     hi = key_of(v)   = 0xffffffff                              if v is NaN
//                      = bits | 0x80000000                       if v >= +0.0 (sign clear)
//                      = ~bits                                   if v <  0.0  (sign set)
//                        where bits = __float_as_uint(v + 0.0f)
//     lo = ~(unsigned)index
//     key = ((u64)hi << 32) | lo
//
//   (a) hi is MONOTONE on the non-NaN floats. The standard trick: IEEE-754 binary32
//       compares like a sign-magnitude integer, so flipping the sign bit of the
//       non-negatives and inverting the negatives yields an unsigned order identical to
//       the float order. Range check: non-negative bits run 0x00000000..0x7f800000 (+inf)
//       -> hi 0x80000000..0xff800000; negative bits run 0x80000001..0xff800000 (-inf)
//       -> hi 0x7fffffff..0x007fffff. Disjoint, and every negative sits below every
//       non-negative.
//   (b) NaN GETS ONE KEY, 0xffffffff, strictly above the largest non-NaN hi (0xff800000,
//       which is +inf). One key for all payloads matters twice: a negative NaN
//       (bits >= 0xff800001) would otherwise map BELOW -inf, and two different NaN
//       payloads would otherwise order by payload instead of tying. With a single key,
//       NaN-vs-NaN is a tie in hi and therefore resolved by lo -- i.e. by index -- which
//       is exactly torch's ``idx_a < idx_b``.
//   (c) -0.0 IS CANONICALIZED. IEEE says -0.0 == +0.0, so torch calls it a tie and takes
//       the lower index; the raw bits (0x80000000 vs 0x00000000) do not. ``v + 0.0f`` is
//       exactly the fix: under round-to-nearest -0.0f + 0.0f == +0.0f and x + 0.0f == x
//       for every other finite x, subnormals included. (This is why NVRTC must not be
//       given --use_fast_math, which would license dropping the add. vtl/nvrtc.py's
//       BASE_FLAGS omit it, and its self-check asserts so.)
//   (d) lo = ~index makes a HIGHER key mean a LOWER index. So when hi ties -- equal
//       values, both-NaN, or the -0.0/+0.0 pair from (c) -- the maximum key is the
//       smallest index, which is torch's tie-break in all three cases.
//   (e) Therefore key(a) > key(b) <=> GreaterOrNan(a, b, ia, ib) for every pair with
//       ia != ib, and max-by-key over the row is torch's argmax. QED.
//
// THE IDENTITY IS 0, which is what lets the cross-block combine be a bare atomicMax into a
// zeroed workspace. Every REAL key is strictly positive: the smallest hi any real element
// can produce is -inf's 0x007fffff (see (a)), so every real key is at least
// 0x007fffff_00000000 > 0. A slice that read nothing emits 0 and cannot win.
//
// SHAPE OF THE COMPUTATION. Three launches on ONE stream, all stream-capture-legal:
//   1. greedy_argmax_reset     ws[0..rows) = 0
//   2. greedy_argmax_partial   grid (SPLIT, rows): each block scans VOCAB/SPLIT of one row
//                              with 16-byte vector loads, reduces to a single packed key,
//                              and does ONE atomicMax into ws[row]
//   3. greedy_argmax_finalize  out[row] = ~(unsigned)(ws[row] & 0xffffffff)
// SPLIT exists because one block per row is one CTA per row: at the served batch of <= 8
// that is 8 of the device's 132 SMs and the row scan is latency-bound at ~10 us. SPLIT=16
// puts 128 CTAs on the same work.
//
// Required defines (vtl/patches/greedy_argmax.py supplies them from the loaded config):
//   VOCAB     vocabulary size, i.e. logits.shape[1]. No default: guessing it would index
//             off the end of the row. 248320 for the shipped Qwen3.5 checkpoint.
//   THREADS   partial-kernel block width. Default 256; must be a multiple of 32.
//   SPLIT     row slices per row, i.e. gridDim.x of the partial kernel. Default 16.
//
// LAYOUT ASSUMPTION: ``logits`` is contiguous [rows, VOCAB], and 16-byte aligned WHEN the
// vector path is compiled (VOCAB % 8 == 0). The Python side re-checks both on every call
// and falls back to torch.argmax if either fails, because a misaligned 16-byte load is a
// fault, not a slow path. The scalar specialization needs only bf16's own 2-byte alignment,
// which every torch tensor has by construction, so the Python check is gated on the vector
// path actually being the one compiled.

#include <cuda_bf16.h>

#ifndef VOCAB
#error "NVRTC: -DVOCAB=<vocab_size> is required"
#endif
#ifndef THREADS
#define THREADS 256
#endif
#ifndef SPLIT
#define SPLIT 16
#endif

#define VEC 8                  // 16-byte loads: 8 x bf16
#define NVEC (VOCAB / VEC)     // whole vectors per row
#define VEC_PATH ((VOCAB % VEC) == 0)

static_assert(VOCAB > 0, "VOCAB must be positive");
static_assert(VOCAB < 0x7fffffff, "VOCAB must leave kNoIndex free as a sentinel");
static_assert(THREADS % 32 == 0, "THREADS must be a whole number of warps");
static_assert(THREADS >= 32 && THREADS <= 1024, "THREADS must be a legal block width");
static_assert(SPLIT >= 1 && SPLIT <= 1024, "SPLIT must be a legal gridDim.x");

// One 16-byte load, same shape as rms_norm_quant.cu's.
struct __align__(16) Vec8 { __nv_bfloat16 v[VEC]; };

// "This accumulator has not seen an element yet." Larger than every legal index (the
// static_assert above), so it also loses every tie if it ever reached the key map -- but it
// never does: pack_key() turns it into the reduction identity instead.
__device__ constexpr int kNoIndex = 0x7fffffff;

// Written as bit patterns because NVRTC's INFINITY comes from a header this translation
// unit deliberately does not pull in.
__device__ __forceinline__ float neg_inf() { return __int_as_float((int)0xff800000); }

// `v != v` rather than isnan(): no <cmath> here, and this compiles to a single set.p
// without depending on which isnan overload NVRTC's -default-device mode picks.
__device__ __forceinline__ bool is_nan(float v) { return v != v; }

// ---------------------------------------------------------------------------------------
// the packed key -- see the proof in the header comment
// ---------------------------------------------------------------------------------------

__device__ __forceinline__ unsigned key_of(float v) {
    if (is_nan(v)) return 0xffffffffu;   // ONE key for every payload and either sign
    v = v + 0.0f;                        // -0.0f + 0.0f == +0.0f; a no-op for everything else
    const unsigned bits = __float_as_uint(v);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ unsigned long long pack_key(float v, int i) {
    if (i == kNoIndex) return 0ull;      // scanned nothing -> the reduction identity
    return ((unsigned long long)key_of(v) << 32) | (unsigned long long)(~(unsigned)i);
}

// ---------------------------------------------------------------------------------------
// the per-thread scan
// ---------------------------------------------------------------------------------------

// One candidate against one incumbent, for a scan whose indices STRICTLY INCREASE.
//
// That monotonicity is the whole reason this is three instructions instead of the full
// pairwise predicate: every tie torch resolves by "lower index" is a tie the incumbent
// already holds, so neither the equal-value case nor the NaN-vs-NaN case needs an index
// compare here. Only the "NaN outranks a non-NaN" clause survives. The full key compare is
// used for every merge that is NOT index-monotone -- across the four accumulators, across
// lanes, across warps, across blocks.
//
// The incumbent starts at (-inf, kNoIndex), and -inf does NOT beat -inf under this
// predicate, so an accumulator whose whole slice is -inf ends still holding kNoIndex. That
// is repaired once per accumulator after the loop (see the head fixup), not per element.
__device__ __forceinline__ void scan_step(float& bv, int& bi, float v, int i) {
    if (v > bv || (is_nan(v) && !is_nan(bv))) { bv = v; bi = i; }
}

// Warp-level tree over PACKED keys. Literal 32/16: `warpSize` is a runtime value, which
// would keep the loop rolled and the shuffle offsets out of the instruction.
__device__ __forceinline__ unsigned long long warp_max(unsigned long long k) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const unsigned long long o = __shfl_down_sync(0xffffffffu, k, off);
        if (o > k) k = o;
    }
    return k;
}

// ---------------------------------------------------------------------------------------
// 1. reset: the workspace is the atomicMax accumulator, so it starts at the identity
// ---------------------------------------------------------------------------------------

// Strided on blockDim so ANY launch width serves; one block, because `rows` is <= a few
// dozen. Separate from the partial kernel because zeroing inside it would race the other
// slices' atomics -- blocks in one grid have no ordering.
extern "C" __global__ void greedy_argmax_reset(
    unsigned long long* __restrict__ ws,   // [>= rows]
    const int rows) {
    for (int r = (int)threadIdx.x; r < rows; r += (int)blockDim.x) ws[r] = 0ull;
}

// ---------------------------------------------------------------------------------------
// 2. partial: one block per (row slice, row); one atomicMax per block
// ---------------------------------------------------------------------------------------

extern "C" __global__ void __launch_bounds__(THREADS) greedy_argmax_partial(
    const __nv_bfloat16* __restrict__ logits,   // [rows, VOCAB], contiguous
    unsigned long long* __restrict__ ws,        // [>= rows], zeroed by greedy_argmax_reset
    const int rows) {
    constexpr int kWarps = (THREADS + 31) / 32;
    __shared__ unsigned long long smem[kWarps];

    const int row = (int)blockIdx.y;
    if (row >= rows) return;   // lets a caller launch a fixed grid for a smaller batch
    const int slice = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;

    // Seeded with the identity BEFORE any warp leader writes, so a launch with
    // blockDim < THREADS (which leaves the high warps' slots unwritten) reads the identity
    // rather than whatever the shared allocation happened to contain.
    if (tid < kWarps) smem[tid] = 0ull;
    __syncthreads();

    const long long base = (long long)row * (long long)VOCAB;
    unsigned long long key;

#if VEC_PATH
    // VOCAB % 8 == 0, so EVERY row base is 16-byte aligned and the whole row is vectors.
    const Vec8* __restrict__ rowv = reinterpret_cast<const Vec8*>(logits + base);
    // Slice bounds by exact division of the VECTOR count, so the slices tile [0, NVEC)
    // with no gap and no overlap for any (NVEC, SPLIT) -- including NVEC < SPLIT, where the
    // low slices are simply empty. long long only to keep NVEC * SPLIT off the int edge;
    // both operands are compile-time constants, so this folds.
    const int vbeg = (int)(((long long)NVEC * slice) / SPLIT);
    const int vend = (int)(((long long)NVEC * (slice + 1)) / SPLIT);

    // FOUR independent accumulators, one per (element mod 4) lane of the Vec8. Each sees a
    // strictly increasing index sequence -- accumulator k takes elements k and k+4 of every
    // vector, and the vectors are walked in order -- so scan_step's monotonicity argument
    // holds per accumulator. Four chains instead of one is what breaks the serial
    // dependency between consecutive compares; they are merged by KEY below, where the
    // indices are no longer monotone.
    float bv0 = neg_inf(), bv1 = neg_inf(), bv2 = neg_inf(), bv3 = neg_inf();
    int bi0 = kNoIndex, bi1 = kNoIndex, bi2 = kNoIndex, bi3 = kNoIndex;

    for (int v = vbeg + tid; v < vend; v += THREADS) {
        const Vec8 x = rowv[v];
        const int i0 = v * VEC;
        scan_step(bv0, bi0, __bfloat162float(x.v[0]), i0 + 0);
        scan_step(bv1, bi1, __bfloat162float(x.v[1]), i0 + 1);
        scan_step(bv2, bi2, __bfloat162float(x.v[2]), i0 + 2);
        scan_step(bv3, bi3, __bfloat162float(x.v[3]), i0 + 3);
        scan_step(bv0, bi0, __bfloat162float(x.v[4]), i0 + 4);
        scan_step(bv1, bi1, __bfloat162float(x.v[5]), i0 + 5);
        scan_step(bv2, bi2, __bfloat162float(x.v[6]), i0 + 6);
        scan_step(bv3, bi3, __bfloat162float(x.v[7]), i0 + 7);
    }

    // THE -inf FIXUP. An accumulator still holding kNoIndex after a NON-EMPTY scan saw only
    // -inf (anything finite, +inf or NaN would have beaten the -inf identity), so its true
    // best is (-inf, its FIRST index) -- which is the head of its strided range. One
    // predicated move per accumulator, off the per-element path. An accumulator whose range
    // was empty keeps kNoIndex and pack_key turns it into the identity.
    const int head = vbeg + tid;
    if (head < vend) {
        const int h0 = head * VEC;
        if (bi0 == kNoIndex) bi0 = h0 + 0;
        if (bi1 == kNoIndex) bi1 = h0 + 1;
        if (bi2 == kNoIndex) bi2 = h0 + 2;
        if (bi3 == kNoIndex) bi3 = h0 + 3;
    }

    key = pack_key(bv0, bi0);
    const unsigned long long k1 = pack_key(bv1, bi1);
    const unsigned long long k2 = pack_key(bv2, bi2);
    const unsigned long long k3 = pack_key(bv3, bi3);
    if (k1 > key) key = k1;
    if (k2 > key) key = k2;
    if (k3 > key) key = k3;
#else
    // Guarded scalar path for a VOCAB the vector path cannot serve. It is not just the
    // leftover elements: with VOCAB % 8 != 0 the row base itself is only 2-byte aligned on
    // odd rows, so there is no aligned bulk to split off. Scalar for the whole slice is the
    // correct answer, and it is still one pass -- other checkpoints work, they just do not
    // get the 16-byte loads. One accumulator, because this path is not the one that ships.
    const __nv_bfloat16* __restrict__ row_p = logits + base;
    const int ibeg = (int)(((long long)VOCAB * slice) / SPLIT);
    const int iend = (int)(((long long)VOCAB * (slice + 1)) / SPLIT);

    float bv0 = neg_inf();
    int bi0 = kNoIndex;
    for (int i = ibeg + tid; i < iend; i += THREADS) {
        scan_step(bv0, bi0, __bfloat162float(row_p[i]), i);
    }
    if (bi0 == kNoIndex && ibeg + tid < iend) bi0 = ibeg + tid;   // the same -inf fixup
    key = pack_key(bv0, bi0);
#endif

    key = warp_max(key);

    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) smem[warp] = key;
    __syncthreads();

    // Full 5-step warp tree over kWarps entries padded with the identity, rather than a
    // log2(kWarps) tree: kWarps is a power of two for every THREADS this ships with, but a
    // partial tree would silently drop the last warp for any THREADS that is not, and the
    // extra shuffles cost nothing next to the row scan.
    if (warp == 0) {
        key = (tid < kWarps) ? smem[tid] : 0ull;
        key = warp_max(key);
        // ONE atomic per block. Commutative and associative by construction (integer max),
        // so the order the SPLIT blocks land in does not change the answer -- which is what
        // makes the split legal under a replayed graph as much as under an eager launch.
        if (tid == 0) atomicMax(&ws[row], key);
    }
}

// ---------------------------------------------------------------------------------------
// 3. finalize: unpack the winning index
// ---------------------------------------------------------------------------------------

// ws[row] is never still 0 for a row in range: the SPLIT slices TILE [0, NVEC) (or
// [0, VOCAB) on the scalar path), so at least one block reads at least one element and
// every real key is > 0. Strided on blockDim like the reset, one block.
extern "C" __global__ void greedy_argmax_finalize(
    const unsigned long long* __restrict__ ws,   // [>= rows]
    long long* __restrict__ out,                 // [rows]
    const int rows) {
    for (int r = (int)threadIdx.x; r < rows; r += (int)blockDim.x) {
        out[r] = (long long)(~(unsigned)(ws[r] & 0xffffffffull));
    }
}
