// Fused GDN decode step, NVRTC-native. ONE launch per layer replacing the pair the
// non-spec decode path runs today (vLLM v0.25.0,
// qwen_gdn_linear_attn._forward_core_decode_non_spec):
//
//   1. causal_conv1d_update            (Triton, mamba/ops/causal_conv1d.py)
//        depthwise conv window CONV=4 over the mixed [q|k|v] projection, silu,
//        conv-state rotation [s0,s1,s2] -> [s1,s2,x]
//   2. fused_recurrent_gated_delta_rule_packed_decode
//        (Triton, fla/ops/fused_recurrent.py:256-479)
//        split q/k/v from the conv output, l2norm q/k IN KERNEL (the call site passes
//        use_qk_l2norm_in_kernel=True), sigmoid-gated delta-rule recurrence updating the
//        fp32 ssm state in place, bf16 output write.
//
// SEMANTICS ARE THE TRITON PAIR'S, REPLICATED OP FOR OP (single token per sequence,
// non-spec, KERNEL_WIDTH=4, seqlen=1, state_len=3):
//
//   conv (per channel c):                            [Triton rounding replicated exactly]
//     p_j   = bf16(tap_j * w_j)          j=0..3, taps = [s0,s1,s2,x]   (bf16*bf16 -> bf16)
//     acc   = ((((bias + p0) + p1) + p2) + p3)                         (fp32 adds, in order)
//     yc    = bf16( acc / (1 + expf(-acc)) )                           (silu fp32, then the
//                                                                       bf16 store both
//                                                                       kernels round-trip)
//     state <- [s1, s2, x]               (x stored RAW, not the conv output)
//   recurrence (per value head hv of key head h = hv / (HV/HK)):
//     q = yc[h*DK .. ]  k = yc[HK*DK + h*DK ..]  v = yc[2*HK*DK + hv*DV ..]   (all f32(bf16))
//     q = q / sqrtf(sum(q^2) + 1e-6);  k likewise;  q = q * scale
//     x    = a[hv] + dt_bias[hv]
//     sp   = x <= 20 ? logf(1 + expf(x)) : x                           (NOT log1pf -- match)
//     g    = -expf(A_log[hv]) * sp
//     beta = f32(bf16(1 / (1 + expf(-b[hv]))))     <- sigmoid ROUNDED THROUGH BF16: the
//                                                     Triton kernel does
//                                                     sigmoid(b).to(b.dtype).to(f32)
//     h    = h * expf(g)
//     v    = (v - sum_k(h[v,k] * k[k])) * beta
//     h   += v * k          (outer product)
//     o    = bf16( sum_k(h[v,k] * q[k]) )
//   state_idx <= 0 (NULL_BLOCK_ID): write zero output rows, touch nothing else --
//   exactly the recurrent kernel's early-out (the conv kernel also skips on 0).
//
// GRID IS (HK, num_tokens), ONE KEY HEAD PER BLOCK -- not one value head. This is a
// correctness constraint, not a tuning choice: the q/k conv channels and their rotating
// conv state are SHARED by the HV/HK sibling value heads. With one block per value head,
// sibling blocks would race the owner's state rotation (reads of s0..s2 against shifted
// writes) with no inter-block ordering. One block per key head makes every conv channel's
// read-modify-write private to a single block, like the stock conv kernel's one-program-
// per-channel layout. The block then runs the recurrence for its HV/HK value heads
// back to back (they share the l2normed q/k anyway).
//
// FP32 STATE MATH IS PINNED WITH __fmul_rn/__fadd_rn: Triton emits separate mul + add
// (mul feeds a tl.sum tree or an elementwise add), never an FMA, and nvcc's default
// contraction would silently fuse ours. No fast-math anywhere. TWO divergences are
// tolerated, and only two:
//   1. fp32 REDUCTION ORDER (the l2norm sums and the two length-DK dots). Triton
//      tree-reduces 128 lanes; we do a per-lane 4-chain + a 32-lane shuffle tree.
//   2. TRANSCENDENTAL LOWERING. expf/logf/sqrtf here vs whatever tl.exp/tl.log/tl.sqrt
//      lower to for the installed Triton (libdevice __nv_expf, or an ex2.approx
//      expansion). Both are <= ~2 ulp fp32. NOTE: FLA_USE_FAST_OPS=1 swaps Triton onto
//      tldevice.fast_expf/fast_logf (fla/ops/op.py) -- still inside the same bound, but
//      the parity test asserts against whatever the installed stack actually runs.
// Both move results by O(1 ulp fp32); bench/test_gdn_decode_step.py bounds it against the
// real Triton pair (state fp32: rtol 1e-4 / atol 1e-5; out bf16: rtol/atol 1e-2 ~ 2 bf16
// ulp). Everything else -- the bf16 narrowings, the softplus branch, the beta round-trip,
// the true-vs-fused multiply-add order -- is replicated exactly.
//
// One global write the pair does that we DELIBERATELY SKIP: causal_conv1d_update writes
// the conv output back over mixed_qkv in place. Nothing reads it afterwards on this path
// (it feeds only kernel 2, which we fused), so the fast path leaves mixed_qkv untouched.
//
// Required defines (vtl/patches/gdn_decode_step.py supplies them from the loaded layer):
//   DK=128 DV=128 HK=16 HV=64 CONV=4 THREADS=128
// Block: (THREADS, 1, 1). The fp32 [DV, DK] state tiles (64 KB each, HV/HK of them per
// block) stream through registers one row per warp iteration, 16B (float4) accesses;
// q/k/v staged in shared memory. H200: HK*T blocks (96 at trace-peak decode concurrency
// ~6) -- HBM-bound on the 4 MB/seq state, which is read+written exactly once.

#include <cuda_bf16.h>

#ifndef DK
#error "NVRTC: -DDK=<head_k_dim> is required"
#endif
#ifndef DV
#error "NVRTC: -DDV=<head_v_dim> is required"
#endif
#ifndef HK
#error "NVRTC: -DHK=<num_k_heads/tp> is required"
#endif
#ifndef HV
#error "NVRTC: -DHV=<num_v_heads/tp> is required"
#endif
#ifndef CONV
#define CONV 4
#endif
#ifndef THREADS
#define THREADS 128
#endif

#define QK_DIM (HK * DK)                    // 2048
#define CONV_DIM (2 * QK_DIM + HV * DV)     // 12288: [q | k | v]
#define GPB (HV / HK)                       // value heads per block (= per key head)
#define WARPS (THREADS / 32)
#define ROWS_PER_WARP (DV / WARPS)

static_assert(CONV == 4, "the conv tap sequence below is written for width 4");
static_assert(THREADS == DK, "one thread per q/k channel: THREADS must equal DK");
static_assert(THREADS == DV, "one thread per v channel per head slice");
static_assert(THREADS % 32 == 0, "whole warps");
static_assert(DV % WARPS == 0, "state rows must split evenly across warps");
// Phase 3 gives every lane of a warp exactly four k/q channels (`s_k[lane*4 + 0..3]`) and
// closes the dot with a full-warp xor ladder, so the length-DK dot is 32 lanes x 4. DK=128
// is therefore a hard constraint of this port, not just an alignment one -- at DK=256 the
// ladder would silently reduce only the first 128 channels.
static_assert(DK == 128, "phase 3 assumes DK == 32 lanes x float4; port it before changing");
static_assert(DK % 4 == 0, "float4 state rows");
static_assert(HV % HK == 0, "grouped value heads");
static_assert(GPB <= THREADS, "gating computed by the first GPB threads");

__device__ constexpr float kL2Eps = 1e-6f;              // fused_recurrent l2norm epsilon
__device__ constexpr float kSoftplusThreshold = 20.0f;  // SOFTPLUS_THRESHOLD in the wrapper

// One channel of the depthwise conv update, Triton rounding replicated (see header).
// Returns the fp32 value of the bf16-rounded conv output and rotates the channel's state.
__device__ __forceinline__ float conv1d_channel(
    const __nv_bfloat16* __restrict__ mixed_qkv, long long qkv_base,
    __nv_bfloat16* __restrict__ conv_state, long long cs_base,
    long long cs_dim_stride, long long cs_tok_stride,
    const __nv_bfloat16* __restrict__ conv_w, long long w_dim_stride,
    long long w_width_stride,
    const __nv_bfloat16* __restrict__ conv_bias,
    int c) {
    const long long sbase = cs_base + (long long)c * cs_dim_stride;
    const __nv_bfloat16 s0 = conv_state[sbase];
    const __nv_bfloat16 s1 = conv_state[sbase + cs_tok_stride];
    const __nv_bfloat16 s2 = conv_state[sbase + 2 * cs_tok_stride];
    const __nv_bfloat16 x = mixed_qkv[qkv_base + c];

    const long long wbase = (long long)c * w_dim_stride;
    const __nv_bfloat16 w0 = conv_w[wbase];
    const __nv_bfloat16 w1 = conv_w[wbase + w_width_stride];
    const __nv_bfloat16 w2 = conv_w[wbase + 2 * w_width_stride];
    const __nv_bfloat16 w3 = conv_w[wbase + 3 * w_width_stride];

    // acc starts at f32(bias) (or 0), then four IN-ORDER fp32 adds of bf16-rounded
    // products -- `acc += matrix_x * matrix_w` in Triton multiplies at bf16.
    float acc = (conv_bias != nullptr) ? __bfloat162float(conv_bias[c]) : 0.0f;
    acc = __fadd_rn(acc, __bfloat162float(__hmul(s0, w0)));
    acc = __fadd_rn(acc, __bfloat162float(__hmul(s1, w1)));
    acc = __fadd_rn(acc, __bfloat162float(__hmul(s2, w2)));
    acc = __fadd_rn(acc, __bfloat162float(__hmul(x, w3)));

    // silu in fp32, then the bf16 round-trip both stock kernels pay through mixed_qkv.
    acc = acc / (1.0f + expf(-acc));
    const float y = __bfloat162float(__float2bfloat16(acc));

    // rotate: [s0,s1,s2] -> [s1,s2,x]; x is stored RAW, as stock does.
    conv_state[sbase] = s1;
    conv_state[sbase + cs_tok_stride] = s2;
    conv_state[sbase + 2 * cs_tok_stride] = x;
    return y;
}

// Block-wide fp32 sum over THREADS values (l2norm). Order differs from Triton's
// 128-lane tree by construction; see the header for the tolerance story.
__device__ __forceinline__ float block_sum(float val, float* smem) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val = __fadd_rn(val, __shfl_down_sync(0xffffffffu, val, off));
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) smem[warp] = val;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = smem[0];
#pragma unroll
        for (int w = 1; w < WARPS; ++w) s = __fadd_rn(s, smem[w]);
        smem[0] = s;
    }
    __syncthreads();
    return smem[0];
}

extern "C" __global__ void __launch_bounds__(THREADS) gdn_decode_step(
    const __nv_bfloat16* __restrict__ mixed_qkv,   // [T, CONV_DIM], row stride qkv_stride
    __nv_bfloat16* __restrict__ conv_state,        // strides (seq, dim, tok) -- handles DS & SD
    const __nv_bfloat16* __restrict__ conv_w,      // [CONV_DIM, CONV] viewed, strides given
    const __nv_bfloat16* __restrict__ conv_bias,   // [CONV_DIM] or null
    const __nv_bfloat16* __restrict__ a,           // [T, HV], row stride a_stride
    const __nv_bfloat16* __restrict__ b,           // [T, HV], row stride b_stride
    const float* __restrict__ A_log,               // [HV] fp32 (wrapper upcasts if needed)
    const float* __restrict__ dt_bias,             // [HV] fp32 (wrapper upcasts if needed)
    float* __restrict__ ssm_state,                 // [lines, HV, DV, DK], token stride given
    __nv_bfloat16* __restrict__ out,               // [T, 1, HV, DV] contiguous
    const int* __restrict__ state_indices,         // [T] int32, stride indices_stride
    const float scale,                             // head_k_dim ** -0.5
    const long long qkv_stride,
    const long long a_stride,
    const long long b_stride,
    const long long cs_seq_stride,
    const long long cs_dim_stride,
    const long long cs_tok_stride,
    const long long w_dim_stride,
    const long long w_width_stride,
    const long long state_tok_stride,
    // NOT 1 in general: in the eager/piecewise path the metadata builder hands both stock
    // kernels `block_table_tensor[:, 0]`, whose stride is the block table's ROW stride.
    // (Only the full-cudagraph path copies it into a contiguous buffer.) Both Triton
    // kernels take this stride as an argument for exactly this reason.
    const long long indices_stride) {
    const int i_h = blockIdx.x;        // key head
    const long long n = blockIdx.y;    // token (== sequence: non-spec decode, 1 tok/seq)
    const int tid = threadIdx.x;

    __shared__ float s_q[DK];
    __shared__ float s_k[DK];
    __shared__ float s_v[GPB * DV];
    __shared__ float s_red[WARPS];
    __shared__ float s_expg[GPB], s_beta[GPB];

    const long long state_idx = (long long)state_indices[n * indices_stride];
    if (state_idx <= 0) {
        // NULL_BLOCK_ID: the recurrent kernel zeroes the output rows and returns; the
        // conv kernel skips entirely. Nothing else may be touched.
#pragma unroll
        for (int j = 0; j < GPB; ++j) {
            const long long o_base = (n * HV + (long long)i_h * GPB + j) * (long long)DV;
            out[o_base + tid] = __float2bfloat16(0.0f);
        }
        return;
    }

    // ---- phase 1: depthwise conv update; every channel private to this block ---------
    const long long qkv_base = n * qkv_stride;
    const long long cs_base = state_idx * cs_seq_stride;

    s_q[tid] = conv1d_channel(mixed_qkv, qkv_base, conv_state, cs_base, cs_dim_stride,
                              cs_tok_stride, conv_w, w_dim_stride, w_width_stride,
                              conv_bias, i_h * DK + tid);
    s_k[tid] = conv1d_channel(mixed_qkv, qkv_base, conv_state, cs_base, cs_dim_stride,
                              cs_tok_stride, conv_w, w_dim_stride, w_width_stride,
                              conv_bias, QK_DIM + i_h * DK + tid);
#pragma unroll
    for (int j = 0; j < GPB; ++j) {
        s_v[j * DV + tid] = conv1d_channel(
            mixed_qkv, qkv_base, conv_state, cs_base, cs_dim_stride, cs_tok_stride,
            conv_w, w_dim_stride, w_width_stride, conv_bias,
            2 * QK_DIM + (i_h * GPB + j) * DV + tid);
    }
    __syncthreads();

    // ---- phase 2: l2norm q/k (folded into the packed-decode kernel), per-head gating -
    const float q0 = s_q[tid];
    const float k0 = s_k[tid];
    const float sum_q = block_sum(__fmul_rn(q0, q0), s_red);
    __syncthreads();   // s_red reuse; also orders the s_q/s_k rewrites below
    const float sum_k = block_sum(__fmul_rn(k0, k0), s_red);

    if (tid < GPB) {
        const int i_hv = i_h * GPB + tid;
        // x = a + dt_bias; softplus with the Triton threshold; g = -exp(A_log) * sp.
        const float a_val = __bfloat162float(a[n * a_stride + i_hv]);
        const float b_val = __bfloat162float(b[n * b_stride + i_hv]);
        const float xg = a_val + dt_bias[i_hv];
        const float sp = (xg <= kSoftplusThreshold) ? logf(1.0f + expf(xg)) : xg;
        const float g = -expf(A_log[i_hv]) * sp;
        s_expg[tid] = expf(g);
        // beta = sigmoid(b) rounded THROUGH bf16 -- .to(b.dtype).to(f32) in Triton.
        const float sig = 1.0f / (1.0f + expf(-b_val));
        s_beta[tid] = __bfloat162float(__float2bfloat16(sig));
    }
    // q = q / sqrt(sum + eps); THEN q *= scale (two separate rounds, Triton order).
    s_q[tid] = __fmul_rn(q0 / sqrtf(sum_q + kL2Eps), scale);
    s_k[tid] = k0 / sqrtf(sum_k + kL2Eps);
    __syncthreads();

    // ---- phase 3: delta-rule recurrence for the GPB value heads, back to back --------
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const float kk0 = s_k[lane * 4 + 0], kk1 = s_k[lane * 4 + 1];
    const float kk2 = s_k[lane * 4 + 2], kk3 = s_k[lane * 4 + 3];
    const float qq0 = s_q[lane * 4 + 0], qq1 = s_q[lane * 4 + 1];
    const float qq2 = s_q[lane * 4 + 2], qq3 = s_q[lane * 4 + 3];

#pragma unroll
    for (int j = 0; j < GPB; ++j) {
        const int i_hv = i_h * GPB + j;
        const float expg = s_expg[j];
        const float beta = s_beta[j];
        float* const h_head =
            ssm_state + state_idx * state_tok_stride + (long long)i_hv * DV * DK;
        const long long o_base = (n * HV + i_hv) * (long long)DV;

        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            const int v = warp * ROWS_PER_WARP + r;
            float4* const row =
                reinterpret_cast<float4*>(h_head + (long long)v * DK) + lane;
            float4 h = *row;

            // h *= exp(g)
            h.x = __fmul_rn(h.x, expg); h.y = __fmul_rn(h.y, expg);
            h.z = __fmul_rn(h.z, expg); h.w = __fmul_rn(h.w, expg);

            // t = sum_k h[v,k] * k[k]   (mul then add, no FMA -- Triton does not fuse)
            float t = __fadd_rn(__fadd_rn(__fmul_rn(h.x, kk0), __fmul_rn(h.y, kk1)),
                                __fadd_rn(__fmul_rn(h.z, kk2), __fmul_rn(h.w, kk3)));
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                t = __fadd_rn(t, __shfl_xor_sync(0xffffffffu, t, off));

            // bv = (v - t) * beta
            const float bv = __fmul_rn(__fadd_rn(s_v[j * DV + v], -t), beta);

            // h += bv * k
            h.x = __fadd_rn(h.x, __fmul_rn(bv, kk0));
            h.y = __fadd_rn(h.y, __fmul_rn(bv, kk1));
            h.z = __fadd_rn(h.z, __fmul_rn(bv, kk2));
            h.w = __fadd_rn(h.w, __fmul_rn(bv, kk3));

            // o[v] = sum_k h[v,k] * q[k]
            float o = __fadd_rn(__fadd_rn(__fmul_rn(h.x, qq0), __fmul_rn(h.y, qq1)),
                                __fadd_rn(__fmul_rn(h.z, qq2), __fmul_rn(h.w, qq3)));
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                o = __fadd_rn(o, __shfl_xor_sync(0xffffffffu, o, off));

            *row = h;   // state written back in place, fp32
            if (lane == 0) out[o_base + v] = __float2bfloat16(o);
        }
    }
}
