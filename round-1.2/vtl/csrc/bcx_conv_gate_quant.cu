// Fused LFM2 short-conv DECODE block: `B*x` -> depthwise causal conv1d (+ state rotation)
// -> `C*` gate -> dynamic per-token fp8 (e4m3) quant. One launch where stock needs three.
//
// Stock decode (vllm .../mamba/short_conv.py forward_cuda) is:
//     Bx_d = (B_d * x_d).contiguous()                      # elementwise mul + copy   [T,D] bf16
//     Bx   = causal_conv1d_update(Bx_d, conv_state, w, b)  # Triton, writes IN PLACE into Bx_d
//     y    = C_d * Bx                                      # + out_proj's own per-token fp8 quant
// i.e. the bf16 [T,D] `Bx` intermediate makes two full round-trips through HBM and three kernels
// are launched per conv layer (x10 layers per step). This kernel reads the raw `in_proj` output
// once, keeps `Bx` in registers, writes it straight into the conv-state ring buffer, and emits the
// fp8 activation + per-token scale that `out_proj`'s cutlass GEMM consumes directly.
//
// SCOPE -- decode only, width 3, reading the first `width - 1 = 2` slots of a state that may be
// ALLOCATED wider. That is exactly LFM2.5 (`conv_L_cache = 3`). Under speculative decode
// MambaStateShapeCalculator.short_conv_state_shape allocates `conv_kernel - 1 + num_spec` slots,
// but the Triton kernel pins `state_len = width - 1` whenever num_accepted_tokens is None
// (causal_conv1d.py:1181-1184), so the extra slots are inert on this path and the kernel is
// correct over them unchanged -- see the note on `supported_for`.
//
// Chain SPEC decode IS handled, by the kSpec instantiation below: one block per REQUEST, a
// per-request token loop, and taps read at `num_accepted - 1` so rejected drafts are rolled back
// instead of committed. Prefill (causal_conv1d_fn) is still NOT handled, nor is the tree-spec
// staging seam (_VTL_CONV_STAGE); the caller keeps the stock path for both.
//
// The kSpec=false semantics below are replicated from _causal_conv1d_update_kernel specialised to
// seqlen=1, KERNEL_WIDTH=3, state_len=2, IS_VARLEN/IS_APC_ENABLED/IS_SPEC_DECODING = False:
//
//   coord = state_indices[t, 0]
//   if coord == null_block_id: row is cudagraph padding -> skip entirely
//   s0, s1 = state[coord, c, 0], state[coord, c, 1]
//   bx     = B[t,c] * x[t,c]                                  <- scalar_t multiply
//   state[coord, c, 0], state[coord, c, 1] = s1, bx           <- the ring-buffer rotation
//   acc    = f32(bias[c]) + f32(w0*s0) + f32(w1*s1) + f32(w2*bx)
//   y      = C[t,c] * scalar_t(acc)
//
// NUMERICS ARE LOAD-BEARING. Triton types `bf16 * bf16 -> bf16` and accumulates into an fp32
// `acc`, then rounds once more on the bf16 store; the products below are narrowed the same way
// (`mul_one`, identical to mul_quant.cu). Do NOT "improve" this to an fp32-product FMA -- it would
// silently diverge from the Triton kernel that the prefill half of the same layer still uses.
// The quant epilogue (amax -> scale -> reciprocal-multiply -> clamp -> hw convert) is character
// for character the one in mul_quant.cu / rms_norm_quant.cu.
//
// NULL-BLOCK DIVERGENCE (deliberate): stock early-returns on padded rows, leaving whatever `B*x`
// wrote in the aliased output buffer -- finite garbage that is discarded downstream. We instead
// write y_fp8 = 0 and y_scale = kMinScale for those rows, so the padded rows of the following GEMM
// are defined rather than uninitialised. State is untouched either way, which is the part that
// matters (a stray write would corrupt a live sequence's ring buffer).
//
//   bcx_conv_gate_quant(Tensor! y_fp8, Tensor! y_scale, Tensor! conv_state, Tensor bcx,
//                       Tensor conv_weight, Tensor? conv_bias, Tensor state_indices,
//                       int null_block_id, Tensor? scale_ub) -> ()
//   bcx_conv_gate_supported(int dim, int width, int state_len) -> bool

#include <torch/all.h>

#include "fp8_common.cuh"

namespace vtl {

namespace {

constexpr int kWidth = 3;  // conv_L_cache on LFM2.5
// Tap slots READ per token, not the allocated ring-buffer width -- the allocation may be wider
// under spec decode and `supported_for` accepts that (see the note there).
constexpr int kStateLen = 2;  // == kWidth - 1
// The tap loop below selects s0 / s1 / bx by index; widening kWidth needs that select widened too.
static_assert(kWidth == 3 && kStateLen == kWidth - 1, "tap select is specialised to width 3");

// scalar_t multiply then narrow, returned as fp32 -- matches Triton's bf16 arithmetic.
// Identical to mul_quant.cu's helper of the same name; kept local so each .cu stays one TU.
template <typename scalar_t>
__device__ __forceinline__ float mul_one(scalar_t a, scalar_t b) {
  return static_cast<float>(a * b);
}

// One block per token, D/kVec threads (256 for bf16 at D=2048), each thread owning kVec
// consecutive channels in registers across the single block-wide amax reduction.
//
// `bcx` is the raw in_proj output [T, 3D]: B = cols [0,D), C = [D,2D), x = [2D,3D) -- matching
// `B, C, x = BCx.chunk(3, dim=-1)`. Three 16-byte vector loads per thread; no chunk view, no
// .contiguous() copy.
//
// WHY kVectorized MATTERS ON H200. At decode the grid is the batch size (1-16 blocks), so this
// kernel is latency- and ISSUE-bound, never bandwidth-bound: its whole payload is ~32 KB and it
// occupies a single SM at batch 1. Read scalar-wise, `conv_state` (kVec x 2) and `conv_weight`
// (kVec x 3) cost 40 loads + 16 stores per thread -- ~15k memory instructions per block, which at
// ~4 LSU issues/cycle is ~2.3 us of pure issue on a kernel that should cost 3-5 us end to end.
// Vectorised they collapse to 5 loads + 2 stores. Both slices are provably 16-byte aligned:
//   * weight is row-major (D, kWidth), so a thread's slice starts at base_c*kWidth elements
//     = kVec*kWidth*threadIdx.x, i.e. 16*kWidth*threadIdx.x BYTES (kVec*sizeof(scalar_t) == 16);
//   * state (when stride_dim == 1) starts at base_c elements = 16*threadIdx.x bytes.
// The scalar instantiation is kept for the non-default "DS" conv-state layout (stride_dim == 2)
// and any non-contiguous conv weight -- correct, just not accelerated. `launch` picks between
// them; both are covered by bench/test_bcx_conv_gate_quant.py.
// kSpec: CHAIN SPECULATIVE DECODE. One block per REQUEST instead of per token, looping the
// request's 1..1+num_spec query tokens serially, with the conv taps read at an offset derived
// from num_accepted_tokens so the rejected drafts of the previous step are rolled back rather
// than committed. Replicates _causal_conv1d_update_kernel's IS_SPEC_DECODING path exactly;
// derivation, with the Triton line numbers, next to the state write below.
//
// The token loop MUST stay inside one block: the taps are loop-carried
// ((tap0, tap1) <- (tap1, bx_t)), and the alternative -- one block per token, each recomputing
// its predecessors' bx from `bcx` -- would race, because block t writes state slot 1+t while
// block t' may still be reading slots off/off+1. One block per request also preserves the
// "every block owns a distinct coord" invariant that makes the state writes safe without any
// cross-block ordering.
//
// kSpec=false constant-folds qs=blockIdx.x, L=1, off=0 and compiles to exactly the pre-spec
// kernel -- that is why this is a template parameter and not a runtime branch.
template <typename scalar_t, bool kVectorized, bool kSpec>
__global__ void __launch_bounds__(kFastMaxThreads) bcx_conv_gate_kernel(
    c10::Float8_e4m3fn* __restrict__ out,       // [T, D]
    float* __restrict__ scales,                 // [T]
    scalar_t* __restrict__ state,               // [num_blocks, D, S], S >= kStateLen
    scalar_t const* __restrict__ bcx,           // [T, 3D]
    scalar_t const* __restrict__ weight,        // [D, kWidth]
    scalar_t const* __restrict__ bias,          // [D] or nullptr
    int32_t const* __restrict__ state_idx,      // [T or num_reqs, *]
    float const* __restrict__ scale_ub,         // scalar or nullptr
    int32_t const* __restrict__ num_accepted,   // [num_reqs], kSpec only
    int32_t const* __restrict__ query_start,    // [num_reqs + 1], kSpec only
    int const D, int64_t const bcx_stride, int64_t const st_blk, int64_t const st_dim,
    int64_t const st_tok, int64_t const w_dim, int64_t const idx_stride,
    int64_t const null_block_id) {
  constexpr int kVec = VecTraits<scalar_t>::kVec;
  using Vec = VecIn<scalar_t, kVec>;

  int64_t const seq = blockIdx.x;  // token index when !kSpec, request index when kSpec
  int const base_c = threadIdx.x * kVec;
  bool const active = base_c < D;

  // Query extent FIRST -- a padded row still has to leave a defined output behind, and that
  // needs qs/L even though the state must not be touched.
  int64_t qs, L;
  if constexpr (kSpec) {
    qs = static_cast<int64_t>(query_start[seq]);
    L = static_cast<int64_t>(query_start[seq + 1]) - qs;
  } else {
    qs = seq;
    L = 1;
  }

  __shared__ float smem_max[32];

  // Uniform across the block (depends only on blockIdx.x), so the early-out below never splits a
  // block across the __syncthreads() inside block_reduce.
  int64_t const coord = static_cast<int64_t>(state_idx[seq * idx_stride]);
  if (coord == null_block_id || L <= 0) {
    // Deliberate divergence from stock (see the NULL-BLOCK note in the file header): define the
    // padded rows rather than leaving whatever B*x wrote in the aliased buffer.
    //
    // Only ever touches THIS request's own rows, so it cannot run past the decode slice into the
    // prefill half. Cudagraph padding is the case that would: those requests get
    // state_indices = NULL_BLOCK_ID (mamba_attn.py:585) AND a degenerate query range, because
    // query_start_loc repeats the final cumulative total past num_reqs
    // (gpu_model_runner.py:2057) -- so L == 0 and the loop below does not execute at all.
    for (int64_t t = 0; t < L; ++t) {
      int64_t const tok = qs + t;
      if (threadIdx.x == 0) scales[tok] = kMinScale;
      if (active) {
        VecOut<kVec> vzero;
#pragma unroll
        for (int j = 0; j < kVec / 2; ++j) vzero.pair[j] = floats_to_fp8x2(0.0f, 0.0f);
        *reinterpret_cast<VecOut<kVec>*>(out + tok * static_cast<int64_t>(D) + base_c) = vzero;
      }
    }
    return;
  }

  // Tap offset. Under chain spec the previous step left
  // [history2..historyM, draft1..draftN] in the ring; accepting `a` tokens means the live window
  // starts at a-1 (causal_conv1d.py:836-854). a==0 cannot happen (the bonus token is always
  // accepted) but clamp rather than read out of bounds if it ever does -- Triton does not.
  int64_t off = 0;
  if constexpr (kSpec) {
    off = static_cast<int64_t>(num_accepted[seq]) - 1;
    if (off < 0) off = 0;
  }

  scalar_t* st_row = nullptr;
  scalar_t const* w_row = nullptr;
  Vec vw[kWidth];
  Vec vt0, vt1;   // the two live taps, loop-carried
  Vec vt1_orig;   // s[off+1] as it was BEFORE the loop -- becomes the new slot 0

  if (active) {
    st_row = state + coord * st_blk + static_cast<int64_t>(base_c) * st_dim;
    w_row = weight + static_cast<int64_t>(base_c) * w_dim;

    // ---- gather this thread's kVec channels: weights, then the taps at `off` ----
    // Indices below are compile-time constants inside the fully unrolled loops, so `vw`/`vt*`
    // stay in registers -- no local-memory spill, no dynamic indexing.
    if constexpr (kVectorized) {
#pragma unroll
      for (int i = 0; i < kWidth; ++i) {
        vw[i] = *reinterpret_cast<Vec const*>(w_row + i * kVec);
      }
      vt0 = *reinterpret_cast<Vec const*>(st_row + off * st_tok);
      vt1 = *reinterpret_cast<Vec const*>(st_row + (off + 1) * st_tok);
    } else {
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        scalar_t const* const w = w_row + static_cast<int64_t>(j) * w_dim;
#pragma unroll
        for (int k = 0; k < kWidth; ++k) {
          int const flat = j * kWidth + k;
          vw[flat / kVec].v[flat % kVec] = w[k];
        }
        scalar_t const* const sp = st_row + static_cast<int64_t>(j) * st_dim;
        vt0.v[j] = sp[off * st_tok];
        vt1.v[j] = sp[(off + 1) * st_tok];
      }
    }
    vt1_orig = vt1;
  }

  for (int64_t t = 0; t < L; ++t) {
    int64_t const tok = qs + t;
    float p[kVec];
    Vec vbx;

    if (active) {
      int64_t const row = tok * bcx_stride + base_c;
      Vec const vb = *reinterpret_cast<Vec const*>(bcx + row);
      Vec const vc = *reinterpret_cast<Vec const*>(bcx + row + D);
      Vec const vx = *reinterpret_cast<Vec const*>(bcx + row + 2 * static_cast<int64_t>(D));

      // ---- the block itself ----
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        scalar_t const s0 = vt0.v[j];
        scalar_t const s1 = vt1.v[j];
        scalar_t const bx = vb.v[j] * vx.v[j];
        vbx.v[j] = bx;

        float acc = (bias != nullptr) ? static_cast<float>(bias[base_c + j]) : 0.0f;
#pragma unroll
        for (int k = 0; k < kWidth; ++k) {
          int const flat = j * kWidth + k;
          scalar_t const w = vw[flat / kVec].v[flat % kVec];
          acc += mul_one(w, (k == 0) ? s0 : (k == 1) ? s1 : bx);
        }
        p[j] = mul_one(vc.v[j], static_cast<scalar_t>(acc));
      }

      // Slot 1+t takes this token's bx. Writing as we go is safe: the taps were read once, above,
      // and slot 0 is written after the loop. Slots > L are never touched, matching the Triton
      // store mask `idx_tokens < state_len` with state_len = L + 1.
      if constexpr (kVectorized) {
        *reinterpret_cast<Vec*>(st_row + (1 + t) * st_tok) = vbx;
      } else {
#pragma unroll
        for (int j = 0; j < kVec; ++j) {
          st_row[static_cast<int64_t>(j) * st_dim + (1 + t) * st_tok] = vbx.v[j];
        }
      }
    } else {
#pragma unroll
      for (int j = 0; j < kVec; ++j) p[j] = 0.0f;
    }

    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < kVec; ++j) amax = fmaxf(amax, fabsf(p[j]));
    amax = block_reduce(amax, MaxOp{}, 0.0f, smem_max);
    if (scale_ub != nullptr) amax = fminf(amax, *scale_ub);
    float const scale = fmaxf(amax / kFp8Max, kMinScale);
    if (threadIdx.x == 0) scales[tok] = scale;
    float const inv_scale = 1.0f / scale;

    if (active) {
      VecOut<kVec> vout;
#pragma unroll
      for (int j = 0; j < kVec; j += 2) {
        vout.pair[j >> 1] = floats_to_fp8x2(p[j] * inv_scale, p[j + 1] * inv_scale);
      }
      *reinterpret_cast<VecOut<kVec>*>(out + tok * static_cast<int64_t>(D) + base_c) = vout;
      // Shift the taps for the next token, exactly as Triton does (`col0 = col1; col1 = x`).
      vt0 = vt1;
      vt1 = vbx;
    }

    // block_reduce's trailing __syncthreads() protects its own read of smem[0], but NOT that
    // read against the NEXT call's write to smem[wid]. `L` is uniform across the block, so this
    // barrier is uniform too, and it costs nothing when L == 1 (the whole non-spec path).
    if (t + 1 < L) __syncthreads();
  }

  // Slot 0 takes what was at off+1, i.e. the window slides left by exactly one regardless of L
  // -- Triton's source mask `(idx_tokens + seqlen) < state_len` admits only idx_tokens == 0
  // (causal_conv1d.py:887-895 with state_len = L + 1). Every block owns a distinct `coord`, so
  // there is no cross-block race on these slots.
  if (active) {
    if constexpr (kVectorized) {
      *reinterpret_cast<Vec*>(st_row) = vt1_orig;  // a straight register-to-memory move
    } else {
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        st_row[static_cast<int64_t>(j) * st_dim] = vt1_orig.v[j];
      }
    }
  }
}

// The shape predicate, shared by the launcher's TORCH_CHECK and the Python-side eligibility gate
// (via the `bcx_conv_gate_supported` op) so the two can never drift apart.
template <typename scalar_t>
bool supported_for(int64_t dim, int64_t width, int64_t state_len) {
  constexpr int kVec = VecTraits<scalar_t>::kVec;
  // `>=`, not `==`: kStateLen is the number of tap slots this kernel READS, not the allocated
  // width of the ring buffer. Speculative decode widens the allocation to `conv_kernel - 1 +
  // num_spec` (MambaStateShapeCalculator.short_conv_state_shape), which an `==` rejected outright
  // -- disabling this kernel on BOTH models the moment num_speculative_tokens went non-zero.
  //
  // A non-spec single-token decode over a [blocks, D, 5] state is byte-identical to one over
  // [blocks, D, 2]. Traced through the stock kernel:
  //   * causal_conv1d.py:1181-1184 overrides state_len to `width - 1` on the HOST whenever
  //     num_accepted_tokens is None -- conv_state.size(2) is only read to validate `>= width-1`;
  //   * mamba_attn.py:496 only populates query_start_loc_d when num_accepted_tokens is not None,
  //     so this call takes the NON-varlen path (seqlen=1) and skips the :823 state_len revision;
  //   * with IS_SPEC_DECODING false the tap offset is 0 (:843-844), so taps are slots 0/1, the
  //     slide writes only slots 0/1 (:887-933), and slots >= 2 are never read or written.
  // The kernel addresses state through runtime strides, so it handles the wider allocation as-is.
  //
  // The TARGET under spec is a different case (per-request token loop + num_accepted tap offset)
  // and is still rejected -- by the caller's `spec_decode_active` gate, not by this predicate.
  if (width != kWidth || state_len < kStateLen) return false;
  if (dim <= 0 || dim % kVec != 0) return false;
  return dim / kVec <= kFastMaxThreads;
}

template <typename scalar_t>
void launch(torch::Tensor& out, torch::Tensor& scales, torch::Tensor& state,
            torch::Tensor const& bcx, torch::Tensor const& weight,
            std::optional<torch::Tensor> const& bias, torch::Tensor const& state_idx,
            std::optional<torch::Tensor> const& scale_ub,
            std::optional<torch::Tensor> const& num_accepted,
            std::optional<torch::Tensor> const& query_start, int64_t null_block_id,
            cudaStream_t stream) {
  constexpr int kVec = VecTraits<scalar_t>::kVec;

  int const D = static_cast<int>(weight.size(0));
  int64_t const num_tokens = bcx.size(0);
  if (num_tokens == 0) return;

  // Both or neither: the offset is meaningless without the extents and vice versa.
  bool const spec = num_accepted.has_value();
  TORCH_CHECK(spec == query_start.has_value(),
              "vtl: bcx_conv_gate_quant needs num_accepted_tokens and query_start_loc together");

  TORCH_CHECK(supported_for<scalar_t>(D, weight.size(1), state.size(2)),
              "vtl: bcx_conv_gate_quant unsupported shape dim=", D, " width=", weight.size(1),
              " state_len=", state.size(2), " (call bcx_conv_gate_supported first)");

  auto* out_p = reinterpret_cast<c10::Float8_e4m3fn*>(out.data_ptr());
  auto* scales_p = scales.data_ptr<float>();
  auto* state_p = state.data_ptr<scalar_t>();
  auto const* bcx_p = bcx.const_data_ptr<scalar_t>();
  auto const* w_p = weight.const_data_ptr<scalar_t>();
  auto const* bias_p = bias.has_value() ? bias->const_data_ptr<scalar_t>() : nullptr;
  auto const* idx_p = state_idx.const_data_ptr<int32_t>();
  auto const* ub_p = scale_ub.has_value() ? scale_ub->const_data_ptr<float>() : nullptr;
  auto const* na_p = spec ? num_accepted->const_data_ptr<int32_t>() : nullptr;
  auto const* qs_p = spec ? query_start->const_data_ptr<int32_t>() : nullptr;

  int64_t const bcx_stride = bcx.stride(0);

  // The three 16-byte vector loads (B, C, x) need every row start 16-byte aligned. D % kVec == 0
  // already makes the +D / +2D column offsets a whole number of vectors.
  TORCH_CHECK(aligned16(bcx_p) && aligned16(out_p) && bcx_stride % kVec == 0,
              "vtl: bcx_conv_gate_quant needs 16B-aligned bcx/out and a kVec-multiple row stride;"
              " got bcx_stride=", bcx_stride);

  int64_t const st_blk = state.stride(0);
  int64_t const st_dim = state.stride(1);
  int64_t const st_tok = state.stride(2);
  int64_t const w_dim = weight.stride(0);

  // 16-byte loads for the state and the taps. Requires the channel axis to be the contiguous one
  // (the default "SD" layout, transposed to (blocks, D, S) -> stride_dim == 1) and a row-major
  // conv weight, plus enough stride alignment that every thread's slice stays 16-byte aligned.
  // "DS" and any exotic weight layout take the scalar instantiation -- correct, just not tuned.
  bool const vectorized = st_dim == 1 && st_tok % kVec == 0 && st_blk % kVec == 0 &&
                          w_dim == kWidth && aligned16(state_p) && aligned16(w_p);

  int const nthreads = D / kVec;
  // One block per REQUEST under spec (the token loop is loop-carried through the taps), one per
  // token otherwise. state_idx rows are indexed by whichever of the two the grid counts, and the
  // two coincide exactly when every query length is 1 -- which is why the non-spec launch is
  // unchanged.
  int64_t const grid_n = spec ? state_idx.size(0) : num_tokens;
  dim3 const grid(grid_n);
  dim3 const block((nthreads + 31) / 32 * 32);
  if (spec) {
    if (vectorized) {
      bcx_conv_gate_kernel<scalar_t, true, true><<<grid, block, 0, stream>>>(
          out_p, scales_p, state_p, bcx_p, w_p, bias_p, idx_p, ub_p, na_p, qs_p, D, bcx_stride,
          st_blk, st_dim, st_tok, w_dim, state_idx.stride(0), null_block_id);
    } else {
      bcx_conv_gate_kernel<scalar_t, false, true><<<grid, block, 0, stream>>>(
          out_p, scales_p, state_p, bcx_p, w_p, bias_p, idx_p, ub_p, na_p, qs_p, D, bcx_stride,
          st_blk, st_dim, st_tok, w_dim, state_idx.stride(0), null_block_id);
    }
  } else if (vectorized) {
    bcx_conv_gate_kernel<scalar_t, true, false><<<grid, block, 0, stream>>>(
        out_p, scales_p, state_p, bcx_p, w_p, bias_p, idx_p, ub_p, nullptr, nullptr, D, bcx_stride,
        st_blk, st_dim, st_tok, w_dim, state_idx.stride(0), null_block_id);
  } else {
    bcx_conv_gate_kernel<scalar_t, false, false><<<grid, block, 0, stream>>>(
        out_p, scales_p, state_p, bcx_p, w_p, bias_p, idx_p, ub_p, nullptr, nullptr, D, bcx_stride,
        st_blk, st_dim, st_tok, w_dim, state_idx.stride(0), null_block_id);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (kernel_sync_enabled()) {
    cudaError_t const err = cudaStreamSynchronize(stream);
    TORCH_CHECK(err == cudaSuccess, "vtl bcx_conv_gate_quant faulted: ", cudaGetErrorString(err),
                " | path=", (vectorized ? "vec" : "scalar"), " dtype=", bcx.scalar_type(),
                " num_tokens=", num_tokens, " D=", D, " bcx_stride=", bcx_stride, " st=(", st_blk,
                ",", st_dim, ",", st_tok, ") w_dim=", w_dim, " kVec=", kVec);
  }
}

}  // namespace

// Shapes the fused kernel actually handles, queried from Python so the eligibility gate does not
// re-hardcode kFastMaxThreads / kVec. Answers for the bf16 vector width, which is also the fp16
// one; the fp32 path (kVec=4) is test-only and strictly more permissive.
// `state_len` here is the ALLOCATED ring-buffer width; anything >= kStateLen is accepted, since
// this kernel only ever touches the first kStateLen slots.
bool bcx_conv_gate_supported(int64_t dim, int64_t width, int64_t state_len) {
  return supported_for<c10::BFloat16>(dim, width, state_len);
}

void bcx_conv_gate_quant(torch::Tensor& y_fp8, torch::Tensor& y_scale, torch::Tensor& conv_state,
                         torch::Tensor const& bcx, torch::Tensor const& conv_weight,
                         std::optional<torch::Tensor> const& conv_bias,
                         torch::Tensor const& state_indices, int64_t null_block_id,
                         std::optional<torch::Tensor> const& scale_ub,
                         std::optional<torch::Tensor> const& num_accepted_tokens,
                         std::optional<torch::Tensor> const& query_start_loc) {
  TORCH_CHECK(y_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "vtl: bcx_conv_gate_quant supports fp8_e4m3 output only, got ", y_fp8.scalar_type());
  TORCH_CHECK(y_fp8.is_contiguous(), "vtl: y_fp8 must be contiguous");
  TORCH_CHECK(y_scale.scalar_type() == at::ScalarType::Float, "vtl: y_scale must be fp32");
  TORCH_CHECK(y_scale.is_contiguous(), "vtl: y_scale must be contiguous");
  TORCH_CHECK(bcx.dim() == 2 && conv_weight.dim() == 2 && conv_state.dim() == 3,
              "vtl: expected bcx [T,3D], conv_weight [D,W], conv_state [blocks,D,S]");
  TORCH_CHECK(bcx.stride(1) == 1, "vtl: bcx must be contiguous in its last dim");
  TORCH_CHECK(conv_weight.stride(1) == 1, "vtl: conv_weight must be contiguous in its last dim");

  int64_t const D = conv_weight.size(0);
  TORCH_CHECK(bcx.size(1) == 3 * D, "vtl: bcx must be [T, 3*dim]; got ", bcx.size(1),
              " for dim=", D);
  TORCH_CHECK(conv_state.size(1) == D, "vtl: conv_state dim mismatch: ", conv_state.size(1),
              " vs ", D);
  TORCH_CHECK(y_fp8.size(0) == bcx.size(0) && y_fp8.size(1) == D,
              "vtl: y_fp8 must be [T, dim] matching bcx");
  TORCH_CHECK(y_scale.numel() == bcx.size(0), "vtl: y_scale must hold one scale per token");
  TORCH_CHECK(state_indices.scalar_type() == at::ScalarType::Int,
              "vtl: state_indices must be int32, got ", state_indices.scalar_type());

  // state_indices rows are per-REQUEST under spec and per-TOKEN otherwise. The two coincide
  // whenever every query length is 1, so a caller that gets this wrong would NOT trip a
  // `size(0) == bcx.size(0)` check on the non-spec path -- hence keying the check on the
  // presence of query_start_loc, which is the thing that actually changes the contract.
  bool const spec = num_accepted_tokens.has_value();
  TORCH_CHECK(spec == query_start_loc.has_value(),
              "vtl: num_accepted_tokens and query_start_loc must be given together");
  if (spec) {
    TORCH_CHECK(query_start_loc->scalar_type() == at::ScalarType::Int &&
                    num_accepted_tokens->scalar_type() == at::ScalarType::Int,
                "vtl: query_start_loc and num_accepted_tokens must be int32");
    TORCH_CHECK(query_start_loc->dim() == 1 && num_accepted_tokens->dim() == 1,
                "vtl: query_start_loc [reqs+1] and num_accepted_tokens [reqs] must be 1-D");
    int64_t const reqs = state_indices.size(0);
    TORCH_CHECK(query_start_loc->numel() == reqs + 1,
                "vtl: query_start_loc must be [num_reqs+1]; got ", query_start_loc->numel(),
                " for ", reqs, " requests");
    TORCH_CHECK(num_accepted_tokens->numel() == reqs,
                "vtl: num_accepted_tokens must be [num_reqs]; got ",
                num_accepted_tokens->numel(), " for ", reqs, " requests");
    TORCH_CHECK(state_indices.dim() == 2,
                "vtl: spec state_indices must be [num_reqs, 1+num_spec]");
    // Slot 1+t is written for t in [0, L) with L <= state_indices.size(1), and the taps sit at
    // off/off+1 with off <= num_accepted-1 <= size(1)-1. Both fit iff the ring holds one more
    // slot than the widest query. Exactly satisfied at 2+num_spec vs 1+(1+num_spec).
    TORCH_CHECK(conv_state.size(2) >= 1 + state_indices.size(1),
                "vtl: conv_state needs >= 1+", state_indices.size(1), " slots for spec decode; got ",
                conv_state.size(2));
  } else {
    TORCH_CHECK(state_indices.size(0) == bcx.size(0),
                "vtl: state_indices must have one row per token");
  }
  TORCH_CHECK(conv_state.scalar_type() == bcx.scalar_type() &&
                  conv_weight.scalar_type() == bcx.scalar_type(),
              "vtl: bcx, conv_weight and conv_state must share a dtype");
  TORCH_CHECK(!conv_bias.has_value() || (conv_bias->scalar_type() == bcx.scalar_type() &&
                                         conv_bias->numel() == D && conv_bias->is_contiguous()),
              "vtl: conv_bias must be a contiguous [dim] tensor of the input dtype");

  c10::cuda::OptionalCUDAGuard const device_guard(bcx.device());
  cudaStream_t const stream = c10::cuda::getCurrentCUDAStream();

  switch (bcx.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch<c10::BFloat16>(y_fp8, y_scale, conv_state, bcx, conv_weight, conv_bias, state_indices,
                            scale_ub, num_accepted_tokens, query_start_loc, null_block_id, stream);
      break;
    case at::ScalarType::Half:
      launch<c10::Half>(y_fp8, y_scale, conv_state, bcx, conv_weight, conv_bias, state_indices,
                        scale_ub, num_accepted_tokens, query_start_loc, null_block_id, stream);
      break;
    case at::ScalarType::Float:
      launch<float>(y_fp8, y_scale, conv_state, bcx, conv_weight, conv_bias, state_indices,
                    scale_ub, num_accepted_tokens, query_start_loc, null_block_id, stream);
      break;
    default:
      TORCH_CHECK(false, "vtl: unsupported input dtype ", bcx.scalar_type());
  }
}

}  // namespace vtl
