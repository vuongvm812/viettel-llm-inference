// Fused int4 lm_head GEMV + argmax for the SPECULATIVE DRAFT model.
//
// Replaces `compute_logits(h).argmax(-1)` on the draft path -- a `[B, N]` logits tensor written
// to HBM and read straight back -- with a single fused reduction that never materialises it.
// At the shipped draft config (N = VTL_DRAFT_VOCAB = 16384 after FR-Spec pruning, K = 1024,
// B <= 8) the logits tensor is only ~0.5 MB, so this is a modest win; it is worth having because
// it also removes the dequant round trip and runs three times per target step.
//
// WHY ITS OWN WEIGHT COPY, and this is the load-bearing design decision.
//
// The production W4A8 path packs weights through `ops.cutlass_encode_and_reorder_int4b`
// (vtl/patches/quant_w4a8.py), a CUTLASS-internal interleaved/swizzled layout. Decoding that
// layout by hand would couple this kernel to a CUTLASS implementation detail, and a mistake
// would NOT surface as a crash or as wrong model output -- the target verifies every drafted
// token, so a wrong draft is simply rejected. It would surface as a slightly worse acceptance
// rate, which is unattributable and would be blamed on the model for weeks.
//
// So this kernel reads a SECOND, independently packed copy in a layout defined right here:
// plain row-major, 8 signed int4 per int32, symmetric per-group scales. At N=16384/K=1024 that
// is 8 MB of int4 + 0.5 MB of fp32 scales (16384 x 1024/128 x 4 B) -- affordable, and it buys
// total independence from
// whatever CUTLASS does. `vtl/patches/lm_head_quant.py` builds it at load time from the same
// pre-pack bf16 rows, and bench/test_draft_argmax.py checks it against a pure-torch oracle.
//
// W4A16, deliberately, not W4A8: the activation is `[B, 1024]` (a few KB) so quantising it saves
// nothing measurable, while keeping it in fp32 avoids a second quantisation error on the one
// number that decides which token gets drafted. Weight traffic -- which is what actually costs
// here -- is identical either way.
//
// LAYOUT
//   qweight      [N, K/8] int32     value k of row n lives in nibble (k % 8) of qweight[n][k/8],
//                                   stored as (v & 0xF) with v in [-8, 7] (sign-magnitude via
//                                   the standard 4-bit two's complement wrap).
//   group_scale  [N, K/G] float     w[n][k] = int4(n,k) * group_scale[n][k / G], G = kGroup.
//
// REDUCTION
// Two launches, and honestly so: stage 1 gives every (batch row, tile of output rows) block a
// partial (max, index); stage 2 reduces those partials. That is the same launch count as the
// stock `cutlass_w4a8_mm` + `argmax` it replaces -- the win here is bandwidth and the absent
// intermediate, not launch count. A single-launch version needs either a cooperative grid sync
// or a last-block-does-the-reduce atomic counter, and the counter carries state across cudagraph
// replays, which is exactly the kind of thing that works in a benchmark and fails in a graph.
//
// TIE-BREAKING must match `torch.argmax`: LOWEST index wins among equal values. Both reduction
// levels below compare `(val > best) || (val == best && idx < best_idx)`.

#include <torch/all.h>

#include <math_constants.h>

#include <limits>

#include "fp8_common.cuh"

namespace vtl {

namespace {

constexpr int kGroup = 128;        // must equal lm_head_quant's DRAFT_ARGMAX_GROUP
constexpr int kPack = 8;           // int4 values per int32
constexpr int kThreads = 256;      // 8 warps
constexpr int kWarps = kThreads / 32;
constexpr int kRowsPerBlock = 64;  // output rows a stage-1 block reduces

__device__ __forceinline__ int nibble_to_int4(uint32_t word, int j) {
  // Sign-extend the 4-bit field at position j. Shifting into the top bits and back down does it
  // in two instructions with no branch.
  int const shifted = static_cast<int>(word << (28 - 4 * j));
  return shifted >> 28;
}

// (value, index) max with torch.argmax tie-breaking: lowest index wins.
__device__ __forceinline__ void arg_max_into(float& best, int& best_idx, float v, int idx) {
  if (v > best || (v == best && idx < best_idx)) {
    best = v;
    best_idx = idx;
  }
}

__device__ __forceinline__ void warp_arg_max(float& best, int& best_idx) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    float const ov = __shfl_xor_sync(0xffffffffu, best, off);
    int const oi = __shfl_xor_sync(0xffffffffu, best_idx, off);
    arg_max_into(best, best_idx, ov, oi);
  }
}

// Stage 1: one block per (batch element, tile of kRowsPerBlock output rows).
// One WARP per output row; lanes stride the K axis by whole int32 words so the qweight loads are
// coalesced across the warp.
template <typename scalar_t>
__global__ void __launch_bounds__(kThreads) draft_argmax_stage1(
    float* __restrict__ part_val,            // [B, tiles]
    int32_t* __restrict__ part_idx,          // [B, tiles]
    scalar_t const* __restrict__ hidden,     // [B, K]
    uint32_t const* __restrict__ qweight,    // [N, K/8]
    float const* __restrict__ group_scale,   // [N, K/G]
    int const N, int const K, int64_t const hidden_stride) {
  int const b = blockIdx.x;
  int const tile = blockIdx.y;
  int const lane = threadIdx.x & 31;
  int const warp = threadIdx.x >> 5;

  int const words = K / kPack;      // int32 words per row
  int const groups = K / kGroup;    // scale entries per row
  scalar_t const* const x = hidden + static_cast<int64_t>(b) * hidden_stride;

  float best = -CUDART_INF_F;
  int best_idx = N;  // sentinel above every real index, so an empty warp never wins a tie

  int const row0 = tile * kRowsPerBlock;
  for (int r = warp; r < kRowsPerBlock; r += kWarps) {
    int const n = row0 + r;
    if (n >= N) break;

    uint32_t const* const wrow = qweight + static_cast<int64_t>(n) * words;
    float const* const srow = group_scale + static_cast<int64_t>(n) * groups;

    float acc = 0.0f;
    for (int w = lane; w < words; w += 32) {
      uint32_t const packed = wrow[w];
      int const k0 = w * kPack;
      // Every k in one int32 word shares a group as long as kGroup % kPack == 0, so the scale
      // is loaded once per word rather than once per value.
      float const scale = srow[k0 / kGroup];
      float partial = 0.0f;
#pragma unroll
      for (int j = 0; j < kPack; ++j) {
        partial += static_cast<float>(x[k0 + j]) * static_cast<float>(nibble_to_int4(packed, j));
      }
      acc += partial * scale;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);

    // Lane 0 holds the row's logit; every lane keeps the running best so the warp reduction
    // below is uniform.
    float const logit = __shfl_sync(0xffffffffu, acc, 0);
    arg_max_into(best, best_idx, logit, n);
  }

  // Reduce within the warp, then across warps through shared memory.
  warp_arg_max(best, best_idx);

  __shared__ float smem_val[kWarps];
  __shared__ int smem_idx[kWarps];
  if (lane == 0) {
    smem_val[warp] = best;
    smem_idx[warp] = best_idx;
  }
  __syncthreads();

  if (warp == 0) {
    best = (lane < kWarps) ? smem_val[lane] : -CUDART_INF_F;
    best_idx = (lane < kWarps) ? smem_idx[lane] : N;
    warp_arg_max(best, best_idx);
    if (lane == 0) {
      int64_t const slot = static_cast<int64_t>(b) * gridDim.y + tile;
      part_val[slot] = best;
      part_idx[slot] = best_idx;
    }
  }
}

// Stage 2: reduce one batch element's partials to a single token id. `tiles` is small (N /
// kRowsPerBlock, e.g. 256), so one block per batch element is plenty.
__global__ void __launch_bounds__(kThreads) draft_argmax_stage2(
    int64_t* __restrict__ out_ids,        // [B]
    float const* __restrict__ part_val,   // [B, tiles]
    int32_t const* __restrict__ part_idx, // [B, tiles]
    int const tiles, int const N) {
  int const b = blockIdx.x;
  int const lane = threadIdx.x & 31;
  int const warp = threadIdx.x >> 5;

  float best = -CUDART_INF_F;
  int best_idx = N;
  for (int t = threadIdx.x; t < tiles; t += kThreads) {
    int64_t const slot = static_cast<int64_t>(b) * tiles + t;
    arg_max_into(best, best_idx, part_val[slot], part_idx[slot]);
  }
  warp_arg_max(best, best_idx);

  __shared__ float smem_val[kWarps];
  __shared__ int smem_idx[kWarps];
  if (lane == 0) {
    smem_val[warp] = best;
    smem_idx[warp] = best_idx;
  }
  __syncthreads();

  if (warp == 0) {
    best = (lane < kWarps) ? smem_val[lane] : -CUDART_INF_F;
    best_idx = (lane < kWarps) ? smem_idx[lane] : N;
    warp_arg_max(best, best_idx);
    // Clamp the sentinel. NaN loses every comparison in arg_max_into (both `NaN > x` and
    // `NaN == x` are false), so an all-NaN row would leave best_idx at its initial `N` -- which
    // is a valid-but-never-scored id on a pruned head and OUT OF RANGE for the embedding lookup
    // on a full one. torch.argmax returns a real index in that case, so fall back to 0.
    if (lane == 0) out_ids[b] = static_cast<int64_t>(best_idx < N ? best_idx : 0);
  }
}

template <typename scalar_t>
void launch(torch::Tensor& out_ids, torch::Tensor const& hidden, torch::Tensor const& qweight,
            torch::Tensor const& group_scale, torch::Tensor& part_val, torch::Tensor& part_idx,
            cudaStream_t stream) {
  int const B = static_cast<int>(hidden.size(0));
  int const K = static_cast<int>(hidden.size(1));
  int const N = static_cast<int>(qweight.size(0));
  int const tiles = (N + kRowsPerBlock - 1) / kRowsPerBlock;

  dim3 const grid1(B, tiles);
  draft_argmax_stage1<scalar_t><<<grid1, kThreads, 0, stream>>>(
      part_val.data_ptr<float>(), part_idx.data_ptr<int32_t>(),
      hidden.const_data_ptr<scalar_t>(),
      reinterpret_cast<uint32_t const*>(qweight.const_data_ptr<int32_t>()),
      group_scale.const_data_ptr<float>(), N, K, hidden.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  draft_argmax_stage2<<<B, kThreads, 0, stream>>>(
      out_ids.data_ptr<int64_t>(), part_val.const_data_ptr<float>(),
      part_idx.const_data_ptr<int32_t>(), tiles, N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (kernel_sync_enabled()) {
    cudaError_t const err = cudaStreamSynchronize(stream);
    TORCH_CHECK(err == cudaSuccess, "vtl draft_argmax faulted: ", cudaGetErrorString(err),
                " | B=", B, " N=", N, " K=", K, " tiles=", tiles);
  }
}

}  // namespace

// True for shapes this kernel handles. Queried from Python so the eligibility gate does not
// re-hardcode the packing constants.
bool draft_argmax_supported(int64_t vocab, int64_t hidden_size) {
  // `vocab % kRowsPerBlock == 0` is NOT required -- stage 1's `break` handles a short tail tile
  // -- but it IS exercised by bench/test_draft_argmax.py's non-multiple case, so do not add the
  // constraint here without removing that coverage.
  return vocab > 0 && hidden_size > 0 && hidden_size % kGroup == 0 &&
         hidden_size % (kPack * 32) == 0 && vocab <= std::numeric_limits<int32_t>::max();
}

int64_t draft_argmax_tiles(int64_t vocab) {
  return (vocab + kRowsPerBlock - 1) / kRowsPerBlock;
}

void draft_argmax(torch::Tensor& out_ids, torch::Tensor const& hidden,
                  torch::Tensor const& qweight, torch::Tensor const& group_scale,
                  torch::Tensor& part_val, torch::Tensor& part_idx) {
  TORCH_CHECK(hidden.dim() == 2, "vtl: hidden must be [B, K]");
  TORCH_CHECK(qweight.dim() == 2 && group_scale.dim() == 2,
              "vtl: qweight [N, K/8] and group_scale [N, K/group] must be 2-D");
  TORCH_CHECK(qweight.scalar_type() == at::ScalarType::Int, "vtl: qweight must be int32");
  TORCH_CHECK(group_scale.scalar_type() == at::ScalarType::Float,
              "vtl: group_scale must be fp32");
  TORCH_CHECK(out_ids.scalar_type() == at::ScalarType::Long, "vtl: out_ids must be int64");
  // The three outputs are indexed as row-major [B, tiles] / [B] from raw pointers, so
  // contiguity and residency are correctness preconditions, not hygiene.
  TORCH_CHECK(out_ids.is_contiguous() && part_val.is_contiguous() && part_idx.is_contiguous(),
              "vtl: out_ids, part_val and part_idx must be contiguous");
  TORCH_CHECK(out_ids.is_cuda() && part_val.is_cuda() && part_idx.is_cuda() && hidden.is_cuda(),
              "vtl: draft_argmax operands must be CUDA tensors");
  TORCH_CHECK(part_val.scalar_type() == at::ScalarType::Float &&
                  part_idx.scalar_type() == at::ScalarType::Int,
              "vtl: part_val must be fp32 and part_idx int32");
  TORCH_CHECK(qweight.is_contiguous() && group_scale.is_contiguous(),
              "vtl: qweight and group_scale must be contiguous");
  TORCH_CHECK(hidden.stride(1) == 1, "vtl: hidden must be contiguous in its last dim");

  int64_t const B = hidden.size(0);
  int64_t const K = hidden.size(1);
  int64_t const N = qweight.size(0);
  TORCH_CHECK(draft_argmax_supported(N, K), "vtl: draft_argmax unsupported shape N=", N,
              " K=", K, " (call draft_argmax_supported first)");
  TORCH_CHECK(qweight.size(1) == K / kPack, "vtl: qweight must be [N, K/8]; got ",
              qweight.size(1), " for K=", K);
  TORCH_CHECK(group_scale.size(0) == N && group_scale.size(1) == K / kGroup,
              "vtl: group_scale must be [N, K/", kGroup, "]");
  TORCH_CHECK(out_ids.numel() == B, "vtl: out_ids must hold one id per row");

  int64_t const tiles = draft_argmax_tiles(N);
  TORCH_CHECK(part_val.numel() >= B * tiles && part_idx.numel() >= B * tiles,
              "vtl: partial workspace too small; need ", B * tiles, " got ", part_val.numel());

  c10::cuda::OptionalCUDAGuard const device_guard(hidden.device());
  cudaStream_t const stream = c10::cuda::getCurrentCUDAStream();

  switch (hidden.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch<c10::BFloat16>(out_ids, hidden, qweight, group_scale, part_val, part_idx, stream);
      break;
    case at::ScalarType::Half:
      launch<c10::Half>(out_ids, hidden, qweight, group_scale, part_val, part_idx, stream);
      break;
    case at::ScalarType::Float:
      launch<float>(out_ids, hidden, qweight, group_scale, part_val, part_idx, stream);
      break;
    default:
      TORCH_CHECK(false, "vtl: unsupported hidden dtype ", hidden.scalar_type());
  }
}

}  // namespace vtl
