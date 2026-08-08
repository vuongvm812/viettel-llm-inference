// Fused mamba "align" pre-copy: state advance + conv-state block migration in ONE launch.
//
// Stock V2 align runs two Triton kernels per step, OUTSIDE the cudagraph, unconditionally
// (vllm/v1/worker/gpu/model_states/mamba_hybrid.py:206-226):
//
//   preprocess_mamba_align_fused_kernel[(cdiv(num_reqs,256),)]   # publish src_col/src_off,
//                                                                # advance state_idx
//   ctx.run_fused_precopy(...) -> precopy_mamba_align_fused_kernel[(num_reqs, num_states)]
//
// The first writes two [max_reqs] scratch arrays whose ONLY consumer is the second, and the
// second fast-exits per request on `src_col < 0 || src_col == dst_col` -- which is every
// step that does not cross a mamba block boundary, i.e. 15 out of every 16 decode steps at
// block_size 16. So the steady state is two launches, a full grid of them, to compute six
// scalars and then do nothing. This kernel is one launch of `num_reqs` blocks that keeps
// `src_col`/`token_bias` in shared memory and never materialises the scratch arrays.
//
// NO INTER-BLOCK RACE, BY CONSTRUCTION. Fusing the two kernels removes the grid-wide
// barrier between them, which is only safe because block `b` reads and writes exactly the
// per-request slot `idx_mapping[b]` and nothing else: a batch is a permutation of distinct
// request slots, so no two blocks touch the same `state_idx` / `num_accepted` element. The
// conv copies are likewise disjoint (distinct physical block ids per request).
//
// SCOPE -- LFM2's SD conv layout only:
//   * `is_conv_state_dim_first() == false`, i.e. the cache is (blocks, state_len, dim) with
//     `conv_width = size(1)` and `inner_size = stride(1)`, so one contiguous byte range per
//     (request, layer) migrates the window;
//   * every state is a conv state (`conv_width > 0`). LFM2.5 has 10 short-conv layers and no
//     temporal/SSM state, so the temporal branch of `_copy_mamba_state_block` is dead here.
// Anything else is refused by `conv_align_supported` and the caller keeps the two Triton
// launches. # ponytail: the DS layout and temporal states are a per-row strided copy and a
// second address mode; they are not in this model, so they are refused rather than written
// blind. Add them if a model with an SSM state ever lands in this repo.
//
// Arithmetic ported verbatim from mamba_utils.py:263-305 (the advance) and :27-130 (the SD
// conv branch of the copy body):
//
//   src_col        = state_idx[req]                       # pre-advance
//   token_bias     = max(num_accepted[req] - 1, 0)
//   computed_after = num_computed[req] + qsl[b+1] - qsl[b]
//   new_state_idx  = cdiv(computed_after, MAMBA_BLOCK) - 1
//   state_idx[req] = new_state_idx
//   if src_col >= 0 and src_col != new_state_idx: num_accepted[req] = 1
//   if src_col >= 0 and src_col != new_state_idx:
//       for each state s:
//           src = base[s] + bt[g][b*btr + src_col] * blk[s] + token_bias*inner[s]*elem[s]
//           dst = base[s] + bt[g][b*btr + new_state_idx] * blk[s]
//           memcpy(dst, src, (conv_width[s] - token_bias) * inner[s] * elem[s])
//
//   conv_align_fused(Tensor! state_idx, Tensor! num_accepted, Tensor idx_mapping,
//                    Tensor num_computed_tokens, Tensor query_start_loc,
//                    Tensor block_table_ptrs, int block_table_stride_req,
//                    Tensor state_base_addrs, Tensor state_block_strides,
//                    Tensor state_elem_sizes, Tensor state_inner_sizes,
//                    Tensor state_conv_widths, Tensor state_group_indices,
//                    int num_reqs, int mamba_block_size) -> ()
//   conv_align_supported(bool dim_first, int num_states) -> bool

#include <torch/all.h>

#include "fp8_common.cuh"

namespace vtl {

namespace {

constexpr int kThreads = 256;

// One block per request. Thread 0 does the (scalar) advance and publishes the copy
// decision; the whole block then serially drains each state's byte range. At LFM2's shape
// that is 10 states x <=8 KiB = 80 KiB moved by 256 threads on a boundary step, and 6
// scalar loads + an early return on the other 15/16 of steps.
__global__ void __launch_bounds__(kThreads) conv_align_fused_kernel(
    int32_t* __restrict__ state_idx,          // [max_reqs] in/out
    int32_t* __restrict__ num_accepted,       // [max_reqs] in/out
    int32_t const* __restrict__ idx_mapping,  // [num_reqs] batch -> req slot (-1 = skip)
    int32_t const* __restrict__ num_computed, // [max_reqs]
    int32_t const* __restrict__ qsl,          // [num_reqs + 1]
    int64_t const* __restrict__ bt_ptrs,      // [num_groups] device addresses of int32 tables
    int64_t const bt_stride_req,
    int64_t const* __restrict__ base_addrs,   // [num_states]
    int64_t const* __restrict__ blk_strides,  // [num_states] bytes per block
    int32_t const* __restrict__ elem_sizes,   // [num_states] bytes
    int64_t const* __restrict__ inner_sizes,  // [num_states] elements per conv position
    int32_t const* __restrict__ conv_widths,  // [num_states]
    int32_t const* __restrict__ group_idx,    // [num_states]
    int const num_states, int const mamba_block_size) {
  __shared__ int s_src_col;
  __shared__ int s_dst_col;
  __shared__ int s_token_bias;

  int const b = blockIdx.x;

  if (threadIdx.x == 0) {
    int const req = idx_mapping[b];
    if (req < 0) {
      s_src_col = -1;  // padded row: nothing to advance, nothing to copy
    } else {
      int const src_col = state_idx[req];
      int const acc = num_accepted[req];
      int const computed_after = num_computed[req] + qsl[b + 1] - qsl[b];
      // cdiv(computed_after, MB) - 1, matching Triton's integer form exactly.
      int const dst_col = (computed_after + mamba_block_size - 1) / mamba_block_size - 1;
      state_idx[req] = dst_col;
      bool const crossed = (src_col >= 0) && (src_col != dst_col);
      if (crossed) num_accepted[req] = 1;
      s_src_col = crossed ? src_col : -1;
      s_dst_col = dst_col;
      s_token_bias = acc > 1 ? acc - 1 : 0;
    }
  }
  __syncthreads();

  int const src_col = s_src_col;
  if (src_col < 0) return;
  int const dst_col = s_dst_col;
  int const token_bias = s_token_bias;

  for (int s = 0; s < num_states; ++s) {
    int const width = conv_widths[s];
    if (width <= token_bias) continue;  // nothing left of the window to migrate

    int64_t const elem = static_cast<int64_t>(elem_sizes[s]);
    int64_t const inner = inner_sizes[s];
    int64_t const blk = blk_strides[s];
    int64_t const base = base_addrs[s];

    int32_t const* const table =
        reinterpret_cast<int32_t const*>(bt_ptrs[group_idx[s]]) + b * bt_stride_req;
    int64_t const src_block = table[src_col];
    int64_t const dst_block = table[dst_col];

    char const* src = reinterpret_cast<char const*>(base + src_block * blk +
                                                    static_cast<int64_t>(token_bias) * inner * elem);
    char* dst = reinterpret_cast<char*>(base + dst_block * blk);
    int64_t const bytes = (static_cast<int64_t>(width) - token_bias) * inner * elem;

    // 16-byte vectorised when both ends and the length allow it, which is the LFM2 case
    // (block stride 8192 B, inner*elem 4096 B, so every address is 4 KiB aligned). The
    // byte loop is the correctness fallback, not dead code: a padded page stride or an
    // odd conv width lands there.
    if ((reinterpret_cast<uintptr_t>(src) % 16) == 0 &&
        (reinterpret_cast<uintptr_t>(dst) % 16) == 0 && (bytes % 16) == 0) {
      int64_t const nvec = bytes >> 4;
      auto const* vsrc = reinterpret_cast<uint4 const*>(src);
      auto* vdst = reinterpret_cast<uint4*>(dst);
      for (int64_t i = threadIdx.x; i < nvec; i += kThreads) vdst[i] = vsrc[i];
    } else {
      for (int64_t i = threadIdx.x; i < bytes; i += kThreads) dst[i] = src[i];
    }
  }
}

}  // namespace

// The shape predicate, queried from Python so the eligibility gate cannot drift from the
// kernel. `dim_first` is `is_conv_state_dim_first()`; `all_conv` is "every state in the
// context's metadata is a conv state" (conv_width > 0).
bool conv_align_supported(bool dim_first, bool all_conv, int64_t num_states) {
  return !dim_first && all_conv && num_states > 0;
}

void conv_align_fused(torch::Tensor& state_idx, torch::Tensor& num_accepted,
                      torch::Tensor const& idx_mapping,
                      torch::Tensor const& num_computed_tokens,
                      torch::Tensor const& query_start_loc,
                      torch::Tensor const& block_table_ptrs, int64_t block_table_stride_req,
                      torch::Tensor const& state_base_addrs,
                      torch::Tensor const& state_block_strides,
                      torch::Tensor const& state_elem_sizes,
                      torch::Tensor const& state_inner_sizes,
                      torch::Tensor const& state_conv_widths,
                      torch::Tensor const& state_group_indices, int64_t num_reqs,
                      int64_t mamba_block_size) {
  TORCH_CHECK(mamba_block_size > 0, "vtl: mamba_block_size must be positive");
  if (num_reqs <= 0) return;

  TORCH_CHECK(state_idx.scalar_type() == at::ScalarType::Int &&
                  num_accepted.scalar_type() == at::ScalarType::Int &&
                  idx_mapping.scalar_type() == at::ScalarType::Int &&
                  num_computed_tokens.scalar_type() == at::ScalarType::Int &&
                  query_start_loc.scalar_type() == at::ScalarType::Int,
              "vtl: conv_align_fused wants int32 state_idx/num_accepted/idx_mapping/"
              "num_computed_tokens/query_start_loc");
  TORCH_CHECK(state_base_addrs.scalar_type() == at::ScalarType::Long &&
                  state_block_strides.scalar_type() == at::ScalarType::Long &&
                  state_inner_sizes.scalar_type() == at::ScalarType::Long &&
                  block_table_ptrs.scalar_type() == at::ScalarType::Long,
              "vtl: conv_align_fused wants int64 addr/stride/inner/table-ptr metadata");
  TORCH_CHECK(state_elem_sizes.scalar_type() == at::ScalarType::Int &&
                  state_conv_widths.scalar_type() == at::ScalarType::Int &&
                  state_group_indices.scalar_type() == at::ScalarType::Int,
              "vtl: conv_align_fused wants int32 elem/width/group metadata");

  int64_t const num_states = state_base_addrs.numel();
  TORCH_CHECK(state_block_strides.numel() == num_states &&
                  state_elem_sizes.numel() == num_states &&
                  state_inner_sizes.numel() == num_states &&
                  state_conv_widths.numel() == num_states &&
                  state_group_indices.numel() == num_states,
              "vtl: conv_align_fused state metadata arrays must all be [num_states]");
  TORCH_CHECK(idx_mapping.numel() >= num_reqs,
              "vtl: idx_mapping must hold at least num_reqs entries");
  TORCH_CHECK(query_start_loc.numel() >= num_reqs + 1,
              "vtl: query_start_loc must hold num_reqs + 1 entries");
  TORCH_CHECK(num_states > 0 && num_states <= (1 << 20), "vtl: implausible num_states=",
              num_states);

  c10::cuda::OptionalCUDAGuard const device_guard(state_idx.device());
  cudaStream_t const stream = c10::cuda::getCurrentCUDAStream();

  conv_align_fused_kernel<<<dim3(static_cast<unsigned>(num_reqs)), dim3(kThreads), 0, stream>>>(
      state_idx.data_ptr<int32_t>(), num_accepted.data_ptr<int32_t>(),
      idx_mapping.const_data_ptr<int32_t>(), num_computed_tokens.const_data_ptr<int32_t>(),
      query_start_loc.const_data_ptr<int32_t>(), block_table_ptrs.const_data_ptr<int64_t>(),
      block_table_stride_req, state_base_addrs.const_data_ptr<int64_t>(),
      state_block_strides.const_data_ptr<int64_t>(),
      state_elem_sizes.const_data_ptr<int32_t>(),
      state_inner_sizes.const_data_ptr<int64_t>(),
      state_conv_widths.const_data_ptr<int32_t>(),
      state_group_indices.const_data_ptr<int32_t>(), static_cast<int>(num_states),
      static_cast<int>(mamba_block_size));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (kernel_sync_enabled()) {
    cudaError_t const err = cudaStreamSynchronize(stream);
    TORCH_CHECK(err == cudaSuccess, "vtl conv_align_fused faulted: ", cudaGetErrorString(err),
                " | num_reqs=", num_reqs, " num_states=", num_states,
                " mamba_block_size=", mamba_block_size,
                " bt_stride_req=", block_table_stride_req);
  }
}

}  // namespace vtl
