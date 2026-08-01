// Registrations for vtl's CUDA kernels. Two flavours:
//
//   vllm_cuda::<op>                 -- our own op, so tests/benchmarks can call it side by
//                                     side with the stock one (importing vtl._C overrides _C
//                                     process-wide, so ours and stock cannot otherwise coexist).
//
//   _C::<op> (CUDA)                 -- overrides vLLM's CUDA kernel for an op vLLM already
//                                     emits. Same op identity, so FUSED_OPS,
//                                     FixFunctionalizationPass, the meta kernel and the
//                                     torch.compile cache key are all untouched; only the
//                                     kernel behind the dispatch key changes. Last registration
//                                     for a (op, key) pair wins (PyTorch warns). Requires
//                                     vLLM's `_C` schema to already exist, which is why the
//                                     patches import vllm._C_stable_libtorch before vtl._C.
//
// Ops (LFM2.5 target: 10 short-conv + 6 GQA-attn layers, SwiGLU MLP, no GDN/MTP/vision):
//   rms_norm_dynamic_per_token_quant           -- override _C  (fused norm+quant, feeds qkv/w13)
//   dynamic_per_token_scaled_fp8_quant         -- override _C  (attn out_proj + attn-out quant)
//   silu_and_mul_dynamic_per_token_quant       -- NEW op, vllm_cuda only (w2/down_proj fusion target;
//                                                 inserted into the graph by a fusion pattern, so
//                                                 its fake/meta kernel is registered in Python).
//   mul_dynamic_per_token_quant                -- NEW op, vllm_cuda only. Fuses the short-conv gate
//                                                 `y = C * Bx` + out_proj input quant (once the conv
//                                                 projections are fp8, see shortconv_quant.py).
//                                                 Called directly from the patched ShortConv decode
//                                                 path (no FX pattern -- short_conv is an opaque op);
//                                                 its fake kernel is registered in Python.
//   bcx_conv_gate_quant / _supported           -- NEW ops, vllm_cuda only. The whole short-conv
//                                                 DECODE block in one launch: `B*x` -> depthwise
//                                                 conv1d + state rotation -> `C*` -> fp8 quant,
//                                                 replacing mul_dynamic_per_token_quant AND the
//                                                 Triton causal_conv1d_update on that half of the
//                                                 batch. `_supported` is the shape predicate the
//                                                 Python gate queries instead of re-hardcoding it.
//   conv_align_fused / conv_align_supported    -- NEW ops, vllm_cuda only. The mamba "align"
//                                                 pre-copy (state advance + conv-window block
//                                                 migration) in one launch instead of the two
//                                                 Triton kernels preprocess_state runs every
//                                                 step. Called from the fork patch
//                                                 mamba_align_precopy.patch.
//   shortconv_decode_mega / _supported /       -- NEW ops, vllm_cuda only. The WHOLE short-conv
//     _grid / _scratch_bytes                      decode block (norm+quant -> in_proj W4A8 ->
//                                                 conv+gate+quant -> out_proj W4A8) in one
//                                                 launch, with a hand-rolled device-wide
//                                                 barrier between phases. `_grid` reports this
//                                                 build's real co-residency budget, which the
//                                                 barrier's safety depends on and which Python
//                                                 cannot compute.

#include <torch/extension.h>
#include <torch/library.h>

namespace vtl {
void rms_norm_dynamic_per_token_quant(torch::Tensor& result, torch::Tensor const& input,
                                      torch::Tensor const& weight, torch::Tensor& scale,
                                      double epsilon,
                                      std::optional<torch::Tensor> const& scale_ub,
                                      std::optional<torch::Tensor> const& residual);

void dynamic_per_token_scaled_fp8_quant(torch::Tensor& result, torch::Tensor const& input,
                                        torch::Tensor& scale,
                                        std::optional<torch::Tensor> const& scale_ub);

void silu_and_mul_dynamic_per_token_quant(torch::Tensor& result, torch::Tensor& scale,
                                          torch::Tensor const& input,
                                          std::optional<torch::Tensor> const& scale_ub);

void mul_dynamic_per_token_quant(torch::Tensor& result, torch::Tensor& scale,
                                 torch::Tensor const& a, torch::Tensor const& b,
                                 std::optional<torch::Tensor> const& scale_ub);

void bcx_conv_gate_quant(torch::Tensor& y_fp8, torch::Tensor& y_scale, torch::Tensor& conv_state,
                         torch::Tensor const& bcx, torch::Tensor const& conv_weight,
                         std::optional<torch::Tensor> const& conv_bias,
                         torch::Tensor const& state_indices, int64_t null_block_id,
                         std::optional<torch::Tensor> const& scale_ub);

bool bcx_conv_gate_supported(int64_t dim, int64_t width, int64_t state_len);

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
                      int64_t mamba_block_size);

bool conv_align_supported(bool dim_first, bool all_conv, int64_t num_states);

void shortconv_decode_mega(torch::Tensor& out, torch::Tensor const& x,
                           std::optional<torch::Tensor> const& norm_weight, double epsilon,
                           torch::Tensor const& in_wq, torch::Tensor const& in_gs,
                           torch::Tensor const& conv_weight,
                           std::optional<torch::Tensor> const& conv_bias,
                           torch::Tensor& conv_state, torch::Tensor const& state_indices,
                           int64_t null_block_id, torch::Tensor const& out_wq,
                           torch::Tensor const& out_gs, torch::Tensor& scratch,
                           int64_t num_blocks);

bool shortconv_decode_mega_supported(int64_t dim, int64_t width, int64_t state_len,
                                     int64_t max_tokens);
int64_t shortconv_decode_mega_grid(int64_t dim);
int64_t shortconv_decode_mega_scratch_bytes(int64_t dim);
}  // namespace vtl

TORCH_LIBRARY(vllm_cuda, m) {
  m.def(
      "rms_norm_dynamic_per_token_quant(Tensor! result, Tensor input, "
      "Tensor weight, Tensor! scale, float epsilon, "
      "Tensor? scale_ub, Tensor!? residual) -> ()");
  m.def(
      "dynamic_per_token_scaled_fp8_quant(Tensor! result, Tensor input, "
      "Tensor! scale, Tensor? scale_ub) -> ()");
  m.def(
      "silu_and_mul_dynamic_per_token_quant(Tensor! result, Tensor! scale, "
      "Tensor input, Tensor? scale_ub) -> ()");
  m.def(
      "mul_dynamic_per_token_quant(Tensor! result, Tensor! scale, "
      "Tensor a, Tensor b, Tensor? scale_ub) -> ()");
  m.def(
      "bcx_conv_gate_quant(Tensor! y_fp8, Tensor! y_scale, Tensor! conv_state, "
      "Tensor bcx, Tensor conv_weight, Tensor? conv_bias, Tensor state_indices, "
      "int null_block_id, Tensor? scale_ub) -> ()");
  m.def(
      "conv_align_fused(Tensor! state_idx, Tensor! num_accepted, Tensor idx_mapping, "
      "Tensor num_computed_tokens, Tensor query_start_loc, Tensor block_table_ptrs, "
      "int block_table_stride_req, Tensor state_base_addrs, Tensor state_block_strides, "
      "Tensor state_elem_sizes, Tensor state_inner_sizes, Tensor state_conv_widths, "
      "Tensor state_group_indices, int num_reqs, int mamba_block_size) -> ()");
  m.def(
      "shortconv_decode_mega(Tensor! out, Tensor x, Tensor? norm_weight, float epsilon, "
      "Tensor in_wq, Tensor in_gs, Tensor conv_weight, Tensor? conv_bias, "
      "Tensor! conv_state, Tensor state_indices, int null_block_id, Tensor out_wq, "
      "Tensor out_gs, Tensor! scratch, int num_blocks) -> ()");
  // Shape/capacity predicates: no tensor arguments, so they cannot be dispatched by device
  // key -- register the implementations right here as catch-alls instead of in the CUDA block
  // below. `shortconv_decode_mega_grid` queries this build's real occupancy, which is why the
  // Python gate cannot compute it itself.
  m.def("bcx_conv_gate_supported(int dim, int width, int state_len) -> bool",
        TORCH_FN(vtl::bcx_conv_gate_supported));
  m.def("conv_align_supported(bool dim_first, bool all_conv, int num_states) -> bool",
        TORCH_FN(vtl::conv_align_supported));
  m.def(
      "shortconv_decode_mega_supported(int dim, int width, int state_len, int max_tokens) "
      "-> bool",
      TORCH_FN(vtl::shortconv_decode_mega_supported));
  m.def("shortconv_decode_mega_grid(int dim) -> int",
        TORCH_FN(vtl::shortconv_decode_mega_grid));
  m.def("shortconv_decode_mega_scratch_bytes(int dim) -> int",
        TORCH_FN(vtl::shortconv_decode_mega_scratch_bytes));
}

TORCH_LIBRARY_IMPL(vllm_cuda, CUDA, m) {
  m.impl("rms_norm_dynamic_per_token_quant", TORCH_FN(vtl::rms_norm_dynamic_per_token_quant));
  m.impl("dynamic_per_token_scaled_fp8_quant",
         TORCH_FN(vtl::dynamic_per_token_scaled_fp8_quant));
  m.impl("silu_and_mul_dynamic_per_token_quant",
         TORCH_FN(vtl::silu_and_mul_dynamic_per_token_quant));
  m.impl("mul_dynamic_per_token_quant", TORCH_FN(vtl::mul_dynamic_per_token_quant));
  m.impl("bcx_conv_gate_quant", TORCH_FN(vtl::bcx_conv_gate_quant));
  m.impl("conv_align_fused", TORCH_FN(vtl::conv_align_fused));
  m.impl("shortconv_decode_mega", TORCH_FN(vtl::shortconv_decode_mega));
}

// Overrides of vLLM's own _C ops (schemas defined by vllm._C_stable_libtorch, imported first).
// The schemas here must match _C's exactly, including the mutable-alias (Tensor!) and optional
// (Tensor?) markers, or the impl is rejected.
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("rms_norm_dynamic_per_token_quant", TORCH_FN(vtl::rms_norm_dynamic_per_token_quant));
  m.impl("dynamic_per_token_scaled_fp8_quant",
         TORCH_FN(vtl::dynamic_per_token_scaled_fp8_quant));
}

// The ops arrive through the static initializers above, at dlopen. This exists only so that
// `import vtl._C` finds a PyInit__C and does not raise.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
