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
// Ops:
//   rms_norm_dynamic_per_token_quant           -- override _C  (fused norm+quant, feeds qkv/gate_up)
//   dynamic_per_token_scaled_fp8_quant         -- override _C  (o_proj + down_proj-quant-half)
//   silu_and_mul_dynamic_per_token_quant       -- NEW op, vllm_cuda only (down_proj fusion target;
//                                                 inserted into the graph by a fusion pattern, so
//                                                 its fake/meta kernel is registered in Python).

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

void mul_sigmoid_dynamic_per_token_quant(torch::Tensor& result, torch::Tensor& scale,
                                         torch::Tensor const& input, torch::Tensor const& gate,
                                         std::optional<torch::Tensor> const& scale_ub);

void gated_rmsnorm(torch::Tensor& result, torch::Tensor const& input,
                   torch::Tensor const& gate, torch::Tensor const& weight, double epsilon,
                   bool gate_is_silu);

void gated_rmsnorm_dynamic_per_token_quant(torch::Tensor& result, torch::Tensor& scale,
                                           torch::Tensor const& input, torch::Tensor const& gate,
                                           torch::Tensor const& weight, double epsilon,
                                           int64_t num_heads, bool gate_is_silu,
                                           std::optional<torch::Tensor> const& scale_ub);
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
      "mul_sigmoid_dynamic_per_token_quant(Tensor! result, Tensor! scale, "
      "Tensor input, Tensor gate, Tensor? scale_ub) -> ()");
  m.def(
      "gated_rmsnorm(Tensor! result, Tensor input, Tensor gate, Tensor weight, "
      "float epsilon, bool gate_is_silu) -> ()");
  m.def(
      "gated_rmsnorm_dynamic_per_token_quant(Tensor! result, Tensor! scale, Tensor input, "
      "Tensor gate, Tensor weight, float epsilon, int num_heads, bool gate_is_silu, "
      "Tensor? scale_ub) -> ()");
}

TORCH_LIBRARY_IMPL(vllm_cuda, CUDA, m) {
  m.impl("rms_norm_dynamic_per_token_quant", TORCH_FN(vtl::rms_norm_dynamic_per_token_quant));
  m.impl("dynamic_per_token_scaled_fp8_quant",
         TORCH_FN(vtl::dynamic_per_token_scaled_fp8_quant));
  m.impl("silu_and_mul_dynamic_per_token_quant",
         TORCH_FN(vtl::silu_and_mul_dynamic_per_token_quant));
  m.impl("mul_sigmoid_dynamic_per_token_quant",
         TORCH_FN(vtl::mul_sigmoid_dynamic_per_token_quant));
  m.impl("gated_rmsnorm", TORCH_FN(vtl::gated_rmsnorm));
  m.impl("gated_rmsnorm_dynamic_per_token_quant",
         TORCH_FN(vtl::gated_rmsnorm_dynamic_per_token_quant));
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
