// Two registrations for one kernel:
//
//   vllm_cuda::rms_norm_dynamic_per_token_quant  -- our own op, so tests and benchmarks
//     can call it side by side with the stock one.
//
//   _C::rms_norm_dynamic_per_token_quant (CUDA)  -- overrides vLLM's CUDA kernel for the
//     op the RMSNormQuantFusionPass already emits. Same op identity, so FUSED_OPS,
//     FixFunctionalizationPass, the meta kernel and the torch.compile cache key are all
//     untouched; only the kernel behind the dispatch key changes. Registering in C++
//     rather than via torch.library.Library keeps the hot path free of Python dispatch.
//
// Overriding is what the last registration for a (op, dispatch key) pair does; PyTorch
// warns about it. It requires vLLM's `_C` schema to already exist, which is why
// vtl/patches/rms_norm_quant.py imports vllm._C_stable_libtorch before vtl._C.

#include <torch/extension.h>
#include <torch/library.h>

namespace vtl {
void rms_norm_dynamic_per_token_quant(torch::Tensor& result, torch::Tensor const& input,
                                      torch::Tensor const& weight, torch::Tensor& scale,
                                      double epsilon,
                                      std::optional<torch::Tensor> const& scale_ub,
                                      std::optional<torch::Tensor> const& residual);
}  // namespace vtl

TORCH_LIBRARY(vllm_cuda, m) {
  m.def(
      "rms_norm_dynamic_per_token_quant(Tensor! result, Tensor input, "
      "Tensor weight, Tensor! scale, float epsilon, "
      "Tensor? scale_ub, Tensor!? residual) -> ()");
}

TORCH_LIBRARY_IMPL(vllm_cuda, CUDA, m) {
  m.impl("rms_norm_dynamic_per_token_quant", TORCH_FN(vtl::rms_norm_dynamic_per_token_quant));
}

TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("rms_norm_dynamic_per_token_quant", TORCH_FN(vtl::rms_norm_dynamic_per_token_quant));
}

// The ops arrive through the static initializers above, at dlopen. This exists only so
// that `import vtl._C` finds a PyInit__C and does not raise.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
