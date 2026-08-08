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
// Ops. All three are architecture-independent: they key off tensor shapes, not off any
// particular model's block structure, so they apply to whatever model is mounted at /model.
//   rms_norm_dynamic_per_token_quant           -- override _C  (fused norm+quant; feeds the
//                                                 qkv and gate/up projections of any
//                                                 pre-norm transformer block)
//   dynamic_per_token_scaled_fp8_quant         -- override _C  (standalone activation quant,
//                                                 for inputs no norm immediately precedes)
//   silu_and_mul_dynamic_per_token_quant       -- NEW op, vllm_cuda only (SwiGLU down-proj
//                                                 fusion target; inserted into the graph by a
//                                                 fusion pattern, so its fake/meta kernel is
//                                                 registered in Python).
//
// ADDING AN OP: declare it in namespace vtl below, m.def() its schema in TORCH_LIBRARY, and
// m.impl() it in the CUDA block. Add the .cu to setup.py's sources. Predicates with no tensor
// arguments cannot be dispatched by device key -- register those as catch-alls in TORCH_LIBRARY
// itself (see the note there). For a kernel you want to iterate on without a full image rebuild,
// use the NVRTC path (vtl/nvrtc.py + vtl/kernels/) instead of this file.

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
  // Shape/capacity predicates belong here rather than in the CUDA block below: with no tensor
  // arguments they cannot be dispatched by device key, so they register as catch-alls, e.g.
  //   m.def("my_kernel_supported(int dim) -> bool", TORCH_FN(vtl::my_kernel_supported));
  // Use this for anything only the built binary can answer (occupancy, scratch size), which
  // the Python gate cannot compute for itself.
}

TORCH_LIBRARY_IMPL(vllm_cuda, CUDA, m) {
  m.impl("rms_norm_dynamic_per_token_quant", TORCH_FN(vtl::rms_norm_dynamic_per_token_quant));
  m.impl("dynamic_per_token_scaled_fp8_quant",
         TORCH_FN(vtl::dynamic_per_token_scaled_fp8_quant));
  m.impl("silu_and_mul_dynamic_per_token_quant",
         TORCH_FN(vtl::silu_and_mul_dynamic_per_token_quant));
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
