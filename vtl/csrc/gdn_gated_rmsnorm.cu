// Per-head gated RMSNorm for the GDN (linear-attention) output path -- the `linear_attn.norm`
// (weight [head_dim]=[128], F32) applied to the gated-delta core output before out_proj, on all
// 18 GDN layers. This is the tractable, self-contained GDN target from Phase 3 (the scan/conv
// are the harder, profile-gated ones).
//
// ASSUMED semantics (standard Mamba2 / vLLM RMSNormGated, forward_native):
//   h   = f32(x) * silu(f32(gate))            silu(z) = z * sigmoid(z)   -- gate in fp32
//   rms = rsqrt(mean_D(h^2) + eps)            reduction over the last dim (head_dim)
//   out = scalar_t( h * rms * weight )        weight is fp32; narrow once at the end
// !!! The exact cast points in vLLM v0.25.0's RMSNormGated must be confirmed by the parity test
// (bench/test_gdn_gated_rmsnorm.py) on the H200 before enabling this kernel. It is gated OFF by
// default (VTL_ENABLE_GDN_KERNELS) until that passes. A group/n_groups variant would change the
// reduction width -- head_dim here is the per-head norm, groups=1.
//
// Op (own op only; wired in Python by replacing RMSNormGated.forward_cuda):
//   gated_rmsnorm(Tensor! result, Tensor input, Tensor gate, Tensor weight, float epsilon) -> ()
//   input/gate/result = [num_rows, D] (num_rows = num_tokens * num_v_heads); weight = [D] fp32.

#include <torch/all.h>

#include "fp8_common.cuh"

namespace vtl {

namespace {

// One block per row (= one (token, head)); one thread per element when D <= 1024, so each
// thread reads x and gate once and holds h in a register across the single sum reduction.
template <typename scalar_t>
__global__ void __launch_bounds__(1024) gated_rmsnorm_kernel(
    scalar_t* __restrict__ out, scalar_t const* __restrict__ input,
    scalar_t const* __restrict__ gate, float const* __restrict__ weight, float const epsilon,
    int const D) {
  int64_t const row = blockIdx.x;
  int const i = threadIdx.x;
  int64_t const off = row * static_cast<int64_t>(D) + i;
  bool const active = i < D;

  float h = 0.0f;
  if (active) {
    float const x = static_cast<float>(input[off]);
    float const z = static_cast<float>(gate[off]);
    h = x * (z / (1.0f + expf(-z)));  // x * silu(gate), in fp32
  }

  __shared__ float smem[32];
  float const ss = block_reduce(h * h, AddOp{}, 0.0f, smem);
  float const rms = rsqrtf(ss / D + epsilon);

  if (active) {
    out[off] = static_cast<scalar_t>(h * rms * weight[i]);
  }
}

// Fallback for D > 1024 (not the qwen3_5 case, head_dim=128): strided, two passes.
template <typename scalar_t>
__global__ void gated_rmsnorm_generic_kernel(
    scalar_t* __restrict__ out, scalar_t const* __restrict__ input,
    scalar_t const* __restrict__ gate, float const* __restrict__ weight, float const epsilon,
    int const D) {
  int64_t const base = static_cast<int64_t>(blockIdx.x) * D;
  __shared__ float smem[32];

  float ss = 0.0f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float const x = static_cast<float>(input[base + i]);
    float const z = static_cast<float>(gate[base + i]);
    float const h = x * (z / (1.0f + expf(-z)));
    ss += h * h;
  }
  ss = block_reduce(ss, AddOp{}, 0.0f, smem);
  float const rms = rsqrtf(ss / D + epsilon);

  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float const x = static_cast<float>(input[base + i]);
    float const z = static_cast<float>(gate[base + i]);
    float const h = x * (z / (1.0f + expf(-z)));
    out[base + i] = static_cast<scalar_t>(h * rms * weight[i]);
  }
}

template <typename scalar_t>
void launch(torch::Tensor const& out, torch::Tensor const& input, torch::Tensor const& gate,
            torch::Tensor const& weight, double epsilon, cudaStream_t stream) {
  int const D = input.size(-1);
  int64_t const num_rows = input.numel() / D;
  if (num_rows == 0) return;

  auto* out_p = out.data_ptr<scalar_t>();
  auto const* in_p = input.const_data_ptr<scalar_t>();
  auto const* gate_p = gate.const_data_ptr<scalar_t>();
  auto const* w_p = weight.const_data_ptr<float>();

  dim3 const grid(num_rows);
  if (D <= 1024) {
    dim3 const block((D + 31) / 32 * 32);
    gated_rmsnorm_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out_p, in_p, gate_p, w_p, static_cast<float>(epsilon), D);
  } else {
    dim3 const block(1024);
    gated_rmsnorm_generic_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out_p, in_p, gate_p, w_p, static_cast<float>(epsilon), D);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (kernel_sync_enabled()) {
    cudaError_t const err = cudaStreamSynchronize(stream);
    TORCH_CHECK(err == cudaSuccess, "vtl gated_rmsnorm faulted: ", cudaGetErrorString(err),
                " | dtype=", input.scalar_type(), " num_rows=", num_rows, " D=", D);
  }
}

}  // namespace

void gated_rmsnorm(torch::Tensor& result, torch::Tensor const& input,
                   torch::Tensor const& gate, torch::Tensor const& weight, double epsilon) {
  TORCH_CHECK(input.is_contiguous() && gate.is_contiguous() && result.is_contiguous(),
              "vtl: gated_rmsnorm requires contiguous input/gate/result");
  TORCH_CHECK(input.sizes() == gate.sizes() && input.sizes() == result.sizes(),
              "vtl: gated_rmsnorm input/gate/result must have equal shape");
  TORCH_CHECK(weight.scalar_type() == at::ScalarType::Float, "vtl: gated_rmsnorm weight fp32");
  TORCH_CHECK(weight.numel() == input.size(-1), "vtl: gated_rmsnorm weight must be [D]");
  TORCH_CHECK(input.scalar_type() == result.scalar_type() &&
                  input.scalar_type() == gate.scalar_type(),
              "vtl: gated_rmsnorm input/gate/result dtype mismatch");

  c10::cuda::OptionalCUDAGuard const device_guard(input.device());
  cudaStream_t const stream = c10::cuda::getCurrentCUDAStream();

  switch (input.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch<c10::BFloat16>(result, input, gate, weight, epsilon, stream);
      break;
    case at::ScalarType::Half:
      launch<c10::Half>(result, input, gate, weight, epsilon, stream);
      break;
    case at::ScalarType::Float:
      launch<float>(result, input, gate, weight, epsilon, stream);
      break;
    default:
      TORCH_CHECK(false, "vtl: unsupported input dtype ", input.scalar_type());
  }
}

}  // namespace vtl
