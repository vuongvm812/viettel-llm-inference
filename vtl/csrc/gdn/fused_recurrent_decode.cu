// GDN decode: one gated-delta-rule recurrence step per sequence (the packed-decode path),
// for the qwen3_5 linear-attention layers. DEFAULT-OFF, parity-gated (VTL_ENABLE_GDN_KERNELS +
// VTL_GDN_RECURRENT); the Python shim routes vLLM's fused_recurrent_gated_delta_rule_packed_decode
// to this ONLY when the tensor layout matches, else falls back to the stock Triton kernel.
//
// Gated delta rule (per head, state S = [Dk, Dv], f32), assumed formula -- MUST be reconciled
// with vLLM v0.25.0's kernel on the box (this is the load-bearing numeric contract):
//   (optional) q,k <- l2norm over Dk
//   S      <- S * exp(g)                         # g = per-head log-decay for this step
//   kS_j    = sum_i k_i * S_ij                    # old prediction, [Dv]
//   vnew_j  = v_j - kS_j                          # delta
//   S_ij   <- S_ij + beta * k_i * vnew_j          # rank-1 update
//   o_j     = sum_i q_i * S_ij                     # output, [Dv]
// Dk = Dv = head_dim = 128 (== blockDim). One block per (sequence, head); thread j owns column j
// of S, so state reads/writes are coalesced across threads. Two passes over the Dk rows: pass A
// decays S in place and accumulates kS; pass B applies the update and accumulates the output.
//
// Op (own op only; wired in Python):
//   gdn_recurrent_decode(Tensor! o, Tensor! state, Tensor q, Tensor k, Tensor v, Tensor g,
//                        Tensor beta, bool qk_l2norm) -> ()
//   q,k [N,H,Dk]; v,o [N,H,Dv]; g,beta [N,H]; state [N,H,Dk,Dv] f32.

#include <torch/all.h>

#include "../fp8_common.cuh"  // block_reduce, AddOp, cuda infra, kernel_sync_enabled

namespace vtl {

namespace {

constexpr float kL2Eps = 1e-6f;

template <typename scalar_t>
__global__ void recurrent_decode_kernel(scalar_t* __restrict__ o,      // [N,H,Dv]
                                        float* __restrict__ state,     // [N,H,Dk,Dv] f32
                                        scalar_t const* __restrict__ q,     // [N,H,Dk]
                                        scalar_t const* __restrict__ k,     // [N,H,Dk]
                                        scalar_t const* __restrict__ v,     // [N,H,Dv]
                                        float const* __restrict__ g,        // [N,H]
                                        float const* __restrict__ beta,     // [N,H]
                                        bool const l2norm, int const H, int const Dk,
                                        int const Dv) {
  int const nh = blockIdx.x;          // n*H + h
  int const j = threadIdx.x;          // column of S / output index, blockDim.x == Dv
  if (j >= Dv) return;

  extern __shared__ float smem[];
  float* q_s = smem;          // [Dk]
  float* k_s = smem + Dk;     // [Dk]
  float* red = smem + 2 * Dk;  // [32] for block_reduce

  int64_t const qk_base = static_cast<int64_t>(nh) * Dk;
  int64_t const v_base = static_cast<int64_t>(nh) * Dv;
  int64_t const s_base = static_cast<int64_t>(nh) * Dk * Dv;

  // Load q,k row into shared (thread j loads element j; valid because Dk == Dv == blockDim).
  float qj = static_cast<float>(q[qk_base + j]);
  float kj = static_cast<float>(k[qk_base + j]);
  if (l2norm) {
    float const nq = rsqrtf(block_reduce(qj * qj, AddOp{}, 0.0f, red) + kL2Eps);
    __syncthreads();
    float const nk = rsqrtf(block_reduce(kj * kj, AddOp{}, 0.0f, red) + kL2Eps);
    __syncthreads();
    qj *= nq;
    kj *= nk;
  }
  q_s[j] = qj;
  k_s[j] = kj;
  __syncthreads();

  float const decay = expf(g[nh]);
  float const b = beta[nh];
  float const vj = static_cast<float>(v[v_base + j]);

  // Pass A: accumulate kS_j = sum_i k_i * (decay * S_ij) WITHOUT writing back. Pass B recomputes
  // the decay from the still-original S, so we do one write of S over the two passes, not two
  // (S is memory-bound; this drops ~25% of its HBM traffic).
  float kS = 0.0f;
  for (int i = 0; i < Dk; ++i) {
    kS += k_s[i] * (state[s_base + static_cast<int64_t>(i) * Dv + j] * decay);
  }
  float const vnew = vj - kS;

  // Pass B: decayed state + rank-1 update (single write), accumulate o_j = sum_i q_i * S_ij.
  float o_acc = 0.0f;
  for (int i = 0; i < Dk; ++i) {
    int64_t const off = s_base + static_cast<int64_t>(i) * Dv + j;
    float const s = state[off] * decay + b * k_s[i] * vnew;
    state[off] = s;
    o_acc += q_s[i] * s;
  }
  o[v_base + j] = static_cast<scalar_t>(o_acc);
}

template <typename scalar_t>
void launch(torch::Tensor& o, torch::Tensor& state, torch::Tensor const& q,
            torch::Tensor const& k, torch::Tensor const& v, torch::Tensor const& g,
            torch::Tensor const& beta, bool l2norm, cudaStream_t stream) {
  int const N = q.size(0);
  int const H = q.size(1);
  int const Dk = q.size(2);
  int const Dv = v.size(2);
  TORCH_CHECK(Dk == Dv, "vtl gdn decode: Dk must equal Dv (block layout assumes it)");
  size_t const shmem = (2 * Dk + 32) * sizeof(float);
  recurrent_decode_kernel<scalar_t><<<N * H, Dv, shmem, stream>>>(
      o.data_ptr<scalar_t>(), state.data_ptr<float>(), q.const_data_ptr<scalar_t>(),
      k.const_data_ptr<scalar_t>(), v.const_data_ptr<scalar_t>(), g.const_data_ptr<float>(),
      beta.const_data_ptr<float>(), l2norm, H, Dk, Dv);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void gdn_recurrent_decode(torch::Tensor& o, torch::Tensor& state, torch::Tensor const& q,
                          torch::Tensor const& k, torch::Tensor const& v, torch::Tensor const& g,
                          torch::Tensor const& beta, bool qk_l2norm) {
  TORCH_CHECK(q.dim() == 3 && v.dim() == 3 && state.dim() == 4, "vtl gdn decode: bad ranks");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                  state.is_contiguous() && o.is_contiguous() && g.is_contiguous() &&
                  beta.is_contiguous(),
              "vtl gdn decode: contiguous tensors required");
  TORCH_CHECK(state.scalar_type() == at::ScalarType::Float, "vtl gdn decode: state must be f32");
  TORCH_CHECK(g.scalar_type() == at::ScalarType::Float && beta.scalar_type() == at::ScalarType::Float,
              "vtl gdn decode: g/beta must be f32");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type() &&
                  q.scalar_type() == o.scalar_type(),
              "vtl gdn decode: q/k/v/o dtype mismatch");
  // Every kernel index is derived from these, so validate them all -- a short tensor is an OOB
  // device write, not a wrong answer.
  int const N = q.size(0), H = q.size(1), Dk = q.size(2), Dv = v.size(2);
  TORCH_CHECK(k.sizes() == q.sizes(), "vtl gdn decode: k must match q [N,H,Dk]");
  TORCH_CHECK(o.sizes() == v.sizes(), "vtl gdn decode: o must match v [N,H,Dv]");
  TORCH_CHECK(v.size(0) == N && v.size(1) == H, "vtl gdn decode: v [N,H,Dv] mismatch");
  TORCH_CHECK(g.dim() == 2 && g.size(0) == N && g.size(1) == H, "vtl gdn decode: g must be [N,H]");
  TORCH_CHECK(beta.sizes() == g.sizes(), "vtl gdn decode: beta must be [N,H]");
  TORCH_CHECK(state.size(0) == N && state.size(1) == H && state.size(2) == Dk &&
                  state.size(3) == Dv,
              "vtl gdn decode: state must be [N,H,Dk,Dv]");
  TORCH_CHECK(Dk == Dv && Dv <= 1024, "vtl gdn decode: need Dk==Dv and Dv<=1024 (blockDim)");
  if (N * H == 0) return;

  c10::cuda::OptionalCUDAGuard const guard(q.device());
  cudaStream_t const stream = c10::cuda::getCurrentCUDAStream();
  switch (q.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch<c10::BFloat16>(o, state, q, k, v, g, beta, qk_l2norm, stream);
      break;
    case at::ScalarType::Half:
      launch<c10::Half>(o, state, q, k, v, g, beta, qk_l2norm, stream);
      break;
    case at::ScalarType::Float:
      launch<float>(o, state, q, k, v, g, beta, qk_l2norm, stream);
      break;
    default:
      TORCH_CHECK(false, "vtl gdn decode: unsupported dtype ", q.scalar_type());
  }
  if (kernel_sync_enabled()) {
    cudaError_t const e = cudaStreamSynchronize(stream);
    TORCH_CHECK(e == cudaSuccess, "vtl gdn_recurrent_decode faulted: ", cudaGetErrorString(e));
  }
}

}  // namespace vtl
