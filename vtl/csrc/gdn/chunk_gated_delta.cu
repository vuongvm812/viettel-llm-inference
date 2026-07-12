// GDN prefill: gated-delta-rule scan over a varlen batch, for the qwen3_5 linear-attention
// layers. DEFAULT-OFF, parity-gated (VTL_ENABLE_GDN_KERNELS + VTL_GDN_CHUNK_SCAN); the Python
// shim routes vLLM's chunk_gated_delta_rule prefill to this ONLY when the layout matches, else
// falls back to the stock Triton/CuteDSL kernel.
//
// SCOPE / honest ceiling: this is a SEQUENTIAL per-sequence recurrence (the exact same gated
// delta rule as the decode step, applied token-by-token), NOT the chunk-parallel WY-representation
// algorithm vLLM's Triton/CuteDSL kernels use. It is correctness-first: numerically it matches the
// recurrence, but it does not exploit intra-sequence parallelism, so on long prefills it will not
// beat the chunked incumbents. It exists to (a) pass parity as the reference CUDA implementation
// and (b) be the base the chunk/WY parallelization (chunk_delta_h / solve_tril / chunk_o) is built
// on IF the Phase-0 profile shows the scan dominant. Enable only where the on-box bench wins.
//
// Recurrence (per head, state S = [Dk, Dv] f32) -- same contract as fused_recurrent_decode.cu,
// which MUST be reconciled with vLLM v0.25.0 on the box:
//   (optional) q,k <- l2norm; S <- S*exp(g_t); kS = k^T S; vnew = v - kS;
//   S <- S + beta * k vnew^T; o_t = q^T S.
// g_t is assumed per-STEP log-decay (as in the decode path). If vLLM passes a cumulative gate,
// the shim must convert it or bail -- documented in gdn_kernels.py.
//
// State lives in the caller's `final_state` buffer (seeded from `initial_state` or zero, updated
// in place per token, left holding the final state). One block per (sequence, head); thread j owns
// column j of S so state R/W is coalesced. Two passes over Dk rows per token.
//
// Op (own op only; wired in Python):
//   gdn_chunk_scan(Tensor! o, Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta,
//                  Tensor query_start_loc, Tensor? initial_state, Tensor! final_state,
//                  bool qk_l2norm) -> ()
//   q,k [L,H,Dk]; v,o [L,H,Dv]; g,beta [L,H]; qsl [S+1]; init/final_state [S,H,Dk,Dv] f32.

#include <torch/all.h>

#include "../fp8_common.cuh"  // block_reduce, AddOp, cuda infra, kernel_sync_enabled

namespace vtl {

namespace {

constexpr float kL2Eps = 1e-6f;

template <typename scalar_t>
__global__ void chunk_scan_kernel(scalar_t* __restrict__ o,           // [L,H,Dv]
                                  float* __restrict__ final_state,     // [S,H,Dk,Dv] working+out
                                  scalar_t const* __restrict__ q,      // [L,H,Dk]
                                  scalar_t const* __restrict__ k,      // [L,H,Dk]
                                  scalar_t const* __restrict__ v,      // [L,H,Dv]
                                  float const* __restrict__ g,         // [L,H]
                                  float const* __restrict__ beta,      // [L,H]
                                  int const* __restrict__ qsl,         // [S+1]
                                  float const* __restrict__ init_state,  // [S,H,Dk,Dv] or null
                                  bool const l2norm, int const H, int const Dk, int const Dv) {
  int const s = blockIdx.x;
  int const h = blockIdx.y;
  int const j = threadIdx.x;  // column of S / output index, blockDim.x == Dv
  if (j >= Dv) return;

  extern __shared__ float smem[];
  float* q_s = smem;           // [Dk]
  float* k_s = smem + Dk;      // [Dk]
  float* red = smem + 2 * Dk;  // [32]

  int64_t const st_base = (static_cast<int64_t>(s) * H + h) * Dk * Dv;  // column-j elems at +i*Dv+j

  // Seed the working state column j from initial_state (or zero).
  for (int i = 0; i < Dk; ++i) {
    int64_t const off = st_base + static_cast<int64_t>(i) * Dv + j;
    final_state[off] = init_state != nullptr ? init_state[off] : 0.0f;
  }

  int const start = qsl[s];
  int const end = qsl[s + 1];

  for (int t = start; t < end; ++t) {
    int64_t const qk_base = (static_cast<int64_t>(t) * H + h) * Dk;
    int64_t const v_base = (static_cast<int64_t>(t) * H + h) * Dv;
    int const gb = t * H + h;

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

    float const decay = expf(g[gb]);
    float const b = beta[gb];
    float const vj = static_cast<float>(v[v_base + j]);

    // Pass A: accumulate kS without writing back; pass B recomputes the decay from the
    // still-original state, so the state is written once per token, not twice (memory-bound).
    float kS = 0.0f;
    for (int i = 0; i < Dk; ++i) {
      int64_t const off = st_base + static_cast<int64_t>(i) * Dv + j;
      kS += k_s[i] * (final_state[off] * decay);
    }
    float const vnew = vj - kS;

    float o_acc = 0.0f;
    for (int i = 0; i < Dk; ++i) {
      int64_t const off = st_base + static_cast<int64_t>(i) * Dv + j;
      float const sval = final_state[off] * decay + b * k_s[i] * vnew;
      final_state[off] = sval;
      o_acc += q_s[i] * sval;
    }
    o[v_base + j] = static_cast<scalar_t>(o_acc);
    __syncthreads();  // q_s/k_s reused next token
  }
}

template <typename scalar_t>
void launch(torch::Tensor& o, torch::Tensor& final_state, torch::Tensor const& q,
            torch::Tensor const& k, torch::Tensor const& v, torch::Tensor const& g,
            torch::Tensor const& beta, torch::Tensor const& qsl,
            std::optional<torch::Tensor> const& initial_state, bool l2norm,
            cudaStream_t stream) {
  int const H = q.size(1);
  int const Dk = q.size(2);
  int const Dv = v.size(2);
  int const S = qsl.numel() - 1;
  TORCH_CHECK(Dk == Dv, "vtl gdn chunk scan: Dk must equal Dv (block layout assumes it)");
  auto const* is_p = initial_state.has_value() ? initial_state->const_data_ptr<float>() : nullptr;
  size_t const shmem = (2 * Dk + 32) * sizeof(float);
  dim3 const grid(S, H);
  chunk_scan_kernel<scalar_t><<<grid, Dv, shmem, stream>>>(
      o.data_ptr<scalar_t>(), final_state.data_ptr<float>(), q.const_data_ptr<scalar_t>(),
      k.const_data_ptr<scalar_t>(), v.const_data_ptr<scalar_t>(), g.const_data_ptr<float>(),
      beta.const_data_ptr<float>(), qsl.const_data_ptr<int>(), is_p, l2norm, H, Dk, Dv);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void gdn_chunk_scan(torch::Tensor& o, torch::Tensor const& q, torch::Tensor const& k,
                    torch::Tensor const& v, torch::Tensor const& g, torch::Tensor const& beta,
                    torch::Tensor const& query_start_loc,
                    std::optional<torch::Tensor> const& initial_state,
                    torch::Tensor& final_state, bool qk_l2norm) {
  TORCH_CHECK(q.dim() == 3 && v.dim() == 3 && final_state.dim() == 4, "vtl gdn chunk: bad ranks");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && o.is_contiguous() &&
                  final_state.is_contiguous() && g.is_contiguous() && beta.is_contiguous() &&
                  query_start_loc.is_contiguous(),
              "vtl gdn chunk: contiguous tensors required");
  TORCH_CHECK(final_state.scalar_type() == at::ScalarType::Float, "vtl gdn chunk: state f32");
  TORCH_CHECK(g.scalar_type() == at::ScalarType::Float && beta.scalar_type() == at::ScalarType::Float,
              "vtl gdn chunk: g/beta must be f32");
  TORCH_CHECK(query_start_loc.scalar_type() == at::ScalarType::Int,
              "vtl gdn chunk: query_start_loc must be int32");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type() &&
                  q.scalar_type() == o.scalar_type(),
              "vtl gdn chunk: q/k/v/o dtype mismatch");
  // Validate everything the kernel indexes off -- a short tensor or a bad S is an OOB write.
  int const L = q.size(0), H = q.size(1), Dk = q.size(2), Dv = v.size(2);
  int const S = static_cast<int>(query_start_loc.numel()) - 1;
  TORCH_CHECK(S >= 1, "vtl gdn chunk: query_start_loc must have >= 2 entries (S>=1)");
  TORCH_CHECK(k.sizes() == q.sizes(), "vtl gdn chunk: k must match q [L,H,Dk]");
  TORCH_CHECK(o.sizes() == v.sizes(), "vtl gdn chunk: o must match v [L,H,Dv]");
  TORCH_CHECK(v.size(0) == L && v.size(1) == H, "vtl gdn chunk: v [L,H,Dv] mismatch");
  TORCH_CHECK(g.dim() == 2 && g.size(0) == L && g.size(1) == H, "vtl gdn chunk: g must be [L,H]");
  TORCH_CHECK(beta.sizes() == g.sizes(), "vtl gdn chunk: beta must be [L,H]");
  TORCH_CHECK(final_state.size(0) == S && final_state.size(1) == H &&
                  final_state.size(2) == Dk && final_state.size(3) == Dv,
              "vtl gdn chunk: final_state must be [S,H,Dk,Dv]");
  if (initial_state.has_value())
    TORCH_CHECK(initial_state->sizes() == final_state.sizes() &&
                    initial_state->scalar_type() == at::ScalarType::Float,
                "vtl gdn chunk: initial_state must be f32 [S,H,Dk,Dv]");
  TORCH_CHECK(Dk == Dv && Dv <= 1024, "vtl gdn chunk: need Dk==Dv and Dv<=1024 (blockDim)");
  // NOTE: query_start_loc VALUES (monotone, in [0,L]) are the varlen-metadata caller's contract;
  // validating them here would force a device->host sync. The kernel indexes tokens in
  // [qsl[s], qsl[s+1]); a caller that violates the contract is a bug in the varlen metadata.

  c10::cuda::OptionalCUDAGuard const guard(q.device());
  cudaStream_t const stream = c10::cuda::getCurrentCUDAStream();
  switch (q.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch<c10::BFloat16>(o, final_state, q, k, v, g, beta, query_start_loc, initial_state,
                            qk_l2norm, stream);
      break;
    case at::ScalarType::Half:
      launch<c10::Half>(o, final_state, q, k, v, g, beta, query_start_loc, initial_state,
                        qk_l2norm, stream);
      break;
    case at::ScalarType::Float:
      launch<float>(o, final_state, q, k, v, g, beta, query_start_loc, initial_state, qk_l2norm,
                    stream);
      break;
    default:
      TORCH_CHECK(false, "vtl gdn chunk: unsupported dtype ", q.scalar_type());
  }
  if (kernel_sync_enabled()) {
    cudaError_t const e = cudaStreamSynchronize(stream);
    TORCH_CHECK(e == cudaSuccess, "vtl gdn_chunk_scan faulted: ", cudaGetErrorString(e));
  }
}

}  // namespace vtl
