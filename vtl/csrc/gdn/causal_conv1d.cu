// GDN causal depthwise conv1d -- custom CUDA for the qwen3_5 linear-attention conv path
// (conv_dim=6144 channels, kernel width 4, activation="silu"), on all 18 GDN layers.
//
// Two ops, both parity-gated and DEFAULT-OFF (VTL_ENABLE_GDN_KERNELS + VTL_GDN_CONV1D). The
// Python shim (vtl/patches/gdn_kernels.py) routes vLLM's `causal_conv1d_fn` / `causal_conv1d_update`
// to these ONLY when the incoming args are in the plain shape these kernels implement, and falls
// back to the stock Triton kernel otherwise -- so the messy paged/spec-decode cache bookkeeping
// (cache_indices, block_idx_*, pad/null slots, block_size_to_align) is handled by stock, never
// mis-replicated here. Parity vs stock is the hard gate before either is enabled.
//
// Convention (mamba / vLLM causal_conv1d): width W, causal, per channel d
//   y[d, t] = act( bias[d] + sum_{k=0..W-1} weight[d, k] * x[d, t - (W-1) + k] )
// so weight[d, W-1] multiplies the current input and weight[d, 0] the oldest (t-(W-1)); positions
// before a sequence's start come from its initial conv_state (the last W-1 inputs) or zero. The
// decode "update" state holds exactly those last W-1 inputs and shifts by one each step.
//
// Ops (own ops only; wired in Python):
//   gdn_causal_conv1d_update(Tensor! x, Tensor! conv_state, Tensor weight, Tensor? bias,
//                            bool silu) -> ()
//     x [B, D] in/out (mutated to y); conv_state [B, D, W-1] in/out (shifted); weight [D, W].
//   gdn_causal_conv1d_fn(Tensor! y, Tensor x, Tensor weight, Tensor? bias,
//                        Tensor query_start_loc, Tensor? initial_state,
//                        Tensor? has_initial_state, Tensor!? final_state, bool silu) -> ()
//     x [D, L] varlen (columns = tokens, concatenated over sequences); y [D, L] out;
//     query_start_loc [S+1]; initial_state/final_state [S, D, W-1]; has_initial_state [S] bool.

#include <torch/all.h>

#include "../fp8_common.cuh"  // c10 cuda includes, kernel_sync_enabled, launch-check macro

namespace vtl {

namespace {

__device__ __forceinline__ float silu_f(float y, bool on) {
  return on ? y / (1.0f + expf(-y)) : y;
}

// ---- decode update: one thread per (batch b, channel d) -------------------------------------
// Reads W-1 state taps + the new input, emits y, shifts the state. W is a template constant so
// the tap loop fully unrolls; qwen3_5 is W=4 (3 state taps).
template <typename scalar_t, int W>
__global__ void conv1d_update_kernel(scalar_t* __restrict__ x,          // [B, D] in/out
                                     scalar_t* __restrict__ conv_state,  // [B, D, W-1]
                                     scalar_t const* __restrict__ weight,  // [D, W]
                                     float const* __restrict__ bias,       // [D] or nullptr
                                     bool const silu, int const B, int const D) {
  int const idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= B * D) return;
  int const b = idx / D;
  int const d = idx % D;

  scalar_t const* w = weight + d * W;
  int64_t const st_base = (static_cast<int64_t>(b) * D + d) * (W - 1);

  float acc = bias != nullptr ? bias[d] : 0.0f;
  float taps[W - 1];
#pragma unroll
  for (int k = 0; k < W - 1; ++k) {
    taps[k] = static_cast<float>(conv_state[st_base + k]);
    acc += static_cast<float>(w[k]) * taps[k];  // w[0..W-2] hit the (oldest..newest) state
  }
  float const xin = static_cast<float>(x[idx]);
  acc += static_cast<float>(w[W - 1]) * xin;  // w[W-1] hits the current input

  x[idx] = static_cast<scalar_t>(silu_f(acc, silu));

  // Shift the window: drop the oldest, append the new input.
#pragma unroll
  for (int k = 0; k < W - 2; ++k) conv_state[st_base + k] = static_cast<scalar_t>(taps[k + 1]);
  conv_state[st_base + (W - 2)] = static_cast<scalar_t>(xin);
}

// ---- prefill: one block per (sequence, channel-tile), one thread per channel -----------------
// Each thread owns a channel and walks that channel's tokens left-to-right, keeping a W-tap
// sliding window in registers (x[d, :] is contiguous, so the per-thread walk is sequential).
// Pre-seq taps come from initial_state when has_initial_state[s], else zero. After the sequence,
// the last W-1 inputs are written to final_state[s].
template <typename scalar_t, int W>
__global__ void conv1d_fn_kernel(scalar_t* __restrict__ y,             // [D, L]
                                 scalar_t const* __restrict__ x,       // [D, L]
                                 scalar_t const* __restrict__ weight,  // [D, W]
                                 float const* __restrict__ bias,       // [D] or nullptr
                                 int const* __restrict__ qsl,          // [S+1]
                                 scalar_t const* __restrict__ init_state,   // [S, D, W-1] or null
                                 bool const* __restrict__ has_init,         // [S] or null
                                 scalar_t* __restrict__ final_state,        // [S, D, W-1] or null
                                 bool const silu, int const D, int64_t const L) {
  int const s = blockIdx.x;
  int const d = blockIdx.y * blockDim.x + threadIdx.x;
  if (d >= D) return;

  int const start = qsl[s];
  int const end = qsl[s + 1];
  // Do NOT early-return on an empty sequence (start==end): the token loop is simply skipped, but
  // we must still write final_state = the seeded window (= passthrough of the initial conv state),
  // or it would be left uninitialised (garbage).

  scalar_t const* w = weight + d * W;
  float wr[W];
#pragma unroll
  for (int k = 0; k < W; ++k) wr[k] = static_cast<float>(w[k]);
  float const b = bias != nullptr ? bias[d] : 0.0f;

  // Sliding window of the last W inputs, window[W-1] = current. Seed the pre-seq taps into
  // window[1..W-1] (NOT window[0..W-2]): the loop shifts left *before* appending the first
  // input, so the taps must sit one slot high to land at window[0..W-2] on token 0. Seeding
  // window[0..W-2] would shift is0 out entirely -> token-0 output off by one tap.
  float window[W];
#pragma unroll
  for (int k = 0; k < W; ++k) window[k] = 0.0f;
  bool const use_init = has_init != nullptr && init_state != nullptr && has_init[s];
  if (use_init) {
    int64_t const is_base = (static_cast<int64_t>(s) * D + d) * (W - 1);
#pragma unroll
    for (int k = 0; k < W - 1; ++k)
      window[k + 1] = static_cast<float>(init_state[is_base + k]);  // oldest..newest -> slots 1..W-1
  }

  int64_t const row = static_cast<int64_t>(d) * L;
  for (int t = start; t < end; ++t) {
    // shift in the current input
#pragma unroll
    for (int k = 0; k < W - 1; ++k) window[k] = window[k + 1];
    window[W - 1] = static_cast<float>(x[row + t]);

    float acc = b;
#pragma unroll
    for (int k = 0; k < W; ++k) acc += wr[k] * window[k];
    y[row + t] = static_cast<scalar_t>(silu_f(acc, silu));
  }

  if (final_state != nullptr) {
    int64_t const fs_base = (static_cast<int64_t>(s) * D + d) * (W - 1);
#pragma unroll
    for (int k = 0; k < W - 1; ++k)
      final_state[fs_base + k] = static_cast<scalar_t>(window[k + 1]);  // last W-1 inputs
  }
}

template <typename scalar_t>
void launch_update(torch::Tensor& x, torch::Tensor& conv_state, torch::Tensor const& weight,
                   std::optional<torch::Tensor> const& bias, bool silu, cudaStream_t stream) {
  int const B = x.size(0);
  int const D = x.size(1);
  int const W = weight.size(1);
  auto const* bias_p = bias.has_value() ? bias->const_data_ptr<float>() : nullptr;
  int const threads = 256;
  int const blocks = (B * D + threads - 1) / threads;
  TORCH_CHECK(W == 4, "vtl gdn conv1d: only width 4 is compiled (qwen3_5)");
  conv1d_update_kernel<scalar_t, 4><<<blocks, threads, 0, stream>>>(
      x.data_ptr<scalar_t>(), conv_state.data_ptr<scalar_t>(),
      weight.const_data_ptr<scalar_t>(), bias_p, silu, B, D);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_fn(torch::Tensor& y, torch::Tensor const& x, torch::Tensor const& weight,
               std::optional<torch::Tensor> const& bias, torch::Tensor const& qsl,
               std::optional<torch::Tensor> const& init_state,
               std::optional<torch::Tensor> const& has_init,
               std::optional<torch::Tensor> const& final_state, bool silu,
               cudaStream_t stream) {
  int const D = x.size(0);
  int64_t const L = x.size(1);
  int const W = weight.size(1);
  int const S = qsl.numel() - 1;
  TORCH_CHECK(W == 4, "vtl gdn conv1d: only width 4 is compiled (qwen3_5)");
  auto const* bias_p = bias.has_value() ? bias->const_data_ptr<float>() : nullptr;
  auto const* is_p = init_state.has_value() ? init_state->const_data_ptr<scalar_t>() : nullptr;
  auto const* hi_p = has_init.has_value() ? has_init->const_data_ptr<bool>() : nullptr;
  auto* fs_p = final_state.has_value() ? final_state->data_ptr<scalar_t>() : nullptr;

  int const threads = 128;
  dim3 const grid(S, (D + threads - 1) / threads);
  conv1d_fn_kernel<scalar_t, 4><<<grid, threads, 0, stream>>>(
      y.data_ptr<scalar_t>(), x.const_data_ptr<scalar_t>(), weight.const_data_ptr<scalar_t>(),
      bias_p, qsl.const_data_ptr<int>(), is_p, hi_p, fs_p, silu, D, L);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void gdn_causal_conv1d_update(torch::Tensor& x, torch::Tensor& conv_state,
                              torch::Tensor const& weight,
                              std::optional<torch::Tensor> const& bias, bool silu) {
  TORCH_CHECK(x.dim() == 2 && conv_state.dim() == 3, "vtl gdn conv1d update: x [B,D], state [B,D,W-1]");
  TORCH_CHECK(x.is_contiguous() && conv_state.is_contiguous() && weight.is_contiguous(),
              "vtl gdn conv1d update: contiguous x/state/weight required");
  TORCH_CHECK(weight.dim() == 2 && weight.size(1) == 4, "vtl gdn conv1d update: weight [D,4]");
  int const B = x.size(0), D = x.size(1);
  TORCH_CHECK(conv_state.size(0) == B && conv_state.size(1) == D &&
                  conv_state.size(2) == weight.size(1) - 1,
              "vtl gdn conv1d update: conv_state must be [B,D,W-1]");
  TORCH_CHECK(weight.size(0) == D && weight.scalar_type() == x.scalar_type() &&
                  conv_state.scalar_type() == x.scalar_type(),
              "vtl gdn conv1d update: weight/state must be [D,*] and match x dtype");
  // bias is read as f32 in the kernel; a bf16 bias would be reinterpreted -> require f32 so a
  // mismatch is a clean error the interceptor catches and bails to stock on, not silent garbage.
  if (bias.has_value())
    TORCH_CHECK(bias->scalar_type() == at::ScalarType::Float && bias->numel() == D,
                "vtl gdn conv1d update: bias must be f32 [D]");
  c10::cuda::OptionalCUDAGuard const g(x.device());
  cudaStream_t const stream = c10::cuda::getCurrentCUDAStream();
  switch (x.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_update<c10::BFloat16>(x, conv_state, weight, bias, silu, stream);
      break;
    case at::ScalarType::Half:
      launch_update<c10::Half>(x, conv_state, weight, bias, silu, stream);
      break;
    case at::ScalarType::Float:
      launch_update<float>(x, conv_state, weight, bias, silu, stream);
      break;
    default:
      TORCH_CHECK(false, "vtl gdn conv1d update: unsupported dtype ", x.scalar_type());
  }
  if (kernel_sync_enabled()) {
    cudaError_t const e = cudaStreamSynchronize(stream);
    TORCH_CHECK(e == cudaSuccess, "vtl gdn_causal_conv1d_update faulted: ", cudaGetErrorString(e));
  }
}

void gdn_causal_conv1d_fn(torch::Tensor& y, torch::Tensor const& x, torch::Tensor const& weight,
                          std::optional<torch::Tensor> const& bias,
                          torch::Tensor const& query_start_loc,
                          std::optional<torch::Tensor> const& initial_state,
                          std::optional<torch::Tensor> const& has_initial_state,
                          std::optional<torch::Tensor> const& final_state, bool silu) {
  TORCH_CHECK(x.dim() == 2 && y.dim() == 2, "vtl gdn conv1d fn: x/y [D,L]");
  TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && weight.is_contiguous() &&
                  query_start_loc.is_contiguous(),
              "vtl gdn conv1d fn: contiguous x/y/weight/qsl required");
  TORCH_CHECK(x.sizes() == y.sizes(), "vtl gdn conv1d fn: x and y shapes differ");
  TORCH_CHECK(query_start_loc.scalar_type() == at::ScalarType::Int,
              "vtl gdn conv1d fn: query_start_loc must be int32");
  int const D = x.size(0), S = static_cast<int>(query_start_loc.numel()) - 1;
  TORCH_CHECK(S >= 1, "vtl gdn conv1d fn: query_start_loc must have >= 2 entries");
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) == D && weight.size(1) == 4 &&
                  weight.scalar_type() == x.scalar_type(),
              "vtl gdn conv1d fn: weight must be [D,4] matching x dtype");
  if (bias.has_value())
    TORCH_CHECK(bias->scalar_type() == at::ScalarType::Float && bias->numel() == D,
                "vtl gdn conv1d fn: bias must be f32 [D]");
  auto check_state = [&](std::optional<torch::Tensor> const& s, char const* nm) {
    if (s.has_value())
      TORCH_CHECK(s->dim() == 3 && s->size(0) == S && s->size(1) == D && s->size(2) == 3 &&
                      s->scalar_type() == x.scalar_type() && s->is_contiguous(),
                  "vtl gdn conv1d fn: ", nm, " must be contiguous [S,D,W-1] matching x dtype");
  };
  check_state(initial_state, "initial_state");
  check_state(final_state, "final_state");
  if (has_initial_state.has_value())
    TORCH_CHECK(has_initial_state->scalar_type() == at::ScalarType::Bool &&
                    has_initial_state->numel() == S,
                "vtl gdn conv1d fn: has_initial_state must be bool [S]");
  c10::cuda::OptionalCUDAGuard const g(x.device());
  cudaStream_t const stream = c10::cuda::getCurrentCUDAStream();
  switch (x.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_fn<c10::BFloat16>(y, x, weight, bias, query_start_loc, initial_state,
                               has_initial_state, final_state, silu, stream);
      break;
    case at::ScalarType::Half:
      launch_fn<c10::Half>(y, x, weight, bias, query_start_loc, initial_state,
                           has_initial_state, final_state, silu, stream);
      break;
    case at::ScalarType::Float:
      launch_fn<float>(y, x, weight, bias, query_start_loc, initial_state, has_initial_state,
                       final_state, silu, stream);
      break;
    default:
      TORCH_CHECK(false, "vtl gdn conv1d fn: unsupported dtype ", x.scalar_type());
  }
  if (kernel_sync_enabled()) {
    cudaError_t const e = cudaStreamSynchronize(stream);
    TORCH_CHECK(e == cudaSuccess, "vtl gdn_causal_conv1d_fn faulted: ", cudaGetErrorString(e));
  }
}

}  // namespace vtl
