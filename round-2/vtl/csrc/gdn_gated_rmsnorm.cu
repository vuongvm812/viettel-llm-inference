// Per-head gated RMSNorm for the GDN (linear-attention) output path -- `linear_attn.norm`
// (RMSNormGated, weight [head_v_dim]) applied per value head to the gated-delta core output
// before out_proj, on all 36 GDN layers of Qwen3.5-122B-A10B (HV=64 x DV=128 per layer).
//
// SEMANTICS -- matched to vLLM v0.25.0 RMSNormGated.forward_static / the FLA Triton
// rmsnorm_fn element-for-element, for the config qwen3_5 uses: norm_before_gate=True,
// group_size=None (one group = head_v_dim), activation in {silu, sigmoid}. Gate AFTER norm:
//   xf   = f32(x)                                     (per (token,head) row, len D)
//   rstd = rsqrtf(mean_D(xf^2) + eps)
//   y    = ((xf * rstd) * f32(w)) * act(f32(z))       all fp32; ONE narrowing at the end
//   act(z) = z*sigmoid(z) if gate_is_silu else sigmoid(z)
// NOTE the weight is f32(w) whatever its storage dtype: v0.25.0 creates the RMSNormGated
// weight in the MODEL dtype (bf16 on this checkpoint), and both stock implementations
// upcast it before the multiply. There is NO bf16 narrowing of the weight product here
// (unlike rms_norm_quant.cu's double rounding) -- the only narrowing is the final cast of
// y to the model dtype, which is the tensor the quant op then re-reads.
//
// Three ops (own vllm_cuda:: ops; wired in Python behind VTL_ENABLE_GDN_KERNELS):
//
//   gated_rmsnorm(Tensor! result, Tensor input, Tensor gate, Tensor weight, float epsilon,
//                 bool gate_is_silu) -> ()
//     Standalone norm. input/gate/result = [num_rows, D]; weight = [D] (fp32 or input dtype).
//
//   gated_rmsnorm_dynamic_per_token_quant(...)  -- round-1.1 compat entry point, kept for
//     parity/A-B only. Per-TOKEN dynamic fp8 quant over the flattened heads. Round-2's
//     checkpoint quantizes activations per (token, 128-group), so this is NOT the round-2
//     hot path; its warp fast path also caps at H<=32 (H=64 here takes the generic kernel).
//
//   gated_rmsnorm_per_group_quant(Tensor! result, Tensor! scale, Tensor input, Tensor gate,
//       Tensor weight, float epsilon, int num_heads, int group_size, bool gate_is_silu) -> ()
//     THE round-2 fusion target: per-head norm above, then fp8-e4m3 quant with per-
//     (token, 128-group) scales exactly as vLLM's `_C::per_token_group_fp8_quant` emits
//     them. group_size must equal D, so ONE quant group == ONE head and the whole fused
//     op is warp-local. Quant numerics matched to csrc/libtorch_stable/.../
//     per_token_group_quant.cu (eps floor 1e-10, NO min-scale clamp, TRUE division,
//     clamp order fminf(fmaxf(v, -448), 448)):
//       yq    = f32(scalar_t(y))                       <- the graph boundary double-rounding
//       amax  = max(1e-10, max_group |yq|)
//       s     = amax / 448
//       out   = fp8(fminf(fmaxf(yq / s, -448), 448))
//     scale is fp32 [num_tokens, H*D/group] with ARBITRARY 2-D strides -- the launcher
//     forwards scale.stride(0/1), so the row-major layout per_token_group_quant_fp8
//     allocates, the column-major one (strides (1, N)) and the TMA-aligned one
//     (strides (1, aligned_N)) are all written correctly by the same kernel.
//
// H200 (sm_90a) tuning of the per-group fast path:
//   - one 16-lane half-warp per (token,head) row for bf16/fp16 (D=128 = 16 lanes x 8-elem
//     16B vectors); fp32 rows take a full warp. Reductions are pure __shfl_xor within the
//     sub-group -- NO shared memory, NO __syncthreads anywhere in the kernel;
//   - x and z are read once and kept in registers across both reductions;
//   - 256-thread blocks pack 16 rows (bf16) for occupancy; grid covers num_rows.
//   - no fast-math: expf/rsqrtf only where stock has them; the quant divide is a TRUE
//     divide because stock's group-quant kernel divides (see fp8_common.cuh's warning
//     about "optimizing" this into a reciprocal multiply).
// The arithmetic here is kept OPERATION-FOR-OPERATION identical to the NVRTC twin
// (vtl/kernels/gated_rmsnorm_group_quant.cu); bench/test_gdn_gated_rmsnorm.py asserts the
// two produce bit-identical fp8 codes and scales.

#include <torch/all.h>

#include "fp8_common.cuh"

namespace vtl {

namespace {

// act(z): silu (z*sigmoid(z)) or sigmoid(z), in fp32 -- matches forward_static (z.float()).
__device__ __forceinline__ float gate_act(float z, bool is_silu) {
  float const s = 1.0f / (1.0f + expf(-z));
  return is_silu ? z * s : s;
}

// Full-warp reduction (all 32 lanes carry valid data or the identity).
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// Quant-group eps floor and fp8 range, matching _C::per_token_group_fp8_quant's defaults
// (fp8_utils.per_token_group_quant_fp8 / MatcherQuantFP8 both hardcode eps=1e-10).
constexpr float kGroupQuantEps = 1e-10f;

// Stock group-quant clamp order: fminf(fmaxf(v, lo), hi). NOT interchangeable with
// fp8_common's clamp_fp8 (fmaxf(lo, fminf(v, hi))): they differ on NaN (stock maps NaN to
// lo, clamp_fp8 to hi). Matched to csrc/libtorch_stable's QuantizeGroup.
__device__ __forceinline__ float group_quant_clamp(float v) {
  return fminf(fmaxf(v, -kFp8Max), kFp8Max);
}

// -------- standalone gated RMSNorm (bf16/fp16/fp32 out) ---------------------------------------

// Fast path: one warp per row, D <= 32*kVec and D % kVec == 0, each lane owns one 16B chunk.
template <typename scalar_t, typename weight_t, int WARPS>
__global__ void gated_rmsnorm_warp_kernel(scalar_t* __restrict__ out,
                                          scalar_t const* __restrict__ input,
                                          scalar_t const* __restrict__ gate,
                                          weight_t const* __restrict__ weight, float const eps,
                                          bool const is_silu, int const D,
                                          int64_t const num_rows) {
  constexpr int kVec = VecTraits<scalar_t>::kVec;
  int const warp = threadIdx.x >> 5;
  int const lane = threadIdx.x & 31;
  int64_t const row = static_cast<int64_t>(blockIdx.x) * WARPS + warp;
  if (row >= num_rows) return;  // uniform across the warp: row depends only on warp id

  int const idx = lane * kVec;
  bool const active = idx < D;
  int64_t const off = row * static_cast<int64_t>(D) + idx;

  float xf[kVec], zf[kVec];
  if (active) {
    VecIn<scalar_t, kVec> vx = *reinterpret_cast<VecIn<scalar_t, kVec> const*>(input + off);
    VecIn<scalar_t, kVec> vz = *reinterpret_cast<VecIn<scalar_t, kVec> const*>(gate + off);
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      xf[j] = static_cast<float>(vx.v[j]);
      zf[j] = static_cast<float>(vz.v[j]);
    }
  } else {
#pragma unroll
    for (int j = 0; j < kVec; ++j) xf[j] = zf[j] = 0.0f;
  }

  float ss = 0.0f;
#pragma unroll
  for (int j = 0; j < kVec; ++j) ss += xf[j] * xf[j];
  ss = warp_reduce_sum(ss);
  float const rstd = rsqrtf(ss / D + eps);

  if (active) {
    VecIn<scalar_t, kVec> vo;
#pragma unroll
    for (int j = 0; j < kVec; ++j)
      vo.v[j] = static_cast<scalar_t>(xf[j] * rstd * static_cast<float>(weight[idx + j]) *
                                      gate_act(zf[j], is_silu));
    *reinterpret_cast<VecIn<scalar_t, kVec>*>(out + off) = vo;
  }
}

// Generic fallback: one block per row, strided, for D that the fast path cannot vectorize.
template <typename scalar_t, typename weight_t>
__global__ void gated_rmsnorm_generic_kernel(scalar_t* __restrict__ out,
                                             scalar_t const* __restrict__ input,
                                             scalar_t const* __restrict__ gate,
                                             weight_t const* __restrict__ weight, float const eps,
                                             bool const is_silu, int const D) {
  int64_t const base = static_cast<int64_t>(blockIdx.x) * D;
  __shared__ float smem[32];

  float ss = 0.0f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float const x = static_cast<float>(input[base + i]);
    ss += x * x;
  }
  ss = block_reduce(ss, AddOp{}, 0.0f, smem);
  float const rstd = rsqrtf(ss / D + eps);

  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float const x = static_cast<float>(input[base + i]);
    float const z = static_cast<float>(gate[base + i]);
    out[base + i] = static_cast<scalar_t>(x * rstd * static_cast<float>(weight[i]) *
                                          gate_act(z, is_silu));
  }
}

// -------- fused gated RMSNorm + per-(token,group) fp8 quant (ROUND-2 HOT PATH) ----------------

// Fast path, D == kGroupD == group_size: one SUB-GROUP of kLanes = D/kVec lanes per
// (token,head) row -- 16 lanes for bf16/fp16, 32 for fp32. Each lane owns one 16-byte
// chunk; both reductions are __shfl_xor within the aligned sub-group, so there is no
// shared memory and no __syncthreads. THREADS/kLanes rows per block.
//
// The arithmetic below is kept operation-for-operation identical to the NVRTC twin
// (vtl/kernels/gated_rmsnorm_group_quant.cu) so the two bit-match: same load order, same
// in-thread accumulation order, same shuffle offsets, same rsqrtf/expf calls, same
// narrow-then-widen, same true division, same clamp order, same pairwise fp8 converter.
template <typename scalar_t, typename weight_t, int D, int THREADS>
__global__ void __launch_bounds__(THREADS) gated_rmsnorm_group_quant_kernel(
    c10::Float8_e4m3fn* __restrict__ out,   // [num_tokens, H*D] == flat [num_rows, D]
    float* __restrict__ scales,             // fp32, strides (ss_tok, ss_grp)
    scalar_t const* __restrict__ input,     // [num_rows, D]
    scalar_t const* __restrict__ gate,      // [num_rows, D]
    weight_t const* __restrict__ weight,    // [D]
    int64_t const ss_tok, int64_t const ss_grp, float const eps, bool const is_silu,
    int const H, int64_t const num_rows) {
  constexpr int kVec = VecTraits<scalar_t>::kVec;
  constexpr int kLanes = D / kVec;  // lanes per row/group
  static_assert(D % kVec == 0, "vector width must divide the head dim");
  static_assert(kLanes >= 2 && kLanes <= 32 && (kLanes & (kLanes - 1)) == 0,
                "sub-group reduction needs a power-of-two lane count within one warp");
  static_assert(THREADS % kLanes == 0, "whole sub-groups per block");
  constexpr int kRowsPerBlock = THREADS / kLanes;

  int const sub = threadIdx.x / kLanes;   // row within the block
  int const lane = threadIdx.x % kLanes;  // lane within the row
  int64_t const row = static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + sub;
  if (row >= num_rows) return;  // uniform within the sub-group (and its shuffle mask)

  // Shuffle mask covering exactly this sub-group's lanes within the warp. Early-exited
  // sub-groups must not appear in an active mask, and xor offsets < kLanes stay inside an
  // aligned kLanes block, so this is safe without any __syncwarp.
  int const lane_in_warp = threadIdx.x & 31;
  unsigned mask;
  if constexpr (kLanes == 32) {
    mask = 0xffffffffu;
  } else {
    mask = ((1u << kLanes) - 1u) << (lane_in_warp & ~(kLanes - 1));
  }

  int const idx = lane * kVec;
  int64_t const off = row * static_cast<int64_t>(D) + idx;

  // --- load once, keep in registers ---
  float xf[kVec], zf[kVec];
  {
    VecIn<scalar_t, kVec> const vx =
        *reinterpret_cast<VecIn<scalar_t, kVec> const*>(input + off);
    VecIn<scalar_t, kVec> const vz =
        *reinterpret_cast<VecIn<scalar_t, kVec> const*>(gate + off);
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      xf[j] = static_cast<float>(vx.v[j]);
      zf[j] = static_cast<float>(vz.v[j]);
    }
  }

  // --- sum(x^2) over the row, sub-group shuffle reduce ---
  float ss = 0.0f;
#pragma unroll
  for (int j = 0; j < kVec; ++j) ss += xf[j] * xf[j];
#pragma unroll
  for (int off_ = kLanes / 2; off_ > 0; off_ >>= 1) ss += __shfl_xor_sync(mask, ss, off_);
  float const rstd = rsqrtf(ss / D + eps);

  // --- y = ((x*rstd)*w)*act(z) in fp32; narrow ONCE to scalar_t; widen for the quant ---
  float yq[kVec];
  float amax = 0.0f;
#pragma unroll
  for (int j = 0; j < kVec; ++j) {
    float const y =
        ((xf[j] * rstd) * static_cast<float>(weight[idx + j])) * gate_act(zf[j], is_silu);
    yq[j] = static_cast<float>(static_cast<scalar_t>(y));  // the RMSNormGated -> quant boundary
    amax = fmaxf(amax, fabsf(yq[j]));
  }
#pragma unroll
  for (int off_ = kLanes / 2; off_ > 0; off_ >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(mask, amax, off_));

  // --- per-group scale, stock numerics: eps floor, /448, NO min-scale clamp ---
  amax = fmaxf(amax, kGroupQuantEps);
  float const y_s = amax / kFp8Max;
  if (lane == 0) {
    int64_t const token = row / H;
    int64_t const grp = row % H;  // group_size == D: one group per head
    scales[token * ss_tok + grp * ss_grp] = y_s;
  }

  // --- quantize: TRUE division (stock divides), stock clamp order, hw pair converter ---
  VecOut<kVec> vout;
#pragma unroll
  for (int j = 0; j < kVec; j += 2) {
    float2 v;
    v.x = group_quant_clamp(yq[j] / y_s);
    v.y = group_quant_clamp(yq[j + 1] / y_s);
    vout.pair[j >> 1] = __nv_cvt_float2_to_fp8x2(v, __NV_SATFINITE, __NV_E4M3);
  }
  *reinterpret_cast<VecOut<kVec>*>(out + off) = vout;
}

// Generic fused fallback: one block per (token,head) row, strided loops, for shapes the
// fast path rejects (D != 128 / misaligned). Requires group_size == D like the fast path;
// re-reads x|z on the second pass. Correctness path only.
template <typename scalar_t, typename weight_t>
__global__ void gated_rmsnorm_group_quant_generic_kernel(
    c10::Float8_e4m3fn* __restrict__ out, float* __restrict__ scales,
    scalar_t const* __restrict__ input, scalar_t const* __restrict__ gate,
    weight_t const* __restrict__ weight, int64_t const ss_tok, int64_t const ss_grp,
    float const eps, bool const is_silu, int const H, int const D) {
  int64_t const row = blockIdx.x;
  int64_t const base = row * static_cast<int64_t>(D);
  __shared__ float smem[32];
  __shared__ float s_scale;

  float ss = 0.0f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float const x = static_cast<float>(input[base + i]);
    ss += x * x;
  }
  ss = block_reduce(ss, AddOp{}, 0.0f, smem);
  float const rstd = rsqrtf(ss / D + eps);

  float amax = 0.0f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float const x = static_cast<float>(input[base + i]);
    float const z = static_cast<float>(gate[base + i]);
    float const y = ((x * rstd) * static_cast<float>(weight[i])) * gate_act(z, is_silu);
    float const yq = static_cast<float>(static_cast<scalar_t>(y));
    amax = fmaxf(amax, fabsf(yq));
  }
  amax = block_reduce(amax, MaxOp{}, 0.0f, smem);
  amax = fmaxf(amax, kGroupQuantEps);
  float const y_s = amax / kFp8Max;
  if (threadIdx.x == 0) {
    int64_t const token = row / H;
    int64_t const grp = row % H;
    scales[token * ss_tok + grp * ss_grp] = y_s;
    s_scale = y_s;
  }
  __syncthreads();
  float const scale = s_scale;

  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float const x = static_cast<float>(input[base + i]);
    float const z = static_cast<float>(gate[base + i]);
    float const y = ((x * rstd) * static_cast<float>(weight[i])) * gate_act(z, is_silu);
    float const yq = static_cast<float>(static_cast<scalar_t>(y));
    __nv_fp8_storage_t const bits =
        __nv_cvt_float_to_fp8(group_quant_clamp(yq / scale), __NV_SATFINITE, __NV_E4M3);
    out[base + i] = c10::Float8_e4m3fn(bits, c10::Float8_e4m3fn::from_bits());
  }
}

// -------- fused gated RMSNorm + per-token fp8 quant (round-1.1 compat) ------------------------

// Fast path: one BLOCK per token, one WARP per head. Each warp norms its head (pure __shfl,
// no sync), keeps the normed row in registers, then a single block-wide max over the H warps
// gives the per-token amax -> scale -> fp8 write. blockDim = H*32 (<= 1024, so H <= 32 --
// the round-2 model's H=64 lands on the generic kernel below; this entry is A/B-only).
template <typename scalar_t, typename weight_t>
__global__ void gated_rmsnorm_quant_warp_kernel(c10::Float8_e4m3fn* __restrict__ out,  // [N, H*D]
                                                float* __restrict__ scales,            // [N]
                                                scalar_t const* __restrict__ input,  // [N*H, D]
                                                scalar_t const* __restrict__ gate,   // [N*H, D]
                                                weight_t const* __restrict__ weight, // [D]
                                                float const* __restrict__ scale_ub,  // scalar/null
                                                float const eps, bool const is_silu, int const H,
                                                int const D) {
  constexpr int kVec = VecTraits<scalar_t>::kVec;
  int const warp = threadIdx.x >> 5;  // head within the token
  int const lane = threadIdx.x & 31;
  int64_t const token = blockIdx.x;
  int const idx = lane * kVec;
  bool const active = idx < D;
  int64_t const in_off = (token * H + warp) * static_cast<int64_t>(D) + idx;

  float nrm[kVec];  // normed value (fp32), kept in registers across the block reduction
  {
    float xf[kVec], zf[kVec];
    if (active) {
      VecIn<scalar_t, kVec> vx = *reinterpret_cast<VecIn<scalar_t, kVec> const*>(input + in_off);
      VecIn<scalar_t, kVec> vz = *reinterpret_cast<VecIn<scalar_t, kVec> const*>(gate + in_off);
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        xf[j] = static_cast<float>(vx.v[j]);
        zf[j] = static_cast<float>(vz.v[j]);
      }
    } else {
#pragma unroll
      for (int j = 0; j < kVec; ++j) xf[j] = zf[j] = 0.0f;
    }
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < kVec; ++j) ss += xf[j] * xf[j];
    ss = warp_reduce_sum(ss);
    float const rstd = rsqrtf(ss / D + eps);
#pragma unroll
    for (int j = 0; j < kVec; ++j)
      nrm[j] = active ? xf[j] * rstd * static_cast<float>(weight[idx + j]) *
                            gate_act(zf[j], is_silu)
                      : 0.0f;
  }

  float amax = 0.0f;
#pragma unroll
  for (int j = 0; j < kVec; ++j) amax = fmaxf(amax, fabsf(nrm[j]));
  __shared__ float smem[32];
  amax = block_reduce(amax, MaxOp{}, 0.0f, smem);  // over all H warps of this token
  if (scale_ub != nullptr) amax = fminf(amax, *scale_ub);
  float const scale = fmaxf(amax / kFp8Max, kMinScale);
  if (threadIdx.x == 0) scales[token] = scale;

  if (active) {
    int64_t const out_off =
        token * static_cast<int64_t>(H) * D + warp * static_cast<int64_t>(D) + idx;
    VecOut<kVec> vout;
#pragma unroll
    for (int j = 0; j < kVec; j += 2)
      vout.pair[j >> 1] = floats_to_fp8x2(nrm[j] / scale, nrm[j + 1] / scale);
    *reinterpret_cast<VecOut<kVec>*>(out + out_off) = vout;
  }
}

// Generic fused fallback: one block per token, loops heads. Two passes over x|z (re-reads), so
// slower than the fast path -- for shapes the fast path rejects (D not vectorizable, or
// H*32 > 1024, which includes this model's H=64). dynamic shared: [H] rms + [32] scratch.
template <typename scalar_t, typename weight_t>
__global__ void gated_rmsnorm_quant_generic_kernel(c10::Float8_e4m3fn* __restrict__ out,
                                                   float* __restrict__ scales,
                                                   scalar_t const* __restrict__ input,
                                                   scalar_t const* __restrict__ gate,
                                                   weight_t const* __restrict__ weight,
                                                   float const* __restrict__ scale_ub,
                                                   float const eps, bool const is_silu,
                                                   int const H, int const D) {
  int64_t const token = blockIdx.x;
  extern __shared__ float sh[];
  float* rms_sh = sh;      // [H]
  float* red = sh + H;     // [32]

  float my_amax = 0.0f;
  for (int h = 0; h < H; ++h) {
    int64_t const base = (token * H + h) * static_cast<int64_t>(D);
    float ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
      float const x = static_cast<float>(input[base + i]);
      ss += x * x;
    }
    ss = block_reduce(ss, AddOp{}, 0.0f, red);
    float const rstd = rsqrtf(ss / D + eps);
    if (threadIdx.x == 0) rms_sh[h] = rstd;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
      float const x = static_cast<float>(input[base + i]);
      float const z = static_cast<float>(gate[base + i]);
      my_amax = fmaxf(my_amax,
                      fabsf(x * rstd * static_cast<float>(weight[i]) * gate_act(z, is_silu)));
    }
    __syncthreads();  // rms_sh[h] visible + red free before the next head
  }

  float amax = block_reduce(my_amax, MaxOp{}, 0.0f, red);
  if (scale_ub != nullptr) amax = fminf(amax, *scale_ub);
  float const scale = fmaxf(amax / kFp8Max, kMinScale);
  if (threadIdx.x == 0) scales[token] = scale;

  for (int h = 0; h < H; ++h) {
    int64_t const base = (token * H + h) * static_cast<int64_t>(D);
    int64_t const obase = token * static_cast<int64_t>(H) * D + h * static_cast<int64_t>(D);
    float const rstd = rms_sh[h];
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
      float const x = static_cast<float>(input[base + i]);
      float const z = static_cast<float>(gate[base + i]);
      out[obase + i] =
          float_to_fp8(x * rstd * static_cast<float>(weight[i]) * gate_act(z, is_silu) / scale);
    }
  }
}

// -------- launchers ---------------------------------------------------------------------------

constexpr int kNormWarpsPerBlock = 8;    // standalone: rows packed per block for occupancy
constexpr int kGroupQuantThreads = 256;  // per-group fast path block width (16 bf16 rows)
constexpr int kGroupD = 128;             // the one fast-path head dim (== quant group)

template <typename scalar_t, typename weight_t>
void launch_norm(torch::Tensor const& out, torch::Tensor const& input, torch::Tensor const& gate,
                 torch::Tensor const& weight, double epsilon, bool is_silu, cudaStream_t stream) {
  constexpr int kVec = VecTraits<scalar_t>::kVec;
  int const D = input.size(-1);
  int64_t const num_rows = input.numel() / D;
  if (num_rows == 0) return;

  auto* out_p = out.data_ptr<scalar_t>();
  auto const* in_p = input.const_data_ptr<scalar_t>();
  auto const* gate_p = gate.const_data_ptr<scalar_t>();
  auto const* w_p = weight.const_data_ptr<weight_t>();
  float const eps = static_cast<float>(epsilon);

  bool const fast = D % kVec == 0 && (D / kVec) <= 32 && aligned16(in_p) && aligned16(gate_p) &&
                    aligned16(out_p);
  if (fast) {
    constexpr int W = kNormWarpsPerBlock;
    dim3 const grid((num_rows + W - 1) / W);
    dim3 const block(W * 32);
    gated_rmsnorm_warp_kernel<scalar_t, weight_t, W>
        <<<grid, block, 0, stream>>>(out_p, in_p, gate_p, w_p, eps, is_silu, D, num_rows);
  } else {
    dim3 const grid(num_rows);
    dim3 const block(std::min((D + 31) / 32 * 32, 1024));
    gated_rmsnorm_generic_kernel<scalar_t, weight_t>
        <<<grid, block, 0, stream>>>(out_p, in_p, gate_p, w_p, eps, is_silu, D);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (kernel_sync_enabled()) {
    cudaError_t const e = cudaStreamSynchronize(stream);
    TORCH_CHECK(e == cudaSuccess, "vtl gated_rmsnorm faulted: ", cudaGetErrorString(e),
                " dtype=", input.scalar_type(), " num_rows=", num_rows, " D=", D);
  }
}

template <typename scalar_t, typename weight_t>
void launch_group_quant(torch::Tensor const& out, torch::Tensor const& scale,
                        torch::Tensor const& input, torch::Tensor const& gate,
                        torch::Tensor const& weight, double epsilon, int H, bool is_silu,
                        cudaStream_t stream) {
  constexpr int kVec = VecTraits<scalar_t>::kVec;
  int const D = input.size(-1);
  int64_t const num_rows = input.size(0);
  if (num_rows == 0) return;

  auto* out_p = reinterpret_cast<c10::Float8_e4m3fn*>(out.data_ptr());
  auto* scale_p = scale.data_ptr<float>();
  auto const* in_p = input.const_data_ptr<scalar_t>();
  auto const* gate_p = gate.const_data_ptr<scalar_t>();
  auto const* w_p = weight.const_data_ptr<weight_t>();
  int64_t const ss_tok = scale.stride(0);
  int64_t const ss_grp = scale.stride(1);
  float const eps = static_cast<float>(epsilon);

  bool const fast =
      D == kGroupD && aligned16(in_p) && aligned16(gate_p) && aligned16(out_p);
  if (fast) {
    constexpr int T = kGroupQuantThreads;
    constexpr int kRowsPerBlock = T / (kGroupD / kVec);
    dim3 const grid((num_rows + kRowsPerBlock - 1) / kRowsPerBlock);
    dim3 const block(T);
    gated_rmsnorm_group_quant_kernel<scalar_t, weight_t, kGroupD, T><<<grid, block, 0, stream>>>(
        out_p, scale_p, in_p, gate_p, w_p, ss_tok, ss_grp, eps, is_silu, H, num_rows);
  } else {
    dim3 const grid(num_rows);
    dim3 const block(std::min((D + 31) / 32 * 32, 1024));
    gated_rmsnorm_group_quant_generic_kernel<scalar_t, weight_t><<<grid, block, 0, stream>>>(
        out_p, scale_p, in_p, gate_p, w_p, ss_tok, ss_grp, eps, is_silu, H, D);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (kernel_sync_enabled()) {
    cudaError_t const e = cudaStreamSynchronize(stream);
    TORCH_CHECK(e == cudaSuccess, "vtl gated_rmsnorm_group_quant faulted: ",
                cudaGetErrorString(e), " path=", (fast ? "fast" : "generic"),
                " dtype=", input.scalar_type(), " num_rows=", num_rows, " H=", H, " D=", D);
  }
}

template <typename scalar_t, typename weight_t>
void launch_norm_quant(torch::Tensor const& out, torch::Tensor const& scale,
                       torch::Tensor const& input, torch::Tensor const& gate,
                       torch::Tensor const& weight, double epsilon, int H, bool is_silu,
                       std::optional<torch::Tensor> const& scale_ub, cudaStream_t stream) {
  constexpr int kVec = VecTraits<scalar_t>::kVec;
  int const D = input.size(-1);
  int64_t const num_rows = input.numel() / D;
  int64_t const num_tokens = num_rows / H;
  if (num_tokens == 0) return;

  auto* out_p = reinterpret_cast<c10::Float8_e4m3fn*>(out.data_ptr());
  auto* scale_p = scale.data_ptr<float>();
  auto const* in_p = input.const_data_ptr<scalar_t>();
  auto const* gate_p = gate.const_data_ptr<scalar_t>();
  auto const* w_p = weight.const_data_ptr<weight_t>();
  auto const* ub_p = scale_ub.has_value() ? scale_ub->const_data_ptr<float>() : nullptr;
  float const eps = static_cast<float>(epsilon);

  bool const fast = D % kVec == 0 && (D / kVec) <= 32 && H * 32 <= 1024 && aligned16(in_p) &&
                    aligned16(gate_p) && aligned16(out_p);
  dim3 const grid(num_tokens);
  if (fast) {
    dim3 const block(H * 32);
    gated_rmsnorm_quant_warp_kernel<scalar_t, weight_t>
        <<<grid, block, 0, stream>>>(out_p, scale_p, in_p, gate_p, w_p, ub_p, eps, is_silu, H, D);
  } else {
    dim3 const block(std::min((D + 31) / 32 * 32, 1024));
    size_t const shmem = (static_cast<size_t>(H) + 32) * sizeof(float);
    gated_rmsnorm_quant_generic_kernel<scalar_t, weight_t>
        <<<grid, block, shmem, stream>>>(out_p, scale_p, in_p, gate_p, w_p, ub_p, eps, is_silu, H,
                                         D);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (kernel_sync_enabled()) {
    cudaError_t const e = cudaStreamSynchronize(stream);
    TORCH_CHECK(e == cudaSuccess, "vtl gated_rmsnorm_quant faulted: ", cudaGetErrorString(e),
                " path=", (fast ? "fast" : "generic"), " dtype=", input.scalar_type(),
                " num_tokens=", num_tokens, " H=", H, " D=", D);
  }
}

// Weight may be stored fp32 OR in the input dtype (v0.25.0 creates it in the model dtype;
// both stock implementations upcast before use, and so do our kernels).
void check_weight_dtype(torch::Tensor const& weight, torch::Tensor const& input,
                        char const* op) {
  TORCH_CHECK(weight.scalar_type() == at::ScalarType::Float ||
                  weight.scalar_type() == input.scalar_type(),
              "vtl: ", op, " weight must be fp32 or match the input dtype, got weight=",
              weight.scalar_type(), " input=", input.scalar_type());
}

// One dispatch for input dtype x weight dtype. `F` is a functor templated on
// <scalar_t, weight_t>.
template <template <typename, typename> class F, typename... Args>
void dispatch_dtypes(torch::Tensor const& input, torch::Tensor const& weight, char const* op,
                     Args&&... args) {
  bool const w_fp32 = weight.scalar_type() == at::ScalarType::Float;
  switch (input.scalar_type()) {
    case at::ScalarType::BFloat16:
      if (w_fp32) {
        F<c10::BFloat16, float>::run(std::forward<Args>(args)...);
      } else {
        F<c10::BFloat16, c10::BFloat16>::run(std::forward<Args>(args)...);
      }
      break;
    case at::ScalarType::Half:
      if (w_fp32) {
        F<c10::Half, float>::run(std::forward<Args>(args)...);
      } else {
        F<c10::Half, c10::Half>::run(std::forward<Args>(args)...);
      }
      break;
    case at::ScalarType::Float:
      F<float, float>::run(std::forward<Args>(args)...);
      break;
    default:
      TORCH_CHECK(false, "vtl: ", op, " unsupported input dtype ", input.scalar_type());
  }
}

template <typename scalar_t, typename weight_t>
struct NormOp {
  static void run(torch::Tensor const& out, torch::Tensor const& input,
                  torch::Tensor const& gate, torch::Tensor const& weight, double epsilon,
                  bool is_silu, cudaStream_t stream) {
    launch_norm<scalar_t, weight_t>(out, input, gate, weight, epsilon, is_silu, stream);
  }
};

template <typename scalar_t, typename weight_t>
struct GroupQuantOp {
  static void run(torch::Tensor const& out, torch::Tensor const& scale,
                  torch::Tensor const& input, torch::Tensor const& gate,
                  torch::Tensor const& weight, double epsilon, int H, bool is_silu,
                  cudaStream_t stream) {
    launch_group_quant<scalar_t, weight_t>(out, scale, input, gate, weight, epsilon, H, is_silu,
                                           stream);
  }
};

template <typename scalar_t, typename weight_t>
struct TokenQuantOp {
  static void run(torch::Tensor const& out, torch::Tensor const& scale,
                  torch::Tensor const& input, torch::Tensor const& gate,
                  torch::Tensor const& weight, double epsilon, int H, bool is_silu,
                  std::optional<torch::Tensor> const& scale_ub, cudaStream_t stream) {
    launch_norm_quant<scalar_t, weight_t>(out, scale, input, gate, weight, epsilon, H, is_silu,
                                          scale_ub, stream);
  }
};

}  // namespace

void gated_rmsnorm(torch::Tensor& result, torch::Tensor const& input, torch::Tensor const& gate,
                   torch::Tensor const& weight, double epsilon, bool gate_is_silu) {
  TORCH_CHECK(input.is_contiguous() && gate.is_contiguous() && result.is_contiguous(),
              "vtl: gated_rmsnorm requires contiguous input/gate/result");
  TORCH_CHECK(input.sizes() == gate.sizes() && input.sizes() == result.sizes(),
              "vtl: gated_rmsnorm input/gate/result must have equal shape");
  check_weight_dtype(weight, input, "gated_rmsnorm");
  TORCH_CHECK(weight.is_contiguous() && weight.numel() == input.size(-1),
              "vtl: gated_rmsnorm weight must be contiguous [D]");
  TORCH_CHECK(input.scalar_type() == result.scalar_type() &&
                  input.scalar_type() == gate.scalar_type(),
              "vtl: gated_rmsnorm input/gate/result dtype mismatch");

  c10::cuda::OptionalCUDAGuard const device_guard(input.device());
  cudaStream_t const stream = c10::cuda::getCurrentCUDAStream();
  dispatch_dtypes<NormOp>(input, weight, "gated_rmsnorm", result, input, gate, weight, epsilon,
                          gate_is_silu, stream);
}

void gated_rmsnorm_per_group_quant(torch::Tensor& result, torch::Tensor& scale,
                                   torch::Tensor const& input, torch::Tensor const& gate,
                                   torch::Tensor const& weight, double epsilon,
                                   int64_t num_heads, int64_t group_size, bool gate_is_silu) {
  TORCH_CHECK(result.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "vtl: gated_rmsnorm_group_quant supports fp8_e4m3 output only, got ",
              result.scalar_type());
  TORCH_CHECK(input.is_contiguous() && gate.is_contiguous() && result.is_contiguous(),
              "vtl: gated_rmsnorm_group_quant requires contiguous input/gate/result");
  TORCH_CHECK(input.dim() == 2 && input.sizes() == gate.sizes(),
              "vtl: gated_rmsnorm_group_quant expects input/gate [num_tokens*H, D]");
  check_weight_dtype(weight, input, "gated_rmsnorm_group_quant");
  int const D = static_cast<int>(input.size(-1));
  int const H = static_cast<int>(num_heads);
  TORCH_CHECK(weight.is_contiguous() && weight.numel() == D,
              "vtl: gated_rmsnorm_group_quant weight must be contiguous [D]");
  TORCH_CHECK(group_size == D,
              "vtl: gated_rmsnorm_group_quant requires group_size == head dim (one quant "
              "group per head), got group=", group_size, " D=", D);
  TORCH_CHECK(H >= 1 && input.size(0) % H == 0,
              "vtl: gated_rmsnorm_group_quant rows must be a multiple of num_heads");
  int64_t const num_tokens = input.size(0) / H;
  TORCH_CHECK(result.numel() == input.numel(),
              "vtl: gated_rmsnorm_group_quant result must be [num_tokens, H*D]");
  TORCH_CHECK(scale.scalar_type() == at::ScalarType::Float,
              "vtl: gated_rmsnorm_group_quant scale must be fp32");
  TORCH_CHECK(scale.dim() == 2 && scale.size(0) == num_tokens && scale.size(1) == H,
              "vtl: gated_rmsnorm_group_quant scale must be [num_tokens, H] (H*D/group "
              "groups per token), got ", scale.sizes());
  TORCH_CHECK(input.scalar_type() == gate.scalar_type(),
              "vtl: gated_rmsnorm_group_quant input/gate dtype mismatch");

  c10::cuda::OptionalCUDAGuard const device_guard(input.device());
  cudaStream_t const stream = c10::cuda::getCurrentCUDAStream();
  dispatch_dtypes<GroupQuantOp>(input, weight, "gated_rmsnorm_group_quant", result, scale, input,
                                gate, weight, epsilon, H, gate_is_silu, stream);
}

void gated_rmsnorm_dynamic_per_token_quant(torch::Tensor& result, torch::Tensor& scale,
                                           torch::Tensor const& input, torch::Tensor const& gate,
                                           torch::Tensor const& weight, double epsilon,
                                           int64_t num_heads, bool gate_is_silu,
                                           std::optional<torch::Tensor> const& scale_ub) {
  TORCH_CHECK(result.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "vtl: gated_rmsnorm_quant supports fp8_e4m3 output only, got ", result.scalar_type());
  TORCH_CHECK(input.is_contiguous() && gate.is_contiguous() && result.is_contiguous(),
              "vtl: gated_rmsnorm_quant requires contiguous input/gate/result");
  TORCH_CHECK(input.sizes() == gate.sizes(), "vtl: gated_rmsnorm_quant input/gate shape mismatch");
  TORCH_CHECK(input.dim() == 2, "vtl: gated_rmsnorm_quant expects input [num_tokens*H, D]");
  check_weight_dtype(weight, input, "gated_rmsnorm_quant");
  int const D = static_cast<int>(input.size(-1));
  int const H = static_cast<int>(num_heads);
  TORCH_CHECK(weight.is_contiguous() && weight.numel() == D,
              "vtl: gated_rmsnorm_quant weight must be contiguous [D]");
  TORCH_CHECK(H >= 1 && input.size(0) % H == 0,
              "vtl: gated_rmsnorm_quant rows must be a multiple of num_heads");
  int64_t const num_tokens = input.size(0) / H;
  TORCH_CHECK(result.numel() == num_tokens * H * D,
              "vtl: gated_rmsnorm_quant result must be [num_tokens, H*D]");
  TORCH_CHECK(scale.scalar_type() == at::ScalarType::Float, "vtl: gated_rmsnorm_quant scale fp32");
  TORCH_CHECK(scale.numel() == num_tokens, "vtl: gated_rmsnorm_quant scale must be [num_tokens]");
  TORCH_CHECK(input.scalar_type() == gate.scalar_type(), "vtl: gated_rmsnorm_quant dtype mismatch");

  c10::cuda::OptionalCUDAGuard const device_guard(input.device());
  cudaStream_t const stream = c10::cuda::getCurrentCUDAStream();
  dispatch_dtypes<TokenQuantOp>(input, weight, "gated_rmsnorm_quant", result, scale, input, gate,
                                weight, epsilon, H, gate_is_silu, scale_ub, stream);
}

}  // namespace vtl
