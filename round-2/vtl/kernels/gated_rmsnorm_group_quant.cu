// NVRTC variant of the fused per-head gated RMSNorm + per-(token,128-group) FP8 quant kernel
// (the GDN out_proj input path -- 36 layers of Qwen3.5-122B-A10B). AOT twin:
// vtl/csrc/gdn_gated_rmsnorm.cu :: gated_rmsnorm_group_quant_kernel. Template rules per
// vtl/kernels/rms_norm_quant.cu: no torch/ATen headers, no host code, shapes as -D macros.
//
// Required defines (vtl/patches/gdn_kernels.py supplies them from the loaded model at the
// first call):
//   DV        head_v_dim (the norm row length; 128 on this model)
//   HV        value heads per rank (64 on this model) -- only used for the scale index
//   GROUP     quant group size; MUST equal DV (one quant group == one head)
//   THREADS   block width (default 256); must be a multiple of DV/8
//   IS_SILU   1: gate = z*sigmoid(z) (silu/swish); 0: gate = sigmoid(z)
//   W_FP32    1: weight is fp32; 0: weight is bf16 (v0.25.0 stores it in the model dtype)
//
// NUMERICS -- matched to RMSNormGated.forward_static + _C::per_token_group_fp8_quant,
// including the double rounding at the op boundary:
//   y    = ((f32(x) * rsqrtf(mean(x^2)+eps)) * f32(w)) * act(f32(z))   (all fp32)
//   yq   = f32(bf16(y))                  <- RMSNormGated returns bf16; the quant re-reads it
//   s    = max(1e-10, max_group|yq|) / 448          (NO min-scale clamp -- group quant
//                                                    floors the AMAX at 1e-10 instead)
//   out  = fp8(fminf(fmaxf(yq / s, -448), 448))     (TRUE division; stock divides. The
//                                                    clamp order differs from
//                                                    rms_norm_quant.cu's on NaN: stock group
//                                                    quant maps NaN to -448, keep it.)
// The arithmetic is kept operation-for-operation identical to the AOT twin (same load and
// accumulation order, same shuffle offsets, same expf/rsqrtf, same pairwise fp8 converter);
// bench/test_gdn_gated_rmsnorm.py asserts the two bit-match on the H200.
//
// Launch (computed in Python): one 16-lane sub-group per (token,head) row;
//   grid  = (ceil(num_rows / (THREADS / (DV/8))), 1, 1)
//   block = (THREADS, 1, 1)
//   args  = (out fp8*, scales f32*, x bf16*, z bf16*, w (W_FP32? f32*: bf16*),
//            ss_tok i64, ss_grp i64, num_rows i64, eps f32)
// scales strides are in ELEMENTS: row-major [N, HV] passes (HV, 1); the column-major layout
// vLLM's cutlass path allocates passes (1, N).

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#ifndef DV
#error "NVRTC: -DDV=<head_v_dim> is required"
#endif
#ifndef HV
#error "NVRTC: -DHV=<num_v_heads_per_rank> is required"
#endif
#ifndef GROUP
#error "NVRTC: -DGROUP=<quant_group_size> is required"
#endif
#ifndef IS_SILU
#error "NVRTC: -DIS_SILU=<0|1> is required (silu/swish vs sigmoid gate)"
#endif
#ifndef THREADS
#define THREADS 256
#endif
#ifndef W_FP32
#define W_FP32 0
#endif

#define VEC 8                 // 16-byte loads: 8 x bf16 per lane
#define LANES (DV / VEC)      // lanes per (token,head) row == per quant group

static_assert(GROUP == DV, "one quant group per head: GROUP must equal DV");
static_assert(DV % VEC == 0, "DV must be a multiple of 8 for the vectorized path");
static_assert(LANES >= 2 && LANES <= 32 && (LANES & (LANES - 1)) == 0,
              "sub-group reduction needs a power-of-two lane count within one warp");
static_assert(THREADS % LANES == 0, "whole sub-groups per block");
static_assert(HV >= 1, "at least one value head");

#define ROWS_PER_BLOCK (THREADS / LANES)

// constexpr, not __constant__: NVRTC modules have no host-side symbol copy. These fold
// into the SASS as immediates.
__device__ constexpr float kFp8Max = 448.0f;
__device__ constexpr float kGroupQuantEps = 1e-10f;  // stock per_token_group_fp8_quant eps

#if W_FP32
typedef float weight_t;
__device__ __forceinline__ float w2f(weight_t w) { return w; }
#else
typedef __nv_bfloat16 weight_t;
__device__ __forceinline__ float w2f(weight_t w) { return __bfloat162float(w); }
#endif

// One 16-byte load.
struct __align__(16) Vec8 { __nv_bfloat16 v[VEC]; };
// 8 fp8 codes = one 8-byte store, written as 4 hardware pair conversions.
struct __align__(8) Fp8Vec8 { __nv_fp8x2_storage_t pair[VEC / 2]; };

// act(z): silu (z*sigmoid(z)) or sigmoid(z), in fp32 -- matches forward_static (z.float()).
__device__ __forceinline__ float gate_act(float z) {
  const float s = 1.0f / (1.0f + expf(-z));
#if IS_SILU
  return z * s;
#else
  return s;
#endif
}

// Stock group-quant clamp order (fminf(fmaxf(v, lo), hi)); NaN maps to lo, as stock does.
__device__ __forceinline__ float group_quant_clamp(float v) {
  return fminf(fmaxf(v, -kFp8Max), kFp8Max);
}

extern "C" __global__ void __launch_bounds__(THREADS) gated_rmsnorm_group_quant(
    __nv_fp8_e4m3* __restrict__ out,      // [num_tokens, HV*DV] == flat [num_rows, DV]
    float* __restrict__ scales,           // fp32, strides (ss_tok, ss_grp) in elements
    const __nv_bfloat16* __restrict__ x,  // [num_rows, DV]
    const __nv_bfloat16* __restrict__ z,  // [num_rows, DV]
    const weight_t* __restrict__ w,       // [DV]
    const long long ss_tok, const long long ss_grp, const long long num_rows,
    const float eps) {
  const int sub = threadIdx.x / LANES;   // row within the block
  const int lane = threadIdx.x % LANES;  // lane within the row
  const long long row = (long long)blockIdx.x * ROWS_PER_BLOCK + sub;
  if (row >= num_rows) return;  // uniform within the sub-group (and its shuffle mask)

  // Shuffle mask covering exactly this sub-group's lanes within the warp. Early-exited
  // sub-groups must not appear in an active mask; xor offsets < LANES stay inside an
  // aligned LANES block.
  const int lane_in_warp = threadIdx.x & 31;
#if (DV / VEC) == 32
  const unsigned mask = 0xffffffffu;
#else
  const unsigned mask = ((1u << LANES) - 1u) << (lane_in_warp & ~(LANES - 1));
#endif

  const int idx = lane * VEC;
  const long long off = row * (long long)DV + idx;

  // --- load once, keep in registers ---
  float xf[VEC], zf[VEC];
  {
    const Vec8 vx = *reinterpret_cast<const Vec8*>(x + off);
    const Vec8 vz = *reinterpret_cast<const Vec8*>(z + off);
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      xf[j] = __bfloat162float(vx.v[j]);
      zf[j] = __bfloat162float(vz.v[j]);
    }
  }

  // --- sum(x^2) over the row, sub-group shuffle reduce ---
  float ss = 0.0f;
#pragma unroll
  for (int j = 0; j < VEC; ++j) ss += xf[j] * xf[j];
#pragma unroll
  for (int off_ = LANES / 2; off_ > 0; off_ >>= 1) ss += __shfl_xor_sync(mask, ss, off_);
  const float rstd = rsqrtf(ss / DV + eps);

  // --- y = ((x*rstd)*w)*act(z) in fp32; narrow ONCE to bf16; widen for the quant ---
  float yq[VEC];
  float amax = 0.0f;
#pragma unroll
  for (int j = 0; j < VEC; ++j) {
    const float y = ((xf[j] * rstd) * w2f(w[idx + j])) * gate_act(zf[j]);
    yq[j] = __bfloat162float(__float2bfloat16(y));  // the RMSNormGated -> quant boundary
    amax = fmaxf(amax, fabsf(yq[j]));
  }
#pragma unroll
  for (int off_ = LANES / 2; off_ > 0; off_ >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(mask, amax, off_));

  // --- per-group scale, stock numerics: eps floor, /448, NO min-scale clamp ---
  amax = fmaxf(amax, kGroupQuantEps);
  const float y_s = amax / kFp8Max;
  if (lane == 0) {
    const long long token = row / HV;
    const long long grp = row % HV;  // GROUP == DV: one quant group per head
    scales[token * ss_tok + grp * ss_grp] = y_s;
  }

  // --- quantize: TRUE division (stock divides), stock clamp order, hw pair converter ---
  Fp8Vec8 vout;
#pragma unroll
  for (int j = 0; j < VEC; j += 2) {
    float2 v;
    v.x = group_quant_clamp(yq[j] / y_s);
    v.y = group_quant_clamp(yq[j + 1] / y_s);
    vout.pair[j >> 1] = __nv_cvt_float2_to_fp8x2(v, __NV_SATFINITE, __NV_E4M3);
  }
  *reinterpret_cast<Fp8Vec8*>(out + off) = vout;
}
