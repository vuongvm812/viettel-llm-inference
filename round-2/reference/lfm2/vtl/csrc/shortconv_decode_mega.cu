// Scoped decode megakernel: the whole LFM2 short-conv block in ONE launch.
//
// Replaces this chain, per layer, 10 layers per step, inside the FULL cudagraph:
//
//   rms_norm_dynamic_per_token_quant  ->  in_proj  (W4A8 GEMM, N=6144, K=2048)
//     ->  bcx_conv_gate_quant         ->  out_proj (W4A8 GEMM, N=2048, K=2048)
//
// At decode M=1..8 every one of those is latency-bound, not throughput-bound: four launches,
// four tail effects, and three round trips of the intermediate activations through HBM.
// Fusing them needs a DEVICE-WIDE barrier between phases -- out_proj cannot start until the
// entire conv output exists -- which is the whole reason megakernel_probe.py shipped first.
//
// THE BARRIER IS HAND-ROLLED, DELIBERATELY. `cooperative_groups::this_grid().sync()` needs
// `cudaLaunchCooperativeKernel`, which torch's op dispatch does not expose and which cannot
// be issued from inside a stream-captured op without extra plumbing; on some toolchains it
// also drags in `-rdc=true`, which torch's BuildExtension cannot device-link (see setup.py's
// note on -dlto). The sense-reversing counter below needs none of that: it is plain atomics
// plus `__threadfence()`, it launches on the current stream like any other kernel, and it is
// stream-capturable. The `cg::grid.sync()` variant is kept behind `-DVTL_MEGA_COOP=1` for
// A/B on a box where the cooperative launch path is wired; runtime always prefers the
// hand-rolled one.
//
// CO-RESIDENCY IS A CORRECTNESS PRECONDITION, NOT A TUNING KNOB. A block that never gets
// scheduled never arrives, and the barrier hangs the device. So the grid is NOT chosen here:
// `shortconv_mega.py` computes it from `cudaOccupancyMaxActiveBlocksPerMultiprocessor` on
// THIS kernel (registers and smem included -- the probe's device ceiling is an upper bound,
// not an answer) times the SM count, and passes it in. `shortconv_decode_mega_grid()` is the
// query it uses. If that number is below `kMinBlocks` the Python gate refuses and the stock
// four-op chain runs.
//
// PHASES (3 barriers):
//   1. rms_norm + per-token fp8 quant       blocks 0..M-1, one per token   -> x_fp8, x_scale
//   2. in_proj W4A8 GEMV, N-split           all blocks, 6144/G rows each   -> bxc (bf16)
//   3. B*x -> depthwise conv1d + state rotation -> C* gate -> fp8 quant
//                                           blocks 0..M-1                  -> y_fp8, y_scale
//   4. out_proj W4A8 GEMV, N-split          all blocks, 2048/G rows each   -> out (bf16)
//
// Phase 1 and 3 are the bodies of rms_norm_quant.cu and bcx_conv_gate_quant.cu, specialised
// to one token per block; phases 2/4 are w4a8_gemv.cuh. N-SPLIT, not split-K: each block owns
// whole output rows, so there is no inter-block reduction and no second barrier per GEMV.
//
// SCRATCH is caller-owned and allocated ONCE at patch import, because a cudagraph bakes
// pointers: x_fp8[8,2048] + x_scale[8] + bxc[8,6144] + y_fp8[8,2048] + y_scale[8] + the
// barrier words. `shortconv_mega.py` lays it out; `kScratchBytes` here is the contract.
//
//   shortconv_decode_mega(Tensor! out, Tensor x, Tensor? norm_weight, float epsilon,
//                         Tensor in_wq, Tensor in_gs, Tensor conv_weight, Tensor? conv_bias,
//                         Tensor! conv_state, Tensor state_indices, int null_block_id,
//                         Tensor out_wq, Tensor out_gs, Tensor! scratch, int num_blocks) -> ()
//   shortconv_decode_mega_grid(int smem_bytes) -> int      # blocks/SM x SM count
//   shortconv_decode_mega_supported(int dim, int width, int state_len, int max_tokens) -> bool

#include <torch/all.h>

#include "fp8_common.cuh"
#include "w4a8_gemv.cuh"

#if defined(VTL_MEGA_COOP) && VTL_MEGA_COOP
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#endif

namespace vtl {

namespace {

constexpr int kMegaThreads = 256;
constexpr int kMegaMaxM = 8;      // decode batch ceiling; above it the stock chain runs
constexpr int kConvWidth = 3;     // LFM2.5 conv_L_cache
constexpr int kConvState = 2;     // == kConvWidth - 1
// Below this the barrier's serialization costs more than the four launches it removes, and
// the grid cannot cover D=2048 one-thread-per-channel (see megakernel_probe.verdict).
constexpr int kMinBlocks = 8;

// ---- device-wide sense-reversing barrier ---------------------------------------------
// Layout inside the scratch header: [0] = arrival counter, [1] = generation.
// Generation is monotonic across launches (never reset), so a stale value from a previous
// replay can never look like "already released".
__device__ __forceinline__ void grid_barrier(unsigned int* __restrict__ state, int nblocks) {
  __syncthreads();  // this block's shared/global writes are complete
  if (threadIdx.x == 0) {
    volatile unsigned int* gen = state + 1;
    __threadfence();  // publish them device-wide BEFORE announcing arrival
    unsigned int const my_gen = *gen;
    if (atomicAdd(state, 1u) == static_cast<unsigned int>(nblocks - 1)) {
      atomicExch(state, 0u);
      __threadfence();
      atomicAdd(state + 1, 1u);
    } else {
      while (*gen == my_gen) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(64);  // back off so the spinner does not starve the arrivers' atomics
#endif
      }
    }
    __threadfence();  // acquire: see every other block's phase output
  }
  __syncthreads();
}

// ---- phase 1: (optional RMSNorm) + dynamic per-token fp8 quant --------------------------
// One block, one token. Body of rms_norm_quant.cu's fast path at ITEMS=1 (D=2048, kVec=8,
// 256 threads): variance over the row, rsqrt, weight multiply, block amax, reciprocal-scale,
// hardware fp8 pair convert. Numerics character-for-character the standalone kernel's.
//
// `w == nullptr` SKIPS THE NORM and quantizes only. That is the shape the live integration
// uses: LFM2's `operator_norm` runs in lfm2.py, one level above `ShortConv.forward`, and
// short_conv.patch's whole point was hoisting it out where RMSNormQuantFusionPass can see it
// -- so by the time the megakernel is reachable the activation is already normalized and the
// only work left before in_proj is the per-token fp8 quant. The norm path stays implemented
// because it is the same eight lines and it is what a future integration one level up needs.
__device__ void phase_rms_norm_quant(c10::BFloat16 const* __restrict__ x,
                                     c10::BFloat16 const* __restrict__ w,
                                     __nv_fp8_storage_t* __restrict__ out_fp8,
                                     float* __restrict__ out_scale, int D, float eps,
                                     float* smem) {
  constexpr int kVec = VecTraits<c10::BFloat16>::kVec;
  using Vec = VecIn<c10::BFloat16, kVec>;
  int const base = threadIdx.x * kVec;
  bool const active = base < D;
  bool const do_norm = (w != nullptr);

  Vec vx, vw;
  float sq = 0.0f;
  if (active) {
    vx = *reinterpret_cast<Vec const*>(x + base);
    if (do_norm) {
      vw = *reinterpret_cast<Vec const*>(w + base);
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        float const f = static_cast<float>(vx.v[j]);
        sq += f * f;
      }
    }
  }
  float inv_rms = 1.0f;
  if (do_norm) {
    sq = block_reduce(sq, AddOp{}, 0.0f, smem);
    inv_rms = rsqrtf(sq / static_cast<float>(D) + eps);
  }

  float p[kVec];
  float amax = 0.0f;
  if (active) {
#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      if (do_norm) {
        // Narrow to bf16 after the norm and after the weight multiply, exactly as stock does.
        c10::BFloat16 const n =
            static_cast<c10::BFloat16>(static_cast<float>(vx.v[j]) * inv_rms);
        p[j] = static_cast<float>(static_cast<c10::BFloat16>(n * vw.v[j]));
      } else {
        p[j] = static_cast<float>(vx.v[j]);
      }
      amax = fmaxf(amax, fabsf(p[j]));
    }
  } else {
#pragma unroll
    for (int j = 0; j < kVec; ++j) p[j] = 0.0f;
  }
  amax = block_reduce(amax, MaxOp{}, 0.0f, smem);
  float const scale = fmaxf(amax / kFp8Max, kMinScale);
  if (threadIdx.x == 0) *out_scale = scale;
  float const inv = 1.0f / scale;

  if (active) {
    VecOut<kVec> vo;
#pragma unroll
    for (int j = 0; j < kVec; j += 2) {
      vo.pair[j >> 1] = floats_to_fp8x2(p[j] * inv, p[j + 1] * inv);
    }
    *reinterpret_cast<VecOut<kVec>*>(out_fp8 + base) = vo;
  }
}

// ---- phase 3: B*x -> conv1d + rotation -> C* -> fp8 quant ------------------------------
// One block, one token. Body of bcx_conv_gate_quant.cu, specialised to the vectorised SD
// path at width 3 / state_len 2 (the shape `bcx_conv_gate_supported` already gates on).
__device__ void phase_conv_gate_quant(c10::BFloat16 const* __restrict__ bcx,
                                      c10::BFloat16 const* __restrict__ weight,
                                      c10::BFloat16 const* __restrict__ bias,
                                      c10::BFloat16* __restrict__ state, int64_t coord,
                                      int64_t st_blk, int64_t st_tok,
                                      __nv_fp8_storage_t* __restrict__ out_fp8,
                                      float* __restrict__ out_scale, int D, float* smem) {
  constexpr int kVec = VecTraits<c10::BFloat16>::kVec;
  using Vec = VecIn<c10::BFloat16, kVec>;
  int const base = threadIdx.x * kVec;
  bool const active = base < D;

  if (coord < 0) {  // cudagraph padding row: defined zeros, state untouched
    if (threadIdx.x == 0) *out_scale = kMinScale;
    if (active) {
      VecOut<kVec> vz;
#pragma unroll
      for (int j = 0; j < kVec / 2; ++j) vz.pair[j] = floats_to_fp8x2(0.0f, 0.0f);
      *reinterpret_cast<VecOut<kVec>*>(out_fp8 + base) = vz;
    }
    return;
  }

  float p[kVec];
  if (active) {
    Vec const vb = *reinterpret_cast<Vec const*>(bcx + base);
    Vec const vc = *reinterpret_cast<Vec const*>(bcx + base + D);
    Vec const vx = *reinterpret_cast<Vec const*>(bcx + base + 2 * D);
    c10::BFloat16* const st_row = state + coord * st_blk + base;
    c10::BFloat16 const* const w_row = weight + static_cast<int64_t>(base) * kConvWidth;

    Vec vw[kConvWidth];
#pragma unroll
    for (int i = 0; i < kConvWidth; ++i) vw[i] = *reinterpret_cast<Vec const*>(w_row + i * kVec);
    Vec const vs0 = *reinterpret_cast<Vec const*>(st_row);
    Vec const vs1 = *reinterpret_cast<Vec const*>(st_row + st_tok);
    Vec vbx;

#pragma unroll
    for (int j = 0; j < kVec; ++j) {
      c10::BFloat16 const s0 = vs0.v[j];
      c10::BFloat16 const s1 = vs1.v[j];
      c10::BFloat16 const bx = vb.v[j] * vx.v[j];
      vbx.v[j] = bx;
      float acc = (bias != nullptr) ? static_cast<float>(bias[base + j]) : 0.0f;
#pragma unroll
      for (int k = 0; k < kConvWidth; ++k) {
        int const flat = j * kConvWidth + k;
        c10::BFloat16 const w = vw[flat / kVec].v[flat % kVec];
        acc += static_cast<float>(w * ((k == 0) ? s0 : (k == 1) ? s1 : bx));
      }
      p[j] = static_cast<float>(vc.v[j] * static_cast<c10::BFloat16>(acc));
    }
    *reinterpret_cast<Vec*>(st_row) = vs1;  // ring rotate [s0,s1] -> [s1,bx]
    *reinterpret_cast<Vec*>(st_row + st_tok) = vbx;
  } else {
#pragma unroll
    for (int j = 0; j < kVec; ++j) p[j] = 0.0f;
  }

  float amax = 0.0f;
#pragma unroll
  for (int j = 0; j < kVec; ++j) amax = fmaxf(amax, fabsf(p[j]));
  amax = block_reduce(amax, MaxOp{}, 0.0f, smem);
  float const scale = fmaxf(amax / kFp8Max, kMinScale);
  if (threadIdx.x == 0) *out_scale = scale;
  float const inv = 1.0f / scale;
  if (active) {
    VecOut<kVec> vo;
#pragma unroll
    for (int j = 0; j < kVec; j += 2) {
      vo.pair[j >> 1] = floats_to_fp8x2(p[j] * inv, p[j + 1] * inv);
    }
    *reinterpret_cast<VecOut<kVec>*>(out_fp8 + base) = vo;
  }
}

// Stage [M][K] fp8 from global into shared, 16 bytes per thread per pass.
__device__ __forceinline__ void stage_activations(__nv_fp8_storage_t* __restrict__ dst,
                                                  __nv_fp8_storage_t const* __restrict__ src,
                                                  int M, int K) {
  int const total = (M * K) >> 4;  // 16-byte units
  auto* d = reinterpret_cast<uint4*>(dst);
  auto const* s = reinterpret_cast<uint4 const*>(src);
  for (int i = threadIdx.x; i < total; i += kMegaThreads) d[i] = s[i];
  __syncthreads();
}

struct MegaScratch {
  unsigned int* barrier;      // [2]
  __nv_fp8_storage_t* x_fp8;  // [kMegaMaxM, D]
  float* x_scale;             // [kMegaMaxM]
  c10::BFloat16* bxc;         // [kMegaMaxM, 3D]
  __nv_fp8_storage_t* y_fp8;  // [kMegaMaxM, D]
  float* y_scale;             // [kMegaMaxM]
};

__global__ void __launch_bounds__(kMegaThreads, 2) shortconv_decode_mega_kernel(
    c10::BFloat16* __restrict__ out, c10::BFloat16 const* __restrict__ x,
    c10::BFloat16 const* __restrict__ norm_w, float const eps,
    uint8_t const* __restrict__ in_wq, float const* __restrict__ in_gs,
    c10::BFloat16 const* __restrict__ conv_w, c10::BFloat16 const* __restrict__ conv_b,
    c10::BFloat16* __restrict__ conv_state, int32_t const* __restrict__ state_idx,
    int64_t const idx_stride, int64_t const null_block_id, int64_t const st_blk,
    int64_t const st_tok, uint8_t const* __restrict__ out_wq,
    float const* __restrict__ out_gs, MegaScratch scr, int const M, int const D,
    int const nblocks) {
  extern __shared__ char smem_raw[];
  auto* smem_f = reinterpret_cast<float*>(smem_raw);                 // [32] reductions
  auto* x_sm = reinterpret_cast<__nv_fp8_storage_t*>(smem_raw + 128);  // [M][D]

  int const N_in = 3 * D;

  // ---- phase 1 ----
  if (blockIdx.x < M) {
    int const t = blockIdx.x;
    phase_rms_norm_quant(x + static_cast<int64_t>(t) * D, norm_w, scr.x_fp8 + t * D,
                         scr.x_scale + t, D, eps, smem_f);
  }
  grid_barrier(scr.barrier, nblocks);

  // ---- phase 2: in_proj. bxc is written as fp32-in-bf16 storage via a narrowing store. ----
  stage_activations(x_sm, scr.x_fp8, M, D);
  {
    int const warp = threadIdx.x >> 5;
    int const nwarps = kMegaThreads >> 5;
    int const lane = threadIdx.x & 31;
    int64_t const wq_stride = static_cast<int64_t>(D) >> 1;
    int const gs_stride = D / kW4A8Group;
    float acc[kMegaMaxM];
    for (int n = blockIdx.x * nwarps + warp; n < N_in; n += nblocks * nwarps) {
      w4a8_gemv_row<kMegaMaxM>(in_wq + static_cast<int64_t>(n) * wq_stride, in_gs + n * gs_stride,
                               x_sm, D, M, D, acc);
      if (lane == 0) {
#pragma unroll
        for (int m = 0; m < kMegaMaxM; ++m) {
          if (m < M) {
            scr.bxc[static_cast<int64_t>(m) * N_in + n] =
                static_cast<c10::BFloat16>(acc[m] * scr.x_scale[m]);
          }
        }
      }
    }
  }
  grid_barrier(scr.barrier, nblocks);

  // ---- phase 3 ----
  if (blockIdx.x < M) {
    int const t = blockIdx.x;
    int64_t const coord = static_cast<int64_t>(state_idx[t * idx_stride]);
    phase_conv_gate_quant(scr.bxc + static_cast<int64_t>(t) * N_in, conv_w, conv_b, conv_state,
                          coord == null_block_id ? -1 : coord, st_blk, st_tok,
                          scr.y_fp8 + t * D, scr.y_scale + t, D, smem_f);
  }
  grid_barrier(scr.barrier, nblocks);

  // ---- phase 4: out_proj ----
  stage_activations(x_sm, scr.y_fp8, M, D);
  {
    int const warp = threadIdx.x >> 5;
    int const nwarps = kMegaThreads >> 5;
    int const lane = threadIdx.x & 31;
    int64_t const wq_stride = static_cast<int64_t>(D) >> 1;
    int const gs_stride = D / kW4A8Group;
    float acc[kMegaMaxM];
    for (int n = blockIdx.x * nwarps + warp; n < D; n += nblocks * nwarps) {
      w4a8_gemv_row<kMegaMaxM>(out_wq + static_cast<int64_t>(n) * wq_stride, out_gs + n * gs_stride,
                               x_sm, D, M, D, acc);
      if (lane == 0) {
#pragma unroll
        for (int m = 0; m < kMegaMaxM; ++m) {
          if (m < M) {
            out[static_cast<int64_t>(m) * D + n] =
                static_cast<c10::BFloat16>(acc[m] * scr.y_scale[m]);
          }
        }
      }
    }
  }
}

#if defined(VTL_MEGA_COOP) && VTL_MEGA_COOP
// Cooperative variant, kept only so the two barriers can be A/B'd where a cooperative launch
// is available. NOT selected at runtime: `cudaLaunchCooperativeKernel` is not reachable
// through torch's op dispatch, and the hand-rolled barrier is already capture-safe.
__global__ void __launch_bounds__(kMegaThreads, 2) shortconv_decode_mega_coop_probe() {
  cg::grid_group grid = cg::this_grid();
  grid.sync();
}
#endif

int smem_bytes_for(int D) {
  // 128 B of reduction scratch (32 warps worth of float, over-provisioned to keep x_sm
  // 16-byte aligned) + the staged [M, D] fp8 activation.
  return 128 + kMegaMaxM * D;
}

}  // namespace

bool shortconv_decode_mega_supported(int64_t dim, int64_t width, int64_t state_len,
                                     int64_t max_tokens) {
  if (width != kConvWidth || state_len != kConvState) return false;
  if (dim <= 0 || dim % (kW4A8Group * 2) != 0) return false;
  // One thread per 8 channels in phases 1/3, and the GEMV wants K == kWarpK per warp pass.
  if (dim / VecTraits<c10::BFloat16>::kVec > kMegaThreads) return false;
  return max_tokens > 0 && max_tokens <= kMegaMaxM;
}

// blocks/SM for THIS kernel at THIS smem, times the SM count -- the co-residency budget the
// hand-rolled barrier needs. Python multiplies nothing and guesses nothing; it just refuses
// when the answer is below kMinBlocks.
int64_t shortconv_decode_mega_grid(int64_t dim) {
  int blocks_per_sm = 0;
  int const smem = smem_bytes_for(static_cast<int>(dim));
  cudaError_t const rc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, shortconv_decode_mega_kernel, kMegaThreads, static_cast<size_t>(smem));
  if (rc != cudaSuccess || blocks_per_sm <= 0) {
    cudaGetLastError();  // clear the sticky error; a failed query is a refusal, not a crash
    return 0;
  }
  int sm_count = 0;
  if (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  int64_t const grid = static_cast<int64_t>(blocks_per_sm) * sm_count;
  return grid < kMinBlocks ? 0 : grid;
}

int64_t shortconv_decode_mega_scratch_bytes(int64_t dim) {
  // barrier[2] u32 (padded to 16B) + x_fp8 + x_scale + bxc + y_fp8 + y_scale, each 16B aligned.
  auto align16 = [](int64_t n) { return (n + 15) / 16 * 16; };
  return align16(8) + align16(kMegaMaxM * dim) + align16(kMegaMaxM * 4) +
         align16(kMegaMaxM * 3 * dim * 2) + align16(kMegaMaxM * dim) + align16(kMegaMaxM * 4);
}

void shortconv_decode_mega(torch::Tensor& out, torch::Tensor const& x,
                           std::optional<torch::Tensor> const& norm_weight, double epsilon,
                           torch::Tensor const& in_wq, torch::Tensor const& in_gs,
                           torch::Tensor const& conv_weight,
                           std::optional<torch::Tensor> const& conv_bias,
                           torch::Tensor& conv_state, torch::Tensor const& state_indices,
                           int64_t null_block_id, torch::Tensor const& out_wq,
                           torch::Tensor const& out_gs, torch::Tensor& scratch,
                           int64_t num_blocks) {
  int64_t const M = x.size(0);
  int64_t const D = x.size(1);
  TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16 &&
                  out.scalar_type() == at::ScalarType::BFloat16 &&
                  conv_state.scalar_type() == at::ScalarType::BFloat16 &&
                  conv_weight.scalar_type() == at::ScalarType::BFloat16,
              "vtl: shortconv_decode_mega is bf16-only");
  TORCH_CHECK(!norm_weight.has_value() ||
                  (norm_weight->scalar_type() == at::ScalarType::BFloat16 &&
                   norm_weight->is_contiguous() && norm_weight->numel() == D),
              "vtl: norm_weight must be a contiguous bf16 [dim] tensor when present");
  TORCH_CHECK(shortconv_decode_mega_supported(D, conv_weight.size(1), conv_state.size(2), M),
              "vtl: shortconv_decode_mega unsupported shape dim=", D,
              " width=", conv_weight.size(1), " state_len=", conv_state.size(2), " M=", M);
  TORCH_CHECK(out.size(0) == M && out.size(1) == D, "vtl: out must be [M, dim]");
  TORCH_CHECK(x.is_contiguous() && out.is_contiguous(),
              "vtl: shortconv_decode_mega needs contiguous x/out");
  TORCH_CHECK(in_wq.scalar_type() == at::ScalarType::Byte &&
                  out_wq.scalar_type() == at::ScalarType::Byte,
              "vtl: packed int4 weights must be uint8 [N, K/2]");
  TORCH_CHECK(in_gs.scalar_type() == at::ScalarType::Float &&
                  out_gs.scalar_type() == at::ScalarType::Float,
              "vtl: group scales must be fp32 [N, K/128]");
  TORCH_CHECK(in_wq.size(0) == 3 * D && in_wq.size(1) == D / 2,
              "vtl: in_proj packed weight must be [3*dim, dim/2]");
  TORCH_CHECK(out_wq.size(0) == D && out_wq.size(1) == D / 2,
              "vtl: out_proj packed weight must be [dim, dim/2]");
  TORCH_CHECK(in_gs.size(0) == 3 * D && in_gs.size(1) == D / kW4A8Group,
              "vtl: in_proj group scales must be [3*dim, dim/128]");
  TORCH_CHECK(out_gs.size(0) == D && out_gs.size(1) == D / kW4A8Group,
              "vtl: out_proj group scales must be [dim, dim/128]");
  TORCH_CHECK(state_indices.scalar_type() == at::ScalarType::Int &&
                  state_indices.size(0) >= M,
              "vtl: state_indices must be int32 with one row per token");
  TORCH_CHECK(scratch.scalar_type() == at::ScalarType::Byte &&
                  scratch.numel() >= shortconv_decode_mega_scratch_bytes(D),
              "vtl: scratch is too small; shortconv_mega.py must allocate ",
              shortconv_decode_mega_scratch_bytes(D), " bytes");
  TORCH_CHECK(conv_state.stride(1) == 1, "vtl: megakernel needs the SD conv-state layout");
  TORCH_CHECK(num_blocks >= kMinBlocks,
              "vtl: shortconv_decode_mega refuses a grid of ", num_blocks,
              " -- every block must be co-resident for the device barrier");

  auto align16 = [](int64_t n) { return (n + 15) / 16 * 16; };
  auto* base = reinterpret_cast<char*>(scratch.data_ptr());
  MegaScratch scr{};
  int64_t off = 0;
  scr.barrier = reinterpret_cast<unsigned int*>(base + off);           off += align16(8);
  scr.x_fp8 = reinterpret_cast<__nv_fp8_storage_t*>(base + off);       off += align16(kMegaMaxM * D);
  scr.x_scale = reinterpret_cast<float*>(base + off);                  off += align16(kMegaMaxM * 4);
  scr.bxc = reinterpret_cast<c10::BFloat16*>(base + off);              off += align16(kMegaMaxM * 3 * D * 2);
  scr.y_fp8 = reinterpret_cast<__nv_fp8_storage_t*>(base + off);       off += align16(kMegaMaxM * D);
  scr.y_scale = reinterpret_cast<float*>(base + off);

  c10::cuda::OptionalCUDAGuard const device_guard(x.device());
  cudaStream_t const stream = c10::cuda::getCurrentCUDAStream();
  int const smem = smem_bytes_for(static_cast<int>(D));

  shortconv_decode_mega_kernel<<<dim3(static_cast<unsigned>(num_blocks)), dim3(kMegaThreads),
                                 smem, stream>>>(
      reinterpret_cast<c10::BFloat16*>(out.data_ptr()), x.const_data_ptr<c10::BFloat16>(),
      norm_weight.has_value() ? norm_weight->const_data_ptr<c10::BFloat16>() : nullptr,
      static_cast<float>(epsilon),
      in_wq.const_data_ptr<uint8_t>(), in_gs.const_data_ptr<float>(),
      conv_weight.const_data_ptr<c10::BFloat16>(),
      conv_bias.has_value() ? conv_bias->const_data_ptr<c10::BFloat16>() : nullptr,
      conv_state.data_ptr<c10::BFloat16>(), state_indices.const_data_ptr<int32_t>(),
      state_indices.stride(0), null_block_id, conv_state.stride(0), conv_state.stride(2),
      out_wq.const_data_ptr<uint8_t>(), out_gs.const_data_ptr<float>(), scr,
      static_cast<int>(M), static_cast<int>(D), static_cast<int>(num_blocks));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (kernel_sync_enabled()) {
    cudaError_t const err = cudaStreamSynchronize(stream);
    TORCH_CHECK(err == cudaSuccess, "vtl shortconv_decode_mega faulted: ",
                cudaGetErrorString(err), " | M=", M, " D=", D, " blocks=", num_blocks,
                " smem=", smem);
  }
}

}  // namespace vtl
