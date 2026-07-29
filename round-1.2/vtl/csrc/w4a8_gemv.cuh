// W4A8 GEMV for the decode megakernel: fp8-e4m3 activation x group-128 int4 weight.
//
// WHY THIS EXISTS INSTEAD OF ops.cutlass_w4a8_mm. Not to beat CUTLASS at GEMM -- to be
// *callable from inside another kernel*. The megakernel needs the two short-conv projections
// as DEVICE FUNCTIONS between its phases; a CUTLASS kernel is a launch. At decode M<=8 the
// GEMM is pure weight streaming anyway (6 MB for in_proj at N=6144,K=2048), so a GEMV that
// hits peak bandwidth is not leaving anything on the table -- there is no reuse to exploit.
//
// WEIGHT LAYOUT IS OURS, AND HAS TO BE. `quant_w4a8.py` stores `weight_packed` after
// `ops.cutlass_encode_and_reorder_int4b` and `weight_group_scale` after
// `ops.cutlass_pack_scale_fp8`: both are CUTLASS's internal mixed-input MMA tile orders, not
// addressable from hand-written code. So `shortconv_mega.py` re-derives a plain view from the
// same quantized values at load time (see `_build_mega_weights`):
//
//   wq   : uint8  [N, K/2]   row-major; byte j = k(2j) in the LOW nibble, k(2j+1) in the HIGH
//                            nibble, each a two's-complement signed int4 in [-8, 7]
//   gs   : float  [N, K/G]   row-major group scale, WITH the per-channel residual already
//                            folded in, so the epilogue is one multiply by the token scale
//
// with G = 128 (`quant_w4a8.GROUP_SIZE`). That costs a second copy of the weights (~84 MB for
// LFM2.5's 10 conv layers) and is the price of a fusable projection.
//
// NUMERICS ARE NOT BIT-EXACT vs CUTLASS, and cannot be: CUTLASS accumulates fp8xint4 products
// inside an MMA in its own tile order. Here each lane sums exactly one 64-wide half-group in
// fp32, scales by that group's scale, and the warp reduces. Parity is checked to a tolerance
// in bench/test_w4a8_gemv.py, against a torch reference built from the same wq/gs.
//
// SHAPE: K must be a multiple of 128 and G, N a multiple of 1 (rows are independent).
// Lane l owns k in [64l, 64l+64), which is exactly half of group l/2 -- so a lane touches ONE
// group scale, and `K == 2048` puts the whole row in one warp pass. Wider K loops the warp.
#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace vtl {

// Group size of the W4A8 weight quantization. Must match quant_w4a8.GROUP_SIZE.
constexpr int kW4A8Group = 128;
// K each lane owns per warp pass. 64 nibbles = 32 packed bytes = 2 aligned 16-byte loads.
constexpr int kLaneK = 64;
constexpr int kWarpK = kLaneK * 32;  // 2048 -- one warp pass covers a full LFM2 row

// Two's-complement nibble -> signed int4 value. Sign-extend bit 3.
__device__ __forceinline__ int nib_to_int4(unsigned nib) {
  return static_cast<int>(nib & 0xFu) - static_cast<int>((nib & 0x8u) << 1);
}

// fp8-e4m3 storage byte -> float, via the hardware converter (the software path in c10 is
// ~10 ALU ops per element, and this runs K times per row).
__device__ __forceinline__ float fp8_to_float(__nv_fp8_storage_t v) {
  __half h;
  h = __nv_cvt_fp8_to_halfraw(v, __NV_E4M3);
  return __half2float(h);
}

// One warp, one output row, all `M` tokens. Returns nothing; writes `acc[m]` (fp32, pre
// token-scale) which the caller scales and stores.
//
// `x_fp8` is [M][K] in SHARED memory (row stride `x_stride`), staged once by the caller so
// the weight stream is the only global traffic. `wq_row`/`gs_row` point at row `n`.
//
// The M loop is INSIDE the weight load on purpose: at M=8 a naive token-outer loop would
// stream the 6 MB of in_proj weights eight times, which is the entire cost of the phase.
template <int kMaxM>
__device__ __forceinline__ void w4a8_gemv_row(const uint8_t* __restrict__ wq_row,
                                              const float* __restrict__ gs_row,
                                              const __nv_fp8_storage_t* __restrict__ x_fp8,
                                              int x_stride, int M, int K, float* acc) {
  int const lane = threadIdx.x & 31;
#pragma unroll
  for (int m = 0; m < kMaxM; ++m) acc[m] = 0.0f;

  for (int k0 = 0; k0 < K; k0 += kWarpK) {
    int const k_lane = k0 + lane * kLaneK;
    if (k_lane >= K) break;
    // 32 packed bytes = the lane's 64 k values, as two 16-byte loads.
    uint4 const* const wsrc = reinterpret_cast<uint4 const*>(wq_row + (k_lane >> 1));
    uint4 const w0 = wsrc[0];
    uint4 const w1 = wsrc[1];
    float const scale = gs_row[k_lane / kW4A8Group];

    uint32_t const words[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
    // The token loop is unrolled over the COMPILE-TIME bound so `acc[m]` and `words[wi]`
    // are register-indexed; a runtime `m` would push both to local memory.
#pragma unroll
    for (int m = 0; m < kMaxM; ++m) {
      if (m >= M) continue;
      __nv_fp8_storage_t const* const xrow = x_fp8 + m * x_stride + k_lane;
      float part = 0.0f;
#pragma unroll
      for (int wi = 0; wi < 8; ++wi) {
        uint32_t const word = words[wi];
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          unsigned const byte = (word >> (8 * b)) & 0xFFu;
          int const k = wi * 8 + b * 2;
          part = fmaf(fp8_to_float(xrow[k]), static_cast<float>(nib_to_int4(byte)), part);
          part = fmaf(fp8_to_float(xrow[k + 1]), static_cast<float>(nib_to_int4(byte >> 4)),
                      part);
        }
      }
      acc[m] = fmaf(part, scale, acc[m]);
    }
  }

  // Warp reduction: every lane ends holding the row's full dot product.
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
    for (int m = 0; m < kMaxM; ++m) {
      acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], off);
    }
  }
}

}  // namespace vtl
