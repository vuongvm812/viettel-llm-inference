// Fused-projection GEMM for the Qwen3.5 MTP head `fc` (ColumnParallelLinear, in=2H=4096,
// out=H=2048): result[M,N] = input[M,K] @ weight[N,K]^T. Runs on every MTP decode step at
// small M (batch*(spec_tokens+1) -- tens of rows), the regime where a cuBLASDx *device* GEMM
// wins by avoiding cuBLAS launch overhead.
//
// STAGE A (this file): correctness-first cuBLASDx tiled GEMM. `input` is the ALREADY
// norm+concatenated [M,K] activation -- the two pre_fc RMSNorms + the cat stay in torch, this
// op replaces only the fc GEMM. Wired behind VTL_ENABLE_MTP_KERNELS (default OFF), armed only
// after `make test-kernel` parity on the H200.
//   ponytail: baseline 64^3 tiles + naive K-loop, no TMA/WGMMA pipelining. Stage B folds the
//   two RMSNorms + cat into the prologue (kills 3 launches + the [M,4096] HBM round-trip) and
//   pipelines the weight loads. Do that ONLY after the on-box A/B shows this beats cuBLAS.
//   ponytail: VTL_FP8 quantizes fc, so the deployed model runs fc in fp8; this bf16 path is the
//   parity baseline. An fp8 GEMM (weights already fp8) is the follow-up that actually halves
//   weight DRAM traffic -- gate it separately after bf16 parity.
//
// !!! cuBLASDx API points I could not compile off-box are marked "VERIFY". On the first on-box
// `make build`, nvcc pins any mismatch to the `.execute()` call site -- fix the Precision arity
// / C arrangement / smem leading dims there. Everything else (tiling, masking, dtype dispatch,
// host checks) is plain CUDA and independent of those.

#include <torch/all.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublasdx.hpp>

#include "fp8_common.cuh"  // c10 cuda guard/stream includes + kernel_sync_enabled + aligned16

namespace vtl {
namespace {

// ---- torch scalar <-> cuBLASDx compute type -------------------------------------------------
// c10::BFloat16 / c10::Half are bit-compatible with __nv_bfloat16 / __half, but we go through
// float on load/store so the smem type is a real CUDA arithmetic type cuBLASDx accepts.
template <typename scalar_t>
struct CudaCompute;
template <>
struct CudaCompute<c10::BFloat16> {
  using type = __nv_bfloat16;
  static __device__ __forceinline__ type from_float(float f) { return __float2bfloat16(f); }
};
template <>
struct CudaCompute<c10::Half> {
  using type = __half;
  static __device__ __forceinline__ type from_float(float f) { return __float2half(f); }
};
template <>
struct CudaCompute<float> {
  using type = float;
  static __device__ __forceinline__ type from_float(float f) { return f; }
};

// Tile shape. M is tiny in decode, so one M-tile usually covers the whole problem (masked).
constexpr int kMT = 64;    // output rows per block-tile (M padded/masked to this)
constexpr int kNT = 64;    // output cols per block-tile
constexpr int kKC = 64;    // K chunk contracted per cuBLASDx execute
constexpr int kBlockDim = 256;
constexpr int kSm = 900;   // H200 / Hopper

// cuBLASDx Block GEMM: C[MT,NT] = A[MT,KC] @ B[KC,NT], accumulating float.
//   A arrangement row_major  -> a_smem[m*KC + k]              (== input tile, contiguous)
//   B arrangement col_major  -> b_smem[n*KC + k] == weight[n][k]  (== the NT semantics: A@Wt^T)
// Mirrors ipm.cuh matmul_NT, extended with mixed precision (bf16/fp16 in, fp32 accumulate).
template <typename cuda_t>
using FcGEMM = decltype(cublasdx::Size<kMT, kNT, kKC>() +
                        // VERIFY: Precision<TA,TB,TC>; fp32 accumulate. If the 3-arg form is not
                        // available in the wheel's cuBLASDx, use Precision<cuda_t>() + fp32 C.
                        cublasdx::Precision<cuda_t, cuda_t, float>() +
                        cublasdx::Type<cublasdx::type::real>() +
                        cublasdx::Function<cublasdx::function::MM>() +
                        cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>() +
                        cublasdx::SM<kSm>() + cublasdx::Block() +
                        cublasdx::BlockDim<kBlockDim>());

template <typename scalar_t>
__global__ void mtp_fc_gemm_kernel(scalar_t* __restrict__ out,          // [M, N] row-major
                                   scalar_t const* __restrict__ inp,    // [M, K] row-major
                                   scalar_t const* __restrict__ wt,     // [N, K] row-major
                                   int const M, int const N, int const K) {
  using cuda_t = typename CudaCompute<scalar_t>::type;
  using CT = CudaCompute<scalar_t>;

  // smem: A tile (cuda_t), B tile (cuda_t), C tile (float). 16B-aligned base.
  extern __shared__ __align__(16) unsigned char raw[];
  cuda_t* a_s = reinterpret_cast<cuda_t*>(raw);
  cuda_t* b_s = a_s + kMT * kKC;
  float* c_s = reinterpret_cast<float*>(b_s + kNT * kKC);  // kMT*kKC + kNT*kKC cuda_t elems in

  int const n0 = blockIdx.x * kNT;  // output column-tile origin
  int const m0 = blockIdx.y * kMT;  // output row-tile origin
  int const tid = threadIdx.x;

  for (int i = tid; i < kMT * kNT; i += kBlockDim) c_s[i] = 0.0f;
  __syncthreads();

  for (int k0 = 0; k0 < K; k0 += kKC) {
    // A tile [MT,KC] row-major, zero-padded past the real M/K.
    for (int i = tid; i < kMT * kKC; i += kBlockDim) {
      int const m = i / kKC, k = i % kKC;
      int const gm = m0 + m, gk = k0 + k;
      float const v =
          (gm < M && gk < K) ? static_cast<float>(inp[static_cast<int64_t>(gm) * K + gk]) : 0.0f;
      a_s[i] = CT::from_float(v);
    }
    // B tile: b_s[n*KC + k] = weight[n0+n][k0+k]  (col-major logical [KC,NT] => A @ weight^T).
    for (int i = tid; i < kNT * kKC; i += kBlockDim) {
      int const n = i / kKC, k = i % kKC;
      int const gn = n0 + n, gk = k0 + k;
      float const v =
          (gn < N && gk < K) ? static_cast<float>(wt[static_cast<int64_t>(gn) * K + gk]) : 0.0f;
      b_s[i] = CT::from_float(v);
    }
    __syncthreads();

    // C += A @ B. beta=1 accumulates across K chunks (C was zeroed above).
    // VERIFY: pointer-API execute expects (alpha, a_smem, b_smem, beta, c_smem) as in ipm.cuh.
    FcGEMM<cuda_t>().execute(1.0f, a_s, b_s, 1.0f, c_s);
    __syncthreads();
  }

  // Epilogue: C tile row-major c_s[m*NT + n] -> out[gm, gn].
  // VERIFY: if cuBLASDx defaults C to col-major, index c_s[n*kMT + m] instead.
  for (int i = tid; i < kMT * kNT; i += kBlockDim) {
    int const m = i / kNT, n = i % kNT;
    int const gm = m0 + m, gn = n0 + n;
    if (gm < M && gn < N)
      out[static_cast<int64_t>(gm) * N + gn] = static_cast<scalar_t>(c_s[i]);
  }
}

template <typename scalar_t>
void launch(torch::Tensor const& out, torch::Tensor const& input, torch::Tensor const& weight,
            cudaStream_t stream) {
  int const M = static_cast<int>(input.size(0));
  int const K = static_cast<int>(input.size(1));
  int const N = static_cast<int>(weight.size(0));
  if (M == 0) return;

  dim3 const grid((N + kNT - 1) / kNT, (M + kMT - 1) / kMT);
  size_t const shmem = (static_cast<size_t>(kMT) * kKC + static_cast<size_t>(kNT) * kKC) *
                           sizeof(typename CudaCompute<scalar_t>::type) +
                       static_cast<size_t>(kMT) * kNT * sizeof(float);

  auto* out_p = out.data_ptr<scalar_t>();
  auto const* in_p = input.const_data_ptr<scalar_t>();
  auto const* w_p = weight.const_data_ptr<scalar_t>();

  mtp_fc_gemm_kernel<scalar_t><<<grid, kBlockDim, shmem, stream>>>(out_p, in_p, w_p, M, N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (kernel_sync_enabled()) {
    cudaError_t const e = cudaStreamSynchronize(stream);
    TORCH_CHECK(e == cudaSuccess, "vtl mtp_fc_gemm faulted: ", cudaGetErrorString(e),
                " dtype=", input.scalar_type(), " M=", M, " N=", N, " K=", K);
  }
}

}  // namespace

void mtp_fc_gemm(torch::Tensor& result, torch::Tensor const& input, torch::Tensor const& weight) {
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() && result.is_contiguous(),
              "vtl: mtp_fc_gemm requires contiguous input/weight/result");
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2 && result.dim() == 2,
              "vtl: mtp_fc_gemm expects 2-D input[M,K], weight[N,K], result[M,N]");
  TORCH_CHECK(input.size(1) == weight.size(1), "vtl: mtp_fc_gemm K mismatch (input K vs weight K)");
  TORCH_CHECK(result.size(0) == input.size(0) && result.size(1) == weight.size(0),
              "vtl: mtp_fc_gemm result must be [M, N]");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type() &&
                  input.scalar_type() == result.scalar_type(),
              "vtl: mtp_fc_gemm input/weight/result dtype must match");
  // Tiles assume a 16-byte-aligned K stride; K=4096 (bf16) satisfies it. Guard anyway.
  TORCH_CHECK(input.size(1) % 8 == 0, "vtl: mtp_fc_gemm expects K a multiple of 8, got ",
              input.size(1));

  c10::cuda::OptionalCUDAGuard const device_guard(input.device());
  cudaStream_t const stream = c10::cuda::getCurrentCUDAStream();

  switch (input.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch<c10::BFloat16>(result, input, weight, stream);
      break;
    case at::ScalarType::Half:
      launch<c10::Half>(result, input, weight, stream);
      break;
    case at::ScalarType::Float:
      launch<float>(result, input, weight, stream);
      break;
    default:
      TORCH_CHECK(false, "vtl: mtp_fc_gemm unsupported dtype ", input.scalar_type());
  }
}

}  // namespace vtl
