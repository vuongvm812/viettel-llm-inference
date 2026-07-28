// vtl._C_w4a8 -- extra CUTLASS W4A8 schedules for the MIG decode shapes.
//
// WHY THIS EXISTS. vLLM's W4A8 kernel ships ten instantiations and a heuristic that keys only
// on M/N/K (w4a8_mm_entry.cu:341-372). It was tuned on a full 132-SM Hopper; the judge runs an
// H200 MIG 1g.18gb slice with ~16 SMs, so the *wave* structure the heuristic implicitly assumes
// is wrong here. Concretely, at decode (M=1-8) after the kernel's swap+transpose:
//
//   layer          CTAs @ TileM=128    waves on 16 SMs    k-iters (K/128)
//   qkv  n3072     24                  1.5  (ragged)      16
//   w13  n16384    128                 8    (fine)        16
//   w2   k8192     16                  1    (deep)        64
//
// A single deep wave (w2) and a ragged 1.5 waves (qkv) are exactly the two cases Stream-K was
// designed for, and `cutlass_w4a8` has no Stream-K arm anywhere. Nine of its ten instantiations
// are also cluster 1x1x1, so TMA multicast is never used. This extension adds the missing arms
// so they can be swept on the box; it changes nothing unless VTL_W4A8_SCHEDULE_V2 is set.
//
// WHY A SEPARATE .so. Built for sm_90a ONLY, with no PTX (deliberate -- a JIT fallback would
// hide an arch mismatch until the first launch). A non-Hopper box therefore cannot load this
// module at all, and it must not be able to take `vtl._C` (the five fused kernels this server
// actually depends on) down with it. `vtl/patches/quant_w4a8.py` imports it only when the
// capability is exactly (9, 0) AND the env is set.
//
// PROVENANCE. Vendored from vLLM v0.25.0
// `csrc/libtorch_stable/quantization/cutlass_w4a8/w4a8_mm_entry.cu` (verified byte-identical to
// tag v0.25.0), converted from the torch stable ABI to the regular one. Every deviation from
// that source is marked `VTL:` below; there are no unmarked changes to the GEMM configuration.
// The three headers under cutlass_extensions/epilogue/ are byte-identical copies of vLLM's.
//
// SWAP + TRANSPOSE, THE THING THAT MAKES TILE NAMES CONFUSING. The kernel passes the WEIGHT as
// CUTLASS operand A and computes {n, m, k, 1}. So TileShape's FIRST dim tiles the linear layer's
// OUTPUT CHANNELS and its SECOND dim tiles TOKENS. "128x16" means 128 output channels x 16
// tokens. A decode-shaped schedule has a small second number.

#include <Python.h>  // must precede any system header (CPython requirement)

#include <torch/all.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/cuda/CUDAGuard.h>

#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/mixed_dtype_utils.hpp"

#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"

#include <cuda_runtime.h>

// Speculative arms, ON by default, so a build that rejects them can be recovered with a single
// -D flag instead of a source edit:
//
//   CUTLASS_NVCC_EXTRA='-DVTL_W4A8_ENABLE_PP=0'   (see setup.py)
//
// PP: the mixed-dtype CollectiveBuilder is specialized on the COOPERATIVE kernel schedule, and
// upstream's own comment says the cooperative epilogue "is the only epi supporting the required
// swap + transpose". Pingpong is therefore expected to static_assert; it is attempted because
// TileM=64 is unreachable any other way (cooperative requires TileM in {128, 256}) and 64 output
// channels x 16 tokens is the only tile that gives w2 (n=2048) more than one wave on 16 SMs.
// N8: GMMA's N=8 atom is legal on Hopper, but the builder may still reject an 8-wide epilogue
// tile. Both are isolated so their failure costs one arm, not the extension.
#ifndef VTL_W4A8_ENABLE_PP
  #define VTL_W4A8_ENABLE_PP 1
#endif
#ifndef VTL_W4A8_ENABLE_N8
  #define VTL_W4A8_ENABLE_N8 1
#endif

namespace vtl::w4a8_v2 {

using namespace cute;

// VTL: local replacement for libtorch_stable/cutlass_extensions/common.hpp's CUTLASS_CHECK,
// which is unusable here -- it is built on STD_TORCH_CHECK, i.e. the stable ABI we just left.
#define VTL_CUTLASS_CHECK(status)                                      \
  {                                                                    \
    cutlass::Status _vtl_err = (status);                               \
    TORCH_CHECK(_vtl_err == cutlass::Status::kSuccess,                 \
                "vtl w4a8 v2: CUTLASS error: ",                        \
                cutlassGetStatusString(_vtl_err));                     \
  }

// -------------------------------------------------------------------------------------
// Static configuration shared across all instantiations
// -------------------------------------------------------------------------------------
using MmaType = cutlass::float_e4m3_t;  // A/scale element type
using QuantType = cutlass::int4b_t;     // B element type (packed int4)

static int constexpr TileShapeK = 128 * 8 / sizeof_bits<MmaType>::value;
static int constexpr ScalePackSize = 8;  // pack 8 scale elements together

// A matrix configuration
using ElementA = MmaType;                   // Element type for A matrix operand
using LayoutA = cutlass::layout::RowMajor;  // Layout type for A matrix operand
using LayoutA_Transpose =
    typename cutlass::layout::LayoutTranspose<LayoutA>::type;
constexpr int AlignmentA =
    128 / cutlass::sizeof_bits<
              ElementA>::value;  // Memory access granularity/alignment of A
                                 // matrix in units of elements (up to 16 bytes)
using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;

// B matrix configuration
using ElementB = QuantType;  // Element type for B matrix operand
using LayoutB =
    cutlass::layout::ColumnMajor;  // Layout type for B matrix operand
using LayoutB_Transpose =
    typename cutlass::layout::LayoutTranspose<LayoutB>::type;
constexpr int AlignmentB =
    128 / cutlass::sizeof_bits<
              ElementB>::value;  // Memory access granularity/alignment of B
                                 // matrix in units of elements (up to 16 bytes)
using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;

// Define the CuTe layout for reordered quantized tensor B
// LayoutAtomQuant places values that will be read by the same thread in
// contiguous locations in global memory. It specifies the reordering within a
// single warp's fragment
using LayoutAtomQuant =
    decltype(cutlass::compute_memory_reordering_atom<MmaType>());
using LayoutB_Reordered = decltype(cute::tile_to_shape(
    LayoutAtomQuant{}, Layout<Shape<int, int, int>, StrideB>{}));

// Group-wise scales
using ElementScale = MmaType;
using LayoutScale = cutlass::layout::RowMajor;

// Per-tok, per-chan scales
using ElementSChannel = float;

// C/D matrix configuration
using ElementC =
    cutlass::bfloat16_t;  // Element type for C and D matrix operands
using LayoutC =
    cutlass::layout::RowMajor;  // Layout type for C and D matrix operands
constexpr int AlignmentC =
    128 / cutlass::sizeof_bits<
              ElementC>::value;  // Memory access granularity/alignment of C
                                 // matrix in units of elements (up to 16 bytes)

using ElementD = ElementC;
using LayoutD = LayoutC;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

// Core kernel configurations
using ElementAccumulator = float;     // Element type for internal accumulation
using ElementCompute = float;         // Element type for epilogue computation
using ArchTag = cutlass::arch::Sm90;  // Tag indicating the minimum SM that
                                      // supports the intended feature
using OperatorClass = cutlass::arch::OpClassTensorOp;  // Operator class tag
using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

// ----------------------------------------------------------------------------
// Kernel template -- Tile/Cluster shapes, kernel/epilogue schedule, scheduler
// ----------------------------------------------------------------------------
// VTL: upstream's W4A8GemmKernel takes <TileShape_MN, ClusterShape_MNK> and hardcodes the
// cooperative schedules + the default (persistent) tile scheduler. The three extra params are
// the whole point of this file; every default below reproduces upstream exactly, so an arm
// declared with two params is byte-for-byte upstream's kernel.
template <class TileShape_MN, class ClusterShape_MNK,
          class KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative,
          class EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative,
          class TileScheduler = cutlass::gemm::PersistentScheduler>
struct W4A8GemmKernelV2 {
  using TileShape =
      decltype(cute::append(TileShape_MN{}, cute::Int<TileShapeK>{}));
  using ClusterShape = ClusterShape_MNK;

  // Epilogue per-tok, per-chan scales
  using ChTokScalesEpilogue =
      typename vllm::c3x::ScaledEpilogue<ElementAccumulator, ElementD,
                                         TileShape>;
  using EVTCompute = typename ChTokScalesEpilogue::EVTCompute;
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, TileShape, ClusterShape, EpilogueTileType,
          ElementAccumulator, ElementSChannel,
          // Transpose layout of D here since we use explicit swap + transpose
          // the void type for C tells the builder to allocate 0 smem for the C
          // matrix. We can enable this if beta == 0 by changing ElementC to
          // void below.
          ElementC, typename cutlass::layout::LayoutTranspose<LayoutC>::type,
          AlignmentC, ElementD,
          typename cutlass::layout::LayoutTranspose<LayoutD>::type, AlignmentD,
          EpilogueSchedule,  // This is the only epi supporting the required
                             // swap + transpose.
          EVTCompute>::CollectiveOp;

  // The Scale information must get paired with the operand that will be scaled.
  // In this example, B is scaled so we make a tuple of B's information and the
  // scale information.
  using CollectiveMainloopShuffled =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass,
          cute::tuple<ElementB, cutlass::Array<ElementScale, ScalePackSize>>,
          LayoutB_Reordered, AlignmentB, ElementA, LayoutA_Transpose,
          AlignmentA, ElementAccumulator, TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  // VTL: 4th GemmUniversal param. Upstream omits it, which defaults to void ->
  // PersistentScheduler; passing PersistentScheduler explicitly is the same kernel.
  using GemmKernelShuffled = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,  // Indicates ProblemShape
      CollectiveMainloopShuffled, CollectiveEpilogue, TileScheduler>;
  using GemmShuffled =
      cutlass::gemm::device::GemmUniversalAdapter<GemmKernelShuffled>;

  using StrideC = typename GemmKernelShuffled::StrideC;
  using StrideD = typename GemmKernelShuffled::StrideD;
  using StrideS = typename CollectiveMainloopShuffled::StrideScale;

  static constexpr bool kStreamK =
      std::is_same_v<TileScheduler, cutlass::gemm::StreamKScheduler>;

  static at::Tensor mm(at::Tensor const& A,
                       at::Tensor const& B,             // already packed
                       at::Tensor const& group_scales,  // already packed
                       int64_t group_size,
                       at::Tensor const& channel_scales,
                       at::Tensor const& token_scales) {
    // TODO: param validation
    int m = A.size(0);
    int k = A.size(1);
    int n = B.size(1);

    // safely cast group_size to int
    TORCH_CHECK(
        group_size > 0 && group_size <= std::numeric_limits<int>::max(),
        "group_size out of supported range for int: ", group_size);
    int const group_size_int = static_cast<int>(group_size);

    // Allocate output
    // VTL: at::cuda guard/stream/empty in place of the stable-ABI trio, and ElementD is
    // hardcoded bf16 (the epilogue emits bf16 and this op has no out_type argument).
    const c10::cuda::CUDAGuard device_guard(A.device());
    auto stream = at::cuda::getCurrentCUDAStream(A.device().index());
    at::Tensor D = at::empty({m, n}, A.options().dtype(at::kBFloat16));
    // prepare arg pointers
    auto A_ptr = static_cast<MmaType const*>(A.const_data_ptr());
    auto B_ptr = static_cast<QuantType const*>(B.const_data_ptr());
    auto D_ptr = static_cast<ElementD*>(D.data_ptr());
    // can we avoid hardcode the 8 here
    auto S_ptr =
        static_cast<cutlass::Array<ElementScale, ScalePackSize> const*>(
            group_scales.const_data_ptr());

    // runtime layout for B
    auto shape_B = cute::make_shape(n, k, 1);
    LayoutB_Reordered layout_B_reordered =
        cute::tile_to_shape(LayoutAtomQuant{}, shape_B);

    // strides
    int const scale_k = cutlass::ceil_div(k, group_size_int);
    StrideA stride_A =
        cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    // Reverse stride here due to swap and transpose
    StrideD stride_D =
        cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(n, m, 1));
    StrideS stride_S = cutlass::make_cute_packed_stride(
        StrideS{}, cute::make_shape(n, scale_k, 1));

    /// Populates a Gemm::Arguments structure from the given arguments
    /// Swap the A and B tensors, as well as problem shapes here.
    using Args = typename GemmShuffled::Arguments;
    using MainloopArguments = typename GemmKernelShuffled::MainloopArguments;
    using EpilogueArguments = typename GemmKernelShuffled::EpilogueArguments;

    MainloopArguments mainloop_arguments{
        B_ptr, layout_B_reordered, A_ptr,         stride_A,
        S_ptr, stride_S,           group_size_int};

    EpilogueArguments epilogue_arguments{
        ChTokScalesEpilogue::prepare_args(channel_scales, token_scales),
        nullptr,
        {},  // no C
        D_ptr,
        stride_D};

    Args arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                   {n, m, k, 1},  // shape
                   mainloop_arguments,
                   epilogue_arguments};

    // VTL: the ONLY behavioural addition. CUTLASS's Stream-K scheduler defaults to
    // DecompositionMode::Heuristic, which falls back to plain data-parallel whenever the tile
    // grid already covers the device -- and at decode our grids are 16-128 CTAs, so the
    // heuristic would silently give us the arm we already have. Forcing StreamK is what makes
    // the *_sk arms mean anything. hw_info is left default on purpose: CUTLASS then queries the
    // SM count, which is what upstream's persistent scheduler does too (on a MIG slice CUDA
    // reports the slice's SMs, which is the number Stream-K should be splitting for -- worth
    // confirming on the box, because a 132 here would size the splits for the wrong device).
    if constexpr (kStreamK) {
      using DecompositionMode =
          typename GemmKernelShuffled::TileScheduler::DecompositionMode;
      arguments.scheduler.decomposition_mode = DecompositionMode::StreamK;
    }

    // Workspace. Stream-K needs this ZEROED (it holds the reduction barriers), which is what
    // initialize() below does -- hence the get/empty/initialize flow is kept verbatim.
    size_t workspace_size = GemmShuffled::get_workspace_size(arguments);
    at::Tensor workspace = at::empty({static_cast<int64_t>(workspace_size)},
                                     A.options().dtype(at::kByte));

    // Run GEMM
    GemmShuffled gemm;
    VTL_CUTLASS_CHECK(gemm.can_implement(arguments));
    VTL_CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
    VTL_CUTLASS_CHECK(gemm.run(stream));

    return D;
  }
};

// ----------------------------------------------------------------------------
// Kernel instantiations and dispatch logic
// ----------------------------------------------------------------------------
// One `using` + one dispatch branch per arm, deliberately not a macro: when an arm has to be
// dropped the diff is a single self-contained block.

// Stream-K, cooperative. The primary hypothesis: w2 (16 CTAs x 64 k-iters on 16 SMs) is one
// deep wave and qkv (24 CTAs) is 1.5 ragged waves; both are textbook Stream-K.
using Kernel_128x16_1x1x1_sk =
    W4A8GemmKernelV2<Shape<_128, _16>, Shape<_1, _1, _1>,
                     cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                     cutlass::epilogue::TmaWarpSpecializedCooperative,
                     cutlass::gemm::StreamKScheduler>;

using Kernel_128x32_1x1x1_sk =
    W4A8GemmKernelV2<Shape<_128, _32>, Shape<_1, _1, _1>,
                     cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                     cutlass::epilogue::TmaWarpSpecializedCooperative,
                     cutlass::gemm::StreamKScheduler>;

// Cluster multicast. HONEST NOTE: after swap+transpose the multicast operand is the
// ACTIVATION, which at decode is 1-8 tokens -- there is very little to share. Swept anyway
// because the pair of CTAs also halves the number of TMA loads issued for it.
using Kernel_128x16_2x1x1 =
    W4A8GemmKernelV2<Shape<_128, _16>, Shape<_2, _1, _1>>;

#if VTL_W4A8_ENABLE_N8
// Narrower token tile than any stock instantiation. At M<=8 the stock 128x16 arm computes and
// discards at least half its epilogue tile.
using Kernel_128x8_1x1x1 = W4A8GemmKernelV2<Shape<_128, _8>, Shape<_1, _1, _1>>;
#endif

#if VTL_W4A8_ENABLE_PP
// VTL: the epilogue schedule here is `TmaWarpSpecialized`, NOT a "pingpong" epilogue -- CUTLASS
// 3.x/4.x has no `cutlass::epilogue::TmaWarpSpecializedPingpong` for non-grouped GEMM; the
// pingpong MAINLOOP pairs with the plain warp-specialized epilogue. See the ENABLE_PP note at
// the top for why this arm is expected to fail to build.
using Kernel_64x16_1x1x1_pp =
    W4A8GemmKernelV2<Shape<_64, _16>, Shape<_1, _1, _1>,
                     cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                     cutlass::epilogue::TmaWarpSpecialized>;

using Kernel_64x32_1x1x1_pp =
    W4A8GemmKernelV2<Shape<_64, _32>, Shape<_1, _1, _1>,
                     cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                     cutlass::epilogue::TmaWarpSpecialized>;
#endif

at::Tensor mm_dispatch_v2(at::Tensor const& A, at::Tensor const& B,
                          at::Tensor const& group_scales, int64_t group_size,
                          at::Tensor const& channel_scales,
                          at::Tensor const& token_scales,
                          std::string const& schedule) {
  if (schedule == "128x16_1x1x1_sk") {
    return Kernel_128x16_1x1x1_sk::mm(A, B, group_scales, group_size,
                                      channel_scales, token_scales);
  } else if (schedule == "128x32_1x1x1_sk") {
    return Kernel_128x32_1x1x1_sk::mm(A, B, group_scales, group_size,
                                      channel_scales, token_scales);
  } else if (schedule == "128x16_2x1x1") {
    return Kernel_128x16_2x1x1::mm(A, B, group_scales, group_size,
                                   channel_scales, token_scales);
#if VTL_W4A8_ENABLE_N8
  } else if (schedule == "128x8_1x1x1") {
    return Kernel_128x8_1x1x1::mm(A, B, group_scales, group_size,
                                  channel_scales, token_scales);
#endif
#if VTL_W4A8_ENABLE_PP
  } else if (schedule == "64x16_1x1x1_pp") {
    return Kernel_64x16_1x1x1_pp::mm(A, B, group_scales, group_size,
                                     channel_scales, token_scales);
  } else if (schedule == "64x32_1x1x1_pp") {
    return Kernel_64x32_1x1x1_pp::mm(A, B, group_scales, group_size,
                                     channel_scales, token_scales);
#endif
  }
  TORCH_CHECK(false, "vtl w4a8 v2: unknown or not-compiled-in schedule: ",
              schedule);
  return {};
}

// Stock `_C::cutlass_w4a8_mm`, reached through the dispatcher because it lives in a different
// .so (vllm._C_stable_libtorch) with no C++ header we can link against.
//
// callBoxed, not typed<>(): `typed<Sig>()` asserts our C++ signature infers to the registered
// schema, and the schema has a `ScalarType?` argument whose C++ inference has moved across
// torch releases. A mismatch there is a hard error at the FIRST forward -- i.e. mid-benchmark.
// IValues are dynamically typed, so this cannot drift; the boxing costs ~1us, on a path that
// only runs for M > threshold (prefill), where the GEMM itself is milliseconds.
at::Tensor stock_mm(at::Tensor const& A, at::Tensor const& B,
                    at::Tensor const& group_scales, int64_t group_size,
                    at::Tensor const& channel_scales,
                    at::Tensor const& token_scales) {
  static const auto op =
      c10::Dispatcher::singleton().findSchemaOrThrow("_C::cutlass_w4a8_mm", "");
  // Trailing Nones are out_type and maybe_schedule: no dtype override, kernel heuristic.
  std::vector<c10::IValue> stack{A,
                                 B,
                                 group_scales,
                                 group_size,
                                 channel_scales,
                                 token_scales,
                                 c10::IValue(),
                                 c10::IValue()};
  op.callBoxed(stack);
  TORCH_CHECK(stack.size() == 1 && stack[0].isTensor(),
              "vtl w4a8 v2: _C::cutlass_w4a8_mm returned an unexpected stack");
  return stack[0].toTensor();
}

// The M branch lives HERE, not in Python, on purpose: `VtlW4A8LinearMethod.apply` is traced by
// support_torch_compile with fullgraph=True, where a branch on the symbolic token count is
// either a graph break (= engine crash) or a recompile per batch bucket. Inside an opaque
// custom op it is neither -- the graph sees one node.
at::Tensor w4a8_mm_v2(at::Tensor const& a, at::Tensor const& b_q,
                      at::Tensor const& group_scales, int64_t group_size,
                      at::Tensor const& channel_scales,
                      at::Tensor const& token_scales,
                      std::string const& schedule, int64_t m_threshold) {
  if (a.size(0) > m_threshold) {
    return stock_mm(a, b_q, group_scales, group_size, channel_scales,
                    token_scales);
  }
  return mm_dispatch_v2(a, b_q, group_scales, group_size, channel_scales,
                        token_scales, schedule);
}

}  // namespace vtl::w4a8_v2

// FRAGMENT, not TORCH_LIBRARY: `vtl._C` owns TORCH_LIBRARY(vllm_cuda) and only one .so may.
// quant_w4a8.py imports vtl._C before vtl._C_w4a8 to keep that ownership unambiguous.
TORCH_LIBRARY_FRAGMENT(vllm_cuda, m) {
  m.def(
      "w4a8_mm_v2(Tensor a, Tensor b_q, Tensor group_scales, int group_size, "
      "Tensor channel_scales, Tensor token_scales, str schedule, "
      "int m_threshold) -> Tensor");
}

TORCH_LIBRARY_IMPL(vllm_cuda, CUDA, m) {
  m.impl("w4a8_mm_v2", TORCH_FN(vtl::w4a8_v2::w4a8_mm_v2));
}

// The ops arrive through the static initializers above, at dlopen. This exists only so that
// `import vtl._C_w4a8` finds an init function and does not raise.
//
// VTL: hand-rolled instead of vtl/csrc/torch_bindings.cpp's PYBIND11_MODULE. That file is a
// .cpp built by the host compiler; this one is a .cu, and pulling pybind11 through nvcc buys
// nothing here (there is nothing to bind -- everything is a dispatcher op) for a real risk of
// a template/compiler-version fight in the one TU we cannot compile off-box. The name is
// spelled out rather than built from TORCH_EXTENSION_NAME because token-pasting a macro into
// PyInit_ needs two levels of indirection to say less clearly.
extern "C" {
static struct PyModuleDef vtl_c_w4a8_module = {
    PyModuleDef_HEAD_INIT, "_C_w4a8", nullptr, -1, nullptr, nullptr, nullptr,
    nullptr, nullptr,
};
PyMODINIT_FUNC PyInit__C_w4a8(void) {
  return PyModule_Create(&vtl_c_w4a8_module);
}
}
