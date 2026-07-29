// vtl._C_w4a8 -- extra CUTLASS W4A8 schedules for the MIG decode shapes.
//
// WHY THIS EXISTS. vLLM's W4A8 kernel ships ten instantiations and a heuristic that keys only
// on M/N/K (w4a8_mm_entry.cu:341-372). It was tuned on a full 132-SM Hopper; the judge runs an
// H200 MIG 1g.18gb slice with ~16 SMs, so the *wave* structure the heuristic implicitly assumes
// is wrong here. It also never takes over prefill, so the two Hopper knobs that require more
// than one TOKEN tile -- cluster multicast of the weights, and raster order -- had no path to
// fire at all; the *_pf arms and the prefill band exist for that.
//
// Concretely, at decode (M=1-8) after the kernel's swap+transpose:
//
//   layer          CTAs @ TileM=128    waves on 16 SMs    k-iters (K/128)
//   qkv  n3072     24                  1.5  (ragged)      16
//   w13  n16384    128                 8    (fine)        16
//   w2   k8192     16                  1    (deep)        64
//
// A single deep wave (w2) and a ragged 1.5 waves (qkv) are exactly the two cases Stream-K was
// designed for, and `cutlass_w4a8` has no Stream-K arm anywhere. Nine of its ten instantiations
// are also cluster 1x1x1, so TMA multicast is never used. This extension adds the missing arms
// so they can be swept on the box; it changes nothing unless VTL_W4A8_SCHEDULE_V2 (decode) or
// VTL_W4A8_SCHEDULE_V2_PREFILL + VTL_W4A8_V2_PREFILL_MAX (prefill) is set.
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
// ATen/cuda/CUDAContext.h pulls cusparse.h, absent from the runtime image;
// only the stream getter is needed.
#include <c10/cuda/CUDAStream.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdlib>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"

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
static int constexpr PackFactor = 8;     // 8 int4s per int32 word of B (w4a8_mm_entry.cu:41)

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
// Hardware info -- queried ONCE per process
// ----------------------------------------------------------------------------
// Upstream leaves Arguments::hw_info default-constructed (sm_count = 0). That is not inert:
// every initialize() then runs KernelHardwareInfo::query_device_multiprocessor_count(), i.e. a
// cudaDeviceGetAttribute round-trip, and this model issues ~50 W4A8 GEMMs per decode step on a
// host-sensitive path. Querying once and caching removes ~50 host round-trips/step.
//
// It also buys the escape hatch this whole extension needs. sm_count is what BOTH schedulers
// size themselves from -- the persistent one launches min(sm_count, tiles) CTAs, and Stream-K
// derives its k-split units and its workspace from it. If a MIG 1g.18gb slice reports 132
// (the physical H200) rather than its own ~16, every persistent grid is 8x oversubscribed and
// Stream-K splits for a device that is not there. Whether MIG reports the slice or the physical
// GPU could not be settled from NVIDIA's docs, so VTL_W4A8_SM_COUNT overrides it without a
// rebuild. The value CUTLASS ends up using is logged at load by quant_w4a8.py.
//
// Function-local static: one CUDA device per process, which is what a MIG slice gives us.
static cutlass::KernelHardwareInfo const& hardware_info() {
  static cutlass::KernelHardwareInfo const info = [] {
    cutlass::KernelHardwareInfo hw;
    char const* env = std::getenv("VTL_W4A8_SM_COUNT");
    int const override_sm = (env && *env) ? std::atoi(env) : 0;
    hw.sm_count = override_sm > 0
                      ? override_sm
                      : cutlass::KernelHardwareInfo::query_device_multiprocessor_count();
    return hw;
  }();
  return info;
}

// Reported through `vllm_cuda::w4a8_sm_count()` so the boot log can state the number CUTLASS is
// actually scheduling against, rather than the one torch reports.
int64_t sm_count() { return hardware_info().sm_count; }

// ----------------------------------------------------------------------------
// Scheduler tuning -- the runtime half of an arm
// ----------------------------------------------------------------------------
// Tile/cluster/schedule are types; decomposition mode, k-split count, reduction mode, raster
// order and swizzle are runtime fields on Arguments::scheduler. Carrying them as a traits type
// keeps one arm = one `using`, and lets the illegal combination below be a compile error.
//
// raster: 0 = leave CUTLASS's Heuristic, 1 = AlongM, 2 = AlongN. NOT an enum, because the enum
// lives on the scheduler type and each arm would have to name its own.
struct SchedDefault {
  static constexpr int kSplits = 1;
  static constexpr bool kSplitK = false;
  static constexpr bool kNondeterministic = false;
  static constexpr int kRaster = 0;
  static constexpr int kMaxSwizzle = 1;
};

// Nondeterministic reduction. Deterministic makes Stream-K peers hand off in K order behind a
// lock; nondeterministic drops that to two rendezvous (first writer, last accumulator). The
// numerical cost is fp32 partials summed out of order -- ~1e-7 relative, four orders below the
// 1e-2 the parity test allows. The REAL cost is that results stop being bit-reproducible
// run-to-run, so a flake and a regression become indistinguishable by re-running. Sweep-only:
// do not make this the shipped default.
struct SchedNondet : SchedDefault {
  static constexpr bool kNondeterministic = true;
};

// Fixed split-K. For w2 (k=8192 -> 64 k-iters over 16 tiles = exactly one deep wave) an even
// 4-way split is a cleaner decomposition than Stream-K's ragged unit assignment. CUTLASS clamps
// to min(splits, sm_count, k_tiles) internally.
struct SchedSplitK4 : SchedDefault {
  static constexpr int kSplits = 4;
  static constexpr bool kSplitK = true;
};

// Weight-major raster. Only meaningful when the token grid has more than one tile, i.e. prefill
// -- see the note on the prefill arms below.
struct SchedRasterN : SchedDefault {
  static constexpr int kRaster = 2;
};

struct SchedRasterNSwizzle4 : SchedDefault {
  static constexpr int kRaster = 2;
  static constexpr int kMaxSwizzle = 4;
};

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
          class TileScheduler = cutlass::gemm::PersistentScheduler,
          class SchedTune = SchedDefault,
          int Stages = 0>  // 0 = CUTLASS's auto carveout, i.e. upstream
struct W4A8GemmKernelV2 {
  using TileShape =
      decltype(cute::append(TileShape_MN{}, cute::Int<TileShapeK>{}));
  using ClusterShape = ClusterShape_MNK;

  // VTL: splits > 1 is a HARD can_implement rejection under forced StreamK -- CUTLASS only
  // honours it for SplitK or Heuristic (sm90_tile_scheduler_stream_k.hpp can_implement). Left
  // to runtime that is a CUTLASS_CHECK throw at the first forward, i.e. mid-benchmark.
  static_assert(SchedTune::kSplits == 1 || SchedTune::kSplitK,
                "vtl w4a8 v2: kSplits > 1 requires kSplitK (StreamK rejects an explicit split)");
  static_assert(SchedTune::kRaster >= 0 && SchedTune::kRaster <= 2,
                "vtl w4a8 v2: kRaster is 0 (Heuristic), 1 (AlongM) or 2 (AlongN)");

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

  // VTL: Stages == 0 keeps upstream's auto carveout verbatim; any other value pins the mainloop
  // pipeline depth. Auto lands around 19 stages at 128x16 (~11.3 KB/stage against Hopper's
  // 227 KB), so the only interesting direction is shallower -- fewer stages means less TMA in
  // flight, which may not matter on the 16-k-iteration shapes.
  using StageCountType = std::conditional_t<
      Stages == 0,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::StageCount<Stages>>;

  // The Scale information must get paired with the operand that will be scaled.
  // In this example, B is scaled so we make a tuple of B's information and the
  // scale information.
  using CollectiveMainloopShuffled =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass,
          cute::tuple<ElementB, cutlass::Array<ElementScale, ScalePackSize>>,
          LayoutB_Reordered, AlignmentB, ElementA, LayoutA_Transpose,
          AlignmentA, ElementAccumulator, TileShape, ClusterShape,
          StageCountType, KernelSchedule>::CollectiveOp;

  // VTL: 4th GemmUniversal param. Upstream omits it, which defaults to void ->
  // PersistentScheduler; passing PersistentScheduler explicitly is the same kernel.
  using GemmKernelShuffled = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,  // Indicates ProblemShape
      CollectiveMainloopShuffled, CollectiveEpilogue, TileScheduler>;
  using GemmShuffled =
      cutlass::gemm::device::GemmUniversalAdapter<GemmKernelShuffled>;

  // VTL: CUTLASS does NOT bounds-check an explicit StageCount -- compute_stage_count_or_override
  // returns the requested number unchanged, and an over-budget kernel first fails inside
  // initialize(), where cudaFuncSetAttribute(MaxDynamicSharedMemorySize) rejects it. That is a
  // CUTLASS_CHECK throw at the first forward of a benchmark. With no H200 to test on, the Docker
  // build is the only place this can be caught, so assert it at compile time.
  //
  // Assert on the KERNEL's SharedStorage, which is the exact byte count CUTLASS hands to
  // cudaFuncSetAttribute. Summing the two collectives' SharedStorage instead over-counts and
  // rejects arms that fit: StageCountAutoCarveout already subtracts the epilogue before sizing
  // the mainloop pipeline, and GemmUniversal overlaps the two in its own storage layout. That
  // version of this assert failed the build on 128x32_1x1x1_sk and 128x8_1x1x1, both of which
  // are upstream-shaped arms that have always compiled.
  // 232448 = cutlass::detail::sm90_smem_capacity_bytes (227 KB, not 228).
  static_assert(sizeof(typename GemmKernelShuffled::SharedStorage) <= 232448,
                "vtl w4a8 v2: kernel smem exceeds Hopper's 227KB/SM; lower Stages");

  using StrideC = typename GemmKernelShuffled::StrideC;
  using StrideD = typename GemmKernelShuffled::StrideD;
  using StrideS = typename CollectiveMainloopShuffled::StrideScale;

  static constexpr bool kStreamK =
      std::is_same_v<TileScheduler, cutlass::gemm::StreamKScheduler>;

  // VTL: reported by w4a8_stages() so the boot log can state the pipeline depth CUTLASS chose,
  // which is otherwise invisible for the Stages == 0 (auto carveout) arms.
  static constexpr int kStages = CollectiveMainloopShuffled::DispatchPolicy::Stages;
  // After swap+transpose the problem is {n, m, k}: CUTLASS's N dim tiles TOKENS.
  static constexpr int kTileN = cute::size<1>(TileShape{});
  static constexpr int kClusterN = cute::size<1>(ClusterShape{});

  static at::Tensor mm(at::Tensor const& A,
                       at::Tensor const& B,             // already packed
                       at::Tensor const& group_scales,  // already packed
                       int64_t group_size,
                       at::Tensor const& channel_scales,
                       at::Tensor const& token_scales) {
    // VTL: upstream's `// TODO: param validation` is answered in check_args(), called once at
    // the op entry (w4a8_mm_v2) so it also covers the M > threshold forward to the stock kernel.
    int m = A.size(0);
    int k = A.size(1);
    int n = B.size(1);

    // VTL: a cluster wider than the token grid is a SILENT 2x regression, not an error.
    // get_grid_shape rounds the logical grid up to a whole number of clusters, blocks_per_problem
    // is computed from the PADDED value, and the scheduler hands those phantom tiles out as valid
    // work -- so the extra CTAs run the whole mainloop and the epilogue predicates their output
    // away. can_implement checks nothing about clusters against the grid, so nothing else catches
    // it. Cheap host-side compare against a GEMM that is microseconds at its cheapest.
    if constexpr (kClusterN > 1) {
      TORCH_CHECK(cutlass::ceil_div(m, kTileN) >= kClusterN,
                  "vtl w4a8 v2: cluster N=", kClusterN, " needs at least that many token tiles, "
                  "but m=", m, " with TileN=", kTileN, " gives ",
                  cutlass::ceil_div(m, kTileN),
                  " -- half the slice would compute discarded output. Use a 1x1x1 arm here.");
    }

    // safely cast group_size to int
    TORCH_CHECK(
        group_size > 0 && group_size <= std::numeric_limits<int>::max(),
        "group_size out of supported range for int: ", group_size);
    int const group_size_int = static_cast<int>(group_size);

    // Allocate output
    // VTL: at::cuda guard/stream/empty in place of the stable-ABI trio, and ElementD is
    // hardcoded bf16 (the epilogue emits bf16 and this op has no out_type argument).
    const c10::cuda::CUDAGuard device_guard(A.device());
    auto stream = c10::cuda::getCurrentCUDAStream(A.device().index());
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

    // VTL: hw_info is the 5th member of the Arguments aggregate; scheduler is the 6th. Upstream
    // brace-initializes only the first four, leaving sm_count = 0 and a device query inside
    // every initialize(). See hardware_info() for why that matters on a MIG slice.
    Args arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                   {n, m, k, 1},  // shape
                   mainloop_arguments,
                   epilogue_arguments,
                   hardware_info()};

    // VTL: the ONLY behavioural addition. CUTLASS's Stream-K scheduler defaults to
    // DecompositionMode::Heuristic, which falls back to plain data-parallel whenever the tile
    // grid already covers the device -- and at decode our grids are 16-128 CTAs, so the
    // heuristic would silently give us the arm we already have. Forcing StreamK is what makes
    // the *_sk arms mean anything.
    if constexpr (kStreamK) {
      using TS = typename GemmKernelShuffled::TileScheduler;
      if constexpr (SchedTune::kSplitK) {
        arguments.scheduler.decomposition_mode = TS::DecompositionMode::SplitK;
        arguments.scheduler.splits = SchedTune::kSplits;
      } else {
        arguments.scheduler.decomposition_mode = TS::DecompositionMode::StreamK;
      }
      if constexpr (SchedTune::kNondeterministic) {
        arguments.scheduler.reduction_mode = TS::ReductionMode::Nondeterministic;
      }
    }

    // Raster order and swizzle live on the BASE scheduler Arguments, so both the persistent and
    // the Stream-K schedulers carry them.
    //
    // Both are provable no-ops at decode and are only wired up for the prefill arms: CUTLASS
    // derives the swizzle from min(tiles_m, tiles_n) and the raster order only reorders a 2-D
    // walk, and at m <= 8 with any TileN >= 8 the token grid is a single tile -- so tiles_n == 1
    // collapses the swizzle to 0 and makes both raster directions enumerate identically.
    if constexpr (SchedTune::kRaster != 0) {
      using RasterOrderOptions =
          typename GemmKernelShuffled::TileScheduler::RasterOrderOptions;
      arguments.scheduler.raster_order = SchedTune::kRaster == 1
                                             ? RasterOrderOptions::AlongM
                                             : RasterOrderOptions::AlongN;
    }
    if constexpr (SchedTune::kMaxSwizzle > 1) {
      arguments.scheduler.max_swizzle_size = SchedTune::kMaxSwizzle;
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

// Nondeterministic Stream-K reduction. Same decomposition as the *_sk arms, minus the K-ordered
// turnstile between peers. Sweep-only -- see SchedNondet for why this must not become the ship.
using Kernel_128x16_1x1x1_sk_nd =
    W4A8GemmKernelV2<Shape<_128, _16>, Shape<_1, _1, _1>,
                     cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                     cutlass::epilogue::TmaWarpSpecializedCooperative,
                     cutlass::gemm::StreamKScheduler, SchedNondet>;

using Kernel_128x32_1x1x1_sk_nd =
    W4A8GemmKernelV2<Shape<_128, _32>, Shape<_1, _1, _1>,
                     cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                     cutlass::epilogue::TmaWarpSpecializedCooperative,
                     cutlass::gemm::StreamKScheduler, SchedNondet>;

// Fixed 4-way split-K. The competing hypothesis to Stream-K for w2's single deep wave: an even
// split costs one reduction and no per-unit bookkeeping, where Stream-K pays for a raggedness
// that a 16-CTA/16-SM problem does not actually have.
using Kernel_128x16_1x1x1_splitk4 =
    W4A8GemmKernelV2<Shape<_128, _16>, Shape<_1, _1, _1>,
                     cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                     cutlass::epilogue::TmaWarpSpecializedCooperative,
                     cutlass::gemm::StreamKScheduler, SchedSplitK4>;

using Kernel_128x32_1x1x1_splitk4 =
    W4A8GemmKernelV2<Shape<_128, _32>, Shape<_1, _1, _1>,
                     cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                     cutlass::epilogue::TmaWarpSpecializedCooperative,
                     cutlass::gemm::StreamKScheduler, SchedSplitK4>;

// Shallower mainloop pipeline. Auto carveout lands near 19 stages at 128x16; these test whether
// that depth is doing anything on the 16-k-iteration shapes (qkv, w13). Expected to be a null --
// the mainloop is DRAM-bound, not latency-bound -- but it is two `using` lines to find out.
using Kernel_128x16_1x1x1_s4 =
    W4A8GemmKernelV2<Shape<_128, _16>, Shape<_1, _1, _1>,
                     cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                     cutlass::epilogue::TmaWarpSpecializedCooperative,
                     cutlass::gemm::PersistentScheduler, SchedDefault, 4>;

using Kernel_128x16_1x1x1_s8 =
    W4A8GemmKernelV2<Shape<_128, _16>, Shape<_1, _1, _1>,
                     cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                     cutlass::epilogue::TmaWarpSpecializedCooperative,
                     cutlass::gemm::PersistentScheduler, SchedDefault, 8>;

// Cluster multicast. HONEST NOTE: after swap+transpose the multicast operand is the
// ACTIVATION, which at decode is 1-8 tokens -- there is very little to share. Swept anyway
// because the pair of CTAs also halves the number of TMA loads issued for it.
//
// This is ClusterM=2. Sharing WEIGHT bytes would need ClusterN=2 (CUTLASS multicasts operand A
// -- the weights, post-swap -- along the cluster's N dim), and the N dim tiles tokens: two CTAs
// share a weight tile only when they compute the same output channels for DIFFERENT token
// tiles. At m <= 8 there is exactly one token tile, so no cluster shape whatsoever shares weight
// bytes at decode. That is a property of the decode shape, not of this configuration -- every
// output channel's weights must cross DRAM once per forward pass regardless of tiling, which is
// the decode roofline. Weight multicast becomes real only at prefill; see the arms below.
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

// ---------------------------------------------------------------------------------------
// Prefill arms -- the only place cluster multicast and raster order are not no-ops
// ---------------------------------------------------------------------------------------
// Both knobs need at least two TOKEN tiles. TileN=128 gives that from m >= 129 onward:
// m=512 -> 4 token tiles, so ClusterN=2 pairs CTAs that want the SAME 128x128 int4 weight tile
// (8 KB per k-iteration) and TMA multicasts it once over the SM-to-SM interconnect instead of
// pulling it twice from HBM. This is the technique the decode arms structurally cannot use.
//
// Raster order matters here for the same reason and is arguably the larger effect. CUTLASS's
// Heuristic picks AlongM whenever tiles_n > tiles_m, which is the common prefill shape -- and
// AlongM walks OUTPUT CHANNELS, so consecutive CTAs share the activation tile and each pulls its
// own weight tile. Per 16-CTA wave per k-iteration that is 16 weight tiles + 1 activation tile
// (~130 KB) against AlongN's 1 + 16 (~40 KB). On a slice whose HBM bandwidth is the binding
// constraint, the stock heuristic optimizes reuse of the operand that is nearly free.
//
// The cluster guard in mm() is what keeps these honest: routed at m <= TileN they raise instead
// of silently burning half the slice.
using Kernel_128x128_1x2x1_pf =
    W4A8GemmKernelV2<Shape<_128, _128>, Shape<_1, _2, _1>,
                     cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                     cutlass::epilogue::TmaWarpSpecializedCooperative,
                     cutlass::gemm::PersistentScheduler, SchedRasterN>;

// Raster/swizzle without the cluster, so a win can be attributed to one or the other. Also the
// arm to use when m is between the thresholds but below 2*TileN, where the cluster cannot fire.
using Kernel_128x128_1x1x1_pf =
    W4A8GemmKernelV2<Shape<_128, _128>, Shape<_1, _1, _1>,
                     cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                     cutlass::epilogue::TmaWarpSpecializedCooperative,
                     cutlass::gemm::PersistentScheduler, SchedRasterNSwizzle4>;

// VTL: the param validation upstream left as a TODO. Every argument below is immediately cast
// to a raw device pointer and handed to a kernel that derives its extents from OTHER arguments,
// so a wrong dtype or a short buffer is an out-of-bounds DEVICE read: the context is poisoned
// and every later request in the same benchmark fails too, far from the call that broke it.
// These are host-side integer compares against a GEMM that is microseconds at its cheapest, and
// they live at the op entry (not in the template) so they cover the stock-forwarding branch too
// and are compiled once rather than per arm.
//
// The contract is the stock kernel's own (w4a8_mm_entry.cu:172-174 for the extents, :380-411
// for what the two packing ops emit):
//   A               [m, k]                       fp8 e4m3, row-major contiguous
//   B               [k/8, n]                     int32, 8 int4s per word, encoded + reordered
//   group_scales    [n * ceil(k/gs) * 8] flat    fp8 e4m3 -- pack_scale_fp8 expands each 8x
//   channel_scales  [n] (or a scalar)            fp32     -- epilogue ColOrScalarLoad
//   token_scales    [m] (or a scalar)            fp32     -- epilogue RowOrScalarLoad
//
// B's STRIDES are deliberately not checked: the kernel ignores them and reads through its own
// tile_to_shape(LayoutAtomQuant, (n,k,1)) layout, so the only thing that bounds the access is
// the element count, which is what we check. Requiring contiguity here would reject the
// column-major buffer the production path actually passes.
void check_args(at::Tensor const& A, at::Tensor const& B,
                at::Tensor const& group_scales, int64_t group_size,
                at::Tensor const& channel_scales,
                at::Tensor const& token_scales) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && group_scales.is_cuda() &&
                  channel_scales.is_cuda() && token_scales.is_cuda(),
              "vtl w4a8 v2: every argument must be a CUDA tensor");
  TORCH_CHECK(B.device() == A.device() &&
                  group_scales.device() == A.device() &&
                  channel_scales.device() == A.device() &&
                  token_scales.device() == A.device(),
              "vtl w4a8 v2: all arguments must be on one device, got A on ",
              A.device(), " and B on ", B.device());

  TORCH_CHECK(A.dim() == 2 && B.dim() == 2,
              "vtl w4a8 v2: A and B must be 2-D, got A.dim()=", A.dim(),
              " B.dim()=", B.dim());
  TORCH_CHECK(A.scalar_type() == at::kFloat8_e4m3fn,
              "vtl w4a8 v2: A must be float8_e4m3fn, got ", A.scalar_type());
  // stride_A is a PACKED stride built from (m, k, 1); a view with any other layout would be
  // read as if it were dense.
  TORCH_CHECK(A.is_contiguous(),
              "vtl w4a8 v2: A must be contiguous (the mainloop assumes a packed row-major "
              "stride); call .contiguous() first");
  TORCH_CHECK(B.scalar_type() == at::kInt,
              "vtl w4a8 v2: B must be int32 (packed int4), got ", B.scalar_type());

  int64_t const m = A.size(0);
  int64_t const k = A.size(1);
  int64_t const n = B.size(1);
  TORCH_CHECK(B.size(0) * PackFactor == k,
              "vtl w4a8 v2: B must be [k/", PackFactor, ", n] for A's k=", k,
              ", got B=[", B.size(0), ", ", n, "]");
  TORCH_CHECK(B.numel() * PackFactor == k * n,
              "vtl w4a8 v2: B holds ", B.numel() * PackFactor,
              " int4s, the GEMM reads k*n=", k * n);
  TORCH_CHECK(group_size > 0 && k % group_size == 0,
              "vtl w4a8 v2: k=", k, " is not a multiple of group_size=", group_size);

  TORCH_CHECK(group_scales.scalar_type() == at::kFloat8_e4m3fn,
              "vtl w4a8 v2: group_scales must be float8_e4m3fn (the output of "
              "cutlass_pack_scale_fp8), got ", group_scales.scalar_type());
  TORCH_CHECK(group_scales.is_contiguous(),
              "vtl w4a8 v2: group_scales must be contiguous");
  int64_t const want_scales = n * (k / group_size) * ScalePackSize;
  TORCH_CHECK(group_scales.numel() == want_scales,
              "vtl w4a8 v2: group_scales has ", group_scales.numel(),
              " elements, expected n*(k/group_size)*", ScalePackSize, " = ",
              want_scales, " -- was it run through cutlass_pack_scale_fp8?");

  TORCH_CHECK(channel_scales.scalar_type() == at::kFloat &&
                  token_scales.scalar_type() == at::kFloat,
              "vtl w4a8 v2: channel_scales and token_scales must be float32, got ",
              channel_scales.scalar_type(), " and ", token_scales.scalar_type());
  TORCH_CHECK(channel_scales.is_contiguous() && token_scales.is_contiguous(),
              "vtl w4a8 v2: channel_scales and token_scales must be contiguous");
  // numel, not shape: production passes channel_scales as [n, 1] and the epilogue's
  // ColOrScalarLoad/RowOrScalarLoad also accept a single broadcast scalar.
  TORCH_CHECK(channel_scales.numel() == n || channel_scales.numel() == 1,
              "vtl w4a8 v2: channel_scales must hold n=", n, " values (or 1), got ",
              channel_scales.numel());
  TORCH_CHECK(token_scales.numel() == m || token_scales.numel() == 1,
              "vtl w4a8 v2: token_scales must hold m=", m, " values (or 1), got ",
              token_scales.numel());
}

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
  } else if (schedule == "128x16_1x1x1_sk_nd") {
    return Kernel_128x16_1x1x1_sk_nd::mm(A, B, group_scales, group_size,
                                         channel_scales, token_scales);
  } else if (schedule == "128x32_1x1x1_sk_nd") {
    return Kernel_128x32_1x1x1_sk_nd::mm(A, B, group_scales, group_size,
                                         channel_scales, token_scales);
  } else if (schedule == "128x16_1x1x1_splitk4") {
    return Kernel_128x16_1x1x1_splitk4::mm(A, B, group_scales, group_size,
                                           channel_scales, token_scales);
  } else if (schedule == "128x32_1x1x1_splitk4") {
    return Kernel_128x32_1x1x1_splitk4::mm(A, B, group_scales, group_size,
                                           channel_scales, token_scales);
  } else if (schedule == "128x16_1x1x1_s4") {
    return Kernel_128x16_1x1x1_s4::mm(A, B, group_scales, group_size,
                                      channel_scales, token_scales);
  } else if (schedule == "128x16_1x1x1_s8") {
    return Kernel_128x16_1x1x1_s8::mm(A, B, group_scales, group_size,
                                      channel_scales, token_scales);
  } else if (schedule == "128x128_1x2x1_pf") {
    return Kernel_128x128_1x2x1_pf::mm(A, B, group_scales, group_size,
                                       channel_scales, token_scales);
  } else if (schedule == "128x128_1x1x1_pf") {
    return Kernel_128x128_1x1x1_pf::mm(A, B, group_scales, group_size,
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

// Pipeline depth of an arm, for the boot log. Only the auto-carveout arms are interesting (their
// stage count is chosen by CUTLASS and otherwise invisible), but every name resolves so the
// caller does not need to know which is which. -1 = unknown name, never throws: this is a
// diagnostic and must not be able to fail a boot that would otherwise serve.
int64_t stages_of(std::string const& schedule) {
  if (schedule == "128x16_1x1x1_sk") return Kernel_128x16_1x1x1_sk::kStages;
  if (schedule == "128x32_1x1x1_sk") return Kernel_128x32_1x1x1_sk::kStages;
  if (schedule == "128x16_1x1x1_sk_nd") return Kernel_128x16_1x1x1_sk_nd::kStages;
  if (schedule == "128x32_1x1x1_sk_nd") return Kernel_128x32_1x1x1_sk_nd::kStages;
  if (schedule == "128x16_1x1x1_splitk4") return Kernel_128x16_1x1x1_splitk4::kStages;
  if (schedule == "128x32_1x1x1_splitk4") return Kernel_128x32_1x1x1_splitk4::kStages;
  if (schedule == "128x16_1x1x1_s4") return Kernel_128x16_1x1x1_s4::kStages;
  if (schedule == "128x16_1x1x1_s8") return Kernel_128x16_1x1x1_s8::kStages;
  if (schedule == "128x128_1x2x1_pf") return Kernel_128x128_1x2x1_pf::kStages;
  if (schedule == "128x128_1x1x1_pf") return Kernel_128x128_1x1x1_pf::kStages;
  if (schedule == "128x16_2x1x1") return Kernel_128x16_2x1x1::kStages;
#if VTL_W4A8_ENABLE_N8
  if (schedule == "128x8_1x1x1") return Kernel_128x8_1x1x1::kStages;
#endif
#if VTL_W4A8_ENABLE_PP
  if (schedule == "64x16_1x1x1_pp") return Kernel_64x16_1x1x1_pp::kStages;
  if (schedule == "64x32_1x1x1_pp") return Kernel_64x32_1x1x1_pp::kStages;
#endif
  return -1;
}

// Smallest token count an arm can legally run: a ClusterN of C needs C token tiles, so
// m > (C-1)*TileN. 1 for every non-clustered arm, which is all of them but one.
//
// WHY THIS IS NOT JUST THE GUARD IN mm(). That guard raises, which is right for a test and
// wrong for a server: with the prefill band open, chunked prefill hands this op every chunk
// size, and a 33-128 token chunk routed to the ClusterN=2 arm would be a 500 in the middle of
// the benchmark. Below the floor the band falls through to the stock kernel instead -- the same
// kernel the band was carved out of, so the fallback is exactly the previous behaviour. The
// guard in mm() stays as the backstop for anything that dispatches an arm directly.
int64_t min_m_of(std::string const& schedule) {
  if (schedule == "128x128_1x2x1_pf") {
    return (Kernel_128x128_1x2x1_pf::kClusterN - 1) * Kernel_128x128_1x2x1_pf::kTileN + 1;
  }
  return 1;
}

// The M branch lives HERE, not in Python, on purpose: `VtlW4A8LinearMethod.apply` is traced by
// support_torch_compile with fullgraph=True, where a branch on the symbolic token count is
// either a graph break (= engine crash) or a recompile per batch bucket. Inside an opaque
// custom op it is neither -- the graph sees one node.
//
// VTL: three bands, not two. Prefill used to go straight to the stock kernel, which meant the
// two knobs that only work with more than one token tile -- cluster multicast of the weights,
// and raster order -- had no path to ever fire. `prefill_max` bounds the new band so a long
// prompt still lands on the stock heuristic rather than an arm tuned for m ~ 512.
//
//   m <= m_threshold                     -> decode arm  (`schedule`)
//   m_threshold < m <= prefill_max       -> prefill arm (`prefill_schedule`)
//   otherwise                            -> stock kernel
//
// An empty schedule string means "that band is off" and falls through to stock, so the default
// configuration (prefill_schedule = "", prefill_max = 0) is byte-for-byte the previous behaviour.
at::Tensor w4a8_mm_v2(at::Tensor const& a, at::Tensor const& b_q,
                      at::Tensor const& group_scales, int64_t group_size,
                      at::Tensor const& channel_scales,
                      at::Tensor const& token_scales,
                      std::string const& schedule, int64_t m_threshold,
                      std::string const& prefill_schedule,
                      int64_t prefill_max) {
  check_args(a, b_q, group_scales, group_size, channel_scales, token_scales);
  int64_t const m = a.size(0);

  if (m <= m_threshold && !schedule.empty()) {
    return mm_dispatch_v2(a, b_q, group_scales, group_size, channel_scales,
                          token_scales, schedule);
  }
  if (m > m_threshold && m <= prefill_max && !prefill_schedule.empty() &&
      m >= min_m_of(prefill_schedule)) {
    return mm_dispatch_v2(a, b_q, group_scales, group_size, channel_scales,
                          token_scales, prefill_schedule);
  }
  return stock_mm(a, b_q, group_scales, group_size, channel_scales,
                  token_scales);
}

}  // namespace vtl::w4a8_v2

// FRAGMENT, not TORCH_LIBRARY: `vtl._C` owns TORCH_LIBRARY(vllm_cuda) and only one .so may.
// quant_w4a8.py imports vtl._C before vtl._C_w4a8 to keep that ownership unambiguous.
TORCH_LIBRARY_FRAGMENT(vllm_cuda, m) {
  m.def(
      "w4a8_mm_v2(Tensor a, Tensor b_q, Tensor group_scales, int group_size, "
      "Tensor channel_scales, Tensor token_scales, str schedule, "
      "int m_threshold, str prefill_schedule, int prefill_max) -> Tensor");
  // Diagnostics, CompositeExplicitAutograd so they resolve without a device dispatch: they take
  // no tensors and are called once at load, from the boot log path.
  m.def("w4a8_sm_count() -> int");
  m.def("w4a8_stages(str schedule) -> int");
}

TORCH_LIBRARY_IMPL(vllm_cuda, CUDA, m) {
  m.impl("w4a8_mm_v2", TORCH_FN(vtl::w4a8_v2::w4a8_mm_v2));
}

TORCH_LIBRARY_IMPL(vllm_cuda, CompositeExplicitAutograd, m) {
  m.impl("w4a8_sm_count", TORCH_FN(vtl::w4a8_v2::sm_count));
  m.impl("w4a8_stages", TORCH_FN(vtl::w4a8_v2::stages_of));
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
