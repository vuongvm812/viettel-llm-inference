"""``vtl_w4a8`` -- int4 weights (group-128, symmetric) + fp8-e4m3 per-token activations.

WHY. The judge runs an H200 **MIG 1g.18gb slice**: 1/7 of the SMs and 1/8 of the memory
slices, i.e. ~600 GB/s instead of 4.8 TB/s. Decode is weight-bandwidth bound, and per decode
step LFM2.5 streams 1035.5 M body params + a bf16 lm_head:

    body    group scales   chan   lm_head   total     @600 GB/s
    bf16 2071   --          --     268      2339 MB     3.90 ms
    fp8  1036   --          1.6    268      1306 MB     2.18 ms   <- vtl_fp8 today
    int4  518   64.8        1.6    268       852 MB     1.42 ms

so ~35% less weight traffic than fp8, ~0.76 ms of GPU time per step.

The group-scale figure is 64.8 MB, NOT the 8.1 MB the raw count suggests: cutlass_pack_scale_fp8
expands every scale into an ``Array<ElementScale, ScalePackSize>`` with ScalePackSize = 8
(w4a8_mm_entry.cu:40, 387 allocates ``numel() * 8``) and the mainloop TMA-loads all eight copies
every step. That single 8x is a seventh of the entire int4 weight budget -- do not "optimize" it
away by re-deriving the number from the scale count.

BE HONEST ABOUT WHETHER THIS PAYS. TPOT ~= max(host, gpu) under async scheduling. The host term
is ~3.4 ms and MIG does not touch it (the 3-core budget is the same either way). Modelling the
slice's GPU term as bandwidth plus a non-scaling fixed launch cost, and adding the ~440 MB/step
of KV traffic W4A8 does not touch, gives roughly 3.1 ms fp8 -> 2.4 ms int4: BOTH under the host
term, i.e. a measured TPOT delta of ~0. And TTFT can only get worse -- at 8192 batched tokens the
kernel re-dequantizes each weight tile once per 256-token N-tile, SM work on an SM-starved slice
that pure-fp8 cutlass_scaled_mm never pays -- and prefill on 19 SMs is COMPUTE bound, which is
exactly where that lands. The base case for this whole patch is "no benefit, slightly worse
TTFT". It must be A/B'd against vtl_fp8 on the MIG slice before it ships: ``VTL_QUANT=vtl_fp8
make up`` then ``make bench``, plus ``bench/eval_quality.py`` for the accuracy half.

IF THAT A/B SHOWS A REAL TTFT REGRESSION, the fix is to hold BOTH weight formats and dispatch on
M in apply() -- fp8 for prefill, int4 for decode -- at ~1 GB extra resident, which the 18 GB
slice can afford. Deliberately NOT built: a shape-dependent Python branch inside a torch.compile
region guards on the symbolic batch dim and risks a recompile per bucket, which is a real cost
paid against a regression nobody has measured yet. Measure first.

HOW. vLLM v0.25.0 already ships the SM90 CUTLASS W4A8 kernel (`csrc/libtorch_stable/
quantization/cutlass_w4a8/w4a8_mm_entry.cu`, `MmaType=float_e4m3_t`, `QuantType=int4b_t`), but
only wires it to ``CompressedTensorsW4A8Fp8``, which loads a *pre-packed* checkpoint produced
offline by llm-compressor. The judge mounts a bf16 checkpoint at ``/model`` that we cannot
swap, and vLLM has no online int4 path (``layers/quantization/online/`` is fp8/int8/mxfp8
only). So we quantize + pack at load time, in ``process_weights_after_loading``, and call
``ops.cutlass_w4a8_mm`` directly.

The pack/encode SEQUENCE follows vLLM's kernel test
(``tests/kernels/quantization/test_cutlass_w4a8.py::cutlass_quantize_and_pack``), but the SCALE
handling follows the production ``CutlassW4A8LinearKernel`` (cutlass.py:76-103). The two differ
and the test is the wrong model to copy: it feeds the GEMM random per-channel scales, which is
fine for checking a kernel against a matching reference and would produce garbage in a server.
RTN (round-to-nearest) rather than GPTQ: no calibration data, no startup budget, ~40 lines. It
costs accuracy -- see the accuracy warning below.

ACCURACY IS UNMEASURED. Group-128 symmetric RTN with no calibration and no zero-points is the
weakest 4-bit recipe there is, and 1.2 B is the parameter-poor end where 4-bit stops being close
to free -- compounding with fp8 activations and an fp8 KV cache. The short-conv ``in_proj`` is
the riskiest layer in the model: its output feeds a persistent recurrent state, so error
accumulates along the sequence instead of washing out per token. There is no eval harness in
this repo (bench/ measures latency and throughput only). Run one before submitting; if quality
regresses, the ladder is in ``VTL_W4A8_IGNORE`` below.

WHAT DOES **NOT** CHANGE. Every vtl CUDA kernel. All five emit fp8-e4m3 + a per-token fp32
scale, which is exactly the activation format W4A8 wants -- the "A8" in W4A8 *is* fp8 e4m3.
``RMSNormQuantFusionPass`` also still fires: we quantize activations with the same
``QuantFP8(static=False, PER_TOKEN)`` that ``Fp8LinearMethod`` uses, so the graph still shows
the ``kFp8DynamicTokenSym`` op that ``compilation/passes/fusion/rms_quant_fusion.py`` matches,
and it gets hoisted into the preceding norm exactly as before.

ENGINE FAILURE, ALREADY PAID FOR ONCE. ``VtlW4A8LinearMethod.apply`` runs inside the region
``support_torch_compile`` traces with **fullgraph=True**. Anything Dynamo cannot put in the
graph -- a ``logging`` call, a module-global mutation, a print, a ``.item()``, a data-dependent
Python branch -- is a graph break, and a graph break under fullgraph is a raised exception at
COMPILE time, not a slow path. The first on-box run of this patch died exactly there: a
once-only ``log_fallback_summary()`` at the top of ``apply()`` killed the engine while compiling
``model.layers.0.feed_forward.w13``, the first quantized layer Dynamo reached. The summary now
fires from ``_install_load_summary()``, a ``BaseModelLoader.load_model`` wrapper. Everything
else in this file is load-time and may log freely; ``apply()`` may not. Note this is a
compile-time trap, so an off-box ``make check`` cannot catch it -- only a real boot can.

SAFETY. ``--quantization=vtl_w4a8`` is a *serve flag*: if this name fails to register, vLLM
aborts at startup with "Invalid quantization method" and we score zero. So registration is
trivial and everything fragile is guarded:

  * ``get_quant_method`` runs for EVERY Linear at load and must never raise.
  * Any layer the CUTLASS kernel cannot implement (K or N not divisible by 128, non-SM90)
    silently falls back to ``vtl_fp8`` -- see [[quant_fp8]].
  * A layer whose quantization throws at load falls back to bf16 rather than killing the load.

Set ``VTL_W4A8_IGNORE`` to a comma-separated substring list to keep layers out of int4; they
fall back to fp8 (not bf16 -- see get_quant_method). If the eval regresses, the ladder is
``VTL_W4A8_IGNORE="conv"`` first, since the recurrent short-conv path is the most error-sensitive
and only costs 168 of the 1036 MB; then ``"conv,attn"``, i.e. MLP-only int4, which still keeps
868 MB (84%) of the win.

``lm_head`` is NOT a LinearBase (it is a ParallelLMHead, i.e. a VocabParallelEmbedding), so it
does not come through the linear path at all. It is handled by [[lm_head_quant]], which this
file dispatches to from the ParallelLMHead branch of get_quant_method -- 268 MB/step, 31.5% of
the post-W4A8 decode weight budget, the largest single tensor left in bf16. ``VTL_W4A8_IGNORE``
still wins over ``VTL_LM_HEAD_QUANT``; that precedence is stated in the branch and nowhere else.

NOTHING IN THIS FILE BELONGS IN A CUDA KERNEL, and the question has been asked once already.
Everything except ``apply()`` runs at model load: quantize, pack, install. ``apply()`` itself is
``QuantFP8`` + ``ops.cutlass_w4a8_mm``, both already CUDA, and the fused short-conv path reaches
``cutlass_w4a8_mm`` by hand. ``pack_int4_rows`` looks like kernel bait and is not -- it runs once
per layer at startup and already executes on-device. A hand-written packer buys startup seconds,
not latency.
"""

from __future__ import annotations

import functools
import logging
import os

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vllm.vtl")

IGNORE_ENV = "VTL_W4A8_IGNORE"

# CUTLASS tile/cluster override for the W4A8 GEMM. UNSET = the kernel's own heuristic, which is
# what we ship; this exists to sweep that heuristic on the box without a rebuild.
#
# Worth having because the heuristic (w4a8_mm_entry.cu:341-372) keys ONLY on M/N/K and is blind
# to SM count -- it was tuned on a full 132-SM Hopper and the judge's MIG 1g.18gb slice has ~19,
# so its tile choice is an open question here, not a settled one.
#
# NAMING: "<TileM>x<TileN>_<ClusterM>x<ClusterN>x<ClusterK>", from
# `Kernel_128x16_1x1x1 = W4A8GemmKernel<Shape<_128,_16>, Shape<_1,_1,_1>>` (w4a8_mm_entry.cu:268).
# TileK is fixed at 128 for fp8 operands (:39). Mind the operand swap: the kernel passes the
# WEIGHT as CUTLASS operand A, so TileM is the output-channel (N of the linear) tile and TileN is
# the token (M of the linear) tile. A decode-shaped schedule is therefore a small SECOND number.
#
# THE TEN LEGAL VALUES (w4a8_mm_entry.cu:278-318). Anything else makes the kernel throw
# STD_TORCH_CHECK at the first forward, so _schedule() validates and ignores unknown strings
# rather than letting a typo take the server down mid-benchmark:
#
#     256x128_1x1x1   128x256_1x1x1   128x256_2x1x1
#     256x64_1x1x1    128x128_1x1x1
#     256x32_1x1x1    128x64_1x1x1
#     256x16_1x1x1    128x32_1x1x1    128x16_1x1x1
#
# The heuristic picks 128x16 / 128x32 for our decode shapes (M<=32) and 128x256_1x1x1 for an
# 8192-token prefill, so those three are the ones worth beating first.
SCHEDULE_ENV = "VTL_W4A8_SCHEDULE"

VALID_SCHEDULES = frozenset({
    "256x128_1x1x1", "256x64_1x1x1", "256x32_1x1x1", "256x16_1x1x1",
    "128x256_2x1x1", "128x256_1x1x1", "128x128_1x1x1", "128x64_1x1x1",
    "128x32_1x1x1", "128x16_1x1x1",
})
# The CUTLASS W4A8 kernel supports exactly one group size (see `can_implement` in
# vllm/model_executor/kernels/linear/mixed_precision/cutlass.py:55). Not a tunable.
GROUP_SIZE = 128
PACK_FACTOR = 8  # int4 values per int32

# Fallback COUNTERS, not one-shot latches. "We shipped fp8 believing we shipped int4" is this
# patch's named failure mode, and a single warning line for ~100 layers cannot tell total
# degradation from a one-layer blip. log_fallback_summary() prints the totals after load.
_fallback_count = 0        # get_quant_method raised -> delegated to fp8
_load_fallback_count = 0   # quantize/pack raised -> layer stayed bf16
_shape_fallback_count = 0  # kernel can_implement said no -> layer uses fp8
_w4a8_layer_count = 0      # layers that actually installed int4 weights
_summary_logged = False


def log_fallback_summary() -> None:
    """One line stating how much of the model is actually int4. Grepped by ``make verify``.

    Deliberately WARNs when nothing quantized: booting fp8 while believing you booted int4 is
    indistinguishable from success in every other signal we emit.

    Idempotent: called from the load_model wrapper, which runs once per process, but the guard
    costs nothing and a duplicated summary would read as two different models being loaded.
    """
    global _summary_logged
    if _summary_logged:
        return
    _summary_logged = True

    fell_back = _shape_fallback_count + _fallback_count + _load_fallback_count
    if _w4a8_layer_count == 0:
        log.warning(
            "vtl: w4a8 quantized 0 layers (fp8/bf16 fallbacks: shape=%d error=%d load=%d) "
            "-- this server is NOT running int4",
            _shape_fallback_count, _fallback_count, _load_fallback_count,
        )
    else:
        log.info(
            "vtl: w4a8 quantized %d layers (fallbacks: shape=%d error=%d load=%d, total=%d)",
            _w4a8_layer_count, _shape_fallback_count, _fallback_count,
            _load_fallback_count, fell_back,
        )


def parse_ignored_layers(raw: str | None) -> list[str]:
    """Comma-separated layer-name substrings to keep out of int4, e.g. ``lm_head,conv``."""
    if not raw:
        return []
    return [name.strip() for name in raw.split(",") if name.strip()]


def validate_schedule(raw: str | None) -> str | None:
    """Normalize ``VTL_W4A8_SCHEDULE``; ``None`` means "let the kernel choose".

    Unknown strings are dropped with a warning rather than passed through. The kernel's own
    reaction to a bad name is STD_TORCH_CHECK at the first forward -- i.e. a typo in an env var
    takes the server down after it has already reported healthy, in the middle of a benchmark.
    Falling back to the heuristic makes a typo cost you one confusing data point instead.
    """
    if not raw or not raw.strip():
        return None
    value = raw.strip()
    if value not in VALID_SCHEDULES:
        log.warning(
            "vtl: %s=%r is not one of %s; using the kernel heuristic instead",
            SCHEDULE_ENV, value, ", ".join(sorted(VALID_SCHEDULES)),
        )
        return None
    return value


@functools.cache
def schedule() -> str | None:
    """THE single read of ``VTL_W4A8_SCHEDULE`` in the whole process.

    Single-source on purpose. The short-conv patch also needs this value, but it lives in the
    gitignored vLLM fork and deliberately carries no ``vtl`` import, so an earlier version had
    both files calling ``os.environ.get`` independently -- two readers of one knob, which drift
    silently. Instead this value is stamped onto each layer as ``_vtl_w4a8_schedule`` during
    process_weights_after_loading, and short_conv reads the layer attribute. Same trick as
    ``_vtl_w4a8_done``. cache=1 read, so a sweep cannot half-apply.
    """
    return validate_schedule(os.environ.get(SCHEDULE_ENV))


def is_ignored(prefix: str, patterns: list[str]) -> bool:
    """SUBSTRING match. vLLM's own ``is_layer_skipped`` is an exact prefix match, so a bare
    ``lm_head`` would never match a nested prefix like ``model.lm_head``. Same semantics as
    [[quant_fp8]] so the two ignore lists behave identically."""
    return bool(prefix) and any(pat and pat in prefix for pat in patterns)


def pack_int4_rows(q, pack_factor: int = PACK_FACTOR):
    """Pack an int4-valued ``[K, N]`` tensor into ``[K/8, N]`` int32, on the GPU.

    Equivalent to ``quant_utils.pack_rows(q & 0xF, 4, K, N)``, which we deliberately do NOT
    call: it round-trips through ``.cpu().numpy().astype(uint32)``, and across the whole model
    that is ~4 GB of host memory inside an 8 GB cgroup, plus a device sync per layer. This is
    the same packing in three lines of torch, on-device.

    Nibble i of the output word holds row ``8*j + i`` -- matching ``pack_rows``' strided
    ``q_w[i::pack_factor]``, which selects rows ``i, 8+i, 16+i, ...`` in j order.

    Built in int64 and narrowed at the end: nibble 7 of a negative int4 sets bit 31, which
    would make the int32 accumulation overflow-wrap. The wrap happens to produce the right bit
    pattern, but relying on that is the kind of thing that breaks silently.
    """
    import torch

    size_k, size_n = q.shape
    assert size_k % pack_factor == 0, f"K={size_k} not divisible by {pack_factor}"
    shifts = (
        4 * torch.arange(pack_factor, device=q.device, dtype=torch.int64)
    ).view(1, pack_factor, 1)
    nibbles = (q.reshape(size_k // pack_factor, pack_factor, size_n).to(torch.int64) & 0xF)
    return (nibbles << shifts).sum(dim=1).to(torch.int32)


def _quantize_and_pack(weight):
    """bf16 ``[N, K]`` -> ``(packed_int4, group_scales_fp8, channel_scales_fp32)``.

    Mirrors ``cutlass_quantize_and_pack`` in vLLM's own kernel test. Note ``quantize_weights``
    groups along dim 0, so it wants the weight K-major (``[K, N]``), the transpose of vLLM's
    ``layer.weight``.
    """
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        GroupShape,
        convert_bf16_scales_to_fp8,
        quantize_weights,
    )
    from vllm.scalar_type import scalar_types

    # fp32 for the quantize math: the division by the group scale in bf16 loses ~3 bits, which
    # at 4-bit output is a visible extra rounding error for free.
    w_kn = weight.t().contiguous().float()  # [N, K] -> [K, N]

    # scalar_types.int4 (NOT uint4b8): signed, no bias, so w_q lands in [-8, 7] directly and we
    # skip the +8/-8 round trip that the compressed-tensors checkpoint path needs.
    _, w_q, w_s, _ = quantize_weights(
        w_kn, scalar_types.int4, group_size=GROUP_SIZE, zero_points=False
    )
    del w_kn

    # pack_int4_rows masks each value to 4 bits internally, so a negative int4 cannot
    # sign-extend over its neighbouring nibbles.
    w_q = pack_int4_rows(w_q)
    w_q = w_q.t().contiguous().t()  # column-major, as the kernel expects
    # Production does this before the same call (cutlass.py:80). unified_encode_int4b_device
    # (w4a8_utils.cu:69) launches with no explicit stream argument, so it is not ordered
    # against whatever stream produced w_q. Weight loading is normally on the default stream
    # and this is probably redundant -- but it is one sync, once per layer, at load time, and
    # the failure it prevents is silently garbled weights.
    torch.accelerator.synchronize()
    packed = ops.cutlass_encode_and_reorder_int4b(w_q)

    # w_s is [K/128, N]; convert_bf16_scales_to_fp8 quantizes per ROW, so it needs the
    # per-output-channel orientation [N, K/128]. It splits each scale into an fp8 group scale
    # (consumed in the GEMM mainloop) and an fp32 per-channel residual (applied in the epilogue).
    # bf16 (not the fp32 we quantized in) because that is the dtype the checkpoint path feeds
    # it, and the group scale is about to be squeezed into fp8 regardless.
    quant_fp8 = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)
    fp8_scales, chan_scales = convert_bf16_scales_to_fp8(
        quant_fp8, w_s.t().contiguous().to(torch.bfloat16)
    )
    group_scales = ops.cutlass_pack_scale_fp8(
        fp8_scales.t().contiguous().to(torch.float8_e4m3fn)
    )
    # chan_scales is fp32 [N, 1] and is passed through UNRESHAPED, exactly as the production
    # CutlassW4A8LinearKernel does (cutlass.py:96-98). vLLM's kernel test happens to pass a
    # 1-D [N] instead; match the path that actually serves models, not the test.
    return packed, group_scales, chan_scales


#: The three ops the W4A8 path needs. They are CONDITIONALLY COMPILED -- vLLM's CMakeLists
#: gates the cutlass_w4a8 sources on sm90a + CUDA>=12, and `_custom_ops.py` itself guards each
#: one with `hasattr(torch.ops._C, ...)` before registering its fake. A vLLM build without them
#: still passes CutlassW4A8LinearKernel.can_implement (it only checks arch/dtype/shape), so
#: without this list we would accept every layer and then fail in the weight transform -- which
#: lands the layer in BF16, i.e. 2x the decode traffic of the fp8 we started from. Checking here
#: instead means a kernel-less image degrades to fp8, which is the correct floor.
_REQUIRED_OPS = (
    "cutlass_w4a8_mm",
    "cutlass_encode_and_reorder_int4b",
    "cutlass_pack_scale_fp8",
)


def w4a8_ops_available() -> bool:
    import torch

    return all(hasattr(torch.ops._C, name) for name in _REQUIRED_OPS)


def _can_implement(input_size: int, output_size: int) -> bool:
    """Ask the real kernel, rather than re-hardcoding its constraints here."""
    import torch

    if not w4a8_ops_available():
        return False
    # MPLinearLayerConfig via the package root, which is the path CompressedTensorsW4A8Fp8
    # itself imports -- the deeper .mixed_precision.MPLinearKernel path is an implementation
    # detail that has already been renamed once in this tree.
    from vllm.model_executor.kernels.linear import MPLinearLayerConfig
    from vllm.model_executor.kernels.linear.mixed_precision.cutlass import (
        CutlassW4A8LinearKernel,
    )
    from vllm.scalar_type import scalar_types

    ok, _reason = CutlassW4A8LinearKernel.can_implement(
        MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(input_size, output_size),
            weight_type=scalar_types.int4,
            act_type=torch.float8_e4m3fn,
            group_size=GROUP_SIZE,
            zero_points=False,
            has_g_idx=False,
            out_type=torch.bfloat16,
        )
    )
    return ok


@functools.cache
def _linear_method_cls():
    """Return the ``VtlW4A8LinearMethod`` class. Imports vLLM lazily -- see module docstring.

    Cached on the FUNCTION, not on the config instance. Caching it as ``self._method_cls``
    would put a class object whose qualname is ``<locals>.VtlW4A8LinearMethod`` inside the
    config -- and pickle serializes classes by reference, so it could not be resolved on the
    other side of a ``spawn``. That is the exact hazard VtlW4A8Config is hoisted to module
    scope to avoid; re-introducing it through an attribute would undo that.
    """
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.linear import LinearMethodBase
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    from vllm.model_executor.parameter import ModelWeightParameter
    from vllm.model_executor.utils import set_weight_attrs

    class VtlW4A8LinearMethod(LinearMethodBase):
        """bf16 weights in, int4 weights out, quantized once at the end of model load."""

        def __init__(self) -> None:
            # Same activation quantizer Fp8LinearMethod uses, which is what keeps
            # RMSNormQuantFusionPass matching. See the module docstring.
            self.quant_fp8 = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)
            # schedule() is cached, so this is one dict lookup per layer at construction and
            # an attribute load per forward -- never an environ read on the hot path.
            self.schedule = schedule()

        def create_weights(
            self,
            layer,
            input_size_per_partition: int,
            output_partition_sizes: list[int],
            input_size: int,
            output_size: int,
            params_dtype,
            **extra_weight_attrs,
        ) -> None:
            # Allocate the ORDINARY bf16 weight so the stock weight_loader fills it from the
            # bf16 checkpoint untouched; we replace it in process_weights_after_loading. This is
            # why no compressed-tensors checkpoint is needed.
            weight_loader = extra_weight_attrs.pop("weight_loader")
            weight = ModelWeightParameter(
                data=torch.empty(
                    sum(output_partition_sizes),
                    input_size_per_partition,
                    dtype=params_dtype,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight", weight)
            # Carry through whatever else the caller attached (UnquantizedLinearMethod does the
            # same). Empty in practice once weight_loader is popped, but dropping it silently is
            # how a future extra_weight_attrs key goes missing.
            set_weight_attrs(weight, extra_weight_attrs)

        def process_weights_after_loading(self, layer) -> None:
            if getattr(layer, "_vtl_w4a8_done", False):
                return
            try:
                packed, group_scales, chan_scales = _quantize_and_pack(layer.weight.data)

                # Everything below is INSIDE the try on purpose. If installation half-completed
                # -- packed tensors present, _vtl_w4a8_done still False -- then apply() would
                # take the bf16 arm while _vtl_w4a8_out_proj in the short-conv patch would see
                # the packed attributes and call cutlass_w4a8_mm on a partly-built layer. That
                # is wrong numerics, not slow numerics. Install-or-nothing.
                #
                # register_parameter, not plain attributes: named_parameters() walkers (memory
                # accounting, weight-load sanity checks) skip bare tensors. Matches production
                # (cutlass.py:97-99).
                layer.register_parameter(
                    "weight_packed", torch.nn.Parameter(packed, requires_grad=False)
                )
                layer.register_parameter(
                    "weight_group_scale",
                    torch.nn.Parameter(group_scales, requires_grad=False),
                )
                layer.register_parameter(
                    "weight_chan_scale",
                    torch.nn.Parameter(chan_scales, requires_grad=False),
                )
                # Release the bf16 original (4x the packed size). Emptied rather than `del`ed so
                # the attribute still exists: plenty of vLLM and vtl code does a bare
                # `layer.weight` lookup.
                #
                # Safe for the TIED lm_head too (see [[lm_head_quant]]): this REBINDS
                # layer.weight, it does not free the storage. embed_tokens.weight is a separate
                # attribute on a separate module pointing at the same tensor, so the embedding
                # lookup keeps working. Reads like a use-after-free; is not one.
                layer.weight = torch.nn.Parameter(
                    torch.empty(0, dtype=layer.weight.dtype, device=layer.weight.device),
                    requires_grad=False,
                )
                # Stamped on the layer so the short-conv patch reads it from here rather than
                # doing its own os.environ.get -- see schedule().
                layer._vtl_w4a8_schedule = schedule()
                layer._vtl_w4a8_done = True

                global _w4a8_layer_count
                _w4a8_layer_count += 1
            except Exception as exc:
                # Leave the layer bf16 and let apply() fall through to a plain matmul. Losing a
                # layer to bf16 is survivable; failing model load is not.
                global _load_fallback_count
                _load_fallback_count += 1
                if _load_fallback_count == 1:
                    log.warning("vtl: w4a8 quantize failed (%s); that layer stays bf16", exc)
                layer._vtl_w4a8_done = False

        def apply(self, layer, x, bias=None):
            # NOTHING HERE MAY GRAPH-BREAK. This body is traced by support_torch_compile with
            # fullgraph=True, so a plain `log.info(...)` is not a slow path, it is an engine
            # crash -- see the ENGINE FAILURE note in the module docstring. The load summary
            # that used to live here now fires from _install_load_summary().
            if not getattr(layer, "_vtl_w4a8_done", False):
                return torch.nn.functional.linear(x, layer.weight, bias)  # bf16 fallback

            x_2d = x.reshape(-1, x.shape[-1])
            out_shape = x.shape[:-1] + (layer.weight_chan_scale.shape[0],)

            # NOTE: the fused short-conv path does NOT come through here -- it calls
            # ops.cutlass_w4a8_mm directly with an already-quantized activation (see
            # _vtl_out_proj_fp8 in vllm_patches/v0.26.0/short_conv.patch). For every other
            # layer the quant below is what RMSNormQuantFusionPass hoists into the norm.
            xq, x_scales = self.quant_fp8(x_2d)
            out = ops.cutlass_w4a8_mm(
                a=xq,
                b_q=layer.weight_packed,
                b_group_scales=layer.weight_group_scale,
                b_group_size=GROUP_SIZE,
                a_token_scales=x_scales,
                b_channel_scales=layer.weight_chan_scale,
                maybe_schedule=self.schedule,
            )
            if bias is not None:
                out.add_(bias)
            return out.reshape(out_shape)

    return VtlW4A8LinearMethod


try:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig as _QuantizationConfig,
    )
except Exception:  # pragma: no cover - vLLM absent (self-check import only)
    _QuantizationConfig = object


class VtlW4A8Config(_QuantizationConfig):
    """``vtl_w4a8`` quant config.

    Defined at MODULE scope (not nested in ``apply``) so it is picklable across ``spawn``: the
    multi-api-server / Rust-frontend engine-core launch pickles ``vllm_config``, which
    references this instance, and the spawned child unpickles at plain import time. A class
    nested in a function has qualname ``apply.<locals>.VtlW4A8Config`` that pickle cannot
    resolve. Same reasoning as [[quant_fp8]]'s VtlFp8Config.
    """

    def __init__(self, ignored_layers: list[str] | None = None) -> None:
        super().__init__()
        self.ignored_layers = (
            ignored_layers
            if ignored_layers is not None
            else parse_ignored_layers(os.environ.get(IGNORE_ENV))
        )
        # Only picklable state lives on the instance. The linear-method CLASS deliberately does
        # not (see _linear_method_cls); VtlFp8Config is module scope, so it does.
        self._fp8_fallback = None

    @classmethod
    def get_name(cls):
        return "vtl_w4a8"

    @classmethod
    def get_supported_act_dtypes(cls):
        import torch

        # bfloat16 only: the CUTLASS W4A8 epilogue emits bf16 (ElementD = bfloat16_t).
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # 75, matching Fp8Config, NOT 90 -- even though the CUTLASS kernel is sm90a-only.
        # vLLM compares this once at config resolution (config/vllm.py:634-641) and raises
        # outright if the GPU is below it, which would bypass every per-layer fallback in this
        # file and turn "degrade to fp8" into "refuse to boot". _can_implement already asks the
        # kernel about capability 90 per layer, which is where that decision belongs.
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        # Empty => weight_utils.get_quant_config() constructs us with no args, which is the
        # bf16-checkpoint path we care about.
        return []

    @classmethod
    def from_config(cls, config: dict) -> "VtlW4A8Config":
        return cls()

    def _fp8(self):
        """Lazily build a VtlFp8Config to delegate un-int4-able layers to."""
        if self._fp8_fallback is None:
            from vtl.patches.quant_fp8 import VtlFp8Config

            self._fp8_fallback = VtlFp8Config()
        return self._fp8_fallback

    def get_quant_method(self, layer, prefix: str):
        # Runs for EVERY Linear at load time -- it must NEVER raise, or the whole engine dies
        # at model load. Every vLLM import is lazy + guarded and any failure degrades to fp8.
        global _fallback_count, _shape_fallback_count
        try:
            from vllm.model_executor.layers.linear import LinearBase
            from vllm.model_executor.layers.vocab_parallel_embedding import (
                ParallelLMHead,
            )

            # BEFORE the LinearBase check: ParallelLMHead is a VocabParallelEmbedding, so it
            # would otherwise fall into the "not ours" arm below and stay bf16 -- which is what
            # it did until [[lm_head_quant]] landed. embed_tokens is a plain
            # VocabParallelEmbedding and is NOT a ParallelLMHead, so it still returns None here
            # and keeps its bf16 table (which the tied head needs anyway).
            if isinstance(layer, ParallelLMHead):
                # THE precedence rule between the two knobs, stated once. VTL_W4A8_IGNORE means
                # bf16 for the head, not fp8: unlike the linear path there is no fp8 fallback to
                # delegate to, and VTL_LM_HEAD_QUANT=fp8 is the explicit way to ask for that rung.
                if is_ignored(prefix, self.ignored_layers):
                    return None
                from vtl.patches.lm_head_quant import lm_head_method

                # prefix selects FR-Spec vocab pruning: only `draft_model.*` is narrowed.
                return lm_head_method(layer, prefix)

            if not isinstance(layer, LinearBase):
                # Not ours (embed_tokens, attention, MoE -- the lm_head branched off above).
                # None is the
                # correct "not ours" signal HERE and only here -- VocabParallelEmbedding maps
                # it to UnquantizedEmbeddingMethod (vocab_parallel_embedding.py:279-280),
                # whereas LinearBase raises on it. See the except arm below.
                return None

            if is_ignored(prefix, self.ignored_layers):
                # Delegate rather than returning UnquantizedLinearMethod: "keep this layer out
                # of int4" means fp8, not bf16. fp8 then applies its OWN ignore list, so a name
                # in both lists still lands in bf16. This is what makes the documented
                # escalation ladder (VTL_W4A8_IGNORE="lm_head,conv,attn" -> MLP-only int4)
                # actually degrade to fp8 instead of doubling those layers' weight traffic.
                return self._fp8().get_quant_method(layer, prefix)

            # K/N divisibility and arch are the kernel's call, not ours.
            #
            # input_size/output_size, NOT the *_per_partition variants. Column/RowParallelLinear
            # do set those before super().__init__() runs this (linear.py:440-441, 1590-1592),
            # but ReplicatedLinear never sets them at all -- it passes input_size straight to
            # create_weights. The full shape is also the TP-invariant thing to ask the kernel
            # about, and the two are equal at our TP=1 anyway.
            if not _can_implement(layer.input_size, layer.output_size):
                _shape_fallback_count += 1
                if _shape_fallback_count == 1:
                    # Logged loudly: "everything quietly became fp8" is this patch's main
                    # silent-failure mode, and a boot log is the only way to notice it. The
                    # total is reported by log_fallback_summary() once loading finishes.
                    log.warning(
                        "vtl: w4a8 kernel cannot implement %s (%dx%d); that layer uses fp8",
                        prefix, layer.input_size, layer.output_size,
                    )
                return self._fp8().get_quant_method(layer, prefix)

            return _linear_method_cls()()
        except Exception as exc:
            _fallback_count += 1
            if _fallback_count == 1:
                log.warning("vtl: w4a8 unavailable (%s); falling back to vtl_fp8", exc)
            try:
                return self._fp8().get_quant_method(layer, prefix)
            except Exception:
                # NOT None: LinearBase.__init__ raises ValueError("All linear layers should
                # support quant method.") on a falsy return (linear.py:277), so the one branch
                # written to never kill the engine would be the one that kills it. bf16 is the
                # honest floor -- slow, but it serves.
                from vllm.model_executor.layers.linear import UnquantizedLinearMethod

                return UnquantizedLinearMethod()


def device_summary() -> str:
    """Name / SM count / HBM of the GPU we are ACTUALLY on. Never raises.

    The entire case for this patch rests on one unverified claim: that the judge runs an H200
    **MIG 1g.18gb slice** (~19 of 132 SMs, ~600 GB/s). Nothing in this repo has ever checked it,
    and it swings the value of W4A8 by ~8x:

        full H200   4.8 TB/s   fp8 1306 MB/step = 0.27 ms -> int4 852 MB = 0.18 ms   (-0.09 ms)
        MIG 1g.18gb ~600 GB/s  fp8 1306 MB/step = 2.18 ms -> int4 852 MB = 1.42 ms   (-0.76 ms)

    Against a ~3.4 ms host term that neither GPU changes, the full-H200 saving is invisible while
    the TTFT and accuracy costs are identical -- i.e. on a full H200 this patch is a pure loss and
    the right answer is ``--quantization=vtl_fp8``. One log line settles which world we are in,
    and it is the container's view (a MIG slice reports its own name and SM count), which is the
    view that matters. Grepped by ``make verify``.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return "no CUDA device"
        p = torch.cuda.get_device_properties(0)
        return (
            f"{p.name}, {p.multi_processor_count} SMs, "
            f"{p.total_memory / 1024**3:.0f} GiB, sm_{p.major}{p.minor}"
        )
    except Exception as exc:
        return f"unknown ({exc})"


def _install_load_summary() -> None:
    """Emit ``log_fallback_summary()`` once after model load, from OUTSIDE the compiled graph.

    It used to be emitted lazily from the first ``apply()``. That looked free -- one global bool
    read, and cudagraph capture would elide it from the steady state -- and it took the engine
    down on the first on-box run: ``apply()`` is traced by ``support_torch_compile`` with
    fullgraph=True, where a ``logging`` call is a graph break and a graph break is an exception.
    The engine died compiling ``model.layers.0.feed_forward.w13``, the first quantized layer
    Dynamo reached, long before any of the fallback counters could have been interesting.

    ``BaseModelLoader.load_model`` is the seam: it is what calls the module-level
    ``process_weights_after_loading``, so by the time it returns every per-layer tally is final,
    and it runs exactly once per process. Patched at the CLASS, not at
    ``model_loader.utils.process_weights_after_loading``, because ``base_loader.py`` does
    ``from ...utils import process_weights_after_loading`` -- a by-value import that rebinding
    the module attribute would never reach.
    """
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader

    if already_patched(BaseModelLoader, "load_model"):
        return
    original = BaseModelLoader.load_model

    def load_model(self, *args, **kwargs):
        model = original(self, *args, **kwargs)
        # A broken summary must never cost us a loaded model.
        try:
            log_fallback_summary()
        except Exception:
            log.exception("vtl: w4a8 fallback summary failed")
        return model

    BaseModelLoader.load_model = mark_patched(load_model, original)


@register_patch("w4a8", default=True)
def apply() -> None:
    from vllm.model_executor.layers.quantization import register_quantization_config

    register_quantization_config("vtl_w4a8")(VtlW4A8Config)

    # Best-effort: without it we lose the one line that says how much of the model is really
    # int4, which is a reporting loss, not a serving one.
    try:
        _install_load_summary()
    except Exception:
        log.exception("vtl: could not install the w4a8 load summary hook")

    # Say plainly, once, whether this image can actually do W4A8. Without it the only symptom
    # of a kernel-less build is a per-layer shape warning that reads like a tuning problem.
    try:
        available = w4a8_ops_available()
    except Exception:
        available = False
    if not available:
        log.warning(
            "vtl: W4A8 CUDA ops absent from this vLLM build (needs sm90a + CUDA>=12); "
            "every layer will serve as vtl_fp8. Rebuild the base image to get int4."
        )

    from vtl.patches.lm_head_quant import mode as lm_head_mode

    # Stated separately from the config line: this is the premise, not a setting.
    log.info("vtl: w4a8 device = %s", device_summary())

    log.info(
        "vtl: registered quantization method 'vtl_w4a8' "
        "(group=%d, ops=%s, schedule=%s, lm_head=%s, ignored=%s)",
        GROUP_SIZE,
        "ok" if available else "MISSING",
        schedule() or "kernel heuristic",
        lm_head_mode(),
        parse_ignored_layers(os.environ.get(IGNORE_ENV)) or "none",
    )


def _self_check() -> None:
    assert parse_ignored_layers(None) == []
    assert parse_ignored_layers("") == []
    assert parse_ignored_layers("lm_head") == ["lm_head"]
    assert parse_ignored_layers(" lm_head , conv ,") == ["lm_head", "conv"]

    # Unset / blank / unknown all mean "kernel heuristic"; only the ten real names pass through.
    assert validate_schedule(None) is None
    assert validate_schedule("") is None
    assert validate_schedule("   ") is None
    assert validate_schedule("128x16_1x1x1") == "128x16_1x1x1"
    assert validate_schedule("  128x256_2x1x1  ") == "128x256_2x1x1"
    assert validate_schedule("128x16") is None          # truncated cluster suffix
    assert validate_schedule("64x16_1x1x1") is None     # not an instantiated tile
    assert validate_schedule("128X16_1X1X1") is None    # case matters to the C++ compare
    assert len(VALID_SCHEDULES) == 10, VALID_SCHEDULES

    # Substring, not exact-prefix -- the whole reason we don't use vLLM's is_layer_skipped.
    assert is_ignored("model.layers.0.lm_head", ["lm_head"]) is True
    assert is_ignored("model.layers.0.feed_forward.w1", ["lm_head"]) is False
    assert is_ignored("", ["lm_head"]) is False
    assert is_ignored("anything", []) is False

    # The one genuinely non-obvious step in pack_int4_rows, checked with no array library so it
    # runs on the dev Mac too: reshaping [K, N] -> [K/8, 8, N] and taking index i along the
    # middle axis must select the SAME rows, in the same order, as pack_rows' strided q_w[i::8].
    for size_k in (8, 32, 2048):
        for i in range(PACK_FACTOR):
            reshaped = [j * PACK_FACTOR + i for j in range(size_k // PACK_FACTOR)]
            strided = list(range(i, size_k, PACK_FACTOR))
            assert reshaped == strided, (size_k, i, reshaped[:4], strided[:4])

    # THE REGRESSION THAT KILLED THE ENGINE ON THE BOX. VtlW4A8LinearMethod.apply is traced with
    # fullgraph=True, so a logging call or a global mutation inside it is a compile-time
    # exception, not a slow path -- and no off-box check can see that, because the class is only
    # built when vLLM is importable and the failure only happens under Dynamo. Reading our own
    # source is crude, and it is the only thing here that would have caught it.
    import pathlib
    import re

    src = pathlib.Path(__file__).read_text()
    body = re.search(
        r"\n( +)def apply\(self, layer, x, bias=None\):\n(.*?)(?=\n\1def |\n\1return )",
        src, re.S,
    )
    assert body, "could not locate VtlW4A8LinearMethod.apply -- update this guard"
    code = "\n".join(
        line for line in body.group(2).splitlines() if not line.lstrip().startswith("#")
    )
    for banned in ("log.", "logger.", "print(", "global "):
        assert banned not in code, (
            f"{banned!r} in the compiled apply() -- fullgraph=True makes that a graph break, "
            "i.e. an engine crash at compile time. Move it to load time."
        )

    try:
        import torch
    except ImportError:
        print("quant_w4a8 self-check ok (torch absent; tensor packing check skipped)")
        return

    # pack_int4_rows must agree with quant_utils.pack_rows, which is the layout the CUTLASS
    # kernel is tested against. Reimplement pack_rows' semantics here (numpy-free) rather than
    # importing vLLM, so this runs on a bare dev box.
    torch.manual_seed(0)
    size_k, size_n = 32, 5
    q = torch.randint(-8, 8, (size_k, size_n), dtype=torch.int32)
    packed = pack_int4_rows(q)
    assert packed.shape == (size_k // PACK_FACTOR, size_n), packed.shape
    assert packed.dtype == torch.int32

    ref = torch.zeros(size_k // PACK_FACTOR, size_n, dtype=torch.int64)
    for i in range(PACK_FACTOR):  # pack_rows: nibble i <- rows i, 8+i, 16+i, ...
        ref |= (q[i::PACK_FACTOR].to(torch.int64) & 0xF) << (4 * i)
    assert torch.equal(packed, ref.to(torch.int32)), "pack order diverges from pack_rows"

    # Round-trip: every nibble must decode back to the signed int4 we put in.
    for i in range(PACK_FACTOR):
        nib = (packed >> (4 * i)) & 0xF
        signed = torch.where(nib >= 8, nib - 16, nib)
        assert torch.equal(signed, q[i::PACK_FACTOR]), f"nibble {i} round-trip failed"

    print("quant_w4a8 self-check ok")


if __name__ == "__main__":
    _self_check()
