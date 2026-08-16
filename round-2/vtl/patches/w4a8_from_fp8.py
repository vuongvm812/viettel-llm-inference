"""``w4a8_from_fp8`` -- requantize an FP8 block-quant checkpoint's dense layers to int4.

WHY. Round-2's target checkpoint (Qwen3.5-122B-A10B-FP8) is already fp8 block-quant
(``weight_block_size=[128,128]``, dynamic group-128 activations). The existing [[quant_w4a8]]
method quantizes int4 FROM BF16 at load -- it registers its own ``vtl_w4a8`` quantization
method, allocates a bf16 weight and RTN-quantizes it in ``process_weights_after_loading``.
Serving an fp8-serialized checkpoint goes through vLLM's stock ``Fp8Config`` instead, so none
of that fires: the dense LinearBase layers (o_proj, in_proj_qkvz/ba, out_proj, the shared
expert's gate_up/down) stay fp8, at 2x the decode weight traffic int4 would need.

HOW. This patch wraps ``Fp8Config.get_quant_method`` (the class the checkpoint's own quant
config resolves to). When the stock config hands back a block-quant ``Fp8LinearMethod`` for a
LinearBase, we substitute a method that:

  * ``create_weights``          -> delegates to the STOCK method, so the fp8 checkpoint loads
                                   through the untouched weight_loader path (fp8 ``weight``
                                   [N, K] + fp32 ``weight_scale_inv`` [ceil(N/128), ceil(K/128)]).
  * ``process_weights_after_loading``
                                -> dequantizes each fp8 block with its scale to fp32, narrows
                                   to bf16, and hands the result to [[quant_w4a8]]'s OWN
                                   ``process_weights_after_loading`` -- the exact RTN int4
                                   group-128 quantize + CUTLASS pack + install sequence the
                                   bf16 path uses (``_quantize_and_pack``, ``pack_int4_rows``,
                                   the v2 schedule stamping). Nothing about the packed layout
                                   is restated here; it is inherited.
  * ``apply``                   -> [[quant_w4a8]]'s apply verbatim (QuantFP8 activation quant +
                                   ``ops.cutlass_w4a8_mm`` / ``vllm_cuda::w4a8_mm_v2``), i.e.
                                   the existing CUTLASS sm_90a GEMM path serves the layer.

WHAT IS DELIBERATELY OUT OF SCOPE. FusedMoE expert weights do not come through
``get_quant_method``-for-LinearBase at all (they get ``Fp8MoEMethod``); they are handled by
[[moe_decode_gemv]]. ``mlp.gate`` and ``shared_expert_gate`` are built with
``quant_config=None`` and never reach Fp8Config -- they stay bf16. Layers the checkpoint
itself excludes from fp8 (lm_head, embeddings, conv1d, in_proj_a/b, gates) come back from the
stock config as unquantized methods and are passed through untouched.

ACCURACY. This is a DOUBLE quantization: bf16 -> fp8 [128,128] blocks (offline, in the
checkpoint) -> int4 group-128 RTN (here). The fp8 step contributes a ~2^-3 relative error per
weight; the int4 step contributes the same +-gs/2 absolute error it would from bf16. Since the
int4 step size is ~2^-3 of the group absmax, the fp8 error is of the same order as the int4
error and roughly doubles it in the worst case rather than compounding multiplicatively.
Whether that survives end-to-end is empirical and is gated by ``bench/eval_quality.py`` on
the GPU host before this ships -- see the validation steps in the test file.

FAIL-SOFT LADDER, in order of preference:
  * anything unexpected BEFORE the dequant (missing/odd-shaped ``weight_scale_inv``, non-fp32
    scales, non-[128,128] blocks, kernel can_implement says no) -> the layer stays on STOCK
    fp8, logged once. This is the "leave the layer on stock fp8" rung.
  * the int4 quantize/pack itself raising AFTER the dequant -> [[quant_w4a8]]'s own fallback
    leaves the layer as the dequantized bf16 (correct, just 2x fp8's traffic), logged once.
  * ``VTL_W4A8_FROM_FP8_IGNORE`` -- comma-separated substring list, same semantics as
    ``VTL_W4A8_IGNORE`` -- keeps named layers on stock fp8 (the quality-regression ladder).
  * this module's ``get_quant_method`` wrapper never raises; any failure returns the stock
    method unchanged.

KNOBS.
    VTL_ENABLE_W4A8_FROM_FP8=1   registry gate; installs the Fp8Config wrapper (default off)
    VTL_W4A8_FROM_FP8            runtime arm; default "1" once the patch is enabled, set 0 to
                                 keep the wrapper installed but requantize nothing (A/B)
    VTL_W4A8_FROM_FP8_IGNORE     substring list of layers to keep on stock fp8
plus everything [[quant_w4a8]] reads (VTL_W4A8_SCHEDULE, VTL_W4A8_SCHEDULE_V2, ...), which
applies unchanged because the packed tensors and apply() are quant_w4a8's own.

The COMPILE-TIME rule from quant_w4a8 carries over: the inherited ``apply`` runs inside a
fullgraph=True torch.compile region, so nothing that graph-breaks (logging, prints, global
mutation) may be added to it. Our override only branches on a load-time constant attribute.
"""

from __future__ import annotations

import functools
import logging
import os

from vtl.patches import quant_w4a8
from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vllm.vtl.w4a8_from_fp8")

RUNTIME_ENV = "VTL_W4A8_FROM_FP8"
IGNORE_ENV = "VTL_W4A8_FROM_FP8_IGNORE"

# The only block geometry this patch understands. Anything else is a different checkpoint
# recipe and stays on stock fp8 (fail-soft, logged once).
BLOCK_N = 128
BLOCK_K = 128

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Counters, same doctrine as quant_w4a8: "we shipped fp8 believing we shipped int4" is the
# named silent failure, and only totals printed after load can distinguish a one-layer blip
# from the whole model quietly staying fp8.
_requant_count = 0          # layers successfully requantized fp8 -> int4
_stock_fallback_count = 0   # validation said no -> layer stayed stock fp8
_wrap_error_count = 0       # get_quant_method wrapper raised -> stock method returned
_summary_logged = False


def runtime_enabled(raw: str | None = None) -> bool:
    """``VTL_W4A8_FROM_FP8``: default ON once the registry gate enabled the patch.

    Two knobs on purpose: the registry gate decides whether the wrapper is installed at all
    (a structural choice), this one lets a sweep disarm the requant without changing the
    patch set (a runtime choice). Unset means armed -- enabling the patch already expressed
    the intent.
    """
    if raw is None:
        raw = os.environ.get(RUNTIME_ENV)
    if raw is None or not raw.strip():
        return True
    return raw.strip().lower() in _TRUTHY


@functools.cache
def ignored_layers() -> list[str]:
    """Substring list, same semantics (and parser) as ``VTL_W4A8_IGNORE``."""
    return quant_w4a8.parse_ignored_layers(os.environ.get(IGNORE_ENV))


def scale_grid(n: int, k: int, block_n: int = BLOCK_N, block_k: int = BLOCK_K) -> tuple[int, int]:
    """Expected ``weight_scale_inv`` shape for an fp8 [N, K] weight with [128,128] blocks."""
    return (-(-n // block_n), -(-k // block_k))


def validate_block_layout(
    weight_shape: tuple[int, ...], scale_shape: tuple[int, ...]
) -> str | None:
    """None if (weight, scale) look like a [128,128] fp8-block pair; else the reason.

    Pure shape math so the self-check can exercise every rejection without torch. The dtype
    checks (fp8 weight, fp32 scale -- an e8m0 scale checkpoint must NOT be dequantized as if
    its scales were fp32) live at the call site where the tensors exist.
    """
    if len(weight_shape) != 2:
        return f"weight is {len(weight_shape)}-D, expected 2-D [N, K]"
    if len(scale_shape) != 2:
        return f"weight_scale_inv is {len(scale_shape)}-D, expected 2-D"
    expected = scale_grid(*weight_shape)
    if tuple(scale_shape) != expected:
        return (
            f"weight_scale_inv shape {tuple(scale_shape)} does not match the "
            f"[{BLOCK_N},{BLOCK_K}]-block grid {expected} of weight {tuple(weight_shape)}"
        )
    return None


def dequant_fp8_block(weight, scale_inv):
    """fp8 ``[N, K]`` x fp32 ``[ceil(N/128), ceil(K/128)]`` -> bf16 ``[N, K]``.

    fp32 for the multiply (the fp8 value is exact in fp32 and the scale is fp32 already);
    bf16 only at the end, because that is the dtype quant_w4a8's ``_quantize_and_pack``
    normalizes from -- it immediately re-widens to fp32 for the RTN math, so the only
    narrowing that matters is this single one. repeat_interleave + crop handles the ragged
    last block of a non-multiple dimension, matching how vLLM's own block-dequant utilities
    broadcast the scale grid.
    """
    import torch

    n, k = weight.shape
    s = scale_inv.to(torch.float32)
    s = s.repeat_interleave(BLOCK_N, dim=0)[:n, :]
    s = s.repeat_interleave(BLOCK_K, dim=1)[:, :k]
    return (weight.to(torch.float32) * s).to(torch.bfloat16)


def _note_stock_fallback(prefix: str, reason) -> None:
    global _stock_fallback_count
    _stock_fallback_count += 1
    if _stock_fallback_count == 1:
        log.warning(
            "vtl: w4a8_from_fp8 cannot requantize %s (%s); that layer stays on stock fp8",
            prefix, reason,
        )


@functools.cache
def _method_cls():
    """The from-fp8 linear method class. Built lazily (imports vLLM) and cached on the
    FUNCTION for the same pickle-across-spawn reason as quant_w4a8's ``_linear_method_cls``:
    only the stock ``Fp8Config`` instance ends up inside the pickled vllm_config, never this
    class, so a ``<locals>`` qualname is survivable -- but keeping one instance of the class
    object per process is still the cheap, predictable shape."""
    import torch

    parent = quant_w4a8._linear_method_cls()

    class VtlW4A8FromFp8LinearMethod(parent):
        """fp8-block checkpoint weights in, [[quant_w4a8]]'s packed int4 out."""

        def __init__(self, stock_method) -> None:
            super().__init__()
            # The stock block-quant Fp8LinearMethod this layer would have used. It handles
            # weight CREATION always (so checkpoint loading is byte-for-byte stock) and
            # processing + apply whenever requantization is declined or fails validation.
            self._stock = stock_method

        def create_weights(self, layer, *args, **kwargs) -> None:
            self._stock.create_weights(layer, *args, **kwargs)

        def process_weights_after_loading(self, layer) -> None:
            if getattr(layer, "_vtl_w4a8_done", False) or getattr(
                layer, "_vtl_from_fp8_stock", False
            ):
                return
            prefix = getattr(layer, "prefix", "") or type(layer).__name__
            try:
                weight = layer.weight
                scale = getattr(layer, "weight_scale_inv", None)
                if scale is None:
                    raise ValueError("layer has no weight_scale_inv")
                if weight.dtype != torch.float8_e4m3fn:
                    raise ValueError(f"weight dtype is {weight.dtype}, not float8_e4m3fn")
                if scale.dtype != torch.float32:
                    # e8m0-scale checkpoints exist; treating their bytes as fp32 would be
                    # silent garbage, so anything non-fp32 goes back to the stock path.
                    raise ValueError(f"weight_scale_inv dtype is {scale.dtype}, not float32")
                reason = validate_block_layout(tuple(weight.shape), tuple(scale.shape))
                if reason is not None:
                    raise ValueError(reason)
                w_bf16 = dequant_fp8_block(weight.data, scale.data)
            except Exception as exc:
                # Stock fp8 is the floor here, and it is a GOOD floor -- the checkpoint was
                # built for it. Restore the untouched stock pipeline for this layer.
                _note_stock_fallback(prefix, exc)
                layer._vtl_from_fp8_stock = True
                self._stock.process_weights_after_loading(layer)
                return

            # Hand the parent a bf16 layer, exactly the shape its bf16-checkpoint path sees.
            # Rebinding (not mutating) so the fp8 storage is released once the parent frees
            # the bf16 weight after packing; the empty rebind of weight_scale_inv keeps the
            # attribute present for any stray reader without pinning ~N*K/16384 floats.
            layer.weight = torch.nn.Parameter(w_bf16, requires_grad=False)
            layer.weight_scale_inv = torch.nn.Parameter(
                torch.empty(0, dtype=torch.float32, device=w_bf16.device),
                requires_grad=False,
            )
            # The parent NEVER raises: on quantize/pack failure it leaves the layer bf16
            # (i.e. the dequantized weights -- correct, just unquantized) and counts it in
            # quant_w4a8._load_fallback_count.
            super().process_weights_after_loading(layer)
            if getattr(layer, "_vtl_w4a8_done", False):
                global _requant_count
                _requant_count += 1

        def apply(self, layer, x, bias=None):
            # Traced with fullgraph=True -- no logging, no globals; the branch is on a
            # per-layer constant stamped at load, so Dynamo specializes it away.
            if getattr(layer, "_vtl_from_fp8_stock", False):
                return self._stock.apply(layer, x, bias)
            return super().apply(layer, x, bias)

    return VtlW4A8FromFp8LinearMethod


def _maybe_wrap(config, layer, prefix: str, method):
    """Return the from-fp8 method when this (config, layer, method) triple qualifies, else
    the stock method unchanged. May raise -- the caller catches and falls back."""
    if not runtime_enabled():
        return method
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

    # Only the plain block-quant linear path. MoE methods, KV-cache methods, unquantized
    # methods (the checkpoint's own exclusion list: lm_head, embeddings, conv1d, ...), and
    # anything a future vLLM adds all pass through untouched.
    if not isinstance(method, Fp8LinearMethod) or not isinstance(layer, LinearBase):
        return method
    if not getattr(method, "block_quant", False):
        return method
    block = list(getattr(method, "weight_block_size", None) or ())
    if block != [BLOCK_N, BLOCK_K]:
        _note_stock_fallback(prefix, f"weight_block_size {block}")
        return method
    if quant_w4a8.is_ignored(prefix, ignored_layers()):
        return method
    if not quant_w4a8.w4a8_ops_available():
        _note_stock_fallback(prefix, "cutlass w4a8 ops absent from this build")
        return method
    # Full (TP-invariant) shape, same reasoning as quant_w4a8.get_quant_method.
    if not quant_w4a8._can_implement(layer.input_size, layer.output_size):
        _note_stock_fallback(
            prefix, f"kernel cannot implement {layer.input_size}x{layer.output_size}"
        )
        return method
    return _method_cls()(method)


def log_summary() -> None:
    """One line stating how much of the fp8 checkpoint actually became int4. Idempotent."""
    global _summary_logged
    if _summary_logged:
        return
    _summary_logged = True
    if _requant_count == 0:
        log.warning(
            "vtl: w4a8_from_fp8 requantized 0 layers (stock-fp8 fallbacks=%d, wrap "
            "errors=%d) -- the dense layers are NOT running int4",
            _stock_fallback_count, _wrap_error_count,
        )
    else:
        log.info(
            "vtl: w4a8_from_fp8 requantized %d fp8-block layers to int4 "
            "(stock-fp8 fallbacks=%d, wrap errors=%d)",
            _requant_count, _stock_fallback_count, _wrap_error_count,
        )


def _install_load_summary() -> None:
    """Same seam as quant_w4a8's summary: ``BaseModelLoader.load_model``, once, after every
    per-layer tally is final. Named-patch marked because THREE patches now legitimately wrap
    this attribute (w4a8, l2_persist, this) -- see registry.already_patched."""
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader

    if already_patched(BaseModelLoader, "load_model", patch="w4a8_from_fp8"):
        return
    original = BaseModelLoader.load_model

    def load_model(self, *args, **kwargs):
        model = original(self, *args, **kwargs)
        try:
            log_summary()
        except Exception:
            log.exception("vtl: w4a8_from_fp8 summary failed")
        return model

    BaseModelLoader.load_model = mark_patched(load_model, original, patch="w4a8_from_fp8")


@register_patch("w4a8_from_fp8", default=False)
def apply() -> None:
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    if already_patched(Fp8Config, "get_quant_method", patch="w4a8_from_fp8"):
        return
    original = Fp8Config.get_quant_method

    def get_quant_method(self, layer, prefix):
        # Runs for EVERY layer at load; a raise here kills the engine, so any failure hands
        # back exactly what stock would have used.
        method = original(self, layer, prefix)
        try:
            return _maybe_wrap(self, layer, prefix, method)
        except Exception as exc:
            global _wrap_error_count
            _wrap_error_count += 1
            if _wrap_error_count == 1:
                log.warning(
                    "vtl: w4a8_from_fp8 wrapper failed for %s (%s); using stock fp8",
                    prefix, exc,
                )
            return method

    Fp8Config.get_quant_method = mark_patched(
        get_quant_method, original, patch="w4a8_from_fp8"
    )

    try:
        _install_load_summary()
    except Exception:
        log.exception("vtl: could not install the w4a8_from_fp8 load summary hook")

    log.info(
        "vtl: w4a8_from_fp8 armed=%s (block=[%d,%d], ignored=%s) -- fp8-block LinearBase "
        "layers will be requantized to int4 at load",
        runtime_enabled(), BLOCK_N, BLOCK_K, ignored_layers() or "none",
    )


def _self_check() -> None:
    # Runtime gate: default ON (the registry gate is the master switch), explicit off works.
    assert runtime_enabled(None) is True
    assert runtime_enabled("") is True
    assert runtime_enabled("1") is True
    assert runtime_enabled("yes") is True
    assert runtime_enabled("0") is False
    assert runtime_enabled("off") is False

    # Scale-grid math, including the ragged-edge (non-multiple) cases.
    assert scale_grid(3072, 3072) == (24, 24)
    assert scale_grid(2048, 3072) == (16, 24)
    assert scale_grid(129, 130) == (2, 2)
    assert scale_grid(1, 1) == (1, 1)

    # Layout validation: exact grid passes, everything else names its reason.
    assert validate_block_layout((3072, 1024), (24, 8)) is None
    assert validate_block_layout((129, 256), (2, 2)) is None
    assert "does not match" in validate_block_layout((3072, 1024), (24, 9))
    assert "does not match" in validate_block_layout((3072, 1024), (8, 24))  # transposed
    assert "2-D" in validate_block_layout((3072,), (24, 8))
    assert "2-D" in validate_block_layout((3072, 1024), (24,))

    # Ignore list shares quant_w4a8's parser and substring semantics.
    assert quant_w4a8.parse_ignored_layers(" conv , attn ") == ["conv", "attn"]
    assert quant_w4a8.is_ignored("model.layers.0.conv1d", ["conv"]) is True

    # The double-quantization error bound this patch's accuracy story rests on, checked in
    # pure python (no numpy -- `make check` runs on a bare host). Simulate e4m3 rounding
    # (3 mantissa bits) + a [1, BLOCK]-ish block scale, then RTN int4 group-128 from BOTH
    # sources. The from-fp8 requant must stay within the direct-from-fp32 error plus the
    # fp8 representation error (triangle inequality on the same values).
    import math
    import random

    def to_e4m3(v: float, scale: float) -> float:
        """Round v/scale to the e4m3 grid (3 mantissa bits, clamp 448), return scale * fp8."""
        x = v / scale
        x = max(-448.0, min(448.0, x))
        if x == 0.0:
            return 0.0
        m, e = math.frexp(abs(x))          # abs(x) = m * 2^e, m in [0.5, 1)
        step = 2.0 ** (e - 4)              # 3 mantissa bits => 8 steps per binade
        snapped = round(abs(x) / step) * step
        return math.copysign(snapped * scale, x)

    def rtn_int4(vals: list[float]) -> list[float]:
        """Group-128 symmetric RTN int4 (absmax/7, clamp [-8, 7]), dequantized."""
        out = []
        for g0 in range(0, len(vals), 128):
            grp = vals[g0 : g0 + 128]
            amax = max(abs(v) for v in grp) or 1e-8
            s = amax / 7.0
            out.extend(max(-8, min(7, round(v / s))) * s for v in grp)
        return out

    rng = random.Random(0)
    w = [rng.gauss(0.0, 0.05) for _ in range(256)]      # two int4 groups
    bscale = max(abs(v) for v in w) / 448.0             # one fp8 block scale
    w_fp8 = [to_e4m3(v, bscale) for v in w]
    fp8_err = max(abs(a - b) for a, b in zip(w_fp8, w))

    deq_direct = rtn_int4(w)
    deq_from_fp8 = rtn_int4(w_fp8)
    direct_err = max(abs(a - b) for a, b in zip(deq_direct, w))
    from_fp8_err = max(abs(a - b) for a, b in zip(deq_from_fp8, w))
    # |requant(fp8(w)) - w| <= |requant(fp8(w)) - fp8(w)| + |fp8(w) - w|, and the first term
    # is bounded by (a hair over) the direct RTN error because the group absmax moved by at
    # most fp8_err. 1.10 covers that absmax drift; the assert is the accuracy story in code.
    assert from_fp8_err <= 1.10 * direct_err + fp8_err, (from_fp8_err, direct_err, fp8_err)
    # And fp8 must not have been a no-op in the synthetic data, or the check proves nothing.
    assert fp8_err > 0.0

    try:
        import torch
    except ImportError:
        print("w4a8_from_fp8 self-check ok (torch absent; dequant tensor check skipped)")
        return

    # dequant_fp8_block must agree with an explicit per-block loop, ragged edge included.
    torch.manual_seed(0)
    n, k = 200, 300  # deliberately NOT multiples of 128: 2x3 scale grid, ragged edges
    w32 = torch.randn(n, k) * 0.1
    scale = torch.rand(scale_grid(n, k)) * 0.01 + 1e-3
    wq = torch.empty(n, k)
    for bi in range(scale.shape[0]):
        for bj in range(scale.shape[1]):
            rs, re = bi * BLOCK_N, min((bi + 1) * BLOCK_N, n)
            cs, ce = bj * BLOCK_K, min((bj + 1) * BLOCK_K, k)
            wq[rs:re, cs:ce] = w32[rs:re, cs:ce] / scale[bi, bj]
    wq8 = wq.clamp(-448, 448).to(torch.float8_e4m3fn)
    got = dequant_fp8_block(wq8, scale)
    ref = torch.empty(n, k)
    for bi in range(scale.shape[0]):
        for bj in range(scale.shape[1]):
            rs, re = bi * BLOCK_N, min((bi + 1) * BLOCK_N, n)
            cs, ce = bj * BLOCK_K, min((bj + 1) * BLOCK_K, k)
            ref[rs:re, cs:ce] = wq8[rs:re, cs:ce].float() * scale[bi, bj]
    assert torch.equal(got, ref.to(torch.bfloat16)), "block dequant disagrees with the loop"

    print("w4a8_from_fp8 self-check ok")


if __name__ == "__main__":
    _self_check()
