"""Quantize the LFM2 short-conv projections (``in_proj``/``out_proj``).

TPOT lever. Decode is memory-bandwidth bound, so per-token weight traffic sets the
decode latency. Every weight in the LFM2.5 body is quantized EXCEPT the 10 short-conv
layers' ``in_proj``/``out_proj`` -- stock ``ShortConv.__init__`` builds them without a
``quant_config`` (see vllm ``.../mamba/short_conv.py``), so they stay bf16: ~336 MB/token,
~9-11% of decode weight traffic, the single largest un-quantized chunk.

Precision-agnostic: it forwards whatever ``quant_config`` is active, so these layers follow
the serve flag -- int4 under ``--quantization=vtl_w4a8`` ([[quant_w4a8]]), fp8 under
``vtl_fp8`` ([[quant_fp8]]).

This patch wraps ``ShortConv.__init__`` and, when a quant_config is active, rebuilds ONLY
``in_proj`` and ``out_proj`` with that config. The depthwise ``conv`` weight is left bf16
(tiny, and the Triton ``causal_conv1d_update`` kernel expects its layout). No vLLM source
edit and no change to the caller (``Lfm2ShortConvDecoderLayer``) -- ShortConv already reads
the global vLLM config in its own ``__init__``.

Nothing else is needed: the active config (``VtlW4A8Config`` / ``VtlFp8Config``) quantizes any
``LinearBase`` whose prefix isn't in its ignore list, and both projections feed the GEMM the
same fp8-e4m3 activation + per-token scale; only the weight side differs.

Both ends reach a fused kernel, but by DIFFERENT routes, and neither was free:

* ``operator_norm``->``in_proj`` fuses via RMSNormQuantFusionPass into rms_norm_quant -- but
  only because of two fork patches. ``ShortConv.forward`` dispatches through
  ``torch.ops.vllm.short_conv`` (direct_register_custom_op), so while ``in_proj`` lived inside
  that opaque op no FX pass could see its activation quant, and ``operator_norm`` sits outside
  it. ``short_conv.patch`` hoists ``in_proj`` out; ``lfm2.patch`` changes
  ``output = torch.empty_like(hidden_states)`` to ``empty_like(residual)`` so the norm output
  stops having a second consumer (pm.register_replacement will not match an internal node with
  extra users). Either patch alone is inert. Before them the 6 attention layers fused and these
  10 did not -- ~671 MB of avoidable prefill traffic at 8192 tokens.
* ``out_proj``'s input never goes through a pass at all: it is quantized by hand inside the op
  by mul_quant / bcx_conv_gate_quant, which write fp8 straight into the staging buffer.

Rebuild-after-init throws away the just-built bf16 linears; that cost is paid once at model
load, never at inference. Set ``VTL_ENABLE_SHORTCONV_QUANT=0`` to keep the conv projections bf16.
"""

from __future__ import annotations

import logging

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vtl")


def _rebuild_projections(self, *, merged_cls, row_cls, quant_config) -> bool:
    """Replace ``self.in_proj``/``self.out_proj`` with fp8-config'd copies.

    Returns True if it rebuilt, False if it left the layers bf16 (no quant_config).
    Mirrors the exact construction args in stock ``ShortConv.__init__`` -- same
    input/output sizes, bias, and prefix -- only adding ``quant_config``. ``self.conv``
    is deliberately untouched.
    """
    if quant_config is None:
        return False
    dim = self.conv_dim
    self.in_proj = merged_cls(
        input_size=dim,
        output_sizes=[dim] * 3,
        bias=self.bias,
        quant_config=quant_config,
        prefix=f"{self.prefix}.in_proj",
    )
    self.out_proj = row_cls(
        input_size=dim,
        output_size=dim,
        bias=self.bias,
        quant_config=quant_config,
        prefix=f"{self.prefix}.out_proj",
    )
    return True


@register_patch("shortconv_quant", default=True)
def apply() -> None:
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        RowParallelLinear,
    )
    from vllm.model_executor.layers.mamba.short_conv import ShortConv

    if already_patched(ShortConv, "__init__"):
        return

    original = ShortConv.__init__

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        try:
            quant_config = get_current_vllm_config().quant_config
            if _rebuild_projections(
                self,
                merged_cls=MergedColumnParallelLinear,
                row_cls=RowParallelLinear,
                quant_config=quant_config,
            ):
                log.info(
                    "vtl: shortconv_quant fp8'd %s in_proj/out_proj", self.prefix
                )
        except Exception:
            # Never break model construction; a failure just leaves this layer bf16.
            log.exception("vtl: shortconv_quant skipped for %s", getattr(self, "prefix", "?"))

    ShortConv.__init__ = mark_patched(__init__, original)
    log.info("vtl: shortconv_quant installed (fp8 short-conv in_proj/out_proj)")


def _self_check() -> None:
    """No vLLM: fakes verify the rebuild logic (which layers, args, skip path)."""
    from types import SimpleNamespace

    class _FakeLinear:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def make(prefix):
        # Fake ShortConv with a pre-existing bf16 conv we must NOT touch.
        return SimpleNamespace(
            conv_dim=2048, bias=False, prefix=prefix,
            conv="BF16_CONV_SENTINEL",
            in_proj="BF16_IN", out_proj="BF16_OUT",
        )

    # quant_config present -> rebuilds in_proj/out_proj, leaves conv alone.
    s = make("model.layers.0.conv")
    assert _rebuild_projections(
        s, merged_cls=_FakeLinear, row_cls=_FakeLinear, quant_config="QC"
    ) is True
    assert isinstance(s.in_proj, _FakeLinear) and isinstance(s.out_proj, _FakeLinear)
    assert s.conv == "BF16_CONV_SENTINEL"  # depthwise conv stays bf16
    assert s.in_proj.kwargs == {
        "input_size": 2048, "output_sizes": [2048, 2048, 2048],
        "bias": False, "quant_config": "QC", "prefix": "model.layers.0.conv.in_proj",
    }
    assert s.out_proj.kwargs == {
        "input_size": 2048, "output_size": 2048,
        "bias": False, "quant_config": "QC", "prefix": "model.layers.0.conv.out_proj",
    }

    # No quant_config -> no-op, layers stay bf16.
    s2 = make("x")
    assert _rebuild_projections(
        s2, merged_cls=_FakeLinear, row_cls=_FakeLinear, quant_config=None
    ) is False
    assert s2.in_proj == "BF16_IN" and s2.out_proj == "BF16_OUT"

    print("shortconv_quant self-check ok")


if __name__ == "__main__":
    _self_check()
