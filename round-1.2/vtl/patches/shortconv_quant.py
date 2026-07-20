"""Quantize the LFM2 short-conv projections (``in_proj``/``out_proj``) to fp8.

TPOT lever. Decode is memory-bandwidth bound, so per-token weight traffic sets the
decode latency. Every weight in the LFM2.5 body is already fp8 EXCEPT the 10 short-conv
layers' ``in_proj``/``out_proj`` -- stock ``ShortConv.__init__`` builds them without a
``quant_config`` (see vllm ``.../mamba/short_conv.py``), so they stay bf16: ~336 MB/token,
~9-11% of decode weight traffic, the single largest un-quantized chunk.

This patch wraps ``ShortConv.__init__`` and, when a quant_config is active, rebuilds ONLY
``in_proj`` and ``out_proj`` with that config. The depthwise ``conv`` weight is left bf16
(tiny, and the Triton ``causal_conv1d_update`` kernel expects its layout). No vLLM source
edit and no change to the caller (``Lfm2ShortConvDecoderLayer``) -- ShortConv already reads
the global vLLM config in its own ``__init__``.

Nothing else is needed: with ``--quantization=vtl_fp8`` the existing ``VtlFp8Config``
(see quant_fp8.py) quantizes any ``LinearBase`` whose prefix isn't in ``VTL_FP8_IGNORE``,
and the existing kernels light up for free -- ``operator_norm``->``in_proj`` fuses via the
RMSNormQuantFusionPass (rms_norm_quant), and ``out_proj``'s (non-norm) input gets the
standalone dynamic_per_token_scaled_fp8_quant kernel.

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
