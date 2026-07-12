"""``vtl_fp8`` -- online FP8 W8A8 with per-channel weight scales.

What stock ``--quantization=fp8`` does to a bf16 checkpoint (vLLM v0.22.1,
``Fp8OnlineLinearMethod.process_weights_after_loading``):

* weights: ``ops.scaled_fp8_quant(w, scale=None)`` -> one scalar scale for the
  whole tensor (``kFp8StaticTensorSym``).
* activations: already per-token on Hopper (``kFp8DynamicTokenSym``), because
  ``cutlass_fp8_supported()`` is true there.

So the only thing left on the table is weight granularity. A per-tensor scale is
set by the single largest element in the whole matrix, so one outlier channel
costs precision in every other channel. Per-output-channel scales fix that and
are *free at runtime*: ``cutlass_scaled_mm`` takes a ``[N, 1]`` ``scale_b`` just
as happily as a scalar, and ``CutlassFP8ScaledMMLinearKernel.can_implement``
returns True unconditionally.

Safety: ``--quantization=vtl_fp8`` is a *serve flag*, so if this name failed to
register vLLM would abort at startup with "Invalid quantization method". The
registration is therefore kept trivial -- a bare ``Fp8Config`` subclass, which
works as long as vLLM has FP8 at all -- and every fragile part (the channelwise
kernel path) lives behind a try/except inside ``get_quant_method``. Worst case we
degrade to stock per-tensor fp8; we never fail to boot.

Set ``VTL_FP8_CHANNELWISE=0`` to force the stock path without touching the flag.
"""

from __future__ import annotations

import logging
import os

from vtl.registry import register_patch

log = logging.getLogger("vtl")

IGNORE_ENV = "VTL_FP8_IGNORE"
CHANNELWISE_ENV = "VTL_FP8_CHANNELWISE"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def parse_ignored_layers(raw: str | None) -> list[str]:
    """Comma-separated layer names to keep in bf16, e.g. ``lm_head,model.layers.0``."""
    if not raw:
        return []
    return [name.strip() for name in raw.split(",") if name.strip()]


def channelwise_enabled(raw: str | None) -> bool:
    if raw is None:
        return True
    return raw.strip().lower() in _TRUTHY


def _build_channelwise_method(quant_config, stock_method):
    """Return a per-channel-weight variant of ``stock_method``, or raise.

    Kept separate so every import that could drift across vLLM versions is inside
    one try/except at the call site.
    """
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization import fp8 as fp8_mod
    from vllm.model_executor.layers.quantization.fp8 import Fp8OnlineLinearMethod
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8StaticChannelSym,
    )
    from vllm.model_executor.utils import replace_parameter

    if not fp8_mod.cutlass_fp8_supported():
        raise RuntimeError("CUTLASS FP8 unavailable; Marlin needs a scalar weight scale")

    class ChannelwiseFp8OnlineLinearMethod(Fp8OnlineLinearMethod):
        def __init__(self, cfg) -> None:
            super().__init__(cfg)
            # Must be set before create_weights() calls init_fp8_linear_kernel(),
            # which selects the kernel from this key.
            self.weight_quant_key = kFp8StaticChannelSym

        def process_weights_after_loading(self, layer) -> None:
            if getattr(layer, "_already_called_process_weights_after_loading", False):
                return
            if self.use_marlin:  # Marlin needs a scalar weight scale
                return super().process_weights_after_loading(layer)

            assert not self.block_quant
            layer.input_scale = None
            # weight is [out_features, in_features]; "per token" over its rows is
            # exactly per-output-channel.
            qweight, weight_scale = ops.scaled_fp8_quant(
                layer.weight, scale=None, use_per_token_if_dynamic=True
            )
            # CUTLASS wants B column-major [K, N]. The kernel's own
            # process_weights_after_loading only pads, it does not transpose.
            replace_parameter(layer, "weight", qweight.t().data)
            replace_parameter(layer, "weight_scale", weight_scale.data)
            self.fp8_linear.process_weights_after_loading(layer)
            layer._already_called_process_weights_after_loading = True

    upgraded = ChannelwiseFp8OnlineLinearMethod(quant_config)
    upgraded.marlin_input_dtype = stock_method.marlin_input_dtype
    return upgraded


@register_patch("fp8", default=True)
def apply() -> None:
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    @register_quantization_config("vtl_fp8")
    class VtlFp8Config(Fp8Config):
        # One warning per process, not one per layer.
        _fallback_logged = False

        def __init__(self, ignored_layers: list[str] | None = None) -> None:
            super().__init__(
                is_checkpoint_fp8_serialized=False,
                activation_scheme="dynamic",
                ignored_layers=(
                    ignored_layers
                    if ignored_layers is not None
                    else parse_ignored_layers(os.environ.get(IGNORE_ENV))
                ),
                weight_block_size=None,
            )

        @classmethod
        def get_name(cls):
            return "vtl_fp8"

        @classmethod
        def get_config_filenames(cls) -> list[str]:
            # Empty => weight_utils.get_quant_config() constructs us with no args,
            # which is the bf16-checkpoint path we care about.
            return []

        @classmethod
        def from_config(cls, config: dict) -> "VtlFp8Config":
            return cls()

        def get_quant_method(self, layer, prefix: str):
            from vllm.model_executor.layers.linear import LinearBase
            from vllm.model_executor.layers.quantization.fp8 import (
                Fp8OnlineLinearMethod,
            )

            method = super().get_quant_method(layer, prefix)
            if not isinstance(layer, LinearBase):
                return method  # attention / MoE: leave stock behaviour alone

            # SUBSTRING ignore -- vLLM's is_layer_skipped is EXACT prefix match
            # (`prefix in ignored_layers`), so a bare "in_proj_a" passed to the parent never
            # matches `...linear_attn.in_proj_a`. Enforce our substrings here so the GDN F32
            # SSM projections (in_proj_a/b) and the vision tower are NEVER fp8-quantized --
            # quantizing the Mamba decay/beta path is a correctness break. Keeps big GDN
            # matmuls (in_proj_qkv/z, out_proj) and dense GEMMs on fp8.
            if prefix and any(pat and pat in prefix for pat in self.ignored_layers):
                from vllm.model_executor.layers.linear import UnquantizedLinearMethod
                return UnquantizedLinearMethod()

            if type(method) is not Fp8OnlineLinearMethod:
                return method  # ignored layer (bf16) or fp8-serialized checkpoint
            if not channelwise_enabled(os.environ.get(CHANNELWISE_ENV)):
                return method

            try:
                return _build_channelwise_method(self, method)
            except Exception as exc:
                if not VtlFp8Config._fallback_logged:
                    VtlFp8Config._fallback_logged = True
                    log.warning(
                        "vtl: channelwise fp8 unavailable (%s); "
                        "falling back to stock per-tensor fp8",
                        exc,
                    )
                return method

    log.info(
        "vtl: registered quantization method 'vtl_fp8' (channelwise=%s, ignored=%s)",
        channelwise_enabled(os.environ.get(CHANNELWISE_ENV)),
        parse_ignored_layers(os.environ.get(IGNORE_ENV)) or "none",
    )


def _self_check() -> None:
    assert parse_ignored_layers(None) == []
    assert parse_ignored_layers("") == []
    assert parse_ignored_layers("lm_head") == ["lm_head"]
    assert parse_ignored_layers(" lm_head , model.layers.0 ,") == [
        "lm_head",
        "model.layers.0",
    ]

    assert channelwise_enabled(None) is True  # on unless explicitly disabled
    assert channelwise_enabled("1") is True
    assert channelwise_enabled("0") is False
    assert channelwise_enabled("off") is False
    print("quant_fp8 self-check ok")


if __name__ == "__main__":
    _self_check()
