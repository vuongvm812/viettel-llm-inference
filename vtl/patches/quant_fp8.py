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
    """Comma-separated linear layers to keep in bf16.

    Values must be **exact** fully-qualified names of ``LinearBase`` modules, e.g.
    ``model.layers.0.mlp.down_proj``. vLLM matches with ``prefix in ignored_layers``
    (``is_layer_skipped``, ``skip_with_substr=False``), so a prefix like
    ``model.layers.0`` matches nothing.

    ``lm_head`` can never match: ``Fp8Config.get_quant_method`` only dispatches on
    ``LinearBase``/``RoutedExperts``/``Attention``, and ``ParallelLMHead`` is a
    ``VocabParallelEmbedding``. With ``tie_word_embeddings=true`` (this checkpoint)
    no ``lm_head`` module is built at all.
    """
    if not raw:
        return []
    return [name.strip() for name in raw.split(",") if name.strip()]


def channelwise_enabled(raw: str | None) -> bool:
    if raw is None:
        return True
    return raw.strip().lower() in _TRUTHY


# get_quant_method() is called once per layer (~250x for 36 layers), so the imports,
# the capability probe and the class body below are built once and reused. Both outcomes
# are cached: on Ampere the probe fails every time and we do not want 250 retries.
_METHOD_CLS: type | None = None
_METHOD_ERR: Exception | None = None


def _channelwise_method_cls() -> type:
    """Cache ``_make_channelwise_cls()``, success or failure."""
    global _METHOD_CLS, _METHOD_ERR
    if _METHOD_ERR is not None:
        raise _METHOD_ERR
    if _METHOD_CLS is None:
        try:
            _METHOD_CLS = _make_channelwise_cls()
        except Exception as exc:
            _METHOD_ERR = exc
            raise
    return _METHOD_CLS


def _make_channelwise_cls() -> type:
    """Build a per-channel-weight subclass of ``Fp8OnlineLinearMethod``, or raise.

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

    return ChannelwiseFp8OnlineLinearMethod


def _build_channelwise_method(quant_config, stock_method):
    """Return a per-channel-weight variant of ``stock_method``, or raise."""
    upgraded = _channelwise_method_cls()(quant_config)
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
    # Values are exact LinearBase FQNs -- vLLM matches with `prefix in ignored_layers`.
    assert parse_ignored_layers("model.layers.0.mlp.down_proj") == [
        "model.layers.0.mlp.down_proj"
    ]
    assert parse_ignored_layers(" model.layers.0.mlp.down_proj , model.layers.1.mlp.gate_up_proj ,") == [
        "model.layers.0.mlp.down_proj",
        "model.layers.1.mlp.gate_up_proj",
    ]

    assert channelwise_enabled(None) is True  # on unless explicitly disabled
    assert channelwise_enabled("1") is True
    assert channelwise_enabled("0") is False
    assert channelwise_enabled("off") is False

    # _channelwise_method_cls() runs once per linear layer (~250x). Both outcomes must be
    # cached: without this, Ampere re-probes cutlass_fp8_supported() on every layer.
    global _METHOD_CLS, _METHOD_ERR, _make_channelwise_cls
    saved = (_METHOD_CLS, _METHOD_ERR, _make_channelwise_cls)
    try:
        calls = []

        _METHOD_CLS = _METHOD_ERR = None
        _make_channelwise_cls = lambda: (calls.append(1), str)[1]  # noqa: E731
        assert _channelwise_method_cls() is str
        assert _channelwise_method_cls() is str
        assert len(calls) == 1, f"success not cached: {len(calls)} builds"

        calls.clear()
        _METHOD_CLS = _METHOD_ERR = None

        def _boom():
            calls.append(1)
            raise RuntimeError("CUTLASS FP8 unavailable")

        _make_channelwise_cls = _boom
        for _ in range(3):
            try:
                _channelwise_method_cls()
            except RuntimeError:
                pass
        assert len(calls) == 1, f"failure not cached: {len(calls)} probes"
    finally:
        _METHOD_CLS, _METHOD_ERR, _make_channelwise_cls = saved

    print("quant_fp8 self-check ok")


if __name__ == "__main__":
    _self_check()
