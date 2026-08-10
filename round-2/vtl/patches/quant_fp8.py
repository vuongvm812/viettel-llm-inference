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

log = logging.getLogger("vllm.vtl.fp8")

IGNORE_ENV = "VTL_FP8_IGNORE"
CHANNELWISE_ENV = "VTL_FP8_CHANNELWISE"
# V2 model runner: vLLM's fp8 kernel priority picks Marlin first on CUDA, but Marlin's
# stable-ABI output alloc (aten::empty + memory_format) crashes under V2's dynamic-shape
# compiled graph. When set, force the CUTLASS fp8 kernel instead. Default off -> V1 untouched.
FORCE_CUTLASS_ENV = "VTL_V2_FORCE_CUTLASS_FP8"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# One-shot warning latch for the channelwise weight-load fallback (per process).
_load_fallback_logged = False
# One-shot warning latch for the force-CUTLASS path (per process).
_force_cutlass_logged = False


def parse_ignored_layers(raw: str | None) -> list[str]:
    """Comma-separated layer names to keep in bf16, e.g. ``lm_head,model.layers.0``."""
    if not raw:
        return []
    return [name.strip() for name in raw.split(",") if name.strip()]


def channelwise_enabled(raw: str | None) -> bool:
    if raw is None:
        return True
    return raw.strip().lower() in _TRUTHY


def _force_cutlass_fp8() -> bool:
    """VTL_V2_FORCE_CUTLASS_FP8 -- default OFF (unset => V1/shipped is byte-identical)."""
    return os.environ.get(FORCE_CUTLASS_ENV, "0").strip().lower() in _TRUTHY


def _build_channelwise_method(quant_config, stock_method):
    """Return a per-channel-weight variant of ``stock_method``, or raise.

    Kept separate so every import that could drift across vLLM versions is inside
    one try/except at the call site.
    """
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization import fp8 as fp8_mod
    # Fp8LinearMethod is the stable base in every version; <=0.22's Fp8OnlineLinearMethod was
    # a subclass, folded back into it by 0.25. Subclass the base and let config drive online.
    from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8StaticChannelSym,
    )
    from vllm.model_executor.utils import replace_parameter

    if not fp8_mod.cutlass_fp8_supported():
        raise RuntimeError("CUTLASS FP8 unavailable; Marlin needs a scalar weight scale")

    class ChannelwiseFp8LinearMethod(Fp8LinearMethod):
        def __init__(self, cfg) -> None:
            super().__init__(cfg)
            # Must be set before create_weights() calls init_fp8_linear_kernel(),
            # which selects the kernel from this key.
            self.weight_quant_key = kFp8StaticChannelSym

        def create_weights(self, layer, *args, **kwargs) -> None:
            # Base create_weights() sets self.fp8_linear + self.use_marlin via
            # init_fp8_linear_kernel(); Marlin wins on CUDA (it's first in
            # _POSSIBLE_FP8_KERNELS). Under V2 that Marlin GEMM crashes at runtime
            # (stable-ABI aten::empty+memory_format under the dynamic-shape graph), so
            # re-select CUTLASS via the kernel's own force_kernel path. force_kernel
            # no-ops if CUTLASS can't implement the layer, so worst case is the
            # original Marlin -- no regression. Gated + fail-closed: never raise at load.
            super().create_weights(layer, *args, **kwargs)
            if not (self.use_marlin and _force_cutlass_fp8()):
                return
            global _force_cutlass_logged
            try:
                from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
                from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
                    CutlassFP8ScaledMMLinearKernel,
                )

                forced = init_fp8_linear_kernel(
                    activation_quant_key=self.activation_quant_key,
                    weight_quant_key=self.weight_quant_key,
                    weight_shape=layer.weight.shape,
                    input_dtype=self.input_dtype,
                    out_dtype=self.out_dtype,
                    module_name=self.__class__.__name__,
                    force_kernel=CutlassFP8ScaledMMLinearKernel,
                )
                if isinstance(forced, CutlassFP8ScaledMMLinearKernel):
                    self.fp8_linear = forced
                    self.use_marlin = False  # -> process_weights takes the CUTLASS branch
                elif not _force_cutlass_logged:
                    _force_cutlass_logged = True
                    log.warning(
                        "vtl: CUTLASS fp8 can't implement %s; keeping Marlin",
                        self.__class__.__name__,
                    )
            except Exception as exc:
                if not _force_cutlass_logged:
                    _force_cutlass_logged = True
                    log.warning("vtl: force-cutlass-fp8 failed (%s); keeping Marlin", exc)

        def process_weights_after_loading(self, layer) -> None:
            # Runs during weight loading -- OUTSIDE get_quant_method's guard -- so it must not
            # crash either: on any drift in the base's attrs/ops we fall back to the stock
            # per-tensor path. (Weights are not mutated before the first call that could fail,
            # so the fallback re-quantizes cleanly from the original bf16 weight.)
            try:
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
            except Exception as exc:
                global _load_fallback_logged
                if not _load_fallback_logged:
                    _load_fallback_logged = True
                    log.warning(
                        "vtl: channelwise weight-load failed (%s); stock per-tensor fp8", exc
                    )
                return super().process_weights_after_loading(layer)

    upgraded = ChannelwiseFp8LinearMethod(quant_config)
    upgraded.marlin_input_dtype = getattr(stock_method, "marlin_input_dtype", None)
    return upgraded


# Fp8Config is the parent. Imported at module scope (guarded) so VtlFp8Config below
# is defined at import time -- see the class docstring for why that must be so. The
# guard keeps the module importable without vLLM for the `make check` self-test, which
# never calls apply() and never instantiates the class.
try:
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config as _Fp8Config
except Exception:  # pragma: no cover - vLLM absent (self-check import only)
    _Fp8Config = object


class VtlFp8Config(_Fp8Config):
    """``vtl_fp8`` quant config.

    Defined at MODULE scope (not nested inside ``apply``) so it is picklable across
    ``spawn``: the multi-api-server / Rust-frontend engine-core launch pickles
    ``vllm_config``, which references this instance. A class nested in a function has
    qualname ``apply.<locals>.VtlFp8Config`` that pickle can't resolve, and the spawned
    child unpickles at plain import time (before ``apply`` runs), so the name must exist
    at module scope. ``apply()`` only registers it.
    """

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
        # Runs for EVERY Linear at load time -- it must NEVER raise, or the whole engine
        # dies at model load. No vLLM symbols are imported at the top: names drift across
        # versions (e.g. Fp8OnlineLinearMethod existed in <=0.22 but was folded into
        # Fp8LinearMethod by 0.25), so every import is lazy + guarded and any failure
        # degrades to whatever the parent returned.
        from vllm.model_executor.layers.linear import LinearBase

        method = super().get_quant_method(layer, prefix)
        if not isinstance(layer, LinearBase):
            return method  # attention / MoE: leave stock behaviour alone

        # SUBSTRING ignore -- vLLM's is_layer_skipped is EXACT prefix match
        # (`prefix in ignored_layers`), so a bare "lm_head" passed to the parent never
        # matches a nested prefix. Enforce our substrings here. Typically the only layer
        # kept in bf16 is `lm_head` (tied to the embeddings; small vocab). Every other
        # linear -- attn qkv/out_proj, MLP w13/w2, and any recurrent block's projections --
        # is an ordinary bf16 GEMM and quantizes to fp8 cleanly. Layers that must NOT be
        # touched (an F32 SSM decay/beta path, a vision tower) belong in VTL_FP8_IGNORE.
        if prefix and any(pat and pat in prefix for pat in self.ignored_layers):
            try:
                from vllm.model_executor.layers.linear import UnquantizedLinearMethod
                return UnquantizedLinearMethod()
            except Exception:
                return method  # can't force bf16 -> at least don't crash

        if not channelwise_enabled(os.environ.get(CHANNELWISE_ENV)):
            return method

        # Upgrade the dynamic/online fp8 linear method to per-CHANNEL weight scales.
        # Fp8LinearMethod is the stable base present in every version; the "online"
        # (bf16-checkpoint, dynamic-activation) behaviour is what our config selects.
        try:
            from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
            if not isinstance(method, Fp8LinearMethod):
                return method  # unquantized / ignored / serialized: leave as is
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


@register_patch("fp8", default=True)
def apply() -> None:
    from vllm.model_executor.layers.quantization import register_quantization_config

    register_quantization_config("vtl_fp8")(VtlFp8Config)

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
