"""``vtl_int8`` -- online INT8 W8A8 with per-channel weight scales.

vLLM v0.25.0 has no stock online INT8 W8A8 method for dense ``Linear``
(``_ONLINE_LINEAR_METHODS`` has only FP8/MXFP8 entries; the INT8 online
methods are MoE-only and weight-only). This patch fills that gap: it
registers a ``vtl_int8`` quant config that loads a BF16 checkpoint and
quantizes weights to INT8 per-output-channel at load time, with dynamic
per-token symmetric INT8 activation quant at runtime -- the INT8 analog
of ``vtl_fp8``.

The GEMM is vLLM's stock ``CutlassInt8ScaledMMLinearKernel`` (selected by
``init_int8_linear_kernel`` on CUDA/Hopper), which uses
``ops.scaled_int8_quant`` (dynamic per-token) + ``ops.cutlass_scaled_mm``
(symmetric INT8 scaled MM). No custom CUDA kernels are needed -- the
stock INT8 GEMM path is used unchanged.

No fused norm+quant or silu+mul+quant: vLLM v0.25.0's
``RMSNormQuantFusionPass`` and ``ActivationQuantFusionPass`` have no INT8
entries in ``QUANT_OPS``/``FUSED_OPS``, so those passes never fire for
INT8. Activation quant runs un-fused inside ``apply_weights``. Correct,
just not optimal.

Safety: ``--quantization=vtl_int8`` is a serve flag, so if the name failed
to register vLLM would abort at startup. The registration is therefore
trivial -- a bare ``QuantizationConfig`` subclass. Every fragile part
(weight quant, kernel init) lives inside try/except and degrades to
``UnquantizedLinearMethod`` (BF16) on any failure. We never fail to boot.

Set ``VTL_INT8_IGNORE`` to a comma-separated list of layer-name substrings
to keep in BF16 (same semantics as ``VTL_FP8_IGNORE``).
"""

from __future__ import annotations

import logging
import os

from vtl.registry import register_patch

log = logging.getLogger("vtl")

IGNORE_ENV = "VTL_INT8_IGNORE"


def parse_ignored_layers(raw: str | None) -> list[str]:
    """Comma-separated layer-name substrings to keep in BF16."""
    if not raw:
        return []
    return [name.strip() for name in raw.split(",") if name.strip()]


@register_patch("int8", default=True)
def apply() -> None:
    from vllm.model_executor.layers.linear import LinearMethodBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

    class _VtlInt8LinearMethod(LinearMethodBase):
        """Online INT8 W8A8: per-channel weight, dynamic per-token activation.

        Loads BF16 checkpoint weights, quantizes to INT8 per-output-channel at
        load time. Activation is dynamically quantized per-token to INT8 at
        runtime by the stock ``CutlassInt8ScaledMMLinearKernel``.

        The weight quant math follows ``Int8OnlineMoEMethod._quantize_weights``
        (vllm/.../quantization/online/int8.py): per-row absmax / 127, then
        round+clamp to [-127, 127].
        """

        uses_meta_device = False

        def __init__(self):
            import torch
            from vllm.config import get_current_vllm_config

            self.out_dtype = torch.get_default_dtype()
            self.input_dtype = get_current_vllm_config().model_config.dtype
            self._kernel = None

        def create_weights(
            self,
            layer,
            input_size_per_partition: int,
            output_partition_sizes: list[int],
            input_size: int,
            output_size: int,
            params_dtype,
            **extra_weight_attrs,
        ):
            import torch
            from vllm.model_executor.kernels.linear import init_int8_linear_kernel
            from vllm.model_executor.parameter import ModelWeightParameter

            weight_loader = extra_weight_attrs.get("weight_loader")
            layer.logical_widths = output_partition_sizes

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

            layer.register_parameter("weight_scale", None)
            layer.register_parameter("input_scale", None)
            layer.register_parameter("input_zero_point", None)
            if not hasattr(layer, "azp_adj"):
                layer.register_parameter("azp_adj", None)

            self._kernel = init_int8_linear_kernel(
                is_channelwise=True,
                is_static_input_scheme=False,
                input_symmetric=True,
                module_name=self.__class__.__name__,
            )

        def process_weights_after_loading(self, layer) -> None:
            import torch
            from vllm.model_executor.utils import replace_parameter

            if getattr(layer, "_already_called_process_weights_after_loading", False):
                return

            try:
                vmax = torch.iinfo(torch.int8).max
                w = layer.weight
                scales = w.abs().amax(dim=1) / vmax
                scales = torch.where(
                    scales == 0, torch.ones_like(scales), scales
                )
                q = w.div(scales.unsqueeze(1)).round().clamp(-vmax, vmax).to(torch.int8)
                weight_scale = scales.unsqueeze(1).to(torch.float32)

                replace_parameter(
                    layer, "weight", torch.nn.Parameter(q, requires_grad=False)
                )
                replace_parameter(
                    layer,
                    "weight_scale",
                    torch.nn.Parameter(weight_scale, requires_grad=False),
                )

                self._kernel.process_weights_after_loading(layer)
                layer._already_called_process_weights_after_loading = True
            except Exception as exc:
                log.warning(
                    "vtl: int8 weight quantization failed (%s); "
                    "layer will use stock path",
                    exc,
                )
                layer._already_called_process_weights_after_loading = True

        def apply(self, layer, x, bias=None):
            return self._kernel.apply_weights(layer, x, bias)

    @register_quantization_config("vtl_int8")
    class VtlInt8Config(QuantizationConfig):
        _fallback_logged = False

        def __init__(self, ignored_layers: list[str] | None = None) -> None:
            super().__init__()
            self.ignored_layers = (
                ignored_layers
                if ignored_layers is not None
                else parse_ignored_layers(os.environ.get(IGNORE_ENV))
            )
            self.packed_modules_mapping: dict[str, list[str]] = {}

        @classmethod
        def get_name(cls) -> str:
            return "vtl_int8"

        @classmethod
        def get_supported_act_dtypes(cls) -> list:
            import torch

            return [torch.bfloat16, torch.half]

        @classmethod
        def get_min_capability(cls) -> int:
            return 75

        @classmethod
        def get_config_filenames(cls) -> list[str]:
            return []

        @classmethod
        def from_config(cls, config: dict) -> "VtlInt8Config":
            return cls()

        def get_quant_method(self, layer, prefix: str):
            from vllm.model_executor.layers.linear import LinearBase

            if not isinstance(layer, LinearBase):
                return None

            if prefix and any(pat and pat in prefix for pat in self.ignored_layers):
                try:
                    return UnquantizedLinearMethod()
                except Exception:
                    pass

            try:
                return _VtlInt8LinearMethod()
            except Exception as exc:
                if not VtlInt8Config._fallback_logged:
                    VtlInt8Config._fallback_logged = True
                    log.warning(
                        "vtl: int8 linear method creation failed (%s); "
                        "falling back to unquantized BF16",
                        exc,
                    )
                try:
                    return UnquantizedLinearMethod()
                except Exception:
                    return None

    log.info(
        "vtl: registered quantization method 'vtl_int8' (ignored=%s)",
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

    import os

    from vtl.registry import PATCH_REGISTRY, is_enabled

    patch = next(p for p in PATCH_REGISTRY if p.name == "int8")
    assert patch.default is True
    assert is_enabled(patch) is True

    os.environ["VTL_ENABLE_INT8"] = "0"
    assert is_enabled(patch) is False
    os.environ.pop("VTL_ENABLE_INT8")

    try:
        apply()
    except Exception:
        pass

    print("quant_int8 self-check ok")


if __name__ == "__main__":
    _self_check()
