"""Fuse `attn_output * sigmoid(gate) -> dynamic_per_token_scaled_fp8_quant` (the o_proj input
path on the 6 full-attn + MTP layers) into one CUDA kernel:
`vllm_cuda::mul_sigmoid_dynamic_per_token_quant`.

Direct parallel to `silu_mul_quant.py` (down_proj), with two structural differences driven by
how vLLM emits the gate multiply:
  * The activation is DECOMPOSED -- `aten.sigmoid` then `aten.mul.Tensor`, NOT a fused
    `silu_and_mul`-style op (there is no `MatcherSiluAndMul` equivalent for it). So the pattern
    is built from primitive torch (`input * torch.sigmoid(gate)`) rather than an activation
    matcher.
  * The two operands are SEPARATE tensors (attn_output, gate) and the output is the full hidden
    width H, not a halved [T,2I] split.

The quant half is identical to the silu case (`MatcherQuantFP8(kFp8DynamicTokenSym)`), so the
one integration snag is that `FUSED_OPS[kFp8DynamicTokenSym]` is already claimed by the silu
patch. We `setdefault` it (only to satisfy the base class's `key in FUSED_OPS` assert when silu
is disabled) and then set `self.FUSED_OP` to OUR op on the pattern instance, so this fusion never
depends on -- or collides with -- the silu patch's dict entry.

REQUIRES `fuse_act_quant: true` in the serve --compilation-config (already set alongside
`fuse_norm_quant`) so ActivationQuantFusionPass runs. Verify it fired on the box:
`Replaced N patterns` (N>0) in the activation_quant pass log.

GATING: this whole GDN-family optimization rides `VTL_ENABLE_GDN_KERNELS` (default OFF) --
`apply()` no-ops unless it is set, so it arms/disarms together with the gated-RMSNorm kernels.
`VTL_MUL_SIGMOID_FUSION=0` keeps the op available (for A/B and tests) but skips the pattern.
"""

from __future__ import annotations

import logging
import os

from vtl.registry import register_patch

log = logging.getLogger("vtl")

OP = "vllm_cuda::mul_sigmoid_dynamic_per_token_quant"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _gdn_enabled() -> bool:
    """The whole GDN kernel family (this + gated_rmsnorm) is gated by one flag, default OFF."""
    return os.environ.get("VTL_ENABLE_GDN_KERNELS", "0").strip().lower() in _TRUTHY


def _fusion_enabled() -> bool:
    return os.environ.get("VTL_MUL_SIGMOID_FUSION", "1").strip().lower() in _TRUTHY


_fake_registered = False


def _register_fake() -> None:
    """FakeTensor rule so the inserted node traces. The op mutates result+scale in place and
    returns (); the fake only has to run without touching data. Idempotent."""
    global _fake_registered
    if _fake_registered:
        return
    import torch

    @torch.library.register_fake("vllm_cuda::mul_sigmoid_dynamic_per_token_quant")
    def _fake(result, scale, input, gate, scale_ub=None):  # noqa: ANN001
        return None

    _fake_registered = True


def _register_fusion_pattern() -> bool:
    """Teach vLLM's ActivationQuantFusionPass to rewrite the per-token
    `mul(attn_output, sigmoid(gate)) -> dynamic_per_token_scaled_fp8_quant` subgraph into our
    fused op. Built from primitive torch ops (no activation matcher exists for sigmoid-gate).

    Safe by construction: the torch pattern matcher only rewrites when the bf16 product is
    consumed *solely* by the quant, and the replacement op is parity-tested against exactly this
    stock composition (bench/test_mul_sigmoid_quant.py). A non-match is a no-op; API drift is
    caught and degrades to the unfused (stock sigmoid+mul + our Phase-1b per-token quant) path.
    """
    import torch
    from torch._higher_order_ops.auto_functionalize import auto_functionalized
    from vllm.compilation.passes.fusion import act_quant_fusion as aqf
    from vllm.compilation.passes.fusion.matcher_utils import MatcherQuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTokenSym,
    )

    fused_op = torch.ops.vllm_cuda.mul_sigmoid_dynamic_per_token_quant.default
    # Only so the base ActivationQuantPattern.__init__ assert (`key in FUSED_OPS`) passes when
    # the silu patch is disabled. Never clobbers silu's entry -- we bind our op per-instance below.
    aqf.FUSED_OPS.setdefault(kFp8DynamicTokenSym, fused_op)

    class _MulSigmoidPerTokenPattern(aqf.ActivationQuantPattern):
        def __init__(self) -> None:
            super().__init__(kFp8DynamicTokenSym)
            # Base set self.FUSED_OP from FUSED_OPS[key] -- which may be the silu op. Rebind to
            # OUR op on this instance so the replacement calls the right kernel regardless of
            # dict state or patch order.
            self.FUSED_OP = fused_op
            self.quant_matcher = MatcherQuantFP8(kFp8DynamicTokenSym)

        def get_inputs(self):
            # Two separate example inputs (attn_output, gate); scale is produced by the quant.
            import torch as _torch

            x = _torch.empty((5, 128), device="cuda", dtype=_torch.bfloat16)
            g = _torch.empty((5, 128), device="cuda", dtype=_torch.bfloat16)
            return [x, g]

        @property
        def pattern(self):
            def _pattern(input, gate):
                # Match the model's `attn_output * torch.sigmoid(gate)` (mul arg order: the
                # non-sigmoid tensor first), then the per-token fp8 quant of that product.
                act = input * torch.sigmoid(gate)
                result_quant, scale = self.quant_matcher(act)
                return result_quant, scale

            return _pattern

        @property
        def replacement(self):
            def _replacement(input, gate):
                result = torch.empty(
                    input.shape, device=input.device, dtype=self.quant_dtype
                )
                scale = torch.empty(
                    (input.shape[0], 1), device=input.device, dtype=torch.float32
                )
                at = auto_functionalized(
                    self.FUSED_OP,
                    result=result,
                    scale=scale,
                    input=input,
                    gate=gate,
                    scale_ub=None,
                )
                return at[1], at[2]  # result, scale (mutated args, in schema order)

            return _replacement

    Pass = aqf.ActivationQuantFusionPass
    if getattr(Pass, "__vtl_osigmoid_patched__", False):
        return True

    orig_init = Pass.__init__

    def __init__(self, config):  # noqa: ANN001
        orig_init(self, config)
        try:
            # Build the pattern at pass-construction time (Matchers read the live vLLM config).
            self.register(_MulSigmoidPerTokenPattern())
        except Exception as exc:  # duplicate registration / API drift -> stay unfused
            log.warning("vtl: o_proj sigmoid-mul pattern register failed (%s); unfused", exc)

    Pass.__init__ = __init__
    Pass.__vtl_osigmoid_patched__ = True
    return True


@register_patch("mul_sigmoid_quant", default=True)
def apply() -> None:
    import torch

    if not _gdn_enabled():
        return  # gated with the rest of the GDN family behind VTL_ENABLE_GDN_KERNELS

    import vllm._C_stable_libtorch  # noqa: F401  -- defines the _C schemas we reference

    import vtl._C  # noqa: F401  -- registers vllm_cuda::mul_sigmoid_dynamic_per_token_quant

    if not hasattr(torch.ops.vllm_cuda, "mul_sigmoid_dynamic_per_token_quant"):
        log.warning("vtl: fused o_proj sigmoid-mul quant op did not register; skipping")
        return

    _register_fake()
    log.info("vtl: fused sigmoid-mul + per-token fp8-quant kernel installed (o_proj)")

    if _fusion_enabled():
        try:
            if _register_fusion_pattern():
                log.info("vtl: o_proj sigmoid-mul fusion pattern registered")
        except Exception as exc:  # never take the server down for a fusion hook
            log.warning("vtl: o_proj sigmoid-mul fusion registration failed (%s); unfused", exc)


def _self_check() -> None:
    import os

    from vtl.registry import PATCH_REGISTRY, is_enabled

    patch = next(p for p in PATCH_REGISTRY if p.name == "mul_sigmoid_quant")
    assert patch.default is True
    assert is_enabled(patch) is True  # registry always selects it; the GDN flag gates apply()

    # apply() is a no-op unless VTL_ENABLE_GDN_KERNELS is armed.
    os.environ.pop("VTL_ENABLE_GDN_KERNELS", None)
    assert _gdn_enabled() is False
    os.environ["VTL_ENABLE_GDN_KERNELS"] = "1"
    assert _gdn_enabled() is True
    os.environ.pop("VTL_ENABLE_GDN_KERNELS")

    assert _fusion_enabled() is True
    os.environ["VTL_MUL_SIGMOID_FUSION"] = "0"
    assert _fusion_enabled() is False
    os.environ.pop("VTL_MUL_SIGMOID_FUSION")

    try:
        apply()  # no vLLM here -> import fails inside, caught by apply_all in prod; ok if it raises
    except Exception:
        pass

    print("mul_sigmoid_quant self-check ok")


if __name__ == "__main__":
    _self_check()
