"""Fuse `silu_and_mul -> dynamic_per_token_scaled_fp8_quant` (the down_proj input path) into
one CUDA kernel: `vllm_cuda::silu_and_mul_dynamic_per_token_quant`.

vLLM v0.25.0 has NO per-token fused SiLU-mul quant op (`ActivationQuantFusionPass` only fuses
the static/per-tensor and block variants). So we introduce our own op (kernel in
vtl/csrc/silu_mul_quant.cu) and insert it into the compiled graph with a replacement pattern
that matches `silu_and_mul(x) -> dynamic_per_token_scaled_fp8_quant(...)`.

Three parts:
  1. Register the op's CUDA kernel  -- import vtl._C.
  2. Register a FakeTensor/meta impl -- so torch.compile can trace the inserted node.
  3. Register a replacement pattern into ActivationQuantFusionPass -- mirrors vLLM's own
     SiluMulFp8StaticQuantPattern but for the per-token key (`kFp8DynamicTokenSym`) that vLLM
     leaves un-fused. Safe by construction: the pattern matcher only rewrites the pair when the
     bf16 silu intermediate feeds nothing but the quant, and the replacement op is
     parity-tested against exactly this stock composition. A non-match is a no-op; any API
     drift is caught and degrades to stock silu + our Phase-1b per-token quant.

REQUIRES `fuse_act_quant: true` in the serve --compilation-config (alongside
`fuse_norm_quant`) so ActivationQuantFusionPass actually runs -- see docker-compose-*.yaml.
Verify on the box that it fired: `Replaced N patterns` (N>0) in the activation_quant pass log.

Env: `VTL_ENABLE_SILU_MUL_QUANT=0` disables the whole patch.
`VTL_SILU_MUL_FUSION=0` keeps the op available (for A/B and tests) but skips step 3.
"""

from __future__ import annotations

import logging
import os

from vtl.registry import register_patch

log = logging.getLogger("vtl")

OP = "vllm_cuda::silu_and_mul_dynamic_per_token_quant"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _fusion_enabled() -> bool:
    return os.environ.get("VTL_SILU_MUL_FUSION", "1").strip().lower() in _TRUTHY


_fake_registered = False


def _register_fake() -> None:
    """FakeTensor rule so the inserted node traces. The op mutates result+scale in place and
    returns (); the fake only has to run without touching data. Idempotent -- registering a
    fake twice for the same op raises."""
    global _fake_registered
    if _fake_registered:
        return
    import torch

    @torch.library.register_fake("vllm_cuda::silu_and_mul_dynamic_per_token_quant")
    def _fake(result, scale, input, scale_ub=None):  # noqa: ANN001
        return None

    _fake_registered = True


def _register_fusion_pattern() -> bool:
    """Teach vLLM's ActivationQuantFusionPass to rewrite the per-token
    `silu_and_mul -> dynamic_per_token_scaled_fp8_quant` pair into our fused op.

    Mirrors vLLM's own `SiluMulFp8StaticQuantPattern` exactly, but for the per-token
    (`kFp8DynamicTokenSym`) key that vLLM leaves un-fused, targeting our op. Safe by
    construction: the torch pattern matcher only rewrites the pair when the bf16 silu
    intermediate is consumed *solely* by the quant (otherwise it does not match), and the
    replacement op is parity-tested against exactly this stock composition
    (bench/test_silu_mul_quant.py). A non-match is a no-op; a match is semantically identical.

    Everything is guarded: any API drift across vLLM versions is caught and degrades to the
    unfused (still-fast: stock silu + our Phase-1b per-token quant) path. Returns True if the
    pattern hook was installed. Requires `fuse_act_quant` in the serve --compilation-config so
    the pass actually runs.
    """
    import torch
    from torch._higher_order_ops.auto_functionalize import auto_functionalized
    from vllm.compilation.passes.fusion import act_quant_fusion as aqf
    from vllm.compilation.passes.fusion.matcher_utils import MatcherQuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTokenSym,
    )

    fused_op = torch.ops.vllm_cuda.silu_and_mul_dynamic_per_token_quant.default
    # ActivationQuantPattern asserts quant_key in QUANT_OPS (it is: per-token ->
    # dynamic_per_token_scaled_fp8_quant) and in FUSED_OPS (it is not, by default -- vLLM has
    # no per-token fused silu op). Register ours so the base class accepts the key.
    aqf.FUSED_OPS[kFp8DynamicTokenSym] = fused_op

    class _SiluMulPerTokenPattern(aqf.ActivationQuantPattern):
        def __init__(self) -> None:
            super().__init__(kFp8DynamicTokenSym)
            self.quant_matcher = MatcherQuantFP8(kFp8DynamicTokenSym)

        def get_inputs(self):
            return list(self.silu_and_mul_matcher.inputs())  # [gate_up]; scale is produced

        @property
        def pattern(self):
            def _pattern(input):
                silu_out = self.silu_and_mul_matcher(input)
                result_quant, scale = self.quant_matcher(silu_out)
                return result_quant, scale

            return _pattern

        @property
        def replacement(self):
            def _replacement(input):
                d = input.shape[-1] // 2
                result = torch.empty(
                    (*input.shape[:-1], d), device=input.device, dtype=self.quant_dtype
                )
                scale = torch.empty(
                    (input.shape[0], 1), device=input.device, dtype=torch.float32
                )
                at = auto_functionalized(
                    self.FUSED_OP, result=result, scale=scale, input=input, scale_ub=None
                )
                return at[1], at[2]  # result, scale (mutated args, in schema order)

            return _replacement

    Pass = aqf.ActivationQuantFusionPass
    if getattr(Pass, "__vtl_pertoken_patched__", False):
        return True

    orig_init = Pass.__init__

    def __init__(self, config):  # noqa: ANN001
        orig_init(self, config)
        try:
            # Build the pattern here (not at plugin-apply time): the Matchers read the current
            # vLLM config, which is only set once the pass is being constructed.
            self.register(_SiluMulPerTokenPattern())
        except Exception as exc:  # duplicate registration / API drift -> stay unfused
            log.warning("vtl: silu-mul per-token pattern register failed (%s); unfused", exc)

    Pass.__init__ = __init__
    Pass.__vtl_pertoken_patched__ = True
    return True


@register_patch("silu_mul_quant", default=True)
def apply() -> None:
    import torch

    import vllm._C_stable_libtorch  # noqa: F401  -- defines the _C schemas we reference

    import vtl._C  # noqa: F401  -- registers vllm_cuda::silu_and_mul_dynamic_per_token_quant

    if not hasattr(torch.ops.vllm_cuda, "silu_and_mul_dynamic_per_token_quant"):
        log.warning("vtl: fused silu-mul quant op did not register; skipping")
        return

    _register_fake()
    log.info("vtl: fused silu-mul + per-token fp8-quant kernel installed (down_proj)")

    if _fusion_enabled():
        try:
            if _register_fusion_pattern():
                log.info("vtl: silu-mul fusion pattern registered (down_proj fused)")
        except Exception as exc:  # never take the server down for a fusion hook
            log.warning("vtl: silu-mul fusion pattern registration failed (%s); unfused", exc)


def _self_check() -> None:
    import os

    from vtl.registry import PATCH_REGISTRY, is_enabled

    patch = next(p for p in PATCH_REGISTRY if p.name == "silu_mul_quant")
    assert patch.default is True
    assert is_enabled(patch) is True

    os.environ["VTL_ENABLE_SILU_MUL_QUANT"] = "0"
    assert is_enabled(patch) is False
    os.environ.pop("VTL_ENABLE_SILU_MUL_QUANT")

    assert _fusion_enabled() is True
    os.environ["VTL_SILU_MUL_FUSION"] = "0"
    assert _fusion_enabled() is False
    os.environ.pop("VTL_SILU_MUL_FUSION")

    try:
        apply()
    except Exception:
        pass

    print("silu_mul_quant self-check ok")


if __name__ == "__main__":
    _self_check()
