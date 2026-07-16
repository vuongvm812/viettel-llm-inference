"""GDN (linear-attention) custom CUDA kernels for the qwen3_5 hot path (18/24 layers).

The GDN output norm is `RMSNormGated` (per head, head_v_dim=128, fp32 weight), applied to the
gated-delta core output before `out_proj`. qwen3_5 configures it as **norm_before_gate=True,
group_size=None, activation in {silu, sigmoid}** -- i.e. `out = rms_norm(x) * act(z)`. The stock
CUDA path runs FLA's Triton `rmsnorm_fn` and then a SEPARATE per-token fp8 quant inside out_proj
(there is no stock CUDA fusion of the two -- only a ROCm/AITER pass exists), so the bf16 norm
output makes a full HBM round-trip on every one of the 18 GDN layers, every token.

Two ways to cut that, both gated by `VTL_ENABLE_GDN_KERNELS` (default OFF -- arm on the box only
after `make test-kernel` parity passes):

  * FUSION (default; `VTL_GDN_RMSNORM_FUSION=1`): register a torch pattern that rewrites the
    compiled subgraph `RMSNormGated(x,z,w) -> reshape(-1, H*D) -> dynamic_per_token_fp8_quant`
    into our single `vllm_cuda::gated_rmsnorm_dynamic_per_token_quant` node (norm + per-token
    quant in one kernel). Mirrors vLLM's own AITER gated-norm fusion, retargeted to CUDA. This
    is the win: it removes the norm-output HBM round-trip. VERIFY on the box that it fired
    (`Replaced N patterns`, N>0) -- the exact v0.25.0 pass/matcher plumbing must be confirmed;
    any drift is caught and degrades to unfused, and we fall back to the eager kernel below.

  * EAGER (`VTL_GDN_RMSNORM_FUSION=0`, or fusion registration failed): monkeypatch
    `RMSNormGated.forward` to call our standalone `vllm_cuda::gated_rmsnorm` kernel. Not a
    fusion, so only an A/B lever / eager-mode fallback -- NOT installed alongside the fusion,
    because a traced monkeypatched forward would replace the RMSNormGated node the pattern needs.

The old version of this file assumed norm_before_gate=False (gate-then-norm) and its shim only
fired when norm_before_gate was False -- so on qwen3_5 it never ran. Fixed here.
"""

from __future__ import annotations

import logging
import os

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vtl")

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _fusion_enabled() -> bool:
    return os.environ.get("VTL_GDN_RMSNORM_FUSION", "1").strip().lower() in _TRUTHY


def _gate_is_silu(activation) -> bool:  # noqa: ANN001
    """swish == silu; only "sigmoid" turns it off. Matches RMSNormGated.forward_static."""
    return str(activation) != "sigmoid"


_fake_registered = False


def _register_fake() -> None:
    """FakeTensor rule for the fused op so the inserted node traces. Idempotent."""
    global _fake_registered
    if _fake_registered:
        return
    import torch

    @torch.library.register_fake("vllm_cuda::gated_rmsnorm_dynamic_per_token_quant")
    def _fake(result, scale, input, gate, weight, epsilon, num_heads, gate_is_silu, scale_ub=None):  # noqa: ANN001, E501
        return None

    _fake_registered = True


def _install_gated_rmsnorm_eager() -> bool:
    """Monkeypatch RMSNormGated.forward to our standalone kernel for the exact case it supports
    (norm_before_gate=True, group_size=None). Defensive: unknown signature/shape -> original."""
    import torch

    from vllm.model_executor.layers.layernorm import RMSNormGated  # may not exist -> caller guards

    if already_patched(RMSNormGated, "forward"):
        return True

    orig_forward = RMSNormGated.forward

    def forward(self, x, gate=None):  # noqa: ANN001
        try:
            weight = self.weight
            eps = getattr(self, "variance_epsilon", getattr(self, "eps", 1e-6))
            group = next(
                (
                    getattr(self, a)
                    for a in ("group_size", "n_groups", "num_groups")
                    if getattr(self, a, None) is not None
                ),
                None,
            )
            norm_before_gate = getattr(self, "norm_before_gate", None)
            is_silu = _gate_is_silu(getattr(self, "activation", "silu"))
            ok = (
                gate is not None
                and norm_before_gate is True  # our kernel = norm-then-gate; None/False -> bail
                and (group is None or group == x.shape[-1])  # no sub-grouping
                and x.is_cuda
                and x.is_contiguous()
                and gate.is_contiguous()
                and gate.shape == x.shape
                and weight.dtype == torch.float32
                and weight.numel() == x.shape[-1]
            )
            if not ok:
                return orig_forward(self, x, gate)
            out = torch.empty_like(x)
            torch.ops.vllm_cuda.gated_rmsnorm(out, x, gate, weight, float(eps), is_silu)
            return out
        except Exception:
            return orig_forward(self, x, gate)

    RMSNormGated.forward = mark_patched(forward, orig_forward)
    return True


def _discover_gdn_shapes(config):  # noqa: ANN001
    """(num_heads, head_dim, eps, is_silu) for each GDN layer, the vLLM way. Empty on any drift."""
    try:
        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
    except Exception:
        return []
    out = []
    try:
        layers = get_layers_from_vllm_config(config, GatedDeltaNetAttention)
    except Exception:
        return []
    for layer in layers.values():
        try:
            tp = getattr(layer, "tp_size", 1) or 1
            h = layer.num_v_heads // tp
            d = layer.head_v_dim
            norm = getattr(layer, "norm", None)
            eps = float(getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-6)))
            is_silu = _gate_is_silu(getattr(norm, "activation", "silu"))
            out.append((int(h), int(d), eps, bool(is_silu)))
        except Exception:
            continue
    # de-dup identical shapes (all 18 layers usually share one)
    return list(dict.fromkeys(out))


def _register_fusion_pattern() -> bool:
    """Register the `RMSNormGated -> reshape -> per-token fp8 quant` -> fused-op rewrite into
    ActivationQuantFusionPass, one pattern per discovered (H,D,eps,act). Best-effort: any import
    or shape-discovery failure returns False (caller falls back to the eager kernel). Never
    corrupts output -- the matcher only fires on an exact structural match and the replacement is
    parity-tested (bench/test_gdn_gated_rmsnorm.py)."""
    import torch
    from torch._higher_order_ops.auto_functionalize import auto_functionalized
    from vllm.compilation.passes.fusion import act_quant_fusion as aqf
    from vllm.compilation.passes.fusion.matcher_utils import (
        MatcherQuantFP8,
        MatcherRMSNormGated,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTokenSym

    fused_op = torch.ops.vllm_cuda.gated_rmsnorm_dynamic_per_token_quant.default

    def _make_pattern(H, D, eps, is_silu):  # noqa: ANN001
        hidden = H * D

        class _GatedNormQuantPattern(aqf.ActivationQuantPattern):
            def __init__(self) -> None:
                super().__init__(kFp8DynamicTokenSym)
                self.FUSED_OP = fused_op
                self.quant_matcher = MatcherQuantFP8(kFp8DynamicTokenSym)
                self.norm_matcher = MatcherRMSNormGated(eps)

            def get_inputs(self):
                x = torch.empty((2 * H, D), device="cuda", dtype=torch.bfloat16)
                z = torch.empty((2 * H, D), device="cuda", dtype=torch.bfloat16)
                w = torch.empty((D,), device="cuda", dtype=torch.float32)
                return [x, z, w]

            @property
            def pattern(self):
                def _pattern(x, z, weight):
                    normed = self.norm_matcher(x, z, weight)
                    merged = normed.reshape(-1, hidden)
                    q, s = self.quant_matcher(merged)
                    return q, s

                return _pattern

            @property
            def replacement(self):
                def _replacement(x, z, weight):
                    n_tokens = x.shape[0] // H
                    result = torch.empty(
                        (n_tokens, hidden), device=x.device, dtype=self.quant_dtype
                    )
                    scale = torch.empty((n_tokens, 1), device=x.device, dtype=torch.float32)
                    at = auto_functionalized(
                        self.FUSED_OP,
                        result=result,
                        scale=scale,
                        input=x,
                        gate=z,
                        weight=weight,
                        epsilon=eps,
                        num_heads=H,
                        gate_is_silu=is_silu,
                        scale_ub=None,
                    )
                    return at[1], at[2]  # result, scale (mutated args, schema order)

                return _replacement

        return _GatedNormQuantPattern

    # Needed so ActivationQuantPattern.__init__'s `key in FUSED_OPS` assert passes standalone.
    aqf.FUSED_OPS.setdefault(kFp8DynamicTokenSym, fused_op)

    Pass = aqf.ActivationQuantFusionPass
    if getattr(Pass, "__vtl_gdn_rmsnorm_patched__", False):
        return True

    orig_init = Pass.__init__

    def __init__(self, config):  # noqa: ANN001
        orig_init(self, config)
        try:
            shapes = _discover_gdn_shapes(config)
            for H, D, eps, is_silu in shapes:
                self.register(_make_pattern(H, D, eps, is_silu)())
            if shapes:
                log.info("vtl: GDN gated-norm+quant fusion patterns registered for %s", shapes)
        except Exception as exc:  # API drift -> stay unfused
            log.warning("vtl: GDN gated-norm fusion register failed (%s); unfused", exc)

    Pass.__init__ = __init__
    Pass.__vtl_gdn_rmsnorm_patched__ = True
    return True


@register_patch("gdn_kernels", default=False)
def apply() -> None:
    import torch

    import vllm._C_stable_libtorch  # noqa: F401  -- defines the _C schemas we reference

    import vtl._C  # noqa: F401  -- registers vllm_cuda::gated_rmsnorm(+_quant)

    if not hasattr(torch.ops.vllm_cuda, "gated_rmsnorm"):
        log.warning("vtl: gated_rmsnorm op did not register; skipping GDN kernels")
        return

    _register_fake()

    installed_fusion = False
    if _fusion_enabled():
        try:
            installed_fusion = _register_fusion_pattern()
            if installed_fusion:
                log.info("vtl: GDN gated-RMSNorm+quant FUSION installed (VERIFY it fired on H200)")
        except Exception as exc:
            log.warning("vtl: GDN gated-norm fusion install failed (%s); trying eager", exc)

    if not installed_fusion:
        try:
            if _install_gated_rmsnorm_eager():
                log.info("vtl: GDN gated-RMSNorm eager kernel installed (VERIFY parity on H200)")
        except Exception as exc:
            log.warning("vtl: GDN gated-RMSNorm install failed (%s); stock kernels kept", exc)


def _self_check() -> None:
    import os

    from vtl.registry import PATCH_REGISTRY, is_enabled

    patch = next(p for p in PATCH_REGISTRY if p.name == "gdn_kernels")
    assert patch.default is False, "GDN kernels are opt-in until parity is verified on the box"
    assert is_enabled(patch) is False

    os.environ["VTL_ENABLE_GDN_KERNELS"] = "1"
    assert is_enabled(patch) is True
    os.environ.pop("VTL_ENABLE_GDN_KERNELS")

    assert _fusion_enabled() is True
    os.environ["VTL_GDN_RMSNORM_FUSION"] = "0"
    assert _fusion_enabled() is False
    os.environ.pop("VTL_GDN_RMSNORM_FUSION")

    assert _gate_is_silu("silu") is True
    assert _gate_is_silu("swish") is True
    assert _gate_is_silu("sigmoid") is False

    print("gdn_kernels self-check ok")


if __name__ == "__main__":
    _self_check()
