"""GDN (linear-attention) custom CUDA kernels for the qwen3_5 hot path (18/24 layers).

Phase 3, profile-gated. This wires the ONE tractable, self-contained GDN kernel shipped so
far -- the per-head gated RMSNorm (`vtl/csrc/gdn_gated_rmsnorm.cu`, the `linear_attn.norm`
applied to the gated-delta core output) -- by replacing `RMSNormGated.forward_cuda`.

DEFAULT OFF (`VTL_ENABLE_GDN_KERNELS` unset => disabled) for two reasons that MUST be closed on
the H200 before enabling:
  1. Parity: the kernel implements the standard Mamba2 RMSNormGated cast points; the exact
     v0.25.0 formula (group count, cast order, whether gate is silu'd in fp32) must be confirmed
     by bench/test_gdn_gated_rmsnorm.py against vLLM's own RMSNormGated.
  2. Hook: the RMSNormGated class path / forward signature in v0.25.0 must be confirmed
     (vllm/model_executor/layers/layernorm.py). The monkeypatch below is written defensively and
     falls back to the original forward on any shape/signature it does not recognise.

The harder GDN kernels (causal_conv1d prefill+update, the chunked/recurrent gated-delta scan)
are the next profile-gated targets -- see .claude/plans/valiant-moseying-wreath.md. They are NOT
shipped here: they need the Phase-0 baseline profile to justify them and the exact vLLM GDN
op signatures, and are wired via `_get_gdn_backend` / `torch.ops.vllm.qwen_gdn_attention_core`
rather than a norm monkeypatch.
"""

from __future__ import annotations

import logging

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vtl")


def _install_gated_rmsnorm() -> bool:
    """Monkeypatch RMSNormGated.forward_cuda to use our kernel when the layout matches.
    Returns True if installed. Defensive: unknown signature/shape -> original forward."""
    import torch

    from vllm.model_executor.layers.layernorm import RMSNormGated  # may not exist -> caller guards

    if already_patched(RMSNormGated, "forward"):
        return True

    orig_forward = RMSNormGated.forward

    def forward(self, x, gate=None):  # noqa: ANN001
        # Only take our path for the EXACT case the kernel supports, and bail to vLLM
        # otherwise. Our kernel computes h = x * silu(gate) then RMS-norms h (gate-THEN-norm,
        # i.e. norm_before_gate == False) over the FULL last dim (no sub-grouping). We fire
        # only when we can positively confirm that shape: a missing/ambiguous attribute means
        # bail, never guess. This is the finding-5 hardening -- the group attr may be named
        # n_groups/num_groups, and a wrong norm_before_gate would silently corrupt output.
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
            ok = (
                gate is not None
                and norm_before_gate is False  # our kernel = gate-then-norm; None -> bail
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
            torch.ops.vllm_cuda.gated_rmsnorm(out, x, gate, weight, float(eps))
            return out
        except Exception:
            return orig_forward(self, x, gate)

    RMSNormGated.forward = mark_patched(forward, orig_forward)
    return True


@register_patch("gdn_kernels", default=False)
def apply() -> None:
    import torch

    import vtl._C  # noqa: F401  -- registers vllm_cuda::gated_rmsnorm

    if not hasattr(torch.ops.vllm_cuda, "gated_rmsnorm"):
        log.warning("vtl: gated_rmsnorm op did not register; skipping GDN kernels")
        return

    try:
        if _install_gated_rmsnorm():
            log.info("vtl: GDN gated-RMSNorm CUDA kernel installed (VERIFY parity on H200)")
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

    print("gdn_kernels self-check ok")


if __name__ == "__main__":
    _self_check()
