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


_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _sub_enabled(name: str) -> bool:
    """A GDN scan/conv sub-kernel is OPT-IN: off unless VTL_GDN_<name> is explicitly truthy, even
    when the parent VTL_ENABLE_GDN_KERNELS is on. These kernels are unverified against vLLM's
    Triton/CuteDSL, so failing CLOSED (a forgotten flag leaves stock in place) is the safe default;
    the operator arms each one only after on-box parity + a bench win."""
    import os

    return os.environ.get(f"VTL_GDN_{name}", "0").strip().lower() in _TRUTHY


def _import_first(paths):
    """Return the first importable module among ``paths``, or None. vLLM moves GDN module
    locations across minors, so we try a few known spellings and bail if none resolve."""
    import importlib

    for p in paths:
        try:
            return importlib.import_module(p)
        except Exception:
            continue
    return None


# The mixer imports causal_conv1d_fn / causal_conv1d_update / the recurrent-decode fn BY NAME, so
# to intercept a call we must rebind the name on the MIXER module's namespace, not just the ops
# module. Candidate module paths (verify on v0.25.0).
_MIXER_PATHS = (
    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
    "vllm.model_executor.layers.mamba.gdn.base",
)


def _intercept(module, attr: str, make_wrapper) -> bool:
    """Rebind ``module.attr`` to ``make_wrapper(original)``, once. The wrapper MUST fall back to
    the original on any shape/dtype it does not positively recognise -- so a wrong arg-mapping can
    never corrupt output, only miss the fast path."""
    orig = getattr(module, attr, None)
    if orig is None or getattr(orig, "__vtl_wrapped__", None) is not None:
        return orig is not None
    wrapper = make_wrapper(orig)
    wrapper.__vtl_wrapped__ = orig
    setattr(module, attr, wrapper)
    return True


def _install_causal_conv1d() -> str:
    """Route causal_conv1d_update (decode) + causal_conv1d_fn (prefill) to our kernels for the
    plain non-paged case; bail to stock otherwise. Arg-mapping is best-effort against the known
    signature and MUST be parity-verified on v0.25.0 -- it bails on any mismatch, so it is safe.
    Returns "routing" (update path live), or "absent" (hook not found)."""
    import torch

    mixer = _import_first(_MIXER_PATHS)
    if mixer is None:
        return "absent"
    installed = False

    def make_update(orig):
        def wrapper(*args, **kwargs):
            try:
                x = args[0] if args else kwargs.get("x")
                conv_state = args[1] if len(args) > 1 else kwargs.get("conv_state")
                weight = args[2] if len(args) > 2 else kwargs.get("weight")
                bias = args[3] if len(args) > 3 else kwargs.get("bias")
                activation = args[4] if len(args) > 4 else kwargs.get("activation")
                csi = args[5] if len(args) > 5 else kwargs.get("conv_state_indices")
                # Only the simplest case: single-token, unpaged, width-4, contiguous.
                if (
                    csi is None
                    and x is not None
                    and x.dim() == 2
                    and conv_state is not None
                    and conv_state.dim() == 3
                    and weight is not None
                    and weight.size(1) == 4
                    and x.is_contiguous()
                    and conv_state.is_contiguous()
                    and (bias is None or bias.dtype == torch.float32)  # our op reads bias as f32
                ):
                    silu = activation in ("silu", "swish")
                    torch.ops.vllm_cuda.gdn_causal_conv1d_update(
                        x, conv_state, weight, bias, silu
                    )
                    return x
            except Exception:
                pass
            return orig(*args, **kwargs)

        return wrapper

    installed |= _intercept(mixer, "causal_conv1d_update", make_update)
    # causal_conv1d_fn (prefill) marshalling depends on the varlen/cache layout, which differs
    # enough across minors that we do NOT auto-route it blind -- the op + parity test exist
    # (torch.ops.vllm_cuda.gdn_causal_conv1d_fn), and the shim is wired here for the box to
    # complete once the exact v0.25.0 signature is confirmed. Until then, prefill conv stays stock.
    # The decode "update" path DOES route, so report "routing".
    return "routing" if installed else "absent"


def _install_recurrent_decode() -> str:
    """Route the packed-decode recurrent scan to our kernel when the tensor layout matches; bail
    to stock otherwise. Best-effort arg-mapping -- verify on v0.25.0. Returns "inert" (hook wired,
    pass-through until the packing is confirmed) or "absent"."""
    mixer = _import_first(_MIXER_PATHS)
    if mixer is None:
        return "absent"

    # The op is available (torch.ops.vllm_cuda.gdn_recurrent_decode); the packed-decode call
    # signature (q/k/v/g/beta/state packing) must be confirmed on the box before auto-routing,
    # so we register the interceptor conservatively: it currently recognises no call shape and
    # bails, leaving stock in place until the mapping is verified. This keeps the wiring present
    # and safe (never corrupts) without guessing the packing.
    def make_wrapper(orig):
        def wrapper(*args, **kwargs):
            return orig(*args, **kwargs)  # TODO(on-box): map to gdn_recurrent_decode once verified

        return wrapper

    ok = _intercept(mixer, "fused_recurrent_gated_delta_rule_packed_decode", make_wrapper)
    return "inert" if ok else "absent"  # hook present but pass-through until arg-map verified


def _install_chunk_scan() -> str:
    """Route the prefill chunked scan to our sequential-recurrence kernel via
    ChunkGatedDeltaRule.forward_cuda, shape-gated with fallback to the original. Verify on-box."""
    mod = _import_first(
        (
            "vllm.model_executor.layers.fla.ops.chunk",
            "vllm.model_executor.layers.fla.ops",
        )
    )
    cls = getattr(mod, "ChunkGatedDeltaRule", None) if mod is not None else None
    if cls is None:
        return "absent"
    if getattr(cls, "__vtl_chunk_patched__", False):
        return "inert"

    orig_forward = getattr(cls, "forward_cuda", None)
    if orig_forward is None:
        return "absent"

    # The forward_cuda(...) argument names/packing must be confirmed on v0.25.0 before we marshal
    # into gdn_chunk_scan (the op + parity test exist). Install a pass-through interceptor now so
    # the hook point is present and A/B-toggleable; it bails to the original until the mapping is
    # verified on the box.
    def forward_cuda(self, *args, **kwargs):
        return orig_forward(self, *args, **kwargs)  # TODO(on-box): map to gdn_chunk_scan

    cls.forward_cuda = forward_cuda
    cls.__vtl_chunk_patched__ = True
    return "inert"  # hook present but pass-through until arg-map verified


@register_patch("gdn_kernels", default=False)
def apply() -> None:
    import torch

    import vtl._C  # noqa: F401  -- registers vllm_cuda::gated_rmsnorm + the GDN ops

    if not hasattr(torch.ops.vllm_cuda, "gated_rmsnorm"):
        log.warning("vtl: gated_rmsnorm op did not register; skipping GDN kernels")
        return

    try:
        if _install_gated_rmsnorm():
            log.info("vtl: GDN gated-RMSNorm CUDA kernel installed (VERIFY parity on H200)")
    except Exception as exc:
        log.warning("vtl: GDN gated-RMSNorm install failed (%s); stock kernels kept", exc)

    # The scan/conv kernels are the highest-risk pieces: default-OFF via the parent flag, each
    # additionally toggleable, each shape-gated to fall back to the stock Triton/CuteDSL kernel.
    # The arg-marshalling from vLLM's mixer calls is best-effort and MUST be parity-verified on
    # v0.25.0 (bench/test_gdn_*.py validate the KERNELS directly, independent of this wiring).
    for name, flag, installer in (
        ("causal_conv1d", "CONV1D", _install_causal_conv1d),
        ("recurrent_decode", "RECURRENT", _install_recurrent_decode),
        ("chunk_scan", "CHUNK_SCAN", _install_chunk_scan),
    ):
        if not _sub_enabled(flag):
            continue
        try:
            status = installer()
            if status == "routing":
                log.info("vtl: GDN %s routing to our kernel (VERIFY parity on H200)", name)
            elif status == "inert":
                log.warning(
                    "vtl: GDN %s hook wired but INERT -- arg-mapping unverified on v0.25.0, "
                    "still using stock. Complete the marshalling on-box before trusting it.", name
                )
            else:  # "absent"
                log.info("vtl: GDN %s hook point not found on this vLLM; stock kept", name)
        except Exception as exc:
            log.warning("vtl: GDN %s install failed (%s); stock kernels kept", name, exc)


def _self_check() -> None:
    import os

    from vtl.registry import PATCH_REGISTRY, is_enabled

    patch = next(p for p in PATCH_REGISTRY if p.name == "gdn_kernels")
    assert patch.default is False, "GDN kernels are opt-in until parity is verified on the box"
    assert is_enabled(patch) is False

    os.environ["VTL_ENABLE_GDN_KERNELS"] = "1"
    assert is_enabled(patch) is True
    os.environ.pop("VTL_ENABLE_GDN_KERNELS")

    # Sub-flags are OPT-IN: OFF unless explicitly armed (fail-closed for unverified kernels).
    assert _sub_enabled("CONV1D") is False
    os.environ["VTL_GDN_CONV1D"] = "1"
    assert _sub_enabled("CONV1D") is True
    os.environ.pop("VTL_GDN_CONV1D")

    # _import_first returns None when nothing resolves (no vLLM here) -> installers no-op safely.
    assert _import_first(("nonexistent.mod.xyz",)) is None

    print("gdn_kernels self-check ok")


if __name__ == "__main__":
    _self_check()
