"""Force / A-B the GDN prefill backend, and optionally route the prefill scan to vtl's own
chunk-scan CUDA op. Both levers ride the ``VTL_ENABLE_GDN_KERNELS`` master switch and are
default-off no-ops, so stock FlashInfer runs unless explicitly armed.

Two env levers (both read directly, like mul_sigmoid_quant.py -- one parent arms the family):

  ``VTL_GDN_PREFILL_BACKEND`` in {stock, flashinfer, cutedsl, triton}   (default ``stock`` -> no-op)
      Pins ``ChunkGatedDeltaRule``'s forward method so the three stock backends can be A/B'd at the
      served shapes without a fork. ``stock`` leaves vLLM's own ``_resolve_gdn_prefill_backend``
      decision untouched. Maps: flashinfer -> forward_cuda, cutedsl -> forward_cutedsl,
      triton -> forward_native.

  ``VTL_GDN_CHUNK_SCAN`` = 0|1   (default 0)
      1 -> monkeypatch ``ChunkGatedDeltaRule.forward_cuda`` to call
      ``torch.ops.vllm_cuda.gdn_chunk_scan`` (vtl's own scan) AND pin the instance onto
      forward_cuda so the route actually runs. Defensive: any shape / arg-convention mismatch falls
      back to the original forward_cuda (stock FlashInfer).

HONEST CEILING (ponytail: this is a scaffold, not a shipping win): the vtl scan's kernel BODY is
today a SEQUENTIAL recurrence (vtl/csrc/gdn/chunk_scan.cu). It will NOT beat the chunk-parallel
FlashInfer/CuteDSL incumbents on long prefills -- it exists to make the op real + parity-testable
(bench/test_gdn_chunk_scan.py) and to be the base for the CUTLASS/CuteDSL WY-parallel body
(docs/plans/gdn-parallel-scan.md). Two conventions MUST be reconciled against vLLM v0.25.0 on the
box before trusting the route -- the shim bails rather than guess:
  * ``g`` gate: the kernel assumes per-STEP log-decay ``S <- S*exp(g_t)``. If vLLM passes a
    cumulative gate, the numbers are wrong -- we only fire when g is [L,H] and leave the box A/B
    (make bench-kernel + trace replay vs stock chunk_gated_delta_rule) as the definitive gate.
  * l2norm-of-q/k placement and beta placement -- matched to the decode recurrence, re-verify.

Do NOT enable VTL_GDN_CHUNK_SCAN in production until the WY-parallel body lands AND wins the on-box
bench. Default off; every failure path degrades to stock.
"""

from __future__ import annotations

import logging
import os

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vtl")

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# backend -> the ChunkGatedDeltaRule method that runs it.
_BACKEND_METHOD = {
    "flashinfer": "forward_cuda",
    "cutedsl": "forward_cutedsl",
    "triton": "forward_native",
}


def _gdn_enabled() -> bool:
    """Whole GDN kernel family is gated by one flag, default OFF. Mirrors mul_sigmoid_quant."""
    return os.environ.get("VTL_ENABLE_GDN_KERNELS", "0").strip().lower() in _TRUTHY


def _backend_choice() -> str:
    return os.environ.get("VTL_GDN_PREFILL_BACKEND", "stock").strip().lower()


def _chunk_scan_enabled() -> bool:
    return os.environ.get("VTL_GDN_CHUNK_SCAN", "0").strip().lower() in _TRUTHY


_fake_registered = False


def _register_fake() -> None:
    """FakeTensor rule for gdn_chunk_scan so a traced call runs. Mutates o+final_state, returns
    (). Idempotent."""
    global _fake_registered
    if _fake_registered:
        return
    import torch

    @torch.library.register_fake("vllm_cuda::gdn_chunk_scan")
    def _fake(o, q, k, v, g, beta, query_start_loc, initial_state, final_state, qk_l2norm):  # noqa: ANN001, E501
        return None

    _fake_registered = True


def _target_method(cls) -> str | None:  # noqa: ANN001
    """Which forward method the instance should use, given the two env levers. None = leave stock.
    Chunk-scan wins: it needs forward_cuda (now our route). Then an explicit backend pin. Only
    returns a name the class actually has."""
    if _chunk_scan_enabled():
        want = "forward_cuda"
    else:
        choice = _backend_choice()
        if choice in ("", "stock"):
            return None
        want = _BACKEND_METHOD.get(choice)
        if want is None:
            log.warning("vtl: VTL_GDN_PREFILL_BACKEND=%r not in %s; leaving stock",
                        choice, sorted(_BACKEND_METHOD))
            return None
    return want if hasattr(cls, want) else None


def _install_chunk_scan_route(cls) -> None:  # noqa: ANN001
    """Wrap ChunkGatedDeltaRule.forward_cuda so it calls our op when VTL_GDN_CHUNK_SCAN=1, with a
    fail-closed fallback to the original on any mismatch. Only replaces the method once."""
    import torch

    if already_patched(cls, "forward_cuda"):
        return
    orig_forward_cuda = cls.forward_cuda

    def forward_cuda(self, q, k, v, g, beta, initial_state, output_final_state,  # noqa: ANN001
                     cu_seqlens=None, chunk_indices=None, chunk_offsets=None,
                     use_qk_l2norm_in_kernel=True, core_attn_out=None, **kw):
        if not _chunk_scan_enabled():
            return orig_forward_cuda(
                self, q, k, v, g, beta, initial_state, output_final_state,
                cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, core_attn_out=core_attn_out, **kw)
        try:
            # Only fire on the exact packed-varlen [L,H,D] layout our kernel supports; anything
            # else (batched leading dim, missing cu_seqlens, g not [L,H]) bails to stock so the
            # convention mismatch never corrupts output.
            ok = (
                cu_seqlens is not None
                and q.dim() == 3 and k.dim() == 3 and v.dim() == 3
                and q.is_cuda and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
                and q.shape == k.shape
                and q.shape[0] == v.shape[0] and q.shape[1] == v.shape[1]
                and q.shape[2] == v.shape[2]  # Dk == Dv (kernel assumes square state)
                and g is not None and g.dim() == 2 and g.shape == (q.shape[0], q.shape[1])
                and beta is not None and beta.shape == g.shape
            )
            if not ok:
                raise NotImplementedError("layout/convention unsupported by vtl gdn_chunk_scan")

            L, H, D = q.shape
            qsl = cu_seqlens.to(torch.int32).contiguous()
            S = int(qsl.numel()) - 1
            init = None
            if initial_state is not None:
                init = initial_state.to(torch.float32).contiguous()
                if tuple(init.shape) != (S, H, D, D):
                    raise NotImplementedError("initial_state shape != [S,H,Dk,Dv]")
            o = torch.empty(L, H, D, dtype=q.dtype, device=q.device)
            final_state = torch.empty(S, H, D, D, dtype=torch.float32, device=q.device)
            torch.ops.vllm_cuda.gdn_chunk_scan(
                o, q.contiguous(), k.contiguous(), v.contiguous(),
                g.to(torch.float32).contiguous(), beta.to(torch.float32).contiguous(),
                qsl, init, final_state, bool(use_qk_l2norm_in_kernel))
            if core_attn_out is not None:  # match stock: copy into the caller's buffer
                o_flat = o.reshape(-1)
                co_flat = core_attn_out.reshape(-1)
                co_flat[: o_flat.numel()].copy_(o_flat)
            return o, final_state
        except Exception as exc:  # any drift -> stock FlashInfer, never crash the model
            log.warning("vtl: gdn_chunk_scan route bailed (%s); stock backend", exc)
            return orig_forward_cuda(
                self, q, k, v, g, beta, initial_state, output_final_state,
                cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, core_attn_out=core_attn_out, **kw)

    cls.forward_cuda = mark_patched(forward_cuda, orig_forward_cuda)


def _install_backend_pin(cls) -> None:  # noqa: ANN001
    """Wrap __init__ so the instance's _forward_method is pinned per the env levers, after vLLM's
    own resolution runs. Idempotent."""
    if already_patched(cls, "__init__"):
        return
    orig_init = cls.__init__

    def __init__(self, *args, **kw):  # noqa: ANN001, N807
        orig_init(self, *args, **kw)
        try:
            want = _target_method(type(self))
            if want is not None:
                self._forward_method = getattr(self, want)
                log.info("vtl: GDN prefill pinned to %s (chunk_scan=%s, backend=%s)",
                         want, _chunk_scan_enabled(), _backend_choice())
        except Exception as exc:
            log.warning("vtl: GDN backend pin failed (%s); stock selection kept", exc)

    cls.__init__ = mark_patched(__init__, orig_init)


@register_patch("gdn_prefill_backend", default=True)
def apply() -> None:
    # Registered default=True so the registry always SELECTS it, but it is inert unless the parent
    # flag is armed AND a lever is actually engaged -- same "one parent arms the family" model as
    # mul_sigmoid_quant.py.
    if not _gdn_enabled():
        return
    if _backend_choice() in ("", "stock") and not _chunk_scan_enabled():
        return  # nothing to force; leave stock vLLM entirely alone

    try:
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import ChunkGatedDeltaRule
    except Exception as exc:  # symbol moved/renamed in this vLLM -> stay stock
        log.warning("vtl: ChunkGatedDeltaRule not importable (%s); GDN prefill backend unchanged",
                    exc)
        return

    if _chunk_scan_enabled():
        try:
            import vllm._C_stable_libtorch  # noqa: F401
            import vtl._C  # noqa: F401  -- registers vllm_cuda::gdn_chunk_scan
            import torch

            if not hasattr(torch.ops.vllm_cuda, "gdn_chunk_scan"):
                log.warning("vtl: gdn_chunk_scan op absent; not routing prefill scan")
            else:
                _register_fake()
                _install_chunk_scan_route(ChunkGatedDeltaRule)
                log.info("vtl: GDN prefill scan routed to vtl gdn_chunk_scan "
                         "(SEQUENTIAL baseline -- VERIFY it wins on the box before shipping)")
        except Exception as exc:
            log.warning("vtl: gdn_chunk_scan route install failed (%s); stock backend", exc)

    _install_backend_pin(ChunkGatedDeltaRule)


def _self_check() -> None:
    from vtl.registry import PATCH_REGISTRY, is_enabled

    patch = next(p for p in PATCH_REGISTRY if p.name == "gdn_prefill_backend")
    # Registered default=True (always selected) but apply() is inert unless the parent is armed.
    assert patch.default is True
    assert is_enabled(patch) is True  # selected by default...

    for key in ("VTL_ENABLE_GDN_KERNELS", "VTL_GDN_PREFILL_BACKEND", "VTL_GDN_CHUNK_SCAN"):
        os.environ.pop(key, None)
    assert _gdn_enabled() is False  # ...but fail-closed: family off
    assert _backend_choice() == "stock"
    assert _chunk_scan_enabled() is False

    os.environ["VTL_ENABLE_GDN_KERNELS"] = "1"
    assert _gdn_enabled() is True
    os.environ["VTL_GDN_PREFILL_BACKEND"] = "cutedsl"
    assert _backend_choice() == "cutedsl"
    os.environ["VTL_GDN_CHUNK_SCAN"] = "on"
    assert _chunk_scan_enabled() is True

    # _target_method: chunk-scan wins (forward_cuda); else explicit backend; else None.
    class _Fake:
        def forward_cuda(self): ...
        def forward_cutedsl(self): ...
        def forward_native(self): ...

    assert _target_method(_Fake) == "forward_cuda"  # chunk scan on
    os.environ["VTL_GDN_CHUNK_SCAN"] = "0"
    assert _target_method(_Fake) == "forward_cutedsl"  # backend=cutedsl
    os.environ["VTL_GDN_PREFILL_BACKEND"] = "stock"
    assert _target_method(_Fake) is None
    os.environ["VTL_GDN_PREFILL_BACKEND"] = "bogus"
    assert _target_method(_Fake) is None  # unknown -> stock

    class _NoCuteDSL:
        def forward_cuda(self): ...

    os.environ["VTL_GDN_PREFILL_BACKEND"] = "cutedsl"
    assert _target_method(_NoCuteDSL) is None  # method absent -> stock

    for key in ("VTL_ENABLE_GDN_KERNELS", "VTL_GDN_PREFILL_BACKEND", "VTL_GDN_CHUNK_SCAN"):
        os.environ.pop(key, None)
    print("gdn_prefill_backend self-check ok")


if __name__ == "__main__":
    _self_check()
