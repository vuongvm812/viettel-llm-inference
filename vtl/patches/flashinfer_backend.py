"""Register + optionally force the vtl FlashInfer backend on H200 (sm_90) + FP8 KV.

On Hopper, vLLM auto-selects ``FLASH_ATTN`` over ``FLASHINFER``
(``CudaPlatformBase._get_backend_priorities``), even though the builtin FlashInfer backend
fully supports sm_90 + ``fp8_e4m3``. This patch:

1. points ``AttentionBackendEnum.FLASHINFER`` at our ``VtlFlashInferBackend`` via
   ``register_backend`` (so ``.get_class()`` resolves to our subclass), and
2. optionally forces selection.

Preferred forcing is the CLI flag ``--attention-backend FLASHINFER`` in the compose
command: ``CudaPlatformBase.get_attn_backend_cls`` validates the explicit backend and
returns ``get_path()``, which respects our override -- no monkeypatch needed. The platform
wrap below is a FALLBACK for when the compose file is frozen; enable it with
``VTL_FI_FORCE_SELECT=1``. It injects FLASHINFER only when selection is ``auto``
(``selected_backend is None``) on sm_90 with an fp8 KV cache and non-MLA, and lets vLLM's
own validation run.

Off by default (enable with ``VTL_ENABLE_FLASHINFER_BACKEND=1``) pending H200 bring-up --
forcing FlashInfer previously crashed this server, most likely a cudagraph/compilation
interaction; bring it up in stages per the plan.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vtl")

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def context_indptr_and_lastlen(seq_lens_np, query_lens_np, page_size: int):
    """Prefix-only paged plan for the ragged-prefill split (vtl/flashinfer/backend.py).

    Given per-request total sequence lengths (prefix + new) and new-token (query) lengths,
    return the block layout that covers ONLY the cached prefix of each request -- so the
    ragged new-token self-attention and the paged prefix-context attention never overlap.

    ``ctx_len = seq_len - query_len`` is the cached prefix length. A request whose prefix
    shares its last block with new tokens is bounded by ``last_page_len`` (= ctx_len within
    that block), so the paged kernel reads exactly the prefix even from a shared block.

    Pure NumPy, fully vectorized (no per-request Python loop) so it is cheap on the hot path
    and testable without a GPU. The block INDICES themselves come from the Triton
    ``_copy_page_indices_kernel`` on device using the returned ``ctx_num_blocks`` indptr.

    Returns ``(ctx_indptr, ctx_num_blocks, ctx_last_page_len)`` as int32 arrays, where
    ``ctx_indptr`` has shape ``(num_reqs + 1,)``.
    """
    seq_lens_np = np.asarray(seq_lens_np)
    query_lens_np = np.asarray(query_lens_np)
    ctx_lens = np.maximum(seq_lens_np - query_lens_np, 0)
    ctx_num_blocks = ((ctx_lens + page_size - 1) // page_size).astype(np.int32)
    ctx_indptr = np.zeros(ctx_lens.shape[0] + 1, dtype=np.int32)
    np.cumsum(ctx_num_blocks, dtype=np.int32, out=ctx_indptr[1:])
    # tokens in the final prefix block: ctx_len - (num_blocks - 1) * page_size, and 0 when
    # a request has no prefix at all.
    last = ctx_lens - (np.maximum(ctx_num_blocks, 1) - 1) * page_size
    ctx_last_page_len = np.where(ctx_num_blocks == 0, 0, last).astype(np.int32)
    return ctx_indptr, ctx_num_blocks, ctx_last_page_len


def _should_force_flashinfer(selected_backend, major, kv_cache_dtype, use_mla) -> bool:
    """True when 'auto' selection on H200 (sm_90) + fp8 KV, non-MLA, should become FlashInfer.

    Only overrides when ``selected_backend is None`` so an explicit ``--attention-backend``
    always wins. Blackwell (major 10) already ranks FlashInfer first, so we leave it alone.
    """
    if selected_backend is not None:
        return False
    if use_mla:
        return False
    if major != 9:
        return False
    if not kv_cache_dtype:
        return False
    return str(kv_cache_dtype).startswith("fp8")


def _wrap_get_attn_backend_cls() -> None:
    from vllm.platforms.cuda import CudaPlatformBase
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if already_patched(CudaPlatformBase, "get_attn_backend_cls"):
        return
    original = CudaPlatformBase.get_attn_backend_cls.__func__  # unwrap classmethod

    def get_attn_backend_cls(cls, selected_backend, attn_selector_config, num_heads=None):
        try:
            cap = cls.get_device_capability()
            major = cap.major if cap is not None else None
            if _should_force_flashinfer(
                selected_backend,
                major,
                getattr(attn_selector_config, "kv_cache_dtype", None),
                getattr(attn_selector_config, "use_mla", False),
            ):
                selected_backend = AttentionBackendEnum.FLASHINFER
                log.info("vtl: forcing FLASHINFER backend (sm_90 + fp8 KV cache)")
        except Exception:
            log.exception("vtl: flashinfer force-select failed; leaving selection to vLLM")
        # Let vLLM validate and resolve -- if FlashInfer is somehow invalid for this
        # config it raises the normal clear error rather than us swallowing it.
        return original(cls, selected_backend, attn_selector_config, num_heads)

    CudaPlatformBase.get_attn_backend_cls = classmethod(
        mark_patched(get_attn_backend_cls, original)
    )


@register_patch("flashinfer_backend", default=False)
def apply() -> None:
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.FLASHINFER,
        "vtl.flashinfer.backend.VtlFlashInferBackend",
    )
    forced = _env_on("VTL_FI_FORCE_SELECT")
    if forced:
        _wrap_get_attn_backend_cls()
    log.info(
        "vtl: flashinfer_backend registered (force_select=%s; prefer --attention-backend "
        "FLASHINFER)",
        forced,
    )


def _self_check() -> None:
    """Pure-python: exercises the force-decision every branch. No vLLM, no GPU."""
    force = _should_force_flashinfer

    # Auto selection on H200 + fp8 KV, non-MLA -> force FlashInfer.
    assert force(None, 9, "fp8_e4m3", False) is True
    assert force(None, 9, "fp8", False) is True
    assert force(None, 9, "fp8_e5m2", False) is True

    # Explicit user selection always wins.
    assert force("FLASH_ATTN", 9, "fp8_e4m3", False) is False
    assert force(object(), 9, "fp8_e4m3", False) is False

    # Wrong device: Ampere (8) unsupported; Blackwell (10) already prefers FlashInfer.
    assert force(None, 8, "fp8_e4m3", False) is False
    assert force(None, 10, "fp8_e4m3", False) is False
    assert force(None, None, "fp8_e4m3", False) is False

    # Not fp8 KV -> leave auto priorities (FlashAttention).
    assert force(None, 9, "auto", False) is False
    assert force(None, 9, None, False) is False
    assert force(None, 9, "", False) is False

    # MLA -> not our path.
    assert force(None, 9, "fp8_e4m3", True) is False

    # Env parsing.
    os.environ["_VTL_TEST_FLAG"] = "on"
    assert _env_on("_VTL_TEST_FLAG") is True
    os.environ["_VTL_TEST_FLAG"] = "0"
    assert _env_on("_VTL_TEST_FLAG") is False
    del os.environ["_VTL_TEST_FLAG"]
    assert _env_on("_VTL_TEST_FLAG") is False

    # Ragged-prefill prefix-only plan (page_size=16). Reference: the paged context wrapper
    # must cover exactly ctx_len = seq_len - query_len tokens per request.
    def ref_indices(seq_lens, query_lens, block_table, page):
        out = []
        for i, (s, q) in enumerate(zip(seq_lens, query_lens)):
            nb = (max(s - q, 0) + page - 1) // page
            out.extend(block_table[i][:nb])
        return out

    page = 16
    seq_lens = np.array([30, 16, 10, 48], dtype=np.int64)   # ctx = 20, 6, 0, 38
    query_lens = np.array([10, 10, 10, 10], dtype=np.int64)
    indptr, num_blocks, last = context_indptr_and_lastlen(seq_lens, query_lens, page)
    # ctx_len 20 -> 2 blocks last=4 ; 6 -> 1 block last=6 ; 0 -> 0 blocks last=0 ;
    # 38 -> 3 blocks last=38-32=6
    assert list(num_blocks) == [2, 1, 0, 3], list(num_blocks)
    assert list(last) == [4, 6, 0, 6], list(last)
    assert list(indptr) == [0, 2, 3, 3, 6], list(indptr)

    # Page-aligned prefix: last_page_len == page_size, not 0.
    _, nb2, last2 = context_indptr_and_lastlen(
        np.array([32]), np.array([16]), page  # ctx_len 16, exactly one full block
    )
    assert list(nb2) == [1] and list(last2) == [16], (list(nb2), list(last2))

    # No prefix anywhere -> everything zero (forward will skip the context wrapper).
    _, nb3, last3 = context_indptr_and_lastlen(np.array([10, 5]), np.array([10, 5]), page)
    assert list(nb3) == [0, 0] and list(last3) == [0, 0]

    # indptr total matches a reference per-request gather count.
    bt = [[100, 101, 102, 103, 104], [200, 201, 202, 203, 204],
          [300, 301, 302, 303, 304], [400, 401, 402, 403, 404]]
    assert int(indptr[-1]) == len(ref_indices(seq_lens, query_lens, bt, page))

    print("flashinfer_backend self-check ok")


if __name__ == "__main__":
    _self_check()
