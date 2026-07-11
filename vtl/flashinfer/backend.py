"""vtl custom FlashInfer v1 attention backend for H200 (sm_90) + FP8 KV cache.

vLLM v0.22.1 already ships a complete FlashInfer v1 backend
(``vllm/v1/attention/backends/flashinfer.py``) that supports sm_90, ``fp8_e4m3`` KV,
head_size 128, tensor-core fp8 decode, pinned CUDA-graph buffers and ``fast_decode_plan``
(sync-free replay planning). It is simply NOT auto-selected on Hopper -- ``FLASH_ATTN``
outranks ``FLASHINFER`` in ``CudaPlatformBase._get_backend_priorities`` for sm_90.

This module subclasses that builtin so ``vtl/patches/flashinfer_backend.py`` can force it
via ``register_backend``, and adds the tunings that are NOT already upstream on our
deployment's path (TP=1, non-DCP, TRTLLM unavailable on Hopper):

* a workspace-buffer floor -- SGLang floors FlashInfer's workspace at 512 MiB for Qwen;
  vLLM defaults to 394 MiB (``VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE``). Tune via
  ``VTL_FI_WORKSPACE_MB`` (default 512).
* an OPTIONAL ragged-prefill + ``merge_attn_states`` extend path (SGLang's prefill split),
  gated behind ``VTL_FI_RAGGED_PREFILL``. It currently falls back to the base single paged
  prefill -- see ``VtlFlashInferMetadataBuilder.build`` for why it is H200-validation-gated.

Everything else (fp8 KV view, bmm1/bmm2 scales, tensor-core wrappers, cudagraph capture)
is inherited unchanged from the builtin and is already correct for this deployment.

This module imports vLLM at top level on purpose: it is only ever imported inside the
vLLM process, lazily, when the registry resolves the class path -- never during the
pure-python self-check (which lives in the patch module).
"""

from __future__ import annotations

import logging
import os

import torch
from vllm import envs
from vllm.v1.attention.backends.flashinfer import (
    FlashInferBackend,
    FlashInferMetadataBuilder,
)

log = logging.getLogger("vtl")

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _workspace_floor_bytes() -> int:
    """SGLang's >=512 MiB FlashInfer workspace floor, overridable via VTL_FI_WORKSPACE_MB."""
    try:
        mb = int(os.environ.get("VTL_FI_WORKSPACE_MB", "512"))
    except ValueError:
        mb = 512
    return max(mb, 0) * 1024 * 1024


class VtlFlashInferMetadataBuilder(FlashInferMetadataBuilder):
    def _get_workspace_buffer(self) -> torch.Tensor:
        # Mirror the base allocation (flashinfer.py:_get_workspace_buffer) but floor the
        # size. The base caches on self._workspace_buffer, so only the first call sizes it.
        if self._workspace_buffer is None:
            size = max(int(envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE), _workspace_floor_bytes())
            self._workspace_buffer = torch.zeros(size, dtype=torch.uint8, device=self.device)
        return self._workspace_buffer

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        if _env_on("VTL_FI_RAGGED_PREFILL"):
            # ponytail: SGLang's prefill split = ragged self-attn(new tokens, causal) +
            # paged context-attn(prefix, non-causal) merged with merge_attn_states. Doing
            # this correctly on a SINGLE GPU requires planning the paged context wrapper over
            # PREFIX-ONLY blocks (context seq_lens = num_computed_tokens) so the just-written
            # new-token KV is not double-counted -- the base BatchDCPPrefillWrapper only does
            # the causal=False context run because in DCP the new KV lives on another rank,
            # so it is NOT reusable as-is here. The failure mode is SILENT output corruption
            # (not a crash), and it cannot be byte-validated off-GPU. Ships as a fallback to
            # the base single paged prefill until proven on the H200.
            # Upgrade path: build prefix-only paged plan from num_computed_tokens, run the
            # ragged new-token wrapper, merge_attn_states, then gate-on by default.
            log.warning(
                "vtl: VTL_FI_RAGGED_PREFILL set but the ragged path is H200-validation-"
                "gated; using the base paged prefill"
            )
        return super().build(common_prefix_len, common_attn_metadata, fast_build)


class VtlFlashInferBackend(FlashInferBackend):
    # Only redirect the builder; get_impl_cls (-> FlashInferImpl) and everything else
    # (supported_kv_cache_dtypes incl. fp8_e4m3, head sizes [64,128,256,512],
    # supports_compute_capability (7,5)..(12,1), get_kv_cache_shape,
    # forward_includes_kv_cache_update=False) are inherited and already correct.
    @staticmethod
    def get_builder_cls() -> type[VtlFlashInferMetadataBuilder]:
        return VtlFlashInferMetadataBuilder
