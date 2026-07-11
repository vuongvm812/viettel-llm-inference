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
* SGLang's ragged-prefill + ``merge_attn_states`` extend path (``VTL_FI_RAGGED_PREFILL``):
  new-token self-attention runs through a contiguous ragged wrapper (no paged-KV gather),
  the cached prefix runs through a paged wrapper over PREFIX-ONLY blocks, and the two are
  merged by log-sum-exp. The base path instead runs one paged prefill over ``[prefix+new]``.
  The prefix is read from cache either way (no recompute); the win is skipping paged
  indirection on the new-token portion for long prefill chunks.

Everything else (fp8 KV view, bmm scales, tensor-core wrappers, cudagraph capture) is
inherited unchanged from the builtin.

CORRECTNESS NOTE on the ragged path: it is gated OFF by default and restricted to
pure-prefill (num_decodes == 0), non-cascade, non-TRTLLM, non-nvfp4, FI-native batches;
anything else falls back to the base ``forward``. Its failure mode is silent output
corruption, not a crash, and it cannot be byte-validated off-GPU -- validate on the H200
before enabling. The index math (prefix-only block layout) is unit-tested in
``vtl/patches/flashinfer_backend.py``; every kernel call mirrors the in-tree
``BatchDCPPrefillWrapper`` reference.

This module imports vLLM at top level on purpose: it is only ever imported inside the
vLLM process, lazily, when the registry resolves the class path.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
import torch
from vllm import envs
from vllm.v1.attention.backends.flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
    FIPrefill,
    FlashInferBackend,
    FlashInferImpl,
    FlashInferMetadataBuilder,
    _copy_page_indices_kernel,
)
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

from vtl.patches.flashinfer_backend import context_indptr_and_lastlen

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


@dataclass
class _VtlRaggedPrefill:
    """Wrappers for the ragged-prefill split, attached to FlashInferMetadata.vtl_ragged."""

    new_wrapper: BatchPrefillWithRaggedKVCacheWrapper  # new-token self attn (causal)
    ctx_wrapper: BatchPrefillWithPagedKVCacheWrapper | None  # prefix context (non-causal)


class VtlFlashInferMetadataBuilder(FlashInferMetadataBuilder):
    _vtl_new_wrapper: BatchPrefillWithRaggedKVCacheWrapper | None = None
    _vtl_ctx_wrapper: BatchPrefillWithPagedKVCacheWrapper | None = None

    def _get_workspace_buffer(self) -> torch.Tensor:
        # Mirror the base allocation (flashinfer.py:_get_workspace_buffer) but floor the
        # size. The base caches on self._workspace_buffer, so only the first call sizes it.
        if self._workspace_buffer is None:
            size = max(int(envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE), _workspace_floor_bytes())
            self._workspace_buffer = torch.zeros(size, dtype=torch.uint8, device=self.device)
        return self._workspace_buffer

    def _ragged_wrappers(self):
        ws = self._get_workspace_buffer()
        if self._vtl_new_wrapper is None:
            self._vtl_new_wrapper = BatchPrefillWithRaggedKVCacheWrapper(ws)
        if self._vtl_ctx_wrapper is None:
            self._vtl_ctx_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                ws, get_kv_cache_layout()
            )
        return self._vtl_new_wrapper, self._vtl_ctx_wrapper

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        md = super().build(common_prefix_len, common_attn_metadata, fast_build)
        md.vtl_ragged = None
        if not _env_on("VTL_FI_RAGGED_PREFILL"):
            return md
        try:
            self._plan_ragged(md, common_prefix_len, common_attn_metadata)
        except Exception:
            log.exception("vtl: ragged-prefill planning failed; using base paged prefill")
            md.vtl_ragged = None
        return md

    def _plan_ragged(self, md, common_prefix_len, common_attn_metadata) -> None:
        # Only the pure-prefill FI-native path. Anything else -> base forward.
        if (
            md.num_prefills == 0
            or md.num_decodes != 0
            or common_prefix_len > 0
            or getattr(md, "use_cascade", False)
            or getattr(self, "is_kvcache_nvfp4", False)
            or not isinstance(md.prefill, FIPrefill)
            or not isinstance(md.prefill.wrapper, BatchPrefillWithPagedKVCacheWrapper)
        ):
            return

        qo_indptr_cpu = common_attn_metadata.query_start_loc_cpu.to(torch.int32)
        seq_lens_np = common_attn_metadata.seq_lens_cpu.numpy()
        query_lens_np = np.diff(qo_indptr_cpu.numpy())
        num_reqs = md.num_prefills

        new_wrapper, ctx_wrapper = self._ragged_wrappers()

        # New-token self-attention: ragged, causal, KV == the new tokens (qo == kv indptr).
        new_wrapper.plan(
            qo_indptr=qo_indptr_cpu,
            kv_indptr=qo_indptr_cpu,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            head_dim_vo=self.head_dim,
            causal=True,
            sm_scale=self.sm_scale,
            window_left=self.window_left,
            logits_soft_cap=self.logits_soft_cap,
            q_data_type=self.q_data_type,
        )

        ctx_indptr_np, ctx_num_blocks_np, ctx_last_np = context_indptr_and_lastlen(
            seq_lens_np, query_lens_np, self.page_size
        )
        if int(ctx_indptr_np[-1]) == 0:
            # No cached prefix in this batch -> new-token self-attention alone is exact.
            md.vtl_ragged = _VtlRaggedPrefill(new_wrapper=new_wrapper, ctx_wrapper=None)
            return

        # Prefix-context attention: paged, non-causal, PREFIX-ONLY blocks. Reuse the base's
        # Triton scatter to fill block indices on device (vectorized, no Python loop).
        device = self.device
        ctx_indptr_gpu = torch.from_numpy(ctx_indptr_np).to(device, non_blocking=True)
        ctx_indices = torch.empty(
            int(ctx_indptr_np[-1]), dtype=torch.int32, device=device
        )
        block_table = common_attn_metadata.block_table_tensor
        _copy_page_indices_kernel[(num_reqs,)](
            ctx_indices,
            block_table,
            block_table.stride(0),
            ctx_indptr_gpu,
            BLOCK_SIZE=1024,
        )
        ctx_wrapper.plan(
            qo_indptr=qo_indptr_cpu,
            paged_kv_indptr=torch.from_numpy(ctx_indptr_np),
            paged_kv_indices=ctx_indices,
            paged_kv_last_page_len=torch.from_numpy(ctx_last_np),
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            page_size=self.page_size,
            causal=False,
            sm_scale=self.sm_scale,
            window_left=self.window_left,
            logits_soft_cap=self.logits_soft_cap,
            q_data_type=self.q_data_type,
            kv_data_type=self.kv_cache_dtype,
        )
        md.vtl_ragged = _VtlRaggedPrefill(new_wrapper=new_wrapper, ctx_wrapper=ctx_wrapper)


class VtlFlashInferImpl(FlashInferImpl):
    def _fp8_kv_view(self, kv_cache: torch.Tensor) -> torch.Tensor:
        if kv_cache.dtype != torch.uint8:
            return kv_cache
        if self.kv_cache_dtype in ("fp8", "fp8_e4m3", torch.float8_e4m3fn):
            return kv_cache.view(torch.float8_e4m3fn)
        if self.kv_cache_dtype in ("fp8_e5m2", torch.float8_e5m2):
            return kv_cache.view(torch.float8_e5m2)
        return kv_cache

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        ragged = None if attn_metadata is None else getattr(attn_metadata, "vtl_ragged", None)
        # output_scale means attn+quant fusion (TRTLLM only) -> never our pure-prefill path.
        if ragged is None or output_scale is not None:
            return super().forward(
                layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale, output_block_scale,
            )

        num_actual = attn_metadata.num_actual_tokens
        kv_cache = self._fp8_kv_view(kv_cache)
        q = query[:num_actual]
        k = key[:num_actual]
        v = value[:num_actual]
        out = output[:num_actual]

        # New-token self-attention over the contiguous (ragged) new K/V. Mirrors the
        # BatchDCPPrefillWrapper new-tokens run: return_lse, transpose lse to (heads, tokens).
        new_out, new_lse = ragged.new_wrapper.run(q, k, v, return_lse=True)
        if ragged.ctx_wrapper is None:
            # new_out is [tokens, heads, head_size]; out may be that or flattened -> reshape.
            out.copy_(new_out.reshape(out.shape))
            return output

        stride_order = FlashInferBackend.get_kv_cache_stride_order()
        kv_cache_permute = kv_cache.permute(*stride_order)
        ctx_out, ctx_lse = ragged.ctx_wrapper.run(
            q,
            kv_cache_permute,
            k_scale=layer._k_scale_float,
            v_scale=layer._v_scale_float,
            return_lse=True,
        )
        merge_attn_states(
            out,
            ctx_out,
            ctx_lse.transpose(0, 1).contiguous(),
            new_out,
            new_lse.transpose(0, 1).contiguous(),
        )
        return output


class VtlFlashInferBackend(FlashInferBackend):
    # Redirect builder + impl to our subclasses; everything else (supported_kv_cache_dtypes
    # incl. fp8_e4m3, head sizes [64,128,256,512], supports_compute_capability (7,5)..(12,1),
    # get_kv_cache_shape, forward_includes_kv_cache_update=False) is inherited and correct.
    @staticmethod
    def get_builder_cls() -> type[VtlFlashInferMetadataBuilder]:
        return VtlFlashInferMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[VtlFlashInferImpl]:
        return VtlFlashInferImpl
