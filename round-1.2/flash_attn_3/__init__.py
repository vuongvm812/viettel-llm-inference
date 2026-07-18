"""Vendored official FlashAttention-3 (Dao-AILab hopper/) built as `flash_attn_3._C`.

The CUDA sources live in ``csrc/`` and are compiled by ``round-1.2/setup.py`` into
the ``flash_attn_3._C`` extension (sm_90a, pruned to LFM2's head_dim=64 bf16/fp8
forward paths). This package exposes the paged-KV decode entry point used by the
``fa3_attention`` vtl patch.

Importing ``flash_attn_3._C`` registers the ``torch.ops.flash_attn_3`` operators.
"""

from flash_attn_3.flash_attn_interface import (  # noqa: F401
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)

__version__ = "3.0.0-vtl-vendored"
