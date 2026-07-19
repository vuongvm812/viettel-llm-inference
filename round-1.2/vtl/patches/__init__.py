"""Patch modules. Importing this package registers every patch it can load.

A module that fails to import (missing vLLM symbol after a version bump, absent
optional dep) is logged and skipped, leaving the rest of the patch set intact.
"""

from __future__ import annotations

import importlib
import logging

log = logging.getLogger("vtl")

# Patch modules, in apply order. Add names here as they land.
# The served model is LFM2.5-1.2B (Lfm2ForCausalLM): 16 layers = 10 short-conv + 6 GQA attention,
# no GDN / gated-delta-net, no attn output gate, no MTP, no vision. The old GDN family
# (gdn_kernels, gdn_prefill_backend) and the attn sigmoid-gate fusion (mul_sigmoid_quant) were
# removed -- they targeted the previous qwen3_5 VL hybrid and are dead here. No conv-gate fusion:
# LFM2's short-conv in_proj/out_proj are bf16 (built without a quant_config), so the `C * Bx` gate
# feeds a bf16 matmul and there is no fp8 quant to fuse into. The remaining patches are all
# model-agnostic and cover LFM2's fp8 paths (attn qkv/out_proj, MLP w13/w2) directly.
_MODULES: tuple[str, ...] = (
    "quant_fp8",
    "rms_norm_quant",
    "dynamic_per_token_quant",
    "silu_mul_quant",
    "kv_cache_manager",
    "radix_cache",      # shadow radix tree (SGLang RadixAttention port), default-off A/B harness
    "sched_policy",
    "msgspec_stream",   # dict+msgspec SSE for simple chat streams (serving-path TPOT)
    "msgspec_json",     # msgspec request-body decode + non-streaming JSON response encode
    "greedy_sampler",   # argmax fast path for plain greedy steps (per-step TPOT)
    "profiler",
)

for _name in _MODULES:
    try:
        importlib.import_module(f"{__name__}.{_name}")
    except Exception:
        log.exception("vtl: patch module %s failed to import, skipping", _name)
