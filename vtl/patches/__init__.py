"""Patch modules. Importing this package registers every patch it can load.

A module that fails to import (missing vLLM symbol after a version bump, absent
optional dep) is logged and skipped, leaving the rest of the patch set intact.
"""

from __future__ import annotations

import importlib
import logging

log = logging.getLogger("vtl")

# Patch modules, in apply order. Add names here as they land.
# inputs_embeds_optional removed: it dropped the inputs_embeds buffer on the assumption of a
# dense text-only model; the served model is a qwen3_5 VL hybrid (supports_mm_inputs=True),
# so the buffer is live and the patch was a dead no-op.
_MODULES: tuple[str, ...] = (
    "quant_fp8",
    "rms_norm_quant",
    "dynamic_per_token_quant",
    "silu_mul_quant",
    "mul_sigmoid_quant",
    "gdn_kernels",
    "gdn_prefill_backend",
    "kv_cache_manager",
    "sched_policy",
)

for _name in _MODULES:
    try:
        importlib.import_module(f"{__name__}.{_name}")
    except Exception:
        log.exception("vtl: patch module %s failed to import, skipping", _name)
