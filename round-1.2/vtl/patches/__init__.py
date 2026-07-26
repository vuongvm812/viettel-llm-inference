"""Patch modules. Importing this package registers every patch it can load.

A module that fails to import (missing vLLM symbol after a version bump, absent
optional dep) is logged and skipped, leaving the rest of the patch set intact.
"""

from __future__ import annotations

import importlib
import logging

log = logging.getLogger("vllm.vtl")

# Patch modules, in apply order. Add names here as they land.
# The served model is LFM2.5-1.2B (Lfm2ForCausalLM): 16 layers = 10 short-conv + 6 GQA attention,
# no GDN / gated-delta-net, no attn output gate, no MTP, no vision. The old GDN family
# (gdn_kernels, gdn_prefill_backend) and the attn sigmoid-gate fusion (mul_sigmoid_quant) were
# removed -- they targeted the previous qwen3_5 VL hybrid and are dead here. `shortconv_quant`
# rebuilds LFM2's short-conv in_proj/out_proj with the active quant_config (stock builds them
# bf16); the existing kernels then cover them -- operator_norm->in_proj via rms_norm_quant (which
# needed the in_proj hoist in short_conv.patch plus the empty_like fix in lfm2.patch before the
# fusion pass could see it at all), and out_proj's (non-norm) `C * Bx` input via the standalone
# dynamic_per_token_quant kernel. `lm_head_quant` covers the last big bf16 tensor, the output
# head, which is not a LinearBase and so never reached the quant configs at all. The
# depthwise conv weight stays bf16. `mul_quant` then fuses that gate multiply into the quant AND
# gives every short-conv layer a single fp8 staging buffer for out_proj (so decode, prefill and
# mixed batches all skip the bf16 vstack), and `bcx_conv_gate` replaces the decode half of that
# path -- B*x, causal_conv1d_update and the gate+quant -- with one kernel. The remaining patches
# are model-agnostic and cover LFM2's other fp8 paths (attn qkv/out_proj, MLP w13/w2) directly.
_MODULES: tuple[str, ...] = (
    "quant_fp8",
    "quant_w4a8",       # int4 weights + fp8 acts; delegates un-int4-able layers back to quant_fp8
    "shortconv_quant",  # quantize the short-conv in_proj/out_proj (reuses the kernels above)
    "rms_norm_quant",
    "dynamic_per_token_quant",
    "silu_mul_quant",
    "qk_norm_rope",     # fused QK-RMSNorm+RoPE on the 6 attn layers (stock kernel, unmatched pass)
    "mul_quant",        # conv-gate fused mul+fp8-quant op (opt-in; wired directly in short_conv)
    "bcx_conv_gate",    # whole short-conv DECODE block in one kernel (needs mul_quant's staging buf)
    "kv_cache_manager",
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
