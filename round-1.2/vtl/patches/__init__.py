"""Patch modules. Importing this package registers every patch it can load.

A module that fails to import (missing vLLM symbol after a version bump, absent
optional dep) is logged and skipped, leaving the rest of the patch set intact.
"""

from __future__ import annotations

import importlib
import logging

log = logging.getLogger("vllm.vtl.patches")

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
    # WS4: must come AFTER kv_cache_manager (it subclasses whatever is bound at apply
    # time, so the plan_request/free_blocks signals survive) and AFTER sched_policy
    # (VTL_RUST_SCHED_FULL supersedes its schedule() wrapper, folding the same SJF key
    # into the Rust loop). Off unless VTL_ENABLE_RUST_SCHED=1 *and* a mode flag is set.
    "rust_sched",
    "msgspec_stream",   # dict+msgspec SSE for simple chat streams (serving-path TPOT)
    "msgspec_json",     # msgspec request-body decode + non-streaming JSON response encode
    "greedy_sampler",   # argmax fast path for plain greedy steps (per-step TPOT)
    "step0_eos_ban",    # mask EOS out of each request's FIRST sampled token (the
                        # 8-failures fix: int4 lm_head argmax -> <|im_end|> at step 0
                        # -> empty stream -> unscoreable). V2 sampler seam.
    "decode_fastpath",  # V2 runner: skip the dead metadata build on repeat pure-decode steps,
                        # + pooled pinned staging instead of pin_memory-per-step
    # AFTER decode_fastpath: nstep reuses its `_fa_write` helper and its builder
    # classification rather than copying either, and its readiness probe reads
    # decode_fastpath's `_C.builders`. The two wrap DISJOINT seams -- decode_fastpath owns
    # prepare_inputs / prepare_attn, nstep owns execute_model / sample_tokens / capture_model
    # -- so neither stacks on the other and the order is about imports, not wrappers.
    "nstep_decode",     # N decode iterations per engine step (armed by rust_sched's commit)
    "conv_align_fused",  # registers vllm_cuda::conv_align_fused; the call site is in the fork
    "shortconv_mega",    # registers vllm_cuda::shortconv_decode_mega + its persistent scratch
    "shm_ipc",          # iceoryx2 shm data plane for the frontend<->EngineCore hop (VTL_SHM_IPC=1)
    "profiler",
    "l2_persist",       # boot probe of the MIG L2 set-aside (+ opt-in persisting window).
                        # LAST on purpose: it stacks a second wrapper on
                        # BaseModelLoader.load_model, and quant_w4a8's summary hook must be
                        # the inner one so the w4a8 tallies are already final when we read
                        # weight_packed.
    "megakernel_probe",  # read-only go/no-go for a cooperative-grid decode megakernel. Same
                         # load_model seam as the two above, so it goes after them; it only
                         # reads device attributes, so the order between them does not matter.
)

for _name in _MODULES:
    try:
        importlib.import_module(f"{__name__}.{_name}")
    except Exception:
        log.exception("vtl: patch module %s failed to import, skipping", _name)
