"""Patch modules. Importing this package registers every patch it can load.

A module that fails to import (missing vLLM symbol after a version bump, absent
optional dep) is logged and skipped, leaving the rest of the patch set intact.
"""

from __future__ import annotations

import importlib
import logging

log = logging.getLogger("vllm.vtl.patches")

# Patch modules, in apply order. Add names here as they land.
#
# EVERY MODULE HERE IS MODEL-AGNOSTIC. That is the invariant this package now keeps: each one
# either matches on a vLLM class/graph pattern or reads the loaded config at runtime, so the set
# applies unchanged to whatever model gets mounted at /model. Anything that needed a specific
# architecture's block structure lives in ../../reference/, out of the build and out of this list.
#
# Coverage of the big bf16 tensors: `quant_fp8` and `quant_w4a8` reach every LinearBase through
# vLLM's quant-config seam; `lm_head_quant` covers the output head, which is NOT a LinearBase and
# so never reaches those configs at all. `rms_norm_quant` / `silu_mul_quant` /
# `dynamic_per_token_quant` are torch.compile fusion passes -- they rewrite norm->quant and
# act->quant pairs wherever the graph contains them, and a model whose graph has no match is a
# silent no-op rather than an error.
#
# ADDING A MODEL-SPECIFIC KERNEL: put the patch module here, register the op in
# vtl/csrc/torch_bindings.cpp (AOT) or drop a .cu in vtl/kernels/ (NVRTC, no rebuild), and gate it
# default-off via @register_patch so a model it does not fit degrades to stock instead of failing.
_MODULES: tuple[str, ...] = (
    "quant_fp8",
    "quant_w4a8",       # int4 weights + fp8 acts; delegates un-int4-able layers back to quant_fp8
    # AFTER quant_w4a8: imports its packing helpers (pack_int4_rows, CUTLASS encode) and extends
    # the method to fp8-block checkpoints (dequant -> RTN int4 group-128). VTL_W4A8_FROM_FP8=1.
    "w4a8_from_fp8",
    "rms_norm_quant",
    # AFTER rms_norm_quant (HANDOFF 4.1): both register fusion patterns into the same
    # RMSNormQuantFusionPass; gdn_kernels adds the RMSNormGated -> group-128-quant pattern
    # (the fusion AITER ships on ROCm and CUDA lacks) for the 36 GDN layers.
    "gdn_kernels",
    "gdn_prefill_backend",  # env pin of the GDN prefill backend (triton/flashinfer/cutedsl A/B)
    # NVRTC re-specialization of the stock block-quant ops at the loaded model's geometry
    # (-DHIDDEN/-DGROUP). Op-identity-preserving: leaves the stock op untouched when NVRTC
    # is off or the compile fails. First production consumer of vtl/nvrtc.py.
    "nvrtc_block_quant",
    # Fused greedy argmax over the vocab, -DVOCAB-specialized. Fills the seams the forked V2
    # sampler and nstep_decode already have (torch.ops.vllm_cuda.greedy_argmax_i64 and
    # nstep's `_ARGMAX`); registers nothing unless the compile succeeds, so both stay on
    # torch.argmax otherwise. Needs VTL_NVRTC=1.
    "greedy_argmax",
    # Fused GDN decode step (conv1d update + gating delta rule, one NVRTC launch per layer,
    # pure non-spec decode only; spec/MTP and prefill fall through to stock Triton).
    "gdn_decode_step",
    # MoE decode grouped-GEMV for the 256-expert layers at M<=VTL_MOE_GEMV_MAX_M; fp8 arm is
    # memory-neutral, int4 arm rides w4a8_from_fp8's packing. Needs VTL_NVRTC=1.
    "moe_decode_gemv",
    "dynamic_per_token_quant",
    "silu_mul_quant",
    "kv_cache_manager",
    "sched_policy",
    # WS4: must come AFTER kv_cache_manager (it subclasses whatever is bound at apply
    # time, so the plan_request/free_blocks signals survive) and AFTER sched_policy
    # (VTL_RUST_SCHED_FULL supersedes its schedule() wrapper, folding the same SJF key
    # into the Rust loop). Off unless VTL_ENABLE_RUST_SCHED=1 *and* a mode flag is set.
    "rust_sched",
    # NOTE: no HTTP-layer patches here. The Rust frontend owns the socket
    # (api_server_rust_frontend.patch routes the entrypoint into
    # run_multi_api_server unconditionally), so starlette/FastAPI/OpenAIServing*
    # objects are never constructed in this process. `msgspec_stream` and
    # `msgspec_json` used to sit here and patched exactly those -- dead weight
    # once the Python api_server stopped starting. Per-token SSE now comes from
    # vtl/vllm_patches/rust-frontend/per_token_stream.patch (VTL_STREAM_PER_TOKEN)
    # and body decode from encode_cache.patch. Anything HTTP-shaped belongs there.
    "greedy_sampler",   # argmax fast path for plain greedy steps (per-step TPOT)
    "step0_eos_ban",    # mask EOS out of each request's FIRST sampled token. Guards a real
                        # failure mode of int4 lm_head: argmax picks the EOS token at step 0
                        # -> empty stream -> the request is unscoreable. V2 sampler seam.
    "decode_fastpath",  # V2 runner: skip the dead metadata build on repeat pure-decode steps,
                        # + pooled pinned staging instead of pin_memory-per-step
    # AFTER decode_fastpath: nstep reuses its `_fa_write` helper and its builder
    # classification rather than copying either, and its readiness probe reads
    # decode_fastpath's `_C.builders`. The two wrap DISJOINT seams -- decode_fastpath owns
    # prepare_inputs / prepare_attn, nstep owns execute_model / sample_tokens / capture_model
    # -- so neither stacks on the other and the order is about imports, not wrappers.
    "nstep_decode",     # N decode iterations per engine step (armed by rust_sched's commit)
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
