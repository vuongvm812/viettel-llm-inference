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
    "rms_norm_quant",
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
