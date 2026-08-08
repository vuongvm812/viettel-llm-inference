"""Rust KV-cache / scheduler core (`vtl_sched`), staged behind four env gates.

WS4. The crate in ``round-2/vtl-sched/`` is a logic-preserving port of vLLM v0.25.0's
``v1/core/`` block metadata, prefix cache, hybrid coordinator and ``schedule()`` decision
loop. This module is the only thing that wires it into a live engine.

Gates (all default OFF, all independent except where noted)::

    VTL_ENABLE_RUST_SCHED=1        install this patch at all (registry gate)
    VTL_RUST_SCHED=1               make Rust AUTHORITATIVE for the KVCacheManager surface
    VTL_RUST_SCHED_FULL=1          also run the Rust schedule() loop (implies VTL_RUST_SCHED)
    VTL_RUST_SCHED_RADIX=1         use the radix index instead of the flat hash map
                                   (same answers; see vtl-sched/src/radix.rs)
    VTL_RUST_SCHED_UFO=1           R6a: batch the per-step stop decision of
                                   update_from_output into ONE Rust call (needs _FULL)
    VTL_RUST_SCHED_TABLE=1         R6b: schedule from the Rust-RESIDENT request table
                                   instead of re-marshalling every running request each
                                   step (needs _FULL *and* _UFO -- the per-step token
                                   deltas ride on update_step)
    VTL_RUST_SCHED_SPEC=1          R6c: precompute the next step on the Rust worker
                                   thread between update_from_output and schedule
                                   (needs _TABLE)
    VTL_SCHED_TIMING=1             log-only p50/p95 of the schedule() phases. Independent
                                   of every other gate.
    VTL_RUST_SCHED_TOKSTORE=1      Port-2: Rust owns each slot's output tokens, counters and
                                   block-hash chain (needs R8 *and* VTL_RUST_HASHER). The
                                   sampled ids cross as numpy, Python's ``Request`` degrades
                                   to three int counters, and the rare paths that need the
                                   real lists materialize them back from the crate.

This replaces the most correctness-critical component in the engine, and a hybrid layout
(full-attention groups alongside mamba groups with ``mamba_cache_mode=align``) has
per-kind rules that fail SILENTLY when wrong -- a mamba state cached at the wrong boundary
produces wrong tokens, not an exception. Each rung above was therefore soaked against a
Python-authoritative mirror before it was flipped on. Those mirrors are gone now that the
parity work is done; ``bench/`` and the crate's own tests are what guard the port. Every
gate still fails CLOSED -- a refusal degrades to the Python scheduler rather than raising.

Composition with the existing patches:
  * ``vtl/patches/kv_cache_manager.py`` rebinds ``sched_mod.KVCacheManager`` to a subclass
    carrying ``plan_request`` / ``free_blocks``. We apply AFTER it and subclass whatever is
    bound at that moment, so those signals survive; in authority mode ``plan_request`` is
    re-pointed at the Rust cache-hit walk (the Python coordinator would be stale).
  * ``vtl/patches/sched_policy.py`` wraps ``Scheduler.schedule`` with the cache-aware SJF
    reorder. ``VTL_RUST_SCHED_FULL=1`` SUPERSEDES that wrapper -- the same SJF key runs
    inside the Rust loop (``sched.rs::reorder_waiting``) so the ordering is preserved, not
    dropped. Which of the two is active is logged at install.

Refusal, not approximation: the port covers exactly two KV cache-spec kinds, full attention
and mamba (``single_type.rs::Kind``). Anything else (KV/EC connectors, LoRA, encoder inputs,
speculative decoding, priority policy, sliding-window / chunked-local / cross-attention
specs, sparse prefix-cache retention, DCP/PCP, KV cache events) makes ``build_config`` /
``schedule_supported`` return a reason string, which is logged once and leaves stock vLLM
in charge.

    VTL_RUST_SCHED_REQUIRE=1  turn that refusal into a BOOT FAILURE.

MODEL-AGNOSTIC, WITH ONE SHARP EDGE. Nothing here reads a model name or a layer count --
the config is derived from ``kv_cache_config.kv_cache_groups`` at runtime, so a plain dense
model (one full-attention group, unitary path) works with no changes, and so does an
attention+align-mamba hybrid. But sliding-window attention is COMMON in current models, and
on such a model this entire port switches itself off behind one log line, which is
indistinguishable from "it ran and did not help" in every latency number. Set
VTL_RUST_SCHED_REQUIRE=1 in bench/CI so that case fails loudly; leave it unset in the
submission, where serving-but-slower beats not serving. Supporting a new kind means a new
``Kind`` variant in vtl-sched/src/single_type.rs plus its arm in ``build_config``.
"""

from __future__ import annotations

import logging
import os

from vtl.registry import already_patched, mark_patched, register_patch

# Must be a child of "vllm.vtl": a bare "vtl" logger's INFO records are dropped.
log = logging.getLogger("vllm.vtl.rust_sched")

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# RequestStatus values the Rust core needs (vllm/v1/request.py).
_ST_WAITING, _ST_RUNNING, _ST_PREEMPTED = 0, 1, 2


def env_on(name: str) -> bool:
    """Env gate parsing. Pure python, exercised by the self-check without the crate."""
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def refuse(reason: str) -> None:
    """Report a refusal to engage the Rust scheduler. Raises under VTL_RUST_SCHED_REQUIRE=1.

    WHY THIS EXISTS. The port covers exactly two KV cache-spec kinds (full attention and
    mamba). Anything else -- sliding-window, chunked-local, cross-attention -- makes
    build_config return a reason, and the default behaviour is to log it and leave stock
    vLLM in charge. That is the RIGHT default for a submission (a served-but-slower engine
    beats a dead one) and the WRONG one for measurement: a whole scheduler quietly not
    running looks, in every latency number, exactly like a scheduler that ran and did not
    help. So bench/CI set VTL_RUST_SCHED_REQUIRE=1 and get a boot failure instead.

    Adding a spec kind means adding a `Kind` variant in vtl-sched/src/single_type.rs AND
    the matching arm in build_config -- not just silencing this.
    """
    if env_on("VTL_RUST_SCHED_REQUIRE"):
        raise RuntimeError(
            f"VTL_RUST_SCHED_REQUIRE=1 but the Rust scheduler cannot engage: {reason}"
        )
    log.warning("rust_sched: NOT ENGAGED -- %s", reason)


def reraise_fatal(exc: BaseException) -> None:
    """Re-raise the two exceptions a guard must never swallow.

    Every guard below catches ``BaseException``, not ``Exception``: a panic inside the
    crate surfaces as ``pyo3_runtime.PanicException``, which derives from BaseException,
    so ``except Exception`` would let a Rust ``assert!`` take down EngineCore. Ctrl-C and
    ``SystemExit`` are the two that must still get through.
    """
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        raise exc


def kv_transfer_configured():
    """``True``/``False`` if vLLM's ambient config can be read, ``None`` if it cannot.

    A KV connector is fatal for authority mode (see ``VtlRustKVCacheManager.__init__``),
    but v0.25.0 constructs the Scheduler outside any ``set_current_vllm_config`` context,
    so this answers ``None`` more often than not. It is the cheap outer layer; the hard
    one is ``pop_blocks_for_free`` refusing outright.
    """
    try:
        from vllm.config import get_current_vllm_config_or_none

        cfg = get_current_vllm_config_or_none()
    except Exception:
        return None
    if cfg is None:
        return None
    return getattr(cfg, "kv_transfer_config", None) is not None


def modes() -> dict:
    """Resolve the gate combination. ``full`` implies ``authority``."""
    full = env_on("VTL_RUST_SCHED_FULL")
    # R6a rides on the full loop: it needs the Rust manager's slot interning, and
    # nothing else in the engine would be Rust-backed without it.
    ufo = full and env_on("VTL_RUST_SCHED_UFO")
    # R6b rides on R6a: the resident table's per-step token delta is applied by
    # `update_step`, so without UFO the table would go stale on the very first decode.
    table = ufo and env_on("VTL_RUST_SCHED_TABLE")
    r8 = (
        ufo
        and env_on("VTL_RUST_SCHED_R8")
        and env_on("VTL_SHM_IPC")
        and env_on("VTL_SHM_IPC_RAW")
    )
    # Port-2: Rust owns the per-slot token list, counters and block-hash chain. Rides on
    # R8 (it IS `update_step_pack`, driven off the sampler's numpy array) and on the Rust
    # hasher -- with two hash implementations live, one would own the prompt blocks and
    # the other the decode blocks, and a divergence between them is a silent prefix-cache
    # key-space fork, not an exception.
    tokstore = (
        r8
        and env_on("VTL_RUST_HASHER")
        and env_on("VTL_RUST_SCHED_TOKSTORE")
    )
    return {
        "authority": env_on("VTL_RUST_SCHED") or full,
        "full": full,
        "radix": env_on("VTL_RUST_SCHED_RADIX"),
        "ufo": ufo,
        "table": table,
        # Speculation can only be consumed by the resident fast path, so it rides on it.
        "spec": table and env_on("VTL_RUST_SCHED_SPEC"),
        "timing": env_on("VTL_SCHED_TIMING"),
        # R8 rides on UFO (it IS update_step, plus the pack) and on the shm raw record
        # being the live wire format -- there is no point building bytes the output
        # thread would have to decode again.
        "r8": r8,
        # B3: block hashing in Rust. Independent of everything above -- it only needs the
        # extension importable, so it stays armable with the scheduler flags off.
        "hasher": env_on("VTL_RUST_HASHER"),
        # C1a: N-step decode burst commitment. Needs the full Rust loop (the commit
        # extends the resident table in the same place the schedule decisions land).
        "nstep": full and env_on("VTL_NSTEP"),
        # The Rust steady-state runner's SCHEDULER half: stash the step layout at commit,
        # apply the crate's verdicts instead of running decide(). Rides on the burst commit
        # (the runner launches the burst's unroll graph) and on R9 (the runner drives
        # `step_pack_locked`, whose output only the R9 residue loop knows how to apply).
        # `VTL_RUST_RUNNER` is NOT read here -- it defaults ON, which `env_on` cannot
        # express; the install resolves it through `rust_runner.mode()`.
        "runner": full and env_on("VTL_NSTEP") and tokstore
                  and env_on("VTL_RUST_SCHED_R9"),
        "tokstore": tokstore,
        # R9: collapse decide()/r8_apply's per-request Python loops into the single
        # update_step_pack_np crossing (+ the crate-side cache_blocks fold). Needs the
        # token store live -- the residue loop's counter writes and the fold's
        # `num_computed - num_output_placeholders` math both assume Rust already owns the
        # per-slot bookkeeping tokstore provides.
        "r9": tokstore and env_on("VTL_RUST_SCHED_R9"),
        # Phase A: stop producing decision payload no consumer on THIS boot reads
        # (`num_common_prefix_blocks` in the crate, the four dead `CachedRequestData`
        # fields in Python). Rides on the full loop -- it is that loop's apply block.
        # The env gate is necessary but not sufficient: `lean_blocked` re-checks the
        # runtime config at first schedule and refuses if any consumer is live.
        "lean": full and env_on("VTL_RUST_SCHED_LEAN"),
        "lean_check": env_on("VTL_SCHED_LEAN_CHECK"),
        # Phase B: the decisions arrive in persistent numpy buffers instead of a per-step
        # PyDict. Rides on `lean` -- the arena carries no `num_common_prefix_blocks` slot,
        # which is only safe on exactly the boots `lean_blocked` clears.
        "arena": (
            full
            and env_on("VTL_RUST_SCHED_LEAN")
            and _arena_env() in ("1", "check")
        ),
        "arena_check": _arena_env() == "check",
        # Phase C: re-serve a prebuilt SchedulerOutput on a pure-decode step. Needs the
        # full loop (it IS that loop's tail) plus the runtime checks in `ring_blocked`.
        "so_ring": full and env_on("VTL_SCHED_SO_RING"),
    }


def _arena_env() -> str:
    """``VTL_SCHED_DECISIONS_ARENA``: ``0``/unset off, ``1`` on, ``check`` = dual-run."""
    raw = os.environ.get("VTL_SCHED_DECISIONS_ARENA", "").strip().lower()
    if raw == "check":
        return "check"
    return "1" if raw in _TRUTHY else "0"


# Shared across every lean step: no V2 consumer reads `num_common_prefix_blocks` at all
# (only `warmup.py` writes its own), so handing out the same list cannot be observed.
_ZERO_COMMON: dict[int, list] = {}


def lean_cached_request_data(cls, running_reqs, req_to_new_blocks):
    """``Scheduler._make_cached_request_data`` with every field ``lean_blocked`` proved
    dead left empty. ``cls`` is passed in (not imported) so this stays testable, and this
    module importable, without vLLM.

    Resumed requests are not a case here: the caller has already folded them into
    ``scheduled_new_reqs`` (that is what ``use_v2_model_runner`` does), so the stock
    ``resumed_req_ids`` set would be empty too.
    """
    req_ids = []
    new_block_ids = []
    num_computed = []
    for r in running_reqs:
        rid = r.request_id
        req_ids.append(rid)
        new_block_ids.append(req_to_new_blocks[rid].get_block_ids(allow_none=True))
        num_computed.append(r.num_computed_tokens)
    return cls(
        req_ids=req_ids,
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=new_block_ids,
        num_computed_tokens=num_computed,
        num_output_tokens=[],
    )


def lean_cached_check(lean, stock) -> str | None:
    """``VTL_SCHED_LEAN_CHECK``: first divergence between the lean and stock payloads, or
    None. Pure -- self-checked.

    Two halves. The live fields must match exactly. The dead fields must be empty in the
    STOCK output, which is the per-step runtime proof of ``lean_blocked``'s static
    predicate -- if a boot ever fills one of them, the predicate missed a consumer.
    (``num_output_tokens`` is deliberately absent from both lists: stock always fills it
    and nothing on an eligible boot reads it.)
    """
    for name in ("req_ids", "new_block_ids", "num_computed_tokens"):
        mine, theirs = getattr(lean, name), getattr(stock, name)
        if mine != theirs:
            return f"{name}: lean {mine!r} vs stock {theirs!r}"
    for name in ("resumed_req_ids", "new_token_ids", "all_token_ids"):
        theirs = getattr(stock, name)
        if theirs:
            return f"{name} is non-empty in the stock payload: {theirs!r}"
    return None


def ring_blocked(scheduler) -> str | None:
    """Reason ``VTL_SCHED_SO_RING`` cannot engage on this boot, or None.

    The ring hands the SAME ``SchedulerOutput`` object back every other step, so the two
    things that must hold are (a) the ring is exactly as deep as the async batch queue, or
    a slot could be mutated while its step is still in flight, and (b) nothing downstream
    keeps a live reference into it -- with the `mp` executor the worker only ever sees a
    pickled COPY, which is what makes the reuse invisible. `uni` hands the runner the very
    object the next step mutates, so it is refused.
    """
    cfg = getattr(scheduler, "vllm_config", None)
    depth = getattr(cfg, "max_concurrent_batches", 0)
    if depth != 2:
        return f"batch queue depth is {depth}, the ring is 2 slots"
    backend = getattr(getattr(cfg, "parallel_config", None), "distributed_executor_backend", None)
    if backend != "mp":
        return f"executor backend {backend!r} may alias the reused SchedulerOutput"
    if not getattr(scheduler, "use_v2_model_runner", False):
        return "V1 model runner"
    if getattr(scheduler, "needs_kv_cache_zeroing", False):
        # `take_new_block_ids()` is a drain, so the fast path would have to call it anyway.
        return "KV cache zeroing needs a fresh new_block_ids_to_zero every step"
    if getattr(scheduler, "enable_return_routed_experts", False):
        return "routed-expert block snapshots are rebuilt every step"
    if not isinstance(getattr(scheduler.encoder_cache_manager, "freed", None), list):
        # The fast path must PEEK at the pending frees; `get_freed_mm_hashes()` drains and
        # would drop them on a step that then refuses the reuse.
        return "encoder cache manager exposes no peekable `freed` list"
    # The ring replays `scheduled_cached_reqs` verbatim, so every CachedRequestData field
    # that is per-step-fresh in stock (`new_token_ids` under PP, `num_output_tokens` under
    # iteration-details logging) must be provably dead -- exactly `lean_blocked`'s
    # predicate, whether or not LEAN itself is enabled. The identity check in `ring_reuse`
    # additionally leans on `schedule_supported` refusing KV connectors: without a
    # connector, `_free_request` always `del self.requests[...]` at finish, which is what
    # makes registry identity authoritative for slot recycling.
    why = lean_blocked(scheduler)
    if why:
        return why
    return None


def ring_reuse(scheduler, slot, sched_running) -> object | None:
    """Phase C: re-serve a prebuilt ``SchedulerOutput``, or None to build a fresh one.

    ``slot`` is ``(scheduler_output, slots, requests)`` as populated two steps ago -- the
    async batch queue holds two in flight, so the slot at this parity has been consumed.

    NOTHING is mutated until the whole predicate has passed; a bail after a partial pass
    would leave half the batch advanced. The five known hazards, in order:

    1. ``finished_req_ids`` / ``preempted_req_ids`` are the slot's OWN sets and stay empty
       (the caller only gets here when the scheduler's are empty), so stock's rebind of
       `self.finished_req_ids` has nothing to protect -- but re-check them, because a
       downstream mutation would otherwise persist into every later step.
    2. ``CachedRequestData._req_id_to_num_output_tokens`` is a ``cached_property`` sized to
       the payload it was first read with -- drop it.
    3. ``has_structured_output_requests`` is ``|=`` in stock, which on a reused object
       would latch; recompute it from scratch.
    4. ``vtl_burst_n`` / ``vtl_sample_in_graph`` are injected by ``commit_burst`` AFTER
       this returns; a stale one would make the runner sample in-graph on a step that
       never committed to it.
    5. ``decode_fastpath``'s ``decode_key`` / ``has_new_blocks`` read plain fields, which
       are coherent by construction once the above hold.
    """
    so, slots, requests = slot
    if len(sched_running) != len(slots):
        return None
    live = scheduler.requests
    for i, (s, num_new) in enumerate(sched_running):
        if num_new != 1 or s != slots[i]:
            return None
        # Slot numbers recycle: request A finishes, request B interns the same slot, and
        # after one clean intervening step every cheap count is zero again -- the slot
        # match alone would replay A's output for B. Identity against the live registry
        # is the check `_free_request`'s `del self.requests[...]` makes authoritative.
        r = requests[i]
        if live.get(r.request_id) is not r:
            return None
    if so.finished_req_ids or so.preempted_req_ids:
        return None
    crd = so.scheduled_cached_reqs
    nc = crd.num_computed_tokens
    structured = False
    defer = scheduler.defer_block_free
    seq = scheduler.sched_step_seq + 1
    for i, request in enumerate(requests):
        # `_make_cached_request_data` reads this BEFORE `_update_after_schedule` advances
        # it, so the payload carries the pre-advance value -- and reading it off the
        # request (rather than `+= 1`) survives a burst that gave tokens back.
        nc[i] = request.num_computed_tokens
        request.num_computed_tokens += 1
        if defer:
            request.last_sched_seq = seq
        request.is_prefill_chunk = request.num_computed_tokens < (
            request.num_tokens + request.num_output_placeholders
        )
        # `_update_after_schedule`'s tail, verbatim -- it only ever DISCARDS here; the add
        # belongs to the admission loop, which by construction did not run this step.
        if not request.is_prefill_chunk:
            scheduler._inflight_prefills.discard(request)
            structured |= request.use_structured_output
    crd.__dict__.pop("_req_id_to_num_output_tokens", None)
    so.has_structured_output_requests = structured
    d = so.__dict__
    d.pop("vtl_burst_n", None)
    d.pop("vtl_sample_in_graph", None)
    if defer:
        scheduler.sched_step_seq = seq
    return so


def arena_check(dict_decisions, running, admitted, blocks, lens, preempted, waiting) -> str | None:
    """``VTL_SCHED_DECISIONS_ARENA=check``: first field where the arena marshalling
    disagrees with the PyDict one, or None. Pure -- self-checked.

    Both come from the SAME `Decisions` (the crate builds them in one call), so any
    difference is a marshalling bug, never a scheduling one.
    """
    for name, mine, theirs in (
        ("scheduled_running", running, dict_decisions["scheduled_running"]),
        ("scheduled_admitted", admitted, dict_decisions["scheduled_admitted"]),
        ("running_new_blocks", blocks, dict_decisions["running_new_blocks"]),
        ("running_new_lens", lens, dict_decisions["running_new_lens"]),
        ("preempted", preempted, dict_decisions["preempted"]),
        ("waiting_order", waiting, dict_decisions["waiting_order"]),
    ):
        # The dict path hands back tuples where the arena hands back tuples too (`zip`),
        # so a plain list compare is exact; `list()` normalizes the dict's own container.
        if list(mine) != list(theirs):
            return f"{name}: arena {list(mine)!r} vs dict {list(theirs)!r}"
    return None


def lean_blocked(scheduler) -> str | None:
    """Reason ``VTL_RUST_SCHED_LEAN`` cannot engage on this boot, or None.

    Each clause names the field the lean path would stop producing and the consumer that
    would then read a wrong value:

    * V1 model runner -- reads `num_common_prefix_blocks` for cascade attention
      (gpu_model_runner.py:2600) and gets `all_token_ids` from the scheduler
      (scheduler.py:1294). The V2 runner reads neither.
    * PP without async scheduling -- the only producer of `new_token_ids`
      (scheduler.py:1279).
    * iteration-detail logging / the torch profiler -- the only callers of
      `compute_iteration_details`, whose `is_context_phase` is the only reader of
      `CachedRequestData.num_output_tokens` (utils.py:813, core.py:448,
      gpu_worker.py:929).

    NOTE (deviation from the plan): `num_output_tokens` is gated on
    `enable_logging_iteration_details` + the profiler, NOT on `log_stats` -- serving runs
    with `log_stats=True`, so keying on it would leave the lean path permanently off.
    """
    if not getattr(scheduler, "use_v2_model_runner", False):
        return "V1 model runner reads num_common_prefix_blocks and all_token_ids"
    cfg = getattr(scheduler, "scheduler_config", None)
    if getattr(scheduler, "use_pp", False) and not getattr(cfg, "async_scheduling", False):
        return "PP without async scheduling needs new_token_ids"
    obs = getattr(scheduler, "observability_config", None)
    if getattr(obs, "enable_logging_iteration_details", False):
        return "iteration-detail logging reads is_context_phase"
    if os.environ.get("VLLM_TORCH_PROFILER_DIR", "").strip():
        return "torch profiler annotations read is_context_phase"
    return None


# --------------------------------------------------------------------------
# config extraction
# --------------------------------------------------------------------------


def spec_signature(spec) -> str:
    """Canonical rendering of a KVCacheSpec, standing in for upstream's ``spec == spec``.

    ``KVCacheSpec`` subclasses are dataclasses, so equal field values == equal specs. We
    render type name + sorted fields; two groups get merged into one attention group iff
    their signatures match, exactly as ``verify_and_split_kv_cache_groups`` does.
    """
    fields = getattr(spec, "__dict__", None)
    if fields is None:
        fields = {s: getattr(spec, s, None) for s in getattr(spec, "__slots__", ())}
    return type(spec).__name__ + repr(sorted((k, repr(v)) for k, v in fields.items()))


def build_config(manager, radix: bool):
    """Flatten a live ``KVCacheManager`` into the plain data the crate accepts.

    Returns ``(config_dict, None)`` or ``(None, reason)``. Never raises.
    """
    try:
        from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
    except Exception as exc:  # pragma: no cover - vLLM always present in the engine
        return None, f"vllm import failed: {exc!r}"

    try:
        kv_cache_config = manager.kv_cache_config
        coordinator = manager.coordinator
        pool = manager.block_pool

        if getattr(pool, "enable_kv_cache_events", False):
            return None, "kv cache events are enabled (not ported)"
        if getattr(pool, "metrics_collector", None) is not None:
            return None, "a KVCacheMetricsCollector is installed (not ported)"
        if getattr(coordinator, "retention_interval", None) is not None:
            return None, "VLLM_PREFIX_CACHE_RETENTION_INTERVAL is set (not ported)"
        if not manager.enable_caching:
            return None, "prefix caching is disabled (nothing to port)"

        groups = []
        for group in kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            if isinstance(spec, MambaSpec):
                kind = "mamba"
                mamba_align = getattr(spec, "mamba_cache_mode", None) == "align"
                num_spec_blocks = int(getattr(spec, "num_speculative_blocks", 0) or 0)
            elif isinstance(spec, FullAttentionSpec):
                kind = "full"
                mamba_align = False
                num_spec_blocks = 0
            else:
                return None, f"unported kv cache spec {type(spec).__name__}"
            groups.append(
                {
                    "kind": kind,
                    "block_size": int(spec.block_size),
                    "is_full_attention": isinstance(spec, FullAttentionSpec),
                    "spec_signature": spec_signature(spec),
                    "mamba_align": mamba_align,
                    "num_speculative_blocks": num_spec_blocks,
                    "use_eagle": bool(getattr(group, "is_eagle_group", False)),
                }
            )

        cfg = {
            "num_blocks": int(kv_cache_config.num_blocks),
            "enable_caching": True,
            "max_model_len": int(manager.max_model_len),
            "scheduler_block_size": int(coordinator.scheduler_block_size),
            "hash_block_size": int(pool.hash_block_size),
            "log_stats": bool(manager.log_stats),
            "watermark": float(manager.watermark_blocks) / max(kv_cache_config.num_blocks, 1),
            "radix": bool(radix),
            "groups": groups,
        }
        return cfg, None
    except Exception as exc:
        return None, f"config extraction failed: {exc!r}"


def schedule_supported(scheduler) -> str | None:
    """Reason the Rust schedule() loop cannot run for this scheduler, or None."""
    checks = (
        ("connector", getattr(scheduler, "connector", None) is not None),
        ("ec_connector", getattr(scheduler, "ec_connector", None) is not None),
        ("lora", getattr(scheduler, "lora_config", None) is not None),
        ("spec_decode", int(getattr(scheduler, "num_spec_tokens", 0) or 0) > 0),
        ("eagle", bool(getattr(scheduler, "use_eagle", False))),
        ("dynamic_sd", getattr(scheduler, "dynamic_sd_lookup", None) is not None),
        ("encoder_decoder", bool(getattr(scheduler, "is_encoder_decoder", False))),
        ("reserve_full_isl", bool(getattr(scheduler, "scheduler_reserve_full_isl", False))),
        (
            "priority_policy",
            str(getattr(getattr(scheduler, "policy", None), "value", "fcfs")) != "fcfs",
        ),
    )
    bad = [name for name, hit in checks if hit]
    return ("unsupported scheduler features: " + ", ".join(bad)) if bad else None


# --------------------------------------------------------------------------
# Rust-side request mirror
# --------------------------------------------------------------------------


class RustMirror:
    """Keeps the Rust core's view of each request's block hashes in sync.

    vLLM computes ``Request.block_hashes`` itself (the hasher is attached at request
    creation, upstream of the KV manager), so the authoritative path never re-hashes in
    Rust -- it just receives the bytes. That removes any chance of a hash divergence in
    serving while keeping the ported hasher (``vtl_sched.block_hashes``) available and
    covered by the crate's own tests.
    """

    def __init__(self, rust):
        self.rust = rust
        self._slots: dict[str, int] = {}
        # slot -> hashes already pushed. `slot()` runs once per request PER STEP, so
        # asking Rust for `num_hashes` there was one FFI crossing per request per step to
        # re-learn a number only this class ever changes. Now it is one crossing per
        # request LIFETIME (on first sight, because a recycled slot's hashes were cleared
        # by `Manager::forget` and assuming that would couple us to its internals).
        self._pushed: dict[int, int] = {}
        # slots whose stop params are interned (UFO); discarded on drop so a recycled
        # slot re-registers. Owned here because drop() is the one recycling point.
        self._stops: set[int] = set()
        # slots whose burst eligibility is interned (commit_burst's `_burst_gate`):
        # value is the cached `lim`, or -1 for a permanently ineligible request.
        # Discarded on drop for the same reason as `_stops`.
        self._burst_lim: dict[int, int] = {}
        # R9: slots proven packable (the 6 `decide()` clauses all passed at least once).
        # Monotonic within a slot's lifetime -- see `decide()`'s docstring on why the
        # `prefill_stats` one-shot latch makes that safe -- and discarded on drop like
        # `_stops`/`_burst_lim` so a recycled slot re-proves itself.
        self._pack_ok: set[int] = set()
        # R9: the previous step's batch derivation, `(req_id_to_index, slots, rows,
        # requests)`, keyed by IDENTITY of the first element. `req_id_to_index` is the
        # SAME dict object across repeat decode steps (hotpath_microopt.patch), so an
        # identity hit means "the exact same requests, same order" without walking them.
        # `None` when nothing is cached; invalidated (see call sites) rather than trusted
        # across anything that could change the request set or its facade state.
        self._batch_cache = None

    def slot(self, request) -> int:
        rid = request.request_id
        slot = self._slots.get(rid)
        if slot is None:
            slot = self.rust.intern(rid)
            self._slots[rid] = slot
            self._pushed[slot] = self.rust.num_hashes(slot)
        hashes = request.block_hashes
        have = self._pushed[slot]
        # Port-2: for a store-owned request Python's list is FROZEN at the count Rust
        # already has (the crate grew the chain itself in `tokens.rs`), so this comparison
        # is False forever and the hash count is sourced from Rust state at zero FFI cost.
        # `tok_materialize` re-syncs both sides when a request goes back to the stock path.
        if len(hashes) > have:
            self.rust.push_hashes(
                slot, b"".join(hashes[have:]), int(request.num_prompt_tokens)
            )
            self._pushed[slot] = len(hashes)
        return slot

    def drop(self, request_id: str) -> None:
        slot = self._slots.pop(request_id, None)
        if slot is not None:
            self._pushed.pop(slot, None)
            self._stops.discard(slot)
            self._burst_lim.pop(slot, None)
            self._pack_ok.discard(slot)
            # A freed slot may have been part of the cached batch; the cheapest correct
            # thing is to drop the whole cache rather than reason about which requests
            # it still names correctly.
            self._batch_cache = None
            self.rust.forget(request_id)


_STATUS_MAP: dict = {}


def status_code(request) -> int:
    """Map ``RequestStatus`` to the small ints the Rust core uses. On the per-step path,
    so the enum lookup is resolved once and then it is a dict hit.

    Unknown statuses REFUSE. v0.25.0 has three more waiting-ish states
    (``WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR``, ``WAITING_FOR_REMOTE_KVS``,
    ``WAITING_FOR_STREAMING_REQ``); defaulting them to RUNNING made the Rust loop admit
    requests the Python scheduler parks in ``skipped_waiting``.
    """
    if not _STATUS_MAP:
        from vllm.v1.request import RequestStatus

        _STATUS_MAP[RequestStatus.WAITING] = _ST_WAITING
        _STATUS_MAP[RequestStatus.RUNNING] = _ST_RUNNING
        _STATUS_MAP[RequestStatus.PREEMPTED] = _ST_PREEMPTED
    try:
        return _STATUS_MAP[request.status]
    except KeyError:
        raise NotImplementedError(
            f"rust_sched: unported RequestStatus {request.status!s}"
        ) from None


# --------------------------------------------------------------------------
# authority mode
# --------------------------------------------------------------------------


class RustBlocks:
    """``KVCacheBlocks`` stand-in over Rust-owned block IDs.

    Only the members vLLM's scheduler actually touches are implemented; anything else
    raises loudly rather than returning something plausible.
    """

    __slots__ = ("_ids",)

    def __init__(self, ids):
        self._ids = tuple(ids)

    @property
    def blocks(self):
        return self._ids

    def get_block_ids(self, allow_none: bool = False):
        if allow_none and all(len(g) == 0 for g in self._ids):
            return None
        return tuple(list(g) for g in self._ids)

    def new_empty(self):
        return RustBlocks(tuple([] for _ in self._ids))

    def __add__(self, other):
        return RustBlocks(
            tuple(list(a) + list(b) for a, b in zip(self._ids, other.blocks))
        )

    def get_unhashed_block_ids(self):
        raise NotImplementedError("KV connectors are not supported by VTL_RUST_SCHED")

    get_unhashed_block_ids_all_groups = get_unhashed_block_ids


# Public ``KVCacheManager`` members (v0.25.0 kv_cache_manager.py) that authority mode
# deliberately INHERITS rather than overrides. Empty today: everything the base exposes is
# either reimplemented against Rust or refused. It exists so that a vLLM bump adding a
# method makes `_refuse_unported` stub it, instead of silently answering from the stale
# python coordinator.
_AUTHORITY_INHERITS: frozenset = frozenset()


def _refuse_unported(cls, base) -> None:
    """Stub every public base method the subclass did not port, with a raiser.

    Same doctrine as ``RustBlocks``: in authority mode the python coordinator, block pool
    and per-request block lists are frozen at construction, so an inherited method does
    not fail -- it returns a confident, stale answer.
    """
    def make_raiser(name):
        def raiser(self, *args, **kwargs):
            raise NotImplementedError(
                f"KVCacheManager.{name} is not ported to VTL_RUST_SCHED; refusing "
                "rather than answering from stale python state"
            )

        raiser.__name__ = name
        return raiser

    for name in dir(base):
        if name.startswith("_") or name in _AUTHORITY_INHERITS or name in vars(cls):
            continue
        if not callable(getattr(base, name, None)):
            continue
        setattr(cls, name, make_raiser(name))


def _install_authority(base, mirror_modes):
    """Rust becomes the source of truth for the KVCacheManager surface."""
    import vtl_sched

    class VtlRustKVCacheManager(base):
        __vtl_rust_authority__ = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if kv_transfer_configured():
                # Split brain: with a KV connector, scheduler.py binds the (now stale)
                # python block_pool into the connector (`:280`) and, when
                # `defer_block_free` is on, drains `pop_blocks_for_free`'s result through
                # `block_pool.free_blocks` (`:2157`) -- which would leak every deferred
                # block out of the Rust pool for good.
                raise RuntimeError(
                    "VTL_RUST_SCHED=1 is not compatible with a KV connector "
                    "(kv_transfer_config is set)"
                )
            cfg, reason = build_config(self, mirror_modes["radix"])
            if reason is not None:
                raise RuntimeError(f"VTL_RUST_SCHED=1 but config unsupported: {reason}")
            self._rust = vtl_sched.KvManager(cfg)
            self._mirror = RustMirror(self._rust)
            # Batch 3: hand the crate's KvManager to shm_ipc's output thread, which is the
            # one place that actually knows this boot's `input_address` (`self.addresses`
            # lives on EngineCoreProc, not here). shm_ipc calls `out_open` itself, lazily,
            # once it starts -- registering here just makes the handle reachable. Cheap and
            # harmless if the gate ends up off: an unread registration is a no-op.
            #
            # UNCONDITIONAL as of the Rust runner: `_process_output_sockets` re-checks
            # VTL_SHM_IPC_RUST_PUB when it READS the handle (shm_ipc.py:795), so gating the
            # WRITE changed nothing for shm_ipc while making the handle invisible to every
            # other consumer -- and `rust_runner.export` needs it to build the runner.
            try:
                from vtl.patches import shm_ipc

                shm_ipc.register_rust_kv(self._rust)
            except Exception:
                log.exception(
                    "rust_sched: could not register the rust kv handle with shm_ipc"
                )
            # R6b/R6c state. Always constructed (cheap); the gates decide who reads it.
            self._vtl_table = TableState()
            self._empty = RustBlocks(tuple([] for _ in range(self._rust.num_groups)))
            # Plain attribute, not a property: the base __init__ assigns it (a getter-only
            # property makes super().__init__ raise). Overwrite the python value with ours.
            self.empty_kv_cache_blocks = self._empty
            log.info(
                "rust_sched: AUTHORITY mode active (%d groups, %d blocks)",
                self._rust.num_groups,
                cfg["num_blocks"],
            )

        # --- helpers --------------------------------------------------------

        def _blocks_of(self, slot):
            return RustBlocks(
                tuple(
                    self._rust.buffer(g)[: self._rust.blocks_into_buffer(slot, g)].tolist()
                    for g in range(self._rust.num_groups)
                )
            )

        def _pending(self):
            return RustBlocks(
                tuple(
                    self._rust.buffer(g)[: self._rust.pending_hit_into_buffer(g)].tolist()
                    for g in range(self._rust.num_groups)
                )
            )

        # --- surface --------------------------------------------------------

        @property
        def usage(self):
            return self._rust.usage

        @property
        def free_blocks(self):
            return self._rust.num_free_blocks

        def plan_request(self, request):
            """The vtl scheduling signal, re-pointed at Rust (the Python coordinator is
            stale in this mode). Same contract as vtl/patches/kv_cache_manager.py."""
            try:
                num_prompt = request.num_prompt_tokens
                if getattr(request, "num_computed_tokens", 0):
                    remaining = max(num_prompt - request.num_computed_tokens, 0)
                else:
                    slot = self._mirror.slot(request)
                    hit = self._rust.peek_cache_hit(slot, int(request.num_tokens))
                    remaining = max(num_prompt - hit, 0)
            except BaseException as exc:
                reraise_fatal(exc)
                return (1 << 60), (1 << 60)
            bs = self.coordinator.scheduler_block_size or 16
            return remaining, -(-remaining // bs)

        def new_step_starts(self):
            self._rust.new_step_starts()

        def make_prefix_cache_stats(self):
            raw = self._rust.take_prefix_cache_stats()
            if raw is None:
                return None
            from vllm.v1.metrics.stats import PrefixCacheStats

            stats = PrefixCacheStats()
            for key, value in raw.items():
                setattr(stats, key, value)
            return stats

        def get_computed_blocks(self, request):
            slot = self._mirror.slot(request)
            num = self._rust.get_computed_blocks(
                slot,
                int(request.num_tokens),
                int(request.num_preemptions),
                bool(request.skip_reading_prefix_cache),
            )
            # The scheduler reads this off the coordinator for the mamba split.
            self.coordinator.num_uncached_common_prefix_tokens = (
                self._rust.num_uncached_common_prefix_tokens
            )
            if num == 0:
                return self._empty, 0
            return self._pending(), num

        def allocate_slots(
            self,
            request,
            num_new_tokens,
            num_new_computed_tokens: int = 0,
            new_computed_blocks=None,
            num_lookahead_tokens: int = 0,
            num_external_computed_tokens: int = 0,
            delay_cache_blocks: bool = False,
            num_encoder_tokens: int = 0,
            full_sequence_must_fit: bool = False,
            reserved_blocks: int = 0,
            has_scheduled_reqs: bool = True,
        ):
            if (
                num_external_computed_tokens
                or delay_cache_blocks
                or num_encoder_tokens
                or full_sequence_must_fit
                or reserved_blocks
            ):
                raise NotImplementedError(
                    "VTL_RUST_SCHED=1 does not support connector / encoder / "
                    "full-sequence-admission allocation paths"
                )
            slot = self._mirror.slot(request)
            ok = self._rust.allocate_slots(
                slot,
                int(num_new_tokens),
                int(num_new_computed_tokens),
                new_computed_blocks is not None,
                int(num_lookahead_tokens),
                int(request.num_computed_tokens),
                int(request.num_tokens),
                status_code(request),
                bool(has_scheduled_reqs),
            )
            if not ok:
                return None
            return RustBlocks(
                tuple(
                    self._rust.buffer(g)[: self._rust.new_blocks_into_buffer(g)].tolist()
                    for g in range(self._rust.num_groups)
                )
            )

        def free(self, request):
            slot = self._mirror.slot(request)
            self._rust.free(slot)
            self._mirror.drop(request.request_id)

        def cache_blocks(self, request, num_computed_tokens):
            self._rust.cache_blocks(self._mirror.slot(request), int(num_computed_tokens))

        def remove_skipped_blocks(self, request_id, total_computed_tokens, num_prompt_tokens=None):
            slot = self._rust.lookup(request_id)
            if slot is not None:
                self._rust.remove_skipped_blocks(slot, int(total_computed_tokens))

        def pop_blocks_for_free(self, request):
            # Defence in depth behind the kv_transfer_config refusal in __init__: the only
            # caller is `_free_request_blocks` under `defer_block_free`, which hands the
            # result to the STALE python `block_pool.free_blocks` (scheduler.py:2157).
            # Returning ints there is an AttributeError; returning anything is a permanent
            # leak from the Rust pool.
            raise RuntimeError(
                "VTL_RUST_SCHED=1 does not support deferred block freeing "
                "(pop_blocks_for_free drains the stale python block pool)"
            )

        def evict_blocks(self, block_ids):
            # Out-of-band pool surgery: every resident entry's block list may have moved
            # under it, and a pending speculation was computed against the old pool.
            self._vtl_table.resync("evict_blocks")
            self._rust.evict_blocks([int(b) for b in block_ids])

        def reset_prefix_cache(self):
            self._vtl_table.resync("reset_prefix_cache")
            # R9: a whole-pool event invalidates whatever the batch cache assumed about
            # cached block state (folded `cache_blocks` calls included).
            self._mirror._batch_cache = None
            return self._rust.reset_prefix_cache()

        def get_num_common_prefix_blocks(self, running_request_id):
            slot = self._rust.lookup(running_request_id)
            if slot is None:
                return [0] * self._rust.num_groups
            return list(self._rust.num_common_prefix_blocks(slot))

        def take_events(self):
            return []

        def get_blocks(self, request_id):
            slot = self._rust.lookup(request_id)
            if slot is None:
                return self._empty
            return self._blocks_of(slot)

        def get_block_ids(self, request_id):
            slot = self._rust.lookup(request_id)
            if slot is None:
                return tuple([] for _ in range(self._rust.num_groups))
            return self._rust.block_ids_lists(slot)

        def get_block_ids_for_computed_tokens(self, request_id, num_computed_tokens):
            slot = self._rust.lookup(request_id)
            if slot is None:
                return tuple([] for _ in range(self._rust.num_groups))
            return self._rust.block_ids_for_computed_tokens(slot, int(num_computed_tokens))

        def create_kv_cache_blocks(self, blocks):
            return RustBlocks(tuple(blocks))

        def take_new_block_ids(self):
            return self._rust.take_new_block_ids()

    _refuse_unported(VtlRustKVCacheManager, base)
    return VtlRustKVCacheManager


# --------------------------------------------------------------------------
# R5 -- the schedule() loop
# --------------------------------------------------------------------------


class TableState:
    """R6b/R6c bookkeeping, one per Rust-backed KVCacheManager.

    It lives on the MANAGER, not the scheduler, because the three writers are spread
    across both wrappers: the schedule() wrapper, the update_from_output() wrapper, and
    the manager's own ``reset_prefix_cache`` / ``evict_blocks``.

    ``dirty`` means "the resident table may not match Python's request objects". It starts
    dirty (nothing is resident yet) and clears ONLY when a marshalled ``Scheduler.schedule``
    completes -- that call is the full resync.

    ``gen`` is the speculation's staleness token: every mutation Python makes that the Rust
    manager cannot see bumps it, and ``take_speculative`` refuses a result computed under
    an older one. THE BUMP SITES, exhaustively (anything else reaches Rust through an
    invalidating ``w()`` guard and needs no counter):

      1. ``Scheduler.add_request``      -- a new request lands in the waiting queue
      2. ``Scheduler.finish_requests``  -- abort/finish from outside the scheduler
      3. ``reset_prefix_cache``         -- also dirties: it is a whole-pool event
      4. ``evict_blocks``               -- ditto
      5. any bail condition observed by the schedule() wrapper (stock vLLM then mutates
         requests and queues with no Rust call at all)
      6. any per-request UFO fallback (status 255 / length mismatch / unregistered slot):
         Python ran check_stop itself, so the table missed that request's token delta
      7. every dirty transition, so a speculation kicked before it cannot be consumed after
    """

    __slots__ = ("dirty", "gen", "armed", "off")

    def __init__(self):
        self.dirty = True
        self.gen = 0
        self.armed = False  # a kick is outstanding and not yet consumed
        self.off = False    # permanent, process-lifetime fallback to the marshalled path

    def bump(self) -> None:
        self.gen += 1
        # A pending speculation can no longer be consumed; dropping `armed` saves the
        # crossing. Rolling it back is not our job -- the next schedule call goes through
        # an invalidating guard either way.
        self.armed = False

    def resync(self, why: str) -> None:
        if not self.dirty:
            self.dirty = True
            log.debug("rust_sched: resident table marked dirty -- %s", why)
        self.bump()

    def fail(self, where: str, exc: BaseException) -> None:
        """Fail-open: one log, then the shipped marshalled behaviour forever."""
        reraise_fatal(exc)
        self.dirty = True
        self.armed = False
        if not self.off:
            self.off = True
            log.exception(
                "rust_sched: resident-table path failed in %s; permanently marshalled", where
            )


class PhaseTimers:
    """VTL_SCHED_TIMING: p50/p95 of the three schedule() phases plus the step gap.

    Deliberately a plain ring of ints per phase -- no numpy, no histogram. 500 samples
    sorted twice per 500 steps is noise next to one scheduling step.
    """

    PHASES = ("marshal", "rust", "apply", "gap")
    EVERY = 500

    def __init__(self):
        self.rings = {p: [] for p in self.PHASES}
        self.n = 0
        self.last_exit = 0

    def add(self, phase: str, ns: int) -> None:
        self.rings[phase].append(ns)

    def tick(self) -> None:
        self.n += 1
        if self.n % self.EVERY:
            return
        parts = []
        for p in self.PHASES:
            ring = self.rings[p]
            if not ring:
                continue
            ring.sort()
            parts.append(
                f"{p} p50={ring[len(ring) // 2] / 1000:.1f}us "
                f"p95={ring[min(len(ring) - 1, int(len(ring) * 0.95))] / 1000:.1f}us"
            )
            ring.clear()
        log.info("rust_sched: schedule() timing over %d steps -- %s", self.EVERY, "  ".join(parts))


def bail_reason(scheduler):
    """Per-step conditions the Rust loop does not model (scheduler.py:637-664).

    Module level because both wrappers consult it: the schedule() wrapper to decide the
    step, and the update_from_output() wrapper to decide whether kicking is safe.
    """
    if getattr(scheduler, "skipped_waiting", None):
        return "blocked requests are parked in skipped_waiting"
    if getattr(scheduler, "num_waiting_for_streaming_input", 0):
        return "paused streaming sessions hold model-runner slots"
    pause = getattr(scheduler, "_pause_state", None)
    if pause is not None and getattr(pause, "name", str(pause)) != "UNPAUSED":
        return f"scheduler is paused ({pause})"
    return None


# --------------------------------------------------------------------------
# C1a -- N-step decode burst: the scheduler side
# --------------------------------------------------------------------------


def burst_blocked_batch(so, cap: int) -> str | None:
    """The batch-level gate MINUS the queue-empty guard. Pure -- self-checked.

    Split out because the N=1 in-graph sampling commit wants exactly this and not the queue
    guard: sampling one token in a graph delays no admission, so a non-empty waiting queue is
    irrelevant to it while it is the whole TTFT argument against a burst.
    """
    if so.scheduled_new_reqs:
        return "new requests admitted this step"
    if so.preempted_req_ids:
        return "a request was preempted"
    if so.scheduled_spec_decode_tokens or so.scheduled_encoder_inputs:
        return "spec-decode or encoder inputs scheduled"
    if so.has_structured_output_requests:
        return "structured output"
    if so.new_block_ids_to_zero:
        return "fresh blocks need zeroing"
    nst = so.num_scheduled_tokens
    if not nst:
        return "empty step"
    if len(nst) > cap:
        return f"batch of {len(nst)} exceeds the captured burst sizes"
    for num in nst.values():
        if num != 1:
            return "not a pure 1-token-per-request decode batch"
    return None


def burst_blocked(so, waiting_nonempty: bool, queue_empty_only: bool, cap: int) -> str | None:
    """Why this step cannot carry a BURST, or ``None`` if it can. Pure -- self-checked.

    Batch-level half of the gate; ``burst_request_blocked`` is the per-request half.
    Refusing an admission step is what keeps the burst's TTFT cost bounded: a new request
    would otherwise wait behind N decode iterations instead of one
    (``VTL_NSTEP_QUEUE_EMPTY_ONLY``, on by default, extends that to a merely non-empty
    queue).
    """
    why = burst_blocked_batch(so, cap)
    if why is not None:
        return why
    if queue_empty_only and waiting_nonempty:
        return "waiting queue is not empty"
    return None


def burst_sampler_blocked(request, computed_before: int, n: int,
                          max_model_len: int) -> str | None:
    """The per-request gate MINUS the align gate: everything about the SAMPLER.

    The burst (and the N=1 in-graph path) samples a bare ``argmax``, so anything that would
    move the token away from it -- temperature, penalties, a logit bias, bad words, an unmet
    ``min_tokens`` -- keeps the request on the stock sampler. Also the length caps, which N=1
    still needs: a request one token from ``max_tokens`` must go through the code that
    truncates, not through a bare argmax.

    Split out because N=1 needs it without the align gate: one token crosses no block
    boundary by construction, so the whole "the block tables the graph baked stay valid"
    argument is vacuous for it.
    """
    sp = request.sampling_params
    if sp is None or request.pooling_params is not None:
        return "not a generation request"
    if getattr(request, "resumable", False) or request.use_structured_output:
        return "resumable or structured-output request"
    if request.num_output_placeholders < 0:
        return "negative placeholder count"
    max_tokens = getattr(request, "max_tokens", None) or 0
    limit = min(max_model_len, request.num_prompt_tokens + max_tokens)
    if computed_before + n > limit:
        return "burst would run past max_tokens / max_model_len"
    if sp.temperature != 0.0:
        return "not greedy"
    if sp.logprobs is not None or sp.prompt_logprobs is not None:
        return "logprobs requested"
    if getattr(sp, "min_tokens", 0):
        # min_tokens is enforced by masking EOS in the logits; a bare argmax ignores it.
        return "min_tokens masking is not applied by the burst argmax"
    if sp.bad_words or sp.allowed_token_ids or sp.logit_bias:
        return "logit-modifying sampling params"
    if sp.presence_penalty or sp.frequency_penalty or sp.repetition_penalty != 1.0:
        return "penalties"
    return None


def burst_request_blocked(request, computed_before: int, n: int, block_size: int,
                          max_model_len: int) -> str | None:
    """Why this request cannot ride a burst, or ``None``. Pure -- self-checked.

    ``computed_before`` is ``num_computed_tokens`` BEFORE ``_update_after_schedule``
    advanced it, i.e. the position of the token this step is about to feed.

    THE ALIGN GATE is the first check and the load-bearing one: with
    ``computed_before % block_size + n <= block_size`` the whole burst lives inside one KV
    block and one mamba state column, so no allocation, no state migration and no block
    table change can happen between iterations. The rest is
    ``burst_sampler_blocked``.
    """
    if computed_before % block_size + n > block_size:
        return "burst would cross a block boundary"
    return burst_sampler_blocked(request, computed_before, n, max_model_len)


def _burst_immutable_blocked(request) -> str | None:
    """The request-immutable subset of ``burst_sampler_blocked``'s clauses -- everything
    sourced from ``sampling_params``/``resumable``/``use_structured_output``, none of
    which change after admission. Split out so ``_burst_gate`` can intern it once per
    slot instead of re-evaluating it every step. Excludes the placeholder count and the
    length-cap arithmetic, both of which move every step.
    """
    sp = request.sampling_params
    if sp is None or request.pooling_params is not None:
        return "not a generation request"
    if getattr(request, "resumable", False) or request.use_structured_output:
        return "resumable or structured-output request"
    if sp.temperature != 0.0:
        return "not greedy"
    if sp.logprobs is not None or sp.prompt_logprobs is not None:
        return "logprobs requested"
    if getattr(sp, "min_tokens", 0):
        return "min_tokens masking is not applied by the burst argmax"
    if sp.bad_words or sp.allowed_token_ids or sp.logit_bias:
        return "logit-modifying sampling params"
    if sp.presence_penalty or sp.frequency_penalty or sp.repetition_penalty != 1.0:
        return "penalties"
    return None


def _burst_gate(mirror, slot: int, request, computed_before: int, n: int,
                max_model_len: int, block_size: int | None = None) -> str | None:
    """``burst_request_blocked``/``burst_sampler_blocked``, with the immutable clauses
    interned per slot in ``mirror._burst_lim`` (RustMirror-owned, popped on ``drop()``
    so a recycled slot re-evaluates). First sight of a slot pays the full
    ``_burst_immutable_blocked`` check once and caches ``lim = min(max_model_len,
    num_prompt_tokens + max_tokens)``, or ``-1`` if permanently ineligible. Every later
    step only re-checks what a step can actually change: the align gate (when
    ``block_size`` is given), the placeholder count, and the length-cap arithmetic
    against the cached ``lim``.

    ``block_size is None`` skips the align gate -- the N=1 in-graph path doesn't need
    it (one token crosses no block boundary; see ``burst_sampler_blocked``'s
    docstring).
    """
    lim = mirror._burst_lim.get(slot)
    if lim is None:
        reason = _burst_immutable_blocked(request)
        if reason is not None:
            mirror._burst_lim[slot] = -1
            return reason
        max_tokens = getattr(request, "max_tokens", None) or 0
        lim = mirror._burst_lim[slot] = min(max_model_len, request.num_prompt_tokens + max_tokens)
    elif lim < 0:
        return "ineligible request (interned)"
    if block_size is not None and computed_before % block_size + n > block_size:
        return "burst would cross a block boundary"
    if request.num_output_placeholders < 0:
        return "negative placeholder count"
    if computed_before + n > lim:
        return "burst would run past max_tokens / max_model_len"
    return None


def burst_steps(computed_before: int, n: int, block_size: int, lim: int,
                ceiling: int) -> int:
    """How many back-to-back bursts of ``n`` this request can take. Pure -- self-checked.

    The Rust runner takes several burst launches in one call, and the reason the align gate
    exists does not change when they are back-to-back: the mamba ``state_indices_tensor_d``
    and the block tables are baked into the captured graphs and re-derived by nothing, so
    the WHOLE span has to stay inside the KV block the first launch started in. ``lim`` is
    the ``min(max_model_len, num_prompt_tokens + max_tokens)`` cap ``_burst_gate`` interned
    for the slot, so a multi-burst step cannot run past ``max_tokens`` either.

    Never returns 0: this is only ever asked about a request whose SINGLE-burst gate
    already passed, and the two answers agree by construction -- ``(block_size -
    computed_before % block_size) // n >= 1`` IS ``computed_before % block_size + n <=
    block_size``.
    """
    if ceiling < 2 or lim < 0:
        return 1
    return max(1, min(
        ceiling,
        (block_size - computed_before % block_size) // n,
        (lim - computed_before) // n,
    ))


def burst_commit(request, delta: int) -> None:
    """Advance one request by the ``delta`` extra tokens the burst will compute.

    Byte-for-byte ``Scheduler._update_after_schedule`` + ``AsyncScheduler``'s placeholder
    bump, applied a second time -- INCLUDING the ordering, which is load-bearing:
    ``is_prefill_chunk`` is computed from the placeholder count BEFORE the bump. The Rust
    resident table applies the same thing via ``Manager::table_burst`` -> ``sched::advance``.
    """
    request.num_computed_tokens += delta
    request.is_prefill_chunk = request.num_computed_tokens < (
        request.num_tokens + request.num_output_placeholders
    )
    if not request.is_prefill_chunk:
        request.num_output_placeholders += delta


def burst_uncommit(request, short: int) -> None:
    """Give back the ``short`` tokens a committed burst did not actually produce.

    THE PREFIX-CACHE GUARD. A burst request that hits EOS at token j < N -- or, if the
    runner ever bails, one that comes back with a single token -- had N tokens' worth of
    ``num_computed_tokens`` committed at schedule time. Left uncorrected, the next
    ``cache_blocks`` would fingerprint positions that were never computed and every later
    request matching that prefix would read uninitialised KV.

    ``AsyncScheduler._update_request_with_output`` calls ``cache_blocks(num_computed -
    num_output_placeholders)`` right after this runs, and both counters are corrected by the
    same amount, so the cached length is exactly the number of real tokens either way.
    """
    request.num_computed_tokens = max(0, request.num_computed_tokens - short)
    request.num_output_placeholders = max(0, request.num_output_placeholders - short)
    request.is_prefill_chunk = request.num_computed_tokens < (
        request.num_tokens + request.num_output_placeholders
    )


def pack_req(slot: int, request) -> tuple:
    """Field order must match ``lib.rs::ReqTuple``."""
    return (
        slot,
        int(request.num_tokens),
        # num_tokens_with_spec, by way of the invariant `schedule_supported` enforces:
        # a spec config refuses the whole install, so `spec_token_ids` is always empty
        # and the two counters are equal. Reading `num_tokens` twice beats an attribute
        # lookup that is a property call on both the facade and stock `Request`.
        int(request.num_tokens),
        int(request.num_computed_tokens),
        int(request.num_output_placeholders),
        int(request.num_prompt_tokens),
        int(getattr(request, "max_tokens", 0) or 0),
        status_code(request),
        int(request.num_preemptions),
        bool(getattr(request, "is_prefill_chunk", False)),
        bool(request.skip_reading_prefix_cache),
    )


# --------------------------------------------------------------------------
# Port-2 -- the Rust-owned token store: the Python half
# --------------------------------------------------------------------------
#
# THE CONSUMER SWEEP THAT MAKES THIS POSSIBLE (2026-07-30, deployed config: V2 runner,
# Rust frontend, R8, UFO, Rust hasher; no PP / LoRA / spec / connector / structured output).
# Only TWO live consumers need the token INTS:
#
#   1. ``NewRequestData.prefill_token_ids`` at admission and at preemption-resume;
#   2. the block hasher's block-aligned slices.
#
# Everything else reads ``num_tokens`` / ``num_output_tokens``, i.e. counts -- ten sites, all
# of which see a plain int just as happily as a ``len()``. And there is exactly ONE writer,
# the bulk ``append_output_token_ids`` in ``_urwo_inner``. So: (2) moves into Rust
# (``tokens.rs`` owns the chain, ``hash.rs`` does the sha256), the counts become plain int
# attributes bumped by the writer, ``_all_token_ids`` stops growing, and (1) is handled by
# MATERIALIZATION -- rebuilding the real lists from prompt + ``slot_tokens(slot)`` on the
# rare paths that need them (admission, preemption, any pack refusal, any UFO fallback).
#
# THE FACADE IS PER INSTANCE, NEVER GLOBAL. ``Request.num_tokens`` is a property on the
# class, and a data descriptor on the class beats an instance ``__dict__`` entry -- so a
# plain attribute cannot shadow it. Rebinding ``request.__class__`` to a subclass whose
# ``num_tokens`` is a plain CLASS attribute (not a descriptor) puts the instance dict back in
# front. Mutating vLLM's own ``Request`` would do this to every request in the process,
# including the ones this patch refuses; the subclass swap is scoped to the requests the
# store actually took over, and materialization swaps it straight back.


class _TokState:
    """Port-2 process state. One engine core per process, so a module global is the whole
    lifetime story -- same shape as ``nstep_decode.BURST``."""

    __slots__ = ("live", "armed", "hash_block_size", "facaded",
                 "materialized", "divergences", "warned")

    def __init__(self) -> None:
        self.live = False          # numpy fast path + facade installation
        self.armed = False         # NONE_HASH handed to the crate
        self.hash_block_size = 0
        self.facaded = 0
        self.materialized = 0
        self.divergences = 0
        self.warned = False

    def disable(self, why: str) -> None:
        """Permanent, boot-lifetime fallback. Requests already facaded keep working: the
        per-request writer branch keys off the FACADE, not off this flag, and the next step
        that cannot take the numpy path materializes them back onto the stock path."""
        if self.live:
            log.error("rust_sched: token store disabled for this boot -- %s", why)
        self.live = False


TOK = _TokState()


class _R9State:
    """R9 process state -- same shape as ``_TokState``/``nstep_decode.BURST``: one engine
    core per process, so a module global is the whole lifetime story.

    ``live`` gates taking the collapsed FFI-plus-residue path (fold_cache=True included).
    It is resolved once at install time from the env gates, then only ever moves towards
    ``False`` -- ``disable()`` is the sole mutator after that, mirroring ``TOK.disable``'s
    boot-lifetime latch.
    """

    __slots__ = ("live", "checks", "warned_kinds")

    def __init__(self) -> None:
        self.live = False
        self.checks = 0
        self.warned_kinds: set = set()

    def disable(self, why: str) -> None:
        if self.live:
            log.error("rust_sched: R9 disabled for this boot -- %s", why)
        self.live = False

    def warn_once(self, kind: str, msg: str, *args) -> None:
        """One ERROR per mismatch KIND per boot -- a soak that finds the same divergence
        every step must not flood the log the way an unbounded one would."""
        if kind not in self.warned_kinds:
            self.warned_kinds.add(kind)
            log.error(msg, *args)


R9 = _R9State()


def _pack_ok_clauses(request) -> bool:
    """The 6-clause check the R8 record has no slot for: a second client index, a
    resumable session's re-enqueue, encoder-cache frees, trace headers, prefill stats or
    lifecycle events. ``decide()`` interns a ``True`` result per slot in
    ``RustMirror._pack_ok`` instead of re-running this every step -- see that field's
    docstring for why the one mutable member (``prefill_stats``, a one-shot latch) is
    still safe to cache forever once cleared. Pure -- self-checked.
    """
    return (
        request.client_index == 0
        and not request.resumable
        and not request.has_encoder_inputs
        and request.trace_headers is None
        and request.prefill_stats is None
        and not request.events
    )


def _r9_cache_hit_counts(cache, r2i, sampled_counts):
    """R9's fast-hit guard, factored out of ``decide()`` so it is unit-testable without a
    live crate: an identity check plus a cheap numpy gather plus a zero-count guard.

    ``cache`` is ``mirror._batch_cache`` (``(r2i, slots, rows, requests)`` or ``None``);
    ``r2i`` is THIS step's ``model_runner_output.req_id_to_index``; ``sampled_counts`` is
    the sampler's per-row count array (``LazySampled.counts``). Returns the gathered
    per-cached-row counts as a plain list on a usable hit, or ``None`` on any miss --
    no cache, an identity mismatch, or a prefill chunk (a zero anywhere in the gather)
    changing which rows belong in the batch this step. Pure -- self-checked.
    """
    if cache is None or cache[0] is not r2i:
        return None
    _, _slots, rows, _requests = cache
    # `0 in list` beats a numpy `(counts == 0).any()` here: the gather is a handful of
    # elements at decode batch sizes, and the numpy reduction's fixed cost dwarfs the scan.
    counts = sampled_counts[rows].tolist()
    return None if 0 in counts else counts


def _r9_maybe_check_fold_skips(rust) -> None:
    """R9: ``cache_fold_skips()`` is the loud version of ``table_with``'s silent skip.
    Cheap to call every step, but every-4096 is cheaper still, and the hazard -- if it
    exists at all -- does not need step-granularity detection."""
    R9.checks += 1
    if R9.checks & 0xFFF:
        return
    try:
        skips = rust.cache_fold_skips()
    except BaseException as exc:
        reraise_fatal(exc)
        return
    if skips:
        R9.disable(f"cache_fold_skips={skips} (table_with silent-skip hazard fired)")


class _Row:
    """One request's sampled tokens, as a length and a promise.

    ``decide()`` needs only ``len()`` and truthiness; ``r8_apply`` hands this straight to
    ``_update_request_with_output``, which on the fast path also needs only the length. So
    the row is never converted to Python ints unless something actually asks for the values
    -- which is exactly the ``tolist()`` this port exists to delete.
    """

    __slots__ = ("arr", "i", "n")

    def __init__(self, arr, i: int, n: int) -> None:
        self.arr = arr
        self.i = i
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __bool__(self) -> bool:
        return self.n > 0

    def tolist(self) -> list:
        row = self.arr[self.i][: self.n]
        # ``ndarray.tolist()`` yields native Python ints, never np.int64 -- which is the
        # whole np.int64 hazard (block-hash inputs are pickled, and np.int64 pickles
        # differently from int, silently forking every prefix-cache key).
        return row.tolist() if hasattr(row, "tolist") else [int(t) for t in row]

    def __iter__(self):
        return iter(self.tolist())


class LazySampled:
    """``ModelRunnerOutput.sampled_token_ids`` that has not been converted to lists yet.

    ``AsyncOutput.get_output`` ends in two ``tolist()`` calls plus a per-row truncation loop;
    with the token store there is nothing left in Python that wants the ints, so the arrays
    are handed through and ``update_step_pack_np`` copies the ids out of the buffer directly.

    Duck-types enough of ``list[list[int]]`` that every OTHER consumer still works:
    ``materialize()`` does exactly what ``get_output`` did, and any step that takes the
    object path (a pack refusal, the ZMQ/non-R8 output arm in ``shm_ipc.py``, stock
    ``update_from_output``) calls it first, so those paths never see a ``_Row``.
    """

    __slots__ = ("arr", "counts", "_rows")

    def __init__(self, arr, counts) -> None:
        self.arr = arr
        self.counts = counts
        self._rows = None

    def materialize(self) -> None:
        if self._rows is not None:
            return
        rows = self.arr.tolist() if hasattr(self.arr, "tolist") else [list(r) for r in self.arr]
        counts = self.counts.tolist() if hasattr(self.counts, "tolist") else list(self.counts)
        for row, n in zip(rows, counts):
            del row[n:]
        self._rows = rows

    def __len__(self) -> int:
        return len(self.counts)

    def __bool__(self) -> bool:
        return len(self.counts) > 0

    def __getitem__(self, i):
        if self._rows is not None:
            return self._rows[i]
        return _Row(self.arr, i, int(self.counts[i]))

    def __iter__(self):
        self.materialize()
        return iter(self._rows)


_FACADE_CLASSES: dict = {}


def facade_class(base):
    """``base`` with the three count properties demoted to plain class attributes.

    Cached per base class: there is one ``Request`` class in practice, and building the
    subclass once keeps ``isinstance`` caches and method resolution stable.
    """
    cls = _FACADE_CLASSES.get(base)
    if cls is None:
        cls = _FACADE_CLASSES[base] = type(
            "VtlTokStore" + base.__name__,
            (base,),
            {
                # Non-descriptors, so the instance __dict__ wins the lookup. The values
                # here are never read: install() writes the instance attributes first.
                "num_tokens": 0,
                "num_output_tokens": 0,
                # A property (data descriptor), so it is NOT in the instance __dict__ and
                # needs no per-token bump: `schedule_supported` refuses any spec config, so
                # `spec_token_ids` is empty for every request the Rust core ever sees and
                # num_tokens_with_spec IS num_tokens. Aliasing rather than deleting keeps
                # stock `Scheduler.schedule` (reachable via the bail path, scheduler.py:474)
                # reading a live value instead of one frozen at facade install.
                "num_tokens_with_spec": property(lambda self: self.num_tokens),
                "__vtl_tokstore__": True,
            },
        )
    return cls


def facaded(request) -> bool:
    return getattr(type(request), "__vtl_tokstore__", False)


def tok_store_init(rust, request, slot: int, hash_block_size: int) -> bool:
    """Hand this slot's token bookkeeping to Rust. ``False`` = refuse, stay on stock.

    The seed is the request's token tail past the last block the Python hasher already
    produced a hash for -- fewer than ``hash_block_size`` tokens -- which is the whole reason
    Rust does not need the prompt: the chain state is (last hash, unhashed tail).
    """
    if (
        request.cache_salt
        or request.mm_features
        or request.lora_request is not None
        or request.prompt_embeds is not None
    ):
        # Exactly the refusal set of ``_install_rust_hasher``'s ``rust_block_hasher``:
        # anything that needs ``extra_keys`` is hashed by stock Python, so Rust cannot
        # continue that chain.
        return False
    hashed = len(request.block_hashes) * hash_block_size
    n_tok = int(request.num_tokens)
    tail = n_tok - hashed
    if tail < 0 or tail >= hash_block_size:
        # The hasher is behind (or ahead of) where the store expects it. Refuse rather
        # than reconcile: a wrong boundary is a poisoned cache key, not an exception.
        return False
    pending = [int(t) for t in request.all_token_ids[hashed:]] if tail else []
    rust.store_init(slot, pending, n_tok, int(request.num_output_tokens))
    return True


def tok_facade_on(request) -> None:
    """Swap in the counter facade. Must run AFTER ``tok_store_init`` agreed."""
    base = type(request)
    n_tok = int(request.num_tokens)
    n_out = int(request.num_output_tokens)
    # The class swap comes first: while `base` is bound, `num_tokens` is a getter-only
    # property and the assignment below would raise. (`num_tokens_with_spec` needs no
    # assignment -- the facade aliases it onto `num_tokens`.)
    request.__class__ = facade_class(base)
    request.num_tokens = n_tok
    request.num_output_tokens = n_out
    request._vtl_tok_base = base
    TOK.facaded += 1


def tok_materialize(kv, request) -> None:
    """Rebuild the real token lists and hash chain, then hand the request back to stock.

    Called on every path that needs the INTS or that is about to let stock code mutate the
    request: a pack refusal, a UFO per-request fallback, preemption, resume/admission. The
    request is permanently stock afterwards (``_vtl_tok_off``) -- re-facading it would mean
    re-deriving a seed tail from state stock may have edited.
    """
    if not facaded(request):
        return
    base = request._vtl_tok_base
    toks = None
    packed = None
    slot = kv._mirror._slots.get(request.request_id)
    if slot is None:
        # The slot was already recycled, so the store cannot answer. The request is being
        # retired on this path (that is the only way its slot goes away), so restoring the
        # class is all that is owed -- but say so, because a silent short list would be a
        # truncated completion.
        log.error(
            "rust_sched: token store cannot materialize %s -- its slot is gone",
            request.request_id,
        )
    else:
        try:
            toks = kv._rust.slot_tokens(slot)
            packed = kv._rust.slot_hashes(slot)
            kv._rust.store_forget(slot)
        except BaseException as exc:
            reraise_fatal(exc)
            log.exception("rust_sched: token store materialization failed for %s",
                          request.request_id)
            TOK.disable("materialization raised")
    for name in ("num_tokens", "num_output_tokens"):
        request.__dict__.pop(name, None)
    request.__class__ = base
    request._vtl_tok_off = True
    request._vtl_tok_on = False
    if toks:
        # In place: `all_token_ids` / `output_token_ids` are ConstantList VIEWS over these
        # two lists, so extending them keeps every existing view correct.
        request._all_token_ids.extend(toks)
        request._output_token_ids.extend(toks)
    if packed is not None and slot is not None:
        # Rust's chain is the complete one; Python's list froze at facade install. Replace
        # it wholesale and tell the mirror the crate already has all of it, or the next
        # `mirror.slot()` would push the tail a second time.
        hashes = [packed[i * 32:(i + 1) * 32] for i in range(len(packed) // 32)]
        request.block_hashes[:] = hashes
        kv._mirror._pushed[slot] = len(hashes)
    TOK.materialized += 1


def store_step_fallback(kv, sampled, batch, why: str) -> None:
    """Put a whole step back on the object path: real lists, real requests.

    Runs BEFORE anything delegates to stock code, which is the contract -- stock
    ``update_from_output`` builds ``EngineCoreOutput.new_token_ids`` out of
    ``sampled[index]`` and stock ``check_stop`` reads ``request.num_output_tokens``.
    """
    if isinstance(sampled, LazySampled):
        sampled.materialize()
    for _rid, _n, request in batch:
        tok_materialize(kv, request)
    if not TOK.warned:
        TOK.warned = True
        log.info("rust_sched: token store step fallback (first time) -- %s", why)


def store_full_fallback(scheduler, kv, scheduler_output, model_runner_output) -> None:
    """``store_step_fallback`` over the whole step, from the SchedulerOutput.

    Used where the batch decide() enumerated is not enough: stock ``update_from_output``
    iterates ``scheduler_output.num_scheduled_tokens``, so any store-owned request in there
    -- including one decide() skipped as finished -- must be materialized first.
    """
    batch = []
    for req_id in scheduler_output.num_scheduled_tokens:
        request = scheduler.requests.get(req_id)
        if request is not None and facaded(request):
            batch.append((req_id, 0, request))
    store_step_fallback(kv, model_runner_output.sampled_token_ids, batch, "stock objects")


def tok_arm(kv) -> bool:
    """Give the crate vLLM's live ``NONE_HASH`` and learn the hash block size. Once."""
    if TOK.armed:
        return True
    from vllm.v1.core import kv_cache_utils

    hbs = int(getattr(kv.block_pool, "hash_block_size", 0) or 0)
    if hbs <= 0:
        return False
    kv._rust.store_arm(bytes(kv_cache_utils.NONE_HASH))
    TOK.hash_block_size = hbs
    TOK.armed = True
    log.info("rust_sched: token store armed (hash_block_size=%d)", hbs)
    return True


def _install_async_output() -> None:
    """Skip ``AsyncOutput.get_output``'s two ``tolist()`` calls when the store is live.

    Wraps the class at runtime rather than editing
    ``vllm/v1/worker/gpu/async_utils.py:49-70`` -- no fork patch, no rebuild. Anything the
    fast path cannot express (logprobs, a nans batch) takes stock ``get_output``, and a
    single failure disables the store for the boot.
    """
    from vllm.v1.worker.gpu.async_utils import AsyncOutput

    if already_patched(AsyncOutput, "get_output", patch="rust_sched_tokstore"):
        return
    original = AsyncOutput.get_output

    def get_output(self):
        if TOK.live and self.logprobs_tensors is None and self.num_nans is None:
            try:
                self.copy_event.synchronize()
                mro = self.model_runner_output
                mro.sampled_token_ids = LazySampled(
                    self.sampled_token_ids, self.num_sampled_tokens_np
                )
                mro.prompt_logprobs_dict = self.prompt_logprobs_dict
                return mro
            except BaseException as exc:
                reraise_fatal(exc)
                log.exception("rust_sched: lazy get_output failed; stock conversion")
                TOK.disable("get_output raised")
        return original(self)

    AsyncOutput.get_output = mark_patched(get_output, original, patch="rust_sched_tokstore")


def _install_async_output_event_ring() -> None:
    """One-liner 5d: ``AsyncOutput.__init__`` allocates a fresh ``torch.cuda.Event``
    every decode step just to sleep-wait on it once in ``get_output``; a small
    pre-created ring removes that allocation from the hot path.

    Verbatim copy of the stock body (``vllm/v1/worker/gpu/async_utils.py:13-47``,
    reference only -- never edited) except the event line, implemented as a monkeypatch
    so it needs no fork patch or rebuild. Correctness: ``AsyncModelRunnerOutput`` overlaps
    at most ``max_concurrent_batches`` (2) steps' copies at once; the ring is 4 slots, so
    reusing slot ``i`` again is always at least 2 calls after the one before it recorded
    -- by then that earlier copy has long since been synchronized on. Reusing an event can
    only make a `.synchronize()` wait a little longer for a copy that was already
    in-flight; it can never return before the copy it is meant to guard finishes. Fail
    open: any exception at install time (no CUDA available, a vLLM signature change)
    leaves the stock per-call ``torch.cuda.Event`` in place.
    """
    try:
        import torch

        from vllm.v1.worker.gpu import async_utils

        AsyncOutput = async_utils.AsyncOutput
        if already_patched(AsyncOutput, "__init__", patch="rust_sched_event_ring"):
            return
        original_init = AsyncOutput.__init__

        ring = [torch.cuda.Event(blocking=True) for _ in range(4)]
        counter = [0]

        def __init__(self, model_runner_output, sampler_output, num_sampled_tokens,
                     main_stream, copy_stream):
            self.model_runner_output = model_runner_output
            self.sampler_output = sampler_output
            self.num_sampled_tokens = num_sampled_tokens
            # The only change from stock: a ring slot instead of a fresh allocation.
            self.copy_event = ring[counter[0] % len(ring)]
            counter[0] += 1

            with async_utils.stream(copy_stream, main_stream):
                copy_stream.wait_stream(main_stream)
                self.sampled_token_ids = async_utils.async_copy_to_np(
                    sampler_output.sampled_token_ids
                )
                self.logprobs_tensors = None
                if sampler_output.logprobs_tensors is not None:
                    self.logprobs_tensors = (
                        sampler_output.logprobs_tensors.to_cpu_nonblocking()
                    )
                self.num_nans = None
                if sampler_output.num_nans is not None:
                    self.num_nans = async_utils.async_copy_to_np(sampler_output.num_nans)
                self.num_sampled_tokens_np = async_utils.async_copy_to_np(num_sampled_tokens)
                self.prompt_logprobs_dict = {
                    k: v.to_cpu_nonblocking() if v is not None else None
                    for k, v in self.model_runner_output.prompt_logprobs_dict.items()
                }
                self.copy_event.record(copy_stream)

        AsyncOutput.__init__ = mark_patched(
            __init__, original_init, patch="rust_sched_event_ring"
        )
        log.info("rust_sched: AsyncOutput event ring installed (4 pre-created events)")
    except BaseException as exc:
        reraise_fatal(exc)
        log.exception(
            "rust_sched: AsyncOutput event ring not installed; stock per-call event"
        )


def _install_update_from_output(scheduler_cls, m: dict):
    """R6a -- one Rust call per step for the whole batch's stop decision.

    THE SEAM, and why it is this one. ``update_from_output`` is 350 lines of
    ``EngineCoreOutput`` assembly, connector bookkeeping and queue surgery, all of it over
    Python objects that cannot cross the boundary. The only part that is arithmetic is
    ``check_stop`` (utils.py:94), reached through the single narrow method
    ``_update_request_with_output`` -- which is also the method ``AsyncScheduler``
    overrides, so patching the base class covers both schedulers via its ``super()`` call.

    So: the outer wrapper reads the whole step's sampled tokens BEFORE the loop and asks
    Rust once; the inner wrapper then applies the precomputed verdict instead of calling
    ``check_stop``. Per-request crossings would cost more than the six integer comparisons
    they replace -- that is the entire reason for the batch shape.

    FAIL-CLOSED AT EVERY JOINT. A request with no interned stop params, a
    ``repetition_detection`` / structured-output / pooling request, a spec-decode step, or
    a ``new_token_ids`` list whose length does not match what the batch was computed from,
    all fall through to stock ``check_stop`` for that request alone.
    """
    import time

    from vllm.v1.core.sched.utils import remove_all
    from vllm.v1.request import RequestStatus

    spec = m["spec"]
    # Port-2. STORE gates the crate-side bookkeeping; TOK.live gates the numpy fast path
    # and the facade, and is the flag that a failure turns off for the boot.
    STORE = m["tokstore"]
    TOK.live = STORE

    # R9: the collapsed FFI + residue loop. `R9.live` decides whether decide() folds
    # `cache_blocks` in and whether `update_from_output` uses the residue loop instead of
    # `r8_apply`. It lives on the module singleton, not a closure cell, because
    # `R9.disable()` must be reachable from inside `decide()`'s per-step checks and stay
    # disabled for the rest of the boot, same contract as `TOK.disable`.
    R9.live = m["r9"]

    # R8 emits `bytes` onto the output queue, which ONLY `shm_ipc`'s replacement output
    # thread understands -- stock's would do `outputs.engine_index = ...` on a bytes and
    # take EngineCore down. The env gates are not proof that module installed
    # (VTL_ENABLE_SHM_IPC=0 disables the registry entry while VTL_SHM_IPC=1 is still set),
    # and shm_ipc applies AFTER rust_sched, so the check is resolved lazily on the first
    # step -- by which time every patch has run. Fail closed.
    _r8 = [m["r8"], False]

    def r8_live() -> bool:
        if not _r8[1]:
            _r8[1] = True
            if _r8[0]:
                try:
                    from vtl.patches import shm_ipc

                    ok = bool(getattr(shm_ipc, "RAW_OUTPUT_PATH", False))
                except Exception:
                    ok = False
                if not ok:
                    log.warning(
                        "rust_sched: R8 disabled -- the shm raw output path is not "
                        "installed, so nothing on the output queue could read the bytes"
                    )
                _r8[0] = ok
        return _r8[0]

    # Batch 3: inline-publish ordering guard. `shm_ipc.PUB` holds two single-writer
    # counters -- `enqueued` bumped here (the engine thread) whenever a step's output goes
    # to the queue, `published` bumped by the output thread after each queue item is
    # actually sent. Publishing straight from the crate is only safe when the two agree
    # (no backlog): otherwise an inline publish could overtake an item still in flight
    # through the queue and reorder the wire stream. Resolved EAGERLY (unlike `r8_live`,
    # which waits on `shm_ipc.apply()`'s state) because `PUB` is a plain module singleton
    # that exists the moment the module imports -- the two counters must start from the
    # same step or their gap never closes.
    _pub_mod = None
    if env_on("VTL_SHM_IPC_RUST_PUB"):
        try:
            from vtl.patches import shm_ipc as _pub_mod
        except Exception:
            log.exception("rust_sched: shm_ipc not importable; VTL_SHM_IPC_RUST_PUB inert")
            _pub_mod = None

    def bump_enqueued() -> None:
        if _pub_mod is not None:
            _pub_mod.PUB.enqueued += 1

    # `out_is_open()` is monotone -- `out_open` only ever sets `Some` (python.rs), nothing
    # closes it -- so latch the first True and stop paying an FFI call plus the shared
    # read-lock every step.
    _out_open_latched = False

    def out_publish_ready(rust) -> bool:
        nonlocal _out_open_latched
        if _pub_mod is None:
            return False
        if not _out_open_latched:
            try:
                if not rust.out_is_open():
                    return False
            except BaseException as exc:
                reraise_fatal(exc)
                return False
            _out_open_latched = True
        return _pub_mod.PUB.enqueued == _pub_mod.PUB.published

    # The Rust steady-state runner's scheduler half. Resolved here rather than in `modes()`
    # because it needs `rust_runner.mode()` (which defaults ON) and, more importantly,
    # because the stash it writes is only applicable on exactly the boots this function's
    # R9 residue path is live on -- the runner commits through `step_pack_locked`, the same
    # crate entry point `update_step_pack_np` takes, so `r9_apply` is what applies it.
    RUNNER = m["runner"]
    runner_mod = None
    if RUNNER:
        try:
            from vtl.patches import rust_runner as runner_mod

            RUNNER = runner_mod.mode() == "on"
        except Exception:
            log.exception("rust_sched: rust_runner not importable; no runner handshake")
            RUNNER = False

    wrapped_ufo = scheduler_cls.update_from_output
    wrapped_urwo = scheduler_cls._update_request_with_output
    monotonic = time.monotonic

    # Mirrors update.rs's status codes; 255 = "Rust has no params for this slot".
    _STATUS_FROM_CODE = {
        1: RequestStatus.FINISHED_STOPPED,
        2: RequestStatus.FINISHED_LENGTH_CAPPED,
    }

    def register(rust, request, slot) -> bool:
        """Intern the immutable half of this request's stop condition. False = refuse."""
        sp = request.sampling_params
        if sp is None or request.pooling_params is not None:
            return False
        # check_stop's two unported branches. `repetition_detection` is an O(n) scan over
        # the whole output with its own tuning knobs; a structured-output request can be
        # terminated by the grammar, which lives entirely in Python.
        if getattr(sp, "repetition_detection", None) is not None:
            return False
        if getattr(request, "structured_output_request", None) is not None:
            return False
        # A resumable streaming session can be stopped, resumed via
        # _update_request_as_session with REPLACED sampling_params, and keep its
        # request_id -- so the interned "immutable half" would go stale on the same
        # slot. Not judge traffic; refuse rather than model re-registration.
        if getattr(request, "resumable", False):
            return False
        max_tokens = getattr(request, "max_tokens", None)
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            return False
        eos = getattr(sp, "eos_token_id", None)
        rust.set_stop_params(
            slot,
            int(getattr(sp, "min_tokens", 0) or 0),
            max_tokens,
            None if eos is None else int(eos),
            [int(t) for t in (sp.stop_token_ids or ())],
        )
        return True

    def store_take_over(kv, request, slot) -> bool:
        """Give this slot's token bookkeeping to Rust. ``False`` = it stays in Python.

        Refusing is cheap and per REQUEST: the step then takes the object path (and any
        request the store already owns is materialized first), so a single unsupported
        request never costs correctness -- only the fast path, for that step.
        """
        if getattr(request, "_vtl_tok_off", False):
            return False
        try:
            if not tok_arm(kv):
                return False
            if not tok_store_init(kv._rust, request, slot, TOK.hash_block_size):
                request._vtl_tok_off = True
                return False
        except BaseException as exc:
            reraise_fatal(exc)
            log.exception("rust_sched: token store take-over failed for %s",
                          request.request_id)
            TOK.disable("store_init raised")
            request._vtl_tok_off = True
            return False
        request._vtl_tok_on = True
        if TOK.live:
            tok_facade_on(request)
        return True

    def step_packable_here(self, scheduler_output) -> bool:
        """The half of ``step_packable`` that does not need the ModelRunnerOutput.

        Split out so the Rust runner's commit-time stash can ask the same question one step
        EARLIER, at schedule time, when the mro does not exist yet. The clauses left to
        ``step_packable`` are all properties of a step's output, and the runner's stash gate
        excludes every one of them by other means: the burst gate refuses logprobs and
        structured output, `_emit` builds no pooler output or routed experts, and
        `kv_connector_output` is `None` exactly when `self.connector is None` (checked
        here).
        """
        return (
            not scheduler_output.has_structured_output_requests
            and not self.log_stats
            and self.connector is None
            and self.finished_req_ids_dict is None
            and not self.defer_block_free
            and not self.enable_kv_cache_events
            and not self.enable_return_routed_experts
            and (self.perf_metrics is None or not self.perf_metrics.is_enabled())
        )

    def step_packable(self, scheduler_output, mro) -> bool:
        """R8 gate: can this whole step's OUTPUT be a single Rust-built raw record?

        Everything the fixed layout cannot express, evaluated on what Rust cannot see.
        Mirrors ``shm_ipc.raw_packable``'s reject list plus the parts of
        ``update_from_output`` that would otherwise be skipped: connector bookkeeping,
        stats, structured output, routed experts, prompt logprobs. A `False` here costs
        nothing -- the step takes the R6a path it took before R8 existed.

        `finished_req_ids_dict is None` is the single-engine case, which is also the only
        case where the record's `finished_requests` table is guaranteed empty; a
        multi-engine boot keeps the stock object path rather than guessing at it.
        """
        return (
            not mro.logprobs
            and not mro.prompt_logprobs_dict
            and not mro.pooler_output
            and mro.num_nans_in_logits is None
            and mro.kv_connector_output is None
            and mro.routed_experts is None
            and mro.cudagraph_stats is None
            and step_packable_here(self, scheduler_output)
        )

    def runner_stash(self, kv, scheduler_output, by_slot, slots, n, steps) -> int:
        """Promise the Rust runner this step. Returns the burst launches GRANTED (0 = none,
        so the step is committed for one burst the worker replays itself). Called from
        ``commit_burst``.

        WHAT THE PROMISE IS: "if you launch, ``step_pack_locked`` will pack, and the
        residue will be applied by ``r9_apply``". So every predicate ``decide()``'s lazy R9
        branch would have checked is checked HERE instead, one step earlier -- the runner
        cannot fall back mid-step, because by then the graph has already fed its tokens
        back into the persistent input buffers.

        The per-slot half is two set lookups: ``mirror._stops`` (the slot has interned stop
        params, so the crate can decide the stop rather than returning 255) and
        ``mirror._pack_ok`` (the six clauses the R8 record has no field for). Both are
        populated by ``decide()`` on an earlier step, so the first decode step of a request
        never stashes -- which is also when its counters are least likely to be interned.

        ``facaded`` is the third: ``r9_apply``'s counter writes ARE the facade branch of
        ``_urwo_inner``, and the crate's token store is what holds the tokens the launch
        will append.
        """
        state = runner_mod.STATE
        if not state.live or state.step is not None or state.done is not None:
            return 0
        if not (R9.live and TOK.live and r8_live()):
            return 0
        if not step_packable_here(self, scheduler_output):
            return 0
        mirror = kv._mirror
        reqs = []
        for slot in slots:
            if slot not in mirror._stops or slot not in mirror._pack_ok:
                return 0
            request = by_slot[slot]
            if not facaded(request):
                return 0
            reqs.append(request)
        publish = out_publish_ready(kv._rust)
        if steps > 1 and not publish:
            # `update_from_output` hands back ONE record per engine index, so the second
            # launch's record would have nowhere to go. With the inline publish live each
            # launch delivers its own and there is nothing left to return -- which is why
            # k > 1 is gated on it instead of on a queue of records.
            steps = 1
        state.step = runner_mod._Stash(
            # From the SLOT order, so the launch site's `key == tuple(ib.req_ids)` check
            # doubles as the proof that slots[i] is the slot of batch row i.
            key=tuple(r.request_id for r in reqs),
            slots=tuple(slots),
            reqs=tuple(reqs),
            n=n,
            steps=steps,
            max_model_len=int(self.max_model_len),
            fold=True,               # == R9.live, checked above
            publish=publish,
        )
        return steps

    def runner_consume(self, done) -> None:
        """Turn the runner's ``_Done`` into exactly the state ``r9_apply`` reads.

        Rust already took the crossing (``step_pack_locked`` IS ``update_step_pack_np``'s
        locked body: token store, resident delta, R8 record, inline publish), so
        ``decide()`` must not run -- it would append the same tokens a second time. What is
        left is the residue loop, which is ``r9_apply`` unchanged.
        """
        stash = done.stash
        n, ran = stash.n, done.ran
        # `run_steps` breaks on the FIRST non-zero verdict, so every launch before the last
        # one accepted the full width; only the last one's verdicts can be short.
        base = (ran - 1) * n
        verdicts = done.verdicts
        # Deferred to the end of the function: `refuse` RAISES under
        # VTL_RUST_RUNNER_REQUIRE, and the state below has to be applied first either way.
        stand_down = None
        if done.exit == "unpacked":
            # Should-never-happen: the crate refuses the pre-flight rather than reach here.
            # The failed launch appended NOTHING to the token store (`store_apply` runs
            # only on a packed step) but did advance the resident table, so drop that
            # launch from the residue, force a table resync, and let `burst_uncommit` give
            # its tokens back -- the same reconcile a stopped burst uses.
            log.error("rust_sched: the runner could not pack launch %d; standing down", ran)
            stand_down = "a launch did not pack"
            self._vtl_ufo_clean = False
            verdicts = [(0, 0, -1)] * len(stash.slots)
            base = max(0, base - n)
        self._vtl_r9_residue = [
            (slot, req, (base + acc, status, stop))
            for slot, req, (acc, status, stop) in zip(stash.slots, stash.reqs, verdicts)
        ]
        # Read off the STASH, not off `scheduler_output.vtl_burst_n`: the stash is what the
        # tokens were committed against, so the shortfall stays right either way.
        self._vtl_burst_n = n * stash.steps
        records = done.records
        self._vtl_r8_record = records[0] if records else None
        self._vtl_r8_published = not records
        if len(records) > 1:
            # `update_from_output` returns ONE record per engine index, so there is nowhere
            # to put the rest. Cannot happen: a multi-launch commit requires the inline
            # publish. If it ever does, say so and stand down rather than drop outputs on
            # every later step too.
            log.error(
                "rust_sched: the runner returned %d unpublished records for one step; "
                "%d cannot be delivered", len(records), len(records) - 1,
            )
            stand_down = "multi-record step without the inline publish"
        if stand_down is not None:
            runner_mod.STATE.refuse(stand_down)

    def decide(self, kv, scheduler_output, model_runner_output, pack: bool):
        """Build the flat batch and take the single crossing. None = nothing portable.

        With ``pack``, the crossing is ``update_step_pack``, which also returns the whole
        step's shm output record; the bytes land in ``self._vtl_r8_record`` (``None`` when
        Rust refused to pack). The verdicts are identical either way.

        With the TOKEN STORE live (Port-2) the crossing is ``update_step_pack_np`` instead:
        the sampler's numpy array goes straight over, the per-request counters come off the
        store, and no token id is ever turned into a Python int. ``lazy`` goes False the
        moment anything in the batch cannot ride that path, and the tail then materializes
        the step back onto the object path before it hands over.

        R9 (needs the token store): ``mirror._batch_cache`` remembers the last step's
        ``(slots, rows, requests)`` keyed on IDENTITY of ``req_id_to_index`` (the
        hotpath_microopt fork patch keeps that dict the SAME OBJECT across repeat decode
        steps). A hit skips the whole per-request loop below -- one numpy gather off the
        sampler's own count array stands in for it -- and folds each slot's ``cache_blocks``
        into the crossing (``fold_cache=True``) instead of a per-request FFI afterwards.
        """
        rust, mirror = kv._rust, kv._mirror
        sampled = model_runner_output.sampled_token_ids
        lazy = TOK.live and pack and isinstance(sampled, LazySampled)

        if R9.live and lazy:
            cache = mirror._batch_cache
            counts = _r9_cache_hit_counts(
                cache, model_runner_output.req_id_to_index, sampled.counts
            )
            # `counts is None` covers every miss reason at once: no cache, an identity
            # mismatch, or a prefill chunk (zero-count row) in an otherwise-identical
            # batch -- fall through to the full loop rather than guess which row it was.
            if counts is not None:
                _, c_slots, c_rows, c_requests = cache
                out, record, published = rust.update_step_pack_np(
                    sampled.arr, c_rows, counts, c_slots, int(self.max_model_len),
                    0, monotonic(), (), fold_cache=True, publish=out_publish_ready(rust),
                )
                self._vtl_r8_record = record
                self._vtl_r8_published = published
                _r9_maybe_check_fold_skips(rust)
                if record is not None or published:
                    self._vtl_r9_residue = list(zip(c_slots, c_requests, out))
                    return {}
                # Rust refused the record for a reason neither the identity nor the
                # zero-count check anticipated (e.g. a slot lost its interned name). The
                # verdicts are still real -- `wrapped_ufo`'s per-request fallback must see
                # them, same contract as the miss path below -- but the cache itself is
                # suspect, so drop it and re-derive from scratch.
                mirror._batch_cache = None
                return {
                    r.request_id: (c, *v)
                    for r, c, v in zip(c_requests, counts, out)
                }

        slots: list[int] = []
        rows: list[int] = []
        counts: list[int] = []
        n_out: list[int] = []
        n_tok: list[int] = []
        expect: list[tuple] = []
        for req_id, index in model_runner_output.req_id_to_index.items():
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue
            gen = sampled[index]
            n_gen = len(gen)
            if not n_gen:
                # No token for this request this step (a prefill chunk). Nothing to apply,
                # and nothing the resident table could have missed.
                continue
            if getattr(request, "async_tokens_to_discard", 0):
                # A spec-decode rollback rewinds num_tokens behind Rust's back.
                self._vtl_ufo_clean = False
                pack = False
                continue
            slot = mirror.slot(request)
            if pack:
                # R9: the six clauses are all request-immutable once true (see
                # `RustMirror._pack_ok`'s docstring on why `prefill_stats` -- the one
                # mutable member -- is still safe to intern), so a proven-packable slot
                # is a set lookup instead of six attribute reads every step.
                if slot in mirror._pack_ok:
                    pass
                elif _pack_ok_clauses(request):
                    mirror._pack_ok.add(slot)
                else:
                    # The record has no slot for any of these: a second client index, a
                    # resumable session's re-enqueue, encoder-cache frees, trace headers,
                    # prefill stats or lifecycle events. Objects for the whole step.
                    pack = False
            # Python-side registry, same rationale as RustMirror._pushed: asking Rust
            # `has_stop_params` here was one FFI crossing per request per step to re-learn
            # a fact only this module changes. register() overwrites, so a lost entry
            # (slot recycled -> discarded in mirror.drop) just re-registers.
            if slot not in mirror._stops:
                if not register(rust, request, slot):
                    self._vtl_ufo_clean = False
                    pack = False
                    continue
                mirror._stops.add(slot)
            if STORE and not getattr(request, "_vtl_tok_on", False) \
                    and not store_take_over(kv, request, slot):
                # This request's tokens stay in Python, so the whole step does: the numpy
                # crossing reads its counters off the store and there would be no entry.
                lazy = False
            slots.append(slot)
            rows.append(index)
            counts.append(n_gen)
            expect.append((req_id, n_gen, request))
            # Read BEFORE anything appends -- exactly the values check_stop would see on
            # its first iteration. Ignored by the numpy arm (the store holds them).
            n_out.append(request.num_output_tokens)
            n_tok.append(request.num_tokens)
        if not slots:
            return None
        if lazy and pack:
            req_only = None
            if R9.live:
                req_only = [r for _rid, _n, r in expect]
                mirror._batch_cache = (model_runner_output.req_id_to_index, slots, rows, req_only)
            # Inline publish is scoped to the R9 residue path (below): only `r9_apply`
            # checks `_vtl_r8_published` to skip the queue put, so attempting it while R9
            # is off would strand the record (never queued, never published in the sense
            # `r8_apply` understands). `and` short-circuits, so `out_publish_ready` (and
            # the `shm_ipc` import it triggers) never runs on a non-R9 boot.
            want_publish = R9.live and out_publish_ready(rust)
            out, record, published = rust.update_step_pack_np(
                sampled.arr, rows, counts, slots, int(self.max_model_len),
                0, monotonic(), (), fold_cache=R9.live, publish=want_publish,
            )
            self._vtl_r8_record = record
            self._vtl_r8_published = published
            if R9.live:
                _r9_maybe_check_fold_skips(rust)
                if record is not None or published:
                    self._vtl_r9_residue = list(zip(slots, req_only, out))
            # `record is None and not published` (Rust refused the pack) is NOT handled here on purpose:
            # `update_from_output` then takes the `wrapped_ufo` arm, whose
            # `store_full_fallback` guard is strictly wider than anything this loop saw.
            # Rust appended nothing to the store on that arm, so Python owns the append.
            return {rid: (n, *v) for (rid, n, _r), v in zip(expect, out)}
        if STORE:
            # Objects for this step: stock code is about to read `sampled[index]` and the
            # requests' real token lists, so both have to exist first.
            store_step_fallback(kv, sampled, expect, "step is not numpy-packable")
        cu: list[int] = [0]
        toks: list[int] = []
        for row, cnt in zip(rows, counts):
            gen = sampled[row]
            toks.extend(gen if len(gen) == cnt else gen[:cnt])
            cu.append(len(toks))
        if pack:
            # engine_index is 0 for every non-DP boot (EngineCoreProc's default; only
            # DPEngineCoreProc passes the rank), and step_packable() already refused the
            # multi-engine case via finished_req_ids_dict. finished stays empty for the
            # same reason -- `EngineCoreOutputs.finished_requests` is only ever filled
            # from finished_req_ids_dict.
            out, record = rust.update_step_pack(
                slots, cu, toks, n_out, n_tok, int(self.max_model_len),
                0, monotonic(), (),
            )
            self._vtl_r8_record = record
        else:
            out = rust.update_step(slots, cu, toks, n_out, n_tok, int(self.max_model_len))
        return {rid: (n, *v) for (rid, n, _r), v in zip(expect, out)}

    def r8_apply(self, model_runner_output):
        """``update_from_output``'s per-request bookkeeping, WITHOUT the output assembly.

        The 90 lines this replaces (scheduler.py:1738-1822 and the branches feeding them)
        are pure `EngineCoreOutput` / `EngineCoreOutputs` construction plus the connector,
        stats, logprobs, routed-expert and structured-output arms that ``step_packable``
        has already proved inert for this step. What survives is exactly the state
        mutation: append the tokens (via the R6a-decided ``_update_request_with_output``),
        retire stopped requests, and drop them from the two queues.

        Returns ``{0: <record bytes>}``; ``EngineCoreProc._process_engine_step`` puts the
        pair on the output queue verbatim and ``shm_ipc._process_output_sockets``
        recognises the ``bytes`` payload. The queue is already typed for it (core.py:916).
        """
        sampled = model_runner_output.sampled_token_ids
        stopped_running = None
        stopped_preempted = None
        for req_id, index in model_runner_output.req_id_to_index.items():
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue
            gen = sampled[index]
            if not gen:
                # A prefill chunk: no tokens, so stock emits no EngineCoreOutput either.
                continue
            status_before = request.status
            _, is_stopped = self._update_request_with_output(request, gen)
            if not is_stopped:
                continue
            # `resumable` is refused by decide(), so _handle_stopped_request is the
            # constant-True branch; call it anyway so a future resumable request cannot
            # silently take a path that skips the streaming re-enqueue.
            if self._handle_stopped_request(request):
                self._free_request(request)
            if status_before == RequestStatus.RUNNING:
                if stopped_running is None:
                    stopped_running = set()
                stopped_running.add(request)
            else:
                if stopped_preempted is None:
                    stopped_preempted = set()
                stopped_preempted.add(request)
        if stopped_running:
            self.running = remove_all(self.running, stopped_running)
        if stopped_preempted:
            self.waiting.remove_requests(stopped_preempted)
        return {0: self._vtl_r8_record}

    def r9_apply(self, residue):
        """R9: the residue loop replacing ``r8_apply`` when ``decide()`` took the
        collapsed FFI path (fast-hit OR the miss path's lazy branch, both set
        ``self._vtl_r9_residue``).

        ``residue`` is ``list[(slot, request, (num_accepted, status, stop_reason))]``,
        built by ``decide()`` straight from arrays it already had -- no re-walk of
        ``req_id_to_index``, no ``sampled[index]`` lookup, no
        ``_update_request_with_output`` call. The four attribute writes are the facade
        branch of ``_urwo_inner`` (the fold already ran ``cache_blocks``, so that
        per-request FFI is gone too); the burst reconcile is ``reconcile_burst`` verbatim,
        read off the verdict instead of re-derived by calling ``_update_request_with_output``;
        the finished-request bookkeeping is ``r8_apply`` verbatim.
        """
        burst_n = self._vtl_burst_n
        stopped_running = None
        stopped_preempted = None
        for slot, request, (acc, status, stop_reason) in residue:
            request.num_tokens += acc
            request.num_output_tokens += acc
            request.num_output_placeholders = max(0, request.num_output_placeholders - acc)
            # `reconcile_burst`'s arithmetic, inlined: a burst that stopped short (or
            # whose runner produced fewer tokens than committed) gives the surplus back
            # to both the request and the resident table.
            short = burst_n - acc
            if short > 0:
                # Resolved here, not before the loop: `short > 0` is the rare arm, so the
                # two attribute lookups stay off the every-step path.
                burst_uncommit(request, short)
                self.kv_cache_manager._rust.table_burst([slot], -short)
            if status == 0:
                continue
            status_before = request.status
            request.status = _STATUS_FROM_CODE[status]
            if stop_reason >= 0:
                request.stop_reason = stop_reason
            # `resumable` is refused by decide(), so _handle_stopped_request is the
            # constant-True branch; call it anyway so a future resumable request cannot
            # silently take a path that skips the streaming re-enqueue.
            if self._handle_stopped_request(request):
                self._free_request(request)
            if status_before == RequestStatus.RUNNING:
                if stopped_running is None:
                    stopped_running = set()
                stopped_running.add(request)
            else:
                if stopped_preempted is None:
                    stopped_preempted = set()
                stopped_preempted.add(request)
        if stopped_running:
            self.running = remove_all(self.running, stopped_running)
        if stopped_preempted:
            self.waiting.remove_requests(stopped_preempted)
        # Batch 3: `decide()` already delivered the record straight from the crate --
        # nothing left to queue. `EngineCoreProc._process_engine_step` does
        # `for output in outputs.items() if outputs else ()`, so an EMPTY dict here means
        # the step's output never reaches `output_queue.put_nowait` at all.
        if self._vtl_r8_published:
            return {}
        return {0: self._vtl_r8_record}

    def maybe_kick(self, kv):
        """Arm the next step's speculation. Called once wrapped_ufo has returned, so every
        ``_free_request`` -> ``kv.free`` of this step has already landed in the Rust pool.

        Refuses on anything the worker cannot model: a non-empty waiting queue (spec.rs
        invariant 3 / journal.rs's scope invariant), a dirty table, a bail condition, or a
        step where any request fell back to Python's check_stop.
        """
        tbl = getattr(kv, "_vtl_table", None)
        core = getattr(self, "_vtl_rust_core", None)
        if tbl is None or core is None or tbl.off:
            return
        if not self._vtl_ufo_clean:
            tbl.resync("a UFO per-request fallback ran this step")
            return
        if tbl.dirty or self.waiting or bail_reason(self) is not None:
            return
        try:
            # `mirror.slot`, not a bare lookup: the tokens this step just appended may
            # have completed a block, and its hash ALREADY exists -- produced by Python's
            # hasher inside append_output_token_ids, or (Port-2) by `tokens.rs` inside
            # update_step_pack_np, in which case `slot()` pushes nothing because the crate
            # already holds the chain. Either way the invariant this call protects holds.
            # Pushing it here rather than at the
            # next schedule() is what makes speculation survive a block boundary -- the
            # worker would otherwise walk a table claiming more full blocks than the crate
            # has hashes for, and the next schedule() would invalidate the result anyway.
            # self.running is the post-update order, i.e. the one schedule() will marshal.
            mirror = kv._mirror
            slots = [mirror.slot(r) for r in self.running]
        except BaseException as exc:
            tbl.fail("kick slot refresh", exc)
            return
        try:
            tbl.armed = bool(core.kick(kv._rust, tbl.gen, slots))
        except BaseException as exc:
            tbl.fail("kick", exc)

    def update_from_output(self, scheduler_output, model_runner_output):
        self._vtl_ufo = None
        self._vtl_r8_record = None
        self._vtl_r8_published = False
        self._vtl_r9_residue = None
        # C1a side channel: how many tokens THIS step's schedule committed to. Read once
        # per step (not per request) so the reconcile stays FIFO-correct under async
        # scheduling, where two steps are in flight and each has its own factor.
        self._vtl_burst_n = getattr(scheduler_output, "vtl_burst_n", 1) or 1
        # False the moment ANY request in this step's batch was decided by Python instead
        # of by update_step -- which is exactly when the resident table missed a token
        # delta and must be resynced before it can be scheduled from again.
        self._vtl_ufo_clean = True
        kv = self.kv_cache_manager
        done = None
        if RUNNER:
            # THE ONE PLACE a scheduled step is counted as applied. `inflight` is the
            # runner's launch interlock -- Rust commits at sample time and Python commits
            # here, so a launch while an earlier step is still unapplied would order this
            # step's tokens ahead of that one's (rust_runner's module docstring).
            state = runner_mod.STATE
            state.step = None
            state.inflight = max(0, state.inflight - 1)
            done = state.done
        sampled = model_runner_output.sampled_token_ids
        if done is not None:
            # Rust ran and committed this step. Skip decide() -- it would append the same
            # tokens again -- and hand `r9_apply` the residue it already knows how to take.
            runner_consume(self, done)
        elif (
            hasattr(kv, "_rust")
            and sampled
            and not scheduler_output.scheduled_spec_decode_tokens
            and not model_runner_output.pooler_output
        ):
            pack = r8_live() and step_packable(self, scheduler_output, model_runner_output)
            try:
                self._vtl_ufo = decide(
                    self, kv, scheduler_output, model_runner_output, pack
                )
            except BaseException as exc:
                reraise_fatal(exc)
                log.exception("rust_sched: UFO batch failed; this step uses check_stop")
                self._vtl_ufo = None
                self._vtl_r8_record = None
                self._vtl_r8_published = False
                self._vtl_r9_residue = None
                # An exception mid-derivation may have left the cache half-built or
                # pointed at requests this step is about to fall back on.
                mirror = getattr(kv, "_mirror", None)
                if mirror is not None:
                    mirror._batch_cache = None
            if self._vtl_ufo is None:
                self._vtl_ufo_clean = False
        elif sampled:
            # Tokens were produced but nothing was portable: stock check_stop ran for the
            # whole batch, so update_step applied no delta at all.
            self._vtl_ufo_clean = False
        try:
            # `or self._vtl_r8_published`: Batch 3's inline-delivered case leaves the
            # record itself `None` (nothing left to copy into a `PyBytes`), so the record
            # alone no longer distinguishes "packed" from "refused" -- see `decide()`.
            if self._vtl_r8_record is not None or self._vtl_r8_published:
                if self._vtl_r9_residue is not None:
                    outputs = r9_apply(self, self._vtl_r9_residue)
                else:
                    outputs = r8_apply(self, model_runner_output)
            else:
                if TOK.live:
                    # THE COMPLETE GUARD. Stock's loop walks
                    # `scheduler_output.num_scheduled_tokens`, which is a SUPERSET of what
                    # decide() looked at -- an already-finished request, or a step decide()
                    # refused outright (`_vtl_ufo is None`), still reaches
                    # `sampled_token_ids[index]` and `request.num_output_tokens` here. So
                    # every store-owned request in the step goes back to stock, not just the
                    # ones decide() enumerated.
                    store_full_fallback(
                        self, kv, scheduler_output, model_runner_output
                    )
                outputs = wrapped_ufo(self, scheduler_output, model_runner_output)
        finally:
            self._vtl_ufo = None
            self._vtl_r8_record = None
            self._vtl_r8_published = False
            self._vtl_r9_residue = None
            self._vtl_burst_n = 1
            if RUNNER:
                # Applied (or dropped, if the branch above raised): either way this step's
                # result must not be seen again, and the next launch is unblocked.
                state.done = None
        if spec:
            maybe_kick(self, kv)
        if outputs:
            # Batch 3 ordering guard: count every item this step hands to
            # `EngineCoreProc._process_engine_step` for `output_queue.put_nowait` -- an
            # inline-published step returns `{}` above and is correctly NOT counted here.
            bump_enqueued()
        return outputs

    def reconcile_burst(self, request, kept: int) -> None:
        """Hand back the tokens a committed burst did not produce. See ``burst_uncommit``.

        Called from ``_update_request_with_output`` -- i.e. INSIDE
        ``AsyncScheduler._update_request_with_output``, before its own placeholder
        subtraction and its ``cache_blocks`` call, which is what makes the cached prefix
        length come out exact on both the stop and the (should-never-happen) short-runner
        path.
        """
        short = self._vtl_burst_n - kept
        if short <= 0:
            return
        burst_uncommit(request, short)
        kv = self.kv_cache_manager
        slot = kv._mirror._slots.get(request.request_id)
        if slot is not None:
            # Same correction into the resident table. No resync: table_burst leaves the
            # entry consistent with the Python object, which is the whole contract.
            kv._rust.table_burst([slot], -short)

    def _update_request_with_output(self, request, new_token_ids):
        if self._vtl_burst_n > 1:
            kept, stopped = _urwo_inner(self, request, new_token_ids)
            reconcile_burst(self, request, len(kept))
            return kept, stopped
        return _urwo_inner(self, request, new_token_ids)

    def _urwo_inner(self, request, new_token_ids):
        decided = self._vtl_ufo
        answer = decided.pop(request.request_id, None) if decided else None
        # 255 = unregistered slot; a length mismatch means something mutated the request
        # between the batch and here. Either way, stock decides.
        if answer is None or answer[2] == 255 or answer[0] != len(new_token_ids):
            # Rust left this slot's table entry untouched; Python owns the delta now -- so
            # the request needs its real token lists back before stock touches it.
            self._vtl_ufo_clean = False
            if facaded(request):
                tok_materialize(self.kv_cache_manager, request)
            if isinstance(new_token_ids, _Row):
                new_token_ids = new_token_ids.tolist()
            return wrapped_urwo(self, request, new_token_ids)
        _, num_keep, status, stop_reason = answer

        if facaded(request):
            # Port-2: the tokens and the hash chain are already in Rust (appended by
            # `update_step_pack_np` for exactly `num_keep`), so the whole writer is two
            # counter bumps -- `num_tokens_with_spec` is a facade property aliasing
            # `num_tokens` and needs no write of its own.
            request.num_tokens += num_keep
            request.num_output_tokens += num_keep
        else:
            # One list-extend + ONE block-hasher catch-up call (the hasher loops over every
            # complete block itself), not num_keep incremental appends: at burst N=4 this
            # deletes 3 hasher invocations per request per step. Skip the slice copy in the
            # common no-truncation case.
            request.append_output_token_ids(
                new_token_ids if num_keep == len(new_token_ids) else new_token_ids[:num_keep]
            )
        if status == 0:
            return new_token_ids, False
        if isinstance(new_token_ids, _Row):
            new_token_ids.n = num_keep      # the `_Row` form of `del ...[num_keep:]`
        else:
            del new_token_ids[num_keep:]
        request.status = _STATUS_FROM_CODE[status]
        if stop_reason >= 0:
            request.stop_reason = stop_reason
        return new_token_ids, True

    # Class-level defaults: `_update_request_with_output` is also reachable from
    # `_update_request_as_session`, i.e. without update_from_output having run first.
    scheduler_cls._vtl_ufo = None
    scheduler_cls._vtl_ufo_clean = True
    scheduler_cls._vtl_r8_record = None
    scheduler_cls._vtl_r8_published = False
    scheduler_cls._vtl_r9_residue = None
    scheduler_cls._vtl_burst_n = 1
    mark_patched(update_from_output, wrapped_ufo, patch="rust_sched_ufo")
    mark_patched(_update_request_with_output, wrapped_urwo, patch="rust_sched_ufo")
    scheduler_cls.update_from_output = update_from_output
    scheduler_cls._update_request_with_output = _update_request_with_output
    if RUNNER:
        # `commit_burst` lives in `_install_full_schedule`'s closure, which cannot see
        # `r8_live` / `out_publish_ready` / this function's R9 view. Publishing the stash
        # writer on the class is the one narrow seam between the two installs; its absence
        # is also how `commit_burst` knows the runner handshake is not available.
        scheduler_cls._vtl_runner_stash = runner_stash
    if STORE:
        _install_preempt_hook(scheduler_cls)
        if TOK.live:
            _install_async_output()
    log.info(
        "rust_sched: UFO batched stop decision active "
        "(kick=%s, r8=%s, tokstore=%s, r9=%s, runner=%s)",
        spec, _r8[0], STORE, R9.live, RUNNER,
    )


def _install_preempt_hook(scheduler_cls) -> None:
    """Materialize before ANY preemption, wherever it comes from.

    Preemption is the one transition that turns a running (possibly store-owned) request
    back into a waiting one, and a resumed request is re-admitted through
    ``NewRequestData.prefill_token_ids`` -- the one consumer that needs the real ints.
    ``_preempt_request`` is the single chokepoint stock vLLM routes every preemption
    through, so hooking it covers the Rust loop's bail path and ``reset_prefix_cache``'s
    forced preemption as well as the ordinary one.
    """
    if already_patched(scheduler_cls, "_preempt_request", patch="rust_sched_tokstore"):
        return
    wrapped = scheduler_cls._preempt_request

    def _preempt_request(self, request, *args, **kwargs):
        if facaded(request):
            tok_materialize(self.kv_cache_manager, request)
        # R9: a preempted request may be part of the cached batch (or about to be
        # re-admitted with a different slot); the cheapest correct thing is to drop the
        # whole cache rather than reason about which rows it still names correctly.
        mirror = getattr(self.kv_cache_manager, "_mirror", None)
        if mirror is not None:
            mirror._batch_cache = None
        return wrapped(self, request, *args, **kwargs)

    scheduler_cls._preempt_request = mark_patched(
        _preempt_request, wrapped, patch="rust_sched_tokstore"
    )


def _install_full_schedule(scheduler_cls, sjf_enabled: bool, m: dict):
    """Replace ``Scheduler.schedule`` with the Rust-driven decision loop."""
    import time

    import vtl_sched
    from vllm.v1.core.sched.output import (
        CachedRequestData,
        NewRequestData,
        SchedulerOutput,
    )
    from vllm.v1.engine import EngineCoreEventType
    from vllm.v1.request import RequestStatus

    # Whatever is bound right now -- typically sched_policy's SJF wrapper. Every fallback
    # path must call THIS, not the unwrapped original: dropping the wrapper would silently
    # turn the reorder off on unsupported steps. The Rust path never calls python at all
    # (`sched.rs::reorder_waiting` reproduces the same key), so no unwrapping is needed.
    wrapped = scheduler_cls.schedule

    # Gates resolved ONCE, into closure cells. An off gate then costs one LOAD_DEREF and a
    # not-taken branch per step, which is why the timing arm below is inline rather than a
    # second copy of this 180-line function.
    TABLE, SPEC = m["table"], m["spec"]
    RESIDENT = TABLE
    TIMING = m["timing"]
    # Phase A. Resolved to its final value at first schedule (`lean_blocked` needs the
    # live scheduler), then constant: the crate is told once via `set_params`.
    LEAN = m["lean"]
    # VTL_SCHED_LEAN_CHECK: build BOTH payloads for this many FULL steps and compare the
    # subset the lean path actually feeds forward, then disarm. Counts down.
    lean_check_left = 32 if (m["lean"] and m["lean_check"]) else 0
    # Phase B. Resolved at first schedule: needs the crate entry points to exist (an older
    # wheel has none) and LEAN to have survived `lean_blocked`.
    ARENA = m["arena"]
    arena_check_left = 32 if (m["arena"] and m["arena_check"]) else 0
    # The six persistent decision buffers, re-read whenever the crate reports a grow.
    arena_bufs = None
    # Phase C. Two slots, one per async-batch-queue parity; `ring_blocked` refuses the gate
    # if the queue is not exactly that deep.
    RING = m["so_ring"]
    ring = [None, None]
    ring_i = 0
    ns = time.monotonic_ns
    timers = PhaseTimers() if TIMING else None

    # C1a. `nstep_decode` is the runner half; it publishes readiness and executes the
    # burst. Importing it here (not at module import) keeps rust_sched loadable when the
    # runner patch is disabled -- `nstep` then resolves to off.
    NSTEP = m["nstep"]
    nstep_mod = None
    if NSTEP:
        try:
            from vtl.patches import nstep_decode as nstep_mod
        except Exception:
            log.exception("rust_sched: nstep runner patch not importable; no bursts")
            NSTEP = False
    NSTEP_QUEUE_EMPTY_ONLY = os.environ.get(
        "VTL_NSTEP_QUEUE_EMPTY_ONLY", "1"
    ).strip().lower() in _TRUTHY

    # The Rust runner's scheduler half. `runner_mod` is only used for the in-flight counter
    # and the multi-step ceiling here -- the stash itself is written by the closure
    # `_install_update_from_output` published on the class (see `commit_burst`).
    RUNNER = m["runner"]
    runner_mod = None
    RUNNER_STEPS = 1
    if RUNNER:
        try:
            from vtl.patches import rust_runner as runner_mod

            RUNNER = runner_mod.mode() == "on"
            RUNNER_STEPS = runner_mod.max_steps()
        except Exception:
            log.exception("rust_sched: rust_runner not importable; no runner handshake")
            RUNNER = False

    def skip_note(self, why: str) -> None:
        """First-reason-only logging: names each new reason once, then stays quiet."""
        seen = self._vtl_burst_skips
        if why not in seen:
            seen.add(why)
            log.info("rust_sched: nstep skipped -- %s", why)

    def runner_steps(self, mirror, by_slot, slots, n) -> int:
        """``burst_steps`` over the whole batch: the smallest headroom wins.

        ``k > 1`` also needs an empty waiting queue -- a multi-burst residency must not
        delay an admission by more than the one burst ``VTL_NSTEP_QUEUE_EMPTY_ONLY``
        already allows.
        """
        if RUNNER_STEPS < 2 or self.waiting:
            return 1
        block_size = self.cache_config.block_size
        k = RUNNER_STEPS
        for slot in slots:
            # `- 1`: num_computed_tokens as it was BEFORE _update_after_schedule advanced
            # it, exactly as `commit_burst` reads it for the gate. `_burst_lim` was interned
            # by that gate a moment ago, so the `-1` default never fires in practice.
            k = burst_steps(
                by_slot[slot].num_computed_tokens - 1, n, block_size,
                mirror._burst_lim.get(slot, -1), k,
            )
            if k < 2:
                return 1
        return k

    def commit_burst(self, kv, so, by_slot, slots) -> None:
        """Decide and commit this step's burst factor -- or, failing that, in-graph N=1
        sampling. Runs after ``_update_after_schedule``.

        TWO COMMITS, ONE GATE. The burst needs the align gate and the queue-empty guard;
        in-graph N=1 sampling needs neither (one token crosses no block boundary and delays
        no admission), so a step the burst refuses can still commit the cheaper rung. The
        shared predicates are ``burst_blocked_batch`` / ``_burst_gate`` (the latter interns
        ``burst_sampler_blocked``'s request-immutable clauses per slot; see its docstring).

        Silent no-op on every ineligible step. On ANY exception the burst is disabled for the
        boot and the step is a plain one-token step: nothing has been mutated at that point
        except, at worst, a partial request loop, which is why the request mutation happens
        in a SECOND pass after every request has been checked.
        """
        n = nstep_mod.burst_factor(so.num_scheduled_tokens)
        one = nstep_mod.sample_in_graph_ready(so.num_scheduled_tokens)
        if n < 2 and not one:
            return
        try:
            batch_why = burst_blocked_batch(so, nstep_mod.MAX_BURST_REQS)
            if batch_why is not None:
                skip_note(self, batch_why)
                return
            mirror = kv._mirror
            # `- 1`: num_computed_tokens as it was BEFORE _update_after_schedule advanced it.
            if n >= 2:
                why = None
                if NSTEP_QUEUE_EMPTY_ONLY and self.waiting:
                    why = "waiting queue is not empty"
                else:
                    block_size = self.cache_config.block_size
                    for slot in slots:
                        request = by_slot[slot]
                        why = _burst_gate(
                            mirror, slot, request, request.num_computed_tokens - 1, n,
                            self.max_model_len, block_size,
                        )
                        if why is not None:
                            break
                if why is None:
                    # The Rust runner's promise for this step, taken BEFORE the commit so
                    # the committed budget can be several bursts wide. A refusal (or no
                    # runner at all) leaves it at one burst, which the worker replays
                    # itself exactly as before; a granted `steps > 1` that the worker then
                    # cannot take is reconciled by `burst_uncommit`, same as a short burst.
                    stash = getattr(self, "_vtl_runner_stash", None)
                    steps = 1
                    if stash is not None:
                        steps = stash(
                            kv, so, by_slot, slots, n,
                            runner_steps(self, mirror, by_slot, slots, n),
                        ) or 1
                    delta = n * steps - 1
                    for slot in slots:
                        burst_commit(by_slot[slot], delta)
                    kv._rust.table_burst(slots, delta)
                    self._vtl_burst_commits = c = getattr(self, "_vtl_burst_commits", 0) + 1
                    if c == 1 or not c & 0x1FFF:
                        log.info(
                            "rust_sched: nstep engaged -- %d bursts committed "
                            "(n=%d, steps=%d, batch=%d)",
                            c, n, steps, len(slots),
                        )
                    so.vtl_burst_n = n * steps
                    return
                skip_note(self, why)
            if one:
                why = None
                for slot in slots:
                    request = by_slot[slot]
                    why = _burst_gate(
                        mirror, slot, request, request.num_computed_tokens - 1, 1,
                        self.max_model_len,
                    )
                    if why is not None:
                        break
                if why is None:
                    # No bookkeeping at all: one token is exactly what the step already
                    # committed to, so only the SAMPLING path changes.
                    if not getattr(self, "_vtl_sample_in_graph_logged", False):
                        self._vtl_sample_in_graph_logged = True
                        log.info("rust_sched: in-graph sampling engaged")
                    so.vtl_sample_in_graph = True
                    return
                skip_note(self, f"in-graph sampling: {why}")
        except BaseException as exc:
            reraise_fatal(exc)
            log.exception("rust_sched: nstep commit failed; bursts disabled for this boot")
            nstep_mod.BURST.disable("scheduler commit raised")

    def schedule(self, *args, **kwargs):
        nonlocal LEAN, lean_check_left, ARENA, arena_check_left, arena_bufs
        nonlocal RING, ring_i
        if RUNNER:
            # Counted BEFORE any fallback return: every step that gets scheduled also gets
            # an `update_from_output` (the engine pops every batch it queues), and the
            # runner's launch interlock is the difference between the two.
            runner_mod.STATE.inflight += 1
        kv = self.kv_cache_manager
        core = getattr(self, "_vtl_rust_core", None)
        if core is None:
            reason = schedule_supported(self)
            if reason is not None or not hasattr(kv, "_rust"):
                if not getattr(self, "_vtl_rust_warned", False):
                    log.warning(
                        "rust_sched: full schedule() disabled -- %s",
                        reason or "KV manager is not Rust-backed",
                    )
                    self._vtl_rust_warned = True
                return wrapped(self, *args, **kwargs)
            core = self._vtl_rust_core = vtl_sched.Scheduler()
            if NSTEP and nstep_mod.BURST.n > 1:
                # Reserve KV headroom for the burst's extra tokens at queue depth 2. The
                # align gate already makes a mid-burst allocation impossible, so this is
                # slack, not a requirement -- but `num_lookahead_tokens` is what every
                # `can_allocate` check reads, and 43x KV headroom makes the reservation
                # free. Plumbed end to end (scheduler.py:908 -> the Rust config key below).
                self.num_lookahead_tokens = max(
                    self.num_lookahead_tokens, 2 * (nstep_mod.BURST.n - 1)
                )
            if LEAN:
                why = lean_blocked(self)
                if why is None:
                    log.info("rust_sched: lean decisions active")
                else:
                    LEAN = False
                    lean_check_left = 0
                    log.warning("rust_sched: lean decisions refused -- %s", why)
            if ARENA and not (LEAN and hasattr(core, "schedule_arena")):
                # Old wheel + new plugin, or the lean predicate refused: both are the
                # shipped dict path, which is still fully wired below.
                ARENA = False
                arena_check_left = 0
                log.warning("rust_sched: decisions arena unavailable; staying on the dict")
            elif ARENA:
                arena_bufs = core.arena_buffers()
                log.info("rust_sched: decisions arena active%s",
                         " (CHECK)" if arena_check_left else "")
            if RING:
                why = ring_blocked(self)
                if why is None:
                    log.info("rust_sched: SchedulerOutput ring active")
                else:
                    RING = False
                    log.warning("rust_sched: SchedulerOutput ring refused -- %s", why)
            # Engine constants: handed over once, not re-parsed from a dict per step.
            core.set_params(
                {
                    "max_num_scheduled_tokens": int(self.max_num_scheduled_tokens),
                    "max_num_running_reqs": int(self.max_num_running_reqs),
                    "max_model_len": int(self.max_model_len),
                    "num_sampled_tokens_per_step": int(self.num_sampled_tokens_per_step),
                    "long_prefill_token_threshold": int(
                        self.scheduler_config.long_prefill_token_threshold or 0
                    ),
                    "enable_chunked_prefill": bool(
                        self.scheduler_config.enable_chunked_prefill
                    ),
                    "need_mamba_block_aligned_split": bool(
                        self.need_mamba_block_aligned_split
                    ),
                    "cache_block_size": int(self.cache_config.block_size),
                    "num_lookahead_tokens": int(self.num_lookahead_tokens),
                    "sjf_reorder": bool(sjf_enabled),
                    # Optional key on the crate side: a wheel older than this plugin
                    # ignores it and keeps sending the full payload, which the apply
                    # block below still reads correctly.
                    "lean_decisions": bool(LEAN),
                }
            )
            log.info(
                "rust_sched: FULL schedule() loop active "
                "(sjf=%s, table=%s, spec=%s, timing=%s)",
                sjf_enabled, TABLE, SPEC, TIMING,
            )

        if TIMING:
            t_enter = ns()
            if timers.last_exit:
                timers.add("gap", t_enter - timers.last_exit)

        # Bail BEFORE `current_step += 1`: the fallback does its own increment.
        bail = bail_reason(self)
        tbl = getattr(kv, "_vtl_table", None) if TABLE else None
        # A dirty table means the marshalled call has to run anyway -- it IS the resync.
        resident = RESIDENT and tbl is not None and not tbl.off and not tbl.dirty
        mirror = kv._mirror
        by_slot = {}
        running = []
        running_slots = []
        waiting = []
        if bail is None:
            try:
                for request in self.running:
                    # `mirror.slot` still pushes new block hashes here, BEFORE scheduling,
                    # exactly as the marshalled path always has -- the resident table
                    # holds counters, never hashes.
                    slot = mirror.slot(request)
                    by_slot[slot] = request
                    running_slots.append(slot)
                    if not resident:
                        running.append(pack_req(slot, request))
                for request in self.waiting:
                    if request.status not in (RequestStatus.WAITING, RequestStatus.PREEMPTED):
                        bail = f"waiting request in status {request.status!s}"
                        break
                    slot = mirror.slot(request)
                    by_slot[slot] = request
                    waiting.append(pack_req(slot, request))
            except NotImplementedError as exc:  # unported RequestStatus on a running req
                bail = str(exc)
        if bail is not None:
            if tbl is not None:
                # Stock vLLM now mutates requests and queues with no Rust call at all.
                tbl.resync(bail)
            seen = getattr(self, "_vtl_rust_bails", None)
            if seen is None:
                seen = self._vtl_rust_bails = set()
            if bail not in seen:
                seen.add(bail)
                log.warning("rust_sched: this step falls back to vLLM -- %s", bail)
            return wrapped(self, *args, **kwargs)

        self.current_step += 1
        if TIMING:
            t_marshal = ns()
            timers.add("marshal", t_marshal - t_enter)

        # Phase B: `counts` is the arena's 8-tuple, `decisions` the PyDict. Exactly one is
        # non-None on any given step, except under CHECK where the crate hands back both
        # marshallings of the SAME decisions (re-running schedule() would mutate twice).
        decisions = None
        counts = None
        if resident:
            try:
                # The worker speculated with an EMPTY waiting slice (spec.rs), and
                # `take_speculative` checks the generation and slot order but NOT the
                # queue -- so refusing here is what makes an admission step safe.
                if SPEC and tbl.armed and not waiting:
                    if ARENA:
                        counts = core.take_speculative_arena(
                            kv._rust, tbl.gen, running_slots, arena_check_left > 0
                        )
                    else:
                        decisions = core.take_speculative(kv._rust, tbl.gen, running_slots)
                tbl.armed = False
                if decisions is None and counts is None:
                    if ARENA:
                        counts = core.schedule_resident_arena(
                            kv._rust, running_slots, waiting, arena_check_left > 0
                        )
                    else:
                        decisions = core.schedule_resident(kv._rust, running_slots, waiting)
            except Exception as exc:
                # The documented failure is "slot N has no resident entry", i.e. the
                # resync signal; every other rejection wants the same answer.
                tbl.resync(f"resident schedule refused ({exc!r})")
                decisions = counts = None
            except BaseException as exc:  # a crate panic
                tbl.fail("schedule_resident", exc)
                decisions = counts = None
            if decisions is None and counts is None:
                running = [pack_req(s, by_slot[s]) for s in running_slots]
        if decisions is None and counts is None:
            if ARENA:
                counts = core.schedule_arena(
                    kv._rust, running, waiting, arena_check_left > 0
                )
            else:
                decisions = core.schedule(kv._rust, running, waiting)
            if tbl is not None:
                # `Scheduler.schedule` rewrites every running entry: this call is the
                # full resync, and the only place `dirty` clears.
                tbl.dirty = False
                tbl.armed = False
        if TIMING:
            t_rust = ns()
            timers.add("rust", t_rust - t_marshal)

        # --- apply the decisions (scheduler.py:1045-1133 in Python) ----------
        timestamp = time.monotonic()
        num_scheduled_tokens: dict[str, int] = {}
        req_to_new_blocks = {}
        scheduled_running_reqs = []
        scheduled_new_reqs = []
        scheduled_resumed_reqs = []

        # scheduler.py:588 keeps only the step's DELTA for a running request (`:983`
        # keeps the full table, but only for a waiting admission). The V2 runner appends
        # new_block_ids without overwriting, so re-sending the whole table every step
        # duplicates rows until the block table overflows.
        #
        # Phase B binds the same six names off strided `[:n]` views instead of the PyDict.
        # `zip` over two `.tolist()`s yields the same (slot, num_new) pairs the dict path
        # yields, so every loop below is written once. Block ids are materialized here --
        # they end up inside a `RustBlocks` that `SchedulerOutput` keeps across steps, and
        # the arena buffer is overwritten by the next step.
        if counts is not None:
            n_run, n_adm, n_blk, n_len, n_pre, n_wait, grew, check_dict = counts
            if grew:
                arena_bufs = core.arena_buffers()
            a_run, a_adm, a_blk, a_lens, a_pre, a_wait = arena_bufs
            sched_running = zip(a_run[0:n_run:2].tolist(), a_run[1:n_run:2].tolist())
            sched_admitted = zip(
                a_adm[0:n_adm:3].tolist(),
                a_adm[1:n_adm:3].tolist(),
                a_adm[2:n_adm:3].tolist(),
            )
            flat = a_blk[:n_blk].tolist()
            lens = a_lens[:n_len].tolist()
            preempted = a_pre[:n_pre].tolist()
            waiting_order = a_wait[:n_wait].tolist()
            if check_dict is not None:
                arena_check_left -= 1
                sched_running = list(sched_running)
                sched_admitted = list(sched_admitted)
                why = arena_check(
                    check_dict, sched_running, sched_admitted, flat, lens,
                    preempted, waiting_order,
                )
                if why is not None:
                    # The dict marshalling in `check_dict` is the trusted source; serve
                    # it for this step and let the dict path own every later one. A raise
                    # here would escape schedule() and fail the request -- the one thing
                    # a soak arm must never do.
                    log.error(
                        "rust_sched: decisions arena diverged (%s); dict path takes over",
                        why,
                    )
                    ARENA = False
                    arena_check_left = 0
                    sched_running = check_dict["scheduled_running"]
                    sched_admitted = check_dict["scheduled_admitted"]
                    flat = check_dict["running_new_blocks"]
                    lens = check_dict["running_new_lens"]
                    preempted = check_dict["preempted"]
                    waiting_order = check_dict["waiting_order"]
                elif not arena_check_left:
                    log.info("rust_sched: VTL_SCHED_DECISIONS_ARENA check clean over 32 steps")
        else:
            sched_running = decisions["scheduled_running"]
            sched_admitted = decisions["scheduled_admitted"]
            flat = decisions["running_new_blocks"]
            lens = decisions["running_new_lens"]
            preempted = decisions["preempted"]
            waiting_order = decisions["waiting_order"]
        # Phase C. The ring is only ever consulted on a step where nothing but decode
        # happened, so all the cheap counts have to be zero first; `ring_reuse` then
        # confirms the running set itself is byte-identical to the slot's.
        if RING:
            # The arena's `zip` is one-shot AND always truthy, so both have to become real
            # lists before the predicate can read them -- and so a refused reuse can still
            # be applied below. On the dict path they already are lists.
            sched_running = list(sched_running)
            sched_admitted = list(sched_admitted)
            slot = ring[ring_i]
            if (
                slot is not None
                and not sched_admitted
                and not preempted
                and not flat
                and not waiting_order
                and not self.waiting
                and not self.finished_req_ids
                and not self.reset_preempted_req_ids
                # PEEKED, not drained: `get_freed_mm_hashes()` clears the list, and the
                # preempt/admit loops below can still append to it this step.
                and not self.encoder_cache_manager.freed
            ):
                so = ring_reuse(self, slot, sched_running)
                if so is not None:
                    ring_i ^= 1
                    if NSTEP:
                        commit_burst(self, kv, so, by_slot, slot[1])
                    if TIMING:
                        t_exit = ns()
                        timers.add("apply", t_exit - t_rust)
                        timers.last_exit = t_exit
                        timers.tick()
                    return so

        num_groups = kv._rust.num_groups
        off = 0
        sched_slots = []
        for i, (slot, num_new) in enumerate(sched_running):
            sched_slots.append(slot)
            request = by_slot[slot]
            scheduled_running_reqs.append(request)
            num_scheduled_tokens[request.request_id] = num_new
            groups = []
            for g in range(num_groups):
                n = lens[i * num_groups + g]
                groups.append(flat[off : off + n])
                off += n
            req_to_new_blocks[request.request_id] = RustBlocks(tuple(groups))

        # Rebuild the waiting queue from the Rust core's view. This is NOT cosmetic: the
        # SJF reorder happens inside the core, so popping the Python deque's front would
        # drop the wrong requests. `waiting_order` is what is LEFT, in final order.
        # Skipped when both sides are empty -- clear-then-rebuild of empty-from-empty is
        # a provable no-op, and that is every steady decode step. Only THAT case: for a
        # non-empty queue the orders can match while the objects differ.
        if waiting_order or self.waiting:
            remaining = [by_slot[s] for s in waiting_order]
            self.waiting.clear()
            for request in remaining:
                self.waiting.add_request(request)

        for slot in preempted:
            request = by_slot[slot]
            # Port-2: `num_computed_tokens = 0` below re-prefills from `_all_token_ids`, so
            # the store has to hand the tokens back first. (This loop IS the Rust loop's
            # `_preempt_request`; the hook on that method covers stock's path.)
            if facaded(request):
                tok_materialize(kv, request)
            self.running.remove(request)
            self.encoder_cache_manager.free(request)
            self._inflight_prefills.discard(request)
            request.status = RequestStatus.PREEMPTED
            request.num_computed_tokens = 0
            request.num_preemptions += 1
            if self.log_stats:
                request.record_event(EngineCoreEventType.PREEMPTED, timestamp)
            # Same one-at-a-time prepend as _preempt_request (scheduler.py:1161).
            self.waiting.prepend_request(request)
            self.reset_preempted_req_ids.add(request.request_id)

        for slot, num_new, num_computed in sched_admitted:
            request = by_slot[slot]
            self.running.append(request)
            if self.log_stats:
                request.record_event(EngineCoreEventType.SCHEDULED, timestamp)
            if request.status == RequestStatus.WAITING:
                scheduled_new_reqs.append(request)
            else:
                scheduled_resumed_reqs.append(request)
            request.status = RequestStatus.RUNNING
            request.num_computed_tokens = num_computed
            num_scheduled_tokens[request.request_id] = num_new
            req_to_new_blocks[request.request_id] = kv.get_blocks(request.request_id)
            if num_computed + num_new < request.num_tokens:
                self._inflight_prefills.add(request)

        total = sum(num_scheduled_tokens.values())
        if self.use_v2_model_runner:
            scheduled_new_reqs.extend(scheduled_resumed_reqs)
            scheduled_resumed_reqs = []
            for r in scheduled_new_reqs:
                # `prefill_token_ids` is one of only two consumers that needs the real ints.
                # Belt to `_install_preempt_hook`'s braces: a resumed request should already
                # be materialized, and an admission was never facaded, so this is normally
                # one `type()` lookup per admitted request.
                if facaded(r):
                    tok_materialize(kv, r)
            new_reqs_data = [
                NewRequestData.from_request(
                    r,
                    req_to_new_blocks[r.request_id].get_block_ids(),
                    r._all_token_ids,
                )
                for r in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    r, req_to_new_blocks[r.request_id].get_block_ids()
                )
                for r in scheduled_new_reqs
            ]
        cached_reqs_data = None
        if LEAN:
            try:
                cached_reqs_data = lean_cached_request_data(
                    CachedRequestData, scheduled_running_reqs, req_to_new_blocks
                )
                if lean_check_left:
                    lean_check_left -= 1
                    why = lean_cached_check(
                        cached_reqs_data,
                        self._make_cached_request_data(
                            scheduled_running_reqs, scheduled_resumed_reqs,
                            num_scheduled_tokens, {}, req_to_new_blocks,
                        ),
                    )
                    if why is not None:
                        raise AssertionError(why)
                    if not lean_check_left:
                        log.info("rust_sched: VTL_SCHED_LEAN_CHECK clean over 32 steps")
            except BaseException as exc:
                reraise_fatal(exc)
                # The crate keeps its `lean_decisions` setting (set_params ran once), so
                # `num_common_prefix_blocks` stays absent -- harmless, nothing reads it.
                LEAN = False
                lean_check_left = 0
                cached_reqs_data = None
                log.exception("rust_sched: lean CachedRequestData failed; back to stock")
        if cached_reqs_data is None:
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                {},
                req_to_new_blocks,
            )
        # Absent = the crate ran the lean epilogue. No V2 consumer reads this at all, so
        # one shared zero list per group count is enough (nothing can mutate it).
        common = None if decisions is None else decisions.get("num_common_prefix_blocks")
        if common is None:
            common = _ZERO_COMMON.get(num_groups)
            if common is None:
                common = _ZERO_COMMON[num_groups] = [0] * num_groups
        else:
            common = list(common)
        if not self.use_v2_model_runner:
            self.prev_step_scheduled_req_ids.clear()
            self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        free_mm_hashes = self.encoder_cache_manager.get_freed_mm_hashes()
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=common,
            preempted_req_ids=self.reset_preempted_req_ids,
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=free_mm_hashes,
            new_block_ids_to_zero=(
                (kv.take_new_block_ids() or None) if self.needs_kv_cache_zeroing else None
            ),
            num_spec_tokens_to_schedule=self.num_spec_tokens,
        )
        if self.defer_block_free and total > 0:
            self.sched_step_seq += 1
        self._update_after_schedule(scheduler_output)
        if RING:
            # Populate THIS parity's slot -- but ONLY from a step that is itself a pure
            # decode. Every payload a later reuse would re-send verbatim has to be empty
            # already: `scheduled_new_reqs` would re-add requests to the runner,
            # `new_block_ids` would re-append block ids, `free_encoder_mm_hashes` would
            # re-free, and a non-empty finished/preempted set would be replayed. Anything
            # else invalidates the slot instead, and the next clean step at this parity
            # refills it.
            #
            # `finished_req_ids` / `preempted_req_ids` are the sets `_update_after_schedule`
            # just ORPHANED by rebinding the scheduler's, so the slot owns them outright --
            # that is hazard 1's whole answer.
            ring[ring_i] = (
                (scheduler_output, tuple(sched_slots), scheduled_running_reqs)
                if (
                    not new_reqs_data
                    and not flat
                    and not free_mm_hashes
                    and not scheduler_output.finished_req_ids
                    and not scheduler_output.preempted_req_ids
                    and total == len(sched_slots)
                    # An empty batch is not worth caching, and reusing one would bump
                    # `sched_step_seq` where stock gates on `total > 0`.
                    and total > 0
                )
                else None
            )
            ring_i ^= 1
        if NSTEP:
            commit_burst(self, kv, scheduler_output, by_slot, sched_slots)
        if TIMING:
            t_exit = ns()
            timers.add("apply", t_exit - t_rust)
            timers.last_exit = t_exit
            timers.tick()
        return scheduler_output

    # Per-reason log de-duplication for the burst gate; class level so the first step
    # does not have to create it.
    scheduler_cls._vtl_burst_skips = set()
    scheduler_cls.schedule = mark_patched(schedule, wrapped)


def _install_rust_hasher() -> None:
    """B3 -- ``get_request_block_hasher``'s product, backed by the ported Rust hasher.

    THE CHEAP RUNG, deliberately. ``Request.block_hashes`` stays the single source of
    truth, ``RustMirror.slot``'s ``push_hashes`` still ships the bytes over, ``maybe_kick``
    ordering is untouched and every consumer is unchanged -- the only thing that moves is
    WHERE the sha256-over-pickle happens. Full removal (Rust owning the list) couples to
    the R6c kick ordering and is out of scope.

    Rebinds the IMPORT-BOUND name in ``vllm.v1.engine.core`` -- patching
    ``kv_cache_utils`` would do nothing, core.py does ``from ... import``.

    Two shapes, because the prompt and the decode tail want different call patterns:
      * request has NO hashes yet (admission, the whole prompt) -> ONE crossing for every
        block via ``vtl_sched.block_hashes``, which chains internally from NONE_HASH;
      * request already has hashes (a decode step completed a block) -> chain from the
        last one with ``hash_block_tokens``. That is one crossing per NEW block, and a
        decode step produces at most one.

    Refusal guard: anything that needs ``extra_keys`` (cache_salt, LoRA, multimodal,
    prompt embeds) keeps the stock hasher for that request. The served path is text-only,
    so this never fires; guessing at extra-key pickling would be a silent cache-poisoning
    bug if it ever did.
    """
    import vtl_sched
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.engine import core as core_mod

    stock_factory = kv_cache_utils.get_request_block_hasher
    if getattr(core_mod.get_request_block_hasher, "__vtl_rust_hasher__", False):
        return

    def factory(hash_block_size: int, caching_hash_fn):
        stock = stock_factory(hash_block_size, caching_hash_fn)
        # NONE_HASH is a module global written by init_none_hash(), which core.py calls
        # immediately before this factory -- so it is readable now and constant after.
        none_hash = bytes(kv_cache_utils.NONE_HASH)
        block_hashes = vtl_sched.block_hashes
        hash_block_tokens = vtl_sched.hash_block_tokens
        BlockHash = kv_cache_utils.BlockHash
        # Present only on a wheel built with Item 2a. `getattr`, not an import guard: an
        # older wheel then simply never takes the numpy arm.
        block_hashes_u32 = getattr(vtl_sched, "block_hashes_u32", None)

        def rust_block_hasher(request):
            if (
                request.cache_salt
                or request.mm_features
                or request.lora_request is not None
                or request.prompt_embeds is not None
            ):
                return stock(request)
            have = len(request.block_hashes)
            start = have * hash_block_size
            num_tokens = request.num_tokens
            if start + hash_block_size > num_tokens:
                return []
            end = num_tokens - (num_tokens - start) % hash_block_size
            tokens = request.all_token_ids
            if have == 0:
                # Prompt-only path (`have == 0` is admission, so `all_token_ids[:end]` IS
                # `prompt_token_ids[:end]`). With the raw ADD record those ids already
                # exist as a uint32 numpy view, so the crate can read them contiguously
                # instead of extracting `end` PyLongs. `.flags.aligned` is the safety
                # contract for the Rust-side `&[u32]`; the record's reserved u16 is what
                # makes it hold, and a False here just takes the list path.
                arr = getattr(request.prompt_token_ids, "numpy_u32", None)
                if (
                    block_hashes_u32 is not None
                    and arr is not None
                    and end <= len(arr)
                    and arr.flags.aligned
                ):
                    hs = block_hashes_u32(none_hash, hash_block_size, arr[:end])
                else:
                    hs = block_hashes(none_hash, hash_block_size, tokens[:end])
                return [BlockHash(h) for h in hs]
            out = []
            parent = request.block_hashes[-1]
            while start < end:
                parent = hash_block_tokens(
                    none_hash, parent, tokens[start : start + hash_block_size]
                )
                out.append(BlockHash(parent))
                start += hash_block_size
            return out

        return rust_block_hasher

    factory.__vtl_rust_hasher__ = True
    core_mod.get_request_block_hasher = factory
    log.info("rust_sched: block hashing routed to vtl_sched (VTL_RUST_HASHER=1)")


def _install_generation_hooks(scheduler_cls):
    """Bump ``TableState.gen`` on the two queue mutations that never reach Rust.

    Everything else the engine does to the KV state goes through an invalidating ``w()``
    guard inside the crate, which drops a pending speculation on its own. These two do
    not: ``add_request`` only appends to a Python deque, and ``finish_requests`` retires
    requests whose blocks may be freed later (or not at all, for a request still waiting).
    """
    wrapped_add = scheduler_cls.add_request
    wrapped_finish = scheduler_cls.finish_requests

    def bump(self):
        tbl = getattr(self.kv_cache_manager, "_vtl_table", None)
        if tbl is not None:
            tbl.bump()

    def add_request(self, request, *args, **kwargs):
        bump(self)
        return wrapped_add(self, request, *args, **kwargs)

    def finish_requests(self, *args, **kwargs):
        bump(self)
        return wrapped_finish(self, *args, **kwargs)

    scheduler_cls.add_request = mark_patched(add_request, wrapped_add, patch="rust_sched_spec")
    scheduler_cls.finish_requests = mark_patched(
        finish_requests, wrapped_finish, patch="rust_sched_spec"
    )


# --------------------------------------------------------------------------
# install
# --------------------------------------------------------------------------


@register_patch("rust_sched", default=False)
def apply() -> None:
    # One-liner 5d: independent of every gate below (no vtl_sched extension needed, no
    # scheduler surface touched) -- see `_install_async_output_event_ring`'s docstring.
    _install_async_output_event_ring()

    m = modes()
    if not (m["authority"] or m["hasher"]):
        log.info("rust_sched: no mode selected, nothing installed")
        return

    try:
        import vtl_sched  # noqa: F401
    except BaseException as exc:
        reraise_fatal(exc)
        # The wheel is built best-effort (network), so a missing extension is a real and
        # tolerable state -- but it is also the single most likely reason a measured arm
        # silently has no Rust scheduler in it at all.
        refuse(f"vtl_sched extension not importable ({exc!r})")
        return

    if m["hasher"]:
        try:
            _install_rust_hasher()
        except BaseException as exc:
            reraise_fatal(exc)
            log.exception("rust_sched: rust block hasher not installed; keeping stock")

    if not m["authority"]:
        return

    import vllm.v1.core.sched.scheduler as sched_mod

    base = sched_mod.KVCacheManager
    if getattr(base, "__vtl_rust_authority__", False):
        return  # idempotent

    if getattr(base, "__vtl_subclass__", False):
        log.info("rust_sched: composing on top of vtl kv_cache_manager (signals preserved)")
    else:
        log.info("rust_sched: vtl kv_cache_manager patch is not installed; subclassing stock")

    sched_mod.KVCacheManager = _install_authority(base, m)
    log.info("rust_sched: installed AUTHORITY manager (VTL_RUST_SCHED=1)")

    if m["authority"]:
        # Newer vLLM defaults scheduler_reserve_full_isl=True; its stock schedule()
        # then calls allocate_slots(full_sequence_must_fit=True), which the authority
        # manager refuses (the admission-fit gate needs the python coordinator the Rust
        # pool replaced). The Rust port models the pre-feature admission (first-chunk
        # check), so force the flag off on the live scheduler: identical semantics to
        # what the port was verified against, and schedule_supported() then lets the
        # full Rust loop engage.
        from vllm.v1.core.sched.scheduler import Scheduler

        if not getattr(Scheduler.__init__, "__vtl_rust_isl__", False):
            orig_init = Scheduler.__init__

            def init(self, *args, **kwargs):
                orig_init(self, *args, **kwargs)
                if getattr(self, "scheduler_reserve_full_isl", False) and hasattr(
                    self.kv_cache_manager, "_rust"
                ):
                    self.scheduler_reserve_full_isl = False
                    log.warning(
                        "rust_sched: forcing scheduler_reserve_full_isl=False "
                        "(admission-fit gate is not modeled by the Rust port)"
                    )

            init.__vtl_rust_isl__ = True
            Scheduler.__init__ = init

    if m["full"]:
        from vllm.v1.core.sched.scheduler import Scheduler

        if already_patched(Scheduler, "schedule") and not getattr(
            Scheduler.schedule, "__vtl_rust_full__", False
        ):
            log.info("rust_sched: SUPERSEDING sched_policy's schedule() wrapper "
                     "(its SJF key now runs inside the Rust loop)")
        sjf = os.environ.get("VTL_ENABLE_SCHED_POLICY", "1").strip().lower() in _TRUTHY
        _install_full_schedule(Scheduler, sjf, m)
        Scheduler.schedule.__vtl_rust_full__ = True
        log.info(
            "rust_sched: resolved R6b/R6c mode -- table=%s spec=%s timing=%s",
            m["table"], m["spec"], m["timing"],
        )

        if m["ufo"] and not already_patched(
            Scheduler, "update_from_output", patch="rust_sched_ufo"
        ):
            _install_update_from_output(Scheduler, m)

        if m["spec"] and not already_patched(
            Scheduler, "add_request", patch="rust_sched_spec"
        ):
            _install_generation_hooks(Scheduler)


# --------------------------------------------------------------------------
# self-check -- runs with neither vLLM nor the compiled crate
# --------------------------------------------------------------------------


def _self_check() -> None:
    saved = {k: os.environ.get(k) for k in (
        "VTL_RUST_SCHED", "VTL_RUST_SCHED_FULL", "VTL_RUST_SCHED_RADIX",
        "VTL_RUST_SCHED_UFO",
        "VTL_RUST_SCHED_TABLE",
        "VTL_RUST_SCHED_SPEC", "VTL_SCHED_TIMING",
        "VTL_RUST_SCHED_R8", "VTL_RUST_HASHER",
        "VTL_SHM_IPC", "VTL_SHM_IPC_RAW", "VTL_NSTEP",
        "VTL_RUST_SCHED_TOKSTORE",
        "VTL_RUST_SCHED_R9",
        "VTL_RUST_SCHED_LEAN", "VTL_SCHED_LEAN_CHECK",
        "VTL_SCHED_DECISIONS_ARENA", "VTL_SCHED_SO_RING",
    )}
    try:
        for k in saved:
            os.environ.pop(k, None)
        assert modes() == {
            "authority": False,
            "full": False, "radix": False, "ufo": False,
            "table": False, "spec": False, "timing": False,
            "r8": False, "hasher": False, "nstep": False, "runner": False,
            "tokstore": False,
            "r9": False,
            "lean": False, "lean_check": False,
            "arena": False, "arena_check": False, "so_ring": False,
        }, modes()
        assert env_on("VTL_RUST_SCHED") is False
        for truthy in ("1", "true", "YES", " on "):
            os.environ["VTL_RUST_SCHED"] = truthy
            assert env_on("VTL_RUST_SCHED") is True, truthy
        os.environ["VTL_RUST_SCHED"] = "0"
        assert env_on("VTL_RUST_SCHED") is False
        # full implies authority, even with VTL_RUST_SCHED unset.
        os.environ.pop("VTL_RUST_SCHED")
        os.environ["VTL_RUST_SCHED_FULL"] = "1"
        assert modes()["authority"] is True and modes()["full"] is True
        # UFO needs the full loop: it reads the Rust manager's slot interning.
        os.environ["VTL_RUST_SCHED_UFO"] = "1"
        assert modes()["ufo"] is True
        # The R6b/R6c ladder: TABLE needs UFO, SPEC needs TABLE.
        os.environ["VTL_RUST_SCHED_TABLE"] = "1"
        os.environ["VTL_RUST_SCHED_SPEC"] = "1"
        assert modes()["table"] is True and modes()["spec"] is True
        os.environ["VTL_RUST_SCHED_UFO"] = "0"
        assert modes()["table"] is False and modes()["spec"] is False, (
            "TABLE must not arm without UFO -- update_step applies its token delta"
        )
        os.environ["VTL_RUST_SCHED_UFO"] = "1"
        os.environ["VTL_RUST_SCHED_FULL"] = "0"
        assert modes()["ufo"] is False, "UFO must not arm without the full Rust loop"
        assert modes()["table"] is False and modes()["spec"] is False
        # VTL_SCHED_TIMING is orthogonal: log-only, no dependency on any of the above.
        os.environ["VTL_SCHED_TIMING"] = "1"
        assert modes()["timing"] is True

        # R8 needs UFO *and* the raw shm wire format -- building bytes the output thread
        # would have to unpack again is strictly worse than building objects.
        os.environ["VTL_RUST_SCHED_FULL"] = "1"
        os.environ["VTL_RUST_SCHED_UFO"] = "1"
        os.environ["VTL_RUST_SCHED_R8"] = "1"
        assert modes()["r8"] is False, "R8 must not arm without the shm raw wire format"
        os.environ["VTL_SHM_IPC"] = "1"
        assert modes()["r8"] is False, "...nor with shm but msgpack records"
        os.environ["VTL_SHM_IPC_RAW"] = "1"
        assert modes()["r8"] is True
        os.environ["VTL_RUST_SCHED_UFO"] = "0"
        assert modes()["r8"] is False, (
            "R8 IS update_step plus a pack; it cannot arm without UFO"
        )
        os.environ["VTL_RUST_SCHED_UFO"] = "1"

        # Phase A's lean payload IS the full loop's apply block, so it needs FULL too.
        os.environ["VTL_RUST_SCHED_LEAN"] = "1"
        assert modes()["lean"] is True
        os.environ["VTL_RUST_SCHED_FULL"] = "0"
        assert modes()["lean"] is False
        os.environ["VTL_RUST_SCHED_FULL"] = "1"

        # Phase B rides on the lean payload: the arena carries no common-prefix slot.
        os.environ["VTL_SCHED_DECISIONS_ARENA"] = "1"
        assert modes()["arena"] is True and modes()["arena_check"] is False
        os.environ["VTL_RUST_SCHED_LEAN"] = "0"
        assert modes()["arena"] is False
        os.environ["VTL_RUST_SCHED_LEAN"] = "1"
        os.environ["VTL_SCHED_DECISIONS_ARENA"] = "check"
        assert modes()["arena"] is True and modes()["arena_check"] is True
        os.environ["VTL_SCHED_DECISIONS_ARENA"] = "0"
        assert modes()["arena"] is False

        # Phase C is the same loop's tail.
        os.environ["VTL_SCHED_SO_RING"] = "1"
        assert modes()["so_ring"] is True
        os.environ["VTL_RUST_SCHED_FULL"] = "0"
        assert modes()["so_ring"] is False
        os.environ["VTL_RUST_SCHED_FULL"] = "1"
        os.environ["VTL_SCHED_SO_RING"] = "0"

        # The N-step commit lives inside the Rust schedule loop, so it needs FULL.
        os.environ["VTL_NSTEP"] = "1"
        assert modes()["nstep"] is True
        os.environ["VTL_RUST_SCHED_FULL"] = "0"
        assert modes()["nstep"] is False
        os.environ["VTL_RUST_SCHED_FULL"] = "1"

        # Port-2 rides on R8 *and* the Rust hasher: two live hash implementations would
        # each own half the chain, and a divergence between them is silent.
        os.environ["VTL_RUST_SCHED_TOKSTORE"] = "1"
        assert modes()["r8"] is True
        assert modes()["tokstore"] is False, "no token store without the Rust hasher"
        os.environ["VTL_RUST_HASHER"] = "1"
        assert modes()["tokstore"] is True
        os.environ["VTL_RUST_SCHED_R8"] = "0"
        assert modes()["tokstore"] is False, (
            "the store IS update_step_pack; it cannot arm without R8"
        )
        os.environ["VTL_RUST_SCHED_R8"] = "1"
        assert modes()["tokstore"] is True

        # R9 rides on the token store: the residue loop's counter writes and the fold's
        # `num_computed - num_output_placeholders` math both assume Rust already owns the
        # per-slot bookkeeping tokstore provides.
        assert modes()["r9"] is False, "R9 needs VTL_RUST_SCHED_R9 too"
        os.environ["VTL_RUST_SCHED_R9"] = "1"
        assert modes()["r9"] is True
        os.environ.pop("VTL_RUST_SCHED_R9")
        os.environ.pop("VTL_RUST_SCHED_TOKSTORE")
        assert modes()["tokstore"] is False
        os.environ["VTL_RUST_SCHED_R9"] = "1"
        assert modes()["r9"] is False, "R9 cannot arm without the token store"
        os.environ.pop("VTL_RUST_SCHED_R9")

        # The Rust runner's scheduler half needs the burst commit (it launches the burst's
        # own graph) AND R9 (its verdicts are applied by the residue loop, nothing else).
        os.environ["VTL_RUST_SCHED_TOKSTORE"] = "1"
        os.environ["VTL_RUST_SCHED_R9"] = "1"
        assert modes()["runner"] is True
        os.environ["VTL_NSTEP"] = "0"
        assert modes()["runner"] is False, "no burst commit, no unroll graph to launch"
        os.environ["VTL_NSTEP"] = "1"
        os.environ["VTL_RUST_SCHED_R9"] = "0"
        assert modes()["runner"] is False, "only r9_apply knows how to apply the verdicts"
        os.environ["VTL_RUST_SCHED_R9"] = "1"
        os.environ["VTL_RUST_SCHED_TOKSTORE"] = "0"
        assert modes()["runner"] is False
        os.environ.pop("VTL_RUST_SCHED_R9")
        os.environ.pop("VTL_RUST_SCHED_TOKSTORE")
        os.environ["VTL_NSTEP"] = "1"

        # The hasher is independent: it needs only the extension, not the scheduler.
        os.environ["VTL_RUST_SCHED_FULL"] = "0"
        assert modes()["hasher"] is True and modes()["full"] is False
        os.environ.pop("VTL_RUST_HASHER")
        assert modes()["hasher"] is False
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # spec_signature: equal field values group together, different ones do not.
    class Spec:
        def __init__(self, block_size, extra):
            self.block_size = block_size
            self.extra = extra

    assert spec_signature(Spec(16, "a")) == spec_signature(Spec(16, "a"))
    assert spec_signature(Spec(16, "a")) != spec_signature(Spec(16, "b"))
    assert spec_signature(Spec(16, "a")) != spec_signature(Spec(32, "a"))

    # (Muted below: several failure paths log at ERROR by design, and `make check`
    # output should stay clean.)
    log.setLevel(logging.CRITICAL)

    # TableState: born dirty (nothing is resident yet); resync bumps the generation AND
    # disarms, so a speculation kicked before it can never be consumed after; fail() is
    # permanent. (Still inside the muted-logger region -- fail() logs at ERROR.)
    tbl = TableState()
    assert tbl.dirty and tbl.gen == 0 and not tbl.armed and not tbl.off
    tbl.dirty, tbl.armed = False, True
    tbl.resync("test")
    assert tbl.dirty and tbl.gen == 1 and not tbl.armed
    tbl.bump()
    assert tbl.gen == 2
    tbl.fail("test", RuntimeError("boom"))
    assert tbl.off and tbl.dirty and not tbl.armed
    try:
        tbl.fail("test", KeyboardInterrupt())
    except KeyboardInterrupt:
        pass
    else:  # pragma: no cover
        raise AssertionError("fail() must not swallow interpreter signals")

    # PhaseTimers: p50/p95 off a sorted ring, and the ring is drained each report.
    pt = PhaseTimers()
    for i in range(100):
        pt.add("rust", i * 1000)
    pt.n = PhaseTimers.EVERY - 1
    pt.tick()
    assert pt.rings["rust"] == [], "the ring must be drained after a report"

    # RustBlocks duck-types the pieces of KVCacheBlocks the scheduler touches.
    rb = RustBlocks(([1, 2], [3]))
    assert rb.get_block_ids() == ([1, 2], [3])
    assert rb.new_empty().get_block_ids() == ([], [])
    assert rb.new_empty().get_block_ids(allow_none=True) is None
    assert (rb + RustBlocks(([9], [8]))).get_block_ids() == ([1, 2, 9], [3, 8])
    try:
        rb.get_unhashed_block_ids()
    except NotImplementedError:
        pass
    else:  # pragma: no cover
        raise AssertionError("connector path must refuse loudly")
    log.setLevel(logging.NOTSET)

    # ---- C1a: the N-step burst gate + the num_computed_tokens arithmetic -------------
    class SO:
        def __init__(self, **kw):
            self.scheduled_new_reqs = []
            self.preempted_req_ids = set()
            self.scheduled_spec_decode_tokens = {}
            self.scheduled_encoder_inputs = {}
            self.has_structured_output_requests = False
            self.new_block_ids_to_zero = None
            self.num_scheduled_tokens = {"a": 1, "b": 1}
            self.__dict__.update(kw)

    assert burst_blocked(SO(), False, True, 8) is None
    for kw in (
        {"scheduled_new_reqs": [object()]},
        {"preempted_req_ids": {"a"}},
        {"scheduled_spec_decode_tokens": {"a": [7]}},
        {"scheduled_encoder_inputs": {"a": [0]}},
        {"has_structured_output_requests": True},
        {"new_block_ids_to_zero": [3]},
        {"num_scheduled_tokens": {}},
        {"num_scheduled_tokens": {"a": 1, "b": 2}},   # a chunked prefill in the batch
        {"num_scheduled_tokens": dict.fromkeys("abcdefghi", 1)},  # 9 > cap
    ):
        assert burst_blocked(SO(**kw), False, True, 8) is not None, kw
    # The waiting queue only blocks when the TTFT guard is on.
    assert burst_blocked(SO(), True, True, 8) == "waiting queue is not empty"
    assert burst_blocked(SO(), True, False, 8) is None
    # ...and the split the N=1 in-graph sampling commit uses does NOT see the queue at all:
    # one token delays no admission, so the TTFT argument that blocks a burst is vacuous.
    assert burst_blocked_batch(SO(), 8) is None
    assert burst_blocked_batch(SO(num_scheduled_tokens={"a": 2}), 8) is not None
    for kw in ({"scheduled_new_reqs": [object()]}, {"preempted_req_ids": {"a"}},
               {"has_structured_output_requests": True}):
        assert burst_blocked_batch(SO(**kw), 8) is not None, kw

    class SP:
        def __init__(self, **kw):
            self.temperature = 0.0
            self.logprobs = None
            self.prompt_logprobs = None
            self.min_tokens = 0
            self.bad_words = None
            self.allowed_token_ids = None
            self.logit_bias = None
            self.presence_penalty = 0.0
            self.frequency_penalty = 0.0
            self.repetition_penalty = 1.0
            self.__dict__.update(kw)

    class REQ:
        def __init__(self, **kw):
            self.sampling_params = SP()
            self.pooling_params = None
            self.resumable = False
            self.use_structured_output = False
            self.num_output_placeholders = 1
            self.num_prompt_tokens = 100
            self.max_tokens = 200
            self.num_tokens = 200
            self.num_computed_tokens = 200
            self.__dict__.update(kw)

    # The align gate, at block_size 16 / N 4: offsets 0..12 fit, 13..15 do not.
    fits = [c for c in range(16) if burst_request_blocked(REQ(), c, 4, 16, 32768) is None]
    assert fits == list(range(13)), fits
    assert len(fits) == 13, "13/16 coverage at N=4 is the number the plan sizes against"
    assert [c for c in range(16)
            if burst_request_blocked(REQ(), c, 8, 16, 32768) is None] == list(range(9))
    assert burst_request_blocked(REQ(), 0, 17, 16, 32768) is not None, "N > block_size"

    # ...and every non-greedy disqualifier, one at a time (offset 0 = align gate open).
    assert burst_request_blocked(REQ(), 0, 4, 16, 32768) is None
    for kw in (
        {"sampling_params": None},
        {"pooling_params": object()},
        {"resumable": True},
        {"use_structured_output": True},
        {"sampling_params": SP(temperature=0.7)},
        {"sampling_params": SP(logprobs=5)},
        {"sampling_params": SP(prompt_logprobs=1)},
        {"sampling_params": SP(min_tokens=8)},
        {"sampling_params": SP(bad_words=["x"])},
        {"sampling_params": SP(allowed_token_ids=[1])},
        {"sampling_params": SP(logit_bias={1: 2.0})},
        {"sampling_params": SP(presence_penalty=0.5)},
        {"sampling_params": SP(frequency_penalty=0.5)},
        {"sampling_params": SP(repetition_penalty=1.1)},
        # max_tokens caps the burst: 99 + 4 > min(32768, 100 + 2)
        {"max_tokens": 2, "num_prompt_tokens": 100},
    ):
        assert burst_request_blocked(REQ(**kw), 99 if "max_tokens" in kw else 0,
                                     4, 16, 32768) is not None, kw
    # ...and max_model_len does the same.
    assert burst_request_blocked(REQ(max_tokens=1 << 20), 32766, 4, 16, 32768) is not None

    # The align gate is the ONLY thing `burst_sampler_blocked` drops: at every offset in the
    # block it says yes where `burst_request_blocked` says "would cross a block boundary",
    # and it still refuses every non-greedy request and both length caps.
    for c in range(16):
        assert burst_sampler_blocked(REQ(), c, 1, 32768) is None, c
    assert [c for c in range(16) if burst_request_blocked(REQ(), c, 4, 16, 32768) is None] \
        == list(range(13)), "the align gate still applies to the BURST"
    assert burst_sampler_blocked(REQ(sampling_params=SP(temperature=0.7)), 0, 1, 32768)
    assert burst_sampler_blocked(REQ(sampling_params=SP(min_tokens=8)), 0, 1, 32768)
    # limit = min(32768, 100 + 2) = 102, so offset 101 still fits and 102 does not.
    assert burst_sampler_blocked(REQ(max_tokens=2), 101, 1, 32768) is None
    assert burst_sampler_blocked(REQ(max_tokens=2), 102, 1, 32768), "max_tokens caps N=1 too"
    assert burst_sampler_blocked(REQ(max_tokens=1 << 20), 32768, 1, 32768) is not None

    # The Rust runner's multi-burst headroom. At block_size 16 / N 4 a step can only take
    # 4 launches from a block-aligned offset, 3 from offset 4, and so on -- and NEVER 0,
    # because it is only ever asked about a request the single-burst gate already cleared.
    for c in range(16):
        k = burst_steps(c, 4, 16, 1 << 30, 8)
        assert k == max(1, (16 - c % 16) // 4), (c, k)
        if burst_request_blocked(REQ(), c, 4, 16, 32768) is None:
            assert k >= 1 and c % 16 + k * 4 <= 16, (c, k)
    assert burst_steps(0, 4, 16, 1 << 30, 8) == 4, "the whole block, and no more"
    assert burst_steps(0, 4, 16, 1 << 30, 2) == 2, "VTL_RUST_RUNNER_STEPS is the ceiling"
    assert burst_steps(0, 4, 16, 1 << 30, 1) == 1, "...and 1 disables the multi-launch step"
    # The length cap binds too: 10 tokens left at N=4 is two launches, not three.
    assert burst_steps(0, 4, 64, 10, 8) == 2
    assert burst_steps(0, 4, 64, 4, 8) == 1
    assert burst_steps(0, 4, 16, -1, 8) == 1, "an un-interned slot never multi-launches"

    # commit -> uncommit is exactly symmetric on both counters, and `is_prefill_chunk` is
    # computed BEFORE the placeholder bump (the ordering `_update_after_schedule` uses).
    r = REQ(num_tokens=100, num_computed_tokens=101, num_output_placeholders=2)
    burst_commit(r, 3)
    assert (r.num_computed_tokens, r.num_output_placeholders) == (104, 5), vars(r)
    assert r.is_prefill_chunk is False, "104 < 100 + 2 is false, and the bump comes after"
    burst_uncommit(r, 3)
    assert (r.num_computed_tokens, r.num_output_placeholders) == (101, 2)
    # THE PREFIX-CACHE INVARIANT. `AsyncScheduler._update_request_with_output` caches
    # `num_computed - num_output_placeholders` blocks; on a plain decode step that comes
    # out at `num_tokens - 1` (every token but the one just generated, whose position has
    # no KV yet). A burst must land on the same value whether it ran to completion or
    # stopped early -- that equality IS the guard against fingerprinting KV that was
    # never computed.
    for kept in (4, 2, 1):
        # Post-_update_after_schedule steady state: T=100, C=101, P=2.
        r = REQ(num_tokens=100, num_computed_tokens=101, num_output_placeholders=2)
        burst_commit(r, 3)                          # committed N=4
        assert (r.num_computed_tokens, r.num_output_placeholders) == (104, 5)
        r.num_tokens += kept                        # `_update_request_with_output` appends
        burst_uncommit(r, 4 - kept)                 # ...then the reconcile
        r.num_output_placeholders -= kept           # ...then AsyncScheduler's subtraction
        assert r.num_output_placeholders >= 0, kept
        assert (
            r.num_computed_tokens - r.num_output_placeholders == r.num_tokens - 1
        ), (kept, vars(r))
    # Saturating, never negative, on an accounting skew.
    r = REQ(num_computed_tokens=1, num_output_placeholders=1)
    burst_uncommit(r, 1000)
    assert (r.num_computed_tokens, r.num_output_placeholders) == (0, 0)

    # ---- C1b: commit_burst's per-slot eligibility intern (`_burst_gate`) --------------
    class FakeRustForget:
        def forget(self, request_id):
            pass

    mirror = RustMirror(FakeRustForget())
    req = REQ(max_tokens=2)  # lim = min(32768, 100 + 2) = 102
    # Miss: full evaluation, lim cached.
    assert _burst_gate(mirror, 5, req, 0, 4, 32768, 16) is None
    assert mirror._burst_lim[5] == 102
    # Hit: the cached lim wins even if an immutable field is mutated afterwards -- a
    # live scheduler never does this, the point is the intern no longer looks.
    req.sampling_params = SP(temperature=0.7)
    assert _burst_gate(mirror, 5, req, 0, 4, 32768, 16) is None
    # ...but the genuinely dynamic clauses still run on every hit.
    assert _burst_gate(mirror, 5, req, 13, 4, 32768, 16) is not None, "align gate"
    req.num_output_placeholders = -1
    assert _burst_gate(mirror, 5, req, 0, 4, 32768, 16) is not None, "placeholder count"
    req.num_output_placeholders = 1
    assert _burst_gate(mirror, 5, req, 99, 4, 32768, 16) is not None, "length cap (lim=102)"
    assert _burst_gate(mirror, 5, req, 98, 4, 32768, 16) is None, "98 + 4 == 102 still fits"
    # block_size=None (the N=1 in-graph path) skips the align gate on a hit too.
    assert _burst_gate(mirror, 5, req, 13, 1, 32768) is None

    # Miss on a permanently-ineligible request caches -1; every later hit short-circuits
    # without re-running `_burst_immutable_blocked`.
    bad = REQ(sampling_params=SP(temperature=0.7))
    assert _burst_gate(mirror, 9, bad, 0, 4, 32768, 16) == "not greedy"
    assert mirror._burst_lim[9] == -1
    bad.sampling_params = SP()  # now "eligible" -- the intern must still refuse
    assert _burst_gate(mirror, 9, bad, 0, 4, 32768, 16) == "ineligible request (interned)"

    # drop() clears the intern so a recycled slot re-evaluates.
    mirror._slots["r1"] = 9
    mirror.drop("r1")
    assert 9 not in mirror._burst_lim

    # ---- R9: packability intern, the batch cache, and the fast-hit guard -------------

    # `_pack_ok_clauses`: the prefill_stats one-shot latch is the only mutable member --
    # refused while set, packable once vLLM's stock `take_prefill_stats()` clears it.
    class PSReq:
        def __init__(self, **kw):
            self.client_index = 0
            self.resumable = False
            self.has_encoder_inputs = False
            self.trace_headers = None
            self.prefill_stats = object()  # non-None: a fresh request's live PrefillStats
            self.events = []
            self.__dict__.update(kw)

    pr = PSReq()
    assert _pack_ok_clauses(pr) is False, "prefill_stats is still set"
    pslot = 40
    if pslot in mirror._pack_ok:
        pass
    elif _pack_ok_clauses(pr):
        mirror._pack_ok.add(pslot)
    assert pslot not in mirror._pack_ok, "must not intern on a refusal"
    pr.prefill_stats = None  # what stock `take_prefill_stats()` leaves behind
    if pslot in mirror._pack_ok:
        pass
    elif _pack_ok_clauses(pr):
        mirror._pack_ok.add(pslot)
    assert pslot in mirror._pack_ok, "the latch cleared -- now interned"
    # Mutating a request afterwards does not un-intern it: the whole point of the cache
    # is to stop looking once a slot is proven.
    pr.client_index = 7
    assert pslot in mirror._pack_ok
    for kw in (
        {"client_index": 1}, {"resumable": True}, {"has_encoder_inputs": True},
        {"trace_headers": object()}, {"prefill_stats": object()}, {"events": [1]},
    ):
        assert _pack_ok_clauses(PSReq(**kw)) is False, kw

    # `mirror.drop()` discards the intern so a recycled slot re-proves itself.
    mirror._slots["ps1"] = pslot
    mirror.drop("ps1")
    assert pslot not in mirror._pack_ok

    # `_r9_cache_hit_counts`: identity, the numpy gather, and the zero-count guard.
    import numpy as np

    r2i = {"a": 0, "b": 1}
    cache = (r2i, [11, 12], [0, 1], ["req-a", "req-b"])
    counts = np.array([1, 1, 4], dtype=np.int64)  # index 2 belongs to some OTHER request
    assert _r9_cache_hit_counts(cache, r2i, counts) == [1, 1]
    assert _r9_cache_hit_counts(None, r2i, counts) is None
    assert _r9_cache_hit_counts(cache, dict(r2i), counts) is None, (
        "identity, not equality -- a freshly-built dict must miss even with equal contents"
    )
    zero = np.array([1, 0], dtype=np.int64)
    assert _r9_cache_hit_counts(cache, r2i, zero) is None, (
        "a zero anywhere forces the miss path -- a prefill chunk may have appeared"
    )

    # Phase A. `lean_blocked`'s clauses, the lean CachedRequestData, and the CHECK-mode
    # comparator -- all pure, so a stub scheduler and a stub dataclass are enough.
    class LeanSched:
        use_v2_model_runner = True
        use_pp = False
        scheduler_config = type("C", (), {"async_scheduling": True})()
        observability_config = type("O", (), {"enable_logging_iteration_details": False})()

    ls = LeanSched()
    assert lean_blocked(ls) is None
    ls.use_pp = True
    assert lean_blocked(ls) is None, "async scheduling makes PP not need new_token_ids"
    ls.scheduler_config = type("C", (), {"async_scheduling": False})()
    assert "PP" in lean_blocked(ls)
    ls.use_pp = False
    ls.observability_config = type("O", (), {"enable_logging_iteration_details": True})()
    assert "iteration-detail" in lean_blocked(ls)
    assert "V1 model runner" in lean_blocked(object())

    class LeanCRD:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class LeanBlocks:
        def __init__(self, ids):
            self.ids = ids

        def get_block_ids(self, allow_none=False):
            return self.ids

    class LeanReq:
        def __init__(self, rid, computed):
            self.request_id = rid
            self.num_computed_tokens = computed

    lreqs = [LeanReq("a", 7), LeanReq("b", 9)]
    lblocks = {"a": LeanBlocks(([1],)), "b": LeanBlocks(None)}
    lean = lean_cached_request_data(LeanCRD, lreqs, lblocks)
    assert lean.req_ids == ["a", "b"]
    assert lean.new_block_ids == [([1],), None]
    assert lean.num_computed_tokens == [7, 9]
    assert (lean.new_token_ids, lean.all_token_ids, lean.num_output_tokens) == ([], {}, [])
    assert lean.resumed_req_ids == set() and lean.resumed_req_ids is not lean.all_token_ids
    stock = LeanCRD(
        req_ids=["a", "b"], resumed_req_ids=set(), new_token_ids=[], all_token_ids={},
        new_block_ids=[([1],), None], num_computed_tokens=[7, 9], num_output_tokens=[3, 4],
    )
    assert lean_cached_check(lean, stock) is None, "num_output_tokens must not be compared"
    stock.num_computed_tokens = [7, 10]
    assert "num_computed_tokens" in lean_cached_check(lean, stock)
    stock.num_computed_tokens = [7, 9]
    stock.all_token_ids = {"a": [1]}
    assert "all_token_ids" in lean_cached_check(lean, stock), (
        "a dead field filled by stock means lean_blocked missed a consumer"
    )

    # Phase C's install predicate: every clause names a consumer the reuse would break.
    class RingSched:
        vllm_config = type("V", (), {
            "max_concurrent_batches": 2,
            "parallel_config": type("P", (), {"distributed_executor_backend": "mp"})(),
        })()
        use_v2_model_runner = True
        needs_kv_cache_zeroing = False
        enable_return_routed_experts = False
        encoder_cache_manager = type("E", (), {"freed": []})()

    assert ring_blocked(RingSched()) is None
    rs = RingSched()
    rs.needs_kv_cache_zeroing = True
    assert "zeroing" in ring_blocked(rs)
    rs = RingSched()
    rs.vllm_config = type("V", (), {
        "max_concurrent_batches": 1,
        "parallel_config": RingSched.vllm_config.parallel_config,
    })()
    assert "batch queue depth" in ring_blocked(rs), (
        "a ring deeper than the queue could be mutated mid-flight"
    )
    rs = RingSched()
    rs.vllm_config = type("V", (), {
        "max_concurrent_batches": 2,
        "parallel_config": type("P", (), {"distributed_executor_backend": "uni"})(),
    })()
    assert "uni" in ring_blocked(rs), "uni hands the runner the object we mutate"

    # `ring_reuse` refuses before it mutates anything: a changed running set, an extra
    # token, or a set some downstream code dirtied.
    class RingReq:
        use_structured_output = False

        def __init__(self, rid, computed, total):
            self.request_id = rid
            self.num_computed_tokens = computed
            self.num_tokens = total
            self.num_output_placeholders = 0
            self.is_prefill_chunk = False

    class RingCRD:
        def __init__(self, n):
            self.num_computed_tokens = [0] * n
            self.__dict__["_req_id_to_num_output_tokens"] = {"stale": 1}

    class RingSO:
        def __init__(self, n):
            self.scheduled_cached_reqs = RingCRD(n)
            self.finished_req_ids = set()
            self.preempted_req_ids = set()
            self.has_structured_output_requests = True
            self.vtl_burst_n = 4
            self.vtl_sample_in_graph = True

    class RingHost:
        defer_block_free = False
        sched_step_seq = 0
        _inflight_prefills = set()

    reqs = [RingReq("a", 10, 11), RingReq("b", 20, 21)]
    rslot = (RingSO(2), (5, 6), reqs)
    host = RingHost()
    host.requests = {r.request_id: r for r in reqs}
    assert ring_reuse(host, rslot, [(5, 1), (7, 1)]) is None, "a changed slot must refuse"
    assert ring_reuse(host, rslot, [(5, 1), (6, 2)]) is None, "a chunk must refuse"
    assert ring_reuse(host, rslot, [(5, 1)]) is None, "a shrunk batch must refuse"
    # Slot numbers recycle: "a" finished and a NEW request interned slot 5 -- the slot
    # list matches byte-for-byte, only the registry identity can tell them apart.
    host.requests["a"] = RingReq("a2", 10, 11)
    assert ring_reuse(host, rslot, [(5, 1), (6, 1)]) is None, "a recycled slot must refuse"
    host.requests["a"] = reqs[0]
    assert [r.num_computed_tokens for r in reqs] == [10, 20], "refusals must not mutate"
    so = ring_reuse(host, rslot, [(5, 1), (6, 1)])
    assert so is rslot[0]
    assert so.scheduled_cached_reqs.num_computed_tokens == [10, 20], (
        "the payload carries the PRE-advance values, like _make_cached_request_data"
    )
    assert [r.num_computed_tokens for r in reqs] == [11, 21]
    assert "_req_id_to_num_output_tokens" not in so.scheduled_cached_reqs.__dict__
    assert so.has_structured_output_requests is False, "recomputed, never |="
    assert not hasattr(so, "vtl_burst_n") and not hasattr(so, "vtl_sample_in_graph")
    rslot[0].finished_req_ids.add("dirty")
    assert ring_reuse(host, rslot, [(5, 1), (6, 1)]) is None, (
        "a dirtied slot set must refuse rather than serve a stale finished list"
    )

    # Phase B's arena comparator: same values compare equal regardless of container.
    d_ok = {
        "scheduled_running": [(1, 2)], "scheduled_admitted": [(3, 4, 5)],
        "running_new_blocks": [7, 8], "running_new_lens": [1, 1],
        "preempted": [9], "waiting_order": [10, 11],
    }
    assert arena_check(d_ok, zip([1], [2]), zip([3], [4], [5]), [7, 8], [1, 1], [9], [10, 11]) is None
    assert "waiting_order" in arena_check(
        d_ok, [(1, 2)], [(3, 4, 5)], [7, 8], [1, 1], [9], [11, 10]
    ), "order matters -- the waiting queue is rebuilt from it verbatim"
    assert "running_new_blocks" in arena_check(
        d_ok, [(1, 2)], [(3, 4, 5)], [7], [1, 1], [9], [10, 11]
    )

    # `RustMirror._batch_cache`: built by decide(), read on the next hit, and cleared by
    # every path that could make it point at the wrong requests.
    mirror._batch_cache = cache
    assert mirror._batch_cache is cache
    mirror._slots["cache-owner"] = 11
    mirror.drop("cache-owner")
    assert mirror._batch_cache is None, "drop() must invalidate the whole cache"

    # `_R9State`: boot-lifetime disable, and warn_once's one-line-per-kind budget.
    r9st = _R9State()
    r9st.live = True
    log.setLevel(logging.CRITICAL)
    r9st.warn_once("slots", "first")
    r9st.warn_once("slots", "second")  # suppressed -- already warned for this kind
    assert r9st.warned_kinds == {"slots"}
    r9st.disable("test")
    log.setLevel(logging.NOTSET)
    assert not r9st.live

    # ---- Port-2: LazySampled / _Row / the facade / materialization --------------------

    # LazySampled over plain nested lists (no numpy needed): `_Row` must answer len and
    # truthiness without converting anything, and materialize() must reproduce exactly what
    # `AsyncOutput.get_output`'s truncation loop produced.
    arr = [[11, 12, 13, 14], [21, 0, 0, 0], [0, 0, 0, 0]]
    ls = LazySampled([row[:] for row in arr], [3, 1, 0])
    assert len(ls) == 3 and bool(ls)
    row = ls[0]
    assert isinstance(row, _Row) and len(row) == 3 and bool(row)
    assert row.tolist() == [11, 12, 13], row.tolist()
    assert list(row) == [11, 12, 13]
    assert not ls[2] and len(ls[2]) == 0
    # `_Row.n = num_keep` is the `del new_token_ids[num_keep:]` of the fast path.
    row.n = 2
    assert len(row) == 2 and row.tolist() == [11, 12]
    ls.materialize()
    assert ls[0] == [11, 12, 13] and ls[1] == [21] and ls[2] == []
    assert list(ls) == [[11, 12, 13], [21], []]
    ls.materialize()  # idempotent
    assert ls[1] == [21]
    assert not isinstance(LazySampled(arr, [1]).arr[0][0], float)

    # The facade: a subclass whose count "properties" are plain class attributes, so the
    # instance dict wins. Getting this wrong is the whole reason it is not a global patch.
    class FakeReq:
        def __init__(self, prompt, outputs=()):
            self.request_id = "r1"
            self._all_token_ids = list(prompt) + list(outputs)
            self._output_token_ids = list(outputs)
            self.block_hashes = []
            self.cache_salt = None
            self.mm_features = []
            self.lora_request = None
            self.prompt_embeds = None
            self.spec_token_ids = []
            self.num_prompt_tokens = len(prompt)

        @property
        def all_token_ids(self):
            return self._all_token_ids

        @property
        def num_tokens(self):
            return len(self._all_token_ids)

        @property
        def num_tokens_with_spec(self):
            return len(self._all_token_ids) + len(self.spec_token_ids)

        @property
        def num_output_tokens(self):
            return len(self._output_token_ids)

    r = FakeReq(range(100), [7, 8])
    assert not facaded(r) and (r.num_tokens, r.num_output_tokens) == (102, 2)
    tok_facade_on(r)
    assert facaded(r), "the subclass swap must take"
    assert type(r) is facade_class(FakeReq), "the facade class is cached per base"
    assert (r.num_tokens, r.num_tokens_with_spec, r.num_output_tokens) == (102, 102, 2)
    # The writer's counter bumps, and `_all_token_ids` deliberately NOT growing.
    for keep in (1, 4):
        before = r.num_tokens
        r.num_tokens += keep
        r.num_output_tokens += keep
        assert r.num_tokens == before + keep
        # The alias tracks without a write of its own -- this is what lets the writer
        # drop the third bump, and what keeps a bail to stock `schedule()` correct.
        assert r.num_tokens_with_spec == r.num_tokens
    assert (r.num_tokens, r.num_output_tokens) == (107, 7)
    assert len(r._all_token_ids) == 102, "the list must stop growing under the facade"

    # Materialization: the store's 5 output tokens land in BOTH lists, the chain is replaced
    # from Rust's packed bytes, the mirror's push watermark moves with it, and the request is
    # bit-for-bit a stock request again -- with num_tokens back to a len().
    class FakeRust:
        def __init__(self, toks, packed):
            self.toks = toks
            self.packed = packed
            self.forgotten = []

        def slot_tokens(self, slot):
            return list(self.toks)

        def slot_hashes(self, slot):
            return self.packed

        def store_forget(self, slot):
            self.forgotten.append(slot)

    class FakeKv:
        def __init__(self, rust, slot):
            self._rust = rust
            self._mirror = type("M", (), {"_slots": {"r1": slot}, "_pushed": {slot: 0}})()

    packed = bytes(range(32)) * 3
    kv = FakeKv(FakeRust([1, 2, 3, 4, 5], packed), 4)
    tok_materialize(kv, r)
    assert not facaded(r) and type(r) is FakeReq
    assert r._all_token_ids[-5:] == [1, 2, 3, 4, 5]
    assert r._output_token_ids == [7, 8, 1, 2, 3, 4, 5]
    assert r.num_tokens == 107 == len(r._all_token_ids), "the facade's count was exact"
    assert r.num_output_tokens == 7
    assert len(r.block_hashes) == 3 and r.block_hashes[0] == packed[:32]
    assert kv._mirror._pushed[4] == 3, "the mirror must not re-push Rust's own chain"
    assert kv._rust.forgotten == [4]
    assert r._vtl_tok_off is True, "a materialized request is permanently stock"
    tok_materialize(kv, r)  # idempotent: not facaded any more, so a no-op
    assert kv._rust.forgotten == [4]

    # THE STEP FALLBACK, end to end: a facaded request plus a LazySampled batch, and after
    # it every consumer of the object path sees real lists on both sides.
    r2 = FakeReq(range(50), [9])
    tok_facade_on(r2)
    r2.num_tokens += 3        # ...matching the 3 tokens the fake store holds below
    r2.num_output_tokens += 3
    lazy = LazySampled([[31, 32, 0], [0, 0, 0]], [2, 0])
    kv2 = FakeKv(FakeRust([4, 5, 6], b"h" * 32), 6)
    TOK.warned = True  # keep the one-shot log out of `make check`'s output
    store_step_fallback(kv2, lazy, [("r1", 2, r2)], "self-check")
    assert lazy[0] == [31, 32] and lazy[1] == []
    assert not facaded(r2) and type(r2) is FakeReq
    assert r2._output_token_ids == [9, 4, 5, 6]
    assert r2.num_tokens == len(r2._all_token_ids) == 54
    assert r2.num_output_tokens == 4
    assert kv2._mirror._pushed[6] == 1

    # THE WIDER GUARD: stock's loop walks `scheduler_output.num_scheduled_tokens`, so a
    # store-owned request decide() never enumerated (already finished, or a step it refused
    # outright) still has to be handed back before stock reads its token lists.
    r3 = FakeReq(range(10), [1])
    tok_facade_on(r3)
    r3.num_tokens += 2
    r3.num_output_tokens += 2

    class FakeSched:
        requests = {"r1": r3, "gone": None}

    class FakeSO:
        num_scheduled_tokens = {"r1": 1, "gone": 1, "never-seen": 1}

    class FakeMRO:
        sampled_token_ids = LazySampled([[5, 0], [0, 0]], [1, 0])

    kv3 = FakeKv(FakeRust([2, 3], b"z" * 32), 9)
    store_full_fallback(FakeSched(), kv3, FakeSO(), FakeMRO())
    assert not facaded(r3), "a request decide() never saw must still be materialized"
    assert r3._output_token_ids == [1, 2, 3]
    assert r3.num_tokens == len(r3._all_token_ids) == 13

    # tok_store_init's refusals: anything the Rust hasher will not hash, and any seed tail
    # that says the two hashers disagree about how many blocks are complete.
    class NoopRust:
        def __init__(self):
            self.init = None

        def store_init(self, slot, pending, n_tok, n_out):
            self.init = (slot, pending, n_tok, n_out)

    nr = NoopRust()
    fresh = FakeReq(range(20), [7])
    fresh.block_hashes = [b"x" * 32]        # 16 tokens hashed, 5-token tail
    assert tok_store_init(nr, fresh, 3, 16) is True
    assert nr.init == (3, [16, 17, 18, 19, 7], 21, 1), nr.init
    for attr, value in (("cache_salt", "s"), ("mm_features", [object()]),
                        ("lora_request", object()), ("prompt_embeds", object())):
        bad = FakeReq(range(20), [7])
        bad.block_hashes = [b"x" * 32]
        setattr(bad, attr, value)
        assert tok_store_init(NoopRust(), bad, 3, 16) is False, attr
    behind = FakeReq(range(40), [7])
    behind.block_hashes = [b"x" * 32]       # 16 hashed but 41 tokens -> a 25-token tail
    assert tok_store_init(NoopRust(), behind, 3, 16) is False, "hasher is behind the store"
    exact = FakeReq(range(32))
    exact.block_hashes = [b"x" * 32, b"y" * 32]
    nr2 = NoopRust()
    assert tok_store_init(nr2, exact, 3, 16) is True
    assert nr2.init == (3, [], 32, 0), "a block-aligned request seeds an empty tail"

    # TOK.disable is permanent for the boot.
    saved_tok = TOK.live
    try:
        TOK.live = True
        log.setLevel(logging.CRITICAL)
        TOK.disable("test")
        log.setLevel(logging.NOTSET)
        assert not TOK.live
    finally:
        TOK.live = saved_tok

    # reraise_fatal: a crate panic (BaseException, not Exception) is swallowed by the
    # guards; interpreter signals still get through.
    class FakePanic(BaseException):
        pass

    reraise_fatal(FakePanic())
    for fatal in (KeyboardInterrupt(), SystemExit()):
        try:
            reraise_fatal(fatal)
        except BaseException as exc:  # noqa: BLE001 - that is the point
            assert exc is fatal
        else:  # pragma: no cover
            raise AssertionError("interpreter signals must propagate")

    # _refuse_unported: authority mode must refuse anything it did not port, rather than
    # inherit a confident answer computed from the frozen python coordinator.
    class FakeBase:
        def ported(self):
            return "python"

        def unported(self):
            return "stale python answer"

    class FakeAuthority(FakeBase):
        def ported(self):
            return "rust"

    _refuse_unported(FakeAuthority, FakeBase)
    assert FakeAuthority().ported() == "rust"
    try:
        FakeAuthority().unported()
    except NotImplementedError as exc:
        assert "unported" in str(exc) and "refusing" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unported base methods must refuse")

    # No vLLM here, so the ambient-config probe must answer "unknown", not crash.
    assert kv_transfer_configured() is None

    # build_config degrades to a reason string instead of raising when vLLM is absent.
    cfg, reason = build_config(object(), False)
    assert cfg is None and isinstance(reason, str) and reason, reason

    # schedule_supported names every blocking feature it finds.
    class Sched:
        connector = object()
        lora_config = object()
        num_spec_tokens = 0

    reason = schedule_supported(Sched())
    assert "connector" in reason and "lora" in reason, reason

    class PlainSched:
        pass

    assert schedule_supported(PlainSched()) is None

    # ---- refuse(): warn by default, RAISE under VTL_RUST_SCHED_REQUIRE=1 ----
    # The default must never raise: a submission that cannot use the Rust scheduler still
    # has to serve. The gate exists so a BENCH cannot silently measure a stock engine.
    os.environ.pop("VTL_RUST_SCHED_REQUIRE", None)
    refuse("unported kv cache spec SlidingWindowSpec")  # must not raise
    for val in ("1", "true", "on", "YES"):
        os.environ["VTL_RUST_SCHED_REQUIRE"] = val
        try:
            refuse("unported kv cache spec SlidingWindowSpec")
        except RuntimeError as e:
            assert "SlidingWindowSpec" in str(e), e   # the reason must survive
        else:
            raise AssertionError(f"VTL_RUST_SCHED_REQUIRE={val!r} must raise")
    os.environ["VTL_RUST_SCHED_REQUIRE"] = "0"
    refuse("still just a warning")  # explicit 0 is off, not "set therefore on"
    os.environ.pop("VTL_RUST_SCHED_REQUIRE", None)

    try:
        import vtl_sched  # noqa: F401

        print("rust_sched self-check ok (vtl_sched extension present)")
    except Exception:
        print("rust_sched self-check ok (vtl_sched extension absent -- pure-python parts only)")


if __name__ == "__main__":
    _self_check()
