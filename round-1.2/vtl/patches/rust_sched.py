"""Rust KV-cache / scheduler core (`vtl_sched`), staged behind four env gates.

WS4. The crate in ``round-1.2/vtl-sched/`` is a logic-preserving port of vLLM v0.25.0's
``v1/core/`` block metadata, prefix cache, hybrid coordinator and ``schedule()`` decision
loop. This module is the only thing that wires it into a live engine.

Gates (all default OFF, all independent except where noted)::

    VTL_ENABLE_RUST_SCHED=1        install this patch at all (registry gate)
    VTL_RUST_SCHED_SHADOW=1        run Rust ALONGSIDE Python every call and compare
    VTL_RUST_SCHED_SHADOW_STRICT=1 make a shadow mismatch raise (TESTS ONLY). In serving
                                   the outer guard catches the AssertionError and simply
                                   disables the mirror, so strict buys nothing there --
                                   it exists so a pytest driving the manager directly
                                   fails on the first divergence instead of counting.
    VTL_RUST_SCHED=1               make Rust AUTHORITATIVE for the KVCacheManager surface
    VTL_RUST_SCHED_FULL=1          also run the Rust schedule() loop (implies VTL_RUST_SCHED)
    VTL_RUST_SCHED_RADIX=1         use the radix index instead of the flat hash map
                                   (same answers; see vtl-sched/src/radix.rs)
    VTL_RUST_SCHED_UFO=1           R6a: batch the per-step stop decision of
                                   update_from_output into ONE Rust call (needs _FULL)
    VTL_RUST_SCHED_UFO_SHADOW=1    keep Python's check_stop authoritative and only log
                                   where the Rust verdict disagrees
    VTL_RUST_SCHED_TABLE=1         R6b: schedule from the Rust-RESIDENT request table
                                   instead of re-marshalling every running request each
                                   step (needs _FULL *and* _UFO -- the per-step token
                                   deltas ride on update_step)
    VTL_RUST_SCHED_TABLE_SHADOW=1  keep the marshalled path authoritative and only log
                                   where the resident table disagrees with pack_req
    VTL_RUST_SCHED_SPEC=1          R6c: precompute the next step on the Rust worker
                                   thread between update_from_output and schedule
                                   (needs _TABLE; mutually exclusive with _TABLE_SHADOW)
    VTL_SCHED_TIMING=1             log-only p50/p95 of the schedule() phases. Independent
                                   of every other gate.
    VTL_RUST_SCHED_TOKSTORE=1      Port-2: Rust owns each slot's output tokens, counters and
                                   block-hash chain (needs R8 *and* VTL_RUST_HASHER). The
                                   sampled ids cross as numpy, Python's ``Request`` degrades
                                   to three int counters, and the rare paths that need the
                                   real lists materialize them back from the crate.
    VTL_RUST_SCHED_TOKSTORE_SHADOW=1  keep Python authoritative and only drive + diff the
                                   Rust store (hash chains + both counters) per step.

Why shadow first: this replaces the most correctness-critical component in the engine,
and LFM2's hybrid layout (6 full-attention + 10 short-conv/mamba groups,
``mamba_cache_mode=align``) has per-kind rules that fail SILENTLY when wrong -- a mamba
state cached at the wrong boundary produces wrong tokens, not an exception. Shadow mode
runs Python as the authority and only logs divergence, so a soak proves parity before any
flag flips.

Composition with the existing patches:
  * ``vtl/patches/kv_cache_manager.py`` rebinds ``sched_mod.KVCacheManager`` to a subclass
    carrying ``plan_request`` / ``free_blocks``. We apply AFTER it and subclass whatever is
    bound at that moment, so those signals survive; in authority mode ``plan_request`` is
    re-pointed at the Rust cache-hit walk (the Python coordinator would be stale).
  * ``vtl/patches/sched_policy.py`` wraps ``Scheduler.schedule`` with the cache-aware SJF
    reorder. ``VTL_RUST_SCHED_FULL=1`` SUPERSEDES that wrapper -- the same SJF key runs
    inside the Rust loop (``sched.rs::reorder_waiting``) so the ordering is preserved, not
    dropped. Which of the two is active is logged at install.

Refusal, not approximation: the port covers exactly the served configuration. Anything
else (KV/EC connectors, LoRA, encoder inputs, speculative decoding, priority policy,
sliding-window / chunked-local / cross-attention specs, sparse prefix-cache retention,
DCP/PCP, KV cache events) makes ``_build_config`` / ``_schedule_supported`` return a
reason string, which is logged once and leaves stock vLLM in charge.
"""

from __future__ import annotations

import logging
import os
import struct

from vtl.registry import already_patched, mark_patched, register_patch

# Must be a child of "vllm.vtl": a bare "vtl" logger's INFO records are dropped.
log = logging.getLogger("vllm.vtl.rust_sched")

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# RequestStatus values the Rust core needs (vllm/v1/request.py).
_ST_WAITING, _ST_RUNNING, _ST_PREEMPTED = 0, 1, 2

# R8 shadow only: the record's f64 timestamp sits at byte 9 (tag + version + u16 reserved
# + u32 engine_index). Layout owner is vtl/patches/shm_ipc.py.
_RAW_TS = struct.Struct("<d")


def env_on(name: str) -> bool:
    """Env gate parsing. Pure python, exercised by the self-check without the crate."""
    return os.environ.get(name, "").strip().lower() in _TRUTHY


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
    table_shadow = table and env_on("VTL_RUST_SCHED_TABLE_SHADOW")
    r8 = (
        ufo
        and env_on("VTL_RUST_SCHED_R8")
        and env_on("VTL_SHM_IPC")
        and env_on("VTL_SHM_IPC_RAW")
    )
    return {
        "shadow": env_on("VTL_RUST_SCHED_SHADOW"),
        "strict": env_on("VTL_RUST_SCHED_SHADOW_STRICT"),
        "authority": env_on("VTL_RUST_SCHED") or full,
        "full": full,
        "radix": env_on("VTL_RUST_SCHED_RADIX"),
        "ufo": ufo,
        "ufo_shadow": env_on("VTL_RUST_SCHED_UFO_SHADOW"),
        "table": table,
        "table_shadow": table_shadow,
        # Shadow keeps the MARSHALLED call authoritative, and that call is what resyncs
        # the table -- so the resident fast path (and therefore speculation, which can
        # only be consumed by it) must stay off while shadowing.
        "spec": table and not table_shadow and env_on("VTL_RUST_SCHED_SPEC"),
        "timing": env_on("VTL_SCHED_TIMING"),
        # R8 rides on UFO (it IS update_step, plus the pack) and on the shm raw record
        # being the live wire format -- there is no point building bytes the output
        # thread would have to decode again. Shadow keeps python authoritative.
        "r8": r8,
        "r8_shadow": r8 and env_on("VTL_RUST_SCHED_R8_SHADOW"),
        # B3: block hashing in Rust. Independent of everything above -- it only needs the
        # extension importable, so it stays armable with the scheduler flags off.
        "hasher": env_on("VTL_RUST_HASHER"),
        # C1a: N-step decode burst commitment. Needs the full Rust loop (the commit
        # extends the resident table in the same place the schedule decisions land).
        "nstep": full and env_on("VTL_NSTEP"),
        # Port-2: Rust owns the per-slot token list, counters and block-hash chain.
        #
        # Rides on R8 (it IS `update_step_pack`, driven off the sampler's numpy array) and on
        # the Rust hasher -- with two hash implementations live, one would own the prompt
        # blocks and the other the decode blocks, and a divergence between them is a silent
        # prefix-cache key-space fork, not an exception.
        #
        # ...and refuses the two OTHER shadow arms outright: both keep Python's
        # `append_output_token_ids` authoritative, and the store has already appended the
        # same tokens by then, so the pair would double-count every token. Those arms are
        # diagnostics; the token store simply is not available while one is on.
        "tokstore": (
            r8
            and not env_on("VTL_RUST_SCHED_R8_SHADOW")
            and not env_on("VTL_RUST_SCHED_UFO_SHADOW")
            and env_on("VTL_RUST_HASHER")
            and env_on("VTL_RUST_SCHED_TOKSTORE")
        ),
        "tokstore_shadow": (
            r8
            and not env_on("VTL_RUST_SCHED_R8_SHADOW")
            and not env_on("VTL_RUST_SCHED_UFO_SHADOW")
            and env_on("VTL_RUST_HASHER")
            and env_on("VTL_RUST_SCHED_TOKSTORE")
            and env_on("VTL_RUST_SCHED_TOKSTORE_SHADOW")
        ),
    }


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
    parity-tested in ``bench/test_rust_sched_parity.py``.
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
# shadow mode
# --------------------------------------------------------------------------


class ShadowState:
    """Mismatch bookkeeping. Monotonic counter, bounded logging, never raises in serving."""

    LOG_BUDGET = 20

    def __init__(self, strict: bool):
        self.strict = strict
        self.mismatches = 0
        self.calls = 0

    def check(self, what: str, py, rs, ctx: str = "") -> None:
        self.calls += 1
        if py == rs:
            return
        self.mismatches += 1
        if self.mismatches <= self.LOG_BUDGET:
            log.error(
                "rust_sched SHADOW MISMATCH #%d in %s: python=%r rust=%r %s",
                self.mismatches,
                what,
                py,
                rs,
                ctx,
            )
        elif self.mismatches == self.LOG_BUDGET + 1:
            log.error("rust_sched: further shadow mismatches suppressed")
        if self.strict:
            raise AssertionError(
                f"rust_sched shadow mismatch in {what}: python={py!r} rust={rs!r} {ctx}"
            )


def _install_shadow(base, mirror_modes):
    """Subclass the currently-bound KVCacheManager and mirror every call into Rust."""
    import vtl_sched

    class VtlShadowKVCacheManager(base):
        __vtl_rust_shadow__ = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._rust = None
            self._shadow = ShadowState(mirror_modes["strict"])
            if kv_transfer_configured():
                log.warning("rust_sched: shadow disabled -- a KV connector is configured")
                return
            cfg, reason = build_config(self, mirror_modes["radix"])
            if reason is not None:
                log.warning("rust_sched: shadow disabled -- %s", reason)
                return
            try:
                self._rust = vtl_sched.KvManager(cfg)
            except BaseException as exc:
                reraise_fatal(exc)
                log.warning("rust_sched: shadow disabled -- crate rejected config: %r", exc)
                return
            self._mirror = RustMirror(self._rust)
            log.info(
                "rust_sched: SHADOW mode active (%d groups, %d blocks, strict=%s)",
                self._rust.num_groups,
                cfg["num_blocks"],
                mirror_modes["strict"],
            )

        # --- mirrored calls -------------------------------------------------

        def new_step_starts(self):
            super().new_step_starts()
            if self._rust is not None:
                self._rust.new_step_starts()

        def get_computed_blocks(self, request):
            blocks, num = super().get_computed_blocks(request)
            if self._rust is not None:
                try:
                    slot = self._mirror.slot(request)
                    rnum = self._rust.get_computed_blocks(
                        slot,
                        int(request.num_tokens),
                        int(request.num_preemptions),
                        bool(request.skip_reading_prefix_cache),
                    )
                    self._shadow.check(
                        "get_computed_blocks.num_tokens", num, rnum, request.request_id
                    )
                    py_ids = blocks.get_block_ids()
                    for g in range(self._rust.num_groups):
                        n = self._rust.pending_hit_into_buffer(g)
                        rs = self._rust.buffer(g)[:n].tolist()
                        self._shadow.check(
                            f"get_computed_blocks.blocks[{g}]",
                            list(py_ids[g]),
                            rs,
                            request.request_id,
                        )
                except BaseException as exc:
                    self._disable("get_computed_blocks", exc)
            return blocks, num

        def allocate_slots(self, request, num_new_tokens, *args, **kwargs):
            result = super().allocate_slots(request, num_new_tokens, *args, **kwargs)
            if self._rust is not None:
                try:
                    self._mirror_allocate(request, num_new_tokens, result, args, kwargs)
                except BaseException as exc:
                    self._disable("allocate_slots", exc)
            return result

        def _mirror_allocate(self, request, num_new_tokens, result, args, kwargs):
            num_new_computed = kwargs.get("num_new_computed_tokens", 0)
            if args:
                num_new_computed = args[0]
            new_computed_blocks = kwargs.get("new_computed_blocks")
            if len(args) > 1:
                new_computed_blocks = args[1]
            lookahead = kwargs.get("num_lookahead_tokens", 0)
            if len(args) > 2:
                lookahead = args[2]
            slot = self._mirror.slot(request)
            ok = self._rust.allocate_slots(
                slot,
                int(num_new_tokens),
                int(num_new_computed),
                new_computed_blocks is not None,
                int(lookahead),
                int(request.num_computed_tokens),
                int(request.num_tokens),
                status_code(request),
                bool(kwargs.get("has_scheduled_reqs", True)),
            )
            self._shadow.check(
                "allocate_slots.fits", result is not None, ok, request.request_id
            )
            if result is None or not ok:
                return
            py_ids = result.get_block_ids()
            for g in range(self._rust.num_groups):
                n = self._rust.new_blocks_into_buffer(g)
                self._shadow.check(
                    f"allocate_slots.new_blocks[{g}]",
                    list(py_ids[g]),
                    self._rust.buffer(g)[:n].tolist(),
                    request.request_id,
                )
            # Eviction order is not directly observable from Python, but it fully
            # determines which block IDs come back next, so ID parity over a long trace
            # IS eviction parity. Free-block counts catch accounting drift immediately.
            self._shadow.check(
                "free_blocks",
                self.block_pool.get_num_free_blocks(),
                self._rust.num_free_blocks,
                request.request_id,
            )

        def cache_blocks(self, request, num_computed_tokens):
            super().cache_blocks(request, num_computed_tokens)
            if self._rust is not None:
                try:
                    slot = self._mirror.slot(request)
                    self._rust.cache_blocks(slot, int(num_computed_tokens))
                except BaseException as exc:
                    self._disable("cache_blocks", exc)

        def free(self, request):
            super().free(request)
            if self._rust is not None:
                try:
                    slot = self._mirror.slot(request)
                    self._rust.free(slot)
                    self._shadow.check(
                        "free.free_blocks",
                        self.block_pool.get_num_free_blocks(),
                        self._rust.num_free_blocks,
                        request.request_id,
                    )
                    self._mirror.drop(request.request_id)
                except BaseException as exc:
                    self._disable("free", exc)

        def get_num_common_prefix_blocks(self, running_request_id):
            out = super().get_num_common_prefix_blocks(running_request_id)
            if self._rust is not None:
                try:
                    slot = self._rust.lookup(running_request_id)
                    if slot is not None:
                        self._shadow.check(
                            "num_common_prefix_blocks",
                            list(out),
                            list(self._rust.num_common_prefix_blocks(slot)),
                            running_request_id,
                        )
                except BaseException as exc:
                    self._disable("common-prefix", exc)
            return out

        def reset_prefix_cache(self):
            out = super().reset_prefix_cache()
            if self._rust is not None:
                try:
                    self._shadow.check(
                        "reset_prefix_cache", out, self._rust.reset_prefix_cache()
                    )
                except BaseException as exc:
                    self._disable("reset_prefix_cache", exc)
            return out

        def evict_blocks(self, block_ids):
            super().evict_blocks(block_ids)
            if self._rust is not None:
                try:
                    self._rust.evict_blocks([int(b) for b in block_ids])
                    self._shadow.check(
                        "evict_blocks.free_blocks",
                        self.block_pool.get_num_free_blocks(),
                        self._rust.num_free_blocks,
                    )
                except BaseException as exc:
                    self._disable("evict_blocks", exc)

        def _disable(self, where: str, exc: BaseException) -> None:
            """Drop the mirror. Never lets a crate panic reach EngineCore."""
            reraise_fatal(exc)
            log.exception("rust_sched: shadow %s failed; mirror disabled", where)
            self._rust = None

        @property
        def rust_shadow_mismatches(self) -> int:
            return self._shadow.mismatches

    return VtlShadowKVCacheManager


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


# pack_req field order, for shadow-mode divergence reports.
_REQ_FIELDS = (
    "slot",
    "num_tokens",
    "num_tokens_with_spec",
    "num_computed_tokens",
    "num_output_placeholders",
    "num_prompt_tokens",
    "max_tokens",
    "status",
    "num_preemptions",
    "is_prefill_chunk",
    "skip_reading_prefix_cache",
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
        int(request.num_tokens_with_spec),
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

    __slots__ = ("live", "shadow", "armed", "hash_block_size", "facaded",
                 "materialized", "divergences", "warned")

    def __init__(self) -> None:
        self.live = False          # numpy fast path + facade installation
        self.shadow = False        # python authoritative, store driven and compared
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
        if self.live or self.shadow:
            log.error("rust_sched: token store disabled for this boot -- %s", why)
        self.live = False
        self.shadow = False


TOK = _TokState()


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
                "num_tokens_with_spec": 0,
                "num_output_tokens": 0,
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
    n_spec = int(request.num_tokens_with_spec)
    n_out = int(request.num_output_tokens)
    # The class swap comes first: while `base` is bound, `num_tokens` is a getter-only
    # property and the assignment below would raise.
    request.__class__ = facade_class(base)
    request.num_tokens = n_tok
    request.num_tokens_with_spec = n_spec
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
    for name in ("num_tokens", "num_tokens_with_spec", "num_output_tokens"):
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
    log.info("rust_sched: token store armed (hash_block_size=%d, shadow=%s)", hbs, TOK.shadow)
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

    shadow = m["ufo_shadow"]
    spec = m["spec"]
    r8_shadow = m["r8_shadow"]
    # Port-2. STORE gates the crate-side bookkeeping (both arms); TOK.live gates the numpy
    # fast path and the facade, and is the flag that a failure turns off for the boot.
    STORE = m["tokstore"]
    TOK.shadow = m["tokstore_shadow"]
    TOK.live = STORE and not TOK.shadow

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
            and not scheduler_output.has_structured_output_requests
            and not self.log_stats
            and self.connector is None
            and self.finished_req_ids_dict is None
            and not self.defer_block_free
            and not self.enable_kv_cache_events
            and not self.enable_return_routed_experts
            and (self.perf_metrics is None or not self.perf_metrics.is_enabled())
        )

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
        """
        rust, mirror = kv._rust, kv._mirror
        sampled = model_runner_output.sampled_token_ids
        lazy = TOK.live and pack and isinstance(sampled, LazySampled)
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
            if pack and not (
                request.client_index == 0
                and not request.resumable
                and not request.has_encoder_inputs
                and request.trace_headers is None
                and request.prefill_stats is None
                and not request.events
            ):
                # The record has no slot for any of these: a second client index, a
                # resumable session's re-enqueue, encoder-cache frees, trace headers,
                # prefill stats or lifecycle events. Objects for the whole step instead.
                pack = False
            slot = mirror.slot(request)
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
                # (`_vtl_tok_on`, not `facaded()`: the SHADOW arm installs no facade, and
                # re-seeding the store every step would mask the drift it exists to catch.)
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
            out, record = rust.update_step_pack_np(
                sampled.arr, rows, counts, slots, int(self.max_model_len),
                0, monotonic(), (),
            )
            self._vtl_r8_record = record
            # `record is None` (Rust refused the pack) is NOT handled here on purpose:
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
        if TOK.shadow:
            self._vtl_tok_shadow = (slots, cu, toks, [v[0] for v in out], expect)
        return {rid: (n, *v) for (rid, n, _r), v in zip(expect, out)}

    def store_compare(self, kv) -> None:
        """VTL_RUST_SCHED_TOKSTORE_SHADOW: Python stays authoritative, Rust is checked.

        Runs AFTER stock's ``append_output_token_ids`` (so the request holds this step's
        tokens and hashes), drives the store with the SAME accepted counts, and diffs the
        two hash chains and both counters per request.

        Deliberately NOT a record byte-compare: running both pack arms in one step would
        apply every verdict twice. Record byte-identity is proved instead by the crate's
        ``record_is_identical_with_the_store`` golden test, which is exhaustive and free.
        """
        state = self._vtl_tok_shadow
        self._vtl_tok_shadow = None
        if state is None:
            return
        slots, cu, toks, accepted, batch = state
        try:
            kv._rust.store_apply(slots, cu, toks, accepted)
            for slot, (rid, _n, request) in zip(slots, batch):
                want = (int(request.num_tokens), int(request.num_output_tokens))
                got = kv._rust.store_counts(slot)
                if got != want:
                    TOK.divergences += 1
                    log.error("rust_sched: token store COUNTER divergence on %s -- "
                              "python=%r rust=%r", rid, want, got)
                py_hashes = b"".join(request.block_hashes)
                rs_hashes = kv._rust.slot_hashes(slot)
                if rs_hashes != py_hashes:
                    TOK.divergences += 1
                    log.error("rust_sched: token store HASH CHAIN divergence on %s -- "
                              "python=%d hashes, rust=%d hashes", rid,
                              len(py_hashes) // 32, len(rs_hashes or b"") // 32)
        except BaseException as exc:
            reraise_fatal(exc)
            log.exception("rust_sched: token store shadow comparison failed")
            TOK.disable("shadow comparison raised")

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
        # C1a side channel: how many tokens THIS step's schedule committed to. Read once
        # per step (not per request) so the reconcile stays FIFO-correct under async
        # scheduling, where two steps are in flight and each has its own factor.
        self._vtl_burst_n = getattr(scheduler_output, "vtl_burst_n", 1) or 1
        # False the moment ANY request in this step's batch was decided by Python instead
        # of by update_step -- which is exactly when the resident table missed a token
        # delta and must be resynced before it can be scheduled from again.
        self._vtl_ufo_clean = True
        kv = self.kv_cache_manager
        sampled = model_runner_output.sampled_token_ids
        if (
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
            if self._vtl_ufo is None:
                self._vtl_ufo_clean = False
        elif sampled:
            # Tokens were produced but nothing was portable: stock check_stop ran for the
            # whole batch, so update_step applied no delta at all.
            self._vtl_ufo_clean = False
        try:
            if self._vtl_r8_record is not None and not r8_shadow:
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
                if self._vtl_r8_record is not None:
                    r8_compare(self, outputs)
            if TOK.shadow and self._vtl_tok_shadow is not None:
                store_compare(self, kv)
        finally:
            self._vtl_ufo = None
            self._vtl_r8_record = None
            self._vtl_burst_n = 1
            self._vtl_tok_shadow = None
        if spec:
            maybe_kick(self, kv)
        return outputs

    def r8_compare(self, outputs) -> None:
        """VTL_RUST_SCHED_R8_SHADOW: Python stays authoritative, Rust's bytes are checked.

        Packs the objects stock just built with the SAME layout the frontend decodes and
        diffs the two byte strings. A mismatch is the only signal that matters -- if these
        agree over a full replay, flipping the authority arm on cannot change the wire.
        """
        rust_bytes = self._vtl_r8_record
        try:
            from vtl.patches.shm_ipc import raw_packable, raw_pack_into, raw_size

            eco = outputs.get(0) if outputs else None
            if eco is None or len(outputs) != 1:
                log.error("rust_sched: R8 shadow -- python produced %d client batches, "
                          "rust produced 1", len(outputs or ()))
                return
            # Rust stamps its own timestamp; copy python's in so the diff is about the
            # payload, not about which side called monotonic() first.
            eco.engine_index = 0
            eco.timestamp = _RAW_TS.unpack_from(rust_bytes, 9)[0]
            if not raw_packable(eco):
                log.error("rust_sched: R8 shadow -- rust packed a batch python refuses")
                return
            buf = bytearray(raw_size(eco))
            raw_pack_into(buf, eco)
            if bytes(buf) != rust_bytes:
                log.error(
                    "rust_sched: R8 shadow BYTE DIVERGENCE\n  python=%s\n  rust  =%s",
                    buf.hex(), rust_bytes.hex(),
                )
        except BaseException as exc:
            reraise_fatal(exc)
            log.exception("rust_sched: R8 shadow comparison failed")

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

        if shadow:
            kept, stopped = wrapped_urwo(self, request, new_token_ids)
            want = status != 0
            if stopped != want or len(kept) != num_keep:
                log.error(
                    "rust_sched: UFO shadow divergence on %s -- python (%d kept, stopped=%s) "
                    "vs rust (%d kept, status=%d)",
                    request.request_id, len(kept), stopped, num_keep, status,
                )
            elif want:
                # stop_reason is compared too: EOS-also-in-stop_token_ids is the one
                # branch where status matches but the precedence choice shows only here.
                rust_reason = stop_reason if stop_reason >= 0 else None
                if (request.status is not _STATUS_FROM_CODE[status]
                        or request.stop_reason != rust_reason):
                    log.error(
                        "rust_sched: UFO shadow status divergence on %s -- python "
                        "(%s, reason=%s) vs rust (%s, reason=%s)",
                        request.request_id, request.status, request.stop_reason,
                        _STATUS_FROM_CODE[status], rust_reason,
                    )
            return kept, stopped

        if facaded(request):
            # Port-2: the tokens and the hash chain are already in Rust (appended by
            # `update_step_pack_np` for exactly `num_keep`), so the whole writer is two
            # counter bumps. `num_tokens_with_spec` moves with `num_tokens` because
            # `spec_token_ids` is empty on every path that reaches here (the Rust schedule
            # loop refuses a spec config outright).
            request.num_tokens += num_keep
            request.num_tokens_with_spec += num_keep
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
    scheduler_cls._vtl_burst_n = 1
    scheduler_cls._vtl_tok_shadow = None
    mark_patched(update_from_output, wrapped_ufo, patch="rust_sched_ufo")
    mark_patched(_update_request_with_output, wrapped_urwo, patch="rust_sched_ufo")
    scheduler_cls.update_from_output = update_from_output
    scheduler_cls._update_request_with_output = _update_request_with_output
    if STORE:
        _install_preempt_hook(scheduler_cls)
        if TOK.live:
            _install_async_output()
    log.info(
        "rust_sched: UFO batched stop decision active "
        "(shadow=%s, kick=%s, r8=%s%s, tokstore=%s%s)",
        shadow, spec, _r8[0], " SHADOW" if r8_shadow else "",
        STORE, " SHADOW" if TOK.shadow else "",
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
        return wrapped(self, request, *args, **kwargs)

    scheduler_cls._preempt_request = mark_patched(
        _preempt_request, wrapped, patch="rust_sched_tokstore"
    )


def _install_full_schedule(scheduler_cls, sjf_enabled: bool, m: dict):
    """Replace ``Scheduler.schedule`` with the Rust-driven decision loop."""
    import time

    import vtl_sched
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
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
    TABLE, TABLE_SHADOW, SPEC = m["table"], m["table_shadow"], m["spec"]
    RESIDENT = TABLE and not TABLE_SHADOW
    TIMING = m["timing"]
    ns = time.monotonic_ns
    timers = PhaseTimers() if TIMING else None
    table_shadow_state = ShadowState(strict=False) if TABLE_SHADOW else None

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

    def shadow_table(kv, running, by_slot):
        """Log where the resident table disagrees with this step's ``pack_req`` tuples.

        Called only from the marshalled arm, and only in TABLE_SHADOW mode: ``table_entry``
        is non-invalidating and therefore reads THROUGH a pending speculation, but
        ``modes()`` forces SPEC off whenever TABLE_SHADOW is on, so none can be pending.
        """
        for us in running:
            them = kv._rust.table_entry(us[0])
            if them == us:
                table_shadow_state.calls += 1
                continue
            rid = by_slot[us[0]].request_id
            if them is None:
                table_shadow_state.check("table_entry", "present", "missing", rid)
                continue
            for name, mine, theirs in zip(_REQ_FIELDS, us, them):
                table_shadow_state.check(f"table.{name}", mine, theirs, rid)

    def skip_note(self, why: str) -> None:
        """First-reason-only logging: names each new reason once, then stays quiet."""
        seen = self._vtl_burst_skips
        if why not in seen:
            seen.add(why)
            log.info("rust_sched: nstep skipped -- %s", why)

    def commit_burst(self, kv, so, by_slot, decisions) -> None:
        """Decide and commit this step's burst factor -- or, failing that, in-graph N=1
        sampling. Runs after ``_update_after_schedule``.

        TWO COMMITS, ONE GATE. The burst needs the align gate and the queue-empty guard;
        in-graph N=1 sampling needs neither (one token crosses no block boundary and delays
        no admission), so a step the burst refuses can still commit the cheaper rung. The
        shared predicates are ``burst_blocked_batch`` / ``burst_sampler_blocked``.

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
            slots = [slot for slot, _ in decisions["scheduled_running"]]
            # `- 1`: num_computed_tokens as it was BEFORE _update_after_schedule advanced it.
            if n >= 2:
                why = None
                if NSTEP_QUEUE_EMPTY_ONLY and self.waiting:
                    why = "waiting queue is not empty"
                else:
                    block_size = self.cache_config.block_size
                    for slot in slots:
                        request = by_slot[slot]
                        why = burst_request_blocked(
                            request, request.num_computed_tokens - 1, n,
                            block_size, self.max_model_len,
                        )
                        if why is not None:
                            break
                if why is None:
                    delta = n - 1
                    for slot in slots:
                        burst_commit(by_slot[slot], delta)
                    kv._rust.table_burst(slots, delta)
                    so.vtl_burst_n = n
                    return
                skip_note(self, why)
            if one:
                why = None
                for slot in slots:
                    request = by_slot[slot]
                    why = burst_sampler_blocked(
                        request, request.num_computed_tokens - 1, 1, self.max_model_len
                    )
                    if why is not None:
                        break
                if why is None:
                    # No bookkeeping at all: one token is exactly what the step already
                    # committed to, so only the SAMPLING path changes.
                    so.vtl_sample_in_graph = True
                    return
                skip_note(self, f"in-graph sampling: {why}")
        except BaseException as exc:
            reraise_fatal(exc)
            log.exception("rust_sched: nstep commit failed; bursts disabled for this boot")
            nstep_mod.BURST.disable("scheduler commit raised")

    def schedule(self, *args, **kwargs):
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
                }
            )
            log.info(
                "rust_sched: FULL schedule() loop active "
                "(sjf=%s, table=%s, table_shadow=%s, spec=%s, timing=%s)",
                sjf_enabled, TABLE, TABLE_SHADOW, SPEC, TIMING,
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

        decisions = None
        if resident:
            try:
                # The worker speculated with an EMPTY waiting slice (spec.rs), and
                # `take_speculative` checks the generation and slot order but NOT the
                # queue -- so refusing here is what makes an admission step safe.
                if SPEC and tbl.armed and not waiting:
                    decisions = core.take_speculative(kv._rust, tbl.gen, running_slots)
                tbl.armed = False
                if decisions is None:
                    decisions = core.schedule_resident(kv._rust, running_slots, waiting)
            except Exception as exc:
                # The documented failure is "slot N has no resident entry", i.e. the
                # resync signal; every other rejection wants the same answer.
                tbl.resync(f"resident schedule refused ({exc!r})")
                decisions = None
            except BaseException as exc:  # a crate panic
                tbl.fail("schedule_resident", exc)
                decisions = None
            if decisions is None:
                running = [pack_req(s, by_slot[s]) for s in running_slots]
        if decisions is None:
            if TABLE_SHADOW and tbl is not None and not tbl.dirty:
                try:
                    shadow_table(kv, running, by_slot)
                except BaseException as exc:
                    tbl.fail("table shadow", exc)
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
        flat = decisions["running_new_blocks"]
        lens = decisions["running_new_lens"]
        num_groups = kv._rust.num_groups
        off = 0
        for i, (slot, num_new) in enumerate(decisions["scheduled_running"]):
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
        remaining = [by_slot[s] for s in decisions["waiting_order"]]
        self.waiting.clear()
        for request in remaining:
            self.waiting.add_request(request)

        for slot in decisions["preempted"]:
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

        for slot, num_new, num_computed in decisions["scheduled_admitted"]:
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
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            {},
            req_to_new_blocks,
        )
        if not self.use_v2_model_runner:
            self.prev_step_scheduled_req_ids.clear()
            self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=list(decisions["num_common_prefix_blocks"]),
            preempted_req_ids=self.reset_preempted_req_ids,
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            new_block_ids_to_zero=(
                (kv.take_new_block_ids() or None) if self.needs_kv_cache_zeroing else None
            ),
            num_spec_tokens_to_schedule=self.num_spec_tokens,
        )
        if self.defer_block_free and total > 0:
            self.sched_step_seq += 1
        self._update_after_schedule(scheduler_output)
        if NSTEP:
            commit_burst(self, kv, scheduler_output, by_slot, decisions)
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
                return [
                    BlockHash(h) for h in block_hashes(none_hash, hash_block_size, tokens[:end])
                ]
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
    m = modes()
    if not (m["shadow"] or m["authority"] or m["hasher"]):
        log.info("rust_sched: no mode selected, nothing installed")
        return

    try:
        import vtl_sched  # noqa: F401
    except BaseException as exc:
        reraise_fatal(exc)
        log.warning("rust_sched: vtl_sched extension not importable (%r); staying on vLLM", exc)
        return

    if m["hasher"]:
        try:
            _install_rust_hasher()
        except BaseException as exc:
            reraise_fatal(exc)
            log.exception("rust_sched: rust block hasher not installed; keeping stock")

    if not (m["shadow"] or m["authority"]):
        return

    import vllm.v1.core.sched.scheduler as sched_mod

    base = sched_mod.KVCacheManager
    if getattr(base, "__vtl_rust_shadow__", False) or getattr(
        base, "__vtl_rust_authority__", False
    ):
        return  # idempotent

    if getattr(base, "__vtl_subclass__", False):
        log.info("rust_sched: composing on top of vtl kv_cache_manager (signals preserved)")
    else:
        log.info("rust_sched: vtl kv_cache_manager patch is not installed; subclassing stock")

    if m["authority"]:
        sched_mod.KVCacheManager = _install_authority(base, m)
        log.info("rust_sched: installed AUTHORITY manager (VTL_RUST_SCHED=1)")
    else:
        sched_mod.KVCacheManager = _install_shadow(base, m)
        log.info("rust_sched: installed SHADOW manager (VTL_RUST_SCHED_SHADOW=1)")

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
            "rust_sched: resolved R6b/R6c mode -- table=%s table_shadow=%s spec=%s timing=%s",
            m["table"], m["table_shadow"], m["spec"], m["timing"],
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
        "VTL_RUST_SCHED", "VTL_RUST_SCHED_SHADOW", "VTL_RUST_SCHED_FULL",
        "VTL_RUST_SCHED_SHADOW_STRICT", "VTL_RUST_SCHED_RADIX",
        "VTL_RUST_SCHED_UFO", "VTL_RUST_SCHED_UFO_SHADOW",
        "VTL_RUST_SCHED_TABLE", "VTL_RUST_SCHED_TABLE_SHADOW",
        "VTL_RUST_SCHED_SPEC", "VTL_SCHED_TIMING",
        "VTL_RUST_SCHED_R8", "VTL_RUST_SCHED_R8_SHADOW", "VTL_RUST_HASHER",
        "VTL_SHM_IPC", "VTL_SHM_IPC_RAW", "VTL_NSTEP",
        "VTL_RUST_SCHED_TOKSTORE", "VTL_RUST_SCHED_TOKSTORE_SHADOW",
    )}
    try:
        for k in saved:
            os.environ.pop(k, None)
        assert modes() == {
            "shadow": False, "strict": False, "authority": False,
            "full": False, "radix": False, "ufo": False, "ufo_shadow": False,
            "table": False, "table_shadow": False, "spec": False, "timing": False,
            "r8": False, "r8_shadow": False, "hasher": False, "nstep": False,
            "tokstore": False, "tokstore_shadow": False,
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
        # The R6b/R6c ladder: TABLE needs UFO, SPEC needs TABLE, and TABLE_SHADOW keeps
        # the marshalled path authoritative so it must switch SPEC off.
        os.environ["VTL_RUST_SCHED_TABLE"] = "1"
        os.environ["VTL_RUST_SCHED_SPEC"] = "1"
        assert modes()["table"] is True and modes()["spec"] is True
        os.environ["VTL_RUST_SCHED_TABLE_SHADOW"] = "1"
        assert modes()["table_shadow"] is True
        assert modes()["spec"] is False, "shadow keeps the marshalled path authoritative"
        os.environ.pop("VTL_RUST_SCHED_TABLE_SHADOW")
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
        assert modes()["r8_shadow"] is False
        os.environ["VTL_RUST_SCHED_R8_SHADOW"] = "1"
        assert modes()["r8_shadow"] is True
        os.environ["VTL_RUST_SCHED_UFO"] = "0"
        assert modes()["r8"] is False and modes()["r8_shadow"] is False, (
            "R8 IS update_step plus a pack; it cannot arm without UFO"
        )
        os.environ["VTL_RUST_SCHED_UFO"] = "1"

        # The N-step commit lives inside the Rust schedule loop, so it needs FULL.
        os.environ["VTL_NSTEP"] = "1"
        assert modes()["nstep"] is True
        os.environ["VTL_RUST_SCHED_FULL"] = "0"
        assert modes()["nstep"] is False
        os.environ["VTL_RUST_SCHED_FULL"] = "1"

        # Port-2 rides on R8 *and* the Rust hasher: two live hash implementations would
        # each own half the chain, and a divergence between them is silent.
        os.environ["VTL_RUST_SCHED_TOKSTORE"] = "1"
        assert modes()["r8_shadow"] is True
        assert modes()["tokstore"] is False, "the R8 shadow arm would double-count tokens"
        os.environ["VTL_RUST_SCHED_R8_SHADOW"] = "0"
        assert modes()["r8"] is True
        assert modes()["tokstore"] is False, "no token store without the Rust hasher"
        os.environ["VTL_RUST_HASHER"] = "1"
        assert modes()["tokstore"] is True
        assert modes()["tokstore_shadow"] is False
        os.environ["VTL_RUST_SCHED_TOKSTORE_SHADOW"] = "1"
        assert modes()["tokstore_shadow"] is True
        os.environ["VTL_RUST_SCHED_UFO_SHADOW"] = "1"
        assert modes()["tokstore"] is False and modes()["tokstore_shadow"] is False, (
            "the UFO shadow arm keeps python's append authoritative -> double count"
        )
        os.environ["VTL_RUST_SCHED_UFO_SHADOW"] = "0"
        os.environ["VTL_RUST_SCHED_R8"] = "0"
        assert modes()["tokstore"] is False and modes()["tokstore_shadow"] is False, (
            "the store IS update_step_pack; it cannot arm without R8"
        )
        os.environ["VTL_RUST_SCHED_R8"] = "1"
        os.environ.pop("VTL_RUST_SCHED_TOKSTORE")
        os.environ.pop("VTL_RUST_SCHED_TOKSTORE_SHADOW")
        assert modes()["tokstore"] is False

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

    # ShadowState: counts, never raises unless strict. (The mismatch path logs at ERROR
    # by design; mute it here so `make check` output stays clean.)
    log.setLevel(logging.CRITICAL)
    st = ShadowState(strict=False)
    st.check("x", 1, 1)
    assert st.mismatches == 0
    st.check("x", 1, 2, "ctx")
    assert st.mismatches == 1
    strict = ShadowState(strict=True)
    try:
        strict.check("y", [1], [2])
    except AssertionError as exc:
        assert "rust_sched shadow mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("strict shadow mode must raise")

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
        r.num_tokens_with_spec += keep
        r.num_output_tokens += keep
        assert r.num_tokens == before + keep
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

    # TOK.disable is permanent for the boot and closes both arms.
    saved_tok = (TOK.live, TOK.shadow)
    try:
        TOK.live = TOK.shadow = True
        log.setLevel(logging.CRITICAL)
        TOK.disable("test")
        log.setLevel(logging.NOTSET)
        assert not TOK.live and not TOK.shadow
    finally:
        TOK.live, TOK.shadow = saved_tok

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

    try:
        import vtl_sched  # noqa: F401

        print("rust_sched self-check ok (vtl_sched extension present)")
    except Exception:
        print("rust_sched self-check ok (vtl_sched extension absent -- pure-python parts only)")


if __name__ == "__main__":
    _self_check()
