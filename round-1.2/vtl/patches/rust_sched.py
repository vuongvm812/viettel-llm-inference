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

from vtl.registry import already_patched, mark_patched, register_patch

# Must be a child of "vllm.vtl": a bare "vtl" logger's INFO records are dropped.
log = logging.getLogger("vllm.vtl.rust_sched")

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# RequestStatus values the Rust core needs (vllm/v1/request.py).
_ST_WAITING, _ST_RUNNING, _ST_PREEMPTED = 0, 1, 2


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
    return {
        "shadow": env_on("VTL_RUST_SCHED_SHADOW"),
        "strict": env_on("VTL_RUST_SCHED_SHADOW_STRICT"),
        "authority": env_on("VTL_RUST_SCHED") or full,
        "full": full,
        "radix": env_on("VTL_RUST_SCHED_RADIX"),
        # R6a rides on the full loop: it needs the Rust manager's slot interning, and
        # nothing else in the engine would be Rust-backed without it.
        "ufo": full and env_on("VTL_RUST_SCHED_UFO"),
        "ufo_shadow": env_on("VTL_RUST_SCHED_UFO_SHADOW"),
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
            self._rust.evict_blocks([int(b) for b in block_ids])

        def reset_prefix_cache(self):
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


def _install_update_from_output(scheduler_cls, shadow: bool):
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
    from vllm.v1.request import RequestStatus

    wrapped_ufo = scheduler_cls.update_from_output
    wrapped_urwo = scheduler_cls._update_request_with_output

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

    def decide(self, kv, scheduler_output, model_runner_output):
        """Build the flat batch and take the single crossing. None = nothing portable."""
        rust, mirror = kv._rust, kv._mirror
        sampled = model_runner_output.sampled_token_ids
        slots: list[int] = []
        cu: list[int] = [0]
        toks: list[int] = []
        n_out: list[int] = []
        n_tok: list[int] = []
        expect: list[tuple[str, int]] = []
        for req_id, index in model_runner_output.req_id_to_index.items():
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue
            gen = sampled[index]
            if not gen or getattr(request, "async_tokens_to_discard", 0):
                continue
            slot = mirror.slot(request)
            # Python-side registry, same rationale as RustMirror._pushed: asking Rust
            # `has_stop_params` here was one FFI crossing per request per step to re-learn
            # a fact only this module changes. register() overwrites, so a lost entry
            # (slot recycled -> discarded in mirror.drop) just re-registers.
            if slot not in mirror._stops:
                if not register(rust, request, slot):
                    continue
                mirror._stops.add(slot)
            slots.append(slot)
            expect.append((req_id, len(gen)))
            toks.extend(gen)
            cu.append(len(toks))
            # Read BEFORE the loop appends anything -- exactly the values check_stop would
            # see on its first iteration.
            n_out.append(request.num_output_tokens)
            n_tok.append(request.num_tokens)
        if not slots:
            return None
        out = rust.update_step(slots, cu, toks, n_out, n_tok, int(self.max_model_len))
        return {rid: (n, *v) for (rid, n), v in zip(expect, out)}

    def update_from_output(self, scheduler_output, model_runner_output):
        self._vtl_ufo = None
        kv = self.kv_cache_manager
        if (
            hasattr(kv, "_rust")
            and model_runner_output.sampled_token_ids
            and not scheduler_output.scheduled_spec_decode_tokens
            and not model_runner_output.pooler_output
        ):
            try:
                self._vtl_ufo = decide(self, kv, scheduler_output, model_runner_output)
            except BaseException as exc:
                reraise_fatal(exc)
                log.exception("rust_sched: UFO batch failed; this step uses check_stop")
                self._vtl_ufo = None
        try:
            return wrapped_ufo(self, scheduler_output, model_runner_output)
        finally:
            self._vtl_ufo = None

    def _update_request_with_output(self, request, new_token_ids):
        decided = self._vtl_ufo
        answer = decided.pop(request.request_id, None) if decided else None
        # 255 = unregistered slot; a length mismatch means something mutated the request
        # between the batch and here. Either way, stock decides.
        if answer is None or answer[2] == 255 or answer[0] != len(new_token_ids):
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

        for token_id in new_token_ids[:num_keep]:
            request.append_output_token_ids(token_id)
        if status == 0:
            return new_token_ids, False
        del new_token_ids[num_keep:]
        request.status = _STATUS_FROM_CODE[status]
        if stop_reason >= 0:
            request.stop_reason = stop_reason
        return new_token_ids, True

    mark_patched(update_from_output, wrapped_ufo, patch="rust_sched_ufo")
    mark_patched(_update_request_with_output, wrapped_urwo, patch="rust_sched_ufo")
    scheduler_cls.update_from_output = update_from_output
    scheduler_cls._update_request_with_output = _update_request_with_output
    log.info("rust_sched: UFO batched stop decision active (shadow=%s)", shadow)


def _install_full_schedule(scheduler_cls, sjf_enabled: bool):
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

    def bail_reason(self):
        """Per-step conditions the Rust loop does not model (scheduler.py:637-664)."""
        if getattr(self, "skipped_waiting", None):
            return "blocked requests are parked in skipped_waiting"
        if getattr(self, "num_waiting_for_streaming_input", 0):
            return "paused streaming sessions hold model-runner slots"
        pause = getattr(self, "_pause_state", None)
        if pause is not None and getattr(pause, "name", str(pause)) != "UNPAUSED":
            return f"scheduler is paused ({pause})"
        return None

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
            log.info("rust_sched: FULL schedule() loop active (sjf=%s)", sjf_enabled)

        # Bail BEFORE `current_step += 1`: the fallback does its own increment.
        bail = bail_reason(self)
        mirror = kv._mirror
        by_slot = {}
        running = []
        waiting = []
        if bail is None:
            try:
                for request in self.running:
                    slot = mirror.slot(request)
                    by_slot[slot] = request
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
            seen = getattr(self, "_vtl_rust_bails", None)
            if seen is None:
                seen = self._vtl_rust_bails = set()
            if bail not in seen:
                seen.add(bail)
                log.warning("rust_sched: this step falls back to vLLM -- %s", bail)
            return wrapped(self, *args, **kwargs)

        self.current_step += 1
        decisions = core.schedule(kv._rust, running, waiting)

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
        return scheduler_output

    scheduler_cls.schedule = mark_patched(schedule, wrapped)


# --------------------------------------------------------------------------
# install
# --------------------------------------------------------------------------


@register_patch("rust_sched", default=False)
def apply() -> None:
    m = modes()
    if not (m["shadow"] or m["authority"]):
        log.info("rust_sched: no mode selected, nothing installed")
        return

    try:
        import vtl_sched  # noqa: F401
    except BaseException as exc:
        reraise_fatal(exc)
        log.warning("rust_sched: vtl_sched extension not importable (%r); staying on vLLM", exc)
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
        _install_full_schedule(Scheduler, sjf)
        Scheduler.schedule.__vtl_rust_full__ = True

        if m["ufo"] and not already_patched(
            Scheduler, "update_from_output", patch="rust_sched_ufo"
        ):
            _install_update_from_output(Scheduler, m["ufo_shadow"])


# --------------------------------------------------------------------------
# self-check -- runs with neither vLLM nor the compiled crate
# --------------------------------------------------------------------------


def _self_check() -> None:
    saved = {k: os.environ.get(k) for k in (
        "VTL_RUST_SCHED", "VTL_RUST_SCHED_SHADOW", "VTL_RUST_SCHED_FULL",
        "VTL_RUST_SCHED_SHADOW_STRICT", "VTL_RUST_SCHED_RADIX",
        "VTL_RUST_SCHED_UFO", "VTL_RUST_SCHED_UFO_SHADOW",
    )}
    try:
        for k in saved:
            os.environ.pop(k, None)
        assert modes() == {
            "shadow": False, "strict": False, "authority": False,
            "full": False, "radix": False, "ufo": False, "ufo_shadow": False,
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
        os.environ["VTL_RUST_SCHED_FULL"] = "0"
        assert modes()["ufo"] is False, "UFO must not arm without the full Rust loop"
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
