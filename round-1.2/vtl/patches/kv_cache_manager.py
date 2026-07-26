"""Custom ``KVCacheManager`` that overrides vLLM's and feeds the scheduler.

This is the exact-tree answer to "put a radix tree on the router." vLLM V1's prefix
cache already IS a radix tree (block hashes are prefix-chained and
``find_longest_cache_hit`` walks the chain, stopping at the first miss), and our
scheduler runs inside the EngineCore process, so it reads that real tree directly. There
is nothing to approximate and no worker to avoid talking to -- an approximate shadow tree
would only add drift, memory, and a background GC thread for a structure we already have
exactly. So instead we SUBCLASS the real manager and add read-only signals the scheduler
consumes for cache-aware + memory-aware ordering. Allocation/caching is 100% inherited;
output is unchanged.

Hybrid-KV note (LFM2): the served model has multiple KV cache groups (full-attention +
short-conv/mamba state), so ``coordinator.find_longest_cache_hit`` / ``block_pool.get_num_free_blocks``
may behave differently than the single-group case these signals assume. That is a PERF concern,
not a correctness one: both are read-only and only feed scheduling ORDER, and both degrade on
any API mismatch (hit-walk failure -> prompt-length; free_blocks failure -> unlimited). Worst
case is a suboptimal admission order, never a wrong result. Re-validate the benefit on the box.

The scheduler constructs ``KVCacheManager`` at exactly one call site
(``Scheduler.__init__``), by the name imported into ``vllm.v1.core.sched.scheduler``.
Rebinding that name to our subclass makes every scheduler (sync and async) build ours,
with zero constructor changes and no worker-side coupling (workers never hold it).

The same subclass is also where we ELIDE two stock manager methods whose results this
configuration computes and nobody reads. Both are pure dead work, both fail closed:

* ``get_num_common_prefix_blocks`` -- called every ``schedule()`` step
  (``scheduler.py:1078-1084``) to fill ``SchedulerOutput.num_common_prefix_blocks``, whose
  ONLY consumer is ``gpu_model_runner.py:4207``, i.e. **V1** cascade attention. The V2 model
  runner never reads the field. It is not already free either: prefix caching + 2 KV groups
  selects ``HybridKVCacheCoordinator``, which does not override the method, so it falls
  through to the base per-group loop (``kv_cache_coordinator.py:319-334``); ``MambaManager``
  returns 0 for free but ``FullAttentionManager`` walks the request's blocks counting
  ``ref_cnt`` (``single_type_kv_cache_manager.py:795-803``). We skip it ONLY when the V2
  runner is provably live -- see ``_cascade_dead``.

* ``estimate_cached_tokens`` -- called once per request on the step that emits its first
  token, i.e. inside TTFT (``scheduler.py:1798-1806``). ``PrefillStats()`` is built
  unconditionally (``request.py:197``), so ``--disable-log-stats`` does NOT suppress it. It
  calls ``get_blocks()`` (allocating a ``KVCacheBlocks``) then loops every allocated block in
  every non-cross-attn group with no early break (``kv_cache_manager.py:659-687``) -- ~270
  blocks for the full-attention group on a 4.3k prompt. We return 0 when stats are off.

Expect microseconds, not milliseconds. This is dead-work removal, not a win: the ref-cnt walk
breaks at the end of the shared prefix (mean concurrency here is 2), and the TTFT term is worth
~1/28th of TPOT per millisecond in the score. It is worth doing because the diff is tiny and
the work is provably unread, not because it will move the number.

!! ``VTL_CHEAP_PREFILL_STATS=0`` is load-bearing if you ever add
``--enable-prompt-tokens-details``. The ``estimate_cached_tokens`` elision self-gates on
``self.log_stats``, which covers the metrics consumer (``metrics/stats.py:390``), but the
SECOND consumer -- ``output_processor.py:640`` -> ``num_cached_tokens`` -> the response's
``prompt_tokens_details.cached_tokens`` -- is NOT gated by ``log_stats``. It is invisible today
only because that serve flag is unset. Add the flag and you must set this knob to 0, or the
field reports 0.

Set ``VTL_ENABLE_KV_CACHE_MANAGER=0`` to serve vLLM's stock manager.
"""

from __future__ import annotations

import logging
import os

from vtl.registry import register_patch

log = logging.getLogger("vllm.vtl")

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_HUGE = 1 << 60  # un-plannable request: pushed to the back / treated as unlimited free
_DEFAULT_BLOCK_SIZE = 16  # only a fallback if the manager exposes no block_size

# Module-level cache for the env half of the V2 probe. The env cannot change mid-process, and
# `vllm.envs` attribute access runs a lambda, so resolve it once. `"unset"` is the
# not-yet-probed sentinel; the resolved value is True / False / None (None = "vLLM derives it").
_ENV_V2: object = "unset"


def _cheap_prefill_stats() -> bool:
    return os.environ.get("VTL_CHEAP_PREFILL_STATS", "1").strip().lower() in _TRUTHY


def _v2_from_env():
    """``VLLM_USE_V2_MODEL_RUNNER`` as a tri-state, or None if it does not decide.

    The env var is ``bool | None`` and None means "derive it from the config"
    (``config/vllm.py:549-568`` also forces V2 for dspark / dflash / diffusion). Never infer
    from None -- an explicit 0 or 1 is the only thing we trust here.
    """
    global _ENV_V2
    if _ENV_V2 == "unset":
        try:
            import vllm.envs as envs

            raw = envs.VLLM_USE_V2_MODEL_RUNNER
        except Exception:
            raw = None
        _ENV_V2 = raw if isinstance(raw, bool) else None
    return _ENV_V2


def _cascade_dead(mgr) -> bool:
    """True when nothing reads ``SchedulerOutput.num_common_prefix_blocks``.

    Resolved once, then cached on the manager instance. Two sources, in order:

    1. An explicit ``_vtl_v2`` set on the manager by something that knows better. Nothing sets
       it today -- ``sched_policy`` used to, from the Scheduler's authoritative
       ``use_v2_model_runner`` (``scheduler.py:290``), but that handshake was reverted along
       with the rest of that commit's per-step scheduler work. The hook stays because it is
       one ``getattr`` and it is how a future caller overrides the env.
    2. The env, but only when explicitly set (see ``_v2_from_env``). This is the live source:
       ``docker-compose.yaml`` always sets ``VLLM_USE_V2_MODEL_RUNNER`` to a literal, so the
       elision resolves on the first call.

    FAILS CLOSED: unresolved -> False -> do the stock walk. A wrong True would silently break
    V1 cascade attention; a wrong False only costs the walk we wanted to skip. The unresolved
    case is deliberately NOT cached, so a later override still lands.
    """
    cached = getattr(mgr, "_vtl_cascade_dead", None)
    if cached is not None:
        return cached
    dead = getattr(mgr, "_vtl_v2", None)
    if dead is None:
        dead = _v2_from_env()
    if dead is None:
        return False
    dead = bool(dead)
    mgr._vtl_cascade_dead = dead
    if dead:
        log.info(
            "vtl: V2 runner confirmed -- eliding the per-step common-prefix block walk "
            "(cascade attention is V1-only, so nothing reads it)"
        )
    return dead


def _num_common_prefix_blocks(mgr, running_request_id, stock):
    """Zeros when cascade metadata is dead, else the stock per-group walk.

    Returns a FRESH list: the caller stores it straight into ``SchedulerOutput``, and stock
    builds a fresh ``[0] * n`` at ``scheduler.py:1078`` too. ~100ns to avoid handing every
    step's output the same aliased list.
    """
    if _cascade_dead(mgr):
        num_groups = getattr(mgr, "num_kv_cache_groups", None)  # cached at :161 upstream
        if num_groups is not None:
            return [0] * num_groups
    return stock(running_request_id)


def _estimate_cached_tokens(mgr, request, stock):
    """0 when the number is unread, else the stock block scan.

    Self-gates on ``mgr.log_stats`` (set from ``scheduler.py:275``) so re-enabling stats
    restores the real computation with no knob to remember. Missing attribute -> assume stats
    are on and do the real work.
    """
    if _cheap_prefill_stats() and not getattr(mgr, "log_stats", True):
        return 0
    return stock(request)


def _plan_request(request, coord_holder, block_size: int):
    """One cache-hit walk -> ``(remaining_prefill, blocks_needed)``.

    ``remaining_prefill`` = uncached prompt tokens (the cache-aware SJF key).
    ``blocks_needed`` = KV blocks this request will still allocate (the memory-aware key).
    Both come from a single ``find_longest_cache_hit`` so the scheduler pays one walk, not
    two. Uses the coordinator directly (NOT ``get_computed_blocks``) to avoid polluting
    ``prefix_cache_stats`` -- the real schedule() loop records those once, and
    double-counting corrupts the hit-rate metric. Degrades to prompt length on any
    failure; never raises.
    """
    try:
        num_prompt = request.num_prompt_tokens
    except Exception:
        return _HUGE, _HUGE
    try:
        # Resumed/preempted reqs already have progress; mirror the loop, don't re-look-up.
        if getattr(request, "num_computed_tokens", 0):
            remaining = max(num_prompt - request.num_computed_tokens, 0)
        else:
            # INDEX, do not unpack: this return grew from 2-tuple (v0.25.0) to 3-tuple in
            # v0.26.0 (+num_uncached_common_prefix_tokens). Unpacking raised ValueError into
            # the `except` below, silently degrading every request to prompt-length ordering
            # with no log line. Indexing survives the next arity change too.
            hit = coord_holder.coordinator.find_longest_cache_hit(
                request.block_hashes, request.num_tokens - 1
            )
            remaining = max(num_prompt - hit[1], 0)
    except Exception:
        remaining = num_prompt  # degrade to shortest-prompt-first; never raise
    bs = block_size if block_size > 0 else _DEFAULT_BLOCK_SIZE
    blocks_needed = (remaining + bs - 1) // bs
    return remaining, blocks_needed


@register_patch("kv_cache_manager", default=True)
def apply() -> None:
    import vllm.v1.core.sched.scheduler as sched_mod

    base = sched_mod.KVCacheManager
    if getattr(base, "__vtl_subclass__", False):
        return  # idempotent: already our subclass

    class VtlKVCacheManager(base):
        """Stock manager + read-only signals for the vtl scheduler. Nothing else changes."""

        __vtl_subclass__ = True

        def plan_request(self, request):
            return _plan_request(
                request, self, getattr(self, "block_size", 0) or _DEFAULT_BLOCK_SIZE
            )

        @property
        def free_blocks(self) -> int:
            try:
                return self.block_pool.get_num_free_blocks()
            except Exception:
                return _HUGE  # unknown -> treat as unlimited, never demote on error

        # The two dead-work elisions. Bodies live at module level so `make check` can drive
        # them with fakes -- this class only exists inside apply(), where it needs vLLM.
        def get_num_common_prefix_blocks(self, running_request_id):
            return _num_common_prefix_blocks(
                self, running_request_id, super().get_num_common_prefix_blocks
            )

        def estimate_cached_tokens(self, request):
            return _estimate_cached_tokens(
                self, request, super().estimate_cached_tokens
            )

    sched_mod.KVCacheManager = VtlKVCacheManager
    log.info(
        "vtl: kv_cache_manager installed (exact-tree, memory-aware signals, "
        "dead common-prefix/prefill-stats scans elided)"
    )


# ponytail: `estimate_cached_tokens -> 0` leaves the rest of the PrefillStats machinery
# running -- the per-request `.set()` at scheduler.py:806-815 and the `prefill_stats` field on
# every first-token EngineCoreOutput. Nulling `request.prefill_stats` at construction (the
# single site is core.py:968 `Request.from_engine_core_request`) would kill both, since every
# reader is `is not None`-guarded. Not done: that is a few attribute writes plus one msgpack
# field, i.e. noise, in exchange for a second patch surface on a class we otherwise never
# touch. Revisit only if a profile puts those writes on the board.


def _self_check() -> None:
    """Runs with no vLLM: pure-python fakes exercise the walk + block math + fallbacks."""

    class FakeCoord:
        def __init__(self, cached):  # cached: {block_hashes_key: num_cached_tokens}
            self.cached = cached

        def find_longest_cache_hit(self, block_hashes, max_len):
            # 3-tuple, matching v0.26.0 (blocks, hit_length, uncached_common_prefix).
            # The fake previously returned the v0.25.0 2-tuple, so `make check` kept
            # passing while the real call site was raising ValueError in production.
            return ([], self.cached.get(block_hashes, 0), 0)

    class FakeKVM:
        def __init__(self, cached):
            self.coordinator = FakeCoord(cached)

    class FakeReq:
        def __init__(self, rid, prompt, computed=0):
            self.request_id = rid
            self.num_prompt_tokens = prompt
            self.num_tokens = prompt
            self.num_computed_tokens = computed
            self.block_hashes = rid

    kvm = FakeKVM({"a": 190, "b": 0})
    # a: 200 prompt, 190 cached -> 10 remaining; blocks = ceil(10/16) = 1
    assert _plan_request(FakeReq("a", 200), kvm, 16) == (10, 1)
    # b: 50 prompt, 0 cached -> 50 remaining; blocks = ceil(50/16) = 4
    assert _plan_request(FakeReq("b", 50), kvm, 16) == (50, 4)
    # exact multiple: 32 remaining / 16 -> 2 blocks, no off-by-one
    assert _plan_request(FakeReq("c", 32), FakeKVM({"c": 0}), 16) == (32, 2)

    # Resumed request keys off its own progress, no lookup.
    assert _plan_request(FakeReq("w", 100, computed=70), None, 16) == (30, 2)

    # Lookup failure degrades to prompt length, never raises.
    class BoomKVM:
        class coordinator:
            @staticmethod
            def find_longest_cache_hit(*_):
                raise RuntimeError("boom")

    assert _plan_request(FakeReq("z", 42), BoomKVM, 16) == (42, 3)

    # Un-plannable request (no num_prompt_tokens) sinks to the back.
    class Bad:
        pass

    assert _plan_request(Bad(), kvm, 16) == (_HUGE, _HUGE)

    # block_size <= 0 falls back to the default, still divides cleanly.
    assert _plan_request(FakeReq("d", 32), FakeKVM({"d": 0}), 0) == (
        32,
        (32 + _DEFAULT_BLOCK_SIZE - 1) // _DEFAULT_BLOCK_SIZE,
    )

    # ---- A: the common-prefix elision ------------------------------------------------
    global _ENV_V2
    saved_env_v2 = _ENV_V2
    _ENV_V2 = None  # env does not decide, so only the handshake can

    class FakeMgr:
        """Just the attributes the two elisions read."""

        def __init__(self, **kw):
            self.num_kv_cache_groups = 2
            self.log_stats = False
            self.__dict__.update(kw)

    def stock_walk(_rid):
        calls.append("walk")
        return [7, 0]

    # Unresolved (no handshake, env says nothing) MUST fall through to the stock walk.
    calls: list[str] = []
    assert _num_common_prefix_blocks(FakeMgr(), "r", stock_walk) == [7, 0]
    assert calls == ["walk"], calls
    # ...and must NOT cache, so a later handshake still lands.
    mgr = FakeMgr()
    assert _num_common_prefix_blocks(mgr, "r", stock_walk) == [7, 0]
    mgr._vtl_v2 = True
    assert _num_common_prefix_blocks(mgr, "r", stock_walk) == [0, 0]

    # Handshake says V1 -> stock walk, and that answer IS cached.
    calls = []
    v1 = FakeMgr(_vtl_v2=False)
    assert _num_common_prefix_blocks(v1, "r", stock_walk) == [7, 0]
    assert _num_common_prefix_blocks(v1, "r", stock_walk) == [7, 0]
    assert calls == ["walk", "walk"], calls
    assert v1._vtl_cascade_dead is False

    # Handshake says V2 -> zeros, one per group, and never the stock walk.
    calls = []
    v2 = FakeMgr(_vtl_v2=True)
    assert _num_common_prefix_blocks(v2, "r", stock_walk) == [0, 0]
    assert calls == [], calls
    # Fresh list every call: the caller stores it in SchedulerOutput.
    assert _num_common_prefix_blocks(v2, "r", stock_walk) is not _num_common_prefix_blocks(
        v2, "r", stock_walk
    )
    # Unknown group count -> cannot build the right shape -> stay stock.
    blind = FakeMgr(_vtl_v2=True)
    del blind.num_kv_cache_groups
    assert _num_common_prefix_blocks(blind, "r", stock_walk) == [7, 0]

    # The env alone resolves it when there is no handshake.
    _ENV_V2 = True
    assert _num_common_prefix_blocks(FakeMgr(), "r", stock_walk) == [0, 0]
    _ENV_V2 = False
    assert _num_common_prefix_blocks(FakeMgr(), "r", stock_walk) == [7, 0]
    _ENV_V2 = saved_env_v2

    # ---- B: the prefill-stats elision ------------------------------------------------
    saved_cheap = os.environ.get("VTL_CHEAP_PREFILL_STATS")

    def stock_estimate(_req):
        calls.append("estimate")
        return 1234

    os.environ.pop("VTL_CHEAP_PREFILL_STATS", None)  # default is ON
    calls = []
    assert _estimate_cached_tokens(FakeMgr(log_stats=False), object(), stock_estimate) == 0
    assert calls == [], calls
    # Stats on -> the number is read, so compute it for real.
    assert (
        _estimate_cached_tokens(FakeMgr(log_stats=True), object(), stock_estimate) == 1234
    )
    # Missing attribute -> assume stats are on, never silently zero a live metric.
    blind = FakeMgr()
    del blind.log_stats
    assert _estimate_cached_tokens(blind, object(), stock_estimate) == 1234
    # Kill switch restores stock even with stats off (the --enable-prompt-tokens-details case).
    os.environ["VTL_CHEAP_PREFILL_STATS"] = "0"
    assert (
        _estimate_cached_tokens(FakeMgr(log_stats=False), object(), stock_estimate) == 1234
    )
    assert calls == ["estimate", "estimate", "estimate"], calls

    if saved_cheap is None:
        os.environ.pop("VTL_CHEAP_PREFILL_STATS", None)
    else:
        os.environ["VTL_CHEAP_PREFILL_STATS"] = saved_cheap

    print("kv_cache_manager self-check ok")


if __name__ == "__main__":
    _self_check()
