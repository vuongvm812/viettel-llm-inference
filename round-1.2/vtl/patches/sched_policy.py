"""Cache-aware shortest-remaining-prefill-first reorder of vLLM's V1 waiting queue.

Workload is prefill-bound (101:1), multi-turn, ~82% prefix-cache hit, chunked
prefill with an 8192-token/step budget. FCFS admits by arrival; a long cold prompt
at the head blocks many short cache-hitting prompts behind it, inflating mean TTFT
and wasting the per-step prefill budget. We reorder ``self.waiting`` each step so the
request with the fewest UNCACHED prompt tokens goes first (SJF on remaining prefill).

Inspired by SGLang's cache-aware scheduler -- NOT its overlap scheduler; vLLM's
native async scheduling already covers overlap (``AsyncScheduler``, on by default).

Output-preserving: same requests, same per-request sampling; only admission order /
TTFT changes. Base ``Scheduler.schedule()`` is inherited unchanged by ``AsyncScheduler``,
so one patch covers both sync and async scheduling.

Hybrid-KV note (LFM2): with multiple KV cache groups (full-attention + short-conv state), the
``find_longest_cache_hit`` / ``usage`` / ``free_blocks`` reads may mis-estimate. Because this
patch ONLY reorders the waiting queue (output is identical), a mis-estimate can only produce a
suboptimal order, never a wrong result -- and every read degrades on failure (to prompt-length /
slack). Re-validate the ordering win on the box; the env flag reverts to stock FCFS.

The reorder runs on EVERY ``schedule()`` call, so it carries three cheap guards to stop it
paying for itself when there is nothing to gain. Each one only ever skips a re-ORDER, so the
worst case of a wrong guard is a stale admission order, never a wrong result:

* the queue is shorter than 2 (the common case here -- mean concurrency is 2, so ``waiting``
  is usually empty);
* the scheduler cannot admit anyone this step anyway because ``running`` is at the cap, which
  is exactly the burst case where the queue is long and sorting it every step is pure waste;
* the queue is the same one we already left in our own order, and the keys are not yet stale
  (see ``_RESORT_INTERVAL``). Without this, a backed-up queue pays n ``find_longest_cache_hit``
  block-hash walks per step to re-derive the order it already has.

Set ``VTL_ENABLE_SCHED_POLICY=0`` to serve stock FCFS.
"""

from __future__ import annotations

import logging
from collections import deque

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vtl")

_HUGE = 1 << 60  # push un-keyable requests to the back, never crash the sort
_USAGE_TIGHT = 0.90  # KV usage above which we prefer requests that fit over pure SJF
_mem_aware_engaged = 0  # how many schedule() calls hit the memory-aware branch

# Bound on how stale a cached SJF key may get. A waiting request's uncached-prefill count
# shrinks as OTHER requests cache blocks, so an unchanged queue can still deserve a new order.
# Re-sorting every N steps costs one sort per N instead of per step and caps the drift at N
# steps -- which, at a few ms per step, is well inside the time it takes a queue to matter.
#
# Module-level, i.e. one set of state per PROCESS, not per Scheduler -- same assumption
# `_mem_aware_engaged` above already makes. EngineCore builds exactly one Scheduler, so this
# holds; a second one in the same process would share the fingerprint (stale order, never a
# wrong result) and would not get the handshake at all (stock walk, also safe).
_RESORT_INTERVAL = 8
_last_sorted: tuple = ()  # request_ids in the order we last left the queue
_last_sort_step = -_RESORT_INTERVAL
_handshook = False  # the one-time use_v2_model_runner handoff to the KV manager


def _v2_handshake(scheduler) -> None:
    """Hand the KV cache manager the Scheduler's authoritative ``use_v2_model_runner``, once.

    ``vtl/patches/kv_cache_manager.py`` uses it to elide the per-step common-prefix block walk
    whose only consumer is V1 cascade attention. The Scheduler is the authoritative source
    (``scheduler.py:290``); the manager cannot see the config itself. Runs before the stock
    ``schedule()``, so the flag is in place before the first walk. Silent on failure -- the
    manager falls back to probing the env, and failing that just does the stock walk.
    """
    global _handshook
    if _handshook:
        return
    _handshook = True
    try:
        scheduler.kv_cache_manager._vtl_v2 = scheduler.use_v2_model_runner
    except Exception:
        pass


def _admission_blocked(scheduler) -> bool:
    """True when the waiting loop cannot admit anyone this step, so ordering is moot.

    Mirrors the base loop's own cap check (``scheduler.py:668-670``). ``getattr`` defaults so a
    renamed attribute degrades to "do the sort" rather than raising.
    """
    cap = getattr(scheduler, "max_num_running_reqs", None)
    if cap is None:
        return False
    num_running = len(getattr(scheduler, "running", ()))
    num_running += getattr(scheduler, "num_waiting_for_streaming_input", 0)
    return num_running >= cap


def _safe_usage(kv_cache_manager) -> float:
    try:
        return float(kv_cache_manager.usage)
    except Exception:
        return 0.0  # unknown pressure -> treat as slack, stay pure cache-aware


def _remaining_prefill(request, kv_cache_manager) -> int:
    """Uncached prompt tokens for ``request``.

    Falls back to prompt length on any failure. Uses the coordinator directly (NOT
    ``get_computed_blocks``) so we do not pollute ``prefix_cache_stats`` -- the real
    ``schedule()`` loop records those once, and double-counting corrupts the hit-rate
    metric we care about.
    """
    try:
        num_prompt = request.num_prompt_tokens
    except Exception:
        return _HUGE
    try:
        # Resumed/preempted reqs already have progress; the loop uses that, not a
        # fresh lookup. Mirror it so the key matches the work actually left.
        if getattr(request, "num_computed_tokens", 0):
            return max(num_prompt - request.num_computed_tokens, 0)
        # INDEX, do not unpack: this return grew from 2-tuple (v0.25.0) to 3-tuple in
        # v0.26.0 (+num_uncached_common_prefix_tokens). Unpacking raised ValueError into
        # the `except` below, silently degrading the whole cache-aware reorder to
        # shortest-prompt-first with no log line. Indexing survives the next arity change.
        hit = kv_cache_manager.coordinator.find_longest_cache_hit(
            request.block_hashes, request.num_tokens - 1
        )
        return max(num_prompt - hit[1], 0)
    except Exception:
        return num_prompt  # degrade to shortest-prompt-first; never raise


def _reorder_waiting(waiting, kv_cache_manager, step: int = 0) -> None:
    """In-place stable reorder of a deque waiting queue.

    Cache-aware when memory is slack (fewest-uncached first, pure SJF); memory-aware when
    tight (requests that won't fit the free block pool are demoted below ones that do, so
    we don't admit a prompt that immediately forces a preempt/recompute). This is the
    single-node analog of SGLang's "cache-aware if balanced, else shortest-queue."

    Signals come from our custom ``VtlKVCacheManager`` (``plan_request``/``free_blocks``).
    If that patch is disabled, we fall back to the standalone cache-aware key so the two
    patches stay independent. Non-deque queues are left untouched; stable sort keeps FCFS
    among ties. deque index 0 is the front, so ascending key = admitted first.
    """
    if not isinstance(waiting, deque) or len(waiting) < 2:
        return

    global _last_sorted, _last_sort_step
    # Already in our order and the keys are still fresh: skip n cache-hit walks plus the sort.
    # getattr, not attribute access: an un-keyable request must not raise out of the
    # fingerprint and cost us the whole reorder -- everything else here degrades, so does this.
    ids = tuple(getattr(r, "request_id", None) for r in waiting)
    if ids == _last_sorted and step - _last_sort_step < _RESORT_INTERVAL:
        return

    plan = getattr(kv_cache_manager, "plan_request", None)
    if plan is None:
        # kv_cache_manager patch off: pure cache-aware SJF, no memory signal available.
        key = lambda r: (0, _remaining_prefill(r, kv_cache_manager))  # noqa: E731
    else:
        free = getattr(kv_cache_manager, "free_blocks", _HUGE)
        tight = _safe_usage(kv_cache_manager) >= _USAGE_TIGHT
        if tight:
            global _mem_aware_engaged
            _mem_aware_engaged += 1
            if _mem_aware_engaged == 1:
                log.info("vtl: sched_policy memory-aware branch engaged (KV under pressure)")

        def key(r):
            remaining, blocks_needed = plan(r)
            # ponytail: `free` is a single snapshot -- it shrinks as we admit, but the
            # sort keys off it once. Good enough for ordering; exact per-step accounting
            # is what the base allocate_slots loop already does. Fit-flag first, so a
            # too-big request sinks below every fitting one without disturbing SJF order
            # among peers. When slack, tight=False -> flag is always 0 -> pure SJF.
            fits = 0 if not tight or blocks_needed <= free else 1
            return (fits, remaining)

    ordered = sorted(waiting, key=key)
    waiting.clear()
    waiting.extend(ordered)
    _last_sorted = tuple(getattr(r, "request_id", None) for r in waiting)
    _last_sort_step = step


@register_patch("sched_policy", default=True)
def apply() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    if already_patched(Scheduler, "schedule"):
        return

    original = Scheduler.schedule

    # *args/**kwargs passthrough: schedule()'s signature drifts across versions -- v0.25.0
    # passes should_throttle_prefills. Forward whatever vLLM gives so the wrapper never breaks
    # the call convention; we only reorder the waiting queue first.
    def schedule(self, *args, **kwargs):
        _v2_handshake(self)
        try:
            # current_step is bumped by the FIRST line of the stock schedule(), so we read
            # the previous step's value. Monotonic is all the staleness TTL needs.
            if not _admission_blocked(self):
                _reorder_waiting(
                    self.waiting,
                    self.kv_cache_manager,
                    getattr(self, "current_step", 0),
                )
        except Exception:
            log.exception("vtl: sched_policy reorder failed, using stock order")
        return original(self, *args, **kwargs)

    Scheduler.schedule = mark_patched(schedule, original)
    log.info(
        "vtl: sched_policy installed (cache-aware SJF, memory-aware when KV tight)"
    )


# ponytail: SJF starves long cold prompts under sustained short-prompt load. The
# ceiling is bounded here -- the workload is ~82% cache-hit and multi-turn (few truly
# cold long prompts), and chunked prefill still advances the head each step. Upgrade
# path if a tail-TTFT regression shows up: add aging -- key = remaining_prefill
# - alpha * (now - request.arrival_time) -- ~2 lines, no new infra. Not implemented:
# no evidence it's needed and it adds a tuning knob (alpha) to babysit.

# ponytail: the unchanged-queue guard caches at QUEUE granularity, not per request -- one
# request joining the queue re-walks the prefix for all of them. Per-request memoisation is
# the finer version (`Request` has no __slots__, so a `_vtl_sjf_key` attribute is available),
# but it needs its own invalidation and the queue here is nearly always 0-1 entries deep.
# Ceiling: O(n) walks on the step a request arrives. Upgrade only if a profile shows it.
#
# ponytail: the INTRA-step double lookup is untouched and cannot be fixed from here -- we walk
# the prefix for each waiting request, then the stock loop walks it again for whichever one it
# admits (scheduler.py:715-760). Deduplicating means handing our result into the stock loop,
# i.e. a vLLM source patch under vllm_patches/, which is a much bigger commitment than this
# whole file. Ceiling: one redundant walk per admitted request.


def _self_check() -> None:
    """Runs with no vLLM: pure-python fakes exercise the ordering + fallbacks."""

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
            self.block_hashes = rid  # key into FakeCoord.cached

    def reorder(q, kvm, step=0):
        """Drive _reorder_waiting from a cold cache, so each case stands alone."""
        global _last_sorted, _last_sort_step
        _last_sorted, _last_sort_step = (), -_RESORT_INTERVAL
        _reorder_waiting(q, kvm, step)

    # a=200 prompt, 190 cached -> 10 left; b=50, 0 cached -> 50; c=100, 100 -> 0
    kvm = FakeKVM({"a": 190, "b": 0, "c": 100})
    a, b, c = FakeReq("a", 200), FakeReq("b", 50), FakeReq("c", 100)
    q = deque([a, b, c])  # FCFS order a,b,c
    reorder(q, kvm)
    assert [r.request_id for r in q] == ["c", "a", "b"], [r.request_id for r in q]

    # Stable tie-break keeps FCFS among equal remaining prefill.
    kvm2 = FakeKVM({"x": 0, "y": 0})
    x, y = FakeReq("x", 30), FakeReq("y", 30)
    q2 = deque([x, y])
    reorder(q2, kvm2)
    assert [r.request_id for r in q2] == ["x", "y"]

    # len<2 and non-deque are no-ops.
    solo = deque([a])
    reorder(solo, kvm)
    assert list(solo) == [a]
    reorder([a, b, c], kvm)  # list, not deque: must not raise

    # Lookup failure degrades to prompt length, never raises.
    class BoomKVM:
        class coordinator:
            @staticmethod
            def find_longest_cache_hit(*_):
                raise RuntimeError("boom")

    assert _remaining_prefill(FakeReq("z", 42), BoomKVM) == 42

    # Resumed request keys off its own progress, no lookup.
    assert _remaining_prefill(FakeReq("w", 100, computed=70), None) == 30

    # Memory-aware path: a manager exposing plan_request/free_blocks/usage.
    class FakeVtlKVM:
        def __init__(self, plans, free, usage):
            self._plans = plans  # rid -> (remaining, blocks_needed)
            self.free_blocks = free
            self.usage = usage

        def plan_request(self, r):
            return self._plans[r.request_id]

    plans = {"big": (5, 100), "small": (10, 1)}  # big: smaller SJF key but won't fit
    big, small = FakeReq("big", 0), FakeReq("small", 0)

    # Slack (usage below threshold): pure cache-aware SJF -> big (remaining 5) first.
    slack = FakeVtlKVM(plans, free=10, usage=0.50)
    q3 = deque([big, small])
    reorder(q3, slack)
    assert [r.request_id for r in q3] == ["big", "small"], [r.request_id for r in q3]

    # Tight (usage >= threshold): big needs 100 blocks > 10 free -> demoted below small.
    tight = FakeVtlKVM(plans, free=10, usage=0.95)
    q4 = deque([big, small])
    reorder(q4, tight)
    assert [r.request_id for r in q4] == ["small", "big"], [r.request_id for r in q4]

    # ---- C: the guards ----------------------------------------------------------------
    class CountingCoord(FakeCoord):
        """Counts cache-hit walks, so a skipped re-sort is observable."""

        walks = 0

        def find_longest_cache_hit(self, block_hashes, max_len):
            CountingCoord.walks += 1
            return super().find_longest_cache_hit(block_hashes, max_len)

    class CountingKVM:  # no plan_request -> the standalone cache-aware SJF path
        def __init__(self, cached):
            self.coordinator = CountingCoord(cached)

        @property
        def walks(self):
            return CountingCoord.walks

    counting = CountingKVM({"p": 0, "r": 0})
    p, r_ = FakeReq("p", 40), FakeReq("r", 10)
    q5 = deque([p, r_])
    reorder(q5, counting, step=100)
    assert [q.request_id for q in q5] == ["r", "p"], [q.request_id for q in q5]
    after_first = counting.walks
    assert after_first == 2, after_first

    # Same queue, same order, within the TTL -> no walks at all.
    _reorder_waiting(q5, counting, step=101)
    assert counting.walks == after_first, counting.walks

    # Past the TTL -> re-sorted, keys refreshed.
    _reorder_waiting(q5, counting, step=100 + _RESORT_INTERVAL)
    assert counting.walks == after_first + 2, counting.walks

    # A membership change busts the cache even inside the TTL.
    walks_before = counting.walks
    q5.append(FakeReq("s", 5))
    _reorder_waiting(q5, counting, step=100 + _RESORT_INTERVAL + 1)
    assert counting.walks == walks_before + 3, counting.walks

    # Admission guard: at the running cap there is nothing to order.
    class FakeSched:
        def __init__(self, running, cap, streaming=0):
            self.running = [None] * running
            self.max_num_running_reqs = cap
            self.num_waiting_for_streaming_input = streaming

    assert _admission_blocked(FakeSched(70, 70)) is True
    assert _admission_blocked(FakeSched(69, 70)) is False
    assert _admission_blocked(FakeSched(69, 70, streaming=1)) is True
    # An unrecognisable scheduler must NOT block -- degrade to doing the sort.
    assert _admission_blocked(object()) is False

    # Handshake: hands the flag over exactly once, and never raises on a stock manager.
    global _handshook

    class FakeMgr:
        pass

    class FakeSchedV2:
        def __init__(self, v2):
            self.kv_cache_manager = FakeMgr()
            self.use_v2_model_runner = v2

    _handshook = False
    s = FakeSchedV2(True)
    _v2_handshake(s)
    assert s.kv_cache_manager._vtl_v2 is True
    s2 = FakeSchedV2(False)
    _v2_handshake(s2)  # already handshook: second scheduler is left alone
    assert not hasattr(s2.kv_cache_manager, "_vtl_v2")
    _handshook = False
    _v2_handshake(object())  # no attributes at all: must not raise
    _handshook = False

    print("sched_policy self-check ok")


if __name__ == "__main__":
    _self_check()
