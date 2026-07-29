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

Set ``VTL_ENABLE_SCHED_POLICY=0`` to serve stock FCFS.
"""

from __future__ import annotations

import logging
from collections import deque

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vllm.vtl.sched_policy")

_HUGE = 1 << 60  # push un-keyable requests to the back, never crash the sort
_USAGE_TIGHT = 0.90  # KV usage above which we prefer requests that fit over pure SJF
_mem_aware_engaged = 0  # how many schedule() calls hit the memory-aware branch


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
        _blocks, num_cached = kv_cache_manager.coordinator.find_longest_cache_hit(
            request.block_hashes, request.num_tokens - 1
        )
        return max(num_prompt - num_cached, 0)
    except Exception:
        return num_prompt  # degrade to shortest-prompt-first; never raise


def _reorder_waiting(waiting, kv_cache_manager) -> None:
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
        try:
            _reorder_waiting(self.waiting, self.kv_cache_manager)
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


def _self_check() -> None:
    """Runs with no vLLM: pure-python fakes exercise the ordering + fallbacks."""

    class FakeCoord:
        def __init__(self, cached):  # cached: {block_hashes_key: num_cached_tokens}
            self.cached = cached

        def find_longest_cache_hit(self, block_hashes, max_len):
            return ([], self.cached.get(block_hashes, 0))

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

    # a=200 prompt, 190 cached -> 10 left; b=50, 0 cached -> 50; c=100, 100 -> 0
    kvm = FakeKVM({"a": 190, "b": 0, "c": 100})
    a, b, c = FakeReq("a", 200), FakeReq("b", 50), FakeReq("c", 100)
    q = deque([a, b, c])  # FCFS order a,b,c
    _reorder_waiting(q, kvm)
    assert [r.request_id for r in q] == ["c", "a", "b"], [r.request_id for r in q]

    # Stable tie-break keeps FCFS among equal remaining prefill.
    kvm2 = FakeKVM({"x": 0, "y": 0})
    x, y = FakeReq("x", 30), FakeReq("y", 30)
    q2 = deque([x, y])
    _reorder_waiting(q2, kvm2)
    assert [r.request_id for r in q2] == ["x", "y"]

    # len<2 and non-deque are no-ops.
    solo = deque([a])
    _reorder_waiting(solo, kvm)
    assert list(solo) == [a]
    _reorder_waiting([a, b, c], kvm)  # list, not deque: must not raise

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
    _reorder_waiting(q3, slack)
    assert [r.request_id for r in q3] == ["big", "small"], [r.request_id for r in q3]

    # Tight (usage >= threshold): big needs 100 blocks > 10 free -> demoted below small.
    tight = FakeVtlKVM(plans, free=10, usage=0.95)
    q4 = deque([big, small])
    _reorder_waiting(q4, tight)
    assert [r.request_id for r in q4] == ["small", "big"], [r.request_id for r in q4]

    print("sched_policy self-check ok")


if __name__ == "__main__":
    _self_check()
