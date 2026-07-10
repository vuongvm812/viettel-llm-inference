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

Set ``VTL_ENABLE_SCHED_POLICY=0`` to serve stock FCFS.
"""

from __future__ import annotations

import logging
from collections import deque

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vtl")

_HUGE = 1 << 60  # push un-keyable requests to the back, never crash the sort


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
    """In-place stable reorder of a deque waiting queue, fewest-uncached first.

    Non-deque queues (``PriorityRequestQueue``, unknown shapes) are left untouched.
    Stable sort keeps FCFS order among ties. deque index 0 is the front
    (``peek_request``/``popleft``), so ascending key = shortest remaining prefill
    admitted first.
    """
    if not isinstance(waiting, deque) or len(waiting) < 2:
        return
    ordered = sorted(waiting, key=lambda r: _remaining_prefill(r, kv_cache_manager))
    waiting.clear()
    waiting.extend(ordered)


@register_patch("sched_policy", default=True)
def apply() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    if already_patched(Scheduler, "schedule"):
        return

    original = Scheduler.schedule

    def schedule(self):
        try:
            _reorder_waiting(self.waiting, self.kv_cache_manager)
        except Exception:
            log.exception("vtl: sched_policy reorder failed, using stock order")
        return original(self)

    Scheduler.schedule = mark_patched(schedule, original)
    log.info(
        "vtl: sched_policy installed (cache-aware shortest-remaining-prefill-first)"
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

    print("sched_policy self-check ok")


if __name__ == "__main__":
    _self_check()
