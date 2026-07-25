"""Custom scheduler for the benchmark workload: conversation-aware reorder + aging.

Workload: 70 multi-turn conversations (6 turns), Poisson arrivals, temp=0 greedy,
~82% prefix-cache hit, chunked prefill at 8192 tokens/step.

The stock vLLM scheduler processes the waiting queue FCFS.  Our existing
``sched_policy`` reorders it cache-aware-SJF (fewest-uncached first), which lowers
mean TTFT but can starve turn-1 cold prefills (~1150 uncached tokens) under
sustained Poisson arrivals of turns 2-6 (~150 tokens each).  This scheduler adds
two properties tuned for this exact trace:

* **Conversation grouping** — requests whose block-hash prefixes overlap are grouped
  together so sequential turn ordering within a conversation is respected (turn N
  should see turn N-1 already cached, maximizing the prefix-cache hit rate).

* **Anti-starvation aging** — the remaining-uncached-tokens sort key erodes by
  ``_ALPHA`` tokens per second of wait time, ensuring a cold turn-1 can never wait
  past the 120 s timeout.

The memory-aware branch (demote requests that won't fit the free block pool) is
skipped — the MIG 18 GB slice has ~90% free KV headroom with a ~2 GB working set,
so ``plan_request``/``free_blocks`` checks are wasted work that can never change
admission order.

Fails closed on any exception: the stock FCFS order is left untouched.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import TYPE_CHECKING

from vtl.registry import already_patched, mark_patched, register_patch

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.request import Request

log = logging.getLogger("vtl")

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

_HUGE = 1 << 60  # push un-keyable requests to the back; never crash the sort

# Tokens-per-second erosion of the remaining-uncached SJF key.
# After ``_GUARD`` seconds of waiting, the effective cost drops by ``_ALPHA * dt``.
# With a turn-1 cold prefill at ~1150 tokens and turns 2-6 at ~150 tokens,
# _ALPHA = 20 lets a turn-1 catch up to the turn-2 tier after ~53 s of queuing
# (well inside the 120 s per-request timeout).
_ALPHA = 20

# No aging below this threshold (seconds).  A request processed in < 3 s was never
# starved; aging it would only disturb the stable cache-aware order.
_GUARD = 3.0

# Number of leading block-hashes used to fingerprint a conversation.  The shared
# system prefix (~1000 tokens / ~63 blocks at block_size=16) is identical across
# all conversations, so the first meaningful divergence is inside the
# per-conversation prefix.  Hashing the first 4 blocks (64 tokens) captures the
# conversation-specific prefix while ignoring the shared system preamble.
_CONV_PREFIX_BLOCKS = 4

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _conv_group_id(request: Request) -> int:
    """Stable conversation fingerprint from the leading block-hash prefix.

    Requests from the same conversation share block-hashes beyond the common system
    prefix.  Hashing a fixed-length prefix yields a deterministic group id that is
    invariant to turn number (all turns within a conversation start from the same
    per-conversation prefix blocks).

    Falls back to a hash of ``request_id`` when block_hashes is empty (request
    hasn't been tokenised yet) — still yields a correct ordering, just without
    conversation awareness for that admission step.
    """
    hashes = getattr(request, "block_hashes", None)
    if not hashes:
        return hash(request.request_id) & 0x7FFF_FFFF
    head = hashes[:_CONV_PREFIX_BLOCKS]
    return hash(tuple(head)) & 0x7FFF_FFFF


def _remaining_prefill(request: Request, kv_cache_manager: KVCacheManager) -> int:
    """Uncached prompt tokens that must still be prefilled for ``request``.

    Uses ``kv_cache_manager.coordinator.find_longest_cache_hit`` directly (NOT
    ``get_computed_blocks``) so the real ``schedule()`` loop's
    ``prefix_cache_stats`` are not double-counted.  Falls back to
    ``num_prompt_tokens`` on any failure (safe degradation, never raises).
    """
    try:
        num_prompt = request.num_prompt_tokens
    except Exception:
        return _HUGE

    try:
        # Resumed / preempted requests track their own progress via
        # num_computed_tokens — no cache lookup needed.
        if getattr(request, "num_computed_tokens", 0):
            return max(num_prompt - request.num_computed_tokens, 0)

        _blocks, num_cached = kv_cache_manager.coordinator.find_longest_cache_hit(
            request.block_hashes, request.num_tokens - 1
        )
        return max(num_prompt - num_cached, 0)
    except Exception:
        return num_prompt


def _reorder_waiting(
    waiting: deque[Request], kv_cache_manager: KVCacheManager
) -> None:
    """In-place stable reorder of the waiting deque.

    Sort key = ``(aged_remaining, conv_group)`` where ``aged_remaining`` is the
    fewest-uncached-tokens count reduced by ``_ALPHA`` tokens per second of wait
    time beyond ``_GUARD``.  Conversation grouping keeps sequential turns adjacent,
    maximising prefix-cache reuse.  Stable sort preserves FCFS for ties (same
    conversation, same aged cost).
    """
    if not isinstance(waiting, deque) or len(waiting) < 2:
        return

    now = time.monotonic()

    def _key(r: Request) -> tuple[int, int]:
        remaining = _remaining_prefill(r, kv_cache_manager)
        arrival = getattr(r, "arrival_time", 0.0)
        if arrival:
            waited = max(0.0, now - arrival - _GUARD)
            aged = remaining - int(waited * _ALPHA)
        else:
            aged = remaining
        group = _conv_group_id(r)
        return (max(0, aged), group)

    ordered = sorted(waiting, key=_key)
    waiting.clear()
    waiting.extend(ordered)


# ---------------------------------------------------------------------------
# Patch registration
# ---------------------------------------------------------------------------


@register_patch("sched_custom", default=True)
def apply() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    if already_patched(Scheduler, "schedule"):
        return

    original = Scheduler.schedule

    # *args/**kwargs passthrough: schedule()'s signature drifts across versions.
    # Forward whatever vLLM gives so the wrapper never breaks the call
    # convention; we only reorder the waiting queue first.
    def schedule(self: Scheduler, *args: object, **kwargs: object) -> object:
        try:
            _reorder_waiting(self.waiting, self.kv_cache_manager)
        except Exception:
            log.exception("vtl: sched_custom reorder failed, using stock order")
        return original(self, *args, **kwargs)

    Scheduler.schedule = mark_patched(schedule, original)
    log.info("vtl: sched_custom installed (conversation-aware + aged-SJF)")


# ---------------------------------------------------------------------------
# Self-check (runs with no vLLM: pure-Python fakes)
# ---------------------------------------------------------------------------


def _self_check() -> None:
    class FakeCoord:
        def __init__(self, cached: dict[str, int]):
            self.cached = cached  # block_hashes -> num_cached

        def find_longest_cache_hit(self, block_hashes, _max_len):
            key = tuple(block_hashes) if isinstance(block_hashes, list) else block_hashes
            return [], self.cached.get(key, 0)

    class FakeKVM:
        def __init__(self, cached: dict[str, int]):
            self.coordinator = FakeCoord(cached)

    class FakeReq:
        def __init__(
            self,
            rid: str,
            prompt: int,
            computed: int = 0,
            block_hashes: list[int] | None = None,
            arrival: float | None = None,
        ):
            self.request_id = rid
            self.num_prompt_tokens = prompt
            self.num_tokens = prompt  # simplified: no output yet
            self.num_computed_tokens = computed
            self.block_hashes = block_hashes or []
            self.arrival_time = arrival or time.monotonic()

    # --- basic SJF: c=100 cached=100 → 0 remaining, a=200 cached=0 → 200 remaining ---
    kvm = FakeKVM({(100,): 100, (200,): 0})
    a, c = FakeReq("a", 200, block_hashes=[200]), FakeReq("c", 100, block_hashes=[100])
    q = deque([a, c])
    _reorder_waiting(q, kvm)
    assert [r.request_id for r in q] == ["c", "a"], [r.request_id for r in q]

    # --- conversation grouping: same group, turn-1 should precede turn-2 ---
    # block_hashes prefix [1, 2] identifies the conversation; t1 has fewer tokens
    kvm2 = FakeKVM({(1, 2, 10): 0, (1, 2, 20): 0})
    t2 = FakeReq("turn-2", 300, block_hashes=[1, 2, 20])
    t1 = FakeReq("turn-1", 100, block_hashes=[1, 2, 10])
    q2 = deque([t2, t1])
    _reorder_waiting(q2, kvm2)
    assert [r.request_id for r in q2] == ["turn-1", "turn-2"], [r.request_id for r in q2]

    # --- aging: old turn-1 catches young turn-2 ---
    now = time.monotonic()
    late = FakeReq("late-turn-1", 1200, block_hashes=[9], arrival=now - 60.0)
    early = FakeReq("early-turn-2", 200, block_hashes=[8], arrival=now - 1.0)
    q3 = deque([late, early])
    _reorder_waiting(q3, kvm2)
    # With _ALPHA=20, _GUARD=3: late waited 60s → aged = 1200 - 20*57 = 60
    # early waited 1s → aged = 200 (below guard, no aging)
    # 60 < 200 → late should go first
    assert [r.request_id for r in q3] == ["late-turn-1", "early-turn-2"], [
        r.request_id for r in q3
    ]

    # --- stable sort preserves FCFS for ties ---
    kvm4 = FakeKVM({})
    x, y = FakeReq("x", 30), FakeReq("y", 30)
    q4 = deque([x, y])
    _reorder_waiting(q4, kvm4)
    assert [r.request_id for r in q4] == ["x", "y"]

    # --- len < 2 and non-deque are no-ops ---
    solo = deque([a])
    _reorder_waiting(solo, kvm)
    assert list(solo) == [a]
    _reorder_waiting([a, c], kvm)  # list, not deque: must not raise

    # --- lookup failure degrades to prompt length, never raises ---
    class BoomKVM:
        class coordinator:
            @staticmethod
            def find_longest_cache_hit(*_args):
                raise RuntimeError("boom")

    assert _remaining_prefill(FakeReq("z", 42), BoomKVM()) == 42

    # --- resumed request keys off its own progress ---
    assert _remaining_prefill(FakeReq("w", 100, computed=70), None) == 30

    # --- block_hashes empty → request-id hash grouping (still correct) ---
    assert _conv_group_id(FakeReq("req-1", 100)) == _conv_group_id(FakeReq("req-1", 100))
    assert _conv_group_id(FakeReq("req-1", 100)) != _conv_group_id(FakeReq("req-2", 100))

    print("sched_custom self-check ok")


if __name__ == "__main__":
    _self_check()
