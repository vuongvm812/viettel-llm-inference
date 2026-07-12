"""Shared prefix-cache planning for the two scheduler patches.

``sched_policy`` (cache-aware SJF reorder) and ``kv_cache_manager`` (memory-aware signals)
both need the same thing for a waiting request: how many of its prompt tokens are NOT already
in the prefix cache (the SJF key), and how many KV blocks that uncached tail will allocate (the
memory key). Both come from ONE ``coordinator.find_longest_cache_hit`` walk -- a traversal of up
to ~1000 prefix-chained block hashes for an 18.7K-token prompt. This module holds the single
copy of that walk, its degrade-to-prompt-length fallbacks, and -- new -- its memoization.

**Memoization.** The walk ran once per waiting request on *every* ``schedule()`` call (the sort
key is evaluated eagerly for each element). But a cold request's uncached-prefill count only
*shrinks* while it waits (the prefix cache grows, never un-caches its hits), so re-walking every
step is wasted work on the single-threaded EngineCore critical path. We walk once, stash the
result on the request, and reuse it. A stash that goes stale (a shared prefix gets cached by
another request mid-wait) is at worst conservative -- the request is ordered as if less of it
were cached -- and both callers are order-only / output-preserving, so a conservative key can
never change a result, only a scheduling position. The resumed/preempted branch
(``num_computed_tokens > 0``) is cheap arithmetic and is always recomputed, ignoring any stash.

Pure-Python, no vLLM imports -- so importing it costs nothing and keeps the two patches
independent (each still degrades on its own if the other is disabled).
"""

from __future__ import annotations

HUGE = 1 << 60  # un-plannable request: pushed to the back / treated as unlimited free
_DEFAULT_BLOCK_SIZE = 16  # only a fallback if the manager exposes no block_size

_STASH_ATTR = "_vtl_remaining"  # memoized cold-walk result, set on the request object


def _remaining(request, kv_cache_manager, num_prompt: int) -> int:
    """Uncached prompt tokens for ``request``. Memoized on the cold path; never raises.

    Uses the coordinator directly (NOT ``get_computed_blocks``) so we do not pollute
    ``prefix_cache_stats`` -- the real ``schedule()`` loop records those once, and
    double-counting corrupts the hit-rate metric.
    """
    # Resumed/preempted reqs already have progress; the loop uses that, not a fresh lookup.
    # Cheap -- always recompute, and ignore any (now-stale) cold stash.
    if getattr(request, "num_computed_tokens", 0):
        return max(num_prompt - request.num_computed_tokens, 0)

    stashed = getattr(request, _STASH_ATTR, None)
    if stashed is not None:
        return stashed

    try:
        _blocks, num_cached = kv_cache_manager.coordinator.find_longest_cache_hit(
            request.block_hashes, request.num_tokens - 1
        )
        remaining = max(num_prompt - num_cached, 0)
    except Exception:
        # Degrade to shortest-prompt-first; do NOT memoize a degraded value so a transient
        # failure self-heals next step (a persistent one just costs the immediately-raising try).
        return num_prompt

    try:
        setattr(request, _STASH_ATTR, remaining)  # skipped silently if Request is __slots__'d
    except Exception:
        pass
    return remaining


def plan_prefill(request, kv_cache_manager, block_size: int | None = None):
    """One cache-hit walk -> ``(remaining_prefill, blocks_needed)``.

    ``remaining_prefill`` = uncached prompt tokens (the cache-aware SJF key).
    ``blocks_needed`` = KV blocks the uncached tail will still allocate (the memory-aware key).
    Degrades to prompt length on any failure; an un-plannable request (no ``num_prompt_tokens``)
    sinks to the back as ``(HUGE, HUGE)``. Never raises.
    """
    try:
        num_prompt = request.num_prompt_tokens
    except Exception:
        return HUGE, HUGE
    remaining = _remaining(request, kv_cache_manager, num_prompt)
    bs = block_size if (block_size and block_size > 0) else _DEFAULT_BLOCK_SIZE
    blocks_needed = (remaining + bs - 1) // bs
    return remaining, blocks_needed


def _self_check() -> None:
    """Runs with no vLLM: pure-python fakes exercise the walk, memoization, and fallbacks."""

    class FakeCoord:
        def __init__(self, cached):  # cached: {block_hashes_key: num_cached_tokens}
            self.cached = cached
            self.calls = 0

        def find_longest_cache_hit(self, block_hashes, max_len):
            self.calls += 1
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

    kvm = FakeKVM({"a": 190, "b": 0})
    # a: 200 prompt, 190 cached -> 10 remaining; blocks = ceil(10/16) = 1
    assert plan_prefill(FakeReq("a", 200), kvm, 16) == (10, 1)
    # b: 50 prompt, 0 cached -> 50 remaining; blocks = ceil(50/16) = 4
    assert plan_prefill(FakeReq("b", 50), kvm, 16) == (50, 4)
    # exact multiple: 32 remaining / 16 -> 2 blocks, no off-by-one
    assert plan_prefill(FakeReq("c", 32), FakeKVM({"c": 0}), 16) == (32, 2)

    # Memoization: a repeated request walks the coordinator exactly ONCE.
    memo_kvm = FakeKVM({"m": 90})
    r = FakeReq("m", 100)
    assert plan_prefill(r, memo_kvm, 16) == (10, 1)
    assert plan_prefill(r, memo_kvm, 16) == (10, 1)
    assert plan_prefill(r, memo_kvm, 16) == (10, 1)
    assert memo_kvm.coordinator.calls == 1, memo_kvm.coordinator.calls
    assert getattr(r, _STASH_ATTR) == 10

    # Resumed request keys off its own progress, no lookup, ignores any stash.
    resumed_kvm = FakeKVM({})
    assert plan_prefill(FakeReq("w", 100, computed=70), resumed_kvm, 16) == (30, 2)
    assert resumed_kvm.coordinator.calls == 0

    # Lookup failure degrades to prompt length, never raises, and is NOT memoized
    # (so it re-attempts next step).
    class BoomCoord:
        def __init__(self):
            self.calls = 0

        def find_longest_cache_hit(self, *_):
            self.calls += 1
            raise RuntimeError("boom")

    class BoomKVM:
        def __init__(self):
            self.coordinator = BoomCoord()

    bkvm = BoomKVM()
    rz = FakeReq("z", 42)
    assert plan_prefill(rz, bkvm, 16) == (42, 3)
    assert plan_prefill(rz, bkvm, 16) == (42, 3)
    assert bkvm.coordinator.calls == 2  # retried, not memoized
    assert getattr(rz, _STASH_ATTR, None) is None

    # Un-plannable request (no num_prompt_tokens) sinks to the back.
    class Bad:
        pass

    assert plan_prefill(Bad(), kvm, 16) == (HUGE, HUGE)

    # block_size <= 0 / None falls back to the default, still divides cleanly.
    assert plan_prefill(FakeReq("d", 32), FakeKVM({"d": 0}), 0) == (
        32,
        (32 + _DEFAULT_BLOCK_SIZE - 1) // _DEFAULT_BLOCK_SIZE,
    )
    assert plan_prefill(FakeReq("e", 32), FakeKVM({"e": 0}), None) == (
        32,
        (32 + _DEFAULT_BLOCK_SIZE - 1) // _DEFAULT_BLOCK_SIZE,
    )

    # A __slots__'d Request (no stash possible) still returns the right answer every call.
    class Slotted:
        __slots__ = ("num_prompt_tokens", "num_tokens", "num_computed_tokens", "block_hashes")

        def __init__(self):
            self.num_prompt_tokens = 100
            self.num_tokens = 100
            self.num_computed_tokens = 0
            self.block_hashes = "s"

    slot_kvm = FakeKVM({"s": 40})
    s = Slotted()
    assert plan_prefill(s, slot_kvm, 16) == (60, 4)
    assert plan_prefill(s, slot_kvm, 16) == (60, 4)
    assert slot_kvm.coordinator.calls == 2  # can't memoize -> re-walks, but correct

    print("_prefix self-check ok")


if __name__ == "__main__":
    _self_check()
