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

Hybrid-KV note: a hybrid model has multiple KV cache groups (full-attention +
short-conv/mamba state), so ``coordinator.find_longest_cache_hit`` / ``block_pool.get_num_free_blocks``
may behave differently than the single-group case these signals assume. That is a PERF concern,
not a correctness one: both are read-only and only feed scheduling ORDER, and both degrade on
any API mismatch (hit-walk failure -> prompt-length; free_blocks failure -> unlimited). Worst
case is a suboptimal admission order, never a wrong result. Re-validate the benefit on the box.

The scheduler constructs ``KVCacheManager`` at exactly one call site
(``Scheduler.__init__``), by the name imported into ``vllm.v1.core.sched.scheduler``.
Rebinding that name to our subclass makes every scheduler (sync and async) build ours,
with zero constructor changes and no worker-side coupling (workers never hold it).

Set ``VTL_ENABLE_KV_CACHE_MANAGER=0`` to serve vLLM's stock manager.
"""

from __future__ import annotations

import logging

from vtl.registry import register_patch

log = logging.getLogger("vllm.vtl.kv_cache_manager")

_HUGE = 1 << 60  # un-plannable request: pushed to the back / treated as unlimited free
_DEFAULT_BLOCK_SIZE = 16  # only a fallback if the manager exposes no block_size


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
            _blocks, num_cached = coord_holder.coordinator.find_longest_cache_hit(
                request.block_hashes, request.num_tokens - 1
            )
            remaining = max(num_prompt - num_cached, 0)
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

    sched_mod.KVCacheManager = VtlKVCacheManager
    log.info("vtl: kv_cache_manager installed (exact-tree, memory-aware signals)")


def _self_check() -> None:
    """Runs with no vLLM: pure-python fakes exercise the walk + block math + fallbacks."""

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

    print("kv_cache_manager self-check ok")


if __name__ == "__main__":
    _self_check()
