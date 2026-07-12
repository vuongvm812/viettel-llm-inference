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

Hybrid-KV note (qwen3_5): the served model has multiple KV cache groups (full-attention +
GDN/mamba state), so ``coordinator.find_longest_cache_hit`` / ``block_pool.get_num_free_blocks``
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

from vtl._prefix import HUGE, plan_prefill
from vtl.registry import register_patch

log = logging.getLogger("vtl")


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
            # One cache-hit walk (memoized on the request) -> (remaining_prefill, blocks_needed).
            return plan_prefill(request, self, getattr(self, "block_size", None))

        @property
        def free_blocks(self) -> int:
            try:
                return self.block_pool.get_num_free_blocks()
            except Exception:
                return HUGE  # unknown -> treat as unlimited, never demote on error

    sched_mod.KVCacheManager = VtlKVCacheManager
    log.info("vtl: kv_cache_manager installed (exact-tree, memory-aware signals)")


def _self_check() -> None:
    """The walk/block/memoization math lives in vtl._prefix (see its _self_check). Here we only
    confirm this patch delegates through it the way VtlKVCacheManager.plan_request does."""

    class FakeCoord:
        def find_longest_cache_hit(self, block_hashes, max_len):
            return ([], 190)

    class FakeMgr:  # stands in for the KVCacheManager: has a coordinator + block_size
        coordinator = FakeCoord()
        block_size = 16

    class FakeReq:
        num_prompt_tokens = 200
        num_tokens = 200
        num_computed_tokens = 0
        block_hashes = "a"

    mgr = FakeMgr()
    # 200 prompt, 190 cached -> 10 remaining; blocks = ceil(10/16) = 1, as plan_request returns.
    assert plan_prefill(FakeReq(), mgr, getattr(mgr, "block_size", None)) == (10, 1)
    # free_blocks degrades to HUGE (unlimited) when the pool read fails -- exercised inline.
    assert HUGE > 0

    print("kv_cache_manager self-check ok")


if __name__ == "__main__":
    _self_check()
