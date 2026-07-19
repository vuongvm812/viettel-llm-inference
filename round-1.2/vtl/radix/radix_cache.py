"""Token-granular RadixCache flow -- port of SGLang's scheduler-facing ``RadixCache``
lifecycle (``sglang/python/sglang/srt/mem_cache/radix_cache.py``:
``match_prefix`` / ``cache_unfinished_req`` / ``cache_finished_req`` / ``inc_lock_ref`` /
``dec_lock_ref`` / ``evict``).

This ties the three SGLang memory-model layers together into the actual RadixAttention flow:

    tree   -- vtl.radix_tree.VtlRadixCache: the radix tree over token ids -> KV slot indices
    pool   -- ReqToTokenPool: request -> its token slots (the attention indirection table)
    alloc  -- TokenToKVPoolAllocator: token-granular KV slot allocator

The lifecycle per request (what the scheduler drives):

    kv_indices, node = match_prefix(token_ids)   # reuse cached prefix slots
    inc_lock_ref(node)                           # pin so the prefix is not evicted mid-run
    ... allocate slots for the uncached suffix, write pool, run prefill/decode ...
    cache_finished_req(req_slot, token_ids, node, cache_protected_len)
        insert token_ids -> slots into the tree (tree takes a ref)
        free the DUPLICATE prefix slots this request re-allocated
        dec_lock_ref(node); free the request slot

At ``page_size=1`` (the token-granular case) there is no partial-page tail, so the
alignment/tail-freeing SGLang does is omitted. ``page_size>1`` and the hybrid mamba group
are on-box frontier work -- see INTEGRATION.md.

Off-box: reuses the pure-python tree, pool, and allocator, so the whole flow runs and is
asserted without torch, vLLM, or a GPU.
"""

from __future__ import annotations

import os
import sys

try:
    from vtl.radix_tree import VtlRadixCache
except ImportError:  # direct-run fallback (python radix_cache.py)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from radix_tree import VtlRadixCache

from vtl.radix.req_to_token_pool import ReqToTokenPool
from vtl.radix.token_allocator import TokenToKVPoolAllocator


class TokenRadixCache:
    def __init__(
        self,
        req_pool: ReqToTokenPool,
        allocator: TokenToKVPoolAllocator,
        page_size: int = 1,
    ):
        assert page_size == 1, "off-box flow supports page_size=1 (token granularity) only"
        self.req_pool = req_pool
        self.allocator = allocator
        self.page_size = page_size
        self.tree = VtlRadixCache(allocator=allocator)

    # ----- match / lock (before the request runs) --------------------------

    def match_prefix(self, token_ids) -> tuple[list, object]:
        """Longest cached prefix: ``(kv_slot_indices, last_node)``. The slots are the
        SHARED cached KV -- the request reuses them instead of recomputing."""
        return self.tree.match_prefix(list(token_ids))

    def inc_lock_ref(self, node) -> None:
        self.tree.inc_lock_ref(node)

    def dec_lock_ref(self, node) -> None:
        if node is not None and node is not self.tree.root_node:
            self.tree.dec_lock_ref(node)

    def evict(self, num_tokens: int) -> int:
        return self.tree.evict(num_tokens)

    # ----- cache on finish / chunk boundary (after the request runs) -------

    def cache_finished_req(
        self, req_slot: int, token_ids, last_node, cache_protected_len: int = 0
    ) -> None:
        """Insert the finished request's tokens into the tree, free the duplicate prefix
        slots it re-allocated, release its lock and its request slot."""
        kv_indices = self.req_pool.read(req_slot, len(token_ids))
        prefix_len = self.tree.insert(list(token_ids), list(kv_indices))
        # Slots for [cache_protected_len, prefix_len) were already in the tree from an
        # earlier request -> this request's copies are duplicates, free them.
        self.allocator.free(kv_indices[cache_protected_len:prefix_len])
        self.dec_lock_ref(last_node)
        self.req_pool.free(req_slot)

    def cache_unfinished_req(
        self, req_slot: int, token_ids, last_node, cache_protected_len: int = 0
    ) -> tuple[list, object, int]:
        """Cache at a chunk boundary: insert, free duplicates, re-point the request at the
        canonical shared slots, and move the lock to the new terminal node.

        Returns ``(new_indices, new_last_node, new_cache_protected_len)``.
        """
        kv_indices = self.req_pool.read(req_slot, len(token_ids))
        prefix_len = self.tree.insert(list(token_ids), list(kv_indices))
        self.allocator.free(kv_indices[cache_protected_len:prefix_len])

        new_indices, new_last_node = self.tree.match_prefix(list(token_ids))
        assert len(new_indices) == len(token_ids), (len(new_indices), len(token_ids))
        # Re-point the request's table at the shared slots (from cache_protected_len on).
        self.req_pool.write(
            req_slot, cache_protected_len, new_indices[cache_protected_len:]
        )
        self.dec_lock_ref(last_node)
        self.inc_lock_ref(new_last_node)
        return new_indices, new_last_node, len(new_indices)


def _self_check() -> None:
    # ----- end-to-end: two requests share a prefix, KV is reused, nothing leaks -----
    alloc = TokenToKVPoolAllocator(size=100)
    pool = ReqToTokenPool(size=10, max_context_len=64)
    cache = TokenRadixCache(pool, alloc, page_size=1)
    full = alloc.available_size()  # 100

    # Request A: tokens [1,2,3,4], no cached prefix. Allocates its own 4 slots.
    a_tokens = [1, 2, 3, 4]
    a_prefix, a_node = cache.match_prefix(a_tokens)
    assert a_prefix == [] and a_node is cache.tree.root_node
    ra = pool.alloc(1)[0]
    sa = alloc.alloc(len(a_tokens))          # [1,2,3,4]
    pool.write(ra, 0, sa)
    assert alloc.available_size() == full - 4
    cache.cache_finished_req(ra, a_tokens, last_node=a_node, cache_protected_len=0)
    # A's 4 slots are now owned by the tree; none freed back yet.
    assert alloc.available_size() == full - 4

    # Request B: tokens [1,2,3,5], shares prefix [1,2,3] with A.
    b_tokens = [1, 2, 3, 5]
    b_prefix, b_node = cache.match_prefix(b_tokens)
    assert b_prefix == [1, 2, 3], b_prefix       # REUSES A's slots 1,2,3 -- the whole point
    cache.inc_lock_ref(b_node)                   # pin the shared prefix during B's run
    rb = pool.alloc(1)[0]
    sb = alloc.alloc(len(b_tokens) - len(b_prefix))  # 1 new slot for token [5] -> [5]
    assert sb == [5]
    pool.write(rb, 0, b_prefix + sb)             # B's row: [1,2,3,5], prefix shared
    # B allocated only 1 new slot (reuse saved 3):
    assert alloc.available_size() == full - 5
    cache.cache_finished_req(rb, b_tokens, last_node=b_node, cache_protected_len=len(b_prefix))
    # No duplicates freed (B reused, didn't copy); [5] now owned by the tree.
    assert alloc.available_size() == full - 5

    # Tree holds [1,2,3] -> {[4], [5]}: 5 unique slots. Evicting reclaims them all.
    freed = cache.evict(1000)
    assert freed == 5, freed
    assert alloc.available_size() == full        # no leak, no double-free

    # ----- lock_ref pins a shared prefix out of eviction -----
    alloc2 = TokenToKVPoolAllocator(size=50)
    pool2 = ReqToTokenPool(size=10, max_context_len=64)
    c2 = TokenRadixCache(pool2, alloc2, page_size=1)
    rc = pool2.alloc(1)[0]
    sc = alloc2.alloc(2)
    pool2.write(rc, 0, sc)
    c2.cache_finished_req(rc, [7, 8], last_node=c2.tree.root_node, cache_protected_len=0)
    _, node78 = c2.match_prefix([7, 8])
    c2.inc_lock_ref(node78)
    assert c2.evict(1000) == 0                    # pinned -> nothing evictable
    c2.dec_lock_ref(node78)
    assert c2.evict(1000) == 2                    # released -> frees

    print("radix_cache (token flow) self-check ok")


if __name__ == "__main__":
    _self_check()
