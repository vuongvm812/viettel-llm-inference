"""Explicit radix tree for vLLM's block-hash prefix cache -- a faithful port of
SGLang's ``RadixCache`` (``sglang/python/sglang/srt/mem_cache/radix_cache.py``).

Why this exists: vLLM V1's prefix cache already IS a radix index -- block hashes
are prefix-chained (``hash(parent_hash, block_tokens)``) and ``find_longest_cache_hit``
walks that chain, one ``get_cached_block`` dict lookup per block, stopping at the
first miss. This module re-expresses that flat ``cached_block_hash_to_id`` index as
the explicit ``TreeNode`` structure SGLang calls "RadixAttention", so it can be
traversed, inspected, and A/B-measured against vLLM's own walk.

Granularity: SGLang's tree is token-granular over a ``ReqToTokenPool`` that vLLM has
no analog for. vLLM shares KV at ``block_size`` granularity, so here one tree *unit*
is one block: ``key`` is a list of block hashes, ``value`` is the matching list of
block ids. Everything SGLang layered on top of the core -- ``extra_key``/LoRA salt,
EAGLE bigram views, HiCache host tiering, SHA256 hash recompute, priority eviction,
kv-cache-event/session mixins -- is dropped; vLLM supplies the hashes and owns the
blocks.

Standalone: no torch, no vLLM import. ``value`` is a plain ``list``; the allocator is
any object with ``.free(value_list)``. Run ``python radix_tree.py`` for the self-check.
"""

from __future__ import annotations

import heapq
from typing import Any, Hashable, Optional


class RadixKey:
    """A sequence of block hashes (one hashable unit per KV block).

    Reduced from SGLang's ``RadixKey``: no ``extra_key`` namespace, no bigram view,
    no page alignment -- every unit is already a whole block, so ``page_size`` is 1.
    """

    __slots__ = ("units",)

    def __init__(self, units: list[Hashable]):
        self.units = units

    def __len__(self) -> int:
        return len(self.units)

    def __getitem__(self, idx: slice) -> "RadixKey":
        return RadixKey(self.units[idx])

    def child_key(self) -> Hashable:
        """Dict key for the first unit -- how a parent indexes its children."""
        return self.units[0]

    def match(self, other: "RadixKey") -> int:
        """Length of the shared leading run of units with ``other``.

        ponytail: linear scan. SGLang gallops with C-level slice compares because its
        keys are long token arrays; block-hash keys are short (prompt_len/block_size),
        so a plain loop is simpler and not measurably slower. Upgrade to bisect only
        if a profile ever says so.
        """
        a, b = self.units, other.units
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i


class TreeNode:
    __slots__ = ("children", "parent", "key", "value", "lock_ref", "last_access")

    def __init__(self) -> None:
        self.children: dict[Hashable, "TreeNode"] = {}
        self.parent: Optional["TreeNode"] = None
        self.key: Optional[RadixKey] = None
        self.value: Optional[list] = None  # None == evicted
        self.lock_ref = 0
        self.last_access = 0

    @property
    def evicted(self) -> bool:
        return self.value is None

    def __lt__(self, other: "TreeNode") -> bool:  # heap tiebreak
        return self.last_access < other.last_access


class VtlRadixCache:
    """Block-granular radix tree over a KV-block allocator.

    ``allocator`` needs only ``.free(value_list)``. In vLLM that is
    ``BlockPool.free_blocks``; in the self-check it is a plain fake. LRU order uses a
    logical clock (a monotonically increasing counter) instead of wall time, so
    eviction order is deterministic and testable.
    """

    def __init__(self, allocator: Optional[Any] = None):
        self.allocator = allocator
        self._clock = 0
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.evictable_leaves: set[TreeNode] = set()
        self.reset()

    @classmethod
    def create_simulated(cls, allocator: Optional[Any] = None) -> "VtlRadixCache":
        """Build a tree with no real KV pool, for simulation/tests."""
        return cls(allocator=allocator)

    def reset(self) -> None:
        self.root_node = TreeNode()
        self.root_node.key = RadixKey([])
        self.root_node.value = []
        self.root_node.lock_ref = 1  # root is never evictable
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.evictable_leaves.clear()

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    # ----- public API -------------------------------------------------------

    def match_prefix(self, units: list[Hashable]) -> tuple[list, TreeNode]:
        """Longest cached prefix of ``units``.

        Returns ``(matched_values, last_node)`` where ``matched_values`` is the flat
        list of block ids for the matched prefix (``len`` == matched block count).
        May split a node if the match ends mid-segment.
        """
        if len(units) == 0:
            return [], self.root_node
        segments, node = self._match_prefix_helper(self.root_node, RadixKey(units))
        flat: list = []
        for seg in segments:
            flat.extend(seg)
        return flat, node

    def match_prefix_len(self, units: list[Hashable]) -> int:
        return len(self.match_prefix(units)[0])

    def insert(self, units: list[Hashable], value: list) -> int:
        """Insert ``units`` -> ``value`` (equal length). Returns the prefix length
        (in units) that was already present."""
        if len(units) == 0:
            return 0
        assert len(units) == len(value), (len(units), len(value))
        prefix_len, _ = self._insert_helper(self.root_node, RadixKey(units), list(value))
        return prefix_len

    def evict(self, num_units: int) -> int:
        """Free up to ``num_units`` blocks from LRU evictable leaves. Returns the
        number actually freed."""
        heap = [(n.last_access, n) for n in self.evictable_leaves]
        heapq.heapify(heap)
        num_evicted = 0
        while num_evicted < num_units and heap:
            _, x = heapq.heappop(heap)
            if self.allocator is not None:
                self.allocator.free(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)
            if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                heapq.heappush(heap, (x.parent.last_access, x.parent))
        return num_evicted

    def inc_lock_ref(self, node: TreeNode) -> None:
        """Pin ``node``..root so its blocks are not evictable while a request uses them."""
        while node is not self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            node = node.parent

    def dec_lock_ref(self, node: TreeNode) -> None:
        while node is not self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            node = node.parent

    def total_size(self) -> int:
        total = 0
        stack = [self.root_node]
        while stack:
            n = stack.pop()
            total += len(n.value)
            for c in n.children.values():
                if not c.evicted:
                    stack.append(c)
        return total

    def pretty_print(self) -> None:
        stack = [(self.root_node, 0)]
        while stack:
            node, indent = stack.pop()
            print(" " * indent, len(node.key), list(node.key.units)[:10],
                  f"r={node.lock_ref}")
            for child in node.children.values():
                stack.append((child, indent + 2))
        print(f"#blocks: {self.total_size()}")

    # ----- internal ---------------------------------------------------------

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        access = self._tick()
        node.last_access = access
        child_key = key.child_key()

        value: list[list] = []
        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]
            child.last_access = access
            prefix_len = child.key.match(key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            value.append(child.value)
            node = child
            key = key[prefix_len:]
            if len(key):
                child_key = key.child_key()
        return value, node

    def _split_node(self, child: TreeNode, split_len: int) -> TreeNode:
        """Split ``child`` at ``split_len``: a new parent holds the shared prefix,
        ``child`` keeps the tail."""
        new_node = TreeNode()
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.last_access = child.last_access
        new_node.key = child.key[:split_len]
        new_node.value = list(child.value[:split_len])
        new_node.children = {child.key.units[split_len]: child}

        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = list(child.value[split_len:])

        new_node.parent.children[new_node.key.child_key()] = new_node
        return new_node

    def _insert_helper(self, node: TreeNode, key: RadixKey, value: list):
        access = self._tick()
        node.last_access = access
        if len(key) == 0:
            return 0, node

        child_key = key.child_key()
        total_prefix = 0
        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            node.last_access = access
            prefix_len = node.key.match(key)
            total_prefix += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]
            if prefix_len < len(node.key):
                node = self._split_node(node, prefix_len)
            if len(key):
                child_key = key.child_key()

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = list(value)
            new_node.last_access = access
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)
            node = new_node
        return total_prefix, node

    def _delete_leaf(self, node: TreeNode) -> None:
        del node.parent.children[node.key.child_key()]
        self.evictable_size_ -= len(node.key)
        self.evictable_leaves.discard(node)
        node.value = None  # mark evicted
        self._update_leaf_status(node.parent)

    def _update_leaf_status(self, node: TreeNode) -> None:
        """A node is an evictable leaf iff it is live, unpinned, and has no live child."""
        if node is self.root_node or node.evicted or node.lock_ref > 0:
            self.evictable_leaves.discard(node)
            return
        for child in node.children.values():
            if not child.evicted:
                self.evictable_leaves.discard(node)
                return
        self.evictable_leaves.add(node)


# ---------------------------------------------------------------------------

class _FakeAllocator:
    def __init__(self) -> None:
        self.freed: list = []

    def free(self, value_list: list) -> None:
        self.freed.extend(value_list)


def _self_check() -> None:
    # ----- SGLang's own __main__ demo, as asserts -----
    # Insert [1,2,3], [1,2,4,5], [1,2,4,5,6,7], [8,9,10,11,12]; value==unit for clarity.
    t = VtlRadixCache.create_simulated(_FakeAllocator())

    def ins(seq):
        return t.insert(list(seq), list(seq))

    assert ins([1, 2, 3]) == 0
    assert ins([1, 2, 3]) == 3            # re-insert: full prefix already present
    assert ins([1, 2, 4, 5]) == 2         # shares [1,2], splits the [1,2,3] node
    assert ins([1, 2, 4, 5, 6, 7]) == 4   # shares [1,2,4,5]
    assert ins([8, 9, 10, 11, 12]) == 0   # disjoint

    # match: [1,2,3,13,14] -> longest cached prefix is [1,2,3]
    vals, node = t.match_prefix([1, 2, 3, 13, 14])
    assert vals == [1, 2, 3], vals
    assert node.key.units == [3], node.key.units  # terminal node is the [3] tail
    # deeper match on a shared branch
    assert t.match_prefix_len([1, 2, 4, 5, 6, 7, 99]) == 6
    # miss at root
    assert t.match_prefix_len([100, 101]) == 0
    # total unique blocks = 3 + 2 + 2 + 5 = 12
    assert t.total_size() == 12, t.total_size()

    # ----- split correctness: values stay aligned to units -----
    t2 = VtlRadixCache.create_simulated()
    t2.insert([10, 20, 30], ["a", "b", "c"])
    t2.insert([10, 20, 40], ["a", "b", "d"])  # splits after [10,20]
    assert t2.match_prefix([10, 20, 30])[0] == ["a", "b", "c"]
    assert t2.match_prefix([10, 20, 40])[0] == ["a", "b", "d"]

    # ----- LRU evict order: least-recently-accessed leaf goes first -----
    t3 = VtlRadixCache.create_simulated(_FakeAllocator())
    t3.insert([1, 2], [1, 2])
    t3.insert([3, 4], [3, 4])
    t3.match_prefix([1, 2])            # touch [1,2] -> newer than [3,4]
    freed = t3.evict(1)                # free ~1 block: the older leaf [3,4]
    assert freed == 2, freed           # one leaf = one segment of 2 blocks
    assert t3.match_prefix_len([3, 4]) == 0
    assert t3.match_prefix_len([1, 2]) == 2

    # ----- lock_ref pins a prefix out of the evictable set -----
    t4 = VtlRadixCache.create_simulated(_FakeAllocator())
    t4.insert([5, 6], [5, 6])
    _, leaf = t4.match_prefix([5, 6])
    t4.inc_lock_ref(leaf)
    assert leaf not in t4.evictable_leaves
    assert t4.protected_size_ == 2 and t4.evictable_size_ == 0
    assert t4.evict(10) == 0           # nothing evictable while pinned
    t4.dec_lock_ref(leaf)
    assert leaf in t4.evictable_leaves
    assert t4.evict(10) == 2           # now it frees

    print("radix_tree self-check ok")


if __name__ == "__main__":
    _self_check()
