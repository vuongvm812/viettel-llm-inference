"""Token-granular KV-slot allocator -- port of SGLang's ``TokenToKVPoolAllocator``
(``sglang/python/sglang/srt/mem_cache/allocator/token.py``).

SGLang shares KV at token granularity: each cached token occupies one slot in the KV
pool, and the radix tree stores lists of these slot indices. vLLM has no token-granular
allocator -- its ``BlockPool`` hands out 16-token blocks. This is the allocator that makes
the SGLang RadixAttention flow possible; on-box it is backed by the real KV pool, off-box
(here) by a plain int free-list so the whole flow is testable without torch or a GPU.

Slot 0 is reserved (padding for cuda-graph padded batches write dummy KV there), matching
SGLang's ``free_pages = arange(1, size+1)``.
"""

from __future__ import annotations


class TokenToKVPoolAllocator:
    def __init__(self, size: int):
        self.size = size
        self.clear()

    def clear(self) -> None:
        # Slot 0 reserved; real slots are 1..size (SGLang: free_pages = arange(1, size+1)).
        self.free_pages: list[int] = list(range(1, self.size + 1))

    def available_size(self) -> int:
        return len(self.free_pages)

    def alloc(self, need_size: int) -> list[int] | None:
        """Return ``need_size`` free slots, or ``None`` if the pool can't satisfy it."""
        if need_size > len(self.free_pages):
            return None
        sel = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return sel

    def free(self, slots) -> None:
        # ponytail: append-to-list free (SGLang cats a tensor). No sort/dedup -- callers
        # must never double-free, same contract as SGLang's need_sort=False path.
        self.free_pages.extend(slots)


def _self_check() -> None:
    a = TokenToKVPoolAllocator(size=8)
    assert a.available_size() == 8
    s = a.alloc(3)
    assert s == [1, 2, 3] and a.available_size() == 5
    assert a.alloc(100) is None  # over-request returns None, leaves pool intact
    assert a.available_size() == 5
    a.free(s)
    assert a.available_size() == 8
    a.clear()
    assert a.available_size() == 8 and a.free_pages[0] == 1  # slot 0 stays reserved
    print("token_allocator self-check ok")


if __name__ == "__main__":
    _self_check()
