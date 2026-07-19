"""Request -> token-slot table -- port of SGLang's ``ReqToTokenPool``
(``sglang/python/sglang/srt/mem_cache/memory_pool.py:242``).

``req_to_token[req_slot, position]`` = the KV slot holding that request's token at that
position. This is the indirection table SGLang's attention backend reads to gather each
request's (possibly shared) KV -- the thing vLLM replaces with a block table. The radix
flow reads a request's slots out of here (``read``) to insert into the tree, and writes the
canonical shared slots back (``write``) after a prefix match so the request points at the
reused KV instead of its own duplicates.

Row 0 is a reserved padding row (SGLang: cuda-graph padded batches default req slots to 0).
Off-box this is a list-of-rows; on-box it becomes an int32 ``[size+1, max_context_len]`` tensor.
"""

from __future__ import annotations


class ReqToTokenPool:
    def __init__(self, size: int, max_context_len: int):
        self.size = size
        self.max_context_len = max_context_len
        self._alloc_size = size + 1  # +1 padding row at index 0
        self.rows: list[list[int]] = [
            [0] * max_context_len for _ in range(self._alloc_size)
        ]
        self.free_slots: list[int] = list(range(1, self._alloc_size))

    def available_size(self) -> int:
        return len(self.free_slots)

    def alloc(self, num_reqs: int) -> list[int] | None:
        """Reserve ``num_reqs`` request slots. Returns their indices or ``None``."""
        if num_reqs > len(self.free_slots):
            return None
        sel = self.free_slots[:num_reqs]
        self.free_slots = self.free_slots[num_reqs:]
        return sel

    def free(self, req_slot: int) -> None:
        self.free_slots.append(req_slot)

    def write(self, req_slot: int, start: int, values) -> None:
        """Set ``req_to_token[req_slot, start:start+len(values)] = values``."""
        vals = list(values)
        self.rows[req_slot][start : start + len(vals)] = vals

    def read(self, req_slot: int, length: int) -> list[int]:
        """The request's KV slots for positions ``[0, length)``."""
        return self.rows[req_slot][:length]

    def clear(self) -> None:
        for row in self.rows:
            for i in range(self.max_context_len):
                row[i] = 0
        self.free_slots = list(range(1, self._alloc_size))


def _self_check() -> None:
    p = ReqToTokenPool(size=4, max_context_len=16)
    assert p.available_size() == 4
    slots = p.alloc(2)
    assert slots == [1, 2] and p.available_size() == 2
    p.write(slots[0], 0, [10, 11, 12])
    assert p.read(slots[0], 3) == [10, 11, 12]
    # partial overwrite from an offset (the cache_unfinished write-back pattern)
    p.write(slots[0], 1, [99])
    assert p.read(slots[0], 3) == [10, 99, 12]
    p.free(slots[0])
    assert p.available_size() == 3
    assert p.alloc(100) is None  # over-request
    print("req_to_token_pool self-check ok")


if __name__ == "__main__":
    _self_check()
