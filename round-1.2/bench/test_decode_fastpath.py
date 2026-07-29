"""Correctness for the pure-decode fast path (vtl/patches/decode_fastpath.py).

Three tiers, because the three failure modes are independent:

  * the PREDICATE (``decode_key`` / ``has_new_blocks``) is pure Python and runs anywhere --
    a false positive here is a wrong answer, so every disqualifier gets its own case;
  * the MAMBA WRITE is checked on GPU against vLLM's own
    ``mamba_get_block_table_tensor`` + the ``_update_metadata_for_cudagraph_capture`` copy,
    which is the code it replaces, bit for bit;
  * the FA3 WRITE has no cheap oracle -- ``get_scheduler_metadata`` IS the reference -- so it
    is covered by ``VTL_DECODE_FASTPATH_SHADOW=1`` against a live engine, not here.

    pytest bench/test_decode_fastpath.py -q
    python bench/test_decode_fastpath.py     # predicate tier only, no pytest/GPU needed
"""

from __future__ import annotations

import sys

import pytest


def _key_fns():
    try:
        from vtl.patches.decode_fastpath import decode_key, has_new_blocks
    except Exception:  # PYTHONPATH without round-1.2 on it
        pytest.skip("vtl not importable")
    return decode_key, has_new_blocks


class SO:
    """The five SchedulerOutput fields the predicate reads, plus the cached-reqs block ids."""

    def __init__(self, **kw):
        self.scheduled_new_reqs = []
        self.finished_req_ids = set()
        self.preempted_req_ids = set()
        self.scheduled_spec_decode_tokens = {}
        self.scheduled_encoder_inputs = {}
        self.new_block_ids_to_zero = None
        self.has_structured_output_requests = False
        self.num_scheduled_tokens = {"a": 1, "b": 1, "c": 1}
        self.scheduled_cached_reqs = type("C", (), {"new_block_ids": [None, None, None]})()
        self.__dict__.update(kw)


DISQUALIFIERS = [
    {"scheduled_new_reqs": [object()]},
    {"finished_req_ids": {"a"}},
    {"preempted_req_ids": {"b"}},
    {"scheduled_spec_decode_tokens": {"a": [11]}},
    {"scheduled_encoder_inputs": {"a": [0]}},
    {"new_block_ids_to_zero": [3, 4]},
    {"has_structured_output_requests": True},
    {"num_scheduled_tokens": {"a": 1, "b": 2, "c": 1}},  # a prefill chunk in the batch
    {"num_scheduled_tokens": {}},
]


def test_predicate():
    decode_key, has_new_blocks = _key_fns()
    assert decode_key(SO()) == ("a", "b", "c")
    for kw in DISQUALIFIERS:
        assert decode_key(SO(**kw)) is None, kw

    # Request ORDER is the cache key, because it becomes idx_mapping/query_start_loc.
    assert decode_key(SO(num_scheduled_tokens={"c": 1, "a": 1, "b": 1})) == ("c", "a", "b")
    # A dropped request keeps 1 token each but must still miss.
    assert decode_key(SO(num_scheduled_tokens={"a": 1, "b": 1})) != decode_key(SO())

    # Fresh blocks stay on the fast path; they only force a block-table re-gather.
    s = SO()
    s.scheduled_cached_reqs = type("C", (), {"new_block_ids": [None, ([77],), None]})()
    assert decode_key(s) == ("a", "b", "c") and has_new_blocks(s)
    assert not has_new_blocks(SO())
    s = SO()
    s.scheduled_cached_reqs = type("C", (), {"new_block_ids": [([],), ([],), ([],)]})()
    assert not has_new_blocks(s)


@pytest.mark.parametrize("num_reqs", [1, 2, 8, 17])
@pytest.mark.parametrize("block_size", [16, 64])
def test_mamba_state_indices_write_matches_stock(num_reqs, block_size):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    try:
        from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
    except Exception as exc:
        pytest.skip(f"vllm not importable: {exc}")
    from vtl.patches.decode_fastpath import _mamba_write

    dev = torch.device("cuda")
    max_blocks = 40
    g = torch.Generator(device="cpu").manual_seed(num_reqs * 100 + block_size)
    block_table = torch.randint(
        0, 9999, (num_reqs, max_blocks), dtype=torch.int32, generator=g
    ).to(dev)
    # Row 0 is a cudagraph padding row (seq_len 0) -- the clamp(min=0) case.
    seq_lens = torch.randint(
        1, max_blocks * block_size, (num_reqs,), dtype=torch.int32, generator=g
    ).to(dev)
    seq_lens[0] = 0

    spec = type("Spec", (), {"block_size": block_size, "num_speculative_blocks": 0})()
    want = mamba_get_block_table_tensor(block_table, seq_lens, spec, "align")

    class B:
        state_indices_tensor_d = torch.full((64, 1), -1, dtype=torch.int32, device=dev)

    scratch = torch.zeros((64, 1), dtype=torch.int64, device=dev)
    _mamba_write(B, block_table, seq_lens, block_size, scratch)

    got = B.state_indices_tensor_d[:num_reqs]
    assert torch.equal(got, want), (got, want)
    # Rows past the batch must be untouched by the fast write.
    assert torch.all(B.state_indices_tensor_d[num_reqs:] == -1)
    # And seq_lens must not have been mutated in place by the arithmetic.
    assert int(seq_lens[0]) == 0


if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
    test_predicate()
    print("decode_fastpath predicate self-check ok")
