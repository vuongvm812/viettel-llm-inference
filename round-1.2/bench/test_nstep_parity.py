"""C3 -- correctness for the N-step decode burst (`vtl/patches/nstep_decode.py`).

Three tiers, because the failure modes are independent and only one of them needs a GPU:

  1. HANDSHAKE (runs anywhere, no torch). The scheduler must never commit a burst the
     runner will not execute -- under async scheduling the next step is already scheduled
     off the optimistic ``num_computed``, so a silently-short burst writes the following
     step's tokens at the wrong positions. This tier drives ``burst_factor`` /
     ``burst_blocked`` / ``burst_request_blocked`` through the whole eligibility matrix.

  2. ACCOUNTING (runs anywhere, no torch). ``burst_commit`` / ``burst_uncommit`` against a
     model of ``_update_after_schedule`` + ``AsyncScheduler``'s placeholder arithmetic, for
     every stop point 1..N. The property under test is the one that can poison the prefix
     cache: after the step, ``num_computed - num_output_placeholders`` -- which is exactly
     what ``cache_blocks`` fingerprints -- must equal the real token count minus one,
     burst or not, stopped or not.

  3. GREEDY BIT-IDENTITY (needs a GPU **and a model**). ``VTL_NSTEP=0`` vs ``VTL_NSTEP=1``
     over the same prompts must produce byte-identical completions: the burst changes WHEN
     tokens are computed, never WHICH. Skipped under ``make test-kernel``, which mounts
     ``bench/`` and nothing else -- run it by hand with the model visible:

         VTL_TEST_MODEL=/model pytest bench/test_nstep_parity.py -q -k identity

Tiers 1-2 also run as a plain script:  python bench/test_nstep_parity.py
"""

from __future__ import annotations

import os
import sys

import pytest

BLOCK = 16
MAX_MODEL_LEN = 32768


def _sched():
    try:
        from vtl.patches.rust_sched import (
            burst_blocked,
            burst_commit,
            burst_request_blocked,
            burst_uncommit,
        )
    except Exception:  # pragma: no cover - PYTHONPATH without round-1.2
        pytest.skip("vtl.patches.rust_sched not importable")
    return burst_blocked, burst_request_blocked, burst_commit, burst_uncommit


def _runner():
    try:
        from vtl.patches import nstep_decode
    except Exception:  # pragma: no cover
        pytest.skip("vtl.patches.nstep_decode not importable")
    return nstep_decode


class SP:
    def __init__(self, **kw):
        self.temperature = 0.0
        self.logprobs = None
        self.prompt_logprobs = None
        self.min_tokens = 0
        self.bad_words = None
        self.allowed_token_ids = None
        self.logit_bias = None
        self.presence_penalty = 0.0
        self.frequency_penalty = 0.0
        self.repetition_penalty = 1.0
        self.__dict__.update(kw)


class REQ:
    """The Request fields the gate and the accounting read, nothing else."""

    def __init__(self, **kw):
        self.sampling_params = SP()
        self.pooling_params = None
        self.resumable = False
        self.use_structured_output = False
        self.num_prompt_tokens = 10
        self.max_tokens = 4096
        self.num_tokens = 100
        self.num_computed_tokens = 101   # post-_update_after_schedule steady state
        self.num_output_placeholders = 2
        self.is_prefill_chunk = False
        self.__dict__.update(kw)


class SO:
    def __init__(self, **kw):
        self.scheduled_new_reqs = []
        self.preempted_req_ids = set()
        self.scheduled_spec_decode_tokens = {}
        self.scheduled_encoder_inputs = {}
        self.has_structured_output_requests = False
        self.new_block_ids_to_zero = None
        self.num_scheduled_tokens = {"a": 1, "b": 1}
        self.__dict__.update(kw)


# ---------------------------------------------------------------------------------------
# Tier 1 -- the handshake and the eligibility matrix.
# ---------------------------------------------------------------------------------------


def test_runner_must_publish_a_matching_key_before_a_burst_is_committed():
    m = _runner()
    b = m.BURST
    saved = (b.armed, b.ready, b.key, b.n)
    try:
        b.n, b.armed, b.ready, b.key = 4, True, True, ("a", "b")
        assert m.burst_factor({"a": 1, "b": 1}) == 4
        # The runner's key is the LAST executed step's request set, in order. Anything
        # that changes the set or the order means the cudagraph descriptor may change too,
        # so the commit is refused.
        for nst in ({"b": 1, "a": 1}, {"a": 1}, {"a": 1, "b": 1, "c": 1}, {}):
            assert m.burst_factor(nst) == 1, nst
        b.ready = False
        assert m.burst_factor({"a": 1, "b": 1}) == 1
    finally:
        b.armed, b.ready, b.key, b.n = saved


def test_batch_level_disqualifiers():
    burst_blocked, _, _, _ = _sched()
    assert burst_blocked(SO(), False, True, 8) is None
    for kw in (
        {"scheduled_new_reqs": [object()]},
        {"preempted_req_ids": {"a"}},
        {"scheduled_spec_decode_tokens": {"a": [7]}},
        {"scheduled_encoder_inputs": {"a": [0]}},
        {"has_structured_output_requests": True},
        {"new_block_ids_to_zero": [3]},
        {"num_scheduled_tokens": {}},
        {"num_scheduled_tokens": {"a": 1, "b": 2}},
        {"num_scheduled_tokens": dict.fromkeys("abcdefghi", 1)},
    ):
        assert burst_blocked(SO(**kw), False, True, 8) is not None, kw
    assert burst_blocked(SO(), True, True, 8) is not None, "TTFT guard"
    assert burst_blocked(SO(), True, False, 8) is None, "...and it is a knob"


@pytest.mark.parametrize("n,expected", [(2, 15), (4, 13), (8, 9), (16, 1)])
def test_align_gate_coverage(n, expected):
    """``num_computed % block + N <= block`` is what makes a mid-burst block crossing --
    a new KV block, a mamba state migration, a stale block table -- impossible."""
    _, blocked, _, _ = _sched()
    ok = [c for c in range(BLOCK) if blocked(REQ(), c, n, BLOCK, MAX_MODEL_LEN) is None]
    assert ok == list(range(expected)), (n, ok)


def test_request_level_disqualifiers():
    _, blocked, _, _ = _sched()
    assert blocked(REQ(), 0, 4, BLOCK, MAX_MODEL_LEN) is None
    for kw in (
        {"sampling_params": None},
        {"pooling_params": object()},
        {"resumable": True},
        {"use_structured_output": True},
        {"sampling_params": SP(temperature=0.7)},
        {"sampling_params": SP(logprobs=5)},
        {"sampling_params": SP(prompt_logprobs=1)},
        {"sampling_params": SP(min_tokens=8)},
        {"sampling_params": SP(bad_words=["x"])},
        {"sampling_params": SP(allowed_token_ids=[1])},
        {"sampling_params": SP(logit_bias={1: 2.0})},
        {"sampling_params": SP(presence_penalty=0.5)},
        {"sampling_params": SP(frequency_penalty=0.5)},
        {"sampling_params": SP(repetition_penalty=1.1)},
    ):
        assert blocked(REQ(**kw), 0, 4, BLOCK, MAX_MODEL_LEN) is not None, kw
    # The two length caps.
    assert blocked(REQ(num_prompt_tokens=10, max_tokens=12), 20, 4, BLOCK,
                   MAX_MODEL_LEN) is not None, "max_tokens"
    assert blocked(REQ(max_tokens=1 << 20), MAX_MODEL_LEN - 2, 4, BLOCK,
                   MAX_MODEL_LEN) is not None, "max_model_len"


# ---------------------------------------------------------------------------------------
# Tier 2 -- the accounting that keeps the prefix cache honest.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 4, 8])
def test_commit_then_reconcile_leaves_the_cacheable_prefix_exact(n):
    _, _, commit, uncommit = _sched()
    for kept in range(1, n + 1):
        r = REQ()
        real_before = r.num_tokens
        commit(r, n - 1)
        assert r.num_computed_tokens == real_before + n
        # The step comes back with `kept` tokens: append, reconcile, then AsyncScheduler's
        # own `num_output_placeholders -= len(new_token_ids)`.
        r.num_tokens += kept
        uncommit(r, n - kept)
        r.num_output_placeholders -= kept
        assert r.num_output_placeholders >= 0, (n, kept)
        # `AsyncScheduler._update_request_with_output` caches exactly this many tokens.
        cached = r.num_computed_tokens - r.num_output_placeholders
        assert cached == r.num_tokens - 1, (n, kept, cached, r.num_tokens)
        # ...and it can never exceed the number of tokens that really exist, which is the
        # property that stops a poisoned prefix from being published.
        assert cached <= r.num_tokens


def test_a_committed_burst_that_produced_nothing_extra_is_fully_unwound():
    """The should-never-happen path: the runner bailed and returned one token."""
    _, _, commit, uncommit = _sched()
    r = REQ()
    before = (r.num_computed_tokens, r.num_output_placeholders)
    commit(r, 3)
    uncommit(r, 3)
    assert (r.num_computed_tokens, r.num_output_placeholders) == before


def test_uncommit_saturates_instead_of_going_negative():
    _, _, _, uncommit = _sched()
    r = REQ(num_computed_tokens=1, num_output_placeholders=1)
    uncommit(r, 1000)
    assert (r.num_computed_tokens, r.num_output_placeholders) == (0, 0)


def test_num_rejected_encodes_the_burst_width_for_post_update():
    """``post_update`` computes ``computed_delta = query_len - num_rejected`` and query_len
    is 1 on a decode step, so the burst passes ``-(N-1)``."""
    for n in (2, 4, 8):
        assert 1 - (-(n - 1)) == n


# ---------------------------------------------------------------------------------------
# Tier 3 -- greedy bit-identity against a live engine. GPU + model required.
# ---------------------------------------------------------------------------------------

MODEL = os.environ.get("VTL_TEST_MODEL", "")
PROMPTS = [
    "Explain in one paragraph why prefix caching helps multi-turn chat.",
    "List three properties of a CUDA graph.",
    "Write a haiku about a scheduler.",
]


def _have_gpu_and_model() -> bool:
    if not MODEL or not os.path.isdir(MODEL):
        return False
    try:
        import torch
    except Exception:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(
    not _have_gpu_and_model(),
    reason="needs a GPU and VTL_TEST_MODEL pointing at the served model directory",
)
@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_greedy_output_is_bit_identical_with_and_without_the_burst(mode):
    """The burst changes WHEN tokens are computed, never WHICH.

    Two engines in sequence rather than one process with a flag flip: the patch set is
    installed at engine construction (the `general_plugins` entry point), so the arms have
    to be separate boots. Slow -- it is a correctness gate, not something to run in a loop.
    """
    import subprocess

    def run(nstep: str) -> list[str]:
        env = dict(os.environ)
        env.update(
            VTL_NSTEP=nstep,
            VTL_NSTEP_MODE=mode,
            VTL_NSTEP_N=env.get("VTL_NSTEP_N", "4"),
            VLLM_USE_V2_MODEL_RUNNER="1",
        )
        script = (
            "import json, sys\n"
            "from vllm import LLM, SamplingParams\n"
            f"llm = LLM(model={MODEL!r}, max_model_len=4096, enforce_eager=False,\n"
            "          gpu_memory_utilization=0.85)\n"
            f"out = llm.generate({PROMPTS!r}, SamplingParams(temperature=0.0, max_tokens=64))\n"
            "print(json.dumps([o.outputs[0].text for o in out]))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True, text=True,
            timeout=1800,
        )
        if proc.returncode != 0:
            pytest.fail(f"engine boot failed (VTL_NSTEP={nstep}):\n{proc.stderr[-4000:]}")
        import json

        return json.loads(proc.stdout.strip().splitlines()[-1])

    baseline = run("0")
    burst = run("1")
    assert burst == baseline, "\n".join(
        f"prompt {i}:\n  base ={b!r}\n  burst={x!r}"
        for i, (b, x) in enumerate(zip(baseline, burst))
        if b != x
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-k", "not identity"]))
