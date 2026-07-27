"""Standalone self-check for the V2 runner's shared constant input buffers.

Tests ``cu_num_logits_arange`` / ``zeros_max_num_reqs`` from the forked V2
``InputBuffers`` (``vllm/v1/worker/gpu/input_batch.py``). ``prepare_inputs``
slices these instead of re-allocating ``torch.arange`` / ``torch.zeros`` on the
device every decode step, so the contract is: a slice must equal what it
replaced, and nothing may ever write through one. Runs inside the container
(torch + vllm present); skips gracefully on a dev box without them.

Collected by ``make test-kernel`` (pytest, inside the image) and runnable standalone:

    python3 round-1.2/bench/test_v2_input_buffers.py
"""

try:
    import torch

    from vllm.v1.worker.gpu.input_batch import InputBuffers

    _SKIP = ""
except Exception as exc:  # torch / vllm not importable on this host
    _SKIP = f"vllm not importable: {exc}"

MAX_NUM_REQS = 70  # matches --max-num-seqs in docker-compose.yaml


def test_constant_buffers_match_what_they_replace() -> None:
    if _SKIP:
        import pytest

        pytest.skip(_SKIP)

    # CPU is enough: this checks values and the read-only contract, not placement.
    bufs = InputBuffers(
        max_num_reqs=MAX_NUM_REQS, max_num_tokens=8192, device=torch.device("cpu")
    )

    # int32 is load-bearing: the Triton consumers tl.load these as int32 pointers.
    assert bufs.cu_num_logits_arange.dtype == torch.int32
    assert bufs.zeros_max_num_reqs.dtype == torch.int32

    # Every batch size the scheduler can produce, including the num_reqs == max edge.
    for num_reqs in range(1, MAX_NUM_REQS + 1):
        cu_num_logits = bufs.cu_num_logits_arange[: num_reqs + 1]
        expanded_local_pos = bufs.zeros_max_num_reqs[:num_reqs]
        assert torch.equal(
            cu_num_logits, torch.arange(num_reqs + 1, dtype=torch.int32)
        ), num_reqs
        assert torch.equal(
            expanded_local_pos, torch.zeros(num_reqs, dtype=torch.int32)
        ), num_reqs

    # The read-only contract. Async scheduling keeps two batches in flight sharing
    # these buffers, so a consumer that writes through a slice corrupts the other
    # batch. Re-checking the whole buffer after all that slicing is what catches it.
    assert torch.equal(
        bufs.cu_num_logits_arange, torch.arange(MAX_NUM_REQS + 1, dtype=torch.int32)
    )
    assert torch.equal(
        bufs.zeros_max_num_reqs, torch.zeros(MAX_NUM_REQS, dtype=torch.int32)
    )

    print("v2 input-buffer constants self-check ok")


if __name__ == "__main__":
    if _SKIP:
        print(f"SKIP ({_SKIP})")
    else:
        test_constant_buffers_match_what_they_replace()
