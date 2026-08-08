"""Standalone self-check for the V2 greedy-sampler fast-path predicate.

Tests ``_vtl_greedy_eligible`` from the forked V2 sampler
(``vllm/v1/worker/gpu/sample/sampler.py``). Runs inside the container (torch +
vllm present); skips gracefully on a dev box without them. No pytest -- one file,
assert-based, matching ``vtl/patches/greedy_sampler.py``'s convention.

    python3 round-2/bench/test_v2_greedy_fastpath.py
"""

import sys

import numpy as np

try:
    from vllm.v1.worker.gpu.sample.sampler import NO_LOGPROBS, _vtl_greedy_eligible
except Exception as exc:  # torch / vllm not importable on this host
    print(f"SKIP (vllm not importable: {exc})")
    sys.exit(0)


def _clean(n=4):
    """A batch that IS a plain greedy step (all defaults, temp 0)."""
    return dict(
        num_draft_tokens=0,
        temperature_np=np.zeros(n, dtype=np.float32),
        max_num_logprobs=NO_LOGPROBS,
        max_per_req_token_ids=0,
        use_penalty_np=np.zeros(n, dtype=bool),
        use_logit_bias_np=np.zeros(n, dtype=bool),
        num_bad_words_np=np.zeros(n, dtype=np.int32),
    )


def main() -> None:
    # Plain greedy batch -> eligible.
    assert _vtl_greedy_eligible(**_clean()) is True

    # Each disqualifier flips it to False, one at a time.
    b = _clean(); b["temperature_np"][2] = 0.7          # one sampling req
    assert _vtl_greedy_eligible(**b) is False
    b = _clean(); b["num_draft_tokens"] = 2             # spec-decode rows
    assert _vtl_greedy_eligible(**b) is False
    b = _clean(); b["max_num_logprobs"] = 5             # logprobs requested
    assert _vtl_greedy_eligible(**b) is False
    b = _clean(); b["max_per_req_token_ids"] = 1        # prompt-logprob token ids
    assert _vtl_greedy_eligible(**b) is False
    b = _clean(); b["use_penalty_np"][0] = True         # rep/freq/presence penalty
    assert _vtl_greedy_eligible(**b) is False
    b = _clean(); b["use_logit_bias_np"][1] = True      # bias / min-tokens / allowed-ids
    assert _vtl_greedy_eligible(**b) is False
    b = _clean(); b["num_bad_words_np"][3] = 2          # bad words
    assert _vtl_greedy_eligible(**b) is False

    # Temperature must be ALL zero, not just any-zero.
    b = _clean(); b["temperature_np"] = np.array([0.0, 0.0, 1.0, 0.0], np.float32)
    assert _vtl_greedy_eligible(**b) is False

    print("v2 greedy fast-path predicate self-check ok")


if __name__ == "__main__":
    main()
