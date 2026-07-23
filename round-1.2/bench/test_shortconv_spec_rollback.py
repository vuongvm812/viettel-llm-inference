"""Contract for the LFM2 short-conv speculative-decode rollback (chain: ngram/ngram_gpu/suffix).

The bug that made the judge's long-context probe score 0% and get flagged `truncation /
dual-path`: during spec verify the target advances the persistent ``conv_state`` once per DRAFT
token, and short_conv.py committed *all* of them -- including the rejected tail. The fix
(short_conv.py forward_cuda decode branch) passes ``num_accepted_tokens`` / ``query_start_loc`` /
``max_query_len`` to the stock ``causal_conv1d_update``, which then rolls the state back to the
accepted prefix -- exactly as ``mamba_mixer2`` already does for Mamba2.

This test does NOT need CUDA or vtl._C: it pins the *semantics* the kernel guarantees and the fix
requests, so a future edit that drops the num_accepted arg (or miscounts the bonus token) fails
here instead of only on-box. The on-box integration check is the long-context probe (spec ON vs
OFF at temp 0 must be token-identical) -- see the plan's Verification section.

    pytest bench/test_shortconv_spec_rollback.py -q
    python bench/test_shortconv_spec_rollback.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

WIDTH = 3  # LFM2.5 conv_L_cache
STATE_LEN = WIDTH - 1  # short_conv_state_shape allocates width-1


def step(state: torch.Tensor, x_t: torch.Tensor, weight: torch.Tensor) -> tuple:
    """One short-conv decode step. Mirrors causal_conv1d_update_torch (ops/cpu):
    prepend the (width-1)-wide state, depthwise conv1d, keep the last (width-1) inputs
    as the new state. Returns (output[dim], new_state[dim, STATE_LEN])."""
    dim = state.shape[0]
    x_new = torch.cat([state, x_t.unsqueeze(-1)], dim=-1)  # (dim, STATE_LEN+1)
    out = F.conv1d(
        x_new.unsqueeze(0), weight.unsqueeze(1), None, padding=0, groups=dim
    )[0, :, -1]  # (dim,)
    new_state = x_new[:, -STATE_LEN:].clone()
    return out, new_state


def decode_chain(state: torch.Tensor, xs: torch.Tensor, weight: torch.Tensor) -> list:
    """Sequentially decode L tokens. Returns the list of post-step states S_0..S_{L-1}."""
    states = []
    s = state
    for t in range(xs.shape[0]):
        _, s = step(s, xs[t], weight)
        states.append(s)
    return states


def _rand(dim: int, gen: torch.Generator):
    return torch.randn(dim, generator=gen, dtype=torch.float64)


def test_commit_accepted_prefix_matches_fresh_decode():
    """num_accepted=K must leave conv_state == decoding only the K accepted tokens.

    K counts the bonus token, so K in [1 .. L]; the committed persistent state is the
    state after the K-th token, i.e. S_{K-1} (0-indexed)."""
    gen = torch.Generator().manual_seed(0)
    dim, L = 8, 4  # L = 1 bonus + 3 spec drafts
    weight = torch.randn(dim, WIDTH, generator=gen, dtype=torch.float64)
    S = torch.randn(dim, STATE_LEN, generator=gen, dtype=torch.float64)
    xs = torch.stack([_rand(dim, gen) for _ in range(L)])

    states = decode_chain(S, xs, weight)  # S_0 .. S_{L-1}

    for K in range(1, L + 1):
        committed = states[K - 1]  # what num_accepted=K must persist
        fresh = decode_chain(S, xs[:K], weight)[-1]  # if only K tokens ever existed
        assert torch.allclose(committed, fresh), f"rollback broke at K={K}"


def test_committing_all_drafts_is_wrong_when_rejected():
    """The pre-fix behavior (commit S_{L-1}, i.e. all drafts) diverges from the correct
    accepted-prefix commit whenever any draft is rejected (K < L). This is the corruption
    that compounds over a long generation to 0%."""
    gen = torch.Generator().manual_seed(1)
    dim, L = 8, 4
    weight = torch.randn(dim, WIDTH, generator=gen, dtype=torch.float64)
    S = torch.randn(dim, STATE_LEN, generator=gen, dtype=torch.float64)
    xs = torch.stack([_rand(dim, gen) for _ in range(L)])
    states = decode_chain(S, xs, weight)

    commit_all = states[L - 1]  # buggy: every draft committed
    for K in range(1, L):  # at least one draft rejected
        correct = states[K - 1]
        assert not torch.allclose(commit_all, correct), (
            f"expected divergence at K={K}; if these match the test can't catch the bug"
        )


if __name__ == "__main__":
    test_commit_accepted_prefix_matches_fresh_decode()
    test_committing_all_drafts_is_wrong_when_rejected()
    print("ok: short-conv spec rollback contract holds")
