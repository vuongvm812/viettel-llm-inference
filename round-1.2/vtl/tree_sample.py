"""Batched greedy tree acceptance — the logic the forked `RejectionSampler.forward` calls.

Wraps the golden-tested per-request `tree_spec.tree_greedy_accept_indices` over a batch carrier and
emits vLLM's rejection-sample output shape: `[batch, max_spec_len + 1]` filled with a placeholder,
each row = the accepted draft tokens along the best path + the bonus token. For a width-1 (chain)
carrier this reduces to exactly vLLM's stock greedy chain accept, so the fork can safely defer to
stock there and only take this path for branching trees.

Greedy only (temperature==0) — matches the round-2 workload (temperature=0). Random/typical tree
sampling (SGLang `tree_speculative_sampling_target_only`) is out of scope until a non-greedy trace
appears; `assert`ed below so it fails loudly rather than silently mis-sampling.

Self-check (needs torch, no vLLM/GPU):  python vtl/tree_sample.py
"""
from __future__ import annotations

try:  # package import vs bare (self-check) — see tree_runtime
    from vtl.tree_spec import tree_greedy_accept_indices
except ImportError:
    from tree_spec import tree_greedy_accept_indices

PLACEHOLDER_TOKEN_ID = -1  # vLLM's rejection_sampler.PLACEHOLDER_TOKEN_ID


def _root_children(tree_mask_row_sums) -> list[int]:
    """Depth-0 nodes (only ancestor is self) = the first-level draft nodes, in index order."""
    return [i for i, s in enumerate(tree_mask_row_sums) if s == 1]


def tree_greedy_accept_batch(carrier: dict, target_pred, root_pred):
    """Greedy tree accept over the whole batch.

    carrier      : tree_runtime.build_carrier output (device tensors).
    target_pred  : int64 [b, max_nodes] — target model's greedy argmax AFTER each draft node
                   (target_pred[r, i] = token the target predicts to follow node i).
    root_pred    : int64 [b] — target's greedy argmax at the anchor (predicts the first draft token).
    Returns (out_tokens int64 [b, max_nodes+1], accept_lens int64 [b]) where accept_lens counts the
    accepted draft tokens + 1 bonus (== vLLM/SGLang convention).
    """
    import torch

    device = carrier["draft_tokens"].device
    b, max_nodes = carrier["draft_tokens"].shape
    draft = carrier["draft_tokens"].tolist()
    nxt = carrier["retrieve_next_token"].tolist()
    sib = carrier["retrieve_next_sibling"].tolist()
    num_nodes = carrier["num_nodes"].tolist()
    mask_sums = carrier["tree_mask"].sum(dim=-1).tolist()  # [b, max_nodes]
    tpred = target_pred.tolist()
    rpred = root_pred.tolist()

    out = torch.full((b, max_nodes + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int64, device=device)
    accept_lens = torch.ones(b, dtype=torch.int64, device=device)
    for r in range(b):
        m = num_nodes[r]
        if m == 0:
            out[r, 0] = rpred[r]  # nothing drafted -> just the target's own next token (bonus)
            continue
        roots = _root_children(mask_sums[r][:m])
        accepted, bonus = tree_greedy_accept_indices(
            draft[r][:m], nxt[r][:m], sib[r][:m], tpred[r][:m], rpred[r], roots,
        )
        for j, node in enumerate(accepted):
            out[r, j] = draft[r][node]
        out[r, len(accepted)] = bonus
        accept_lens[r] = len(accepted) + 1
    return out, accept_lens


def _selfcheck():
    import torch

    try:
        from vtl.tree_runtime import build_carrier
    except ImportError:
        from tree_runtime import build_carrier

    # Tree: 0->{1,2}; 1->{3}; 3->{4}; 2->{5}  (same shape as tree_spec._selfcheck)
    node_tokens = [10, 11, 12, 13, 14, 15]
    parent = [-1, 0, 0, 1, 3, 2]
    carrier = build_carrier([(node_tokens, parent)], seq_lens=[100], max_nodes=6)

    def run(root_pred, target_pred):
        tp = torch.tensor([target_pred], dtype=torch.int64)
        rp = torch.tensor([root_pred], dtype=torch.int64)
        out, alen = tree_greedy_accept_batch(carrier, tp, rp)
        n = int(alen[0])
        return out[0, :n].tolist()

    # accept 0->1->3 then bonus (target diverges): emitted = [10,11,13, bonus]
    assert run(10, [11, 13, 0, 99, 0, 0]) == [10, 11, 13, 99], run(10, [11, 13, 0, 99, 0, 0])
    # accept 0->2->5: emitted = [10,12,15, bonus]
    assert run(10, [12, 0, 15, 0, 0, 77]) == [10, 12, 15, 77], run(10, [12, 0, 15, 0, 0, 77])
    # no root match -> accept nothing, bonus = root_pred
    assert run(77, [0, 0, 0, 0, 0, 0]) == [77], run(77, [0, 0, 0, 0, 0, 0])

    # width-1 chain reduces to stock chain accept: full accept + bonus
    chain = ([5, 6, 7], [-1, 0, 1])
    cc = build_carrier([chain], seq_lens=[3], max_nodes=3)
    out, alen = tree_greedy_accept_batch(cc, torch.tensor([[6, 7, 42]]), torch.tensor([5]))
    assert out[0, : int(alen[0])].tolist() == [5, 6, 7, 42], out[0].tolist()

    # empty tree (nothing drafted) -> just the bonus
    empty = build_carrier([([], [])], seq_lens=[9], max_nodes=3)
    out, alen = tree_greedy_accept_batch(empty, torch.zeros(1, 3, dtype=torch.int64), torch.tensor([88]))
    assert int(alen[0]) == 1 and out[0, 0].item() == 88
    print("PASS: tree_sample — batched greedy tree accept == oracle, chain reduction, empty tree")


if __name__ == "__main__":
    _selfcheck()
