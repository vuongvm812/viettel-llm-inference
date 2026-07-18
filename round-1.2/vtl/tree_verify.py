"""vLLM-facing adapter for the tree-verify forks — keeps the .patch diffs to a 3-line divert.

The forked vLLM readers (`RejectionSampler.forward`, `flash_attn`, `ShortConv.forward_cuda`) call
into here so the hardware-coupled index mapping lives in one testable place instead of smeared across
patch files. Everything that is layout-dependent is marked `ON-BOX:` and guarded by a loud assert.

Pure logic (`scatter_node_preds`) is unit-tested off-box; the vLLM-object entry points are thin.

Self-check (needs torch, no vLLM/GPU):  python vtl/tree_verify.py
"""
from __future__ import annotations


def is_chain(carrier) -> bool:
    """Width-1 tree -> stock verify is exact. Chain <=> no node has a next-sibling."""
    sib = carrier.get("retrieve_next_sibling") if carrier is not None else None
    if sib is None:
        return True
    return bool((sib == -1).all().item())


def scatter_node_preds(flat_preds, cu_num_draft_tokens, num_nodes, max_nodes):
    """Flat per-node target predictions -> padded [b, max_nodes].

    vLLM lays target logits/predictions out flat over the batch, segmented by `cu_num_draft_tokens`
    (cumulative draft-token counts). Node i of request r lives at flat index cu[r-1]+i, in the same
    node order the carrier uses (drafter insertion order). This regroups them into [b, max_nodes].
    """
    import torch

    b = len(num_nodes)
    device = flat_preds.device
    out = torch.zeros((b, max_nodes), dtype=torch.int64, device=device)
    cu = cu_num_draft_tokens.tolist() if hasattr(cu_num_draft_tokens, "tolist") else list(cu_num_draft_tokens)
    start = 0
    for r in range(b):
        end = int(cu[r])
        m = int(num_nodes[r])
        # ON-BOX: confirm flat[start:end] is exactly request r's nodes in carrier order. Asserted so a
        # layout mismatch fails here rather than silently accepting the wrong tokens.
        assert end - start == m, f"tree layout mismatch: seg {end - start} != num_nodes {m} (req {r})"
        if m:
            out[r, :m] = flat_preds[start:end]
        start = end
    return out


def accept(metadata, target_logits, bonus_token_ids):
    """Greedy tree acceptance from vLLM's RejectionSampler.forward state.

    Returns output_token_ids [b, max_nodes+1] (vLLM rejection-sample shape) for a BRANCHING tree, or
    None to signal "use stock rejection_sample" (chain / no carrier). Greedy only (temperature==0).
    """
    from vtl.tree_sample import tree_greedy_accept_batch

    carrier = getattr(metadata, "vtl_tree", None)
    if is_chain(carrier):
        return None
    flat_argmax = target_logits.argmax(dim=-1)  # [num_tokens]
    target_pred = scatter_node_preds(
        flat_argmax, metadata.cu_num_draft_tokens, carrier["num_nodes"].tolist(),
        carrier["draft_tokens"].shape[1],
    )
    # root_pred = the target's greedy token at each request's anchor position. In vLLM's chain layout
    # this is the bonus logits' argmax when zero draft tokens are accepted; reuse bonus_token_ids as
    # the per-request anchor prediction. ON-BOX: verify bonus_token_ids[r] is the anchor argmax (it is
    # for greedy chain accept); if a trace uses temperature>0 this whole path is skipped upstream.
    root_pred = bonus_token_ids.view(-1)[: carrier["draft_tokens"].shape[0]]
    out, _ = tree_greedy_accept_batch(carrier, target_pred, root_pred)
    return out


def _selfcheck():
    import torch

    # scatter: 2 requests, num_nodes [3, 2], cu [3, 5], max_nodes 4
    flat = torch.tensor([10, 11, 12, 20, 21], dtype=torch.int64)
    got = scatter_node_preds(flat, torch.tensor([3, 5]), [3, 2], 4)
    assert got.tolist() == [[10, 11, 12, 0], [20, 21, 0, 0]], got.tolist()

    # layout mismatch fails loudly
    try:
        scatter_node_preds(flat, torch.tensor([2, 5]), [3, 2], 4)
        assert False, "expected layout assert"
    except AssertionError as e:
        assert "layout mismatch" in str(e), e

    # is_chain: chain carrier True, branching False
    try:
        from vtl.tree_runtime import build_carrier
    except ImportError:
        from tree_runtime import build_carrier
    assert is_chain(build_carrier([([1, 2], [-1, 0])], [5], 2)) is True
    assert is_chain(build_carrier([([1, 2, 3], [-1, 0, 0])], [5], 3)) is False
    assert is_chain(None) is True
    print("PASS: tree_verify — scatter_node_preds, layout assert, is_chain")


if __name__ == "__main__":
    _selfcheck()
