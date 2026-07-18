"""
Tree speculative decoding core logic (Phases 2-5 of the SGLang ngram-v2 -> vLLM fork).

Pure-logic building blocks (framework-free where possible), validated against the
golden oracles in vtl/tree_verify_ref.py and bench/spike_*.py:

  reconstruct_indices        - Python port of sgl-kernel reconstructIndicesFromTreeMask
  tree_greedy_accept_indices - acceptance walked via retrieve_next_token/sibling (SGLang path),
                               cross-checked against the parent-based reference oracle
  conv_commit_window         - short-conv gather-commit (last K-1 accepted-path Bx)
  flatten_batch_trees        - per-request trees -> padded batch buffers for the runner
  cascade_tree_attention     - the FA3 cascade (prefix + expand + LSE merge); torch, box-only

The vLLM wiring (monkeypatch of runner/backend/sampler) lives in vtl/patches/tree_spec.py.

Self-check (no torch/vllm):  python vtl/tree_spec.py
"""
from __future__ import annotations

from collections import defaultdict


# ----------------------------------------------------------------------------- indices

def reconstruct_indices(parent: list[int], seq_len: int):
    """Port of sgl-kernel `reconstructIndicesFromTreeMask` (ngram_utils.cu) from parent ptrs.

    Returns dict of per-node arrays:
      positions          - seq_len + depth(node)          (absolute position for RoPE/attn)
      retrieve_index     - identity node->flat index (kept for parity with the CUDA contract)
      retrieve_next_token- first child index, or -1
      retrieve_next_sibling - next same-parent sibling index, or -1
      depth              - tree depth of node (root-level draft token = 0)
    Node ordering is the drafter's insertion order; children/siblings are the smallest
    index greater than the node (matches build_tree's root-first insertion).
    """
    n = len(parent)
    depth = [0] * n
    for i in range(n):
        d, j = 0, parent[i]
        while j != -1:
            d += 1
            j = parent[j]
        depth[i] = d
    positions = [seq_len + depth[i] for i in range(n)]
    retrieve_index = list(range(n))
    retrieve_next_token = [-1] * n
    retrieve_next_sibling = [-1] * n
    children = defaultdict(list)
    for i, p in enumerate(parent):
        children[p].append(i)
    for i in range(n):
        kids = children.get(i, [])
        if kids:
            retrieve_next_token[i] = min(kids)
    for p, kids in children.items():
        s = sorted(kids)
        for a, b in zip(s, s[1:]):
            retrieve_next_sibling[a] = b
    return {
        "positions": positions,
        "retrieve_index": retrieve_index,
        "retrieve_next_token": retrieve_next_token,
        "retrieve_next_sibling": retrieve_next_sibling,
        "depth": depth,
    }


def parent_from_mask(mask: list[list[bool]]) -> list[int]:
    """Recover parent pointers from an ancestor tree_mask (mask[i][j] = j ancestor of i)."""
    n = len(mask)
    parent = [-1] * n
    for i in range(n):
        ancestors = [j for j in range(n) if j != i and mask[i][j]]
        # parent = the deepest ancestor = the one that has the most ancestors itself
        if ancestors:
            parent[i] = max(ancestors, key=lambda j: sum(mask[j]))
    return parent


# -------------------------------------------------------------------------- acceptance

def tree_greedy_accept_indices(
    node_tokens: list[int],
    retrieve_next_token: list[int],
    retrieve_next_sibling: list[int],
    target_pred: list[int],
    root_pred: int,
    root_children: list[int],
) -> tuple[list[int], int]:
    """Greedy acceptance walked via the SGLang retrieve-index structure.

    `root_children` = the first-level draft node indices (parent == -1), needed because the
    retrieve arrays only link within the tree; the root's "next_token" is supplied separately.
    Returns (accepted_node_ids, bonus_token) — must equal the parent-based reference oracle.
    """
    accepted: list[int] = []
    expected = root_pred
    # first level: scan root_children as a sibling chain
    cur = root_children[0] if root_children else -1
    # build a quick sibling-walk over root_children
    root_next = {}
    for a, b in zip(sorted(root_children), sorted(root_children)[1:]):
        root_next[a] = b
    while True:
        # find, among the current sibling chain, the node whose token == expected
        match = -1
        node = cur
        first_level = not accepted
        while node != -1:
            if node_tokens[node] == expected:
                match = node
                break
            node = root_next.get(node, -1) if first_level else retrieve_next_sibling[node]
        if match == -1:
            break
        accepted.append(match)
        expected = target_pred[match]
        cur = retrieve_next_token[match]  # descend to first child
        if cur == -1:
            break
    return accepted, expected


# ------------------------------------------------------------------------- conv commit

def conv_commit_window(accepted_path, staged_bx, init_state, state_len):
    """LFM2 short-conv gather-commit: the last `state_len` conv-inputs along the accepted path.

    accepted_path : accepted node ids (root->leaf order)
    staged_bx     : dict node_id -> Bx vector, staged during the tree verify forward
    init_state    : the cached conv window before the tree (list of `state_len` vectors)
    Returns (committed_window, effective_len) where effective_len = len(accepted_path) is the
    scalar the `align`-mode block-migration still needs (path length), kept consistent with the
    gathered contents. Validated in bench/spike_tree_conv_commit.py.
    """
    timeline = list(init_state) + [staged_bx[n] for n in accepted_path]
    return timeline[-state_len:], len(accepted_path)


# ------------------------------------------------------------------- batch flattening

def flatten_batch_trees(trees, seq_lens, max_nodes, pad_token=0):
    """Per-request (node_tokens, parent) -> padded batch buffers for the runner/backend.

    Returns dict of row-major [batch, max_nodes] (or [batch, max_nodes, max_nodes]) lists:
      draft_tokens, positions, retrieve_next_token, retrieve_next_sibling, num_nodes,
      tree_mask (bool [b, max_nodes, max_nodes], ancestor incl self).
    Fixed max_nodes keeps every buffer static-shaped so cudagraph can capture the verify step.
    """
    b = len(trees)
    draft = [[pad_token] * max_nodes for _ in range(b)]
    pos = [[0] * max_nodes for _ in range(b)]
    nxt = [[-1] * max_nodes for _ in range(b)]
    sib = [[-1] * max_nodes for _ in range(b)]
    mask = [[[False] * max_nodes for _ in range(max_nodes)] for _ in range(b)]
    num_nodes = [0] * b
    for r, (node_tokens, parent) in enumerate(trees):
        m = min(len(node_tokens), max_nodes)
        num_nodes[r] = m
        idx = reconstruct_indices(parent[:m], int(seq_lens[r]))
        for i in range(m):
            draft[r][i] = node_tokens[i]
            pos[r][i] = idx["positions"][i]
            nxt[r][i] = idx["retrieve_next_token"][i] if idx["retrieve_next_token"][i] < m else -1
            sib[r][i] = idx["retrieve_next_sibling"][i] if idx["retrieve_next_sibling"][i] < m else -1
            j = i
            while j != -1:
                mask[r][i][j] = True
                j = parent[j]
    return {
        "draft_tokens": draft,
        "positions": pos,
        "retrieve_next_token": nxt,
        "retrieve_next_sibling": sib,
        "tree_mask": mask,
        "num_nodes": num_nodes,
    }


# --------------------------------------------------------- cascade attention (torch/box)

def cascade_tree_attention(q, k_prefix, v_prefix, k_tree, v_tree, tree_mask):
    """FA3 cascade tree attention (validated in bench/spike_fa3_tree_cascade.py).

    Reference-shaped torch implementation: prefix stage (all queries see committed KV) +
    expand stage (each query sees only tree ancestors) merged by LSE. The production Phase-3
    backend replaces the two `sdpa` calls with `flash_attn_varlen_func` on paged KV (a
    block_size=1 draft region) but the merge math is identical. Requires torch; box-only.
    """
    import math

    import torch

    def sdpa(qq, kk, vv, mask):
        scale = 1.0 / math.sqrt(qq.shape[-1])
        scores = (qq @ kk.transpose(-1, -2)) * scale
        scores = scores.masked_fill(~mask, float("-inf"))
        lse = torch.logsumexp(scores, dim=-1)
        return torch.softmax(scores, dim=-1) @ vv, lse

    t = q.shape[0]
    out_p, lse_p = sdpa(q, k_prefix, v_prefix, torch.ones(t, k_prefix.shape[0], dtype=torch.bool, device=q.device))
    out_t, lse_t = sdpa(q, k_tree, v_tree, tree_mask)
    m = torch.maximum(lse_p, lse_t)
    w1 = torch.exp(lse_p - m).unsqueeze(-1)
    w2 = torch.exp(lse_t - m).unsqueeze(-1)
    return (out_p * w1 + out_t * w2) / (w1 + w2)


# ------------------------------------------------------------------------------- checks

def _selfcheck():
    from tree_verify_ref import tree_greedy_accept  # parent-based oracle
    from ngram_tree import tree_mask_from_parent

    # Tree: 0->{1,2}; 1->{3}; 3->{4}; 2->{5}
    parent = [-1, 0, 0, 1, 3, 2]
    node_tokens = [10, 11, 12, 13, 14, 15]

    idx = reconstruct_indices(parent, seq_len=100)
    assert idx["depth"] == [0, 1, 1, 2, 3, 2], idx["depth"]
    assert idx["positions"] == [100, 101, 101, 102, 103, 102]
    assert idx["retrieve_next_token"] == [1, 3, 5, 4, -1, -1], idx["retrieve_next_token"]
    assert idx["retrieve_next_sibling"] == [-1, 2, -1, -1, -1, -1], idx["retrieve_next_sibling"]

    # parent_from_mask round-trips
    assert parent_from_mask(tree_mask_from_parent(parent)) == parent

    # index-based acceptance must equal the parent-based oracle over several targets
    root_children = [i for i, p in enumerate(parent) if p == -1]
    cases = [
        (10, [11, 13, 0, 0, 0, 0]),   # accept 0->1->3
        (10, [12, 0, 15, 0, 0, 0]),   # accept 0->2->5
        (10, [11, 99, 0, 0, 0, 0]),   # accept 0->1, target then diverges
        (77, [0, 0, 0, 0, 0, 0]),     # no root match
        (10, [14, 0, 0, 0, 0, 0]),    # wants a non-child (no cross-branch jump)
    ]
    for root_pred, target_pred in cases:
        ref_acc, ref_bonus = tree_greedy_accept(node_tokens, parent, target_pred, root_pred)
        acc, bonus = tree_greedy_accept_indices(
            node_tokens, idx["retrieve_next_token"], idx["retrieve_next_sibling"],
            target_pred, root_pred, root_children,
        )
        assert acc == ref_acc and bonus == ref_bonus, (root_pred, acc, ref_acc, bonus, ref_bonus)

    # conv gather-commit: last state_len accepted-path Bx
    staged = {n: [float(n)] for n in range(6)}
    win, eff = conv_commit_window([0, 1, 3], staged, init_state=[[-1.0], [-2.0]], state_len=2)
    assert win == [[1.0], [3.0]] and eff == 3, (win, eff)
    win2, eff2 = conv_commit_window([0], staged, init_state=[[-1.0], [-2.0]], state_len=2)
    assert win2 == [[-2.0], [0.0]] and eff2 == 1, (win2, eff2)  # short path pulls from init_state

    # batch flatten: shapes + a spot check
    flat = flatten_batch_trees([(node_tokens, parent)], seq_lens=[100], max_nodes=8)
    assert flat["num_nodes"] == [6]
    assert flat["draft_tokens"][0][:6] == node_tokens
    assert flat["tree_mask"][0][3][:6] == [True, True, False, True, False, False]  # node 3 sees 0,1,3

    print("PASS: tree_spec — indices, index-based accept == oracle, conv-commit, batch flatten")


if __name__ == "__main__":
    _selfcheck()
