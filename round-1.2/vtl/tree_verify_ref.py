"""
Reference tree-verify oracle — the golden test for the vLLM-core fork (Phases 4-5).

`tree_greedy_accept` is a pure-Python reference for SGLang's verify_tree_greedy: given a
draft tree and the target model's greedy prediction at each node, it returns the accepted
root->leaf path (+ bonus token). Every fork implementation of the tree rejection sampler
(vllm/v1/sample/rejection_sampler.py) and the conv-state commit (which consumes the accepted
path) MUST reproduce this. Silent-corruption is the failure mode, so this oracle is the
non-negotiable check: drive a toy tree through both and assert identical accepted paths.

Composes with the other references:
  drafter          -> vtl/ngram_tree.py  (build_tree, tree_mask_from_parent)
  tree attention   -> bench/spike_fa3_tree_cascade.py  (cascade == dense masked)
  conv-state commit-> bench/spike_tree_conv_commit.py  (path-gather == chain)
  acceptance       -> here

Self-check (no deps):  python vtl/tree_verify_ref.py
"""
from __future__ import annotations

from collections import defaultdict


def tree_greedy_accept(
    node_tokens: list[int],
    parent: list[int],
    target_pred: list[int],
    root_pred: int,
) -> tuple[list[int], int]:
    """Greedy tree acceptance.

    Args:
        node_tokens: draft token id at each tree node.
        parent: parent node index per node (-1 for a first-level draft token).
        target_pred: target_pred[i] = the token the target model greedily predicts to
            FOLLOW node i (argmax of target logits at node i's position).
        root_pred: the target's greedy prediction for the FIRST draft token (argmax at
            the anchor / last committed token).

    Returns:
        (accepted_node_ids, bonus_token). accepted_node_ids is the accepted root->leaf
        path (possibly empty); bonus_token is the target's own next token after the last
        accepted node (always emitted — the guaranteed 1-token progress).
    """
    by_parent: dict[int, list[int]] = defaultdict(list)
    for i, p in enumerate(parent):
        by_parent[p].append(i)

    accepted: list[int] = []
    expected = root_pred  # the token the target wants next
    cur = -1  # start above the roots
    while True:
        match = next((c for c in by_parent.get(cur, []) if node_tokens[c] == expected), None)
        if match is None:
            break
        accepted.append(match)
        cur = match
        expected = target_pred[match]  # what the target wants after the accepted node
    return accepted, expected  # `expected` here is the bonus token after the last accept


def emitted_tokens(node_tokens, accepted, bonus):
    return [node_tokens[i] for i in accepted] + [bonus]


def _selfcheck():
    # Tree: 0(A) -> {1(B), 2(C)}; 1 -> {3(D)}    tokens: A=10 B=11 C=12 D=13
    node_tokens = [10, 11, 12, 13]
    parent = [-1, 0, 0, 1]

    # Full accept down A->B->D, then target wants E(=99) which isn't a child -> bonus 99.
    target_pred = [11, 13, 0, 99]  # after0->B, after1->D, after2->(dontcare), after3->E
    acc, bonus = tree_greedy_accept(node_tokens, parent, target_pred, root_pred=10)
    assert acc == [0, 1, 3] and bonus == 99, (acc, bonus)
    assert emitted_tokens(node_tokens, acc, bonus) == [10, 11, 13, 99]

    # Accept A at the root, then the target diverges into the C branch (node 2, a child of
    # 0): accept 0 then 2, bonus = node 2's pred. (C is a 2nd-token alternative, not a root.)
    acc, bonus = tree_greedy_accept(node_tokens, parent, [12, 0, 55, 0], root_pred=10)
    assert acc == [0, 2] and bonus == 55, (acc, bonus)

    # Target's first token matches no root draft: nothing accepted, bonus = root_pred.
    acc, bonus = tree_greedy_accept(node_tokens, parent, [0, 0, 0, 0], root_pred=77)
    assert acc == [] and bonus == 77, (acc, bonus)

    # Accept stops when the wanted token exists in the tree but not as a CHILD of the
    # current node (no cross-branch jump): A accepted, target wants D(13) but D is a child
    # of 1, not of 0 -> stop at 0, bonus 13.
    acc, bonus = tree_greedy_accept(node_tokens, parent, [13, 0, 0, 0], root_pred=10)
    assert acc == [0] and bonus == 13, (acc, bonus)

    print("PASS: tree_greedy_accept — full/partial/branch/no-match + no cross-branch jump")


if __name__ == "__main__":
    _selfcheck()
