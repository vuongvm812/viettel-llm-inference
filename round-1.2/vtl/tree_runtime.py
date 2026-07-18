"""Torch glue between the framework-free tree logic (vtl/tree_spec.py) and vLLM's runtime.

`tree_spec.py` is pure-Python (list-based) so it stays unit-testable without torch. This module
turns its outputs into device tensors the forked vLLM readers (attention / sampler / conv commit)
consume, and packs/reads the `md.vtl_tree` carrier. Everything here is CPU-testable off-box; the
only vLLM-specific coupling is the field names, kept in one place.

Self-check (needs torch, no vLLM/GPU):  python vtl/tree_runtime.py
"""
from __future__ import annotations

try:  # package import (vllm plugin) vs bare import (python vtl/tree_runtime.py self-check)
    from vtl.tree_spec import flatten_batch_trees
except ImportError:
    from tree_spec import flatten_batch_trees


# The carrier: a dict of device tensors attached to SpecDecodeMetadata as `md.vtl_tree`. Static
# [b, max_nodes] shapes (padded) so cudagraph can capture the verify step. Dtypes chosen to match
# vLLM's index tensors (int64) and attention masks (bool).
_INT_KEYS = ("draft_tokens", "positions", "retrieve_next_token", "retrieve_next_sibling")


def tree_to_tensors(flat: dict, device="cpu") -> dict:
    """flatten_batch_trees() output (lists) -> dict of device tensors.

    draft_tokens/positions/retrieve_next_token/retrieve_next_sibling : int64 [b, max_nodes]
    tree_mask   : bool  [b, max_nodes, max_nodes]  (ancestor incl. self)
    num_nodes   : int32 [b]
    """
    import torch

    out = {k: torch.tensor(flat[k], dtype=torch.int64, device=device) for k in _INT_KEYS}
    out["tree_mask"] = torch.tensor(flat["tree_mask"], dtype=torch.bool, device=device)
    out["num_nodes"] = torch.tensor(flat["num_nodes"], dtype=torch.int32, device=device)
    return out


def build_carrier(trees, seq_lens, max_nodes, device="cpu") -> dict:
    """Full producer: per-request (node_tokens, parent) trees -> device-tensor carrier.

    Used by the metadata hook / the `_calc_spec_decode_metadata` fork. `trees` and `seq_lens` are
    the stash the proposer leaves on `runner.drafter` (`_vtl_pending_trees` / `_vtl_seq_lens`).
    """
    return tree_to_tensors(flatten_batch_trees(trees, seq_lens, max_nodes), device=device)


def is_chain_carrier(carrier: dict) -> bool:
    """True iff every request's tree is linear (no branching) -> stock chain verify is exact.

    Chain <=> no node has a next-sibling. Lets the width-1 milestone route through stock paths
    unchanged (the P2 regression gate: carrier present but behavior bit-identical to no-tree).
    """
    sib = carrier.get("retrieve_next_sibling")
    if sib is None:
        return True
    return bool((sib == -1).all().item())


def _selfcheck():
    import torch

    # Two requests: a linear chain, and a branching tree 0->{1,2}; 1->{3}.
    chain = ([10, 11, 12], [-1, 0, 1])
    branch = ([20, 21, 22, 23], [-1, 0, 0, 1])
    carrier = build_carrier([chain, branch], seq_lens=[5, 7], max_nodes=4, device="cpu")

    assert carrier["draft_tokens"].shape == (2, 4)
    assert carrier["draft_tokens"].dtype == torch.int64
    assert carrier["tree_mask"].shape == (2, 4, 4)
    assert carrier["tree_mask"].dtype == torch.bool
    assert carrier["num_nodes"].tolist() == [3, 4]
    # chain padded with pad_token 0 past node 3
    assert carrier["draft_tokens"][0].tolist() == [10, 11, 12, 0]
    # positions = seq_len + depth: chain depths 0,1,2 -> 5,6,7 ; branch depths 0,1,1,2 -> 7,8,8,9
    assert carrier["positions"][0][:3].tolist() == [5, 6, 7]
    assert carrier["positions"][1].tolist() == [7, 8, 8, 9]
    # branch mask: node 3 sees ancestors 0,1,3
    assert carrier["tree_mask"][1][3].tolist() == [True, True, False, True]

    # is_chain_carrier: pure-chain batch True; batch containing a branch False
    only_chain = build_carrier([chain], seq_lens=[5], max_nodes=4)
    assert is_chain_carrier(only_chain) is True
    assert is_chain_carrier(carrier) is False  # branch has a sibling (node 2 sib of 1)
    print("PASS: tree_runtime — carrier tensorization, positions/mask, is_chain_carrier")


if __name__ == "__main__":
    _selfcheck()
