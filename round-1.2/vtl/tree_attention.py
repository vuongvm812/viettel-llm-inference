"""Tree-attention masking — the logic the forked FlashAttention/FlashInfer verify path calls.

The verify forward packs `num_nodes` draft queries per request. Each draft node must attend to
(a) ALL committed KV (the prefix, length seq_len) and (b) only its own tree ancestors among the
draft KV. SGLang builds a boolean `custom_mask` of shape [num_nodes, seq_len + num_nodes] per
request (prefix columns all True, then the num_nodes×num_nodes ancestor block = our tree_mask) and
hands it to the flashinfer prefill wrapper (FULL_MASK). This module builds that mask; the reference
`tree_attention_ref` proves the mask yields the same output as the spike-validated cascade merge.

ON-BOX: the exact packing of these per-request masks into the flashinfer/FA `custom_mask` buffer
(flattened + qo/kv indptr, block_size=1 draft KV page layout) is backend-specific and must be
confirmed on the H200 — see vtl/vllm_patches/v0.25.0/flash_attn.patch. The *mask contents* here are
backend-independent and unit-tested.

Self-check (needs torch, no vLLM/GPU):  python vtl/tree_attention.py
"""
from __future__ import annotations

try:  # package vs bare import — see tree_runtime
    from vtl.tree_spec import cascade_tree_attention
except ImportError:
    from tree_spec import cascade_tree_attention


def build_full_mask_rows(carrier: dict, seq_lens):
    """Per-request boolean attention mask [num_nodes, seq_len + num_nodes].

    Column layout per request: [ seq_len committed-KV columns (all True) | num_nodes tree columns
    (= tree_mask, ancestor incl. self) ]. Returns a list of bool tensors (one per request), variable
    KV length — the flash_attn fork flattens/pads these into the backend `custom_mask`.
    """
    import torch

    device = carrier["tree_mask"].device
    tree_mask = carrier["tree_mask"]  # [b, max_nodes, max_nodes] bool
    num_nodes = carrier["num_nodes"].tolist()
    rows = []
    for r, sl in enumerate(seq_lens):
        m = num_nodes[r]
        sl = int(sl)
        prefix = torch.ones((m, sl), dtype=torch.bool, device=device)
        treeblk = tree_mask[r, :m, :m]
        rows.append(torch.cat([prefix, treeblk], dim=1))  # [m, sl + m]
    return rows


def tree_attention_ref(q, k_all, v_all, mask):
    """Dense masked attention reference: softmax(q k^T / sqrt d) over the masked KV, @ v.

    q       : [t, d]  draft-node queries
    k_all,v_all : [kv, d]  concat of [committed KV ; tree KV]
    mask    : [t, kv] bool (from build_full_mask_rows)
    Correctness anchor for the backend fork — must equal the cascade merge (validated below).
    """
    import math

    import torch

    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k_all.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v_all


def _selfcheck():
    import torch

    try:
        from vtl.tree_runtime import build_carrier
    except ImportError:
        from tree_runtime import build_carrier

    torch.manual_seed(0)
    # Tree 0->{1,2}; 1->{3}. seq_len (committed) = 4, num_nodes = 4, head dim 8.
    node_tokens = [10, 11, 12, 13]
    parent = [-1, 0, 0, 1]
    seq_len, m, d = 4, 4, 8
    carrier = build_carrier([(node_tokens, parent)], seq_lens=[seq_len], max_nodes=m)

    rows = build_full_mask_rows(carrier, seq_lens=[seq_len])
    mask = rows[0]
    assert mask.shape == (m, seq_len + m), mask.shape
    # every node attends to all 4 committed columns
    assert bool(mask[:, :seq_len].all())
    # node 3's draft columns = ancestors {0,1,3}
    assert mask[3, seq_len:].tolist() == [True, True, False, True]
    # node 2's draft columns = {0,2}
    assert mask[2, seq_len:].tolist() == [True, False, True, False]

    # the masked dense attention must equal the spike-validated cascade (prefix + tree, LSE-merged)
    q = torch.randn(m, d, dtype=torch.float64)
    k_prefix = torch.randn(seq_len, d, dtype=torch.float64)
    v_prefix = torch.randn(seq_len, d, dtype=torch.float64)
    k_tree = torch.randn(m, d, dtype=torch.float64)
    v_tree = torch.randn(m, d, dtype=torch.float64)
    tree_block = mask[:, seq_len:]

    dense = tree_attention_ref(q, torch.cat([k_prefix, k_tree]), torch.cat([v_prefix, v_tree]), mask)
    casc = cascade_tree_attention(q, k_prefix, v_prefix, k_tree, v_tree, tree_block)
    assert torch.allclose(dense, casc, atol=1e-9), (dense - casc).abs().max()
    print("PASS: tree_attention — full-mask contents, dense == cascade merge")


if __name__ == "__main__":
    _selfcheck()
