"""
Phase-0 spike (Crux 2): tree attention == cascade(prefix, tree) merged by LSE.

Validates the decomposition SGLang uses to run tree verify on FA3/Hopper WITHOUT
a custom-mask kernel (flashattention_backend.py topk>1 path). Full masked
attention over [committed_KV ++ tree_KV] with a tree mask is split into:
  (1) prefix stage  — every draft query attends to ALL committed KV (plain causal),
  (2) expand stage  — each draft query attends only to its tree ANCESTORS (a
                      per-query KV subset; SGLang realizes this by permuting paged
                      KV indices so ancestors sort to the front + per-query seqlen),
then the two are merged by log-sum-exp. If this holds, tree verify on H200 needs
only two standard FA3 calls + an LSE merge — the mask never enters a kernel.

Pure torch (CPU fine). Run on the box:  python bench/spike_fa3_tree_cascade.py

NOTE: this validates the math of the decomposition (the load-bearing claim),
independent of the flash-attn API. The Phase-3 backend work then implements stage
(2) via vLLM's paged block_table with a block_size=1 draft region + cascade merge.
"""
import math

import torch

torch.manual_seed(0)


def sdpa(q, k, v, mask):
    """Masked softmax attention. q:[Q,d] k,v:[K,d] mask:[Q,K] bool. -> out:[Q,d], lse:[Q]."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~mask, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    out = torch.softmax(scores, dim=-1) @ v
    return out, lse


def lse_merge(out1, lse1, out2, lse2):
    """Flash cascade merge of two partial attentions by log-sum-exp."""
    m = torch.maximum(lse1, lse2)
    w1 = torch.exp(lse1 - m).unsqueeze(-1)
    w2 = torch.exp(lse2 - m).unsqueeze(-1)
    return (out1 * w1 + out2 * w2) / (w1 + w2)


def ancestor_mask(parent):
    """[T,T] bool: mask[i][j] = j on i's root-path (ancestor incl. self)."""
    n = len(parent)
    m = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        j = i
        while j != -1:
            m[i, j] = True
            j = parent[j]
    return m


def run(device="cpu", dtype=torch.float32):
    d, P = 16, 7                       # head dim, committed (prefix) length
    parent = [-1, 0, 0, 1, 3, 2]       # draft tree: 0->{1,2}; 1->{3}; 3->{4}; 2->{5}
    T = len(parent)
    tree_mask = ancestor_mask(parent).to(device)

    q = torch.randn(T, d, device=device, dtype=dtype)
    k_pre = torch.randn(P, d, device=device, dtype=dtype)
    v_pre = torch.randn(P, d, device=device, dtype=dtype)
    k_tree = torch.randn(T, d, device=device, dtype=dtype)
    v_tree = torch.randn(T, d, device=device, dtype=dtype)

    # Reference: one dense attention over [prefix ++ tree] with the combined mask.
    K = torch.cat([k_pre, k_tree], 0)
    V = torch.cat([v_pre, v_tree], 0)
    full_mask = torch.cat([torch.ones(T, P, dtype=torch.bool, device=device), tree_mask], dim=1)
    ref, _ = sdpa(q, K, V, full_mask)

    # Cascade: prefix stage (all queries see all committed KV) + expand stage
    # (ancestors only), merged by LSE.
    out_p, lse_p = sdpa(q, k_pre, v_pre, torch.ones(T, P, dtype=torch.bool, device=device))
    out_t, lse_t = sdpa(q, k_tree, v_tree, tree_mask)
    casc = lse_merge(out_p, lse_p, out_t, lse_t)

    err = (casc - ref).abs().max().item()
    tol = 1e-5 if dtype == torch.float32 else 2e-3
    assert err < tol, f"cascade != dense masked attention on {device}/{dtype}: max err {err}"
    return err


if __name__ == "__main__":
    err = run("cpu", torch.float32)
    print(f"PASS: cascade+LSE == dense masked tree attention (cpu/fp32, max err {err:.2e})")
    if torch.cuda.is_available():
        err_h = run("cuda", torch.bfloat16)
        print(f"PASS: same on cuda/bf16 (max err {err_h:.2e}) — the H200 verify dtype")
    print("  => tree verify = 2 standard attentions + LSE merge; no FA4/custom-mask kernel needed.")
