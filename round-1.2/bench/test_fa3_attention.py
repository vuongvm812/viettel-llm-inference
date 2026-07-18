"""Correctness + micro-benchmark for the vendored official FA3 paged-KV kernel.

Needs a GPU + the built `flash_attn_3._C`. Run inside the container:

    pytest bench/test_fa3_attention.py -q          # FA3 vs pure-torch SDPA oracle
    python bench/test_fa3_attention.py             # A/B bench: FA3 vs vLLM's bundled FA

LFM2.5-1.2B GQA shapes: 32 q-heads / 8 kv-heads, head_dim=64, bf16, causal. This exercises
the exact entry point the fa3_attention patch uses -- flash_attn_with_kvcache with a paged
KV cache (page_table + cache_seqlens) -- for a full prefill and a decode step.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

H, H_KV, D = 32, 8, 64
DTYPE = torch.bfloat16
BLOCK = 16
SCALE = 1.0 / math.sqrt(D)
RTOL, ATOL = 2e-2, 2e-2


def _fa3():
    try:
        from flash_attn_3 import flash_attn_with_kvcache

        return flash_attn_with_kvcache
    except Exception as exc:  # pragma: no cover - image without the built ext
        pytest.skip(f"flash_attn_3 unavailable: {exc}")


def _build_paged(seqlens: list[int]):
    """Per-seq contiguous KV -> a shared paged cache + page_table + cache_seqlens.

    Returns (k_cache, v_cache, page_table, cache_seqlens, per_seq_kv) where per_seq_kv is
    the list of (k_i, v_i) contiguous tensors used to build the oracle.
    """
    max_blocks = max((s + BLOCK - 1) // BLOCK for s in seqlens)
    total_blocks = sum((s + BLOCK - 1) // BLOCK for s in seqlens)
    dev = "cuda"
    k_cache = torch.zeros(total_blocks, BLOCK, H_KV, D, dtype=DTYPE, device=dev)
    v_cache = torch.zeros(total_blocks, BLOCK, H_KV, D, dtype=DTYPE, device=dev)
    page_table = torch.full((len(seqlens), max_blocks), -1, dtype=torch.int32, device=dev)
    cache_seqlens = torch.tensor(seqlens, dtype=torch.int32, device=dev)
    per_seq_kv = []
    blk = 0
    for i, s in enumerate(seqlens):
        k_i = torch.randn(s, H_KV, D, dtype=DTYPE, device=dev)
        v_i = torch.randn(s, H_KV, D, dtype=DTYPE, device=dev)
        per_seq_kv.append((k_i, v_i))
        nb = (s + BLOCK - 1) // BLOCK
        for j in range(nb):
            page_table[i, j] = blk
            lo, hi = j * BLOCK, min((j + 1) * BLOCK, s)
            k_cache[blk, : hi - lo] = k_i[lo:hi]
            v_cache[blk, : hi - lo] = v_i[lo:hi]
            blk += 1
    return k_cache, v_cache, page_table, cache_seqlens, per_seq_kv


def _sdpa_oracle(q_i, k_i, v_i, causal):
    """Reference GQA attention for one sequence. q_i:(sq,H,D) k/v_i:(sk,H_KV,D)."""
    rep = H // H_KV
    k = k_i.repeat_interleave(rep, dim=1)  # (sk, H, D)
    v = v_i.repeat_interleave(rep, dim=1)
    q = q_i.transpose(0, 1).float()  # (H, sq, D)
    k = k.transpose(0, 1).float()
    v = v.transpose(0, 1).float()
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=SCALE)
    return out.transpose(0, 1).to(DTYPE)  # (sq, H, D)


def _run_case(seqlens, query_lens, causal):
    fa3 = _fa3()
    k_cache, v_cache, page_table, cache_seqlens, per_seq_kv = _build_paged(seqlens)
    b = len(seqlens)
    sq = query_lens[0]
    assert all(q == sq for q in query_lens), "batched form needs equal query_len"
    q = torch.randn(b, sq, H, D, dtype=DTYPE, device="cuda")

    out = fa3(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        softmax_scale=SCALE,
        causal=causal,
    )
    if isinstance(out, tuple):
        out = out[0]

    for i in range(b):
        k_i, v_i = per_seq_kv[i]
        # The query tokens are the last sq positions of the sequence.
        ref = _sdpa_oracle(q[i], k_i, v_i, causal=causal)
        torch.testing.assert_close(out[i], ref, rtol=RTOL, atol=ATOL)


def test_fa3_prefill_parity():
    # Full prefill: query_len == seq_len, standard causal.
    _run_case(seqlens=[256], query_lens=[256], causal=True)


def test_fa3_decode_parity():
    # Decode step: 1 query attends the whole cached context (bottom-right causal).
    _run_case(seqlens=[128], query_lens=[1], causal=True)


def _bench():
    import time

    fa3 = _fa3()
    try:
        from vllm.vllm_flash_attn import flash_attn_with_kvcache as vllm_fa
    except Exception:
        vllm_fa = None

    for tag, seqlens, sq in (("prefill", [2048], 2048), ("decode", [4096] * 32, 1)):
        k_cache, v_cache, page_table, cache_seqlens, _ = _build_paged(seqlens)
        q = torch.randn(len(seqlens), sq, H, D, dtype=DTYPE, device="cuda")

        def call(fn):
            return fn(
                q=q, k_cache=k_cache, v_cache=v_cache, page_table=page_table,
                cache_seqlens=cache_seqlens, softmax_scale=SCALE, causal=True,
            )

        def timeit(fn, iters=50):
            for _ in range(10):
                call(fn)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                call(fn)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / iters * 1e6  # us

        us_fa3 = timeit(fa3)
        line = f"[{tag}] FA3(upstream)={us_fa3:8.1f}us"
        if vllm_fa is not None:
            try:
                line += f"  vLLM-bundled={timeit(vllm_fa):8.1f}us"
            except Exception as exc:
                line += f"  vLLM-bundled=n/a ({exc})"
        print(line)


if __name__ == "__main__":
    _bench()
