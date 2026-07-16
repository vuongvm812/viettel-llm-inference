#!/usr/bin/env python3
"""Characterize the request trace: token lengths, prefill:decode ratio, shared prefix, KV footprint.

Every serving flag in docker-compose-optimized.yaml is derived from these numbers,
so re-run this whenever the trace changes.

    python bench/trace_stats.py [--trace data/input/trace-round2.jsonl] [--model ../hf-model]

LFM2.5-1.2B is a hybrid: only the 6 full-attention layers grow a KV cache per token; the 10
short-conv layers hold a fixed-size conv state per sequence (conv_dim=2048 x L_cache=3), which is
constant, not per-token, so it does not factor into the per-token KV growth below.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# LFM2.5-1.2B: 6 full-attention layers (of 16), 8 KV heads (GQA), head_dim 64, 2 bytes/elem bf16.
KV_BYTES_PER_TOKEN = 2 * 6 * 8 * 64 * 2  # K and V, attention layers only


def prompt_token_counts(rows: list[dict], tokenizer) -> list[int]:
    return [
        sum(len(tokenizer.encode(m["content"]).ids) for m in r["body"]["messages"])
        for r in rows
    ]


def token_sequences(rows: list[dict], tokenizer) -> list[list[int]]:
    """The token sequence each request's KV cache actually holds."""
    return [
        tokenizer.encode("".join(m["content"] for m in r["body"]["messages"])).ids
        for r in rows
    ]


def prefix_cache_stats(seqs: list[list[int]], block_size: int = 16) -> tuple[int, int]:
    """Simulate vLLM's block-hash prefix cache in arrival order.

    Mirrors ``vllm.v1.core.kv_cache_utils.hash_block_tokens``: each block's key is
    ``hash(parent_block_hash, block_token_ids)``, so the key identifies the whole
    prefix, not just the block. That chained hash in a flat dict *is* a radix tree
    at block granularity -- which is why vLLM has no literal tree.

    Chaining keeps this O(1) per block. Accumulating the prefix as a tuple instead
    would be O(L^2) per sequence and never finishes at block_size=1.

    Returns ``(unique_blocks, hit_blocks)``. ``unique_blocks * block_size`` is the
    retained working set: the KV that must stay resident to get every hit. Compare
    it against the GPU's KV budget before reaching for CPU/NVMe offload.

    A trailing partial block is not counted; vLLM only caches full blocks.
    """
    seen: set[int] = set()
    unique = hits = 0
    for seq in seqs:
        parent = 0  # stands in for vLLM's NONE_HASH
        for start in range(0, len(seq) - block_size + 1, block_size):
            parent = hash((parent, tuple(seq[start : start + block_size])))
            if parent in seen:
                hits += 1
            else:
                seen.add(parent)
                unique += 1
    return unique, hits


def percentile(sorted_vals: list[int], q: float) -> int:
    return sorted_vals[int(q * (len(sorted_vals) - 1))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="data/input/trace-round2.jsonl")
    ap.add_argument("--model", default="../hf-model")
    ap.add_argument("--max-position-embeddings", type=int, default=32768)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--kv-gb", type=float, default=120.0, help="KV bytes available on the GPU")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    rows = [json.loads(line) for line in open(args.trace)]

    lens = prompt_token_counts(rows, tokenizer)
    ordered = sorted(lens)
    max_tokens = max(r["body"].get("max_tokens") or 0 for r in rows)
    prefill, decode = sum(lens), max_tokens * len(rows)
    turns = {len([m for m in r["body"]["messages"] if m["role"] == "user"]) for r in rows}

    print(f"requests            {len(rows)}")
    print(
        f"prompt tokens       min={ordered[0]:,}  p50={percentile(ordered, .5):,}  "
        f"p90={percentile(ordered, .9):,}  max={ordered[-1]:,}"
    )
    print(f"user turns/request  {min(turns)}..{max(turns)}")
    print(f"prefill tokens      {prefill:,}")
    print(f"decode tokens       {decode:,}  (max_tokens={max_tokens})")
    print(f"prefill:decode      {prefill / max(decode, 1):.0f}:1")

    needed = ordered[-1] + max_tokens
    print(f"\nrequired --max-model-len  >= {needed:,}")
    if needed > args.max_position_embeddings:
        print(
            f"  !! exceeds max_position_embeddings ({args.max_position_embeddings:,}); "
            "needs RoPE scaling or VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
        )

    seqs = token_sequences(rows, tokenizer)
    unique, hits = prefix_cache_stats(seqs, args.block_size)
    working_set = unique * args.block_size * KV_BYTES_PER_TOKEN
    print(f"\nprefix cache (block_size={args.block_size})")
    print(f"  hit rate          {hits / (unique + hits):.1%} of blocks")
    print(f"  prefill avoided   {hits * args.block_size:,} tokens")
    print(f"  retained KV       {working_set / 1e9:.1f} GB bf16 / "
          f"{working_set / 2e9:.1f} GB fp8  <- the offload working set")

    per_req = percentile(ordered, .5) * KV_BYTES_PER_TOKEN
    budget = args.kv_gb * 1e9
    print(f"\nKV per token        {KV_BYTES_PER_TOKEN / 1024:.0f} KB (bf16)")
    print(f"KV per request @p50 {per_req / 1e9:.2f} GB")
    print(
        f"concurrent requests {budget / per_req:.0f} (bf16)  ->  "
        f"{budget / (per_req / 2):.0f} (fp8 KV)"
    )
    print(f"\nGPU KV budget       {args.kv_gb:.0f} GB")
    if working_set < budget:
        print(
            f"  the entire reusable working set fits in GPU with "
            f"{budget / working_set:.0f}x headroom -> CPU/NVMe offload has nothing to do"
        )
    else:
        print("  working set exceeds GPU KV -> offload tiers can raise the hit rate")


def _self_check() -> None:
    class FakeEncoding:
        def __init__(self, ids):
            self.ids = ids

    class FakeTokenizer:
        """One token per character, so token math is checkable by eye."""

        def encode(self, text):
            return FakeEncoding([ord(c) for c in text])

    tok = FakeTokenizer()
    rows = [
        {"body": {"messages": [{"content": "SHARED"}, {"content": "aaa"}]}},
        {"body": {"messages": [{"content": "SHARED"}, {"content": "bb"}]}},
    ]
    assert prompt_token_counts(rows, tok) == [9, 8]

    # Two requests sharing a 4-token prefix, block_size 2: request 1 materializes
    # blocks [ab][cd]; request 2 reuses both, then adds one of its own.
    seqs = [list(b"abcdXX"), list(b"abcdYY")]
    unique, hits = prefix_cache_stats(seqs, block_size=2)
    assert (unique, hits) == (4, 2), (unique, hits)

    # A block whose *prefix* differs is not a hit, even though the block matches.
    seqs = [list(b"abZZ"), list(b"cdZZ")]
    unique, hits = prefix_cache_stats(seqs, block_size=2)
    assert (unique, hits) == (4, 0), (unique, hits)

    # Identical requests: everything after the first is a hit.
    seqs = [list(b"abcd"), list(b"abcd")]
    assert prefix_cache_stats(seqs, block_size=2) == (2, 2)

    # A trailing partial block is not counted (vLLM only caches full blocks).
    assert prefix_cache_stats([list(b"abc")], block_size=2) == (1, 0)

    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert percentile([1, 2, 3, 4, 5], 1.0) == 5
    print("trace_stats self-check ok")


if __name__ == "__main__":
    import sys

    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
