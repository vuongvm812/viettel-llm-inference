#!/usr/bin/env python3
"""Characterize the request trace: token lengths, prefill:decode ratio, shared prefix, KV footprint.

Every serving flag in docker-compose-optimized.yaml is derived from these numbers,
so re-run this whenever the trace changes.

    python bench/trace_stats.py [--trace data/input/trace-round1.jsonl] [--model hf-model]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Qwen2-3B: 36 layers, 2 KV heads (GQA), head_dim 128, 2 bytes/elem for bf16/fp16.
KV_BYTES_PER_TOKEN = 2 * 36 * 2 * 128 * 2  # K and V


def prompt_token_counts(rows: list[dict], tokenizer) -> list[int]:
    return [
        sum(len(tokenizer.encode(m["content"]).ids) for m in r["body"]["messages"])
        for r in rows
    ]


def common_token_prefix(rows: list[dict], tokenizer, sample: int = 20) -> int:
    """Longest token prefix shared by the first `sample` requests.

    This is what --enable-prefix-caching can actually reuse. Compare against
    mean prompt length to size the win.
    """
    ids = [
        tokenizer.encode("".join(m["content"] for m in r["body"]["messages"])).ids
        for r in rows[:sample]
    ]
    shortest = min(len(x) for x in ids)
    i = 0
    while i < shortest and len({x[i] for x in ids}) == 1:
        i += 1
    return i


def percentile(sorted_vals: list[int], q: float) -> int:
    return sorted_vals[int(q * (len(sorted_vals) - 1))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="data/input/trace-round1.jsonl")
    ap.add_argument("--model", default="hf-model")
    ap.add_argument("--max-position-embeddings", type=int, default=32768)
    ap.add_argument("--kv-gb", type=float, default=120.0, help="KV bytes available on the GPU")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    rows = [json.loads(line) for line in open(args.trace)]

    lens = prompt_token_counts(rows, tokenizer)
    ordered = sorted(lens)
    max_tokens = max(r["body"].get("max_tokens") or 0 for r in rows)
    prefill, decode = sum(lens), max_tokens * len(rows)

    print(f"requests            {len(rows)}")
    print(
        f"prompt tokens       min={ordered[0]:,}  p50={percentile(ordered, .5):,}  "
        f"p90={percentile(ordered, .9):,}  max={ordered[-1]:,}"
    )
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

    shared = common_token_prefix(rows, tokenizer)
    mean_len = prefill / len(rows)
    saved = shared * (len(rows) - 1)
    print(f"\nshared token prefix {shared:,}  ({shared / mean_len:.0%} of the mean prompt)")
    print(f"prefix-cache saves  {saved:,} prefill tokens ({saved / prefill:.0%} of all prefill)")

    per_req = percentile(ordered, .5) * KV_BYTES_PER_TOKEN
    budget = args.kv_gb * 1e9
    print(f"\nKV per token        {KV_BYTES_PER_TOKEN / 1024:.0f} KB (bf16)")
    print(f"KV per request @p50 {per_req / 1e9:.2f} GB")
    print(
        f"concurrent requests {budget / per_req:.0f} (bf16)  ->  "
        f"{budget / (per_req / 2):.0f} (fp8 KV)"
    )


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
    assert common_token_prefix(rows, tok) == 6, common_token_prefix(rows, tok)

    # A prefix that runs to the end of the shortest input must not over-count.
    rows2 = [
        {"body": {"messages": [{"content": "ab"}]}},
        {"body": {"messages": [{"content": "abcd"}]}},
    ]
    assert common_token_prefix(rows2, tok) == 2

    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert percentile([1, 2, 3, 4, 5], 1.0) == 5
    print("trace_stats self-check ok")


if __name__ == "__main__":
    import sys

    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
