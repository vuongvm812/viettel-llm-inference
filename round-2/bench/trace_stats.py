#!/usr/bin/env python3
"""Characterize the request trace: token lengths, prefill:decode ratio, shared prefix, KV footprint.

Every serving flag in docker-compose-optimized.yaml is derived from these numbers,
so re-run this whenever the trace changes.

    python bench/trace_stats.py [--trace data/input/trace-round2.jsonl] [--model ../hf-model]

KV geometry is read from the model's own config.json (--model), not hardcoded. On a hybrid
model only the full-attention layers grow a KV cache per token; SSM/conv layers hold a
fixed-size state per sequence, which is constant rather than per-token and so does not
factor into the per-token growth below.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Used only when config.json cannot be read (no model dir). Stated, not silent.
FALLBACK_KV_BYTES_PER_TOKEN = 2 * 6 * 8 * 64 * 2


def kv_bytes_per_token(model_dir: str | Path) -> tuple[int, str]:
    """Bytes of KV cache one token adds, from config.json. Returns (bytes, how-we-got-it).

    Counts ONLY the attention layers: `layer_types` is HF's per-layer tag on hybrid models
    ("full_attention" vs a recurrent kind), so a hybrid charges KV for its attention layers
    alone. A plain dense model has no such key and every layer counts.
    """
    try:
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
    except Exception as e:
        return FALLBACK_KV_BYTES_PER_TOKEN, f"fallback ({e.__class__.__name__}: no config.json)"

    text = cfg.get("text_config", cfg)  # multimodal wrappers nest the LM config
    n_layers = int(text.get("num_hidden_layers", 0))
    types = text.get("layer_types")
    attn_layers = sum(1 for t in types if "attention" in str(t)) if types else n_layers

    n_heads = int(text.get("num_attention_heads", 0))
    kv_heads = int(text.get("num_key_value_heads", n_heads) or n_heads)
    head_dim = int(text.get("head_dim", 0) or (text.get("hidden_size", 0) // n_heads if n_heads else 0))
    # bf16/fp16 KV. fp8 kv-cache-dtype halves this at RUNTIME, but the trace's working-set
    # figure is about how much the trace touches, so keep the unquantized number here.
    elem = 2

    if not (attn_layers and kv_heads and head_dim):
        return FALLBACK_KV_BYTES_PER_TOKEN, "fallback (config.json missing KV fields)"
    return (
        2 * attn_layers * kv_heads * head_dim * elem,
        f"{attn_layers} attn layers x {kv_heads} kv heads x {head_dim} head_dim x {elem}B, K+V",
    )


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
    kv_per_token, kv_how = kv_bytes_per_token(args.model)
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
    working_set = unique * args.block_size * kv_per_token
    print(f"\nprefix cache (block_size={args.block_size})")
    print(f"  hit rate          {hits / (unique + hits):.1%} of blocks")
    print(f"  prefill avoided   {hits * args.block_size:,} tokens")
    print(f"  retained KV       {working_set / 1e9:.1f} GB bf16 / "
          f"{working_set / 2e9:.1f} GB fp8  <- the offload working set")

    per_req = percentile(ordered, .5) * kv_per_token
    budget = args.kv_gb * 1e9
    print(f"\nKV per token        {kv_per_token / 1024:.0f} KB (bf16)  [{kv_how}]")
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

    # ---- KV geometry is read from config.json, not assumed ----
    import tempfile

    def _cfg(d):
        tmp = tempfile.mkdtemp()
        (Path(tmp) / "config.json").write_text(json.dumps(d))
        return tmp

    # Dense: every layer holds KV. 2 * 4 layers * 2 kv heads * 8 head_dim * 2B = 256.
    got, how = kv_bytes_per_token(_cfg({
        "num_hidden_layers": 4, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 8,
    }))
    assert got == 256, (got, how)

    # Hybrid: layer_types tags 2 of 4 as attention, so KV halves. This is the whole
    # point of reading layer_types -- charging all 4 would overstate the working set 2x.
    got, how = kv_bytes_per_token(_cfg({
        "num_hidden_layers": 4, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 8,
        "layer_types": ["conv", "full_attention", "conv", "full_attention"],
    }))
    assert got == 128, (got, how)

    # head_dim derived from hidden_size when absent; MHA when num_key_value_heads absent.
    got, _ = kv_bytes_per_token(_cfg({
        "num_hidden_layers": 1, "num_attention_heads": 4, "hidden_size": 32,
    }))
    assert got == 2 * 1 * 4 * 8 * 2, got

    # Nested multimodal config still finds the LM fields.
    got, _ = kv_bytes_per_token(_cfg({
        "text_config": {"num_hidden_layers": 1, "num_attention_heads": 2, "head_dim": 4}
    }))
    assert got == 2 * 1 * 2 * 4 * 2, got

    # No model dir: falls back, and SAYS it fell back rather than reporting a fake number.
    got, how = kv_bytes_per_token("/nonexistent-model-dir")
    assert got == FALLBACK_KV_BYTES_PER_TOKEN and "fallback" in how, (got, how)
    got, how = kv_bytes_per_token(_cfg({"foo": 1}))
    assert got == FALLBACK_KV_BYTES_PER_TOKEN and "fallback" in how, (got, how)

    print("trace_stats self-check ok")


if __name__ == "__main__":
    import sys

    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
