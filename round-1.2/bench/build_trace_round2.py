#!/usr/bin/env python3
"""Regenerate ``data/input/trace-round2.jsonl`` to match the phase-2 grading distribution.

Reads the grading spec ``data/input/trace_grading_public.jsonl`` (one row per turn:
``conv_id, turn_idx, in_warmup, timestamp_ms, think_ms, in_chars, in_tokens_est, out_tokens_max``,
no message bodies) and emits a replayable trace (``request_id, timestamp_ms, workload_type, body``)
whose bodies carry real message text sized -- via the LFM2.5 tokenizer -- to each turn's
``in_tokens_est``.

Workload model (why this shape). The spec has 70 conversations x 6 turns, and ``in_tokens_est`` is
FLAT (~4,000) across a conversation's turns rather than growing. So a turn is not the full accreting
history; it is a fixed per-conversation context re-sent each turn plus a small varying question. We
reproduce that exactly:

    messages = [ {system: GLOBAL_PREAMBLE + CTX[conv]} , {user: Q[conv, turn]} ]

  * GLOBAL_PREAMBLE  -- one fixed block shared by ALL conversations. Cached process-wide after the
    very first request; this is what vtl/warmup_healthcheck.py primes.
  * CTX[conv]        -- a ~3.6k-token context unique to each conversation (distinct prefix, so no
    accidental cross-conversation cache hits). Identical across that conversation's 6 turns, so it
    is a prefix-cache hit on turns 1..5 -- the intra-conversation reuse that dominates phase-2.
  * Q[conv, turn]    -- a small (~80-token) question that varies per turn (the only new prefill).

Timing. ``timestamp_ms`` is the absolute arrival of a conversation (set on turn 0; turns 1..5 carry
0 in the spec because the grader think-paces them off the previous response). Our replay.py is
open-loop, so we synthesize each turn's absolute arrival as
``conv_arrival + turn_idx * (think_ms + NOMINAL_DECODE_MS)`` and emit records sorted by arrival.
NOMINAL_DECODE_MS is a knob (rough time to stream out_tokens_max); it only affects open-loop queue
spacing, not prefill/cache behaviour.

Deterministic: all text is sliced from the round-1 corpus at index-derived offsets (no RNG), so
reruns are byte-identical. Corpus = the message content already in ``data/input/trace-round1.jsonl``.

    python bench/build_trace_round2.py                 # from round-1.2/: writes data/input/trace-round2.jsonl
    python bench/build_trace_round2.py --self-check    # no model/IO; checks the assembly logic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_NAME = "LFM2.5-1.2B-Instruct"
PREAMBLE_TOKENS = 300  # global shared system preamble (warmup-primeable)
QUESTION_TOKENS = 80  # per-turn varying user question (the only new prefill each turn)
NOMINAL_DECODE_MS = 2000  # rough stream time for out_tokens_max; open-loop pacing knob only


def load_spec(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def load_corpus_ids(trace_round1: str, tok) -> list[int]:
    """All message text from the round-1 trace, tokenized once into a flat id pool."""
    chunks: list[str] = []
    for line in open(trace_round1):
        if not line.strip():
            continue
        body = json.loads(line)["body"]
        chunks.extend(m["content"] for m in body["messages"])
    return tok.encode("\n\n".join(chunks)).ids


def sized_text(pool: list[int], start: int, ntok: int, tok) -> str:
    """Decode ``ntok`` tokens from ``pool`` starting at ``start`` (wrapping if needed)."""
    n = len(pool)
    ids = [pool[(start + i) % n] for i in range(ntok)]
    return tok.decode(ids)


def build_records(spec: list[dict], pool: list[int], tok) -> list[dict]:
    from collections import defaultdict

    # One fixed global preamble, shared by every conversation.
    preamble = "You are a helpful assistant. Use the provided context to answer.\n\n" + sized_text(
        pool, 0, PREAMBLE_TOKENS, tok
    )

    by_conv: dict[int, list[dict]] = defaultdict(list)
    for row in spec:
        by_conv[row["conv_id"]].append(row)

    staged: list[tuple[int, dict]] = []  # (abs_ts, record) before sorting/re-id
    for conv_id, turns in by_conv.items():
        turns = sorted(turns, key=lambda r: r["turn_idx"])
        target0 = turns[0]["in_tokens_est"]
        ctx_tokens = max(target0 - PREAMBLE_TOKENS - QUESTION_TOKENS, 1)
        # Unique, non-overlapping context slice per conversation; a per-conv header guarantees a
        # distinct prefix even if slices ever wrap and overlap.
        ctx_start = (conv_id + 1) * (ctx_tokens + QUESTION_TOKENS + PREAMBLE_TOKENS)
        context = f"[Reference document for session {conv_id}]\n" + sized_text(
            pool, ctx_start, ctx_tokens, tok
        )
        system_content = preamble + "\n\n" + context

        conv_arrival = turns[0]["timestamp_ms"]
        for row in turns:
            t = row["turn_idx"]
            # Distinct question per (conv, turn) -- the only varying prefill.
            q_start = ctx_start + ctx_tokens + t * QUESTION_TOKENS
            question = "Question: " + sized_text(pool, q_start, QUESTION_TOKENS, tok)
            abs_ts = conv_arrival + t * (row["think_ms"] + NOMINAL_DECODE_MS)
            record = {
                "timestamp_ms": abs_ts,
                "workload_type": "conversation",
                "body": {
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": question},
                    ],
                    "max_tokens": row["out_tokens_max"],
                    "temperature": 0,
                    "seed": 42,
                },
            }
            staged.append((abs_ts, record))

    staged.sort(key=lambda x: x[0])
    records = []
    for request_id, (_, rec) in enumerate(staged):
        records.append({"request_id": request_id, **rec})
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="data/input/trace_grading_public.jsonl")
    ap.add_argument("--corpus", default="data/input/trace-round1.jsonl")
    ap.add_argument("--model", default="../hf-model")
    ap.add_argument("--out", default="data/input/trace-round2.jsonl")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    spec = load_spec(args.spec)
    pool = load_corpus_ids(args.corpus, tok)
    records = build_records(spec, pool, tok)

    with open(args.out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # Report so the operator can sanity-check against the spec.
    lens = [sum(len(tok.encode(m["content"]).ids) for m in r["body"]["messages"]) for r in records]
    lens.sort()
    convs = {r["request_id"] for r in records}
    assert all(r["body"]["model"] == MODEL_NAME for r in records), "model name must be LFM2.5"
    assert len(records) == len(spec), "one record per spec turn"
    print(
        f"wrote {len(records)} records to {args.out}\n"
        f"  prompt tokens  min={lens[0]:,}  p50={lens[len(lens) // 2]:,}  max={lens[-1]:,}\n"
        f"  (spec target ~4,000/turn; {len(convs)} requests over "
        f"{len({(r['request_id']) for r in records})} ids)"
    )


def _self_check() -> None:
    """No model/IO: fake a tokenizer (1 token/char) and spec, exercise assembly + timing."""

    class FakeEncoding:
        def __init__(self, ids):
            self.ids = ids

    class FakeTok:
        def encode(self, text):
            return FakeEncoding([ord(c) for c in text])

        def decode(self, ids):
            return "".join(chr(i) for i in ids)

    tok = FakeTok()
    pool = list(range(65, 91)) * 2000  # A..Z repeated, plenty of tokens
    spec = [
        {"conv_id": 0, "turn_idx": 0, "timestamp_ms": 0, "think_ms": 3000,
         "in_tokens_est": 500, "out_tokens_max": 200},
        {"conv_id": 0, "turn_idx": 1, "timestamp_ms": 0, "think_ms": 3000,
         "in_tokens_est": 500, "out_tokens_max": 200},
        {"conv_id": 1, "turn_idx": 0, "timestamp_ms": 100, "think_ms": 3000,
         "in_tokens_est": 500, "out_tokens_max": 200},
    ]
    recs = build_records(spec, pool, tok)
    assert len(recs) == 3
    assert [r["request_id"] for r in recs] == [0, 1, 2], "sequential ids after sort"
    # conv 0 turn 0 arrives at 0; conv 0 turn 1 at 0 + (3000+2000) = 5000; conv 1 turn 0 at 100.
    assert [r["timestamp_ms"] for r in recs] == [0, 100, 5000], [r["timestamp_ms"] for r in recs]
    # Within conv 0, the system message (preamble+context) is identical across turns -> prefix reuse.
    c0 = [r for r in recs if r["body"]["messages"][0]["content"].find("session 0") >= 0]
    assert len(c0) == 2 and c0[0]["body"]["messages"][0]["content"] == c0[1]["body"]["messages"][0]["content"]
    # Different conversations have distinct contexts (distinct prefixes -> no cross-conv hits).
    sys0 = recs[0]["body"]["messages"][0]["content"]
    sys_conv1 = next(r for r in recs if "session 1" in r["body"]["messages"][0]["content"])
    assert sys0 != sys_conv1["body"]["messages"][0]["content"]
    # The user question varies across a conversation's turns.
    q = [r["body"]["messages"][1]["content"] for r in recs if "session 0" in r["body"]["messages"][0]["content"]]
    assert q[0] != q[1], "per-turn question must vary"
    assert all(r["body"]["model"] == MODEL_NAME for r in recs)
    print("build_trace_round2 self-check ok")


if __name__ == "__main__":
    import sys

    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
