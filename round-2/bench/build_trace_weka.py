#!/usr/bin/env python3
"""Build ``data/input/trace-weka.jsonl`` — a replay.py trace derived from the REAL grading
corpus (HF ``semianalysisai/cc-traces-weka-062126``, the SemiAnalysis Weka Claude Code
sessions that aiperf's locked ``inferencex-agentx-mvp`` scenario replays for grading).

Why this exists: ``trace-round2.jsonl`` (the authored synthetic trace) predates the BTC
round-2 grading spec and no longer matches it (HANDOFF §6.3). `make warm` bakes compile
caches and PGO/BOLT trains the Rust frontend on whatever trace they replay — those two
flows must see grading-shaped traffic. `make bench` / CI stay on the synthetic trace.

The corpus carries NO text. Each session is a timeline of model requests
``{t, model, in, out, hash_ids, think_time, ...}`` where ``hash_ids`` is the prompt as a
sequence of 64-token KV-cache block hashes: equal ids mean byte-identical 64-token
blocks, so the ids encode the entire prefix-reuse structure. We synthesize deterministic
text per hash id (same id -> same text, across the whole trace), which reproduces:

  * true input sizes           (64 tokens per block, tokenizer-exact when one is given)
  * true prefix-cache topology (shared hash prefix -> shared prompt byte prefix)
  * true output lengths        (``max_tokens`` = recorded ``out``, ``ignore_eos`` like grading)
  * true arrival shape         (recorded think-time gaps, capped at the scenario's 10 s
                                idle guard so a replay never sits idle for hours)

Subagent groups flatten into the same timeline as parallel requests (their inner
requests carry absolute session timestamps). ``--sessions N`` session-trees start at
t=0 together — the standing approximation of the scenario's "N trees live at all times".
Traces whose peak ``in+out`` exceeds ``--max-context`` are dropped whole, mirroring
aiperf's ``--max-context-length`` trace filter (grading: 204,800).

    python bench/build_trace_weka.py                  # stream from HF, write the trace
    python bench/build_trace_weka.py --corpus /path/to/traces.jsonl
    python bench/build_trace_weka.py --tokenizer ../hf-model   # tokenizer-exact blocks
    python bench/build_trace_weka.py --self-check     # no network, no tokenizer

The output is generated, not committed (gitignored): run `make trace-weka`.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "input" / "trace-weka.jsonl"
CORPUS_URL = (
    "https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126"
    "/resolve/main/traces.jsonl"
)
BLOCK_TOKENS = 64  # corpus invariant: in == 64 * len(hash_ids)

# Common English words -> mostly one BPE token each with a leading space, so the
# no-tokenizer fallback of BLOCK_TOKENS words per block lands near 64 real tokens.
_WORDS = (
    "the of and to in is that it for on with as at by from up about into over after "
    "file line code test build run make change read write open close list find call "
    "return value name path type case block start end next first last new old good "
    "data set get put add remove check point work part time way day use place right "
    "small large long short high low own same other each which their there when where "
    "result error state count order group number system server request output input"
).split()


_BLOCK_CACHE: dict = {}


def _block_text(hash_id, tokenizer=None):
    """Deterministic ~64-token text for one KV block. Same hash id -> same text.
    Memoized: blocks recur constantly (that is the prefix-cache structure), and the
    tokenizer-exact path re-encodes on every miss."""
    cached = _BLOCK_CACHE.get(hash_id)
    if cached is not None:
        return cached
    rng = random.Random(f"weka-block:{hash_id}")
    words = rng.choices(_WORDS, k=BLOCK_TOKENS)
    text = " ".join(words)
    if tokenizer is not None:
        ids = tokenizer.encode(" " + text, add_special_tokens=False).ids
        if len(ids) > BLOCK_TOKENS:
            text = tokenizer.decode(ids[:BLOCK_TOKENS]).strip()
        else:  # pad with extra words until exactly BLOCK_TOKENS, then cut
            while len(ids) < BLOCK_TOKENS:
                text += " " + rng.choice(_WORDS)
                ids = tokenizer.encode(" " + text, add_special_tokens=False).ids
            text = tokenizer.decode(ids[:BLOCK_TOKENS]).strip()
    _BLOCK_CACHE[hash_id] = text
    return text


def _model_requests(trace):
    """Flatten one session: main-stream requests + subagent inner requests, sorted by
    absolute session time. Subagent inner ``t`` is absolute (matches the group's) in the
    corpus; guard for a relative encoding anyway."""
    out = []
    for r in trace["requests"]:
        if r.get("type") == "subagent":
            g_t = r.get("t", 0.0)
            for inner in r.get("requests", []):
                t = inner.get("t", g_t)
                if t < g_t - 1.0:  # looks relative to the group start
                    t = g_t + t
                out.append(dict(inner, t=t))
        elif "hash_ids" in r:
            out.append(r)
    out.sort(key=lambda r: r["t"])
    return out


def _peak_context(reqs):
    return max((r["in"] + r["out"] for r in reqs), default=0)


def _compress_gaps(reqs, idle_cap_s):
    """Recorded sessions span hours of human think time. Rebuild arrivals with every
    inter-arrival gap capped (the scenario's global idle guard, approximated), keeping
    simultaneous requests simultaneous. Returns [(arrival_s, req), ...]."""
    out = []
    arrival = 0.0
    prev_t = None
    for r in reqs:
        if prev_t is not None:
            arrival += min(max(r["t"] - prev_t, 0.0), idle_cap_s)
        prev_t = r["t"]
        out.append((arrival, r))
    return out


def _build_records(traces, sessions, max_requests, idle_cap_s, model, tokenizer, ignore_eos):
    """Interleave the selected sessions (all starting at t=0) into replay.py records."""
    merged = []
    for si, trace in enumerate(traces[:sessions]):
        for arrival, r in _compress_gaps(_model_requests(trace), idle_cap_s):
            merged.append((arrival, si, r))
    merged.sort(key=lambda x: (x[0], x[1]))
    merged = merged[:max_requests]

    records = []
    for i, (arrival, _si, r) in enumerate(merged):
        blocks = [_block_text(h, tokenizer) for h in r["hash_ids"]]
        messages = []
        if len(blocks) > 1:
            messages.append({"role": "system", "content": "\n".join(blocks[:-1])})
        messages.append({"role": "user", "content": blocks[-1]})
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max(int(r["out"]), 1),
        }
        if ignore_eos:
            body["ignore_eos"] = True  # grading runs with ignore_eos:true (full decodes)
        records.append(
            {
                "request_id": i,
                "timestamp_ms": int(arrival * 1000),
                "workload_type": "weka",
                "body": body,
            }
        )
    return records


def _iter_corpus(corpus):
    """Yield parsed traces. ``corpus`` is a local path, or None to stream from HF —
    streaming lets the caller stop early without downloading the full 1.85 GB file."""
    if corpus:
        with open(corpus) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return
    req = urllib.request.Request(CORPUS_URL)
    with urllib.request.urlopen(req) as resp:
        for line in io.TextIOWrapper(resp, encoding="utf-8"):
            if line.strip():
                yield json.loads(line)


def _select_traces(corpus, max_context, pool_size, seed):
    """First ``pool_size`` corpus traces passing the peak-context filter (corpus order is
    stable, so this is deterministic), then a seeded shuffle picks the session order."""
    pool = []
    scanned = 0
    for trace in _iter_corpus(corpus):
        scanned += 1
        reqs = _model_requests(trace)
        if not reqs:
            continue
        if _peak_context(reqs) <= max_context:
            pool.append(trace)
            if len(pool) >= pool_size:
                break
    print(f"scanned {scanned} traces, pool of {len(pool)} within max-context")
    random.Random(seed).shuffle(pool)
    return pool


def _load_tokenizer(path):
    from tokenizers import Tokenizer

    p = Path(path)
    tok_file = p / "tokenizer.json" if p.is_dir() else p
    return Tokenizer.from_file(str(tok_file))


# ---------------------------------------------------------------- self-check


def _fake_trace(trace_id, reqs):
    return {"id": trace_id, "requests": reqs}


def _fake_req(t, hash_ids, out=10, typ="s"):
    return {"t": t, "type": typ, "model": "m", "in": 64 * len(hash_ids), "out": out,
            "hash_ids": hash_ids}


def self_check():
    # 1. peak-context filter drops the oversized trace whole
    small = _fake_trace("a", [_fake_req(0.0, [1, 2]), _fake_req(5.0, [1, 2, 3])])
    big = _fake_trace(
        "b", [_fake_req(0.0, [9]), _fake_req(1.0, list(range(100, 100 + 4000)))]
    )  # 4000 blocks = 256k tokens > cap
    kept = [t for t in (small, big) if _peak_context(_model_requests(t)) <= 204_800]
    assert [t["id"] for t in kept] == ["a"], "peak-context filter"

    # 2. subagent inner requests flatten into the timeline at absolute time
    with_sub = _fake_trace(
        "c",
        [
            _fake_req(0.0, [1]),
            {"t": 50.0, "type": "subagent",
             "requests": [_fake_req(50.0, [7], typ="n"), _fake_req(60.0, [7, 8], typ="n")]},
            _fake_req(100.0, [1, 2]),
        ],
    )
    flat = _model_requests(with_sub)
    assert [r["t"] for r in flat] == [0.0, 50.0, 60.0, 100.0], "flatten order"

    # 3. gap cap: a 3600 s think pause becomes idle_cap; simultaneity is preserved
    reqs = [_fake_req(0.0, [1]), _fake_req(0.0, [2]), _fake_req(3600.0, [3])]
    comp = _compress_gaps(reqs, idle_cap_s=10.0)
    assert comp[0][0] == comp[1][0] == 0.0 and comp[2][0] == 10.0, "gap cap"

    # 4. records: schema, sorted timestamps, shared hash prefix -> shared text prefix,
    #    max_tokens from `out`, determinism
    tr = _fake_trace("d", [_fake_req(0.0, [1, 2, 3], out=5), _fake_req(9.0, [1, 2, 3, 4], out=0)])
    recs1 = _build_records([tr], 1, 100, 10.0, "M", None, True)
    recs2 = _build_records([tr], 1, 100, 10.0, "M", None, True)
    assert recs1 == recs2, "determinism"
    assert [sorted(r) for r in recs1] == [["body", "request_id", "timestamp_ms", "workload_type"]] * 2
    assert recs1[0]["timestamp_ms"] <= recs1[1]["timestamp_ms"], "sorted arrivals"
    sys0 = recs1[0]["body"]["messages"][0]["content"]
    sys1 = recs1[1]["body"]["messages"][0]["content"]
    assert sys1.startswith(sys0), "shared hash prefix must be a shared byte prefix"
    assert recs1[0]["body"]["max_tokens"] == 5 and recs1[1]["body"]["max_tokens"] == 1
    assert all(r["body"]["ignore_eos"] is True for r in recs1)

    # 5. block text: deterministic per id, distinct across ids, ~64 words fallback
    assert _block_text(42) == _block_text(42) and _block_text(42) != _block_text(43)
    assert len(_block_text(42).split()) == BLOCK_TOKENS

    print("self-check OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--corpus", help="local traces.jsonl (default: stream from HF)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--model", default="Qwen3.5-122B-A10B-FP8")
    ap.add_argument("--tokenizer", help="HF model dir (tokenizer.json) for exact 64-token blocks")
    ap.add_argument("--sessions", type=int, default=5,
                    help="concurrent session-trees (grading: concurrency 5)")
    ap.add_argument("--pool", type=int, default=15,
                    help="qualifying traces to collect before the seeded pick")
    ap.add_argument("--max-requests", type=int, default=400)
    ap.add_argument("--max-context", type=int, default=204_800,
                    help="drop traces whose peak in+out exceeds this (grading cap)")
    ap.add_argument("--idle-cap", type=float, default=10.0,
                    help="cap inter-arrival gaps at this many seconds (scenario idle guard)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ignore-eos", action="store_true",
                    help="omit ignore_eos:true (if a frontend rejects the extra field)")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        self_check()
        return

    tokenizer = _load_tokenizer(args.tokenizer) if args.tokenizer else None
    if tokenizer is None:
        print("NOTE: no --tokenizer; block sizing falls back to 64 common words/block")

    traces = _select_traces(args.corpus, args.max_context, args.pool, args.seed)
    if len(traces) < args.sessions:
        sys.exit(f"FAIL: only {len(traces)} traces within --max-context, need {args.sessions}")
    records = _build_records(traces, args.sessions, args.max_requests, args.idle_cap,
                             args.model, tokenizer, not args.no_ignore_eos)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    n_blocks = [
        sum(len(m["content"].split("\n")) if m["role"] == "system" else 1
            for m in r["body"]["messages"])
        for r in records
    ]
    est_in = sorted(b * BLOCK_TOKENS for b in n_blocks)
    n = len(records)
    span = records[-1]["timestamp_ms"] / 1000 if records else 0
    print(f"wrote {n} requests to {out} ({out.stat().st_size / 1e6:.1f} MB), "
          f"span {span:.0f}s, est input tokens: "
          f"p50 {est_in[n // 2]} p90 {est_in[int(n * 0.9)]} max {est_in[-1]}")


if __name__ == "__main__":
    main()
