#!/usr/bin/env python3
"""Byte-parity gate for output-changing options (flash attention, quantization).

The service reproduces llama-cli's greedy (temp=0) output byte-for-byte — a sacred
invariant. Any option that can perturb the logits must clear this gate before shipping:
capture full completions from a reference server and a candidate server, then diff.

    # server running with flash_attention: auto  (the reference)
    python bench/parity.py capture --target http://localhost:8000 --out ref.json
    # restart the server with flash_attention: on (the candidate)
    python bench/parity.py capture --target http://localhost:8000 --out on.json
    python bench/parity.py compare ref.json on.json   # exit 1 on any mismatch

Stdlib only (urllib, sequential) — parity cares about output, not timing, so it needs
no aiohttp/event loop. `self-test` checks the compare logic offline (no server).
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

# ponytail: 4 lines duplicated from replay.py rather than `import replay`, which drags in
# aiohttp/metrics at module load — parity needs neither.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACE = REPO_ROOT / "data" / "input" / "trace-round1.jsonl"


def load_trace(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def capture(target, trace, limit):
    """Replay the trace non-streamed and return {request_id: completion_text}."""
    records = load_trace(trace)
    if limit:
        records = records[:limit]
    out = {}
    for rec in records:
        body = dict(rec["body"])
        body["stream"] = False  # want the whole completion, not deltas
        req = urllib.request.Request(
            f"{target}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            obj = json.load(resp)
        out[str(rec.get("request_id"))] = obj["choices"][0]["message"]["content"]
    return out


def compare(a, b):
    """List request_ids whose completions differ. Raise if the two runs replayed
    different request sets (else a dropped request would masquerade as parity)."""
    if a.keys() != b.keys():
        only_a = sorted(a.keys() - b.keys())
        only_b = sorted(b.keys() - a.keys())
        raise SystemExit(f"request sets differ: only in A={only_a}, only in B={only_b}")
    return [rid for rid in a if a[rid] != b[rid]]


def _self_test():
    assert compare({"1": "x"}, {"1": "x"}) == []
    assert compare({"1": "x", "2": "y"}, {"1": "x", "2": "z"}) == ["2"]
    try:
        compare({"1": "x"}, {"2": "x"})
    except SystemExit:
        pass
    else:
        raise AssertionError("mismatched request sets must raise")
    print("parity self-test ok")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture", help="replay the trace and save completions")
    c.add_argument("--target", required=True, help="base URL, e.g. http://localhost:8000")
    c.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    c.add_argument("--out", required=True, help="output JSON path")
    c.add_argument("--limit", type=int, help="only replay the first N records")
    d = sub.add_parser("compare", help="diff two capture files; exit 1 on any mismatch")
    d.add_argument("a")
    d.add_argument("b")
    sub.add_parser("self-test", help="check the compare logic offline")
    args = p.parse_args()

    if args.cmd == "capture":
        out = capture(args.target, args.trace, args.limit)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"captured {len(out)} completions → {args.out}")
    elif args.cmd == "compare":
        a = json.loads(Path(args.a).read_text())
        b = json.loads(Path(args.b).read_text())
        diffs = compare(a, b)
        if diffs:
            print(f"BYTE-PARITY FAILED: {len(diffs)}/{len(a)} completions differ")
            for rid in diffs[:5]:
                print(f"  request {rid}: A={a[rid][:60]!r} B={b[rid][:60]!r}")
            sys.exit(1)
        print(f"byte-parity OK: {len(a)} completions identical")
    else:  # self-test
        _self_test()


if __name__ == "__main__":
    main()
