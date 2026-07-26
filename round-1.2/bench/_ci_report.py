#!/usr/bin/env python3
"""Read bench-open.json + bench-closed-*.json, print format A (per-mode detail) + B (table).

Called by the bench CI workflow. Outputs human-readable tables to stdout and a
::VTL_BENCH:: JSON marker to stderr for `make ci-bench` to extract.
"""
from __future__ import annotations

import json
import sys
from glob import glob
from pathlib import Path

try:
    from metrics import aggregate
except ImportError:
    from bench.metrics import aggregate

# ERS constants from the scoring spec.
F_TTFT, C_TTFT = 10.0, 400.0  # ms
F_TPOT, C_TPOT = 1.0, 10.0    # ms
GAMMA = 2.0
W = 0.5


def _clamp(x, lo, hi):
    return max(lo, min(x, hi))


def _ers_one(ttft_ms, tpot_ms):
    if ttft_ms is None or tpot_ms is None:
        return 0.0
    x_ttft = _clamp((C_TTFT - ttft_ms) / (C_TTFT - F_TTFT), 0.0, 1.0)
    x_tpot = _clamp((C_TPOT - tpot_ms) / (C_TPOT - F_TPOT), 0.0, 1.0)
    s_ttft = x_ttft ** GAMMA
    s_tpot = x_tpot ** GAMMA
    return W * s_ttft + (1.0 - W) * s_tpot


def _ers(run):
    ok = [r for r in run["records"] if r.get("success") and r.get("ttft") and r.get("itl_mean")]
    if not ok:
        return 0.0
    return sum(_ers_one(r["ttft"] * 1000, r["itl_mean"] * 1000) for r in ok) / len(run["records"])


def _fmt(v, suffix=""):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.1f}{suffix}" if suffix in ("ms",) and v < 100 else f"{v:.0f}{suffix}"
    return str(v)


def _row(label, cells):
    return f"  {label:<13}" + "".join(f"{c:>10}" for c in cells)


def _load(path):
    run = json.loads(Path(path).read_text())
    run["_agg"] = aggregate(run["records"], run["wall_time"])
    run["_ers"] = _ers(run)
    return run


def main():
    # open-loop
    open_path = Path("bench-open.json")
    open_run = _load(open_path) if open_path.exists() else None

    # closed-loop sweep (sorted by concurrency)
    closed_runs = []
    for p in sorted(glob("bench-closed-*.json")):
        closed_runs.append(_load(p))

    runs = [r for r in [open_run, *closed_runs] if r is not None]
    if not runs:
        print("ERROR: no bench results found", file=sys.stderr)
        sys.exit(1)

    # --- Table B: all modes side by side ---
    headers = [r["mode"] for r in runs]
    aggs = [r["_agg"] for r in runs]
    ers = [r["_ers"] for r in runs]

    print()
    print("=== Bench Results (B) ===")
    print(_row("", headers))
    print("  " + "-" * (13 + 10 * len(runs)))
    for name, key in (("TTFT p50", "ttft_ms"), ("TTFT p99", "ttft_ms"),
                       ("TPOT p50", "itl_ms"), ("TPOT p99", "itl_ms")):
        cells = [_fmt(a[key][name.split()[-1]], "ms") for a in aggs]
        print(_row(name, cells))
    cells = [f"{a['output_tok_s']:.0f}" for a in aggs]
    print(_row("tok/s", cells))
    cells = [f"{a['req_s']:.2f}" for a in aggs]
    print(_row("req/s", cells))
    cells = [f"{a['n_success']}/{a['n_total']}" for a in aggs]
    print(_row("ok", cells))
    cells = [f"{e:.3f}" for e in ers]
    print(_row("ERS", cells))
    print()

    # --- Detail A: per-mode breakdown ---
    print("=== Bench Details (A) ===")
    for i, run in enumerate(runs):
        a = run["_agg"]
        e = run["_ers"]
        print(f"--- {headers[i]} ---")
        print(f"  ok {a['n_success']}/{a['n_total']}  "
              f"wall {a['wall_time_s']:.0f}s  "
              f"tok/s {a['output_tok_s']:.0f}  "
              f"req/s {a['req_s']:.2f}  "
              f"ERS {e:.3f}")
        ttft = a["ttft_ms"]
        print(f"  TTFT  p50:{_fmt(ttft['p50'],'ms')}  p90:{_fmt(ttft['p90'],'ms')}  "
              f"p95:{_fmt(ttft['p95'],'ms')}  p99:{_fmt(ttft['p99'],'ms')}  "
              f"mean:{_fmt(ttft['mean'],'ms')}")
        tpot = a["itl_ms"]
        print(f"  TPOT  p50:{_fmt(tpot['p50'],'ms')}  p90:{_fmt(tpot['p90'],'ms')}  "
              f"p95:{_fmt(tpot['p95'],'ms')}  p99:{_fmt(tpot['p99'],'ms')}  "
              f"mean:{_fmt(tpot['mean'],'ms')}")
        e2e = a["e2e_ms"]
        print(f"  E2E   p50:{_fmt(e2e['p50'],'ms')}  p90:{_fmt(e2e['p90'],'ms')}  "
              f"p95:{_fmt(e2e['p95'],'ms')}  p99:{_fmt(e2e['p99'],'ms')}  "
              f"mean:{_fmt(e2e['mean'],'ms')}")
        if run.get("spec_decode"):
            spec = run["spec_decode"]
            print(f"  spec  drafts:{spec.get('num_drafts',0)}  "
                  f"accept:{spec.get('acceptance_rate',0):.2f}  "
                  f"len:{spec.get('mean_accept_len',0):.2f}")
        print()

    # Marker for ci-bench extraction
    summary = {
        "modes": {
            r["mode"]: {"ers": r["_ers"], **{k: r["_agg"][k] for k in ("n_success", "n_total")}}
            for r in runs
        }
    }
    print(f"::VTL_BENCH::{json.dumps(summary)}")


if __name__ == "__main__":
    main()
