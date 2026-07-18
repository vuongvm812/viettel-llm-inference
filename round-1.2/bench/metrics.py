"""Aggregate per-request bench records into latency/throughput stats.

Pure functions only (no I/O) so the reporting logic has a runnable self-check:
    python bench/metrics.py
Latencies are stored in seconds on each record and reported in milliseconds.
"""

from statistics import fmean

PCTS = (50, 90, 95, 99)


def percentile(values, q):
    """Linear-interpolated percentile. q in [0, 100]. None for empty input."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (q / 100) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (rank - lo)


def _lat_stats(seconds):
    """{p50,p90,p95,p99,mean} in ms from a list of second-valued latencies."""
    ms = [v * 1000 for v in seconds]
    out = {f"p{q}": percentile(ms, q) for q in PCTS}
    out["mean"] = fmean(ms) if ms else None
    return out


def aggregate(records, wall_time, spec=None):
    """Roll per-request records + wall clock into a summary dict.

    Reports the primary metric set (TTFT/ITL/E2E percentiles, tok/s, req/s). `itl_ms` IS the
    per-output-token latency (TPOT) — the two are the same measurement. `spec` (optional) is a
    server-side spec-decode summary from spec_decode_stats(), folded in as `spec_decode`.
    ponytail: design.md's secondary "queue delay" (arrival → first server activity) is
    omitted — it's an approximation of TTFT minus a warm baseline, redundant with TTFT here.
    """
    ok = [r for r in records if r.get("success")]

    def col(key):
        return [r[key] for r in ok if r.get(key) is not None]

    total_out = sum(r.get("output_tokens", 0) for r in ok)
    return {
        "n_total": len(records),
        "n_success": len(ok),
        "n_error": len(records) - len(ok),
        "wall_time_s": wall_time,
        "total_output_tokens": total_out,
        "output_tok_s": total_out / wall_time if wall_time > 0 else 0.0,
        "req_s": len(ok) / wall_time if wall_time > 0 else 0.0,
        "ttft_ms": _lat_stats(col("ttft")),
        "itl_ms": _lat_stats(col("itl_mean")),  # == TPOT (time per output token)
        "e2e_ms": _lat_stats(col("e2e")),
        "spec_decode": spec,
    }


# -------------------------------------------------------------- spec-decode acceptance

# vLLM exposes these as Prometheus Counters; the client appends `_total`. Populated only when
# the server runs WITHOUT --disable-log-stats (the production compose disables it, so acceptance
# is n/a there — run a stats-enabled server for the A/B measurement).
SPEC_COUNTERS = {
    "num_drafts": "vllm:spec_decode_num_drafts_total",
    "num_draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "num_accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
}


def parse_spec_counters(prom_text):
    """Sum each spec-decode counter across its label series in Prometheus text. None if absent."""
    if not prom_text:
        return None
    wanted = {v: k for k, v in SPEC_COUNTERS.items()}
    totals = {k: 0.0 for k in SPEC_COUNTERS}
    seen = False
    for line in prom_text.splitlines():
        if not line or line[0] == "#":
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        key = wanted.get(name)
        if key is None:
            continue
        try:
            totals[key] += float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
        seen = True
    return totals if seen else None


def spec_decode_stats(before, after):
    """Deltas + acceptance from two parse_spec_counters snapshots. None if either is missing."""
    if not before or not after:
        return None
    d = {k: after[k] - before[k] for k in after}
    draft, drafts, acc = d["num_draft_tokens"], d["num_drafts"], d["num_accepted_tokens"]
    return {
        "num_drafts": drafts,
        "num_draft_tokens": draft,
        "num_accepted_tokens": acc,
        # fraction of drafted tokens the target accepted (higher = better drafter)
        "acceptance_rate": acc / draft if draft > 0 else None,
        # mean accepted draft tokens per decode step that drafted (excludes the bonus token)
        "mean_accept_len": acc / drafts if drafts > 0 else None,
    }


def _selfcheck():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4, 5], 0) == 1
    assert percentile([1, 2, 3, 4, 5], 100) == 5
    assert percentile([], 50) is None
    assert percentile([10, 20], 50) == 15  # interpolated midpoint

    recs = [
        {"success": True, "ttft": 0.1, "itl_mean": 0.01, "e2e": 2.0, "output_tokens": 200},
        {"success": True, "ttft": 0.3, "itl_mean": 0.03, "e2e": 4.0, "output_tokens": 100},
        {"success": False, "error": "boom", "output_tokens": 0},
    ]
    agg = aggregate(recs, wall_time=10.0)
    assert agg == {**agg, "n_total": 3, "n_success": 2, "n_error": 1}
    assert agg["total_output_tokens"] == 300
    assert agg["output_tok_s"] == 30.0
    assert agg["req_s"] == 0.2
    assert agg["ttft_ms"]["p50"] == 200.0  # midpoint of 100ms and 300ms
    assert agg["ttft_ms"]["mean"] == 200.0
    assert agg["spec_decode"] is None  # default: no spec stats

    # spec-decode counter parsing: sum across label series, ignore comments/other metrics
    prom = (
        "# HELP vllm:spec_decode_num_drafts_total drafts\n"
        '# TYPE vllm:spec_decode_num_drafts_total counter\n'
        'vllm:spec_decode_num_drafts_total{model_name="m",engine="0"} 100.0\n'
        'vllm:spec_decode_num_draft_tokens_total{model_name="m",engine="0"} 300.0\n'
        'vllm:spec_decode_num_accepted_tokens_total{model_name="m",engine="0"} 210.0\n'
        'vllm:other_metric_total{x="y"} 999.0\n'
    )
    snap = parse_spec_counters(prom)
    assert snap == {"num_drafts": 100.0, "num_draft_tokens": 300.0, "num_accepted_tokens": 210.0}, snap
    assert parse_spec_counters("# nothing here\nvllm:foo 1.0") is None  # no spec counters -> None
    assert parse_spec_counters("") is None

    # deltas: 300 drafted, 210 accepted over 100 draft-steps -> 0.70 accept, 2.1 tokens/draft
    before = {"num_drafts": 0.0, "num_draft_tokens": 0.0, "num_accepted_tokens": 0.0}
    st = spec_decode_stats(before, snap)
    assert abs(st["acceptance_rate"] - 0.70) < 1e-9 and abs(st["mean_accept_len"] - 2.1) < 1e-9, st
    assert spec_decode_stats(None, snap) is None  # stats disabled on server -> graceful None
    print("metrics self-check OK")


if __name__ == "__main__":
    _selfcheck()
