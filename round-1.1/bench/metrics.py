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


def aggregate(records, wall_time):
    """Roll per-request records + wall clock into a summary dict.

    Reports the primary metric set (TTFT/ITL/E2E percentiles, tok/s, req/s).
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
        "itl_ms": _lat_stats(col("itl_mean")),
        "e2e_ms": _lat_stats(col("e2e")),
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
    print("metrics self-check OK")


if __name__ == "__main__":
    _selfcheck()
