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


# ------------------------------------------------------------------------- ERS (the score)

# The judge's Effective Request Score. Mirrors HANDOFF.md "Scoring"; these are the judge's
# numbers, not ours -- do not tune them to make an arm look good.
#
#   s_ttft = clamp((C_ttft - TTFT) / (C_ttft - F_ttft), 0, 1) ** gamma
#   s_tpot = clamp((C_tpot - TPOT) / (C_tpot - F_tpot), 0, 1) ** gamma
#   S      = w * s_ttft + (1 - w) * s_tpot
#   ERS    = mean(S) over ALL requests, failures included at S = 0
#
# WHY THIS LIVES HERE. TTFT and TPOT move in opposite directions under almost every knob we
# sweep (chunking prefill trades one for the other), so an arm that improves both is rare and
# reading the two columns separately cannot rank arms. ERS is the only number that can.
# The exchange rate is steep and unintuitive: TPOT's window is 9 ms wide and TTFT's is 390 ms,
# so from a 5 ms TPOT / 100 ms TTFT operating point, **shaving 1 ms off TPOT is worth ~34 ms off
# TTFT** (pinned by the self-check). An arm that costs 20 ms of TTFT to buy 1 ms of TPOT wins.
ERS_PARAMS = {
    "f_ttft_ms": 10.0,
    "c_ttft_ms": 400.0,
    "f_tpot_ms": 1.0,
    "c_tpot_ms": 10.0,
    "gamma": 2.0,
    "w_ttft": 0.5,
}


def _component(value_ms, floor_ms, ceil_ms, gamma):
    """One clamped, gamma-shaped sub-score in [0, 1]. Higher is better."""
    frac = (ceil_ms - value_ms) / (ceil_ms - floor_ms)
    return (0.0 if frac < 0.0 else 1.0 if frac > 1.0 else frac) ** gamma


def score_request(ttft_ms, tpot_ms, params=None):
    """Per-request {s_ttft, s_tpot, score}. Both inputs in MILLISECONDS.

    ``ttft_ms=None`` means the request never produced a first token -- that is a failure, and
    the judge scores the whole request 0, not just the TTFT half.

    ``tpot_ms=None`` on an otherwise-successful request means there was no inter-token gap to
    measure (a single-token response; ``replay.py`` divides by ``n_tokens - 1``). There is no
    latency to penalise, so s_tpot is 1.0 -- but ``ers()`` counts these separately as
    ``n_no_tpot`` because a run full of them would show a great score for the wrong reason.
    Cannot occur on this trace (max_tokens is 200-300); the guard is for degenerate runs.
    """
    p = params or ERS_PARAMS
    if ttft_ms is None:
        return {"s_ttft": 0.0, "s_tpot": 0.0, "score": 0.0}
    s_ttft = _component(ttft_ms, p["f_ttft_ms"], p["c_ttft_ms"], p["gamma"])
    s_tpot = (
        1.0
        if tpot_ms is None
        else _component(tpot_ms, p["f_tpot_ms"], p["c_tpot_ms"], p["gamma"])
    )
    w = p["w_ttft"]
    return {"s_ttft": s_ttft, "s_tpot": s_tpot, "score": w * s_ttft + (1 - w) * s_tpot}


def ers(records, params=None):
    """ERS over a replay run's records. Latencies on records are in SECONDS.

    Denominator is every record, not just the successful ones: the judge averages over the
    requests it sent, so an error drags the mean down linearly. Reporting ERS over successes
    only would hide exactly the failure mode that costs the most.

    ``s_ttft_mean`` / ``s_tpot_mean`` are broken out so a moved ERS can be attributed. Do not
    read them as independently meaningful -- only ``ers`` is the score.
    """
    if not records:
        return None
    scored, no_tpot = [], 0
    for r in records:
        if not r.get("success") or not r.get("output_tokens"):
            scored.append({"s_ttft": 0.0, "s_tpot": 0.0, "score": 0.0})
            continue
        ttft, itl = r.get("ttft"), r.get("itl_mean")
        if itl is None:
            no_tpot += 1
        scored.append(
            score_request(
                None if ttft is None else ttft * 1000,
                None if itl is None else itl * 1000,
                params,
            )
        )
    return {
        "ers": fmean(s["score"] for s in scored),
        "s_ttft_mean": fmean(s["s_ttft"] for s in scored),
        "s_tpot_mean": fmean(s["s_tpot"] for s in scored),
        "n_scored": len(scored),
        "n_zero": sum(1 for s in scored if s["score"] == 0.0),
        "n_no_tpot": no_tpot,
    }


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
        # THE number to rank arms by. TTFT and TPOT alone cannot order two arms that trade one
        # against the other, which is most of them.
        "ers": ers(records),
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

    # ---- ERS ----------------------------------------------------------------
    # Endpoints: at/below the floor scores 1, at/above the ceiling scores 0.
    assert score_request(10.0, 1.0)["score"] == 1.0
    assert score_request(5.0, 0.5)["score"] == 1.0  # better than floor still clamps to 1
    assert score_request(400.0, 10.0)["score"] == 0.0
    assert score_request(9999.0, 9999.0)["score"] == 0.0
    # gamma=2 is what makes the tail hurt: halfway to the ceiling scores a QUARTER, not a half.
    mid = score_request(205.0, 5.5)  # midpoint of both windows
    assert abs(mid["s_ttft"] - 0.25) < 1e-9 and abs(mid["s_tpot"] - 0.25) < 1e-9, mid
    assert abs(mid["score"] - 0.25) < 1e-9
    # w=0.5: a perfect TTFT with a ceiling TPOT is worth exactly half.
    assert abs(score_request(10.0, 10.0)["score"] - 0.5) < 1e-9
    # No first token == the whole request scores 0, not just its TTFT half.
    assert score_request(None, 1.0) == {"s_ttft": 0.0, "s_tpot": 0.0, "score": 0.0}

    # THE EXCHANGE RATE, pinned so it cannot rot: from TPOT 5ms / TTFT 100ms, actually shaving
    # one millisecond off TPOT is worth ~34 ms off TTFT. This is why arms are ranked by ers and
    # not by reading the TTFT and TPOT columns. (The *instantaneous* derivative ratio there is
    # ~31; the finite 1 ms step is worth more because s_tpot is convex. 34 is the one that
    # matches what an arm delivers, so it is the one asserted.)
    base = score_request(100.0, 5.0)["score"]
    d_tpot = score_request(100.0, 4.0)["score"] - base  # 1 ms of TPOT
    d_ttft = score_request(99.0, 5.0)["score"] - base  # 1 ms of TTFT
    assert 33.0 < d_tpot / d_ttft < 36.0, d_tpot / d_ttft

    # Failures are in the denominator: one perfect request + one error == 0.5, never 1.0.
    two = ers([
        {"success": True, "ttft": 0.010, "itl_mean": 0.001, "output_tokens": 200},
        {"success": False, "error": "boom", "output_tokens": 0},
    ])
    assert two["ers"] == 0.5 and two["n_scored"] == 2 and two["n_zero"] == 1, two
    # A 200-response with zero tokens is a judge failure mode too, not a success.
    assert ers([{"success": True, "ttft": 0.01, "itl_mean": 0.001, "output_tokens": 0}])["ers"] == 0.0
    # Single-token response: no inter-token gap to penalise, but it is counted and reported.
    one = ers([{"success": True, "ttft": 0.010, "itl_mean": None, "output_tokens": 1}])
    assert one["ers"] == 1.0 and one["n_no_tpot"] == 1, one
    assert ers([]) is None
    assert agg["ers"]["n_scored"] == 3  # folded into aggregate(), failures included

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
