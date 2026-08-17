"""Convert an aiperf AgentX run into the repo's bench-run JSON schema.

    python bench/aiperf_adapter.py --artifact-dir bench-aiperf/<run>/ --out bench-aiperf.json
    python bench/aiperf_adapter.py --selfcheck

aiperf (`aiperf profile --scenario inferencex-agentx-mvp`, see `make bench-aiperf`) writes two
artifacts we care about: `profile_export.jsonl` -- one MetricRecordInfo object per request --
and `profile_export_aiperf.json`, aggregates only, which carries `submission_valid`. Per-request
records live ONLY in the JSONL, so that is what gets converted; the aggregate JSON is read just
for the validity verdict. The output is the same shape replay.py writes ({"records": [...],
"wall_time": s, "mode": str}), so metrics.aggregate and _ci_report's ERS work unchanged.

Field mapping (source names verified against the aiperf docs, 2026-08 -- re-verify against the
first real run and update this table if the schema moved):

    repo field         aiperf source                                     conversion
    ttft          (s)  metrics.time_to_first_token                  (ms) /1000
    itl_mean      (s)  metrics.inter_token_latency; fallback mean of
                       metrics.inter_chunk_latency (list)           (ms) /1000
    e2e           (s)  metrics.request_latency                      (ms) /1000
    output_tokens      metrics.output_token_count, fallback
                       metrics.output_sequence_length                    int
    success            error == null and not metadata.was_cancelled
    send_time     (s)  metadata.request_start_ns - min(start_ns)         /1e9
    wall_time     (s)  max(request_end_ns) - min(request_start_ns)       /1e9

Defensive by design, because the schema is confirmed-at-first-run: metric values may be plain
numbers or {"value": .., "unit": ..} objects (converted per unit when present); warmup-phase
records are dropped; a record with no error but also no TTFT is a FAILURE, never silently
skipped -- _ers divides by the full record count, so dropping it would inflate the score.

Pure stdlib on purpose (same rule as sweep_report.py): runs on the host, no aiperf install
needed -- it only parses files.
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import fmean

RECORDS_NAME = "profile_export.jsonl"
AGGREGATE_NAME = "profile_export_aiperf.json"
DEFAULT_MODE = "aiperf-agentx-c5"

# Divisors to seconds by declared unit. aiperf latency metrics default to ms; timestamps to ns.
_UNIT_TO_S = {"s": 1.0, "sec": 1.0, "seconds": 1.0, "ms": 1e3, "milliseconds": 1e3,
              "us": 1e6, "microseconds": 1e6, "ns": 1e9, "nanoseconds": 1e9}


def _raw(value):
    """(number, unit-or-None) from a plain number or a {"value","unit"} MetricValue object."""
    if isinstance(value, dict):
        return value.get("value"), value.get("unit")
    return value, None


def _seconds(value, default_unit="ms"):
    """A metric value as seconds, or None. Unknown units fall back to default_unit."""
    num, unit = _raw(value)
    if num is None:
        return None
    return float(num) / _UNIT_TO_S.get(unit or default_unit, _UNIT_TO_S[default_unit])


def _count(value):
    num, _ = _raw(value)
    return int(num) if num is not None else None


def _first(mapping, *keys):
    for k in keys:
        if mapping.get(k) is not None:
            return mapping[k]
    return None


def _is_warmup(meta):
    phase = str(meta.get("benchmark_phase", "")).lower()
    return bool(meta.get("is_warmup")) or "warmup" in phase


def convert_record(obj):
    """One repo-schema record (replay.py's do_request shape) from one MetricRecordInfo.

    None if the record is a warmup-phase row (excluded from scoring by the scenario).
    """
    metrics = obj.get("metrics") or {}
    meta = obj.get("metadata") or {}
    if _is_warmup(meta):
        return None

    error = obj.get("error")
    if isinstance(error, dict):
        error = ": ".join(str(error[k]) for k in ("type", "code", "message") if error.get(k))

    ttft = _seconds(metrics.get("time_to_first_token"))
    itl = _seconds(_first(metrics, "inter_token_latency", "inter_token_latency_avg"))
    if itl is None:
        chunks = metrics.get("inter_chunk_latency")
        if isinstance(chunks, list) and chunks:
            vals = [_seconds(c) for c in chunks]
            vals = [v for v in vals if v is not None]
            itl = fmean(vals) if vals else None
    e2e = _seconds(_first(metrics, "request_latency", "request_latency_avg"))
    out_tokens = _count(_first(metrics, "output_token_count", "output_sequence_length"))

    success = error is None and not meta.get("was_cancelled")
    if success and ttft is None:
        # No error reported but nothing was measured either -- score it as a failure rather
        # than dropping it (a dropped record silently inflates ERS via the denominator).
        success, error = False, "aiperf record has no time_to_first_token"
    if error is None and meta.get("was_cancelled"):
        error = "cancelled by aiperf"

    start_ns, _ = _raw(meta.get("request_start_ns"))
    end_ns, _ = _raw(meta.get("request_end_ns"))
    request_id = _first(meta, "x_request_id", "request_id")
    conv = meta.get("conversation_id")
    turn = meta.get("turn_index")
    if request_id is None and conv is not None:
        request_id = f"{conv}:{turn}"

    return {
        "request_id": request_id,
        "workload_type": "agentx",
        "send_time": None,  # filled in by convert_run once min(start_ns) is known
        "_start_ns": start_ns,
        "_end_ns": end_ns,
        "success": bool(success),
        "status": None,
        "error": error,
        "ttft": ttft,
        "e2e": e2e,
        "itl_mean": itl,
        "output_tokens": out_tokens or 0,
        "token_basis": "aiperf",
    }


def find_submission_valid(node):
    """Depth-first search for a `submission_valid` key; None if absent. The docs do not pin
    where in profile_export_aiperf.json it lives, so look everywhere."""
    if isinstance(node, dict):
        if "submission_valid" in node:
            return bool(node["submission_valid"])
        for v in node.values():
            found = find_submission_valid(v)
            if found is not None:
                return found
    elif isinstance(node, list):
        for v in node:
            found = find_submission_valid(v)
            if found is not None:
                return found
    return None


def convert_run(jsonl_lines, aggregate_obj=None, mode=DEFAULT_MODE):
    records = []
    for i, line in enumerate(jsonl_lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"line {i + 1} of {RECORDS_NAME} is not JSON: {exc}")
        rec = convert_record(obj)
        if rec is not None:
            records.append(rec)
    if not records:
        raise SystemExit(f"no non-warmup records in {RECORDS_NAME}")

    starts = [r["_start_ns"] for r in records if r["_start_ns"] is not None]
    ends = [r["_end_ns"] for r in records if r["_end_ns"] is not None]
    t0 = min(starts) if starts else None
    wall = (max(ends) - t0) / 1e9 if ends and t0 is not None else None
    for r in records:
        r["send_time"] = (r.pop("_start_ns") - t0) / 1e9 if t0 is not None and r["_start_ns"] is not None else r.pop("_start_ns")
        r.pop("_end_ns")
    if wall is None:
        # Degenerate export with no timestamps: fall back to the sum of e2e so tok/s stays
        # finite; the latency metrics (the scored ones) are unaffected.
        wall = sum(r["e2e"] or 0.0 for r in records)

    return {
        "target": None,
        "trace": "aiperf:inferencex-agentx-mvp",
        "mode": mode,
        "wall_time": wall,
        "spec_decode": None,
        "submission_valid": find_submission_valid(aggregate_obj) if aggregate_obj else None,
        "records": records,
    }


def _locate(artifact_dir, name):
    """The artifact-dir layout (flat vs timestamped subdirs) is confirmed-at-first-run, so
    accept the file at any depth; newest wins if aiperf wrote several."""
    root = Path(artifact_dir)
    if root.name == name and root.is_file():
        return root
    hits = sorted(root.rglob(name), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def main():
    p = argparse.ArgumentParser(description=f"Convert an aiperf artifact dir into replay.py's run schema.")
    p.add_argument("--artifact-dir", required=True, help="aiperf --artifact-dir output (any depth)")
    p.add_argument("--out", required=True, help="output JSON path (e.g. bench-aiperf.json)")
    p.add_argument("--mode-label", default=DEFAULT_MODE, help="run['mode'] for the report tables")
    args = p.parse_args()

    jsonl = _locate(args.artifact_dir, RECORDS_NAME)
    if jsonl is None:
        raise SystemExit(f"no {RECORDS_NAME} under {args.artifact_dir} -- not an aiperf artifact dir?")
    agg_path = _locate(args.artifact_dir, AGGREGATE_NAME)
    aggregate_obj = json.loads(agg_path.read_text()) if agg_path else None

    run = convert_run(jsonl.read_text().splitlines(), aggregate_obj, mode=args.mode_label)
    Path(args.out).write_text(json.dumps(run, indent=2))

    ok = sum(1 for r in run["records"] if r["success"])
    print(f"{jsonl} -> {args.out}: {ok}/{len(run['records'])} ok, wall {run['wall_time']:.0f}s",
          file=sys.stderr)
    if run["submission_valid"] is False:
        print("\n*** SUBMISSION INVALID: aiperf flagged this run (context overflow >1%, "
              "cancellation, or unsafe override). Scores from it are not comparable. ***\n",
              file=sys.stderr)
        sys.exit(2)
    if run["submission_valid"] is None and aggregate_obj is not None:
        print(f"warning: no submission_valid key found in {AGGREGATE_NAME}", file=sys.stderr)


def _selfcheck():
    from _ci_report import _ers

    def line(metrics=None, meta=None, error=None):
        return json.dumps({"metrics": metrics or {}, "metadata": meta or {}, "error": error})

    lines = [
        # plain-number ms metrics
        line({"time_to_first_token": 250.0, "inter_token_latency": 10.0,
              "request_latency": 3250.0, "output_token_count": 300},
             {"request_start_ns": 1_000_000_000, "request_end_ns": 4_250_000_000,
              "x_request_id": "r1"}),
        # MetricValue objects with explicit units (seconds), token count as object
        line({"time_to_first_token": {"value": 0.5, "unit": "s"},
              "inter_chunk_latency": [{"value": 20.0, "unit": "ms"}, {"value": 40.0, "unit": "ms"}],
              "request_latency": {"value": 9.5, "unit": "s"},
              "output_sequence_length": {"value": 100}},
             {"request_start_ns": 2_000_000_000, "request_end_ns": 11_500_000_000,
              "conversation_id": "c7", "turn_index": 3}),
        # errored request
        line({}, {"request_start_ns": 3_000_000_000},
             error={"type": "HTTPError", "code": 500, "message": "boom"}),
        # cancelled at benchmark end
        line({"time_to_first_token": 100.0}, {"was_cancelled": True}),
        # no error, no TTFT -> must become a failure, not vanish
        line({}, {"request_start_ns": 4_000_000_000}),
        # warmup row -> dropped entirely
        line({"time_to_first_token": 1.0}, {"benchmark_phase": "warmup"}),
    ]
    run = convert_run(lines, {"run_info": {"submission_valid": True}}, mode="test")

    assert run["mode"] == "test" and run["submission_valid"] is True
    assert len(run["records"]) == 5, len(run["records"])  # warmup dropped, failures kept
    r1, r2, r3, r4, r5 = run["records"]

    assert r1["success"] and (r1["ttft"], r1["itl_mean"], r1["e2e"]) == (0.25, 0.01, 3.25)
    assert r1["output_tokens"] == 300 and r1["request_id"] == "r1"
    assert r1["send_time"] == 0.0
    assert r2["success"] and r2["ttft"] == 0.5 and abs(r2["itl_mean"] - 0.03) < 1e-12
    assert r2["output_tokens"] == 100 and r2["request_id"] == "c7:3"
    assert r2["send_time"] == 1.0
    assert not r3["success"] and "HTTPError" in r3["error"] and "boom" in r3["error"]
    assert not r4["success"] and r4["error"] == "cancelled by aiperf"
    assert not r5["success"] and "no time_to_first_token" in r5["error"]
    assert abs(run["wall_time"] - 10.5) < 1e-9, run["wall_time"]  # 11.5s end - 1.0s start

    # ERS on the converted run: failures stay in the denominator. r1 is at the round-2 floor
    # on TPOT (10 ms -> x=90/92) but not TTFT (250 ms -> x=5750/5800); r2 scores lower.
    score = _ers(run)
    per = [0.5 * (5750 / 5800) ** 2 + 0.5 * (90 / 92) ** 2,
           0.5 * (5500 / 5800) ** 2 + 0.5 * (70 / 92) ** 2]
    assert abs(score - sum(per) / 5) < 1e-12, (score, sum(per) / 5)

    # metrics.aggregate accepts the converted records unchanged
    from metrics import aggregate
    agg = aggregate(run["records"], run["wall_time"])
    assert agg["n_total"] == 5 and agg["n_success"] == 2
    assert agg["ttft_ms"]["mean"] == (250.0 + 500.0) / 2

    # submission_valid search + invalid detection
    assert find_submission_valid({"a": [{"deep": {"submission_valid": False}}]}) is False
    assert find_submission_valid({"a": 1}) is None

    print("aiperf_adapter self-check OK")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--selfcheck"]:
        _selfcheck()
    else:
        main()
