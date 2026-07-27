"""Print a side-by-side ours-vs-vLLM table from two replay.py output JSONs.

    python bench/compare.py vllm.json ours.json
"""

import json
import sys
from pathlib import Path

from metrics import aggregate, score_request

FAIRNESS_NOTES = """\
Reading this table:
  - Rank by ERS, not by the TTFT/TPOT columns. Most knobs we sweep trade one against the
    other, so the columns routinely disagree about which arm won. From TPOT 5ms / TTFT 100ms,
    shaving 1 ms off TPOT is worth ~34 ms off TTFT (pinned in metrics.py's self-check).
  - The noise floor is BOOT-TO-BOOT (~0.5 ms TPOT), not rep-to-rep (~0.05 ms). Two arms that
    differ by less than that are indistinguishable no matter how many requests each replayed.
    Add boots, not reps. One boot per arm is one sample, not a result.
  - Arms must differ in ONE thing. Every arm here should come from the same image digest and
    the same compose, overridden by exactly the flag under test (see `make arm`). A rebuilt or
    re-pinned image between arms is the one confounder this harness cannot detect.
  - `scored 0` counts requests the judge would score zero — errors, timeouts, empty outputs.
    They are invisible in the latency percentiles (a failed request has no TTFT to land in a
    p99) yet each costs a full 1/N. An arm that looks fast may just have answered fewer.
    A trailing `!` means some request had no measurable TPOT (single-token output).
  - Token count basis: output tok/s counts one streamed SSE content chunk = one token. A
    stream's usage.completion_tokens is used instead when present — mixing the two bases
    across runs is flagged below. Note ITL is always chunk-based, so it does NOT survive
    --stream-interval > 1; see the warning in HANDOFF.md before setting that flag."""


def _token_basis(run):
    """Which token-count basis a run used ('chunks', 'usage', or 'mixed')."""
    bases = {r.get("token_basis") for r in run["records"] if r.get("success")}
    bases.discard(None)
    if not bases:
        return "chunks"  # nothing succeeded; default
    return bases.pop() if len(bases) == 1 else "mixed"


def load(path):
    run = json.loads(Path(path).read_text())
    run["_agg"] = aggregate(run["records"], run["wall_time"])
    return run


def _fmt(v, suffix="", places=0):
    return "—" if v is None else f"{v:.{places}f}{suffix}"


def _row(label, cells):
    return f"  {label:<20}" + "".join(f"{c:>18}" for c in cells)


def _fmt_ers(e, key="ers"):
    return "—" if not e else f"{e[key]:.4f}"


BOOT_NOISE_TPOT_MS = 0.5  # measured boot-to-boot spread; rep-to-rep is ~0.05 ms


def _noise_floor_ers(agg):
    """What BOOT_NOISE_TPOT_MS of TPOT is worth in ERS at THIS run's operating point.

    Not a constant: s_tpot is quadratic in TPOT, so the same half-millisecond is worth ~0.020
    ERS at a 6.5 ms TPOT and ~0.036 at 4 ms. Returns None if the run has no usable latencies.
    """
    ttft, tpot = agg["ttft_ms"]["mean"], agg["itl_ms"]["mean"]
    if ttft is None or tpot is None:
        return None
    return (
        score_request(ttft, tpot)["score"]
        - score_request(ttft, tpot + BOOT_NOISE_TPOT_MS)["score"]
    )


def _fmt_zero(e):
    """Requests that scored 0. Called out because they are invisible in the latency
    percentiles -- a failed request has no TTFT to appear in a p99 -- yet each one costs a
    full 1/N of the score. An arm that looks fast here may simply have answered fewer."""
    if not e:
        return "—"
    return f"{e['n_zero']}/{e['n_scored']}" + ("" if not e["n_no_tpot"] else " !")


def print_table(runs):
    headers = [r["target"].split("://", 1)[-1] for r in runs]
    aggs = [r["_agg"] for r in runs]

    print()
    print(_row("", headers))
    print("  " + "-" * (20 + 18 * len(runs)))
    # TPOT (time-per-output-token) == mean inter-token latency (itl_ms); E2E == end-to-end latency.
    # TPOT gets 2 decimals: its whole scoring window is 9 ms wide (floor 1, ceiling 10), so
    # rounding to the nearest millisecond throws away ~1/9 of the score per digit -- a 0.5 ms
    # boot-to-boot difference would print as 0 or 1. TTFT's window is 390 ms; 1 ms is noise there.
    for name, key, places in (
        ("TTFT", "ttft_ms", 0),
        ("TPOT", "itl_ms", 2),
        ("E2E", "e2e_ms", 0),
    ):
        for pct in ("p50", "p99"):
            cells = [_fmt(a[key][pct], "ms", places) for a in aggs]
            print(_row(f"{name} {pct}", cells))
    # The tail is what gamma=2 punishes, and p99 alone can hide a bimodal ITL distribution.
    print(_row("TPOT mean", [_fmt(a["itl_ms"]["mean"], "ms", 2) for a in aggs]))
    # ERS first among the summary rows: it is the only one that can ORDER two arms. TTFT and
    # TPOT trade against each other under most knobs we sweep, so the columns above routinely
    # disagree about which arm won. s_ttft/s_tpot are shown to attribute a move, not to rank.
    print(_row("ERS", [_fmt_ers(a["ers"]) for a in aggs]))
    print(_row("  s_ttft", [_fmt_ers(a["ers"], "s_ttft_mean") for a in aggs]))
    print(_row("  s_tpot", [_fmt_ers(a["ers"], "s_tpot_mean") for a in aggs]))
    print(_row("  scored 0", [_fmt_zero(a["ers"]) for a in aggs]))
    print(_row("output tok/s", [f"{a['output_tok_s']:.1f}" for a in aggs]))
    print(_row("req/s", [f"{a['req_s']:.2f}" for a in aggs]))
    print(_row("success", [f"{a['n_success']}/{a['n_total']}" for a in aggs]))
    print(_row("errors", [str(a["n_error"]) for a in aggs]))
    print(_row("mode", [r["mode"] for r in runs]))
    print()

    bases = [_token_basis(r) for r in runs]
    if len(set(bases)) > 1:
        print("  WARNING: runs used different token-count bases " + str(dict(zip(headers, bases)))
              + " — tok/s is NOT comparable (usage tokens vs SSE chunks). Align the bases.")
        print()

    modes = {r["mode"] for r in runs}
    if len(modes) > 1:
        print(f"  WARNING: runs used different replay modes {sorted(modes)} — open-loop and "
              "closed-loop\n           impose different arrival patterns, so ERS is NOT "
              "comparable across them.")
        print()

    best = max(
        (a for a in aggs if a["ers"]), key=lambda a: a["ers"]["ers"], default=None
    )
    if best is not None and len(aggs) > 1:
        spread = best["ers"]["ers"] - min(a["ers"]["ers"] for a in aggs if a["ers"])
        print(f"  Best ERS: {headers[aggs.index(best)]} ({best['ers']['ers']:.4f}), "
              f"spread {spread:.4f} across {len(aggs)} arms.")
        # One boot per arm cannot resolve a difference smaller than the boot-to-boot floor.
        # Convert that floor (~0.5 ms of TPOT) into ERS *at this run's operating point* rather
        # than hardcoding it: s_tpot is quadratic, so the same 0.5 ms is worth 0.020 ERS at a
        # 6.5 ms TPOT and 0.036 at 4 ms. A fixed threshold would call real wins null at one end
        # and noise significant at the other.
        floor = _noise_floor_ers(best)
        if floor is not None and spread < floor:
            print(f"  ...which is INSIDE the boot-to-boot noise floor (0.5 ms TPOT ≈ "
                  f"{floor:.4f} ERS here).\n     This is a null result, not a winner. "
                  "Add boots before believing it.")
        print()
    print(FAIRNESS_NOTES)


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python bench/compare.py <run1.json> [run2.json ...]")
    runs = [load(p) for p in sys.argv[1:]]
    print_table(runs)


if __name__ == "__main__":
    main()
