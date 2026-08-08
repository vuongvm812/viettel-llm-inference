"""Group a `make sweep-schedule` run by arm and say which arms actually beat the baseline.

    python bench/sweep_report.py bench-sched-*.json
    python bench/sweep_report.py --selfcheck

compare.py prints one column per FILE, all labelled `localhost:8000`. A schedule sweep is 11
arms x 3 boots = 33 indistinguishable columns, with a spread computed across boots AND arms
together -- exactly the comparison the "add boots, not reps" rule calls meaningless. This
groups by arm first and measures every delta against the BASELINE arm's own boot-to-boot
spread, because that floor is arm- and run-dependent: the 0.5 ms TPOT figure is folklore, and
the first boot of each schedule arm additionally pays an AOT/inductor recompile (flipping
VTL_W4A8_SCHEDULE changes the selected kernel, which invalidates /opt/vtl/cache).

Filenames come from the sweep-schedule make target: bench-sched-<arm>-b<boot>.json. One replay
per boot -- boots are the samples, reps are not (rep-to-rep is ~0.05 ms, boot-to-boot is ~0.5).

Modelled on bench/ab_summary.py from branch gnhf/goal-minimize-scored-7c79fe, minus its ERS
ranking: metrics.aggregate on this branch has no `ers` key, so the two scored terms are
reported directly (1 ms TPOT ~= 28 ms TTFT if you need to trade them off by hand).

Pure stdlib on purpose -- the dev box cannot import aiohttp, so anything needing the venv has
to run inside the image. Run this on the host.
"""

import json
import re
import sys
from pathlib import Path

from _ci_report import _ers
from metrics import aggregate

NAME_RE = re.compile(r"bench-sched-(?P<arm>.+)-b(?P<boot>\d+)\.json$")
BASELINE = "heuristic"  # the sentinel arm = kernel heuristic = what submissions ship

# (label, metrics.aggregate key, decimal places). p50 for both: the sweep asks which schedule
# is faster, and the tails are dominated by prefill scheduling, not by the tile choice.
METRICS = (("TPOT p50", "itl_ms", 3), ("TTFT p50", "ttft_ms", 1), ("ERS", "ers", 4))


def makefile_schedules(makefile):
    """``{'STOCK_SCHEDULES': [...], ...}`` from the root Makefile, one key per arm list.

    Backslash continuations are joined first -- the lists are wrapped across several lines, and a
    regex that stops at the newline would silently see half the arms.
    """
    text = makefile.read_text().replace("\\\n", " ")
    found = {}
    for name in ("STOCK_SCHEDULES", "V2_SCHEDULES", "V2_PREFILL_SCHEDULES"):
        m = re.search(rf"^{name}\s*[:?]?=\s*(.*)$", text, re.M)
        found[name] = m.group(1).split() if m else []
    return found


def parse_name(path):
    """('128x16_1x1x1', '2') from 'bench-sched-128x16_1x1x1-b2.json'. None if not a sweep file."""
    m = NAME_RE.search(Path(path).name)
    return (m["arm"], m["boot"]) if m else None


def mean(xs):
    return sum(xs) / len(xs)


def spread(xs):
    """Peak-to-peak. With 3 boots a stdev is a fiction; the range is what we actually saw."""
    return max(xs) - min(xs) if len(xs) > 1 else 0.0


def collect(paths):
    """(  {arm: [agg per boot]}, [skipped paths]  ) from replay.py output JSONs."""
    arms, skipped = {}, []
    for p in paths:
        key = parse_name(p)
        if not key:
            skipped.append(p)
            continue
        run = json.loads(Path(p).read_text())
        agg = aggregate(run["records"], run["wall_time"])
        # The scored quantity, so an arm that trades TPOT against TTFT can be ranked without
        # doing the 1 ms ~= 28 ms conversion by hand. `_ers` divides by the FULL record count,
        # so failed requests drag the arm down exactly as the scoring spec intends. Stuffed
        # under a "p50" key because that is the shape `summarize` reads; it is a mean, not a
        # median -- the label below says ERS, not ERS p50, for that reason.
        agg["ers"] = {"p50": _ers(run)}
        arms.setdefault(key[0], []).append(agg)
    return arms, skipped


def summarize(arms, baseline=BASELINE):
    """([(arm, n_boots, {key: (mean, spread, delta, inside_floor)})], {key: floor}).

    `delta` is vs the baseline arm's mean (None for the baseline itself, or when there is no
    baseline in the run). `inside_floor` True means the arm moved the metric by less than the
    baseline arm moved it just by rebooting -- not a result, whatever the mean says.
    """
    per = {
        arm: {key: [a[key]["p50"] for a in aggs if a[key]["p50"] is not None]
              for _, key, _ in METRICS}
        for arm, aggs in arms.items()
    }
    base = per.get(baseline)
    floors = {key: (spread(base[key]) if base and base[key] else None) for _, key, _ in METRICS}

    rows = []
    for arm in sorted(per, key=lambda a: (a != baseline, a)):  # baseline first, then alphabetical
        cells = {}
        for _, key, _ in METRICS:
            values = per[arm][key]
            if not values:
                cells[key] = None
                continue
            m = mean(values)
            delta = None if arm == baseline or not (base and base[key]) else m - mean(base[key])
            floor = floors[key]
            inside = None if delta is None or floor is None else abs(delta) <= floor
            cells[key] = (m, spread(values), delta, inside)
        rows.append((arm, len(arms[arm]), cells))
    return rows, floors


def render(rows, floors, baseline=BASELINE):
    print()
    print(f"  {'arm':<20}{'boots':>6}" + "".join(f"{label:>30}" for label, _, _ in METRICS))
    print("  " + "-" * (26 + 30 * len(METRICS)))
    for arm, boots, cells in rows:
        line = f"  {arm:<20}{boots:>6}"
        for _, key, places in METRICS:
            cell = cells[key]
            if cell is None:
                text = "—"
            else:
                m, sp, delta, inside = cell
                text = f"{m:.{places}f} ±{sp:.{places}f}"
                if delta is not None:
                    text += f" {delta:+.{places}f}{'~' if inside else '!'}"
            line += f"{text:>30}"
        print(line)
    print()
    if all(f is None for f in floors.values()):
        print(f"  no `{baseline}` arm in this run — deltas are unanchored. Boot the baseline too.")
        print()
        return
    floor_text = ", ".join(
        # ERS is a dimensionless score, not a duration -- the `ms` suffix is per-metric.
        f"{label} {floors[key]:.{places}f}{'' if key == 'ers' else 'ms'}"
        for label, key, places in METRICS
        if floors[key] is not None
    )
    print(f"  ± is peak-to-peak across boots. This run's own noise floor (the `{baseline}` arm's")
    print(f"  boot-to-boot spread): {floor_text}.")
    print("  ~ = delta is inside that floor: NOT a result — add boots (BOOTS=n) or drop the arm.")
    print("  ! = delta clears the floor. Rank those, and remember 1 ms TPOT ~= 28 ms TTFT.")
    print("  DIRECTION: TPOT/TTFT are lower-is-better, ERS is HIGHER-is-better — so a winning")
    print("  arm shows NEGATIVE TPOT/TTFT deltas and a POSITIVE ERS delta. ERS is the scored")
    print("  quantity; when it disagrees with TPOT, ERS wins.")
    print()


def main():
    # `--baseline NAME` because the sentinel arm is only called `heuristic` for the W4A8
    # schedule sweep; a topology sweep's baseline is the shipped config (`mp`). Everything
    # downstream already takes `baseline=` -- only this entry point hardcoded it.
    argv = sys.argv[1:]
    baseline = BASELINE
    paths = []
    i = 0
    while i < len(argv):
        if argv[i] == "--baseline":
            if i + 1 >= len(argv):
                sys.exit("--baseline needs an arm name")
            baseline, i = argv[i + 1], i + 2
            continue
        if argv[i].startswith("--baseline="):
            baseline, i = argv[i].split("=", 1)[1], i + 1
            continue
        paths.append(argv[i])
        i += 1
    if not paths:
        sys.exit("usage: python bench/sweep_report.py [--baseline ARM] bench-sched-*.json")
    arms, skipped = collect(paths)
    if skipped:
        print(f"  skipped (not a bench-sched-<arm>-b<n>.json): {skipped}")
    if not arms:
        sys.exit("  nothing to summarise")
    if baseline not in arms:
        # Not fatal -- the table is still readable -- but every delta and the noise floor
        # silently vanish, which reads like "no arm moved anything".
        print(f"  WARNING: baseline arm {baseline!r} not among {sorted(arms)}; "
              "no deltas and no noise floor will be computed")
    rows, floors = summarize(arms, baseline=baseline)
    render(rows, floors, baseline=baseline)


def _check_schedule_names():
    """The arm names live in two places -- the Makefile boots them, quant_w4a8 validates them.

    A name in only one list costs a whole boot and does not look like a bug: quant_w4a8 warns
    and falls back to the kernel heuristic, so the arm just quietly reports the baseline's
    numbers. Cheap to check here, since --selfcheck already runs off-box in `make check`.
    """
    here = Path(__file__).resolve()
    makefile = here.parents[2] / "Makefile"
    if not makefile.exists():
        print(f"  (schedule cross-check skipped: no {makefile})")
        return
    sys.path.insert(0, str(here.parents[1]))  # the round dir, so `vtl` imports off-box
    try:
        from vtl.patches.quant_w4a8 import (
            VALID_SCHEDULES,
            VALID_SCHEDULES_V2,
            VALID_SCHEDULES_V2_PREFILL,
        )
    except ImportError as exc:  # a bare bench/ checkout: report, don't fail
        print(f"  (schedule cross-check skipped: {exc})")
        return

    names = makefile_schedules(makefile)
    for key, valid in (("STOCK_SCHEDULES", VALID_SCHEDULES),
                       ("V2_SCHEDULES", VALID_SCHEDULES_V2),
                       ("V2_PREFILL_SCHEDULES", VALID_SCHEDULES_V2_PREFILL)):
        swept = names[key]
        assert swept, f"Makefile {key} is empty or unparsed -- the sweep would boot no arms"
        unknown = sorted(set(swept) - set(valid))
        assert not unknown, (
            f"Makefile {key} names {unknown}, which quant_w4a8 rejects: those boots would run "
            "the kernel heuristic and read as duplicate baselines"
        )


def _selfcheck():
    import tempfile

    _check_schedule_names()

    assert parse_name("bench-sched-128x16_1x1x1-b2.json") == ("128x16_1x1x1", "2")
    assert parse_name("x/y/bench-sched-heuristic-b1.json") == ("heuristic", "1")
    assert parse_name("bench-sched-128x16_1x1x1_sk-b3.json") == ("128x16_1x1x1_sk", "3")
    assert parse_name("bench-open.json") is None
    assert spread([1.0]) == 0.0 and spread([1.0, 1.5, 1.2]) == 0.5

    def write(d, arm, boot, itl_s, ttft_s):
        rec = {"success": True, "ttft": ttft_s, "itl_mean": itl_s, "e2e": 1.0,
               "output_tokens": 10}
        p = Path(d) / f"bench-sched-{arm}-b{boot}.json"
        p.write_text(json.dumps({"records": [rec], "wall_time": 1.0}))
        return str(p)

    with tempfile.TemporaryDirectory() as d:
        paths = []
        # baseline: 4.0/4.2/4.1 ms TPOT -> mean 4.1, boot-to-boot floor 0.2 ms
        for boot, itl in enumerate((0.0040, 0.0042, 0.0041), start=1):
            paths.append(write(d, "heuristic", boot, itl, 0.060))
        # a real win: -0.6 ms, well outside the 0.2 ms floor
        for boot in (1, 2, 3):
            paths.append(write(d, "128x16_1x1x1", boot, 0.0035, 0.060))
        # noise: -0.05 ms, inside the floor
        for boot in (1, 2, 3):
            paths.append(write(d, "128x32_1x1x1_sk", boot, 0.00405, 0.060))
        paths.append(str(Path(d) / "bench-open.json"))  # unrelated file must be skipped, not read

        arms, skipped = collect(paths)
        assert skipped == [paths[-1]], skipped
        assert {a: len(v) for a, v in arms.items()} == {
            "heuristic": 3, "128x16_1x1x1": 3, "128x32_1x1x1_sk": 3}, arms

        rows, floors = summarize(arms)
        assert [r[0] for r in rows] == ["heuristic", "128x16_1x1x1", "128x32_1x1x1_sk"], rows
        assert abs(floors["itl_ms"] - 0.2) < 1e-9, floors
        assert floors["ttft_ms"] == 0.0, floors  # identical TTFT across boots

        base, win, null = rows
        assert base[2]["itl_ms"][2] is None                     # baseline has no delta
        assert abs(base[2]["itl_ms"][0] - 4.1) < 1e-9, base
        assert abs(win[2]["itl_ms"][2] - -0.6) < 1e-9, win      # -0.6 ms vs a 0.2 ms floor
        assert win[2]["itl_ms"][3] is False                     # clears it
        assert abs(null[2]["itl_ms"][2] - -0.05) < 1e-9, null
        assert null[2]["itl_ms"][3] is True                     # inside the floor -> flagged

        render(rows, floors)                                    # must not raise
        # no baseline booted -> deltas unanchored, still renders
        arms.pop("heuristic")
        rows, floors = summarize(arms)
        assert all(f is None for f in floors.values()), floors
        render(rows, floors)

    print("sweep_report self-check OK")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--selfcheck"]:
        _selfcheck()
    else:
        main()
