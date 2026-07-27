"""Group `make ab` boots by arm and say whether the arm delta clears the boot-to-boot floor.

    python bench/ab_summary.py bench-ab-*.json

compare.py prints one column per FILE, all labelled `localhost:8000`. With 3 boots per arm
that is six indistinguishable columns and a spread computed across boots and arms together --
which is exactly the comparison the "add boots, not reps" rule says is meaningless. This
groups by arm first, so the arm delta is measured against the WITHIN-ARM boot spread that the
run itself observed, instead of against the hardcoded 0.5 ms folklore number.

Filenames come from the `ab` make target: bench-ab-<arm>-r<round>-<rep>.json. Reps inside one
boot are averaged (they are ~0.05 ms apart); boots are the samples.

Pure stdlib on purpose -- the dev box cannot import aiohttp, so anything needing the venv has
to run inside the image. Run this on the host.
"""

import json
import re
import sys
from pathlib import Path

from metrics import aggregate

NAME_RE = re.compile(r"bench-ab-(?P<arm>.+)-r(?P<round>[^-]+)-(?P<rep>[^-]+)\.json$")


def parse_name(path):
    """('2048', '1') from 'bench-ab-2048-r1-2.json'. None if it is not an ab file."""
    m = NAME_RE.search(Path(path).name)
    return (m["arm"], m["round"]) if m else None


def mean(xs):
    return sum(xs) / len(xs)


def group_boots(samples):
    """[(arm, boot, value)] -> {arm: [per-boot value]}, reps within a boot averaged."""
    by_boot = {}
    for arm, boot, value in samples:
        by_boot.setdefault((arm, boot), []).append(value)
    arms = {}
    for (arm, _boot), values in by_boot.items():
        arms.setdefault(arm, []).append(mean(values))
    return arms


def spread(xs):
    """Peak-to-peak. With 3 boots a stdev is a fiction; the range is what we actually saw."""
    return max(xs) - min(xs) if len(xs) > 1 else 0.0


def verdict(arms):
    """(best_arm, delta, floor, is_win). floor = the largest within-arm boot spread observed.

    A delta smaller than the noise a single arm produced by rebooting is not a result, no
    matter how clean the means look.
    """
    means = {a: mean(v) for a, v in arms.items()}
    best = max(means, key=means.get)
    delta = means[best] - min(means.values())
    floor = max(spread(v) for v in arms.values())
    return best, delta, floor, delta > floor


def main():
    paths = sys.argv[1:]
    if not paths:
        sys.exit("usage: python bench/ab_summary.py bench-ab-*.json")

    rows, skipped = [], []
    for p in paths:
        key = parse_name(p)
        if not key:
            skipped.append(p)
            continue
        run = json.loads(Path(p).read_text())
        agg = aggregate(run["records"], run["wall_time"])
        if not agg["ers"]:
            skipped.append(p)
            continue
        rows.append((key, agg))

    if skipped:
        print(f"  skipped (not a bench-ab-<arm>-r<n>-<i>.json, or nothing scored): {skipped}")
    if not rows:
        sys.exit("  nothing to summarise")

    metrics = [
        ("ERS", lambda a: a["ers"]["ers"], 4, ""),
        ("TTFT p50", lambda a: a["ttft_ms"]["p50"], 0, "ms"),
        ("TPOT mean", lambda a: a["itl_ms"]["mean"], 2, "ms"),
        ("TPOT p99", lambda a: a["itl_ms"]["p99"], 2, "ms"),
    ]
    arm_names = sorted({k[0] for k, _ in rows})
    print()
    print(f"  {'':<12}" + "".join(f"{a:>22}" for a in arm_names))
    print("  " + "-" * (12 + 22 * len(arm_names)))
    for label, pick, places, unit in metrics:
        arms = group_boots([(k[0], k[1], pick(a)) for k, a in rows if pick(a) is not None])
        cells = []
        for name in arm_names:
            v = arms.get(name)
            cells.append(
                "—" if not v
                else f"{mean(v):.{places}f}{unit} ±{spread(v):.{places}f} (n={len(v)})"
            )
        print(f"  {label:<12}" + "".join(f"{c:>22}" for c in cells))
    print()

    ers_arms = group_boots([(k[0], k[1], a["ers"]["ers"]) for k, a in rows])
    if len(ers_arms) < 2:
        print("  one arm only — nothing to compare.")
        return
    best, delta, floor, win = verdict(ers_arms)
    print(f"  Best ERS: arm {best}, delta {delta:.4f} over the worst arm.")
    print(f"  Within-arm boot spread (this run's own noise floor): {floor:.4f} ERS.")
    print("  -> " + ("WIN: delta clears the floor. Keep it." if win else
                     "NULL: delta is inside the floor. Not a result — add boots or revert."))
    print()


def _selfcheck():
    assert parse_name("bench-ab-2048-r1-2.json") == ("2048", "1")
    assert parse_name("x/bench-ab-FLASH-ATTN-r3-1.json") == ("FLASH-ATTN", "3")
    assert parse_name("bench-arm-st2048.json") is None
    # two reps in boot r1 average to one sample; r2 is a second sample
    assert group_boots([("a", "1", 1.0), ("a", "1", 3.0), ("a", "2", 4.0)]) == {"a": [2.0, 4.0]}
    # delta 0.105 against a within-arm spread of 0.02 -> win
    best, delta, floor, win = verdict({"a": [1.00, 1.02], "b": [1.11, 1.12]})
    assert (best, win) == ("b", True) and abs(delta - 0.105) < 1e-9 and abs(floor - 0.02) < 1e-9
    # same delta, but arm a alone swung 0.20 between boots -> null
    assert verdict({"a": [0.91, 1.11], "b": [1.11, 1.12]})[3] is False
    print("ab_summary selfcheck ok")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--selfcheck"]:
        _selfcheck()
    else:
        main()
