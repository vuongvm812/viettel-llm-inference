#!/usr/bin/env python3
"""Regenerate ``data/input/trace_grading_public.jsonl`` to match the phase-2 grading workload
spec ``data/input/grading-workload-spec.json``.

The workload is fixed multi-turn (field meanings from the grading harness):
  * ``shared_system_prefix_tokens``      -- system prefix, IDENTICAL across every conversation
                                            (a global prefix-cache hit); present in every turn.
  * ``per_conversation_prefix_tokens``   -- the conversation's private context, introduced at
                                            turn 1 and carried as context on every later turn.
  * ``new_user_tokens_per_turn``         -- the user's prompt each turn; turn 1 ALSO carries the
                                            two prefix blocks above.
  * ``output_tokens_per_turn_pinned``    -- pinned output length per turn (drives out_tokens_max).

So the per-turn prompt (in_tokens_est) GROWS as the thread accumulates one (user, output) pair
per turn:

    in_tokens_est(turn) = shared + conv + turn*(user + output) + user          # turn is 0-indexed
                        = 2000 + turn*450 + 150  ->  2150, 2600, 3050, 3500, 3950, 4400

Only the token-shape fields (in_tokens_est, in_chars, out_tokens_max) come from the spec. The
Poisson(seed 42) arrival schedule, warmup flags, and conversation/turn structure are carried
from the existing public spec unchanged -- the new spec revised token SIZES, not the arrival
process. Deterministic; no tokenizer or model needed.

    python bench/build_grading_spec.py                # writes data/input/trace_grading_public.jsonl
    python bench/build_grading_spec.py --self-check    # checks the growth formula, no IO
"""
from __future__ import annotations

import argparse
import json

CHARS_PER_TOKEN = 3  # the spec's in_chars/in_tokens_est ratio (kept for parity with the harness)


def in_tokens_for_turn(turn_idx: int, spec: dict) -> int:
    """Prompt tokens at a 0-indexed turn: both prefix blocks + one (user, output) pair per prior
    turn + this turn's new user tokens."""
    base = spec["shared_system_prefix_tokens"] + spec["per_conversation_prefix_tokens"]
    per_turn = spec["new_user_tokens_per_turn"] + spec["output_tokens_per_turn_pinned"]
    return base + turn_idx * per_turn + spec["new_user_tokens_per_turn"]


def rewrite(rows: list[dict], spec: dict) -> list[dict]:
    out = []
    for r in rows:
        it = in_tokens_for_turn(r["turn_idx"], spec)
        out.append({
            **r,
            "in_chars": it * CHARS_PER_TOKEN,
            "in_tokens_est": it,
            "out_tokens_max": spec["output_tokens_per_turn_pinned"],
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="data/input/grading-workload-spec.json")
    ap.add_argument("--carry", default="data/input/trace_grading_public.jsonl",
                    help="existing public spec to carry arrivals/warmup/structure from")
    ap.add_argument("--out", default="data/input/trace_grading_public.jsonl")
    args = ap.parse_args()

    spec = json.load(open(args.spec))
    rows = [json.loads(line) for line in open(args.carry) if line.strip()]
    new = rewrite(rows, spec)

    # Guard: structure carried from `carry` must still match the spec's declared totals.
    assert len(new) == spec["total_requests"], (len(new), spec["total_requests"])
    assert len({r["conv_id"] for r in new}) == spec["num_conversations"]
    assert all(r["turn_idx"] < spec["user_turns_per_conversation"] for r in new)

    with open(args.out, "w") as f:
        for r in new:
            f.write(json.dumps(r) + "\n")

    last = spec["user_turns_per_conversation"] - 1
    ins = sorted(r["in_tokens_est"] for r in new)
    print(
        f"wrote {len(new)} rows -> {args.out}\n"
        f"  in_tokens_est grows {in_tokens_for_turn(0, spec)}..{in_tokens_for_turn(last, spec)} "
        f"(observed min={ins[0]} max={ins[-1]})\n"
        f"  out_tokens_max pinned at {spec['output_tokens_per_turn_pinned']}"
    )


def _self_check() -> None:
    spec = {
        "shared_system_prefix_tokens": 1000, "per_conversation_prefix_tokens": 1000,
        "new_user_tokens_per_turn": 150, "output_tokens_per_turn_pinned": 300,
        "user_turns_per_conversation": 6, "num_conversations": 70, "total_requests": 420,
    }
    # Growth formula matches the hand-computed table.
    assert [in_tokens_for_turn(t, spec) for t in range(6)] == [2150, 2600, 3050, 3500, 3950, 4400]

    rows = [{"conv_id": c, "turn_idx": t, "in_warmup": c < 15, "timestamp_ms": c * 100,
             "think_ms": 3000, "in_chars": 1, "in_tokens_est": 1, "out_tokens_max": 1}
            for c in range(70) for t in range(6)]
    nr = rewrite(rows, spec)
    conv0 = [r for r in nr if r["conv_id"] == 0]
    assert [r["in_tokens_est"] for r in conv0] == [2150, 2600, 3050, 3500, 3950, 4400]
    assert all(r["out_tokens_max"] == 300 for r in nr)
    assert all(r["in_chars"] == r["in_tokens_est"] * 3 for r in nr)
    # Carried fields (arrivals/warmup) are untouched.
    assert nr[0]["timestamp_ms"] == 0 and nr[0]["in_warmup"] is True
    print("build_grading_spec self-check ok")


if __name__ == "__main__":
    import sys

    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
