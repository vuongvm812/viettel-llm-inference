"""Print a side-by-side ours-vs-vLLM table from two replay.py output JSONs.

    python bench/compare.py vllm.json ours.json
"""

import json
import sys
from pathlib import Path

from metrics import aggregate

FAIRNESS_NOTES = """\
Fairness notes (see docs/design/benchmark/design.md):
  - Weights: both sides run BF16 Qwen3.5-2B (vLLM from HF safetensors, ours from a lossless
    BF16 GGUF conversion). Precision matches — still spot-check a few outputs, not just latency.
  - Prefix caching: vLLM runs --enable-prefix-caching; ours has shared-prefix KV only from P4
    onward. Benchmarking before P4 gives vLLM the prefix-cache edge — flag it.
  - Same trace, arrival schedule, and max_tokens/temp/seed for both — the controlled part.
  - Token count basis: output tok/s counts one streamed SSE content chunk = one token (valid
    since vLLM and our runtime both stream 1 token/chunk). A stream's usage.completion_tokens
    is used instead only if present — mixing the two bases across targets is flagged below."""


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


def _fmt(v, suffix=""):
    return "—" if v is None else f"{v:.0f}{suffix}"


def _row(label, cells):
    return f"  {label:<20}" + "".join(f"{c:>18}" for c in cells)


def print_table(runs):
    headers = [r["target"].split("://", 1)[-1] for r in runs]
    aggs = [r["_agg"] for r in runs]

    print()
    print(_row("", headers))
    print("  " + "-" * (20 + 18 * len(runs)))
    for name, key in (("TTFT", "ttft_ms"), ("ITL", "itl_ms"), ("E2E", "e2e_ms")):
        for pct in ("p50", "p99"):
            cells = [_fmt(a[key][pct], "ms") for a in aggs]
            print(_row(f"{name} {pct}", cells))
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

    if any(":8001" in r["target"] or "localhost:8001" in r["target"] for r in runs):
        print("  NOTE: :8001 is our runtime; if it is still the P1 mock backend, its latency/"
              "tokens\n        are placeholder, not real-model numbers (needs P2+ on Linux+GPU).")
        print()
    print(FAIRNESS_NOTES)


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python bench/compare.py <run1.json> [run2.json ...]")
    runs = [load(p) for p in sys.argv[1:]]
    print_table(runs)


if __name__ == "__main__":
    main()
