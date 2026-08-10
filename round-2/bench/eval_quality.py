"""Greedy-output agreement between two server configurations.

The gap this fills: bench/ measures latency and throughput only, so every quantization decision
in this repo -- vtl_fp8, vtl_w4a8, VTL_LM_HEAD_QUANT -- has shipped with its ACCURACY unmeasured.
Group-128 symmetric RTN int4 with no calibration is the weakest 4-bit recipe there is, 1.2 B is
the parameter-poor end where 4-bit stops being close to free, and it compounds with fp8
activations and an fp8 KV cache. This is the check that gates those knobs.

CAPTURE, THEN COMPARE -- two passes, not two live targets. One GPU serves one config at a time,
so running a reference and a candidate side by side is not possible on the judge's box. Same
shape as bench/replay.py -> bench/compare.py:

    VTL_QUANT=vtl_fp8 make up                                       # reference config
    python bench/eval_quality.py --target http://localhost:8000 --out ref.json
    make down

    VTL_LM_HEAD_QUANT=int4 make up                                  # candidate config
    python bench/eval_quality.py --target http://localhost:8000 --out int4.json
    make down

    python bench/eval_quality.py --compare ref.json int4.json

WHAT IT MEASURES. temperature=0, so a numerically identical server is byte-identical and any
divergence is real. Reported as (a) exact-match rate over the sampled prompts and (b) where the
first divergence lands, as a fraction of the reference length. Both matter and neither alone is
enough: a 60% match rate whose divergences all start at word 40 of 64 is a different animal from
one that diverges at word 2. Greedy decoding also AMPLIFIES -- one flipped argmax changes every
token after it -- so treat the match rate as a sensitive tripwire, not as an accuracy percentage.
A perceptible-quality question needs a real eval; this answers "did the weights move".

WHITESPACE WORDS, NOT TOKENS. Scoring by tokenizer id would be more faithful and would drag
transformers into a bench/ that currently needs only aiohttp. Words are a monotone proxy: they
cannot report agreement where there is none.
"""

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACE = REPO_ROOT / "data" / "input" / "trace-round2.jsonl"


def load_trace(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def first_divergence(ref: str, cand: str) -> int:
    """Index of the first differing whitespace word; ``-1`` if one is a prefix of the other.

    A pure prefix is NOT a divergence in content -- it is a length difference, which at a fixed
    max_tokens means the two runs hit the stop condition at different points. Reported
    separately so it cannot be mistaken for a wrong token.
    """
    a, b = ref.split(), cand.split()
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1


def score(pairs: list[tuple[str, str]]) -> dict:
    """Aggregate (reference, candidate) output pairs into the reported metrics.

    Pure, so the self-check can pin it without a server. ``divergence_frac`` is averaged over
    the DIVERGING pairs only -- folding the matches in as 1.0 would let a high match rate mask
    how early the failures start, which is the number that decides whether a knob is safe.
    """
    if not pairs:
        return {"n": 0, "exact": 0, "exact_rate": None, "prefix_only": 0,
                "diverged": 0, "divergence_frac_mean": None, "divergence_word_mean": None}

    exact = prefix_only = 0
    fracs: list[float] = []
    words: list[int] = []
    for ref, cand in pairs:
        if ref == cand:
            exact += 1
            continue
        idx = first_divergence(ref, cand)
        if idx < 0:
            prefix_only += 1  # same words as far as both go, different lengths
            continue
        words.append(idx)
        n_ref = len(ref.split())
        fracs.append(idx / n_ref if n_ref else 0.0)

    return {
        "n": len(pairs),
        "exact": exact,
        "exact_rate": exact / len(pairs),
        "prefix_only": prefix_only,
        "diverged": len(fracs),
        "divergence_frac_mean": statistics.fmean(fracs) if fracs else None,
        "divergence_word_mean": statistics.fmean(words) if words else None,
    }


async def capture(target: str, records: list, max_tokens: int, concurrency: int) -> list[dict]:
    """Collect one greedy completion per record. Non-streaming: only the final text matters."""
    import aiohttp

    sem = asyncio.Semaphore(concurrency)
    out: list[dict | None] = [None] * len(records)

    async def one(session, i, rec):
        body = dict(rec["body"])
        # Pin every sampling knob the trace might carry. A record with temperature=0.7 would
        # otherwise make two identical servers "disagree" and read as a quantization regression.
        body.update(temperature=0, top_p=1, max_tokens=max_tokens, stream=False)
        body.pop("seed", None)
        async with sem:
            try:
                async with session.post(f"{target}/v1/chat/completions", json=body) as resp:
                    if resp.status != 200:
                        out[i] = {"id": rec.get("request_id"), "error": (await resp.text())[:200]}
                        return
                    obj = await resp.json()
                    out[i] = {
                        "id": rec.get("request_id"),
                        "text": obj["choices"][0]["message"]["content"] or "",
                    }
            except Exception as exc:  # a dead server must not lose the rows already captured
                out[i] = {"id": rec.get("request_id"), "error": repr(exc)[:200]}

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*(one(session, i, r) for i, r in enumerate(records)))
    return [r for r in out if r is not None]


def compare(ref_path: str, cand_path: str) -> int:
    ref = {r["id"]: r for r in json.load(open(ref_path))["results"]}
    cand = {r["id"]: r for r in json.load(open(cand_path))["results"]}

    errors = [i for i in ref.keys() & cand.keys()
              if "error" in ref[i] or "error" in cand[i]]
    pairs = [(ref[i]["text"], cand[i]["text"])
             for i in ref.keys() & cand.keys()
             if "error" not in ref[i] and "error" not in cand[i]]
    s = score(pairs)

    print(f"reference {ref_path}")
    print(f"candidate {cand_path}")
    print(f"  compared         {s['n']}"
          + (f"  ({len(errors)} skipped: request errored)" if errors else ""))
    if not s["n"]:
        print("  nothing to compare -- captures share no request ids")
        return 1
    print(f"  exact match      {s['exact']}/{s['n']}  ({s['exact_rate']:.1%})")
    print(f"  prefix-only diff {s['prefix_only']}  (same words, different length)")
    print(f"  diverged         {s['diverged']}")
    if s["diverged"]:
        print(f"  first divergence at word {s['divergence_word_mean']:.1f} on average, "
              f"{s['divergence_frac_mean']:.1%} through the reference")
    # Not a pass/fail threshold: greedy decoding amplifies, so there is no defensible cutoff to
    # hardcode. Print, and let the operator weigh it against the latency the knob bought.
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target", default="http://localhost:8000")
    p.add_argument("--trace", default=str(DEFAULT_TRACE))
    p.add_argument("--limit", type=int, default=64, help="prompts to sample (default 64)")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--out", help="capture mode: write results here")
    p.add_argument("--compare", nargs=2, metavar=("REF", "CAND"))
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()

    if args.self_check:
        _self_check()
        return 0
    if args.compare:
        return compare(*args.compare)
    if not args.out:
        p.error("one of --out (capture) or --compare REF CAND is required")

    records = load_trace(args.trace)[: args.limit]
    results = asyncio.run(capture(args.target, records, args.max_tokens, args.concurrency))
    failed = sum("error" in r for r in results)
    with open(args.out, "w") as f:
        json.dump({"target": args.target, "trace": args.trace,
                   "max_tokens": args.max_tokens, "results": results}, f)
    print(f"captured {len(results) - failed}/{len(results)} -> {args.out}"
          + (f"  ({failed} errored)" if failed else ""))
    return 1 if failed == len(results) else 0


def _self_check() -> None:
    assert first_divergence("a b c", "a b c") == -1
    assert first_divergence("a b c", "a b") == -1        # prefix, not a content divergence
    assert first_divergence("a b c", "a x c") == 1
    assert first_divergence("a b", "x b") == 0
    assert first_divergence("", "") == -1

    s = score([])
    assert s["n"] == 0 and s["exact_rate"] is None

    # All identical: no divergence stats at all, rather than a misleading 0.0.
    s = score([("a b", "a b"), ("c d", "c d")])
    assert (s["exact"], s["exact_rate"], s["diverged"]) == (2, 1.0, 0)
    assert s["divergence_frac_mean"] is None

    # A prefix difference is counted separately and must NOT inflate `diverged`.
    s = score([("a b c d", "a b")])
    assert (s["exact"], s["prefix_only"], s["diverged"]) == (0, 1, 0)

    # The mean is over DIVERGING pairs only: one exact + one diverging at word 2 of 4 is 50%,
    # not the 25% that folding the match in as 1.0 would give.
    s = score([("a b c d", "a b c d"), ("a b c d", "a b X d")])
    assert s["exact_rate"] == 0.5 and s["diverged"] == 1
    assert s["divergence_word_mean"] == 2
    assert abs(s["divergence_frac_mean"] - 0.5) < 1e-9

    # Divergence at word 0 of a long reference is the loud case; it must round to ~0, not to 1.
    s = score([("a b c d e f g h", "X b c d e f g h")])
    assert s["divergence_frac_mean"] == 0.0

    print("eval_quality self-check ok")


if __name__ == "__main__":
    sys.exit(main())
