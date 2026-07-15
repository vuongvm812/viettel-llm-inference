"""Phase-0 profiler: where does the prefill/decode time actually go?

Every real kernel win in this repo is gated on an on-box profile (see
docs/plans/gdn-parallel-scan.md, the README) and none exists yet. This drives vLLM's
built-in torch profiler around a real-trace replay, then buckets the captured CUDA
kernels into the categories the plan reasons about:

    gdn_scan      GDN linear-attn prefill scan + conv + recurrent decode (18/24 layers)
    full_attn     the 6 full-attention layers (FlashInfer prefill/decode)
    gemm          FP8 / bf16 projection + FFN matmuls
    quant_fusion  the vtl fused norm/act -> fp8 quant kernels
    other         rope, embedding, sampling, copies, memset, ...

The ranked table is the deliverable that decides Phase 2: if `gdn_scan` is a top-2
cost the chunk-parallel WY kernel is justified; if `full_attn` dominates, the fp8 /
cascade attention work in Phase 1 is the priority.

Usage (server must be launched with VLLM_TORCH_PROFILER_DIR set -- `make profile` does this):

    # drive the profiler around a replay, then parse the newest trace it wrote
    python3 bench/profile_trace.py --target http://localhost:8000 \
        --profile-dir ./bench-profile --limit 8 --concurrency 8

    # just re-bucket an already-captured chrome trace (no server needed)
    python3 bench/profile_trace.py --trace-file ./bench-profile/<...>.pt.trace.json.gz

    python3 bench/profile_trace.py --self-check      # no GPU, no server

The buckets are HEURISTIC (name substrings). Real v0.25.0 kernel names are only known
on the box, so the report always also prints the top unbucketed kernels -- reclassify
by editing BUCKETS below once you see them.
"""

import argparse
import asyncio
import glob
import gzip
import json
import os
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

# `replay` (and its aiohttp dep) is imported lazily in the replay path only, so
# --self-check / --trace-file parsing run with no HTTP deps (and stay in `make check`).

# (bucket, regex) in PRIORITY order: a kernel name is charged to the FIRST match.
# gdn before attn (delta-rule has its own GEMMs), quant before gemm (norm/quant names
# must not fall into the generic gemm bucket). Edit these once real names are known.
BUCKETS: tuple[tuple[str, str], ...] = (
    ("gdn_scan", r"gated_delta|delta_rule|chunk_scan|fused_recurrent|causal_conv1d|solve_tril|\bgdn\b|chunk_o|chunk_state|chunk_a"),
    ("full_attn", r"flashinfer|batch_?prefill|batch_?decode|paged|flash_?attn|\bmha\b|\bmla\b|prefill_kernel|decode_kernel|_attention"),
    ("quant_fusion", r"rms_?norm|layer_?norm|dynamic_per_token|silu_and_mul|mul_sigmoid|scaled_fp8|\bquant\b|gated_rmsnorm|rotary|\bnorm\b"),
    ("gemm", r"cutlass|gemm|scaled_mm|wgmma|\bsm90\b|\bsm80\b|matmul|nvjet|hgemm|s16816|ampere_|_linear|cublas|triton_.*mm"),
)
_COMPILED = tuple((name, re.compile(rx, re.IGNORECASE)) for name, rx in BUCKETS)


def bucket_of(kernel_name: str) -> str:
    for name, rx in _COMPILED:
        if rx.search(kernel_name):
            return name
    return "other"


def parse_chrome_trace(path: Path) -> dict[str, float]:
    """Sum GPU-kernel durations (microseconds) by kernel name from a chrome trace."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        doc = json.load(f)
    events = doc.get("traceEvents", doc) if isinstance(doc, dict) else doc
    by_name: dict[str, float] = defaultdict(float)
    for ev in events:
        if not isinstance(ev, dict):
            continue
        # torch profiler tags device kernels with cat=="kernel"; dur is in microseconds.
        if ev.get("ph") == "X" and ev.get("cat") == "kernel" and "dur" in ev:
            by_name[ev.get("name", "?")] += float(ev["dur"])
    return dict(by_name)


def summarize(by_name: dict[str, float], top: int = 40) -> dict:
    buckets: dict[str, float] = defaultdict(float)
    per_bucket_names: dict[str, list] = defaultdict(list)
    for kname, us in by_name.items():
        b = bucket_of(kname)
        buckets[b] += us
        per_bucket_names[b].append((kname, us))
    total = sum(by_name.values()) or 1.0
    ranked_buckets = sorted(buckets.items(), key=lambda kv: -kv[1])
    ranked_kernels = sorted(by_name.items(), key=lambda kv: -kv[1])[:top]
    return {
        "total_us": total,
        "buckets": [
            {"bucket": b, "us": us, "pct": 100 * us / total} for b, us in ranked_buckets
        ],
        "top_kernels": [
            {"name": n, "us": us, "pct": 100 * us / total, "bucket": bucket_of(n)}
            for n, us in ranked_kernels
        ],
    }


def print_report(s: dict) -> None:
    print(f"\n=== prefill/decode kernel cost ({s['total_us'] / 1e3:.1f} ms GPU total) ===")
    print(f"{'bucket':<14}{'ms':>12}{'%':>8}")
    for row in s["buckets"]:
        print(f"{row['bucket']:<14}{row['us'] / 1e3:>12.1f}{row['pct']:>7.1f}%")
    print(f"\n--- top {len(s['top_kernels'])} kernels (reclassify BUCKETS from these) ---")
    print(f"{'%':>6}{'ms':>10}  {'bucket':<13}kernel")
    for row in s["top_kernels"]:
        nm = row["name"][:80]
        print(f"{row['pct']:>5.1f}%{row['us'] / 1e3:>10.1f}  {row['bucket']:<13}{nm}")


def _post(url: str) -> None:
    """POST with no body; tolerate 404 (endpoint only exists when the profiler dir is set)."""
    try:
        req = urllib.request.Request(url, method="POST", data=b"")
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 (local target)
            r.read()
    except Exception as e:  # noqa: BLE001
        print(f"warn: POST {url} failed ({e}); is VLLM_TORCH_PROFILER_DIR set on the server?",
              file=sys.stderr)


async def _drive_replay(target: str, records: list, concurrency: int) -> None:
    import aiohttp

    from replay import do_request  # lazy: keeps aiohttp out of --self-check

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=600)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as s:
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        q: asyncio.Queue = asyncio.Queue()
        for r in records:
            q.put_nowait(r)

        async def worker():
            while True:
                try:
                    rec = q.get_nowait()
                except asyncio.QueueEmpty:
                    return
                await do_request(s, target, rec, t0)

        await asyncio.gather(*(worker() for _ in range(concurrency)))


def _newest_trace(profile_dir: Path) -> Path | None:
    cands = glob.glob(str(profile_dir / "**" / "*.json*"), recursive=True)
    cands = [c for c in cands if c.endswith((".json", ".json.gz"))]
    if not cands:
        return None
    return Path(max(cands, key=os.path.getmtime))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default="http://localhost:8000")
    p.add_argument("--trace", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "input" / "trace-round1.jsonl")
    p.add_argument("--limit", type=int, default=8, help="requests to replay under the profiler")
    p.add_argument("--concurrency", type=int, default=8, help="closed-loop workers (exercise batched/wave behavior)")
    p.add_argument("--profile-dir", type=Path, default=Path("bench-profile"),
                   help="host dir the server's VLLM_TORCH_PROFILER_DIR maps to")
    p.add_argument("--trace-file", type=Path, help="skip the replay; just re-bucket this chrome trace")
    p.add_argument("--out", type=Path, default=Path("bench-profile-summary.json"))
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()

    if args.self_check:
        _selfcheck()
        return

    trace_file = args.trace_file
    if trace_file is None:
        from replay import load_trace  # lazy: keeps aiohttp out of --self-check

        records = load_trace(args.trace)[: args.limit]
        print(f"profiling {len(records)} requests @ conc={args.concurrency} -> {args.target}", file=sys.stderr)
        _post(f"{args.target}/start_profile")
        asyncio.run(_drive_replay(args.target, records, args.concurrency))
        _post(f"{args.target}/stop_profile")
        # vLLM flushes the trace asynchronously on stop; give it a moment, then pick the newest.
        import time

        for _ in range(30):
            trace_file = _newest_trace(args.profile_dir)
            if trace_file is not None:
                break
            time.sleep(1)
        if trace_file is None:
            sys.exit(f"no trace file appeared in {args.profile_dir}; check the mount + VLLM_TORCH_PROFILER_DIR")
        print(f"parsing {trace_file}", file=sys.stderr)

    by_name = parse_chrome_trace(trace_file)
    if not by_name:
        sys.exit(f"no GPU kernel events in {trace_file} (wrong file? profiler captured host-only?)")
    s = summarize(by_name, top=args.top)
    print_report(s)
    args.out.write_text(json.dumps(s, indent=2))
    print(f"\nsummary -> {args.out}", file=sys.stderr)


def _selfcheck() -> None:
    # bucket routing: priority order + fall-through to 'other'.
    assert bucket_of("void chunk_gated_delta_rule_fwd_kernel<...>") == "gdn_scan"
    assert bucket_of("flashinfer::BatchPrefillWithPagedKVCacheKernel") == "full_attn"
    assert bucket_of("_C::rms_norm_dynamic_per_token_quant") == "quant_fusion"
    assert bucket_of("cutlass_scaled_mm_sm90_fp8") == "gemm"
    assert bucket_of("causal_conv1d_fwd_kernel") == "gdn_scan"
    assert bucket_of("elementwise_kernel<AddFunctor>") == "other"
    # a delta-rule GEMM stays in gdn_scan, not gemm (priority).
    assert bucket_of("gated_delta_rule_gemm") == "gdn_scan"

    events = [
        {"ph": "X", "cat": "kernel", "name": "chunk_gated_delta_rule_fwd", "dur": 100.0},
        {"ph": "X", "cat": "kernel", "name": "flashinfer::BatchPrefill", "dur": 60.0},
        {"ph": "X", "cat": "kernel", "name": "cutlass_scaled_mm", "dur": 40.0},
        {"ph": "X", "cat": "cpu_op", "name": "aten::add", "dur": 999.0},  # not a kernel -> ignored
        {"ph": "M", "name": "process_name"},                              # metadata -> ignored
    ]
    doc = {"traceEvents": events}
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(doc, f)
        tmp = f.name
    by_name = parse_chrome_trace(Path(tmp))
    os.unlink(tmp)
    assert by_name == {"chunk_gated_delta_rule_fwd": 100.0, "flashinfer::BatchPrefill": 60.0, "cutlass_scaled_mm": 40.0}, by_name
    s = summarize(by_name)
    assert abs(s["total_us"] - 200.0) < 1e-6, s["total_us"]
    top_bucket = s["buckets"][0]
    assert top_bucket["bucket"] == "gdn_scan" and abs(top_bucket["pct"] - 50.0) < 1e-6, s["buckets"]
    print("profile_trace self-check ok")


if __name__ == "__main__":
    main()
