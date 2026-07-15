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

This is only the PARSER. The capture is done by vtl/patches/profiler.py inside the worker
(this build's vLLM stripped the /start_profile endpoint), driven by `make profile`:
boot -> wait healthy -> touch bench-profile/.arm -> replay load -> worker dumps
bench-profile/vtl-trace-<pid>.json. Then this script buckets that trace.

    make profile                                     # capture + parse end to end (needs H200)

    python3 bench/profile_trace.py --profile-dir ./bench-profile   # parse newest capture
    python3 bench/profile_trace.py --trace-file ./bench-profile/vtl-trace-147.json
    python3 bench/profile_trace.py --self-check      # no GPU, no server

The buckets are HEURISTIC (name substrings). Real v0.25.0 kernel names are only known
on the box, so the report always also prints the top unbucketed kernels -- reclassify
by editing BUCKETS below once you see them.
"""

import argparse
import glob
import gzip
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# (bucket, regex) in PRIORITY order: a kernel name is charged to the FIRST match.
# gdn before attn (delta-rule has its own GEMMs), quant before gemm (norm/quant names
# must not fall into the generic gemm bucket). Edit these once real names are known.
# Priority order (first match wins). Tuned against the real v0.25.0 kernel names seen in the
# first on-box capture (2026-07-16). gdn/attn first so their internal GEMM-ish kernels aren't
# stolen by the generic gemm bucket; gemm before quant so `*_marlin_gemm_*` GEMMs (even when
# their fused name also mentions rms_norm) land in gemm, not quant_fusion.
BUCKETS: tuple[tuple[str, str], ...] = (
    # FLA chunked gated-delta-rule (prefill) + conv + recurrent decode -- the 18 GDN layers.
    ("gdn_scan", r"gated_delta|delta_rule|causal_conv1d|fused_recurrent|sigmoid_gating|solve_tril|"
                 r"chunk_fwd|chunk_o\b|chunk_state|chunk_scaled_dot|recompute_w|fused_post_conv|"
                 r"merge_\d+x\d+.*inverse|\bgdn\b|chunk_scan"),
    # the 6 full-attention layers (FlashInfer prefill/decode).
    ("full_attn", r"flashinfer|batch_?prefill|batch_?decode|paged_?attn|flash_?attn|\bmha\b|\bmla\b|_attention"),
    # projection + FFN matmuls. marlin::Marlin is FP8-Marlin (the current fallback path); cutlass
    # scaled_mm / s16816 / gemv are the native ones. reshape_and_cache = KV write, keep with gemm-ish.
    ("gemm", r"\bmarlin\b|cutlass|scaled_mm|_gemm\b|\bgemm|gemv|wgmma|\bsm90\b|s16816|ampere_.*gemm|"
             r"hgemm|nvjet|cublas|reshape_and_cache"),
    # the vtl fused norm/act -> fp8 quant kernels + rope/rmsnorm.
    ("quant_fusion", r"rms_?norm|layer_?norm|dynamic_per_token|silu_and_mul|mul_sigmoid|scaled_fp8|"
                     r"\bquant\b|gated_rmsnorm|rope|rotary|\bnorm\b"),
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


def _best_trace(profile_dir: Path) -> Path | None:
    """The chrome trace vtl/patches/profiler.py wrote. Pick the LARGEST json (a real
    kernel-rich capture dwarfs any stray/empty file, e.g. from a non-worker process)."""
    cands = [
        Path(c)
        for c in glob.glob(str(profile_dir / "**" / "vtl-trace-*.json*"), recursive=True)
        if c.endswith((".json", ".json.gz"))
    ]
    if not cands:  # fall back to any *.json* if the naming ever changes
        cands = [
            Path(c)
            for c in glob.glob(str(profile_dir / "**" / "*.json*"), recursive=True)
            if c.endswith((".json", ".json.gz")) and "summary" not in Path(c).name
        ]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_size)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile-dir", type=Path, default=Path("bench-profile"),
                   help="dir the worker wrote vtl-trace-<pid>.json into (mounted from /profile)")
    p.add_argument("--trace-file", type=Path, help="parse this specific chrome trace instead of scanning --profile-dir")
    p.add_argument("--out", type=Path, default=Path("bench-profile-summary.json"))
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()

    if args.self_check:
        _selfcheck()
        return

    trace_file = args.trace_file or _best_trace(args.profile_dir)
    if trace_file is None:
        sys.exit(
            f"no vtl-trace-*.json in {args.profile_dir}. Did the worker arm+dump? "
            "`make profile` touches .arm after the server is healthy, then replays load; "
            "check the server logs for 'vtl: profiler wrote ...'."
        )
    print(f"parsing {trace_file} ({trace_file.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)

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
    # real names from the 2026-07-16 capture that were misfiled the first time:
    assert bucket_of("void marlin::Marlin<...>(int4 const*, ...)") == "gemm"
    assert bucket_of("chunk_fwd_kernel_o") == "gdn_scan"
    assert bucket_of("recompute_w_u_fwd_kernel") == "gdn_scan"
    assert bucket_of("fused_sigmoid_gating_delta_rule_update_kernel") == "gdn_scan"
    assert bucket_of("merge_16x16_to_64x64_inverse_kernel") == "gdn_scan"
    assert bucket_of("ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x6_tn") == "gemm"

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
