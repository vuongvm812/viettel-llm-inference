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
# Priority order (first match wins). Tuned against real v0.25.0 kernel names.
#
# HARDWARE CAVEAT: the first capture (2026-07-16) was on an RTX 3060 (Ampere, sm_86), which has
# NO FP8 tensor cores -- so cutlass_fp8_supported() is False and every FP8 Linear fell back to
# FP8-Marlin (`void marlin::Marlin<...>`, ~70% of GPU time there). That is EXPECTED on Ampere and
# tells us nothing about the H200 target: on Hopper (sm_90) those GEMMs run native FP8
# cutlass_scaled_mm and Marlin does not appear. So the 3060 bucket shares are NOT representative --
# re-run `make profile` on the H200 before drawing any Phase-1/2 conclusion. The gdn/quant kernel
# names below (FLA chunk_*, causal_conv1d, rms_norm) DO run on both, so those buckets carry over.
#
# gdn/attn first so their internal GEMM-ish kernels aren't stolen by the generic gemm bucket;
# gemm before quant so `*_marlin_gemm_*` GEMMs (even when their fused name also mentions rms_norm)
# land in gemm, not quant_fusion.
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


def load_events(path: Path) -> list:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        doc = json.load(f)
    return doc.get("traceEvents", doc) if isinstance(doc, dict) else doc


def kernel_totals(events: list) -> dict[str, float]:
    """Sum GPU-kernel durations (microseconds) by kernel name."""
    by_name: dict[str, float] = defaultdict(float)
    for ev in events:
        if not isinstance(ev, dict):
            continue
        # torch profiler tags device kernels with cat=="kernel"; dur is in microseconds.
        if ev.get("ph") == "X" and ev.get("cat") == "kernel" and "dur" in ev:
            by_name[ev.get("name", "?")] += float(ev["dur"])
    return dict(by_name)


def parse_chrome_trace(path: Path) -> dict[str, float]:
    return kernel_totals(load_events(path))


def _union_us(spans: list[tuple[float, float]]) -> float:
    """Total length of the union of [start, end) spans -- kernels on different streams
    overlap, so summing durations overstates how long the GPU was actually busy."""
    total = 0.0
    cur_s = cur_e = None
    for s, e in sorted(spans):
        if cur_e is None or s > cur_e:
            total += 0.0 if cur_e is None else cur_e - cur_s
            cur_s, cur_e = s, e
        elif e > cur_e:
            cur_e = e
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def host_summary(events: list, steps: int | None = None) -> dict:
    """The host-vs-GPU split: how much of the captured window the GPU was idle.

    ``vtl/patches/profiler.py`` captures ProfilerActivity.CPU as well as CUDA, but this
    script only ever summed kernel time -- so the one question every remaining tuning
    decision rests on (is TPOT the host term or the GPU term?) could not be answered from
    a capture. gpu_busy is the UNION of kernel spans, wall is first-kernel-start to
    last-kernel-end, and the difference is time the GPU had nothing to run.

    CPU operators are ranked by SELF time (child spans subtracted), so ``aten::linear``
    does not shadow the ``aten::mm`` inside it. Note ``with_stack=False`` in the capture,
    so pure-Python frames are invisible: whatever the ranked aten ops do not account for
    is interpreter/vLLM-Python overhead.
    """
    spans: list[tuple[float, float]] = []
    cpu_by_tid: dict = defaultdict(list)
    for ev in events:
        if not isinstance(ev, dict) or ev.get("ph") != "X" or "dur" not in ev:
            continue
        ts, dur = float(ev["ts"]), float(ev["dur"])
        cat = ev.get("cat")
        # memcpy/memset occupy the device too -- charging them to the host gap would
        # invent host cost out of a D2H sync.
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            spans.append((ts, ts + dur))
        elif cat == "cpu_op":
            cpu_by_tid[(ev.get("pid"), ev.get("tid"))].append((ts, dur, ev.get("name", "?")))

    gpu_busy = _union_us(spans)
    wall = (max(e for _, e in spans) - min(s for s, _ in spans)) if spans else 0.0

    self_us: dict[str, float] = defaultdict(float)
    # Only the busiest thread: that is the engine-core loop (scheduler + runner), and
    # summing self time across threads can exceed wall, which makes the residual below
    # meaningless.
    busiest = max(cpu_by_tid.values(), key=lambda evs: sum(d for _, d, _ in evs), default=[])
    stack: list[tuple[float, str]] = []  # (end, name)
    for ts, dur, name in sorted(busiest, key=lambda t: (t[0], -t[1])):
        while stack and stack[-1][0] <= ts:
            stack.pop()
        if stack:
            self_us[stack[-1][1]] -= dur  # charge the child to its parent
        self_us[name] += dur
        stack.append((ts + dur, name))

    ranked = sorted(self_us.items(), key=lambda kv: -kv[1])
    aten_self = sum(self_us.values())
    return {
        "wall_us": wall,
        "gpu_busy_us": gpu_busy,
        "host_gap_us": wall - gpu_busy,
        "gpu_busy_pct": 100 * gpu_busy / wall if wall else 0.0,
        "steps": steps,
        "cpu_op_self_us": aten_self,
        # Engine-thread time inside no ATen op at all: interpreter frames, vLLM's own
        # Python, ZMQ/msgpack. with_stack=False cannot see it and neither can a chrome
        # trace -- if this dominates, the tool to reach for is VTL_PROFILE_PY=1
        # (cProfile in vtl/patches/profiler.py), not a finer kernel bucket.
        "python_residual_us": max(0.0, wall - aten_self),
        "top_cpu_ops": [{"name": n, "self_us": us} for n, us in ranked[:20] if us > 0],
    }


def print_host_report(h: dict) -> None:
    wall, steps = h["wall_us"], h["steps"]
    per = (lambda us: f"{us / steps / 1e3:>10.3f}") if steps else (lambda us: f"{'-':>10}")
    print(f"\n=== host vs GPU ({wall / 1e3:.1f} ms wall"
          + (f", {steps} steps" if steps else "") + ") ===")
    print(f"{'':<14}{'ms':>12}{'%':>8}{'ms/step':>11}")
    for label, us in (("gpu busy", h["gpu_busy_us"]), ("host gap", h["host_gap_us"])):
        print(f"{label:<14}{us / 1e3:>12.1f}{100 * us / wall if wall else 0:>7.1f}%{per(us)}")
    print("\n--- engine-thread host time breakdown ---")
    for label, us in (("aten ops", h["cpu_op_self_us"]), ("python/other", h["python_residual_us"])):
        print(f"{label:<14}{us / 1e3:>12.1f}{100 * us / wall if wall else 0:>7.1f}%{per(us)}")
    if h["top_cpu_ops"]:
        print("\n--- top CPU ops by SELF time (python frames not captured) ---")
        print(f"{'ms':>10}{'ms/step':>11}  op")
        for row in h["top_cpu_ops"]:
            print(f"{row['self_us'] / 1e3:>10.1f}{per(row['self_us'])}  {row['name'][:70]}")


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
    p.add_argument("--steps", type=int, default=None,
                   help="execute_model calls in the capture (VTL_PROFILE_STEPS, default 20). "
                        "Only used to divide the host/GPU totals into per-step numbers.")
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

    events = load_events(trace_file)  # loaded once: the kernel and host passes share it
    by_name = kernel_totals(events)
    if not by_name:
        sys.exit(f"no GPU kernel events in {trace_file} (wrong file? profiler captured host-only?)")
    s = summarize(by_name, top=args.top)
    print_report(s)
    h = host_summary(events, steps=args.steps)
    print_host_report(h)
    s["host"] = h
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

    # union, not sum: two overlapping streams must not double-count GPU busy time.
    assert _union_us([(0.0, 10.0), (5.0, 20.0), (30.0, 35.0)]) == 25.0
    host_events = [
        # wall 0..100; kernels busy 0-10 and 5-20 (union 20) and 60-70 -> 30 busy, 70 idle.
        {"ph": "X", "cat": "kernel", "name": "k1", "ts": 0.0, "dur": 10.0},
        {"ph": "X", "cat": "kernel", "name": "k2", "ts": 5.0, "dur": 15.0},
        {"ph": "X", "cat": "kernel", "name": "k3", "ts": 60.0, "dur": 10.0},
        # nested CPU ops on one thread: linear 100us contains mm 80us -> self 20 / 80.
        {"ph": "X", "cat": "cpu_op", "name": "aten::linear", "ts": 0.0, "dur": 100.0, "pid": 1, "tid": 1},
        {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 10.0, "dur": 80.0, "pid": 1, "tid": 1},
    ]
    h = host_summary(host_events, steps=2)
    assert h["gpu_busy_us"] == 30.0, h
    assert h["wall_us"] == 70.0 and h["host_gap_us"] == 40.0, h
    self_by_name = {r["name"]: r["self_us"] for r in h["top_cpu_ops"]}
    assert self_by_name == {"aten::mm": 80.0, "aten::linear": 20.0}, self_by_name
    # 100us of aten inside a 70us wall window -> residual clamps at 0, never negative.
    assert h["cpu_op_self_us"] == 100.0 and h["python_residual_us"] == 0.0, h
    print("profile_trace self-check ok")


if __name__ == "__main__":
    main()
