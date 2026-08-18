# Benchmark harness

Two paths, different jobs. The grading-fidelity path is what shipping decisions are
justified on; the synthetic path is the fast, runs-anywhere regression signal. Background
and the full spec: [`../HANDOFF.md` §6](../HANDOFF.md#6-official-grading-workload--scoring-round-2-btc-spec).

## Path 1 — grading fidelity: aiperf AgentX (`make bench-aiperf`, H200 only)

Replays the SemiAnalysis Weka corpus (real multi-turn Claude Code sessions) through NVIDIA
aiperf's locked `inferencex-agentx-mvp` scenario — the exact workload and flags the BTC
grades with, except their hidden seed (`AIPERF_SEED`, default 0; sweep a few).

```bash
pip install -r bench/requirements-aiperf.txt   # heavy; H200 host venv only
make bench-aiperf                              # 900 s run + ERS report
make bench-aiperf AIPERF_LIMIT=8 AIPERF_DURATION=120   # smoke
```

The target chains three steps: `aiperf profile ...` → `aiperf_adapter.py` (converts the
per-request `profile_export.jsonl` into the repo run schema, honoring `submission_valid`)
→ `_ci_report.py` (ERS tables). The Weka dataset auto-downloads from HuggingFace on first
run.

First-real-run checklist (two things the docs don't pin down; the Makefile target has the
same note):

1. If aiperf rejects an explicit flag as duplicate/conflicting with the scenario preset,
   delete the explicit copy from the `bench-aiperf` target and note which.
2. Confirm the adapter's field mapping (module docstring table) against the actual
   `profile_export.jsonl`, and where `submission_valid` sits in `profile_export_aiperf.json`.

## Path 2 — synthetic trace (`make bench`, runs anywhere)

Open-loop replay of `data/input/trace-round2.jsonl` (420 authored requests) honoring
per-record arrival times, plus a closed-loop sweep at 1/8/32/128. SSE streaming for
TTFT/ITL.

```bash
pip install -r bench/requirements.txt          # aiohttp only
python bench/replay.py --target http://localhost:8000 --out run.json
python bench/compare.py a.json b.json          # side-by-side + fairness notes
```

Extra flags: `--closed-loop N`, `--limit N` (smoke), `--trace PATH`, `--timeout SECONDS`.

Its arrival/token/prefix statistics predate the round-2 spec: use it for relative
regressions and CI, never for final tuning calls.

## Weka-derived trace for `make warm` + PGO/BOLT (`make trace-weka`)

`build_trace_weka.py` converts the REAL grading corpus (HF
`semianalysisai/cc-traces-weka-062126` — KV-block hash traces, no text) into a replay.py
trace at `data/input/trace-weka.jsonl` (generated, gitignored). It synthesizes
deterministic text per 64-token block hash, so input sizes, prefix-cache topology,
output lengths (`max_tokens` + `ignore_eos`, as grading runs), and think-time arrivals
(capped at the scenario's 10 s idle guard) all match grading shape. Traces whose peak
context exceeds 204,800 are dropped whole, mirroring aiperf's `--max-context-length`.

`make warm` (cache baking) and the vllm-fork PGO/BOLT training stages replay THIS trace
— they shape what the shipped image is optimized for, so they must not run on the stale
synthetic trace. `make bench`/CI deliberately stay on Path 2.

Run `make trace-weka` once per checkout (streams the corpus from HF, stops early; with
`hf-model/` metadata present — `make model-fetch-meta` — blocks are sized
tokenizer-exactly, else a word-count fallback is used).

## Notes

- Output tokens are counted from streamed content deltas (uniform across targets); `usage`
  is used only if the stream provides it. aiperf runs use server-reported counts
  (`--use-server-token-count`).
- Self-checks (all off-box, no GPU): `python bench/metrics.py`,
  `python bench/sweep_report.py --selfcheck`, `python bench/aiperf_adapter.py --selfcheck`
  — all wired into `make check`.
