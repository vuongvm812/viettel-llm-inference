# Round-2: Weka-derived trace for `make warm` + PGO/BOLT

## Context

The BTC round page (§3.1) grades by replaying **real Claude Code sessions from the
SemiAnalysis Weka corpus** (HF `semianalysisai/cc-traces-weka-062126`) through NVIDIA
aiperf's locked `inferencex-agentx-mvp` scenario: 900 s, concurrency 5 session-trees,
204,800 context cap, `ignore_eos:true`. `make bench-aiperf` already mirrors that exactly,
and HANDOFF §6.3 already said the authored synthetic trace
(`data/input/trace-round2.jsonl`) predates the spec and no longer matches grading.

But two flows that shape the **shipped image** still replayed the synthetic trace:

1. `make warm` — bakes torch.compile/Triton/CUDA-graph caches into the image. Warmed on
   2-4k-context prefill-biased traffic, the judge's real long-context decode-heavy load
   hits cold compile paths during the graded 900 s.
2. vllm-fork **PGO/BOLT** — the Rust frontend's profile-guided hot/cold split was
   trained on the same wrong shapes.

## What changed

- **`round-2/bench/build_trace_weka.py`** (new): converts the corpus into the replay.py
  trace schema at `data/input/trace-weka.jsonl` (generated, gitignored). The corpus
  carries no text — each request is `{t, model, in, out, hash_ids, think_time, ...}`
  where `hash_ids` are 64-token KV-block hashes encoding the entire prefix-reuse
  structure. The converter synthesizes deterministic text per hash id (same id → same
  text), which reproduces true input sizes (tokenizer-exact with `--tokenizer`, e.g.
  `hf-model/` from `make model-fetch-meta`), true prefix-cache topology, true output
  lengths (`max_tokens` = recorded `out`, plus `ignore_eos:true` like grading;
  `--no-ignore-eos` if a frontend rejects the field), and recorded think-time arrivals
  capped at the scenario's 10 s idle guard. Subagent inner requests flatten into the
  timeline as parallel requests; 5 session-trees (seeded pick from the first qualifying
  traces, streamed from HF with early stop — no 1.85 GB download) start at t=0 together.
  Traces whose peak `in+out` exceeds 204,800 are dropped whole, mirroring aiperf's
  `--max-context-length` filter. `--self-check` is wired into `make check`.
- **`make trace-weka`** (new target): runs the converter; auto-passes `--tokenizer
  $(PGO_HFMODEL)` when its `tokenizer.json` exists.
- **`make warm`**: now replays `WARM_TRACE ?= $(WEKA_TRACE)` (guard: "run `make
  trace-weka` first"), `WARM_REQS` default 32 → 64 for context-length coverage.
  `TRACE`/`make bench`/CI stay on the synthetic trace, by decision.
- **PGO/BOLT**: `Dockerfile.vllm-fork` COPYs `trace-weka.jsonl` (dockerignore negation
  added); `pgo_train.sh` and the BOLT stage replay it; `pgo-hfmodel-ctx` gate checks it
  exists. Stale "closed-loop 8 ≈ trace peak ~6" comment updated to the grading rationale
  (5 trees + subagent fan-out).

First generated build (seed 0, no tokenizer): 229 requests / 95 MB / 492 s span, input
tokens p50 83k, p90 134k, max 169k; 29 of the first 40 requests share >1 kB prompt
prefixes with earlier ones. Regenerate with the model tokenizer on the GPU box for
token-exact block sizing.

## Verification

- `python bench/build_trace_weka.py --self-check` — green (in `make check`).
- Generated trace loads via `bench/replay.py`'s own `load_trace`, timestamps sorted,
  schema matches, prefix sharing confirmed (above).
- GPU box: `make trace-weka && make warm` completes; harvested `docker/cache/` should
  now contain long-context compile artifacts.
- `make vllm-fork` (PGO on) completes with the training replay serving the Weka trace.
- Before/after `make bench-aiperf` is the arbiter of whether grading-shaped warm/PGO
  moved ERS.
