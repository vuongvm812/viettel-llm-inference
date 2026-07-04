# viettel-llm-inference

A custom LLM inference runtime in Rust, built on an **LMAX Disruptor** (lock-free ring
buffer) serving fabric. Goal: match or beat **vLLM** on latency/throughput under a tight
**3 CPU cores + 1 GPU** budget, serving an OpenAI-compatible API.

The transformer forward pass runs in **llama.cpp** (via `llama-cpp-2` FFI, GGUF-quantized
Qwen3.5-2B); this project is the serving, scheduling, and continuous-batching layer around it.

> Status: **design phase.** The architecture and roadmap are documented under `docs/`;
> implementation has not started (see `docs/ROADMAP.md`, phase P0).

## Architecture at a glance

Three pinned cores connected by four one-directional Disruptor rings, passing zero-copy slot
handles (payloads live in a pre-allocated slab, never crossing a ring):

| Core | Role | Wait strategy |
|------|------|---------------|
| 0 | Web I/O & SSE streaming (`tokio` + `hyper`) | event-driven (epoll) |
| 1 | Tokenize / detokenize (llama.cpp vocab) | spin-with-hint |
| 2 | Fast loop: scheduler + dynamic batcher + KV monitor, drives GPU `decode()` | busy-spin |

Full write-up: **`docs/GENERAL_ARCHITECTURE.md`**.

## Repository layout

```
docs/
  GENERAL_ARCHITECTURE.md    architecture, core mapping, ring topology, request flow
  ROADMAP.md                 phased delivery plan (P0–P7)
  design/                    per-component design docs
    web-io/            fast-loop/          disruptor-pipeline/   benchmark/
    text-processing/   inference-backend/  build-optimization/
services/
  crates/
    disruptor-rs/            vendored LMAX Disruptor port (the serving fabric)
data/
  input/trace-round1.jsonl   120 OpenAI requests; benchmark trace + PGO/BOLT training data
docker-compose.yml           vLLM baseline (Qwen3.5-2B, OpenAI API, port 8000)
```

## Baseline

`docker-compose up` starts the vLLM baseline (vLLM `v0.22.1`, OpenAI-compatible,
`Qwen3.5-2B`, 256K context, prefix caching, single GPU) on port `8000` — the target to beat.

## Next steps

See `docs/ROADMAP.md`. **P0** sets up the `services/` Cargo workspace, unblocks the
`disruptor-rs` build (resolve the missing `ring-core` / feature-gate `lossy`), and scaffolds
the `inference-runtime` crate.
