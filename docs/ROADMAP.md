# Roadmap

Phased delivery. Each phase lists **goal**, **exit criteria**, and **risks**. Ship the
simplest thing that proves the phase, then move up. Phases P1–P4 are validated on macOS for
correctness; core-pinning and perf numbers are only meaningful on the Linux target (P5+).

---

## P0 — Unblock the build & set up the workspace

**Goal.** Get a compiling Cargo workspace with the Disruptor crate and an empty service crate.

- Create `services/Cargo.toml` workspace: members `crates/disruptor-rs`, `crates/inference-runtime` (new).
- **Resolve the `ring-core` blocker.** `crates/disruptor-rs/Cargo.toml` declares
  `ring-core = { path = "../ring-core" }` (missing crate) for its `lossy` module, which we do
  **not** use. Fix by feature-gating `lossy` off (add a `lossy` Cargo feature, `#[cfg(feature
  = "lossy")]` the module + the `ring-core`/`bytemuck` deps) or by stubbing `ring-core`.
  Prefer the feature gate — smallest change, keeps upstream intact.
- Scaffold `crates/inference-runtime`: `main.rs`, module stubs for the 3 cores + slab + rings.

**Exit criteria.** `cargo build` succeeds from `services/`. `cargo test -p disruptor` passes.

**Risks.** `lossy` may be entangled in `lib.rs` re-exports — gating must cover the `pub use`.

---

## P1 — Pipeline skeleton with a mock backend (no GPU)

**Goal.** Prove the 4-ring topology and the slab end to end, with a *mock* model that emits
deterministic tokens. No llama.cpp yet.

- Implement `RequestSlab` + lock-free free-list; the `RingEvent { slot, kind }` type.
- Wire R1–R4 with `disruptor-rs`; pin cores 0/1/2 (no-op on mac).
- Core 0: minimal `tokio` + `hyper`/`axum` OpenAI endpoint, SSE streaming, egress poller.
- Core 1: byte-length "tokenizer" mock (or real llama.cpp vocab if ready) + detokenize mock.
- Core 2: mock "decode" that returns N canned tokens per request, exercising R3/R4.

**Exit criteria.** `curl` an OpenAI request → receive a streamed SSE response through all
four rings; slot returns to the free-list; no deadlock under the 120-request trace replayed
open-loop.

**Risks.** Cyclic-looking topology (tokens loop back to Core 1) — verify each ring is strictly
one-directional and generation feedback stays intra-Core-2 (no ring cycle → no deadlock).

---

## P2 — llama.cpp FFI integration, single sequence

**Goal.** Real GPU inference for one request at a time.

- Add `llama-cpp-2`; load GGUF Qwen3.5-2B, `LlamaModel` (shared `Arc`) + `LlamaContext`
  (owned by Core 2), `n_threads = 1`, full GPU offload.
- Core 1 tokenize/detokenize via llama.cpp vocab.
- Core 2: prefill + greedy sample (`temp=0`, `seed=42` for determinism) one seq to EOS/max.

**Exit criteria.** A single trace request returns text matching a direct `llama-cli` run
(determinism check). TTFT/tokens measured on Linux+GPU.

**Risks.** FFI Send/Sync boundary (model shared read-only, context single-threaded on Core 2);
exact `llama-cpp-2` method names for KV/sampling must be confirmed against the crate version.

---

## P3 — Continuous batching

**Goal.** Iteration-level batching of many concurrent sequences (the vLLM-style win).

- Core 2 scheduler: running set + pending queue; each iteration builds one `llama_batch`
  across all active seqs, `decode()`, sample per seq, emit tokens, retire finished seqs.
- Dynamic batcher: cap by max batch size and per-iteration token budget.

**Exit criteria.** Replaying the 120-request trace, multiple requests are in-flight per
`decode()`; throughput (tok/s) scales above the P2 single-seq number.

**Risks.** Head-of-line blocking between prefill (long 40K prompt) and decode steps — may need
chunked prefill or separate prefill/decode scheduling (note for P7).

---

## P4 — Shared-prefix KV caching

**Goal.** Compute the shared 39K-char system prompt **once** (parity with vLLM's
`--enable-prefix-caching`, needed for a fair benchmark).

- Detect shared prefix by hash of leading tokens; reserve a prefix seq, prefill once.
- Per request: `kv_cache_seq_cp(prefix_seq → new_seq, 0..prefix_len)`, then prefill only the
  user suffix at `pos = prefix_len`.
- GPU/KV monitor: track free KV cells, admit new seqs only while memory allows.

**Exit criteria.** On the trace, the system prompt is prefilled once (verified by prefill token
count); TTFT for requests 2..120 drops sharply vs P3.

**Risks.** KV cell accounting for shared cells; eviction when the prefix seq must be reclaimed
under memory pressure.

---

## P5 — Benchmark harness & vLLM comparison

**Goal.** Head-to-head latency/throughput vs vLLM on the same trace.

- Python asyncio + aiohttp replayer (`design/benchmark/design.md`): honor `timestamp_ms`
  open-loop arrivals, POST `body` verbatim, collect TTFT, TPOT/ITL, e2e latency, tok/s, req/s,
  percentiles. Same script targets `:8000` (vLLM) and our port.
- Bring up vLLM via `docker-compose up`; run both; produce a comparison report.

**Exit criteria.** One command produces a side-by-side table (ours vs vLLM) on `trace-round1.jsonl`.

**Risks.** Fairness — same model quantization matters (GGUF vs vLLM's format); note in the report.

---

## P6 — PGO → LTO → BOLT

**Goal.** Squeeze the Rust binary using the trace as training data.

- `[profile.release]`: `lto="fat"`, `codegen-units=1`, `panic="abort"`.
- PGO: `-Cprofile-generate` build → replay trace → `llvm-profdata merge` → `-Cprofile-use`.
- BOLT (Linux only): `-Wl,--emit-relocs` → `perf record` the replay → `perf2bolt` → `llvm-bolt`.
- `build-pgo-bolt.sh` orchestrates all stages (`design/build-optimization/design.md`).

**Exit criteria.** Optimized binary beats the plain `--release` build on the P5 metrics
(measure, don't assume). BOLT step documented as Linux-only.

**Risks.** BOLT needs a representative profile — the 120-request trace may under-cover cold
paths; note coverage limits. Optimizes our code only, not `libllama`.

---

## P7 — Stretch

- Radix/prefix-tree cache for arbitrary shared prefixes (generalize P4).
- WebSocket streaming alongside SSE.
- Chunked prefill / prefill-decode disaggregation to fix P3 head-of-line blocking.
- `libllama` build tuning (its own PGO, CUDA graph capture, quantization sweep).
- Multi-GPU / tensor-parallel (out of scope at 1 GPU).
