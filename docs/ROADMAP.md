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

**Design refs.** [Disruptor Pipeline & Request Slab](design/disruptor-pipeline/design.md).

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

**Design refs.** [Disruptor Pipeline & Request Slab](design/disruptor-pipeline/design.md),
[Core 0: Web I/O & Streaming](design/web-io/design.md),
[Core 1: Text Processing](design/text-processing/design.md),
[Core 2: The Fast Loop](design/fast-loop/design.md).

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

**Design refs.** [Inference Backend (llama.cpp via FFI)](design/inference-backend/design.md),
[Core 1: Text Processing](design/text-processing/design.md).

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

**Design refs.** [Core 2: The Fast Loop](design/fast-loop/design.md),
[Inference Backend (llama.cpp via FFI)](design/inference-backend/design.md).

**Exit criteria.** Replaying the 120-request trace, multiple requests are in-flight per
`decode()`; throughput (tok/s) scales above the P2 single-seq number.

> **Proof status.** The mock pipeline tests prove the *scheduling* invariants
> (interleaving, `max_batch_seqs`/KV caps, per-iteration `max_batch_tokens` deferral,
> staggered retirement, over-`n_ctx` reject, backlog drain) in the sandbox. The
> *performance* claim — ≥2 active seqs per real `decode()` and tok/s scaling over P2 — is
> **not yet proven**: it needs the Linux/GPU target. Concretely, on target add (a) a
> decode-count assertion inside `Decoder::step` — exactly one `ctx.decode()` per step,
> over `active.len()` seqs, so a regression that decodes each seq separately is caught
> (the mock has no `decode()` to count, so this is target-only); and (b) a P2-vs-P3 tok/s
> run on the 120-request trace.
>
> **Deferred deliverable (target box only).** This instrumentation cannot run in the
> CPU/macOS dev sandbox (no `cmake`/`libllama`/GPU/GGUF — see the P2 build constraint),
> so it is an explicit open task, not shippable here:
> - [ ] `--features llama` bench: assert exactly one `ctx.decode()` per `Decoder::step`
>   over `active.len()` seqs (guards against per-seq decode regression).
> - [ ] `--features llama` bench: P2 vs P3 tok/s on the 120-request trace (short/moderate
>   prompts — the long-prompt trace stays 1 seq/decode until P4/P7, see the caveat below).
>
> **Long-prompt caveat (target trace).** The 120-request trace uses ~40K-token prompts.
> With `max_batch_tokens` bounding prefill per iteration, a long prompt is admitted only
> by the idle-progress exception and the *next* long prompt defers until the running set
> drains to idle (the HOL guard — see `core2::run`). So for the long-prompt trace,
> concurrency stays at **1 seq per `decode()`** regardless of hardware: the throughput win
> materializes only once **P4** (shared-prefix KV drops effective prefill cost to
> prompt-minus-shared-prefix) or **P7** (chunked prefill splits one prompt across
> iterations) lands. P3's multi-seq concurrency is demonstrated on short/moderate prompts;
> treat the long-prompt throughput number as a P4/P7 exit criterion, not a P3 one.

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

**Design refs.** [Core 2: The Fast Loop](design/fast-loop/design.md),
[Inference Backend (llama.cpp via FFI)](design/inference-backend/design.md).

**Exit criteria.** On the trace, the system prompt is prefilled once (verified by prefill token
count); TTFT for requests 2..120 drops sharply vs P3.

> **Proof status.** The mock pipeline tests prove the *accounting* invariant in the
> sandbox: the shared prefix is charged **once**, so more requests batch than independent
> prompts under the same limits — both the `max_batch_tokens` head-of-line guard
> (`shared_prefix_relaxes_token_budget_hol`, the P3 long-prompt caveat this phase relaxes)
> and the KV-reservation cap (`shared_prefix_admits_more_under_kv_pressure`), plus
> single-entry fallback correctness (`distinct_or_short_prompts_fall_back`) and the pure
> `shared_prefix_len` match logic. Detection is a **fixed-K window** (`runtime.shared_prefix_tokens`,
> on by default) matched by exact token-slice compare — one cached prefix (v1); hashing +
> multi-prefix radix tree is P7.
>
> **Deferred deliverable (target box only).** The real `kv_cache_seq_cp` path lives in
> `--features llama`, which the CPU/macOS dev sandbox cannot build (no `cmake`/`libllama`/
> GPU/GGUF — see the P2 build constraint). So these are explicit open tasks, not shippable
> here:
> - [ ] `--features llama` bench: assert the system prompt is prefilled **exactly once**
>   (prefill token count == `prefix_len + Σ suffix` over the trace).
> - [ ] `--features llama` bench: TTFT for requests 2..120 drops sharply vs P3 on the trace.
> - [ ] Confirm the exact `llama-cpp-2` KV-copy spelling (`kv_cache_seq_cp` vs a
>   `kv_self_*`/memory API) and that it shares cell references (so a request's `seq_rm`
>   leaves the shared prefix resident).

**Risks.** KV cell accounting for shared cells (handled: prefix reserved once, per-seq
reservation excludes it); eviction when the prefix seq must be reclaimed under memory pressure
(out of scope in v1 — single held prefix, re-established after any full KV wipe; multi-prefix
eviction is P7).

---

## P5 — Benchmark harness & vLLM comparison

**Goal.** Head-to-head latency/throughput vs vLLM on the same trace.

- Python asyncio + aiohttp replayer (`design/benchmark/design.md`): honor `timestamp_ms`
  open-loop arrivals, POST `body` verbatim, collect TTFT, TPOT/ITL, e2e latency, tok/s, req/s,
  percentiles. Same script targets `:8000` (vLLM) and our port.
- Bring up vLLM via `docker-compose up`; run both; produce a comparison report.

**Design refs.** [Benchmark Harness](design/benchmark/design.md).

**Exit criteria.** One command produces a side-by-side table (ours vs vLLM) on `trace-round1.jsonl`.

> **Proof status.** The harness is built and verified end-to-end in the sandbox against the
> live local runtime (`bench/`): `bench/metrics.py` self-check passes; `bench/replay.py`
> replays all 120 trace requests open-loop (and `--closed-loop N`) against `:8001` with
> 0 errors and populated TTFT/ITL/E2E/tok-s; `bench/compare.py` prints the side-by-side table
> + fairness notes. The comparison is target-agnostic — only the vLLM side needs the target box.
>
> **Deferred deliverable (target box only, Linux+GPU).** vLLM cannot run in the CPU/macOS dev
> sandbox (needs a GPU), and our `:8001` is still the P1 mock (placeholder latency/tokens until
> P2+), so the actual head-to-head is an explicit open task:
> - [ ] `docker-compose up` the vLLM baseline; `bench/replay.py --target :8000 --out vllm.json`.
> - [ ] Replay against a P2+ real-model `:8001`; `bench/compare.py vllm.json ours.json`.
> - [ ] Capture the report and note quantization/prefix-cache fairness caveats (compare.py prints them).

**Risks.** Fairness — same model quantization matters (GGUF vs vLLM's format); note in the report.

---

## P6 — PGO → LTO → BOLT

**Goal.** Squeeze the Rust binary using the trace as training data.

- `[profile.release]`: `lto="fat"`, `codegen-units=1`, `panic="abort"`.
- PGO: `-Cprofile-generate` build → replay trace → `llvm-profdata merge` → `-Cprofile-use`.
- BOLT (Linux only): `-Wl,--emit-relocs` → `perf record` the replay → `perf2bolt` → `llvm-bolt`.
- `build-pgo-bolt.sh` orchestrates all stages (`design/build-optimization/design.md`).

**Design refs.** [Build Optimization (PGO → LTO → BOLT)](design/build-optimization/design.md).

**Exit criteria.** Optimized binary beats the plain `--release` build on the P5 metrics
(measure, don't assume). BOLT step documented as Linux-only.

> **Proof status.** Built and runnable in the sandbox: `[profile.release]` (fat LTO,
> `codegen-units=1`, `panic="abort"`) in `services/Cargo.toml`; `services/.cargo/config.toml`
> (`target-cpu=native`); the graceful-shutdown seam (`core0::serve_until` + `shutdown_signal`,
> tokio `signal` feature) that makes a *clean* process exit possible — LLVM flushes PGO
> `*.profraw` only via a libc `atexit` handler a killed process never runs, so without it PGO
> produces **zero** data (`serve_until_returns_on_shutdown` guards the wiring). `build-pgo-bolt.sh`
> orchestrates the full two-pass PGO build, replays the trace to train it, merges, rebuilds with
> `profile-use`, and prints the `bench/compare.py` baseline-vs-optimized table; `make build/pgo`.
>
> **Deferred deliverable (target box only).** Two things cannot be proven in the CPU/macOS dev
> sandbox running the P1 **mock** backend (synthetic latency/tokens — see the P5 caveat):
> - [ ] BOLT stage (`-Wl,--emit-relocs` → `perf record` → `perf2bolt` → `llvm-bolt`). **ELF/Linux
>   only** (Mach-O unsupported); the script auto-skips it on macOS / when `perf`+`llvm-bolt` are
>   absent. Needs the Linux deploy box.
> - [ ] A *meaningful* perf delta: rerun `build-pgo-bolt.sh` against a `--features llama` real-model
>   build on Linux+GPU and confirm the optimized binary beats plain `--release` on the P5 metrics.
>   The sandbox comparison table exercises the pipeline but is not a real signal.

**Risks.** BOLT needs a representative profile — the 120-request trace may under-cover cold
paths; the script loops the trace (`PGO_REPLAY_LOOPS`, default 3) to densify coverage — note
coverage limits. Optimizes our code only, not `libllama`.

---

## P7 — Stretch

- Radix/prefix-tree cache for arbitrary shared prefixes (generalize P4).
- WebSocket streaming alongside SSE.
- Chunked prefill / prefill-decode disaggregation to fix P3 head-of-line blocking.
- `libllama` build tuning (its own PGO, CUDA graph capture, quantization sweep).
- Multi-GPU / tensor-parallel (out of scope at 1 GPU).

**Design refs.** [Core 2: The Fast Loop](design/fast-loop/design.md) (radix cache, chunked prefill),
[Core 0: Web I/O & Streaming](design/web-io/design.md) (WebSocket),
[Inference Backend (llama.cpp via FFI)](design/inference-backend/design.md) (`libllama` tuning).
