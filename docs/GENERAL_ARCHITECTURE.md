# General Architecture — Disruptor-based Rust LLM Inference Runtime

## 1. Problem & goal

Serve an OpenAI-compatible LLM inference API that competes with **vLLM** on latency and
throughput, on a machine with **3 CPU cores + 1 GPU**. The differentiator is the serving
fabric: instead of async task queues + locks, we use the **LMAX Disruptor** (lock-free,
pre-allocated ring buffers, `services/crates/disruptor-rs/`) to move requests between three
pinned cores with minimal coordination latency.

**Baseline** (`docker-compose.yml`): vLLM `v0.22.1`, OpenAI-compatible, `Qwen3.5-2B`, 256K
context, `--enable-prefix-caching`, single GPU, port `8000`.

**Model execution** is delegated to **llama.cpp** via FFI (`llama-cpp-2` → `libllama`,
Qwen3.5-2B dense transformer, **BF16** — a lossless BF16 GGUF conversion of the HF weights, not
a lower-bit quant). Our runtime is the *serving + scheduling + batching* layer; the GPU forward
pass is llama.cpp's `decode()`.

## 2. Design principles (hard constraints)

| Principle | How it's enforced |
|---|---|
| No blocking / no lock | Inter-core coordination is only via Disruptor rings (atomic cursors, no mutexes). The one blocking call is `decode()` on Core 2 — that is *doing the GPU work*, not waiting on a peer. |
| No redundant heap allocation | Requests live in a **pre-allocated `RequestSlab`**; rings carry a `Copy` handle `{ slot, kind }`, never the payload. Events in each ring are pre-allocated at startup (Disruptor factory). |
| No copy / no clone (hot path) | Prompt/token/output buffers are owned by the slab and referenced by index; nothing large crosses a ring. `MultiProducer` handles are the only intentional `clone` (one per producer thread, cheap). |
| Static polymorphism only | `disruptor-rs` is fully monomorphized (no `dyn`). Handlers are generic closures; the backend is a concrete `LlamaContext`, not a trait object. |

## 3. Core mapping

Real thread pinning happens only on Linux (`pin_at_core` is a **no-op on macOS** — see §7).

| Core | Role | Work type | Wait strategy | Owns |
|---|---|---|---|---|
| **0** | Web I/O & streaming | I/O-bound | event-driven (epoll), **not** busy-spin | `tokio` runtime, sockets, slot↔connection map |
| **1** | Text processing (tokenize + detokenize) | CPU-bound, bursty | `BusySpinWithSpinLoopHint` (dedicated core) | `Arc<LlamaModel>` (const vocab) |
| **2** | The fast loop (scheduler + batcher + KV monitor) | ultra-low-latency spin | `BusySpin` | mutable `LlamaContext` (KV cache) |

**Wait-strategy rationale** — `BusySpin` burns a whole core, so only Core 2 (the latency-
critical loop that must react to GPU completion instantly) uses it. Core 0 must stay
event-driven or it starves its own async I/O; it drains its ingress ring cooperatively.
Core 1 is dedicated and bursty, so a spin-with-hint (or short spin-then-park) minimizes
hand-off latency without a second full-spin core.

## 4. Ring topology (4 one-directional Disruptor rings)

```
                 R1 (MPSC)                 R2 (SPSC)
   Core 0  ───────────────▶  Core 1  ───────────────▶  Core 2
  HTTP in    slot handle     tokenize   slot handle    scheduler
  (multi-                                               (fast loop,
   producer)                                             decode+sample)
      ▲                          ▲                           │
      │  R4 (SPSC)               │  R3 (SPSC)                 │
      └──────────────────────────┴───────────────────────────┘
       SSE egress             detokenize        token/finish handle
      (Chunk|Finish)
```

- **R1** Core 0 → Core 1: new request admitted. Multi-producer (many HTTP connections).
- **R2** Core 1 → Core 2: request tokenized, ready to schedule.
- **R3** Core 2 → Core 1: generated token(s) to detokenize + completion.
- **R4** Core 1 → Core 0: text chunk / finish, streamed to the client.

**Continued-generation feedback stays inside Core 2** — the next decode step reads the KV
cache (on GPU) and Core 2's own scheduler state. R3/R4 only produce user-visible text and
signal completion; they are not in the generation loop's critical path.

Rings are Disruptors built with `disruptor::build_multi_producer` (R1) /
`build_single_producer` (R2–R4). Details and exact wiring: `design/disruptor-pipeline/design.md`.

## 5. Zero-copy slab model

A single `RequestSlab` = `Vec<RequestState>` sized to max in-flight requests, allocated at
startup, with a lock-free free-list (Treiber stack over atomics, or an index ring).

```
RequestState {
    tokens:      preallocated token buffer (prompt ids)
    out_bytes:   preallocated output byte buffer (detokenized text)
    seq_id:      llama.cpp sequence id (assigned by Core 2)
    conn:        egress handle (Core 0 owns the socket side)
    n_generated, max_tokens, params, state machine flags ...
}
```

Ring event (all four rings) is small and `Copy`:

```
struct RingEvent { slot: u32, kind: EventKind }   // EventKind: New | Token | Finish | ...
```

The ~40K-char prompt is written **once** into the slab by Core 0 and never copied through a
ring. See `design/disruptor-pipeline/design.md`.

## 6. End-to-end flow of one request

1. **Core 0** accepts an HTTP POST `/v1/chat/completions`, claims a free `slot`, writes the
   request `body` into the slab, registers the connection's egress handle, then
   `publish`es `{slot, New}` on **R1**.
2. **Core 1** tokenizes the prompt (llama.cpp vocab) into the slab's token buffer, then
   `publish`es `{slot, New}` on **R2**.
3. **Core 2** admits the sequence if KV memory allows (else it waits in the scheduler's
   pending queue). It sets up **shared-prefix KV** (prefill the 39K system prompt once, then
   `kv_cache_seq_cp` into the new seq), prefills the user suffix, and enters the decode loop.
   Each iteration batches all active sequences into one `llama_batch`, calls `decode()`
   (GPU), samples one token per sequence, and for each new token `publish`es
   `{slot, Token}` on **R3**. On EOS / `max_tokens` it frees the seq's KV and publishes
   `{slot, Finish}`.
4. **Core 1** detokenizes each token to a text piece (appended into the slab), publishes
   `{slot, Token}` / `{slot, Finish}` on **R4**.
5. **Core 0** streams each piece as an SSE `data:` chunk; on `Finish` it sends the
   terminal chunk, closes the stream, and returns the `slot` to the free-list.

## 7. Platform caveats

- **macOS dev vs Linux target.** `disruptor-rs::pin_at_core` is a silent no-op on macOS
  (`src/affinity.rs`). Topology and correctness can be developed/tested on macOS, but core
  placement and the 3-core performance story are only valid on the Linux deployment box.
- **BOLT** post-link optimization is ELF/Linux-only (see `design/build-optimization/design.md`).
- **llama.cpp threads.** With full GPU offload, set `n_threads = 1` — the GPU does the
  compute; extra CPU threads would contend with our 3 pinned cores.

## 8. Mapping to the disruptor-rs API (verified against `src/`)

- Build: `build_multi_producer(size, factory, wait_strategy)` /
  `build_single_producer(...)`; `size` is power-of-two (multi-producer requires ≥ 64).
- Pin + wire: `.pin_at_core(n).thread_name(..).handle_events_with(closure)`; `.and_then()`
  inserts a barrier for interdependent stages; `.new_event_poller()` hands back a poller for
  cores that own their own loop (Core 0, Core 2).
- Publish: `Producer::{publish, batch_publish, try_publish, try_batch_publish}`;
  `MultiProducer` is `Clone` (one handle per producer thread).
- Consume (owned loop): `EventPoller::{poll, take(limit), has_available}`; the returned
  `EventGuard` must be dropped to advance the cursor (releases backpressure).
- Wait strategies: `BusySpin`, `BusySpinWithSpinLoopHint`, `Sleep`.

## 9. Component docs

| Component | Doc |
|---|---|
| Core 0 web I/O & streaming | `design/web-io/design.md` |
| Core 1 tokenize/detokenize | `design/text-processing/design.md` |
| Core 2 fast loop | `design/fast-loop/design.md` |
| llama.cpp backend | `design/inference-backend/design.md` |
| Ring pipeline + slab | `design/disruptor-pipeline/design.md` |
| Benchmark harness | `design/benchmark/design.md` |
| PGO/LTO/BOLT build | `design/build-optimization/design.md` |

Phased delivery: `ROADMAP.md`.
