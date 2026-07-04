# P1 Implementation Plan — Pipeline skeleton with a mock backend

Proves the 4-ring Disruptor topology + zero-copy slab end to end with a **mock** model
(deterministic canned tokens, no llama.cpp). Validated on macOS (pinning is a no-op here).

## Modules (all in `crates/inference-runtime/src/`)

- `rings.rs` — `RingEvent { slot: u32, kind: EventKind }`, `EventKind { New | Token(u32) |
  Finish(FinishReason) }`, `FinishReason { Eos | MaxTokens | Error }`. `Copy`, ~8 bytes. Shared
  by all four rings.
- `slab.rs` — `RequestState` (prompt, max_tokens, tokens) +
  `Slab { slots: Box<[UnsafeCell<RequestState>]> }` with `unsafe impl Sync`. Single-writer-per-
  stage; the ring hand-off (release/acquire) is the synchronization edge. `unsafe fn slot_mut`.
- `pipeline.rs` — builds the 4 rings, spawns Core 1 + Core 2 threads, hands the caller the
  Core-0-side handles (R1 producer, R4 poller). Reused by both `main` (real Core 0) and the test.
- `core1.rs` — `run(...)`: poll R1 → mock tokenize (one token per prompt byte) → publish R2;
  poll R3 → mock detokenize (`id as u8`) → publish R4 carrying the byte **in the event**.
- `core2.rs` — `run(...)`: poll R2 → admit; each iteration emit one canned byte per active seq on
  R3 until `min(max_tokens, CANNED.len())`, then `Finish`. No KV, no llama.
- `core0.rs` — tokio current-thread + axum. `POST /v1/chat/completions` (SSE), `GET /health`.
  Ingress: validate → pop slot → write prompt/max_tokens/conn → publish R1. Egress: R4 poller
  task drains R4, streams each event's byte as an SSE chunk, `Finish` → return slot to free-list.

## Free-list

Plain `Vec<u32>` owned by Core 0 only (design's SPSC note — nobody else claims/returns slots).
In `main` it is `Arc<Mutex<Vec<u32>>>` shared between the two Core-0 tasks (handler + poller).

## Ring wiring (verified `disruptor-rs` API)

| Ring | Build | Producer (owner) | Poller (owner) |
|---|---|---|---|
| R1 | `build_multi_producer` | Core 0 | Core 1 |
| R2 | `build_single_producer` | Core 1 | Core 2 |
| R3 | `build_single_producer` | Core 2 | Core 1 |
| R4 | `build_single_producer` | Core 1 | Core 0 |

`new_event_poller()` splits each ring into (poller, producer). Poller types:
`EventPoller<RingEvent, MultiProducerBarrier>` (R1) / `…, SingleProducerBarrier>` (R2–R4).
Producers passed as `impl Producer<RingEvent>`.

## RED test (TDD) — `tests/pipeline.rs`

The design's `demo()` self-check without HTTP: play Core 0 from the test thread — push K=120
synthetic requests through R1, drain R4, collect per-slot output. Assert: every slot returns to
the free-list, and each request's streamed bytes equal the canned reply (ordered, no interleave).
This proves no deadlock + slot recycle + one-directional rings.

## Detok → egress handoff (deviation from web-io/disruptor-pipeline design)

The design docs stream by having Core 0 read the slab's `out_bytes` as Core 1 appends. That is
**racy for streaming**: a slot is in the detokenize stage (Core 1 appending token N+1) and the
egress stage (Core 0 reading token N) at the same time — two cores on one slot, violating
single-writer-per-stage, and a realloc of `out_bytes` would dangle Core 0's slice (UB). P1 instead
carries the detokenized byte **in the R4 event**, so no cross-core buffer is shared. A `u32` only
holds one byte, so **P2 (real multi-byte vocab pieces) needs a proper handoff**: a per-slot output
buffer with capacity that *never* reallocates, where Core 0 reads only indices below an
acquire-synchronized watermark. Update `web-io/design.md` and `disruptor-pipeline/design.md`
accordingly.

## Deferred (ponytail)

- `generation` ABA guard: omitted — slots aren't reused until `Finish`+free, so no ABA in P1.
  Add with cancellation/timeout.
- Real tokenizer/llama vocab: P2.
- Prefix-cache / batching: P3–P4.
