# Design — Disruptor Pipeline & Request Slab

The lock-free fabric connecting the three cores. This is the heart of the runtime; the other
components plug into it.

## Responsibilities

- Move request *handles* (not payloads) between cores with minimal latency, no locks, no
  per-message heap allocation.
- Own the pre-allocated request storage (`RequestSlab`) and its lock-free free-list.
- Define the ring event types and the exact `disruptor-rs` wiring for each of the 4 rings.

## The event type (crosses every ring)

```rust
#[derive(Clone, Copy)]
struct RingEvent {
    slot: u32,        // index into RequestSlab
    kind: EventKind,
}

#[derive(Clone, Copy)]
enum EventKind {
    New,              // R1, R2: a fresh request to process
    Token(u32),       // R3, R4: one generated token id / piece for `slot`
    Finish(FinishReason), // stream complete: Eos | MaxTokens | Error
}
```

`RingEvent` is `Copy`, ~8 bytes — no heap, no `Drop`. This is the *only* thing that travels
through a ring. All four rings share the type (a single enum keeps handlers monomorphic and
lets `Token`/`Finish` ride the same ring).

> For decode, Core 2 may emit several tokens per iteration across sequences — use
> `batch_publish(n, ..)` on R3 to publish a burst in one shot (Disruptor's batch path is the
> throughput sweet spot per the crate benchmarks).

## RequestSlab — pre-allocated, zero-copy storage

```rust
struct RequestState {
    // filled by Core 0
    body:       RequestParams,        // model, max_tokens, temp, seed, stop, ...
    prompt:     Box<str>,             // written once by Core 0; never copied through a ring
    conn:       EgressHandle,         // Core 0 owns the socket side (see web-io)
    // filled by Core 1
    tokens:     Vec<i32>,             // prompt token ids (llama vocab); capacity pre-reserved
    // filled by Core 2
    seq_id:     i32,                  // llama.cpp sequence id
    n_prompt:   u32,
    n_generated:u32,
    // filled by Core 1 (detokenize) / drained by Core 0
    out_bytes:  Vec<u8>,             // appended text pieces; capacity pre-reserved
    // lifecycle
    generation: u32,                 // ABA guard for the free-list / stale events
}

struct RequestSlab {
    slots: Box<[UnsafeCell<RequestState>]>,  // fixed capacity = MAX_IN_FLIGHT, allocated at startup
    free:  FreeList,                          // lock-free
}
```

- `slots` allocated **once** at startup. `MAX_IN_FLIGHT` is bounded by KV memory anyway
  (see fast-loop), so the slab is small (e.g. 256).
- Each slot is written by exactly one core at a time as the handle advances through the
  pipeline (Core 0 → 1 → 2 → 1 → 0). This single-writer-at-a-stage invariant is what makes
  `UnsafeCell` sound without locks — the ring hand-off is the synchronization edge (a
  Disruptor publish/consume pair is a release/acquire, matching the crate's memory model).
- `prompt`/`tokens`/`out_bytes` own their buffers; capacity is pre-reserved so steady-state
  generation appends without reallocating. (A prompt larger than reserved capacity reallocates
  once on ingress — acceptable, off the decode hot path.)

### Free-list (lock-free)

Treiber stack of slot indices over an `AtomicU32` head with a `generation` ABA guard, or a
dedicated SPSC index ring. Claim on Core 0 ingress (`pop`), return on Core 0 after `Finish`
(`push`). Only Core 0 pushes/pops → contention is trivial; a simple atomic stack suffices.

> ponytail: single-producer/single-consumer free-list (only Core 0 touches it). If ingress
> ever moves off Core 0, switch to a Treiber stack with ABA guard — the `generation` field is
> already there for it.

## Ring-by-ring wiring (against the verified `disruptor-rs` API)

Sizes are powers of two; multi-producer requires ≥ 64.

| Ring | Producer | Consumer | Builder | Wait strategy (consumer thread) |
|---|---|---|---|---|
| R1 | Core 0 (many conns) | Core 1 tokenize | `build_multi_producer` | Core 1: `BusySpinWithSpinLoopHint` |
| R2 | Core 1 | Core 2 schedule | `build_single_producer` | Core 2: `BusySpin` |
| R3 | Core 2 | Core 1 detokenize | `build_single_producer` | Core 1: `BusySpinWithSpinLoopHint` |
| R4 | Core 1 | Core 0 egress | `build_single_producer` | Core 0: **poller**, not spin |

- **R1** is multi-producer because each HTTP connection task may publish. Core 0 holds one
  `MultiProducer` and `clone()`s a handle per producer context.
- **R2/R3** are single-producer (one core each side). Core 1 hosts two consumers (R1→ and
  R3→) and two producers (→R2, →R4); these are independent rings, not one Disruptor with
  interdependent stages, so **no `.and_then()`** is needed here — the topology is a cycle of
  separate rings, which the crate's linear/DAG builder can't express as one graph anyway.
- **R4's consumer runs on Core 0** which must stay event-driven. Use `new_event_poller()` and
  drive `poller.poll()` / `has_available()` cooperatively inside the tokio loop (see web-io),
  rather than a managed busy-spin thread that would starve async I/O.

### Handler/consumer forms

- Core 1 and Core 2 that **own their thread** can use either managed handlers
  (`handle_events_with`, Disruptor spawns + pins the thread) or `new_event_poller` (we own the
  loop). Core 1 needs to both consume *and* produce onto other rings inside one thread → use
  **`new_event_poller`** for R1 and R3 and drive both plus the `libllama` vocab calls in one
  Core-1 loop. Same for Core 2 (poll R2, run decode, publish R3).
- Core 0 uses `new_event_poller` for R4, integrated into tokio.
- Managed `handle_events_with` + `.pin_at_core(n)` is used only if a stage is a pure sink; our
  cores are all produce-and-consume, so pollers dominate. Pin the poller threads manually via
  `core_affinity` (the crate pins managed threads for you, but not poller threads — you own them).

## Backpressure & flow control

- A ring fills when its consumer lags. `Producer::publish`/`batch_publish` **spin until
  slots free**; `try_publish` returns `RingBufferFull`. On the hot decode path use `try_*` and
  let the scheduler skip emitting rather than stall Core 2's GPU loop.
- A poller that stops polling applies backpressure upstream (the crate enforces this). Core 2
  must keep R2 drained (admit into pending queue quickly) so Core 1 never stalls.
- End-to-end admission is really governed by KV memory (fast-loop), not ring depth; rings are
  sized generously (e.g. 1024) so they are never the bottleneck.

## Shutdown

Dropping the last `MultiProducer`/`SingleProducer` signals consumers to drain then stop
(crate semantics). Graceful shutdown: stop accepting on Core 0, let in-flight slots finish,
drop producers, join poller loops.

## Invariants / checks (for implementation)

- Single-writer-per-stage on each slot (assert `generation` matches on every ring receive;
  stale event → drop).
- Each ring is strictly one-directional; the only "loop" is generation feedback **inside**
  Core 2 (no ring), so the graph is acyclic across rings → no deadlock.
- A minimal `demo()` self-check: push K synthetic requests through R1→R2→R3→R4 with a mock
  Core 2, assert all K slots return to the free-list and out_bytes are ordered per slot.
