# Design — Core 0: Web I/O & Streaming

The "Netty equivalent" in Rust. Pinned to **Core 0**. I/O-bound, **event-driven (epoll),
never busy-spin** — spinning here would starve the async runtime.

## Responsibilities

- OpenAI-compatible HTTP server (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`).
- Ingress: parse request → claim a slab slot → publish `{slot, New}` on **R1** (multi-producer).
- Egress: drain **R4** cooperatively → stream text pieces to the right connection as SSE.
- Own the socket side of each request (the `EgressHandle` stored in the slab).

## Runtime

- `tokio` **current-thread** runtime (single OS thread, pinned to Core 0 via `core_affinity`)
  + `hyper`/`axum`. One core, one reactor; concurrency is via async tasks, not threads.
- Rust equivalent of the user's "Netty + SSE/WebSocket streamer". SSE is the v1 protocol
  (matches OpenAI/vLLM `stream: true`); WebSocket is a P7 stretch.

## Ingress path

```
POST /v1/chat/completions
  → deserialize body (serde) into RequestParams
  → slot = slab.free.pop()            // if none: 503 / queue (backpressure)
  → write body + prompt into slab[slot]; register EgressHandle
  → r1_producer.publish(|e| *e = RingEvent{slot, New})
  → keep the HTTP response open for streaming
```

- One `MultiProducer` for R1, `clone()`d into each connection task (crate: `MultiProducer:
  Clone`). Publishing is lock-free.
- The ~40K-char prompt is written **once** into `slab[slot].prompt`; only `{slot, New}`
  (8 bytes) goes on the ring.
- If the free-list is empty (all slots in flight), either return `503` or park the request in
  a small bounded queue — a deliberate admission cap tied to `MAX_IN_FLIGHT`.

## Egress path (the tricky part: no busy-spin on Core 0)

R4's consumer must run on Core 0 **without** a busy-spin thread. Use `new_event_poller()` and
drive it from inside the tokio loop:

```rust
// A tokio task that yields to the reactor between polls.
loop {
    match r4_poller.poll() {
        Ok(mut guard) => {
            for ev in &mut guard {                 // EventGuard yields &RingEvent
                let s = &slab[ev.slot];
                match ev.kind {
                    Token(_) => s.conn.send_chunk(s.newly_appended_out_bytes()),
                    Finish(reason) => { s.conn.finish(reason); slab.free.push(ev.slot); }
                    _ => {}
                }
            }
        }                                          // drop(guard) advances the cursor
        Err(Polling::NoEvents) => tokio::task::yield_now().await,   // let epoll run
        Err(Polling::Shutdown) => break,
    }
}
```

- `has_available()` (cheap, no fence) can gate whether to `poll()` at all, so an idle system
  parks on epoll instead of hot-looping. If latency demands it, poll a few times before
  yielding (short spin, then cooperate) — but never a dedicated full-spin thread on Core 0.
- **Dropping the `EventGuard` is mandatory** — it commits the consumed sequence and releases
  R4 backpressure.

## slot ↔ connection mapping

The `EgressHandle` lives **in the slab** (`slab[slot].conn`), so egress needs no side map —
the ring event carries the `slot`, the slot carries the connection. `EgressHandle` is the
sender half of a per-connection stream (e.g. `hyper` body channel / `tokio::sync::mpsc` feeding
the SSE response body). Sending a chunk is the one unavoidable allocation boundary (bytes go
to the kernel anyway); keep pieces batched to reduce syscalls without hurting TTFT.

## SSE framing

- Each `Token` → one `data: {OpenAI delta chunk}\n\n` frame (or a small batch of tokens per
  frame to cut overhead — trade TTFT/ITL smoothness vs syscalls; make it a knob).
- `Finish` → final `data: {..finish_reason..}\n\n` then `data: [DONE]\n\n`, close body, return
  the slot to the free-list.
- Non-streaming requests (`stream:false`) buffer `out_bytes` in the slab and reply once on `Finish`.

## Determinism / correctness

- Pass `temperature`, `seed`, `max_tokens`, `stop` straight through to the slab so Core 2 can
  reproduce vLLM's deterministic decoding (trace uses `temp=0, seed=42`).
- Validate the request at this trust boundary (model name, `max_tokens` bounds, JSON shape)
  before claiming a slot — reject early, don't burn a slot on a bad request.

## Open questions for implementation

- Exact `hyper` streaming body type for SSE (channel vs `Stream` impl) — pick the one that
  lets Core 0's poller push bytes without `await` per token.
- Whether to pin the tokio reactor thread explicitly (`core_affinity::set_for_current`) at
  startup — yes on Linux; no-op on macOS.
