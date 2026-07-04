# Design — Core 1: Text Processing (Tokenize + Detokenize)

Pinned to **Core 1**. CPU-bound, bursty. Sits between Core 0 and Core 2 on the ingress side
and between Core 2 and Core 0 on the egress side. One thread, one loop, two rings in, two
rings out.

## Responsibilities

- **Tokenize** (R1 → R2): prompt text → token ids, using llama.cpp's own vocab.
- **Detokenize** (R3 → R4): generated token ids → UTF-8 text pieces.
- Do this without heap churn on the hot path (buffers live in the slab).

## Why llama.cpp's vocab (not a separate tokenizer)

Tokenize/detokenize must use the **exact vocab the model was trained/quantized with**, or
token ids won't match the GGUF model → garbage output. llama.cpp exposes tokenize and
token-to-piece over the model's const vocab. We therefore hold a shared **`Arc<LlamaModel>`**
(the same model Core 2 loads) and call vocab functions on it.

**Send/Sync safety.** Tokenize / `token_to_piece` read only the immutable vocab; they do not
touch the `LlamaContext` or KV cache. So sharing `LlamaModel` read-only across Core 1 and
Core 2 is sound while Core 2 mutates its own `LlamaContext`. (Confirm the `llama-cpp-2` types
are `Send + Sync` for the model handle; wrap in a thin `unsafe`-audited newtype if the crate
is conservative.)

## The Core 1 loop (owns its thread; two pollers, two producers)

```rust
core_affinity::set_for_current(core_1);          // no-op on macOS
loop {
    // ingress: tokenize
    if let Ok(mut g) = r1_poller.poll() {
        for ev in &mut g {                        // ev.kind == New
            let s = &mut slab[ev.slot];
            model.tokenize_into(&s.prompt, &mut s.tokens);   // append into pre-reserved buf
            r2_producer.publish(|e| *e = RingEvent{slot: ev.slot, kind: New});
        }
    }
    // egress: detokenize
    if let Ok(mut g) = r3_poller.poll() {
        for ev in &mut g {                        // ev.kind == Token(id) | Finish
            let s = &mut slab[ev.slot];
            match ev.kind {
                Token(id) => {
                    model.token_to_piece_into(id, &mut s.out_bytes);  // append UTF-8
                    r4_producer.publish(|e| *e = RingEvent{slot: ev.slot, kind: Token(id)});
                }
                Finish(r) => r4_producer.publish(|e| *e = RingEvent{slot: ev.slot, kind: Finish(r)}),
                _ => {}
            }
        }
    }
    // both idle → spin hint (dedicated core) then loop
    std::hint::spin_loop();
}
```

- Uses `new_event_poller()` for R1 and R3 so both consume + both produce happen in one Core-1
  thread. Pin the thread manually (`core_affinity`) since poller threads aren't pinned by the crate.
- **Drop each `EventGuard`** (end of the `for` scope) to advance cursors / release backpressure.

## Wait strategy

Dedicated core, bursty load (40 req/s, but each prompt ~40K chars). Options:

- `BusySpinWithSpinLoopHint` inside the manual loop (shown above) — lowest hand-off latency,
  acceptable because Core 1 is dedicated.
- **Spin-then-park** refinement: spin for a bounded number of empty iterations, then a short
  `Sleep`/`park_timeout` to avoid cooking the core when fully idle. Recommended default —
  tokenizing 40K-char prompts is real work, but between bursts the core shouldn't spin at 100%.

Make the backoff a tuning knob; measure TTFT impact on Linux.

## Buffers — no heap churn on the hot path

- `s.tokens` (prompt ids) and `s.out_bytes` (output text) are owned by the slab with
  **pre-reserved capacity**. Tokenize/detokenize *append* into them — no per-token allocation.
- A prompt longer than the reserved token capacity reallocates `s.tokens` once at ingress
  (off the decode hot path) — acceptable.

## Detokenize batching vs TTFT (trade-off)

Core 2 can emit tokens one-at-a-time (best TTFT/ITL) or in small bursts via R3 `batch_publish`
(fewer ring ops, fewer SSE frames). Detokenize honors whatever granularity arrives. Streaming
smoothness (per-token) vs efficiency (per-burst) is a knob shared with Core 0's SSE framing;
default to per-token for the first token (fast TTFT) and allow small bursts thereafter.

## Correctness notes

- Multi-byte UTF-8: a single token may be a partial code point; `token_to_piece` handles byte
  fallback, but the SSE layer must not split a frame mid-code-point — buffer partial bytes in
  `out_bytes` and only emit complete UTF-8 (llama.cpp piece output is already byte-safe; verify).
- `demo()` self-check: tokenize a known string, detokenize back, assert round-trip equals the
  original for ASCII and a multi-byte sample.
