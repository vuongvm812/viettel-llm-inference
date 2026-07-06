//! Core 2 — the fast loop (P3: continuous batching).
//!
//! Scheduler + backend driver. Drains tokenized requests off R2 into a pending
//! queue, admits them into the [`Decoder`]'s running set (up to `max_batch_seqs`
//! and a per-iteration prompt-token budget), then `step`s the whole running set
//! once per iteration — many sequences share one `decode()`, new ones join the
//! next step, finished ones retire immediately (the vLLM-style throughput win).
//! The mock decoder replays canned bytes; the real one batches prefill + greedy
//! decode on an owned `LlamaContext`. Emission (R3) and shutdown are
//! backend-independent. See `design/fast-loop/`.

use crate::affinity;
use crate::backend::{Admit, Decoder, DecoderInit};
use crate::rings::RingEvent;
use crate::slab::Slab;
use disruptor::{EventPoller, Polling, Producer, SingleProducerBarrier};
use std::collections::VecDeque;
use std::sync::Arc;

/// Run the Core 2 loop until R2 is shut down and every sequence has drained.
pub fn run(
    core: usize,
    slab: Arc<Slab>,
    mut r2: EventPoller<RingEvent, SingleProducerBarrier>,
    mut r3: impl Producer<RingEvent>,
    init: DecoderInit,
) {
    affinity::pin(core); // no-op on macOS
    // Build the decoder here, on Core 2's own thread: the real `LlamaContext` is
    // single-threaded and thread-affine, so it must never be created elsewhere
    // and moved in.
    let mut decoder = Decoder::new(init);
    // Size the pending queue to the slab so a full backlog never reallocates on the
    // hot loop.
    let mut pending: VecDeque<u32> = VecDeque::with_capacity(slab.capacity());
    let mut r2_shutdown = false;

    loop {
        // Ingest: non-blocking drain of R2 into the pending queue.
        match r2.poll() {
            Ok(mut guard) => {
                for ev in &mut guard {
                    pending.push_back(ev.slot);
                }
            }
            Err(Polling::NoEvents) => {}
            Err(Polling::Shutdown) => r2_shutdown = true,
        }

        // Admit: fill the running set while it has room and the shared KV budget allows.
        //
        // P7 chunked prefill removed the per-iteration prefill-token admission gate: a
        // long prompt is admitted immediately as a Prefilling seq and consumes a chunk of
        // its prompt per decode step (the `max_batch_tokens` budget now lives *inside*
        // `step`, bounding prompt tokens prefilled per unified batch — not whether to
        // admit). So admission is purely capacity + KV reservation; no prompt ever
        // serializes the running set behind a head-of-line prefill.
        while decoder.has_capacity() {
            let Some(&slot) = pending.front() else { break };
            match decoder.admit(slot, &slab, &mut r3) {
                Admit::Admitted(_) => {
                    pending.pop_front();
                }
                // KV budget full right now (or the matching prefix is still mid-establish):
                // leave it queued and let the decode step below make progress (retire a
                // seq / finish an establisher's prefill), then retry next iteration.
                // ponytail: strict FIFO — a large front request that can't fit blocks
                // smaller later ones that could, underfilling the batch. Upgrade path is
                // a bounded scan over `pending` (admit fitting ones) with an age/scan
                // limit to keep the front from starving; deferred until fill-rate bites.
                Admit::Deferred => break,
                // Can never fit (backend already published Finish(Error)): drop it.
                Admit::Rejected => {
                    pending.pop_front();
                }
            }
        }

        // Decode one step over the whole running set (this is the batching).
        // `step` emits the whole per-step token burst in one `r3.batch_publish` (the
        // design's R3 throughput sweet spot) — the burst is `active.len()` events,
        // which is <= max_batch_seqs <= max_inflight <= ring_size (config-enforced),
        // so it always fits and can never be a batch larger than the ring. Retire
        // `Finish`es (a minority) are published individually. This is bounded
        // backpressure, not a deadlock risk: the only cyclic wait would be Core 1
        // blocked on an R2 publish while Core 2 is blocked on R3, but R2 can never fill
        // — each slot has at most one outstanding R2 entry per lifecycle, so live R2
        // entries <= max_inflight <= ring_size, and Core 1 never blocks publishing to
        // R2. R3 pressure therefore only ever traces back to Core 0 draining R4, which
        // is independent and always makes progress.
        if !decoder.is_idle() {
            decoder.step(&slab, &mut r3);
        }

        if r2_shutdown && pending.is_empty() && decoder.is_idle() {
            break;
        }
        std::hint::spin_loop();
    }
}
