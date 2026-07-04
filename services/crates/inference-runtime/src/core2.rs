//! Core 2 — the fast loop (P2).
//!
//! Scheduler + backend driver. Drains tokenized requests off R2 into a pending
//! queue and runs them through a [`Decoder`] one sequence at a time (P2 =
//! single sequence; continuous batching is P3). The mock decoder replays canned
//! bytes; the real one prefills + greedy-decodes on an owned `LlamaContext`.
//! Emission (R3) and shutdown are backend-independent. See `design/fast-loop/`.

use crate::affinity;
use crate::backend::{Decoder, DecoderInit};
use crate::rings::RingEvent;
use crate::slab::Slab;
use disruptor::{EventPoller, Polling, Producer, SingleProducerBarrier};
use std::collections::VecDeque;
use std::sync::Arc;

/// Run the Core 2 loop until R2 is shut down and the pending queue is drained.
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
    let mut pending: VecDeque<u32> = VecDeque::new();
    let mut r2_shutdown = false;

    loop {
        // Admit newly tokenized requests (non-blocking drain of R2).
        match r2.poll() {
            Ok(mut guard) => {
                for ev in &mut guard {
                    pending.push_back(ev.slot);
                }
            }
            Err(Polling::NoEvents) => {}
            Err(Polling::Shutdown) => r2_shutdown = true,
        }

        // Process one sequence to completion (single-seq P2). Re-poll R2 between
        // sequences so ingress never blocks on a full ring.
        if let Some(slot) = pending.pop_front() {
            decoder.run_sequence(slot, &slab, &mut r3);
        }

        if r2_shutdown && pending.is_empty() {
            break;
        }
        std::hint::spin_loop();
    }
}
