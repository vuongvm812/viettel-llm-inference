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
use crate::backend::{prompt_len, Admit, Decoder, DecoderInit};
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
    max_batch_tokens: u32,
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

        // Admit: fill the running set while it has room, the shared KV budget allows
        // it (the decoder's reservation), and this iteration's prefill-token budget
        // isn't spent. Bounding prefill tokens per iteration keeps a burst of long
        // prompts from starving the decode of already-active seqs — we peek the next
        // prompt's length *before* prefilling it, and stop once the budget is spent.
        // ponytail: a single prompt bigger than the whole budget still admits when
        // nothing's been admitted yet — one long prefill still blocks decode for its
        // duration (head-of-line blocking). Splitting one prefill across iterations
        // (chunked prefill) is P7.
        let mut admitted_tokens = 0u32;
        while decoder.has_capacity() {
            let Some(&slot) = pending.front() else { break };
            let plen = prompt_len(&slab, slot) as u32;
            if admitted_tokens > 0 && admitted_tokens.saturating_add(plen) > max_batch_tokens {
                break; // budget spent — run a decode step before prefilling more
            }
            match decoder.admit(slot, &slab, &mut r3) {
                Admit::Admitted(n_prompt) => {
                    pending.pop_front();
                    admitted_tokens = admitted_tokens.saturating_add(n_prompt as u32);
                }
                // KV budget full right now: leave it queued and let the decode step
                // below retire a seq and free cells, then retry next iteration.
                Admit::Deferred => break,
                // Can never fit (backend already published Finish(Error)): drop it.
                Admit::Rejected => {
                    pending.pop_front();
                }
            }
        }

        // Decode one step over the whole running set (this is the batching).
        if !decoder.is_idle() {
            decoder.step(&slab, &mut r3);
        }

        if r2_shutdown && pending.is_empty() && decoder.is_idle() {
            break;
        }
        std::hint::spin_loop();
    }
}
