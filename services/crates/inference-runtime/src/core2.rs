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
        // isn't spent.
        //
        // Invariant: `max_batch_tokens` is a *per-iteration prefill-token budget* — it
        // caps how many prompt tokens Core 2 prefills before running the next decode
        // step, so a burst of long prompts can't starve the decode of already-active
        // seqs. It is NOT a total batch-token budget (active decode tokens aren't
        // charged against it). We peek the next prompt's length *before* prefilling and
        // stop once admitting it would blow the budget.
        //
        // Idle-progress exception: when the running set is empty and we've admitted
        // nothing yet this iteration, admit the first prompt even if it exceeds the
        // budget — otherwise a prompt larger than the whole budget could never start.
        // A long prompt that arrives *while sequences are already decoding* now waits
        // (defers to a later iteration) instead of blocking active decode with a full
        // prefill. It admits once the set drains to idle. Splitting one prefill across
        // iterations (chunked prefill) is the P7 fix that removes even the idle stall.
        let mut admitted_tokens = 0u32;
        while decoder.has_capacity() {
            let Some(&slot) = pending.front() else { break };
            // SAFETY: `slot` sits in Core 2's `pending` (arrived via R2, no R3 publish
            // yet) → Core 2 solely owns it.
            let plen = unsafe { slab.prompt_len(slot) } as u32;
            let over_budget = admitted_tokens.saturating_add(plen) > max_batch_tokens;
            let idle_progress = decoder.is_idle() && admitted_tokens == 0;
            if over_budget && !idle_progress {
                break; // budget spent (or a long prefill deferred) — decode first
            }
            match decoder.admit(slot, &slab, &mut r3) {
                Admit::Admitted(n_prompt) => {
                    pending.pop_front();
                    admitted_tokens = admitted_tokens.saturating_add(n_prompt as u32);
                }
                // KV budget full right now: leave it queued and let the decode step
                // below retire a seq and free cells, then retry next iteration.
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
        // `step` emits tokens/finishes with *blocking* R3 publishes. That's bounded
        // backpressure, not a deadlock risk: the only cyclic wait would be Core 1
        // blocked on an R2 publish while Core 2 is blocked on R3, but R2 can never fill
        // — each slot has at most one outstanding R2 entry per lifecycle, so live R2
        // entries <= max_inflight <= ring_size (config-enforced), and Core 1 never
        // blocks publishing to R2. R3 pressure therefore only ever traces back to Core 0
        // draining R4, which is independent and always makes progress. Batching the
        // per-step emission into one `r3.batch_publish` (design note) is a target-box
        // perf refinement, not a correctness fix — and must chunk to <= ring_size to
        // avoid a batch that can never fit.
        if !decoder.is_idle() {
            decoder.step(&slab, &mut r3);
        }

        if r2_shutdown && pending.is_empty() && decoder.is_idle() {
            break;
        }
        std::hint::spin_loop();
    }
}
