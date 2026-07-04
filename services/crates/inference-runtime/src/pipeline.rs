//! Ring wiring + worker spawn (P1).
//!
//! Builds the 4 one-directional Disruptor rings, spawns the Core 1 and Core 2
//! threads, and hands the caller the Core-0-side handles (R1 producer to publish
//! ingress, R4 poller to drain egress). Reused by both `main` (real Core 0 over
//! HTTP) and the pipeline integration test (Core 0 played by the test thread).

use crate::config::Config;
use crate::rings::RingEvent;
use crate::slab::Slab;
use crate::{core1, core2};
use disruptor::{
    build_multi_producer, build_single_producer, BusySpin, BusySpinWithSpinLoopHint, EventPoller,
    MultiProducer, SingleConsumerBarrier, SingleProducerBarrier,
};
use std::sync::Arc;
use std::thread::JoinHandle;

/// R1 ingress producer (owned by Core 0). Multi-producer: one clone per HTTP task.
pub type IngressProducer = MultiProducer<RingEvent, SingleConsumerBarrier>;
/// R4 egress poller (owned by Core 0), drained cooperatively from the tokio loop.
pub type EgressPoller = EventPoller<RingEvent, SingleProducerBarrier>;

/// Core-0-side handles plus the worker thread joins.
pub struct Pipeline {
    pub ingress: IngressProducer,
    pub egress: EgressPoller,
    pub core1: JoinHandle<()>,
    pub core2: JoinHandle<()>,
}

/// Build the rings and spawn Core 1 + Core 2. Dropping [`Pipeline::ingress`]
/// starts a clean shutdown that cascades R1→R2→R3→R4 and lets both threads join.
pub fn spawn(cfg: &Config, slab: Arc<Slab>) -> Pipeline {
    let size = cfg.runtime.ring_size as usize;
    let (c1_core, c2_core) = (cfg.runtime.cores.text, cfg.runtime.cores.fast_loop);

    // Load the backends before wiring: the real `load` reads the GGUF and leaks
    // the model to `'static` so Core 1 (vocab) and Core 2 (context) can share it.
    // The mock `load` is a no-op. Startup fails loudly here on a bad model.
    let (text, decoder_init) = crate::backend::load(cfg);

    // Every consumer here is an EventPoller (non-blocking `poll()`), so the wait
    // strategy passed below governs only *producer* back-pressure spin, not
    // consumption. The consume-side wait is the hand-coded loop in each core
    // (spin_loop on cores 1/2, cooperative sleep on Core 0). Strategies still
    // match the design table for when producers block on a full ring.
    //
    // Build each ring, splitting it into (poller for the consumer, producer for
    // the producer). R1 is multi-producer (many HTTP conns); R2–R4 single.
    let (r1_poller, r1_builder) =
        build_multi_producer(size, RingEvent::default, BusySpinWithSpinLoopHint).new_event_poller();
    let ingress = r1_builder.build();

    let (r2_poller, r2_builder) =
        build_single_producer(size, RingEvent::default, BusySpin).new_event_poller();
    let r2_producer = r2_builder.build();

    let (r3_poller, r3_builder) =
        build_single_producer(size, RingEvent::default, BusySpinWithSpinLoopHint).new_event_poller();
    let r3_producer = r3_builder.build();

    let (r4_poller, r4_builder) =
        build_single_producer(size, RingEvent::default, BusySpin).new_event_poller();
    let r4_producer = r4_builder.build();

    let slab1 = Arc::clone(&slab);
    let core1 = std::thread::Builder::new()
        .name("core1-text".into())
        .spawn(move || {
            core1::run(c1_core, slab1, r1_poller, r2_producer, r3_poller, r4_producer, text);
        })
        .expect("spawn core1");

    let slab2 = slab;
    let max_batch_tokens = cfg.runtime.max_batch_tokens;
    let core2 = std::thread::Builder::new()
        .name("core2-fastloop".into())
        .spawn(move || {
            core2::run(c2_core, slab2, r2_poller, r3_producer, decoder_init, max_batch_tokens);
        })
        .expect("spawn core2");

    Pipeline {
        ingress,
        egress: r4_poller,
        core1,
        core2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CANNED_REPLY;
    use crate::rings::{EventKind, RingEvent};
    use crate::slab::Slab;
    use disruptor::{Polling, Producer};

    fn test_cfg(max_inflight: u32, ring_size: u32) -> Config {
        // Defaults big enough not to constrain the older tests: cap == slots, budget
        // and n_ctx generous.
        test_cfg_with(max_inflight, ring_size, max_inflight, 100_000, 1_000_000)
    }

    /// Config with the P3 batching knobs (and `n_ctx`) exposed, so tests can force the
    /// max-batch-seqs cap and the KV-reservation defer path.
    fn test_cfg_with(
        max_inflight: u32,
        ring_size: u32,
        max_batch_seqs: u32,
        max_batch_tokens: u32,
        n_ctx: u32,
    ) -> Config {
        let yaml = format!(
            r#"
server: {{ host: "127.0.0.1", port: 0 }}
model: {{ gguf_path: "x", n_ctx: {n_ctx}, n_threads: 1, n_gpu_layers: -1 }}
runtime:
  max_inflight: {max_inflight}
  ring_size: {ring_size}
  max_batch_seqs: {max_batch_seqs}
  max_batch_tokens: {max_batch_tokens}
  cores: {{ web_io: 0, text: 1, fast_loop: 2 }}
"#
        );
        serde_yaml::from_str(&yaml).expect("cfg")
    }

    /// The design's `demo()` self-check: drive K requests through R1→R2→R3→R4
    /// with the test thread playing Core 0. A capacity smaller than K forces slot
    /// recycling under backpressure — proving no deadlock, that every slot returns
    /// to the free-list, and that each request's output is its full canned reply.
    #[test]
    fn pipeline_streams_all_requests_and_recycles_slots() {
        const CAP: u32 = 8;
        const K: usize = 120;
        let cfg = test_cfg(CAP, 1024);
        let slab = Arc::new(Slab::new(CAP as usize, 128, 128, 256));

        let Pipeline {
            mut ingress,
            mut egress,
            core1,
            core2,
        } = spawn(&cfg, Arc::clone(&slab));

        let mut free: Vec<u32> = (0..CAP).rev().collect();
        let mut outputs: Vec<Vec<u8>> = vec![Vec::new(); K];
        let mut slot_req: Vec<usize> = vec![usize::MAX; CAP as usize];
        // Core-0-side read cursor into each slot's `out_bytes` (reset on claim).
        let mut cursor: Vec<usize> = vec![0; CAP as usize];
        let mut next_req = 0usize;
        let mut finished = 0usize;

        while finished < K {
            // Publish while slots and requests remain (Core 0 ingress).
            while next_req < K {
                let Some(slot) = free.pop() else { break };
                // SAFETY: slot just came off the free-list; no core holds it.
                unsafe {
                    let s = slab.slot_mut(slot);
                    s.reset();
                    s.prompt.push_str("hello");
                    s.max_tokens = 200; // > CANNED_REPLY.len() → full reply, Eos
                }
                cursor[slot as usize] = 0;
                slot_req[slot as usize] = next_req;
                ingress.publish(|e| {
                    *e = RingEvent {
                        slot,
                        kind: EventKind::New,
                    }
                });
                next_req += 1;
            }

            // Drain egress (Core 0 egress): a Token event carries the count of
            // newly-committed complete-UTF-8 bytes; read them from the slab.
            match egress.poll() {
                Ok(mut guard) => {
                    for ev in &mut guard {
                        let req = slot_req[ev.slot as usize];
                        match ev.kind {
                            EventKind::Piece(delta) => {
                                let c = cursor[ev.slot as usize];
                                // SAFETY: Core 1 committed + published [c, c+delta).
                                let piece = unsafe { slab.read_committed(ev.slot, c, delta as usize) }
                                    .expect("committed range in bounds");
                                outputs[req].extend_from_slice(piece);
                                cursor[ev.slot as usize] = c + delta as usize;
                            }
                            EventKind::Finish(_) => {
                                free.push(ev.slot);
                                finished += 1;
                            }
                            EventKind::Token(_) | EventKind::New => {}
                        }
                    }
                }
                Err(Polling::NoEvents) => std::hint::spin_loop(),
                Err(Polling::Shutdown) => break,
            }
        }

        // Clean shutdown: drop ingress → cascade R1→R2→R3→R4, join workers.
        drop(ingress);
        core1.join().expect("core1 join");
        core2.join().expect("core2 join");

        assert_eq!(finished, K, "every request finished");
        assert_eq!(free.len(), CAP as usize, "every slot returned to free-list");
        for (r, out) in outputs.iter().enumerate() {
            assert_eq!(out, CANNED_REPLY, "request {r} streamed the full canned reply");
        }
    }

    /// Continuous batching (P3): with several requests admitted at once, Core 2 must
    /// interleave their decode steps — many sequences in flight per `decode()` — rather
    /// than running each to completion before the next. Observed from Core 0's egress:
    /// count how many slots are "open" (streamed a `Piece`, not yet `Finish`) at the
    /// same time; the peak must exceed 1. Under the P2 single-shot loop the peak is
    /// exactly 1 (slot N fully finishes before slot N+1 starts) → this fails until the
    /// fast loop batches.
    #[test]
    fn continuous_batching_runs_multiple_seqs_concurrently() {
        const CAP: u32 = 4;
        const K: usize = 4;
        let cfg = test_cfg(CAP, 1024);
        let slab = Arc::new(Slab::new(CAP as usize, 128, 128, 256));
        let Pipeline { mut ingress, mut egress, core1, core2 } = spawn(&cfg, Arc::clone(&slab));

        // Admit all K at once (CAP == K, so no recycling) — full canned reply each.
        for slot in 0..CAP {
            // SAFETY: each slot came off no ring yet; test owns it.
            unsafe {
                let s = slab.slot_mut(slot);
                s.reset();
                s.prompt.push_str("hello");
                s.max_tokens = 200; // > CANNED_REPLY.len() → full reply, Eos
            }
            ingress.publish(|e| *e = RingEvent { slot, kind: EventKind::New });
        }

        let mut open: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut max_open = 0usize;
        let mut finished = 0usize;
        while finished < K {
            match egress.poll() {
                Ok(mut guard) => {
                    for ev in &mut guard {
                        match ev.kind {
                            EventKind::Piece(_) => {
                                open.insert(ev.slot);
                                max_open = max_open.max(open.len());
                            }
                            EventKind::Finish(_) => {
                                open.remove(&ev.slot);
                                finished += 1;
                            }
                            EventKind::Token(_) | EventKind::New => {}
                        }
                    }
                }
                Err(Polling::NoEvents) => std::hint::spin_loop(),
                Err(Polling::Shutdown) => break,
            }
        }
        drop(ingress);
        core1.join().expect("core1 join");
        core2.join().expect("core2 join");

        assert_eq!(finished, K, "every request finished");
        assert!(
            max_open >= 2,
            "expected ≥2 sequences streaming concurrently (continuous batching), saw peak {max_open}"
        );
    }

    /// The *continuous* half of continuous batching: sequences admitted together but
    /// with different `max_tokens` must retire at different steps while the rest keep
    /// decoding — a sequence still streams `Piece`s after another has already
    /// `Finish`ed. Also pins that interleaving never corrupts a slot's output: each
    /// request's bytes are exactly its truncated canned reply.
    #[test]
    fn heterogeneous_max_tokens_retire_mid_batch() {
        const CAP: u32 = 4;
        let max_toks = [4u32, 8, 16, 200]; // distinct targets → staggered retirement
        let cfg = test_cfg(CAP, 1024);
        let slab = Arc::new(Slab::new(CAP as usize, 128, 128, 256));
        let Pipeline { mut ingress, mut egress, core1, core2 } = spawn(&cfg, Arc::clone(&slab));

        let reply_len = CANNED_REPLY.len() as u32;
        for slot in 0..CAP {
            // SAFETY: each slot came off no ring yet; test owns it.
            unsafe {
                let s = slab.slot_mut(slot);
                s.reset();
                s.prompt.push_str("hi");
                s.max_tokens = max_toks[slot as usize];
            }
            ingress.publish(|e| *e = RingEvent { slot, kind: EventKind::New });
        }

        let mut outputs: Vec<Vec<u8>> = vec![Vec::new(); CAP as usize];
        let mut cursor = vec![0usize; CAP as usize];
        let mut finished: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut retire_mid_batch = false;
        while finished.len() < CAP as usize {
            match egress.poll() {
                Ok(mut guard) => {
                    for ev in &mut guard {
                        match ev.kind {
                            EventKind::Piece(delta) => {
                                // A still-open slot streaming after another already
                                // finished = retirement happened mid-batch.
                                if !finished.is_empty() && !finished.contains(&ev.slot) {
                                    retire_mid_batch = true;
                                }
                                let c = cursor[ev.slot as usize];
                                let piece = unsafe { slab.read_committed(ev.slot, c, delta as usize) }
                                    .expect("committed range in bounds");
                                outputs[ev.slot as usize].extend_from_slice(piece);
                                cursor[ev.slot as usize] = c + delta as usize;
                            }
                            EventKind::Finish(_) => {
                                finished.insert(ev.slot);
                            }
                            EventKind::Token(_) | EventKind::New => {}
                        }
                    }
                }
                Err(Polling::NoEvents) => std::hint::spin_loop(),
                Err(Polling::Shutdown) => break,
            }
        }
        drop(ingress);
        core1.join().expect("core1 join");
        core2.join().expect("core2 join");

        for slot in 0..CAP as usize {
            let target = max_toks[slot].min(reply_len) as usize;
            assert_eq!(
                outputs[slot], &CANNED_REPLY[..target],
                "slot {slot} streamed exactly its truncated reply under batching"
            );
        }
        assert!(
            retire_mid_batch,
            "a sequence kept streaming after another finished (staggered retirement)"
        );
    }

    /// Drive K requests through a pipeline whose batch cap `< K` and assert the two
    /// admission limits hold: peak concurrency never exceeds `cap` (excess requests
    /// wait in `pending`), and every request still finishes with its full reply (the
    /// backlog drains as seqs retire). `limit_by_seqs` toggles whether the cap comes
    /// from `max_batch_seqs` or from the shared-KV reservation (`n_ctx`).
    fn assert_batch_cap(limit_by_seqs: bool) {
        const CAP: u32 = 4;
        const K: usize = 4;
        const EXPECT: usize = 2;
        // Reservation per mock seq = prompt_len + max_tokens. Prompt "hello" = 5,
        // max_tokens 200 → 205; n_ctx = 2*205 admits exactly 2 at a time.
        let cfg = if limit_by_seqs {
            test_cfg_with(CAP, 1024, EXPECT as u32, 100_000, 1_000_000)
        } else {
            test_cfg_with(CAP, 1024, 8, 100_000, 2 * (5 + 200))
        };
        let slab = Arc::new(Slab::new(CAP as usize, 128, 128, 256));
        let Pipeline { mut ingress, mut egress, core1, core2 } = spawn(&cfg, Arc::clone(&slab));

        for slot in 0..CAP {
            // SAFETY: each slot came off no ring yet; test owns it.
            unsafe {
                let s = slab.slot_mut(slot);
                s.reset();
                s.prompt.push_str("hello");
                s.max_tokens = 200;
            }
            ingress.publish(|e| *e = RingEvent { slot, kind: EventKind::New });
        }

        let mut outputs: Vec<Vec<u8>> = vec![Vec::new(); K];
        let mut cursor = vec![0usize; CAP as usize];
        let mut open: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut max_open = 0usize;
        let mut finished = 0usize;
        while finished < K {
            match egress.poll() {
                Ok(mut guard) => {
                    for ev in &mut guard {
                        match ev.kind {
                            EventKind::Piece(delta) => {
                                open.insert(ev.slot);
                                max_open = max_open.max(open.len());
                                let c = cursor[ev.slot as usize];
                                let piece = unsafe { slab.read_committed(ev.slot, c, delta as usize) }
                                    .expect("committed range in bounds");
                                outputs[ev.slot as usize].extend_from_slice(piece);
                                cursor[ev.slot as usize] = c + delta as usize;
                            }
                            EventKind::Finish(_) => {
                                open.remove(&ev.slot);
                                finished += 1;
                            }
                            EventKind::Token(_) | EventKind::New => {}
                        }
                    }
                }
                Err(Polling::NoEvents) => std::hint::spin_loop(),
                Err(Polling::Shutdown) => break,
            }
        }
        drop(ingress);
        core1.join().expect("core1 join");
        core2.join().expect("core2 join");

        assert_eq!(finished, K, "every request finished (backlog drained past the cap)");
        assert_eq!(
            max_open, EXPECT,
            "peak concurrency should be capped at {EXPECT} (limit_by_seqs={limit_by_seqs})"
        );
        for out in &outputs {
            assert_eq!(out, CANNED_REPLY, "each request streamed its full reply despite staging");
        }
    }

    /// `max_batch_seqs` caps how many sequences decode together; excess wait and drain.
    #[test]
    fn max_batch_seqs_caps_concurrency() {
        assert_batch_cap(true);
    }

    /// The shared-KV reservation caps concurrency: with `max_batch_seqs` high but a
    /// small `n_ctx`, admission defers once reserved cells would exceed `n_ctx`.
    #[test]
    fn kv_reservation_caps_concurrency() {
        assert_batch_cap(false);
    }

    /// `max_tokens` truncation landing *inside* a multi-byte code point: the UTF-8
    /// gate holds the incomplete lead byte, and only `flush_tail` (at Finish) can
    /// emit it. Asserts the reassembled output is byte-exact `CANNED_REPLY[..N]` —
    /// i.e. the held byte was flushed, not dropped. Pick N so byte N-1 is a lone
    /// UTF-8 lead byte (the first byte of a 2+ byte char).
    #[test]
    fn truncation_mid_codepoint_flushes_the_held_lead_byte() {
        // Find an N where CANNED_REPLY[N-1] is a lead byte (0b11xxxxxx) and
        // CANNED_REPLY[N] is a continuation (0b10xxxxxx) — a split code point.
        let n = (1..CANNED_REPLY.len())
            .find(|&i| CANNED_REPLY[i - 1] & 0xC0 == 0xC0 && CANNED_REPLY[i] & 0xC0 == 0x80)
            .expect("canned reply has a multi-byte char to split");

        const CAP: u32 = 2;
        let cfg = test_cfg(CAP, 1024);
        let slab = Arc::new(Slab::new(CAP as usize, 128, 128, 256));
        let Pipeline { mut ingress, mut egress, core1, core2 } = spawn(&cfg, Arc::clone(&slab));

        // SAFETY: slot 0 came off no ring yet; test owns it.
        unsafe {
            let s = slab.slot_mut(0);
            s.reset();
            s.prompt.push_str("hi");
            s.max_tokens = n as u32; // truncate exactly at the split
        }
        ingress.publish(|e| *e = RingEvent { slot: 0, kind: EventKind::New });

        let mut out: Vec<u8> = Vec::new();
        let mut cursor = 0usize;
        let mut done = false;
        while !done {
            match egress.poll() {
                Ok(mut guard) => {
                    for ev in &mut guard {
                        match ev.kind {
                            EventKind::Piece(delta) => {
                                let piece = unsafe { slab.read_committed(0, cursor, delta as usize) }
                                    .expect("committed range in bounds");
                                out.extend_from_slice(piece);
                                cursor += delta as usize;
                            }
                            EventKind::Finish(_) => done = true,
                            _ => {}
                        }
                    }
                }
                Err(Polling::NoEvents) => std::hint::spin_loop(),
                Err(Polling::Shutdown) => break,
            }
        }
        drop(ingress);
        core1.join().expect("core1 join");
        core2.join().expect("core2 join");

        // The held lead byte at N-1 must be present → exactly N bytes, matching the prefix.
        assert_eq!(out, &CANNED_REPLY[..n], "flush_tail emitted the held partial byte");
    }

    /// Output-buffer overflow must finish as `Error`, not `Eos`/`MaxTokens`: a
    /// clipped reply is a failure, and the slot must still recycle. Uses an
    /// `out_cap` far smaller than the reply so detok overflows. Also exercises the
    /// backend-failure → Finish(Error) → slot-recycle path end to end.
    #[test]
    fn output_overflow_finishes_as_error_and_recycles() {
        use crate::rings::FinishReason;
        const CAP: u32 = 2;
        let cfg = test_cfg(CAP, 1024);
        // out_cap = 4 bytes, reply is ~37 → overflow part-way.
        let slab = Arc::new(Slab::new(CAP as usize, 128, 128, 4));
        let Pipeline { mut ingress, mut egress, core1, core2 } = spawn(&cfg, Arc::clone(&slab));

        // SAFETY: slot 0 unused; test owns it.
        unsafe {
            let s = slab.slot_mut(0);
            s.reset();
            s.prompt.push_str("hi");
            s.max_tokens = 200; // full reply → overflows the 4-byte out_bytes
        }
        ingress.publish(|e| *e = RingEvent { slot: 0, kind: EventKind::New });

        let mut finish: Option<FinishReason> = None;
        let mut freed = false;
        while finish.is_none() {
            match egress.poll() {
                Ok(mut guard) => {
                    for ev in &mut guard {
                        if let EventKind::Finish(r) = ev.kind {
                            finish = Some(r);
                            freed = ev.slot == 0;
                        }
                    }
                }
                Err(Polling::NoEvents) => std::hint::spin_loop(),
                Err(Polling::Shutdown) => break,
            }
        }
        drop(ingress);
        core1.join().expect("core1 join");
        core2.join().expect("core2 join");

        assert_eq!(finish, Some(FinishReason::Error), "overflow → Error, not success");
        assert!(freed, "slot recycled after an errored stream");
    }
}
