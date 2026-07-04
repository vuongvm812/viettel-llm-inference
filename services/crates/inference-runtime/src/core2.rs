//! Core 2 — the fast loop, mock backend (P1).
//!
//! Scheduler + emitter. No llama.cpp, no KV: admits tokenized requests off R2 and
//! emits a deterministic canned reply, one byte-token per active seq per iteration,
//! on R3 — exercising the R3/R4 half of the topology. Real prefill/decode is P2+.
//! See `design/fast-loop/`.

use crate::affinity;
use crate::rings::{EventKind, FinishReason, RingEvent};
use crate::slab::Slab;
use disruptor::{EventPoller, Polling, Producer, SingleProducerBarrier};
use std::sync::Arc;

/// The canned reply every mock request streams (as raw byte-tokens).
pub const CANNED_REPLY: &[u8] = b"Hello from the mock backend. This is a canned streamed response.\n";

/// Per-sequence decode state (Core 2 local — never touches the slab).
struct ActiveSeq {
    slot: u32,
    emitted: u32,
    /// Total tokens to emit = min(max_tokens, CANNED_REPLY.len()).
    target: u32,
    /// Whether the target was capped by max_tokens (→ MaxTokens) or the reply
    /// ran out (→ Eos).
    capped_by_max: bool,
}

/// Run the Core 2 loop until R2 is shut down and all active seqs have finished.
pub fn run(
    core: usize,
    slab: Arc<Slab>,
    mut r2: EventPoller<RingEvent, SingleProducerBarrier>,
    mut r3: impl Producer<RingEvent>,
) {
    affinity::pin(core); // no-op on macOS
    let mut active: Vec<ActiveSeq> = Vec::new();
    let mut r2_shutdown = false;

    loop {
        // 1. Admit newly tokenized requests (non-blocking drain of R2).
        match r2.poll() {
            Ok(mut guard) => {
                for ev in &mut guard {
                    // SAFETY: this slot reached Core 2 via R2 → we own it now.
                    let max_tokens = unsafe { slab.slot_mut(ev.slot) }.max_tokens;
                    let reply_len = CANNED_REPLY.len() as u32;
                    let target = max_tokens.min(reply_len);
                    if target == 0 {
                        // max_tokens == 0 slips past validation: finish empty, don't decode.
                        r3.publish(|e| {
                            *e = RingEvent {
                                slot: ev.slot,
                                kind: EventKind::Finish(FinishReason::MaxTokens),
                            }
                        });
                        continue;
                    }
                    active.push(ActiveSeq {
                        slot: ev.slot,
                        emitted: 0,
                        target,
                        capped_by_max: max_tokens < reply_len,
                    });
                }
            }
            Err(Polling::NoEvents) => {}
            Err(Polling::Shutdown) => r2_shutdown = true,
        }

        // 2. One "decode" step: emit the next byte-token for every active seq.
        for seq in &mut active {
            let id = CANNED_REPLY[seq.emitted as usize] as u32;
            r3.publish(|e| {
                *e = RingEvent {
                    slot: seq.slot,
                    kind: EventKind::Token(id),
                }
            });
            seq.emitted += 1;

            if seq.emitted >= seq.target {
                let reason = if seq.capped_by_max {
                    FinishReason::MaxTokens
                } else {
                    FinishReason::Eos
                };
                r3.publish(|e| {
                    *e = RingEvent {
                        slot: seq.slot,
                        kind: EventKind::Finish(reason),
                    }
                });
            }
        }
        active.retain(|s| s.emitted < s.target);

        if r2_shutdown && active.is_empty() {
            break;
        }
        std::hint::spin_loop();
    }
}
