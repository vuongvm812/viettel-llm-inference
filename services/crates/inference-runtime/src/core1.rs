//! Core 1 — text processing, mock backend (P1).
//!
//! Sits on both the ingress and egress sides. Tokenize (R1→R2): mock = one token
//! per prompt byte. Detokenize (R3→R4): mock = `id as u8` appended to `out_bytes`,
//! the exact inverse of Core 2's byte-tokens. Real llama.cpp vocab is P2. See
//! `design/text-processing/`.

use crate::affinity;
use crate::rings::{EventKind, RingEvent};
use crate::slab::Slab;
use disruptor::{EventPoller, MultiProducerBarrier, Polling, Producer, SingleProducerBarrier};
use std::sync::Arc;

/// Run the Core 1 loop until both its input rings (R1, R3) are shut down.
pub fn run(
    core: usize,
    slab: Arc<Slab>,
    mut r1: EventPoller<RingEvent, MultiProducerBarrier>,
    r2: impl Producer<RingEvent>,
    mut r3: EventPoller<RingEvent, SingleProducerBarrier>,
    mut r4: impl Producer<RingEvent>,
) {
    affinity::pin(core); // no-op on macOS
    // Held in an Option so we can drop the R2 producer the moment R1 shuts down.
    // That drop is what lets Core 2 (and then this loop via R3) shut down — Core 1
    // must not keep R2's producer alive while waiting on R3, or the two deadlock.
    let mut r2 = Some(r2);

    loop {
        // Ingress: tokenize R1 → R2.
        match r1.poll() {
            Ok(mut guard) => {
                for ev in &mut guard {
                    // Scope the &mut so the slab borrow ends BEFORE the publish —
                    // after publish, Core 2 owns this slot. Keeping `s` live across
                    // the publish would be one edit away from a cross-thread &mut alias.
                    // SAFETY: slot arrived via R1 → Core 1 owns the tokenize stage.
                    {
                        let s = unsafe { slab.slot_mut(ev.slot) };
                        s.tokens.clear();
                        s.tokens.extend(s.prompt.bytes().map(|b| b as i32));
                    }
                    if let Some(p) = r2.as_mut() {
                        p.publish(|e| {
                            *e = RingEvent {
                                slot: ev.slot,
                                kind: EventKind::New,
                            }
                        });
                    }
                }
            }
            Err(Polling::NoEvents) => {}
            Err(Polling::Shutdown) => r2 = None, // cascade shutdown downstream
        }

        // Egress: detokenize R3 → R4.
        match r3.poll() {
            Ok(mut guard) => {
                for ev in &mut guard {
                    match ev.kind {
                        EventKind::Token(id) => {
                            // Mock detokenize: token id → one output byte. The piece
                            // rides the R4 event so Core 0 never reads a buffer Core 1
                            // is still appending to (that would race). Real vocab pieces
                            // are multi-byte and need a slab-based handoff — see P2.
                            let piece = id as u8;
                            r4.publish(|e| {
                                *e = RingEvent {
                                    slot: ev.slot,
                                    kind: EventKind::Token(piece as u32),
                                }
                            });
                        }
                        EventKind::Finish(_) => {
                            r4.publish(|e| *e = *ev);
                        }
                        EventKind::New => {} // never emitted on R3
                    }
                }
            }
            // R3 shuts down only after Core 2 finished every seq and dropped its
            // producer — by then all output has flowed to R4, so we can exit.
            Err(Polling::NoEvents) => {}
            Err(Polling::Shutdown) => break,
        }

        std::hint::spin_loop();
    }
}
