//! Mock backend (default build). Deterministic byte-codec, no GPU/model.
//!
//! Tokenize = one token per prompt byte; detokenize = token id back to that byte.
//! Decode = replay [`CANNED_REPLY`] as byte-tokens. The canned reply deliberately
//! carries multi-byte UTF-8 (`café ☕`) so the pipeline's partial-code-point
//! framing is exercised without a real vocab.

use super::{Admit, BackendError};
use crate::config::Config;
use crate::rings::{EventKind, FinishReason, RingEvent};
use crate::slab::Slab;
use disruptor::Producer;

/// The canned reply every mock request streams, as raw UTF-8 byte-tokens. Mixes
/// ASCII with 2-byte (`é`) and 3-byte (`☕`) code points on purpose.
pub const CANNED_REPLY: &[u8] = "Hello from the mock backend. café ☕\n".as_bytes();

/// Build the mock backends. No model to load, so this is infallible. Reads the
/// continuous-batching cap + KV budget from config (parity with the real `load`).
pub fn load(cfg: &Config) -> (TextBackend, DecoderInit) {
    (
        TextBackend,
        DecoderInit {
            max_batch_seqs: cfg.runtime.effective_max_batch_seqs() as usize,
            // The mock has no real KV cache, but it mirrors the same admission
            // accounting against `n_ctx` so the reservation/defer path is testable
            // without a GPU.
            n_ctx: cfg.model.n_ctx as usize,
        },
    )
}

/// Core 1 text codec (mock): byte ↔ token id.
#[derive(Clone, Copy)]
pub struct TextBackend;

impl TextBackend {
    /// Tokenize: one token id per prompt byte, appended into `out`. Never fails
    /// (Result mirrors the real vocab codec's fallible signature).
    pub fn tokenize(&self, prompt: &str, out: &mut Vec<i32>) -> Result<(), BackendError> {
        out.extend(prompt.bytes().map(|b| b as i32));
        Ok(())
    }

    /// Detokenize: token id → its single byte, written into `piece`.
    pub fn token_bytes(&self, id: u32, piece: &mut Vec<u8>) -> Result<(), BackendError> {
        piece.clear();
        piece.push(id as u8);
        Ok(())
    }
}

/// Startup handles for the mock decoder — the batching cap and KV budget.
pub struct DecoderInit {
    pub max_batch_seqs: usize,
    pub n_ctx: usize,
}

/// One sequence in the mock running set: walks [`CANNED_REPLY`] one byte per step.
struct MockSeq {
    slot: u32,
    /// Next index into `CANNED_REPLY[..target]` to emit.
    cursor: u32,
    /// Emit up to this many bytes = `max_tokens.min(reply_len)`.
    target: u32,
    /// Original request cap, to decide the finish reason (MaxTokens vs Eos).
    max_tokens: u32,
    /// KV cells this seq reserved (`n_prompt + max_tokens`), released on retire.
    reserve: usize,
}

/// Core 2 decoder (mock): continuous batching over a running set. Each `step`
/// advances every active sequence by one canned byte-token, so many sequences stream
/// interleaved on R3 — exercising the batched fast loop without a GPU. Per-request
/// output is byte-identical to P2's single-shot replay, just interleaved.
pub struct Decoder {
    active: Vec<MockSeq>,
    max_batch_seqs: usize,
    /// Total KV cells shared across the running set (mirrors the real backend so the
    /// admission/reservation logic is exercised in the sandbox).
    n_ctx: usize,
    /// Sum of `reserve` over the running set.
    reserved: usize,
}

impl Decoder {
    pub fn new(init: DecoderInit) -> Self {
        Decoder {
            active: Vec::with_capacity(init.max_batch_seqs),
            max_batch_seqs: init.max_batch_seqs,
            n_ctx: init.n_ctx,
            reserved: 0,
        }
    }

    /// Room in the running set for another sequence.
    pub fn has_capacity(&self) -> bool {
        self.active.len() < self.max_batch_seqs
    }

    /// No active sequences → nothing to step.
    pub fn is_idle(&self) -> bool {
        self.active.is_empty()
    }

    /// Admit a slot into the running set, honouring the shared KV budget exactly like
    /// the real backend: reserve `n_prompt + max_tokens` cells up front (never start a
    /// seq that can't finish), defer while the budget is full, reject a request that
    /// can't fit `n_ctx` even alone. A zero-length target finishes immediately.
    pub fn admit<P: Producer<RingEvent>>(&mut self, slot: u32, slab: &Slab, r3: &mut P) -> Admit {
        // SAFETY: slot arrived via R2 → Core 2 owns it now.
        let s = unsafe { slab.slot_mut(slot) };
        let max_tokens = s.max_tokens;
        let prompt_len = s.tokens.len();
        let target = max_tokens.min(CANNED_REPLY.len() as u32);
        if target == 0 {
            r3.publish(|e| *e = RingEvent { slot, kind: EventKind::Finish(FinishReason::MaxTokens) });
            return Admit::Admitted(prompt_len);
        }
        // saturating_add mirrors the real backend's overflow-safe reservation math.
        let needed = prompt_len.saturating_add(max_tokens as usize);
        if needed > self.n_ctx {
            // Can't fit even alone → permanent reject (parity with ContextOverflow).
            r3.publish(|e| *e = RingEvent { slot, kind: EventKind::Finish(FinishReason::Error) });
            return Admit::Rejected;
        }
        if self.reserved + needed > self.n_ctx {
            // Fits alone but not alongside the current running set → wait for a retire.
            return Admit::Deferred;
        }
        self.reserved += needed;
        self.active.push(MockSeq { slot, cursor: 0, target, max_tokens, reserve: needed });
        Admit::Admitted(prompt_len)
    }

    /// One decode step over all active sequences: emit each one's next canned
    /// byte-token, then retire any that reached its target (with the same
    /// `MaxTokens`/`Eos` reason as P2), releasing its KV reservation.
    pub fn step<P: Producer<RingEvent>>(&mut self, _slab: &Slab, r3: &mut P) {
        let reply_len = CANNED_REPLY.len() as u32;
        let reserved = &mut self.reserved;
        self.active.retain_mut(|seq| {
            let byte = CANNED_REPLY[seq.cursor as usize];
            r3.publish(|e| *e = RingEvent { slot: seq.slot, kind: EventKind::Token(byte as u32) });
            seq.cursor += 1;
            if seq.cursor < seq.target {
                return true; // keep decoding
            }
            let reason = if seq.max_tokens < reply_len {
                FinishReason::MaxTokens
            } else {
                FinishReason::Eos
            };
            r3.publish(|e| *e = RingEvent { slot: seq.slot, kind: EventKind::Finish(reason) });
            *reserved -= seq.reserve;
            false // retire
        });
    }
}
