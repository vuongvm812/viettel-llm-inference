//! Mock backend (default build). Deterministic byte-codec, no GPU/model.
//!
//! Tokenize = one token per prompt byte; detokenize = token id back to that byte.
//! Decode = replay [`CANNED_REPLY`] as byte-tokens. The canned reply deliberately
//! carries multi-byte UTF-8 (`café ☕`) so the pipeline's partial-code-point
//! framing is exercised without a real vocab.

use super::BackendError;
use crate::config::Config;
use crate::rings::{EventKind, FinishReason, RingEvent};
use crate::slab::Slab;
use disruptor::Producer;

/// The canned reply every mock request streams, as raw UTF-8 byte-tokens. Mixes
/// ASCII with 2-byte (`é`) and 3-byte (`☕`) code points on purpose.
pub const CANNED_REPLY: &[u8] = "Hello from the mock backend. café ☕\n".as_bytes();

/// Build the mock backends. No model to load, so this is infallible and ignores
/// config (parity with the real `load`, which reads `cfg.model`).
pub fn load(_cfg: &Config) -> (TextBackend, DecoderInit) {
    (TextBackend, DecoderInit)
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

/// Startup handles for the mock decoder — none needed.
pub struct DecoderInit;

/// Core 2 decoder (mock): replays the canned reply for one sequence.
pub struct Decoder;

impl Decoder {
    pub fn new(_init: DecoderInit) -> Self {
        Decoder
    }

    /// Decode one sequence to completion: emit each canned byte-token on R3,
    /// then `Finish`. Truncates at `max_tokens` (→ `MaxTokens`) else runs the
    /// reply out (→ `Eos`), matching the P1 semantics.
    pub fn run_sequence<P: Producer<RingEvent>>(&mut self, slot: u32, slab: &Slab, r3: &mut P) {
        // SAFETY: slot arrived via R2 → Core 2 owns it now.
        let max_tokens = unsafe { slab.slot_mut(slot) }.max_tokens;
        let reply_len = CANNED_REPLY.len() as u32;
        let target = max_tokens.min(reply_len);
        if target == 0 {
            r3.publish(|e| *e = RingEvent { slot, kind: EventKind::Finish(FinishReason::MaxTokens) });
            return;
        }
        for &byte in &CANNED_REPLY[..target as usize] {
            r3.publish(|e| *e = RingEvent { slot, kind: EventKind::Token(byte as u32) });
        }
        let reason = if max_tokens < reply_len {
            FinishReason::MaxTokens
        } else {
            FinishReason::Eos
        };
        r3.publish(|e| *e = RingEvent { slot, kind: EventKind::Finish(reason) });
    }
}
