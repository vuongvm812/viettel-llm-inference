//! Mock backend (default build). Deterministic byte-codec, no GPU/model.
//!
//! Tokenize = one token per prompt byte; detokenize = token id back to that byte.
//! Decode = replay [`CANNED_REPLY`] as byte-tokens. The canned reply deliberately
//! carries multi-byte UTF-8 (`café ☕`) so the pipeline's partial-code-point
//! framing is exercised without a real vocab.

use super::{shared_prefix_len, Admit, BackendError};
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
            shared_prefix_tokens: cfg.runtime.shared_prefix_tokens as usize,
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
    /// Shared-prefix window K (P4); `0` disables (P3 behavior).
    pub shared_prefix_tokens: usize,
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
    /// KV cells this seq reserved (`suffix + max_tokens`, where suffix excludes any
    /// shared prefix), released on retire. The shared prefix's cells are reserved
    /// once (in `Decoder::reserved`) and are *not* part of this per-seq amount.
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
    /// Sum of per-seq `reserve` over the running set, plus the shared prefix's `K`
    /// cells (reserved once when the prefix is established, never released).
    reserved: usize,
    /// Shared-prefix window K (P4); `0` disables sharing (P3 behavior).
    prefix_k: usize,
    /// The established shared-prefix tokens (the first `prefix_k`), or `None` until
    /// the first eligible request establishes it. Single entry (v1): a non-matching
    /// prompt falls back to a full-prefill; multi-prefix/radix is P7.
    prefix: Option<Vec<i32>>,
}

impl Decoder {
    pub fn new(init: DecoderInit) -> Self {
        Decoder {
            active: Vec::with_capacity(init.max_batch_seqs),
            max_batch_seqs: init.max_batch_seqs,
            n_ctx: init.n_ctx,
            reserved: 0,
            prefix_k: init.shared_prefix_tokens,
            prefix: None,
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

    /// Tokens [`admit`](Self::admit) would actually prefill for `slot`: the full prompt,
    /// unless it *shares an already-established* prefix, in which case only the unshared
    /// suffix (the K prefix cells are `kv_cache_seq_cp`'d, not re-decoded). Read-only —
    /// establishing the prefix (first eligible request) still reports the full prompt, so
    /// the establisher pays its real cost. Core 2 peeks this for the prefill budget.
    pub fn effective_prefill_len(&self, slot: u32, slab: &Slab) -> usize {
        // SAFETY: `slot` sits in Core 2's `pending` → Core 2 solely owns it.
        let tokens = unsafe { slab.slot_tokens(slot) };
        let prompt_len = tokens.len();
        let shares = self.prefix_k > 0
            && prompt_len >= self.prefix_k
            && self
                .prefix
                .as_ref()
                .is_some_and(|p| shared_prefix_len(tokens, p, self.prefix_k) == self.prefix_k);
        if shares {
            prompt_len - self.prefix_k
        } else {
            prompt_len
        }
    }

    /// Drop the resident shared prefix, returning its K reserved cells. Called only
    /// when the decoder is idle and a fallback request can't fit alongside the prefix
    /// (see [`admit`](Self::admit)) — the prefix would otherwise never be reclaimed
    /// (v1 has no eviction), wedging the fittable request. No-op if none is resident.
    fn evict_prefix(&mut self) {
        if self.prefix.take().is_some() {
            self.reserved = self.reserved.saturating_sub(self.prefix_k);
        }
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
            return Admit::Admitted(0); // nothing prefilled (parity with the real backend)
        }
        // Reject uses the request's *total* footprint (`prompt_len + max_tokens` =
        // prefix + suffix + generation): sharing a prefix lowers how many requests run
        // together, not whether one fits `n_ctx` alone. saturating_add mirrors the real
        // backend's overflow-safe reservation math.
        let footprint = prompt_len.saturating_add(max_tokens as usize);
        if footprint > self.n_ctx {
            // Can't fit even alone → permanent reject (parity with ContextOverflow).
            r3.publish(|e| *e = RingEvent { slot, kind: EventKind::Finish(FinishReason::Error) });
            return Admit::Rejected;
        }
        // Shared-prefix (P4): if the first K tokens match the cached prefix (or this is
        // the first eligible request, which *establishes* it), only the suffix +
        // generation is charged per request; the K prefix cells are reserved once and
        // never released. `l` is K on a share/establish, else 0 (full-prefill fallback).
        let l = if self.prefix_k > 0 && prompt_len >= self.prefix_k {
            match &self.prefix {
                None => self.prefix_k, // first eligible request establishes the prefix
                Some(p) => shared_prefix_len(&s.tokens, p, self.prefix_k),
            }
        } else {
            0
        };
        let establishing = l > 0 && self.prefix.is_none();
        let suffix = prompt_len - l;
        // Per-seq reservation, released on retire (the shared prefix's K stays).
        let reserve = suffix.saturating_add(max_tokens as usize);
        // Cells this admission adds to the running total. An establisher charges the full
        // footprint (its own prompt + generation; the K prefix cells it donates are a
        // subset), which == the reject bound, so an idle establisher never defers. A
        // sharer charges just its suffix + generation (prefix already counted).
        let inc = if establishing { footprint } else { reserve };
        if self.reserved.saturating_add(inc) > self.n_ctx {
            if self.active.is_empty() {
                // Idle → no retire will free cells; the only reserved cells are a
                // resident prefix this fallback request doesn't fit alongside (an
                // establisher/sharer always fits when idle). Evict it so a request the
                // reject bound already OK'd still admits, instead of wedging forever.
                self.evict_prefix();
            } else {
                // Fits alone but not alongside the running set → wait for a retire.
                return Admit::Deferred;
            }
        }
        if establishing {
            self.prefix = Some(s.tokens[..self.prefix_k].to_vec());
        }
        self.reserved += inc;
        self.active.push(MockSeq { slot, cursor: 0, target, max_tokens, reserve });
        // Effective prefill = suffix only when sharing an established prefix; the full
        // prompt when establishing (we decode the K prefix + suffix) or on fallback.
        let effective_prefill = if establishing || l == 0 { prompt_len } else { suffix };
        Admit::Admitted(effective_prefill)
    }

    /// One decode step over all active sequences: emit each one's next canned
    /// byte-token, then retire any that reached its target (with the same
    /// `MaxTokens`/`Eos` reason as P2), releasing its KV reservation.
    pub fn step<P: Producer<RingEvent>>(&mut self, _slab: &Slab, r3: &mut P) {
        let n = self.active.len();
        if n == 0 {
            return;
        }
        // Emit the whole per-step token burst in one R3 batch (the design's throughput
        // path). `n == active.len() <= max_batch_seqs <= ring_size`, so it always fits.
        r3.batch_publish(n, |iter| {
            for (e, seq) in iter.zip(self.active.iter()) {
                let byte = CANNED_REPLY[seq.cursor as usize];
                *e = RingEvent { slot: seq.slot, kind: EventKind::Token(byte as u32) };
            }
        });
        // Advance + retire. Finishes (a minority) stay individual publishes.
        let reply_len = CANNED_REPLY.len() as u32;
        let reserved = &mut self.reserved;
        self.active.retain_mut(|seq| {
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
