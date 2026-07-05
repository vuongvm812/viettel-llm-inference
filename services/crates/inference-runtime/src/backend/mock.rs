//! Mock backend (default build). Deterministic byte-codec, no GPU/model.
//!
//! Tokenize = one token per prompt byte; detokenize = token id back to that byte.
//! Decode = replay [`CANNED_REPLY`] as byte-tokens. The canned reply deliberately
//! carries multi-byte UTF-8 (`café ☕`) so the pipeline's partial-code-point
//! framing is exercised without a real vocab.

use super::{admission_plan, shared_reuse_len, Admit, BackendError, Verdict};
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

    /// Tokens [`admit`](Self::admit) would actually decode for `slot` — full prompt minus
    /// any reused prefix cells. Core 2 peeks this for the prefill budget; it consumes the
    /// same [`shared_reuse_len`] the plan does, so peek and admit can't disagree.
    pub fn effective_prefill_len(&self, slot: u32, slab: &Slab) -> usize {
        // SAFETY: `slot` sits in Core 2's `pending` → Core 2 solely owns it (read only).
        let tokens = unsafe { slab.slot_tokens(slot) };
        tokens.len() - shared_reuse_len(self.prefix_k, self.prefix.as_deref(), tokens)
    }

    /// Drop the resident shared prefix, returning its K reserved cells. Called only
    /// when the decoder is idle and a fallback request can't fit alongside the prefix
    /// (see [`admission_plan`]) — the prefix would otherwise never be reclaimed (v1 has
    /// no eviction), wedging the fittable request. No-op if none is resident.
    fn evict_prefix(&mut self) {
        if self.prefix.take().is_some() {
            self.reserved = self.reserved.saturating_sub(self.prefix_k);
        }
    }

    /// Admit a slot into the running set. The KV/share accounting is the shared
    /// [`admission_plan`] oracle (identical to the real backend); this only applies the
    /// mock's side effects (establish the prefix, reserve cells, push a cursor). A
    /// zero-length target finishes immediately.
    pub fn admit<P: Producer<RingEvent>>(&mut self, slot: u32, slab: &Slab, r3: &mut P) -> Admit {
        // SAFETY: slot arrived via R2 → Core 2 owns it now.
        let s = unsafe { slab.slot_mut(slot) };
        let max_tokens = s.max_tokens;
        let target = max_tokens.min(CANNED_REPLY.len() as u32);
        if target == 0 {
            r3.publish(|e| *e = RingEvent { slot, kind: EventKind::Finish(FinishReason::MaxTokens) });
            return Admit::Admitted(0); // nothing prefilled (parity with the real backend)
        }
        let plan = admission_plan(
            self.prefix_k,
            self.prefix.as_deref(),
            self.reserved,
            self.n_ctx,
            self.active.is_empty(),
            &s.tokens,
            max_tokens,
        );
        match plan.verdict {
            Verdict::Reject => {
                r3.publish(|e| *e = RingEvent { slot, kind: EventKind::Finish(FinishReason::Error) });
                Admit::Rejected
            }
            Verdict::Defer => Admit::Deferred,
            Verdict::Admit => {
                if plan.evict_prefix {
                    self.evict_prefix();
                }
                if plan.establishing {
                    self.prefix = Some(s.tokens[..self.prefix_k].to_vec());
                }
                self.reserved += plan.inc;
                self.active.push(MockSeq { slot, cursor: 0, target, max_tokens, reserve: plan.reserve });
                Admit::Admitted(plan.effective)
            }
        }
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
