//! Mock backend (default build). Deterministic byte-codec, no GPU/model.
//!
//! Tokenize = one token per prompt byte; detokenize = token id back to that byte.
//! Decode = replay [`CANNED_REPLY`] as byte-tokens. The canned reply deliberately
//! carries multi-byte UTF-8 (`café ☕`) so the pipeline's partial-code-point
//! framing is exercised without a real vocab.

use super::{
    admission_plan, step_prefill_budget, Admit, BackendError, NodeId, PrefixTrie, Verdict,
    MAX_PREFIX_SEQS, MAX_TRIE_NODES,
};
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
            // P7 chunked prefill: the per-step prefill-chunk budget (was an admission
            // gate; now bounds prompt tokens prefilled per decode step).
            max_batch_tokens: cfg.runtime.max_batch_tokens as usize,
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
    /// Per-step prefill-chunk budget (P7 chunked prefill).
    pub max_batch_tokens: usize,
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
    /// P7 chunked prefill: prompt tokens still to prefill. `> 0` → **Prefilling** (emits
    /// nothing, consumes a chunk of the per-step budget); `0` → **Decoding** (emits one
    /// canned byte per step). Set to the *effective* prefill count at admit (suffix-only
    /// when a prefix was shared), so a long prompt is spread across steps instead of
    /// prefilled in one shot behind the head-of-line guard.
    prompt_remaining: usize,
    /// This seq establishes a radix-cache fork (P7): the trie node to mark `ready` once its
    /// prefill completes, so deferred matching requests can then share it. `None` otherwise.
    establishes: Option<NodeId>,
}

/// Core 2 decoder (mock): continuous batching over a running set. Each `step`
/// advances every active sequence by one canned byte-token, so many sequences stream
/// interleaved on R3 — exercising the batched fast loop without a GPU. Per-request
/// output is byte-identical to P2's single-shot replay, just interleaved.
pub struct Decoder {
    active: Vec<MockSeq>,
    max_batch_seqs: usize,
    /// Total KV cells shared across the running set (mirrors the real backend so the
    /// admission/reservation logic is exercised in the sandbox). Accumulates each admit's
    /// `inc` and releases each retire's `reserve`; the leftover after establishers retire is
    /// the resident prefix cells (also reflected by `trie.reserved_cells()`).
    n_ctx: usize,
    reserved: usize,
    /// Radix cache of shared prefixes (P7) — variable-length longest match, nested forks,
    /// LRU eviction. Replaces P4's single fixed-K prefix. The mock drives the same trie the
    /// real backend does; only KV side effects differ.
    trie: PrefixTrie,
    /// Monotonic stand-in for a llama prefix seq id (the mock has no real KV; the value only
    /// tags trie nodes and is echoed back by eviction).
    next_seq: i32,
    /// Per-step prefill-chunk budget (P7).
    max_batch_tokens: usize,
}

impl Decoder {
    pub fn new(init: DecoderInit) -> Self {
        Decoder {
            active: Vec::with_capacity(init.max_batch_seqs),
            max_batch_seqs: init.max_batch_seqs,
            n_ctx: init.n_ctx,
            reserved: 0,
            trie: PrefixTrie::new(MAX_PREFIX_SEQS, init.shared_prefix_tokens),
            next_seq: 0,
            max_batch_tokens: init.max_batch_tokens,
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

    fn next_seq(&mut self) -> i32 {
        let s = self.next_seq;
        self.next_seq += 1;
        s
    }

    /// Evict resident prefix cells (idle only) until the request's `footprint` fits `n_ctx`,
    /// subtracting each evicted prefix's cells from `reserved` in lock-step so the counter
    /// stays exact. Called on the [`admission_plan`] `evict_prefix` path.
    fn evict_to_fit(&mut self, footprint: usize) {
        while self.reserved.saturating_add(footprint) > self.n_ctx {
            match self.trie.evict_lru_leaf() {
                Some((_seq, cells)) => self.reserved = self.reserved.saturating_sub(cells),
                None => break, // nothing left to evict
            }
        }
    }

    /// Admit a slot into the running set. The KV/share accounting is the shared
    /// [`admission_plan`] oracle (identical to the real backend); this only applies the
    /// mock's side effects (record the path, materialize a fork, reserve cells, push a
    /// Prefilling seq). A zero-length target finishes immediately.
    ///
    /// P7: the request records its prompt path in the radix trie (so future requests can
    /// fork against it), reuses the deepest ready prefix, optionally caches a newly-discovered
    /// shared fork, and joins as **Prefilling** (`prompt_remaining = plan.effective`) —
    /// [`step`](Self::step) prefills a chunk per iteration. The readiness gate defers a request
    /// matching a prefix whose establisher is still Prefilling (unmaterialized cells).
    pub fn admit<P: Producer<RingEvent>>(&mut self, slot: u32, slab: &Slab, r3: &mut P) -> Admit {
        // SAFETY: slot arrived via R2 → Core 2 owns it now.
        let s = unsafe { slab.slot_mut(slot) };
        let max_tokens = s.max_tokens;
        let n_prompt = s.tokens.len();
        let target = max_tokens.min(CANNED_REPLY.len() as u32);
        // Nothing to generate (or empty prompt) → finish immediately (parity with the real
        // backend's `n_prompt == 0 || max_tokens == 0` guard).
        if n_prompt == 0 || target == 0 {
            r3.publish(|e| *e = RingEvent { slot, kind: EventKind::Finish(FinishReason::MaxTokens) });
            return Admit::Admitted(0);
        }
        // Reject a request that can't fit `n_ctx` even alone BEFORE any trie work: the trie
        // walk + structural insert (a full-prompt edge Vec) are wasted on the fast loop for a
        // request that `admission_plan` would reject anyway, and an uncapped prompt would
        // accrete a multi-MB edge into the arena. Same bound `admission_plan` uses.
        if n_prompt.saturating_add(max_tokens as usize) > self.n_ctx {
            r3.publish(|e| *e = RingEvent { slot, kind: EventKind::Finish(FinishReason::Error) });
            return Admit::Rejected;
        }
        // Record this request's path so future requests can fork against it (discovers
        // variable-length shared prefixes). Cheap structural nodes only, no KV.
        self.trie.insert_structural(&s.tokens, MAX_TRIE_NODES);

        // The full reuse/establish decision (incl. the readiness gate) is the shared
        // `PrefixTrie::plan_reuse` oracle — one trie walk, identical to the real backend.
        let reuse = self.trie.plan_reuse(&s.tokens, n_prompt);
        // Readiness gate: matching a prefix whose establisher is still Prefilling → wait.
        // The establisher always makes prefill progress (per-step budget >= 1), so bounded.
        if reuse.defer_unready {
            return Admit::Deferred;
        }

        let plan = admission_plan(
            reuse.reused,
            reuse.establish_new_cells,
            self.reserved,
            self.n_ctx,
            self.active.is_empty(),
            n_prompt,
            max_tokens,
        );
        match plan.verdict {
            Verdict::Reject => {
                r3.publish(|e| *e = RingEvent { slot, kind: EventKind::Finish(FinishReason::Error) });
                Admit::Rejected
            }
            Verdict::Defer => Admit::Deferred,
            Verdict::Admit => {
                // A real reuse (`reused > 0`) → keep that prefix hot against LRU eviction.
                // Gated on `plan.reused` (not just `match_node`) so the evict-prefix path —
                // which resets `reused` to 0 but leaves `match_node` set — doesn't bump an LRU
                // tick the real backend never bumps (keeps the two eviction orders identical).
                if plan.reused > 0 {
                    if let Some(node) = reuse.match_node {
                        self.trie.touch(node);
                    }
                }
                let mut establishes = None;
                if plan.evict_prefix {
                    // Idle pressure: drop resident prefixes to fit; admits as a full prefill
                    // (plan already reset reused/establishing).
                    self.evict_to_fit(n_prompt.saturating_add(max_tokens as usize));
                } else if plan.establishing {
                    let f = reuse.fork.expect("establishing ⇒ a fork was found");
                    let seq = self.next_seq();
                    let ok = self.trie.materialize(f, seq);
                    debug_assert!(ok, "can_materialize checked in plan_reuse before establish");
                    establishes = Some(f);
                }
                self.reserved += plan.inc;
                self.active.push(MockSeq {
                    slot,
                    cursor: 0,
                    target,
                    max_tokens,
                    reserve: plan.reserve,
                    prompt_remaining: plan.effective,
                    establishes,
                });
                Admit::Admitted(plan.effective)
            }
        }
    }

    /// One unified batch step (P7 chunked prefill): emit one canned byte for each
    /// **Decoding** seq (`prompt_remaining == 0`) and retire finished ones, then spend
    /// the per-step prefill budget advancing **Prefilling** seqs — many prompts prefill
    /// interleaved with decode, so a long prompt no longer blocks the running set. A
    /// Prefilling seq whose prompt completes this step becomes Decoding *next* step
    /// (mirrors the real backend sampling the first token at completion, emitting it on
    /// the following step). Unlike P3, this does not early-return on an all-Prefilling
    /// set: cold-start prompts must still prefill with nothing yet decoding.
    pub fn step<P: Producer<RingEvent>>(&mut self, _slab: &Slab, r3: &mut P) {
        // Decode phase: one token per Decoding seq, in one R3 burst (the throughput
        // path). `n_decode <= active.len() <= max_batch_seqs <= ring_size`, so it fits.
        let n_decode = self.active.iter().filter(|s| s.prompt_remaining == 0).count();
        if n_decode > 0 {
            r3.batch_publish(n_decode, |iter| {
                let decoding = self.active.iter().filter(|s| s.prompt_remaining == 0);
                for (e, seq) in iter.zip(decoding) {
                    let byte = CANNED_REPLY[seq.cursor as usize];
                    *e = RingEvent { slot: seq.slot, kind: EventKind::Token(byte as u32) };
                }
            });
        }
        // Advance + retire Decoding seqs (Prefilling seqs pass through untouched here).
        // Finishes (a minority) stay individual publishes.
        let reply_len = CANNED_REPLY.len() as u32;
        let reserved = &mut self.reserved;
        self.active.retain_mut(|seq| {
            if seq.prompt_remaining > 0 {
                return true; // Prefilling — handled in the prefill phase below
            }
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
            // `saturating_sub` + assert: the accounting is exact (each `reserve` was added at
            // admit), but a saturating floor turns any future drift into a benign under-count
            // rather than a `usize` wrap → permanent-reject wedge.
            debug_assert!(*reserved >= seq.reserve, "reserved underflow on retire");
            *reserved = reserved.saturating_sub(seq.reserve);
            false // retire
        });
        // Prefill phase: share the per-step budget across Prefilling seqs. The mock has no
        // hardware batch, so `n_batch = usize::MAX` and the budget is purely
        // `max_batch_tokens` (the n_batch dimension is unit-tested in `step_prefill_budget`).
        // Mark each completed establisher's fork ready inline (deferred sharers can then use
        // it) — a fork completes at most once per request, so this never allocates.
        let mut budget = step_prefill_budget(n_decode, usize::MAX, self.max_batch_tokens);
        let mut i = 0;
        while i < self.active.len() && budget > 0 {
            if self.active[i].prompt_remaining > 0 {
                let take = self.active[i].prompt_remaining.min(budget);
                self.active[i].prompt_remaining -= take;
                budget -= take;
                if self.active[i].prompt_remaining == 0 {
                    if let Some(node) = self.active[i].establishes {
                        self.trie.set_ready(node); // its fork cells are now materialized
                    }
                }
            }
            i += 1;
        }
        // Cross-check the two sources of truth for prefix cells at a quiescent point: when the
        // running set is idle, `reserved` holds only resident prefix cells, which must equal
        // `trie.reserved_cells()`. Catches any accounting drift in tests (default build).
        debug_assert!(
            !self.active.is_empty() || self.reserved == self.trie.reserved_cells(),
            "reserved ({}) != trie.reserved_cells ({}) when idle",
            self.reserved,
            self.trie.reserved_cells(),
        );
    }
}
