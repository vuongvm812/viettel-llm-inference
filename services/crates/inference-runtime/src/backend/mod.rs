//! Model backend seam — the one place mock and real llama.cpp diverge (P2).
//!
//! Two concrete, cfg-selected types (no `dyn`, static dispatch only):
//! - [`TextBackend`] — Core 1's tokenize / detokenize.
//! - [`Decoder`] — Core 2's prefill + greedy decode of one sequence.
//!
//! Default build uses the byte-codec mock so the pipeline compiles and tests run
//! without a GPU or GGUF model. `--features llama` swaps in real inference over
//! `libllama` (Metal on macOS, CUDA on Linux). Everything else in the pipeline —
//! rings, slab, UTF-8 framing — is backend-independent.

mod prefix_trie;
pub use prefix_trie::{NodeId, PrefixTrie};

/// Max materialized (KV-owning) prefix nodes in the radix cache (P7) — bounds the reserved
/// llama seq ids beyond the request pool. Small: realistic serving has a handful of distinct
/// system prompts. // ponytail: fixed cap + LRU evict; raise if distinct-prefix count grows.
#[cfg_attr(not(feature = "llama"), allow(dead_code))]
pub const MAX_PREFIX_SEQS: usize = 8;

/// Cap on total radix-trie nodes (structural + materialized). Bounds the arena under a stream
/// of distinct prompts; over it, new sharing simply isn't discovered (full-prefill fallback).
#[cfg_attr(not(feature = "llama"), allow(dead_code))]
pub const MAX_TRIE_NODES: usize = 4096;

#[cfg(not(feature = "llama"))]
mod mock;
#[cfg(not(feature = "llama"))]
pub use mock::*;

#[cfg(feature = "llama")]
mod llama;
#[cfg(feature = "llama")]
pub use llama::*;

/// Outcome of [`Decoder::admit`] — the admission decision for one pending slot.
/// Backend-independent so Core 2's scheduler loop can match on it without knowing
/// which backend is compiled in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admit {
    /// Admitted into the running set; carries the *effective* prefill-token count —
    /// the full prompt normally, but only the unshared suffix when a shared prefix
    /// was reused (P4). Core 2 charges it against the per-iteration prefill-token
    /// budget, so shared-prefix requests don't serialize behind the HOL guard.
    Admitted(usize),
    /// KV budget is full *right now* — leave the slot in `pending` and retry after a
    /// decode step frees cells by retiring a sequence. Only returned when the running
    /// set is non-empty, so an empty pipeline always makes progress.
    Deferred,
    /// Cannot ever be admitted (its prompt + generation exceeds `n_ctx` alone). The
    /// backend already published `Finish(Error)` on R3; Core 2 drops it from `pending`.
    Rejected,
}

/// Concrete backend error (no `Box<dyn Error>` — keeps the "static polymorphism"
/// discipline and lets callers match). Feature-independent: the mock never
/// constructs one; the real backend maps `llama-cpp-2` errors into it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(not(feature = "llama"), allow(dead_code))] // variants built by the real backend
pub enum BackendError {
    /// `str_to_token` failed — prompt could not be tokenized.
    Tokenize(String),
    /// `token_to_bytes` failed — a generated token could not be detokenized.
    Detokenize(String),
    /// `decode()` / batch build failed on the GPU forward pass.
    Decode(String),
    /// Prompt (+ requested generation) exceeds the model's context window.
    ContextOverflow { needed: usize, n_ctx: usize },
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::Tokenize(e) => write!(f, "tokenize: {e}"),
            BackendError::Detokenize(e) => write!(f, "detokenize: {e}"),
            BackendError::Decode(e) => write!(f, "decode: {e}"),
            BackendError::ContextOverflow { needed, n_ctx } => {
                write!(f, "context overflow: need {needed} tokens, n_ctx={n_ctx}")
            }
        }
    }
}

impl std::error::Error for BackendError {}

/// Length of the longest prefix of `bytes` that is complete UTF-8 — i.e. does not
/// end in the middle of a multi-byte code point. Used by Core 1 to stream only
/// whole characters (a detok piece can split a code point across tokens).
pub fn complete_utf8_len(bytes: &[u8]) -> usize {
    match std::str::from_utf8(bytes) {
        Ok(_) => bytes.len(),
        Err(e) => e.valid_up_to(),
    }
}

/// Admission verdict for one request (backend-independent so Core 2 matches on it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "llama"), allow(dead_code))]
pub enum Verdict {
    /// Admit into the running set — apply [`AdmissionPlan`]'s side effects.
    Admit,
    /// KV full *now*; leave queued and retry after a retire (running set non-empty).
    Defer,
    /// Can't ever fit `n_ctx` alone — the backend publishes `Finish(Error)`, drops it.
    Reject,
}

/// The pure admission decision for one request — the KV-cell accounting, computed once and
/// consumed by *both* backends so the mock stays an exact oracle for the (target-only) real
/// backend. Each backend owns only its side effects (real KV copy/decode vs. the mock
/// cursor). P7 generalizes the inputs from a fixed-K single prefix to the radix trie's
/// variable, per-request `reused` / `establish_new_cells` (see [`admission_plan`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "llama"), allow(dead_code))]
pub struct AdmissionPlan {
    pub verdict: Verdict,
    /// Prefix cells to copy from the resident prefix trie (0 = full prefill). Pre-capped.
    pub reused: usize,
    /// This request caches a new shared-prefix (radix fork): it prefills as usual, then
    /// donates the fork's cells, which stay reserved after it retires.
    pub establishing: bool,
    /// Evict resident prefixes before admitting (idle + wouldn't otherwise fit).
    pub evict_prefix: bool,
    /// Cells to add to the running `reserved` total on admit.
    pub inc: usize,
    /// Cells to release when this seq retires — an establisher keeps its fork's cells
    /// reserved (they survive its own `seq_rm`); everyone else releases their whole reservation.
    pub reserve: usize,
    /// Tokens actually decoded — the effective prefill charged to the token budget.
    pub effective: usize,
}

/// Decide how to admit a request of `n_prompt` tokens given the decoder's current KV state.
/// Pure and backend-independent; see [`AdmissionPlan`]. The caller runs the [`PrefixTrie`]
/// first and passes:
/// - `reused`: cells the request copies from a resident ready prefix (already capped to
///   `<= n_prompt - 1` so it still decodes ≥ 1 token for the first-sample logits; `0` = full prefill).
/// - `establish_new_cells`: `Some(new)` when this request caches a radix fork, where `new`
///   is the fork's *incremental* cells over its resident ancestor (they stay reserved after
///   retire); `None` for a plain sharer/fallback.
///
/// `reserved` is the running KV reservation, `active_empty` whether the running set is idle
/// (gates prefix eviction). Callers handle finish-now (`max_tokens == 0`) before calling.
#[cfg_attr(not(feature = "llama"), allow(dead_code))] // used by the real decoder + mock + tests
pub fn admission_plan(
    reused: usize,
    establish_new_cells: Option<usize>,
    reserved: usize,
    n_ctx: usize,
    active_empty: bool,
    n_prompt: usize,
    max_tokens: u32,
) -> AdmissionPlan {
    let gen = max_tokens as usize;
    // Reject uses the full footprint: sharing lowers how many run together, not whether
    // one fits `n_ctx` alone.
    let footprint = n_prompt.saturating_add(gen);
    if footprint > n_ctx {
        return AdmissionPlan {
            verdict: Verdict::Reject,
            reused: 0,
            establishing: false,
            evict_prefix: false,
            inc: 0,
            reserve: 0,
            effective: n_prompt,
        };
    }
    let decoded = n_prompt - reused; // tokens this request actually prefills
    let inc = decoded.saturating_add(gen);
    // An establisher donates `new_cells` to the fork (they survive its own retire); a
    // sharer/fallback releases its whole reservation on retire.
    let reserve = match establish_new_cells {
        Some(new_cells) => inc.saturating_sub(new_cells),
        None => inc,
    };
    let mut plan = AdmissionPlan {
        verdict: Verdict::Admit,
        reused,
        establishing: establish_new_cells.is_some(),
        evict_prefix: false,
        inc,
        reserve,
        effective: decoded,
    };
    if reserved.saturating_add(inc) > n_ctx {
        if active_empty {
            // Idle → no retire will free cells; the only reserved cells are resident
            // prefixes this request doesn't fit alongside. Evict them and admit as a plain
            // full prefill (it can't share/keep cells we're dropping) — the reject bound
            // already guaranteed the footprint fits, so it admits instead of wedging.
            plan.evict_prefix = true;
            plan.reused = 0;
            plan.establishing = false;
            plan.inc = footprint;
            plan.reserve = footprint;
            plan.effective = n_prompt;
        } else {
            plan.verdict = Verdict::Defer;
        }
    }
    plan
}

/// Per-step prefill-token budget for the unified batch (P7 chunked prefill). Decode
/// tokens are already resident — one per Decoding seq, priority — and prefill fills the
/// rest of the hardware batch (`n_batch`), capped by the configured per-step
/// `max_batch_tokens`. Returns 0 iff decode alone fills `n_batch`; `>= 1` whenever
/// `n_batch > n_decode`, which guarantees a lone long prompt always makes progress and
/// the loop can't wedge on an all-Prefilling running set.
///
/// Liveness: `n_decode <= active.len() <= max_batch_seqs <= n_batch` (llama's `load`
/// sets `n_batch = n_ctx.min(2048).max(max_batch_seqs)`), so any Prefilling seq present
/// implies `n_decode < active.len() <= n_batch` → budget `>= 1`. The mock has no
/// hardware batch, so it passes `n_batch = usize::MAX` and the budget is purely
/// `max_batch_tokens` (that dimension is exercised here, not in the mock).
#[cfg_attr(not(feature = "llama"), allow(dead_code))] // used by the real decoder + mock + tests
pub fn step_prefill_budget(n_decode: usize, n_batch: usize, max_batch_tokens: usize) -> usize {
    n_batch.saturating_sub(n_decode).min(max_batch_tokens)
}

/// Index of the maximum logit (temp=0 greedy pick). First-max wins on ties, which
/// matches llama.cpp's greedy sampler — the determinism the P2 exit criterion
/// (byte-match vs `llama-cli`) depends on. Lives here (not in the feature-gated
/// backend) so it stays unit-tested in the default build. Returns 0 on an empty
/// slice — callers must only pass logits from a position with logits enabled.
#[cfg_attr(not(feature = "llama"), allow(dead_code))] // used by the real decoder + tests
pub fn argmax(logits: &[f32]) -> i32 {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best as i32
}

#[cfg(test)]
mod tests {
    use super::{admission_plan, argmax, step_prefill_budget, Verdict};

    #[test]
    fn step_prefill_budget_prioritizes_decode_and_caps() {
        // Idle (no decode) → the full configured per-step prefill budget.
        assert_eq!(step_prefill_budget(0, 512, 8), 8, "budget-bound when n_batch is generous");
        // Decode tokens are resident first; prefill gets what's left of n_batch...
        assert_eq!(step_prefill_budget(4, 6, 100), 2, "n_batch-bound after decode fills 4/6");
        // ...and 0 once decode alone fills the hardware batch.
        assert_eq!(step_prefill_budget(6, 6, 100), 0, "decode fills the batch → no prefill room");
        // Progress guarantee: whenever n_batch > n_decode, a lone long prompt advances.
        assert!(step_prefill_budget(2, 512, 100) >= 1, "a Prefilling seq always gets >= 1 token");
    }

    const K: usize = 4;

    #[test]
    fn admission_plan_fallback_full_prefill() {
        // No sharing (reused 0, not establishing) → full prompt, whole reservation released.
        let p = admission_plan(0, None, 0, 100, true, 5, 3);
        assert_eq!(p.verdict, Verdict::Admit);
        assert!(!p.establishing && p.reused == 0 && !p.evict_prefix);
        assert_eq!((p.inc, p.reserve, p.effective), (8, 8, 5)); // footprint 5+3
    }

    #[test]
    fn admission_plan_establish_reserves_fork_once() {
        // Establishing a K-cell fork (reused 0): full prefill, and it leaves exactly K
        // cells reserved after it retires (inc - reserve == K = the donated fork).
        let p = admission_plan(0, Some(K), 0, 100, true, 6, 2);
        assert!(p.establishing && p.reused == 0);
        assert_eq!(p.effective, 6, "establisher decodes the full prompt");
        assert_eq!(p.inc, 8, "footprint 6+2");
        assert_eq!(p.reserve, 4, "inc - new_cells(K) → leaves K reserved");
        assert_eq!(p.inc - p.reserve, K, "fork charged exactly once");
    }

    #[test]
    fn admission_plan_nested_establish_charges_incremental_cells() {
        // Establishing a deeper fork on top of a resident ancestor: reuse the ancestor's
        // 4 cells, decode the suffix, donate only the *incremental* 2 fork cells.
        let p = admission_plan(4, Some(2), 4, 100, false, 8, 2);
        assert!(p.establishing && p.reused == 4);
        assert_eq!(p.effective, 4, "decodes suffix 8-4");
        assert_eq!(p.inc, 6, "decoded 4 + gen 2");
        assert_eq!(p.inc - p.reserve, 2, "only the 2 incremental fork cells stay reserved");
    }

    #[test]
    fn admission_plan_share_charges_suffix_only() {
        // Resident prefix matches: reuse K, decode only the suffix, release all on retire.
        let p = admission_plan(K, None, K, 100, false, 6, 2);
        assert!(!p.establishing && p.reused == 4);
        assert_eq!(p.effective, 2, "suffix only (6 - 4)");
        assert_eq!((p.inc, p.reserve), (4, 4), "suffix 2 + gen 2");
    }

    #[test]
    fn admission_plan_rejects_over_ctx() {
        assert_eq!(admission_plan(0, None, 0, 10, true, 5, 100).verdict, Verdict::Reject);
    }

    #[test]
    fn admission_plan_defers_when_busy_but_evicts_when_idle() {
        // Resident prefix (reserved=K) + a fallback whose footprint fits alone (8 ≤ 10)
        // but not atop the prefix (4+8 > 10). Busy → defer; idle → evict + admit.
        let busy = admission_plan(0, None, K, 10, false, 5, 3);
        assert_eq!(busy.verdict, Verdict::Defer);
        let idle = admission_plan(0, None, K, 10, true, 5, 3);
        assert_eq!(idle.verdict, Verdict::Admit);
        assert!(idle.evict_prefix && !idle.establishing && idle.reused == 0);
        assert_eq!((idle.inc, idle.effective), (8, 5), "admits as a full-prefill fallback");
    }

    #[test]
    fn argmax_picks_first_max_for_greedy_determinism() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
        assert_eq!(argmax(&[0.5, 0.5]), 0, "ties → first index (matches llama greedy)");
        assert_eq!(argmax(&[-2.0, -0.5, -3.0]), 1, "all negative");
        assert_eq!(argmax(&[]), 0, "empty → 0 (caller must avoid)");
    }
}
