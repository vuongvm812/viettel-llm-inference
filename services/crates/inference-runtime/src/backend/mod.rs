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

/// Shared-prefix length for P4 KV caching: `k` if the first `k` tokens of `tokens`
/// exactly match the cached `prefix`, else `0` (no sharing → full prefill fallback).
///
/// A prompt shorter than `k`, a cached prefix shorter than `k`, or `k == 0` (feature
/// off) all yield `0`. The exact token-slice compare is collision-free — the design
/// doc's "hash first K tokens" is only needed for a multi-entry cache (P7's radix
/// tree); v1 keeps a single cached prefix, where a direct compare is simpler and
/// can't false-match. Lives here (not the feature-gated backend) so it stays
/// unit-tested in the default build and is shared by both backends.
#[cfg_attr(not(feature = "llama"), allow(dead_code))] // used by the real decoder + tests
pub fn shared_prefix_len(tokens: &[i32], prefix: &[i32], k: usize) -> usize {
    if k > 0 && tokens.len() >= k && prefix.len() >= k && tokens[..k] == prefix[..k] {
        k
    } else {
        0
    }
}

/// Cells a request may *reuse* from an already-established shared prefix: the matched
/// length, capped at `n_prompt - 1` so the request still decodes ≥ 1 token in its own
/// sequence (needed for the first-sample logits). Returns `0` when establishing
/// (`prefix` is `None`), on a mismatch, or when too short — those decode the full
/// prompt. Single source of truth for both the admission accounting and Core 2's
/// budget peek, so they can't disagree (e.g. on the exact-`k` edge).
#[cfg_attr(not(feature = "llama"), allow(dead_code))] // used by the real decoder + tests
pub fn shared_reuse_len(prefix_k: usize, prefix: Option<&[i32]>, tokens: &[i32]) -> usize {
    match prefix {
        Some(p) => shared_prefix_len(tokens, p, prefix_k).min(tokens.len().saturating_sub(1)),
        None => 0,
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

/// The pure admission decision for one request — the shared-prefix (P4) share detection
/// plus KV-cell accounting — computed once and consumed by *both* backends. Extracting
/// it keeps the mock an exact oracle for the (sandbox-unbuildable) real backend: they
/// can no longer drift (e.g. the exact-`k` cap that once lived only in the real path).
/// Each backend owns only its side effects (real KV copy/decode vs. the mock cursor).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "llama"), allow(dead_code))]
pub struct AdmissionPlan {
    pub verdict: Verdict,
    /// Prefix cells to copy from the resident prefix (0 = full prefill). Already capped.
    pub reused: usize,
    /// This request establishes the prefix (full prefill, then donate `k` cells).
    pub establishing: bool,
    /// Evict the resident prefix before admitting (idle + wouldn't otherwise fit).
    pub evict_prefix: bool,
    /// Cells to add to the running `reserved` total on admit.
    pub inc: usize,
    /// Cells to release when this seq retires (an establisher leaves its `k` prefix
    /// cells reserved; everyone else releases their whole reservation).
    pub reserve: usize,
    /// Tokens actually decoded — the effective prefill charged to the token budget.
    pub effective: usize,
}

/// Decide how to admit `tokens` (already tokenized, `max_tokens > 0`) given the
/// decoder's current KV state. Pure and backend-independent; see [`AdmissionPlan`].
/// `prefix` is the established shared-prefix tokens (`None` until established),
/// `reserved` the running KV reservation, `active_empty` whether the running set is
/// idle (gates prefix eviction). Callers handle finish-now cases before calling.
#[cfg_attr(not(feature = "llama"), allow(dead_code))] // used by the real decoder + tests
pub fn admission_plan(
    prefix_k: usize,
    prefix: Option<&[i32]>,
    reserved: usize,
    n_ctx: usize,
    active_empty: bool,
    tokens: &[i32],
    max_tokens: u32,
) -> AdmissionPlan {
    let n_prompt = tokens.len();
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
    let eligible = prefix_k > 0 && n_prompt >= prefix_k;
    let establishing = eligible && prefix.is_none();
    let reused = shared_reuse_len(prefix_k, prefix, tokens); // 0 when establishing/fallback
    let decoded = n_prompt - reused; // tokens this request actually prefills
                                     // An establisher physically occupies the full footprint (its `k` prefix cells
                                     // become permanent once donated); a sharer/fallback occupies only what it
                                     // decodes + generates.
    let inc = if establishing { footprint } else { decoded.saturating_add(gen) };
    // Released on retire: an establisher keeps its `k` prefix cells reserved (they
    // survive its own `seq_rm`); everyone else releases the whole reservation.
    let reserve = if establishing { footprint.saturating_sub(prefix_k) } else { decoded.saturating_add(gen) };

    let mut plan = AdmissionPlan {
        verdict: Verdict::Admit,
        reused,
        establishing,
        evict_prefix: false,
        inc,
        reserve,
        effective: decoded,
    };
    if reserved.saturating_add(inc) > n_ctx {
        if active_empty {
            // Idle → no retire will free cells; the only reserved cells are a resident
            // prefix this request doesn't fit alongside. Evict it and admit as a plain
            // fallback (it can't share a prefix we're dropping) — the reject bound
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
    use super::{admission_plan, argmax, shared_prefix_len, shared_reuse_len, Verdict};

    const K: usize = 4;
    const PFX: [i32; 4] = [1, 2, 3, 4];

    #[test]
    fn admission_plan_fallback_full_prefill() {
        // No prefix feature (k=0) → full prompt, whole reservation released on retire.
        let p = admission_plan(0, None, 0, 100, true, &[1, 2, 3, 4, 5], 3);
        assert_eq!(p.verdict, Verdict::Admit);
        assert!(!p.establishing && p.reused == 0 && !p.evict_prefix);
        assert_eq!((p.inc, p.reserve, p.effective), (8, 8, 5)); // footprint 5+3
    }

    #[test]
    fn admission_plan_establish_reserves_prefix_once() {
        // First eligible request: full prefill (effective = whole prompt), and it leaves
        // exactly K cells reserved after it retires (inc - reserve == K).
        let p = admission_plan(K, None, 0, 100, true, &[1, 2, 3, 4, 9, 9], 2);
        assert!(p.establishing && p.reused == 0);
        assert_eq!(p.effective, 6, "establisher decodes the full prompt");
        assert_eq!(p.inc, 8, "footprint 6+2");
        assert_eq!(p.reserve, 4, "footprint - K → leaves K reserved");
        assert_eq!(p.inc - p.reserve, K, "prefix charged exactly once");
    }

    #[test]
    fn admission_plan_share_charges_suffix_only() {
        // Established prefix matches: reuse K, decode only the suffix.
        let p = admission_plan(K, Some(&PFX), K, 100, false, &[1, 2, 3, 4, 9, 9], 2);
        assert!(!p.establishing && p.reused == 4);
        assert_eq!(p.effective, 2, "suffix only (6 - 4)");
        assert_eq!((p.inc, p.reserve), (4, 4), "suffix 2 + gen 2");
    }

    #[test]
    fn admission_plan_exact_k_establish_leaves_k_not_k_minus_one() {
        // The regression the mock couldn't catch: prompt is *exactly* K tokens. An
        // establisher must still leave K cells reserved (inc - reserve == K), not K-1.
        let p = admission_plan(K, None, 0, 100, true, &PFX, 3);
        assert!(p.establishing && p.reused == 0);
        assert_eq!(p.effective, K, "full prompt decoded");
        assert_eq!(p.inc - p.reserve, K, "prefix charged exactly once, even at prompt==K");
    }

    #[test]
    fn admission_plan_exact_k_share_keeps_one_token_to_decode() {
        // Prompt exactly K and matches: reuse K-1 so 1 token is still decoded for logits.
        let p = admission_plan(K, Some(&PFX), K, 100, false, &PFX, 3);
        assert_eq!(p.reused, K - 1, "capped so ≥1 token decodes");
        assert_eq!(p.effective, 1);
    }

    #[test]
    fn admission_plan_rejects_over_ctx() {
        assert_eq!(admission_plan(0, None, 0, 10, true, &[1, 2, 3, 4, 5], 100).verdict, Verdict::Reject);
    }

    #[test]
    fn admission_plan_defers_when_busy_but_evicts_when_idle() {
        // Resident prefix (reserved=K) + a fallback whose footprint fits alone (8 ≤ 10)
        // but not atop the prefix (4+8 > 10). Busy → defer; idle → evict + admit.
        let busy = admission_plan(K, Some(&PFX), K, 10, false, &[9, 9, 9, 9, 9], 3);
        assert_eq!(busy.verdict, Verdict::Defer);
        let idle = admission_plan(K, Some(&PFX), K, 10, true, &[9, 9, 9, 9, 9], 3);
        assert_eq!(idle.verdict, Verdict::Admit);
        assert!(idle.evict_prefix && !idle.establishing && idle.reused == 0);
        assert_eq!((idle.inc, idle.effective), (8, 5), "admits as a full-prefill fallback");
    }

    #[test]
    fn shared_reuse_len_caps_and_falls_back() {
        assert_eq!(shared_reuse_len(K, Some(&PFX), &[1, 2, 3, 4, 9]), 4, "match, room for suffix");
        assert_eq!(shared_reuse_len(K, Some(&PFX), &PFX), 3, "exact K → capped to K-1");
        assert_eq!(shared_reuse_len(K, Some(&PFX), &[9, 9, 9, 9, 9]), 0, "mismatch");
        assert_eq!(shared_reuse_len(K, None, &[1, 2, 3, 4, 9]), 0, "establishing decodes all");
    }

    #[test]
    fn argmax_picks_first_max_for_greedy_determinism() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
        assert_eq!(argmax(&[0.5, 0.5]), 0, "ties → first index (matches llama greedy)");
        assert_eq!(argmax(&[-2.0, -0.5, -3.0]), 1, "all negative");
        assert_eq!(argmax(&[]), 0, "empty → 0 (caller must avoid)");
    }

    #[test]
    fn shared_prefix_len_matches_first_k_tokens() {
        let prefix = [1, 2, 3, 4];
        // Exact match on the first K → share K.
        assert_eq!(shared_prefix_len(&[1, 2, 3, 4, 9, 9], &prefix, 4), 4);
        // Prompt longer than K but first K identical → still K (only the window matters).
        assert_eq!(shared_prefix_len(&[1, 2, 3, 4, 5, 6, 7], &prefix, 4), 4);
        // First K differ → no sharing (single-entry v1 fallback).
        assert_eq!(shared_prefix_len(&[1, 2, 0, 4, 9], &prefix, 4), 0);
        // Prompt shorter than K → can't share.
        assert_eq!(shared_prefix_len(&[1, 2, 3], &prefix, 4), 0);
        // Cached prefix shorter than K → can't share.
        assert_eq!(shared_prefix_len(&[1, 2, 3, 4], &[1, 2, 3], 4), 0);
        // K == 0 → feature off, never shares.
        assert_eq!(shared_prefix_len(&[1, 2, 3, 4], &prefix, 0), 0);
    }
}
