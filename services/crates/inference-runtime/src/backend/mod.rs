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

use crate::slab::Slab;

/// Outcome of [`Decoder::admit`] — the admission decision for one pending slot.
/// Backend-independent so Core 2's scheduler loop can match on it without knowing
/// which backend is compiled in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admit {
    /// Admitted into the running set; carries the prompt-token count it prefilled
    /// (Core 2 charges it against the per-iteration prefill-token budget).
    Admitted(usize),
    /// KV budget is full *right now* — leave the slot in `pending` and retry after a
    /// decode step frees cells by retiring a sequence. Only returned when the running
    /// set is non-empty, so an empty pipeline always makes progress.
    Deferred,
    /// Cannot ever be admitted (its prompt + generation exceeds `n_ctx` alone). The
    /// backend already published `Finish(Error)` on R3; Core 2 drops it from `pending`.
    Rejected,
}

/// Prompt-token count of a pending slot, for Core 2's admission token budget. Reads
/// only `tokens.len()` (written by Core 1 during tokenize).
///
/// SAFETY: `slot` is sitting in Core 2's `pending` queue (it arrived via R2 and Core 2
/// has not published anything on R3 for it yet), so Core 2 solely owns it here.
pub fn prompt_len(slab: &Slab, slot: u32) -> usize {
    unsafe { slab.slot_mut(slot) }.tokens.len()
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
    use super::argmax;

    #[test]
    fn argmax_picks_first_max_for_greedy_determinism() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
        assert_eq!(argmax(&[0.5, 0.5]), 0, "ties → first index (matches llama greedy)");
        assert_eq!(argmax(&[-2.0, -0.5, -3.0]), 1, "all negative");
        assert_eq!(argmax(&[]), 0, "empty → 0 (caller must avoid)");
    }
}
