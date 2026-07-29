//! R6a — port of the stop decision in `Scheduler.update_from_output`, batched per step.
//!
//! Mirrors, exactly:
//!   * `vllm/v1/core/sched/utils.py:94-130`  `check_stop`
//!   * `vllm/v1/core/sched/scheduler.py:1897` `_update_request_with_output`'s append loop
//!
//! WHY BATCHED, AND WHY ONLY THIS. `check_stop` is six integer comparisons; calling into
//! Rust once per request per step to run them would cost more in PyO3 crossings than the
//! comparisons themselves. So the whole step crosses once: flat `slots` / `cu_lens` /
//! `token_ids` in, a flat verdict list out. Everything that needs a Python object —
//! `append_output_token_ids` (the detokenizer and the block hasher both read it),
//! `EngineCoreOutput`, queue removal, `_free_request` — stays in Python. This module owns
//! the DECISION, not the bookkeeping.
//!
//! WHY THE COUNTERS ARE INPUTS, NOT STATE. `num_output_tokens` and `num_tokens` are mutated
//! by Python in a dozen places (preemption resets, spec-decode rejection, KV-load failure
//! rollback). Mirroring them here would be a drift bug waiting for a rare branch; they are
//! two more ints per request per step on a path that already ships a token list. Only the
//! genuinely immutable half — the sampling params — is interned per slot.
//!
//! NOT PORTED, and refused by the caller instead of guessed: `repetition_detection`
//! (`check_sequence_repetition` is an O(n) scan over the whole output with its own tuning
//! knobs), pooling requests, and spec decode (the caller gates on
//! `scheduled_spec_decode_tokens` being empty, so `new_token_ids` is always the single
//! sampled token here — the loop below still handles n>1 because nothing costs extra).

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

/// `RequestStatus` codes crossing the boundary. Python maps them back to the enum.
pub const NOT_STOPPED: u8 = 0;
pub const FINISHED_STOPPED: u8 = 1;
pub const FINISHED_LENGTH_CAPPED: u8 = 2;

/// No `stop_reason`. vLLM leaves the attribute at `None`; only the stop-token-id branch
/// sets it, and token ids are non-negative.
pub const NO_STOP_REASON: i64 = -1;

/// The immutable half of a request's stop condition, interned once per slot.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StopParams {
    pub min_tokens: usize,
    pub max_tokens: usize,
    /// `sampling_params.eos_token_id`; `None` when the model/request has none.
    pub eos_token_id: Option<i64>,
    /// `sampling_params.stop_token_ids`. A linear scan, like vLLM's `in (... or ())`:
    /// these lists are 0-2 long in practice, so a set would be slower.
    pub stop_token_ids: SmallVec<[i64; 4]>,
}

/// One request's answer: how many of the offered tokens survive, and why it stopped.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Verdict {
    /// Tokens to append. Equals `tokens.len()` unless a stop truncated the tail.
    pub num_accepted: u32,
    pub status: u8,
    pub stop_reason: i64,
}

/// `check_stop`, with the counters already advanced past `last_token_id`.
///
/// Branch order is load-bearing: EOS wins over an explicit stop id that happens to equal
/// it (only the status differs, but `stop_reason` does not get set), and both win over the
/// length cap — a request whose last token is EOS on exactly `max_tokens` reports STOPPED,
/// not LENGTH_CAPPED, and vLLM's finish_reason is derived from the status.
fn check_stop(
    p: &StopParams,
    last_token_id: i64,
    num_output_tokens: usize,
    num_tokens: usize,
    max_model_len: usize,
) -> Option<(u8, i64)> {
    if num_output_tokens < p.min_tokens {
        return None;
    }
    if Some(last_token_id) == p.eos_token_id {
        return Some((FINISHED_STOPPED, NO_STOP_REASON));
    }
    if p.stop_token_ids.contains(&last_token_id) {
        return Some((FINISHED_STOPPED, last_token_id));
    }
    if num_tokens >= max_model_len || num_output_tokens >= p.max_tokens {
        return Some((FINISHED_LENGTH_CAPPED, NO_STOP_REASON));
    }
    None
}

/// `_update_request_with_output`'s loop: append, check, trim on the first stop.
///
/// `num_output_tokens` / `num_tokens` are the values BEFORE this step's tokens are
/// appended, exactly as Python holds them at the call site.
pub fn apply_tokens(
    p: &StopParams,
    tokens: &[i64],
    mut num_output_tokens: usize,
    mut num_tokens: usize,
    max_model_len: usize,
) -> Verdict {
    for (i, &tok) in tokens.iter().enumerate() {
        num_output_tokens += 1;
        num_tokens += 1;
        if let Some((status, stop_reason)) =
            check_stop(p, tok, num_output_tokens, num_tokens, max_model_len)
        {
            return Verdict {
                num_accepted: (i + 1) as u32,
                status,
                stop_reason,
            };
        }
    }
    Verdict {
        num_accepted: tokens.len() as u32,
        status: NOT_STOPPED,
        stop_reason: NO_STOP_REASON,
    }
}

/// Slot -> stop params. Lives on `Manager` so `forget()` drops it with the interned id.
#[derive(Default)]
pub struct StopTable {
    params: FxHashMap<u32, StopParams>,
    /// Scratch reused across steps so the hot path allocates nothing.
    pub out: Vec<Verdict>,
}

impl StopTable {
    pub fn set(&mut self, slot: u32, params: StopParams) {
        self.params.insert(slot, params);
    }

    pub fn has(&self, slot: u32) -> bool {
        self.params.contains_key(&slot)
    }

    pub fn forget(&mut self, slot: u32) {
        self.params.remove(&slot);
    }

    /// One step. `cu_lens` is the exclusive prefix sum of per-request token counts, so
    /// request `i` owns `token_ids[cu_lens[i]..cu_lens[i + 1]]` (`cu_lens[0] == 0`).
    ///
    /// An unregistered slot yields `status = u8::MAX`, which the caller must read as
    /// "fall back to Python for this request" rather than "not stopped".
    #[allow(clippy::too_many_arguments)]
    pub fn update_step(
        &mut self,
        slots: &[u32],
        cu_lens: &[u32],
        token_ids: &[i64],
        num_output_tokens: &[u32],
        num_tokens: &[u32],
        max_model_len: usize,
    ) -> Result<&[Verdict], String> {
        let n = slots.len();
        if cu_lens.len() != n + 1 || num_output_tokens.len() != n || num_tokens.len() != n {
            return Err(format!(
                "update_step arity mismatch: {n} slots, {} cu_lens, {} out_toks, {} toks",
                cu_lens.len(),
                num_output_tokens.len(),
                num_tokens.len()
            ));
        }
        if cu_lens.first() != Some(&0) || cu_lens[n] as usize != token_ids.len() {
            return Err("update_step cu_lens must start at 0 and end at token_ids.len()".into());
        }
        self.out.clear();
        self.out.reserve(n);
        for i in 0..n {
            let (lo, hi) = (cu_lens[i] as usize, cu_lens[i + 1] as usize);
            // hi is checked against token_ids.len() too: [0, 5, 3] with 3 tokens passes the
            // first/last checks but would slice out of bounds at i=0 before i=1's lo > hi
            // fires -- a panic where the contract promises an Err.
            if lo > hi || hi > token_ids.len() {
                return Err("update_step cu_lens must be non-decreasing and bounded".into());
            }
            match self.params.get(&slots[i]) {
                Some(p) => self.out.push(apply_tokens(
                    p,
                    &token_ids[lo..hi],
                    num_output_tokens[i] as usize,
                    num_tokens[i] as usize,
                    max_model_len,
                )),
                None => self.out.push(Verdict {
                    num_accepted: 0,
                    status: u8::MAX,
                    stop_reason: NO_STOP_REASON,
                }),
            }
        }
        Ok(&self.out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> StopParams {
        StopParams {
            min_tokens: 0,
            max_tokens: 8,
            eos_token_id: Some(2),
            stop_token_ids: SmallVec::from_slice(&[13, 99]),
        }
    }

    const MML: usize = 1000;

    #[test]
    fn no_stop() {
        let v = apply_tokens(&params(), &[7], 0, 100, MML);
        assert_eq!(v, Verdict { num_accepted: 1, status: NOT_STOPPED, stop_reason: -1 });
    }

    #[test]
    fn eos_stops_without_a_reason() {
        let v = apply_tokens(&params(), &[2], 0, 100, MML);
        assert_eq!(v, Verdict { num_accepted: 1, status: FINISHED_STOPPED, stop_reason: -1 });
    }

    #[test]
    fn stop_token_id_carries_its_reason() {
        let v = apply_tokens(&params(), &[99], 0, 100, MML);
        assert_eq!(v, Verdict { num_accepted: 1, status: FINISHED_STOPPED, stop_reason: 99 });
    }

    #[test]
    fn max_tokens_caps() {
        // 7 already emitted; this one makes 8 == max_tokens.
        let v = apply_tokens(&params(), &[7], 7, 100, MML);
        assert_eq!(v.status, FINISHED_LENGTH_CAPPED);
        assert_eq!(v.num_accepted, 1);
    }

    #[test]
    fn max_model_len_caps() {
        let v = apply_tokens(&params(), &[7], 0, MML - 1, MML);
        assert_eq!(v.status, FINISHED_LENGTH_CAPPED);
    }

    #[test]
    fn min_tokens_suppresses_every_stop() {
        let mut p = params();
        p.min_tokens = 4;
        // EOS, a stop id and the length cap all fire below min_tokens -> none of them count.
        assert_eq!(apply_tokens(&p, &[2], 0, 100, MML).status, NOT_STOPPED);
        assert_eq!(apply_tokens(&p, &[99], 1, 100, MML).status, NOT_STOPPED);
        assert_eq!(apply_tokens(&p, &[7], 2, MML - 1, MML).status, NOT_STOPPED);
        // ...and the first token at or past min_tokens does.
        assert_eq!(apply_tokens(&p, &[2], 3, 100, MML).status, FINISHED_STOPPED);
    }

    #[test]
    fn eos_beats_an_identical_stop_id_and_the_length_cap() {
        let mut p = params();
        p.stop_token_ids = SmallVec::from_slice(&[2]);
        // Same token in both lists: EOS wins, so stop_reason stays unset.
        assert_eq!(apply_tokens(&p, &[2], 7, 100, MML).stop_reason, NO_STOP_REASON);
        assert_eq!(apply_tokens(&p, &[2], 7, 100, MML).status, FINISHED_STOPPED);
    }

    #[test]
    fn multi_token_trims_at_the_first_stop() {
        // Spec-decode-shaped input: 4 offered, EOS third -> keep 3, drop the tail.
        let v = apply_tokens(&params(), &[5, 6, 2, 7], 0, 100, MML);
        assert_eq!(v.num_accepted, 3);
        assert_eq!(v.status, FINISHED_STOPPED);
    }

    #[test]
    fn no_eos_configured_never_stops_on_it() {
        let mut p = params();
        p.eos_token_id = None;
        assert_eq!(apply_tokens(&p, &[2], 0, 100, MML).status, NOT_STOPPED);
    }

    #[test]
    fn table_batches_and_flags_unregistered_slots() {
        let mut t = StopTable::default();
        t.set(0, params());
        t.set(1, params());
        // slot 2 deliberately unregistered.
        let out = t
            .update_step(&[0, 1, 2], &[0, 1, 2, 3], &[7, 2, 5], &[0, 0, 0], &[10, 10, 10], MML)
            .unwrap()
            .to_vec();
        assert_eq!(out[0].status, NOT_STOPPED);
        assert_eq!(out[1].status, FINISHED_STOPPED);
        assert_eq!(out[2].status, u8::MAX, "unregistered slot must be refused, not answered");

        t.forget(1);
        assert!(!t.has(1) && t.has(0));

        // Arity is validated, not trusted: a short cu_lens is a caller bug, not a panic.
        assert!(t.update_step(&[0], &[0], &[7], &[0], &[10], MML).is_err());
        assert!(t.update_step(&[0], &[1, 2], &[7], &[0], &[10], MML).is_err());
        assert!(t.update_step(&[0], &[0, 5], &[7], &[0], &[10], MML).is_err());
    }

    #[test]
    fn empty_token_list_is_a_no_op() {
        let mut t = StopTable::default();
        t.set(0, params());
        let out = t.update_step(&[0], &[0, 0], &[], &[3], &[10], MML).unwrap();
        assert_eq!(out[0], Verdict { num_accepted: 0, status: NOT_STOPPED, stop_reason: -1 });
    }
}
