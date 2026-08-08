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

// ------------------------------------------------------------------------------------------
// R8 -- the raw shm output record, built here instead of by Python.
//
// Byte-identical to `vtl/patches/shm_ipc.py`'s `raw_pack_into` (little-endian, unaligned,
// tag byte first); the layout comment lives there and the GOLDEN VECTORS below are the same
// hex strings that module's `_self_check` asserts, so a drift on either side fails a test
// rather than silently handing the Rust frontend a record it decodes into garbage.
//
// A THIRD implementation of this layout is the frontend's decoder in
// `vtl/vllm_patches/rust-frontend/shm_ipc.patch`. All three share these vectors.
// ------------------------------------------------------------------------------------------

pub const TAG_RAW: u8 = b'R';
pub const RAW_VERSION: u8 = 1;
pub const FLAG_FINISH_REASON: u8 = 0b0000_0001;
pub const FLAG_STOP_TOKEN: u8 = 0b0000_0010;

/// `FinishReason` (vllm/v1/engine/__init__.py) for the two statuses `check_stop` can reach.
/// `ABORT`/`ERROR`/`REPETITION` never come out of [`check_stop`], so they are not mapped:
/// a request that finishes for one of those reasons is retired on a path R8 refuses.
pub const FINISH_STOP: u8 = 0;
pub const FINISH_LENGTH: u8 = 1;

/// `Verdict::status` -> `FinishReason`. `None` for a request that did not stop.
pub fn finish_reason(status: u8) -> Option<u8> {
    match status {
        FINISHED_STOPPED => Some(FINISH_STOP),
        FINISHED_LENGTH_CAPPED => Some(FINISH_LENGTH),
        _ => None,
    }
}

/// Gather each request's accepted tokens out of the sampler's `[nrows, ncols]` array.
///
/// Two callers, one implementation: `update_step_pack_np` (source = a numpy array the
/// sampler wrote) and the Rust model runner (source = its own pinned D2H buffer). The layout
/// and the bounds rules are identical, and the row/column arithmetic here is exactly the
/// kind of thing that fails silently -- a wrong `row * ncols` reads another request's tokens
/// and still returns a plausible id -- so it lives OUTSIDE the `python` feature to stay in
/// the default `cargo test` gate the Dockerfile runs.
///
/// Returns an owned `Vec` because the numpy caller must copy before releasing the GIL.
pub fn gather_sampled(
    arr: &[i64],
    nrows: usize,
    ncols: usize,
    rows: &[u32],
    counts: &[u32],
) -> Result<Vec<i64>, String> {
    if rows.len() != counts.len() {
        return Err(format!(
            "gather_sampled arity mismatch: {} rows, {} counts",
            rows.len(),
            counts.len()
        ));
    }
    let mut flat: Vec<i64> = Vec::with_capacity(rows.len() * 2);
    for i in 0..rows.len() {
        let row = rows[i] as usize;
        let cnt = counts[i] as usize;
        if row >= nrows || cnt > ncols {
            return Err(format!(
                "gather_sampled row {row} / count {cnt} outside [{nrows}, {ncols}]"
            ));
        }
        let base = row * ncols;
        // Belt and braces: `nrows`/`ncols` are what the caller CLAIMS the array is. If they
        // over-describe a short buffer, the slice below would panic (or, in the runner's
        // case, read past a pinned allocation) -- so check against the real length too.
        if base + cnt > arr.len() {
            return Err(format!(
                "gather_sampled slice {base}+{cnt} past the {}-element array",
                arr.len()
            ));
        }
        flat.extend_from_slice(&arr[base..base + cnt]);
    }
    Ok(flat)
}

/// Streaming writer for one TAG_RAW record into a reused buffer.
///
/// Streaming rather than "build a Vec of rows then serialize": the row count is known
/// before the loop starts, so the header can be written first and every row appended in
/// place -- no per-step intermediate allocation, and the buffer survives across steps.
#[derive(Default)]
pub struct RawPacker {
    pub buf: Vec<u8>,
    /// Rows/finished ids still owed, so a truncated record cannot reach the wire.
    owed_out: u32,
    owed_fin: u32,
}

impl RawPacker {
    /// Start a record. Clears the buffer and writes the 24-byte header.
    pub fn begin(&mut self, engine_index: u32, timestamp: f64, n_out: u32, n_fin: u32) {
        let buf = &mut self.buf;
        buf.clear();
        buf.reserve(24 + 32 * n_out as usize);
        buf.push(TAG_RAW);
        buf.push(RAW_VERSION);
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&engine_index.to_le_bytes());
        buf.extend_from_slice(&timestamp.to_le_bytes());
        buf.extend_from_slice(&n_out.to_le_bytes());
        buf.extend_from_slice(&n_fin.to_le_bytes());
        self.owed_out = n_out;
        self.owed_fin = n_fin;
    }

    /// One `EngineCoreOutput`. `num_nans_in_logits` is always 0: a non-None nans batch is
    /// not raw-packable in Python either, so the caller falls back before reaching here.
    pub fn push_output(
        &mut self,
        request_id: &str,
        tokens: &[i64],
        finish: Option<u8>,
        stop_reason: Option<i64>,
    ) -> Result<(), String> {
        if self.owed_out == 0 {
            return Err("RawPacker: more outputs pushed than declared".into());
        }
        self.owed_out -= 1;
        let id = request_id.as_bytes();
        let mut flags = 0u8;
        let finish_byte = match finish {
            Some(f) => {
                flags |= FLAG_FINISH_REASON;
                f
            }
            None => 0,
        };
        let stop = match stop_reason {
            Some(s) => {
                flags |= FLAG_STOP_TOKEN;
                u32::try_from(s).map_err(|_| format!("stop_reason {s} does not fit u32"))?
            }
            None => 0,
        };
        let buf = &mut self.buf;
        buf.extend_from_slice(&(id.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // num_nans_in_logits
        buf.push(flags);
        buf.push(finish_byte);
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&stop.to_le_bytes());
        buf.extend_from_slice(id);
        for &tok in tokens {
            let t = u32::try_from(tok).map_err(|_| format!("token id {tok} does not fit u32"))?;
            buf.extend_from_slice(&t.to_le_bytes());
        }
        Ok(())
    }

    /// One entry of the trailing `finished_requests` table. Callers push in SORTED order --
    /// the Python packer sorts a set, and the bytes have to match.
    pub fn push_finished(&mut self, request_id: &str) -> Result<(), String> {
        if self.owed_fin == 0 {
            return Err("RawPacker: more finished ids pushed than declared".into());
        }
        self.owed_fin -= 1;
        let id = request_id.as_bytes();
        self.buf.extend_from_slice(&(id.len() as u32).to_le_bytes());
        self.buf.extend_from_slice(id);
        Ok(())
    }

    /// The finished record. `Err` if the declared counts were not filled -- a short record
    /// would decode as garbage on the frontend, so it must never leave this type.
    pub fn finish(&self) -> Result<&[u8], String> {
        if self.owed_out != 0 || self.owed_fin != 0 {
            return Err(format!(
                "RawPacker: record incomplete ({} outputs, {} finished ids missing)",
                self.owed_out, self.owed_fin
            ));
        }
        Ok(&self.buf)
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
mod gather_tests {
    use super::gather_sampled;

    /// `[3 rows, 4 cols]`, values encode (row, col) so a mis-strided read is obvious.
    fn arr() -> Vec<i64> {
        (0..3).flat_map(|r| (0..4).map(move |c| r * 100 + c)).collect()
    }

    #[test]
    fn gathers_variable_counts_from_the_right_rows() {
        let out = gather_sampled(&arr(), 3, 4, &[0, 2, 1], &[1, 3, 2]).unwrap();
        // row 0 x1, row 2 x3, row 1 x2 -- in REQUEST order, not row order.
        assert_eq!(out, vec![0, 200, 201, 202, 100, 101]);
    }

    #[test]
    fn a_zero_count_contributes_nothing() {
        // A request that accepted no token still occupies a slot; it must not shift the
        // ones after it.
        let out = gather_sampled(&arr(), 3, 4, &[0, 1, 2], &[1, 0, 1]).unwrap();
        assert_eq!(out, vec![0, 200]);
    }

    #[test]
    fn full_width_rows_round_trip() {
        let out = gather_sampled(&arr(), 3, 4, &[1], &[4]).unwrap();
        assert_eq!(out, vec![100, 101, 102, 103]);
    }

    #[test]
    fn out_of_range_row_is_an_error_not_a_wrong_answer() {
        assert!(gather_sampled(&arr(), 3, 4, &[3], &[1]).is_err());
    }

    #[test]
    fn count_wider_than_the_array_is_refused() {
        assert!(gather_sampled(&arr(), 3, 4, &[0], &[5]).is_err());
    }

    #[test]
    fn a_short_buffer_is_caught_even_when_the_shape_looks_fine() {
        // The caller claims [3, 4] but hands over 8 elements. Row 2 is in range per the
        // shape and past the end of the buffer -- this is the runner's failure mode if a
        // pinned allocation is sized from a stale batch.
        let short = vec![0i64; 8];
        assert!(gather_sampled(&short, 3, 4, &[2], &[1]).is_err());
    }

    #[test]
    fn mismatched_arity_is_refused() {
        assert!(gather_sampled(&arr(), 3, 4, &[0, 1], &[1]).is_err());
    }

    #[test]
    fn empty_batch_yields_empty() {
        assert_eq!(gather_sampled(&arr(), 3, 4, &[], &[]).unwrap(), Vec::<i64>::new());
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

    // ---- R8 record packer -----------------------------------------------------------
    //
    // The two hex strings below are copied VERBATIM from
    // `round-2/vtl/patches/shm_ipc.py` (`GOLDEN_PLAIN_DECODE` / `GOLDEN_FINISHED`),
    // which `make check` asserts against the Python packer. Same bytes on both sides or
    // both test suites fail.

    const GOLDEN_PLAIN_DECODE: &str = concat!(
        "52", "01", "0000", "07000000", "000000000000f03f", "01000000", "00000000",
        "06000000", "01000000", "00000000", "00", "00", "0000", "00000000", "7265712d3161",
        "05000000",
    );

    const GOLDEN_FINISHED: &str = concat!(
        "52", "01", "0000", "00000000", "0000000000000000", "02000000", "02000000",
        "05000000", "02000000", "00000000", "03", "01", "0000", "02000000", "7265712d61",
        "03000000", "04000000",
        "05000000", "00000000", "00000000", "01", "02", "0000", "00000000", "7265712d62",
        "05000000", "7265712d61",
        "05000000", "7265712d62",
    );

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn plain_decode_record_matches_the_python_golden_vector() {
        let mut p = RawPacker::default();
        p.begin(7, 1.0, 1, 0);
        p.push_output("req-1a", &[5], None, None).unwrap();
        assert_eq!(hex(p.finish().unwrap()), GOLDEN_PLAIN_DECODE);
    }

    #[test]
    fn finished_record_matches_the_python_golden_vector() {
        let mut p = RawPacker::default();
        p.begin(0, 0.0, 2, 2);
        // finish_reason=1 (LENGTH), stop_reason=2, two tokens.
        p.push_output("req-a", &[3, 4], Some(1), Some(2)).unwrap();
        // finish_reason=2 (ABORT in Python's enum -- the packer just carries the byte),
        // no tokens, no stop_reason.
        p.push_output("req-b", &[], Some(2), None).unwrap();
        // Python sorts the finished SET; the packer here trusts the caller's order, so the
        // caller must sort. This is that order.
        p.push_finished("req-a").unwrap();
        p.push_finished("req-b").unwrap();
        assert_eq!(hex(p.finish().unwrap()), GOLDEN_FINISHED);
    }

    #[test]
    fn record_buffer_is_reused_without_leaking_the_previous_record() {
        let mut p = RawPacker::default();
        p.begin(0, 0.0, 1, 0);
        p.push_output("aaaaaaaaaaaaaaaaaaaa", &[1, 2, 3, 4, 5], None, None).unwrap();
        let long = p.finish().unwrap().len();
        p.begin(7, 1.0, 1, 0);
        p.push_output("req-1a", &[5], None, None).unwrap();
        assert!(p.finish().unwrap().len() < long);
        assert_eq!(hex(p.finish().unwrap()), GOLDEN_PLAIN_DECODE);
    }

    #[test]
    fn an_incomplete_or_overfull_record_is_refused_not_shipped() {
        let mut p = RawPacker::default();
        p.begin(0, 0.0, 2, 1);
        p.push_output("a", &[1], None, None).unwrap();
        assert!(p.finish().is_err(), "1 of 2 outputs written");
        p.push_output("b", &[1], None, None).unwrap();
        assert!(p.finish().is_err(), "finished id still owed");
        p.push_finished("a").unwrap();
        assert!(p.finish().is_ok());
        assert!(p.push_output("c", &[1], None, None).is_err());
        assert!(p.push_finished("b").is_err());
    }

    #[test]
    fn out_of_range_ids_are_errors_not_truncations() {
        let mut p = RawPacker::default();
        p.begin(0, 0.0, 1, 0);
        assert!(p.push_output("a", &[1 << 33], None, None).is_err());
        p.begin(0, 0.0, 1, 0);
        assert!(p.push_output("a", &[-1], None, None).is_err());
        p.begin(0, 0.0, 1, 0);
        assert!(p.push_output("a", &[1], None, Some(-2)).is_err());
    }

    #[test]
    fn utf8_request_ids_are_length_prefixed_in_bytes() {
        let mut p = RawPacker::default();
        p.begin(0, 0.0, 1, 0);
        p.push_output("req-é", &[1], None, None).unwrap();
        let rec = p.finish().unwrap();
        // 24 header + 20 row head + 6 utf-8 bytes ("req-" + 2-byte é) + 4 token bytes.
        assert_eq!(rec.len(), 24 + 20 + 6 + 4);
        assert_eq!(u32::from_le_bytes(rec[24..28].try_into().unwrap()), 6);
    }

    #[test]
    fn finish_reason_maps_only_the_two_statuses_check_stop_can_reach() {
        assert_eq!(finish_reason(NOT_STOPPED), None);
        assert_eq!(finish_reason(FINISHED_STOPPED), Some(FINISH_STOP));
        assert_eq!(finish_reason(FINISHED_LENGTH_CAPPED), Some(FINISH_LENGTH));
        assert_eq!(finish_reason(u8::MAX), None);
    }

    #[test]
    fn empty_token_list_is_a_no_op() {
        let mut t = StopTable::default();
        t.set(0, params());
        let out = t.update_step(&[0], &[0, 0], &[], &[3], &[10], MML).unwrap();
        assert_eq!(out[0], Verdict { num_accepted: 0, status: NOT_STOPPED, stop_reason: -1 });
    }
}
