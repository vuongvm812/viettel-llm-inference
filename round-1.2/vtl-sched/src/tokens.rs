//! Per-slot token bookkeeping, owned by Rust instead of by `Request._all_token_ids`.
//!
//! WHY THIS EXISTS. `update_step_pack` already receives every sampled token of every step.
//! The only Python consumers that need the token INTS are (a) `NewRequestData.
//! prefill_token_ids` at admission/resume and (b) the block hasher; everything else needs
//! counts. So the store keeps the counts and the hash chain here, and Python's lists stop
//! growing -- which deletes the two `tolist()`s in `AsyncOutput.get_output`, the
//! per-request `toks.extend` in `decide()`, the `append_output_token_ids` extend, and the
//! per-step Python hasher invocation.
//!
//! WHAT IT KEEPS, and why each piece is the minimum:
//!   * `out`     -- output tokens appended since the store took over the slot. ONLY used by
//!                  the rare-path `slot_tokens()` materialization (preemption/resume, any
//!                  pack refusal). Bounded by `max_tokens`.
//!   * `pending` -- the tokens after the last hashed block boundary, always
//!                  `< hash_block_size` of them. This is the whole reason the store does not
//!                  need the prompt: the chain state is (last hash, unhashed tail), and
//!                  Python seeds the tail once at activation.
//!   * counters  -- `num_tokens` / `num_output_tokens`, the values `check_stop` and the
//!                  resident table read.
//!
//! The hash chain itself lives in `Manager::reqs[slot].hashes`, NOT here: that is the vec
//! `push_hashes` / `num_hashes` already maintain, so `RustMirror` keeps working unchanged
//! and there is exactly one list of block hashes per slot in the crate.

use rustc_hash::FxHashMap;

use crate::hash::{hash_block_tokens, Digest32};

/// One slot's token state. See the module docs for why these three fields are enough.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SlotTokens {
    pub out: Vec<i64>,
    pub pending: Vec<i64>,
    pub num_tokens: usize,
    pub num_output_tokens: usize,
}

/// Slot -> [`SlotTokens`], plus the two hashing constants.
///
/// `none_hash` is `None` until Python hands over vLLM's live `NONE_HASH`; an unarmed store
/// refuses to activate a slot rather than hashing against a guessed seed (a silent
/// prefix-cache key-space split, i.e. a 0% hit rate nobody notices).
#[derive(Default)]
pub struct TokenStore {
    slots: FxHashMap<u32, SlotTokens>,
    none_hash: Option<Digest32>,
    hash_block_size: usize,
}

impl TokenStore {
    pub fn new(hash_block_size: usize) -> Self {
        Self {
            slots: FxHashMap::default(),
            none_hash: None,
            hash_block_size,
        }
    }

    pub fn arm(&mut self, none_hash: Digest32) {
        self.none_hash = Some(none_hash);
    }

    #[inline]
    pub fn armed(&self) -> bool {
        self.none_hash.is_some()
    }

    #[inline]
    pub fn hash_block_size(&self) -> usize {
        self.hash_block_size
    }

    /// Take over one slot. `pending` is the request's token tail past the last block the
    /// Python hasher already produced a hash for, so `pending.len() < hash_block_size`
    /// always holds -- a longer tail means Python's hasher and this one disagree about how
    /// many blocks are complete, which is refused rather than reconciled.
    pub fn init(
        &mut self,
        slot: u32,
        pending: &[i64],
        num_tokens: usize,
        num_output_tokens: usize,
    ) -> Result<(), String> {
        if self.none_hash.is_none() {
            return Err("token store is not armed (no NONE_HASH)".into());
        }
        if self.hash_block_size == 0 {
            return Err("token store needs a non-zero hash_block_size".into());
        }
        if pending.len() >= self.hash_block_size {
            return Err(format!(
                "token store seed tail of {} tokens is not shorter than hash_block_size {}",
                pending.len(),
                self.hash_block_size
            ));
        }
        self.slots.insert(
            slot,
            SlotTokens {
                out: Vec::new(),
                pending: pending.to_vec(),
                num_tokens,
                num_output_tokens,
            },
        );
        Ok(())
    }

    #[inline]
    pub fn active(&self, slot: u32) -> bool {
        self.slots.contains_key(&slot)
    }

    /// `(num_tokens, num_output_tokens)`, the two counters Python's facade mirrors.
    #[inline]
    pub fn counts(&self, slot: u32) -> Option<(usize, usize)> {
        self.slots
            .get(&slot)
            .map(|s| (s.num_tokens, s.num_output_tokens))
    }

    /// Output tokens only -- what Python appends to `_output_token_ids` / `_all_token_ids`
    /// when it materializes the request back onto the stock path.
    #[inline]
    pub fn tokens(&self, slot: u32) -> Option<&[i64]> {
        self.slots.get(&slot).map(|s| s.out.as_slice())
    }

    #[inline]
    pub fn pending(&self, slot: u32) -> Option<&[i64]> {
        self.slots.get(&slot).map(|s| s.pending.as_slice())
    }

    /// Slot recycling and the materialization hand-back. Idempotent.
    #[inline]
    pub fn forget(&mut self, slot: u32) {
        self.slots.remove(&slot);
    }

    /// Append `tokens` to one slot and run the block-hash catch-up into `hashes`.
    ///
    /// `hashes` is the slot's live `ReqState::hashes`, i.e. the same vec `push_hashes`
    /// appends to and `num_hashes` measures -- so a caller that mixes the two paths still
    /// sees one consistent chain.
    ///
    /// A block boundary chains from `hashes.last()`, exactly as
    /// `get_request_block_hasher`'s loop does; the first block of a request chains from
    /// `NONE_HASH` because `hash_block_tokens` substitutes it for an absent parent.
    pub fn append(
        &mut self,
        slot: u32,
        tokens: &[i64],
        hashes: &mut Vec<Digest32>,
    ) -> Result<(), String> {
        let none_hash = self
            .none_hash
            .ok_or_else(|| "token store is not armed (no NONE_HASH)".to_string())?;
        let bs = self.hash_block_size;
        let st = self
            .slots
            .get_mut(&slot)
            .ok_or_else(|| format!("token store has no entry for slot {slot}"))?;
        st.out.reserve(tokens.len());
        for &tok in tokens {
            st.out.push(tok);
            st.pending.push(tok);
            st.num_tokens += 1;
            st.num_output_tokens += 1;
            if st.pending.len() == bs {
                let parent = hashes.last().copied();
                let h = hash_block_tokens(
                    &none_hash,
                    parent.as_ref().map(|p| &p[..]),
                    &st.pending,
                    None,
                );
                hashes.push(h);
                st.pending.clear();
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::{hash_request_tokens, none_hash_from_seed};

    fn armed(bs: usize) -> TokenStore {
        let mut s = TokenStore::new(bs);
        s.arm(none_hash_from_seed("0"));
        s
    }

    #[test]
    fn unarmed_store_refuses_to_activate() {
        let mut s = TokenStore::new(4);
        assert!(!s.armed());
        assert!(s.init(0, &[], 0, 0).is_err());
        s.arm([7u8; 32]);
        assert!(s.init(0, &[], 0, 0).is_ok());
    }

    #[test]
    fn a_seed_tail_at_or_past_the_block_size_is_refused() {
        let mut s = armed(4);
        assert!(s.init(0, &[1, 2, 3], 3, 0).is_ok());
        assert!(s.init(1, &[1, 2, 3, 4], 4, 0).is_err());
    }

    #[test]
    fn counters_track_every_appended_token() {
        let mut s = armed(4);
        s.init(0, &[], 100, 0).unwrap();
        let mut hashes = Vec::new();
        s.append(0, &[5, 6], &mut hashes).unwrap();
        assert_eq!(s.counts(0), Some((102, 2)));
        s.append(0, &[7], &mut hashes).unwrap();
        assert_eq!(s.counts(0), Some((103, 3)));
        assert_eq!(s.tokens(0), Some(&[5i64, 6, 7][..]));
    }

    #[test]
    fn the_chain_matches_hashing_the_whole_sequence_at_once() {
        // A request whose prompt is [1,2,3,4,5] with block size 4: Python hashed block
        // [1,2,3,4] and seeded the store with the tail [5]. Six decode tokens follow.
        let nh = none_hash_from_seed("0");
        let all: Vec<i64> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let want = hash_request_tokens(&nh, 4, &all, None);

        let mut hashes = hash_request_tokens(&nh, 4, &all[..4], None);
        assert_eq!(hashes.len(), 1);
        let mut s = armed(4);
        s.init(0, &all[4..5], 5, 0).unwrap();
        for tok in &all[5..] {
            s.append(0, &[*tok], &mut hashes).unwrap();
        }
        assert_eq!(hashes, want[..hashes.len()].to_vec());
        assert_eq!(hashes.len(), all.len() / 4);
        // The unhashed tail is exactly the leftover.
        assert_eq!(s.pending(0), Some(&[9i64, 10, 11][..]));
    }

    #[test]
    fn a_bulk_append_hashes_the_same_as_one_token_at_a_time() {
        let nh = none_hash_from_seed("0");
        let mut bulk = armed(4);
        let mut one = armed(4);
        bulk.init(0, &[1], 1, 0).unwrap();
        one.init(0, &[1], 1, 0).unwrap();
        let mut hb: Vec<Digest32> = Vec::new();
        let mut ho: Vec<Digest32> = Vec::new();
        bulk.append(0, &[2, 3, 4, 5, 6, 7, 8, 9], &mut hb).unwrap();
        for t in 2..10i64 {
            one.append(0, &[t], &mut ho).unwrap();
        }
        assert_eq!(hb, ho);
        assert_eq!(hb.len(), 2);
        assert_eq!(bulk.counts(0), one.counts(0));
        // ...and both agree with the reference hasher over [1..9].
        let all: Vec<i64> = (1..10).collect();
        assert_eq!(hb, hash_request_tokens(&nh, 4, &all, None)[..2].to_vec());
    }

    #[test]
    fn forget_recycles_a_slot_with_no_residue() {
        let mut s = armed(4);
        s.init(0, &[1], 1, 0).unwrap();
        let mut hashes = Vec::new();
        s.append(0, &[2, 3, 4], &mut hashes).unwrap();
        assert!(s.active(0));
        s.forget(0);
        assert!(!s.active(0));
        assert_eq!(s.counts(0), None);
        assert_eq!(s.tokens(0), None);
        assert!(s.append(0, &[5], &mut hashes).is_err(), "a dead slot refuses");
        // A recycled slot starts clean, not from the dead request's tokens.
        s.init(0, &[], 0, 0).unwrap();
        assert_eq!(s.counts(0), Some((0, 0)));
        assert_eq!(s.tokens(0), Some(&[][..]));
    }
}
