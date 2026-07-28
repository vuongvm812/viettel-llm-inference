//! `VTL_RUST_SCHED_RADIX=1` variant of the prefix-cache index.
//!
//! What this is: an alternative index over the SAME keys as `HashIndex`
//! (`BlockHashToBlockMap`, block_pool.py:34) — a two-level radix bucket on the
//! leading 16 bits of the block hash, then a linear scan of the bucket. Every
//! operation returns exactly what the flat map returns (asserted by
//! `radix_matches_hash_index` below), so switching the flag cannot change a
//! scheduling decision — only how the lookup is performed.
//!
//! What this is NOT: the token-granular sub-block matching from the
//! `radix-attention-port` branch. That changes *hit lengths* (matching inside a
//! block), which is a semantic change and therefore out of scope for a
//! logic-preserving port — vLLM's own block-hash chain is already a radix tree at
//! block granularity, and the measured upside of going finer was +0.1pp.
//!
//! ponytail: bucket + linear scan, not a compressed byte trie. The bucket holds
//! ~num_blocks / 65536 entries (well under one for any pool we run), so the scan is
//! O(1) in practice. Upgrade path if a pool ever exceeds ~1e6 cached blocks: widen
//! the bucket prefix to 24 bits, one constant.

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::block_pool::HashKey;

type Bucket = SmallVec<[(HashKey, u32); 2]>;

#[derive(Default)]
pub struct RadixIndex {
    buckets: FxHashMap<u16, Bucket>,
    len: usize,
}

impl RadixIndex {
    #[inline]
    fn prefix(key: &HashKey) -> u16 {
        u16::from_be_bytes([key.hash[0], key.hash[1]])
    }

    pub fn get_one_block(&self, key: &HashKey) -> Option<u32> {
        let b = self.buckets.get(&Self::prefix(key))?;
        b.iter().find(|(k, _)| k == key).map(|(_, v)| *v)
    }

    pub fn contain(&self, key: &HashKey, block_id: u32) -> bool {
        match self.buckets.get(&Self::prefix(key)) {
            None => false,
            Some(b) => b.iter().any(|(k, v)| k == key && *v == block_id),
        }
    }

    pub fn insert(&mut self, key: HashKey, block_id: u32) {
        let b = self.buckets.entry(Self::prefix(&key)).or_default();
        if b.iter().any(|(k, v)| *k == key && *v == block_id) {
            return;
        }
        if !b.iter().any(|(k, _)| *k == key) {
            self.len += 1;
        }
        b.push((key, block_id));
    }

    pub fn pop(&mut self, key: &HashKey, block_id: u32) -> Option<u32> {
        let prefix = Self::prefix(key);
        let b = self.buckets.get_mut(&prefix)?;
        let idx = b.iter().position(|(k, v)| k == key && *v == block_id)?;
        let (_, v) = b.remove(idx);
        if !b.iter().any(|(k, _)| k == key) {
            self.len -= 1;
        }
        if b.is_empty() {
            self.buckets.remove(&prefix);
        }
        Some(v)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn clear(&mut self) {
        self.buckets.clear();
        self.len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_pool::HashIndex;
    use crate::hash::{Digest32, HASH_LEN};

    fn key(seed: u8, group: u32) -> HashKey {
        let mut hash: Digest32 = [0; HASH_LEN];
        hash[0] = seed;
        hash[1] = seed.wrapping_mul(7);
        hash[31] = seed;
        HashKey { hash, group }
    }

    /// The whole justification for shipping the flag: identical answers, always.
    #[test]
    fn radix_matches_hash_index() {
        let mut flat = HashIndex::default();
        let mut radix = RadixIndex::default();
        // Deterministic pseudo-random op stream; includes duplicate-hash collisions.
        let mut state: u32 = 12345;
        let mut next = || {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            (state >> 16) as u8
        };
        for _ in 0..2000 {
            let k = key(next(), (next() % 3) as u32);
            let b = next() as u32;
            match next() % 3 {
                0 => {
                    flat.insert(k, b);
                    radix.insert(k, b);
                }
                1 => {
                    assert_eq!(flat.pop(&k, b), radix.pop(&k, b));
                }
                _ => {
                    assert_eq!(flat.get_one_block(&k), radix.get_one_block(&k));
                    assert_eq!(flat.contain(&k, b), radix.contain(&k, b));
                }
            }
            assert_eq!(flat.len(), radix.len());
        }
    }
}
