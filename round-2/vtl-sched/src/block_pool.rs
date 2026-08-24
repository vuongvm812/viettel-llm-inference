//! Port of `vllm/v1/core/block_pool.py` + the `FreeKVCacheBlockQueue` / `KVCacheBlock`
//! half of `vllm/v1/core/kv_cache_utils.py`.
//!
//! Mirrors:
//!   * `kv_cache_utils.py:117` `KVCacheBlock`            -> [`Block`]
//!   * `kv_cache_utils.py:179` `FreeKVCacheBlockQueue`   -> [`BlockArena`] (index arena, same
//!                                                          fake head/tail sentinels)
//!   * `block_pool.py:34`      `BlockHashToBlockMap`     -> [`HashIndex`] (+ `radix.rs` variant)
//!   * `block_pool.py:144`     `BlockPool`               -> [`BlockPool`]
//!
//! Everything is index-based over one `Vec<Block>` sized at construction from the startup
//! KV budget, so no per-step heap allocation happens in the free-list / refcount paths.
//!
//! Not ported (constructor rejects them, see `config.rs`): KV cache events
//! (`enable_kv_cache_events`) and `KVCacheMetricsCollector`. Both are inert on this stack
//! and are pure side channels — porting them would add allocation to the hot path for
//! output nobody reads.

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::hash::{Digest32, HASH_LEN};
use crate::journal::{Journal, Undo};
use crate::radix::{Bucket, RadixIndex};

pub const NIL: u32 = u32::MAX;

/// `BlockHashWithGroupId` (kv_cache_utils.py:49): the block hash bytes with a 4-byte
/// big-endian group id appended. Kept unpacked; `to_bytes()` reproduces the wire form.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct HashKey {
    pub hash: Digest32,
    pub group: u32,
}

impl HashKey {
    pub fn to_bytes(&self) -> [u8; HASH_LEN + 4] {
        let mut out = [0u8; HASH_LEN + 4];
        out[..HASH_LEN].copy_from_slice(&self.hash);
        out[HASH_LEN..].copy_from_slice(&self.group.to_be_bytes());
        out
    }
}

#[derive(Clone, Debug)]
pub struct Block {
    pub ref_cnt: i32,
    pub hash: Option<HashKey>,
    pub hash_num_tokens: Option<u32>,
    pub prev: u32,
    pub next: u32,
    pub is_null: bool,
}

impl Default for Block {
    fn default() -> Self {
        Block {
            ref_cnt: 0,
            hash: None,
            hash_num_tokens: None,
            prev: NIL,
            next: NIL,
            is_null: false,
        }
    }
}

/// The block arena plus the doubly-linked free queue that threads through it.
///
/// vLLM allocates two extra `KVCacheBlock(block_id=-1)` sentinels; we place them at
/// indices `n` (head) and `n + 1` (tail) of the same arena so the link math is uniform.
///
/// The speculative undo journal lives here because [`BlockArena::get_mut`] is the single
/// choke point every block write in the crate goes through — ref counts, hashes and the
/// free-queue links all land in the same `Block`, so one recording site covers the whole
/// allocator. `journal` is `None` in steady state; the check is one predictable branch.
pub struct BlockArena {
    pub blocks: Vec<Block>,
    pub num_free_blocks: usize,
    pub(crate) journal: Option<Journal>,
    head: u32,
    tail: u32,
}

impl BlockArena {
    pub fn new(num_blocks: usize) -> Self {
        let mut blocks = vec![Block::default(); num_blocks + 2];
        let head = num_blocks as u32;
        let tail = num_blocks as u32 + 1;
        for i in 0..num_blocks {
            if i > 0 {
                blocks[i].prev = (i - 1) as u32;
            }
            if i + 1 < num_blocks {
                blocks[i].next = (i + 1) as u32;
            }
        }
        if num_blocks > 0 {
            blocks[head as usize].next = 0;
            blocks[0].prev = head;
            blocks[tail as usize].prev = (num_blocks - 1) as u32;
            blocks[num_blocks - 1].next = tail;
        } else {
            blocks[head as usize].next = tail;
            blocks[tail as usize].prev = head;
        }
        BlockArena {
            blocks,
            num_free_blocks: num_blocks,
            journal: None,
            head,
            tail,
        }
    }

    #[inline]
    pub fn get(&self, id: u32) -> &Block {
        &self.blocks[id as usize]
    }

    /// THE write choke point. Every mutation of any block — links, ref count, hash —
    /// goes through here, which is what makes the speculative journal exhaustive over the
    /// allocator without a recording call at each of the ~30 individual assignments.
    #[inline]
    pub fn get_mut(&mut self, id: u32) -> &mut Block {
        if let Some(j) = self.journal.as_mut() {
            j.push(Undo::Block(id, self.blocks[id as usize].clone()));
        }
        &mut self.blocks[id as usize]
    }

    #[inline]
    fn set_free(&mut self, n: usize) {
        if let Some(j) = self.journal.as_mut() {
            j.push(Undo::FreeCount(self.num_free_blocks));
        }
        self.num_free_blocks = n;
    }

    /// `FreeKVCacheBlockQueue.popleft` (kv_cache_utils.py:231).
    pub fn popleft(&mut self) -> Result<u32, String> {
        let first = self.blocks[self.head as usize].next;
        if first == self.tail || first == NIL {
            if self.num_free_blocks != 0 {
                return Err(format!(
                    "num_free_blocks ({}) is out of sync with the free list",
                    self.num_free_blocks
                ));
            }
            return Err("No free blocks available".into());
        }
        let next = self.blocks[first as usize].next;
        if next == NIL {
            return Err("Invalid block found in popleft() without a valid next_free_block".into());
        }
        let head = self.head;
        self.get_mut(head).next = next;
        self.get_mut(next).prev = head;
        let b = self.get_mut(first);
        b.prev = NIL;
        b.next = NIL;
        self.set_free(self.num_free_blocks - 1);
        Ok(first)
    }

    /// `FreeKVCacheBlockQueue.popleft_n` (kv_cache_utils.py:268).
    pub fn popleft_n(&mut self, n: usize, out: &mut Vec<u32>) {
        if n == 0 {
            return;
        }
        debug_assert!(self.num_free_blocks >= n);
        self.set_free(self.num_free_blocks - n);
        let mut curr = self.blocks[self.head as usize].next;
        for _ in 0..n {
            debug_assert_ne!(curr, NIL);
            out.push(curr);
            let last = curr;
            curr = self.blocks[curr as usize].next;
            let b = self.get_mut(last);
            b.prev = NIL;
            b.next = NIL;
        }
        if curr != NIL {
            let head = self.head;
            self.get_mut(head).next = curr;
            self.get_mut(curr).prev = head;
        }
    }

    /// `FreeKVCacheBlockQueue.remove` (kv_cache_utils.py:301).
    pub fn remove(&mut self, id: u32) -> Result<(), String> {
        let (prev, next) = {
            let b = &self.blocks[id as usize];
            (b.prev, b.next)
        };
        if prev == NIL || next == NIL {
            return Err(format!("remove() called on an invalid block: {id}"));
        }
        self.get_mut(prev).next = next;
        self.get_mut(next).prev = prev;
        let b = self.get_mut(id);
        b.prev = NIL;
        b.next = NIL;
        self.set_free(self.num_free_blocks - 1);
        Ok(())
    }

    /// `FreeKVCacheBlockQueue.prepend_n` (kv_cache_utils.py:344).
    pub fn prepend_n(&mut self, ids: &[u32]) {
        if ids.is_empty() {
            return;
        }
        let first = self.blocks[self.head as usize].next;
        let mut prev = self.head;
        for &id in ids {
            self.get_mut(id).prev = prev;
            self.get_mut(prev).next = id;
            prev = id;
        }
        self.get_mut(prev).next = first;
        self.get_mut(first).prev = prev;
        self.set_free(self.num_free_blocks + ids.len());
    }

    /// `FreeKVCacheBlockQueue.append_n` (kv_cache_utils.py:365).
    pub fn append_n(&mut self, ids: &[u32]) {
        if ids.is_empty() {
            return;
        }
        let mut last = self.blocks[self.tail as usize].prev;
        for &id in ids {
            self.get_mut(id).prev = last;
            self.get_mut(last).next = id;
            last = id;
        }
        self.get_mut(last).next = self.tail;
        let tail = self.tail;
        self.get_mut(tail).prev = last;
        self.set_free(self.num_free_blocks + ids.len());
    }

    /// `FreeKVCacheBlockQueue.get_all_free_blocks` — test/debug only.
    pub fn free_order(&self) -> Vec<u32> {
        let mut out = Vec::new();
        let mut curr = self.blocks[self.head as usize].next;
        while curr != NIL && curr != self.tail {
            out.push(curr);
            curr = self.blocks[curr as usize].next;
        }
        out
    }
}

/// `BlockHashToBlockMap` (block_pool.py:34). The `One | Many` split mirrors upstream's
/// `KVCacheBlock | dict[int, KVCacheBlock]` union, which exists to keep GC pressure down.
#[derive(Debug, Clone)]
pub(crate) enum Entry {
    One(u32),
    Many(SmallVec<[u32; 4]>),
}

/// One key's pre-mutation mapping, recorded by the speculative journal. Opaque on
/// purpose: the two index implementations restore different things (a map entry vs a
/// whole radix bucket plus the distinct-key count), and neither is expressible as an
/// inverse `insert`/`pop` — `One`->`Many` promotion is not reversible by popping.
#[derive(Debug)]
pub(crate) enum IndexSnapshot {
    Hash(Option<Entry>),
    Radix(Option<Bucket>, usize),
}

#[derive(Default)]
pub struct HashIndex {
    map: FxHashMap<HashKey, Entry>,
}

impl HashIndex {
    pub fn get_one_block(&self, key: &HashKey) -> Option<u32> {
        match self.map.get(key)? {
            Entry::One(b) => Some(*b),
            Entry::Many(v) => v.first().copied(),
        }
    }

    pub fn contain(&self, key: &HashKey, block_id: u32) -> bool {
        match self.map.get(key) {
            None => false,
            Some(Entry::One(b)) => *b == block_id,
            Some(Entry::Many(v)) => v.contains(&block_id),
        }
    }

    pub fn insert(&mut self, key: HashKey, block_id: u32) {
        let promote = match self.map.get_mut(&key) {
            None => {
                self.map.insert(key, Entry::One(block_id));
                return;
            }
            Some(Entry::One(old)) => {
                if *old == block_id {
                    return;
                }
                *old
            }
            Some(Entry::Many(v)) => {
                if !v.contains(&block_id) {
                    v.push(block_id);
                }
                return;
            }
        };
        let mut v: SmallVec<[u32; 4]> = SmallVec::new();
        v.push(promote);
        v.push(block_id);
        self.map.insert(key, Entry::Many(v));
    }

    /// `BlockHashToBlockMap.pop` (block_pool.py:107).
    pub fn pop(&mut self, key: &HashKey, block_id: u32) -> Option<u32> {
        match self.map.remove(key)? {
            Entry::One(b) => {
                if b == block_id {
                    Some(b)
                } else {
                    // Single block ID mismatch: put it back (upstream keeps this for safety).
                    self.map.insert(*key, Entry::One(b));
                    None
                }
            }
            Entry::Many(mut v) => {
                let found = v.iter().position(|&b| b == block_id).map(|i| v.remove(i));
                if !v.is_empty() {
                    self.map.insert(*key, Entry::Many(v));
                }
                found
            }
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }

    pub(crate) fn snapshot(&self, key: &HashKey) -> Option<Entry> {
        self.map.get(key).cloned()
    }

    pub(crate) fn restore(&mut self, key: &HashKey, snap: Option<Entry>) {
        match snap {
            Some(e) => {
                self.map.insert(*key, e);
            }
            None => {
                self.map.remove(key);
            }
        }
    }

    /// Order-independent content dump for [`crate::manager::Manager::state_fingerprint`].
    pub(crate) fn entries(&self) -> Vec<(HashKey, SmallVec<[u32; 4]>)> {
        self.map
            .iter()
            .map(|(k, e)| {
                let v: SmallVec<[u32; 4]> = match e {
                    Entry::One(b) => SmallVec::from_slice(&[*b]),
                    Entry::Many(v) => v.clone(),
                };
                (*k, v)
            })
            .collect()
    }
}

/// Selectable index implementation. `Hash` is upstream's flat map; `Radix` is the
/// token-granular trie variant (`VTL_RUST_SCHED_RADIX=1`) — same keys, same answers,
/// different data structure. Parity between them is asserted in `radix.rs` tests.
pub enum Index {
    Hash(HashIndex),
    Radix(RadixIndex),
}

impl Index {
    pub fn get_one_block(&self, key: &HashKey) -> Option<u32> {
        match self {
            Index::Hash(i) => i.get_one_block(key),
            Index::Radix(i) => i.get_one_block(key),
        }
    }
    pub fn contain(&self, key: &HashKey, block_id: u32) -> bool {
        match self {
            Index::Hash(i) => i.contain(key, block_id),
            Index::Radix(i) => i.contain(key, block_id),
        }
    }
    pub fn insert(&mut self, key: HashKey, block_id: u32) {
        match self {
            Index::Hash(i) => i.insert(key, block_id),
            Index::Radix(i) => i.insert(key, block_id),
        }
    }
    pub fn pop(&mut self, key: &HashKey, block_id: u32) -> Option<u32> {
        match self {
            Index::Hash(i) => i.pop(key, block_id),
            Index::Radix(i) => i.pop(key, block_id),
        }
    }
    pub fn clear(&mut self) {
        match self {
            Index::Hash(i) => i.clear(),
            Index::Radix(i) => i.clear(),
        }
    }

    /// Speculative journal support. The radix variant snapshots the WHOLE bucket the key
    /// falls in: neighbouring keys that share the 16-bit prefix are restored along with
    /// it, which is still exact because their own snapshots (taken earlier) are replayed
    /// after this one in the reversed pass.
    pub(crate) fn snapshot(&self, key: &HashKey) -> IndexSnapshot {
        match self {
            Index::Hash(i) => IndexSnapshot::Hash(i.snapshot(key)),
            Index::Radix(i) => {
                let (b, n) = i.snapshot(key);
                IndexSnapshot::Radix(b, n)
            }
        }
    }

    pub(crate) fn restore(&mut self, key: &HashKey, snap: IndexSnapshot) {
        match (self, snap) {
            (Index::Hash(i), IndexSnapshot::Hash(e)) => i.restore(key, e),
            (Index::Radix(i), IndexSnapshot::Radix(b, n)) => i.restore(key, b, n),
            // Unreachable: the index implementation is fixed at construction.
            _ => unreachable!("index snapshot kind does not match the live index"),
        }
    }

    pub(crate) fn entries(&self) -> Vec<(HashKey, SmallVec<[u32; 4]>)> {
        match self {
            Index::Hash(i) => i.entries(),
            Index::Radix(i) => i.entries(),
        }
    }
}

pub struct BlockPool {
    pub arena: BlockArena,
    pub num_gpu_blocks: usize,
    pub enable_caching: bool,
    pub hash_block_size: usize,
    pub null_block: u32,
    pub(crate) cached: Index,
    /// `cached_block_hashes_by_block` (block_pool.py:186): secondary hash keys for a block
    /// that already owns a primary hash (partial-block aliases).
    pub(crate) cached_by_block: FxHashMap<u32, SmallVec<[HashKey; 2]>>,
    /// Scratch buffers reused across calls — the "no per-step heap allocation" discipline.
    scratch_ids: Vec<u32>,
    scratch_with_hash: Vec<u32>,
    scratch_without_hash: Vec<u32>,
}

impl BlockPool {
    pub fn new(
        num_gpu_blocks: usize,
        enable_caching: bool,
        hash_block_size: usize,
        radix: bool,
    ) -> Result<Self, String> {
        if num_gpu_blocks == 0 {
            return Err("num_gpu_blocks must be > 0".into());
        }
        let mut arena = BlockArena::new(num_gpu_blocks);
        // block_pool.py:191 — the null block is popped out of the free queue at init.
        let null_block = arena.popleft()?;
        arena.get_mut(null_block).is_null = true;
        Ok(BlockPool {
            arena,
            num_gpu_blocks,
            enable_caching,
            hash_block_size,
            null_block,
            cached: if radix {
                Index::Radix(RadixIndex::default())
            } else {
                Index::Hash(HashIndex::default())
            },
            cached_by_block: FxHashMap::default(),
            scratch_ids: Vec::with_capacity(64),
            scratch_with_hash: Vec::with_capacity(64),
            scratch_without_hash: Vec::with_capacity(64),
        })
    }

    /// `BlockPool.get_cached_block` (block_pool.py:199). Cache miss on ANY group -> None.
    pub fn get_cached_block(
        &self,
        block_hash: &Digest32,
        group_ids: &[u32],
        out: &mut SmallVec<[u32; 4]>,
    ) -> bool {
        out.clear();
        for &group in group_ids {
            let key = HashKey {
                hash: *block_hash,
                group,
            };
            match self.cached.get_one_block(&key) {
                Some(b) => out.push(b),
                None => {
                    out.clear();
                    return false;
                }
            }
        }
        true
    }

    /// Stage S: is `block_id` STILL the pool's block for `key`?
    ///
    /// The staleness primitive the connector phase split needs. A probe recorded in phase 1
    /// lists blocks that were cache hits AT THAT MOMENT; between the phases (and inside
    /// phase 2, between two candidates) an allocation can evict one of them --
    /// [`Self::get_new_blocks`] calls [`Self::maybe_evict_cached_block`] on every block it
    /// pops, which clears `Block::hash` and pops the index entry -- and hand it to another
    /// request. Consuming such a hit would `touch` a block a second owner already holds.
    ///
    /// BOTH halves are checked because both move independently: eviction clears the block's
    /// own key, and the index entry is what a different block cached under the same hash
    /// would take over.
    ///
    /// `key` is the block's PRIMARY hash as of the walk (`Block::hash`). A block that was
    /// matched under a SECONDARY key (`cached_by_block`, the partial-block alias) still
    /// answers correctly: `_remove_cached_block_hashes` pops the primary and every alias
    /// together, so the primary key going away is exactly the event this test is for.
    pub fn block_still_cached_as(&self, block_id: u32, key: &HashKey) -> bool {
        self.arena.get(block_id).hash == Some(*key) && self.cached.contain(key, block_id)
    }

    /// Record the pre-mutation mapping of `key` when a speculative run is in flight.
    #[inline]
    fn journal_index(&mut self, key: &HashKey) {
        if self.arena.journal.is_some() {
            // Snapshot first, THEN take the journal borrow: `cached` and `arena` are
            // disjoint fields but the borrow checker still wants them sequenced.
            let snap = self.cached.snapshot(key);
            if let Some(j) = self.arena.journal.as_mut() {
                j.push(Undo::Index(*key, snap));
            }
        }
    }

    #[inline]
    fn journal_cached_by_block(&mut self, block_id: u32) {
        if self.arena.journal.is_some() {
            let old = self.cached_by_block.get(&block_id).cloned();
            if let Some(j) = self.arena.journal.as_mut() {
                j.push(Undo::CachedByBlock(block_id, old));
            }
        }
    }

    /// `BlockPool._remove_cached_block_hashes` (block_pool.py:484).
    fn remove_cached_block_hashes(&mut self, block_id: u32) -> usize {
        let mut keys: SmallVec<[HashKey; 3]> = SmallVec::new();
        if let Some(h) = self.arena.get(block_id).hash {
            keys.push(h);
        }
        if self.cached_by_block.contains_key(&block_id) {
            self.journal_cached_by_block(block_id);
        }
        if let Some(extra) = self.cached_by_block.remove(&block_id) {
            keys.extend(extra.into_iter());
        }
        if keys.is_empty() {
            return 0;
        }
        let mut removed = 0;
        for key in &keys {
            let key = *key;
            self.journal_index(&key);
            if self.cached.pop(&key, block_id).is_some() {
                removed += 1;
            }
        }
        let b = self.arena.get_mut(block_id);
        b.hash = None;
        b.hash_num_tokens = None;
        removed
    }

    /// `BlockPool._insert_block_hash` (block_pool.py:520).
    fn insert_block_hash(&mut self, key: HashKey, block_id: u32, num_tokens: Option<u32>) {
        if self.arena.get(block_id).hash == Some(key) {
            return;
        }
        if self.cached.contain(&key, block_id) {
            return;
        }
        if self.arena.get(block_id).hash.is_none() {
            let b = self.arena.get_mut(block_id);
            b.hash = Some(key);
            b.hash_num_tokens = num_tokens;
        } else {
            self.journal_cached_by_block(block_id);
            self.cached_by_block.entry(block_id).or_default().push(key);
        }
        self.journal_index(&key);
        self.cached.insert(key, block_id);
    }

    /// `BlockPool.cache_full_blocks` (block_pool.py:226), events path omitted.
    ///
    /// `block_hashes` are the request's hashes at `hash_block_size` granularity; when
    /// `block_size > hash_block_size` the caller strides them (see
    /// `BlockHashListWithBlockSize`, kv_cache_utils.py:2225).
    #[allow(clippy::too_many_arguments)]
    pub fn cache_full_blocks(
        &mut self,
        req_blocks: &[u32],
        block_hashes: &[Digest32],
        hash_stride: usize,
        num_cached_blocks: usize,
        num_full_blocks: usize,
        block_size: usize,
        group_id: u32,
        block_mask: Option<&[bool]>,
    ) {
        if num_cached_blocks >= num_full_blocks {
            return;
        }
        // block_pool.py:274 `assert len(block_hashes) >= num_full_blocks`
        assert!(
            block_hashes.len() >= num_full_blocks * hash_stride,
            "not enough block hashes: {} for {} full blocks (stride {})",
            block_hashes.len(),
            num_full_blocks,
            hash_stride
        );
        for i in 0..(num_full_blocks - num_cached_blocks) {
            let blk = req_blocks[num_cached_blocks + i];
            if self.arena.get(blk).is_null {
                continue;
            }
            if let Some(mask) = block_mask {
                if !mask[i] {
                    continue;
                }
            }
            // `new_block_hashes = block_hashes[num_cached_blocks:]` then `[i]`, at the
            // group's block-size stride.
            let hash_idx = (num_cached_blocks + i + 1) * hash_stride - 1;
            let block_hash = block_hashes[hash_idx];
            let num_hash_tokens = ((num_cached_blocks + i + 1) * block_size) as u32;
            let key = HashKey {
                hash: block_hash,
                group: group_id,
            };
            if self.arena.get(blk).hash.is_some() {
                // Partial->full promotion of the same cache block.
                debug_assert!(
                    self.arena.get(blk).hash_num_tokens.unwrap_or(0) < num_hash_tokens
                );
                self.remove_cached_block_hashes(blk);
            }
            self.insert_block_hash(key, blk, Some(num_hash_tokens));
        }
    }

    /// `BlockPool.cache_partial_block` (block_pool.py:358), events path omitted.
    /// Only reachable when a group's `block_size > hash_block_size`.
    pub fn cache_partial_block(
        &mut self,
        block_id: u32,
        block_hashes: &[Digest32],
        num_tokens: usize,
        group_id: u32,
        block_size: usize,
    ) -> Option<HashKey> {
        if self.arena.get(block_id).is_null {
            return None;
        }
        assert!(block_size > self.hash_block_size);
        assert!(block_size % self.hash_block_size == 0);
        assert!(num_tokens % block_size != 0);
        assert!(num_tokens % self.hash_block_size == 0);
        let num_hash_blocks = num_tokens / self.hash_block_size;
        assert!(num_hash_blocks > 0 && num_hash_blocks <= block_hashes.len());
        let key = HashKey {
            hash: block_hashes[num_hash_blocks - 1],
            group: group_id,
        };
        let already_cached =
            self.arena.get(block_id).hash == Some(key) || self.cached.contain(&key, block_id);
        let promote = !already_cached
            && self.arena.get(block_id).hash.is_some()
            && self
                .arena
                .get(block_id)
                .hash_num_tokens
                .is_some_and(|n| (n as usize) < num_hash_blocks * self.hash_block_size);
        if promote {
            self.remove_cached_block_hashes(block_id);
        }
        self.insert_block_hash(
            key,
            block_id,
            Some((num_hash_blocks * self.hash_block_size) as u32),
        );
        Some(key)
    }

    /// `BlockPool.get_new_blocks` (block_pool.py:542).
    pub fn get_new_blocks(&mut self, num_blocks: usize, out: &mut Vec<u32>) -> Result<(), String> {
        if num_blocks > self.get_num_free_blocks() {
            return Err(format!(
                "Cannot get {num_blocks} free blocks from the pool"
            ));
        }
        let start = out.len();
        self.arena.popleft_n(num_blocks, out);
        for i in start..out.len() {
            let id = out[i];
            if self.enable_caching {
                self.maybe_evict_cached_block(id);
            }
            debug_assert_eq!(self.arena.get(id).ref_cnt, 0);
            self.arena.get_mut(id).ref_cnt += 1;
        }
        Ok(())
    }

    /// `BlockPool._maybe_evict_cached_block` (block_pool.py:574).
    pub fn maybe_evict_cached_block(&mut self, block_id: u32) -> bool {
        let evicted = self.remove_cached_block_hashes(block_id);
        if evicted == 0 {
            return false;
        }
        true
    }

    /// `BlockPool.touch` (block_pool.py:597).
    ///
    /// The null placeholder is NOT ref-counted here, unlike Python. Python bumps it on
    /// touch and decrements it on free and simply tolerates the drift (`is_null` keeps
    /// it out of the free queue either way); a drifting counter on this side would trip
    /// `free_blocks`' double-free assert in debug and wrap in release, so both paths
    /// skip it and its counter stays honestly at 0. The one reader that can observe the
    /// difference is `get_num_common_prefix_blocks` (single_type.rs), whose ref-count
    /// scan now always breaks at a leading null instead of coincidentally continuing
    /// when Python's drifted count equals num_reqs -- the conservative answer, and moot
    /// on lean boots, which skip that call entirely.
    pub fn touch(&mut self, blocks: &[u32]) -> Result<(), String> {
        for &id in blocks {
            let b = self.arena.get(id);
            if b.is_null {
                continue;
            }
            if b.ref_cnt == 0 {
                // Freed-but-cached block: pull it out of the eviction queue. A failure
                // here means the free list and the ref counts disagree -- unwinding is
                // the only safe answer, since the next allocation would hand this block
                // to a second request.
                self.arena.remove(id)?;
            }
            self.arena.get_mut(id).ref_cnt += 1;
        }
        Ok(())
    }

    /// `BlockPool.free_blocks` (block_pool.py:614). `ordered` is in *eviction* order,
    /// i.e. already reversed by the caller.
    pub fn free_blocks(&mut self, ordered: &[u32]) {
        let mut with_hash = std::mem::take(&mut self.scratch_with_hash);
        let mut without_hash = std::mem::take(&mut self.scratch_without_hash);
        with_hash.clear();
        without_hash.clear();
        for &id in ordered {
            let b = self.arena.get_mut(id);
            // Mamba-align rows pad request tables with the null placeholder, so it
            // arrives here on every hybrid free -- skip it (see `touch`), which is what
            // keeps the double-free assert meaningful for real blocks.
            if b.is_null {
                continue;
            }
            debug_assert!(b.ref_cnt > 0, "double free of block {id}");
            b.ref_cnt -= 1;
            if b.ref_cnt == 0 {
                if b.hash.is_none() {
                    without_hash.push(id);
                } else {
                    with_hash.push(id);
                }
            }
        }
        // Blocks without a hash can never serve a prefix hit -> evict them first.
        self.arena.prepend_n(&without_hash);
        self.arena.append_n(&with_hash);
        self.scratch_with_hash = with_hash;
        self.scratch_without_hash = without_hash;
    }

    /// `BlockPool.evict_blocks` (block_pool.py:637).
    pub fn evict_blocks(&mut self, block_ids: &[u32]) {
        for &id in block_ids {
            assert!((id as usize) < self.num_gpu_blocks, "invalid block_id {id}");
            self.maybe_evict_cached_block(id);
        }
    }

    /// `BlockPool.reset_prefix_cache` (block_pool.py:656).
    pub fn reset_prefix_cache(&mut self) -> bool {
        // Whole-arena sweep: deliberately NOT journaled. Unreachable while a speculative
        // run is armed (see the scope invariant in `journal.rs`) — every PyO3 entry point
        // rolls the speculation back before it mutates.
        debug_assert!(self.arena.journal.is_none());
        let num_used = self.num_gpu_blocks - self.get_num_free_blocks();
        if num_used != 1 {
            return false;
        }
        self.cached.clear();
        self.cached_by_block.clear();
        for b in self.arena.blocks.iter_mut() {
            b.hash = None;
            b.hash_num_tokens = None;
        }
        true
    }

    #[inline]
    pub fn get_num_free_blocks(&self) -> usize {
        self.arena.num_free_blocks
    }

    /// `BlockPool.get_usage` (block_pool.py:700) — the null block is excluded.
    pub fn get_usage(&self) -> f64 {
        let total = self.num_gpu_blocks - 1;
        if total == 0 {
            return 0.0;
        }
        1.0 - (self.get_num_free_blocks() as f64 / total as f64)
    }

    pub fn take_scratch(&mut self) -> Vec<u32> {
        let mut v = std::mem::take(&mut self.scratch_ids);
        v.clear();
        v
    }

    pub fn put_scratch(&mut self, v: Vec<u32>) {
        self.scratch_ids = v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(n: u8) -> Digest32 {
        [n; HASH_LEN]
    }

    #[test]
    fn free_queue_is_fifo_by_block_id_initially() {
        let arena = BlockArena::new(5);
        assert_eq!(arena.free_order(), vec![0, 1, 2, 3, 4]);
        assert_eq!(arena.num_free_blocks, 5);
    }

    #[test]
    fn popleft_remove_append_roundtrip() {
        let mut a = BlockArena::new(4);
        assert_eq!(a.popleft().unwrap(), 0);
        assert_eq!(a.free_order(), vec![1, 2, 3]);
        a.remove(2).unwrap();
        assert_eq!(a.free_order(), vec![1, 3]);
        assert_eq!(a.num_free_blocks, 2);
        a.append_n(&[2]);
        assert_eq!(a.free_order(), vec![1, 3, 2]);
        a.prepend_n(&[0]);
        assert_eq!(a.free_order(), vec![0, 1, 3, 2]);
        assert_eq!(a.num_free_blocks, 4);
    }

    #[test]
    fn popleft_n_matches_repeated_popleft() {
        let mut a = BlockArena::new(6);
        let mut got = Vec::new();
        a.popleft_n(3, &mut got);
        assert_eq!(got, vec![0, 1, 2]);
        assert_eq!(a.free_order(), vec![3, 4, 5]);
        let mut b = BlockArena::new(6);
        let each: Vec<u32> = (0..3).map(|_| b.popleft().unwrap()).collect();
        assert_eq!(each, got);
        assert_eq!(b.free_order(), a.free_order());
    }

    #[test]
    fn popleft_on_empty_errors_not_panics() {
        let mut a = BlockArena::new(1);
        assert_eq!(a.popleft().unwrap(), 0);
        assert!(a.popleft().is_err());
    }

    #[test]
    fn null_block_is_reserved_and_usage_excludes_it() {
        let p = BlockPool::new(4, true, 16, false).unwrap();
        assert_eq!(p.null_block, 0);
        assert!(p.arena.get(p.null_block).is_null);
        assert_eq!(p.get_num_free_blocks(), 3);
        assert_eq!(p.get_usage(), 0.0);
    }

    #[test]
    fn eviction_order_unhashed_first() {
        let mut p = BlockPool::new(6, true, 16, false).unwrap();
        let mut got = Vec::new();
        p.get_new_blocks(3, &mut got).unwrap();
        assert_eq!(got, vec![1, 2, 3]);
        // Give block 2 a hash so it lands at the tail (kept longer), 1 and 3 at the head.
        p.insert_block_hash(HashKey { hash: h(9), group: 0 }, 2, Some(16));
        p.free_blocks(&[3, 2, 1]);
        // without_hash prepended (3, 1) then with_hash appended (2), tail was [4, 5].
        assert_eq!(p.arena.free_order(), vec![3, 1, 4, 5, 2]);
    }

    #[test]
    fn refcount_touch_and_free() {
        let mut p = BlockPool::new(6, true, 16, false).unwrap();
        let mut got = Vec::new();
        p.get_new_blocks(1, &mut got).unwrap();
        let b = got[0];
        assert_eq!(p.arena.get(b).ref_cnt, 1);
        p.insert_block_hash(HashKey { hash: h(1), group: 0 }, b, Some(16));
        p.free_blocks(&[b]);
        assert_eq!(p.arena.get(b).ref_cnt, 0);
        let free_before = p.get_num_free_blocks();
        // Touching a ref_cnt==0 cached block pulls it back out of the free queue.
        p.touch(&[b]);
        assert_eq!(p.arena.get(b).ref_cnt, 1);
        assert_eq!(p.get_num_free_blocks(), free_before - 1);
    }

    #[test]
    fn allocation_evicts_the_cached_hash() {
        let mut p = BlockPool::new(4, true, 16, false).unwrap();
        let mut got = Vec::new();
        p.get_new_blocks(1, &mut got).unwrap();
        let b = got[0];
        let key = HashKey { hash: h(7), group: 0 };
        p.insert_block_hash(key, b, Some(16));
        p.free_blocks(&[b]);
        assert!(p.cached.get_one_block(&key).is_some());
        // Drain the pool: b comes back around and must lose its hash on reallocation.
        let mut all = Vec::new();
        p.get_new_blocks(p.get_num_free_blocks(), &mut all).unwrap();
        assert!(all.contains(&b));
        assert!(p.cached.get_one_block(&key).is_none());
    }

    #[test]
    fn hash_index_union_semantics() {
        let mut idx = HashIndex::default();
        let k = HashKey { hash: h(3), group: 1 };
        idx.insert(k, 10);
        assert_eq!(idx.get_one_block(&k), Some(10));
        idx.insert(k, 11);
        // Duplicated hash: first inserted wins the lookup (dict insertion order).
        assert_eq!(idx.get_one_block(&k), Some(10));
        assert!(idx.contain(&k, 11));
        assert_eq!(idx.pop(&k, 10), Some(10));
        assert_eq!(idx.get_one_block(&k), Some(11));
        assert_eq!(idx.pop(&k, 99), None);
        assert_eq!(idx.pop(&k, 11), Some(11));
        assert!(idx.is_empty());
    }

    #[test]
    fn get_cached_block_needs_every_group() {
        let mut p = BlockPool::new(8, true, 16, false).unwrap();
        p.insert_block_hash(HashKey { hash: h(5), group: 0 }, 1, Some(16));
        let mut out = SmallVec::new();
        assert!(!p.get_cached_block(&h(5), &[0, 1], &mut out));
        assert!(out.is_empty());
        p.insert_block_hash(HashKey { hash: h(5), group: 1 }, 2, Some(16));
        assert!(p.get_cached_block(&h(5), &[0, 1], &mut out));
        assert_eq!(out.as_slice(), &[1, 2]);
    }

    #[test]
    fn reset_prefix_cache_requires_only_null_held() {
        let mut p = BlockPool::new(4, true, 16, false).unwrap();
        let mut got = Vec::new();
        p.get_new_blocks(1, &mut got).unwrap();
        assert!(!p.reset_prefix_cache());
        p.insert_block_hash(HashKey { hash: h(1), group: 0 }, got[0], Some(16));
        p.free_blocks(&got);
        assert!(p.reset_prefix_cache());
        assert!(p.arena.get(got[0]).hash.is_none());
    }
}
