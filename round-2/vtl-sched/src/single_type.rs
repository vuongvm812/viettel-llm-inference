//! Port of `vllm/v1/core/single_type_kv_cache_manager.py` for the two spec kinds the
//! two kinds vLLM exposes as cache specs: full attention, and mamba-family state
//! (`mamba_cache_mode="align"`). A model's layer counts never reach this code.
//!
//! Mirrors:
//!   * `:33`   `SingleTypeKVCacheManager`      -> [`TypeManager`] (shared paths)
//!   * `:102`  `get_num_blocks_to_allocate`    -> [`TypeManager::get_num_blocks_to_allocate`]
//!   * `:183`  `add_local_computed_blocks`     -> [`TypeManager::add_local_computed_blocks`]
//!   * `:279`  `allocate_new_blocks`           -> [`TypeManager::allocate_new_blocks`]
//!   * `:319`  `cache_blocks`                  -> [`TypeManager::cache_blocks`]
//!   * `:480`  `_remove_blocks_in_range`       -> [`TypeManager::remove_blocks_in_range`]
//!   * `:507`  `remove_skipped_blocks`         -> [`TypeManager::remove_skipped_blocks`]
//!   * `:564`  `FullAttentionManager`          -> [`Kind::FullAttention`] arms
//!   * `:1026` `MambaManager`                  -> [`Kind::Mamba`] arms
//!
//! HYBRID WARNING (why this file is the risky one): the two kinds disagree on almost
//! every rule — mamba scans the hash chain right-to-left and keeps exactly one state
//! block (`:1066`), skips `num_computed_tokens - 1` tokens instead of 0 (`:1335`),
//! rewrites its own block list in align mode (`:1265`), and refuses blocks another
//! request cached in the same step (`:1196`). Getting any of these wrong corrupts
//! output silently rather than raising.
//!
//! Deliberately NOT ported (the config gate in `config.rs` rejects them at construction
//! so they can never be silently wrong): sliding-window / chunked-local / cross-attention /
//! sink managers, `reachable_block_mask` sparse retention
//! (`VLLM_PREFIX_CACHE_RETENTION_INTERVAL`), DCP/PCP block-size scaling, the
//! recycling-aware admission cap (`_max_admission_blocks_per_request`, only set for SWA /
//! chunked-local), and external/connector computed blocks.

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::block_pool::{BlockPool, HashKey};
use crate::hash::Digest32;
use crate::journal::Undo;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    FullAttention,
    Mamba,
}

/// `cdiv` (vllm/utils/math_utils.py).
#[inline]
pub fn cdiv(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// Python floor division, which differs from Rust `/` for negatives
/// (`-1 // 16 == -1`, and mamba's `get_num_skipped_tokens` does go negative).
#[inline]
fn floordiv(a: i64, b: i64) -> i64 {
    a.div_euclid(b)
}

/// Strided view over a request's `hash_block_size`-granular block hashes, reproducing
/// `BlockHashListWithBlockSize` (kv_cache_utils.py:2225).
#[derive(Clone, Copy)]
pub struct HashView<'a> {
    pub hashes: &'a [Digest32],
    pub stride: usize,
}

impl<'a> HashView<'a> {
    #[inline]
    pub fn len(&self) -> usize {
        self.hashes.len() / self.stride
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    #[inline]
    pub fn get(&self, idx: usize) -> &'a Digest32 {
        &self.hashes[(idx + 1) * self.stride - 1]
    }
}

pub struct TypeManager {
    pub kind: Kind,
    pub block_size: usize,
    pub group_id: u32,
    pub scheduler_block_size: usize,
    pub enable_caching: bool,
    /// Set by the coordinator after grouping (kv_cache_coordinator.py:596).
    pub use_eagle: bool,
    /// Whether the spec is one of the types whose new block IDs need zeroing
    /// (single_type_kv_cache_manager.py:304).
    pub tracks_new_block_ids: bool,
    pub mamba_align: bool,
    pub num_speculative_blocks: usize,
    pub null_block: u32,

    pub req_to_blocks: FxHashMap<u32, Vec<u32>>,
    pub num_cached_block: FxHashMap<u32, usize>,
    pub new_block_ids: Vec<u32>,
    /// MambaManager only (`:1031`).
    pub cached_blocks_this_step: FxHashSet<HashKey>,
    /// MambaManager align mode only (`:1037`).
    pub last_state_block_idx: FxHashMap<u32, usize>,
    pub allocated_block_reqs: FxHashSet<u32>,

    /// Per-request capacity reserved on first touch so the steady-state decode append
    /// never reallocates.
    reserve_blocks: usize,
    scratch: Vec<u32>,
}

impl TypeManager {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kind: Kind,
        block_size: usize,
        group_id: u32,
        scheduler_block_size: usize,
        enable_caching: bool,
        mamba_align: bool,
        num_speculative_blocks: usize,
        null_block: u32,
        max_model_len: usize,
    ) -> Self {
        TypeManager {
            kind,
            block_size,
            group_id,
            scheduler_block_size,
            enable_caching,
            use_eagle: false,
            tracks_new_block_ids: kind == Kind::FullAttention,
            mamba_align,
            num_speculative_blocks,
            null_block,
            req_to_blocks: FxHashMap::default(),
            num_cached_block: FxHashMap::default(),
            new_block_ids: Vec::with_capacity(256),
            cached_blocks_this_step: FxHashSet::default(),
            last_state_block_idx: FxHashMap::default(),
            allocated_block_reqs: FxHashSet::default(),
            reserve_blocks: cdiv(max_model_len, block_size) + num_speculative_blocks + 2,
            scratch: Vec::with_capacity(64),
        }
    }

    /// Record an op on the speculative journal, if one is armed. `pool` is threaded into
    /// every mutating method purely so the journal (which lives on the arena, the one
    /// object every allocator path already holds) is reachable from here.
    #[inline]
    fn jrn(&self, pool: &mut BlockPool, op: impl FnOnce() -> Undo) {
        if let Some(j) = pool.arena.journal.as_mut() {
            j.push(op());
        }
    }

    #[inline]
    fn blocks_mut(&mut self, pool: &mut BlockPool, req: u32) -> &mut Vec<u32> {
        // `defaultdict(list)` — the entry must be created on read, because
        // `get_num_common_prefix_blocks` counts `len(self.req_to_blocks)`.
        if pool.arena.journal.is_some() && !self.req_to_blocks.contains_key(&req) {
            let g = self.group_id;
            self.jrn(pool, || Undo::ReqBlocksInsert(g, req));
        }
        let cap = self.reserve_blocks;
        self.req_to_blocks
            .entry(req)
            .or_insert_with(|| Vec::with_capacity(cap))
    }

    /// `blocks_mut` for callers that are about to APPEND: records the length to truncate
    /// back to.
    #[inline]
    fn blocks_mut_for_push(&mut self, pool: &mut BlockPool, req: u32) -> &mut Vec<u32> {
        if pool.arena.journal.is_some() {
            let (g, len) = (self.group_id, self.blocks(req).len());
            let existed = self.req_to_blocks.contains_key(&req);
            if let Some(j) = pool.arena.journal.as_mut() {
                if !existed {
                    j.push(Undo::ReqBlocksInsert(g, req));
                }
                j.push(Undo::ReqBlocksLen(g, req, len));
            }
        }
        let cap = self.reserve_blocks;
        self.req_to_blocks
            .entry(req)
            .or_insert_with(|| Vec::with_capacity(cap))
    }

    #[inline]
    pub fn blocks(&self, req: u32) -> &[u32] {
        self.req_to_blocks.get(&req).map(|v| &v[..]).unwrap_or(&[])
    }

    /// `get_num_skipped_tokens` (`:546` full attention, `:1335` mamba).
    #[inline]
    fn num_skipped_tokens(&self, num_computed_tokens: i64) -> i64 {
        match self.kind {
            Kind::FullAttention => 0,
            Kind::Mamba => num_computed_tokens - 1,
        }
    }

    /// `_get_num_evictable_blocks` (`:99`).
    fn num_evictable(&self, pool: &BlockPool, blocks: &[u32]) -> usize {
        blocks
            .iter()
            .filter(|&&b| {
                let blk = pool.arena.get(b);
                blk.ref_cnt == 0 && !blk.is_null
            })
            .count()
    }

    /// `get_num_blocks_to_allocate` (`:102`, mamba override `:1186`).
    pub fn get_num_blocks_to_allocate(
        &self,
        pool: &BlockPool,
        req: u32,
        num_tokens: usize,
        new_computed_blocks: &[u32],
        total_computed_tokens: usize,
        num_tokens_main_model: usize,
    ) -> usize {
        if self.kind == Kind::Mamba {
            if let Some(&last) = new_computed_blocks.last() {
                if let Some(h) = pool.arena.get(last).hash {
                    if self.cached_blocks_this_step.contains(&h) {
                        // Mamba cannot reuse a state another request cached this step.
                        return pool.num_gpu_blocks + 1;
                    }
                }
            }
            if self.mamba_align {
                return self.mamba_align_blocks_to_allocate(
                    pool,
                    req,
                    num_tokens_main_model,
                    new_computed_blocks,
                );
            }
        }

        let mut num_tokens = num_tokens;
        if self.kind == Kind::Mamba && self.num_speculative_blocks > 0 {
            // Non-align mamba reserves the speculative tail (`:1208`).
            num_tokens += self.block_size * self.num_speculative_blocks;
        }

        let num_required_blocks = cdiv(num_tokens, self.block_size) as i64;
        let num_req_blocks = self.blocks(req).len() as i64;

        if self.num_cached_block.contains_key(&req) {
            debug_assert!(new_computed_blocks.is_empty());
            return (num_required_blocks - num_req_blocks).max(0) as usize;
        }

        let num_skipped_tokens = self.num_skipped_tokens(total_computed_tokens as i64);
        let num_local_computed_blocks = new_computed_blocks.len() as i64 + num_req_blocks;
        let num_skipped_blocks = floordiv(num_skipped_tokens, self.block_size as i64);
        let num_new_blocks =
            (num_required_blocks - num_skipped_blocks.max(num_local_computed_blocks)).max(0);

        let num_skipped_new_computed = (num_skipped_blocks - num_req_blocks).max(0) as usize;
        let tail = if num_skipped_new_computed >= new_computed_blocks.len() {
            &[][..]
        } else {
            &new_computed_blocks[num_skipped_new_computed..]
        };
        num_new_blocks as usize + self.num_evictable(pool, tail)
    }

    /// Mamba `mamba_cache_mode="align"` branch of `get_num_blocks_to_allocate` (`:1220`).
    fn mamba_align_blocks_to_allocate(
        &self,
        pool: &BlockPool,
        req: u32,
        num_tokens_main_model: usize,
        new_computed_blocks: &[u32],
    ) -> usize {
        let num_required_blocks =
            cdiv(num_tokens_main_model, self.block_size) + self.num_speculative_blocks;
        let mut num_new_blocks = num_required_blocks as i64
            - new_computed_blocks.len() as i64
            - self.blocks(req).len() as i64;
        if num_new_blocks > 0 {
            num_new_blocks = if self.allocated_block_reqs.contains(&req) {
                1
            } else {
                1 + self.num_speculative_blocks as i64
            };
        }
        num_new_blocks.max(0) as usize + self.num_evictable(pool, new_computed_blocks)
    }

    /// `add_local_computed_blocks` (`:183`). `num_external_computed_tokens` is always 0
    /// here (connectors are rejected by the config gate).
    pub fn add_local_computed_blocks(
        &mut self,
        pool: &mut BlockPool,
        req: u32,
        new_computed_blocks: &[u32],
        num_local_computed_tokens: usize,
    ) -> Result<(), String> {
        debug_assert!(self.blocks(req).is_empty());
        let num_skipped_tokens = self.num_skipped_tokens(num_local_computed_tokens as i64);
        let num_skipped_blocks = floordiv(num_skipped_tokens, self.block_size as i64).max(0) as usize;
        let kept: &[u32] = if num_skipped_blocks > 0 {
            if num_skipped_blocks >= new_computed_blocks.len() {
                &[]
            } else {
                &new_computed_blocks[num_skipped_blocks..]
            }
        } else {
            new_computed_blocks
        };
        if self.enable_caching {
            pool.touch(kept)?;
        } else {
            debug_assert!(kept.is_empty());
        }
        let null = self.null_block;
        let n = {
            let blocks = self.blocks_mut_for_push(pool, req);
            for _ in 0..num_skipped_blocks {
                blocks.push(null);
            }
            blocks.extend_from_slice(kept);
            blocks.len()
        };
        let (g, old) = (self.group_id, self.num_cached_block.get(&req).copied());
        self.jrn(pool, || Undo::NumCachedBlock(g, req, old));
        self.num_cached_block.insert(req, n);
        Ok(())
    }

    /// `allocate_new_blocks` (`:279`, mamba override `:1253`). Returns the new blocks,
    /// appended to `out`.
    pub fn allocate_new_blocks(
        &mut self,
        pool: &mut BlockPool,
        req: u32,
        num_tokens: usize,
        num_tokens_main_model: usize,
        out: &mut Vec<u32>,
    ) -> Result<(), String> {
        if self.kind == Kind::Mamba && self.mamba_align {
            return self.mamba_align_allocate(pool, req, num_tokens_main_model, out);
        }
        let mut num_tokens = num_tokens;
        if self.kind == Kind::Mamba && self.num_speculative_blocks > 0 {
            num_tokens += self.block_size * self.num_speculative_blocks;
        }
        let num_required_blocks = cdiv(num_tokens, self.block_size);
        let have = self.blocks_mut(pool, req).len();
        if num_required_blocks <= have {
            return Ok(());
        }
        let num_new = num_required_blocks - have;
        let start = out.len();
        pool.get_new_blocks(num_new, out)?;
        let (g, len) = (self.group_id, have);
        self.jrn(pool, || Undo::ReqBlocksLen(g, req, len));
        self.req_to_blocks
            .get_mut(&req)
            .expect("entry created above")
            .extend_from_slice(&out[start..]);
        if self.tracks_new_block_ids {
            let n = self.new_block_ids.len();
            self.jrn(pool, || Undo::NewBlockIdsLen(g, n));
            self.new_block_ids.extend_from_slice(&out[start..]);
        }
        Ok(())
    }

    /// Mamba align-mode allocation (`:1265`). This one rewrites `req_to_blocks` in place:
    /// earlier state blocks are nulled out and the speculative tail is recycled.
    fn mamba_align_allocate(
        &mut self,
        pool: &mut BlockPool,
        req: u32,
        num_tokens_main_model: usize,
        out: &mut Vec<u32>,
    ) -> Result<(), String> {
        let num_spec = self.num_speculative_blocks;
        let num_required_blocks = cdiv(num_tokens_main_model, self.block_size) + num_spec;
        let null = self.null_block;
        let g = self.group_id;
        let blocks_allocated = self.allocated_block_reqs.contains(&req);
        let prev_block_len = self.blocks_mut(pool, req).len();
        if num_required_blocks <= prev_block_len {
            return Ok(());
        }

        if blocks_allocated {
            // The running state always lives at the last (1 + num_speculative) block.
            let old = self.last_state_block_idx.get(&req).copied();
            self.jrn(pool, || Undo::LastStateIdx(g, req, old));
            self.last_state_block_idx
                .insert(req, prev_block_len - 1 - num_spec);
        } else if prev_block_len > 0 {
            // A fresh request that hit the prefix cache: the last block holds the hit state.
            let old = self.last_state_block_idx.get(&req).copied();
            self.jrn(pool, || Undo::LastStateIdx(g, req, old));
            self.last_state_block_idx.insert(req, prev_block_len - 1);
        }

        let num_skipped_blocks = num_required_blocks - num_spec - 1;
        {
            let armed = pool.arena.journal.is_some();
            let mut ops: Vec<Undo> = Vec::new();
            let blocks = self.req_to_blocks.get_mut(&req).expect("entry created above");
            if armed {
                ops.push(Undo::ReqBlocksLen(g, req, blocks.len()));
            }
            if prev_block_len < num_skipped_blocks {
                for _ in prev_block_len..num_skipped_blocks {
                    blocks.push(null);
                }
            }
            if blocks_allocated {
                for block_idx in (prev_block_len - num_spec)..prev_block_len {
                    if block_idx < num_skipped_blocks {
                        let b = blocks[block_idx];
                        blocks.push(b);
                        if armed {
                            ops.push(Undo::ReqBlocksSet(g, req, block_idx, blocks[block_idx]));
                        }
                        blocks[block_idx] = null;
                    } else {
                        break;
                    }
                }
            }
            if let Some(j) = pool.arena.journal.as_mut() {
                for op in ops {
                    j.push(op);
                }
            }
        }
        let cur_len = self.blocks_mut(pool, req).len();
        let num_new_blocks = num_required_blocks - cur_len;
        if blocks_allocated {
            debug_assert!(num_new_blocks <= 1);
        } else {
            debug_assert!(num_new_blocks <= num_spec + 1);
        }
        let mut fresh = std::mem::take(&mut self.scratch);
        fresh.clear();
        pool.get_new_blocks(num_new_blocks, &mut fresh)?;
        self.blocks_mut_for_push(pool, req)
            .extend_from_slice(&fresh);
        self.scratch = fresh;
        let was = self.allocated_block_reqs.contains(&req);
        self.jrn(pool, || Undo::AllocatedReq(g, req, was));
        self.allocated_block_reqs.insert(req);
        // `return req_blocks[prev_block_len:]` — includes the recycled speculative blocks.
        out.extend_from_slice(&self.blocks(req)[prev_block_len..]);
        Ok(())
    }

    /// `cache_blocks` (`:319`, mamba wrapper `:1343`).
    pub fn cache_blocks(
        &mut self,
        pool: &mut BlockPool,
        req: u32,
        hashes: HashView<'_>,
        num_tokens: usize,
    ) {
        let num_cached_blocks = self.num_cached_block.get(&req).copied().unwrap_or(0);
        let num_full_blocks = num_tokens / self.block_size;
        if num_cached_blocks >= num_full_blocks {
            return;
        }
        // `reachable_block_mask` returns None for full attention always, and for mamba
        // whenever retention is dense (the only configuration this port accepts).
        // Lend the Vec out instead of cloning it: no per-step allocation. The take/insert
        // pair is a net no-op on the map entry (contents are read-only in between), so
        // only the entry's possible CREATION needs journaling — `blocks_mut` does that.
        let blocks = std::mem::take(self.blocks_mut(pool, req));
        pool.cache_full_blocks(
            &blocks,
            hashes.hashes,
            hashes.stride,
            num_cached_blocks,
            num_full_blocks,
            self.block_size,
            self.group_id,
            None,
        );
        let (g, old) = (self.group_id, self.num_cached_block.get(&req).copied());
        self.jrn(pool, || Undo::NumCachedBlock(g, req, old));
        self.num_cached_block.insert(req, num_full_blocks);

        if self.kind == Kind::Mamba {
            for &b in &blocks[num_cached_blocks..num_full_blocks] {
                let blk = pool.arena.get(b);
                if blk.is_null {
                    continue;
                }
                if let Some(h) = blk.hash {
                    let was = self.cached_blocks_this_step.contains(&h);
                    self.jrn(pool, || Undo::CachedThisStepInsert(g, h, was));
                    self.cached_blocks_this_step.insert(h);
                }
            }
        }
        self.req_to_blocks.insert(req, blocks);
    }

    /// `_remove_blocks_in_range` (`:480`).
    fn remove_blocks_in_range(
        &mut self,
        pool: &mut BlockPool,
        req: u32,
        first_block: usize,
        last_block: usize,
    ) {
        if !self.req_to_blocks.contains_key(&req) || first_block >= last_block {
            return;
        }
        let null = self.null_block;
        let g = self.group_id;
        let mut freed = std::mem::take(&mut self.scratch);
        freed.clear();
        {
            let armed = pool.arena.journal.is_some();
            let mut ops: Vec<Undo> = Vec::new();
            let blocks = self.req_to_blocks.get_mut(&req).unwrap();
            let last_block = last_block.min(blocks.len());
            let mut i = last_block;
            while i > first_block {
                i -= 1;
                if blocks[i] == null {
                    break;
                }
                freed.push(blocks[i]);
                if armed {
                    ops.push(Undo::ReqBlocksSet(g, req, i, blocks[i]));
                }
                blocks[i] = null;
            }
            if let Some(j) = pool.arena.journal.as_mut() {
                for op in ops {
                    j.push(op);
                }
            }
        }
        if !freed.is_empty() {
            pool.free_blocks(&freed);
        }
        self.scratch = freed;
    }

    /// `remove_skipped_blocks` (`:507`, mamba override `:1143`).
    pub fn remove_skipped_blocks(
        &mut self,
        pool: &mut BlockPool,
        req: u32,
        total_computed_tokens: usize,
    ) {
        let mut computed = total_computed_tokens as i64;
        if self.kind == Kind::Mamba {
            // Assume every draft token is rejected so we never free a needed state.
            computed = (computed - self.num_speculative_blocks as i64).max(0);
        }
        let g = self.group_id;
        let num_skipped_tokens = self.num_skipped_tokens(computed);
        if num_skipped_tokens > 0 {
            let mut num_skipped_blocks =
                floordiv(num_skipped_tokens, self.block_size as i64) as usize;
            num_skipped_blocks = num_skipped_blocks.min(self.blocks_mut(pool, req).len());
            self.remove_blocks_in_range(pool, req, 0, num_skipped_blocks);
        }
        if self.kind == Kind::Mamba && self.mamba_align {
            let last_state = self.last_state_block_idx.get(&req).copied();
            let limit = cdiv(computed as usize, self.block_size) as i64 - 1;
            if let Some(idx) = last_state {
                if (idx as i64) < limit {
                    let null = self.null_block;
                    let hit = match self.req_to_blocks.get(&req) {
                        Some(blocks) if idx < blocks.len() && blocks[idx] != null => {
                            Some(blocks[idx])
                        }
                        _ => None,
                    };
                    if let Some(b) = hit {
                        self.jrn(pool, || Undo::ReqBlocksSet(g, req, idx, b));
                        self.req_to_blocks.get_mut(&req).unwrap()[idx] = null;
                        pool.free_blocks(&[b]);
                    }
                }
            }
        }
    }

    /// `pop_blocks_for_free` (`:384`, mamba override `:1329`).
    pub fn pop_blocks_for_free(&mut self, pool: &mut BlockPool, req: u32, out: &mut Vec<u32>) {
        let g = self.group_id;
        if self.kind == Kind::Mamba && self.mamba_align {
            let was = self.allocated_block_reqs.contains(&req);
            self.jrn(pool, || Undo::AllocatedReq(g, req, was));
            self.allocated_block_reqs.remove(&req);
            let old = self.last_state_block_idx.get(&req).copied();
            self.jrn(pool, || Undo::LastStateIdx(g, req, old));
            self.last_state_block_idx.remove(&req);
        }
        if pool.arena.journal.is_some() {
            if let Some(v) = self.req_to_blocks.get(&req) {
                let v = v.clone();
                self.jrn(pool, || Undo::ReqBlocksRemove(g, req, v));
            }
        }
        if let Some(v) = self.req_to_blocks.remove(&req) {
            out.extend_from_slice(&v);
        }
        let old = self.num_cached_block.get(&req).copied();
        self.jrn(pool, || Undo::NumCachedBlock(g, req, old));
        self.num_cached_block.remove(&req);
    }

    /// `free` (`:402`) — reverse order so tail blocks are evicted first.
    pub fn free(&mut self, pool: &mut BlockPool, req: u32) {
        let mut v = std::mem::take(&mut self.scratch);
        v.clear();
        self.pop_blocks_for_free(pool, req, &mut v);
        v.reverse();
        pool.free_blocks(&v);
        self.scratch = v;
    }

    /// `get_num_common_prefix_blocks` (`:614` full attention, `:1180` mamba -> always 0).
    pub fn get_num_common_prefix_blocks(&mut self, pool: &mut BlockPool, req: u32) -> usize {
        if self.kind == Kind::Mamba {
            return 0; // cascade attention is not supported by mamba
        }
        let num_reqs = {
            self.blocks_mut(pool, req);
            self.req_to_blocks.len() as i32
        };
        let mut n = 0;
        for &b in self.blocks(req) {
            if pool.arena.get(b).ref_cnt == num_reqs {
                n += 1;
            } else {
                break;
            }
        }
        n
    }

    pub fn take_new_block_ids(&mut self, out: &mut Vec<u32>) {
        out.extend_from_slice(&self.new_block_ids);
        self.new_block_ids.clear();
    }

    pub fn new_step_starts(&mut self, pool: &mut BlockPool) {
        if self.kind == Kind::Mamba {
            if pool.arena.journal.is_some() && !self.cached_blocks_this_step.is_empty() {
                let (g, set) = (self.group_id, self.cached_blocks_this_step.clone());
                self.jrn(pool, || Undo::CachedThisStep(g, set));
            }
            self.cached_blocks_this_step.clear();
        }
    }

    /// `FullAttentionManager.find_longest_cache_hit` (`:566`) and
    /// `MambaManager.find_longest_cache_hit` (`:1042`).
    ///
    /// Appends the hit blocks for each group in `group_ids` to `out[i]`.
    #[allow(clippy::too_many_arguments)]
    pub fn find_longest_cache_hit(
        kind: Kind,
        pool: &BlockPool,
        hashes: HashView<'_>,
        max_length: usize,
        group_ids: &[u32],
        block_size: usize,
        drop_eagle_block: bool,
        alignment_tokens: usize,
        out: &mut [Vec<u32>],
    ) {
        debug_assert_eq!(out.len(), group_ids.len());
        for o in out.iter_mut() {
            o.clear();
        }
        let max_num_blocks = max_length / block_size;
        let mut cached: SmallVec<[u32; 4]> = SmallVec::new();

        match kind {
            Kind::FullAttention => {
                let n = max_num_blocks.min(hashes.len());
                for i in 0..n {
                    if pool.get_cached_block(hashes.get(i), group_ids, &mut cached) {
                        for (o, c) in out.iter_mut().zip(cached.iter()) {
                            o.push(*c);
                        }
                    } else {
                        break;
                    }
                }
                if drop_eagle_block && !out[0].is_empty() {
                    for o in out.iter_mut() {
                        o.pop();
                    }
                }
                while block_size != alignment_tokens
                    && !out[0].is_empty()
                    && (out[0].len() * block_size) % alignment_tokens != 0
                {
                    for o in out.iter_mut() {
                        o.pop();
                    }
                }
            }
            Kind::Mamba => {
                // Right-to-left: only the LAST matching state block is useful.
                let n = max_num_blocks.min(hashes.len());
                for i in (0..n).rev() {
                    if pool.get_cached_block(hashes.get(i), group_ids, &mut cached) {
                        if block_size != alignment_tokens
                            && ((i + 1) * block_size) % alignment_tokens != 0
                        {
                            continue;
                        }
                        for (o, c) in out.iter_mut().zip(cached.iter()) {
                            // The hit length is read as len(blocks) * block_size, so the
                            // skipped prefix is padded with null blocks.
                            for _ in 0..i {
                                o.push(pool.null_block);
                            }
                            o.push(*c);
                        }
                        break;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::HASH_LEN;

    fn h(n: u8) -> Digest32 {
        [n; HASH_LEN]
    }

    fn view(v: &[Digest32]) -> HashView<'_> {
        HashView {
            hashes: v,
            stride: 1,
        }
    }

    fn seed_cache(pool: &mut BlockPool, hashes: &[Digest32], group: u32) -> Vec<u32> {
        let mut ids = Vec::new();
        pool.get_new_blocks(hashes.len(), &mut ids).unwrap();
        for i in 0..hashes.len() {
            pool.cache_full_blocks(&ids, &hashes[..i + 1], 1, i, i + 1, 16, group, None);
        }
        ids
    }

    #[test]
    fn full_attention_hit_stops_at_first_miss() {
        let mut pool = BlockPool::new(64, true, 16, false).unwrap();
        let chain = [h(1), h(2), h(3)];
        // Cache only the first two links.
        let ids = seed_cache(&mut pool, &chain[..2], 0);
        let mut out = vec![Vec::new()];
        TypeManager::find_longest_cache_hit(
            Kind::FullAttention,
            &pool,
            view(&chain),
            3 * 16,
            &[0],
            16,
            false,
            16,
            &mut out,
        );
        assert_eq!(out[0], ids);
    }

    #[test]
    fn full_attention_respects_max_length_and_alignment() {
        let mut pool = BlockPool::new(64, true, 16, false).unwrap();
        let chain = [h(1), h(2), h(3)];
        seed_cache(&mut pool, &chain, 0);
        let mut out = vec![Vec::new()];
        TypeManager::find_longest_cache_hit(
            Kind::FullAttention,
            &pool,
            view(&chain),
            2 * 16,
            &[0],
            16,
            false,
            16,
            &mut out,
        );
        assert_eq!(out[0].len(), 2, "max_length caps the hit");

        // alignment_tokens = 32 with block_size 16 must round the hit down to 2 blocks.
        TypeManager::find_longest_cache_hit(
            Kind::FullAttention,
            &pool,
            view(&chain),
            3 * 16,
            &[0],
            16,
            false,
            32,
            &mut out,
        );
        assert_eq!(out[0].len(), 2);
    }

    /// The hybrid trap: mamba returns ONE real block padded with nulls, not a run.
    #[test]
    fn mamba_returns_only_the_last_state_padded_with_nulls() {
        let mut pool = BlockPool::new(64, true, 16, false).unwrap();
        let chain = [h(1), h(2), h(3)];
        let ids = seed_cache(&mut pool, &chain, 0);
        let mut out = vec![Vec::new()];
        TypeManager::find_longest_cache_hit(
            Kind::Mamba,
            &pool,
            view(&chain),
            3 * 16,
            &[0],
            16,
            false,
            16,
            &mut out,
        );
        assert_eq!(out[0].len(), 3, "hit length must still read as 3 blocks");
        assert_eq!(out[0][0], pool.null_block);
        assert_eq!(out[0][1], pool.null_block);
        assert_eq!(out[0][2], ids[2], "only the newest state block is real");
    }

    #[test]
    fn mamba_skips_unaligned_boundaries() {
        let mut pool = BlockPool::new(64, true, 16, false).unwrap();
        let chain = [h(1), h(2), h(3)];
        // Only the 3rd link (token 48, not a multiple of 32) and the 2nd (token 32) cached.
        let mut ids = Vec::new();
        pool.get_new_blocks(3, &mut ids).unwrap();
        pool.cache_full_blocks(&ids, &chain, 1, 0, 3, 16, 0, None);
        let mut out = vec![Vec::new()];
        TypeManager::find_longest_cache_hit(
            Kind::Mamba,
            &pool,
            view(&chain),
            3 * 16,
            &[0],
            16,
            false,
            32, // scheduler_block_size = lcm(16, 32)
            &mut out,
        );
        // i=2 -> (2+1)*16 = 48, 48 % 32 != 0 -> skipped; i=1 -> 32 % 32 == 0 -> hit.
        assert_eq!(out[0].len(), 2);
        assert_eq!(out[0][1], ids[1]);
    }

    #[test]
    fn mamba_get_num_skipped_tokens_keeps_only_last_state() {
        let mut pool = BlockPool::new(64, true, 16, false).unwrap();
        let mut m = TypeManager::new(Kind::Mamba, 16, 1, 16, true, true, 0, pool.null_block, 512);
        let mut out = Vec::new();
        // 64 tokens of main-model work -> align mode allocates 1 state block + nulls.
        m.allocate_new_blocks(&mut pool, 7, 64, 64, &mut out).unwrap();
        assert_eq!(m.blocks(7).len(), 4);
        assert_eq!(&m.blocks(7)[..3], &[pool.null_block; 3]);
        assert_ne!(m.blocks(7)[3], pool.null_block);
    }

    #[test]
    fn full_attention_common_prefix_counts_shared_refs() {
        let mut pool = BlockPool::new(64, true, 16, false).unwrap();
        let mut m =
            TypeManager::new(Kind::FullAttention, 16, 0, 16, true, false, 0, pool.null_block, 512);
        let mut out = Vec::new();
        m.allocate_new_blocks(&mut pool, 1, 32, 32, &mut out).unwrap();
        let shared = m.blocks(1).to_vec();
        // A second request sharing the first block only.
        pool.touch(&shared[..1]);
        m.req_to_blocks.insert(2, vec![shared[0]]);
        assert_eq!(m.get_num_common_prefix_blocks(&mut pool, 1), 1);
    }
}
