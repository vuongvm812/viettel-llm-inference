//! Port of `vllm/v1/core/kv_cache_coordinator.py`.
//!
//! Mirrors:
//!   * `:61`  `KVCacheCoordinator`             -> [`Coordinator`] (shared fan-out over managers)
//!   * `:130` `get_num_blocks_to_allocate`     -> [`Coordinator::get_num_blocks_to_allocate`]
//!   * `:187` `allocate_new_computed_blocks`   -> [`Coordinator::allocate_new_computed_blocks`]
//!   * `:233` `allocate_new_blocks`            -> [`Coordinator::allocate_new_blocks`]
//!   * `:427` `UnitaryKVCacheCoordinator`      -> the `!is_hybrid` path
//!   * `:514` `HybridKVCacheCoordinator`       -> [`Coordinator::find_longest_cache_hit`]
//!   * `:560` `verify_and_split_kv_cache_groups` -> [`Coordinator::split_attention_groups`]
//!   * `:602` `HybridKVCacheCoordinator.cache_blocks` -> [`Coordinator::cache_blocks`]
//!
//! Group ORDER is load-bearing: every per-group output is a tuple indexed by
//! `kv_cache_group_id`, and `attention_groups` is sorted full-attention-first (`:592`)
//! because full attention's left-to-right scan bounds the other groups.
//!
//! `KVCacheCoordinatorNoPrefixCache` (`:377`) is covered by the `enable_caching == false`
//! early-out in [`Coordinator::find_longest_cache_hit`] plus
//! [`Coordinator::get_num_common_prefix_blocks`]'s zero fill.

use crate::block_pool::BlockPool;
use crate::hash::Digest32;
use crate::single_type::{HashView, Kind, TypeManager};

/// `SpecGroup` (kv_cache_coordinator.py:499): KV cache groups sharing one spec, looked
/// up jointly.
pub struct SpecGroup {
    pub kind: Kind,
    pub block_size: usize,
    pub is_full_attention: bool,
    pub group_ids: Vec<u32>,
    pub use_eagle: bool,
}

/// `(kind, block_size, is_full_attention, spec_signature, use_eagle)` per KV cache group.
/// The signature is the Python side's canonical rendering of the spec dataclass, so
/// grouping here is exactly upstream's `group.spec == spec`.
pub type SpecInput = (Kind, usize, bool, String, bool);

pub struct Coordinator {
    pub pool: BlockPool,
    pub managers: Vec<TypeManager>,
    pub attention_groups: Vec<SpecGroup>,
    pub scheduler_block_size: usize,
    pub hash_block_size: usize,
    pub enable_caching: bool,
    pub is_hybrid: bool,
    /// Marconi-style hint consumed by `_mamba_block_aligned_split` (scheduler.py:730).
    pub num_uncached_common_prefix_tokens: usize,

    /// Per-group hit buffers and loop state, all reused across calls.
    hit_blocks: Vec<Vec<u32>>,
    scratch_group: Vec<Vec<u32>>,
    seen: Vec<bool>,
    eagle_verified: Vec<bool>,
}

#[inline]
fn hash_view(hashes: &[Digest32], block_size: usize, hash_block_size: usize) -> HashView<'_> {
    HashView {
        hashes,
        stride: block_size / hash_block_size,
    }
}

impl Coordinator {
    pub fn new(
        pool: BlockPool,
        managers: Vec<TypeManager>,
        specs: Vec<SpecInput>,
        scheduler_block_size: usize,
        hash_block_size: usize,
        enable_caching: bool,
    ) -> Self {
        let n = managers.len();
        let mut c = Coordinator {
            pool,
            managers,
            attention_groups: Vec::new(),
            scheduler_block_size,
            hash_block_size,
            enable_caching,
            is_hybrid: n > 1,
            num_uncached_common_prefix_tokens: 0,
            hit_blocks: (0..n).map(|_| Vec::with_capacity(64)).collect(),
            scratch_group: (0..n).map(|_| Vec::with_capacity(64)).collect(),
            seen: vec![false; n],
            eagle_verified: Vec::new(),
        };
        c.split_attention_groups(&specs);
        c.eagle_verified = vec![false; c.attention_groups.len()];
        c
    }

    /// `verify_and_split_kv_cache_groups` (`:560`).
    fn split_attention_groups(&mut self, specs: &[SpecInput]) {
        let mut sigs: Vec<String> = Vec::new();
        for (i, (kind, block_size, is_full, sig, use_eagle)) in specs.iter().enumerate() {
            if let Some(pos) = sigs.iter().position(|s| s == sig) {
                self.attention_groups[pos].group_ids.push(i as u32);
                if *use_eagle {
                    self.attention_groups[pos].use_eagle = true;
                }
            } else {
                sigs.push(sig.clone());
                self.attention_groups.push(SpecGroup {
                    kind: *kind,
                    block_size: *block_size,
                    is_full_attention: *is_full,
                    group_ids: vec![i as u32],
                    use_eagle: *use_eagle,
                });
            }
        }
        // Stable sort, full attention first (`:592`).
        self.attention_groups.sort_by_key(|g| !g.is_full_attention);
        for g in &self.attention_groups {
            if g.use_eagle {
                for &gid in &g.group_ids {
                    self.managers[gid as usize].use_eagle = true;
                }
            }
        }
    }

    /// One lookup pass for `attention_groups[idx]`; returns the resulting hit length.
    fn lookup_group(
        &mut self,
        idx: usize,
        hashes: &[Digest32],
        max_length: usize,
        drop_eagle_block: bool,
        alignment_tokens: usize,
    ) -> usize {
        let g = &self.attention_groups[idx];
        let view = hash_view(hashes, g.block_size, self.hash_block_size);
        let mut out = std::mem::take(&mut self.scratch_group);
        TypeManager::find_longest_cache_hit(
            g.kind,
            &self.pool,
            view,
            max_length,
            &g.group_ids,
            g.block_size,
            drop_eagle_block,
            alignment_tokens,
            &mut out[..g.group_ids.len()],
        );
        let hit = out[0].len() * g.block_size;
        for (i, gid) in g.group_ids.iter().enumerate() {
            std::mem::swap(&mut self.hit_blocks[*gid as usize], &mut out[i]);
            self.seen[*gid as usize] = true;
        }
        self.scratch_group = out;
        hit
    }

    /// `find_longest_cache_hit` — unitary (`:480`) or hybrid fixed point (`:630`).
    /// Returns the hit length in tokens; per-group blocks land in [`Self::hit_blocks`].
    pub fn find_longest_cache_hit(
        &mut self,
        hashes: &[Digest32],
        max_cache_hit_length: usize,
    ) -> usize {
        for b in self.hit_blocks.iter_mut() {
            b.clear();
        }
        self.seen.iter_mut().for_each(|s| *s = false);
        self.eagle_verified.iter_mut().for_each(|s| *s = false);
        if !self.enable_caching {
            return 0;
        }

        if !self.is_hybrid {
            let (drop_eagle, block_size) = {
                let g = &self.attention_groups[0];
                (g.use_eagle, g.block_size)
            };
            // Unitary: alignment_tokens == the group's own block size (`:492`).
            return self.lookup_group(0, hashes, max_cache_hit_length, drop_eagle, block_size);
        }

        let mut hit_length = max_cache_hit_length;
        let mut longest_hit_length = 0usize;
        let is_simple_hybrid =
            self.attention_groups.len() == 2 && self.attention_groups[0].is_full_attention;
        let alignment = self.scheduler_block_size;

        loop {
            let mut curr_hit_length = hit_length;
            for idx in 0..self.attention_groups.len() {
                let (block_size, is_full, use_eagle, first_gid) = {
                    let g = &self.attention_groups[idx];
                    (g.block_size, g.is_full_attention, g.use_eagle, g.group_ids[0] as usize)
                };
                if is_full && self.seen[first_gid] {
                    // Full attention is downward-closed: trim, do not re-look-up.
                    curr_hit_length = curr_hit_length / block_size * block_size;
                    continue;
                }
                let drop_eagle_block = use_eagle && !self.eagle_verified[idx];
                let max_length = if drop_eagle_block {
                    (curr_hit_length + block_size).min(max_cache_hit_length)
                } else {
                    curr_hit_length
                };
                let new_hit_length =
                    self.lookup_group(idx, hashes, max_length, drop_eagle_block, alignment);
                if drop_eagle_block {
                    self.eagle_verified[idx] = true;
                } else if new_hit_length < curr_hit_length {
                    self.eagle_verified.iter_mut().for_each(|v| *v = false);
                }
                curr_hit_length = new_hit_length;
                longest_hit_length = longest_hit_length.max(curr_hit_length);
            }
            if curr_hit_length >= hit_length {
                break;
            }
            hit_length = curr_hit_length;
            if is_simple_hybrid {
                break;
            }
        }

        // Truncate full-attention blocks to the final hit length (`:727`).
        if self.attention_groups[0].is_full_attention {
            let bs = self.attention_groups[0].block_size;
            let num_blocks = hit_length / bs;
            for i in 0..self.attention_groups[0].group_ids.len() {
                let gid = self.attention_groups[0].group_ids[i] as usize;
                if self.hit_blocks[gid].len() > num_blocks {
                    self.hit_blocks[gid].truncate(num_blocks);
                }
            }
        }
        self.num_uncached_common_prefix_tokens = longest_hit_length.saturating_sub(hit_length);
        hit_length
    }

    pub fn hit_blocks(&self, group: usize) -> &[u32] {
        &self.hit_blocks[group]
    }

    /// Move the pending hit into a caller-owned buffer (so `find_longest_cache_hit` can
    /// run again for the next request without losing it).
    pub fn take_hit_blocks(&mut self, out: &mut [Vec<u32>]) {
        for (i, o) in out.iter_mut().enumerate() {
            o.clear();
            o.extend_from_slice(&self.hit_blocks[i]);
        }
    }

    /// `get_num_blocks_to_allocate` (`:130`).
    pub fn get_num_blocks_to_allocate(
        &self,
        req: u32,
        num_tokens: usize,
        new_computed_blocks: &[Vec<u32>],
        total_computed_tokens: usize,
        num_tokens_main_model: usize,
    ) -> usize {
        let mut total = 0usize;
        for (i, m) in self.managers.iter().enumerate() {
            total += m.get_num_blocks_to_allocate(
                &self.pool,
                req,
                num_tokens,
                &new_computed_blocks[i],
                total_computed_tokens,
                num_tokens_main_model,
            );
        }
        total
    }

    /// `allocate_new_computed_blocks` (`:187`). The two-phase external-block dance
    /// (`:225`) is absent: connectors are rejected by the config gate.
    pub fn allocate_new_computed_blocks(
        &mut self,
        req: u32,
        new_computed_blocks: &[Vec<u32>],
        num_local_computed_tokens: usize,
    ) -> Result<(), String> {
        if self
            .managers
            .iter()
            .any(|m| m.num_cached_block.contains_key(&req))
        {
            debug_assert!(new_computed_blocks.iter().all(|b| b.is_empty()));
            return Ok(());
        }
        for i in 0..self.managers.len() {
            self.managers[i].add_local_computed_blocks(
                &mut self.pool,
                req,
                &new_computed_blocks[i],
                num_local_computed_tokens,
            )?;
        }
        Ok(())
    }

    /// `allocate_new_blocks` (`:233`). Per-group new blocks land in `out`.
    pub fn allocate_new_blocks(
        &mut self,
        req: u32,
        num_tokens: usize,
        num_tokens_main_model: usize,
        out: &mut [Vec<u32>],
    ) -> Result<(), String> {
        for i in 0..self.managers.len() {
            out[i].clear();
            self.managers[i].allocate_new_blocks(
                &mut self.pool,
                req,
                num_tokens,
                num_tokens_main_model,
                &mut out[i],
            )?;
        }
        Ok(())
    }

    /// `cache_blocks` — base (`:268`) or hybrid alignment-aware (`:602`).
    pub fn cache_blocks(&mut self, req: u32, hashes: &[Digest32], num_computed_tokens: usize) {
        let aligned = if self.is_hybrid {
            num_computed_tokens / self.scheduler_block_size * self.scheduler_block_size
        } else {
            num_computed_tokens
        };
        for i in 0..self.managers.len() {
            let bs = self.managers[i].block_size;
            let mut num_tokens_to_cache = aligned;
            if self.is_hybrid && self.managers[i].use_eagle && aligned > 0 {
                num_tokens_to_cache = num_computed_tokens.min(aligned + bs);
            }
            let view = hash_view(hashes, bs, self.hash_block_size);
            self.managers[i].cache_blocks(&mut self.pool, req, view, num_tokens_to_cache);
        }
    }

    pub fn free(&mut self, req: u32) {
        for i in 0..self.managers.len() {
            self.managers[i].free(&mut self.pool, req);
        }
    }

    pub fn pop_blocks_for_free(&mut self, req: u32, out: &mut Vec<u32>) {
        let pool = &mut self.pool;
        for m in self.managers.iter_mut() {
            m.pop_blocks_for_free(pool, req, out);
        }
    }

    pub fn remove_skipped_blocks(&mut self, req: u32, total_computed_tokens: usize) {
        for i in 0..self.managers.len() {
            self.managers[i].remove_skipped_blocks(&mut self.pool, req, total_computed_tokens);
        }
    }

    pub fn get_num_common_prefix_blocks(&mut self, req: u32, out: &mut Vec<usize>) {
        out.clear();
        if !self.enable_caching {
            out.extend(std::iter::repeat(0).take(self.managers.len()));
            return;
        }
        let pool = &mut self.pool;
        for m in self.managers.iter_mut() {
            let n = m.get_num_common_prefix_blocks(pool, req);
            out.push(n);
        }
    }

    pub fn new_step_starts(&mut self) {
        let pool = &mut self.pool;
        for m in self.managers.iter_mut() {
            m.new_step_starts(pool);
        }
    }

    pub fn take_new_block_ids(&mut self, out: &mut Vec<u32>) {
        for m in self.managers.iter_mut() {
            m.take_new_block_ids(out);
        }
    }
}
