//! Port of `vllm/v1/core/kv_cache_manager.py` — the surface `sched/scheduler.py` consumes.
//!
//! Mirrors:
//!   * `:114` `KVCacheManager.__init__`      -> [`Manager::new`]
//!   * `:185` `usage`                        -> [`Manager::usage`]
//!   * `:194` `make_prefix_cache_stats`      -> [`Manager::take_prefix_cache_stats`]
//!   * `:206` `get_computed_blocks`          -> [`Manager::get_computed_blocks`]
//!   * `:248` `allocate_slots`               -> [`Manager::allocate_slots`]
//!   * `:466` `free`                         -> [`Manager::free`]
//!   * `:476` `remove_skipped_blocks`        -> [`Manager::remove_skipped_blocks`]
//!   * `:495` `pop_blocks_for_free`          -> [`Manager::pop_blocks_for_free`]
//!   * `:508` `evict_blocks`                 -> [`Manager::evict_blocks`]
//!   * `:516` `reset_prefix_cache`           -> [`Manager::reset_prefix_cache`]
//!   * `:532` `get_num_common_prefix_blocks` -> [`Manager::get_num_common_prefix_blocks`]
//!   * `:592` `get_blocks` / `:596` `get_block_ids` / `:600` `get_block_ids_for_computed_tokens`
//!   * `:620` `cache_blocks`                 -> [`Manager::cache_blocks`]
//!   * `:637` `take_new_block_ids` / `:644` `new_step_starts`
//!
//! `take_events` is not ported: `enable_kv_cache_events` is rejected at construction and
//! the Python wrapper returns `[]` for it.

use rustc_hash::FxHashMap;

use crate::block_pool::BlockPool;
use crate::config::Config;
use crate::coordinator::Coordinator;
use crate::hash::Digest32;
use crate::single_type::{cdiv, Kind, TypeManager};

/// Request status values the allocator's watermark branch cares about
/// (`RequestStatus`, vllm/v1/request.py).
pub const STATUS_WAITING: u8 = 0;
pub const STATUS_RUNNING: u8 = 1;
pub const STATUS_PREEMPTED: u8 = 2;

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixCacheStats {
    pub requests: u64,
    pub queries: u64,
    pub hits: u64,
    pub preempted_requests: u64,
    pub preempted_queries: u64,
    pub preempted_hits: u64,
    pub reset: bool,
}

impl PrefixCacheStats {
    fn record(&mut self, num_tokens: u64, num_hits: u64, preempted: bool) {
        if preempted {
            self.preempted_requests += 1;
            self.preempted_queries += num_tokens;
            self.preempted_hits += num_hits;
        } else {
            self.requests += 1;
            self.queries += num_tokens;
            self.hits += num_hits;
        }
    }
}

#[derive(Default)]
pub struct ReqState {
    /// `Request.block_hashes` at `hash_block_size` granularity, pushed in from Python.
    pub hashes: Vec<Digest32>,
    pub num_prompt_tokens: usize,
}

pub struct Manager {
    pub coord: Coordinator,
    pub cfg: Config,
    pub watermark_blocks: usize,

    /// Request-id interning. Python request ids are strings; the hot path uses u32.
    ids: FxHashMap<String, u32>,
    names: Vec<String>,
    free_slots: Vec<u32>,
    pub reqs: FxHashMap<u32, ReqState>,

    stats: PrefixCacheStats,

    /// Reusable per-group buffers (allocated once at construction).
    pub pending_hit: Vec<Vec<u32>>,
    pub new_blocks: Vec<Vec<u32>>,
    pub empty_groups: Vec<Vec<u32>>,
    pub scratch_flat: Vec<u32>,
    pub common_prefix: Vec<usize>,
    /// Last `get_computed_blocks` result, per request slot.
    hit_len: FxHashMap<u32, usize>,
}

impl Manager {
    pub fn new(cfg: Config) -> Result<Self, String> {
        let pool = BlockPool::new(
            cfg.num_blocks,
            cfg.enable_caching,
            cfg.hash_block_size,
            cfg.radix,
        )?;
        let null = pool.null_block;
        let n = cfg.groups.len();
        let managers: Vec<TypeManager> = cfg
            .groups
            .iter()
            .enumerate()
            .map(|(i, g)| {
                TypeManager::new(
                    g.kind,
                    g.block_size,
                    i as u32,
                    cfg.scheduler_block_size,
                    cfg.enable_caching,
                    g.mamba_align,
                    g.num_speculative_blocks,
                    null,
                    cfg.max_model_len,
                )
            })
            .collect();
        let specs = cfg
            .groups
            .iter()
            .map(|g| {
                (
                    g.kind,
                    g.block_size,
                    g.is_full_attention,
                    g.spec_signature.clone(),
                    g.use_eagle,
                )
            })
            .collect();
        let coord = Coordinator::new(
            pool,
            managers,
            specs,
            cfg.scheduler_block_size,
            cfg.hash_block_size,
            cfg.enable_caching,
        );
        let watermark_blocks = (cfg.watermark * cfg.num_blocks as f64) as usize;
        Ok(Manager {
            coord,
            watermark_blocks,
            ids: FxHashMap::default(),
            names: Vec::new(),
            free_slots: Vec::new(),
            reqs: FxHashMap::default(),
            stats: PrefixCacheStats::default(),
            pending_hit: (0..n).map(|_| Vec::with_capacity(64)).collect(),
            new_blocks: (0..n).map(|_| Vec::with_capacity(64)).collect(),
            empty_groups: (0..n).map(|_| Vec::new()).collect(),
            scratch_flat: Vec::with_capacity(256),
            common_prefix: Vec::with_capacity(n),
            hit_len: FxHashMap::default(),
            cfg,
        })
    }

    // ---- request registry -------------------------------------------------

    pub fn intern(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.ids.get(name) {
            return id;
        }
        let id = match self.free_slots.pop() {
            Some(slot) => {
                self.names[slot as usize] = name.to_string();
                slot
            }
            None => {
                self.names.push(name.to_string());
                (self.names.len() - 1) as u32
            }
        };
        self.ids.insert(name.to_string(), id);
        self.reqs.entry(id).or_default();
        id
    }

    pub fn lookup(&self, name: &str) -> Option<u32> {
        self.ids.get(name).copied()
    }

    pub fn forget(&mut self, name: &str) {
        if let Some(id) = self.ids.remove(name) {
            if let Some(st) = self.reqs.get_mut(&id) {
                st.hashes.clear();
                st.num_prompt_tokens = 0;
            }
            self.hit_len.remove(&id);
            self.free_slots.push(id);
        }
    }

    /// Append newly-computed block hashes for a request (Python owns the hasher).
    pub fn push_hashes(&mut self, req: u32, packed: &[u8], num_prompt_tokens: usize) {
        let st = self.reqs.entry(req).or_default();
        st.num_prompt_tokens = num_prompt_tokens;
        let n = packed.len() / 32;
        st.hashes.reserve(n);
        for i in 0..n {
            let mut d: Digest32 = [0; 32];
            d.copy_from_slice(&packed[i * 32..(i + 1) * 32]);
            st.hashes.push(d);
        }
    }

    pub fn num_hashes(&self, req: u32) -> usize {
        self.reqs.get(&req).map(|s| s.hashes.len()).unwrap_or(0)
    }

    // ---- KVCacheManager surface -------------------------------------------

    pub fn usage(&self) -> f64 {
        self.coord.pool.get_usage()
    }

    pub fn num_free_blocks(&self) -> usize {
        self.coord.pool.get_num_free_blocks()
    }

    /// `make_prefix_cache_stats` (`:194`).
    pub fn take_prefix_cache_stats(&mut self) -> Option<PrefixCacheStats> {
        if !self.cfg.log_stats {
            return None;
        }
        Some(std::mem::take(&mut self.stats))
    }

    /// `get_computed_blocks` (`:206`). Returns `num_new_computed_tokens`; the blocks stay
    /// in `pending_hit` (per group) until `allocate_slots` consumes them.
    pub fn get_computed_blocks(
        &mut self,
        req: u32,
        num_tokens: usize,
        num_preemptions: u32,
        skip_reading_prefix_cache: bool,
    ) -> usize {
        for b in self.pending_hit.iter_mut() {
            b.clear();
        }
        if !self.cfg.enable_caching || skip_reading_prefix_cache {
            self.hit_len.insert(req, 0);
            return 0;
        }
        // All tokens hitting the cache would leave nothing to produce logits from,
        // hence `num_tokens - 1`.
        let max_cache_hit_length = num_tokens.saturating_sub(1);
        let hashes = std::mem::take(&mut self.reqs.entry(req).or_default().hashes);
        let hit = self
            .coord
            .find_longest_cache_hit(&hashes, max_cache_hit_length);
        self.reqs.get_mut(&req).unwrap().hashes = hashes;
        self.coord.take_hit_blocks(&mut self.pending_hit);
        if self.cfg.log_stats {
            self.stats
                .record(num_tokens as u64, hit as u64, num_preemptions > 0);
        }
        self.hit_len.insert(req, hit);
        hit
    }

    /// Cache-hit walk that does NOT touch `prefix_cache_stats` — the read-only signal
    /// `vtl/patches/kv_cache_manager.py::plan_request` needs.
    pub fn peek_cache_hit(&mut self, req: u32, num_tokens: usize) -> usize {
        if !self.cfg.enable_caching {
            return 0;
        }
        let hashes = std::mem::take(&mut self.reqs.entry(req).or_default().hashes);
        let hit = self
            .coord
            .find_longest_cache_hit(&hashes, num_tokens.saturating_sub(1));
        self.reqs.get_mut(&req).unwrap().hashes = hashes;
        hit
    }

    /// `allocate_slots` (`:248`). Returns `Some(())` when the request fits (new blocks
    /// land in `self.new_blocks`), `None` when it does not.
    ///
    /// Arguments narrowed to what the served configuration can produce: no external
    /// (connector) tokens, no encoder tokens, no `delay_cache_blocks`, no
    /// `full_sequence_must_fit`, no `reserved_blocks`. The config gate rejects the
    /// features that would set them.
    #[allow(clippy::too_many_arguments)]
    pub fn allocate_slots(
        &mut self,
        req: u32,
        num_new_tokens: usize,
        num_new_computed_tokens: usize,
        use_pending_hit: bool,
        num_lookahead_tokens: usize,
        num_computed_tokens: usize,
        num_request_tokens: usize,
        status: u8,
        has_scheduled_reqs: bool,
    ) -> Result<bool, String> {
        if num_new_tokens == 0 {
            return Err("num_new_tokens must be greater than 0".into());
        }
        // kv_cache_manager.py:428 gates `allocate_new_computed_blocks` on
        // `new_computed_block_list is not self.empty_kv_cache_blocks.blocks`. That
        // singleton is returned by `create_kv_cache_blocks` exactly when every group's
        // hit is empty, so the identity test is equivalent to this emptiness test —
        // and getting it wrong would set `num_cached_block` for a zero-hit request,
        // silently changing the next step's block accounting.
        let has_hit = use_pending_hit && self.pending_hit.iter().any(|v| !v.is_empty());
        let new_computed: &[Vec<u32>] = if has_hit {
            &self.pending_hit
        } else {
            &self.empty_groups
        };

        let num_local_computed_tokens = num_computed_tokens + num_new_computed_tokens;
        let total_computed_tokens = num_local_computed_tokens.min(self.cfg.max_model_len);

        let watermark_blocks = if has_scheduled_reqs
            && (status == STATUS_WAITING || status == STATUS_PREEMPTED)
        {
            self.watermark_blocks
        } else {
            0
        };

        let num_tokens_main_model = total_computed_tokens + num_new_tokens;
        let num_tokens_need_slot =
            (num_tokens_main_model + num_lookahead_tokens).min(self.cfg.max_model_len);

        // Must run before allocating so freed blocks are available (`:404`).
        self.coord.remove_skipped_blocks(req, total_computed_tokens);

        let num_blocks_to_allocate = self.coord.get_num_blocks_to_allocate(
            req,
            num_tokens_need_slot,
            new_computed,
            num_local_computed_tokens,
            num_tokens_main_model,
        );

        let available_blocks = self.coord.pool.get_num_free_blocks();
        if num_blocks_to_allocate + watermark_blocks > available_blocks {
            return Ok(false);
        }

        if has_hit {
            let pending = std::mem::take(&mut self.pending_hit);
            let res = self
                .coord
                .allocate_new_computed_blocks(req, &pending, num_local_computed_tokens);
            self.pending_hit = pending;
            res?;
        }

        let mut out = std::mem::take(&mut self.new_blocks);
        let res = self
            .coord
            .allocate_new_blocks(req, num_tokens_need_slot, num_tokens_main_model, &mut out);
        self.new_blocks = out;
        res?;

        if !self.cfg.enable_caching {
            return Ok(true);
        }

        // Only "finalized" tokens are cached (`:458`).
        let num_tokens_to_cache = (total_computed_tokens + num_new_tokens).min(num_request_tokens);
        self.cache_blocks(req, num_tokens_to_cache);
        Ok(true)
    }

    /// `cache_blocks` (`:620`).
    pub fn cache_blocks(&mut self, req: u32, num_computed_tokens: usize) {
        if !self.cfg.enable_caching {
            return;
        }
        let hashes = std::mem::take(&mut self.reqs.entry(req).or_default().hashes);
        self.coord.cache_blocks(req, &hashes, num_computed_tokens);
        self.reqs.get_mut(&req).unwrap().hashes = hashes;
    }

    /// `free` (`:466`).
    pub fn free(&mut self, req: u32) {
        self.coord.free(req);
        self.hit_len.remove(&req);
    }

    pub fn pop_blocks_for_free(&mut self, req: u32, out: &mut Vec<u32>) {
        self.coord.pop_blocks_for_free(req, out);
    }

    pub fn remove_skipped_blocks(&mut self, req: u32, total_computed_tokens: usize) {
        self.coord.remove_skipped_blocks(req, total_computed_tokens);
    }

    pub fn evict_blocks(&mut self, block_ids: &[u32]) {
        self.coord.pool.evict_blocks(block_ids);
    }

    /// `reset_prefix_cache` (`:516`).
    pub fn reset_prefix_cache(&mut self) -> bool {
        if !self.coord.pool.reset_prefix_cache() {
            return false;
        }
        if self.cfg.log_stats {
            self.stats.reset = true;
        }
        true
    }

    pub fn get_num_common_prefix_blocks(&mut self, req: u32) -> &[usize] {
        let mut out = std::mem::take(&mut self.common_prefix);
        self.coord.get_num_common_prefix_blocks(req, &mut out);
        self.common_prefix = out;
        &self.common_prefix
    }

    /// `get_blocks` (`:592`) for one group.
    pub fn group_blocks(&self, req: u32, group: usize) -> &[u32] {
        self.coord.managers[group].blocks(req)
    }

    /// `get_block_ids_for_computed_tokens` (`:600`): attention groups clip to the blocks
    /// covering `num_computed_tokens`; other kinds return everything.
    pub fn num_blocks_for_computed_tokens(
        &self,
        req: u32,
        group: usize,
        num_computed_tokens: usize,
    ) -> usize {
        let m = &self.coord.managers[group];
        let all = m.blocks(req).len();
        if m.kind == Kind::FullAttention {
            all.min(cdiv(num_computed_tokens, m.block_size))
        } else {
            all
        }
    }

    pub fn take_new_block_ids(&mut self, out: &mut Vec<u32>) {
        self.coord.take_new_block_ids(out);
    }

    pub fn new_step_starts(&mut self) {
        self.coord.new_step_starts();
    }

    /// Drain the eviction victims recorded since the last call (shadow-mode comparison).
    pub fn take_evicted(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.coord.pool.evicted_this_step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, GroupConfig};
    use crate::hash::{Digest32, HASH_LEN};

    fn hybrid_cfg(num_blocks: usize) -> Config {
        Config {
            num_blocks,
            enable_caching: true,
            max_model_len: 4096,
            scheduler_block_size: 16,
            hash_block_size: 16,
            log_stats: true,
            watermark: 0.0,
            radix: false,
            groups: vec![
                GroupConfig {
                    kind: Kind::FullAttention,
                    block_size: 16,
                    is_full_attention: true,
                    spec_signature: "full".into(),
                    mamba_align: false,
                    num_speculative_blocks: 0,
                    use_eagle: false,
                },
                GroupConfig {
                    kind: Kind::Mamba,
                    block_size: 16,
                    is_full_attention: false,
                    spec_signature: "mamba".into(),
                    mamba_align: true,
                    num_speculative_blocks: 0,
                    use_eagle: false,
                },
            ],
        }
    }

    fn packed(hashes: &[Digest32]) -> Vec<u8> {
        hashes.iter().flat_map(|h| h.iter().copied()).collect()
    }

    fn chain(n: usize, salt: u8) -> Vec<Digest32> {
        (0..n)
            .map(|i| {
                let mut d: Digest32 = [salt; HASH_LEN];
                d[0] = i as u8;
                d
            })
            .collect()
    }

    #[test]
    fn hybrid_second_request_reuses_the_prefix() {
        let mut m = Manager::new(hybrid_cfg(256)).unwrap();
        let hs = chain(4, 1);
        // First request: 64 tokens, no hit, allocate + cache.
        let a = m.intern("a");
        m.push_hashes(a, &packed(&hs), 64);
        assert_eq!(m.get_computed_blocks(a, 64, 0, false), 0);
        assert!(m
            .allocate_slots(a, 64, 0, true, 0, 0, 64, STATUS_WAITING, false)
            .unwrap());
        // Full attention holds 4 blocks; mamba (align) holds 3 nulls + 1 state.
        assert_eq!(m.group_blocks(a, 0).len(), 4);
        assert_eq!(m.group_blocks(a, 1).len(), 4);

        // An identical 64-token request gets NOTHING: full attention would match 3
        // blocks (capped at num_tokens - 1 = 63) but the only cached mamba state sits
        // at token 64, so the hybrid fixed point collapses the hit to 0. This is the
        // whole reason `_mamba_block_aligned_split` + `num_uncached_common_prefix_tokens`
        // exist, and getting it "helpfully" wrong would corrupt output.
        m.new_step_starts();
        let b = m.intern("b");
        m.push_hashes(b, &packed(&hs), 64);
        assert_eq!(m.get_computed_blocks(b, 64, 0, false), 0);
        assert_eq!(m.coord.num_uncached_common_prefix_tokens, 48);
        assert!(m.pending_hit[0].is_empty());

        // A longer request sharing the same 64-token prefix DOES hit, on both groups.
        let hs2 = {
            let mut v = hs.clone();
            v.push([42; HASH_LEN]);
            v
        };
        let c = m.intern("c");
        m.push_hashes(c, &packed(&hs2), 80);
        assert_eq!(m.get_computed_blocks(c, 80, 0, false), 64);
        assert_eq!(m.pending_hit[0].len(), 4);
        assert_eq!(m.pending_hit[1].len(), 4, "mamba pads with nulls to the same length");
        assert_eq!(m.pending_hit[1][0], m.coord.pool.null_block);
        assert_ne!(m.pending_hit[1][3], m.coord.pool.null_block);

        let stats = m.take_prefix_cache_stats().unwrap();
        assert_eq!((stats.requests, stats.queries, stats.hits), (3, 208, 64));
    }

    #[test]
    fn allocate_slots_refuses_when_the_pool_is_full() {
        let mut m = Manager::new(hybrid_cfg(8)).unwrap();
        let a = m.intern("a");
        m.push_hashes(a, &packed(&chain(64, 3)), 1024);
        assert_eq!(m.get_computed_blocks(a, 1024, 0, false), 0);
        // 1024 tokens needs 64 full-attn blocks; only 7 are free.
        assert!(!m
            .allocate_slots(a, 1024, 0, true, 0, 0, 1024, STATUS_WAITING, false)
            .unwrap());
        assert_eq!(m.coord.pool.get_num_free_blocks(), 7, "no partial allocation");
    }

    #[test]
    fn free_returns_every_block() {
        let mut m = Manager::new(hybrid_cfg(64)).unwrap();
        let a = m.intern("a");
        m.push_hashes(a, &packed(&chain(2, 5)), 32);
        m.get_computed_blocks(a, 32, 0, false);
        m.allocate_slots(a, 32, 0, true, 0, 0, 32, STATUS_WAITING, false)
            .unwrap();
        let before = m.coord.pool.get_num_free_blocks();
        assert!(before < 63);
        m.free(a);
        assert_eq!(m.coord.pool.get_num_free_blocks(), 63);
    }

    #[test]
    fn watermark_only_applies_to_waiting_requests() {
        let mut cfg = hybrid_cfg(64);
        cfg.watermark = 0.5; // 32 blocks reserved
        let mut m = Manager::new(cfg).unwrap();
        let a = m.intern("a");
        m.push_hashes(a, &packed(&chain(40, 7)), 640);
        m.get_computed_blocks(a, 640, 0, false);
        // 640 tokens -> 40 full-attn blocks + 1 mamba state = 41; free = 63; 41 + 32 > 63.
        assert!(!m
            .allocate_slots(a, 640, 0, true, 0, 0, 640, STATUS_WAITING, true)
            .unwrap());
        // Same allocation for a RUNNING request skips the watermark.
        assert!(m
            .allocate_slots(a, 640, 0, true, 0, 0, 640, STATUS_RUNNING, true)
            .unwrap());
    }
}
