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

use std::hash::{Hash, Hasher};

use rustc_hash::{FxHashMap, FxHasher};
use smallvec::SmallVec;

use crate::block_pool::BlockPool;
use crate::config::Config;
use crate::coordinator::Coordinator;
use crate::hash::Digest32;
use crate::journal::{Journal, Undo};
use crate::single_type::{cdiv, Kind, TypeManager};

/// R6b: the resident mirror of the `Request` fields the scheduling loop reads. Same
/// struct as the marshalled tuple — sharing it is the point, so a resident step and a
/// resync step cannot disagree about what a field means.
pub use crate::sched::SchedReq as SchedEntry;

/// Request status values the allocator's watermark branch cares about
/// (`RequestStatus`, vllm/v1/request.py).
pub const STATUS_WAITING: u8 = 0;
pub const STATUS_RUNNING: u8 = 1;
pub const STATUS_PREEMPTED: u8 = 2;

/// Field-by-field hash of a table entry. Hand-written rather than `#[derive(Hash)]` on
/// `SchedReq` so the fingerprint is stable if the struct ever gains a field that is not
/// part of the scheduling decision.
fn hash_entry<H: Hasher>(e: &SchedEntry, h: &mut H) {
    e.slot.hash(h);
    e.num_tokens.hash(h);
    e.num_tokens_with_spec.hash(h);
    e.num_computed_tokens.hash(h);
    e.num_output_placeholders.hash(h);
    e.num_prompt_tokens.hash(h);
    e.max_tokens.hash(h);
    e.status.hash(h);
    e.num_preemptions.hash(h);
    e.is_prefill_chunk.hash(h);
    e.skip_reading_prefix_cache.hash(h);
}

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

    /// R6a: slot -> stop params + the per-step batched stop decision. Lives here because
    /// slot interning does, so `forget` drops both halves of a request in one place.
    pub stops: crate::update::StopTable,

    /// Phase 2: slot -> "the R8 record has a field for everything this request needs"
    /// (`rust_sched.py::_pack_ok_clauses`), interned at request-ADD time on the input
    /// thread instead of re-derived on the request's first decode step under the GIL.
    ///
    /// NOT in the resident table, deliberately: that table is JOURNALED, so a speculative
    /// schedule would have to undo an entry an add-time write put there — and the two have
    /// nothing to do with each other (this is a property of the request, not of a
    /// scheduling step). It lives beside `stops`/`tokens` for the reason those do: `forget`
    /// is the one place a request's every half is dropped, and a recycled slot must not
    /// inherit the dead request's answer.
    ///
    /// A map rather than a set: `Some(false)` ("registered, not packable") and `None`
    /// ("never registered") route the Python reader to different arms — cache the answer
    /// vs. do the full derivation.
    pack_ok: FxHashMap<u32, bool>,

    /// R8: reused buffer for the raw shm output record. Lives here (not in a local) so a
    /// step packs into last step's allocation.
    pub raw: crate::update::RawPacker,

    /// Port-2: slot -> token counters + unhashed tail (`tokens.rs`). Lives here for the
    /// same reason `stops` does -- `forget` must drop every half of a request in one place,
    /// and the hash catch-up writes into `reqs[slot].hashes`.
    pub tokens: crate::tokens::TokenStore,
    /// Scratch for `update_step_pack_store`'s flat inputs, reused across steps.
    tok_cu: Vec<u32>,
    tok_nout: Vec<u32>,
    tok_ntok: Vec<u32>,

    /// R6b: slot -> the scheduling loop's view of the request. Dense (slots are interned
    /// with reuse), so a `Vec` beats a map on the hot lookup. Written wholesale by the
    /// marshalled [`crate::sched::ScheduleCore::schedule`] (the resync path), then kept
    /// current by the post-schedule commit and `update_step`.
    pub(crate) table: Vec<Option<SchedEntry>>,

    /// Reusable per-group buffers (allocated once at construction).
    pub pending_hit: Vec<Vec<u32>>,
    pub new_blocks: Vec<Vec<u32>>,
    pub empty_groups: Vec<Vec<u32>>,
    pub scratch_flat: Vec<u32>,
    pub common_prefix: Vec<usize>,
    /// Last `get_computed_blocks` result, per request slot.
    pub(crate) hit_len: FxHashMap<u32, usize>,

    /// R9: steps where `update_step_pack_store`'s `fold_cache` found a slot with no
    /// resident table entry (or a non-RUNNING one) and skipped its `cache_blocks` fold.
    /// `table_with`'s silent skip (`:315`) is correct-by-construction for the paths that
    /// use it today, but folding `cache_blocks` in leans on the SAME silence, so this
    /// counter is the loud version: it should stay 0 for the lifetime of a correct boot,
    /// and `rust_sched.py` polls it to trip the R9 disable latch if it ever isn't.
    pub cache_fold_skips: u64,
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
            stops: Default::default(),
            pack_ok: FxHashMap::default(),
            raw: Default::default(),
            tokens: crate::tokens::TokenStore::new(cfg.hash_block_size),
            tok_cu: Vec::with_capacity(16),
            tok_nout: Vec::with_capacity(16),
            tok_ntok: Vec::with_capacity(16),
            table: Vec::new(),
            pending_hit: (0..n).map(|_| Vec::with_capacity(64)).collect(),
            new_blocks: (0..n).map(|_| Vec::with_capacity(64)).collect(),
            empty_groups: (0..n).map(|_| Vec::new()).collect(),
            scratch_flat: Vec::with_capacity(256),
            common_prefix: Vec::with_capacity(n),
            hit_len: FxHashMap::default(),
            cache_fold_skips: 0,
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

    /// Reverse of [`Manager::intern`]: the request id a slot currently holds.
    ///
    /// `names` keeps a STALE string in a freed slot (`forget` only pushes the slot onto
    /// `free_slots`), so the entry is cross-checked against the forward map. A recycled or
    /// never-interned slot answers `None` rather than a dead request's id -- which, on the
    /// R8 path, would otherwise put the wrong id on the wire.
    pub fn name(&self, slot: u32) -> Option<&str> {
        let name = self.names.get(slot as usize)?;
        if self.ids.get(name.as_str()) == Some(&slot) {
            Some(name.as_str())
        } else {
            None
        }
    }

    pub fn forget(&mut self, name: &str) {
        if let Some(id) = self.ids.remove(name) {
            if let Some(st) = self.reqs.get_mut(&id) {
                st.hashes.clear();
                st.num_prompt_tokens = 0;
            }
            self.hit_len.remove(&id);
            self.stops.forget(id);
            // Phase 2: same reason as `stops` -- a recycled slot must not answer with the
            // dead request's pack-ok bit, which would let `decide()` skip a derivation the
            // new occupant never had run for it.
            self.pack_ok.remove(&id);
            // Port-2: the token store is slot-keyed too, so a recycled slot must not
            // inherit the dead request's counters, tail or output tokens.
            self.tokens.forget(id);
            self.table_clear(id);
            self.free_slots.push(id);
        }
    }

    /// Phase 2: intern this request's whole add-time registration in one shot -- the slot,
    /// its stop params and its pack-ok bit. Returns the slot, which is the same one
    /// `RustMirror.slot()` will resolve later (`intern` is idempotent per request id).
    pub fn set_request_meta(
        &mut self,
        name: &str,
        pack_ok: bool,
        params: crate::update::StopParams,
    ) -> u32 {
        let slot = self.intern(name);
        self.stops.set(slot, params);
        self.pack_ok.insert(slot, pack_ok);
        slot
    }

    /// The pack-ok bit for a slot, or `None` when this slot was never preregistered (or has
    /// been recycled since). `None` is the caller's signal to do the Python derivation.
    pub fn request_meta(&self, slot: u32) -> Option<bool> {
        self.pack_ok.get(&slot).copied()
    }

    // ---- R6b: the resident request table ----------------------------------

    /// Overwrite one slot's entry. Journaled, because the post-schedule commit runs
    /// inside the speculative window.
    pub fn table_set(&mut self, slot: u32, entry: SchedEntry) {
        let i = slot as usize;
        if self.table.len() <= i {
            self.table.resize(i + 1, None);
        }
        if let Some(j) = self.coord.pool.arena.journal.as_mut() {
            j.push(Undo::TableEntry(slot, self.table[i]));
        }
        self.table[i] = Some(entry);
    }

    /// `free` deliberately does NOT clear the table: it is also the preemption path
    /// inside the scheduling loop, where the entry must survive to be re-stamped
    /// PREEMPTED by the post-schedule commit. Request death goes through `forget`.
    pub fn table_clear(&mut self, slot: u32) {
        let i = slot as usize;
        if i < self.table.len() {
            if let Some(j) = self.coord.pool.arena.journal.as_mut() {
                j.push(Undo::TableEntry(slot, self.table[i]));
            }
            self.table[i] = None;
        }
    }

    #[inline]
    pub fn table_get(&self, slot: u32) -> Option<SchedEntry> {
        self.table.get(slot as usize).copied().flatten()
    }

    /// Journal-only: restores a slot verbatim, growing the table if the speculative run
    /// was what created it.
    pub(crate) fn table_restore(&mut self, slot: u32, entry: Option<SchedEntry>) {
        let i = slot as usize;
        if self.table.len() <= i {
            self.table.resize(i + 1, None);
        }
        self.table[i] = entry;
    }

    /// Apply the post-schedule commit to one slot (see [`crate::sched`]).
    pub(crate) fn table_with<F: FnOnce(&mut SchedEntry)>(&mut self, slot: u32, f: F) {
        let Some(mut e) = self.table_get(slot) else {
            return;
        };
        f(&mut e);
        self.table_set(slot, e);
    }

    /// FxHash over the entries at `slots`, in the given order. Cheap enough to call every
    /// step from Python as a spot check against its own request objects.
    pub fn table_fingerprint(&self, slots: &[u32]) -> u64 {
        let mut h = FxHasher::default();
        for &s in slots {
            match self.table_get(s) {
                Some(e) => {
                    1u8.hash(&mut h);
                    hash_entry(&e, &mut h);
                }
                None => 0u8.hash(&mut h),
            }
        }
        h.finish()
    }

    // ---- speculation support (see `journal.rs` / `spec.rs`) ---------------

    #[inline]
    pub fn journal_armed(&self) -> bool {
        self.coord.pool.arena.journal.is_some()
    }

    /// Start recording. Panics if already armed — double-arming would silently discard
    /// the first journal and make the state unrecoverable.
    pub fn arm_journal(&mut self) {
        assert!(!self.journal_armed(), "journal armed twice");
        self.coord.pool.arena.journal = Some(Journal::default());
    }

    /// Stop recording and THROW AWAY the journal: the speculative run is being kept.
    pub fn commit_journal(&mut self) {
        self.coord.pool.arena.journal = None;
    }

    /// Stop recording and reverse-apply. State afterwards is bit-identical to the
    /// pre-arm state (asserted by `state_fingerprint` in the property tests).
    pub fn rollback_journal(&mut self) {
        if let Some(j) = self.coord.pool.arena.journal.take() {
            j.rollback(self);
        }
    }

    /// True when the in-flight journal has grown past its soft cap — the signal for the
    /// scheduling loop to abandon a speculative run instead of ballooning memory.
    pub fn journal_over_cap(&self) -> bool {
        self.coord
            .pool
            .arena
            .journal
            .as_ref()
            .is_some_and(|j| j.over_soft_cap())
    }

    /// Order-sensitive hash of EVERYTHING the scheduler can observe. Exists to prove the
    /// journal: `fingerprint == fingerprint_after_arm_run_rollback`. O(pool size), so it
    /// is a test/diagnostic call, never a hot path.
    pub fn state_fingerprint(&self) -> u64 {
        let mut h = FxHasher::default();
        let pool = &self.coord.pool;
        for b in pool.arena.blocks.iter() {
            b.ref_cnt.hash(&mut h);
            match &b.hash {
                Some(k) => {
                    1u8.hash(&mut h);
                    k.hash.hash(&mut h);
                    k.group.hash(&mut h);
                }
                None => 0u8.hash(&mut h),
            }
            b.hash_num_tokens.hash(&mut h);
            b.prev.hash(&mut h);
            b.next.hash(&mut h);
            b.is_null.hash(&mut h);
        }
        pool.arena.num_free_blocks.hash(&mut h);
        pool.arena.free_order().hash(&mut h);

        let mut idx = pool.cached.entries();
        idx.sort_by(|a, b| (a.0.hash, a.0.group).cmp(&(b.0.hash, b.0.group)));
        for (k, ids) in &idx {
            k.hash.hash(&mut h);
            k.group.hash(&mut h);
            ids.as_slice().hash(&mut h);
        }
        let mut cbb: Vec<_> = pool
            .cached_by_block
            .iter()
            .map(|(b, ks)| (*b, ks.iter().map(|k| (k.hash, k.group)).collect::<Vec<_>>()))
            .collect();
        cbb.sort();
        cbb.hash(&mut h);

        for m in self.coord.managers.iter() {
            let mut rb: Vec<_> = m.req_to_blocks.iter().map(|(r, v)| (*r, v.clone())).collect();
            rb.sort();
            rb.hash(&mut h);
            let mut nc: Vec<_> = m.num_cached_block.iter().map(|(r, n)| (*r, *n)).collect();
            nc.sort();
            nc.hash(&mut h);
            m.new_block_ids.hash(&mut h);
            let mut cs: Vec<_> = m
                .cached_blocks_this_step
                .iter()
                .map(|k| (k.hash, k.group))
                .collect();
            cs.sort();
            cs.hash(&mut h);
            let mut ls: Vec<_> = m.last_state_block_idx.iter().map(|(r, n)| (*r, *n)).collect();
            ls.sort();
            ls.hash(&mut h);
            let mut ar: Vec<_> = m.allocated_block_reqs.iter().copied().collect();
            ar.sort();
            ar.hash(&mut h);
        }

        self.coord.num_uncached_common_prefix_tokens.hash(&mut h);
        let mut hl: Vec<_> = self.hit_len.iter().map(|(r, n)| (*r, *n)).collect();
        hl.sort();
        hl.hash(&mut h);
        for (i, e) in self.table.iter().enumerate() {
            match e {
                Some(e) => {
                    i.hash(&mut h);
                    hash_entry(e, &mut h);
                }
                None => (i, 0u8).hash(&mut h),
            }
        }
        h.finish()
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
        if self.journal_armed() {
            let old = self.hit_len.get(&req).copied();
            self.coord
                .pool
                .arena
                .journal
                .as_mut()
                .unwrap()
                .push(Undo::HitLen(req, old));
        }
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

    /// R6a + R6b: one step's stop decisions AND the resident table's update-time delta.
    ///
    /// Mirrors `AsyncScheduler._update_request_with_output` (async_scheduler.py:52): the
    /// real tokens that survived `check_stop` are appended (so `num_tokens` grows by
    /// exactly `num_accepted`) and `num_output_placeholders` drops by the same amount.
    ///
    /// `num_tokens[i]` is Python's authoritative pre-append count, so the entry is SYNCED
    /// to it rather than blindly incremented — the same number of instructions, and it
    /// makes every out-of-band mutation Python performs (spec rejection, KV-load
    /// rollback, a resumed request) self-heal on the next step instead of drifting.
    /// Slots that Rust refused (`status == u8::MAX`) are left alone: Python ran stock
    /// `check_stop` for those and owns the result. The placeholder decrement saturates
    /// where vLLM asserts `>= 0`: the table is a cache of Python's state, so clamping and
    /// letting the next full resync repair it beats taking the engine down.
    pub fn update_step(
        &mut self,
        slots: &[u32],
        cu_lens: &[u32],
        token_ids: &[i64],
        num_output_tokens: &[u32],
        num_tokens: &[u32],
        max_model_len: usize,
    ) -> Result<&[crate::update::Verdict], String> {
        self.stops.update_step(
            slots,
            cu_lens,
            token_ids,
            num_output_tokens,
            num_tokens,
            max_model_len,
        )?;
        for (i, &slot) in slots.iter().enumerate() {
            let v = self.stops.out[i];
            if v.status == u8::MAX {
                continue;
            }
            let accepted = v.num_accepted as usize;
            let authoritative = num_tokens[i] as usize + accepted;
            self.table_with(slot, |e| {
                let delta = authoritative as i64 - e.num_tokens as i64;
                e.num_tokens = authoritative;
                e.num_tokens_with_spec = (e.num_tokens_with_spec as i64 + delta).max(0) as usize;
                e.num_output_placeholders = e.num_output_placeholders.saturating_sub(accepted);
            });
        }
        Ok(&self.stops.out)
    }

    /// C1a: the N-step burst's extra tokens, applied to the resident table.
    ///
    /// The Rust half of the post-schedule commit ([`crate::sched::ScheduleCore::commit`]'s
    /// `advance`) already moved each slot by the ONE token `num_scheduled_tokens` says was
    /// scheduled. A committed burst executes `delta` more in the same step, so both
    /// counters move again and `is_prefill_chunk` is recomputed exactly as `advance` does.
    ///
    /// `delta` is negative on the reconcile path -- a burst request that stopped (or whose
    /// runner produced fewer tokens than committed) gives the surplus back. Subtraction
    /// saturates at 0: the table is a cache of Python's state, and clamping plus the next
    /// full resync beats taking the engine down on an accounting skew.
    ///
    /// Slots with no resident entry are skipped: they are exactly the slots Python owns
    /// this step, and `TableState` is already marked dirty for them.
    pub fn table_burst(&mut self, slots: &[u32], delta: i64) {
        for &slot in slots {
            self.table_with(slot, |e| {
                if delta >= 0 {
                    // Literally one more `_update_after_schedule` of `delta` tokens --
                    // same function, so the placeholder-bump ordering cannot drift.
                    crate::sched::advance(e, delta as usize, delta as usize);
                } else {
                    let d = delta.unsigned_abs() as usize;
                    e.num_computed_tokens = e.num_computed_tokens.saturating_sub(d);
                    e.num_output_placeholders = e.num_output_placeholders.saturating_sub(d);
                    e.is_prefill_chunk =
                        e.num_computed_tokens < e.num_tokens + e.num_output_placeholders;
                }
            });
        }
    }

    /// R8: [`Manager::update_step`] plus the complete TAG_RAW output record.
    ///
    /// One row per slot, in the order given -- every slot the caller passes has at least
    /// one offered token (the caller drops empty ones), and `apply_tokens` never accepts
    /// zero of a non-empty list, so "one slot = one `EngineCoreOutput`" holds exactly as
    /// it does in `Scheduler.update_from_output`.
    ///
    /// Returns `Ok((verdicts, None))` -- verdicts still applied, no record -- whenever any
    /// slot was refused (`status == u8::MAX`) or has no interned name. Python then builds
    /// the objects itself for the whole step; a PARTIAL record is never produced, because
    /// the wire format has no way to say "this row is somebody else's problem".
    ///
    /// `finished` is handed in rather than inferred: `finished_req_ids_dict` also collects
    /// aborts and earlier steps' frees, which this crate cannot see. It must arrive SORTED
    /// (Python sorts a set) or the bytes will not match the Python packer.
    #[allow(clippy::too_many_arguments)]
    pub fn update_step_pack(
        &mut self,
        slots: &[u32],
        cu_lens: &[u32],
        token_ids: &[i64],
        num_output_tokens: &[u32],
        num_tokens: &[u32],
        max_model_len: usize,
        engine_index: u32,
        timestamp: f64,
        finished: &[String],
    ) -> Result<(&[crate::update::Verdict], bool), String> {
        let packed = self.pack_inner(
            slots,
            cu_lens,
            token_ids,
            num_output_tokens,
            num_tokens,
            max_model_len,
            engine_index,
            timestamp,
            finished,
        )?;
        Ok((&self.stops.out, packed))
    }

    /// [`Manager::update_step_pack`] without the verdict borrow, so the store variant can
    /// run it and then keep mutating `self`. The ONE implementation of the record layout:
    /// both public entry points go through here, which is what makes the bytes identical
    /// with and without the token store (asserted by `record_is_identical_with_the_store`).
    #[allow(clippy::too_many_arguments)]
    fn pack_inner(
        &mut self,
        slots: &[u32],
        cu_lens: &[u32],
        token_ids: &[i64],
        num_output_tokens: &[u32],
        num_tokens: &[u32],
        max_model_len: usize,
        engine_index: u32,
        timestamp: f64,
        finished: &[String],
    ) -> Result<bool, String> {
        self.update_step(
            slots,
            cu_lens,
            token_ids,
            num_output_tokens,
            num_tokens,
            max_model_len,
        )?;
        // Disjoint field borrows: `stops.out` and `names`/`ids` read, `raw` written.
        if self.stops.out.iter().any(|v| v.status == u8::MAX) {
            return Ok(false);
        }
        let raw = &mut self.raw;
        raw.begin(
            engine_index,
            timestamp,
            slots.len() as u32,
            finished.len() as u32,
        );
        for (i, &slot) in slots.iter().enumerate() {
            let name = match self.names.get(slot as usize) {
                Some(n) if self.ids.get(n.as_str()) == Some(&slot) => n.as_str(),
                _ => return Ok(false),
            };
            let v = self.stops.out[i];
            let lo = cu_lens[i] as usize;
            let hi = lo + v.num_accepted as usize;
            let stop = (v.stop_reason >= 0).then_some(v.stop_reason);
            if raw
                .push_output(
                    name,
                    &token_ids[lo..hi],
                    crate::update::finish_reason(v.status),
                    stop,
                )
                .is_err()
            {
                return Ok(false);
            }
        }
        for req_id in finished {
            if raw.push_finished(req_id).is_err() {
                return Ok(false);
            }
        }
        Ok(raw.finish().is_ok())
    }

    // ---- Port-2: the token store ------------------------------------------

    /// Arm the store with vLLM's live `NONE_HASH`. Until this runs, `store_init` refuses.
    pub fn store_arm(&mut self, none_hash: Digest32) {
        self.tokens.arm(none_hash);
    }

    /// Take over one slot's token bookkeeping. `pending` is the request's token tail past
    /// the last block hash Python already produced.
    pub fn store_init(
        &mut self,
        slot: u32,
        pending: &[i64],
        num_tokens: usize,
        num_output_tokens: usize,
    ) -> Result<(), String> {
        self.tokens.init(slot, pending, num_tokens, num_output_tokens)
    }

    /// Output tokens the store has appended for this slot. The rare-path materialization.
    pub fn slot_tokens(&self, slot: u32) -> Option<&[i64]> {
        self.tokens.tokens(slot)
    }

    /// The slot's whole block-hash chain, 32 bytes per hash. Python rebuilds
    /// `Request.block_hashes` from this when it hands a request back to the stock path.
    pub fn slot_hashes(&self, slot: u32) -> Option<Vec<u8>> {
        let st = self.reqs.get(&slot)?;
        let mut out = Vec::with_capacity(st.hashes.len() * crate::hash::HASH_LEN);
        for h in &st.hashes {
            out.extend_from_slice(h);
        }
        Some(out)
    }

    /// Append this step's ACCEPTED tokens to the store and run the block-hash catch-up.
    ///
    /// Called by the authoritative numpy pack path. `accepted[i]` is `Verdict::num_accepted`, i.e. stop truncation already
    /// applied -- a token past a stop was never computed into KV and must not be hashed.
    pub fn store_apply(
        &mut self,
        slots: &[u32],
        cu_lens: &[u32],
        token_ids: &[i64],
        accepted: &[u32],
    ) -> Result<(), String> {
        let n = slots.len();
        if cu_lens.len() != n + 1 || accepted.len() != n {
            return Err(format!(
                "store_apply arity mismatch: {n} slots, {} cu_lens, {} accepted",
                cu_lens.len(),
                accepted.len()
            ));
        }
        // Disjoint field borrows: the store mutates `tokens`, the chain lives in `reqs`.
        let Manager { tokens, reqs, .. } = self;
        for i in 0..n {
            let lo = cu_lens[i] as usize;
            let hi = lo + accepted[i] as usize;
            if lo > hi || hi > token_ids.len() || hi > cu_lens[i + 1] as usize {
                return Err("store_apply accepted count is out of range".into());
            }
            let st = reqs
                .get_mut(&slots[i])
                .ok_or_else(|| format!("store_apply: slot {} has no request state", slots[i]))?;
            tokens.append(slots[i], &token_ids[lo..hi], &mut st.hashes)?;
        }
        Ok(())
    }

    /// [`Manager::update_step_pack`] with the per-request counters sourced from the token
    /// store instead of from Python, and this step's kept tokens appended to it.
    ///
    /// The record is built by the same [`Manager::pack_inner`], so the bytes are identical
    /// to the non-store path for identical inputs. On a refused slot (`status == u8::MAX`,
    /// which `pack_inner` reports as "not packed") NOTHING is appended: Python is about to
    /// materialize the whole step onto the stock path, and it owns the append from there.
    ///
    /// `fold_cache`: R9. When the step packed, additionally run each slot's `cache_blocks`
    /// INSIDE this call (see [`Manager::fold_cache_blocks`]) instead of Python doing it one
    /// FFI crossing per request afterwards. `false` reproduces today's behaviour exactly —
    /// no store append, no fold either, is unaffected either way.
    #[allow(clippy::too_many_arguments)]
    pub fn update_step_pack_store(
        &mut self,
        slots: &[u32],
        counts: &[u32],
        token_ids: &[i64],
        max_model_len: usize,
        engine_index: u32,
        timestamp: f64,
        finished: &[String],
        fold_cache: bool,
    ) -> Result<(&[crate::update::Verdict], bool), String> {
        let n = slots.len();
        if counts.len() != n {
            return Err(format!(
                "update_step_pack_store arity mismatch: {n} slots, {} counts",
                counts.len()
            ));
        }
        // Take the scratch out so the slices can be handed to the `&mut self` calls below;
        // put it back before returning so the next step reuses this step's allocation.
        let mut cu = std::mem::take(&mut self.tok_cu);
        let mut nout = std::mem::take(&mut self.tok_nout);
        let mut ntok = std::mem::take(&mut self.tok_ntok);
        cu.clear();
        nout.clear();
        ntok.clear();
        cu.push(0);
        let mut result = Ok(false);
        for i in 0..n {
            match self.tokens.counts(slots[i]) {
                Some((num_tokens, num_output_tokens)) => {
                    ntok.push(num_tokens as u32);
                    nout.push(num_output_tokens as u32);
                }
                None => {
                    result = Err(format!("token store has no entry for slot {}", slots[i]));
                    break;
                }
            }
            cu.push(cu[i] + counts[i]);
        }
        if result.is_ok() {
            result = if cu[n] as usize != token_ids.len() {
                Err(format!(
                    "update_step_pack_store expected {} flat tokens, got {}",
                    cu[n],
                    token_ids.len()
                ))
            } else {
                self.pack_inner(
                    slots,
                    &cu,
                    token_ids,
                    &nout,
                    &ntok,
                    max_model_len,
                    engine_index,
                    timestamp,
                    finished,
                )
                .and_then(|packed| {
                    if packed {
                        let acc: SmallVec<[u32; 16]> =
                            self.stops.out.iter().map(|v| v.num_accepted).collect();
                        self.store_apply(slots, &cu, token_ids, &acc)?;
                        if fold_cache {
                            self.fold_cache_blocks(slots);
                        }
                    }
                    Ok(packed)
                })
            };
        }
        self.tok_cu = cu;
        self.tok_nout = nout;
        self.tok_ntok = ntok;
        let packed = result?;
        Ok((&self.stops.out, packed))
    }

    /// Would [`Manager::update_step_pack_store`] be able to PACK these slots? A pre-flight
    /// for the Rust model runner, and the reason it exists is ordering: the runner asks
    /// BEFORE it launches, because after the launch the graph has already fed its tokens
    /// back into the persistent input buffers and there is no longer a step to decline.
    ///
    /// Mirrors `pack_inner`'s two refusals that are a property of the SLOT rather than of
    /// this step's tokens: no interned stop params (the verdict would be `u8::MAX` and the
    /// record would be dropped) and no live interned name (nothing to key the record row
    /// on). What is left there -- a `RawPacker` write failing -- is a buffer fault, not a
    /// state the caller could have avoided.
    pub fn runner_packable(&self, slots: &[u32]) -> bool {
        slots.iter().all(|&slot| {
            self.stops.has(slot)
                && match self.names.get(slot as usize) {
                    Some(n) => self.ids.get(n.as_str()) == Some(&slot),
                    None => false,
                }
        })
    }

    /// [`Manager::runner_packable`] plus the identity an UPDATE-TIME commit needs: every
    /// slot must still hold the request the launch was planned for.
    ///
    /// WHY THE NAMES. Under the depth-2 batch queue the pinned-ring commit for step k runs
    /// after `schedule(k + 1)`, so between the launch and the commit a request in the batch
    /// can finish (`update(k - 1)` frees its slot) and the freed slot can be handed to a NEW
    /// request. `runner_packable` alone would then say yes -- the slot has stop params and a
    /// live name, just a different request's -- and the commit would append step k's tokens
    /// to somebody else's store. Checking the names is what makes the decline happen instead,
    /// and a decline costs only the Python `decide()` for that step.
    /// INVARIANT this guard rests on: request ids are unique for the process lifetime
    /// (both the stock api_server and the vllm-rs frontend stamp every request with a
    /// fresh uuid; client-supplied ids never reach the engine core verbatim). A freed
    /// slot re-assigned to a DIFFERENT id fails the name compare below; a re-used id
    /// would not, and only that invariant rules it out -- if an id-recycling frontend
    /// ever appears, this needs a slot epoch instead.
    pub fn runner_owns(&self, slots: &[u32], names: &[String]) -> bool {
        slots.len() == names.len()
            && slots.iter().zip(names).all(|(&slot, want)| {
                self.stops.has(slot)
                    && self.names.get(slot as usize).map(String::as_str) == Some(want.as_str())
                    && self.ids.get(want.as_str()) == Some(&slot)
            })
    }

    /// R9: the per-request `cache_blocks` crossing folded into the packed step.
    ///
    /// Mirrors `AsyncScheduler._update_request_with_output`'s trailing call
    /// (async_scheduler.py:71-74): `cache_blocks(num_computed_tokens -
    /// num_output_placeholders)`, for every RUNNING slot this step accepted tokens for.
    /// By the time this runs, [`Manager::update_step`] (called from `pack_inner`, above)
    /// has already applied THIS step's placeholder decrement to the resident table
    /// (`:764`), so `e.num_computed_tokens - e.num_output_placeholders` here is exactly
    /// Python's post-decrement argument — no second read of Python state is needed.
    ///
    /// Every slot in `slots` has `status != u8::MAX` by construction (the caller only
    /// reaches this when `pack_inner` returned `packed == true`, which itself requires
    /// no refused slot), so that check is a documented invariant, not a live branch.
    /// A slot with no resident entry, or a non-RUNNING one (freed / preempted this
    /// step), is the `table_with` silent-skip hazard (`:315`) applied to a NEW caller —
    /// `cache_fold_skips` counts it rather than skipping quietly, so a boot where it
    /// ever fires is loud instead of a silent stale-cache accumulation.
    fn fold_cache_blocks(&mut self, slots: &[u32]) {
        for (i, &slot) in slots.iter().enumerate() {
            if self.stops.out[i].status == u8::MAX {
                continue;
            }
            match self.table_get(slot) {
                Some(e) if e.status == STATUS_RUNNING => {
                    let n = e.num_computed_tokens.saturating_sub(e.num_output_placeholders);
                    self.cache_blocks(slot, n);
                }
                _ => self.cache_fold_skips += 1,
            }
        }
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

    // ---- R8 -------------------------------------------------------------------------

    fn stop_params() -> crate::update::StopParams {
        crate::update::StopParams {
            min_tokens: 0,
            max_tokens: 8,
            eos_token_id: Some(2),
            stop_token_ids: Default::default(),
        }
    }

    #[test]
    fn name_refuses_a_recycled_or_unknown_slot() {
        let mut m = Manager::new(hybrid_cfg(64)).unwrap();
        let a = m.intern("a");
        assert_eq!(m.name(a), Some("a"));
        assert_eq!(m.name(a + 7), None);
        m.forget("a");
        // The string is still in `names`, but the slot is free -- it must not answer.
        assert_eq!(m.name(a), None);
        let b = m.intern("b");
        assert_eq!(b, a, "the slot is recycled, which is what makes the check load-bearing");
        assert_eq!(m.name(b), Some("b"));
    }

    #[test]
    fn update_step_pack_builds_the_record_and_still_applies_verdicts() {
        let mut m = Manager::new(hybrid_cfg(64)).unwrap();
        let a = m.intern("req-a");
        m.stops.set(a, stop_params());
        let (verdicts, packed) = m
            .update_step_pack(&[a], &[0, 1], &[7], &[0], &[10], 1000, 3, 0.5, &[])
            .unwrap();
        assert_eq!(verdicts[0].num_accepted, 1);
        assert_eq!(verdicts[0].status, crate::update::NOT_STOPPED);
        assert!(packed);
        let rec = m.raw.finish().unwrap();
        assert_eq!(rec[0], crate::update::TAG_RAW);
        assert_eq!(u32::from_le_bytes(rec[4..8].try_into().unwrap()), 3); // engine_index
        assert_eq!(u32::from_le_bytes(rec[16..20].try_into().unwrap()), 1); // n_out
        assert_eq!(u32::from_le_bytes(rec[20..24].try_into().unwrap()), 0); // n_fin
        assert_eq!(&rec[44..49], b"req-a");
    }

    #[test]
    fn update_step_pack_trims_the_record_at_a_stop_and_carries_finished_ids() {
        let mut m = Manager::new(hybrid_cfg(64)).unwrap();
        let a = m.intern("req-a");
        m.stops.set(a, stop_params());
        // EOS (2) is the second of three offered tokens -> the record must carry 2.
        let (verdicts, packed) = m
            .update_step_pack(
                &[a],
                &[0, 3],
                &[5, 2, 9],
                &[0],
                &[10],
                1000,
                0,
                0.0,
                &["req-a".to_string()],
            )
            .unwrap();
        assert_eq!(verdicts[0].num_accepted, 2);
        assert!(packed);
        let rec = m.raw.finish().unwrap();
        assert_eq!(u32::from_le_bytes(rec[28..32].try_into().unwrap()), 2, "n_tok");
        assert_eq!(rec[36], crate::update::FLAG_FINISH_REASON, "no stop_reason for EOS");
        assert_eq!(rec[37], crate::update::FINISH_STOP);
        // header + row head + "req-a" + 2 tokens + (u32 len + "req-a")
        assert_eq!(rec.len(), 24 + 20 + 5 + 8 + 4 + 5);
    }

    #[test]
    fn table_burst_moves_both_counters_and_gives_them_back() {
        let mut m = Manager::new(hybrid_cfg(64)).unwrap();
        let a = m.intern("a");
        // Steady-state decode entry as `commit`/`advance` leaves it: one token scheduled,
        // one already in flight, so num_computed is one past num_tokens and there are two
        // outstanding placeholders.
        let mut e = SchedEntry::default();
        e.num_tokens = 100;
        e.num_computed_tokens = 101;
        e.num_output_placeholders = 2;
        e.is_prefill_chunk = false;
        m.table_set(a, e);

        // A committed N=4 burst adds the 3 tokens the scheduler did not schedule.
        m.table_burst(&[a], 3);
        let got = m.table_get(a).unwrap();
        assert_eq!(got.num_computed_tokens, 104);
        assert_eq!(got.num_output_placeholders, 5);
        assert!(!got.is_prefill_chunk, "104 < 100 + 5 is false");

        // The burst stopped after 2 of 4 -> hand back 2, both counters, symmetrically.
        m.table_burst(&[a], -2);
        let got = m.table_get(a).unwrap();
        assert_eq!(got.num_computed_tokens, 102);
        assert_eq!(got.num_output_placeholders, 3);
        assert_eq!(
            got.is_prefill_chunk,
            got.num_computed_tokens < got.num_tokens + got.num_output_placeholders
        );

        // Saturating, not panicking, on an accounting skew.
        m.table_burst(&[a], -1000);
        let got = m.table_get(a).unwrap();
        assert_eq!(got.num_computed_tokens, 0);
        assert_eq!(got.num_output_placeholders, 0);

        // An unknown slot is skipped, not an error.
        m.table_burst(&[a + 9], 3);
    }

    #[test]
    fn update_step_pack_refuses_a_slot_with_no_interned_stop_params() {
        let mut m = Manager::new(hybrid_cfg(64)).unwrap();
        let a = m.intern("req-a");
        let b = m.intern("req-b");
        m.stops.set(a, stop_params());
        // b has no params -> status 255 -> the WHOLE step falls back to Python.
        let (verdicts, packed) = m
            .update_step_pack(&[a, b], &[0, 1, 2], &[7, 7], &[0, 0], &[10, 10], 1000, 0, 0.0, &[])
            .unwrap();
        assert_eq!(verdicts[1].status, u8::MAX);
        assert!(!packed, "a partial record must never be produced");
    }

    // ---- Port-2: the token store ----------------------------------------------------

    /// A manager with the store armed and one slot taken over at `(num_tokens, 0)`.
    fn store_mgr(name: &str, num_tokens: usize) -> (Manager, u32) {
        let mut m = Manager::new(hybrid_cfg(64)).unwrap();
        m.store_arm(crate::hash::none_hash_from_seed("0"));
        let slot = m.intern(name);
        m.stops.set(slot, stop_params());
        m.store_init(slot, &[], num_tokens, 0).unwrap();
        (m, slot)
    }

    #[test]
    fn record_is_identical_with_the_store() {
        // THE GOLDEN EQUIVALENCE. Same request, same tokens, same counters: the store path
        // must emit byte-identical bytes to the path Python's counters drive, or R8's wire
        // format silently forks between the two arms.
        let mut plain = Manager::new(hybrid_cfg(64)).unwrap();
        let a = plain.intern("req-a");
        plain.stops.set(a, stop_params());
        plain
            .update_step_pack(&[a], &[0, 2], &[5, 6], &[3], &[103], 1000, 0, 1.5, &[])
            .unwrap();
        let want = plain.raw.finish().unwrap().to_vec();

        let (mut m, slot) = store_mgr("req-a", 103);
        m.store_init(slot, &[], 103, 3).unwrap();
        let (verdicts, packed) = m
            .update_step_pack_store(&[slot], &[2], &[5, 6], 1000, 0, 1.5, &[], false)
            .unwrap();
        assert!(packed);
        assert_eq!(verdicts[0].num_accepted, 2);
        assert_eq!(m.raw.finish().unwrap(), &want[..]);
        // ...and the counters advanced by exactly the accepted tokens.
        assert_eq!(m.tokens.counts(slot), Some((105, 5)));
        assert_eq!(m.slot_tokens(slot), Some(&[5i64, 6][..]));
    }

    #[test]
    fn record_is_identical_with_the_store_and_fold_cache_on() {
        // R9: `fold_cache` must not perturb the record. It only adds an internal
        // `cache_blocks` call after the verdicts and the store append -- neither of which
        // the packer reads from -- so the bytes must match the `fold_cache=false` golden
        // exactly. `slot` has no resident table entry here, which also exercises
        // `cache_fold_skips` (see `cache_fold_skips_non_running_and_counts` for the
        // non-skip half of that contract).
        let mut plain = Manager::new(hybrid_cfg(64)).unwrap();
        let a = plain.intern("req-a");
        plain.stops.set(a, stop_params());
        plain
            .update_step_pack(&[a], &[0, 2], &[5, 6], &[3], &[103], 1000, 0, 1.5, &[])
            .unwrap();
        let want = plain.raw.finish().unwrap().to_vec();

        let (mut m, slot) = store_mgr("req-a", 103);
        m.store_init(slot, &[], 103, 3).unwrap();
        let (verdicts, packed) = m
            .update_step_pack_store(&[slot], &[2], &[5, 6], 1000, 0, 1.5, &[], true)
            .unwrap();
        assert!(packed);
        assert_eq!(verdicts[0].num_accepted, 2);
        assert_eq!(m.raw.finish().unwrap(), &want[..]);
        assert_eq!(m.tokens.counts(slot), Some((105, 5)));
        assert_eq!(
            m.cache_fold_skips, 1,
            "no resident table entry for this slot -- counted, not silent"
        );
    }

    #[test]
    fn cache_fold_matches_explicit_cache_blocks() {
        // Same scenario driven two ways: `fold_cache=true` folds `cache_blocks` into the
        // packed step; `fold_cache=false` plus an explicit call afterwards reproduces BY
        // HAND what Python does today (`AsyncScheduler._update_request_with_output`'s
        // trailing `cache_blocks(num_computed_tokens - num_output_placeholders)`,
        // async_scheduler.py:71-74). Both must leave the pool in an IDENTICAL state --
        // proven by a third, freshly-admitted request sharing the exact prefix seeing the
        // exact same cache hit.
        fn run(fold_cache: bool) -> (usize, usize) {
            let mut m = Manager::new(hybrid_cfg(256)).unwrap();
            m.store_arm(crate::hash::none_hash_from_seed("0"));
            let hs = chain(4, 1);
            let a = m.intern("a");
            // Prompt: 48 tokens (3 full blocks), admitted and cached in one shot.
            m.push_hashes(a, &packed(&hs[..3]), 48);
            assert_eq!(m.get_computed_blocks(a, 48, 0, false), 0);
            assert!(m
                .allocate_slots(a, 48, 0, true, 0, 0, 48, STATUS_WAITING, false)
                .unwrap());
            m.store_init(a, &[], 48, 0).unwrap();
            // Schedule-time allocation for the 16-token burst -- `num_request_tokens` is
            // still 48 here (Python's `request.num_tokens` before the output arrives), so
            // this allocates the 4th physical block WITHOUT caching it yet (matching
            // `allocate_slots`'s own `min(total_computed + new, num_request_tokens)` clip).
            assert!(m
                .allocate_slots(a, 16, 0, false, 0, 48, 48, STATUS_RUNNING, true)
                .unwrap());
            // The decode-produced 4th block's hash, pushed the way `RustMirror.slot()`
            // would once the block hasher has it -- BEFORE the FFI crossing that applies
            // the tokens, exactly as `decide()` orders it.
            m.push_hashes(a, &packed(&hs[3..]), 48);

            // A resident table entry as it would read right after schedule() advanced it
            // for a committed 16-token burst this step: computed already at 64 (48 prompt
            // + 16 decode), placeholders still at 16 -- NOT yet decremented by this call.
            m.table_set(
                a,
                SchedEntry {
                    slot: a,
                    num_tokens: 48,
                    num_tokens_with_spec: 48,
                    num_computed_tokens: 64,
                    num_output_placeholders: 16,
                    num_prompt_tokens: 48,
                    max_tokens: 1000,
                    status: STATUS_RUNNING,
                    num_preemptions: 0,
                    is_prefill_chunk: false,
                    skip_reading_prefix_cache: false,
                },
            );
            m.stops.set(
                a,
                crate::update::StopParams {
                    min_tokens: 0,
                    max_tokens: 1000,
                    eos_token_id: Some(999_999),
                    stop_token_ids: Default::default(),
                },
            );
            let decode_tokens: Vec<i64> = (500..516).collect();
            let (verdicts, packed_ok) = m
                .update_step_pack_store(
                    &[a], &[16], &decode_tokens, 100_000, 0, 0.0, &[], fold_cache,
                )
                .unwrap();
            assert!(packed_ok);
            assert_eq!(verdicts[0].num_accepted, 16, "no stop -- every token kept");
            if !fold_cache {
                // What Python's per-request trailing call does today: same subtraction,
                // same argument, one FFI crossing later instead of inside this one.
                let e = m.table_get(a).unwrap();
                let n = e.num_computed_tokens - e.num_output_placeholders;
                m.cache_blocks(a, n);
            }

            // A fresh, longer request sharing the exact 64-token prefix.
            let hs5 = {
                let mut v = hs.clone();
                v.push([9; HASH_LEN]);
                v
            };
            let c = m.intern("c");
            m.push_hashes(c, &packed(&hs5), 80);
            let hit = m.get_computed_blocks(c, 80, 0, false);
            (hit, m.coord.pool.get_num_free_blocks())
        }

        let folded = run(true);
        let explicit = run(false);
        assert_eq!(folded, explicit, "fold_cache=true must match an explicit cache_blocks");
        assert_eq!(folded.0, 64, "the decode-completed 4th block must be cache-visible");
    }

    #[test]
    fn cache_fold_skips_non_running_and_counts() {
        // `table_with`'s silent skip (`:315`) is correct-by-construction for its existing
        // callers; folding `cache_blocks` in leans on the SAME silence, so this is the loud
        // version -- `cache_fold_skips` must count every slot the fold could not apply to.
        let (mut m, a) = store_mgr("req-a", 10);

        // No resident table entry at all yet.
        let (_, packed) = m
            .update_step_pack_store(&[a], &[1], &[7], 1000, 0, 0.0, &[], true)
            .unwrap();
        assert!(packed);
        assert_eq!(m.cache_fold_skips, 1);

        // A resident entry that exists but is not RUNNING (e.g. preempted this step) is
        // the same hazard: skip and count, never fold into a stale/wrong slot.
        m.table_set(
            a,
            SchedEntry {
                slot: a,
                status: STATUS_PREEMPTED,
                num_computed_tokens: 11,
                num_output_placeholders: 0,
                ..SchedEntry::default()
            },
        );
        let (_, packed) = m
            .update_step_pack_store(&[a], &[1], &[8], 1000, 0, 0.0, &[], true)
            .unwrap();
        assert!(packed);
        assert_eq!(m.cache_fold_skips, 2);

        // A genuinely RUNNING resident entry does NOT skip.
        m.table_set(
            a,
            SchedEntry {
                slot: a,
                status: STATUS_RUNNING,
                num_computed_tokens: 12,
                num_output_placeholders: 0,
                ..SchedEntry::default()
            },
        );
        m.update_step_pack_store(&[a], &[1], &[9], 1000, 0, 0.0, &[], true)
            .unwrap();
        assert_eq!(m.cache_fold_skips, 2, "a RUNNING entry folds -- no new skip");
    }

    #[test]
    fn the_store_path_truncates_at_a_stop_everywhere() {
        let (mut m, slot) = store_mgr("req-a", 10);
        // EOS (2) is the second of three -> record, counters and store all keep 2.
        let (verdicts, packed) = m
            .update_step_pack_store(&[slot], &[3], &[5, 2, 9], 1000, 0, 0.0, &[], false)
            .unwrap();
        assert_eq!(verdicts[0].num_accepted, 2);
        assert!(packed);
        assert_eq!(u32::from_le_bytes(m.raw.buf[28..32].try_into().unwrap()), 2);
        assert_eq!(m.tokens.counts(slot), Some((12, 2)));
        assert_eq!(m.slot_tokens(slot), Some(&[5i64, 2][..]), "the rejected 9 is gone");
    }

    #[test]
    fn a_refused_slot_leaves_the_store_untouched() {
        // Python is about to materialize the whole step; appending here would double-count.
        let (mut m, a) = store_mgr("req-a", 10);
        let b = m.intern("req-b");
        m.store_init(b, &[], 10, 0).unwrap(); // ...but no stop params
        let (verdicts, packed) = m
            .update_step_pack_store(&[a, b], &[1, 1], &[7, 7], 1000, 0, 0.0, &[], false)
            .unwrap();
        assert_eq!(verdicts[1].status, u8::MAX);
        assert!(!packed);
        assert_eq!(m.tokens.counts(a), Some((10, 0)), "no append on a refused step");
        assert_eq!(m.slot_tokens(a), Some(&[][..]));
    }

    #[test]
    fn the_store_path_refuses_a_slot_it_never_took_over() {
        let (mut m, _a) = store_mgr("req-a", 10);
        let b = m.intern("req-b");
        m.stops.set(b, stop_params());
        assert!(m
            .update_step_pack_store(&[b], &[1], &[7], 1000, 0, 0.0, &[], false)
            .is_err());
        // Arity and flat-length mismatches are Errs, not panics or short records.
        let (mut m2, a2) = store_mgr("req-a", 10);
        assert!(m2
            .update_step_pack_store(&[a2], &[1, 1], &[7], 1000, 0, 0.0, &[], false)
            .is_err());
        assert!(m2
            .update_step_pack_store(&[a2], &[2], &[7], 1000, 0, 0.0, &[], false)
            .is_err());
    }

    #[test]
    fn the_store_hash_chain_matches_push_hashes_over_the_same_tokens() {
        // hash_block_size is 16 here. A request with a 16-token prompt has one pushed hash
        // and an empty tail; 16 decode tokens then complete exactly one more block, and it
        // must equal what the reference hasher produces over the whole 32-token sequence.
        let nh = crate::hash::none_hash_from_seed("0");
        let all: Vec<i64> = (100..132).collect();
        let want = crate::hash::hash_request_tokens(&nh, 16, &all, None);
        assert_eq!(want.len(), 2);

        let mut m = Manager::new(hybrid_cfg(64)).unwrap();
        m.store_arm(nh);
        let slot = m.intern("req-a");
        m.stops.set(
            slot,
            crate::update::StopParams {
                min_tokens: 0,
                max_tokens: 1000,
                eos_token_id: None,
                stop_token_ids: Default::default(),
            },
        );
        m.push_hashes(slot, &packed(&[want[0]]), 16);
        assert_eq!(m.num_hashes(slot), 1);
        m.store_init(slot, &[], 16, 0).unwrap();
        for tok in &all[16..] {
            m.update_step_pack_store(&[slot], &[1], &[*tok], 100000, 0, 0.0, &[], false)
                .unwrap();
        }
        assert_eq!(m.num_hashes(slot), 2, "one new full block");
        let hashes = m.slot_hashes(slot).unwrap();
        assert_eq!(hashes.len(), 64);
        assert_eq!(&hashes[..32], &want[0][..]);
        assert_eq!(&hashes[32..], &want[1][..], "the chain must match the reference hasher");
        assert_eq!(m.tokens.counts(slot), Some((32, 16)));
    }

    #[test]
    fn store_apply_agrees_with_the_incremental_pack_path() {
        // `store_apply` is the batch token-append `update_step_pack` calls internally.
        // Driving 16 tokens through it in ONE call must land the same chain and the same
        // counters as 16 incremental `update_step_pack_store` steps.
        let (mut auth, a) = store_mgr("req-a", 16);
        for t in 200..216i64 {
            auth.update_step_pack_store(&[a], &[1], &[t], 100000, 0, 0.0, &[], false)
                .unwrap();
        }
        let (mut batch, b) = store_mgr("req-a", 16);
        let toks: Vec<i64> = (200..216).collect();
        let acc = vec![16u32];
        batch.store_apply(&[b], &[0, 16], &toks, &acc).unwrap();
        assert_eq!(auth.slot_hashes(a), batch.slot_hashes(b));
        assert_eq!(auth.tokens.counts(a), batch.tokens.counts(b));
        assert_eq!(auth.slot_tokens(a), batch.slot_tokens(b));
        // Out-of-range accepted counts are refused rather than slicing past the batch.
        assert!(batch.store_apply(&[b], &[0, 1], &toks, &[2]).is_err());
        assert!(batch.store_apply(&[b], &[0, 1], &toks, &[1, 1]).is_err());
    }

    #[test]
    fn forget_clears_the_store_so_a_recycled_slot_starts_clean() {
        let (mut m, a) = store_mgr("req-a", 10);
        m.update_step_pack_store(&[a], &[1], &[7], 1000, 0, 0.0, &[], false)
            .unwrap();
        assert_eq!(m.tokens.counts(a), Some((11, 1)));
        m.forget("req-a");
        assert_eq!(m.tokens.counts(a), None);
        let b = m.intern("req-b");
        assert_eq!(b, a, "the slot is recycled -- that is the hazard being tested");
        assert!(m.slot_tokens(b).is_none());
        assert_eq!(m.num_hashes(b), 0, "forget also cleared the chain");
    }

    #[test]
    fn runner_owns_catches_a_slot_that_changed_hands() {
        // The update-time commit's real hazard: between the launch and the commit the
        // request finished, its slot was freed, and `schedule(k + 1)` handed the SAME slot
        // to a new request. `runner_packable` cannot see that -- the slot has stop params
        // and a live name again -- so the commit needs the names it planned against.
        let (mut m, a) = store_mgr("req-a", 10);
        let names = vec!["req-a".to_string()];
        assert!(m.runner_packable(&[a]));
        assert!(m.runner_owns(&[a], &names));
        m.forget("req-a");
        assert!(!m.runner_owns(&[a], &names), "a freed slot owns nothing");
        let b = m.intern("req-b");
        m.stops.set(b, stop_params());
        assert_eq!(b, a, "the slot is recycled -- that is the hazard being tested");
        assert!(m.runner_packable(&[b]), "packable says yes to the NEW request");
        assert!(!m.runner_owns(&[b], &names), "...but it is not the one we launched for");
        assert!(!m.runner_owns(&[b], &[]), "arity mismatches are a decline, not a panic");
    }

    #[test]
    fn add_time_registration_interns_both_halves_and_dies_with_the_request() {
        let mut m = Manager::new(hybrid_cfg(64)).unwrap();
        assert_eq!(m.request_meta(0), None, "nothing is registered yet");

        let a = m.set_request_meta("req-a", true, stop_params());
        assert_eq!(m.request_meta(a), Some(true));
        assert!(m.stops.has(a), "one call must set BOTH halves");
        // Idempotent per request id: `RustMirror.slot()` must resolve the same slot later.
        assert_eq!(m.intern("req-a"), a);
        assert_eq!(m.set_request_meta("req-a", false, stop_params()), a);
        assert_eq!(m.request_meta(a), Some(false), "a re-register overwrites");

        // `forget` is the one teardown chokepoint: a recycled slot must answer `None`, not
        // the dead request's bit -- otherwise `decide()` would skip a derivation the new
        // occupant never had run for it.
        m.forget("req-a");
        assert_eq!(m.request_meta(a), None);
        let b = m.intern("req-b");
        assert_eq!(b, a, "the slot is recycled -- that is the hazard being tested");
        assert_eq!(m.request_meta(b), None);
    }

    #[test]
    fn an_unarmed_store_refuses_instead_of_hashing_against_a_guess() {
        let mut m = Manager::new(hybrid_cfg(64)).unwrap();
        let a = m.intern("req-a");
        assert!(m.store_init(a, &[], 10, 0).is_err());
    }
}
