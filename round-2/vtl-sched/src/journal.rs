//! Exact-undo journal for the speculative scheduling run (`spec.rs`).
//!
//! WHY A JOURNAL AND NOT A CLONE. The speculative run has to mutate the REAL [`Manager`]:
//! every block id it hands out comes from the exact pop order of the free queue and the
//! exact eviction order of the prefix cache, so a run against a copy would only be
//! bit-identical if the copy were promoted to the real state on a hit — which means
//! deep-cloning the whole arena (`num_blocks` * `Block`, plus the prefix-cache index) on
//! EVERY kick. On the served pool that is megabytes per step. Journaling the handful of
//! cells a decode step actually touches is two orders of magnitude cheaper.
//!
//! WHY "OLD VALUE" AND NOT "INVERSE OP". Every entry records the value a cell held
//! immediately BEFORE it was overwritten. Rolling back is one reversed pass that stores
//! each recorded value back. This is exact by construction and needs no reasoning about
//! whether an operation's inverse exists (`HashIndex`'s `One`->`Many` promotion, for
//! instance, has no exact inverse expressible as a `pop`), and duplicates are harmless:
//! the LAST value written back for a cell is the FIRST one that was recorded, i.e. the
//! pre-speculation value. Ordering only matters within a container, and one reversed pass
//! over a single op list preserves it everywhere.
//!
//! SCOPE INVARIANT — read this before adding a recording site. The journal is armed
//! **only** around [`crate::sched::ScheduleCore::schedule_resident`] driven by the spec
//! thread, with an EMPTY waiting queue (`run()` refuses an armed run otherwise). Every
//! PyO3 entry point that mutates state rolls back and disarms first, so no other code path
//! can ever run armed. That is what bounds the recording sites to the running-queue half
//! of the scheduling loop:
//!
//!   * `new_step_starts`         -> mamba `cached_blocks_this_step`
//!   * `allocate_slots`          -> `remove_skipped_blocks`, `allocate_new_blocks`,
//!                                  `cache_blocks` and everything they reach in the pool
//!   * the preemption path       -> `free` (block release + per-request map removal)
//!   * the epilogue              -> `get_num_common_prefix_blocks`'s defaultdict touch
//!   * the post-schedule commit  -> the resident request table
//!
//! The prefix-cache LOOKUP half (`get_computed_blocks`, `add_local_computed_blocks`,
//! `pool.touch`, the SJF reorder, `prefix_cache_stats`) is unreachable under that
//! invariant and is deliberately NOT journaled. If speculation is ever extended to a
//! non-empty waiting queue, those sites must be journaled first — the refusal in `run()`
//! is what makes that a compile-time-visible choice rather than a silent corruption.

use rustc_hash::FxHashSet;
use smallvec::SmallVec;

use crate::block_pool::{Block, HashKey, IndexSnapshot};
use crate::manager::{Manager, SchedEntry};

/// Soft cap on recorded operations. Not a correctness bound (recording continues past it
/// so a rollback in flight stays exact) — it is the signal `run()` polls to abandon a
/// speculative run that is touching far more state than a decode step ever should.
pub const SOFT_CAP: usize = 1 << 16;

/// One cell's pre-mutation value. Group ids and request slots are `u32` throughout.
#[derive(Debug)]
pub enum Undo {
    // ---- block_pool -------------------------------------------------------
    /// `arena.blocks[id]` — covers ref counts, hashes AND the free-queue links, which is
    /// why every arena write funnels through `BlockArena::get_mut`.
    Block(u32, Block),
    FreeCount(usize),
    /// Whatever the prefix-cache index mapped `key` to (representation included).
    Index(HashKey, IndexSnapshot),
    CachedByBlock(u32, Option<SmallVec<[HashKey; 2]>>),

    // ---- single_type (per KV cache group) ---------------------------------
    ReqBlocksLen(u32, u32, usize),
    ReqBlocksSet(u32, u32, usize, u32),
    /// The `defaultdict(list)` entry did not exist before.
    ReqBlocksInsert(u32, u32),
    ReqBlocksRemove(u32, u32, Vec<u32>),
    NumCachedBlock(u32, u32, Option<usize>),
    NewBlockIdsLen(u32, usize),
    /// Whole-set snapshot: `new_step_starts` clears it, so there is nothing finer to record.
    CachedThisStep(u32, FxHashSet<HashKey>),
    CachedThisStepInsert(u32, HashKey, bool),
    LastStateIdx(u32, u32, Option<usize>),
    AllocatedReq(u32, u32, bool),

    // ---- manager ----------------------------------------------------------
    HitLen(u32, Option<usize>),
    TableEntry(u32, Option<SchedEntry>),
}

#[derive(Default, Debug)]
pub struct Journal {
    ops: Vec<Undo>,
}

impl Journal {
    #[inline]
    pub fn push(&mut self, op: Undo) {
        self.ops.push(op);
    }

    #[inline]
    pub fn over_soft_cap(&self) -> bool {
        self.ops.len() > SOFT_CAP
    }

    /// Reverse-apply every recorded value. `m` must be the manager the journal was armed
    /// on; afterwards its [`Manager::state_fingerprint`] equals the pre-arm one.
    pub fn rollback(self, m: &mut Manager) {
        for op in self.ops.into_iter().rev() {
            match op {
                Undo::Block(id, b) => m.coord.pool.arena.blocks[id as usize] = b,
                Undo::FreeCount(n) => m.coord.pool.arena.num_free_blocks = n,
                Undo::Index(key, snap) => m.coord.pool.cached.restore(&key, snap),
                Undo::CachedByBlock(id, v) => match v {
                    Some(v) => {
                        m.coord.pool.cached_by_block.insert(id, v);
                    }
                    None => {
                        m.coord.pool.cached_by_block.remove(&id);
                    }
                },

                Undo::ReqBlocksLen(g, req, n) => {
                    if let Some(v) = m.coord.managers[g as usize].req_to_blocks.get_mut(&req) {
                        v.truncate(n);
                    }
                }
                Undo::ReqBlocksSet(g, req, idx, old) => {
                    if let Some(v) = m.coord.managers[g as usize].req_to_blocks.get_mut(&req) {
                        v[idx] = old;
                    }
                }
                Undo::ReqBlocksInsert(g, req) => {
                    m.coord.managers[g as usize].req_to_blocks.remove(&req);
                }
                Undo::ReqBlocksRemove(g, req, v) => {
                    m.coord.managers[g as usize].req_to_blocks.insert(req, v);
                }
                Undo::NumCachedBlock(g, req, old) => {
                    let t = &mut m.coord.managers[g as usize].num_cached_block;
                    match old {
                        Some(n) => {
                            t.insert(req, n);
                        }
                        None => {
                            t.remove(&req);
                        }
                    }
                }
                Undo::NewBlockIdsLen(g, n) => {
                    m.coord.managers[g as usize].new_block_ids.truncate(n)
                }
                Undo::CachedThisStep(g, set) => {
                    m.coord.managers[g as usize].cached_blocks_this_step = set
                }
                Undo::CachedThisStepInsert(g, key, was_present) => {
                    if !was_present {
                        m.coord.managers[g as usize]
                            .cached_blocks_this_step
                            .remove(&key);
                    }
                }
                Undo::LastStateIdx(g, req, old) => {
                    let t = &mut m.coord.managers[g as usize].last_state_block_idx;
                    match old {
                        Some(n) => {
                            t.insert(req, n);
                        }
                        None => {
                            t.remove(&req);
                        }
                    }
                }
                Undo::AllocatedReq(g, req, was_present) => {
                    let t = &mut m.coord.managers[g as usize].allocated_block_reqs;
                    if was_present {
                        t.insert(req);
                    } else {
                        t.remove(&req);
                    }
                }

                Undo::HitLen(req, old) => match old {
                    Some(n) => {
                        m.hit_len.insert(req, n);
                    }
                    None => {
                        m.hit_len.remove(&req);
                    }
                },
                Undo::TableEntry(slot, old) => m.table_restore(slot, old),
            }
        }
    }
}
