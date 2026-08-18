//! Speculative precompute: run the next `schedule()` on a background thread while the
//! engine is still busy with the current step, so the scheduler's latency on the critical
//! path collapses to a dictionary build.
//!
//! THE SHAPE. Python knows the running set the moment `update_from_output` finishes and,
//! for a pure decode step, nothing between then and the next `schedule()` changes what the
//! scheduler will decide. So it `kick()`s: the running slot order and a generation counter
//! go to a parked worker, which runs the REAL `schedule_resident` and parks again. The
//! next `schedule()` calls `take_speculative(gen, slots)`; if the generation and the slot
//! order still match, the already-computed decisions come straight back.
//!
//! WHY THE REAL STATE AND NOT A COPY. Every block id in the decisions comes out of the
//! free queue's exact pop order and the prefix cache's exact eviction order. A run against
//! a copy is only usable if the copy becomes the state on a hit — which means deep-cloning
//! the arena and the index every kick. So the speculative run mutates the real
//! [`Manager`], and an undo journal (`journal.rs`) makes that reversible. On a hit the
//! journal is DISCARDED: the speculative run WAS the real run. On a miss it is
//! reverse-applied and the state is bit-identical to before.
//!
//! THE THREE INVARIANTS. Break any of them and the engine serves corrupted KV state:
//!
//!   1. MUTUAL EXCLUSION. `manager`, `core` and `spec` live behind ONE `Mutex`. The worker
//!      holds it for the whole speculative run, so no Python call can interleave with a
//!      half-applied schedule.
//!   2. INVALIDATE-BEFORE-MUTATE. Every PyO3 entry point that changes state calls
//!      [`Shared::invalidate`] first: a pending speculation is rolled back and dropped
//!      before the mutation lands. That is also what guarantees the journal is disarmed
//!      everywhere except inside a speculative run — the scope invariant `journal.rs`
//!      relies on to bound its recording sites.
//!   3. EXACT CONSUME. `take_speculative` returns the stored decisions only if the
//!      generation, the slot order AND the params are identical to the kick. Anything else
//!      is a miss: roll back, return `None`, let the caller schedule directly.
//!
//! REFUSE, DON'T APPROXIMATE. The worker aborts a speculative run — rolling back what it
//! recorded and counting a miss — when the waiting queue is non-empty (its prefix-cache
//! lookup paths are not journaled), when the journal outgrows its soft cap, when the
//! scheduling loop returns an error, or when it panics. Speculation is a latency
//! optimisation; every one of those paths falls back to the direct call, which is always
//! correct.
//!
//! FAIL-OPEN. A panic in the worker is caught while the lock is still held, rolled back,
//! and permanently disables speculation. A poisoned mutex (a panic on the Python side) is
//! recovered with `into_inner` and also disables speculation. In both cases the direct
//! path keeps serving.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;

use crate::manager::Manager;
use crate::sched::{Decisions, Params, ScheduleCore};

/// The mutable core. Everything a scheduling step touches, under one lock.
pub struct Shared {
    pub manager: Manager,
    pub core: ScheduleCore,
    pub spec: SpecState,
    /// Batch 3: the crate-owned output channel, opened by `KvManager::out_open`. Lives here
    /// (rather than on `KvManager` directly) so `update_step_pack_np`'s `py.allow_threads`
    /// closure -- which already holds this same lock for the pack -- can publish without a
    /// second mutex acquisition. `None` until `out_open` succeeds; stays `None` forever
    /// without the `shm` cargo feature.
    #[cfg(feature = "shm")]
    pub out: Option<crate::out::OutChannel>,
}

#[derive(Default)]
pub struct SpecState {
    pub pending: Option<PendingSpec>,
    pub gen_at_kick: u64,
    pub hits: u64,
    pub misses: u64,
    pub rollbacks: u64,
    /// Set forever after a worker panic or a poisoned lock. Direct scheduling continues.
    pub disabled: bool,
}

pub struct PendingSpec {
    pub running_slots: Vec<u32>,
    pub params: Params,
    /// Snapshotted so a later direct call cannot clobber `core.decisions` underneath it.
    pub decisions: Decisions,
}

impl Shared {
    pub fn new(manager: Manager) -> Self {
        Shared {
            manager,
            core: ScheduleCore::new(),
            spec: SpecState::default(),
            #[cfg(feature = "shm")]
            out: None,
        }
    }

    /// Invariant 2. Call this at the top of every state-mutating entry point.
    #[inline]
    pub fn invalidate(&mut self) {
        if self.spec.pending.is_some() {
            self.rollback_pending();
        }
    }

    pub fn rollback_pending(&mut self) {
        self.spec.pending = None;
        self.manager.rollback_journal();
        self.spec.rollbacks += 1;
    }

    /// Invariant 3. `Some` hands back exactly what the worker computed and COMMITS the
    /// speculative mutations (the journal is dropped, not applied).
    pub fn take_speculative(
        &mut self,
        generation: u64,
        running_slots: &[u32],
        params: &Params,
    ) -> Option<Decisions> {
        let Some(p) = self.spec.pending.as_ref() else {
            self.spec.misses += 1;
            return None;
        };
        if generation != self.spec.gen_at_kick
            || p.running_slots.as_slice() != running_slots
            || &p.params != params
        {
            self.rollback_pending();
            self.spec.misses += 1;
            return None;
        }
        let p = self.spec.pending.take().unwrap();
        self.manager.commit_journal();
        self.spec.hits += 1;
        Some(p.decisions)
    }
}

/// One unit of work for the worker.
struct KickMsg {
    generation: u64,
    running_slots: Vec<u32>,
    params: Params,
}

#[derive(Default)]
struct Mailbox {
    msg: Option<KickMsg>,
    stop: bool,
}

/// Owns the worker thread. Dropping it asks the worker to exit; it is never joined —
/// the process is tearing down and the worker holds nothing but an `Arc`.
pub struct SpecDriver {
    mail: Arc<(Mutex<Mailbox>, Condvar)>,
    /// Mirrors `SpecState::disabled` without needing the big lock, so `kick` can bail out
    /// of even trying.
    disabled: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl SpecDriver {
    /// Spawn the worker. Called lazily on the first `kick` so an engine that never
    /// speculates never pays for a thread.
    pub fn spawn(shared: Arc<Mutex<Shared>>) -> std::io::Result<Self> {
        let mail: Arc<(Mutex<Mailbox>, Condvar)> = Arc::new(Default::default());
        let disabled = Arc::new(AtomicBool::new(false));
        let (w_mail, w_disabled) = (mail.clone(), disabled.clone());
        let handle = std::thread::Builder::new()
            .name("vtl-sched-spec".into())
            .spawn(move || worker(shared, w_mail, w_disabled))?;
        Ok(SpecDriver {
            mail,
            disabled,
            handle: Some(handle),
        })
    }

    #[inline]
    pub fn is_disabled(&self) -> bool {
        self.disabled.load(Ordering::Relaxed)
    }

    /// Hand the worker a job and return immediately. A job still sitting in the mailbox
    /// is overwritten — the newer running set is the one worth speculating on.
    pub fn kick(&self, generation: u64, running_slots: Vec<u32>, params: Params) {
        let (lock, cv) = &*self.mail;
        let mut m = lock.lock().unwrap_or_else(|e| e.into_inner());
        m.msg = Some(KickMsg {
            generation,
            running_slots,
            params,
        });
        drop(m);
        cv.notify_one();
    }
}

impl Drop for SpecDriver {
    fn drop(&mut self) {
        let (lock, cv) = &*self.mail;
        {
            let mut m = lock.lock().unwrap_or_else(|e| e.into_inner());
            m.stop = true;
        }
        cv.notify_one();
        // Deliberately not joined: the worker may be mid-run holding the state lock, and
        // blocking a Python GC pass on it buys nothing at teardown.
        let _ = self.handle.take();
    }
}

/// Parked on a condvar, never busy-spinning: the box has 3 vCPUs and a spinning worker
/// would take one of them away from the engine for the entire step.
fn worker(shared: Arc<Mutex<Shared>>, mail: Arc<(Mutex<Mailbox>, Condvar)>, disabled: Arc<AtomicBool>) {
    let (lock, cv) = &*mail;
    loop {
        let msg = {
            let mut m = lock.lock().unwrap_or_else(|e| e.into_inner());
            while m.msg.is_none() && !m.stop {
                m = cv.wait(m).unwrap_or_else(|e| e.into_inner());
            }
            if m.stop {
                return;
            }
            m.msg.take().unwrap()
        };
        if disabled.load(Ordering::Relaxed) {
            continue;
        }
        let mut sh = shared.lock().unwrap_or_else(|e| e.into_inner());
        // catch_unwind INSIDE the guard: unwinding out of it would poison the mutex and
        // strand the half-applied journal with no way to reverse it.
        let ok = catch_unwind(AssertUnwindSafe(|| run_speculative(&mut sh, msg))).is_ok();
        if !ok {
            sh.manager.rollback_journal();
            sh.spec.pending = None;
            sh.spec.disabled = true;
            disabled.store(true, Ordering::Relaxed);
        }
    }
}

fn run_speculative(sh: &mut Shared, msg: KickMsg) {
    if sh.spec.disabled {
        return;
    }
    // A kick that lands on top of an un-consumed one: the old speculation is worthless
    // and its journal must go back before a new one is armed.
    sh.invalidate();

    // `ScheduleCore::inconsistent` is a DIAGNOSTIC counter, not scheduling state, so the
    // journal does not record it and a rollback cannot put it back. Left alone, a
    // speculative run that skips a broken entry bumps it, and the direct run that redoes
    // the same step bumps it again — the probe in `rust_sched.py` would read two skips for
    // one broken entry, on the one branch it can afford to sit on.
    //
    // Restored UNCONDITIONALLY, including on the hit path, and that undercounts exactly one
    // case: a consumed speculation IS the real step (`take_speculative` drops the journal
    // rather than replaying the loop), so its skips are never re-counted by anyone. That is
    // the right trade for what the counter is for. It is a wedge probe, read only on steps
    // that scheduled nothing at all, and an entry broken enough to be skipped stays broken
    // — the next DIRECT run counts it. Speculation is also only ever kicked on a healthy
    // quiet step (clean table, empty waiting queue, no bail), which is the least likely
    // place for the skip to originate. Getting the arithmetic right on the rollback path
    // matters more than the last count on the hit path.
    let saved = sh.core.inconsistent;

    sh.manager.arm_journal();
    let res = sh
        .core
        .schedule_resident(&mut sh.manager, &msg.running_slots, &[], &msg.params)
        .cloned();
    match res {
        Ok(decisions) => {
            sh.spec.gen_at_kick = msg.generation;
            sh.spec.pending = Some(PendingSpec {
                running_slots: msg.running_slots,
                params: msg.params,
                decisions,
            });
        }
        Err(_) => {
            // Refused (non-empty waiting, journal over cap) or a genuine scheduling
            // error. Either way the direct path will redo it and report properly.
            sh.manager.rollback_journal();
            sh.spec.misses += 1;
        }
    }
    sh.core.inconsistent = saved;
}

/// Lock the shared state, recovering from (and permanently disabling speculation after) a
/// poisoned mutex. Fail-open: a poisoned lock must not take the engine down.
pub fn lock_shared(shared: &Mutex<Shared>) -> MutexGuard<'_, Shared> {
    match shared.lock() {
        Ok(g) => g,
        Err(poisoned) => {
            let mut g = poisoned.into_inner();
            if !g.spec.disabled {
                g.spec.disabled = true;
                // The panic happened with the lock held, so a journal may be armed and
                // half-applied. Reverse what was recorded and never speculate again.
                g.spec.pending = None;
                g.manager.rollback_journal();
            }
            g
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::{Config, GroupConfig};
    use crate::manager::STATUS_RUNNING;
    use crate::sched::SchedReq;
    use crate::single_type::Kind;

    fn assert_send<T: Send>() {}

    #[test]
    fn core_state_is_send() {
        // `py.allow_threads` and the spec worker both need this; a `Rc` sneaking into any
        // of these types would break both at once.
        assert_send::<Manager>();
        assert_send::<ScheduleCore>();
        assert_send::<Shared>();
        assert_send::<Decisions>();
    }

    fn cfg() -> Config {
        Config {
            num_blocks: 512,
            enable_caching: true,
            max_model_len: 4096,
            scheduler_block_size: 16,
            hash_block_size: 16,
            log_stats: false,
            watermark: 0.0,
            radix: false,
            groups: vec![GroupConfig {
                kind: Kind::FullAttention,
                block_size: 16,
                is_full_attention: true,
                spec_signature: "full".into(),
                mamba_align: false,
                num_speculative_blocks: 0,
                use_eagle: false,
            }],
        }
    }

    fn params() -> Params {
        Params {
            max_num_scheduled_tokens: 8192,
            max_num_running_reqs: 64,
            max_model_len: 4096,
            num_sampled_tokens_per_step: 1,
            long_prefill_token_threshold: 0,
            enable_chunked_prefill: true,
            need_mamba_block_aligned_split: false,
            cache_block_size: 16,
            num_lookahead_tokens: 0,
            sjf_reorder: false,
            sjf_usage_tight: 0.90,
            lean_decisions: false,
            burst_max_reqs: 0,
        }
    }

    /// The `inconsistent` counter is a wedge probe, not scheduling state: the journal does
    /// not record it, so a speculative run that bumps it and is then rolled back would leave
    /// the bump behind and the direct re-run would count the SAME broken entry twice.
    #[test]
    fn a_speculative_run_does_not_double_count_an_inconsistent_entry() {
        let mut kv = Manager::new(cfg()).unwrap();
        let slot = kv.intern("broken");
        // C past T + P is the skew `ScheduleCore::run`'s `checked_sub` refuses to schedule
        // against: it counts the entry and skips it.
        kv.table_set(
            slot,
            SchedReq {
                slot,
                num_tokens: 65,
                num_tokens_with_spec: 65,
                num_computed_tokens: 100,
                num_output_placeholders: 0,
                num_prompt_tokens: 64,
                max_tokens: 128,
                status: STATUS_RUNNING,
                num_preemptions: 0,
                is_prefill_chunk: false,
                skip_reading_prefix_cache: false,
            },
        );
        let mut sh = Shared::new(kv);

        // A DIRECT run counts it -- that is the behaviour the Python probe reads.
        sh.core
            .schedule_resident(&mut sh.manager, &[slot], &[], &params())
            .unwrap();
        assert_eq!(sh.core.inconsistent, 1);

        // A speculative run over the same entry must leave the count exactly where the
        // direct runs put it, whether or not it is ever consumed.
        run_speculative(
            &mut sh,
            KickMsg {
                generation: 1,
                running_slots: vec![slot],
                params: params(),
            },
        );
        assert_eq!(
            sh.core.inconsistent, 1,
            "a speculative run must not add to the skip count"
        );
        assert!(sh.spec.pending.is_some(), "the speculation itself still stands");

        // ...and the rollback the miss path takes does not disturb it either.
        sh.rollback_pending();
        assert_eq!(sh.core.inconsistent, 1);

        // The next direct run is what counts the still-broken entry again: the counter stays
        // monotone and an entry that keeps being skipped keeps being reported.
        sh.core
            .schedule_resident(&mut sh.manager, &[slot], &[], &params())
            .unwrap();
        assert_eq!(sh.core.inconsistent, 2);
    }
}
