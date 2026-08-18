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
//!   4. NO RUN INSIDE THE PHASE-SPLIT WINDOW. Stage S's connector step is two locked calls
//!      with a Python call in between, so mutual exclusion does not cover it: the worker
//!      declines while [`Shared::phase_split_open`] is set, and phase 2 refuses to run if
//!      the window was closed underneath it. See that field for what a run in the gap does.
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
    /// Stage S: a connector step has run phase 1 and is waiting for phase 2.
    ///
    /// THE FOURTH INVARIANT, and the one the other three do not cover. Mutual exclusion
    /// holds the lock for the duration of a CALL; the connector step is two calls with a
    /// Python call (`get_num_new_matched_tokens`) in between, and the lock is FREE for that
    /// whole window. A speculation kicked before the window opened is picked up by the
    /// worker inside it and runs `schedule_resident` on the SAME `ScheduleCore` — clearing
    /// `decisions`, clearing the recorded probes (`Manager::probe_clear`) and overwriting
    /// the carried token budget — after which phase 2 admits against a state that is no
    /// longer the one phase 1 left, with no probe to consume and a budget from a different
    /// step. Invariant 2 does not help: `run_schedule_phase2`'s own `invalidate()` would
    /// roll the speculation back and disarm the journal BEFORE
    /// `refuse_split_under_speculation` could see it, so the refusal never fires.
    ///
    /// So the window is explicit. The worker declines while it is open ([`run_speculative`])
    /// and phase 2 refuses to run when it is not ([`Shared::claim_phase_split`]) — the
    /// second half is what turns any OTHER intervening mutation into a loud error instead
    /// of a silently mis-scheduled step.
    ///
    /// `rust_sched.py::spec_blocked` also refuses to kick at all under a live connector, so
    /// in the shipped configuration the worker is never even offered the job; this is the
    /// layer that does not depend on Python getting that right.
    pub phase_split_open: bool,
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
            phase_split_open: false,
            #[cfg(feature = "shm")]
            out: None,
        }
    }

    /// Phase 1 completed: the crate now holds step state (recorded probes, the carried
    /// token budget, a half-built `Decisions`) that only phase 2 may consume.
    ///
    /// Set on SUCCESS only. A phase 1 that returned an error left nothing for phase 2 and
    /// the wrapper falls back, so the window must stay shut.
    #[inline]
    pub fn open_phase_split(&mut self) {
        self.phase_split_open = true;
    }

    /// Phase 2 entry: take the window, or say who took it first.
    ///
    /// Deliberately checked BEFORE `invalidate()`: rolling a speculation back first would
    /// erase the very evidence (`Manager::journal_armed`) that
    /// `ScheduleCore::refuse_split_under_speculation` looks for, which is how a speculative
    /// run could clobber the window and leave no trace.
    pub fn claim_phase_split(&mut self) -> Result<(), String> {
        if !self.phase_split_open {
            return Err(
                "the phase-split window was closed underneath (a speculative run or \
                 another caller intervened)"
                    .into(),
            );
        }
        self.phase_split_open = false;
        Ok(())
    }

    /// Shut the window without consuming it. Every non-split scheduling entry point calls
    /// this, so a step that bails between the phases cannot leave a stale window open for
    /// the next one to walk into.
    #[inline]
    pub fn close_phase_split(&mut self) {
        self.phase_split_open = false;
    }

    /// Invariant 2. Call this at the top of every state-mutating entry point.
    ///
    /// Deliberately does NOT touch `phase_split_open`: it is called by the `w()` guards of
    /// every mutating pymethod, and the ones that can legitimately land between the phases
    /// (the input thread's `set_request_meta`) only roll journals back — they touch neither
    /// `core.decisions` nor `Manager::probe_hit`. Closing the window here would turn those
    /// into spurious phase-2 failures; leaving it open is safe precisely because they
    /// mutate nothing phase 2 carries.
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
    // Stage S: the connector's phase-split window is open, i.e. the engine thread is
    // between `schedule_phase1` and `schedule_phase2` and the lock it dropped is the one
    // this worker just took. Running here would clear the recorded probes and the carried
    // budget out from under phase 2 (see `Shared::phase_split_open`). DROP the job rather
    // than wait for the window: by the time it closes the running set this was kicked for
    // is one step old anyway, and `take_speculative` would miss on the generation.
    //
    // Counted as a miss and otherwise silent -- this is the hot path, and under the shipped
    // configuration `spec_blocked` means it is never even reached.
    if sh.phase_split_open {
        sh.spec.misses += 1;
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
            connector: false,
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
            connector: false,
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

    /// Invariant 4, both directions. The connector step drops the lock between its two
    /// phases, so a kicked worker can take it and run `schedule_resident` on the SAME
    /// `ScheduleCore` -- clearing the decisions, clearing the probes and overwriting the
    /// carried budget. This is the guard that stops it, and the guard that notices if
    /// something else did.
    #[test]
    fn a_speculative_run_declines_inside_the_phase_split_window() {
        let mut c = cfg();
        c.connector = true;
        let mut p = params();
        p.connector = true;
        let mut kv = Manager::new(c).unwrap();
        // One cached prompt and a candidate that re-sends it, so phase 1 has a real probe
        // to record and phase 2 has something to lose.
        let base = kv.intern("base");
        let hashes: Vec<u8> = (0..4u8)
            .flat_map(|i| {
                let mut d = [1u8; crate::hash::HASH_LEN];
                d[0] = i;
                d.to_vec()
            })
            .collect();
        kv.push_hashes(base, &hashes, 64);
        let dup = kv.intern("dup");
        kv.push_hashes(dup, &hashes, 64);
        let mut sh = Shared::new(kv);

        let waiting = [SchedReq {
            slot: base,
            num_tokens: 64,
            num_tokens_with_spec: 64,
            num_prompt_tokens: 64,
            max_tokens: 128,
            is_prefill_chunk: true,
            ..SchedReq::default()
        }];
        {
            let Shared { manager, core, .. } = &mut sh;
            core.schedule(manager, &[], &waiting, &p).unwrap();
        }

        // A phase 2 with no window open is a wiring bug, not a slow step: refuse it rather
        // than admit against whatever state happens to be lying around.
        assert!(sh.claim_phase_split().is_err(), "no window, no phase 2");

        // Phase 1, exactly as `run_schedule_phase1` drives it.
        let probe = [SchedReq {
            slot: dup,
            num_tokens: 64,
            num_tokens_with_spec: 64,
            num_prompt_tokens: 64,
            max_tokens: 128,
            is_prefill_chunk: true,
            ..SchedReq::default()
        }];
        {
            let Shared { manager, core, .. } = &mut sh;
            core.schedule_phase1(manager, &[], &p, &probe).unwrap();
        }
        sh.open_phase_split();
        let recorded = sh.manager.probe_hits();
        assert_eq!(recorded.len(), 1, "the probe is on the books");

        // THE RACE: the worker picks up a kick while the window is open.
        let misses = sh.spec.misses;
        run_speculative(
            &mut sh,
            KickMsg {
                generation: 1,
                running_slots: vec![],
                params: p,
            },
        );
        assert!(sh.spec.pending.is_none(), "the worker must not have run");
        assert_eq!(sh.spec.misses, misses + 1, "...and must count the declined job");
        assert_eq!(
            sh.manager.probe_hits(),
            recorded,
            "the recorded probe is exactly as phase 1 left it"
        );

        // Phase 2 then claims the window and closes it behind itself.
        assert!(sh.claim_phase_split().is_ok());
        {
            let Shared { manager, core, .. } = &mut sh;
            core.schedule_phase2(manager, &p, &probe, &[], 0).unwrap();
        }
        assert!(
            !sh.phase_split_open,
            "the window is one-shot: the next phase 2 must not walk into it"
        );
        assert!(sh.claim_phase_split().is_err());

        // ...and with the window shut, the worker runs as it always did.
        run_speculative(
            &mut sh,
            KickMsg {
                generation: 2,
                running_slots: vec![],
                params: p,
            },
        );
        assert!(sh.spec.pending.is_some(), "speculation is not disabled, only fenced");
    }

    /// The single-call paths are whole steps: they must not leave a window behind for a
    /// phase 2 that was never paired with a phase 1.
    #[test]
    fn an_unsplit_schedule_closes_the_phase_split_window() {
        let kv = Manager::new(cfg()).unwrap();
        let mut sh = Shared::new(kv);
        sh.open_phase_split();
        sh.close_phase_split();
        {
            let Shared { manager, core, .. } = &mut sh;
            core.schedule(manager, &[], &[], &params()).unwrap();
        }
        assert!(sh.claim_phase_split().is_err());
    }
}
