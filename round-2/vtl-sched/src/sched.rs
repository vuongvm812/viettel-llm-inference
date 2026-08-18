//! R5 — port of `Scheduler.schedule()`'s decision loop
//! (`vllm/v1/core/sched/scheduler.py:396`), behind `VTL_RUST_SCHED_FULL=1`.
//!
//! Mirrors:
//!   * `:338`  `_mamba_block_aligned_split`  -> [`mamba_block_aligned_split`]
//!   * `:441`  running-queue loop            -> [`ScheduleCore::schedule`] part 1
//!   * `:534`  `allocate_slots` + preemption -> [`ScheduleCore::schedule`] preempt loop
//!   * `:636`  waiting-queue loop            -> [`ScheduleCore::schedule`] part 2
//!   * `:1140` `_preempt_request`            -> [`Decisions::preempted`] (Python applies it)
//!   * `vtl/patches/sched_policy.py::_reorder_waiting` -> [`ScheduleCore::reorder_waiting`]
//!
//! SCOPE: this returns DECISIONS ONLY. `SchedulerOutput` assembly, request mutation and
//! queue surgery stay in Python — its consumers are Python objects and moving them
//! across the boundary would cost more than the loop saves.
//!
//! Everything the served configuration cannot produce is rejected by the caller before
//! this runs (see `rust_sched.py::_schedule_supported`): KV/EC connectors, LoRA, encoder
//! inputs, speculative decoding, priority policy, structured output, pause states, DP
//! prefill throttling, PP decode cadence, `scheduler_reserve_full_isl`. Their branches
//! are therefore absent from this loop by construction, not by accident.

use crate::manager::{Manager, ProbeTake, STATUS_PREEMPTED, STATUS_RUNNING, STATUS_WAITING};

/// Mirror of the `Request` fields the scheduling loop reads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SchedReq {
    pub slot: u32,
    pub num_tokens: usize,
    pub num_tokens_with_spec: usize,
    pub num_computed_tokens: usize,
    pub num_output_placeholders: usize,
    pub num_prompt_tokens: usize,
    pub max_tokens: usize,
    pub status: u8,
    pub num_preemptions: u32,
    pub is_prefill_chunk: bool,
    pub skip_reading_prefix_cache: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    pub max_num_scheduled_tokens: usize,
    pub max_num_running_reqs: usize,
    pub max_model_len: usize,
    pub num_sampled_tokens_per_step: usize,
    pub long_prefill_token_threshold: usize,
    pub enable_chunked_prefill: bool,
    pub need_mamba_block_aligned_split: bool,
    pub cache_block_size: usize,
    pub num_lookahead_tokens: usize,
    /// `VTL_ENABLE_SCHED_POLICY` — cache-aware SJF reorder of the waiting queue.
    pub sjf_reorder: bool,
    pub sjf_usage_tight: f64,
    /// `VTL_RUST_SCHED_LEAN` — drop decision payload no V2 consumer reads. Currently:
    /// skip the `num_common_prefix_blocks` epilogue (compute AND marshal). Optional on
    /// the wire (`set_params` defaults it false), so an older plugin keeps working.
    pub lean_decisions: bool,
    /// Largest batch an N-step burst was captured for (`nstep_decode.MAX_BURST_REQS`), or
    /// 0 when the burst is off / the plugin is older than this wheel — which makes
    /// [`Decisions::burst_eligible`] permanently false, i.e. Python re-derives it exactly
    /// as it did before this port.
    pub burst_max_reqs: usize,
    /// A KV connector is live (`Config::connector`'s twin on the schedule-loop side).
    ///
    /// Stage S made this load-bearing. It is the permission for the two things only a
    /// connector can produce: a non-empty cache-hit PROBE in
    /// [`ScheduleCore::schedule_phase1`], and a non-empty EXTERNAL slice in
    /// [`ScheduleCore::schedule_phase2`]. Both refuse loudly when it is false, so a plugin
    /// that sends connector work to a crate configured without one gets an error rather
    /// than a silently mis-scheduled step. False is still the shipped connector-off value
    /// and every path below is then exactly what it was before Stage S.
    pub connector: bool,
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Decisions {
    /// `(slot, num_new_tokens)` for requests already in `running`.
    pub scheduled_running: Vec<(u32, usize)>,
    /// Blocks `allocate_slots` handed out THIS STEP for each `scheduled_running` entry,
    /// flattened group-major: entry `i` group `g` has `running_new_lens[i * G + g]` ids,
    /// consecutive in `running_new_blocks`. v0.25.0 stores exactly this delta for running
    /// requests (scheduler.py:588) and the FULL table only for waiting admissions
    /// (`:983`); the V2 runner appends `new_block_ids` without overwriting, so handing
    /// back the whole table every step duplicates the block table.
    pub running_new_blocks: Vec<u32>,
    pub running_new_lens: Vec<u32>,
    /// `(slot, num_new_tokens, num_computed_tokens)` admitted from `waiting`; `status`
    /// at admission time decides new vs resumed on the Python side.
    pub scheduled_admitted: Vec<(u32, usize, usize)>,
    /// Slots popped off the tail of `running`, in preemption order.
    pub preempted: Vec<u32>,
    /// Final `waiting`-queue order (slots) after the SJF reorder, front first.
    pub waiting_order: Vec<u32>,
    /// Stage S: `(slot, num_computed = local + external, num_external)` for every waiting
    /// candidate this step PARKED on an async connector load (scheduler.py:948-967).
    ///
    /// Not scheduled and not admitted: the blocks are allocated and the local hit is
    /// installed, but no tokens are computed for it this step. Python owns the rest of
    /// stock's tail — `update_state_after_alloc`, the `WAITING_FOR_REMOTE_KVS` status, the
    /// `num_computed_tokens` write, `_inflight_prefills.add` and the `skipped_waiting`
    /// prepend — because every one of those touches a Python object.
    ///
    /// Always EMPTY without a connector, which is what keeps `decisions_dict` and every
    /// consumer of it unchanged on a connector-off boot.
    pub parked_external: Vec<(u32, usize, usize)>,
    /// Stage S: waiting candidates this step CONSUMED from the queue without deciding
    /// anything for them, because the local cache hit phase 1 recorded for them no longer
    /// holds (`Manager::take_probe_hit` -> `ProbeTake::Stale`) AND the connector had
    /// already been told that hit length.
    ///
    /// WHY THEY CANNOT JUST BE ADMITTED ON A FRESH WALK. The connector's answer is keyed to
    /// the hit it was asked about: `update_state_after_alloc` loads
    /// `offload_keys[num_locally_computed_gpu_blocks..]` and asserts the local boundary it
    /// finds in the block records matches `num_locally_computed_tokens`
    /// (offloading/scheduler.py:711-761). Re-walking moves that boundary, so admitting
    /// would either trip the assert or load the wrong key window into the wrong blocks.
    ///
    /// Python moves them to `skipped_waiting`, exactly as it does for the connector's own
    /// "ask me again later" (scheduler.py:743) — which is safe for the same reason: the
    /// next `get_num_new_matched_tokens` clears the request's per-group block ids and
    /// re-runs the lookup from scratch (offloading/scheduler.py:670-691).
    ///
    /// Always empty without a connector: a probe is only ever recorded for one.
    pub probe_stale: Vec<u32>,
    pub num_common_prefix_blocks: Vec<usize>,
    /// May the step this decision describes carry an N-step burst, as far as the SCHEDULER
    /// can tell? The conjuncts are exactly the ones `nstep_decode._publish_ready` used to
    /// re-derive in numpy at sample(k) — no prefill in the batch, every scheduled count 1,
    /// no preemption, batch within the captured burst sizes — and they belong here for a
    /// reason beyond cost: `_publish_ready` was PREDICTING step k+1's batch from step k's,
    /// while this is computed on the batch it actually describes.
    ///
    /// Spec/draft tokens need no clause of their own: a spec step schedules
    /// `1 + num_draft` tokens for a running request, so the all-counts-are-1 test already
    /// excludes it (and `rust_sched.py::schedule_supported` refuses spec decode outright).
    /// What stays in Python is only what never crosses into the crate: encoder inputs,
    /// structured output and `new_block_ids_to_zero`.
    pub burst_eligible: bool,
}

impl Decisions {
    fn clear(&mut self) {
        self.scheduled_running.clear();
        self.running_new_blocks.clear();
        self.running_new_lens.clear();
        self.scheduled_admitted.clear();
        self.preempted.clear();
        self.waiting_order.clear();
        self.parked_external.clear();
        self.probe_stale.clear();
        self.num_common_prefix_blocks.clear();
        self.burst_eligible = false;
    }
}

/// `_mamba_block_aligned_split` (scheduler.py:338). `use_eagle` is always false here.
pub fn mamba_block_aligned_split(
    req: &SchedReq,
    num_new_tokens: usize,
    num_new_local_computed_tokens: usize,
    num_uncached_common_prefix_tokens: usize,
    block_size: usize,
) -> usize {
    let num_computed_tokens = req.num_computed_tokens + num_new_local_computed_tokens;
    let prefill_end = req.num_prompt_tokens.max(req.num_tokens.saturating_sub(1));
    if num_computed_tokens >= prefill_end {
        return num_new_tokens;
    }
    let mut num_new_tokens = num_new_tokens;
    let last_cache_position = req.num_tokens - req.num_tokens % block_size;
    let after = num_computed_tokens + num_new_tokens;
    if after < last_cache_position {
        num_new_tokens = num_new_tokens / block_size * block_size;
    } else if num_computed_tokens < last_cache_position && last_cache_position < after {
        num_new_tokens = last_cache_position - num_computed_tokens;
    }
    // Marconi cache admission: schedule exactly the uncached common prefix.
    if num_uncached_common_prefix_tokens >= block_size
        && num_new_tokens > num_uncached_common_prefix_tokens
    {
        num_new_tokens = num_uncached_common_prefix_tokens / block_size * block_size;
    }
    num_new_tokens
}

pub struct ScheduleCore {
    pub decisions: Decisions,
    /// Running entries the loop refused because `num_computed_tokens` was past
    /// `num_tokens_with_spec + num_output_placeholders` -- see the `checked_sub` in
    /// [`ScheduleCore::run`]. Monotonic for the life of the boot and exposed to Python as
    /// `KvManager::inconsistent_skips` so a step that scheduled nothing can say WHY: a
    /// batch that stops making progress with this counter rising is a bookkeeping skew the
    /// resident table has absorbed, not a scheduler that ran out of budget.
    pub(crate) inconsistent: u64,
    running: Vec<SchedReq>,
    waiting: Vec<SchedReq>,
    keys: Vec<(u8, usize, usize)>,
    reorder_buf: Vec<SchedReq>,
    /// Stage S: the token budget, carried across the phase boundary.
    ///
    /// A local in the un-split loop; a field now, because the running arm (phase 1) spends
    /// it and the waiting arm (phase 2) spends what is left, with a Python call — the
    /// connector's `get_num_new_matched_tokens` — in between. Written at the top of every
    /// phase 1, so a phase 2 that never saw its phase 1 cannot inherit a stale budget from
    /// two steps ago; it inherits the LAST phase 1's, which is the closest thing to a
    /// correct answer and is unreachable anyway (the wrapper calls them in pairs).
    budget: usize,
}

impl Default for ScheduleCore {
    fn default() -> Self {
        Self::new()
    }
}

impl ScheduleCore {
    pub fn new() -> Self {
        ScheduleCore {
            decisions: Decisions::default(),
            inconsistent: 0,
            running: Vec::with_capacity(64),
            waiting: Vec::with_capacity(256),
            keys: Vec::with_capacity(256),
            reorder_buf: Vec::with_capacity(256),
            budget: 0,
        }
    }

    /// `vtl/patches/sched_policy.py::_reorder_waiting` — cache-aware shortest-remaining-
    /// prefill-first, with the memory-aware demotion when KV usage is tight. Stable, so
    /// FCFS survives among ties.
    fn reorder_waiting(&mut self, kv: &mut Manager, params: &Params) {
        if self.waiting.len() < 2 {
            return;
        }
        let tight = kv.usage() >= params.sjf_usage_tight;
        let free = kv.num_free_blocks();
        let block_size = params.cache_block_size.max(1);
        self.keys.clear();
        for (i, r) in self.waiting.iter().enumerate() {
            let remaining = if r.num_computed_tokens > 0 {
                r.num_prompt_tokens.saturating_sub(r.num_computed_tokens)
            } else {
                let hit = kv.peek_cache_hit(r.slot, r.num_tokens);
                r.num_prompt_tokens.saturating_sub(hit)
            };
            let blocks_needed = remaining.div_ceil(block_size);
            let fits = if !tight || blocks_needed <= free { 0u8 } else { 1u8 };
            self.keys.push((fits, remaining, i));
        }
        self.keys.sort();
        // Apply the permutation. `keys` carries the original index so this is stable.
        self.reorder_buf.clear();
        for &(_, _, i) in &self.keys {
            self.reorder_buf.push(self.waiting[i]);
        }
        std::mem::swap(&mut self.waiting, &mut self.reorder_buf);
    }

    /// The marshalled entry point: `running` / `waiting` are queue-ordered snapshots
    /// packed by `rust_sched.py::pack_req`.
    ///
    /// This is ALSO the resident table's full-resync path — every running request's
    /// packed fields are written into the table before the loop runs, so a step that
    /// bails to Python and comes back still finds a table that matches reality.
    pub fn schedule(
        &mut self,
        kv: &mut Manager,
        running: &[SchedReq],
        waiting: &[SchedReq],
        params: &Params,
    ) -> Result<&Decisions, String> {
        self.load_running(kv, running);
        self.waiting.clear();
        self.waiting.extend_from_slice(waiting);
        self.run(kv, params)
    }

    /// The marshalled running load, shared by [`Self::schedule`] and
    /// [`Self::schedule_phase1`] so the resync semantics cannot drift between them.
    fn load_running(&mut self, kv: &mut Manager, running: &[SchedReq]) {
        self.running.clear();
        self.running.extend_from_slice(running);
        for r in running {
            kv.table_set(r.slot, *r);
        }
    }

    /// The resident running load, shared by [`Self::schedule_resident`] and
    /// [`Self::schedule_phase1_resident`].
    fn load_running_resident(&mut self, kv: &Manager, running_slots: &[u32]) -> Result<(), String> {
        self.running.clear();
        self.running.reserve(running_slots.len());
        for &slot in running_slots {
            match kv.table_get(slot) {
                Some(e) => self.running.push(e),
                None => {
                    return Err(format!(
                        "slot {slot} has no resident entry; a full resync is required"
                    ))
                }
            }
        }
        Ok(())
    }

    /// R6b: the same loop, reading the running set out of the Rust-resident table instead
    /// of a marshalled slice. `running_slots` carries Python's queue ORDER, which is
    /// load-bearing twice over — FCFS admission and tail-pop preemption both depend on
    /// it — so it stays Python's authority even though the fields no longer cross.
    pub fn schedule_resident(
        &mut self,
        kv: &mut Manager,
        running_slots: &[u32],
        waiting: &[SchedReq],
        params: &Params,
    ) -> Result<&Decisions, String> {
        self.load_running_resident(kv, running_slots)?;
        self.waiting.clear();
        self.waiting.extend_from_slice(waiting);
        self.run(kv, params)
    }

    /// THE loop body, shared by both entry points so they cannot drift apart. `running`
    /// and `waiting` are already loaded.
    ///
    /// STAGE S: this is now literally `phase1_body` followed by `phase2_body` with an empty
    /// probe and an empty external slice, in the same order, with nothing between them. It
    /// is the PARITY ANCHOR for the split — a connector-off boot runs exactly the
    /// instructions it ran before, and the two public phase entry points below differ from
    /// it only in what they load and in the two slices they are allowed to be non-empty.
    fn run(&mut self, kv: &mut Manager, params: &Params) -> Result<&Decisions, String> {
        // Speculation's scope invariant (see `journal.rs`): the waiting half of the loop
        // reaches prefix-cache lookup paths that are deliberately not journaled. Refuse
        // rather than approximate. Deliberately still HERE and not in `phase1_body`: it has
        // to be answered before `decisions.clear()` / `new_step_starts()` mutate anything,
        // and only this level knows the waiting queue is already loaded.
        if kv.journal_armed() && !self.waiting.is_empty() {
            return Err("speculation refuses a non-empty waiting queue".into());
        }
        self.phase1_body(kv, params, &[])?;
        self.phase2_body(kv, params, &[], 0)
    }

    /// Stage S, phase 1 (marshalled running): the running arm and the connector's
    /// cache-hit probe. See [`Self::schedule_phase2`] for why the step is cut here.
    ///
    /// `probe` is the waiting candidates whose local cache hit the connector needs BEFORE
    /// it can answer `get_num_new_matched_tokens` — `num_computed_tokens == 0` candidates,
    /// capped by the caller at the admission budget. Their hits are parked per slot in the
    /// manager (`Manager::probe_hit`) and phase 2 consumes them instead of re-walking.
    ///
    /// The waiting queue is CLEARED here: the candidate list arrives with phase 2 (Python
    /// cannot assemble it before the connector queries, which need this call's answer), so
    /// anything left over from the previous step must not be scheduled off.
    pub fn schedule_phase1(
        &mut self,
        kv: &mut Manager,
        running: &[SchedReq],
        params: &Params,
        probe: &[SchedReq],
    ) -> Result<(), String> {
        Self::refuse_split_under_speculation(kv)?;
        self.load_running(kv, running);
        self.waiting.clear();
        self.phase1_body(kv, params, probe)
    }

    /// [`Self::schedule_phase1`] with the running set read from the resident table, the
    /// twin of [`Self::schedule_resident`].
    pub fn schedule_phase1_resident(
        &mut self,
        kv: &mut Manager,
        running_slots: &[u32],
        params: &Params,
        probe: &[SchedReq],
    ) -> Result<(), String> {
        Self::refuse_split_under_speculation(kv)?;
        self.load_running_resident(kv, running_slots)?;
        self.waiting.clear();
        self.phase1_body(kv, params, probe)
    }

    /// Stage S, phase 2: the waiting arm and the epilogue, with the connector's answers.
    ///
    /// WHY THE STEP IS CUT IN TWO. `get_num_new_matched_tokens` is a Python call on the
    /// connector, and it takes the request's LOCAL cache hit as its argument
    /// (scheduler.py:735-741). The hit comes out of the crate and the answer goes back into
    /// the crate's admission arithmetic, so somewhere between the two the GIL has to be
    /// held and the crate lock released. That seam is this split: phase 1 ends with the
    /// hits, Python asks the connector, phase 2 admits.
    ///
    /// `external` is `(slot, num_external_computed_tokens, is_async)` for the candidates
    /// Python queried; a slot with no entry is an ordinary candidate. `base_reserved` is
    /// `_inflight_prefill_reserved_blocks()` (scheduler.py:2400) computed over the prefills
    /// that were ALREADY in flight — the loads this loop parks are added to it here, in
    /// order, exactly as stock's `_inflight_prefills.add` makes the next iteration see them.
    ///
    /// The SJF reorder runs here rather than in phase 1 for the same reason the queue is
    /// cleared there: the candidates do not exist yet at phase 1. That makes its two
    /// memory-pressure keys (`usage`, `num_free_blocks`) read AFTER the running arm's
    /// allocations instead of before — a deliberate, connector-only divergence from the
    /// un-split loop (which is the parity anchor and keeps reading them before).
    pub fn schedule_phase2(
        &mut self,
        kv: &mut Manager,
        params: &Params,
        waiting: &[SchedReq],
        external: &[(u32, usize, bool)],
        base_reserved: usize,
    ) -> Result<&Decisions, String> {
        Self::refuse_split_under_speculation(kv)?;
        self.waiting.clear();
        self.waiting.extend_from_slice(waiting);
        if params.sjf_reorder {
            self.reorder_waiting(kv, params);
        }
        self.phase2_body(kv, params, external, base_reserved)
    }

    /// The split entry points are the CONNECTOR path, and the connector path never runs on
    /// the speculation worker (`spec.rs` only ever calls `schedule_resident` with an empty
    /// waiting queue, and `rust_sched.py::spec_blocked` refuses to kick at all while a
    /// connector is live). An armed journal here would mean the two have been wired
    /// together, and half of what phase 1 does — the probe's cache-hit walk — is exactly
    /// what `journal.rs` does not record.
    fn refuse_split_under_speculation(kv: &Manager) -> Result<(), String> {
        if kv.journal_armed() {
            return Err("speculation refuses the connector phase split".into());
        }
        Ok(())
    }

    /// Phase 1's body: the running arm plus the connector probe. `running` is loaded.
    fn phase1_body(
        &mut self,
        kv: &mut Manager,
        params: &Params,
        probe: &[SchedReq],
    ) -> Result<(), String> {
        self.decisions.clear();

        kv.new_step_starts();
        if params.sjf_reorder {
            self.reorder_waiting(kv, params);
        }

        let mut token_budget = params.max_num_scheduled_tokens;

        // ---- 1. RUNNING requests (scheduler.py:441) ------------------------
        let mut req_index = 0usize;
        while req_index < self.running.len() && token_budget > 0 {
            let request = self.running[req_index];

            // Async scheduling: the previous step already reached max_tokens.
            //
            // ADDITION FORM, deliberately. Python writes this as `num_computed_tokens + 2 -
            // num_output_placeholders >= num_prompt_tokens + max_tokens` in signed
            // arithmetic; in usize the `- num_output_placeholders` is a wrap waiting for a
            // `P > C + 2` entry, and a wrapped left-hand side is an ENORMOUS number that
            // makes the skip fire for a request nowhere near its cap (in debug it panics
            // instead, which kills the resident path for the rest of the boot). Moving the
            // subtrahend to the other side is the same inequality over the integers for
            // every state that can legitimately occur -- both sides are only ever compared,
            // never stored -- and is total over all of usize. The `P > 0` guard stays: it is
            // what limits this branch to the async path, where a placeholder means a token
            // the previous step promised.
            if request.num_output_placeholders > 0
                && request.num_computed_tokens + 2
                    >= request.num_prompt_tokens
                        + request.max_tokens
                        + request.num_output_placeholders
            {
                req_index += 1;
                continue;
            }

            // The work derivation, and the one subtraction in this loop that CANNOT be
            // rearranged away: an entry whose `num_computed_tokens` has run past
            // `num_tokens_with_spec + num_output_placeholders` is broken bookkeeping (the
            // `C == T + P - 1` invariant `rust_sched.py::burst_invariant_broken` guards),
            // and there is no defensible number to schedule for it. Wrapping would hand
            // `allocate_slots` a count near usize::MAX -- garbage blocks, or an error that
            // fails the whole step; panicking (which is what plain `-` does in debug) kills
            // the resident path permanently for one bad entry. Counting it and skipping is
            // the third answer: this request stalls, every other request in the batch is
            // scheduled normally, and Python's probe on the counter resyncs the table --
            // which is the only thing that can actually repair the entry.
            let Some(mut num_new_tokens) = (request.num_tokens_with_spec
                + request.num_output_placeholders)
                .checked_sub(request.num_computed_tokens)
            else {
                self.inconsistent = self.inconsistent.saturating_add(1);
                req_index += 1;
                continue;
            };
            if params.long_prefill_token_threshold > 0
                && params.long_prefill_token_threshold < num_new_tokens
            {
                num_new_tokens = params.long_prefill_token_threshold;
            }
            num_new_tokens = num_new_tokens.min(token_budget);
            // BUG-COMPAT NOTE: python computes `max_model_len - num_computed_tokens -
            // num_sampled_tokens_per_step` in signed arithmetic, so an over-length request
            // propagates a NEGATIVE num_new_tokens past the `== 0` check into
            // allocate_slots (which then raises). saturating_sub clamps to 0, which the
            // `== 0` check turns into a skip. Deliberate: the python path is an engine
            // crash, ours is a no-op, and the served config caps num_tokens at
            // max_model_len upstream so neither branch can be reached.
            num_new_tokens = num_new_tokens.min(
                params
                    .max_model_len
                    .saturating_sub(request.num_computed_tokens + params.num_sampled_tokens_per_step),
            );

            if params.need_mamba_block_aligned_split {
                num_new_tokens = mamba_block_aligned_split(
                    &request,
                    num_new_tokens,
                    0,
                    0,
                    params.cache_block_size,
                );
            }

            if num_new_tokens == 0 {
                req_index += 1;
                continue;
            }

            // allocate_slots + FCFS preemption (scheduler.py:534).
            let mut allocated;
            loop {
                allocated = kv.allocate_slots(
                    request.slot,
                    num_new_tokens,
                    0,
                    false,
                    params.num_lookahead_tokens,
                    request.num_computed_tokens,
                    request.num_tokens,
                    request.status,
                    true,
                    // No connector work in the running arm: external tokens, delayed
                    // caching and reserved headroom are all admission-time concerns.
                    0,
                    false,
                    0,
                )?;
                if allocated {
                    break;
                }
                let Some(victim) = self.running.pop() else {
                    break;
                };
                kv.free(victim.slot);
                self.decisions.preempted.push(victim.slot);
                if victim.slot == request.slot {
                    // Preempted ourselves: nothing left to free.
                    break;
                }
            }
            if !allocated {
                break;
            }

            self.decisions
                .scheduled_running
                .push((request.slot, num_new_tokens));
            for g in 0..kv.new_blocks.len() {
                let nb = &kv.new_blocks[g];
                self.decisions.running_new_lens.push(nb.len() as u32);
                self.decisions.running_new_blocks.extend_from_slice(nb);
            }
            token_budget -= num_new_tokens;
            req_index += 1;
            if kv.journal_over_cap() {
                return Err("speculation journal exceeded its soft cap".into());
            }
        }
        // What the waiting arm has left to spend, whether it runs in this call (`run`) or
        // in a later `schedule_phase2`.
        self.budget = token_budget;

        // ---- 1b. The connector's cache-hit probe (scheduler.py:687-721) ----
        // Phase 2 consumes these instead of re-walking, so the hit the connector was asked
        // about and the hit `allocate_slots` is given are the same walk — which matters
        // because a walk is not idempotent in general (another candidate's admission can
        // move blocks under it).
        kv.probe_clear();
        if !probe.is_empty() {
            if !params.connector {
                return Err("a cache-hit probe was requested without a live connector".into());
            }
            for r in probe {
                if r.num_computed_tokens != 0 {
                    // Stock only reaches the hit walk (and therefore the connector query)
                    // on the `request.num_computed_tokens == 0` arm (scheduler.py:685); a
                    // resumed request's KV is already local.
                    return Err(
                        "a probed candidate must have num_computed_tokens == 0".into()
                    );
                }
                kv.probe_computed_blocks(
                    r.slot,
                    r.num_tokens,
                    r.num_preemptions,
                    r.skip_reading_prefix_cache,
                );
            }
        }
        Ok(())
    }

    /// Phase 2's body: the waiting arm and the epilogue. `waiting` is loaded (and
    /// reordered, if it was going to be).
    fn phase2_body(
        &mut self,
        kv: &mut Manager,
        params: &Params,
        external: &[(u32, usize, bool)],
        base_reserved: usize,
    ) -> Result<&Decisions, String> {
        let mut token_budget = self.budget;
        if !external.is_empty() && !params.connector {
            return Err("external computed tokens without a live connector".into());
        }

        // ---- 2. WAITING requests (scheduler.py:636) ------------------------
        let mut admitted_from_waiting = 0usize;
        // The in-loop half of `_inflight_prefill_reserved_blocks` (scheduler.py:2400):
        // stock adds each parked load to `_inflight_prefills` immediately (`:966`), so the
        // NEXT candidate's reservation already counts it. `base_reserved` carries the
        // prefills that were in flight before this step; this carries the ones this loop
        // parked.
        let mut parked_reserved = 0usize;
        if self.decisions.preempted.is_empty() {
            while admitted_from_waiting < self.waiting.len() && token_budget > 0 {
                if self.running.len() >= params.max_num_running_reqs {
                    break;
                }
                let request = self.waiting[admitted_from_waiting];
                // Read once, up here: the staleness arm below has to know whether the
                // connector was told anything about this candidate before it decides what a
                // stale probe costs.
                let ext_entry = external.iter().find(|e| e.0 == request.slot).copied();

                // scheduler.py:684 zeroes the Marconi hint per iteration and only refills
                // it (`:730`) right after a fresh `get_computed_blocks`. Reading the
                // coordinator unconditionally would feed the PREVIOUS request's hint into
                // `mamba_block_aligned_split` whenever num_computed_tokens != 0.
                //
                // Stage S: a candidate the PROBE already walked for (phase 1) consumes that
                // walk instead of running a second one — same three numbers, and the blocks
                // go back into `pending_hit` where `allocate_slots` reads them. Without a
                // connector the store is always empty and this is the fresh walk it always
                // was, in the same place, with the same coordinator read after it.
                //
                // AND a recorded walk can go STALE inside this very loop: the candidates
                // ahead of this one have been allocating, and `get_new_blocks` evicts the
                // cached-but-free blocks it pops. `ProbeTake::Stale` is that, and it splits
                // by whether the connector already saw the number.
                let recorded = if request.num_computed_tokens == 0 {
                    kv.take_probe_hit(request.slot)
                } else {
                    None
                };
                if recorded == Some(ProbeTake::Stale)
                    && ext_entry.is_some_and(|(_, num_external, _)| num_external > 0)
                {
                    // The connector's `keys_to_load` window is keyed to the hit it was
                    // asked about; a fresh walk moves that boundary and admitting on the
                    // new numbers corrupts the load. Consume the candidate from the queue
                    // and hand it back to Python for `skipped_waiting` — see
                    // `Decisions::probe_stale`. The budget is untouched: nothing was
                    // scheduled and nothing was allocated for it.
                    self.decisions.probe_stale.push(request.slot);
                    admitted_from_waiting += 1;
                    continue;
                }
                let (num_new_local_computed_tokens, use_pending_hit, uncached_common) =
                    if request.num_computed_tokens == 0 {
                        match recorded {
                            Some(ProbeTake::Fresh {
                                hit,
                                uncached_common,
                            }) => (hit, true, uncached_common),
                            // No probe (a connector-off boot, or a candidate the cap did
                            // not reach), or a stale one for a candidate the connector was
                            // told nothing about: walk it now. That is exactly what stock
                            // does for every candidate, and a stale probe's only cost here
                            // is the duplicated walk.
                            None | Some(ProbeTake::Stale) => {
                                let hit = kv.get_computed_blocks(
                                    request.slot,
                                    request.num_tokens,
                                    request.num_preemptions,
                                    request.skip_reading_prefix_cache,
                                );
                                (hit, true, kv.coord.num_uncached_common_prefix_tokens)
                            }
                        }
                    } else {
                        (0, false, 0)
                    };
                let num_computed_tokens = if request.num_computed_tokens == 0 {
                    num_new_local_computed_tokens
                } else {
                    request.num_computed_tokens
                };

                // ---- 2a. The parked async load (scheduler.py:797-967) ------
                // Everything about it is decided before the ordinary arithmetic runs,
                // because stock's `load_kv_async` branch sets `num_new_tokens = 0` and then
                // skips the split, the encoder work and the lookahead slots outright.
                if let Some((_, num_external, is_async)) = ext_entry {
                    if num_external > 0 && !is_async {
                        // A SYNCHRONOUS external hit is a shape this port has never had a
                        // path for: it would have to flow through the ordinary
                        // `allocate_slots` call below with `delay_cache_blocks=false`, and
                        // the OffloadingConnector does not produce it (its
                        // `get_num_new_matched_tokens` returns `(n, True)` for a hit).
                        return Err(
                            "sync external tokens are unreachable for OffloadingConnector".into(),
                        );
                    }
                    if is_async {
                        if num_external == 0 {
                            // scheduler.py:799 asserts exactly this.
                            return Err(
                                "an async KV load must carry external computed tokens".into(),
                            );
                        }
                        if request.num_computed_tokens != 0 {
                            // The connector is only ever queried on the `C == 0` arm, so a
                            // parked load with a resumed request is a wiring bug upstream.
                            return Err(
                                "an async KV load needs num_computed_tokens == 0".into(),
                            );
                        }
                        let total_computed = num_new_local_computed_tokens + num_external;
                        if total_computed > request.num_tokens {
                            // scheduler.py:757's assert.
                            return Err(
                                "external + local computed tokens exceed the request".into(),
                            );
                        }
                        let ok = kv.allocate_external(
                            request.slot,
                            num_new_local_computed_tokens,
                            num_external,
                            request.num_tokens,
                            request.status,
                            !self.running.is_empty(),
                            base_reserved + parked_reserved,
                        )?;
                        if !ok {
                            // Stock breaks out of the whole waiting loop when
                            // `allocate_slots` returns None (`:918`).
                            break;
                        }
                        // The reservation this load now holds, for the candidates behind it
                        // (`_request_remaining_blocks`, scheduler.py:2387 — read AFTER
                        // `num_computed_tokens` was set, which is what `:965` does).
                        parked_reserved +=
                            kv.remaining_blocks(request.slot, request.num_tokens, total_computed);
                        self.decisions.parked_external.push((
                            request.slot,
                            total_computed,
                            num_external,
                        ));
                        // Consumed from the front of the queue exactly like an admission
                        // (stock pops it and prepends it to `step_skipped_waiting`), so it
                        // does not reappear in `waiting_order`. The token budget is
                        // untouched: a parked load computes nothing this step.
                        admitted_from_waiting += 1;
                        continue;
                    }
                }

                let mut num_new_tokens = request.num_tokens - num_computed_tokens;
                if params.long_prefill_token_threshold > 0
                    && params.long_prefill_token_threshold < num_new_tokens
                {
                    num_new_tokens = params.long_prefill_token_threshold;
                }
                if !params.enable_chunked_prefill && num_new_tokens > token_budget {
                    break;
                }
                num_new_tokens = num_new_tokens.min(token_budget);
                if num_new_tokens == 0 {
                    return Err("num_new_tokens must be > 0 for a waiting request".into());
                }

                if params.need_mamba_block_aligned_split {
                    num_new_tokens = mamba_block_aligned_split(
                        &request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        uncached_common,
                        params.cache_block_size,
                    );
                    if num_new_tokens == 0 {
                        break;
                    }
                }

                let ok = kv.allocate_slots(
                    request.slot,
                    num_new_tokens,
                    num_new_local_computed_tokens,
                    use_pending_hit,
                    params.num_lookahead_tokens,
                    request.num_computed_tokens,
                    request.num_tokens,
                    request.status,
                    !self.running.is_empty(),
                    // Still all three at their inert defaults, now by ELIMINATION rather
                    // than by the absence of a connector: an async load took the parked arm
                    // above and returned, and a synchronous external hit is refused there.
                    // What is left here computes tokens this step, so its blocks are cached
                    // immediately and it reserves nothing for anyone.
                    0,
                    false,
                    0,
                )?;
                if !ok {
                    break;
                }

                debug_assert!(
                    request.status == STATUS_WAITING || request.status == STATUS_PREEMPTED
                );
                self.running.push(request);
                self.decisions.scheduled_admitted.push((
                    request.slot,
                    num_new_tokens,
                    num_computed_tokens,
                ));
                token_budget -= num_new_tokens;
                admitted_from_waiting += 1;
            }
        }

        // ---- 3. Epilogue (scheduler.py:1035) -------------------------------
        // Under `lean_decisions` the vec is left EMPTY, which is also how `decisions_dict`
        // knows to omit the key: the non-lean arms always push one entry per group.
        if !params.lean_decisions {
            let mut common = std::mem::take(&mut self.decisions.num_common_prefix_blocks);
            common.clear();
            if let Some(first) = self.running.first() {
                common.extend_from_slice(kv.get_num_common_prefix_blocks(first.slot));
            } else {
                common.extend(std::iter::repeat(0).take(kv.coord.managers.len()));
            }
            self.decisions.num_common_prefix_blocks = common;
        }
        // Waiting requests that were admitted are dropped from the front; the rest keep
        // their (possibly reordered) order so Python can rewrite its deque verbatim.
        for r in &self.waiting[admitted_from_waiting..] {
            self.decisions.waiting_order.push(r.slot);
        }

        // ---- 4. Burst eligibility (see `Decisions::burst_eligible`) --------
        // Read off the decisions this loop just made, so it costs a handful of integer
        // comparisons over a batch that peaks at 8 on the scored trace. `burst_max_reqs`
        // of 0 (burst off, or a plugin that does not send the key) makes it permanently
        // false, which is the "Python re-derives it" fallback.
        let d = &mut self.decisions;
        d.burst_eligible = params.burst_max_reqs > 0
            && d.preempted.is_empty()
            && d.scheduled_admitted.is_empty()
            // Stage S: a parked load is an admission that has not happened yet — it is
            // promoted (and then admitted) as soon as its transfer lands, and a burst would
            // delay that by N-1 iterations. Same TTFT argument as `scheduled_admitted`.
            // Always empty without a connector, so this changes no connector-off verdict.
            && d.parked_external.is_empty()
            && !d.scheduled_running.is_empty()
            && d.scheduled_running.len() <= params.burst_max_reqs
            && d.scheduled_running.iter().all(|&(_, n)| n == 1);

        self.commit(kv, params);
        Ok(&self.decisions)
    }

    /// R6b post-schedule commit — the Rust half of what `rust_sched.py` does to the
    /// Python `Request` objects right after `schedule()` returns, applied to the resident
    /// table so the next step can read the running set from Rust.
    ///
    /// Ports, in the order Python applies them:
    ///   * the preempt loop (rust_sched.py, mirroring `_preempt_request`, scheduler.py:1212)
    ///   * `Scheduler._update_after_schedule` (scheduler.py:1236) — advance
    ///     `num_computed_tokens`, then recompute `is_prefill_chunk`
    ///   * `AsyncScheduler._update_after_schedule` (async_scheduler.py:20) — bump
    ///     `num_output_placeholders` by `num_sampled_tokens_per_step`, but ONLY for
    ///     requests that are no longer a prefill chunk, reading the value super() just set
    ///
    /// Preempted and scheduled slots are disjoint by construction: the preempt loop only
    /// pops requests at or after `req_index`, and the waiting loop is gated on nothing
    /// having been preempted.
    fn commit(&mut self, kv: &mut Manager, params: &Params) {
        let sampled = params.num_sampled_tokens_per_step;
        for i in 0..self.decisions.preempted.len() {
            let slot = self.decisions.preempted[i];
            kv.table_with(slot, |e| {
                e.status = STATUS_PREEMPTED;
                e.num_computed_tokens = 0;
                // The placeholders counted tokens the step this request was evicted from
                // will not produce for it, and no later path clears them. The admission
                // arithmetic below never reads P (the waiting arm derives `num_tokens -
                // num_computed_tokens`), so a stranded count costs nothing at re-admission
                // and everything AFTER it: the running loop's `num_tokens_with_spec +
                // num_output_placeholders - num_computed_tokens` is inflated by exactly
                // that count for every later step, scheduling tokens no one promised.
                // Zeroed alongside `num_computed_tokens`, a resumed request is back on
                // plain prefill arithmetic. The token still in flight for this slot drains
                // against the zero in `update_step`, whose `saturating_sub` makes the drain
                // total; Python's preempt loop does the same three writes and clamps the
                // same way (`rust_sched.py`, mirroring `_preempt_request`).
                e.num_output_placeholders = 0;
                // With C back at 0 this is true for any request with a prompt; computed
                // rather than assumed so the degenerate empty-prompt case cannot be wrong.
                e.is_prefill_chunk = e.num_computed_tokens < e.num_tokens;
                e.num_preemptions += 1;
            });
        }
        for i in 0..self.decisions.scheduled_running.len() {
            let (slot, num_new) = self.decisions.scheduled_running[i];
            kv.table_with(slot, |e| advance(e, num_new, sampled));
        }
        for i in 0..self.decisions.scheduled_admitted.len() {
            let (slot, num_new, num_computed) = self.decisions.scheduled_admitted[i];
            // The admitted request was marshalled in `waiting`; this is where it enters
            // the resident table. `self.running` holds it (pushed at admission).
            if let Some(r) = self.running.iter().rev().find(|r| r.slot == slot) {
                let mut e = *r;
                e.status = STATUS_RUNNING;
                e.num_computed_tokens = num_computed;
                advance(&mut e, num_new, sampled);
                kv.table_set(slot, e);
            }
        }
    }
}

/// `_update_after_schedule`'s per-request arithmetic, both scheduler layers.
///
/// The generic port: it runs for real prefill chunks as well as decodes, which is why the
/// placeholder bump keeps its condition -- a chunk that leaves `num_computed_tokens` short
/// of `num_tokens + num_output_placeholders` produces no token this step and must not
/// claim a placeholder for one. The burst commit has a narrower contract and its own
/// function, [`burst_advance`].
#[inline]
pub(crate) fn advance(e: &mut SchedReq, num_new_tokens: usize, num_sampled_tokens_per_step: usize) {
    e.num_computed_tokens += num_new_tokens;
    e.is_prefill_chunk = e.num_computed_tokens < e.num_tokens + e.num_output_placeholders;
    if !e.is_prefill_chunk {
        e.num_output_placeholders += num_sampled_tokens_per_step;
    }
}

/// The N-step burst's per-request commit -- the twin of `rust_sched.py`'s `burst_commit`,
/// applied to the resident table by [`crate::manager::Manager::table_burst`].
///
/// Same three writes as [`advance`] in the same order (`is_prefill_chunk` is computed
/// BEFORE the placeholder bump, and that ordering is load-bearing on both sides), with the
/// bump made unconditional. Both twins are gated by Python's `burst_invariant_broken`,
/// which admits only the steady-decode shape `C == T + P - 1` with `is_prefill_chunk`
/// clear; on that shape `C + delta >= T + P` holds for every `delta >= 1`, so the
/// condition can only ever be true for an entry the gate already refused -- and for such an
/// entry taking it is the harm, not the safety: `C` advances while `P` does not, and the
/// next step's `num_tokens_with_spec + num_output_placeholders - num_computed_tokens`
/// wraps in usize and skips the request forever.
///
/// Unconditional is also what makes `table_burst`'s negative arm an exact inverse of its
/// positive one on both counters, which the Python rollback depends on.
#[inline]
pub(crate) fn burst_advance(e: &mut SchedReq, delta: usize) {
    e.num_computed_tokens += delta;
    e.is_prefill_chunk = e.num_computed_tokens < e.num_tokens + e.num_output_placeholders;
    e.num_output_placeholders += delta;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, GroupConfig};
    use crate::hash::{Digest32, HASH_LEN};
    use crate::single_type::Kind;

    fn cfg(num_blocks: usize) -> Config {
        Config {
            num_blocks,
            enable_caching: true,
            max_model_len: 4096,
            scheduler_block_size: 16,
            hash_block_size: 16,
            log_stats: false,
            watermark: 0.0,
            radix: false,
            connector: false,
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

    fn params() -> Params {
        Params {
            max_num_scheduled_tokens: 8192,
            max_num_running_reqs: 64,
            max_model_len: 4096,
            num_sampled_tokens_per_step: 1,
            long_prefill_token_threshold: 0,
            enable_chunked_prefill: true,
            need_mamba_block_aligned_split: true,
            cache_block_size: 16,
            num_lookahead_tokens: 0,
            sjf_reorder: false,
            sjf_usage_tight: 0.90,
            lean_decisions: false,
            burst_max_reqs: 0,
            connector: false,
        }
    }

    fn req(slot: u32, prompt: usize) -> SchedReq {
        SchedReq {
            slot,
            num_tokens: prompt,
            num_tokens_with_spec: prompt,
            num_computed_tokens: 0,
            num_output_placeholders: 0,
            num_prompt_tokens: prompt,
            max_tokens: 128,
            status: STATUS_WAITING,
            num_preemptions: 0,
            is_prefill_chunk: true,
            skip_reading_prefix_cache: false,
        }
    }

    fn seed(m: &mut Manager, name: &str, slot_tokens: usize, salt: u8) -> u32 {
        let slot = m.intern(name);
        let n = slot_tokens / 16;
        let packed: Vec<u8> = (0..n)
            .flat_map(|i| {
                let mut d: Digest32 = [salt; HASH_LEN];
                d[0] = i as u8;
                d.to_vec()
            })
            .collect();
        m.push_hashes(slot, &packed, slot_tokens);
        slot
    }

    #[test]
    fn admits_waiting_requests_within_the_token_budget() {
        let mut kv = Manager::new(cfg(512)).unwrap();
        let a = seed(&mut kv, "a", 64, 1);
        let b = seed(&mut kv, "b", 64, 2);
        let mut core = ScheduleCore::new();
        let mut p = params();
        p.max_num_scheduled_tokens = 100;
        core.schedule(&mut kv, &[], &[req(a, 64), req(b, 64)], &p)
            .unwrap();
        // 64 + 64 = 128 > 100: the second is chunked to the remaining 36, then
        // mamba-aligned down to 32.
        assert_eq!(core.decisions.scheduled_admitted.len(), 2);
        assert_eq!(core.decisions.scheduled_admitted[0].1, 64);
        assert_eq!(core.decisions.scheduled_admitted[1].1, 32);
    }

    #[test]
    fn preempts_from_the_tail_when_kv_is_exhausted() {
        // 12 usable blocks; each 64-token request holds 4 attention + 1 mamba block.
        let mut kv = Manager::new(cfg(13)).unwrap();
        let mut core = ScheduleCore::new();
        let p = params();
        let a = seed(&mut kv, "a", 64, 1);
        let b = seed(&mut kv, "b", 64, 2);
        core.schedule(&mut kv, &[], &[req(a, 64), req(b, 64)], &p)
            .unwrap();
        assert_eq!(core.decisions.scheduled_admitted.len(), 2);
        assert_eq!(kv.num_free_blocks(), 2, "10 of 12 blocks held");

        let running: Vec<SchedReq> = [a, b]
            .iter()
            .map(|&s| {
                let mut r = req(s, 64);
                r.num_computed_tokens = 64;
                r.status = crate::manager::STATUS_RUNNING;
                r.num_tokens = 65;
                r.num_tokens_with_spec = 65;
                r.is_prefill_chunk = false;
                r
            })
            .collect();

        // Decode step: each request needs 2 more blocks but only 2 are free. The first
        // takes them; the second cannot allocate and preempts itself off the tail.
        core.schedule(&mut kv, &running, &[], &p).unwrap();
        assert_eq!(core.decisions.scheduled_running.len(), 1);
        assert_eq!(core.decisions.scheduled_running[0].0, running[0].slot);
        assert_eq!(core.decisions.preempted, vec![running[1].slot]);
        assert!(kv.num_free_blocks() >= 5, "the victim's blocks came back");
    }

    /// A pure decode batch: `n` running requests, one token each, nothing waiting.
    fn decode_batch(kv: &mut Manager, n: u32) -> Vec<SchedReq> {
        (0..n)
            .map(|i| {
                let slot = seed(kv, &format!("d{i}"), 64, i as u8 + 1);
                let mut r = req(slot, 64);
                r.num_computed_tokens = 64;
                r.status = crate::manager::STATUS_RUNNING;
                r.num_tokens = 65;
                r.num_tokens_with_spec = 65;
                r.is_prefill_chunk = false;
                r
            })
            .collect()
    }

    #[test]
    fn burst_eligible_is_a_pure_one_token_decode_batch_within_the_cap() {
        let mut kv = Manager::new(cfg(512)).unwrap();
        let mut core = ScheduleCore::new();
        let mut p = params();
        p.burst_max_reqs = 8;
        let running = decode_batch(&mut kv, 3);
        core.schedule(&mut kv, &running, &[], &p).unwrap();
        assert_eq!(core.decisions.scheduled_running.len(), 3);
        assert!(core.decisions.burst_eligible);

        // ...and 0 (burst off / an older plugin) is the "Python re-derives it" fallback.
        p.burst_max_reqs = 0;
        core.schedule(&mut kv, &running, &[], &p).unwrap();
        assert!(!core.decisions.burst_eligible);
    }

    #[test]
    fn burst_eligible_refuses_a_batch_wider_than_the_captured_sizes() {
        let mut kv = Manager::new(cfg(512)).unwrap();
        let mut core = ScheduleCore::new();
        let mut p = params();
        p.burst_max_reqs = 2;
        let running = decode_batch(&mut kv, 3);
        core.schedule(&mut kv, &running, &[], &p).unwrap();
        assert!(!core.decisions.burst_eligible, "3 > the cap of 2");
        p.burst_max_reqs = 3;
        core.schedule(&mut kv, &running, &[], &p).unwrap();
        assert!(core.decisions.burst_eligible, "exactly at the cap is fine");
    }

    #[test]
    fn burst_eligible_refuses_an_admission_a_prefill_chunk_and_an_empty_step() {
        let mut kv = Manager::new(cfg(512)).unwrap();
        let mut core = ScheduleCore::new();
        let mut p = params();
        p.burst_max_reqs = 8;

        // Nothing scheduled at all: there is no step to burst.
        core.schedule(&mut kv, &[], &[], &p).unwrap();
        assert!(!core.decisions.burst_eligible);

        // A waiting admission -- the prefill the burst must not delay.
        let w = seed(&mut kv, "w", 64, 9);
        core.schedule(&mut kv, &[], &[req(w, 64)], &p).unwrap();
        assert_eq!(core.decisions.scheduled_admitted.len(), 1);
        assert!(!core.decisions.burst_eligible);

        // A running request still finishing its prefill schedules > 1 token.
        let mut chunk = decode_batch(&mut kv, 1);
        chunk[0].num_computed_tokens = 32;
        chunk[0].num_tokens = 64;
        chunk[0].num_tokens_with_spec = 64;
        chunk[0].is_prefill_chunk = true;
        core.schedule(&mut kv, &chunk, &[], &p).unwrap();
        assert_eq!(core.decisions.scheduled_running[0].1, 32);
        assert!(!core.decisions.burst_eligible);
    }

    #[test]
    fn burst_eligible_refuses_a_step_that_preempted() {
        // Same 12-usable-block squeeze as `preempts_from_the_tail_when_kv_is_exhausted`:
        // the surviving request IS a 1-token decode, but the victim has to be re-admitted
        // and a burst would delay that by N-1 iterations.
        let mut kv = Manager::new(cfg(13)).unwrap();
        let mut core = ScheduleCore::new();
        let mut p = params();
        p.burst_max_reqs = 8;
        let a = seed(&mut kv, "a", 64, 1);
        let b = seed(&mut kv, "b", 64, 2);
        core.schedule(&mut kv, &[], &[req(a, 64), req(b, 64)], &p)
            .unwrap();
        let running: Vec<SchedReq> = [a, b]
            .iter()
            .map(|&s| {
                let mut r = req(s, 64);
                r.num_computed_tokens = 64;
                r.status = crate::manager::STATUS_RUNNING;
                r.num_tokens = 65;
                r.num_tokens_with_spec = 65;
                r.is_prefill_chunk = false;
                r
            })
            .collect();
        core.schedule(&mut kv, &running, &[], &p).unwrap();
        assert_eq!(core.decisions.scheduled_running.len(), 1);
        assert!(!core.decisions.preempted.is_empty());
        assert!(!core.decisions.burst_eligible);
    }

    #[test]
    fn sjf_reorder_puts_the_shortest_remaining_prefill_first() {
        let mut kv = Manager::new(cfg(512)).unwrap();
        let long = seed(&mut kv, "long", 512, 1);
        let short = seed(&mut kv, "short", 64, 2);
        let mut core = ScheduleCore::new();
        let mut p = params();
        p.sjf_reorder = true;
        p.max_num_scheduled_tokens = 64;
        core.schedule(&mut kv, &[], &[req(long, 512), req(short, 64)], &p)
            .unwrap();
        assert_eq!(
            core.decisions.scheduled_admitted[0].0, short,
            "the 64-token prompt must jump the 512-token one"
        );
    }

    #[test]
    fn lean_decisions_skips_the_common_prefix_epilogue() {
        // Empty is the signal `decisions_dict` reads to omit the key entirely; every
        // non-lean arm pushes one entry per KV group, so it can never be empty by accident.
        let mut kv = Manager::new(cfg(512)).unwrap();
        let a = seed(&mut kv, "a", 64, 1);
        let mut core = ScheduleCore::new();
        let mut p = params();

        core.schedule(&mut kv, &[], &[req(a, 64)], &p).unwrap();
        assert_eq!(core.decisions.num_common_prefix_blocks.len(), 2);
        let admitted = core.decisions.scheduled_admitted.clone();

        p.lean_decisions = true;
        let mut kv2 = Manager::new(cfg(512)).unwrap();
        let a2 = seed(&mut kv2, "a", 64, 1);
        let mut core2 = ScheduleCore::new();
        core2.schedule(&mut kv2, &[], &[req(a2, 64)], &p).unwrap();
        assert!(core2.decisions.num_common_prefix_blocks.is_empty());
        // ...and nothing else moved.
        assert_eq!(core2.decisions.scheduled_admitted, admitted);
    }

    #[test]
    fn running_new_blocks_are_the_step_delta_not_the_whole_table() {
        // scheduler.py:588 stores only what THIS step's allocate_slots returned for a
        // running request. Re-handing the full table makes the V2 runner's
        // append_block_ids(overwrite=False) duplicate every row, every step.
        let p = params();
        let mut kv = Manager::new(cfg(512)).unwrap();
        let a = seed(&mut kv, "a", 64, 1);
        let mut core = ScheduleCore::new();
        core.schedule(&mut kv, &[], &[req(a, 64)], &p).unwrap();
        let before: Vec<usize> = (0..2).map(|g| kv.group_blocks(a, g).len()).collect();

        let mut r = req(a, 64);
        r.num_computed_tokens = 64;
        r.num_tokens = 65;
        r.num_tokens_with_spec = 65;
        r.status = crate::manager::STATUS_RUNNING;
        r.is_prefill_chunk = false;
        core.schedule(&mut kv, &[r], &[], &p).unwrap();
        assert_eq!(core.decisions.scheduled_running.len(), 1);
        assert_eq!(core.decisions.scheduled_running[0].1, 1);

        // Independent replay: the same manager state driven straight through
        // allocate_slots must produce exactly the recorded spans.
        let mut kv2 = Manager::new(cfg(512)).unwrap();
        let a2 = seed(&mut kv2, "a", 64, 1);
        let mut core2 = ScheduleCore::new();
        core2.schedule(&mut kv2, &[], &[req(a2, 64)], &p).unwrap();
        kv2.new_step_starts();
        assert!(kv2
            .allocate_slots(
                a2, 1, 0, false, 0, 64, 65, crate::manager::STATUS_RUNNING, true, 0, false, 0,
            )
            .unwrap());

        let d = &core.decisions;
        assert_eq!(d.running_new_lens.len(), 2, "one length per group per request");
        let mut off = 0usize;
        for g in 0..2 {
            let n = d.running_new_lens[g] as usize;
            let span = &d.running_new_blocks[off..off + n];
            off += n;
            assert_eq!(n, kv.group_blocks(a, g).len() - before[g], "group {g} growth");
            assert_eq!(span, &kv2.new_blocks[g][..], "group {g} == allocate_slots delta");
        }
        assert_eq!(off, d.running_new_blocks.len());
        assert!(
            d.running_new_blocks.len() < before[0] + before[1],
            "the delta must be smaller than the table it extends"
        );
    }

    #[test]
    fn marconi_hint_does_not_leak_into_a_resumed_request() {
        // scheduler.py:684 zeroes num_uncached_common_prefix_tokens per iteration and
        // only refills it after a fresh get_computed_blocks (`:730`). A resumed request
        // (num_computed_tokens > 0) skips that call, so reading the coordinator would
        // splice the PREVIOUS request's hint into its mamba-aligned split.
        let mut p = params();
        p.max_num_scheduled_tokens = 8192;
        let mut kv = Manager::new(cfg(512)).unwrap();
        let mut core = ScheduleCore::new();

        let base = seed(&mut kv, "base", 64, 1);
        core.schedule(&mut kv, &[], &[req(base, 64)], &p).unwrap();
        assert_eq!(core.decisions.scheduled_admitted.len(), 1);

        // `dup` re-sends the identical 64 tokens: the full-attention chain matches 3
        // blocks but the only cached mamba state sits at token 64, so the hit collapses
        // to 0 and leaves a 48-token uncached common prefix behind.
        let dup = seed(&mut kv, "dup", 64, 1);
        let mut resumed = req(seed(&mut kv, "resumed", 200, 9), 200);
        resumed.num_computed_tokens = 32;
        resumed.status = STATUS_PREEMPTED;

        core.schedule(&mut kv, &[], &[req(dup, 64), resumed], &p)
            .unwrap();
        assert_eq!(kv.coord.num_uncached_common_prefix_tokens, 48, "hint is live");
        assert_eq!(core.decisions.scheduled_admitted.len(), 2);
        // 200 - 200 % 16 = 192 is the last cacheable position, so the chunk is 192 - 32.
        // With the stale hint it would have been floored to 48.
        assert_eq!(core.decisions.scheduled_admitted[1].1, 160);
    }

    /// B4a: an entry with more placeholders than `C + 2` is exactly the state the old
    /// subtraction form wrapped on (debug: panicked on). The addition form has to reach a
    /// verdict without either, and the rest of the batch has to be scheduled around it.
    #[test]
    fn a_placeholder_count_past_the_computed_tokens_does_not_wrap_the_max_tokens_guard() {
        let mut kv = Manager::new(cfg(512)).unwrap();
        let mut core = ScheduleCore::new();
        let p = params();
        let mut running = decode_batch(&mut kv, 2);
        // P = 70 with C = 64: `C + 2 - P` is -4 over the integers, i.e. a wrap in usize.
        // Over the integers the guard is false (66 < 64 + 128 + 70 either way), so the
        // request stays schedulable -- and `T + P - C` is 65 + 70 - 64 = 71, a real count.
        running[0].num_output_placeholders = 70;
        core.schedule(&mut kv, &running, &[], &p).unwrap();
        assert_eq!(core.inconsistent, 0, "a large P is not a broken entry");
        assert_eq!(core.decisions.scheduled_running.len(), 2);
        assert_eq!(core.decisions.scheduled_running[0].1, 71);
        assert_eq!(core.decisions.scheduled_running[1].1, 1, "the healthy one is untouched");
    }

    /// B4b: `C` past `T + P` is the skew that used to wrap the work derivation into a
    /// near-`usize::MAX` count (or panic in debug). It must be counted and skipped, with the
    /// rest of the batch scheduled normally.
    #[test]
    fn computed_tokens_past_num_tokens_plus_placeholders_is_counted_and_skipped() {
        let mut kv = Manager::new(cfg(512)).unwrap();
        let mut core = ScheduleCore::new();
        let p = params();
        let mut running = decode_batch(&mut kv, 2);
        // T = 65, P = 0, C = 100: the entry promises 65 tokens and claims 100 computed.
        running[0].num_computed_tokens = 100;
        core.schedule(&mut kv, &running, &[], &p).unwrap();
        assert_eq!(core.inconsistent, 1, "the broken entry is counted, once");
        assert_eq!(
            core.decisions.scheduled_running.len(),
            1,
            "...and skipped, while the healthy request in the same batch is scheduled"
        );
        assert_eq!(core.decisions.scheduled_running[0].0, running[1].slot);
        // Monotonic across steps -- the Python probe watches it for an INCREASE.
        core.schedule(&mut kv, &running, &[], &p).unwrap();
        assert_eq!(core.inconsistent, 2);
    }

    /// The addition form is the same predicate as the subtraction form on every state that
    /// can legitimately occur, which is the whole justification for rewriting it. Spot-check
    /// the two against each other over a grid of healthy values (`P <= C + 2`, so the old
    /// form is even computable).
    #[test]
    fn the_max_tokens_guard_matches_the_subtraction_form_on_healthy_values() {
        for c in 0..40usize {
            for p in 0..=(c + 2) {
                for prompt in 0..8usize {
                    for max_tokens in 0..8usize {
                        let old = c + 2 - p >= prompt + max_tokens;
                        let new = c + 2 >= prompt + max_tokens + p;
                        assert_eq!(old, new, "C={c} P={p} prompt={prompt} max={max_tokens}");
                    }
                }
            }
        }
    }

    // ---- Stage S: the phase split ---------------------------------------------------

    /// A connector-off boot must not be able to tell that the loop was cut in two. The
    /// unsplit `schedule()` and an explicit `schedule_phase1(probe=[]) ->
    /// schedule_phase2(waiting, external=[])` are run against two managers built from the
    /// same seed, and BOTH the decisions and the whole manager state have to match.
    ///
    /// `sjf_reorder` stays off here on purpose: the public `schedule_phase2` runs the
    /// reorder itself (its candidates do not exist at phase 1), so its two memory-pressure
    /// keys are read after the running arm's allocations rather than before. That is the
    /// one documented divergence of the split, and it is connector-only — `run()` still
    /// reorders where it always did, which the `sjf_reorder_*` test above pins.
    #[test]
    fn the_phase_split_is_bit_identical_to_the_unsplit_loop_without_a_connector() {
        // A tiny LCG so the fixtures are varied and reproducible without a dev-dependency.
        let mut rng = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move |n: u64| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng % n
        };
        for scenario in 0..24u32 {
            let blocks = 24 + next(200) as usize;
            let n_run = next(4) as usize;
            let n_wait = next(4) as usize;
            let budget = 16 + 32 * next(8) as usize;
            let mut p = params();
            p.max_num_scheduled_tokens = budget;
            p.max_num_running_reqs = 1 + next(5) as usize;
            p.lean_decisions = next(2) == 1;

            // The two fixtures are built by the same closure, so they are identical up to
            // the entry point under test.
            let build = |prompts: &[(usize, u8)]| {
                let mut kv = Manager::new(cfg(blocks)).unwrap();
                let mut reqs = Vec::new();
                for (i, &(len, salt)) in prompts.iter().enumerate() {
                    let slot = seed(&mut kv, &format!("r{i}"), len, salt);
                    reqs.push(req(slot, len));
                }
                (kv, reqs)
            };
            let prompts: Vec<(usize, u8)> = (0..n_run + n_wait)
                .map(|i| ((1 + next(6) as usize) * 32, (scenario as u8) * 8 + i as u8 + 1))
                .collect();

            let (mut kv_a, reqs_a) = build(&prompts);
            let (mut kv_b, reqs_b) = build(&prompts);
            // The running half is put in a decode-ish state; the waiting half is fresh.
            let running = |reqs: &[SchedReq]| -> Vec<SchedReq> {
                reqs[..n_run]
                    .iter()
                    .map(|r| {
                        let mut r = *r;
                        r.num_computed_tokens = r.num_tokens;
                        r.num_tokens += 1;
                        r.num_tokens_with_spec = r.num_tokens;
                        r.status = STATUS_RUNNING;
                        r.is_prefill_chunk = false;
                        r
                    })
                    .collect()
            };
            let (run_a, run_b) = (running(&reqs_a), running(&reqs_b));
            let (wait_a, wait_b) = (&reqs_a[n_run..], &reqs_b[n_run..]);

            let mut core_a = ScheduleCore::new();
            let unsplit = core_a
                .schedule(&mut kv_a, &run_a, wait_a, &p)
                .unwrap()
                .clone();

            let mut core_b = ScheduleCore::new();
            core_b.schedule_phase1(&mut kv_b, &run_b, &p, &[]).unwrap();
            let split = core_b
                .schedule_phase2(&mut kv_b, &p, wait_b, &[], 0)
                .unwrap()
                .clone();

            assert_eq!(unsplit, split, "scenario {scenario}: decisions");
            assert_eq!(
                kv_a.state_fingerprint(),
                kv_b.state_fingerprint(),
                "scenario {scenario}: manager state"
            );
            assert!(split.parked_external.is_empty(), "no connector, no parked loads");
        }
    }

    /// The same equivalence for the RESIDENT entry point, which reads the running set out
    /// of the table instead of a marshalled slice.
    #[test]
    fn the_resident_phase_split_is_bit_identical_to_schedule_resident() {
        let p = params();
        let mut kv_a = Manager::new(cfg(512)).unwrap();
        let mut kv_b = Manager::new(cfg(512)).unwrap();
        let mut core_a = ScheduleCore::new();
        let mut core_b = ScheduleCore::new();
        let (a, b) = (seed(&mut kv_a, "a", 64, 1), seed(&mut kv_b, "a", 64, 1));
        // One marshalled step first, to populate both resident tables identically.
        core_a.schedule(&mut kv_a, &[], &[req(a, 64)], &p).unwrap();
        core_b.schedule(&mut kv_b, &[], &[req(b, 64)], &p).unwrap();
        let w_a = seed(&mut kv_a, "w", 128, 9);
        let w_b = seed(&mut kv_b, "w", 128, 9);

        let unsplit = core_a
            .schedule_resident(&mut kv_a, &[a], &[req(w_a, 128)], &p)
            .unwrap()
            .clone();
        core_b
            .schedule_phase1_resident(&mut kv_b, &[b], &p, &[])
            .unwrap();
        let split = core_b
            .schedule_phase2(&mut kv_b, &p, &[req(w_b, 128)], &[], 0)
            .unwrap()
            .clone();
        assert_eq!(unsplit, split);
        assert_eq!(kv_a.state_fingerprint(), kv_b.state_fingerprint());
    }

    /// The probe's hit is what phase 2 allocates against — it must not walk the cache a
    /// second time, and the blocks recorded in phase 1 have to reach `allocate_slots`.
    #[test]
    fn phase2_consumes_the_probe_hit_instead_of_re_walking() {
        let mut p = params();
        p.connector = true;
        let mut c = cfg(512);
        c.connector = true;
        let mut kv = Manager::new(c).unwrap();
        let mut core = ScheduleCore::new();
        // `base` installs 64 tokens' worth of blocks in the prefix cache; `dup` re-sends
        // the identical prompt, so its probe finds them.
        let base = seed(&mut kv, "base", 64, 1);
        core.schedule(&mut kv, &[], &[req(base, 64)], &p).unwrap();
        let dup = seed(&mut kv, "dup", 64, 1);

        core.schedule_phase1(&mut kv, &[], &p, &[req(dup, 64)]).unwrap();
        let hits = kv.probe_hits();
        assert_eq!(hits.len(), 1);
        let (slot, hit) = hits[0];
        assert_eq!(slot, dup);
        assert!(hit > 0, "the identical prompt must hit the prefix cache");

        let d = core
            .schedule_phase2(&mut kv, &p, &[req(dup, 64)], &[], 0)
            .unwrap()
            .clone();
        assert_eq!(d.scheduled_admitted.len(), 1);
        assert_eq!(
            d.scheduled_admitted[0].2, hit,
            "num_computed_tokens is the probe's hit"
        );
        // Consumed: nothing is left behind for a later step to re-install.
        assert!(kv.probe_hits().is_empty());
    }

    #[test]
    fn phase2_parks_an_async_candidate_and_admits_the_next() {
        let mut p = params();
        p.connector = true;
        let mut c = cfg(512);
        c.connector = true;
        let mut kv = Manager::new(c).unwrap();
        let mut core = ScheduleCore::new();
        let park = seed(&mut kv, "park", 128, 1);
        let plain = seed(&mut kv, "plain", 64, 2);
        let waiting = [req(park, 128), req(plain, 64)];

        core.schedule_phase1(&mut kv, &[], &p, &waiting).unwrap();
        // The connector claims 64 of the parked request's tokens, asynchronously.
        let d = core
            .schedule_phase2(&mut kv, &p, &waiting, &[(park, 64, true)], 0)
            .unwrap()
            .clone();

        assert_eq!(d.parked_external, vec![(park, 64, 64)], "local hit is 0 here");
        assert_eq!(
            d.scheduled_admitted.len(),
            1,
            "the parked load does not stop the queue"
        );
        assert_eq!(d.scheduled_admitted[0].0, plain);
        // A parked request is consumed from the queue: Python parks it in
        // `skipped_waiting`, so it must not come back in `waiting_order`.
        assert!(d.waiting_order.is_empty());
        // ...and it holds blocks for the tokens the transfer will land.
        assert!(kv.group_blocks(park, 0).len() >= 4);
        assert!(!d.burst_eligible, "a parked load is an admission in waiting");
    }

    #[test]
    fn phase2_refuses_synchronous_external_tokens() {
        let mut p = params();
        p.connector = true;
        let mut c = cfg(512);
        c.connector = true;
        let mut kv = Manager::new(c).unwrap();
        let mut core = ScheduleCore::new();
        let a = seed(&mut kv, "a", 64, 1);
        core.schedule_phase1(&mut kv, &[], &p, &[req(a, 64)]).unwrap();
        let err = core
            .schedule_phase2(&mut kv, &p, &[req(a, 64)], &[(a, 32, false)], 0)
            .unwrap_err();
        assert!(err.contains("sync external tokens"), "{err}");

        // ...and an async load with nothing to load is stock's assert at :799.
        core.schedule_phase1(&mut kv, &[], &p, &[req(a, 64)]).unwrap();
        let err = core
            .schedule_phase2(&mut kv, &p, &[req(a, 64)], &[(a, 0, true)], 0)
            .unwrap_err();
        assert!(err.contains("must carry external computed tokens"), "{err}");

        // ...and external work at all without a live connector is a wiring bug.
        let mut off = params();
        off.connector = false;
        let err = core
            .schedule_phase2(&mut kv, &off, &[req(a, 64)], &[(a, 32, true)], 0)
            .unwrap_err();
        assert!(err.contains("without a live connector"), "{err}");
    }

    /// scheduler.py:966 adds each parked load to `_inflight_prefills` immediately, so the
    /// NEXT candidate's `reserved_blocks` already counts it. The crate does that in-loop:
    /// with a pool sized to hold exactly one of the two loads' reservations, the second
    /// must be refused by the reserved gate — and admitted once the reservation is gone.
    #[test]
    fn parked_loads_accumulate_their_reservation_within_the_loop() {
        let mut p = params();
        p.connector = true;
        let build = || {
            // 26 usable blocks. A 256-token request parks on 9 (8 attention + 1 mamba for
            // the 128 tokens the transfer will land) and RESERVES 9 more for the rest of
            // its sequence, so the pool holds one load plus its reservation and no more.
            let mut c = cfg(27);
            c.connector = true;
            let mut kv = Manager::new(c).unwrap();
            let a = seed(&mut kv, "pa", 256, 1);
            let b = seed(&mut kv, "pb", 256, 2);
            (kv, a, b)
        };

        let (mut kv, a, b) = build();
        let mut core = ScheduleCore::new();
        let waiting = [req(a, 256), req(b, 256)];
        eprintln!("free before {}", kv.num_free_blocks());
        core.schedule_phase1(&mut kv, &[], &p, &waiting).unwrap();
        let d = core
            .schedule_phase2(&mut kv, &p, &waiting, &[(a, 128, true), (b, 128, true)], 0)
            .unwrap()
            .clone();
        assert_eq!(
            d.parked_external.len(),
            1,
            "the first load's reservation must gate the second: {:?}",
            d.parked_external
        );
        assert_eq!(d.parked_external[0].0, a);

        // The same second candidate, alone, fits — so the refusal above was the
        // RESERVATION and not the pool being out of blocks.
        let (mut kv2, _a2, b2) = build();
        let mut core2 = ScheduleCore::new();
        let solo = [req(b2, 256)];
        core2.schedule_phase1(&mut kv2, &[], &p, &solo).unwrap();
        let d2 = core2
            .schedule_phase2(&mut kv2, &p, &solo, &[(b2, 128, true)], 0)
            .unwrap()
            .clone();
        assert_eq!(d2.parked_external.len(), 1);

        // ...and `base_reserved` (the prefills already in flight when the step started) is
        // the same gate from the other side: one whole pool's worth refuses even the first.
        let (mut kv3, a3, _b3) = build();
        let mut core3 = ScheduleCore::new();
        let one = [req(a3, 256)];
        core3.schedule_phase1(&mut kv3, &[], &p, &one).unwrap();
        let d3 = core3
            .schedule_phase2(&mut kv3, &p, &one, &[(a3, 128, true)], 64)
            .unwrap()
            .clone();
        assert!(d3.parked_external.is_empty(), "base_reserved gates the first load");
    }

    /// C3. A frozen probe is only as good as the pool it was walked against, and phase 2
    /// keeps allocating between the walk and the consume: an earlier candidate's
    /// `get_new_blocks` evicts cached-but-free blocks as it pops them, and one of those can
    /// be a block a later candidate's probe still lists. Consuming it would `touch` a block
    /// the earlier candidate now owns -- two requests, one block.
    ///
    /// The two arms split on whether the CONNECTOR was told the stale number, because that
    /// is what decides whether a fresh walk is still admissible.
    fn stale_probe_fixture() -> (Manager, ScheduleCore, Params, u32, u32) {
        // ONE full-attention group, so the block arithmetic below is exact. 19 usable
        // blocks (block 0 is the null placeholder), laid out deliberately: the free queue's
        // ORDER is what decides which blocks an allocation evicts, and the eviction has to
        // land on a block the probe recorded.
        let mut c = cfg(20);
        c.groups.truncate(1);
        c.connector = true;
        let mut p = params();
        p.connector = true;
        p.need_mamba_block_aligned_split = false;

        let mut kv = Manager::new(c).unwrap();
        let mut core = ScheduleCore::new();
        // `filler` takes 1..10 and KEEPS them: it spends the unhashed head of the queue, so
        // everything freed below lands ahead of what is left.
        let filler = seed(&mut kv, "filler", 160, 3);
        core.schedule(&mut kv, &[], &[req(filler, 160)], &p).unwrap();
        // `base` takes 11..14 and caches all four; `extra` takes 15,16.
        let base = seed(&mut kv, "base", 64, 1);
        core.schedule(&mut kv, &[], &[req(base, 64)], &p).unwrap();
        let extra = seed(&mut kv, "extra", 32, 5);
        core.schedule(&mut kv, &[], &[req(extra, 32)], &p).unwrap();
        // Freed hashed blocks are APPENDED in eviction order (`TypeManager::free` reverses
        // the request's list), so the queue is now 17,18,19, 14,13,12,11, 16,15 -- the
        // three untouched blocks, then `base`'s, then `extra`'s.
        kv.free(base);
        kv.free(extra);

        // `dup` re-sends `base`'s prompt: its probe hits 3 blocks (the walk caps the hit at
        // `num_tokens - 1`), namely 11,12,13. `hog` needs 5 blocks and pops 17,18,19,14,13
        // -- evicting 13 out from under `dup`'s recorded hit, and leaving 12,11,16,15 free,
        // which is exactly what `dup` needs to be admitted on a re-walk.
        let hog = seed(&mut kv, "hog", 80, 7);
        let dup = seed(&mut kv, "dup", 64, 1);
        let waiting = [req(hog, 80), req(dup, 64)];
        core.schedule_phase1(&mut kv, &[], &p, &waiting).unwrap();
        let hits: Vec<(u32, usize)> = kv.probe_hits();
        let dup_hit = hits.iter().find(|h| h.0 == dup).unwrap().1;
        assert_eq!(dup_hit, 48, "the probe recorded a 3-block hit");
        (kv, core, p, hog, dup)
    }

    #[test]
    fn a_stale_probe_is_skipped_when_the_connector_was_told_its_hit() {
        let (mut kv, mut core, p, hog, dup) = stale_probe_fixture();
        let waiting = [req(hog, 80), req(dup, 64)];
        // The connector answered for `dup` against the 48-token hit; its `keys_to_load`
        // window is keyed to that boundary, so a fresh walk cannot be substituted.
        let d = core
            .schedule_phase2(&mut kv, &p, &waiting, &[(dup, 16, true)], 0)
            .unwrap()
            .clone();
        assert_eq!(d.scheduled_admitted.len(), 1, "the hog is admitted");
        assert_eq!(d.scheduled_admitted[0].0, hog);
        assert_eq!(d.probe_stale, vec![dup], "the stale candidate is handed back");
        assert!(d.parked_external.is_empty(), "and NOT parked on the load");
        assert!(
            !d.waiting_order.contains(&dup),
            "it was consumed from the queue: Python owns it now (skipped_waiting)"
        );
    }

    #[test]
    fn a_stale_probe_without_external_tokens_re_walks_to_the_smaller_hit() {
        let (mut kv, mut core, p, hog, dup) = stale_probe_fixture();
        let waiting = [req(hog, 80), req(dup, 64)];
        // No connector answer for `dup`: nothing outside the crate has seen the stale
        // number, so the honest fix is the walk stock would have done anyway.
        let d = core
            .schedule_phase2(&mut kv, &p, &waiting, &[], 0)
            .unwrap()
            .clone();
        assert!(d.probe_stale.is_empty());
        assert_eq!(d.scheduled_admitted.len(), 2, "both are admitted");
        assert_eq!(d.scheduled_admitted[1].0, dup);
        assert_eq!(
            d.scheduled_admitted[1].2, 32,
            "the third hit block was evicted, so the fresh walk stops at two"
        );
    }

    /// The other half of the same guard: a probe nothing disturbed is NOT stale, so the
    /// revalidation cannot be paying for itself by simply refusing every hit.
    #[test]
    fn an_undisturbed_probe_still_consumes_its_recorded_hit() {
        let mut c = cfg(16);
        c.groups.truncate(1);
        c.connector = true;
        let mut p = params();
        p.connector = true;
        p.need_mamba_block_aligned_split = false;
        let mut kv = Manager::new(c).unwrap();
        let mut core = ScheduleCore::new();
        let base = seed(&mut kv, "base", 64, 1);
        core.schedule(&mut kv, &[], &[req(base, 64)], &p).unwrap();
        let dup = seed(&mut kv, "dup", 64, 1);
        core.schedule_phase1(&mut kv, &[], &p, &[req(dup, 64)]).unwrap();
        let d = core
            .schedule_phase2(&mut kv, &p, &[req(dup, 64)], &[], 0)
            .unwrap()
            .clone();
        assert!(d.probe_stale.is_empty());
        assert_eq!(d.scheduled_admitted[0].2, 48, "the recorded hit was used as-is");
    }

    #[test]
    fn mamba_split_aligns_chunks_to_the_block_size() {
        let r = req(0, 1000);
        // Mid-prefill chunk gets floored to a block multiple.
        assert_eq!(mamba_block_aligned_split(&r, 100, 0, 0, 16), 96);
        // A chunk crossing the last cacheable position is snapped to it.
        let mut r2 = req(0, 1000);
        r2.num_computed_tokens = 980;
        assert_eq!(mamba_block_aligned_split(&r2, 20, 0, 0, 16), 12);
        // Decode (past the prefill end) is untouched.
        let mut r3 = req(0, 100);
        r3.num_computed_tokens = 100;
        assert_eq!(mamba_block_aligned_split(&r3, 1, 0, 0, 16), 1);
    }
}
