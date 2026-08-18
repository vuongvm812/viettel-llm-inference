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

use crate::manager::{Manager, STATUS_PREEMPTED, STATUS_RUNNING, STATUS_WAITING};

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
    /// The waiting arm needs NO branch of its own for it: a connector that only serves
    /// SYNCHRONOUS hits changes nothing here, and an ASYNC load never reaches this loop —
    /// `schedule_supported` refuses a scheduler with a connector attached, and the parked
    /// request path lives entirely in Python (`Manager::allocate_external`). The field is
    /// carried so the loop can refuse loudly rather than silently mis-schedule if that
    /// ever changes.
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
        self.running.clear();
        self.running.extend_from_slice(running);
        for r in running {
            kv.table_set(r.slot, *r);
        }
        self.waiting.clear();
        self.waiting.extend_from_slice(waiting);
        self.run(kv, params)
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
        self.waiting.clear();
        self.waiting.extend_from_slice(waiting);
        self.run(kv, params)
    }

    /// THE loop body, shared by both entry points so they cannot drift apart. `running`
    /// and `waiting` are already loaded.
    fn run(&mut self, kv: &mut Manager, params: &Params) -> Result<&Decisions, String> {
        // Speculation's scope invariant (see `journal.rs`): the waiting half of the loop
        // reaches prefix-cache lookup paths that are deliberately not journaled. Refuse
        // rather than approximate.
        if kv.journal_armed() && !self.waiting.is_empty() {
            return Err("speculation refuses a non-empty waiting queue".into());
        }
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

        // ---- 2. WAITING requests (scheduler.py:636) ------------------------
        let mut admitted_from_waiting = 0usize;
        if self.decisions.preempted.is_empty() {
            while admitted_from_waiting < self.waiting.len() && token_budget > 0 {
                if self.running.len() >= params.max_num_running_reqs {
                    break;
                }
                let request = self.waiting[admitted_from_waiting];

                let (num_new_local_computed_tokens, use_pending_hit) =
                    if request.num_computed_tokens == 0 {
                        let hit = kv.get_computed_blocks(
                            request.slot,
                            request.num_tokens,
                            request.num_preemptions,
                            request.skip_reading_prefix_cache,
                        );
                        (hit, true)
                    } else {
                        (0, false)
                    };
                let num_computed_tokens = if request.num_computed_tokens == 0 {
                    num_new_local_computed_tokens
                } else {
                    request.num_computed_tokens
                };
                // scheduler.py:684 zeroes this per iteration and only refills it (`:730`)
                // right after a fresh `get_computed_blocks`. Reading the coordinator
                // unconditionally would feed the PREVIOUS request's Marconi hint into
                // `mamba_block_aligned_split` whenever num_computed_tokens != 0.
                let uncached_common = if request.num_computed_tokens == 0 {
                    kv.coord.num_uncached_common_prefix_tokens
                } else {
                    0
                };

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
                    // An ASYNC connector load is the only producer of these three, and it
                    // never enters this loop (`Params::connector`). A synchronous connector
                    // hit contributes no external tokens either.
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
