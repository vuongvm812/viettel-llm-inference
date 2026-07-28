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

use crate::manager::{Manager, STATUS_PREEMPTED, STATUS_WAITING};

/// Mirror of the `Request` fields the scheduling loop reads.
#[derive(Clone, Copy, Debug, Default)]
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

#[derive(Clone, Copy, Debug)]
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
}

#[derive(Default, Debug)]
pub struct Decisions {
    /// `(slot, num_new_tokens)` for requests already in `running`.
    pub scheduled_running: Vec<(u32, usize)>,
    /// `(slot, num_new_tokens, num_computed_tokens)` admitted from `waiting`; `status`
    /// at admission time decides new vs resumed on the Python side.
    pub scheduled_admitted: Vec<(u32, usize, usize)>,
    /// Slots popped off the tail of `running`, in preemption order.
    pub preempted: Vec<u32>,
    /// Final `waiting`-queue order (slots) after the SJF reorder, front first.
    pub waiting_order: Vec<u32>,
    pub num_common_prefix_blocks: Vec<usize>,
    pub token_budget_left: usize,
}

impl Decisions {
    fn clear(&mut self) {
        self.scheduled_running.clear();
        self.scheduled_admitted.clear();
        self.preempted.clear();
        self.waiting_order.clear();
        self.num_common_prefix_blocks.clear();
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

    /// The scheduling loop. `running` and `waiting` are queue-ordered snapshots.
    pub fn schedule(
        &mut self,
        kv: &mut Manager,
        running: &[SchedReq],
        waiting: &[SchedReq],
        params: &Params,
    ) -> Result<(), String> {
        self.decisions.clear();
        self.running.clear();
        self.running.extend_from_slice(running);
        self.waiting.clear();
        self.waiting.extend_from_slice(waiting);

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
            if request.num_output_placeholders > 0
                && request.num_computed_tokens + 2 - request.num_output_placeholders
                    >= request.num_prompt_tokens + request.max_tokens
            {
                req_index += 1;
                continue;
            }

            let mut num_new_tokens = request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens;
            if params.long_prefill_token_threshold > 0
                && params.long_prefill_token_threshold < num_new_tokens
            {
                num_new_tokens = params.long_prefill_token_threshold;
            }
            num_new_tokens = num_new_tokens.min(token_budget);
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
            token_budget -= num_new_tokens;
            req_index += 1;
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
                let uncached_common = kv.coord.num_uncached_common_prefix_tokens;

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
        let mut common = std::mem::take(&mut self.decisions.num_common_prefix_blocks);
        common.clear();
        if let Some(first) = self.running.first() {
            common.extend_from_slice(kv.get_num_common_prefix_blocks(first.slot));
        } else {
            common.extend(std::iter::repeat(0).take(kv.coord.managers.len()));
        }
        self.decisions.num_common_prefix_blocks = common;
        self.decisions.token_budget_left = token_budget;
        // Waiting requests that were admitted are dropped from the front; the rest keep
        // their (possibly reordered) order so Python can rewrite its deque verbatim.
        for r in &self.waiting[admitted_from_waiting..] {
            self.decisions.waiting_order.push(r.slot);
        }
        Ok(())
    }
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
        assert_eq!(core.decisions.token_budget_left, 4);
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
