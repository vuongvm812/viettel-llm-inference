//! R6b + speculation: the tests that would catch a drift between the two scheduling
//! entry points, or a journal that does not undo exactly what it recorded.
//!
//! All PyO3-free — the Docker builder runs `cargo test --release` in an image with no
//! libpython, and that gate is the whole reason the crate core has no PyO3 dependency.
//!
//! The harness below is a deliberate re-implementation of what `rust_sched.py` does to
//! the Python `Request` objects around each `schedule()` / `update_from_output()`. Running
//! it in `Marshalled` mode reproduces today's shipped path; running the same event stream
//! in `Resident` mode reproduces R6b. If the Rust-side post-schedule commit disagrees with
//! Python's bookkeeping by one token anywhere, the two streams diverge and the test says so.

use rustc_hash::{FxHashMap, FxHashSet};

use vtl_sched::config::{Config, GroupConfig};
use vtl_sched::hash::{Digest32, HASH_LEN};
use vtl_sched::manager::{Manager, STATUS_PREEMPTED, STATUS_RUNNING, STATUS_WAITING};
use vtl_sched::sched::{Decisions, Params, SchedReq, ScheduleCore};
use vtl_sched::single_type::Kind;
use vtl_sched::spec::Shared;
use vtl_sched::update::StopParams;

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

fn cfg(num_blocks: usize, radix: bool) -> Config {
    Config {
        num_blocks,
        enable_caching: true,
        max_model_len: 2048,
        scheduler_block_size: 16,
        hash_block_size: 16,
        log_stats: false,
        watermark: 0.0,
        radix,
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
        max_num_scheduled_tokens: 512,
        max_num_running_reqs: 6,
        max_model_len: 2048,
        num_sampled_tokens_per_step: 1,
        long_prefill_token_threshold: 0,
        enable_chunked_prefill: true,
        need_mamba_block_aligned_split: true,
        cache_block_size: 16,
        num_lookahead_tokens: 0,
        sjf_reorder: false,
        sjf_usage_tight: 0.90,
        lean_decisions: false,
        // The burst gate is a `rust_sched.py` handshake, not part of the resident/spec
        // parity this suite covers; 0 keeps `burst_eligible` false throughout.
        burst_max_reqs: 0,
    }
}

/// Deterministic LCG — same generator style as the existing `radix.rs` parity test, so a
/// failure is reproducible from the seed alone.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

// ---------------------------------------------------------------------------
// harness
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// `Scheduler.schedule` with packed running snapshots (the shipped path).
    Marshalled,
    /// `Scheduler.schedule_resident` reading the running set out of the Rust table.
    Resident,
}

struct Harness {
    kv: Manager,
    core: ScheduleCore,
    params: Params,
    mode: Mode,
    /// Python's authority: what the `Request` objects would hold.
    reqs: FxHashMap<u32, SchedReq>,
    num_output_tokens: FxHashMap<u32, usize>,
    running: Vec<u32>,
    waiting: Vec<u32>,
    names: FxHashMap<u32, String>,
    next_id: usize,
    /// Slots whose sampled token is still on the GPU. ASYNC SCHEDULING is the whole point
    /// of `num_output_placeholders`: the engine runs `schedule(N)` before
    /// `update_from_output(N - 1)` lands, so the scheduler has to reason about tokens that
    /// exist only as a count. A harness that applied the output synchronously would never
    /// exercise that field, and every placeholder bug would sail through.
    inflight: Vec<u32>,
    /// Slots that were preempted and have not been re-admitted since. Their table entries
    /// stay resident (`Manager::free` deliberately does not clear the table), and nothing
    /// else reads them until re-admission — so they are exactly where a preempt-time field
    /// write can rot unnoticed. `assert_table_matches` walks them alongside `running`.
    preempted: FxHashSet<u32>,
}

impl Harness {
    fn new(mode: Mode, num_blocks: usize, radix: bool) -> Self {
        Harness {
            kv: Manager::new(cfg(num_blocks, radix)).unwrap(),
            core: ScheduleCore::new(),
            params: params(),
            mode,
            reqs: FxHashMap::default(),
            num_output_tokens: FxHashMap::default(),
            running: Vec::new(),
            waiting: Vec::new(),
            names: FxHashMap::default(),
            next_id: 0,
            inflight: Vec::new(),
            preempted: FxHashSet::default(),
        }
    }

    fn admit_new(&mut self, prompt: usize, max_tokens: usize, salt: u8) {
        let name = format!("r{}", self.next_id);
        self.next_id += 1;
        let slot = self.kv.intern(&name);
        let packed: Vec<u8> = (0..prompt / 16)
            .flat_map(|i| {
                let mut d: Digest32 = [salt; HASH_LEN];
                d[0] = i as u8;
                d[1] = salt;
                d.to_vec()
            })
            .collect();
        self.kv.push_hashes(slot, &packed, prompt);
        self.kv.stops.set(
            slot,
            StopParams {
                min_tokens: 0,
                max_tokens,
                eos_token_id: None,
                stop_token_ids: Default::default(),
            },
        );
        self.reqs.insert(
            slot,
            SchedReq {
                slot,
                num_tokens: prompt,
                num_tokens_with_spec: prompt,
                num_computed_tokens: 0,
                num_output_placeholders: 0,
                num_prompt_tokens: prompt,
                max_tokens,
                status: STATUS_WAITING,
                num_preemptions: 0,
                is_prefill_chunk: true,
                skip_reading_prefix_cache: false,
            },
        );
        self.num_output_tokens.insert(slot, 0);
        self.names.insert(slot, name);
        self.waiting.push(slot);
    }

    fn drop_request(&mut self, slot: u32) {
        self.kv.free(slot);
        if let Some(n) = self.names.remove(&slot) {
            self.kv.forget(&n);
        }
        self.reqs.remove(&slot);
        self.num_output_tokens.remove(&slot);
        self.running.retain(|&s| s != slot);
        self.waiting.retain(|&s| s != slot);
        self.preempted.remove(&slot);
        // A dead request's in-flight frame is dropped, exactly as `update_from_output`
        // skips `self.requests.get(req_id) is None`.
        self.inflight.retain(|&s| s != slot);
    }

    fn schedule(&mut self) -> Decisions {
        let waiting: Vec<SchedReq> = self.waiting.iter().map(|s| self.reqs[s]).collect();
        let d = match self.mode {
            Mode::Marshalled => {
                let running: Vec<SchedReq> = self.running.iter().map(|s| self.reqs[s]).collect();
                self.core
                    .schedule(&mut self.kv, &running, &waiting, &self.params)
            }
            Mode::Resident => {
                let running = self.running.clone();
                self.core
                    .schedule_resident(&mut self.kv, &running, &waiting, &self.params)
            }
        };
        d.unwrap().clone()
    }

    /// One full engine step in ASYNC order: `schedule(N)` first, then the output of
    /// `schedule(N - 1)` lands.
    fn apply(&mut self, d: &Decisions) {
        let issued = self.apply_schedule(d);
        let prev = std::mem::replace(&mut self.inflight, issued);
        self.apply_output(&prev);
    }

    /// The queue surgery + `_update_after_schedule` half. Returns the slots that will
    /// produce a token this step.
    fn apply_schedule(&mut self, d: &Decisions) -> Vec<u32> {
        let sampled = self.params.num_sampled_tokens_per_step;

        self.waiting = d.waiting_order.clone();
        for &slot in &d.preempted {
            self.running.retain(|&s| s != slot);
            let r = self.reqs.get_mut(&slot).unwrap();
            r.status = STATUS_PREEMPTED;
            r.num_computed_tokens = 0;
            // The placeholders count tokens the step this request was evicted from will
            // never produce for it. Production Python zeroes them at both preempt sites
            // (`rust_sched.py`'s `_preempt_request` hook and the crate-driven preempt loop)
            // and `sched::ScheduleCore::commit` does the same to the table, so a harness
            // that left them standing would model the PRE-fix Python and pass a crate that
            // had quietly stopped zeroing.
            r.num_output_placeholders = 0;
            // ...and with C back at 0 any request with a prompt is a prefill chunk again.
            r.is_prefill_chunk = r.num_computed_tokens < r.num_tokens;
            r.num_preemptions += 1;
            self.waiting.insert(0, slot);
            self.preempted.insert(slot);
        }
        for &(slot, _, num_computed) in &d.scheduled_admitted {
            self.running.push(slot);
            self.preempted.remove(&slot);
            let r = self.reqs.get_mut(&slot).unwrap();
            r.status = STATUS_RUNNING;
            r.num_computed_tokens = num_computed;
        }
        // `_update_after_schedule` + the AsyncScheduler override, over every scheduled req.
        let scheduled: Vec<(u32, usize)> = d
            .scheduled_running
            .iter()
            .copied()
            .chain(d.scheduled_admitted.iter().map(|&(s, n, _)| (s, n)))
            .collect();
        for &(slot, num_new) in &scheduled {
            let r = self.reqs.get_mut(&slot).unwrap();
            r.num_computed_tokens += num_new;
            r.is_prefill_chunk = r.num_computed_tokens < r.num_tokens + r.num_output_placeholders;
            if !r.is_prefill_chunk {
                r.num_output_placeholders += sampled;
            }
        }

        scheduled
            .iter()
            .map(|&(s, _)| s)
            .filter(|s| !self.reqs[s].is_prefill_chunk)
            .collect()
    }

    /// `update_from_output` for the tokens sampled one step ago.
    fn apply_output(&mut self, decoding: &[u32]) {
        // A slot freed since it was issued is already gone from `inflight`.
        if decoding.is_empty() {
            return;
        }
        let cu: Vec<u32> = (0..=decoding.len() as u32).collect();
        let toks: Vec<i64> = decoding.iter().map(|&s| (s as i64) + 1000).collect();
        let n_out: Vec<u32> = decoding
            .iter()
            .map(|s| self.num_output_tokens[s] as u32)
            .collect();
        let n_tok: Vec<u32> = decoding.iter().map(|s| self.reqs[s].num_tokens as u32).collect();
        let verdicts = self
            .kv
            .update_step(decoding, &cu, &toks, &n_out, &n_tok, self.params.max_model_len)
            .unwrap()
            .to_vec();

        let mut finished = Vec::new();
        for (i, &slot) in decoding.iter().enumerate() {
            let v = verdicts[i];
            assert_ne!(v.status, u8::MAX, "slot {slot} lost its interned stop params");
            let acc = v.num_accepted as usize;
            let r = self.reqs.get_mut(&slot).unwrap();
            r.num_tokens += acc;
            r.num_tokens_with_spec += acc;
            // SATURATING, because a preempt lands between the schedule that promised this
            // token and the update that delivers it: the preempt arm above zeroed P while
            // the token was still in flight, and the drain then has nothing left to take.
            // `Manager::update_step` clamps the same way (`saturating_sub`), as do
            // `rust_sched.py`'s `pack_req` and `_update_request_with_output`.
            r.num_output_placeholders = r.num_output_placeholders.saturating_sub(acc);
            *self.num_output_tokens.get_mut(&slot).unwrap() += acc;
            if v.status != 0 {
                finished.push(slot);
            }
        }
        for slot in finished {
            self.drop_request(slot);
        }
    }

    /// In `Resident` mode the table IS the running set — assert it never drifts from what
    /// Python believes, so a failure points at the exact field rather than a diff of
    /// scheduling decisions ten steps later.
    ///
    /// PREEMPTED SLOTS ARE CHECKED TOO. Their entries stay resident and a re-admission
    /// overwrites them wholesale from the marshalled waiting queue, so a preempt-time field
    /// the crate stopped writing would never show up in a scheduling decision — but it
    /// would rot in the table for as long as the request sits in the queue, and anything
    /// reading the table directly (`table_get`, `table_fingerprint`, an update landing on
    /// the in-flight token) sees the stale value. Walking them is what makes the placeholder
    /// zeroing in `ScheduleCore::commit`'s preempt closure load-bearing for this suite.
    fn assert_table_matches(&self) {
        if self.mode != Mode::Resident {
            return;
        }
        for &slot in &self.running {
            let want = self.reqs[&slot];
            let got = self.kv.table_get(slot).expect("running slot must be resident");
            assert_eq!(got, want, "resident table drifted on slot {slot}");
        }
        for &slot in &self.preempted {
            let want = self.reqs[&slot];
            let got = self
                .kv
                .table_get(slot)
                .expect("a preempted slot keeps its entry (`free` does not clear the table)");
            assert_eq!(got, want, "resident table drifted on preempted slot {slot}");
        }
    }

    fn step(&mut self, rng: &mut Rng) -> Decisions {
        // 0-2 arrivals per step, prompt 16..=256 tokens, 1..=6 output tokens.
        for _ in 0..rng.below(3) {
            let prompt = ((rng.below(16) + 1) * 16) as usize;
            let max_tokens = (rng.below(6) + 1) as usize;
            self.admit_new(prompt, max_tokens, (rng.below(4) + 1) as u8);
        }
        let d = self.schedule();
        self.apply(&d);
        self.assert_table_matches();
        // A random abort: the other way a request leaves, and the one that exercises
        // free/forget against the table.
        if !self.running.is_empty() && rng.below(10) == 0 {
            let i = rng.below(self.running.len() as u64) as usize;
            let slot = self.running[i];
            self.drop_request(slot);
        }
        if rng.below(37) == 0 {
            self.kv.reset_prefix_cache();
        }
        d
    }
}

// ---------------------------------------------------------------------------
// 1. resident vs marshalled parity
// ---------------------------------------------------------------------------

fn parity(seed: u64, steps: usize, num_blocks: usize, radix: bool) {
    let mut a = Harness::new(Mode::Marshalled, num_blocks, radix);
    let mut b = Harness::new(Mode::Resident, num_blocks, radix);
    let (mut ra, mut rb) = (Rng(seed), Rng(seed));
    for i in 0..steps {
        let da = a.step(&mut ra);
        let db = b.step(&mut rb);
        assert_eq!(da, db, "seed {seed}: decisions diverged at step {i}");
        assert_eq!(
            a.kv.state_fingerprint(),
            b.kv.state_fingerprint(),
            "seed {seed}: KV state diverged at step {i}"
        );
    }
    assert!(b.next_id > 20, "the stream must actually admit requests");
}

#[test]
fn resident_table_matches_marshalled_snapshots() {
    for seed in [1u64, 7, 42, 1234, 99991] {
        parity(seed, 120, 512, false);
    }
}

#[test]
fn resident_parity_under_preemption_pressure() {
    // A pool small enough that the preempt loop fires constantly — the branch where
    // running-queue order and the table's PREEMPTED stamping have to agree.
    for seed in [3u64, 17, 555] {
        parity(seed, 120, 40, false);
    }
}

#[test]
fn resident_parity_with_the_radix_index() {
    parity(2024, 120, 512, true);
}

// ---------------------------------------------------------------------------
// 2. the placeholder / num_computed transition matrix
// ---------------------------------------------------------------------------

/// One chunked prefill, then decode. Checks the exact points vLLM's two
/// `_update_after_schedule` layers touch: `num_computed_tokens` advances by the scheduled
/// count, `is_prefill_chunk` is recomputed from the NEW value, and the placeholder bump
/// happens only once the request has stopped being a prefill chunk.
#[test]
fn commit_transitions_across_the_chunked_prefill_boundary() {
    let mut h = Harness::new(Mode::Resident, 512, false);
    h.params.max_num_scheduled_tokens = 64;
    h.admit_new(160, 4, 1);
    let slot = 0u32;

    // Chunk 1: 64 of 160 computed, still a prefill chunk -> no placeholder.
    let d = h.schedule();
    assert_eq!(d.scheduled_admitted, vec![(slot, 64, 0)]);
    h.apply(&d);
    let e = h.kv.table_get(slot).unwrap();
    assert_eq!((e.num_computed_tokens, e.num_output_placeholders), (64, 0));
    assert!(e.is_prefill_chunk);

    // Chunk 2: 128 of 160.
    let d = h.schedule();
    assert_eq!(d.scheduled_running, vec![(slot, 64)]);
    h.apply(&d);
    let e = h.kv.table_get(slot).unwrap();
    assert_eq!((e.num_computed_tokens, e.num_output_placeholders), (128, 0));
    assert!(e.is_prefill_chunk);

    // Chunk 3 finishes the prefill (32 tokens): no longer a chunk, so ONE placeholder is
    // taken out for the token this step will sample.
    let d = h.schedule();
    assert_eq!(d.scheduled_running, vec![(slot, 32)]);
    h.apply(&d);
    let e = h.kv.table_get(slot).unwrap();
    // Async: the sampled token is still in flight, so it exists only as a PLACEHOLDER.
    assert_eq!(
        (e.num_computed_tokens, e.num_tokens, e.num_output_placeholders),
        (160, 160, 1)
    );
    assert!(!e.is_prefill_chunk);

    // Step 4 schedules the (still placeholder-only) decode token, and the step-3 output
    // lands during it: +1 real token, one placeholder drained, one fresh one taken.
    let d = h.schedule();
    assert_eq!(d.scheduled_running, vec![(slot, 1)]);
    h.apply(&d);
    let e = h.kv.table_get(slot).unwrap();
    assert_eq!(
        (e.num_computed_tokens, e.num_tokens, e.num_output_placeholders),
        (161, 161, 1)
    );
}

/// The async-scheduling shape: the placeholder is bumped at schedule time and only comes
/// back down when `update_step` reports how many tokens really landed.
#[test]
fn placeholder_is_bumped_at_schedule_and_drained_at_update() {
    let mut kv = Manager::new(cfg(512, false)).unwrap();
    let mut core = ScheduleCore::new();
    let p = params();
    let slot = kv.intern("a");
    kv.push_hashes(slot, &[7u8; HASH_LEN], 16);
    kv.stops.set(
        slot,
        StopParams {
            min_tokens: 0,
            max_tokens: 8,
            eos_token_id: None,
            stop_token_ids: Default::default(),
        },
    );
    let req = SchedReq {
        slot,
        num_tokens: 16,
        num_tokens_with_spec: 16,
        num_computed_tokens: 0,
        num_output_placeholders: 0,
        num_prompt_tokens: 16,
        max_tokens: 8,
        status: STATUS_WAITING,
        num_preemptions: 0,
        is_prefill_chunk: true,
        skip_reading_prefix_cache: false,
    };
    core.schedule(&mut kv, &[], &[req], &p).unwrap();
    let e = kv.table_get(slot).unwrap();
    assert_eq!(e.num_computed_tokens, 16);
    assert_eq!(e.num_output_placeholders, 1, "one sampled token is in flight");
    assert!(!e.is_prefill_chunk);

    // The token lands.
    kv.update_step(&[slot], &[0, 1], &[5], &[0], &[16], p.max_model_len)
        .unwrap();
    let e = kv.table_get(slot).unwrap();
    assert_eq!((e.num_tokens, e.num_output_placeholders), (17, 0));

    // ... and a step where nothing was scheduled leaves the entry alone.
    let before = kv.table_get(slot).unwrap();
    core.schedule_resident(&mut kv, &[], &[], &p).unwrap();
    assert_eq!(kv.table_get(slot).unwrap(), before);
}

/// `max_tokens` tail: the verdict trims and reports the stop, and the placeholder is
/// drained by the number of tokens that actually landed, not the number offered.
#[test]
fn max_tokens_tail_drains_only_the_accepted_tokens() {
    let mut kv = Manager::new(cfg(512, false)).unwrap();
    let slot = kv.intern("a");
    kv.stops.set(
        slot,
        StopParams {
            min_tokens: 0,
            max_tokens: 2,
            eos_token_id: None,
            stop_token_ids: Default::default(),
        },
    );
    let mut e = SchedReq {
        slot,
        num_tokens: 20,
        num_tokens_with_spec: 20,
        num_computed_tokens: 20,
        num_output_placeholders: 3,
        num_prompt_tokens: 16,
        max_tokens: 2,
        status: STATUS_RUNNING,
        num_preemptions: 0,
        is_prefill_chunk: false,
        skip_reading_prefix_cache: false,
    };
    e.num_tokens = 17; // 16 prompt + 1 output already emitted
    e.num_tokens_with_spec = 17;
    kv.table_set(slot, e);

    // Three offered, but max_tokens=2 stops after the first -> 1 accepted.
    let v = kv
        .update_step(&[slot], &[0, 3], &[5, 6, 7], &[1], &[17], 2048)
        .unwrap()[0];
    assert_eq!(v.num_accepted, 1);
    assert_eq!(v.status, vtl_sched::update::FINISHED_LENGTH_CAPPED);
    let e = kv.table_get(slot).unwrap();
    assert_eq!((e.num_tokens, e.num_output_placeholders), (18, 2));
}

/// `_preempt_request`'s field writes, applied to the table.
///
/// Not observable through the parity harness's DECISIONS — a preempted request re-enters
/// through the MARSHALLED waiting queue, which overwrites its entry wholesale — but the
/// table must never report a stale RUNNING request to anyone who reads it directly, and the
/// placeholder count is the field with no other witness at all: nothing in the admission
/// arithmetic reads it, so a preempt closure that stopped zeroing it would cost nothing
/// until the step AFTER re-admission. This test is where that write is pinned down with a
/// token genuinely in flight (P == 1); `assert_table_matches`'s preempted-slot walk covers
/// the rest of the entry on every randomized step.
#[test]
fn preempted_slots_are_stamped_in_the_table() {
    // 12 usable blocks; each 64-token request holds 4 attention + 1 mamba block, and a
    // decode step needs one more of each. Only one of the two can grow.
    let mut h = Harness::new(Mode::Resident, 13, false);
    h.admit_new(64, 16, 1);
    h.admit_new(64, 16, 2);
    let d = h.schedule();
    assert_eq!(d.scheduled_admitted.len(), 2);
    h.apply(&d);

    let d = h.schedule();
    assert_eq!(d.preempted.len(), 1, "the tail request must be preempted");
    let victim = d.preempted[0];
    let e = h.kv.table_get(victim).unwrap();
    assert_eq!(e.status, STATUS_PREEMPTED);
    assert_eq!(e.num_computed_tokens, 0);
    assert_eq!(e.num_preemptions, 1);
    // The placeholder the victim carried for the token step 1 put in flight goes with it.
    // Left standing it would inflate the running loop's `num_tokens_with_spec +
    // num_output_placeholders - num_computed_tokens` for every step AFTER re-admission —
    // the admission arithmetic itself never reads P, which is why nothing else catches it.
    assert_eq!(e.num_output_placeholders, 0, "the preempt closure must zero P");
    assert!(e.is_prefill_chunk, "C back at 0 makes it a prefill chunk again");
    // ...and Python's half of the same three writes, compared against the table entry.
    // This is the one place the comparison happens with a placeholder actually in flight:
    // the randomized parity harness only ever preempts requests that are still prefilling
    // (P == 0), so its `assert_table_matches` walk cannot reach this state on its own.
    let victim_before = h.reqs[&victim];
    assert_eq!(victim_before.num_output_placeholders, 1, "a token was in flight for it");
    h.apply_schedule(&d);
    h.assert_table_matches();
    // ... and the survivor is still RUNNING with its tokens advanced.
    let (kept, n) = d.scheduled_running[0];
    let e = h.kv.table_get(kept).unwrap();
    assert_eq!(e.status, STATUS_RUNNING);
    assert_eq!(e.num_computed_tokens, 64 + n);
}

/// A slot Rust has no stop params for must leave the table untouched: Python ran stock
/// `check_stop` for it and owns the resulting counters.
#[test]
fn unregistered_slot_leaves_the_table_alone() {
    let mut kv = Manager::new(cfg(512, false)).unwrap();
    let slot = kv.intern("a");
    let e = SchedReq {
        slot,
        num_tokens: 20,
        num_tokens_with_spec: 20,
        num_computed_tokens: 20,
        num_output_placeholders: 1,
        num_prompt_tokens: 16,
        max_tokens: 8,
        status: STATUS_RUNNING,
        num_preemptions: 0,
        is_prefill_chunk: false,
        skip_reading_prefix_cache: false,
    };
    kv.table_set(slot, e);
    let v = kv.update_step(&[slot], &[0, 1], &[5], &[4], &[20], 2048).unwrap()[0];
    assert_eq!(v.status, u8::MAX);
    assert_eq!(kv.table_get(slot).unwrap(), e);
}

// ---------------------------------------------------------------------------
// 3. journal rollback
// ---------------------------------------------------------------------------

fn rollback_property(seed: u64, num_blocks: usize, radix: bool) {
    let mut h = Harness::new(Mode::Resident, num_blocks, radix);
    let mut shadow = Harness::new(Mode::Resident, num_blocks, radix);
    let (mut r1, mut r2) = (Rng(seed), Rng(seed));

    for i in 0..80 {
        // Speculate on `h` and throw it away; `shadow` never speculates. Both must stay
        // bit-identical forever after.
        let before = h.kv.state_fingerprint();
        let running = h.running.clone();
        h.kv.arm_journal();
        let res = h
            .core
            .schedule_resident(&mut h.kv, &running, &[], &h.params)
            .map(|_| ());
        h.kv.rollback_journal();
        assert_eq!(
            h.kv.state_fingerprint(),
            before,
            "seed {seed}: rollback was not exact at step {i} (spec result: {res:?})"
        );

        let d1 = h.step(&mut r1);
        let d2 = shadow.step(&mut r2);
        assert_eq!(d1, d2, "seed {seed}: a rolled-back speculation changed step {i}");
        assert_eq!(h.kv.state_fingerprint(), shadow.kv.state_fingerprint());
    }
}

#[test]
fn speculative_rollback_is_bit_identical() {
    for seed in [11u64, 202, 3003, 65537] {
        rollback_property(seed, 512, false);
    }
}

#[test]
fn speculative_rollback_is_bit_identical_under_eviction_and_preemption() {
    for seed in [5u64, 77] {
        rollback_property(seed, 40, false);
    }
}

#[test]
fn speculative_rollback_is_bit_identical_with_the_radix_index() {
    // The radix index journals a whole bucket per touched key; this is the test that
    // says the `len` bookkeeping and the shared-prefix aliasing survive an undo.
    rollback_property(4242, 512, true);
    rollback_property(909, 40, true);
}

/// The documented refusal: an armed run with a non-empty waiting queue aborts instead of
/// touching prefix-cache paths the journal does not cover.
#[test]
fn speculation_refuses_a_non_empty_waiting_queue() {
    let mut h = Harness::new(Mode::Resident, 512, false);
    h.admit_new(64, 4, 1);
    let waiting: Vec<SchedReq> = h.waiting.iter().map(|s| h.reqs[s]).collect();
    let before = h.kv.state_fingerprint();
    h.kv.arm_journal();
    let res = h
        .core
        .schedule_resident(&mut h.kv, &[], &waiting, &h.params)
        .map(|_| ());
    assert!(res.is_err(), "an armed run with waiting requests must refuse");
    h.kv.rollback_journal();
    assert_eq!(h.kv.state_fingerprint(), before);
}

// ---------------------------------------------------------------------------
// 4. the speculation hit path
// ---------------------------------------------------------------------------

/// Committing a speculative run must land the engine in exactly the state — and hand back
/// exactly the decisions — a direct call would have produced.
#[test]
fn committed_speculation_equals_a_direct_call() {
    for seed in [8u64, 313, 77777] {
        let mut direct = Harness::new(Mode::Resident, 256, false);
        let mut spec = Harness::new(Mode::Resident, 256, false);
        let (mut r1, mut r2) = (Rng(seed), Rng(seed));
        // Warm both up identically so there is real state to speculate against.
        for _ in 0..25 {
            direct.step(&mut r1);
            spec.step(&mut r2);
        }
        assert_eq!(direct.kv.state_fingerprint(), spec.kv.state_fingerprint());

        for _ in 0..25 {
            let running = spec.running.clone();

            // Speculative: arm, run, then COMMIT (drop the journal) — what a `kick`
            // followed by a matching `take_speculative` does.
            spec.kv.arm_journal();
            let d_spec = spec
                .core
                .schedule_resident(&mut spec.kv, &running, &[], &spec.params)
                .unwrap()
                .clone();
            spec.kv.commit_journal();

            // Direct: no journal at all.
            let d_direct = direct
                .core
                .schedule_resident(&mut direct.kv, &running, &[], &direct.params)
                .unwrap()
                .clone();

            assert_eq!(d_spec, d_direct);
            assert_eq!(direct.kv.state_fingerprint(), spec.kv.state_fingerprint());
            direct.apply(&d_direct);
            spec.apply(&d_spec);
            assert_eq!(direct.kv.state_fingerprint(), spec.kv.state_fingerprint());
        }
    }
}

/// End to end through the real worker thread: kick, wait for the pending result, consume
/// it, and compare against the direct answer.
#[test]
fn spec_thread_round_trip() {
    use std::sync::{Arc, Mutex};
    use vtl_sched::spec::SpecDriver;

    let mut warm = Harness::new(Mode::Resident, 256, false);
    let mut rng = Rng(31337);
    for _ in 0..20 {
        warm.step(&mut rng);
    }
    let running = warm.running.clone();
    let p = warm.params;

    // The oracle: what a direct call on an identical state would decide.
    let mut oracle = Harness::new(Mode::Resident, 256, false);
    let mut rng2 = Rng(31337);
    for _ in 0..20 {
        oracle.step(&mut rng2);
    }
    let want = oracle
        .core
        .schedule_resident(&mut oracle.kv, &running, &[], &p)
        .unwrap()
        .clone();

    let shared = Arc::new(Mutex::new(Shared::new(warm.kv)));
    let driver = SpecDriver::spawn(shared.clone()).unwrap();
    driver.kick(9, running.clone(), p);

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        if shared.lock().unwrap().spec.pending.is_some() {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "spec worker never produced a result");
        std::thread::yield_now();
    }

    let mut sh = shared.lock().unwrap();
    // A generation mismatch is a miss, and must roll back cleanly.
    assert!(sh.take_speculative(10, &running, &p).is_none());
    assert!(sh.spec.pending.is_none());
    assert_eq!(sh.manager.state_fingerprint(), oracle_pre_fingerprint(&running));

    // Re-kick and consume properly.
    drop(sh);
    driver.kick(11, running.clone(), p);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        if shared.lock().unwrap().spec.pending.is_some() {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "spec worker never produced a result");
        std::thread::yield_now();
    }
    let mut sh = shared.lock().unwrap();
    let got = sh.take_speculative(11, &running, &p).expect("generation matched");
    assert_eq!(got, want);
    assert_eq!(sh.manager.state_fingerprint(), oracle.kv.state_fingerprint());
    assert_eq!((sh.spec.hits, sh.spec.misses), (1, 1));
    assert!(sh.spec.rollbacks >= 1);
}

/// The pre-speculation fingerprint of a freshly replayed harness — used to prove the
/// generation-mismatch path put everything back.
fn oracle_pre_fingerprint(_running: &[u32]) -> u64 {
    let mut h = Harness::new(Mode::Resident, 256, false);
    let mut rng = Rng(31337);
    for _ in 0..20 {
        h.step(&mut rng);
    }
    h.kv.state_fingerprint()
}

// ---------------------------------------------------------------------------
// 5. Send bounds (also asserted inside `spec.rs`, kept here for the public types)
// ---------------------------------------------------------------------------

#[test]
fn public_state_types_are_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Manager>();
    assert_send::<ScheduleCore>();
    assert_send::<Shared>();
    assert_send::<Decisions>();
    assert_send::<Params>();
}
