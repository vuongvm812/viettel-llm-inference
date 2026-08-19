# Round-2: NSTEP regression investigation (2026-08-18)

Two symptoms reported on-box, on a config where NSTEP=1 "worked yesterday":

1. **`VTL_NSTEP=0`**: the engine serves normally, but logs once per boot:

   ```
   ERROR [vllm.vtl.rust_sched] rust_sched: resident-table path failed in kick; permanently marshalled
     File ".../vtl/patches/rust_sched.py", line 2693, in maybe_kick
       tbl.armed = bool(core.kick(kv._rust, tbl.gen, slots))
   RuntimeError: Already borrowed
   ```

2. **`VTL_NSTEP=1`**: the server wedges mid-decode — requests pinned RUNNING, 0 tok/s,
   no exception (the hang `stall_dump.py` was written for, reproduced under both
   `VTL_NSTEP_MODE=graph` and `eager`, gone with `VTL_NSTEP=0`).

These are **two different bugs**. Both are explained below; neither is caused by the
config edits (NVRTC off, log-stats on, KV offload off) — those only changed timing.

---

## Finding 1 — `Already borrowed` in `kick`: a cross-thread PyO3 pycell borrow race

**Definitive root cause.** Introduced by `da9ea46` (2026-08-10, "add-time prereg");
latent since, fail-open, so easy to miss.

### The two threads

**Input thread** (stock ZMQ input thread, or `shm_ipc.py`'s `_run_input_thread` at
`shm_ipc.py:845`) — with `VTL_RUST_PREREG=1` (compose default), every request arrival
runs the `preprocess_add_request` hook (`rust_sched.py:2933`), which calls:

```rust
// vtl-sched/src/python.rs:370
fn set_request_meta(&self, py: Python<'_>, ...) -> u32 {
    let shared = self.shared.clone();
    py.allow_threads(move || {
        let mut sh = lock_shared(&shared);   // <-- can PARK here
        ...
    })
}
```

A `&self` pymethod holds a **shared borrow of the `KvManager` pycell for the whole
call** — including across `py.allow_threads`. Releasing the GIL does *not* release the
pycell borrow. The hook's docstring ("the input thread never parks on the crate mutex
while holding it") solved the GIL half of the hazard but not the borrow half.

The park is not rare or short: the `vtl-sched-spec` worker holds the big
`Arc<Mutex<Shared>>` for an **entire speculative `schedule_resident` run**
(`spec.rs::worker`, lock taken around the whole `run_speculative`). Every `kick` arms
the worker, so right after each step there is a window in which an arriving request's
`set_request_meta` sits parked on that mutex — GIL released, pycell borrow held.

**Engine thread** — at the end of `update_from_output`, `maybe_kick`
(`rust_sched.py:2693`) calls:

```rust
// vtl-sched/src/python.rs:1437
fn kick(&self, kv: &mut KvManager, generation: u64, running_slots: Vec<u32>) -> PyResult<bool>
```

`&mut KvManager` makes PyO3 attempt an **exclusive** borrow of the same pycell. With
the input thread's shared borrow still live → `BorrowMutError` → PyO3 raises
`RuntimeError("Already borrowed")`.

This is the *only* possible collision direction: every other engine-side crossing that
borrows `KvManager` runs while the engine holds the GIL, and the input thread can only
touch the pycell while *it* holds the GIL — so the engine's own borrows can't be
observed by anyone. `kick` is the one `&mut KvManager` entry point (the crate's own
comment at `python.rs:1016` warns about exactly this, but it covers `&mut self`
methods, and `kick` takes the exclusive borrow through an *argument*). The `&mut` is
needed only to lazily store `kv.driver = Some(SpecDriver)` on first kick.

### Consequence

`tbl.fail("kick", exc)` (`rust_sched.py:1071`) is fail-open: one ERROR log, then
`tbl.off = True` **for the rest of the process** — resident-table scheduling and
speculation are dead for the boot, silently degrading to the marshalled path. The
server keeps serving (as seen in the log: generation continues at 130–160 tok/s), so
this is a *performance* regression, not a crash — but it discards two of round-2's
main scheduler wins whenever the race fires.

### Why it "appeared today"

It is probabilistic per request arrival (arrival must land inside a spec-worker mutex
hold, and the next `maybe_kick` must fire before the worker releases). It logs at most
once per boot. Nothing in the last day's commits touched this path — today's config
edits (log-stats on → extra per-step work between update and kick; KV offload off;
different trace/arrival pattern from the new Weka-trace warm) merely shifted the
timing enough to hit it.

### Fix (recommended)

Make `kick` take `kv: &KvManager` and move `driver` behind interior mutability, e.g.
`driver: Mutex<Option<SpecDriver>>` (or spawn the driver eagerly at the
`39f4f88` scheduler-init rendezvous, where the lazy spawn is no longer buying
anything). Shared borrows never conflict, so the race disappears without touching the
prereg hook. A defensive complement on the Python side: in `maybe_kick`, treat
`RuntimeError("Already borrowed")` as *skip this step's kick* rather than
`tbl.fail(...)` — the condition is transient, and permanently marshalling on it is a
disproportionate response.

---

## Finding 2 — the NSTEP=1 hang: batch-5 bursts are one day old

**"Yesterday NSTEP=1 worked" is explained by what NSTEP=1 was actually doing
yesterday: almost nothing at steady state.**

`e15bab5` (2026-08-17 19:00, "capture batch 5") changed two things at once:

* compose `cudagraph_capture_sizes`: `[1,2,4,8]` → `[1,2,4,5,8]`
* `nstep_decode.BURST_SIZES` seed: `(1,2,4,8)` → `(1,2,4,5,8)`

With `--max-num-seqs=5`, **batch 5 is the steady-state decode width**. Before this
commit there was no FULL decode graph at 5, so batch 5 had no burst rung: at steady
state `burst_factor()` returned 1 and NSTEP=1 degenerated to plain one-token steps.
The commit message says so explicitly ("batch 5 -- the steady state -- had NO full
graph and NO nstep burst rung"), and the *older* comment it replaced had deliberately
deferred adding sizes as "unvalidated-path risk [that] waits for a run that can afford
to fail".

So since yesterday evening, for the first time, the steady-state batch rides the whole
newly-unlocked ladder: batch-5 FULL decode dispatch, the burst rung at 5, the
continuation-graph multi-launch (`VTL_RUST_RUNNER_STEPS=8` residency), unroll/fold and
in-graph sampling at 5. The hang is a liveness bug somewhere in that
first-time-exercised path. That it reproduces under `eager` too (per
`stall_dump.py`'s docstring) localizes it to the **mode-independent half** — the
scheduler burst commit / placeholder reconcile / runner interlock — not the graph
capture itself.

Ruled out for the hang:

* NVRTC changes (`b030e95`, `e847c94`) — reproduced with `VTL_NVRTC=0`.
* Finding 1 — it is fail-open and the engine demonstrably keeps stepping after it.
  (Worth confirming from the NSTEP=1 logs that the hang occurs without the
  "permanently marshalled" line preceding it.)

---

## Finding 3 — config foot-guns in the tested configuration

* `NSTEP`, `NSTEP_MODE`, `STALL_DUMP` (unprefixed) are read by **nothing** in this
  tree. The live knobs are `VTL_NSTEP`, `VTL_NSTEP_MODE`, and for the watchdog
  `VTL_ENABLE_STALL_DUMP` / `VTL_STALL_DUMP_SECS`. If the A/B was driven by the
  unprefixed names, it was actually toggling nothing — re-verify which runs used which
  values via the boot log lines (`rust_sched: ...` / `vtl: nstep ...` /
  `vtl: stall watchdog armed`).
* `STALL_DUMP=0` therefore did **not** disable the flight recorder (it defaults ON) —
  which is good: it is the tool for Finding 2.
* Compose `environment:` in map form needs `KEY: "value"`; `KEY=value` lines belong to
  the list form. Mixing the two in one block is a YAML error.

---

## Recommended next steps

1. **Capture the hang.** Re-run with `VTL_NSTEP=1` and stall_dump armed (default; set
   `VTL_STALL_DUMP_SECS=10` to shorten the wait) and collect stderr: after ~10–20 s of
   wedge it prints per-request counters, the `BURST` handshake state, and
   all-thread Python stacks — that names the wedged line directly.
2. **Confirm the trigger.** With `VTL_NSTEP=1`, revert only
   `cudagraph_capture_sizes` to `[1,2,4,8]`. Expectation: the hang disappears
   (steady-state batch 5 loses its rung again). Then walk the ladder to isolate the
   rung: `VTL_RUST_RUNNER_STEPS=1` → `VTL_NSTEP_UNROLL=0` → `VTL_NSTEP_FOLD_T1=0` →
   `VTL_SAMPLE_IN_GRAPH=0` → `VTL_NSTEP_MODE=eager`.
3. **Fix Finding 1** (the `kick` borrow) regardless — it silently costs the
   resident-table and speculation wins on any boot where the race fires.

---

## Addendum (2026-08-18, follow-up investigation): confirmed mechanisms and fixes

A deeper pass traced the complete burst lifecycle and landed fixes on this branch.
The silent-wedge mechanism is confirmed: in `vtl-sched/src/sched.rs`'s running loop,
a request that fails the scheduling guards is skipped with `continue` **without
leaving `running`** — when every running request fails, `schedule()` returns empty
steps forever (requests RUNNING, 0 tok/s, no exception). The guards fail when the
async counter invariant `num_computed == num_tokens + num_output_placeholders - 1`
breaks. The happy-path burst arithmetic is sound; these were the paths that broke
the pairing, each now fixed:

1. **`commit_burst` exception window** — requests and the resident table were
   mutated *before* `so.vtl_burst_n` was stamped; a raise in the window left an
   unannounced `+delta` the update side never reconciles. Now: gates and the runner
   stash run first, the mutation is an all-or-nothing window
   (`burst_commit_all`) with an exact rollback, and the stamp follows immediately.
2. **`burst_commit`'s conditional placeholder bump** — on a skewed entry it
   advanced `num_computed_tokens` without `num_output_placeholders` every step,
   compounding. Now unconditional (Rust twin: `sched::burst_advance`), and both
   commit arms refuse any request outside the steady-decode shape
   (`burst_invariant_broken`) with a one-shot alarm + table resync.
3. **Preemption left `num_output_placeholders` set** while zeroing
   `num_computed_tokens` — the stale count re-marshalled through `pack_req` and
   wrapped the crate's usize guard on re-admission (request skipped forever). All
   preempt paths (Python loop, `_preempt_request` hook, `sched.rs` closure) now
   zero it.
4. **usize wraps** in the running loop (`C + 2 - P`, `T_spec + P - C`) — rewritten
   in addition/checked form; a broken entry is counted
   (`KvManager::inconsistent_skips`) and surfaced via a one-shot error + resync
   probe on the empty-step branch instead of wedging silently.
5. **Burst could land `C` exactly on `max_model_len`**, making the request
   unschedulable one step before its length cap — the gate now caps at
   `max_model_len - 1`.
6. **Runner stash lifecycle** — `update_from_output`'s unconditional
   `state.step = None` raced the worker's `sample(k)` and destroyed
   `schedule(k)`'s stash (the runner rarely engaged); non-burst `sample_tokens`
   exits leaked the stash for the boot. Stashes are now seq-owned
   (`take_step_for` / `clear_step_upto`, seq carried via `_Burst.pending_seq`).
7. **Multi-launch counts D2H race** — `nsampled_k`/`nrejected_k` were handed to
   `SamplerOutput` as views of persistent buffers while the async copy was in
   flight; a corrupted count produced `short < 0`, which the reconcile skips. Now
   cloned, matching the existing `accum` clone rationale.
8. **Eager arm replayed a stale `b.desc`** with no shape guard — now validated
   (FULL, this padded batch, decode shape) with a 1-token shortfall fallback.
9. **`ring_reuse` omitted AsyncScheduler's placeholder bump** (latent under the
   `uni` executor, a guaranteed wedge under `mp`) — bump added.
10. **The kick borrow race (Finding 1)** — fixed at the source: no pymethod or
    argument takes `KvManager` exclusively any more (`driver` behind its own
    mutex, scratch `Vec` in a `RefCell`), and `maybe_kick` treats a PyO3 borrow
    refusal as skip-this-step instead of permanently marshalling.

`stall_dump` now prints real `_Burst` fields plus the `rust_runner.STATE` line
(the old field list named attributes that never existed), with a self-check that
cross-checks the printed names against the live `__slots__`.

---

## Second incident (2026-08-18 evening): first-prefill hang / NotImplementedError crash under the shipped compose

The serve line gained `--kv-offloading-size=64 --kv-offloading-backend=native`
(judge-specified 2026-08-17, commit 3823158). Those two flags configure a vLLM V1 **KV
connector** — the boot log shows `Initializing KVConnectorBase_V1` and an
`OffloadingSpec` — and a connector cannot coexist with the Rust KV authority: they would
share no block pool, and the connector's allocation and KV-transfer paths are not ported.

The boot guard that was supposed to catch this missed it twice over. It probed only
`cfg.kv_transfer_config`, which the offloading connector does not use (it arrives through
the engine args), and it is tri-state: the `None` it returns when no ambient `VllmConfig`
is in scope was read by `if kv_transfer_configured():` as "no connector". Authority mode
installed anyway. `schedule_supported()` *did* detect it later, off
`scheduler.connector` — that is the reliable signal, and it only exists after the
Scheduler is built, i.e. after the KVCacheManager already decided.

Two outcomes were observed from the same cause: a **hang** in the first prefill wave
(stall dump showed a launch-queue-blocked stack in whatever op happened to be running —
the `nvrtc_block_quant` frame was a bystander, exonerated by reproducing the hang with
`VTL_ENABLE_NVRTC_BLOCK_QUANT=0`), and a **crash** ~30 s in with
`rust_sched.py NotImplementedError: VTL_RUST_SCHED=1 does not support connector /
encoder / full-sequence-admission allocation paths`, raised from stock
`Scheduler.schedule` → `allocate_slots` on the connector path.

Fix on this branch, in `round-2/vtl/patches/rust_sched.py`: (1) `kv_connector_configured()`
probes every field shape a connector can arrive in (`kv_transfer_config`, a kv-offloading
config object, and the raw offloading knobs on either `cfg` or `cfg.cache_config`), each
probe independent so an unknown layout degrades instead of raising; (2) on a detected
connector the authority manager **stands down** — `self.__class__ = base` after
`super().__init__`, which sheds every override and every `_refuse_unported` raiser at once
— and the boot serves the stock scheduler + offloading rather than refusing to start;
(3) a backstop at the first `schedule()` raises immediately if a connector is present and
the KV manager is still Rust-backed (the `None`-detection case), because by then the Rust
pool is live and standing down would split-brain it.

**Open config decision:** connector XOR Rust stack. With the flags as shipped,
`VTL_RUST_SCHED_FULL`, the resident table, the N-step burst and the Rust runner are all
inactive and none of their measurements apply. Either drop the `--kv-offloading-*` flags
or set `VTL_RUST_SCHED=0` and own the choice.

---

## Connector support (stages Q/R/A/S/U1, 2026-08-18)

The "connector XOR Rust stack" decision above is **resolved**: for one configuration the
answer is now *both*. `OffloadingConnector` on the `native` backend — `kv_role=kv_both`,
`CPUOffloadingSpec`, no offloaded `block_size`, `kv_load_failure_policy=recompute` — is
ported, and authority mode stays engaged for it. Every other connector (and every other
configuration of this one) still stands down to the stock scheduler, loudly.

### What now works

* **Q/R/A** — the crate carries external allocation (`Manager::allocate_external`,
  `remaining_blocks`, `block_meta`) and a `KVCacheBlock`-shaped block-record view over
  Rust-owned ids (`RustBlocks` + `_BlockRecordView`), which is what
  `update_state_after_alloc` reads.
* **S** — `schedule()` drives the connector protocol itself: `schedule_phase1` (running
  arm + the cache-hit probe for every waiting candidate) → `get_num_new_matched_tokens`
  per candidate in Python → `schedule_phase2` (external allocation, async loads parked in
  `skipped_waiting` as `WAITING_FOR_REMOTE_KVS`) → `build_connector_meta`, unconditional
  and in stock's place.
* **U1a** — the deferred-free queue. `kv_role=kv_both` + async scheduling makes stock set
  `defer_block_free=True`, so a request finishing with a step still in flight is freed via
  `pop_blocks_for_free` and drained later. Both halves are ported:
  `Manager::free_block_ids` (one reverse over the flat cross-group list, as
  `scheduler.py:2157` does — *not* `TypeManager::free`'s per-group reverse) and a
  `_drain_deferred_frees` / `_request_remaining_blocks` pair installed on the scheduler
  class **only when the connector is live**. The manager's `pop_blocks_for_free` also drops
  the `RustMirror` slot, which the deferred path would otherwise leak for the boot —
  nothing else on that path calls `kv.free()`.
* **U1a guard** — a non-empty `kv_connector_output.invalid_block_ids` raises immediately.
  `OffloadingConnector` never produces one (it does not override
  `get_block_ids_with_load_errors`; the base returns an empty set), and stock's own repair
  path is single-group-only (`(req_block_ids,) = get_block_ids(req_id)`, the
  `TODO (davidb): add support for hybrid memory allocator`), so refusing is inherited stock
  behaviour on this hybrid model, not a regression.

### The compose line this requires

```yaml
- '--kv-transfer-config={"kv_load_failure_policy": "recompute"}'
```

The default is `fail` (`config/kv_transfer.py:69`), which the allow-list refuses. Nothing
else goes in that JSON: `VllmConfig._post_init_kv_transfer_config` synthesizes
`kv_connector=OffloadingConnector` and `kv_role=kv_both` on top of it. Without the line the
boot stands authority mode down and serves stock + offloading.

### What stays excluded

* **tokstore (Port-2)** — its facade freezes `request.block_hashes`, which the connector
  walks on every `build_connector_meta`. Off for the whole boot under a live connector;
  a *declared scope exclusion*, so it never raises, `VTL_RUST_SCHED_REQUIRE` included, and
  it names itself as `tokstore=OFF(connector (block_hashes must stay live))` in RUNGS.
* **r8 / r9** — `step_packable` already refuses a step with a connector or
  `defer_block_free`, so the packed output record and the collapsed residue loop are inert
  here. Documented trade; re-porting them under `defer_block_free` is an optional
  follow-up.
* **lean / arena / SO-ring** — refused on a connector-live boot (the connector rebuilds its
  metadata from every `SchedulerOutput`, and the arena would have to split its persistent
  buffers across the phase seam).
* **every other connector** — LMCache, NIXL, SimpleCPUOffload, `kv_producer`/`kv_consumer`,
  an offloaded `block_size`, an EC transfer config: authority mode stands down.

### On-box validation arms

| Arm | One-liner |
| --- | --- |
| A | Shipped compose as-is: expect `RUNGS authority=on full=on ... connector=OffloadingConnector/native tokstore=OFF(connector ...)` in the boot log, then a clean bench run. |
| B | Same, `-e VTL_RUST_SCHED_REQUIRE=1`: any silent demotion becomes a boot failure — this is the arm that proves the numbers are the port's. |
| C | `-e VTL_NSTEP=1` (default) vs `-e VTL_NSTEP=0`: the burst must still commit with the connector live (`rust_sched: nstep engaged`). |
| D | `--disable-log-stats` removed for one run: read the connector's external hit rate off the prefix-cache stats and confirm the CPU cache is actually serving. |
| E | Connector-off regression: delete the three `--kv-*offloading/transfer*` lines, re-run the same trace, and compare — the port must not have cost the connector-less path anything. |
| F | SIMPLE offload stand-down: `-e VLLM_USE_SIMPLE_KV_OFFLOAD=1` must log the stand-down and serve stock. |
| G | NIXL stand-down: `--kv-transfer-config={"kv_connector":"NixlConnector","kv_role":"kv_both"}` must stand down, not hang or raise. |
| H | Abort/preempt soak: cancel requests mid-prefill and mid-decode under load, and force preemption (small `--max-num-seqs`) — this is the arm that exercises `pop_blocks_for_free` + the deferred drain, so watch `usage`/free-block count for a leak over ~10 min. |

---

## Third incident (2026-08-19): `make warm` / `make up` wedge at the first mixed decode+prefill wave, connector-live boot

First on-box run of the shipped compose with the connector port active (the
`--kv-transfer-config={"kv_load_failure_policy": "recompute"}` line landed with stage S/U1,
so this boot is validation arm A's first actual execution). The server boots, warms up
line 0 of the trace normally (~30 tok/s single-stream decode), then wedges the moment the
5-wide warmup burst (`VTL_WARMUP_CONCURRENCY=5`) transitions from prefill into its first
decode steps. All 5 warmup POSTs time out at 20 s, the RustFrontend auto-aborts their
dropped streams, and the engine never recovers — `make warm` and `make up` both fail on it.

### Reading the stall dump

```
STALL -- 5 running request(s) made no progress for 20s (waiting=0)
STALL req=...1db02510 status=RUNNING num_tokens=2145 computed=2146 placeholders=2 output=0 prefill_chunk=False
STALL req=...39c4e8f9 status=RUNNING num_tokens=2143 computed=2143 placeholders=1 output=0 prefill_chunk=False
STALL req=...83f13c60 status=RUNNING num_tokens=2144 computed=2144 placeholders=1 output=0 prefill_chunk=False
STALL req=...265fdc41 status=RUNNING num_tokens=2145 computed=2145 placeholders=1 output=0 prefill_chunk=False
STALL req=...0f409366 status=RUNNING num_tokens=2602 computed=1759 placeholders=0 output=0 prefill_chunk=True
STALL BURST: armed=True ready=False key=() n=4 mode='graph' pending=0 pending1=False pending_seq=0 rust_steps=8
STALL RUNNER: live=False refused=True why='nstep captured no unroll graphs; there is nothing launchable' inflight=2 pending=0 stash_seq=None
```

**The scheduler-side bookkeeping is CLEAN.** Every request satisfies the async invariant
`C == T + P - 1` exactly (2146 = 2145+2−1, 2143 = 2143+1−1, 2144, 2145; the prefill chunk
carries P=0 by design — chunks claim no placeholder, `rust_sched.py:899-918`). This is
**not** the second-incident wedge (`schedule()` returning empty steps against guard-failing
entries): steps were being scheduled right up to the freeze. The placeholder pattern
(2,1,1,1,0) plus `RUNNER inflight=2` decodes as exactly the normal depth-2 batch queue:

* step *k−1*: request `...1db02510` (first of the wave to finish prefill) takes its first
  decode token, the other four run their final/next prefill chunks (no placeholders);
* step *k*: all four finished requests decode (placeholders 1 each, `...1db02510` now 2),
  `...0f409366` continues chunking (843 tokens left of 2602).

Both steps are executed but neither is applied (`inflight=2` — incremented per
`schedule()`, decremented only in `update_from_output`, `rust_sched.py:5040`/`:4271`),
because the engine thread never returns from `sample_tokens` of the older one.

**Where the engine thread actually is.** The main-thread stack bottoms out at
`nstep_decode.py:1315` — the *non-burst* path (`out = original(...)`; no burst was
committed for this step) — then stock sampling, then:

```
File ".../vtl/patches/step0_eos_ban.py", line 88 in __call__
```

Line 88 is `idx = torch.from_numpy(rows).to(logits.device)`. That `.to()` is a *blocking*
pageable H2D copy: torch's `copy_` with `non_blocking=False` ends in a stream
synchronize, so this line waits for the **entire current stream to drain** — including the
whole forward of the step being sampled and the next step's forward already enqueued
behind it. It sat there for the full 20 s window with the progress signature frozen.
`rows is not None` (prefill rows present in the sampled batch) is consistent with step
*k−1*'s shape — decode row + final prefill chunks.

**Conclusion on mechanism:** the `step0_eos_ban` frame is a *bystander* (same class as the
second incident's `nvrtc_block_quant` frame — it is just the first host call that has to
wait for the stream). The real wedge is **device-side: GPU work enqueued by the first
mixed decode+prefill batch of the boot never completes** (a spinning/garbage-length kernel,
or a cross-stream event that never fires). Everything host-side downstream (sampling,
AsyncOutput D2H, update, abort processing) is queued behind it, which is why even the
frontend's aborts at 01:20:02 change nothing.

### What is new on this boot (suspect list, ranked)

1. **Connector + Rust authority both live for the first time.** This exact compose
   (offloading flags + the `recompute` line + the S/U1 port) had never run on-box; the
   validation arms A–H above were written *to be run*. By the wedge point the connector has
   real work in flight for the first time: line 0's ~2.1k-token prefix has been offloaded,
   and the 5 wave requests (which share that prefix) are the first candidates that can
   probe/hit the CPU cache and the first concurrent offload writes. A block-id or
   event-ordering disagreement between the port's allocation/metadata
   (`allocate_external`, `RustBlocks` view, `build_connector_meta` inputs) and the
   connector worker's GPU⇄CPU transfer streams is the same *class* of failure as the
   second incident's split-brain hang — moved from the first prefill wave to the first
   offload/hit activity.
2. **First mixed decode+prefill batch of the boot.** Line 0 only ever presented pure
   prefill then pure batch-1 decode. Step *k−1* is the first batch mixing decode rows with
   prefill chunks — the shape `jit_monitor` flagged 2 s before the freeze
   (`Triton kernel JIT compilation during inference: fused_moe_kernel`). Decode-band
   kernels (`gdn_decode_step`, `moe_decode_gemv`, fused argmax) and the GDN align-mode
   chunking all see this composition for the first time; a garbage seq-len/state index fed
   to a data-dependent-loop kernel (FA3 with `max_seqlen_k` baked at `max_model_len`, the
   FLA chunk kernels) is indistinguishable from a hang.
3. The JIT compile itself is *not* the wedge (the warning logs after compilation; a
   compile blocks the host, and the host was seen blocked in sampling, not in Triton).

### Secondary findings (not the wedge, but fix/collect anyway)

* **`RUNNER refused: 'nstep captured no unroll graphs'`** — a degrade, not a failure: the
  Rust runner stands down (`rust_runner.py:544`) and Python keeps the step. But it means
  the prologue and/or unroll rung failed capture at boot (`mode='graph'` proves body
  graphs captured; `demote("fold")` on a prologue failure also clears the unroll rung,
  `nstep_decode.py:252-259`). The boot log has the exact lines: look for
  `vtl: nstep <what> capture failed at num_reqs=...` (with traceback) /
  `vtl: nstep <rung> disabled for this boot` and the summary
  `vtl: nstep captured N burst body graph(s) ... prologue=[...] unroll=[...]`. Collect them
  — on the last known-good boot the runner presumably armed, so this may be a second
  regression riding along.
* **`BURST ... rust_steps=8` in the dump is cosmetic**: `demote()` clears the graphs but
  not `rust_steps`; it no longer means multi-launch steps are possible.
* **`rust_sched: token store step fallback (first time) -- step is not numpy-packable`**
  is expected on a connector-live boot (tokstore is a declared connector exclusion;
  `store_take_over` refuses per request and every step then takes the object path) — but
  the logged reason is misleading: the step is "not packable" because the store never took
  the requests over, not because of the step's shape. Consider logging the tokstore
  stand-down reason instead when `TOK.off_why` is set.
* `BURST armed=True ready=False key=()` is correct behaviour here: no pure-decode
  steady-state step had happened yet, so no burst was ever committed — the wedge is
  upstream of everything nstep does at runtime.

### Next steps (in information-value order)

1. **Collect, from the failing box**: (a) the full boot stderr (`docker compose logs`)
   for the nstep capture lines and any connector/rust_sched warnings — especially the
   `RUNGS authority=... connector=...` line proving which mode actually served; (b) the
   *second* stall dump (the watchdog re-fires every 120 s) — an identical stack proves a
   blocked host thread (GPU wedge), a moving one would mean spinning; (c) `nvidia-smi`
   during the wedge — SM util pinned at 100 % = runaway kernel, ~0 % = launch-queue /
   event deadlock.
2. **Bracket the connector port** (the two arms answer different questions):
   * delete all three `--kv-offloading*/--kv-transfer-config` lines (doc arm E): full Rust
     stack, no connector. Warm passes ⇒ the wedge involves the connector.
   * keep the flags, set `VTL_RUST_SCHED=0`: connector on stock scheduler. Warm passes ⇒
     the wedge is in the port, not the connector per se.
3. **`VTL_NSTEP=0` + `VTL_ENABLE_NSTEP_DECODE=0`** — expected *no change* (the burst never
   engaged); if the wedge disappears, that finding invalidates the analysis above and
   points back at capture-time state corruption.
4. **Kernel candidates for the mixed batch**, one at a time:
   `VTL_ENABLE_GDN_DECODE_STEP=0`, `VTL_ENABLE_MOE_DECODE_GEMV=0`,
   `VTL_ENABLE_GREEDY_ARGMAX=0`/`VTL_V2_GREEDY_ARGMAX_KERNEL=0`, `VTL_NVRTC=0`.
5. `VTL_ENABLE_STEP0_EOS_BAN=0` only to confirm the bystander status (the block should
   move to the next sync point, e.g. AsyncOutput's D2H). Independent of this incident,
   `step0_eos_ban`'s per-prefill-step `torch.from_numpy(rows).to(logits.device)` is a
   hidden full-stream sync on the hot path and is worth replacing with a persistent
   device-side index buffer.

### Bisect result (2026-08-19, on-box) and what it does and does not prove

`VTL_ENABLE_RUST_SCHED=0` with everything else unchanged — connector flags included —
**serves the warmup wave cleanly** (mitigation applied locally on the box; the repo compose
is unchanged). Two conclusions and one open split:

* **The kernels are exonerated.** With the patch off, the exact same first-time shapes ran
  (the mixed decode+prefill wave, the freshly JIT'd `fused_moe_kernel` config,
  `gdn_decode_step`, `moe_decode_gemv`, the fused argmax) under the same live connector,
  and nothing hung. The never-completing GPU work is *induced by what the Rust scheduler
  feeds the step*, not by a kernel bug in isolation. Consistent mechanism: a wrong block
  id or count from the scheduler becomes an out-of-bounds/aliased device write (the
  zeroing list, a block-table row, a slot mapping), which corrupts unrelated device state,
  and a later kernel loops on the garbage.
* **The hang needs the rust_sched patch.** But the single arm cannot split two remaining
  hypotheses, because the round-2 connector commits (`ec48f4e`..`cb21b98`) changed shared
  crate/py code as well as connector-only code:
    * **H1 — port × connector interplay** (stage S/U1: phase-split protocol, probe
      consumption, external allocation, deferred frees, `build_connector_meta` inputs);
    * **H2 — a regression in shared rust_sched code** from the same commits, reachable
      without a connector (the last known-good connector-less boot predates `ec48f4e`).
  **The decisive arm:** delete the three `--kv-offloading*/--kv-transfer-config` lines and
  boot with the full Rust stack on. Passes ⇒ H1; hangs ⇒ H2.

### Scheduler flight recorder (landed on this branch)

Because the wedge is enqueued GPU work, the stall dump's thread stacks can only ever catch
a bystander host call blocked on a stream sync — the state that names the culprit is what
the last few `schedule()` calls fed the step. `rust_sched` now carries a small host-side
ring (`FLIGHT`, 48 entries, `VTL_SCHED_FLIGHT=0` to disable, negligible cost) recording
per step: the scheduled `(request, num_new, C, P)` rows, the **span (count, min, max) of
the new-block ids and of the `new_block_ids_to_zero` list**, admission/preemption/parked/
probe-stale counts, plus `pop_blocks_for_free` / deferred-free `drain` events and `bail`
steps. `stall_dump` prints the ring in every wedge dump (`vtl: STALL FLIGHT ...` lines).

On the next wedge repro, read the last two `sched` entries against the request counters:
an id span past the pool size, a zeroing span covering another request's blocks, a `drain`
overlapping a live allocation, or a `num_new` that disagrees with `C/T/P` names the
failing sub-path directly. Collect `nvidia-smi` during the wedge in the same run
(~0 % SM = copy/event deadlock, pinned 100 % = compute kernel on garbage metadata).

Also landed, diagnostics-honesty fixes from this incident's triage: `nstep demote()` now
resets `rust_steps` to 1 (the stall dump advertised 8-launch steps on a boot whose runner
had stood down), and the token-store fallback log names `TOK.off_why` when the store was
off for the whole boot (the bare "step is not numpy-packable" misdirected triage once).

---

## Fourth dump (2026-08-19 02:20): the flight recorder names the trigger step

First wedge captured with the recorder armed. It is the same wedge, and the recorder
answers the question the thread stacks structurally could not.

### What the ring shows

```
t-20.6s .. t-20.4s  sched step=896..904  total=5   (5-row decode, num_new=1 each)
t-20.4s             sched step=905       total=2
t-20.4s             sched step=906       total=3059  adm=1 asked=1 zero=(1, 8, 8)
t-20.2s             sched step=907       total=7803  adm=3 asked=3 zero=(3, 1, 20)
                    <nothing for 20 s>
```

**The trigger is step 907: 2 decode rows + THREE simultaneous unchunked prefills**
(2146 + 2600 + 3055 = 7801 tokens) for **7803 total**, against
`--max-num-batched-tokens=8192`. The eleven steps before it ran in ~0.4 s (**~36 ms per
5-token decode step**); step 907's forward then consumed **the entire 20 s window and did
not return**.

That ratio is the finding. 7803 tokens is ~1560x a decode step's token count, but this is
a 122B-A10B MoE on an H200 — a healthy 7.8k-token prefill belongs in the few-hundred-
milliseconds range, so **step 907 is one to two orders of magnitude slower than it should
be**. This is not (or not only) a deadlock; it is a path whose cost explodes with batch
width, which is exactly why every wedge so far has landed on the first wide mixed batch of
the boot and never on the decode steps before it.

### Scheduler bookkeeping: clean again, and the admission is stock-legal

Every request satisfies the async invariant `C == T + P - 1` (3059 = 3058+2-1,
2146 = 2146+1-1, 2600, 3055, and the decode row 318 = 317+2-1). The admission itself is
what stock would do: 7803 fits the 8192 budget, and the in-flight-prefill reservation that
could have throttled it is — in stock (`scheduler.py:895-901`) — applied **only on the
`load_kv_async` arm**, which this step does not take (`park=0`). The port passes the same
`base_reserved` only into `allocate_external`. **So "the Rust scheduler over-admitted" is
ruled out: this batch is legal and stock-shaped.**

### Where the forward is stuck

```
nvrtc_block_quant.py:180 _eager_gq   <- vtl's EAGER FALLBACK
nvrtc_block_quant.py:290 _gq_impl
fp8_utils.py:637 per_token_group_quant_fp8
fused_moe/utils.py:148 _fp8_quantize -> no_dp_ep.prepare -> modular_kernel.apply
fp8.py:844 apply
moe_decode_gemv.py:755 moe_apply     (M=7803 > VTL_MOE_GEMV_MAX_M=8 -> stock apply)
... qwen3_next.py:589 forward ... model_runner.py:1380 execute_model
```

`_gq_impl` is vtl's override of the `_C` block-quant op, and it took **`_eager_gq`** — the
pure-PyTorch fallback taken whenever the call falls outside the compiled envelope. That
path is fully vectorised (no Python loop), so being caught there means the host is blocked
in it, not spinning — and what blocks in it is allocation: it materialises fp32 temporaries
proportional to the batch (~100 MB per call at this width), and at
`--gpu-memory-utilization=0.95` the caching allocator has little headroom, so those
allocations drive `cudaFree` — **an implicit device synchronize** — repeatedly, per MoE
layer.

That single mechanism explains every observation on record: invisible at decode widths,
catastrophic on the first wide batch, and it is why the *previous* dump was caught in
`step0_eos_ban`'s `torch.from_numpy(rows).to(device)` — another allocation-and-sync on the
same starved allocator. Both frames are bystanders of the same starvation.

### The tension to resolve, stated plainly

This does not by itself explain the `VTL_ENABLE_RUST_SCHED=0` result, because
`nvrtc_block_quant` installs independently of the scheduler. Three readings survive, and
the arms below separate them: (a) the stock-scheduler run never built a batch this wide,
so it never paid the cliff; (b) the stock scheduler chunks these three prefills across
steps where the Rust loop packs them into one; (c) two contributing factors. Note the
scheduler is a *trigger* under all three — the batch width is what detonates the path.

### Landed on this branch

* **The envelope refusal now names its cause.** `vtl: <op> call outside compiled envelope
  (<which predicate>); EAGER FALLBACK ... (this call: <shape>)` — ten predicates could
  refuse and the old message named none of them, so one boot line now says which to fix.
  The accepted path is untouched (the reason is computed only on the refusing arm).
* **Recorder sharpened**: the dump prints 24 ring entries instead of 12 (12 was exactly
  consumed by a run of decode steps, hiding any `pop_free`/`drain` behind them), and the
  kv-zeroing list is recorded **verbatim** instead of as a `(count, min, max)` span — a
  span cannot answer "is this id one a live request still reads", which is the collision
  the field exists to detect. In this dump step 906 zeroed id 8 for a request still live
  at 907, whose span was `(3, 1, 20)`: the span cannot say whether 8 was re-issued.

### Next arms, in order

1. **`VTL_ENABLE_NVRTC_BLOCK_QUANT=0`, full Rust stack ON.** The single highest-value
   test: if warm passes, the eager fallback is the wedge and the scheduler only supplied
   the batch width.
2. **Grep the boot log for `outside compiled envelope`.** With the change above, one line
   names the failing predicate and the shape. If it never appears, the fallback is *not*
   firing and hypothesis 1 is dead — check instead whether the whole boot ran eagerly from
   the start (`launchers is None`, i.e. the NVRTC compile failed).
3. Re-run the wedge with the sharpened recorder and read the verbatim `zero=` ids against
   the live request set.
4. `nvidia-smi` during the wedge: pinned ~100 % SM with the host in `_eager_gq` is
   consistent with allocator-sync starvation; ~0 % points at a copy/event deadlock instead.

---

## Fifth dump (2026-08-19 02:59): `VTL_ENABLE_NVRTC_BLOCK_QUANT=0` still wedges — the eager-fallback theory is dead

The arm was run and the server still hangs, so the block-quant eager fallback is **not**
the wedge. The dump is the most informative yet, because with that patch out of the way the
blocked frame moved to **stock vLLM code**:

```
qwen_gdn_linear_attn.py:1513 _forward_core      <- initial_state = ssm_state[prefill_state_indices]
qwen_gdn_linear_attn.py:1731 qwen_gdn_attention_core
... qwen3_next.py:589 forward ... model_runner.py:1380 execute_model
```

Line 1513 is the GDN state gather on the **prefill** branch (`num_prefills > 0`), one
statement before `chunk_gated_delta_rule` — the FLA chunked scan. Three dumps have now
been caught at three different frames (`step0_eos_ban`'s `.to(device)`, `_eager_gq`'s fp32
temporaries, and now this gather); all three are **allocation sites**, which is what a host
parks on when the device is not draining. The frames are bystanders; the device is.

### The recorder now isolates the trigger to a batch SHAPE

The 24-entry ring caught the whole approach, and it contains the control case this
investigation had been missing:

```
step=583..602  total=5      5-row decode          20 steps in ~0.3 s (~15 ms/step)
step=603       total=3
step=604       total=0      (drained)
step=605       total=5201   adm=2  zero=[17, 13]   2 prefills, NO decode rows  -> FINE
step=606       total=7805   adm=3  zero=[9, 5, 4]  3 prefills + 2 decode rows  -> WEDGE
               <nothing for 21.4 s>
```

**Step 605 is the control.** 5201 tokens of pure prefill completed normally; step 606 —
2 decode rows plus three fresh full prefills (2145 + 2600 + 3058) — never returned. The
previous run's wedge was the same shape (step 907: 2 decode + 3 prefill, 7803 tokens), and
so was the first incident's. The trigger is therefore **not batch width alone** but a
**mixed decode+prefill GDN batch with three concurrent prefills** — which is exactly the
case where the GDN metadata builder peels decodes off the front and builds a separate
prefill tail (`prefill_state_indices` / `prefill_has_initial_state` / `chunk_indices`,
qwen_gdn_linear_attn.py:1506-1508).

The verbatim zeroing ids also clear a hypothesis: `[17, 13]` then `[9, 5, 4]` — five
distinct GDN state blocks, no id re-issued to a second live sequence.

### What is now ruled out, by audit against the v0.25.0 source

* **Bookkeeping** — `C == T + P - 1` holds for all five requests, again.
* **Over-admission** — 7805 fits the 8192 budget, and stock's in-flight-prefill
  reservation applies **only** on the `load_kv_async` arm (`scheduler.py:895-901`), which
  this step does not take (`park=0`).
* **The mamba block-aligned split** — the crate's `mamba_block_aligned_split` mirrors
  `Scheduler._mamba_block_aligned_split` (scheduler.py:338-394) statement for statement,
  including the Marconi clause, and is fed the same `cache_config.block_size` and the same
  `need_mamba_block_aligned_split` flag from the live scheduler.
* **The mamba+connector prefix-cache special case** (scheduler.py:687-726, where stock
  swaps the converging walk for `find_longest_cache_hit_per_group` and takes
  `max(per_group_hits)`) — the crate models both walks and selects on `Config::connector`,
  and in any case the run reports **0.0 % local and 0.0 % external prefix-cache hit rate**,
  so no hit path is producing a number at all.
* **`nvrtc_block_quant`** — disabled for this run; still wedges.

### The one scheduler-side input still unresolved

Stock truncates every prefill chunk to `num_tokens - num_tokens % block_size`. That is a
**no-op when `block_size` exceeds the request** and a **real truncation when it does not**:
at `block_size = 4096` a 2145-token prefill is scheduled whole (what the ring shows), but
at `block_size = 2048` stock would have scheduled 2048 and left the tail for the next step
— i.e. it would never have built step 606 at all. The port mirrors the arithmetic, so the
verdict rests entirely on the live `cache_config.block_size`, and that value appears in no
dump and no log line this investigation has.

**Landed for it:** the flight ring now carries a static header, emitted as its first line
forever (rolling entries age out in about a second of decode steps, these never change):

```
FLIGHT params: block=<cache_config.block_size> mamba_align=<bool> max_tok=8192 max_seqs=5
               long_prefill=<n> chunked=True connector=<bool>
```

It is populated from the very dict handed to the crate, so the header and the crate's
parameters cannot drift. The next dump answers the question without another round trip.

### Next arms

1. **Re-run and read the `FLIGHT params` line.** If `block` is smaller than the prefills
   (~2.1k-3.1k), stock would have split them and the port did not — a port bug, and the
   fix is in the split's call site. If `block` is larger, the split is a no-op for both and
   the scheduler is exonerated on this point.
2. **`nvidia-smi` during the wedge** — still the cheapest bisect of the remaining space,
   and still not collected: pinned ~100 % SM = a kernel looping on bad chunk metadata;
   ~0 % = a copy/event deadlock or allocator starvation.
3. **Shrink the trigger directly**: `--max-num-batched-tokens=4096` (or `--max-num-seqs=3`)
   caps concurrent prefills below the three that every wedge has needed. If that alone
   serves cleanly with the full Rust stack on, it both confirms the shape and is a shippable
   mitigation that keeps the stack.
4. Only then: instrument the GDN metadata (`num_prefills`, `num_decodes`,
   `prefill_query_start_loc`, `chunk_indices` extents) for the wedged step.

---

## Sixth pass (2026-08-19): full audit of the scheduler against v0.25.0 — what is eliminated, and the fix

No new run. This pass reads the wedged path end to end against the pinned vLLM source and
settles what the port can and cannot be responsible for.

### Eliminated, each verified against v0.25.0 source

| Hypothesis | Verdict |
| --- | --- |
| Bookkeeping skew | `C == T + P - 1` holds for all five requests in every dump. |
| Over-admission | 7805 ≤ 8192; stock's in-flight-prefill reservation applies **only** on the `load_kv_async` arm (`scheduler.py:895-901`), and `park=0`. |
| Mamba block-aligned split | The crate mirrors `Scheduler._mamba_block_aligned_split` (`scheduler.py:338-394`) statement for statement, on the same `cache_config.block_size` and the same flag. **And it is derivably a no-op here**: for a 3058-token prefill to be scheduled whole — which the ring shows — `block_size` must exceed 3058, and at that size stock schedules it whole too. |
| `max_num_partial_prefills` | Defaults to 1 but is **not enforced anywhere in V1** (`grep` over `vllm/v1/` is empty), so it is not a divergence. |
| Batch ordering (decodes must precede prefills) | The V2 runner sorts: `req_ids = sorted(num_tokens_per_req, key=...)` (`gpu/model_runner.py:858`). Decodes land first for **any** scheduler output. `_may_reorder_batch` exists only in the V1 runner and is not on this path. |
| GDN metadata construction | `gdn_attn.py:333-403` builds `chunk_indices`/`chunk_offsets` from the same rebased `prefill_query_start_loc_cpu` it passes as `cu_seqlens`, and slices `has_initial_state[num_decodes:]` to match. Self-consistent. |
| GDN decode branch | `qwen_gdn_linear_attn.py:1480-1499` sizes the recurrent kernel from `cu_seqlens[:num_decodes+1]` and 2-token `q`; the full `ssm_state_indices` it also passes is indexed only over those rows. Correct. |
| `vtl` GDN patches | `gdn_decode_step` refuses any batch with `num_prefills != 0` (`:150`); `gdn_prefill_backend` at the shipped `auto` is **not installed at all**; `gdn_kernels` touches the output norm, not the scan. |
| `decode_fastpath` | Its predicate excludes new/finished requests, multi-token requests and a non-empty zeroing list — all present at the wedge step — so it stands down and re-primes. |
| `nvrtc_block_quant` | Disabled for the 02:59 run; still wedges. |

**The port is faithful on every scheduling rule that could produce this batch.** Stock can
build the same batch; it simply gets fewer chances to. That is why the honest framing is
"the Rust loop is the trigger, not necessarily the defect".

### The confirmed trigger, stated exactly

Every wedge needs a GDN batch that is **mixed** (`num_decodes > 0 and num_prefills > 0`)
**and carries three concurrent multi-thousand-token prefills**. The 02:59 ring supplies the
control: step 605 = 5201 tokens of *pure* prefill → fine; step 606 = 2 decode + 3 prefill,
7805 tokens → wedge. `num_decodes > 0 and num_prefills > 0` is precisely the condition at
`gdn_attn.py:340` that peels decodes off and rebases the prefill `cu_seqlens`, and the
frame the host parks in (`qwen_gdn_linear_attn.py:1513`) sits one statement inside that
branch. Pure-decode steps (hundreds of them) and pure-prefill steps both pass.

### The fix, in the order to try it

**1. `VTL_RUST_SCHED_FULL=0` — env only, no rebuild, try this first.**
This is the documented revert ladder's first rung (`FULL->0 -> RUST->0 -> ENABLE->0`). It
hands `schedule()` back to stock Python while **keeping** the Rust KV authority, the
resident table, the hasher, shm-IPC, the kernels and the quant track — far more of the
stack than `VTL_ENABLE_RUST_SCHED=0` retains. It also splits the remaining hypothesis
space in one run: serves ⇒ the defect is in the Rust `schedule()` decisions; still wedges ⇒
it is in the KV authority's block/state-id allocation (note `gdn_attn.py:219`,
`non_spec_state_indices_tensor = block_table_tensor[:, 0]` — the GDN state slot **is** the
first block of the mamba group's block table, so a wrong id there is read straight into the
kernel).

**2. `VTL_SCHED_MIXED_PREFILL_CAP=1` — new on this branch, needs a rebuild.**
Caps how many waiting requests may be admitted on a step that already carries running
(decode) rows, which removes the trigger shape directly: the prefills land on the following
step(s), where they are either pure prefill or mixed with fewer rows. Costs a little TTFT
in a burst, nothing at steady state, and keeps `VTL_RUST_SCHED_FULL=1`.

Default **0 (off, = stock)**, deliberately: it is a shape mitigation, not a root-cause fix,
and the audit above says the port is not provably wrong here. The dangerous part is
bookkeeping, not policy — a withheld candidate appears in no crate decision list while the
rebuild clears `self.waiting` wholesale from `waiting_order`, so anything not re-added is a
lost request. Both rebuild arms re-add the withheld tail behind the crate's leftovers,
preserving FCFS, and `_self_check` models the truncate-then-restore arithmetic and asserts
total preservation.

**3. `VTL_ENABLE_RUST_SCHED=0`** remains the last-resort mitigation already in hand.

### What would still settle the root cause

Only two observations, neither of which this branch can produce on its own: `nvidia-smi`
during a wedge (~100 % SM = a kernel looping on its chunk metadata; ~0 % = a copy/event
deadlock or allocator starvation), and the `FLIGHT params` line now emitted in every dump.
