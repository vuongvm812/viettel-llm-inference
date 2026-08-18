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
