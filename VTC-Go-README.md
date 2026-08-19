# Round 2 — Implementations Summary

This document lists the round-2 work landed on `main` (August 8–17, 2026) and what each
implementation means. The work falls into six groups.

## 1. Workspace bootstrap — model-agnostic round-2 (`147a811`, `aec6c28`)

Round-1.2 was welded to LFM2.5-1.2B, but round 2's model wasn't chosen yet. This created
`round-2/` as a copy of round-1.2's workspace with all architecture-specific work stripped
out:

- 20 LFM2-specific files (patch modules, CUDA sources, vLLM patches, bench tests, docs)
  moved to `round-2/reference/lfm2/`, which the build excludes.
- Model identity removed everywhere: the warm-up healthcheck reads the served name from the
  server's own `/v1/models`, `trace_stats` derives KV geometry from `config.json`,
  `quant_w4a8` logs observed shapes instead of asserting one model's, and `profile_trace`
  buckets by architecture family.
- A per-round `round.mk` include so each round pins its OWN forked-vLLM digest — round-2's
  patch set no longer carries the model-specific patches, so sharing round-1.2's digest
  would silently serve a fork with patches this round does not ship.
- `VTL_RUST_SCHED_REQUIRE=1`: a Rust scheduler that silently switches itself off is
  indistinguishable from "ran and did not help" in every latency number, so for bench/CI
  the refusal becomes a boot failure instead.
- `HANDOFF.md` documents the state of the workspace for whoever picks it up.

## 2. Serving-path Rust-port optimizations (`f061687`, `3da5745`)

A three-item latency batch on the scheduler/input path, all behind flags shipped on in
`docker-compose.yaml` (code-level defaults stay off, so a boot without compose degrades to
stock):

- **Schedule-marshalling collapse** — three rungs:
  - `VTL_RUST_SCHED_LEAN`: drop dead marshalled fields (`token_budget_left`,
    `num_common_prefix_blocks`), skip empty waiting-queue rebuilds, lean
    `CachedRequestData`.
  - `VTL_SCHED_DECISIONS_ARENA`: persistent numpy buffers replace the per-step PyDict
    crossing the Rust→Python boundary; check mode falls back to the dict path on
    divergence.
  - `VTL_SCHED_SO_RING`: a 2-slot `SchedulerOutput` ring for pure-decode steps, with a
    live-request identity check so a recycled scheduler slot can never replay a finished
    request's output.
- **Raw ADD records on the shm channel** (`VTL_SHM_IPC_RAW_ADD`): prompt token ids land as
  a zero-copy u32 view instead of ~4400 individual PyLongs; golden frame pinned in both
  Rust and Python.
- **Frontend encode cache** (`VTL_FRONTEND_ENCODE_CACHE` + `VTL_ENCODE_CACHE_SPLIT`): a
  byte-prefix memcmp cache keyed per conversation so re-sent conversation prefixes skip
  re-tokenization; split output only trusted after the self-verify budget completes equal.
- r9 micro-opts (list-scan zero guard, latched `out_is_open`, aliased
  `num_tokens_with_spec`, hoisted `_rust` lookup) and a `block_pool` fix (the null
  placeholder is no longer ref-counted, fixing pre-existing crate and parity test
  failures).
- A second review pass (`3da5745`) tightened locking and allocation behavior: the ring's
  blocked-clauses now include the lean clauses, arena entry points take `&self` behind a
  `RefCell` (no exclusive borrow across the GIL-released window), and the encode cache's
  large allocations and evictions moved outside the mutex.

## 3. The Rust CUDA-graph runner, M1–M4 (`67ed854`, `d2e5f37`, `d2e5c2d`, `ab3b362`)

The centerpiece: moving the decode-step launch loop out of Python into a Rust crate that
replays pre-captured CUDA graphs.

- **M1 — export the whole-burst graph handles** (`67ed854`): the exported table must carry
  nstep's *unroll* graphs (the ones that end a step with tokens in `BURST.accum`, the
  buffer the pre-planned D2H reads), not the stock forward-only graphs that never write it.
  A shape with no unroll graph gets a NULL-handle row so the runner declines it; `export`
  refuses outright when the unroll rung was demoted at boot; the armed log prints WHICH
  sizes are launchable, since that is the auditable number.
- **M2 — one Rust launch per committed burst** (`d2e5f37`): the crate does the graph
  launch, the pre-planned pinned D2H, and the whole step commit through the same
  `step_pack_locked` entry point the Python path uses. Discovered "hazard 8": with async
  scheduling's depth-2 batch queue (`schedule(k) → sample(k) → update(k-1)`), committing at
  sample time can append step k's tokens ahead of step k-1's — so the launch was
  interlocked on `inflight == 1`, meaning it armed but never engaged under the served
  config.
- **M3 — `VTL_RUST_RUNNER_REQUIRE`** (`d2e5c2d`): mirrors `VTL_RUST_SCHED_REQUIRE` — a
  runner that quietly never armed looks exactly like one that ran and did not help, so an
  A/B arm could measure the Python step and report a win. The flag turns silent refusal
  into a boot failure; compose documents that ARMED IS NOT ENGAGED.
- **M4 — continuation graph + the multi-step Rust loop** (`ab3b362`): relaunching the
  unroll graph back-to-back silently re-emits launch 1's first token (its prologue argmaxes
  a hidden-states buffer only the stock replay refreshes). The continuation graph is the
  same bodies with no prologue, feeding off the previous launch's own final argmax,
  shadow-verified with TWO launches because a single-replay shadow cannot see back-to-back
  staleness. Launch *i* fills accumulator columns `[i*n, (i+1)*n)` so the multi-launch step
  is one contiguous token block; a pre-flight `runner_packable` declines a step while
  declining is still free.

## 4. Making the runner engage and be correct

- **Codex-review fixes for the runner wiring** (`7e382e9`): the "unpacked" recovery
  zero-accepted the whole batch and let worker state advance past the store — the batch is
  now retired as length-capped instead. A second finding (stale padded `idx_map` rows) was
  refuted and documented.
- **Update-time commit — hazard 8, option b** (`d6c3327`): the fix that lets the runner
  actually engage. Commit moves from SAMPLE time to UPDATE time, where steps ARE ordered,
  behind a pinned ring of 2 buffers + 2 events: `Runner::launch(slot)` enqueues only,
  `Runner::commit(slot)` event-syncs, gathers, and packs. `Manager::runner_owns` re-checks
  at commit time that every slot still holds the request the launch planned for.
- **Codex-review fixes for update-time commit** (`b1d5a22`): a post-launch raise could fall
  into the sampler's fallback and double-run sampling (now an explicit `PostLaunchError`);
  ownership-check and pack ran under different locks (a TOCTOU, now one `MutexGuard`);
  `ring_reuse` cleared two stamps but not `vtl_runner_seq`.
- **Runner arming order** (`39f4f88`): on a server boot, `capture_model` runs before vLLM
  constructs the Scheduler, so the runner could never see the Rust KV manager at capture
  time and `REQUIRE=1` failed every boot. Arming is now a two-phase rendezvous completed at
  scheduler init; permanent impossibilities still latch refused at capture.
- **Classify priming** (`e34e80d`): stock populates `aot_sliding_window` only inside the
  first REAL forward build, but graph capture runs dummy builds — so nstep's one-shot
  classification silently bailed every boot, costing the whole burst-graph ladder. It is
  now primed in place at classify time, and every remaining bail logs a reason.
- **Multi-step residency, add-time prereg, lazy tokens, Rust shm input** (`da9ea46`) —
  a four-point latency batch:
  1. Multi-step CUDA-graph residency under update-mode commit: `launch(steps, width)`
     issues k back-to-back launches with ONE D2H and ONE event; `commit(steps)` syncs once
     and packs per step, masking stopped slots out of later steps. `VTL_RUST_RUNNER_STEPS=8`
     is live.
  2. Crate-side burst eligibility: `Decisions.burst_eligible` computed in
     `schedule_resident`, crossing via the arena; both numpy reductions dropped from
     `_publish_ready`.
  3. Add-time registration (`VTL_RUST_PREREG=1`): request slot interned and stop params set
     from a `preprocess_add_request` hook on the input thread, so `decide()`'s miss arm
     seeds its mirrors from it.
  4. Input path: `_LazyAllTokens` replaces the ~4400-PyLong `tolist`
     (`VTL_SHM_IPC_LAZY_COPY=1`); a new `input.rs` iceoryx2 subscriber splits raw-ADD
     frames in Rust (`VTL_SHM_IPC_RUST_SUB=1`), with the Python loop retained as the
     fallback rung.

(`8c41a31` / `1761691` were a waive-then-revert of `VTL_RUST_RUNNER_REQUIRE` in the
localtest overlay — net zero on `main`.)

## 5. CI / collaboration plan for the H200 environment (`dc7f8d8` → `db4643a`)

Docs and workflows for working with the on-site team's constrained VPS access: a round-2
CI + collaboration plan (`dc7f8d8`), VPS CI (`d7d0be2`) and jump-host CI (`65b6dc6`),
dropping the derived runner image since the stock image ships the tools (`aaa8225`), images
moved to the on-site team's Docker Hub account (`472e23c`), a "case 3" console-only VPS
flow — paste-in, no copy-out, degraded egress (`8b88c64`) — finally collapsed to the one
workflow the access rules actually permit (`db4643a`).

## 6. Official round-2 spec adoption (`6fe94e1`, PR #45)

The competition's official spec landed:

- **Scoring**: new ERS band — TTFT floor/ceiling 200/6000 ms, TPOT 8/100 ms (gamma and w
  unchanged) in `_ci_report.py`; every hand-derived exchange-rate figure refreshed
  (1 ms TPOT ≈ 63 ms TTFT now).
- **Workload**: grading is aiperf's `inferencex-agentx-mvp` scenario (Weka corpus, 900 s,
  concurrency 5, ctx 204800, hidden seed). Added `make bench-aiperf`, an
  aiperf→repo-schema adapter with selfcheck, and CI-report pickup of
  `bench-aiperf*.json`. The synthetic-trace path is kept for CI/iteration.
- **Docs**: HANDOFF gains section 6 (spec, band math, serving-config open items — including
  the blocking `max-model-len` 32768 < 204800 placeholder); round-2 README and
  bench/README de-staled from round-1 text.

## In one sentence

Round 2 built a model-agnostic serving workspace, ported the hot scheduler/input paths and
then the decode launch loop itself into Rust (CUDA-graph replay with multi-step
residency), hardened it through several review passes until it correctly engages under
async scheduling, and aligned benching with the official aiperf-based spec.
