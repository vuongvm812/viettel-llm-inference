# Full Rust model runner — feasibility + design

2026-08-08 investigation (three parallel code sweeps: per-step Python inventory, CUDA-graph
ownership, reusable Rust surface). Verdict: **feasible**, no custom capture machinery and no
new C++ extension needed. But a topology contradiction in the served config gates everything
and must be resolved first (§1).

## 0. The one fact that makes it feasible

torch 2.11.0 (the exact pin in `vllm/vllm-openai:v0.25.0`) exposes the raw graph handles as
Python ints: `torch.cuda.CUDAGraph.raw_cuda_graph_exec()` → `cudaGraphExec_t`
(`torch/cuda/graphs.py:174-179`). vLLM uses `keep_graph=False`, instantiates at
`capture_end`, and never re-instantiates or resets a decode graph — the handle is stable for
the process lifetime.

And vLLM's steady-state FULL replay is *literally one launch*: `run_fullgraph` is two
asserts, a no-op offloader sync, and `self.graphs[desc].replay()`
(`cudagraph_utils.py:399-412`). No per-replay bookkeeping exists to reimplement. Dispatch is
a precomputed dict (`_candidates`, padding baked in at init) — flattenable into a static
Rust table at boot.

So: **capture stays in Python at boot; after `capture_model()` we export
`{BatchExecutionDescriptor → exec_handle}` + the raw stream (`torch.cuda.current_stream().cuda_stream`),
and Rust drives `cuGraphLaunch` via dlopen'd libcuda.** Precedent for driver-API use already
in-tree: `vtl/nvrtc.py` (`cuModuleLoadData`/`cuLaunchKernel` from Python).

## 1. Ground truth that gates the design: mp vs uni

`docker-compose.yaml:59` forces `--distributed-executor-backend=mp` → EngineCore and the GPU
worker are **separate processes**, talking via pickle over shm message queues, two RPCs per
step. Consequences, all verified in code:

| Feature | Designed for | Under served `mp` |
|---|---|---|
| N-step burst (`VTL_NSTEP*`) | uni | **DEAD** — `BURST.armed` is a module global set in WorkerProc, read in EngineCoreProc (`nstep_decode.py:819-831` vs `:213-238`). `commit_burst` bails every step. |
| `VTL_SAMPLE_IN_GRAPH` | uni | **DEAD** — same handshake. |
| R9 fast-hit path | uni | **DEAD** — keyed on `id(req_id_to_index)` (`rust_sched.py:1734`); pickle never preserves identity across processes, so it misses every step and the full `decide()` loop runs. (R8/tokstore/inline-publish still work.) |
| `VTL_SCHED_SO_RING` | mp | **requires mp** (`rust_sched.py:325-327`) — the pickle copy is what makes handing back the same `SchedulerOutput` object safe. |
| `_publish_ready` probe | uni | 12 predicates + 2 numpy reductions per step in WorkerProc, result unreadable cross-process — pure waste under mp. |

The stack is split-brained: burst/R9 assume uni, the ring assumes mp. `nstep_parity` tests
pass because they use in-process `LLM(...)`. **A Rust runner resolves this decisively toward
uni**: with the runner and the scheduler in one process, the pickle hops disappear, the ring
becomes obsolete (the runner consumes the decision arena directly), and burst/R9 come alive.

Also confirmed: `decode_fastpath` Part 1 and the whole Rust scheduler require
`--mamba-cache-mode=align` for a hybrid model (crate refuses at boot otherwise); compose
must carry it for such models.

## 2. What the runner must absorb (per steady decode step, WorkerProc)

From the full inventory, the GPU-touching work outside the graph:

- **Pre:** `apply_staged_writes` (Triton + UVA copies), `prepare_pos_seq_lens` (Triton),
  `combine_sampled_and_draft_tokens` (Triton, allocates `logits_indices` fresh),
  `gather_block_tables` (Triton, only on new blocks), `compute_slot_mappings` (Triton,
  in-place), mamba align preprocess (2 Triton), FA3 `get_scheduler_metadata` (C++ →
  persistent buffer), one `idx_mapping` H2D (fresh alloc each step).
- **Graph:** one `cudaGraphLaunch`.
- **Post:** `hidden_states[logits_indices]` gather + `compute_logits` + argmax,
  `post_update` (Triton), mamba postprocess (Triton), `AsyncOutput` D2H (3 copies on a side
  stream, fresh CPU tensors, event sync on another thread).

**Key insight: `nstep_decode._burst_body` already proves the entire step body is capturable
as ONE graph** (`nstep_decode.py:295-355`: feedback → slot mappings → FA3 AOT schedule →
forward → logits → argmax), for sizes (1,2,4,8), unpadded. The Rust runner should not
re-launch these kernels individually — it replays the whole-step graph.

## 3. Architecture

Python keeps: boot, model load, torch.compile, capture, **prefill and every non-steady step**
(stock path preserved, degrade-not-fail). Rust takes the steady-state loop.

### Phase 0 — flip the topology (no Rust; prerequisite; independently valuable)
Drop `--distributed-executor-backend=mp` → UniProcExecutor. SO_RING self-refuses under uni
(already coded). Burst, in-graph sampling, and R9 fast-hit come alive — **this alone
recovers a large share of the prize with a compose-line change** and is the Phase D A/B the
handoff already lists. Measure on the H200 (≥3 boots/arm) before writing any Rust.

### Phase 1 — Rust owns graph replay
New `cuda` feature in `vtl-sched` (dlopen libcuda at runtime — builder stage has no CUDA
headers and needs none; ~8 symbols: `cuGraphLaunch`, `cuEventRecord/Synchronize/Create`,
`cuMemcpyDtoHAsync`, `cuStreamWaitEvent`). Boot: Python walks
`runner.cudagraph_manager.graphs` (+ nstep's burst graphs), exports
`(desc, raw_cuda_graph_exec(), stream)` into a Rust dispatch table mirroring
`_candidates`' padding logic. Per step: one FFI `run_step(num_reqs, num_tokens, uniform,
loras)` → dispatch + launch. Small win alone; proves the Rust↔CUDA plumbing and the parity
harness.

### Phase 2 — Rust owns the whole steady step
Generalize the `_burst_body`-style whole-step capture (accept padded batches, all capture
sizes — dispatch per §5 of the graph study, not burst's exact-match rule). Rust then:
writes the few host inputs, launches the step graph, issues a **pre-planned D2H**
(`cuMemcpyDtoHAsync` into a pinned Rust-owned buffer + event) replacing `AsyncOutput`'s
fresh-tensor copies, and on event completion feeds sampled ids straight into
`update_step_pack_np`'s Rust core (same process under uni) → stop-check → shm publish.
Steady-state step = **zero Python**.

### Phase 3 — Rust-owned multi-step loop
Python's busy loop blocks in one FFI call that runs N steps back-to-back in Rust, exiting on
any non-steady signal. The shm input side is already iceoryx2 — Rust can poll the request
listener natively to detect arrivals without waking Python. Endgame: the Python process is
boot + prefill only, matching the frontier-audit endgame note.

## 4. Hazard checklist (from the inventory — each needs an explicit answer in the port)

1. `torch.inference_mode()` and current-stream are **thread-local** — a Rust thread calling
   ATen (avoid: launch via driver API only) or recording on the wrong stream breaks
   ordering; replicate `copy_stream.wait_stream(main_stream)` with `cuStreamWaitEvent`.
2. `copy_event.synchronize()` currently runs on a different thread with an explicitly set
   CUDA device context — Rust must `cuCtxSetCurrent` or stay on one thread.
3. `_PubCounters` (`shm_ipc.py:161-180`) is lock-free *because of the GIL* — moving the
   publisher fully into Rust needs real atomics (already natural in the crate).
4. Module-state caches (`decode_fastpath._C`, `BURST`, R9/TOK latches, `_ZERO_COMMON`,
   facade `__class__` swap) assume a single-threaded step loop — the Rust loop must not
   interleave with a Python step.
5. A Rust spin loop on the 3-vCPU judge box starves the shm threads — park on events, never
   spin.
6. `step_eplb_after` is a decorator, easy to lose in a port (inert here, but verify).
7. `get_offloader().sync_prev_onload()` — verify no-op for the served config on-box.
8. **The batch queue, found while wiring M2 and the one that gates Phase 2.** Rust commits
   a step's tokens inside `sample_tokens` (`step_pack_locked`: token store, resident delta,
   `cache_blocks`); Python commits them inside `update_from_output`. Async scheduling runs a
   batch queue of depth `max_concurrent_batches` = pp_size + 1 = **2** on the V2 runner
   (`config/vllm.py:490`), and `step_with_batch_queue` is `schedule(k) -> execute(k) ->
   sample(k) -> update(k-1)`. So `sample(k)` runs while step k-1 is still unapplied, and a
   Rust commit there would append step k's tokens AHEAD of step k-1's: scrambled output and
   a block-hash chain over positions in the wrong order, with nothing to crash. Mixing the
   two commit points is unsafe in BOTH directions, and "all steps Rust" cannot bootstrap
   (the first step of a busy period is a prefill, which Python must commit). So the launch
   is interlocked on `rust_runner.STATE.inflight == 1` and the runner idles under the served
   config. **Phase 2 is therefore not a step-level optimisation at all — it only pays once
   the runner owns the loop** (Phase 3 / `max_concurrent_batches` 1), which is also the only
   configuration where its blocking `cuEventSynchronize` inside `sample_tokens` is not
   giving up the overlap async scheduling exists to buy. The other way out, if the batch
   queue must stay, is to make Rust hand the tokens back for Python to commit at update time
   (a pinned ring of 2 instead of one buffer) — that keeps the pre-planned D2H and the
   `decide()` elimination but not the multi-step loop.

   **RESOLVED (2026-08-09) via option (b), update-time commit.** `VTL_RUST_RUNNER_COMMIT`
   picks the commit point; compose ships `"update"`. The per-step FFI is split in two:
   `Runner::launch(slot, ...)` runs at sample time and only ENQUEUES (`cuGraphLaunch` ->
   `cuMemcpyDtoHAsync` into pinned ring slot `slot` -> `cuEventRecord`; no wait, no commit),
   and `Runner::commit(slot, ...)` runs inside `update_from_output` (`cuEventSynchronize` ->
   `gather_sampled` -> `step_pack_locked`). Updates arrive in strict step order, so there is
   no out-of-order hazard and no `inflight` interlock, and the depth-2 async overlap is
   preserved -- which is also why the ring is 2 (= `max_concurrent_batches`). Python owns the
   ring slots through a FIFO of `(slot, stash)` on `rust_runner.STATE` and declines a launch
   when both are outstanding. The commit re-checks that every slot still holds the request
   the launch planned for (`Manager::runner_owns`): between `sample(k)` and `update(k)` a
   request can finish at `update(k-1)` and `schedule(k+1)` can hand its slot to a new one. A
   decline there costs only the Python `decide()` -- a launch commits nothing, and the
   worker's `AsyncOutput` is still carrying the same tokens, which is what makes every
   fallback on this path safe. Kept from Phase 2: `cuGraphLaunch` in place of torch's replay
   dispatch, the pre-planned pinned D2H, and the `decide()` collapse at update time. Given
   up: the multi-step residency loop (`run_steps` / `VTL_RUST_RUNNER_STEPS`) and its
   continuation-graph capture rung, both now sample-mode-only and off in the served config.

## 5. Repo hygiene found during the investigation (fix regardless)

- The root `vllm/` checkout is **v0.26.0**, and `gen.sh` regenerates
  `vtl/vllm_patches/v0.25.0/*.patch` from it — the patches are v0.26-shaped and stale
  vs the tree they claim to patch. Regenerate from `vllm-v0.25.0-edited/`.
- v0.26 removed `AttentionStatePair`; anything reading `mgr._vtl_attn_states` dies silently
  on an upgrade. The Rust runner keys on `BatchExecutionDescriptor` and survives.

## 6. Sizing and order

Under the round-2 scoring band (F/C_tpot = 8/100 ms) TPOT is worth at most ~0.011 ERS/ms — an
order of magnitude below the round-1 figure this document was sized against, so the same
0.22 ms/knob A/B floor now gates only ~0.002 ERS; re-derive the per-knob thresholds before
committing effort. Phase 0 is a compose line
and plausibly the single biggest TPOT lever currently available (it re-enables three shipped
optimizations at once). Phases 1–2 are the actual runner (~1–2 days + ~1–2 weeks incl.
parity); Phase 3 is polish (~1 week). Do not start Phase 1 until Phase 0's uni A/B confirms
the topology on the H200 — the runner's design (same-process arena handoff) depends on it.
