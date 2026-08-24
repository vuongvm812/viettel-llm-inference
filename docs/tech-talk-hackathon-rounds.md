# From `docker compose up` to a Rust CUDA-graph runner

**A 30-minute tech-sharing talk on the Viettel "AI Race" LLM-inference hackathon (task 3)**

> One repo, 383 commits, three rounds, three completely different problems:
> a prefill-bound full H200, a latency-band MIG slice, and a 122B MoE agent workload.
> This talk walks the ladder we climbed — vanilla flag tuning → forked vLLM +
> custom CUDA → Rust ports of the scheduler and the decode launch loop — and the
> lessons each rung taught us.

| # | Section | Time |
|---|---------|------|
| 1 | The assignment: three rounds, three problems | 5 min |
| 2 | The vanilla phase: tuning the plain vLLM image | 5 min |
| 3 | Going custom: forked vLLM, CUDA kernels, Rust ports | 13 min |
| 4 | Results | 2 min |
| 5 | Lessons | 5 min |

---

## 1. The assignment (5 min)

### How the competition works

- The judge pulls **a registry image + our `docker-compose.yaml`**, mounts the model
  at `/model`, and scrapes an OpenAI-compatible server on `:8000`. No build context,
  no local mounts. The entrypoint, model path, and served-model-name lines are locked.
- Submissions are **live-endpoint registrations** (`airace endpoint --task llm --url ...`)
  with a limited attempt budget. Registering a dead endpoint burns an attempt.
  If the server dies mid-grading-run, that attempt scores 0.
- Custom images are allowed — forked vLLM, plugins, baked caches — but two things got
  competitors flagged as **cheating**: ngram/prompt-lookup speculative decoding
  (output-identical at temp 0, worth ~7.4 points/ms on paper), and truncating
  `--max-model-len` below the model's context (fails the judge's long-context probe).

### Three rounds, three completely different problems

| | Round 1.1 | Round 1.2 | Round 2 |
|---|---|---|---|
| **Model** | Qwen3.5-2B (VL hybrid, GDN linear attention) | LiquidAI/LFM2.5-1.2B (10 short-conv + 6 GQA layers) | Qwen3.5-122B-A10B-FP8 (MoE 256-expert top-8, 36 GDN + 12 full-attn layers) |
| **Hardware** | Full H200 (~120 GB KV budget) | **H200 MIG 1g.18gb: 18 GB, 16 SMs, 3 vCPU, 8 GB RAM** | Full H200 141 GB, 22-core Xeon |
| **Workload** | 120 reqs, p50 prompt 18.7k tok, **101:1 prefill-bound**, 82.4% prefix hit | 70 convs × 6 turns = 420 reqs, ~4.7k ctx, ~82% prefix hit | aiperf `inferencex-agentx-mvp`: **real Claude Code session traces**, 900 s, concurrency 5, ctx 204,800, hidden seed |
| **Scoring** | Latency percentiles + throughput | ERS: TTFT 10/400 ms, TPOT **1/10 ms**, γ=2, w=0.5 | ERS: TTFT 200/6000 ms, TPOT **8/100 ms** |
| **What mattered** | Prefill throughput + prefix cache | **Host overhead per decode step** | Finishing requests; TTFT tail; MoE bandwidth |

**Speaker notes:**

- The ERS formula (both rounds): `S_request = w·s_ttft + (1−w)·s_tpot`, where each
  component is `clamp((Ceiling − x)/(Ceiling − Floor), 0, 1)^γ`, averaged over ALL
  requests — a failed request scores 0 but stays in the denominator. γ=2 means tails
  are punished quadratically.
- Round 1.2's band is what drove the whole custom-code arc: with a 1–10 ms TPOT band,
  **1 ms of TPOT was worth ~28–37 ms of TTFT** (~8.6 score points at our operating
  point). The GPU floor on the 16-SM slice was ~1 ms/step; observed TPOT was ~3 ms.
  So ~2 ms/token was **host-side Python overhead** — that is the entire motivation
  for the Rust ports.
- Round 2 flipped the economics: the TPOT band widened 8–100 ms, so the marginal value
  of a TPOT millisecond dropped ~10× (dERS/dTPOT ≈ 0.011/ms). Sub-millisecond
  micro-optimizations became near-noise; what mattered was request completion and tails.
- Round 2 also had a logistics twist: one VM per team behind a jump host, **interactive
  SSH only** (no scp/rsync/tunnels), internet only via a proxy. The Dockerfile's network
  fetches are deliberately non-fatal, so building on the VM silently produces a
  **gutted image** that benches worse for untraceable reasons. Rule: **the VM never
  builds, it pulls** — images built in GitHub CI, pushed to Docker Hub, pulled by digest.

---

## 2. The vanilla phase: plain vLLM param tuning (5 min)

### The false start worth admitting

We didn't start with vLLM. The first ~2 weeks built a **from-scratch Rust inference
engine** — llama.cpp FFI, a 3-pinned-core LMAX-Disruptor pipeline, continuous batching,
shared-prefix KV, PGO/LTO/BOLT. It was killed in one commit
(`60c44cd "remove custom rust inference"`) and replaced with tuning stock
`vllm/vllm-openai`. Matching a mature engine's whole feature surface is not a
two-week project; the right move was to make vLLM fast, not to replace it.

### What plain-image tuning actually bought (round 1.1)

All of this is flags and env vars on the stock image — no code:

- **`--enable-prefix-caching` — "the single biggest win, and it is lossless."**
  The trace re-sends a byte-identical 6,388-token system prompt; block-level hit rate
  82.4%, eliminating 1.99M of 2.41M prefill tokens. Measured: the hit rate is 82.4% at
  *every* block size 1→64, so we did **not** tune block_size or build a radix tree —
  the residual was worth 0.03%.
- **`--max-model-len` sized to the trace, not the model**: longest prompt 27,331 tok →
  32768, not 262144 (which won't even boot) and not the baseline 16384 (which rejects
  most of the trace).
- **`--max-num-seqs` swept 256 → 16**, with the verdict written into the compose:
  *"Pareto optimum: 12 cut tbt 18→17 ms but blew ttft_p95 14k→20k (−0.3). 20 raised
  tbt. Frontier is ~62.5 here."*
- **fp8 quantization + `--kv-cache-dtype=fp8_e4m3`**: prefill is GEMM-bound and 99% of
  tokens are prefill; Hopper fp8 tensor cores act on all of them.
- **Chunked prefill + `--max-num-batched-tokens=8192`**: small prefill chunks stop a
  new turn's prefill from spiking in-flight decode latency (γ=2 punishes that tail).
- Things measured and **rejected**: KV offload to CPU/NVMe (working set 15.7 GB vs
  120 GB budget = 8× headroom, and the miss path is ~700× slower than HBM);
  speculative decoding (decode is ~1% of tokens in a 101:1 trace — "measured, not
  assumed"); `--async-scheduling` passed explicitly (it's already the default; passing
  it only converts warnings into boot failures).

### Image-level tuning without touching vLLM code

- **jemalloc, latency-tuned not RSS-tuned**: `dirty_decay_ms:-1` never returns pages
  (no purge syscalls on the alloc path), percpu arenas, THP metadata. Known risk:
  RSS only grows and the judge caps RAM at 8 GB with no swap → validate peak RSS
  before submitting or the OOM-kill is unscoreable.
- **Thread budgets for a 3-vCPU cgroup**: at TP=1 vLLM's UniProcExecutor never
  calls the code that sizes thread pools to the quota — torch sees the host's 64+
  cores. `OMP_NUM_THREADS=3`, `OMP_WAIT_POLICY=PASSIVE` (a spinning libgomp worker
  steals a third of the host budget from the decode loop).
- **The one knob swept against the real judge**: Rust-frontend tokio workers at
  1/2/3 scored **71.5 / 72.5 / 72.1**. The compose comment: *"2 is a measured
  optimum; do not re-derive it from theory."*
- Baked warm caches in the image (`torch.compile`, Triton, AOT), and the
  **healthcheck doubles as prefix-cache warm-up** — it replays trace prefixes, so
  readiness and a hot cache arrive together.
- Even `sys.setswitchinterval`: CPython's default GIL switch interval is 5 ms —
  longer than an entire decode step. Lowered to 0.2 ms.

### The measurement discipline (this is the real content of the vanilla phase)

- **3 boots per arm minimum.** Boot-to-boot noise floor ~0.5 ms TPOT; a knob has to
  clear ~0.22 ms to prove itself.
- The ledger kept the nulls: `--max-num-scheduled-tokens` 2048 vs 8192 measured **null
  and directionally backwards** vs prediction; `use_inductor_graph_partition` measured
  **worse** and cleared the noise floor — knob deleted.
- `docker-compose.yaml` became the engineering log: every flag carries its rationale,
  its measurement status, and a one-line revert.

---

## 3. Going custom (13 min)

**Framing:** on the round-1.2 MIG slice, decode was ~1 ms GPU + ~2 ms host per step.
Under async scheduling, TPOT ≈ max(host, gpu) — **the host IS the TPOT.** Everything
in this section attacks either host time or memory bandwidth.

### 3a. The forked vLLM (3 min)

Not a real fork — a **`patch -p1` overlay onto stock v0.25.0 site-packages**, with the
compiled `.so`s untouched and a version assert so a base-image bump fails the build
loudly. Everything that *could* be a runtime monkey-patch is one (a plugin with ~30
`VTL_ENABLE_*`-gated modules; `VTL_DISABLE=1` = provably stock). The patches exist
only where monkey-patching is structurally impossible:

- **`short_conv.patch`** — the short-conv decode body runs inside an opaque custom op;
  there is no Python seam to wrap. Rewired to call our fused kernels.
- **`lfm2.patch` — one line**, `torch.empty_like(residual)` instead of
  `empty_like(hidden_states)`. The wrong allocation source made the FX node have a
  user outside the fusion pattern, so **the RMSNorm+quant fusion silently never fired
  on 10 of 16 layers**. One line unblocked a whole fusion pass.
- **`hotpath_microopt.patch`** — line-level edits inside per-step hot loops (dead
  kernel launches, contextmanager allocations, identity caches). No function boundary
  to wrap.
- **`api_server_rust_frontend.patch`** — the `if __name__ == "__main__"` block isn't
  importable, therefore isn't patchable at runtime.

The **Rust frontend** (vLLM's own `vllm-rs`) got source patches before `cargo build`:
sonic-rs (SIMD JSON) request parsing borrowing the body bytes; **iceoryx2 shared-memory
IPC** replacing ZMQ on the data plane (ZMQ kept for handshake and as the degrade path),
with golden byte-frames pinned in both Rust and Python so layout drift fails the image
build; and **per-token SSE streaming** — mandatory, because once the engine emits N
tokens per record, a chunk-counting grader would report TPOT **N× worse than reality**.
Plus PGO where the mock engine is paced to production cadence — PGO trained at memory
speed ships a *slower* binary.

### 3b. Custom CUDA kernels (4 min)

Eight kernels for round 1.2 (`vtl/csrc/`), all sharing three design rules:
**(1)** numerics match stock element-for-element — deliberately including
double-rounding, because "more accurate" shifts the fp8 amax and per-token scale:
*that is a different kernel, not a better one*; **(2)** every kernel has a
`*_supported()` gate that **refuses** anything outside the served layout rather than
approximating; **(3)** parity tests against the stock op chain, not just golden values.

The ladder, in increasing ambition:

1. **Fused elementwise+quant** (`rms_norm_quant`, `silu_mul_quant`, `mul_quant`,
   `dynamic_per_token_quant`): kill bf16 intermediates' HBM round-trips and a launch
   each. Stock RMSNorm+quant read the input three times; ours once.
2. **`bcx_conv_gate_quant`** — one launch replacing three per conv layer × 10 layers:
   B·x → depthwise causal conv + state-ring rotation → C· gate → fp8, with `Bx` never
   leaving registers.
3. **`conv_align_fused`** — stock ran two full-grid Triton launches *outside the CUDA
   graph, every step*, to compute six scalars and then fast-exit 15 of 16 times.
   Fused to one launch.
4. **W4A8 CUTLASS extensions** — vLLM's 10 GEMM instantiations and heuristic were
   tuned on a 132-SM Hopper; on a 16-SM slice the wave structure is wrong (1.5 ragged
   waves, 64-deep k-loops). Added Stream-K, small-tile, and TMA-multicast arms.
5. **The short-conv decode megakernel** (`shortconv_decode_mega.cu`, 504 lines): the
   entire decode block — RMSNorm+quant → W4A8 GEMV → conv+gate+quant → W4A8 GEMV — in
   **one launch with three hand-rolled device-wide sense-reversing barriers**. The
   W4A8 GEMV exists as a *device function* precisely to be callable between barriers.
   Co-residency of all blocks is a **correctness precondition** (a non-resident block
   hangs the device), so the grid comes from the occupancy API on the real kernel, and
   below the minimum the Python gate refuses and the stock 4-op chain runs.

**The honest caveat, straight from the tree:** kernel-level *speed* was modelled, not
measured. The `quant_w4a8` docstring computes weight-traffic savings and then states
its own base case: *"TPOT ≈ max(host, gpu); the host term is ~3.4 ms, so the base case
for this whole patch is no benefit, slightly worse TTFT. It must be A/B'd."* Round 2
proved the point in reverse: the GDN fused epilogue was **reverted after measurement**
— it cost ~0.25 ms/step of host time to save ~0.8 µs/step of HBM.

Round-2 kernels went NVRTC (runtime-compiled, per-op fallback ladder NVRTC → AOT →
stock): greedy argmax gated on a boot-time bit-exact parity check against
`torch.argmax`, MoE GEMV band tuning, rms-norm block-quant.

### 3c. The Rust ports (6 min)

#### The scheduler crate (`vtl-sched`, ~8,000 lines)

A **logic-preserving port** of vLLM v0.25.0's block pool, KV-cache manager, prefix
cache, and the `schedule()` decision loop — each module docstring cites the upstream
file:line it mirrors. The interesting engineering is at the boundary:

- **Only primitives cross the FFI.** Never a vLLM object; block IDs go through
  persistent Rust-owned numpy buffers that Python slices. Per-step decisions moved
  from a PyDict to a persistent numpy **arena**.
- **The prefix hash had to match bit-for-bit** — a hand-written pickle-protocol-5
  emitter + SHA-256, because a silently different key space zeroes an 82% hit rate.
  Unsupported hash algorithms error at construction instead of diverging.
- **`overflow-checks = true` in release**: an integer wrap in block accounting must be
  a catchable panic, not corrupted state the engine keeps serving from.
- **Refusal, not approximation**: a construction-time config gate hands back one
  logged reason and leaves stock vLLM in charge for anything outside the served
  config (LoRA, connectors, spec-decode, sliding windows...).
- **Speculative precompute with an undo journal**: a parked worker runs the *real*
  `schedule()` GIL-free between steps, mutating the real state (a copy is useless —
  block IDs come out of exact free-queue pop order); on a miss the journal is
  reverse-applied bit-identically.

#### N-step burst decode (`nstep_decode`)

The host tax is per *engine step*, not per token — so emit **N tokens per step**.
The safety trick is the **align gate**: commit a burst only when
`num_computed % block_size + N ≤ block_size`. That one inequality keeps the whole
burst inside one KV block and one mamba state column: no new block allocation, baked
block tables stay valid, state indices stay correct. At block_size 16 and N=4 it
covers 13/16 of decode steps. The burst body runs with zero device syncs and zero
host reads, and an "in-graph ladder" captures progressively more of it into a single
CUDA-graph replay — down to the whole N-token burst as **one `replay()` call**, with
each capture failure demoting one rung and logging why.

Deliberately **nothing spec-decode-shaped is configured** — it reuses spec-decode's
multi-token plumbing with none of its config, because a spec-decode submission was
flagged as cheating.

#### The round-2 Rust CUDA-graph runner (M1–M4)

The centerpiece: move the decode-step **launch loop itself** out of Python.

- **The enabling fact:** torch 2.11 exposes `raw_cuda_graph_exec()` as a plain int,
  and vLLM never re-instantiates a decode graph. So Python keeps boot, capture, and
  prefill; Rust replays via dlopen'd libcuda — `cuGraphLaunch`, a pre-planned pinned
  D2H, and the step commit.
- **M1 taught: export the right graphs.** The stock forward-only graph never writes
  the accumulator the D2H reads — the runner needs nstep's unroll graphs, and a shape
  without one gets a NULL row so the runner *declines* instead of replaying stale data.
- **M2 discovered "hazard 8".** Async scheduling runs a depth-2 queue
  (`schedule(k) → sample(k) → update(k-1)`); committing at sample time can append
  step k's tokens **ahead of** step k-1's — scrambled output with nothing to crash.
  The interim interlock meant the runner **armed, self-checked, logged healthy, and
  never engaged**.
- **M4: the continuation graph.** Relaunching the unroll graph back-to-back silently
  re-emits launch 1's first token (its prologue reads a buffer only stock replay
  refreshes). Fix: same graph bodies, no prologue, feeding off the previous launch's
  own argmax — **shadow-verified with TWO launches**, because a single-replay shadow
  structurally cannot see back-to-back staleness.
- **Update-time commit** resolved hazard 8: `launch()` at sample time only enqueues;
  `commit()` runs inside `update_from_output`, where steps ARE ordered — behind a
  2-slot pinned ring with ownership re-checked at commit time (a request can finish
  and its slot be reissued between sample and update).
- **Multi-step residency**: 8 back-to-back graph launches, **one D2H, one event,
  one sync** — the GPU stays resident across 8 tokens while Python does nothing.

Getting it to actually run took as long as building it: a boot-ordering rendezvous
(capture runs before the Scheduler exists), a classify-priming fix (stock populates a
field only on the first *real* forward, and capture only does dummy ones — so the
whole burst ladder silently bailed every boot), and several review passes (§5).

---

## 4. Results (2 min)

| Round | Score trajectory | Notes |
|---|---|---|
| 1.1 | frontier **~62.5** | at `--max-num-seqs=16`; TTFT p95 ~14 s (18.7k-tok prompts) |
| 1.2 | **71.5 → 72.5 → 73.91 → 74.03 → 74.55** | TTFT p50/p95 down to 31/44 ms; TBT stuck at 3 ms; `failed=8` throughout |
| 2 | **89.86**, then a cap experiment at **83.61** | 900 s aiperf runs, same seed |

**Speaker notes:**

- Round 1.2's uncomfortable pair of facts: TBT stayed at 3 ms across the last three
  submissions, which the trace math says means **the N-step burst was likely dormant
  on the judge box** — and the single largest known scoring item was never a kernel:
  8 failed requests ≈ **+1.4 points**, root-caused to the int4 lm_head emitting a
  step-0 EOS (empty stream = request scores 0). A one-line HTTP keep-alive fix
  (60 → 600 s) sat documented and unapplied.
- Round 2's scored A/B is a good closing exhibit: capping mixed prefill at 1 improved
  TTFT p95 by 27%, but **`n_scored` dropped 165 → 88** — by Little's law, mean request
  latency went 27 → 51 s. The doc splits the regression into a fixable gate bug (the
  cap accidentally disabled the burst) and an irreducible cost (one 7.8k-token prefill
  becomes three steps a decode row waits behind). Scoring is about *finishing
  requests*, not about any single latency metric.

---

## 5. Lessons (5 min)

1. **Silent fallback is the enemy.** A 9k-line Rust component that switches itself
   off looks *identical in every latency number* to one that ran and didn't help —
   an A/B can measure the Python path and report a win. Hence `*_REQUIRE=1` flags
   that turn silent refusal into boot failure (bench/CI only — in a submission,
   serving-but-slower beats not serving), "**ARMED IS NOT ENGAGED**" written into the
   compose, fusion passes that must report `Replaced N patterns` because N=0 looks
   exactly like success, and the proxy-built "gutted image" rule.

2. **Size the prize in score points before writing code.** 1 ms of TPOT was worth
   ~8.6 points in round 1.2 and ~1 point in round 2 — the same optimization changed
   value ~10× when the band changed. The megakernel's honest prize was launch
   overhead only (~0.14 ms); the GDN epilogue fusion was reverted for costing 0.25 ms
   of host to save 0.8 µs of HBM. A wall-clock A/B beats a profiler's call counts.

3. **The biggest wins were often one line.** Dropping the mp executor backend,
   capturing batch-size 5, keep-alive 60→600, tokio workers=2, `empty_like(residual)`.
   Weeks of Rust competed with single compose lines — and the compose lines usually won
   on ROI.

4. **"Yesterday it worked" is usually a statement about coverage.** NSTEP "worked"
   the day before only because batch 5 had no captured graph, so the feature
   degenerated to a no-op; adding one capture size exercised the whole ladder for the
   first time — and the hang lived there. Relatedly: 17% of scored tokens decoded
   outside the captured graph set. Know what your feature *actually ran on*.

5. **The FFI boundary is where the bugs live.** A PyO3 `&self` method holds the
   pycell borrow across `allow_threads` — releasing the GIL does not release the
   borrow (the "kick borrow race": fail-open, one log line, whole subsystem silently
   degraded for the boot). An ownership check and a pack under different locks is a
   TOCTOU. Handing numpy *views* of persistent buffers to async consumers while a
   D2H is in flight corrupts counts. Every one of these was found by a dedicated
   adversarial review pass, not by tests.

6. **Debugging a wedged GPU: the stack frames are bystanders.** Three hang dumps
   blamed three different Python lines — all allocation sites, i.e. exactly where a
   host parks when the device won't drain. Stack dumps structurally cannot name
   enqueued-GPU-work bugs, so the fix was a **flight recorder**: a 48-entry ring of
   scheduler decisions, block IDs verbatim (a span can't tell you if a freed block is
   still read), and a params header emitted from the very dict handed to the crate so
   the header cannot drift from reality.

7. **Numerics and metrics are part of the contract.** The fp8 kernels preserve
   stock's double-rounding on purpose — "more accurate" changes the per-token scale
   and is a different kernel. Per-token SSE streaming had to ship with the burst or
   the grader would report TPOT N× worse. Optimize the thing being measured, and
   don't change what the measurement means.

8. **Ship everything behind a flag with a one-line revert, and keep the ledger.**
   The compose file *is* the engineering log: every knob carries its rationale,
   measurement status (including "UNPROVEN ON THIS BOX"), and revert. Shadow-mode
   twins (Python authoritative, Rust diffed) made an 8k-line port safe to land
   incrementally — and were deleted once the port was proven. The nulls and the
   reverts were recorded as carefully as the wins; that's what made the next
   decision cheap.

---

## Sources (all in this repo)

- `round-1.2/HANDOFF.md`, `round-2/HANDOFF.md` §6 — mission briefs, ERS math, constraints
- `round-2/RUST-RUNNER.md` — runner design + the 8 hazards
- `VTC-Go-README.md` — round-2 implementation summary
- `round-1.2/docker-compose.yaml`, `round-1.1/docker-compose-optimized.yaml` — the annotated flag ledgers
- `docs/round-1.2-latency-optimization-plan.md`, `docs/round-1.2-nstep-r8-microopt-plan.md`, `round-1.2/docs/round-1.2-port-frontier-3.md` — score-point economics per operating point
- `docs/round-2-nstep-regression-investigation.md` — the flight recorder, the wedge, the scored cap=1 A/B
- `round-1.2/bench/_ci_report.py`, `round-2/bench/_ci_report.py` — the ERS reference implementations
