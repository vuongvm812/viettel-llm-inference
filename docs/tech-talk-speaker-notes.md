# Speaker notes — "From `docker compose up` to a Rust CUDA-graph runner"

Companion to [`tech-talk-hackathon-rounds.md`](tech-talk-hackathon-rounds.md). Keyed to its
block numbers `01`–`21`. The script is the spine; this file is everything you might need to
say out loud or answer in Q&A. Numbers here are fact-checked against the repo — where a claim
is an estimate, a projection, or an inference, it says so.

---

## Open — the missing 2 ms (blocks 01–03, 1.5 min)

### 01 · Title

- One repo, 388 commits, 2026-07-04 → 2026-08-26. Three rounds, three unrelated problems:
  a prefill-bound trace and a latency-band chat workload (both on an 18 GB H200 MIG slice),
  then a 122B MoE agent workload on a full H200.
- The talk is a ladder: flags → allocator/env tuning → targeted patches → custom kernels →
  Rust ports → moving the CUDA-graph launch loop itself out of Python.

### 02 · The false start

- The first ~2 weeks built a from-scratch Rust inference engine: llama.cpp FFI, a
  3-pinned-core LMAX-Disruptor pipeline, continuous batching, shared-prefix KV, PGO/LTO/BOLT.
- Killed in one commit — `60c44cd "remove custom rust inference"` — and replaced with tuning
  the stock `vllm/vllm-openai` image.
- The lesson to state now and pay off in block 21: matching a mature engine's whole feature
  surface is not a two-week project. The move was to make vLLM fast, not to replace it.

### 03 · The question

- On the round-1.2 MIG slice, observed TPOT was ~3 ms/token. The GPU floor was ~1 ms/step
  (0.6 GB of W4 weights at ~600 GB/s on the slice). So ~2 ms/token was host-side Python.
- Do not answer it yet. Block 10 answers it. Everything between is the setup.

---

## The game (blocks 04–06, 3 min)

### 04 · The contract

- The judge pulls **a registry image + our `docker-compose.yaml`**, mounts the model at
  `/model`, and scrapes an OpenAI-compatible server on `:8000`. No build context, no local
  mounts. Entrypoint, model path and served-model-name lines are locked.
- Submissions are live-endpoint registrations (`airace endpoint --task llm --url ...`) against
  a finite attempt budget. Registering a dead endpoint burns an attempt; if the server dies
  mid-grading-run, that attempt scores 0.
- Round 2's logistics twist: one VM per team behind a jump host, **interactive SSH only** (no
  scp/rsync/tunnels), internet only through a proxy.
- The line worth quoting verbatim, from `docs/ci/README.md`: our Dockerfile's network fetches
  are all deliberately non-fatal, so behind a whitelist proxy a VM build does not fail — it
  *silently produces a gutted image* (no Rust scheduler wheel, no `vtl._C_w4a8`, possibly
  glibc malloc) that benches worse for reasons nobody would trace back to the network.
  **"So the VM never builds. It pulls."** Images are built in GitHub CI, pushed to Docker Hub,
  pulled by digest.

### 05 · The scoreboard

- ERS: `S_request = w·s_ttft + (1−w)·s_tpot`, each component
  `clamp((Ceiling − x)/(Ceiling − Floor), 0, 1)^γ`, averaged over **all** requests — a failed
  request scores 0 but stays in the denominator. γ=2 punishes tails quadratically.
- Round 1.2: TTFT 10/400 ms, TPOT **1/10 ms**, γ=2, w=0.5. Round 2: TTFT 200/6000 ms,
  TPOT **8/100 ms** (`round-2/HANDOFF.md:574-575`).
- Round 1.2's narrow band is what drove the whole custom-code arc: 1 ms of TPOT was worth
  ~28–37 ms of TTFT, ≈ 8.6 score points at our operating point.
- The 28× vs 37× spread is **not** a contradiction — γ=2 makes the marginal gradient
  operating-point-dependent, and the two figures were computed at different points in the
  optimization timeline (`docs/round-1.2-latency-optimization-plan.md:5` vs
  `docs/round-1.2-hotpath-batch-plan.md:5`). Say "roughly 30×" and move on unless asked.
- Round 2 flipped the economics: the band widened ~10×, so dERS/dTPOT ≈ 0.011/ms. Sub-ms
  micro-optimization became near-noise; request completion and tails were what mattered.

### 06 · Three rounds, three bottlenecks

| | Round 1.1 | Round 1.2 | Round 2 |
|---|---|---|---|
| Model | Qwen3.5-2B (VL hybrid, GDN linear attention) | LFM2.5-1.2B (10 short-conv + 6 GQA layers) | Qwen3.5-122B-A10B-FP8 (MoE 256-expert top-8, 36 GDN + 12 full-attn) |
| Hardware | H200 MIG 1g.18gb — 18 GB, 16 SMs, 3 vCPU, 8 GB RAM | same slice | full H200 141 GB, 22-core Xeon |
| Workload | 120 reqs, p50 prompt 18.7k tok, **101:1 prefill-bound**, 82.4% prefix hit | 70 convs × 6 turns = 420 reqs, ~4.7k ctx, ~82% prefix hit | aiperf `inferencex-agentx-mvp` on the SemiAnalysis Weka corpus — real multi-turn Claude Code sessions, 900 s, concurrency 5, ctx 204,800, hidden seed |
| Bottleneck | prefill throughput + prefix cache | host overhead per decode step | finishing requests; TTFT tail; MoE bandwidth |

---

## vLLM in one slide (block 07, 2.5 min)

Full primer: [`vllm-architecture-primer.md`](vllm-architecture-primer.md), which follows the
vLLM team's ["Anatomy of vLLM"](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm).

- vLLM runs as **two processes**: an OpenAI-compatible frontend (parse, tokenize, stream,
  detokenize) talking over ZMQ to the **EngineCore** — a loop where the **scheduler** picks
  which requests join the next *step* (one forward pass; continuous batching means requests
  join and leave every step), the **KV-cache manager** maps their tokens to paged KV blocks
  (which is what makes prefix caching and chunked prefill cheap), and the **GPU model runner**
  builds the step's inputs in Python and launches the forward, replaying pre-captured CUDA
  graphs to cut launch overhead.
- Prefill sets **TTFT**, decode sets **TPOT** — the two numbers ERS scores — and every
  per-step host cost the runner pays in Python lands directly on TPOT.
- Why the map matters: every optimization in this talk lives on one of those boxes. Blocks
  08–09 tune the scheduler's and KV manager's knobs; 11–15 attack the forward pass; 16–18
  replace the frontend, the scheduler, and finally the CUDA-graph launch loop itself.

---

## Rung 1 — flags only (blocks 08–09, 3.5 min)

### 08 · The flags that mattered

All of this is flags and env vars on the stock image — no code.

- **`--enable-prefix-caching`** — "the single biggest win, and it is lossless." The round-1.1
  trace re-sends a byte-identical 6,388-token system prompt; block-level hit rate 82.4%,
  eliminating 1.99M of 2.41M prefill tokens. The hit rate is 82.4% at *every* block size
  1→64, so we did **not** tune block_size or build a radix tree — the residual was 0.03%.
- **`--max-model-len` sized to the trace, not the model**: longest prompt 27,331 tok → 32768,
  not 262144 (won't boot) and not the baseline 16384 (rejects most of the trace).
- **`--max-num-seqs` swept 256 → 16**, verdict written into the compose
  (`round-1.1/docker-compose.yaml:42`): *"Pareto optimum: 12 cut tbt 18→17 ms but blew
  ttft_p95 14k→20k (−0.3). 20 raised tbt. Frontier is ~62.5 here."*
- **fp8 weights + `--kv-cache-dtype=fp8_e4m3`**: prefill is GEMM-bound and 99% of tokens are
  prefill; Hopper fp8 tensor cores act on all of them.
- **Chunked prefill + `--max-num-batched-tokens=8192`**: small chunks stop a new turn's
  prefill from spiking in-flight decode latency, which is exactly the tail γ=2 punishes.
- Image-level tuning worth one anecdote each if you have room: jemalloc tuned for latency
  (`dirty_decay_ms:-1` — never return pages, no purge syscalls on the alloc path; known cost
  is RSS only grows, against an 8 GB cap with no swap); `OMP_NUM_THREADS=3` +
  `OMP_WAIT_POLICY=PASSIVE`, because at TP=1 vLLM's UniProcExecutor never sizes thread pools
  to the cgroup quota and torch sees the host's 64+ cores; warm `torch.compile`/Triton/AOT
  caches baked into the image; the healthcheck doubling as prefix-cache warm-up so readiness
  and a hot cache arrive together; and the crowd-pleaser — CPython's default GIL switch
  interval is 5 ms, longer than an entire decode step, so `sys.setswitchinterval(0.0002)`.

### 09 · The ledger and its nulls

- **3 boots per arm minimum.** Boot-to-boot noise floor ~0.5 ms TPOT, so a knob has to clear
  ~0.22 ms to prove itself.
- Nulls stayed in the ledger: `--max-num-scheduled-tokens` 2048 vs 8192 measured **null and
  directionally backwards** vs prediction; `use_inductor_graph_partition` measured **worse**
  and cleared the noise floor — knob deleted.
- Measured and rejected: **KV offload** to CPU/NVMe (the repo's "8× headroom" numbers were
  measured on the team's full-H200 dev box; on the 18 GB judge slice the honest version is
  that the fp8 working set of 7.8 GB still fits the ~13 GB KV budget, and the miss path is
  ~700× slower than HBM either way); **speculative decoding** (decode is ~1% of tokens in a
  101:1 trace — measured, not assumed — and the ngram variant was banned as cheating);
  `--async-scheduling` passed explicitly (already the default; passing it only converts
  warnings into boot failures).
- The number worth stating: **17% of our round-1.2 scored tokens decoded outside the captured
  CUDA-graph set**, silently falling back to eager. Shapes without a captured graph are a
  performance cliff you will not see unless you look.
- `docker-compose.yaml` became the engineering log: every flag carries its rationale, its
  measurement status, and a one-line revert.

---

## Rung 2 — the host is the TPOT (blocks 10–13, 4.5 min)

### 10 · The answer to block 03

- Decode on the slice: ~1 ms GPU + ~2 ms host per step. Under async scheduling
  TPOT ≈ max(host, gpu) — **the host IS the TPOT.** Everything in this rung and the next two
  attacks either host time or memory bandwidth.
- An earlier, more pessimistic host estimate of ~3.4 ms appears in
  `round-1.2/vtl/patches/quant_w4a8.py:20-21`; it predates the later host-side work. If asked,
  the honest answer is that the host term was measured at different points and shrank.

### 11 · Not a fork

- Not a real fork — a **`patch -p1` overlay onto stock v0.25.0 site-packages**, compiled
  `.so`s untouched, with a version assert so a base-image bump fails the build loudly.
- Everything that *could* be a runtime monkey-patch is one: a plugin with ~30
  `VTL_ENABLE_*`-gated modules, and `VTL_DISABLE=1` gives a provably stock engine.
- Patches exist only where monkey-patching is structurally impossible:
  - `short_conv.patch` — the short-conv decode body runs inside an opaque custom op; no Python
    seam to wrap. Rewired to call our fused kernels.
  - `lfm2.patch` — one line; see block 12.
  - `hotpath_microopt.patch` — line-level edits inside per-step hot loops (dead kernel
    launches, contextmanager allocations, identity caches). No function boundary to wrap.
  - `api_server_rust_frontend.patch` — the `if __name__ == "__main__"` block isn't importable,
    therefore isn't patchable at runtime.

### 12 · The one-line win

- `lfm2.patch` is a single line: `torch.empty_like(residual)` instead of
  `empty_like(hidden_states)`. The wrong allocation source gave the FX node a user outside the
  fusion pattern, so **the RMSNorm+quant fusion silently never fired on 10 of 16 layers**.
- **Do not claim a millisecond figure.** No measured before/after latency delta for this fix
  exists in the repo — only the mechanism plus a modelled ~671 MB of avoidable prefill traffic
  at 8192 tokens (`round-1.2/vtl/patches/shortconv_quant.py:22-27`). The story is "one line
  unblocked a whole fusion pass," not "one line bought X ms."

### 13 · The Rust frontend

- vLLM's own `vllm-rs`, with source patches before `cargo build`: sonic-rs (SIMD JSON) request
  parsing borrowing the body bytes; **iceoryx2 shared-memory IPC** replacing ZMQ on the data
  plane (ZMQ kept for handshake and as the degrade path), with golden byte-frames pinned in
  both Rust and Python so layout drift fails the image build; and **per-token SSE streaming**.
- Per-token SSE is mandatory, not an optimization: once the engine emits N tokens per record, a
  chunk-counting grader reports TPOT **N× worse than reality**.
- PGO where the mock engine is paced to production cadence — PGO trained at memory speed ships
  a *slower* binary.
- **The one knob swept against the real judge**: worker threads at 1/2/3 scored
  **71.5 / 72.5 / 72.1** — 1 starves the request path, 3 pays contention on a 3-core budget.
  Two corrections to the old draft: these are three arms of **one sweep**, not a score
  trajectory, and the swept variables are `VLLM_RS_ZMQ_WORKER_THREADS` /
  `VLLM_RS_REQUEST_WORKER_THREADS` (`round-1.2/docker-compose.yaml:170-175`) — *not*
  `TOKIO_WORKER_THREADS`, which is pinned separately at 3 and sizes only the HTTP runtime.

---

## Rung 3 — kernels (blocks 14–15, 2.5 min)

### 14 · The rules

Eight kernels for round 1.2 (`vtl/csrc/`), all sharing three design rules:

1. Numerics match stock element-for-element — deliberately including double-rounding, because
   "more accurate" shifts the fp8 amax and per-token scale: *that is a different kernel, not a
   better one.*
2. Every kernel has a `*_supported()` gate that **refuses** anything outside the served layout
   rather than approximating it.
3. Parity tests against the stock op chain, not just golden values.

The ladder, in increasing ambition — one sentence each, do not enumerate live:

- Fused elementwise+quant (`rms_norm_quant`, `silu_mul_quant`, `mul_quant`,
  `dynamic_per_token_quant`): kill bf16 intermediates' HBM round-trips and a launch each.
  Stock RMSNorm+quant read the input three times; ours once.
- `bcx_conv_gate_quant` — one launch replacing three per conv layer × 10 layers, with `Bx`
  never leaving registers.
- `conv_align_fused` — stock ran two full-grid Triton launches *outside the CUDA graph, every
  step*, to compute six scalars and then fast-exit 15 of 16 times. Fused to one launch.
- W4A8 CUTLASS extensions — vLLM's 10 GEMM instantiations and heuristic were tuned on a
  132-SM Hopper; on a 16-SM slice the wave structure is wrong (1.5 ragged waves, 64-deep
  k-loops). Added Stream-K, small-tile and TMA-multicast arms.
- The short-conv decode megakernel (`shortconv_decode_mega.cu`, 504 lines): the entire decode
  block — RMSNorm+quant → W4A8 GEMV → conv+gate+quant → W4A8 GEMV — in **one launch with three
  hand-rolled device-wide sense-reversing barriers**. Co-residency of all blocks is a
  *correctness precondition* (a non-resident block hangs the device), so the grid comes from
  the occupancy API on the real kernel, and below the minimum the Python gate refuses and the
  stock 4-op chain runs.
- Round-2 kernels went NVRTC (runtime-compiled, per-op fallback ladder NVRTC → AOT → stock):
  greedy argmax gated on a boot-time bit-exact parity check against `torch.argmax`, MoE GEMV
  band tuning, rms-norm block-quant.

### 15 · The one we didn't build, the one we reverted

This is the honest half of the kernel story. **Kernel-level speed was modelled, not measured.**
The `quant_w4a8` docstring computes weight-traffic savings and then states its own base case:
*"TPOT ≈ max(host, gpu); the host term is ~3.4 ms, so the base case for this whole patch is no
benefit, slightly worse TTFT. It must be A/B'd."*

- **The one we didn't build.** `round-2/docs/improvements/gdn-decode-megakernel.md` is a
  design doc that argues *against itself*: intermediate HBM traffic is only 0.5% of one GDN
  layer's 133 MB, so collapsing 180 launches to 36 is worth ~0.14 ms — about 0.0012 ERS — at
  best, "at a plausible g ≈ 1 µs." It gates the build behind three measured go/no-go criteria
  via `megakernel_probe.py`, and the criteria never cleared. Economics first, then build.
  (The old draft of this talk misattributed this 0.14 ms to the round-1.2 short-conv
  megakernel as a measured result. It is a round-2 projection for a kernel that was never
  built. Do not repeat that.)
- **The one we reverted.** Round 2's GDN fused epilogue was reverted after measurement
  (`a22c31d`/`fac50cd`): it cost ~0.25 ms/step of host time to save ~0.8 µs/step of HBM —
  `round-2/docs/improvements/gdn-epilogue-fusion.md:46-51` calls it *"a 300× loss."* Label it
  as round 2 when you say it; it is not a round-1.2 number.

---

## Rung 4 — the launch loop leaves Python (blocks 16–18, 4.5 min)

### 16 · N-step burst and the align gate

- The host tax is per *engine step*, not per token — so emit **N tokens per step**.
- The safety trick is the **align gate**: commit a burst only when
  `num_computed % block_size + N ≤ block_size`. That one inequality keeps the whole burst
  inside one KV block and one mamba state column: no new block allocation, baked block tables
  stay valid, state indices stay correct. At block_size 16 and N=4 it covers 13/16 of steps.
- The burst body runs with zero device syncs and zero host reads, and an "in-graph ladder"
  captures progressively more of it into a single CUDA-graph replay — down to the whole
  N-token burst as **one `replay()` call** — each capture failure demoting one rung and
  logging why.
- Deliberately **nothing spec-decode-shaped is configured**: it reuses spec-decode's
  multi-token plumbing with none of its config, because a spec-decode submission was flagged
  as cheating.
- The Rust scheduler crate behind it (`vtl-sched`, ~8,000 lines) is a **logic-preserving port**
  of vLLM v0.25.0's block pool, KV-cache manager, prefix cache and `schedule()` loop, each
  module docstring citing the upstream file:line it mirrors. Boundary details worth having:
  only primitives cross the FFI (block IDs move through persistent Rust-owned numpy buffers
  Python slices, and per-step decisions moved from a PyDict to a persistent arena); the prefix
  hash had to match **bit-for-bit**, so a hand-written pickle-protocol-5 emitter + SHA-256,
  because a silently different key space zeroes an 82% hit rate; `overflow-checks = true` in
  release, so an integer wrap in block accounting is a catchable panic rather than corrupted
  state the engine keeps serving from; a construction-time config gate that **refuses** with
  one logged reason and leaves stock vLLM in charge for anything outside the served config
  (LoRA, connectors, spec-decode, sliding windows); and speculative precompute with an undo
  journal — a parked worker runs the *real* `schedule()` GIL-free between steps, mutating real
  state (a copy is useless, block IDs come out of exact free-queue pop order), and on a miss
  the journal is reverse-applied bit-identically.

### 17 · The runner

- **The enabling fact:** torch 2.11 exposes `raw_cuda_graph_exec()` as a plain int, and vLLM
  never re-instantiates a decode graph. So Python keeps boot, capture and prefill; Rust
  replays via dlopen'd libcuda — `cuGraphLaunch`, a pre-planned pinned D2H, and the commit.
- **M1 taught: export the right graphs.** The stock forward-only graph never writes the
  accumulator the D2H reads, so the runner needs nstep's unroll graphs; a shape without one
  gets a NULL row and the runner *declines* rather than replaying stale data.
- **M4: the continuation graph.** Relaunching the unroll graph back-to-back silently re-emits
  launch 1's first token, because its prologue reads a buffer only stock replay refreshes. Fix:
  same graph bodies, no prologue, feeding off the previous launch's own argmax — and
  **shadow-verified with TWO launches**, because a single-replay shadow structurally cannot
  see back-to-back staleness.
- **Multi-step residency**: `VTL_RUST_RUNNER_STEPS=8` is live in the submission compose
  (`round-2/docker-compose.yaml:549`) — 8 back-to-back graph launches, one D2H, one event, one
  sync. The GPU stays resident across 8 tokens while Python does nothing.
- Getting it to run took as long as building it: a boot-ordering rendezvous (capture runs
  before the Scheduler exists), and a classify-priming fix — stock populates a field only on
  the first *real* forward, and capture only does dummy ones, so the whole burst ladder
  silently bailed every boot.

### 18 · Hazard 8

- `round-2/RUST-RUNNER.md:105-157` lists all eight hazards. Number 8 is the one to tell.
- Async scheduling runs a depth-2 queue: `schedule(k) → sample(k) → update(k-1)`. Committing
  at sample time can append step k's tokens **ahead of** step k−1's — scrambled output, with
  nothing to crash.
- The interim interlock meant the runner **armed, self-checked, logged healthy, and never
  engaged**. That is the shape of the whole lesson in block 20.
- Fix: **update-time commit.** `launch()` at sample time only enqueues; `commit()` runs inside
  `update_from_output`, where steps *are* ordered — behind a 2-slot pinned ring with ownership
  re-checked at commit time, because a request can finish and its slot be reissued between
  sample and update.

---

## What actually scored (blocks 19–20, 3 min)

### 19 · The payoff slide

Round 2's score history, reconstructed from git (`5f1aeb5`, `3416eac`, `f65376d`) and
`docs/round-2-nstep-regression-investigation.md:874`:

| Score | What was actually running |
|---|---|
| 81.62 | the vanilla submission |
| **89.26** | still `image: vllm/vllm-openai:v0.25.0` — **stock**. The whole diff vs 81.62 is `VLLM_USE_RUST_FRONTEND: "1"` (vLLM's own flag) plus `--quantization=fp8` |
| 89.86 | the full custom stack — kernels, the `vtl-sched` port, the N-step burst, the Rust CUDA-graph runner |
| 83.61 | the cap=1 experiment (block 20 / below) |

**Two flags on an unmodified image bought +7.6 points. Everything we built after that bought
under +0.6.** If someone doubts it, `git show 3416eac:round-2/docker-compose-rust-peak-perf.yaml`
has no vtl wheel in it at all — and that commit's own message mislabels it "Rust submission
file peak," which is how the finding stayed buried.

Round 1.x for context: round 1.1 frontier ~62.5; round 1.2 ran 72.5 (best sweep arm) → 73.91 →
74.03 → 74.55, with TTFT p50/p95 down to 31/44 ms and TBT stuck at 3 ms. Note the scores are
not comparable across rounds — different models, hardware, bands and workloads.

Also worth telling here: **the largest single known scoring item in round 1.2 was never a
kernel.** 8 failed requests were estimated at ≈ +1.4 points, root-caused to the int4 lm_head
emitting a step-0 EOS (empty stream = the request scores 0,
`round-1.2/vtl/patches/step0_eos_ban.py:3-6`), and a one-line HTTP keep-alive fix (60 → 600 s,
for trace inter-turn gaps that tail past 30 s) sat documented and unapplied — still `"60"` in
`round-1.2/docker-compose.yaml:156` today. Two honesty caveats: `failed=8` is evidenced for the
last three scored submissions, not all five; and the +1.4 is the docs' own estimate, never
confirmed by a clean A/B, since failures persisted at 74.55 even after the EOS ban shipped in
`8015b81`.

### 20 · What we cannot prove

The uncomfortable pair, framed as a measurement limit rather than a verdict:

- **Round 1.2:** TBT stayed at 3 ms across the last three submissions. The trace-verified burst
  math says a working N=4 burst puts TBT at ~1.8 ms, so the burst was probably not engaging on
  the judge box — but `round-1.2/docs/round-1.2-next-ports-plan.md:3-8` is explicit that this
  is **inference by elimination**, not a measured root cause, and engagement counters had been
  removed in `741d7e6`, so engagement was only inferable from TBT at all.
- **Round 2:** the compose ships `VTL_RUST_RUNNER_REQUIRE: "0"`
  (`round-2/docker-compose.yaml:569`), which permits a silent fallback to Python that is
  invisible in ERS numbers. No in-repo log confirms the runner was live during a scored run.
  The repo says why the flag exists, and it is the best line in the whole talk:
  **"a runner that quietly never armed looks exactly like one that ran and did not help."**
- So the honest close on attribution: one 900 s run on a hidden seed cannot resolve 0.6 points,
  we had no judge-box logs, and we had removed or defaulted-off the instrumentation that would
  have told us. Vuong was also remote for round 2, which is a fine place for the joke and a
  real reason the on-box debugging loop was slow.

**The scored A/B that closes the section** — capping mixed prefill at 1 improved TTFT p95 by
27.0% (1888.3 → 1379.1 ms) but dropped `n_scored` 165 → 88, i.e. by Little's law mean request
latency went 27.3 → 51.1 s and the score went 89.86 → 83.61. The doc splits the regression into
a fixable gate bug (the cap accidentally disabled the burst, fixed in `da32c4b`) and an
irreducible cost (one 7.8k-token prefill becomes three steps a decode row waits behind).
**Scoring is about finishing requests, not about any single latency metric.**

---

## Close (block 21, 2 min)

1. **Map the architecture first.** Before touching code, find where the time actually goes:
   prefill sets TTFT, decode sets TPOT, and under async scheduling TPOT ≈ max(host, GPU) — so
   on our box the *host* was the TPOT and no kernel could fix that. The best wins fell out of
   the map, not the profiler: drop the mp executor, capture the missing batch size, fix one
   `empty_like` line.
2. **Measure on the real box.** Our dev-box numbers were tuned for 132 SMs; the judge ran 16.
   Small-call profiling overstates dispatch overhead, so a wall-clock A/B on the target beats a
   profiler's call counts — and it needs discipline: 3 boots per arm, know your noise floor
   (~0.5 ms), keep the nulls. Predicted wins measured backwards more than once.
3. **Earn the right to rewrite.** Climb in order: config flags → allocator/env tuning →
   targeted patches → and only then kernels, Rust ports, a CUDA-graph runner. We learned this
   twice: once by deleting a two-week engine (block 02), and once by measuring that two flags
   on the stock image outscored the entire custom stack (block 19). Weeks of Rust competed with
   single compose lines, and the compose lines usually won on ROI.
4. **Make engagement provable, or your wins are unfalsifiable.** This is the one we would fix
   first next time. We removed the burst's engagement counters and shipped the runner with
   `REQUIRE=0`, so two centerpiece systems cannot be shown to have run in any scored
   submission. Instrumentation that proves a fast path *engaged* is not optional telemetry —
   without it a win and a no-op are the same observation.

If there is time, the tactical tier lives in the sources: non-blocking beats locking (on a
3-vCPU budget one spin loop starved the IPC threads; park on events, `OMP_WAIT_POLICY=PASSIVE`
for the same reason; where locks were unavoidable, allocations and evictions moved outside the
mutex so the critical section is one hash insert); memory utilization is a first-class axis
(jemalloc tuned for latency, stack and registers over heap and HBM, zero-copy views instead of
~4,400 boxed PyLongs per request, persistent buffers instead of per-step allocation — most
"CPU" overhead was allocation and copying in disguise); and Rust buys speed of iteration, not
exemption from concurrency review (the compiler does not see a pycell borrow held across a GIL
release, a TOCTOU across two locks, or a view of a buffer with a D2H still in flight).

---

## Q&A ammunition

- **The GDN-batch wedge.** `docs/round-2-nstep-regression-investigation.md:231-806` is a
  multi-day forensic arc: a device-side hang chased through six passes of hypothesis
  elimination (bookkeeping, over-admission, mamba split, GDN metadata, kernel-by-kernel
  exoneration), a purpose-built `FLIGHT` ring-buffer flight recorder, and a root cause of three
  concurrent GDN prefills in one mixed batch. The best concrete answer to "how do you debug on
  hardware you can barely reach."
- **The cap=1 timing caveat.** The gate fix `da32c4b` (2026-08-19 05:00 UTC) precedes the
  scored measurement note (11:34) in git order, yet the doc says the measured run lacked the
  fix. The explanation is that the scored run used a pinned image digest built before the fix
  was committed. Worth stating if anyone reads the timestamps.
- **Why not just use a bigger slice / TP>1?** Round hardware was a single GPU or slice; tensor
  parallelism and disaggregated prefill/decode were never on the table.
- **Why not spec-decode?** The ngram/prompt-lookup variant is output-identical at temp 0, which
  is exactly why the judges banned it. The N-step burst reuses the multi-token plumbing and
  configures nothing spec-decode-shaped.

## Sources

`round-1.2/HANDOFF.md`, `round-2/HANDOFF.md` (mission briefs, ERS math, constraints) ·
`round-2/RUST-RUNNER.md` (runner design, all 8 hazards) · `VTC-Go-README.md` (round-2
implementation summary) · `round-1.2/docker-compose.yaml`,
`round-1.1/docker-compose-optimized.yaml` (the annotated flag ledgers) ·
`docs/round-1.2-latency-optimization-plan.md`, `docs/round-1.2-nstep-r8-microopt-plan.md`,
`round-1.2/docs/round-1.2-port-frontier-3.md` (score-point economics per operating point) ·
`docs/round-2-nstep-regression-investigation.md` (flight recorder, the wedge, the cap=1 A/B) ·
`docs/ci/README.md` (build in the cloud, run on the VM) · `round-1.2/bench/_ci_report.py`,
`round-2/bench/_ci_report.py` (the ERS reference implementations) ·
[`vllm-architecture-primer.md`](vllm-architecture-primer.md) and
["Anatomy of vLLM"](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm).
