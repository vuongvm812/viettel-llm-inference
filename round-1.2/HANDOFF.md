# HANDOFF — Round 1.2 Mission Brief

## BLOCKER: none of the 2026-07-27 work is in the pinned image

`docker-compose.yaml:6` pins `unseenablefuture/awesome-badger@sha256:c1ade3ac…`, built before any
of it. The vLLM changes reach the image ONLY through `Dockerfile.vllm-fork`'s patch step, and
there are now 15 patches where that image has 10. The judge runs `docker-compose.yaml` alone —
`docker-compose-optimized.yaml` is a local overlay. Until `make vllm-fork PUSH=1` runs and BOTH
digests are re-pinned (`VLLM_FORK_DIGEST` in `Makefile`, the image in `docker-compose.yaml`),
everything below is a no-op on the scored run.

## Drafter latency work, 2026-07-27 — LANDED, NONE OF IT BOOTED YET

Six changes aimed at making the LFM2.5-350M draft arm cheap enough to be worth its 4 model
invocations per decode step. Every one is behind a `VTL_*` env var, all default ON, all set
explicitly in `docker-compose.yaml`. **Nothing here has been compiled or booted** — there is no
GPU, torch, pytest or nvcc on the dev box. `bash vtl/vllm_patches/gen.sh` passes and the
CPU-runnable self-checks pass; that is the whole of the evidence so far.

**The headline finding, which nobody had noticed:** `bcx_conv_gate_quant` — the flagship fused
short-conv decode kernel — has been **dead code since `num_speculative_tokens` went non-zero**.
Spec decode widens the conv state to `conv_kernel - 1 + num_spec` = 5, and the kernel's shape
predicate demanded `state_len == 2` exactly, so it silently rejected *both* models.
`bench/test_bcx_conv_gate_quant.py:224` was asserting that rejection as if it were correct.

| # | Change | Flag (default) | Where |
|---|---|---|---|
| 1 | `supported_for` takes `state_len >= kStateLen`. The kernel only ever reads the first `width-1` slots, and `causal_conv1d_update` pins `state_len = width-1` on the host whenever `num_accepted_tokens is None` (`causal_conv1d.py:1181-1184`), so the wider allocation was always inert on this path. Restores the fused kernel for the **drafter**. | `VTL_ENABLE_BCX_CONV_GATE=1` | `vtl/csrc/bcx_conv_gate_quant.cu` |
| 2 | `kSpec` template: one block per REQUEST, per-request token loop, taps at `num_accepted-1`. Restores it for the **target** under spec (~20 launches/step). New optional op args `num_accepted_tokens`/`query_start_loc`. | `VTL_BCX_SPEC=1` | same `.cu`, `torch_bindings.cpp`, `short_conv.py` |
| 3 | FR-Spec: prune the DRAFT lm_head to its first K token ids before quantize/pack. ~19% of the drafter's weights, read 3x per target step. Lossless — the target verifies every drafted token, so a token outside the window is simply never proposed. | `VTL_DRAFT_VOCAB=16384` | `vtl/patches/lm_head_quant.py` |
| 4 | Drafter FULL cudagraphs for its uniform decode steps. It is pinned to PIECEWISE, which on a 16-layer hybrid splits at every conv AND attn op — ~17 replays per draft step against the target's 1. **INCOMPLETE, shipped OFF** — see below. | `VTL_DRAFT_FULL_CUDAGRAPH=0` | `llm_base_proposer.py`, `mamba_attn.py`, `gpu_model_runner.py` |
| 5 | Multi-verification-length FULL cudagraph keys. **NO schedule ships** — see below. | `VTL_DYNAMIC_SD_FULL_CG=1` (inert without a schedule) | `cudagraph_dispatcher.py`, `config/vllm.py`, `gpu_model_runner.py` |
| 6 | `make sweep-spec` answers "does K=3 pay?" by deriving a throwaway overlay from the submitted compose (one sed on the K literal), K ∈ {0..4}, K=0 = spec OFF. `--disable-log-stats` is stripped from the OVERLAY only. | — | `Makefile` |
| 7 | Fused int4 GEMV + argmax over the pruned draft head. Skips the `[rows, vocab]` logits tensor. Reads its OWN plain int4 packing, NOT the CUTLASS-reordered production weights — see below. | `VTL_DRAFT_FUSED_ARGMAX=1` | `vtl/csrc/draft_argmax.cu`, `lm_head_quant.py`, `llm_base_proposer.py` |
| 8 | PEARL: **installs nothing.** The rollback primitive (`_ConvSnapshot`, `_draft_conv_states`) is kept and tested; the overlap is not reachable safely from this seam — see below. | `VTL_ENABLE_PEARL=0` | `vtl/patches/pearl.py` |

**The dynamic-K schedule was pulled back out, and this is the trap to know about.** A review
caught that registering FULL cudagraph keys per verification length is only half the job: the
CAPTURE pass cannot produce them. `_warmup_and_capture` drops `desc.num_reqs` and `_dummy_run`
hardcodes `max_query_len = self.uniform_decode_query_len`, so every FULL capture lands on the
static length regardless of which descriptor asked for it. A K=0 step (batch 9-70) would then
dispatch FULL, find no captured graph, and `CUDAGraphWrapper` would raise **during serving** —
capturing is globally disabled by that point. That is a mid-run crash in the submitted config,
not a slow path.

So: no schedule in `docker-compose.yaml`, `uniform_decode_query_lens` is pinned to a single
entry, and `_maybe_override_dynamic_sd_cudagraph_mode` now only skips the stock PIECEWISE
downgrade when the schedule is DEGENERATE (every entry equal to the static K). Anyone adding a
real schedule gets the stock downgrade — correct, just slower — instead of a crash. Finishing it
properly means threading `desc.num_reqs` into `_dummy_run` with
`max_query_len = num_tokens // num_reqs` AND relaxing `mamba_attn`'s
`assert m.max_query_len == self.vtl_capture_query_len` for the target.

Also from that review, and worth not re-breaking: `--disable-log-stats` had to go BACK into the
submitted compose. `vtl/patches/kv_cache_manager.py:160` self-gates its prefill block-scan
elision on `log_stats` being off, so removing the flag puts a per-request block walk back inside
TTFT. `make sweep-spec` strips it from its derived overlay instead, which is where the
measurement belongs.

**Change 4 is deliberately OFF and is the one piece of unfinished work here.** Everything around
it is done — the dispatcher emits `FULL_AND_PIECEWISE` keys at `uniform_decode_query_len=1`, the
draft model gets its own `CUDAGraphWrapper(FULL)`, the descriptor is threaded into
`set_forward_context`, the mamba builders take a per-builder `vtl_capture_query_len`, and the
drafter joins the runner's FULL capture pass. The missing piece is the capture metadata:
`SpecDecodeBaseProposer.dummy_run` still enters `set_forward_context(None, ...)`, so no draft
builder's `build_for_cudagraph_capture` is ever called. PIECEWISE tolerates that (attention runs
between graph pieces); FULL does not, because attention is inside the graph. Turning the flag on
as-is would capture against absent metadata.

To finish: give `dummy_run` a capture path that builds a uniform-decode `CommonAttentionMetadata`
(`num_reqs = num_tokens`, query_len 1), calls `build_for_cudagraph_capture` on each entry of
`draft_attn_groups`, and passes the resulting per-layer dict to `set_forward_context` instead of
`None`. Then flip the default. Two ordering traps already handled and worth not re-breaking:
`initialize_cudagraph_keys` runs BEFORE `draft_attn_groups` exists (so the per-builder capture
length is applied at the end of the drafter's `initialize_attn_backend` instead), and the base
`_determine_batch_execution_and_padding` now returns a 4-tuple, which `step3p5.py` and
`dflash.py` also unpack.

**Why the fused argmax carries its own weight copy.** The production path packs through
`ops.cutlass_encode_and_reorder_int4b`, a CUTLASS-internal interleaved layout. Decoding it by
hand would fail SILENTLY: a wrong draft token is not wrong output — the target verifies every
one — so a mistake surfaces only as a worse acceptance rate, weeks later, blamed on the model.
So `draft_argmax.cu` defines its own plain row-major packing (8 signed int4 per int32, symmetric
per-group scales), `lm_head_quant.pack_draft_argmax_weight` builds it at load from the same
pre-pack bf16 rows, and `bench/test_draft_argmax.py` checks it against a pure-torch oracle
including `torch.argmax`'s lowest-index tie-breaking. Costs ~8 MB at K=16384; buys total
independence from a CUTLASS implementation detail. W4A16 not W4A8 on purpose: the activation is
a few KB, so quantizing it saves nothing and only adds error to the number that picks the token.

**Why PEARL installs nothing.** Two independent findings, both from review, both structural:

1. **No safe overlap from that seam.** Running the drafter inside `torch.cuda.stream(...)` makes
   every tensor it allocates owned by that stream; `wait_stream` orders *execution*, not
   deallocation. Several escape — `out` into `self._draft_token_ids` (read later on a *third*
   stream that never waits on ours), `self._draft_probs` into the next step's rejection sampler
   (that one corrupts **acceptance**, silently), and the drafter's lazily-created persistent
   buffers, which would stay side-stream-owned for the process lifetime. vLLM flags this exact
   hazard in `v1/worker/gpu/spec_decode/utils.py` and `pp_utils.py`, and every side stream it
   actually uses is a D2H copy into a *pre-allocated* buffer, never a model forward.
2. **Nothing to roll back without speculation.** Wrapping the call to snapshot and drop state
   nothing rewinds is pure cost (~20 launches) on the host-bound path it was meant to relieve.

A third bug was found and is worth recording because it would have been invisible: the snapshot
derived its rows from `block_table[:, 0]`, but under `mamba_cache_mode=align` that column is
`NULL_BLOCK_ID = 0` for everything but the first block — so the rollback would have saved and
restored the scratch null block while the live rows stayed advanced. A rollback that silently
does nothing is worse than none, because the caller believes it happened.

Real post-verify needs the drafter hoisted above `_sample` in `execute_model` and driven from a
guessed prefix. Before building that, measure: the chain needs **K+1** tokens (the bonus token
the drafter never saw), so reuse is `p^(K+1)` ≈ 0.24 at p=0.7, and on ~19 SMs the ~76% miss path
is charged at nearly full price and is *slower* than baseline. Gate: build only if `p^(K+1) > 0.5`
and the drafter exceeds ~35% of step time.

**Owed before this can be trusted** (in addition to the v0.26.0 rows below):
1. `make test-kernel` — the new spec cases in `bench/test_bcx_conv_gate_quant.py` A/B the rollback
   against `causal_conv1d_update` on the whole `[blocks, D, 5]` ring buffer. **That is the only
   thing standing between change 2 and silent recurrent-state corruption.**
2. The temp-0 long-context probe, spec ON vs OFF (row 4 below). Changes 1 and 2 both touch the
   conv-state rollback — the exact defect that produced the original 0% long-context score.
3. `make bench` and read `accept` / `mean_accept_len`. If acceptance is below ~40% per token the
   whole draft arm is negative and `num_speculative_tokens` should drop; `make sweep-spec`
   (K ∈ {0,1,2,3,4}, K=0 = spec OFF) answers it in one command.
4. Grep the boot log for `short-conv fused decode kernel NOT eligible` — it fires once, names the
   layer, and is the only signal that change 1 did not take. Logged at WARNING deliberately: the
   shipped compose runs `VLLM_LOGGING_LEVEL=WARNING`, where an info line would be invisible in
   exactly the deployment that needs it.
5. **Re-pin BOTH digests.** None of this reaches the judge otherwise: the vLLM changes only enter
   the image through `Dockerfile.vllm-fork`'s patch step, and `docker-compose.yaml:6` still pins
   the pre-change `@sha256:c1ade3ac…`. `make vllm-fork PUSH=1`, then re-pin `VLLM_FORK_DIGEST` in
   `Makefile` and the image digest in `docker-compose.yaml`. The judge runs
   `docker-compose.yaml` alone — `docker-compose-optimized.yaml` is a local overlay only.

## OPEN: on-box verification owed for the vLLM v0.26.0 upgrade (2026-07-26)

Row 1 is DONE (`VLLM_FORK_DIGEST=@sha256:c5bea8bf…` in `Makefile`, main image re-pinned to
`@sha256:030c8c94…` in `docker-compose.yaml`). Rows 2–7 still need the H200 and are **not**
optional. Off-box, `make check` and `bash vtl/vllm_patches/gen.sh` both pass as of 2026-07-26
(all six v0.26.0 patches apply clean to a pristine `v0.26.0` tree, the parked one too).

| # | Command | Catches |
|---|---|---|
| 1 | `make vllm-fork PUSH=1` — **RE-OPENED 2026-07-26** | patch/version drift; the `patch --dry-run` gate. The pinned `VLLM_FORK_DIGEST=@sha256:c5bea8bf…` predates `mamba_utils.patch` + the spec-rollback hunks in `short_conv.patch`/`lfm2.patch`, so it must be rebuilt and BOTH digests re-pinned (fork in `Makefile`, main image in `docker-compose.yaml`). Nothing else changes for the shipped config — the new hunks are inert at `num_spec=0`. |
| 2 | `make build && make up && make verify` | plugin loaded, quant methods registered, async scheduling on, **fusion-replaced-N-patterns count** (a drop = a fusion patch silently stopped matching) |
| 3 | `make test-kernel` | the `vtl._C` kernels and the two hijacked `_C` op schemas |
| 4 | `bench/eval_quality.py`, capturing against the **v0.25.0 image** then the v0.26.0 one | **The important one.** Greedy temp-0 output parity. Several vtl patches reimplement vLLM internals verbatim (`qk_norm_rope` replaces `Lfm2Attention.forward`; `greedy_sampler` replaces `Sampler.forward`), and their failure mode is wrong tokens with no exception. Nothing else catches that. |
| 5 | `make bench` / `make ab` vs the v0.25.0 image | whether the upgrade is latency-neutral. Add **boots, not reps** — the noise floor is boot-to-boot (~0.5 ms TPOT). |
| 6 | peak RSS under the 8 GB cap | jemalloc runs `decay:-1` (never returns pages) and v0.26.0 bumps Transformers to 5.13.0 |
| 7 | `python3 -m pytest` of `vtl/patches/qk_norm_rope.py` inside the image | its numeric parity check only runs where there is a GPU **and** a model dir; confirm it actually executes rather than printing "skipped" |

### Leading latency candidate to sweep once the above is green

**`--max-num-scheduled-tokens` — LIVE at 2048, still unmeasured. This is the one behavioural
change in the artifact that no measurement backs.**
`--max-num-batched-tokens=8192` does **not** chunk our prefills: the longest prompt is 4,281
tokens, so every turn-1 prefill runs whole, in one step, alongside in-flight decodes. This flag
caps the scheduler's per-step budget *without* resizing worker buffers / compile ranges /
`max_in_flight_tokens`.

`docker-compose.yaml` ships a literal `2048`. **Do not make it `${VTL_MAX_SCHED_TOKENS:-2048}`
or any other interpolation** — this compose file is deployed to the judge as-is, where an
unresolved `${VAR}` is a boot failure. Sweep by editing the literal.

2048 is well **below** the longest prompt, so it does change behaviour: a 4,281-token turn-1
prefill now takes 3 scheduler steps instead of 1, trading ~2 extra iterations of TTFT for a
flatter in-flight decode ITL tail. Bracket it on both sides — `8192` is the "flag absent" arm
(== `max_num_batched_tokens` == the pre-v0.26.0 fallback), then `4096`, then `1024`.

At 2048 the prefill genuinely chunks, so `--max-num-partial-prefills` /
`--long-prefill-token-threshold` are now live enough to arbitrate rather than being the no-ops
they were at 4096 — sweep those only if 2048 or 1024 wins.

Watch the ITL **tail**, not the mean — gamma=2 is what makes this worth doing. Add boots, not
reps. Note this flag made the file v0.26.0-only; rollback means dropping the line too.

**`--kv-cache-memory-bytes` — deliberately NOT set, needs a number off the box.** It doesn't
speed anything up; it pins KV sizing so memory profiling stops contributing to the boot-to-boot
A/B noise floor. Guessing the value either wastes VRAM or OOMs at allocation, so read it from a
real boot first (vLLM logs the resolved KV cache size at startup), then pin that value for the
duration of a sweep so every arm gets identical capacity.

**Do NOT set `--stream-interval` > 1.** Its docstring is aimed straight at us ("a larger value
(e.g. 10) reduces host overhead ... by batching multiple tokens before sending"), and the host
saving is real. But `bench/replay.py:88-99` counts **SSE deltas**, not tokens
(`n_tokens += 1` per content chunk), and computes `itl_mean = (last_tok - first_tok) /
(n_tokens - 1)`. Batching 10 tokens per chunk therefore multiplies *measured* TPOT ~10x: from
~4-6 ms to 40-60 ms, past the 10 ms ceiling, i.e. `s_tpot = 0` on every request. Note
`out_tokens` prefers `usage.completion_tokens` but `itl_mean` does not — usage does not rescue
it. Only revisit if the judge is confirmed to normalise ITL by token count.

**Do NOT set `VLLM_USE_BREAKABLE_CUDAGRAPH=1`.** `vllm/config/vllm.py:1201-1207` maps it to
`CompilationMode.NONE` ("Equivalent to -cc.mode=none"), which disables the whole torch.compile
pipeline and takes `fuse_norm_quant` + `fuse_act_quant` — i.e. the custom kernel work — with it.
LFM2 is not in its auto-enable allowlist. Same for `fuse_attn_quant`: it cannot match our
dynamic per-token fp8 activation quant, and enabling it downgrades `cudagraph_mode` from
FULL_AND_PIECEWISE to FULL.

**Inert here, checked so nobody re-checks:** `--prefill-schedule-interval` lives in
`DPEngineCoreProc`, which asserts `is_moe` — data-parallel only, dead at TP=1.
`--max-num-partial-prefills` / `--long-prefill-token-threshold` bite only once prefills actually
chunk. That is no longer hypothetical: `--max-num-scheduled-tokens` now ships at **2048**, so a
4,281-token prompt spans 3 steps and these two have something to arbitrate. Sweep them only
after the `VTL_MAX_SCHED_TOKENS` bracket picks a winner — at 4096 or 8192 they go back to being
no-ops. `--mamba-cache-dtype` does control the
short-conv state (`short_conv.py:576`, `lfm2.py:437`), but fp8 would save ~80 KB/step against
~850 MB of weight traffic (~0.02%), leaves the derived block size at 16, and the vtl
`bcx_conv_gate_quant` / `mul_quant` kernels assume a bf16 conv state — negligible upside, real
risk. `--kv-cache-memory-bytes` / `--num-gpu-blocks-override` don't speed anything up but pin KV
sizing, which removes memory-profiling variance from the boot-to-boot A/B noise floor.

**Diagnostic worth wiring — mechanism CORRECTED 2026-07-26.** `--profiler-config` takes
`delay_iterations`, `warmup_iterations`, `max_iterations`, `active_iterations` and
`ignore_frontend`, i.e. a **bounded in-run torch profile**. It does **not** remove the need for
`/start_profile` — that endpoint is still the trigger. What it does is *mount* it:
`entrypoints/serve/profile/api_router.py:36-44` only calls `app.include_router(router)` when
`profiler_config.profiler is not None`. That is exactly why the served build 404s today —
`VLLM_TORCH_PROFILER_DIR` is unset, so the route was never registered. So:

    --profiler-config='{"profiler":"torch","torch_profiler_dir":"/profile",
                        "ignore_frontend":true,"delay_iterations":200,"max_iterations":50}'
    curl -XPOST :8000/start_profile   # …replay… ; curl -XPOST :8000/stop_profile

v0.26.0 adds `capture_torch_profiler` (traces cudagraph capture itself) and
`detailed_trace_annotation`. `docker-compose.profile.yaml` deliberately does **not** wire this —
a compose overlay's `command:` replaces the base list rather than appending, so it would mean
duplicating every serve flag; prefer the vtl profiler and reach for this only for vLLM's own
annotations. **Scope caveat, same as `make profile`:** the consumers are
`gpu_worker.py` and `async_llm.py` — worker and frontend. `Scheduler.schedule()` is not
instrumented by either, so the three elisions above still ship unmeasured, and this cannot
attribute scheduler-side host time. It *can* attribute the ~4 ms host-bound decode step inside
`execute_model`, which is the bigger half.

**Dead scheduler work, now elided (2026-07-26).** Three pieces of per-step/per-request work that
this configuration computed and no consumer read. All are microsecond-scale — this is dead-work
removal, not a win, and an indistinguishable A/B is the expected result, not a failure.

- `KVCacheManager.estimate_cached_tokens()` on the step that emits each request's first token
  (`scheduler.py:1798-1806`) — `get_blocks()` plus a loop over every allocated block in every KV
  group, ~270 blocks for the full-attention group on a 4.3k prompt, landing in TTFT. Earlier
  notes called this an unavoidable v0.26.0 regression. It is not: `vtl/patches/kv_cache_manager.py`
  returns 0 when `scheduler.log_stats` is false, which `--disable-log-stats` gives us. See the
  `--enable-prompt-tokens-details` warning in `docker-compose.yaml` before re-enabling anything.
- `KVCacheManager.get_num_common_prefix_blocks()` every `schedule()` call
  (`scheduler.py:1078-1084`) — fills `SchedulerOutput.num_common_prefix_blocks`, whose only
  consumer is `gpu_model_runner.py:4207`, i.e. **V1** cascade attention. The V2 runner never
  reads it. Not already free either: prefix caching + 2 groups selects
  `HybridKVCacheCoordinator`, which does not override the method, so `FullAttentionManager` walks
  the request's blocks counting `ref_cnt`. Elided only when V2 is *proven* live — read from an
  explicitly-set `VLLM_USE_V2_MODEL_RUNNER` (`docker-compose.yaml` always sets it to a literal),
  fail-closed-to-stock if it does not resolve.
- **REVERTED 2026-07-26.** Guards on our own waiting-queue reorder — skip when the running cap
  is hit, plus a queue fingerprint with an 8-step re-sort TTL, plus a `sched_policy`→manager
  handshake carrying the Scheduler's `use_v2_model_runner`. Measured **slower** and were backed
  out; `sched_policy.py` is again the pre-guard reorder (only the `len < 2` return). They could
  not have paid for themselves: `waiting` is almost always shorter than 2 at this concurrency,
  so `len < 2` fired first on nearly every step and the guards only added per-step work to the
  path they were meant to shorten. The burst case they targeted does not occur on this trace
  (mean 2 / peak 8 concurrent generations). The elision above lost only its handshake source
  and still resolves from the env. Do not re-add caching here without a queue deep enough to
  sort in the first place.

**`make profile` was broken and is fixed (2026-07-26).** `vtl/patches/profiler.py` imported the
**V1** `GPUModelRunner` while the server runs `VLLM_USE_V2_MODEL_RUNNER=1`, so the worker built
`vllm.v1.worker.gpu.model_runner.GPUModelRunner` (`gpu_worker.py:402-410`) and the wrapper never
fired — it logged `"profiler installed"` every boot and captured nothing. It now wraps both runner
classes (only one is ever constructed). Scope caveat: it wraps the **worker's** `execute_model`, so
the scheduler — which lives in the engine-core process — still does not appear in the trace. The
three elisions above therefore ship without direct measurement.

**Expectation management:** the upgrade's original justification (#46384 `--prefix-match-unit`)
was found to be a **no-op on this model** — see the derivation in `docker-compose.yaml`. Do not
expect a TTFT win. What v0.26.0 actually buys is incidental host-side work (#48641 drops an fp32
logit copy from the sample path, #48143 an allocation from the SSM metadata build, #46647 moves
iteration logging off the engine-core loop) plus staying current. If step 5 shows a regression,
the rollback is the digest in `Makefile` (image-level only — `vtl/vllm_patches/v0.25.0/` was
deleted, so a source-level rollback needs `git revert`).

### Drafter work: the two items that were nearly dropped (2026-07-27)

Both the fused GEMV+argmax and PEARL were initially skipped on expected-value grounds and then
built anyway. The EV reasoning is still correct and worth keeping, because it is what should
decide whether to KEEP them once there are numbers:

**Fused argmax** is the smaller prize by an order of magnitude. After FR-Spec pruning the logits
tensor is `8 x 16384` fp32 (~0.5 MB), so the fusion is worth order 15 us/step against pruning's
~150 us. It is built and behind `VTL_DRAFT_FUSED_ARGMAX`; if a profile does not show
`compute_logits`/`argmax` on the critical path, turning it off costs nothing. What made it worth
building rather than skipping is that the risky part — decoding CUTLASS's reordered layout — is
avoidable entirely by carrying an independent 8 MB copy.

**PEARL** is the weakest item on the list and is shipped OFF for that reason, not because it is
unfinished plumbing. The arithmetic: a non-tree chain must speculate **K+1** tokens (the bonus
token the drafter never saw), so reuse is `p^(K+1)` — about 0.24 at p=0.7. On a ~19 SM MIG slice
there is no idle GPU to overlap into, so the ~76% miss path is charged at nearly full price and
is *slower* than baseline. The snapshot rollback currently in `pearl.py` costs ~20 launches to
save the ~30 being overlapped; the zero-copy fix is `bcx_conv_gate_quant`'s kSpec sliding window
retargeted at the drafter, which also needs the draft model built at `num_spec + 1` state width —
new coupling in the `short_conv`/`lfm2`/`mamba_utils` three-file lockstep that
`bench/test_shortconv_spec_rollback.py` exists to protect.

Decision procedure, ~20 lines and worth running before touching either again: histogram
`num_accepted` off the rejection sampler and time `drafter.propose` against the target forward
with CUDA events. That yields `p` and the drafter's share of step time, which is all that matters.

### Speculative decoding — LIVE in the submitted compose, unmeasured (2026-07-26)

**Read this first: the shipped arm is now a hybrid LFM2.5-350M draft model, not ngram.**
`docker-compose.yaml` carries
`--speculative-config={"method":"draft_model","model":"LiquidAI/LFM2.5-350M","num_speculative_tokens":3,"quantization":"vtl_w4a8"}`
as an **active** line, with the `ngram_gpu` variant commented directly beneath it as the
one-literal revert. `VLLM_USE_V2_MODEL_RUNNER` is already `"0"`, which `draft_model` needs for
the same reason ngram did. Three things below were corrected on the same day:

1. **The short-conv rollback was INERT until this commit.** Upstream PR #44296 is a three-file
   fix; only two shipped. `gpu_model_runner` never named `ShortConvAttentionMetadataBuilder` in
   the `use_spec_decode` gate that injects `num_accepted_tokens`, so the builder saw `None`,
   `forward_cuda` took the non-spec path, and `mamba_attn`'s `decode_threshold` collapsed to 1
   (every spec row misclassified as a *prefill*). Fixed in
   `v0.26.0/gpu_model_runner.patch`; `gen.sh` now captures that file, and
   `bench/test_shortconv_spec_rollback.py` has a fourth case that fails if it is ever dropped
   again. **Consequence: any previously-submitted image with the ngram line live was corrupting
   conv state.** Run go/no-go (1) on the ngram arm before assuming otherwise.
2. **A hybrid draft model is no longer impossible** (this file used to say it was). vLLM issue
   #49112 is that blocker; `llm_base_proposer_multigroup.patch` +
   `mamba_groups_hybrid_draft.patch` + the runner patch fix it — see
   `vtl/vllm_patches/not-applied/README.md` for what was ported and, more usefully, what turned
   out **not** to need porting.
3. **The drafter must be quantized or it cannot win.** A draft model does *not* inherit
   `--quantization`; without the `"quantization":"vtl_w4a8"` key it loads bf16 at ~0.71 GB of
   weight traffic per draft token, against a target step of ~0.85 GB in w4a8 — break-even would
   need ~3.3 of 4 tokens accepted. At w4a8 the drafter is ~0.34 GB and break-even lands near 55%
   acceptance. Needs no patch: `vllm.config.utils.replace` re-runs `VllmConfig.__post_init__`,
   which builds `quant_config` from the *draft* `model_config`, and `vtl_w4a8` quantizes RTN at
   load. Sweep `vtl_w4a8` / `vtl_fp8` / no key, and `num_speculative_tokens_per_batch_size`
   (adaptive gamma) — all inside the one JSON literal.

The original root-cause writeup follows and is still accurate.

### Speculative decoding — root cause of the old ban (2026-07-26)

**The "truncation / dual-path cheating" flag was never a detector.** It was the judge's
inference from the long-context probe scoring 0% while the short scored trace looked healthy —
the signature of a server with two code paths. The actual defect is an upstream vLLM bug on
this model:

- LFM2 has 10 short-conv layers, each carrying a persistent `conv_state`.
- Under chain spec-decode the target advances that state once per **draft** token.
- Stock `ShortConv.forward_cuda` calls `causal_conv1d_update` with **no** `num_accepted_tokens`
  / `query_start_loc` / `max_query_len` (verified against pristine `v0.26.0`, line 282), so
  **rejected drafts are committed to the state and never rolled back**. `mamba_mixer2.py:991`
  passes exactly those three args for exactly this reason; short-conv just never did.
- The error is small per step and compounds with length: clean at 4k, garbage at 32k. Hence a
  passing benchmark and a 0% long-context probe from one code path.

Fixed and now in the applied set: `v0.26.0/short_conv.patch` (pass the three kernel args,
capture `num_spec`, widen `get_state_shape`, bypass the fused `bcx_conv_gate_quant` while spec
is active — it has no `num_accepted` path), `lfm2.patch` and `mamba_utils.patch` (thread
`num_spec` through the shape planner; all three must move together or boot raises). Inert at
`num_spec=0`, so the shipped non-spec artifact is unchanged. Contract test:
`bench/test_shortconv_spec_rollback.py` (in-image, skips on the dev box).

**Second blocker, and it is the one that bites first:** vLLM v0.26.0's **V2 model runner does
not implement ngram/ngram_gpu/suffix at all** — `config/vllm.py:2154-2165` lists them as
unsupported and `_validate_v2_model_runner` **raises at startup**, so with
`VLLM_USE_V2_MODEL_RUNNER=1` the process dies before binding `:8000` and every request scores
zero. `draft_model` is in the same rejected set, so this applies to the shipped arm too. Spec-decode
here means the **V1** runner. Two things in `docker-compose.yaml` must always move together:

1. the live `--speculative-config=…` line in the `command:` block;
2. `VLLM_USE_V2_MODEL_RUNNER: "0"` in `environment:`.

(Literal edits, not env interpolation — see the `--max-num-scheduled-tokens` note above for
why this file carries no `${VAR}`.) Flipping to V1 makes `v2_greedy_sampler.patch` + `mamba_hybrid_postprocess.patch` inert and
re-activates `vtl/patches/greedy_sampler.py` (the V1 fast path, already spec-safe). The
`kv_cache_manager` common-prefix elision self-disables under V1 by reading that same
`VLLM_USE_V2_MODEL_RUNNER: "0"`, which is correct — V1 cascade attention is the one consumer
of that field. So edit 2 covers both.

**Go/no-go, both mandatory before this ships:**
1. Long-context probe token-identical spec ON vs OFF at temperature 0. This is the exact check
   the old flag was a proxy for; nothing else re-earns the trust.
2. TPOT actually drops on the MIG 1g.18gb slice. The prior is favourable — decode is
   weight-traffic bound (852 MB/step at batch ≤ 8), so verifying k+1 tokens is nearly free
   while step count falls with acceptance — but the drafter runs on the co-dominant host term
   and the fused conv-gate kernel is off under spec. Also log the acceptance rate: this trace
   generates prose rather than copying from the prompt, which is ngram's weak case.

Whichever way (2) lands, (1) is worth running once regardless: it is the only direct evidence
that the dual-path root cause is actually gone.

### Compile-knob survey, 2026-07-26 (read before touching `--compilation-config`)

Two settle old questions, one is the only unset knob left:

**`-O3` is byte-identical to the `-O2` default.** `OPTIMIZATION_LEVEL_03` and
`OPTIMIZATION_LEVEL_02` are the same dict, field for field (`vllm/config/vllm.py:252-296`). The
`-O3` in `docker-compose.yaml` buys exactly nothing and costs exactly nothing — do not attribute
any measurement to it, and do not "upgrade" to it as a fix. (`-O3` parses to
`--optimization-level 3`, a field entirely separate from `--compilation-config`
(`utils/argparse_utils.py:322-326`), so the two flags do **not** fight; the explicit pass_config
wins because level defaults only fill fields still `None`.)

**`fuse_attn_quant` cannot turn itself on.** The level dicts set it to `IS_QUANTIZED`, and
`IS_QUANTIZED = False` is hardcoded at `vllm/config/vllm.py:97-104` pending issue #25689. The
existing "do not enable it" warning is still correct, just unreachable by accident.

**`use_inductor_graph_partition` — the one compile knob nothing sets, forced `False` by every
optimization level.** With it off, the FX graph is split at `_attention_ops`
(`config/compilation.py:764-780`), which includes **`vllm::short_conv`** — so on LFM2 the graph
splits at all 16 layers (10 conv + 6 attn), and every custom pass runs per-subgraph. With it on,
partitioning moves to Inductor codegen *after* passes and fusions, so `fuse_norm_quant` /
`fuse_act_quant` see the whole graph (`config/compilation.py:520-531`).

Honest prior: **probably a null result.** Walk an LFM2 layer — `operator_norm → in_proj →
short_conv → out_proj → ffn_norm → w13 → silu_mul → down_proj` — and both fusions we depend on
already sit inside one subgraph; the conv-gate/out_proj fusion is a hand-written vtl kernel, not
an Inductor pass, so it does not care either way. Run it anyway because it is one JSON key and
`make verify` already prints the number that would move:

    -cc='{"pass_config":{"fuse_norm_quant":true,"fuse_act_quant":true},
          "cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,32],
          "use_inductor_graph_partition":true}'

Read the **fusion-replaced-N-patterns** count first: unchanged ⇒ no cross-boundary fusion
existed ⇒ stop, do not bother with `make ab`. Needs torch ≥ 2.9 (`compilation.py:994-1001`).

**Not worth it: filling the `cudagraph_capture_sizes` gaps.** `[1,2,4,8,16,32]` pads a batch of
5-7 up to 8. Peak concurrency on this trace is ~8 and the mean is 2, and decode is weight-traffic
bound (852 MB/step) where batch size barely moves the GPU term. 16 and 32 are never reached at
all; dropping them saves capture time and VRAM, not latency.


## Hardware

| Spec | Value |
|------|-------|
| GPU | MiG H200 profile — 18 GB VRAM, 16 SMs, ~600 GB/s bandwidth |
| CPU | 3 vCPU (CFS quota) |
| RAM | 8 GB (no swap) |
| Driver | NVIDIA 590.x (CUDA 13.x) |
| OS | Ubuntu 24.04 LTS |
| Base image | `linux/amd64` only — must not build on arm64 |

## Model

**LiquidAI/LFM2.5-1.2B-Instruct** — hybrid architecture:

| Property | Value |
|----------|-------|
| Hidden size | 2048 |
| Intermediate | 12288 |
| Layers | 16 (10 short-conv + 6 GQA full-attention) |
| KV heads | 8, head_dim=64 |
| Short-conv dim | 2048, cache_len=3 |
| Max position | 128000 (capped at `--max-model-len=32768`) |
| Tokenizer | LFM2 chat template, no special tokens beyond `<|im_start|>`/`<|im_end|>` |

**KV footprint (attention layers only):**
- BF16: 2 × 6 layers × 8 heads × 64 dim × 2 bytes = **12 KB/token**
- FP8 KV: **6 KB/token**

## Workload

| Parameter | Value |
|-----------|-------|
| Conversations | 70 (independent, simultaneous) |
| Turns per conversation | 6 |
| Total requests | 70 × 6 = **420** |
| Shared system prefix | 1,000 tokens (identical across all conversations) |
| Per-conversation prefix | 1,000 tokens (turn 1 only) |
| New user tokens per turn | 150 |
| Output tokens per turn | 300 |
| Max context (turn 6) | ~2,000 (prefixes) + 6 × 450 (in+out) = **~4,700 tokens** |
| Arrival | Poisson, seed 42 |
| Sampling | Greedy (temperature=0), no logprobs, no tools |

**Prefix cache:** Turn N reuses turns 0..N−1 within the same conversation. Simulated block-level hit rate: **~82%** — the single biggest win in the workload. The KV working set at 70 concurrent max-context sequences is ~2.0 GB FP8, well within the 18 GB budget.

**Prefill-to-decode token ratio:** ~3:1 in raw tokens, but prefix caching eliminates ~82% of prefill work per block.

**Files:**
- `data/input/trace-round2.jsonl` — 420 request records (OpenAI chat format)
- `data/input/trace_grading_public.jsonl` — per-request token counts + metadata
- `data/input/grading-workload-spec.json` — workload specification

## Scoring (ERS — Effective Request Score)

```
ERS = (1 / N) × Σ S_request,i    ∈ [0, 1]

For each successful request:

  S_request = w × s_ttft + (1 − w) × s_tpot

  s_ttft = [clamp((C_ttft − TTFT) / (C_ttft − F_ttft), 0, 1)]^γ
  s_tpot = [clamp((C_tpot − TPOT_mean) / (C_tpot − F_tpot), 0, 1)]^γ
```

### Parameters

| Symbol | Meaning | Value |
|--------|---------|-------|
| F_ttft | TTFT floor (best possible) | 10 ms |
| C_ttft | TTFT ceiling (score=0 beyond) | 400 ms |
| F_tpot | TPOT floor (best possible) | 1 ms |
| C_tpot | TPOT ceiling (score=0 beyond) | 10 ms |
| γ | Power exponent (quadratic penalty) | 2 |
| w | TTFT weight | 0.5 |

### Failure Modes (score = 0 per request)

- HTTP error (non-200)
- Timeout (120s per request, from send to last SSE chunk)
- Empty output (0 tokens returned)
- Connection/parse exception

### Key Scoring Properties

- **γ=2** penalizes tail latency quadratically — a request at 50% of ceiling scores 0.25 (not 0.50).
- **TPOT ceiling is 40× stricter than TTFT** (10 ms vs 400 ms). A 1 ms decode latency spike costs 40× more than a 1 ms prefill latency spike in raw score.
- **Weight is symmetric (w=0.5)** — TTFT and TPOT contribute equally when normalized.
- **One failure = 0 for that request** — errors drag the average down linearly.
- **No warm-up** from the judge — first wave of requests after healthy signal is cold at the HTTP layer. The prefix cache is pre-warmed by the healthcheck (see `vtl/warmup_healthcheck.py`).

## Optimization Surface

### Quantization (load-bearing)

- **W4A8** with per-channel weight scales (`vtl_fp8` quant method, INT4 weights + FP8 activations). Stock vLLM uses per-tensor scales; per-channel improves accuracy at zero runtime cost.
- **FP8 KV cache** (`fp8_e4m3`) — halves KV memory vs BF16, allowing more concurrent sequences.
- Ignored layers: `lm_head` stays in BF16 (tied to embeddings, small vocab).
- Short-conv projections (`in_proj`/`out_proj` across 10 layers) must be explicitly rebuilt with FP8 quant config — stock builds them BF16.

### CUDA Kernel Fusions (decode throughput)

| Fusion | What | Impact |
|--------|------|--------|
| RMSNorm + FP8 quant (+ residual) | `operator_norm` → qkv/in_proj, `ffn_norm` → w13 | Single-pass replacement for 3-pass stock kernel |
| SiLU-mul + FP8 quant | `gate|up` → `silu_and_mul` → `down_proj` input quant | Eliminates BF16 intermediate tensor in HBM |
| Conv-gate mul + FP8 quant | `y = C × Bx` + `out_proj` input quant | Eliminates BF16 `y` tensor, saves one kernel launch per conv layer |
| Standalone per-token FP8 quant | `o_proj` input, weight quantization | Optimized replacement for stock, coarsened threading |

### Scheduling

- **Prefix caching** (vLLM block-hash radix tree, `--enable-prefix-caching`) — lossless, O(1) per block.
- **Chunked prefill** (`--enable-chunked-prefill`) — breaks long prefills into small compute chunks, protecting decode latency.
- **Cache-aware SJF reorder** — waiting queue sorted by fewest uncached prefill tokens. Memory-aware when KV is tight.
- **Async scheduling** — vLLM's default overlap scheduler, no flag needed.

### Decode/Prefill Tuning Levers

| Lever | Current | Notes |
|-------|---------|-------|
| `--max-num-batched-tokens` | 8192 | Tradeoff: smaller = safer TPOT but more prefill steps |
| `--max-num-seqs` | 70 | Matches 70 conversations; peak concurrency under Poisson is lower |
| `cudagraph_capture_sizes` | `[1,2,4,8,16,32]` | Tune for actual batch size distribution |
| `cudagraph_mode` | `FULL_AND_PIECEWISE` | PIECEWISE avoids recompilation for unused sizes |
| Speculative decoding | **UN-BANNED 2026-07-26, still OFF by default** | The old "cheating" flag was the judge inferring two code paths from a 0% long-context probe, not a detector on `--speculative-config`. Root cause and fix: short-conv `conv_state` rollback, now in the applied patch set. Two env vars turn it on (`VTL_SPEC` + `VTL_V2_RUNNER=0`); see the block in `docker-compose.yaml` and the section below. |

### Frontend (per-request overhead)

- **msgspec JSON** — replaces stdlib `json` for request parsing and non-streaming response serialization.
- **msgspec SSE** — per-token streaming chunks use `msgspec.json.encode` instead of pydantic `model_dump_json`.
- **Rust frontend** — vllm-rs rebuilt with fat-LTO, codegen-units=1, sonic_rs parser. Optional PGO via CPU mock engine. Stick with stack-allocated unions only — heap boxing defeats the purpose.
- **Greedy sampler fast path** — argmax shortcut when no logprobs/penalties/tools are active (100% of this workload).

### System/Runtime

- **Jemalloc** — `LD_PRELOAD` with latency tuning: no decay (never returns pages), metadata THP, percpu arenas. Watch RSS under 8 GB cap.
- **AOT + standalone compile** — `VLLM_USE_AOT_COMPILE=1`, `VLLM_USE_STANDALONE_COMPILE=1`.
- **CUDA graph** — `FULL_AND_PIECEWISE` with capture sizes tuned per batch shape.

### Memory & KV

- **KV offload to CPU/NVMe** is available but OFF by default — KV working set (~2 GB FP8) fits in 18 GB with headroom.
- **PagedAttention** — vLLM's block-level memory management, 16-token blocks by default.

### What NOT to Touch

| Component | Reason |
|-----------|--------|
| `mamba_cache_mode` | Must be `align` — LFM2 raises NotImplementedError on `all` |
| Custom draft model training | Potential overfitting to accuracy gate questions → invalid solution |
| Tensor parallelism | TP=1 — no benefit on single GPU |
| Chat template / tokenizer | Must match LFM2.5-1.2B-Instruct exactly per rules |

## Constraints

| Constraint | Value | Impact |
|-----------|-------|--------|
| VRAM budget | 18 GB (`--gpu-memory-utilization=0.90`) | Model ~1.2 GB FP8, KV ~2 GB FP8, CUDA graph cache ~2-3 GB, headroom ~10 GB |
| Host RAM | 8 GB | Jemalloc `decay:-1` means RSS only grows — validate peak RSS before submitting |
| vCPU | 3 | Tokio workers = 2-3 (trade decode vs HTTP) |
| `/dev/shm` | Billed against 8 GB RAM cap | Keep `shm_size` low (TP=1 needs little) |
| Platform | `linux/amd64` only | Builds on arm64 produce unrunnable images |
| Timeout | 120s per request, no retry | Decode throughput is safe (300 tok × ~10ms = 3s); queue depth under Poisson arrival matters |
| Startup errors | Not counted in score | But startup time counts against readiness |
| Custom image | Allowed | vLLM fork + VTL plugin + baked caches is fully supported |
| Draft model weights | Allowed in image | But must not be tuned to overfit accuracy gate questions |
