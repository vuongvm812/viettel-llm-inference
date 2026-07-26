# HANDOFF — Round 1.2 Mission Brief

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

### Speculative decoding — un-banned, wired, unmeasured (2026-07-26)

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
zero. Only eagle/eagle3/mtp/dflash/dspark survive V2, and all of those need a draft model that
a hybrid LFM2 target cannot host. So spec-decode here means the **V1** runner. Two edits in
`docker-compose.yaml`, and they must move together:

1. uncomment the `--speculative-config={"method":"ngram",…}` line in the `command:` block;
2. set `VLLM_USE_V2_MODEL_RUNNER: "0"` in `environment:`.

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
