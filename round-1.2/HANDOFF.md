# HANDOFF — Round 1.2 Mission Brief

## OPEN: on-box verification owed for the vLLM v0.26.0 upgrade (2026-07-26)

The upgrade landed with only the off-box gates run. Everything below needs the H200 and is
**not** optional — the fork image has not been built even once on v0.26.0.

| # | Command | Catches |
|---|---|---|
| 1 | `make vllm-fork PUSH=1` | patch/version drift; the `patch --dry-run` gate. Then set `VLLM_FORK_DIGEST=@sha256:<digest>` in the Makefile — `make push` refuses to ship without it. |
| 2 | `make build && make up && make verify` | plugin loaded, quant methods registered, async scheduling on, **fusion-replaced-N-patterns count** (a drop = a fusion patch silently stopped matching) |
| 3 | `make test-kernel` | the `vtl._C` kernels and the two hijacked `_C` op schemas |
| 4 | `bench/eval_quality.py`, capturing against the **v0.25.0 image** then the v0.26.0 one | **The important one.** Greedy temp-0 output parity. Several vtl patches reimplement vLLM internals verbatim (`qk_norm_rope` replaces `Lfm2Attention.forward`; `greedy_sampler` replaces `Sampler.forward`), and their failure mode is wrong tokens with no exception. Nothing else catches that. |
| 5 | `make bench` / `make ab` vs the v0.25.0 image | whether the upgrade is latency-neutral. Add **boots, not reps** — the noise floor is boot-to-boot (~0.5 ms TPOT). |
| 6 | peak RSS under the 8 GB cap | jemalloc runs `decay:-1` (never returns pages) and v0.26.0 bumps Transformers to 5.13.0 |
| 7 | `python3 -m pytest` of `vtl/patches/qk_norm_rope.py` inside the image | its numeric parity check only runs where there is a GPU **and** a model dir; confirm it actually executes rather than printing "skipped" |

### Leading latency candidate to sweep once the above is green

**`--max-num-scheduled-tokens`** (new CLI flag in v0.26.0). `--max-num-batched-tokens=8192` does
**not** chunk our prefills — the longest prompt is 4,281 tokens, so every turn-1 prefill runs
whole, in one step, alongside in-flight decodes. This flag caps the scheduler's per-step budget
*without* resizing worker buffers / compile ranges / `max_in_flight_tokens`. Sweep
8192 (control) / 4096 / 2048 / 1024 with `make ab`; watch the ITL **tail**, not just the mean,
since gamma=2 punishes it. Adding it drops rollback compatibility with the v0.25.0 image.

**Do NOT set `VLLM_USE_BREAKABLE_CUDAGRAPH=1`.** `vllm/config/vllm.py:1201-1207` maps it to
`CompilationMode.NONE` ("Equivalent to -cc.mode=none"), which disables the whole torch.compile
pipeline and takes `fuse_norm_quant` + `fuse_act_quant` — i.e. the custom kernel work — with it.
LFM2 is not in its auto-enable allowlist. Same for `fuse_attn_quant`: it cannot match our
dynamic per-token fp8 activation quant, and enabling it downgrades `cudagraph_mode` from
FULL_AND_PIECEWISE to FULL.

**Small regression to expect:** v0.26.0 adds `KVCacheManager.estimate_cached_tokens()` on the
step that emits each request's first token (`scheduler.py:1805`), a Python loop over every
allocated block in every KV group. It is NOT gated by `--disable-log-stats` (`PrefillStats()` is
built unconditionally at `request.py:197`). At our ~4.7k max context that is ~600 trivial
iterations once per request, landing in TTFT — tens of microseconds, but one-directional.

**Expectation management:** the upgrade's original justification (#46384 `--prefix-match-unit`)
was found to be a **no-op on this model** — see the derivation in `docker-compose.yaml`. Do not
expect a TTFT win. What v0.26.0 actually buys is incidental host-side work (#48641 drops an fp32
logit copy from the sample path, #48143 an allocation from the SSM metadata build, #46647 moves
iteration logging off the engine-core loop) plus staying current. If step 5 shows a regression,
the rollback is the digest in `Makefile` (image-level only — `vtl/vllm_patches/v0.25.0/` was
deleted, so a source-level rollback needs `git revert`).


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
| Speculative decoding | OFF | ngram (free, no weights) or eagle/medusa (needs draft heads). Greedy temp=0 workload is ideal for ngram |

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
| VRAM budget | 18 GB (`--gpu-memory-utilization=0.95`) | Model ~1.2 GB FP8, KV ~2 GB FP8, CUDA graph cache ~2-3 GB, headroom ~10 GB |
| Host RAM | 8 GB | Jemalloc `decay:-1` means RSS only grows — validate peak RSS before submitting |
| vCPU | 3 | Tokio workers = 2-3 (trade decode vs HTTP) |
| `/dev/shm` | Billed against 8 GB RAM cap | Keep `shm_size` low (TP=1 needs little) |
| Platform | `linux/amd64` only | Builds on arm64 produce unrunnable images |
| Timeout | 120s per request, no retry | Decode throughput is safe (300 tok × ~10ms = 3s); queue depth under Poisson arrival matters |
| Startup errors | Not counted in score | But startup time counts against readiness |
| Custom image | Allowed | vLLM fork + VTL plugin + baked caches is fully supported |
| Draft model weights | Allowed in image | But must not be tuned to overfit accuracy gate questions |
