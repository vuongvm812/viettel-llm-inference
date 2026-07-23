# HANDOFF — Round 1.2 Mission Brief

## Hardware

| Spec | Value |
|------|-------|
| GPU | MiG H200 profile — 18 GB VRAM |
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

- **W8A8 FP8** with per-channel weight scales (`vtl_fp8` quant method). Stock vLLM uses per-tensor scales; per-channel improves accuracy at zero runtime cost.
- **FP8 KV cache** (`fp8_e4m3`) — halves KV memory, allowing 2× more concurrent sequences.
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
- **CUDA lazy loading** — `CUDA_MODULE_LOADING=LAZY` reduces startup time.
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
