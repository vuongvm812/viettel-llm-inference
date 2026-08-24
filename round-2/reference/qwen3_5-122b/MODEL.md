# Qwen/Qwen3.5-122B-A10B-FP8 — round-2 model reference

This directory sits next to `reference/lfm2/` (frozen phase-2 material) but is the **current**
round's model reference. `config.json` here is a verbatim copy of the checkpoint's config
(fetched 2026-08-16 via `make model-fetch-meta`); everything below is derived from it plus the
shard listing on HF (127.2 GB, 39 safetensors shards).

## Geometry

| | |
|---|---|
| Architecture | `Qwen3_5MoeForConditionalGeneration` (`model_type: qwen3_5_moe`) |
| Text layers | 48 = **36 GDN linear-attention + 12 full-attention** (`full_attention_interval: 4`, every 4th layer) |
| GDN (linear attn) | 16 key heads × 128, 64 value heads × 128; conv1d kernel 4; **fp32 SSM state** (`mamba_ssm_dtype: float32`) ≈ **154 MB/seq** across the 36 layers; `attn_output_gate: true` |
| Full attention | 32 Q heads / **2 KV heads**, head_dim **256**; `partial_rotary_factor: 0.25`, interleaved mrope (`mrope_section [11,11,10]`), rope_theta 1e7 |
| MoE (every layer) | 256 experts, top-8 routed, `moe_intermediate_size 1024`; + shared expert 1024 with sigmoid gate; `mlp_only_layers: []` |
| Hidden / vocab | hidden 3072; vocab **248320**; eos 248044; embeddings untied |
| Context | `max_position_embeddings: 262144` |
| MTP | 1 dedicated full-attention drafter layer (`mtp_num_hidden_layers: 1`, shared embeddings) |
| Vision tower | 27 layers (hidden 1152, patch 16) — **skipped at serve time**; text-only serving |

## Quantization

FP8 block-quant checkpoint: `quant_method: fp8`, `weight_block_size [128,128]`, dynamic
per-group-128 activation scales (`activation_scheme: dynamic`, nothing per-tensor). 321
modules stay **bf16** (`modules_to_not_convert`): `lm_head`, `embed_tokens`, and per GDN layer
`conv1d`, `in_proj_a`, `in_proj_b`, plus every `mlp.gate` and `shared_expert_gate` (router
gates). The big weights — routed/shared experts, attention projections — are fp8.

## Memory math on 1× H200 (141 GB)

- **Weights: 127.2 GB** on disk ≈ resident (fp8 majors + bf16 exclusions; lm_head alone is
  248320×3072×2B = 1.53 GB).
- At `--gpu-memory-utilization 0.95`: 141 × 0.95 ≈ **134 GB usable → ~6.7 GB** for everything
  that is not weights.
- **GDN state** (allocated per scheduled seq, fp32): ≈154 MB/seq →
  3.7 GB @ max-num-seqs 24, 4.9 GB @ 32.
- **KV cache** (only the 12 full-attn layers pay): 12 × 2 KV heads × 256 × 2 (K+V) × 2B =
  **24 KB/token bf16**. What's left after GDN state (~1–3 GB) is 40k–120k tokens of KV — at
  the trace's required `--max-model-len ≥ 4663` that is ~113 MB/seq, i.e. KV is not the
  binding constraint; the fp32 GDN state is.
- Net: fits **only** with `max-num-seqs ≈ 24–32` and `gpu-memory-utilization ≈ 0.95`.
  Sweep both (see checklist) — 0.97 risks allocator OOM under CUDA-graph capture, 0.93 may
  not leave enough KV to hold the trace's working set.
- Caveat: `bench/trace_stats.py`'s KV lines assume 48 full-attention layers (96 KB/token) —
  4× pessimistic for this hybrid model; use the 24 KB/token figure above.

## Decode traffic per token

- Routed experts: 8 × 3 matrices × (3072×1024) fp8 × 48 layers ≈ **3.6 GB/token** (batch=1
  worst case; batching amortizes only where requests hit the same experts).
- lm_head: **1.53 GB** bf16.
- So single-seq decode is entirely weight-bandwidth-bound; raising batch toward the
  max-num-seqs cap is nearly free latency-wise until experts stop overlapping.

## What vLLM v0.25.0 provides stock

- `Qwen3_5MoeForConditionalGeneration` **and** the `Qwen3_5MoeMTP` drafter are registered —
  arch loads without patches.
- GDN linear attention runs on **FLA Triton kernels** (no CUDA-core kernel).
- The `RMSNormGated → group-quant` fusion for the GDN path exists **only on ROCm** in
  v0.25.0 — on CUDA the gated norm and the dynamic group-128 activation quant run unfused.
  That is a known optimization target for vtl, not something the stock image already does.

## GPU-host bring-up checklist (H200)

1. `make model-fetch` — full 127.2 GB / 39-shard download into repo-root `hf-model/`
   (resumable; needs ≥140 GB free; `HF_MAX_WORKERS` to tune parallelism).
2. `make vllm-src` — pristine v0.25.0 reference tree at repo-root `vllm/` for
   `vtl/vllm_patches/gen.sh` (its `V025=...` path).
3. Boot ladder:
   a. **Stock boot**: `VTL_DISABLE=1` — proves weights + arch + memory fit before any vtl
      code is in the loop.
   b. **Full vtl boot** + `make verify` — expect
      `rust_sched: AUTHORITY mode active (2 groups` (full-attention + mamba/GDN cache
      groups; `NOT ENGAGED` means you are measuring the stock scheduler).
   c. `make test-kernel` — includes the **first GPU run of the NVRTC path**
      (`bench/test_nvrtc.py` compile+numerics half; only `--self-check` has run off-box).
4. Memory sweep: `max-num-seqs {16, 24, 32}` × `gpu-memory-utilization {0.93, 0.95, 0.97}` —
   record boot success + KV blocks allocated; pick the largest stable config.
5. Baseline `make bench`, **≥3 boots** (boot-to-boot spread is the noise floor).
6. `bench/eval_quality.py` reference run against the baseline server.
7. `make vllm-fork ROUND=round-2 PUSH=1`, then pin the printed digest in
   `round-2/round.mk` (`VLLM_FORK_TAG`).

## Trace status (regenerated on the dev box, 2026-08-16)

`round-2/data/input/trace-round2.jsonl` was regenerated **with the real Qwen3.5 tokenizer**
(meta-only fetch is enough — `build_trace_round2.py` needs only `hf-model/tokenizer.json`
via the `tokenizers` package; no torch/GPU):

    cd round-2 && python3 bench/build_trace_round2.py \
        --model ../hf-model --model-name Qwen3.5-122B-A10B-FP8

Result: 420 records; shared prefix 990 tok identical across all 70 conversations; prompt
tokens min 2,127 / p50 3,024 / p90 4,356 / max 4,363; prefill 1,362,005 tok vs decode
126,000 (11:1); prefix-cache block hit rate 82.8%; required `--max-model-len ≥ 4,663`.
`trace_grading_public.jsonl` is the *input* spec of the generator and is unchanged.
The `body.model` field is now `Qwen3.5-122B-A10B-FP8` — `--served-model-name` in
`docker-compose.yaml` must match or every replayed request 404s.
