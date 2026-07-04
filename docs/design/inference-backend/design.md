# Design — Inference Backend (llama.cpp via FFI)

The GPU compute layer. We do **not** reimplement the transformer; `libllama` runs the forward
pass. Our runtime owns scheduling/batching around it. Crate: **`llama-cpp-2`** (safe wrapper
over `libllama`), **Qwen3.5-2B** dense transformer, **BF16 (unquantized)** — converted from the
HF safetensors to a BF16 GGUF (`convert_hf_to_gguf.py --outtype bf16`; see `script/setup.sh`).

> Exact method names below track the `llama.cpp` C API; confirm the `llama-cpp-2` version's
> spelling at implementation time (the crate renames some C functions). The *mechanisms* are stable.

## Initialization (startup, before cores spin up)

```rust
let backend = LlamaBackend::init();                       // once per process
let model = Arc::new(LlamaModel::load_from_file(
    &backend, gguf_path, model_params
        .with_n_gpu_layers(ALL)                           // full offload → GPU does compute
));
let ctx = model.new_context(&backend, context_params
    .with_n_ctx(N_CTX)                                    // KV capacity; bounds concurrency
    .with_n_batch(N_BATCH)                                // max tokens per decode call
    .with_n_threads(1)                                    // GPU does the work; don't fight our 3 cores
    .with_n_seq_max(MAX_BATCH_SEQS));                     // concurrent sequences
```

- `model` → shared `Arc<LlamaModel>` for Core 1 (vocab) and Core 2 (context).
- `ctx` → owned exclusively by **Core 2**. `LlamaContext` is single-threaded.
- `n_gpu_layers = ALL`: everything on GPU, matching vLLM. CPU stays free for the 3 pinned cores.

## Send/Sync boundary (the one subtle FFI point)

| Handle | Owner | Sharing | Why safe |
|---|---|---|---|
| `LlamaModel` | shared | `Arc`, read by Core 1 + Core 2 | vocab + weights are immutable after load |
| `LlamaContext` | Core 2 only | never shared | KV cache is mutable; single-writer |
| `LlamaBackend` | process | init once, held for lifetime | global init |

If `llama-cpp-2` doesn't mark `LlamaModel: Sync`, wrap in an audited newtype asserting the
read-only-vocab invariant. Never share `LlamaContext`.

## Tokenize / detokenize (called from Core 1)

- `model.str_to_token(text, AddBos::...)` → `Vec<LlamaToken>` (tokenize prompt).
- `model.token_to_bytes(token, Special::...)` / `token_to_str` → piece bytes (detokenize).
- Both read only the vocab → safe concurrently with Core 2's `decode()`.

## Batch build + decode (Core 2, per iteration)

- `LlamaBatch::new(n_batch, n_seq_max)`; per active seq add its next input token:
  `batch.add(token, pos, &[seq_id], /*logits=*/true)`.
- `ctx.decode(&mut batch)` → runs the GPU forward for all sequences in the batch at once
  (this is where continuous batching pays off).
- `ctx.candidates_ith(i)` / `ctx.get_logits_ith(i)` → logits for the i-th requested position,
  fed to the sampler.

## KV cache & shared-prefix ops (the parity feature)

llama.cpp's KV cache is cell-based and sequence-aware — this is what makes shared-prefix
caching a few calls instead of a subsystem:

- **Prefill prefix once** into a reserved `prefix_seq` (decode the 39K system-prompt tokens).
- **Share**: `ctx.kv_cache_seq_cp(prefix_seq, new_seq, p0=0, p1=prefix_len)` — new seq reuses
  the prefix's KV cells; no recompute.
- **Prefill suffix**: decode only the user-content tokens for `new_seq` at `pos = prefix_len`.
- **Free on finish**: `ctx.kv_cache_seq_rm(seq_id, p0=-1, p1=-1)` — reclaim all cells for a
  finished seq.
- **Capacity/monitor**: track used vs `n_ctx` cells to gate admission (fast-loop `KvMonitor`).
  (If the crate exposes KV cell counts, use them; else account manually from admitted tokens.)

> Confirm the crate's exact spelling: recent `llama.cpp` renamed `llama_kv_cache_seq_*` to a
> `llama_kv_self_*` / memory API; `llama-cpp-2` may expose either. The copy/remove/capacity
> semantics are what we depend on.

## Sampling

- Deterministic path (trace: `temp=0`): greedy argmax over logits → reproduces vLLM output.
- General path: `LlamaSampler` chain (top-k/top-p/temp) seeded per seq (`seed=42`) for
  reproducibility. Per-seq sampler state (no cross-request coupling).

## Sizing & constraints

- `N_CTX` is the total KV budget shared across all concurrent sequences (llama.cpp cells are
  global, not per-seq). With ~40K-token prompts, `N_CTX` and `MAX_BATCH_SEQS` directly bound
  how many requests run at once — the same KV-memory ceiling vLLM hits. Tune on the GPU.
- Shared prefix dramatically lowers effective per-request KV (the 39K prefix is stored once).

## Build/link notes

- `llama-cpp-2` links `libllama` (CUDA on Linux, Metal on macOS) — needs the accelerator
  toolchain + the BF16 GGUF on the deploy box. The GGUF is a **lossless BF16** conversion of
  the HF weights (not a lower-bit quant), so it matches vLLM's BF16 precision — good for
  benchmark fairness; note the conversion in the report. (Metal's BF16 support is newer than
  CUDA's — confirm the kernels run BF16 rather than silently upcasting on the mac dev box.)
- `libllama` is a C++ dependency; our PGO/BOLT optimizes **our Rust binary**, not `libllama`
  (its own tuning is P7).

## Correctness

- Determinism check (P2 exit): one trace request through the backend equals a direct
  `llama-cli` run with the same GGUF + `temp=0` + `seed=42`.
- KV leak check: after N requests, used KV cells return to the prefix-only baseline (every
  `seq_rm` fires).
