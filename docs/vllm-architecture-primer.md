# vLLM architecture primer: the machine we spent three rounds modifying

Companion to [`tech-talk-hackathon-rounds.md`](tech-talk-hackathon-rounds.md) §2.
Structure and terminology follow the vLLM team's
["Anatomy of vLLM"](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) — read that for
the full walkthrough; this file keeps only what the talk builds on, and points each
concept at the place in our stack where we tuned, forked, or replaced it.

## The process split

```
client ──HTTP──▶ frontend (OpenAI-compatible API server / AsyncLLM)
                   parse · tokenize · stream SSE · detokenize
                        │  ZMQ (msgpack frames)
                        ▼
                 EngineCore process
                   ┌──────────────────────────────────────────┐
                   │ loop (one iteration = one "step"):       │
                   │   scheduler.schedule()                   │
                   │   ──▶ model executor / GPU model runner  │
                   │       (build inputs · forward · sample)  │
                   │   update_from_output()                   │
                   └──────────────────────────────────────────┘
```

- The **frontend** and the **EngineCore** are separate processes joined by ZMQ.
  A "step" is one forward pass of the whole in-flight batch.
- **Where we lived on this seam:** the Rust frontend (vLLM's own `vllm-rs`, plus our
  source patches — sonic-rs parsing, iceoryx2 shared-memory IPC replacing ZMQ on the
  data plane, per-token SSE) and the `vtl-sched` scheduler port both sit exactly on
  this boundary (talk §4a, §4c).

## Scheduler + continuous batching

- The scheduler keeps **waiting** and **running** queues and decides *which requests
  go into the next engine step*, prioritizing decodes already in flight. Requests
  join and leave the batch **every step** — that is continuous batching; there is no
  "wait for the batch to finish."
- Each request has two phases: **prefill** (process the whole prompt, produce the
  first token → sets **TTFT**) and **decode** (one token per step → sets **TPOT**).
  These are the two numbers ERS scores, which is why the scheduler's decisions are
  score decisions.
- **Where we lived here:** `--max-num-seqs`, `--max-num-batched-tokens` sweeps (talk
  §3); the ~8,000-line logic-preserving Rust port of `schedule()` and its speculative
  precompute (talk §4c); the round-2 mixed-prefill admission cap A/B (talk §5).

## Paged KV cache → prefix caching, chunked prefill

- Attention needs every past token's K/V. vLLM stores them in fixed-size **blocks**
  (default 16 tokens); the **KV-cache manager** owns a `free_block_queue` pool sized
  to fit VRAM (`--gpu-memory-utilization`), and per-request **block tables** map
  token positions → blocks (PagedAttention). No contiguous per-request allocation,
  no fragmentation.
- **Prefix caching** falls out of the block structure: blocks are content-hashed, so
  a prompt sharing a prefix with an earlier one reuses those KV blocks instead of
  recomputing — "the single biggest win, and it is lossless" in round 1.1 (82.4%
  block hit rate; talk §3).
- **Chunked prefill** splits a long prompt across several steps so one prefill can't
  monopolize a step and spike every in-flight decode's latency — with γ=2, exactly
  the tail ERS punishes hardest (talk §3, §5).
- **Where we lived here:** the Rust block pool / KV-cache manager / prefix cache port,
  including the bit-for-bit pickle-protocol-5 + SHA-256 prefix hash (a silently
  different key space zeroes an 82% hit rate; talk §4c); the N-step burst's align
  gate, which exists precisely to keep a burst inside one KV block (talk §4c).

## GPU model runner + CUDA graphs

- The **model runner** does the per-step host work: gather the step's token IDs and
  positions, write block tables, launch the forward, sample. All of it is Python,
  and it repeats **every step** — so on fast decode it is pure per-token overhead.
- **CUDA graphs**: launching hundreds of kernels from Python each step costs more
  than the kernels themselves at small batch sizes. vLLM pre-captures the decode
  forward per batch size and **replays** the whole sequence as one graph launch.
  Shapes without a captured graph fall back to eager — silently slower (17% of our
  round-1.2 scored tokens decoded outside the captured set; talk §6).
- **Where we lived here:** this is the load-bearing box of the whole talk. On the
  16-SM MIG slice, decode was ~1 ms GPU + ~2 ms host — "the host IS the TPOT" — which
  motivated the hot-path patches (§4a), the N-step burst (§4c), and finally the
  round-2 Rust CUDA-graph runner that moves the launch loop itself out of Python via
  `raw_cuda_graph_exec()` + `cuGraphLaunch` (§4c).

## Noticeable concepts, one line each

- **Speculative decoding** — a cheap draft proposes tokens, the main model verifies
  with accept/reject; several tokens per forward pass when accepts run hot. The
  ngram/prompt-lookup variant is output-identical at temp 0 — which is exactly why
  the judges banned it as cheating, and why our N-step burst deliberately configures
  **nothing spec-decode-shaped** while reusing the multi-token plumbing (talk §4c).
- **Quantization** — weights and/or KV in lower precision (our fp8 weights +
  `--kv-cache-dtype=fp8_e4m3` in round 1.1; online-RTN W4A8 into vLLM's SM90 CUTLASS
  kernel in round 1.2): less HBM traffic and more tensor-core throughput, priced in
  accuracy and in kernel complexity (talk §3, §4b).
- **The memory/batching knobs** — `--max-model-len` (per-request KV reservation;
  size it to the workload, not the model card), `--max-num-seqs` (batch width),
  `--max-num-batched-tokens` (prefill chunk size): the three levers the vanilla
  phase swept, each with a measured verdict in the compose ledger (talk §3).
- **Beyond one GPU** (not used by us — round hardware was a single GPU or slice):
  tensor parallelism via `MultiProcExecutor` worker processes, and disaggregated
  prefill/decode instances joined by KV-transfer connectors.
