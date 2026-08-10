# viettel-llm-inference

An OpenAI-compatible inference server that beats the stock vLLM baseline on a 120-request
trace, scored on **both** latency (TTFT / ITL / E2E percentiles) and throughput (tok/s, req/s).

Rather than reimplementing an engine, we run stock `vllm/vllm-openai` and patch it in-process
through vLLM's [plugin system](https://docs.vllm.ai/en/latest/design/plugin_system/). The
`vtl` package registers a `vllm.general_plugins` entry point; vLLM calls `vtl.plugin:register`
in every process before it does any work.

## Know the workload before tuning it

```
python bench/trace_stats.py
```

The trace is **prefill-bound, 101:1** — 2,414,302 prefill tokens against 24,000 decode tokens,
median prompt 18,707 tokens, `max_tokens=200`, greedy. The conversations are **multi-turn** (1–6
user turns) on top of a byte-identical 6,388-token system prompt, so later turns replay their own
earlier history: the block-level prefix-cache hit rate is **82.4%**, eliminating 1,988,864 prefill
tokens.

This is why the flags look the way they do:

- `--max-model-len=32768` — the longest prompt is 27,331 tokens and needs 27,531 with its
  completion. `16384` rejects most of the trace; `262144` exceeds the model's derived limit and
  vLLM refuses to boot.
- `--enable-prefix-caching` — the single biggest win, and it is lossless. vLLM keys each block on
  `hash(parent_block_hash, block_tokens)`, so the key encodes the whole prefix: a hash-chained
  prefix index, block-granular rather than SGLang's token-level radix tree. The gap is a prefix
  ending mid-block, worth **741 tokens (0.03%)** here — the hit rate is 82.4% at every `block_size`
  from 1 to 64, so don't tune it and don't build a radix tree.
- **Async scheduling (SGLang's "zero-overhead batch scheduler") is on by default** — do *not* pass
  `--async-scheduling`. Passing it turns any incompatibility from a warning into a startup
  `ValueError`. Confirm from the log: `Asynchronous scheduling is enabled.`
- `--quantization=vtl_fp8` — prefill is GEMM-compute-bound, so Hopper's FP8 tensor cores act on
  99% of the tokens.
- Speculative decoding acts only on the decode phase, i.e. ~1% of the tokens. Measured, not assumed.
- **KV offload is off.** The reusable working set is 15.7 GB against a ~120 GB budget — 8x headroom,
  so nothing is ever evicted and there is nothing to promote back. See `docker-compose.offload.yaml`.

## Layout

| Path | What |
|---|---|
| `vtl/` | The plugin. `plugin.py` is the entry point, `registry.py` the patch registry, `patches/` the patches. |
| `bench/` | `trace_stats.py` (workload characterization), `replay.py` (open/closed-loop replay), `metrics.py`, `compare.py`. |
| `Dockerfile` | Bakes the plugin **wheel** into the vLLM image. A bind-mount would not register the entry point. |
| `docker-compose.yaml` | **The submission.** Registry-only: no build context, judge provides `/model`, serves `:8000`. Single source of truth for every serve flag and env var. |
| `docker-compose-optimized.yaml` | Local-dev overlay — swaps the pinned digest for the `:dev` tag. Stacks on top of `docker-compose.yaml`; not standalone. |
| `docker-compose.localtest.yaml` | Local overlay — builds the image and mounts `hf-model/`. Not submitted. |

## Patches

Each patch registers into `PATCH_REGISTRY` under a name and is gated by `VTL_ENABLE_<NAME>`.
`register()` never raises: a patch that fails is logged and skipped, degrading to stock vLLM.
`VTL_DISABLE=1` turns the whole overlay off without rebuilding.

## Develop

```
make check      # self-checks; no GPU, no vLLM, no server needed
make up         # build + run locally against hf-model/ (Linux + GPU)
make warm       # warm the torch.compile/Triton caches on a GPU, bake into the image
make bench      # open-loop replay + closed-loop sweep at 1/8/32/128
make push       # push and print the digest to pin in the compose file
```

Everything except `make check` needs a Linux box with the H200.

**Build on the GPU box.** `vllm/vllm-openai` is multi-arch, so an unpinned `docker build` on an
arm64 Mac silently produces an arm64 image that the amd64 H200 host refuses to run. The
Dockerfile, compose, and `make build/push` all pin `linux/amd64`, so a Mac build is *correct* —
but it runs every `RUN` under qemu emulation and pulls a ~10 GB foreign base. Native is faster.
