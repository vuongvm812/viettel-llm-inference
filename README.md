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

**101:1 is a token ratio, not a time ratio.** A roofline on the real shapes (36 layers, 16 query
heads over 2 KV heads, 18 KB/token of fp8 KV) puts a burst's prefill near 1.0 s and its 200 decode
steps near 0.43 s — decode is roughly **30% of GPU time**, and 1.44 ms of each 2.14 ms decode step
is spent reading KV. Decode is memory-bound, not compute-bound. That is what makes speculative
decoding worth having: the extra draft query rows ride along on KV reads already being paid for.

**Arrival times matter as much as token counts.** `timestamp_ms = turn*5000 + conv*25`, so the 120
requests are 20 conversations × 6 turns arriving as **six bursts of 20, five seconds apart**, over
25.5 s. Peak concurrency is **20** and the GPU idles ~80% of the wall clock. Throughput across the
trace is therefore *arrival-bound* — only the tail can move it. Optimize latency and throughput
follows; never trade latency for batch size here.

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
- `--speculative-config={"method":"ngram_gpu",...}` — decode is ~30% of GPU time (see above), and
  it is KV-bandwidth-bound, so drafting is nearly free. Under greedy + standard rejection sampling
  an accepted draft must equal the argmax token, so **output is bit-identical** to non-speculative
  decode. It **must** be `ngram_gpu`: `method="ngram"` silently *disables async scheduling* (vLLM
  warns and continues), which is exactly the overhead-hiding we need on the judge's 3-core box.
  `make verify` asserts both. Drop the flag if `Mean acceptance length` (which counts the bonus
  token, so a useless drafter reads `1.00`) comes in under ~1.10.
- `--compilation-config={"pass_config":{"fuse_attn_quant":true}}` — vLLM hardcodes
  `IS_QUANTIZED = False`, which force-disables attention+fp8-quant fusion for *every* model
  (upstream #25689). We are fp8, so we turn it back on. It merges per-field with the `-O2` defaults.
- **Do not pass `cudagraph_mode`.** `optimization_level` defaults to `-O2`, which already resolves
  it to `FULL_AND_PIECEWISE`. Setting it explicitly measures as a no-op.
- **Do not pass `--api-server-count`.** It is read only by the `vllm serve` entrypoint; under the
  pinned `python3 -m vllm.entrypoints.openai.api_server` it parses and is ignored.
- **Cascade attention is not used.** `disable_cascade_attn` defaults to `True` in 0.22.1, and vLLM's
  own perf model only picks cascade above ~65 concurrent decodes — we peak at 20, where
  FlashDecoding's extra CTAs win. Speculative decoding force-disables it anyway.
- **KV offload is off.** The reusable working set is 15.7 GB against a ~120 GB budget — 8x headroom,
  so nothing is ever evicted and there is nothing to promote back. See `docker-compose.offload.yaml`.

## Layout

| Path | What |
|---|---|
| `vtl/` | The plugin. `plugin.py` is the entry point, `registry.py` the patch registry, `patches/` the patches. |
| `bench/` | `trace_stats.py` (workload characterization), `replay.py` (open/closed-loop replay), `metrics.py`, `compare.py`. |
| `Dockerfile` | Bakes the plugin **wheel** into the vLLM image. A bind-mount would not register the entry point. |
| `docker-compose-optimized.yaml` | The submission. Registry-only: no build context, judge provides `/model`, serves `:8000`. |
| `docker-compose.localtest.yaml` | Local overlay — builds the image and mounts `hf-model/`. Not submitted. |

## Patches

Each patch registers into `PATCH_REGISTRY` under a name and is gated by `VTL_ENABLE_<NAME>`.
`register()` never raises: a patch that fails is logged and skipped, degrading to stock vLLM.
`VTL_DISABLE=1` turns the whole overlay off without rebuilding.

## Develop

```
make check      # self-checks; no GPU, no vLLM, no server needed
make up         # build + run locally against hf-model/ (Linux + GPU)
make verify     # post-boot assertions against a live container -- run this after `make up`
make warm       # warm the torch.compile/Triton caches on a GPU, bake into the image
make bench      # open-loop replay + closed-loop sweep at 1/8/32/128
make push       # push and print the digest to pin in the compose file
```

`make up` exports `VTL_GPU_MEM_UTIL` / `VTL_MAX_NUM_SEQS` / `VTL_KV_DTYPE` to fit a small dev
card; unset (the judge's case) they resolve to the H200 values baked into the compose file. On a
pre-Hopper card `vtl_fp8` falls back to Marlin and `fuse_attn_quant` is inert, so **latency and
throughput measured there mean nothing** — `make verify` prints `WARN` when that happens. What a
dev card *can* prove: the server boots, async scheduling survives spec decode, greedy output is
unchanged, and the n-gram acceptance rate (which depends only on the tokens, not the GPU).

`make warm` deliberately does *not* use those overrides: compile cache keys include device
capability and the cudagraph capture sizes, so it must run on the H200 with submission defaults.

Everything except `make check` needs a Linux box with the H200.

**Build on the GPU box.** `vllm/vllm-openai` is multi-arch, so an unpinned `docker build` on an
arm64 Mac silently produces an arm64 image that the amd64 H200 host refuses to run. The
Dockerfile, compose, and `make build/push` all pin `linux/amd64`, so a Mac build is *correct* —
but it runs every `RUN` under qemu emulation and pulls a ~10 GB foreign base. Native is faster.
