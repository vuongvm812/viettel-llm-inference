# Design — Benchmark Harness

Head-to-head latency/throughput of **our runtime vs vLLM** on the same trace. Python asyncio +
aiohttp (mirrors vLLM's own `benchmark_serving.py` — the standard for LLM serving benchmarks).
One script, two targets.

## Inputs

- Trace: `data/input/trace-round1.jsonl` — 120 records, each
  `{request_id, timestamp_ms, workload_type, body}` where `body` is a ready-to-POST OpenAI
  chat-completions payload (shared ~39K-char system prompt, `max_tokens=200`, `temp=0, seed=42`).
- Targets: vLLM at `http://localhost:8000` (from `docker-compose.yml`), our runtime at its port.
- Endpoint: `/v1/chat/completions` with `"stream": true` (needed for TTFT/ITL).

## Load model: open-loop, honor arrival times

Replay respects `timestamp_ms` (relative arrivals ~25ms apart ≈ 40 req/s) — **open-loop**, so
a slow server builds a real queue (the honest way to compare throughput under load).

```python
t0 = loop.time()
async def fire(rec):
    await asyncio.sleep(max(0, rec["timestamp_ms"]/1000 - (loop.time() - t0)))
    await do_request(session, rec)           # schedule at arrival, don't wait for prior replies
await asyncio.gather(*(fire(r) for r in records))
```

- **Do not** wait for a response before sending the next request (that would be closed-loop and
  hide queueing). Each request is fired at its arrival time regardless of in-flight ones.
- Also support a `--closed-loop N` mode (N concurrent, send-on-completion) for a saturation
  sweep — secondary; open-loop trace replay is the primary comparison.

## Per-request measurement (streaming)

Parse the SSE stream and stamp:

- **TTFT** — arrival → first token chunk.
- **ITL / TPOT** — inter-token latency; mean = (last_token_time − first_token_time)/(n_out−1).
- **E2E latency** — arrival → final chunk (`[DONE]`).
- **output tokens** — count streamed deltas (and/or trust `usage` if present).
- **success/error** — HTTP status, timeouts, malformed stream.

## Aggregate metrics (report)

- Latency percentiles p50/p90/p95/p99 + mean for TTFT, ITL, E2E.
- **Throughput**: output tok/s (total generated / wall time), request/s (completed / wall time).
- Under open-loop: also **queue delay** (arrival → first server activity, approximated by TTFT
  minus a warm baseline).
- Errors / incomplete streams.

Emit both a human table and machine JSON:

```
                    vLLM (:8000)     ours (:PORT)
TTFT p50 / p99        …ms / …ms        …ms / …ms
ITL  p50 / p99        …ms / …ms        …ms / …ms
E2E  p50 / p99        …ms / …ms        …ms / …ms
output tok/s          …               …
req/s                 …               …
```

## Fairness notes (must appear in the report)

- **Weights/quantization differ**: vLLM serves the full/original format; we serve GGUF-quantized
  Qwen3.5-2B. Note the quantization level — it affects both speed and output. For an
  apples-to-apples quality check, compare a few outputs, not just latency.
- **Prefix caching**: vLLM has `--enable-prefix-caching`; our v1 (P4) has shared-prefix KV, so
  both compute the 39K system prompt once. If benchmarking before P4, flag that vLLM has the
  prefix-cache edge.
- Same trace, same arrival schedule, same `max_tokens/temp/seed` for both → the controlled part.

## Usage

```
python bench/replay.py --target http://localhost:8000 --trace data/input/trace-round1.jsonl --out vllm.json
python bench/replay.py --target http://localhost:PORT --trace data/input/trace-round1.jsonl --out ours.json
python bench/compare.py vllm.json ours.json         # prints the table
```

## Reuse as PGO/BOLT driver

The same replay drives training-data collection for PGO/BOLT (`build-optimization/design.md`):
point `--target` at the instrumented build and replay the trace to exercise the hot paths.

## Dependencies

`aiohttp` only (stdlib `asyncio`, `json`, `statistics`). No heavy deps; a `requirements.txt`
with `aiohttp` suffices. Keep percentiles in-process (`statistics.quantiles` or a tiny
hand-rolled percentile) — no need for a metrics framework.
