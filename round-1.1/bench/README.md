# Benchmark harness (P5)

Head-to-head latency/throughput of **our runtime vs vLLM** on the same 120-request trace
(`data/input/trace-round1.jsonl`). Open-loop replay honoring per-record arrival times, SSE
streaming for TTFT/ITL. Spec: [`docs/design/benchmark/design.md`](../docs/design/benchmark/design.md).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate   # optional
pip install -r bench/requirements.txt                # aiohttp only
```

## Run

```bash
# our runtime (port 8001 per config/default-config.yaml)
make run/inference                      # in another shell — starts our server
python bench/replay.py --target http://localhost:8001 --out ours.json

# vLLM baseline (port 8000) — Linux + GPU only
docker-compose up                       # in another shell — starts vLLM
python bench/replay.py --target http://localhost:8000 --out vllm.json

# side-by-side table + fairness notes
python bench/compare.py vllm.json ours.json
```

Extra flags: `--closed-loop N` (saturation sweep, N concurrent, send-on-completion),
`--limit N` (first N records, smoke test), `--trace PATH`, `--timeout SECONDS`.

## Notes

- **vLLM needs a Linux box with a GPU** — it cannot run in the macOS dev sandbox. The replayer
  is target-agnostic and is verified locally against our runtime; the actual vLLM side-by-side
  run is a target-box task.
- Our runtime is the **P1 mock** until P2+ lands — its latency/token numbers are placeholders,
  not real-model output. `compare.py` prints this caveat when a `:8001` target is present.
- Output tokens are counted from streamed content deltas (uniform across targets); `usage` is
  used only if the stream provides it.
- Self-check the reporting math: `python bench/metrics.py`.
