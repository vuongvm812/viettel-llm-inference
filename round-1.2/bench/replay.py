"""Open-loop trace replayer for OpenAI-compatible /v1/chat/completions endpoints.

Mirrors vLLM's benchmark_serving.py: replays data/input/trace-round1.jsonl against a
target, honoring per-record arrival times (open-loop — a slow server builds a real
queue), and streams SSE to stamp TTFT / ITL / E2E / output tokens per request.

    python bench/replay.py --target http://localhost:8001 --out ours.json
    python bench/replay.py --target http://localhost:8000 --out vllm.json   # vLLM
    python bench/replay.py --target http://localhost:8001 --closed-loop 4 --out cl.json

See docs/design/benchmark/design.md.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import aiohttp

from metrics import aggregate

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACE = REPO_ROOT / "data" / "input" / "trace-round2.jsonl"


def load_trace(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


async def do_request(session, url, rec, t0):
    """POST one request, parse the SSE stream, return a measurement record."""
    loop = asyncio.get_running_loop()
    send = loop.time()  # arrival proxy: in open-loop we sleep until arrival before sending
    result = {
        "request_id": rec.get("request_id"),
        "workload_type": rec.get("workload_type"),
        "send_time": send - t0,
        "success": False,
        "status": None,
        "error": None,
        "ttft": None,
        "e2e": None,
        "itl_mean": None,
        "output_tokens": 0,
    }

    try:
        body = dict(rec["body"])
        body["stream"] = True  # required for TTFT/ITL; our mock streams regardless
        async with session.post(f"{url}/v1/chat/completions", json=body) as resp:
            result["status"] = resp.status
            if resp.status != 200:
                result["error"] = (await resp.text())[:200]
                return result

            first_tok = last_tok = None
            n_tokens = 0
            usage_tokens = None
            saw_done = False
            async for raw in resp.content:  # StreamReader iterates line by line
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    saw_done = True
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    result["error"] = "malformed SSE chunk"
                    return result
                if not isinstance(obj, dict):
                    continue  # ignore non-object data lines (e.g. `data: 42`)
                usage = obj.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    usage_tokens = usage["completion_tokens"]
                choices = obj.get("choices") or []
                content = (choices[0].get("delta") or {}).get("content") if choices else None
                if content:
                    now = loop.time()
                    if first_tok is None:
                        first_tok = now
                    last_tok = now
                    n_tokens += 1

        done = loop.time()
        # Count streamed deltas for parity across targets; trust usage only if given.
        out_tokens = usage_tokens if usage_tokens is not None else n_tokens
        result["output_tokens"] = out_tokens
        result["token_basis"] = "usage" if usage_tokens is not None else "chunks"
        result["e2e"] = done - send
        if first_tok is not None:
            result["ttft"] = first_tok - send
        if n_tokens > 1:
            result["itl_mean"] = (last_tok - first_tok) / (n_tokens - 1)
        # A broken target must not masquerade as a good run: require a clean [DONE]
        # and at least one token, else record it as a failure.
        if not saw_done:
            result["error"] = "stream ended without [DONE]"
        elif out_tokens == 0:
            result["error"] = "no output tokens (non-SSE or empty response?)"
        else:
            result["success"] = True
    except asyncio.TimeoutError:
        result["error"] = "timeout"
    except aiohttp.ClientError as e:
        result["error"] = f"{type(e).__name__}: {e}"
    except Exception as e:  # never let one request abort the whole gather
        result["error"] = f"{type(e).__name__}: {e}"
    return result


async def run_open_loop(session, url, records):
    loop = asyncio.get_running_loop()
    t0 = loop.time()

    async def fire(rec):
        delay = rec.get("timestamp_ms", 0) / 1000 - (loop.time() - t0)
        if delay > 0:
            await asyncio.sleep(delay)
        return await do_request(session, url, rec, t0)

    results = await asyncio.gather(*(fire(r) for r in records))
    return results, loop.time() - t0


async def run_closed_loop(session, url, records, concurrency):
    """N workers, send-on-completion — saturation sweep (secondary mode)."""
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    queue = asyncio.Queue()
    for r in records:
        queue.put_nowait(r)
    results = []

    async def worker():
        while True:
            try:
                rec = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            results.append(await do_request(session, url, rec, t0))

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return results, loop.time() - t0


async def main_async(args):
    records = load_trace(args.trace)
    if args.limit:
        records = records[: args.limit]
    mode = f"closed-loop:{args.closed_loop}" if args.closed_loop else "open-loop"
    print(f"replaying {len(records)} requests → {args.target} ({mode})", file=sys.stderr)

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=args.timeout)
    connector = aiohttp.TCPConnector(limit=0)  # unlimited concurrent conns (open-loop bursts)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        if args.closed_loop:
            results, wall = await run_closed_loop(session, args.target, records, args.closed_loop)
        else:
            results, wall = await run_open_loop(session, args.target, records)

    run = {
        "target": args.target,
        "trace": str(args.trace),
        "mode": mode,
        "wall_time": wall,
        "records": results,
    }
    Path(args.out).write_text(json.dumps(run, indent=2))

    agg = aggregate(results, wall)
    print(
        f"done: {agg['n_success']}/{agg['n_total']} ok, {agg['n_error']} err, "
        f"{agg['output_tok_s']:.1f} tok/s, {agg['req_s']:.2f} req/s, "
        f"TTFT p50 {agg['ttft_ms']['p50'] or 0:.0f}ms → {args.out}",
        file=sys.stderr,
    )


def main():
    p = argparse.ArgumentParser(description="Replay a trace against an OpenAI-compatible target.")
    p.add_argument("--target", required=True, help="base URL, e.g. http://localhost:8001")
    p.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    p.add_argument("--out", required=True, help="output JSON path")
    p.add_argument("--closed-loop", type=int, metavar="N",
                   help="closed-loop with N concurrent workers (default: open-loop)")
    p.add_argument("--limit", type=int, help="only replay the first N records (smoke test)")
    p.add_argument("--timeout", type=float, default=600, help="per-request socket read timeout (s)")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
