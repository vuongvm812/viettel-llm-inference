"""Warmup script: trigger torch.compile / Triton / FlashInfer JIT + CUDA graph
capture before the judge sends traffic. Runs inside the container at startup,
after vLLM's /health is up.

Sends shape-covering requests matching the trace's prompt distribution:
  - 4 sequential requests at ~100 / ~6400 / ~18000 / ~27000 tokens (max_tokens=2)
  - 20 concurrent requests at ~6400 tokens (max_tokens=2)

torch.compile keys off tensor shapes, not content, so synthetic prompts suffice.
Errors are logged and ignored -- warmup is best-effort.
"""

import concurrent.futures
import json
import sys
import urllib.request

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen3.5-2B"
TIMEOUT = 120


def _make_prompt(target_chars: int) -> str:
    word = "context buffer process queue layer "
    return word * (target_chars // len(word) + 1)


def _send(messages: list[dict], max_tokens: int = 2) -> bool:
    body = json.dumps(
        {
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 42,
        }
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            resp.read()
        return True
    except Exception as exc:
        print(f"  warmup error: {exc}", file=sys.stderr)
        return False


def main() -> None:
    # ~1.3 chars/token: cover the trace's prompt-length distribution.
    sizes = [
        (130, "small (~100 tok)"),
        (8300, "sys-prompt (~6400 tok)"),
        (23400, "p50 (~18000 tok)"),
        (35100, "max (~27000 tok)"),
    ]

    for chars, label in sizes:
        text = _make_prompt(chars)
        ok = _send([{"role": "user", "content": text}])
        print(f"  warmup {label}: {'ok' if ok else 'failed'}", file=sys.stderr)

    # Batch of 20 concurrent at system-prompt size -- triggers batched-prefill
    # graph + decode CUDA graph capture at the burst concurrency level.
    text = _make_prompt(8300)
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
        futures = [
            pool.submit(_send, [{"role": "user", "content": text}])
            for _ in range(20)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    n_ok = sum(results)
    print(f"  warmup batch-20: {n_ok}/20 ok", file=sys.stderr)

    print("warmup complete", file=sys.stderr)


if __name__ == "__main__":
    main()
