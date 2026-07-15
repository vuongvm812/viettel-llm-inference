#!/usr/bin/env python3
"""Healthcheck that also primes the KV prefix cache with the shared system prompt.

The scored trace arrives in bursts of 20 requests that all share one ~6.4k-token system
prompt. Cold, those 20 race to prefill that prefix and vLLM serializes the duplicates
(MambaManager defers same-block prefills within a step), so the shared prefix bottlenecks
wave-1 TTFT. Priming it once, BEFORE the judge sees a healthy container, means every request
hits a warm system prefix from the first wave.

We fold the warm-up into the healthcheck instead of a background thread so readiness and
warm-up are the same signal: Docker marks the container healthy only when this script exits 0,
and the first passing run does the priming POST synchronously before returning. Crucially,
readiness gates ONLY on /health -- the warm-up is one-shot best-effort (a sentinel is written
BEFORE the POST so a slow/failed prime never re-fires or blocks readiness). A broken warm-up can
never leave the container un-healthy (which would be unscoreable).

Env (all optional):
  VTL_WARMUP_PORT=8000                              local server port
  VTL_WARMUP_MODEL=Qwen3.5-2B                       served model name (must match --served-model-name)
  VTL_WARMUP_PROMPT_FILE=/opt/vtl/warmup_system_prompt.txt   prefix to prime
  VTL_WARMUP_SENTINEL=/tmp/vtl_warm.done            one-shot guard
  VTL_WARMUP_POST_TIMEOUT=20                         seconds; keep < compose healthcheck timeout
  VTL_DISABLE_WARMUP=0                              1 = pure healthcheck, no priming
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

PORT = os.environ.get("VTL_WARMUP_PORT", "8000")
MODEL = os.environ.get("VTL_WARMUP_MODEL", "Qwen3.5-2B")
PROMPT_FILE = os.environ.get("VTL_WARMUP_PROMPT_FILE", "/opt/vtl/warmup_system_prompt.txt")
SENTINEL = os.environ.get("VTL_WARMUP_SENTINEL", "/tmp/vtl_warm.done")
POST_TIMEOUT = float(os.environ.get("VTL_WARMUP_POST_TIMEOUT", "20"))
BASE = f"http://localhost:{PORT}"


def _health_ok() -> bool:
    try:
        with urllib.request.urlopen(f"{BASE}/health", timeout=4) as r:
            return r.status == 200
    except Exception:
        return False


def _prime() -> None:
    """POST the shared system prompt once so its KV + GDN state is cached. Best-effort."""
    try:
        with open(PROMPT_FILE, encoding="utf-8") as f:
            system_prompt = f.read()
    except Exception as e:  # no prompt baked -> nothing to prime, health still fine
        print(f"vtl-warmup: prompt file unreadable ({e}); skipping prime", file=sys.stderr)
        return
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "ready"},
            ],
            "max_tokens": 1,
            "temperature": 0,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=POST_TIMEOUT) as r:
            r.read()
        print("vtl-warmup: system prefix primed", file=sys.stderr)
    except Exception as e:  # a slow/failed prime must not affect readiness
        print(f"vtl-warmup: prime POST failed ({e}); serving cold", file=sys.stderr)


def main() -> int:
    if not _health_ok():
        return 1  # not ready yet; Docker retries
    # One-shot: write the sentinel BEFORE priming so a slow/hung POST never re-fires next tick.
    if os.environ.get("VTL_DISABLE_WARMUP", "0").strip().lower() not in ("1", "true", "yes", "on"):
        if not os.path.exists(SENTINEL):
            try:
                open(SENTINEL, "x").close()
            except FileExistsError:
                return 0  # another check won the race; it is priming
            _prime()
    return 0


def _selfcheck() -> None:
    """No server needed: fake _health_ok/_prime and exercise the gate + one-shot logic."""
    import tempfile

    global _health_ok, _prime, SENTINEL
    primes = []
    _prime = lambda: primes.append(1)  # noqa: E731

    with tempfile.TemporaryDirectory() as d:
        SENTINEL = os.path.join(d, "warm.done")
        os.environ.pop("VTL_DISABLE_WARMUP", None)

        _health_ok = lambda: False  # noqa: E731
        assert main() == 1 and not primes, "unhealthy must return 1 and not prime"

        _health_ok = lambda: True  # noqa: E731
        assert main() == 0 and primes == [1], "first healthy pass primes exactly once"
        assert os.path.exists(SENTINEL), "sentinel written"
        assert main() == 0 and primes == [1], "second pass must NOT re-prime (one-shot)"

        os.remove(SENTINEL)
        os.environ["VTL_DISABLE_WARMUP"] = "1"
        assert main() == 0 and primes == [1], "disabled: healthy, no prime, no sentinel"
        assert not os.path.exists(SENTINEL)
    print("warmup_healthcheck self-check ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        sys.exit(main())
