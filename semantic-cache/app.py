"""Tier 3 — semantic cache proxy in front of vLLM (no fork).

Embeds the prompt, returns a stored completion on a near-duplicate hit,
otherwise forwards to vLLM and caches the reply. OpenAI-compatible passthrough
for everything else. Only helps when the workload has repeated/similar prompts;
on unique prompts it only adds embedding latency — set CACHE_ENABLED=0 then.

Env: UPSTREAM, SIM_THRESHOLD, MAX_ENTRIES, CACHE_ENABLED, EMBED_MODEL
"""
import os

import httpx
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

UPSTREAM = os.environ.get("UPSTREAM", "http://localhost:8000")
THRESHOLD = float(os.environ.get("SIM_THRESHOLD", "0.95"))
MAX_ENTRIES = int(os.environ.get("MAX_ENTRIES", "2000"))
ENABLED = os.environ.get("CACHE_ENABLED", "1") == "1"
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

app = FastAPI()
_client = httpx.AsyncClient(timeout=None)

# in-memory semantic cache: parallel arrays of unit-norm embeddings + responses.
# ponytail: O(n) cosine scan, in-process, FIFO evict. Swap for FAISS/Redis if
# entries or QPS outgrow a single process.
_emb: list = []
_resp: list = []
_embedder = None


def embed(text: str) -> np.ndarray:
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(model_name=EMBED_MODEL)
    v = np.asarray(list(_embedder.embed([text]))[0], dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n else v


def key_text(body: dict) -> str:
    return "\n".join(f"{m.get('role')}:{m.get('content')}" for m in body.get("messages") or [])


def lookup(vec: np.ndarray):
    if not _emb:
        return None
    sims = np.array([float(vec @ e) for e in _emb])
    i = int(sims.argmax())
    return _resp[i] if sims[i] >= THRESHOLD else None


def store(vec: np.ndarray, resp: dict):
    if len(_emb) >= MAX_ENTRIES:
        _emb.pop(0)
        _resp.pop(0)
    _emb.append(vec)
    _resp.append(resp)


@app.get("/health")
async def health():
    try:
        r = await _client.get(f"{UPSTREAM}/health")
        return JSONResponse({"status": "ok", "upstream": r.status_code})
    except Exception as e:
        return JSONResponse({"status": "degraded", "error": str(e)}, status_code=503)


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    # streaming or disabled → transparent proxy, no caching
    if body.get("stream") or not ENABLED:
        return await _proxy_chat(body, stream=bool(body.get("stream")))
    vec = embed(key_text(body))
    hit = lookup(vec)
    if hit is not None:
        return JSONResponse({**hit, "cached": True})
    r = await _client.post(f"{UPSTREAM}/v1/chat/completions", json=body)
    data = r.json()
    if r.status_code == 200:
        store(vec, data)
    return JSONResponse(data, status_code=r.status_code)


async def _proxy_chat(body: dict, stream: bool):
    if not stream:
        r = await _client.post(f"{UPSTREAM}/v1/chat/completions", json=body)
        return JSONResponse(r.json(), status_code=r.status_code)

    async def gen():
        async with _client.stream("POST", f"{UPSTREAM}/v1/chat/completions", json=body) as up:
            async for chunk in up.aiter_raw():
                yield chunk

    return StreamingResponse(gen(), media_type="text/event-stream")


# everything else (/v1/models, /v1/completions, ...) → transparent passthrough
@app.api_route("/{path:path}", methods=["GET", "POST"])
async def passthrough(path: str, request: Request):
    r = await _client.request(
        request.method, f"{UPSTREAM}/{path}",
        content=await request.body(),
        headers={"content-type": request.headers.get("content-type", "application/json")},
    )
    try:
        return JSONResponse(r.json(), status_code=r.status_code)
    except Exception:
        return JSONResponse({"raw": r.text}, status_code=r.status_code)


if __name__ == "__main__":
    # ponytail self-check: cosine hit/miss + FIFO cap, no network/model needed
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    store(a, {"id": "A"})
    assert lookup(a) == {"id": "A"}, "exact match must hit"
    assert lookup(b) is None, "orthogonal must miss at threshold"
    for i in range(MAX_ENTRIES + 5):
        store(a, {"id": i})
    assert len(_emb) == MAX_ENTRIES, "FIFO cap must hold"
    print("ok")
