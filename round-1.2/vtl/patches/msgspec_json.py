"""Full-flow msgspec JSON for the OpenAI frontend.

msgspec (already a vLLM dep -- the engine IPC uses ``msgspec.msgpack``) replaces the two
remaining stdlib-``json`` chokepoints on the request/response path, so the whole HTTP flow
is msgspec end to end:

* **Request body decode** -- ``starlette.requests.Request.json`` is ``json.loads(body)``
  (starlette ``requests.py``); FastAPI calls it (``fastapi/routing.py``) to get the dict it
  then validates against ``ChatCompletionRequest``. We swap in ``msgspec.json.decode``.
* **Every non-streaming JSON response** -- ``starlette.responses.JSONResponse.render`` is a
  ``json.dumps(...).encode()``; it serializes the chat non-stream body
  (``JSONResponse(content=generator.model_dump())``), ``/v1/models``, health, and error
  responses. We swap in ``msgspec.json.encode``. Patching ``render`` on the base class covers
  the explicit ``JSONResponse(...)`` constructions AND FastAPI's default response class, so
  no ``default_response_class`` / ``route_class`` wiring is needed (those don't retro-apply to
  vLLM's already-registered sub-router routes anyway).

The per-token streaming SSE path is handled separately in ``msgspec_stream`` -- the two
patches together make the frontend JSON fully msgspec.

Reliability first (a broken request/response = a 0-scoring request): decode failures are
re-raised as ``json.JSONDecodeError`` so FastAPI still returns a clean 422 (it only catches
that type), and encode failures fall back to stock ``json.dumps`` so an object msgspec can't
serialize degrades instead of 500-ing.
"""

from __future__ import annotations

import json
import logging

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vllm.vtl.msgspec_json")


def _decode_body(body, decode):
    """bytes -> parsed JSON via ``decode``; any failure -> ``json.JSONDecodeError``.

    FastAPI wraps ``request.json()`` in ``except json.JSONDecodeError`` to return 422; msgspec
    raises its own error type, so we map every decode failure onto that so the 422 path holds.
    A body msgspec can't parse is a malformed body -- 422 is the correct outcome.
    """
    try:
        return decode(body)
    except Exception as e:  # msgspec.DecodeError et al.
        doc = body.decode("utf-8", "replace") if isinstance(body, (bytes, bytearray)) else str(body)
        raise json.JSONDecodeError(str(e), doc, 0) from e


def _encode_or_fallback(content, encode, fallback):
    """``encode(content)`` bytes, or ``fallback()`` if msgspec can't serialize the value.

    Never let response serialization raise: an unencodable type degrades to the stock
    ``json.dumps`` render instead of turning into a 500 / broken response.
    """
    try:
        return encode(content)
    except Exception:
        return fallback()


@register_patch("msgspec_json", default=True)
def apply() -> None:
    import msgspec

    from starlette.requests import Request
    from starlette.responses import JSONResponse

    decode = msgspec.json.decode
    encode = msgspec.json.encode

    # --- incoming: request body decode ---
    if not already_patched(Request, "json"):
        original_json = Request.json

        async def _json(self):
            if not hasattr(self, "_json"):
                self._json = _decode_body(await self.body(), decode)
            return self._json

        Request.json = mark_patched(_json, original_json)

    # --- outgoing: every non-streaming JSON response ---
    if not already_patched(JSONResponse, "render"):
        original_render = JSONResponse.render

        def render(self, content):
            return _encode_or_fallback(
                content, encode, lambda: original_render(self, content)
            )

        JSONResponse.render = mark_patched(render, original_render)

    log.info("vtl: msgspec_json installed (request decode + JSON response encode)")


def _self_check() -> None:
    """Runs with no msgspec / starlette / vLLM: json stubs exercise the two helpers."""
    ok = lambda d: json.dumps(d).encode("utf-8")  # noqa: E731

    # decode: valid body round-trips
    assert _decode_body(b'{"a": 1, "s": "x"}', json.loads) == {"a": 1, "s": "x"}

    # decode: malformed body maps onto json.JSONDecodeError (FastAPI's 422 path)
    for bad in (b"{not json", b"", b"\xff\xfe"):
        try:
            _decode_body(bad, json.loads)
            raise AssertionError(f"expected JSONDecodeError for {bad!r}")
        except json.JSONDecodeError:
            pass

    # encode: normal content is serialized by the fast encoder
    called = {"fb": False}

    def fb():
        called["fb"] = True
        return b"FALLBACK"

    out = _encode_or_fallback({"a": 1}, ok, fb)
    assert out == b'{"a": 1}' and not called["fb"], out

    # encode: an unencodable value falls back to the stock renderer, never raises
    def boom(_):
        raise TypeError("cannot encode")

    out2 = _encode_or_fallback(object(), boom, fb)
    assert out2 == b"FALLBACK" and called["fb"], out2

    print("msgspec_json self-check ok")


if __name__ == "__main__":
    _self_check()
