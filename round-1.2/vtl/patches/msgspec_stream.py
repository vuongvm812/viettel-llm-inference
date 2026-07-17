"""Fast per-token SSE serialization for the OpenAI chat-completions stream.

The judge measures TTFT/TPOT at the HTTP client, so the per-token streaming chunk is on
the scored path. Stock vLLM builds a ``ChatCompletionStreamResponse`` (with a nested
``DeltaMessage`` + choice) pydantic object per token and calls ``model_dump_json``. This
patch replaces that -- for the common "simple" request shape only -- with a plain-dict
build + ``msgspec.json.encode``, skipping the pydantic construction entirely (msgspec is
already a vLLM dep: the engine IPC uses ``msgspec.msgpack``, so zero new dependency).

Any request that is NOT simple is delegated untouched to the stock generator, so behaviour
is identical there. "Simple" = no tool/reasoning parser, ``n<=1``, no logprobs, no echo, no
return_token_ids/return_prompt_text, no output logging, and (if usage is requested) no
per-request metrics / prompt-token-details. The served workload (greedy chat, temperature 0,
no logprobs/tools/n) is 100% simple; the guards exist so an unusual private-trace request
still gets exact stock behaviour rather than a wrong stream.

Correctness bar: the emitted SSE is JSON the OpenAI client parses to the same deltas as the
stock generator (validated build-vs-build; greedy + seed => deterministic completions). A
bug here malforms the stream, so the fast path mirrors the stock control flow exactly:
first role chunk, per-token content deltas, the empty chunked-prefill skip, finish handling,
optional continuous/final usage, then ``[DONE]``. Errors are caught and turned into a stock
error chunk + ``[DONE]`` so a failure can never break the connection into a transport error.
"""

from __future__ import annotations

import json
import logging
import time

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vtl")

_OBJ = "chat.completion.chunk"


def _is_simple(serving, request) -> bool:
    """True if the request maps cleanly to the plain-dict fast path.

    Every branch the stock generator has for tools/reasoning/logprobs/echo/n>1/token-ids/
    output-logging is excluded here; those requests fall back to the original method.
    """
    return (
        getattr(serving, "parser_cls", None) is None
        and (request.n is None or request.n == 1)
        and not request.logprobs
        and not getattr(request, "echo", False)
        and not getattr(request, "return_token_ids", False)
        and not getattr(request, "return_prompt_text", False)
        and not getattr(serving, "enable_log_outputs", False)
    )


async def _fast_stream(
    serving,
    request,
    result_generator,
    request_id,
    model_name,
    request_metadata,
    inc_usage,
    inc_cont,
    encode,
    usage_info_cls,
):
    """Stock ``chat_completion_stream_generator`` simple-case flow, dict + ``encode``.

    ``encode`` is injected (``msgspec.json.encode`` at runtime, ``json.dumps`` in the
    self-check) so this whole async flow is testable without msgspec or vLLM.
    """
    created = int(time.time())
    num_prompt_tokens = 0
    prev_num_tokens = 0
    first = True
    finish_sent = False
    fp = getattr(serving, "system_fingerprint", None)

    def sse(chunk):
        return "data: " + encode(chunk).decode("utf-8") + "\n\n"

    try:
        async for res in result_generator:
            if res.prompt_token_ids is not None:
                num_prompt_tokens = len(res.prompt_token_ids)
                if res.encoder_prompt_token_ids is not None:
                    num_prompt_tokens += len(res.encoder_prompt_token_ids)

            if first:
                first = False
                role = serving.get_chat_request_role(request)
                choice = {
                    "index": 0,
                    "delta": {"role": role, "content": ""},
                    "logprobs": None,
                    "finish_reason": None,
                }
                chunk = {
                    "id": request_id,
                    "object": _OBJ,
                    "created": created,
                    "choices": [choice],
                    "model": model_name,
                }
                if inc_cont:
                    chunk["usage"] = {
                        "prompt_tokens": num_prompt_tokens,
                        "completion_tokens": 0,
                        "total_tokens": num_prompt_tokens,
                    }
                yield sse(chunk)

            for output in res.outputs:
                if finish_sent:
                    continue

                delta_text = output.text
                # Chunked-prefill case: don't emit empty chunks (mirror stock skip).
                if not delta_text and not output.token_ids and not prev_num_tokens:
                    continue

                prev_num_tokens += len(output.token_ids)

                if output.finish_reason is None:
                    choice = {
                        "index": 0,
                        "delta": {"content": delta_text},
                        "logprobs": None,
                        "finish_reason": None,
                    }
                else:
                    # Raises on a retryable 'error' finish (stock semantics).
                    serving._raise_if_error(output.finish_reason, request_id)
                    choice = {
                        "index": 0,
                        "delta": {"content": delta_text},
                        "logprobs": None,
                        "finish_reason": output.finish_reason or "stop",
                        "stop_reason": output.stop_reason,
                    }
                    finish_sent = True

                chunk = {
                    "id": request_id,
                    "object": _OBJ,
                    "created": created,
                    "choices": [choice],
                    "model": model_name,
                }
                if not inc_usage and fp is not None and choice["finish_reason"] is not None:
                    chunk["system_fingerprint"] = fp
                if inc_cont:
                    chunk["usage"] = {
                        "prompt_tokens": num_prompt_tokens,
                        "completion_tokens": prev_num_tokens,
                        "total_tokens": num_prompt_tokens + prev_num_tokens,
                    }
                yield sse(chunk)

        if inc_usage:
            chunk = {
                "id": request_id,
                "object": _OBJ,
                "created": created,
                "choices": [],
                "model": model_name,
                "usage": {
                    "prompt_tokens": num_prompt_tokens,
                    "completion_tokens": prev_num_tokens,
                    "total_tokens": num_prompt_tokens + prev_num_tokens,
                },
            }
            if fp is not None:
                chunk["system_fingerprint"] = fp
            yield sse(chunk)

        # Middleware reads this aggregate usage off request_metadata.
        request_metadata.final_usage_info = usage_info_cls(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=prev_num_tokens,
            total_tokens=num_prompt_tokens + prev_num_tokens,
        )
    except Exception as e:  # never break the connection: emit a stock error chunk
        try:
            if type(e).__name__ == "GenerationError":
                data = serving._convert_generation_error_to_streaming_response(e)
            else:
                data = serving.create_streaming_error_response(e)
            yield "data: " + data + "\n\n"
        except Exception:
            log.exception("vtl: msgspec_stream error-response build failed")

    yield "data: [DONE]\n\n"


@register_patch("msgspec_stream", default=True)
def apply() -> None:
    import msgspec.json

    from vllm.entrypoints.openai.chat_completion.protocol import UsageInfo
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.entrypoints.serve.utils.api_utils import should_include_usage

    if already_patched(OpenAIServingChat, "chat_completion_stream_generator"):
        return

    original = OpenAIServingChat.chat_completion_stream_generator
    encode = msgspec.json.encode

    async def chat_completion_stream_generator(
        self,
        request,
        result_generator,
        request_id,
        model_name,
        conversation,
        tokenizer,
        request_metadata,
        chat_template_kwargs=None,
        mm_token_counts=None,
    ):
        fast = _is_simple(self, request)
        inc_usage = inc_cont = False
        if fast:
            inc_usage, inc_cont = should_include_usage(
                request.stream_options, self.enable_force_include_usage
            )
            # Usage + metrics/prompt-details is extra surface we don't replicate.
            if (inc_usage or inc_cont) and (
                getattr(self, "enable_per_request_metrics", False)
                or getattr(self, "enable_prompt_tokens_details", False)
            ):
                fast = False

        if not fast:
            async for c in original(
                self,
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
                chat_template_kwargs,
                mm_token_counts,
            ):
                yield c
            return

        async for c in _fast_stream(
            self,
            request,
            result_generator,
            request_id,
            model_name,
            request_metadata,
            inc_usage,
            inc_cont,
            encode,
            UsageInfo,
        ):
            yield c

    OpenAIServingChat.chat_completion_stream_generator = mark_patched(
        chat_completion_stream_generator, original
    )
    log.info("vtl: msgspec_stream installed (dict+msgspec SSE for simple chat streams)")


def _self_check() -> None:
    """Runs with no vLLM and no msgspec: fakes + a json encoder exercise the flow."""
    import asyncio

    class Req:
        response_role = "assistant"

        def __init__(self, n=1, logprobs=False):
            self.n = n
            self.logprobs = logprobs
            self.echo = False
            self.return_token_ids = False
            self.return_prompt_text = False
            self.stream_options = None
            self.add_generation_prompt = True
            self.messages = [{"role": "user"}]

    class Serving:
        parser_cls = None
        enable_log_outputs = False
        system_fingerprint = None

        def get_chat_request_role(self, request):
            return "assistant"

        def _raise_if_error(self, finish_reason, request_id):
            if finish_reason == "error":
                raise RuntimeError("boom")

        def create_streaming_error_response(self, e):
            return '{"error":"x"}'

        def _convert_generation_error_to_streaming_response(self, e):
            return '{"error":"gen"}'

    class Out:
        index = 0

        def __init__(self, text, token_ids, finish_reason=None, stop_reason=None):
            self.text = text
            self.token_ids = token_ids
            self.finish_reason = finish_reason
            self.stop_reason = stop_reason

    class Res:
        prompt = None
        num_cached_tokens = 0
        encoder_prompt_token_ids = None

        def __init__(self, outputs, prompt_token_ids=None):
            self.outputs = outputs
            self.prompt_token_ids = prompt_token_ids

    class Meta:
        final_usage_info = None

    class Usage:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    enc = lambda d: json.dumps(d).encode("utf-8")  # noqa: E731

    # _is_simple gating
    assert _is_simple(Serving(), Req()) is True
    assert _is_simple(Serving(), Req(logprobs=True)) is False
    assert _is_simple(Serving(), Req(n=2)) is False

    async def agen(items):
        for it in items:
            yield it

    async def run(seq, inc_u=False, inc_cont=False):
        meta = Meta()
        chunks = []
        async for c in _fast_stream(
            Serving(), Req(), agen(seq), "id1", "m", meta, inc_u, inc_cont, enc, Usage
        ):
            chunks.append(c)
        return chunks, meta

    seq = [
        Res([Out("", [], None)], prompt_token_ids=[1, 2, 3]),  # role chunk; content skipped
        Res([Out("Hel", [10], None)]),
        Res([Out("lo", [11], None)]),
        Res([Out("", [12], "stop", 12)]),  # finish
    ]

    chunks, meta = asyncio.run(run(seq))
    assert chunks[-1] == "data: [DONE]\n\n", chunks[-1]
    payloads = [json.loads(c[len("data: ") :].strip()) for c in chunks[:-1]]
    # SSE frame shape
    for c in chunks:
        assert c.startswith("data: ") and c.endswith("\n\n"), c
    # first chunk is the role delta
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    # content concatenates to the full completion
    content = "".join(p["choices"][0]["delta"].get("content", "") for p in payloads[1:])
    assert content == "Hello", content
    # finish reason lands on the last data chunk only
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
    assert all(p["choices"][0]["finish_reason"] is None for p in payloads[:-1])
    # usage accounting reached the middleware object
    assert meta.final_usage_info.prompt_tokens == 3
    assert meta.final_usage_info.completion_tokens == 3

    # include_usage emits a trailing usage-only chunk before [DONE]
    chunks2, _ = asyncio.run(run(seq, inc_u=True))
    usage_p = json.loads(chunks2[-2][len("data: ") :].strip())
    assert usage_p["choices"] == [] and usage_p["usage"]["completion_tokens"] == 3

    # a raising finish is caught and still terminates with [DONE]
    bad = [Res([Out("x", [10], "error", None)], prompt_token_ids=[1])]
    chunks3, _ = asyncio.run(run(bad))
    assert chunks3[-1] == "data: [DONE]\n\n"
    assert any('"error"' in c for c in chunks3)

    print("msgspec_stream self-check ok")


if __name__ == "__main__":
    _self_check()
