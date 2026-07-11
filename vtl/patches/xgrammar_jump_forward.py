"""XGrammar jump-forward decoding for vLLM V1 structured outputs.

When a structured request's grammar forces a deterministic continuation (JSON
keys, punctuation, e.g. ``{"name":"``), those tokens don't need to be *sampled* --
they're the only legal next tokens. Stock vLLM still spends one memory-bound decode
step per forced token. Jump-forward emits them all at once: we append the forced
tokens to the request's output, advance the grammar over them, and let vLLM's
existing chunked-prefill path compute their KV in ONE compute-bound forward next
step. K decode steps collapse into 1 prefill.

Mechanism (verified against v0.22.1): a running request whose ``num_tokens`` exceeds
``num_computed_tokens`` has the gap scheduled as a chunked prefill; only the last
scheduled position per request is sampled (``logits_indices = query_start_loc[1:]-1``),
so the appended tokens are pure forward-pass input, never resampled. We hook the
post-pass of ``Scheduler.update_from_output`` -- the one seam that holds both the
request and the per-step emitted ``new_token_ids`` (jump tokens are real output text
that must reach the detokenizer exactly once).

Correctness is byte-exact under greedy decoding via a strict safe-seam gate: we only
jump when re-tokenizing ``last_token_text + jump_string`` reproduces the natural
tokenization (last token unchanged, suffix round-trips). Any other case, any tokenizer
quirk, any exception -> skip the jump and fall back to stock decode. Jump-forward is
purely additive; the sampled output is already correct without it.

Async scheduling: ``AsyncScheduler`` pipelines ``schedule(N+1)`` ahead of
``update_from_output(N)``, so appending mid-pipeline reorders the in-flight sampled
token against our jumped tokens. Async-safe coordination (placeholder/next-step
accounting) is a GPU-spike item; until it is proven on hardware this patch jumps ONLY
under the sync ``Scheduler`` and is a no-op under async. AsyncScheduler inherits
``update_from_output`` unchanged, so the single wrap below covers both; the runtime
guard below picks the safe path.

Off by default. Enable with ``VTL_ENABLE_XGRAMMAR_JUMP_FORWARD=1`` once byte-parity and
the async accounting have been validated on the H200.
"""

from __future__ import annotations

import logging

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vtl")


def _is_async(scheduler) -> bool:
    """True for AsyncScheduler instances -- jump-forward stays off there for now.

    ponytail: class-name check avoids importing AsyncScheduler (keeps this module
    vLLM-less for the self-check). Upgrade path: implement placeholder/next-step
    accounting so the append is safe mid-pipeline, then drop this guard (Phase 2a.2).
    """
    if type(scheduler).__name__ == "AsyncScheduler":
        return True
    return bool(getattr(scheduler, "use_async_scheduling", False))


def _iter_outputs(engine_core_outputs):
    """Yield every per-request EngineCoreOutput across all client batches.

    ``update_from_output`` returns ``dict[int, EngineCoreOutputs]`` keyed by
    client_index; each ``EngineCoreOutputs.outputs`` is a list of EngineCoreOutput.
    """
    for batch in engine_core_outputs.values():
        for eco in getattr(batch, "outputs", None) or ():
            yield eco


def _safe_jump_ids(tokenizer, last_id: int, jump_str: str) -> list[int]:
    """Token ids for ``jump_str`` that ARE the natural greedy continuation after
    ``last_id`` -- or ``[]`` if the seam is unsafe. Never raises.

    Independently tokenizing ``jump_str`` can diverge from how the model would split
    it glued to the previous token, which would corrupt output and streaming
    detokenization. We re-tokenize ``last_token_text + jump_str`` and accept the
    result ONLY when the last token id is unchanged and the remainder round-trips to
    exactly ``jump_str``. That guarantees byte-identical greedy output.
    """
    try:
        last_txt = tokenizer.decode([last_id])
        merged = tokenizer.encode(last_txt + jump_str, add_special_tokens=False)
        if merged and merged[0] == last_id and tokenizer.decode(merged[1:]) == jump_str:
            return list(merged[1:])
    except Exception:
        pass
    return []


def _apply_jump(scheduler, eco) -> int:
    """Extend one request by its grammar-forced deterministic continuation.

    Returns the number of tokens jumped (0 if none). Self-guarded: any failure for
    this request skips its jump and leaves its already-correct sampled output intact.
    """
    try:
        if eco.finish_reason is not None:
            return 0  # request finished this step; nothing more to force
        req = scheduler.requests.get(eco.request_id)
        if req is None or req.spec_token_ids:
            return 0  # gone, or spec-decode in flight (first cut skips overlap)

        som = scheduler.structured_output_manager
        if not som.should_advance(req):
            return 0  # non-structured, or grammar not active yet (e.g. still reasoning)

        sor = req.structured_output_request
        grammar = getattr(sor, "grammar", None) if sor is not None else None
        if grammar is None or grammar.is_terminated():
            return 0

        matcher = getattr(grammar, "matcher", None)
        find = getattr(matcher, "find_jump_forward_string", None)
        if find is None:
            return 0  # xgrammar too old for jump-forward; self-disable for this req

        jump_str = find()
        if not jump_str or not req.output_token_ids:
            return 0

        jids = _safe_jump_ids(som.tokenizer, req.output_token_ids[-1], jump_str)
        if not jids:
            return 0

        # Never overshoot the request's own budget.
        room = req.max_tokens - len(req.output_token_ids)
        if room <= 0:
            return 0
        jids = jids[:room]

        # Advance the grammar over the forced tokens. These are grammar-deterministic,
        # so accept must succeed; if it somehow doesn't, undo the partial advance and
        # skip -- never append tokens the FSM didn't take.
        before = grammar.num_processed_tokens
        if not grammar.accept_tokens(eco.request_id, jids):
            grammar.rollback(grammar.num_processed_tokens - before)
            return 0

        req.append_output_token_ids(jids)  # grows num_tokens -> next schedule() prefills gap
        eco.new_token_ids.extend(jids)     # emit to the detokenizer exactly once, this step
        return len(jids)
    except Exception:
        log.exception("vtl: jump_forward failed for one request; skipping it")
        return 0


@register_patch("xgrammar_jump_forward", default=False)
def apply() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    if already_patched(Scheduler, "update_from_output"):
        return

    original = Scheduler.update_from_output

    def update_from_output(self, scheduler_output, model_runner_output):
        outs = original(self, scheduler_output, model_runner_output)
        try:
            if not _is_async(self):
                jumped = 0
                for eco in _iter_outputs(outs):
                    jumped += _apply_jump(self, eco)
                if jumped:
                    log.debug("vtl: jump_forward skipped %d decode step(s)", jumped)
        except Exception:
            log.exception("vtl: jump_forward post-pass failed; using stock output")
        return outs

    Scheduler.update_from_output = mark_patched(update_from_output, original)
    log.info("vtl: xgrammar_jump_forward installed (sync scheduler; async gated to GPU spike)")


def _self_check() -> None:
    """Runs with no vLLM/GPU: pure-python fakes exercise the jump logic + every skip."""

    class CharTok:
        """Char-level tokenizer: one id per char, clean round-trip at every seam."""

        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

        def decode(self, ids):
            return "".join(chr(i) for i in ids)

    class MergeTok(CharTok):
        """Tokenizer that MERGES the seam char with the first jump char -> unsafe seam."""

        def encode(self, text, add_special_tokens=False):
            # Collapse a leading two-char run into a single fused id so merged[0] != last_id.
            if len(text) >= 2:
                return [10_000] + [ord(c) for c in text[2:]]
            return [ord(c) for c in text]

    class Matcher:
        def __init__(self, jump, terminated=False, raises=False):
            self._jump = jump
            self._terminated = terminated
            self._raises = raises

        def find_jump_forward_string(self):
            if self._raises:
                raise RuntimeError("boom")
            return self._jump

    class Grammar:
        def __init__(self, jump, terminated=False, raises=False, accept=True):
            self.matcher = Matcher(jump, terminated, raises)
            self._terminated = terminated
            self._accept = accept
            self.num_processed_tokens = 0
            self.accepted: list[int] = []

        def is_terminated(self):
            return self._terminated

        def accept_tokens(self, rid, ids):
            if not self._accept:
                return False
            self.accepted.extend(ids)
            self.num_processed_tokens += len(ids)
            return True

        def rollback(self, n):
            for _ in range(n):
                if self.accepted:
                    self.accepted.pop()
            self.num_processed_tokens -= n

    class SOR:
        def __init__(self, grammar):
            self.grammar = grammar

    class Req:
        def __init__(self, out, grammar, max_tokens=1000, spec=None):
            self._out = list(out)
            self.output_token_ids = self._out
            self.structured_output_request = SOR(grammar) if grammar else None
            self.max_tokens = max_tokens
            self.spec_token_ids = spec or []

        def append_output_token_ids(self, ids):
            self._out.extend(ids)

    class SOM:
        def __init__(self, tok, advance=True):
            self.tokenizer = tok
            self._advance = advance

        def should_advance(self, req):
            return self._advance

    class Sched:
        def __init__(self, req, som):
            self.requests = {"r": req} if req else {}
            self.structured_output_manager = som

    class ECO:
        def __init__(self, finish=None):
            self.request_id = "r"
            self.new_token_ids = []
            self.finish_reason = finish

    def run(req, som, eco):
        return _apply_jump(Sched(req, som), eco)

    # 1. Safe seam: "ab" + jump "cd" -> ['c','d'] appended to output + eco, grammar advanced.
    g = Grammar("cd")
    req = Req([ord("a"), ord("b")], g)
    eco = ECO()
    n = run(req, SOM(CharTok()), eco)
    assert n == 2, n
    assert req.output_token_ids[-2:] == [ord("c"), ord("d")], req.output_token_ids
    assert eco.new_token_ids == [ord("c"), ord("d")], eco.new_token_ids
    assert g.accepted == [ord("c"), ord("d")], g.accepted

    # 2. Unsafe seam: retokenization fuses the boundary -> nothing appended.
    g = Grammar("cd")
    req = Req([ord("a"), ord("b")], g)
    eco = ECO()
    assert run(req, SOM(MergeTok()), eco) == 0
    assert eco.new_token_ids == [] and g.accepted == []

    # 3. finish_reason set -> skip.
    req = Req([ord("a")], Grammar("cd"))
    assert run(req, SOM(CharTok()), ECO(finish="stop")) == 0

    # 4. spec tokens in flight -> skip.
    req = Req([ord("a")], Grammar("cd"), spec=[7])
    assert run(req, SOM(CharTok()), ECO()) == 0

    # 5. should_advance False (non-structured / reasoning) -> skip.
    req = Req([ord("a")], Grammar("cd"))
    assert run(req, SOM(CharTok(), advance=False), ECO()) == 0

    # 6. terminated grammar -> skip.
    req = Req([ord("a")], Grammar("cd", terminated=True))
    assert run(req, SOM(CharTok()), ECO()) == 0

    # 7. empty jump string -> skip.
    req = Req([ord("a")], Grammar(""))
    assert run(req, SOM(CharTok()), ECO()) == 0

    # 8. max_tokens clamp: room for only 1 more token truncates the jump.
    g = Grammar("cd")
    req = Req([ord("a"), ord("b")], g, max_tokens=3)  # room = 3 - 2 = 1
    eco = ECO()
    assert run(req, SOM(CharTok()), eco) == 1
    assert eco.new_token_ids == [ord("c")], eco.new_token_ids
    # ...and no room at all -> skip.
    req = Req([ord("a"), ord("b")], Grammar("cd"), max_tokens=2)
    assert run(req, SOM(CharTok()), ECO()) == 0

    # 9. exception inside (matcher raises) -> isolated, returns 0, nothing appended.
    g = Grammar("cd", raises=True)
    req = Req([ord("a")], g)
    eco = ECO()
    assert run(req, SOM(CharTok()), eco) == 0
    assert eco.new_token_ids == []

    # 10. grammar rejects the forced tokens -> undo partial advance, skip.
    g = Grammar("cd", accept=False)
    req = Req([ord("a")], g)
    eco = ECO()
    assert run(req, SOM(CharTok()), eco) == 0
    assert req.output_token_ids == [ord("a")] and eco.new_token_ids == []

    # 11. async scheduler guard: _is_async trips on the class name.
    class AsyncScheduler:  # noqa: N801 -- mimics vLLM's class name
        pass

    assert _is_async(AsyncScheduler()) is True
    assert _is_async(Sched(None, None)) is False

    print("xgrammar_jump_forward self-check ok")


if __name__ == "__main__":
    _self_check()
