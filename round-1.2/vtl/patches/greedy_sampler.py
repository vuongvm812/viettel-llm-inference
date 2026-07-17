"""Greedy-only fast path for ``Sampler.forward``.

The served workload is greedy (temperature 0) with no logprobs, no penalties, no tools --
so every step, ``Sampler.forward`` casts the full [num_reqs, vocab] logits to float32, runs
an (empty) logits-processor pass, dispatches through ``sample()``, and builds logprobs
machinery it will throw away, only to end at an argmax. This patch short-circuits that: when
the batch is provably a plain greedy step (see ``_should_fastpath``), it returns the argmax
directly.

Small win (per-step overhead, ~1% of TPOT) but on the scored path and lossless. It is only
taken when NOTHING can move the argmax away from ``logits.argmax`` -- no penalties, no
bad-words, no allowed-token mask, no non-argmax-invariant logits processors, no thinking
budget, no logprobs, no spec bonus token. Any of those present => fall through to stock
``forward``. argmax is invariant to the float32 upcast, so skipping the cast is exact.
"""

from __future__ import annotations

import logging

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vtl")


def _should_fastpath(sm, predict_bonus_token: bool) -> bool:
    """True iff the step is a pure greedy argmax with no logit-modifying state.

    Mirrors every branch in ``Sampler.apply_logits_processors`` / the logprobs prologue that
    could change the sampled token; if all are inert, ``argmax`` is the exact result.
    """
    return (
        not predict_bonus_token
        and sm.all_greedy
        and sm.max_num_logprobs is None
        and not sm.logprob_token_ids
        and sm.no_penalties
        and sm.allowed_token_ids_mask is None
        and not sm.bad_words_token_ids
        and not sm.logitsprocs.non_argmax_invariant
        and getattr(sm, "thinking_budget_state_holder", None) is None
    )


@register_patch("greedy_sampler", default=True)
def apply() -> None:
    import torch

    from vllm.v1.outputs import SamplerOutput
    from vllm.v1.sample.sampler import Sampler

    if already_patched(Sampler, "forward"):
        return

    original = Sampler.forward

    def forward(
        self,
        logits,
        sampling_metadata,
        predict_bonus_token: bool = False,
        logprobs_mode_override=None,
    ):
        if _should_fastpath(sampling_metadata, predict_bonus_token):
            tok = logits.argmax(dim=-1).view(-1).to(torch.int32).unsqueeze(-1)
            return SamplerOutput(sampled_token_ids=tok, logprobs_tensors=None)
        return original(
            self, logits, sampling_metadata, predict_bonus_token, logprobs_mode_override
        )

    Sampler.forward = mark_patched(forward, original)
    log.info("vtl: greedy_sampler installed (argmax fast path for plain greedy steps)")


def _self_check() -> None:
    """No torch needed: exercises the guard with fake SamplingMetadata objects."""

    class Procs:
        non_argmax_invariant: list = []

    class SM:
        def __init__(self, **kw):
            self.all_greedy = True
            self.max_num_logprobs = None
            self.logprob_token_ids = None
            self.no_penalties = True
            self.allowed_token_ids_mask = None
            self.bad_words_token_ids = {}
            self.logitsprocs = Procs()
            self.thinking_budget_state_holder = None
            self.__dict__.update(kw)

    # clean greedy step -> fast path
    assert _should_fastpath(SM(), predict_bonus_token=False) is True

    # every blocker individually disqualifies
    assert _should_fastpath(SM(all_greedy=False), False) is False
    assert _should_fastpath(SM(max_num_logprobs=0), False) is False
    assert _should_fastpath(SM(max_num_logprobs=5), False) is False
    assert _should_fastpath(SM(logprob_token_ids={0: [1]}), False) is False
    assert _should_fastpath(SM(no_penalties=False), False) is False
    assert _should_fastpath(SM(allowed_token_ids_mask=object()), False) is False
    assert _should_fastpath(SM(bad_words_token_ids={0: [[1]]}), False) is False
    assert _should_fastpath(SM(thinking_budget_state_holder=object()), False) is False
    assert _should_fastpath(SM(), predict_bonus_token=True) is False

    procs = Procs()
    procs.non_argmax_invariant = [object()]
    assert _should_fastpath(SM(logitsprocs=procs), False) is False

    # argmax correctness (only if torch is importable in this env)
    try:
        import torch

        logits = torch.tensor([[0.1, 0.9, 0.2], [3.0, 1.0, 2.0]])
        tok = logits.argmax(dim=-1).view(-1).to(torch.int32).unsqueeze(-1)
        assert tok.tolist() == [[1], [0]], tok.tolist()
        assert tok.dtype == torch.int32 and tok.shape == (2, 1)
    except ImportError:
        pass

    print("greedy_sampler self-check ok")


if __name__ == "__main__":
    _self_check()
