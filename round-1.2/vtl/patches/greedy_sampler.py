"""Greedy-only fast path for the sampler, for BOTH model runners.

The served workload is greedy (temperature 0) with no logprobs, no penalties, no tools --
so every step the stock sampler casts the full [num_reqs, vocab] logits to float32, runs
an (empty) logits-processor pass, dispatches through ``sample()``, and builds logprobs
machinery it will throw away, only to end at an argmax. This patch short-circuits that: when
the batch is provably a plain greedy step, it returns the argmax directly.

Small win (per-step overhead, ~1% of TPOT) but on the scored path and lossless.

TWO runners, two entry points. ``VLLM_USE_V2_MODEL_RUNNER=1`` selects Model Runner V2, whose
sampler is a DIFFERENT class with a different contract:

    V1  vllm.v1.sample.sampler.Sampler          nn.Module, forward(logits, sampling_metadata)
                                                -> vllm.v1.outputs.SamplerOutput
    V2  vllm.v1.worker.gpu.sample.sampler.Sampler   plain class, __call__(logits, input_batch)
                                                -> vllm.v1.worker.gpu.sample.output.SamplerOutput

Both are patched here so one image can A/B the runners. Each is independently guarded: if a
runner's module is missing, that half is skipped and the other still installs.

ARGMAX-INVARIANCE (why the fast path is exact, not an approximation). We skip the float32
upcast, temperature scaling, ``min_p``, ``top_k`` and ``top_p`` -- all of which either rescale
uniformly or mask only tokens strictly below the max, so none can displace the argmax. The
three transforms that CAN displace it -- penalties, logit bias / allowed-token masks, and
bad-words -- are guarded and force a fall-through, as do logprobs (which need the real
distribution) and a spec bonus token. Any guard failing => stock path.
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


def _should_fastpath_v2(sampler, idx_mapping_np, no_logprobs_sentinel) -> bool:
    """True iff a Model Runner V2 step is a pure greedy argmax with no logit-modifying state.

    Each predicate mirrors the *same* check the corresponding V2 state object uses internally
    to decide it can skip its own kernel, so we can never disagree with it about inertness:
    ``penalties.py`` ``apply_penalties``, ``logit_bias.py`` ``apply_logit_bias``,
    ``bad_words.py`` ``apply_bad_words``. ``temperature == 0`` is V2's definition of greedy
    (``states.py`` ``any_greedy``); we require it of EVERY request in the batch, not any.
    """
    import numpy as np

    states = sampler.sampling_states
    return (
        # num_nans must be computed from the raw logits, before sampling -- if it is wanted,
        # we cannot skip the prologue.
        not sampler.compute_nans
        # No logprobs of either flavour: both need the real (processed) distribution.
        and states.max_num_logprobs(idx_mapping_np) == no_logprobs_sentinel
        and sampler.logprob_token_ids_state.max_num_token_ids(idx_mapping_np) == 0
        # Every request greedy.
        and bool(np.all(states.temperature.np[idx_mapping_np] == 0.0))
        # The three transforms that can actually move the argmax.
        and not bool(np.any(sampler.penalties_state.use_penalty[idx_mapping_np]))
        and not bool(np.any(sampler.logit_bias_state.use_logit_bias[idx_mapping_np]))
        and int(sampler.bad_words_state.num_bad_words.np[idx_mapping_np].max(initial=0)) == 0
    )


@register_patch("greedy_sampler", default=True)
def apply() -> None:
    installed = []
    for name, install in (("v1", _apply_v1), ("v2", _apply_v2)):
        try:
            if install():
                installed.append(name)
        except ImportError:
            continue  # that runner is not present in this vLLM build; the other still applies
    if not installed:
        log.warning("vtl: greedy_sampler found neither sampler class; not installed")
        return
    log.info("vtl: greedy_sampler installed for runner(s): %s", ", ".join(installed))


def _apply_v1() -> bool:
    """Patch the V1 runner's ``Sampler.forward``. Returns True if newly installed."""
    import torch

    from vllm.v1.outputs import SamplerOutput
    from vllm.v1.sample.sampler import Sampler

    if already_patched(Sampler, "forward"):
        return False

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
    return True


def _apply_v2() -> bool:
    """Patch the Model Runner V2 ``Sampler.__call__``. Returns True if newly installed."""
    import torch

    from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
    from vllm.v1.worker.gpu.sample.output import SamplerOutput
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS

    if already_patched(Sampler, "__call__"):
        return False

    original = Sampler.__call__

    def __call__(self, logits, input_batch):
        if not _should_fastpath_v2(self, input_batch.idx_mapping_np, NO_LOGPROBS):
            return original(self, logits, input_batch)

        # int64 to match the stock paths (flashinfer_sample casts explicitly; gumbel_sample
        # returns index dtype). A narrower dtype here would diverge downstream.
        sampled = logits.argmax(dim=-1).view(-1).to(torch.int64)
        # Not skippable: SamplerOutput carries these, and chunked-prefill requests that are
        # not done prefilling must still be reported as producing no output token.
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            input_batch.seq_lens.new_ones(input_batch.num_reqs),
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.req_states.prefill_len.gpu,
        )
        return SamplerOutput(
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    Sampler.__call__ = mark_patched(__call__, original)
    return True


def _self_check_v2() -> None:
    """No vLLM needed: exercises the V2 guard with fake sampler state objects.

    Every blocker must individually force a fall-through -- a guard that silently stops
    blocking is how this patch would start changing sampled tokens.
    """
    try:
        import numpy as np
    except ImportError:
        print("greedy_sampler v2 self-check SKIPPED (no numpy)")
        return

    NO_LOGPROBS = -1
    idx = np.array([0, 1])

    class Arr:  # mimics vLLM's CpuGpuBuffer: the numpy side is all we touch
        def __init__(self, values):
            self.np = np.array(values)

    class V2:
        def __init__(self, **kw):
            self.compute_nans = False
            self._max_logprobs = NO_LOGPROBS
            self._max_token_ids = 0
            self.sampling_states = self  # flat fake: one object plays every role
            self.logprob_token_ids_state = self
            self.temperature = Arr([0.0, 0.0])
            self.penalties_state = self
            self.use_penalty = np.array([False, False])
            self.logit_bias_state = self
            self.use_logit_bias = np.array([False, False])
            self.bad_words_state = self
            self.num_bad_words = Arr([0, 0])
            self.__dict__.update(kw)

        def max_num_logprobs(self, _idx):
            return self._max_logprobs

        def max_num_token_ids(self, _idx):
            return self._max_token_ids

    ok = _should_fastpath_v2(V2(), idx, NO_LOGPROBS)
    assert ok is True, "clean greedy V2 step must take the fast path"

    blockers = {
        "compute_nans": dict(compute_nans=True),
        "logprobs": dict(_max_logprobs=5),
        "logprob_token_ids": dict(_max_token_ids=3),
        "one request sampling": dict(temperature=Arr([0.0, 0.7])),
        "penalties": dict(use_penalty=np.array([False, True])),
        "logit_bias": dict(use_logit_bias=np.array([True, False])),
        "bad_words": dict(num_bad_words=Arr([0, 2])),
    }
    for label, kw in blockers.items():
        assert _should_fastpath_v2(V2(**kw), idx, NO_LOGPROBS) is False, f"{label} must block"

    # Requests outside idx_mapping_np must not block: the guard is per-scheduled-request.
    unscheduled = V2(use_penalty=np.array([False, False, True]), temperature=Arr([0.0, 0.0, 0.9]))
    assert _should_fastpath_v2(unscheduled, idx, NO_LOGPROBS) is True

    print("greedy_sampler v2 self-check ok")


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

    _self_check_v2()

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
