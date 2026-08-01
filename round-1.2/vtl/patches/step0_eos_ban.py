"""Ban EOS as a request's FIRST sampled token (V2 sampler seam).

8/420 judge requests fail with an empty stream: under the int4 lm_head the greedy
argmax lands on ``<|im_end|>`` (id 7) at step 0, the stream closes with zero content
tokens, and the request is unscoreable (the diagnosed +1.4pt failure mode). Masking
EOS out of the step-0 logits forces at least one real token. A request whose step-0
argmax was NOT EOS is byte-identical -- masking a token the argmax didn't pick cannot
move the argmax -- and a request whose step-0 argmax WAS EOS was failing anyway.

Placement: the first output token is sampled on the request's FINAL prefill chunk,
and only the stock ``Sampler.__call__`` path ever samples it -- the nstep burst
commits on pure 1-token decode steps only, so its bare argmaxes never see a step-0
row. One wrapper covers every path. ``is_prefilling_np`` rows are exactly the rows
sampling their first token; non-final chunk rows also match, but their sampled token
is a discarded placeholder, so masking them is harmless.

Cost: one numpy ``any()`` per non-burst step; the masking write itself runs only on
prefill steps (~2 per request lifetime). Knob: ``VTL_STEP0_EOS_ID`` (comma list,
default "7"). Any error disables the mask for the boot and falls through to stock.
"""

from __future__ import annotations

import logging
import os

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vllm.vtl.step0_eos_ban")


def _parse_ids(raw: str) -> tuple[int, ...]:
    """Comma-separated non-negative token ids; anything malformed -> ()."""
    try:
        ids = tuple(int(part) for part in raw.split(",") if part.strip())
    except ValueError:
        return ()
    return ids if all(i >= 0 for i in ids) else ()


def _step0_rows(is_prefilling_np, num_reqs: int, logits_rows: int):
    """Row indices sampling their first token, or None if none / rows don't map 1:1.

    ``logits_rows != num_reqs`` means expanded rows (spec decode, logprobs) -- the
    prefill flags no longer index the logits, so fail open rather than mask wrong rows.
    """
    if logits_rows != num_reqs:
        return None
    flags = is_prefilling_np[:num_reqs]
    if not flags.any():
        return None
    import numpy as np

    return np.nonzero(flags)[0]


@register_patch("step0_eos_ban", default=True)
def apply() -> None:
    import torch

    from vllm.v1.worker.gpu.sample.sampler import Sampler

    if already_patched(Sampler, "__call__"):
        return

    ids = _parse_ids(os.environ.get("VTL_STEP0_EOS_ID", "7"))
    if not ids:
        log.warning("vtl: step0_eos_ban off -- VTL_STEP0_EOS_ID empty or malformed")
        return

    original = Sampler.__call__
    live = [True]  # ponytail: list cell, not a class attr -- one flag per boot

    def __call__(self, logits, input_batch):
        if live[0]:
            try:
                rows = _step0_rows(
                    input_batch.is_prefilling_np, input_batch.num_reqs, logits.shape[0]
                )
                if rows is not None:
                    if max(ids) >= logits.shape[1]:
                        live[0] = False
                        log.error(
                            "vtl: step0_eos_ban disabled -- id %d outside vocab %d",
                            max(ids), logits.shape[1],
                        )
                    else:
                        idx = torch.from_numpy(rows).to(logits.device)
                        for tok in ids:
                            logits[idx, tok] = float("-inf")
            except Exception:
                live[0] = False
                log.exception("vtl: step0_eos_ban disabled -- masking raised")
        return original(self, logits, input_batch)

    Sampler.__call__ = mark_patched(__call__, original)
    log.info("vtl: step0_eos_ban installed (ids=%s)", ids)


def _self_check() -> None:
    import numpy as np

    assert _parse_ids("7") == (7,)
    assert _parse_ids("7,2") == (7, 2)
    assert _parse_ids("") == ()
    assert _parse_ids("x") == ()
    assert _parse_ids("-1") == ()

    # no prefill rows -> None (the every-decode-step path is a single any())
    flags = np.zeros(8, dtype=np.bool_)
    assert _step0_rows(flags, 4, 4) is None
    # expanded logits rows -> fail open
    flags[1] = True
    assert _step0_rows(flags, 4, 5) is None
    # prefill rows -> their indices
    flags[3] = True
    assert _step0_rows(flags, 4, 4).tolist() == [1, 3]
    # stale flags beyond num_reqs are invisible
    flags[:] = False
    flags[6] = True
    assert _step0_rows(flags, 4, 4) is None

    try:
        import torch

        logits = torch.tensor([[0.1, 0.9, 0.2], [3.0, 1.0, 2.0]])
        rows = _step0_rows(np.array([True, False]), 2, 2)
        idx = torch.from_numpy(rows)
        logits[idx, 1] = float("-inf")
        # masked row moved off token 1; unmasked row untouched
        assert logits.argmax(dim=-1).tolist() == [2, 0]
    except ImportError:
        pass

    print("step0_eos_ban self-check ok")


if __name__ == "__main__":
    _self_check()
