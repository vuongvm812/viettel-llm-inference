"""Reject degenerate ``structured_outputs`` at request validation (not in the engine loop).

Consolidates the intent of sndr PN387 (vllm#45346) and PN523: a request carrying a
*degenerate* structured-output spec -- ``json_object=False``, an empty-string ``json`` /
``grammar`` / ``regex`` / ``structural_tag``, or an empty ``choice`` list -- reaches the
xgrammar compiler inside the ``EngineCore`` step loop and raises there. A raise in that
loop is not a per-request 400; it kills the engine (``EngineDead``), taking every in-flight
request down with it. On a judged box that scores as unscoreable.

We wrap ``SamplingParams._validate_structured_outputs`` (called from ``verify()`` at request
admission) and reject these shapes early with a clear ``ValueError`` -> a clean 400 for the
one bad request, engine untouched. Well-formed structured requests are inspected and passed
straight through to the stock validator unchanged.

Signature-agnostic (``*args, **kwargs``) so a v-bump to the validator's arg list doesn't
silently unpatch us. Never raises except the intended ``ValueError`` on a degenerate spec.

``VTL_ENABLE_REJECT_DEGENERATE_STRUCTURED_OUTPUTS=0`` serves stock (a degenerate request then
crashes the engine instead of 400ing).
"""

from __future__ import annotations

import logging

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vtl")

# Fields whose empty string is a compiler crash, not a valid constraint.
_STR_FIELDS = ("json", "grammar", "regex", "structural_tag")


def _reject_degenerate(so) -> None:
    """Raise ValueError if ``so`` is a degenerate structured-output spec. No-op otherwise."""
    if so is None:
        return
    # json_object is a flag: only True is meaningful. False falls through vLLM's
    # backend selection to a raise deep in the step loop.
    if getattr(so, "json_object", None) is False:
        raise ValueError(
            "structured_outputs.json_object must be True if set; omit it otherwise"
        )
    for field in _STR_FIELDS:
        val = getattr(so, field, None)
        if isinstance(val, str) and val.strip() == "":
            raise ValueError(
                f"structured_outputs.{field} must be a non-empty string if set"
            )
    if getattr(so, "choice", None) == []:
        raise ValueError("structured_outputs.choice must be a non-empty list if set")


@register_patch("reject_degenerate_structured_outputs", default=True)
def apply() -> None:
    from vllm.sampling_params import SamplingParams

    if already_patched(SamplingParams, "_validate_structured_outputs"):
        return

    original = SamplingParams._validate_structured_outputs

    def _validate_structured_outputs(self, *args, **kwargs):
        # Reject before the stock validator so degenerate specs never reach the compiler.
        _reject_degenerate(getattr(self, "structured_outputs", None))
        return original(self, *args, **kwargs)

    SamplingParams._validate_structured_outputs = mark_patched(
        _validate_structured_outputs, original
    )
    log.info("vtl: reject_degenerate_structured_outputs installed (degenerate spec -> 400, not EngineDead)")


def _self_check() -> None:
    """No vLLM: a duck-typed structured-outputs spec exercises every reject + the pass-through."""

    class SO:
        def __init__(self, **kw):
            self.json = kw.get("json")
            self.grammar = kw.get("grammar")
            self.regex = kw.get("regex")
            self.structural_tag = kw.get("structural_tag")
            self.choice = kw.get("choice")
            self.json_object = kw.get("json_object")

    def rejects(so) -> bool:
        try:
            _reject_degenerate(so)
            return False
        except ValueError:
            return True

    # Degenerate shapes -> rejected.
    assert rejects(SO(json_object=False))
    assert rejects(SO(json=""))
    assert rejects(SO(json="   "))  # whitespace-only is still empty to the compiler
    assert rejects(SO(grammar=""))
    assert rejects(SO(regex=""))
    assert rejects(SO(structural_tag=""))
    assert rejects(SO(choice=[]))

    # Valid shapes -> passed through untouched.
    assert not rejects(None)
    assert not rejects(SO(json='{"type":"object"}'))
    assert not rejects(SO(json={"type": "object"}))  # dict schema is fine
    assert not rejects(SO(json_object=True))
    assert not rejects(SO(grammar="root ::= \"a\""))
    assert not rejects(SO(choice=["a", "b"]))
    assert not rejects(SO(regex="[0-9]+"))

    print("reject_degenerate_structured_outputs self-check ok")


if __name__ == "__main__":
    _self_check()
