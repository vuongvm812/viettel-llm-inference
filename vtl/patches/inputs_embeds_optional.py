"""Drop the dead ``inputs_embeds`` buffer on text-only models.

vLLM's ``GPUModelRunner.__init__`` unconditionally allocates a persistent
``(max_num_tokens, inputs_embeds_size)`` buffer -- GPU tensor plus a pinned-CPU
mirror -- for ``self.inputs_embeds`` (v0.22.1 ``gpu_model_runner.py:743``). For a
text-only model this buffer is never read or written: every dereference of
``self.inputs_embeds`` in that file is behind ``self.supports_mm_inputs`` or
``self.enable_prompt_embeds`` (the ``_preprocess`` text-only branch uses token ids,
and the DP-dummy randomizer only touches it when ``input_ids is None``). Our model
is dense text-only Qwen2, so the buffer is pure dead allocation:

  max_num_tokens 8192 x inputs_embeds_size 2048 x 2 bytes (bf16) = ~32 MiB GPU
  + ~32 MiB pinned CPU.

Freed before the memory profiler runs, that VRAM becomes KV-cache headroom.

Port of Genesis/SNDR **PN35** (``sndr/engines/vllm/patches/worker/
pn35_inputs_embeds_optional.py``), itself a backport of upstream vllm#35975 by
AjAnubolu. Logic preserved exactly: keep the buffer iff
``supports_mm_inputs or enable_prompt_embeds``; otherwise ``None``. sndr text-patches
the vLLM source before the allocation; ``vtl`` monkeypatches instead, so we let the
original ``__init__`` allocate and then release the buffer on the text-only path --
same guard, same end state (the transient init-time alloc is returned to the caching
allocator before profiling; the 32 MiB is noise on an H200 either way).

Set ``VTL_ENABLE_INPUTS_EMBEDS_OPTIONAL=0`` to keep stock behavior.
"""

from __future__ import annotations

import logging

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vtl")


def _drop_dead_inputs_embeds(runner) -> None:
    """Null ``runner.inputs_embeds`` when the model can't use it (text-only).

    Mirrors the PN35 / vllm#35975 guard: multimodal or prompt-embeds models keep
    the buffer (original behavior); everything else drops it. No-op if the buffer
    was never allocated. Never raises on a missing attribute -- an absent
    ``supports_mm_inputs``/``enable_prompt_embeds`` reads as False (text-only).
    """
    if getattr(runner, "inputs_embeds", None) is None:
        return
    if getattr(runner, "supports_mm_inputs", False) or getattr(
        runner, "enable_prompt_embeds", False
    ):
        return  # buffer is live for this model -- leave it exactly as stock allocated it
    runner.inputs_embeds = None


@register_patch("inputs_embeds_optional", default=True)
def apply() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if already_patched(GPUModelRunner, "__init__"):
        return

    original_init = GPUModelRunner.__init__

    def __init__(self, *args, **kwargs):
        # Original first (its errors are real vLLM errors -- must propagate). Only
        # our post-init cleanup is guarded: on any surprise we keep the stock buffer.
        original_init(self, *args, **kwargs)
        try:
            _drop_dead_inputs_embeds(self)
        except Exception:
            log.exception(
                "vtl: inputs_embeds_optional post-init hook failed, keeping stock buffer"
            )

    GPUModelRunner.__init__ = mark_patched(__init__, original_init)
    log.info(
        "vtl: inputs_embeds_optional installed (drop dead inputs_embeds buffer on text-only models)"
    )


# ponytail: alloc-then-free instead of never-alloc, because a runtime monkeypatch
# can't skip one statement mid-__init__ the way sndr's source text-patch does. The
# ceiling is one transient 32 MiB alloc at startup, released before KV profiling.
# Upgrade path if that 32 MiB ever matters: sniff the `numpy=False` _make_buffer call
# and return None from a wrapped _make_buffer -- fragile, not worth it at this scale.


def _self_check() -> None:
    """Runs with no vLLM: pure-python fakes exercise the guard + fallbacks."""

    class FakeRunner:
        def __init__(self, *, mm, prompt, embeds="<buffer>"):
            self.supports_mm_inputs = mm
            self.enable_prompt_embeds = prompt
            self.inputs_embeds = embeds

    # Text-only: buffer dropped.
    r = FakeRunner(mm=False, prompt=False)
    _drop_dead_inputs_embeds(r)
    assert r.inputs_embeds is None, r.inputs_embeds

    # Multimodal: buffer kept untouched.
    r = FakeRunner(mm=True, prompt=False)
    _drop_dead_inputs_embeds(r)
    assert r.inputs_embeds == "<buffer>", r.inputs_embeds

    # Prompt-embeds: buffer kept untouched.
    r = FakeRunner(mm=False, prompt=True)
    _drop_dead_inputs_embeds(r)
    assert r.inputs_embeds == "<buffer>", r.inputs_embeds

    # Already None (e.g. buffer never allocated): stays None, no crash.
    r = FakeRunner(mm=False, prompt=False, embeds=None)
    _drop_dead_inputs_embeds(r)
    assert r.inputs_embeds is None

    # Missing attrs degrade to text-only, never raise.
    class Bare:
        inputs_embeds = "<buffer>"

    b = Bare()
    _drop_dead_inputs_embeds(b)
    assert b.inputs_embeds is None

    print("inputs_embeds_optional self-check ok")


if __name__ == "__main__":
    _self_check()
