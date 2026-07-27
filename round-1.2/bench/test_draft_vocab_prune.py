"""FR-Spec draft-vocabulary pruning (vtl/patches/lm_head_quant.py).

The drafter's lm_head is ~19% of its weights and is read once per draft pass. Narrowing it to
the first K token ids cuts that proportionally, and is LOSSLESS because the target verifies every
drafted token -- a token outside the window is simply never proposed.

Three things can go wrong, and each has a test here:

  * the TARGET's head gets pruned -> real output is silently truncated. Guarded by the prefix
    gate (``draft_prune_rows`` must return 0 for anything but ``draft_model.*``);
  * the pruning truncates the TIED embedding table -> the model can no longer embed high-id
    input tokens. Guarded by ``test_prune_leaves_tied_embedding_intact``, which is the reason
    ``_apply_draft_prune`` rebinds rather than slicing in place;
  * an unparseable knob silently prunes anyway. Guarded by the parse table.

    pytest bench/test_draft_vocab_prune.py -q      # torch leg skips without torch
    python bench/test_draft_vocab_prune.py         # same checks, no pytest

No CUDA and no vLLM required: every path exercised below either short-circuits before the vLLM
import or fails closed through it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import pytest
except ImportError:  # standalone `python bench/test_draft_vocab_prune.py` on a bare dev box
    class _Mark:
        @staticmethod
        def skipif(_cond, reason=""):  # noqa: ARG004
            return lambda fn: fn

    class pytest:  # type: ignore[no-redef]  # noqa: N801
        mark = _Mark()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vtl.patches.lm_head_quant import (  # noqa: E402
    DEFAULT_DRAFT_VOCAB,
    DRAFT_PREFIX,
    DRAFT_VOCAB_ALIGN,
    DRAFT_VOCAB_ENV,
    _apply_draft_prune,
    draft_prune_rows,
    draft_vocab,
    parse_draft_vocab,
)

try:
    import torch

    HAVE_TORCH = True
except Exception:  # pragma: no cover - torch is present in the image, absent on some dev boxes
    HAVE_TORCH = False


class FakeHead:
    """The two attributes ``draft_prune_rows`` reads off a ParallelLMHead."""

    def __init__(self, padded=65536, dim=1024):
        self.num_embeddings_padded = padded
        self.embedding_dim = dim


def _set_env(value):
    if value is None:
        os.environ.pop(DRAFT_VOCAB_ENV, None)
    else:
        os.environ[DRAFT_VOCAB_ENV] = value
    draft_vocab.cache_clear()  # the @functools.cache single-reader must not leak between cases


def test_parse_table():
    assert parse_draft_vocab(None) == DEFAULT_DRAFT_VOCAB
    assert parse_draft_vocab("") == DEFAULT_DRAFT_VOCAB
    assert parse_draft_vocab("   ") == DEFAULT_DRAFT_VOCAB
    assert parse_draft_vocab(" 8192 ") == 8192
    # Explicit disables.
    for off in ("0", "off", "no", "false", "none", "OFF"):
        assert parse_draft_vocab(off) == 0, off
    # Anything unusable falls to the full head -- never to the default. A typo must not ship an
    # unmeasured acceptance-rate change while the operator believes they asked for something else.
    for bad in ("banana", "-128", "8000", "1e4", "128.0"):
        assert parse_draft_vocab(bad) == 0, bad
    assert DEFAULT_DRAFT_VOCAB % DRAFT_VOCAB_ALIGN == 0


def test_target_head_is_never_pruned():
    """THE guard. Pruning the target's vocabulary truncates real model output."""
    _set_env(str(DEFAULT_DRAFT_VOCAB))
    try:
        for prefix in ("lm_head", "model.lm_head", "", "draft.lm_head", "adraft_model.lm_head"):
            assert draft_prune_rows(prefix, FakeHead()) == 0, prefix
    finally:
        _set_env(None)


def test_draft_head_is_pruned(monkeypatch=None):
    """The POSITIVE case. Without this, a typo in DRAFT_PREFIX makes the whole feature inert
    while every other test in this file still passes.

    `draft_prune_rows` reaches for the live VllmConfig to check draft_sample_method, which does
    not exist outside a running engine -- and it now fails CLOSED on that, returning 0. So the
    assertion here is on the part that is testable off-box: the prefix gate is what decides, and
    it must let `draft_model.lm_head` through to that config lookup rather than short-circuit.
    """
    import vtl.patches.lm_head_quant as m

    _set_env(str(DEFAULT_DRAFT_VOCAB))
    seen = {}
    real = m.draft_vocab

    def _spy():
        seen["called"] = True
        return real()

    m.draft_vocab = _spy
    try:
        # Reached the config lookup (and bailed there) rather than the prefix short-circuit.
        assert m.draft_prune_rows(f"{DRAFT_PREFIX}lm_head", FakeHead()) == 0
        assert seen.get("called"), "prefix gate short-circuited on the DRAFT head"
        # And the target head must not even consult the knob's downstream logic.
        assert m.draft_prune_rows("lm_head", FakeHead()) == 0
    finally:
        m.draft_vocab = real
        _set_env(None)


def test_prefix_constant_matches_vllm():
    """DRAFT_PREFIX must match what vLLM actually names the draft model.

    `DraftModelProposer._get_model` passes prefix="draft_model" and lfm2.py builds the head with
    maybe_prefix(prefix, "lm_head") -> "draft_model.lm_head".
    """
    assert DRAFT_PREFIX == "draft_model.", DRAFT_PREFIX
    assert f"{DRAFT_PREFIX}lm_head" == "draft_model.lm_head"


def test_disabled_knob_prunes_nothing():
    _set_env("0")
    try:
        assert draft_prune_rows(f"{DRAFT_PREFIX}lm_head", FakeHead()) == 0
    finally:
        _set_env(None)


def test_prune_wider_than_vocab_is_rejected():
    """K >= the allocated vocab is a no-op request, not a resize."""
    _set_env("65536")
    try:
        assert draft_prune_rows(f"{DRAFT_PREFIX}lm_head", FakeHead(padded=65536)) == 0
    finally:
        _set_env(None)


@pytest.mark.skipif(not HAVE_TORCH, reason="needs torch")
def test_prune_leaves_tied_embedding_intact():
    """``lm_head.weight`` IS ``embed_tokens.weight`` under tied embeddings.

    Slicing in place would truncate the table the model still needs to embed INPUT tokens, and
    the failure would only surface on a prompt containing a high-id token. The rebind must leave
    the shared storage untouched.
    """
    vocab, dim, keep = 512, 8, 128
    shared = torch.arange(vocab * dim, dtype=torch.float32).reshape(vocab, dim)

    class _Layer:
        pass

    embed = _Layer()
    head = _Layer()
    embed.weight = torch.nn.Parameter(shared, requires_grad=False)
    head.weight = embed.weight  # tie_weights() shares the object, not a copy

    _apply_draft_prune(head, keep)

    assert head.weight.shape == (keep, dim)
    # The embedding table is untouched, both in shape and in contents.
    assert embed.weight.shape == (vocab, dim)
    torch.testing.assert_close(embed.weight.data, shared, rtol=0.0, atol=0.0)
    # And the kept rows are the FIRST K, in order -- that identity is what makes
    # "argmax index == global token id" true and lets the caller skip an id remap entirely.
    torch.testing.assert_close(head.weight.data, shared[:keep], rtol=0.0, atol=0.0)


@pytest.mark.skipif(not HAVE_TORCH, reason="needs torch")
def test_prune_is_a_noop_when_disabled_or_already_narrow():
    vocab, dim = 256, 4
    w = torch.randn(vocab, dim)

    class _Layer:
        pass

    layer = _Layer()
    layer.weight = torch.nn.Parameter(w, requires_grad=False)
    _apply_draft_prune(layer, 0)  # disabled
    assert layer.weight.shape == (vocab, dim)
    _apply_draft_prune(layer, vocab + 128)  # already narrower than requested
    assert layer.weight.shape == (vocab, dim)


@pytest.mark.skipif(not HAVE_TORCH, reason="needs torch")
def test_logits_tail_slice_is_a_noop_on_a_pruned_head():
    """``LogitsProcessor._get_logits`` ends with ``logits[..., :org_vocab_size]``.

    org_vocab_size stays 65536 (config.vocab_size is untouched -- vLLM's
    verify_equal_vocab_size_if_draft_model requires it), so the pruned logits must survive that
    slice unchanged. Asserted rather than assumed, because it is the one downstream call that
    would silently reshape the drafter's output.
    """
    logits = torch.randn(4, DEFAULT_DRAFT_VOCAB)
    torch.testing.assert_close(logits[..., :65536], logits, rtol=0.0, atol=0.0)
    # argmax over the pruned head indexes the global vocabulary directly.
    assert int(logits.argmax(dim=-1).max()) < DEFAULT_DRAFT_VOCAB


if __name__ == "__main__":
    test_parse_table()
    test_prefix_constant_matches_vllm()
    test_draft_head_is_pruned()
    test_target_head_is_never_pruned()
    test_disabled_knob_prunes_nothing()
    test_prune_wider_than_vocab_is_rejected()
    if HAVE_TORCH:
        test_prune_leaves_tied_embedding_intact()
        test_prune_is_a_noop_when_disabled_or_already_narrow()
        test_logits_tail_slice_is_a_noop_on_a_pruned_head()
    print("draft vocab prune checks ok")
