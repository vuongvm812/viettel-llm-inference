"""fp8 the LFM2 ``lm_head`` by untying it from the input embedding.

TPOT lever. lm_head is ~268 MB/token of bf16 weight traffic -- ~15.7% of decode, the
single largest un-quantized chunk once the body is fp8 (see quant_fp8.py). Decode is
memory-bandwidth bound, so cutting it cuts per-token latency.

Why this isn't an ignore-list edit: ``ParallelLMHead`` subclasses ``VocabParallelEmbedding``,
not ``LinearBase``, so ``VtlFp8Config.get_quant_method`` skips it structurally -- and with
``tie_embedding: true`` (LFM2.5) ``lm_head.weight`` is the *same Parameter object* as
``embed_tokens.weight``, so quantizing it in place would corrupt the bf16 embedding gather.

The untie: ``ops.scaled_fp8_quant`` is out-of-place, so the fp8 ``qweight`` it returns is fresh
storage. ``replace_parameter(layer, "weight", ...)`` rebinds ONLY ``lm_head``'s attribute to it
(setattr, not in-place copy), leaving ``embed_tokens.weight`` bf16 and untouched. Result: a
private fp8 lm_head weight (+~134 MB) alongside the retained bf16 embedding.

Opt-in via ``VTL_ENABLE_FP8_LM_HEAD`` (default OFF): logits are the most quality-sensitive GEMM
in the model, so ship this behind an A/B. TP=1 only (our deploy); any miss or error falls back
to the stock bf16 lm_head -- never a silently corrupted one. See [[quant_fp8]].
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("vtl")

ENABLE_ENV = "VTL_ENABLE_FP8_LM_HEAD"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def fp8_lm_head_enabled() -> bool:
    return os.environ.get(ENABLE_ENV, "").strip().lower() in _TRUTHY


def _fp8_quantize_lm_head(layer, scaled_fp8_quant, replace_parameter) -> bool:
    """Untie + per-output-channel fp8-quantize ``layer.weight`` in place on ``layer``.

    ``scaled_fp8_quant`` is out-of-place, so ``qweight`` is fresh storage: replacing the
    parameter breaks any ``embed_tokens`` alias without mutating the shared tensor. Idempotent.
    Returns True if it quantized, False if it was already done.
    """
    if getattr(layer, "_vtl_fp8_lm_head_done", False):
        return False
    orig_ptr = layer.weight.data_ptr()
    # weight is [vocab, hidden]; per-token over its rows == per-output-channel.
    qweight, weight_scale = scaled_fp8_quant(
        layer.weight, scale=None, use_per_token_if_dynamic=True
    )
    # Fresh storage guarantees the shared embedding weight was not touched.
    assert qweight.data_ptr() != orig_ptr, "fp8 lm_head quant aliased the shared weight"
    # CUTLASS wants B column-major [K, N] = [hidden, vocab].
    replace_parameter(layer, "weight", qweight.t().data)
    replace_parameter(layer, "weight_scale", weight_scale.data)
    layer._vtl_fp8_lm_head_done = True
    return True


def _build_method_cls():
    """Build the fp8 lm_head quant method (lazy: needs vLLM symbols)."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )
    from vllm.model_executor.utils import replace_parameter

    class VtlFp8LMHeadMethod(UnquantizedEmbeddingMethod):
        """Unquantized embedding method + fp8 weight/apply for the logits GEMM.

        Inherits ``create_weights``/``embedding``/``tie_weights`` unchanged (the weight is
        built, tied, and loaded exactly as stock); only the post-load compress and the logits
        ``apply`` are overridden.
        """

        def process_weights_after_loading(self, layer) -> None:
            from vllm import _custom_ops as ops

            try:
                _fp8_quantize_lm_head(layer, ops.scaled_fp8_quant, replace_parameter)
                log.info(
                    "vtl: fp8 lm_head installed (untied), weight=%s",
                    tuple(layer.weight.shape),
                )
            except Exception:
                # Quant runs only after a successful call, so a failure leaves the bf16
                # alias intact; apply() falls back on the missing done-flag.
                log.exception("vtl: fp8 lm_head quantize failed; leaving bf16")

        def apply(self, layer, x, bias=None):
            if not getattr(layer, "_vtl_fp8_lm_head_done", False):
                return super().apply(layer, x, bias)  # quant skipped/failed -> bf16
            from vllm import _custom_ops as ops

            x_fp8, x_scale = ops.scaled_fp8_quant(
                x, scale=None, use_per_token_if_dynamic=True
            )
            return ops.cutlass_scaled_mm(
                x_fp8,
                layer.weight,
                scale_a=x_scale,
                scale_b=layer.weight_scale,
                out_dtype=x.dtype,
                bias=bias,
            )

    return VtlFp8LMHeadMethod


def maybe_lm_head_method(layer):
    """Return a fp8 lm_head method for an applicable ``ParallelLMHead``, else None.

    Fully guarded -- called from ``get_quant_method``, which must never raise.
    """
    if not fp8_lm_head_enabled():
        return None
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

        if not isinstance(layer, ParallelLMHead):
            return None
        if get_tensor_model_parallel_world_size() != 1:
            log.warning("vtl: fp8 lm_head needs TP=1; leaving lm_head bf16")
            return None
        return _build_method_cls()()
    except Exception:
        log.exception("vtl: fp8 lm_head routing failed; leaving lm_head bf16")
        return None


def _self_check() -> None:
    """No vLLM: fakes verify the untie logic (fresh storage, idempotency, no embed mutation)."""

    class _FakeTensor:
        def __init__(self, ptr, shape=(65536, 2048)):
            self._ptr = ptr
            self.shape = shape

        def data_ptr(self):
            return self._ptr

        def t(self):
            return _FakeTensor(self._ptr, (self.shape[1], self.shape[0]))

        @property
        def data(self):
            return self

    calls = {}

    def fake_quant(weight, scale=None, use_per_token_if_dynamic=False):
        # Out-of-place: returns tensors at DIFFERENT addresses than the input.
        return _FakeTensor(0xF8, weight.shape), _FakeTensor(0x5C, (weight.shape[0], 1))

    def fake_replace(layer, name, new_data):
        calls[name] = new_data
        setattr(layer, name, new_data)

    class _Layer:
        pass

    # Shared bf16 weight, aliased by both lm_head and embed_tokens (same object/ptr).
    shared = _FakeTensor(0xB16)
    lm = _Layer()
    lm.weight = shared
    embed_weight = shared  # the tie: same object

    assert _fp8_quantize_lm_head(lm, fake_quant, fake_replace) is True
    # lm_head now points at fresh fp8 storage; the shared embed weight is untouched.
    assert lm.weight.data_ptr() == 0xF8
    assert embed_weight.data_ptr() == 0xB16
    assert lm.weight.data_ptr() != embed_weight.data_ptr()
    assert calls["weight_scale"].shape == (65536, 1)
    assert lm.weight.shape == (2048, 65536)  # transposed [hidden, vocab] for CUTLASS
    assert getattr(lm, "_vtl_fp8_lm_head_done") is True

    # Idempotent: a second call is a no-op.
    assert _fp8_quantize_lm_head(lm, fake_quant, fake_replace) is False

    # Gating: disabled by default -> no method.
    os.environ.pop(ENABLE_ENV, None)
    assert fp8_lm_head_enabled() is False
    os.environ[ENABLE_ENV] = "1"
    assert fp8_lm_head_enabled() is True
    os.environ.pop(ENABLE_ENV, None)

    print("fp8_lm_head self-check ok")


if __name__ == "__main__":
    _self_check()
