"""Quantize LFM2's ``lm_head`` -- the last large bf16 tensor in the decode path.

WHY. After [[quant_w4a8]] shrinks every ``LinearBase`` to int4, the per-decode-step weight
budget is 852 MB and ``lm_head`` is **268 MB of it (31.5%)** -- the single biggest remaining
chunk, and it grew in relative terms precisely because W4A8 shrank everything else:

    lm_head          bytes/step   vs bf16     @600 GB/s (the judge's MIG 1g.18gb slice)
    bf16   65536x2048   268.4 MB      --        0.447 ms
    fp8                 134.5 MB   -134 MB      0.224 ms
    int4                 75.9 MB   -192 MB      0.127 ms

int4 = 67.1 MB packed + 8.4 MB group scales (1.05 M scales x ScalePackSize 8, see the
[[quant_w4a8]] docstring -- do NOT re-derive this from the scale count) + 0.26 MB channel scales.

WHY IT IS A SEPARATE FILE. ``lm_head`` is not a ``LinearBase``. It is a ``ParallelLMHead``
(a ``VocabParallelEmbedding``), so ``VtlW4A8Config.get_quant_method`` returned ``None`` for it
and it stayed bf16. Reaching it needs a different create_weights, a tie_weights contract, and a
second (fp8) rung that has no equivalent anywhere else in the tree -- none of which belongs in
quant_w4a8.py's already-long linear path.

WHY THIS WORKS WITHOUT UNTYING THE EMBEDDINGS. config.json has ``tie_embedding: true``, and
``Lfm2ForCausalLM`` calls ``lm_head.tie_weights(embed_tokens)`` (lfm2.py:479), which routes to
``QuantizeMethodBase.tie_weights`` -> ``layer.weight = embed_tokens.weight``. Four facts make
quantizing the head safe anyway:

  * ``_get_logits`` calls ``lm_head.quant_method.apply(lm_head, hidden_states, bias=...)``
    (logits_processor.py:93), so installing a method here is all that is needed.
  * ``is_embedding_layer`` is ``type(self) is VocabParallelEmbedding``
    (vocab_parallel_embedding.py:284), which is False for ParallelLMHead -- so unlike
    ``embed_tokens`` we do NOT have to implement ``embedding()``.
  * ``model_loader/utils.py:101`` calls ``process_weights_after_loading`` on every module with a
    ``quant_method``, lm_head included, after load_weights has finished.
  * Releasing the bf16 original REBINDS ``lm_head.weight``; it does not free the storage, because
    ``embed_tokens.weight`` is a separate module attribute pointing at the same tensor. The
    embedding lookup keeps working. This reads like a use-after-free and is not one.

So the resident cost is +76 MB (int4) on top of the 268 MB embedding table we keep either way.

THE LADDER. ``VTL_LM_HEAD_QUANT`` = ``int4`` (default) | ``fp8`` | ``off``. The output head is the
single most accuracy-sensitive weight in the model and group-128 symmetric RTN with no calibration
is the weakest 4-bit recipe there is, so the middle rung is not optional decoration -- it is the
difference between "one env var" and "rebuild" when the eval regresses. Run
``bench/eval_quality.py`` across all three before trusting any of them.

``VTL_W4A8_IGNORE`` still wins: a substring match on the prefix forces bf16 regardless of this
knob. Two knobs that both mean "keep lm_head out of int4" is exactly the drift hazard
``quant_w4a8.schedule()`` exists to avoid, so the precedence is stated in one place -- the
``ParallelLMHead`` branch of ``VtlW4A8Config.get_quant_method`` -- and nowhere else.

EVERY FAILURE MODE IS bf16, NOT A CRASH. ``get_quant_method`` runs at model load and must never
raise. An unknown mode string, a shape the CUTLASS kernel cannot implement, a missing op, a throw
during quantization: all of them leave the head bf16 and log. "We shipped bf16 believing we
shipped int4" is the failure this file is most likely to have, so the outcome is logged from
``process_weights_after_loading`` -- i.e. after the weights actually changed -- and grepped by
``make verify``.
"""

from __future__ import annotations

import functools
import logging
import os

log = logging.getLogger("vllm.vtl.lm_head_quant")

MODE_ENV = "VTL_LM_HEAD_QUANT"
VALID_MODES = ("int4", "fp8", "off")
DEFAULT_MODE = "int4"


def parse_mode(raw: str | None) -> str:
    """Normalize ``VTL_LM_HEAD_QUANT``. Unset -> the shipped default; unknown -> ``off``.

    Unknown falls to ``off`` rather than the default on purpose. Every other value of this knob
    changes model OUTPUT, and a typo that silently lands on int4 would ship an unmeasured
    accuracy change while the operator believes they asked for something else. bf16 is the only
    answer that cannot be wrong.
    """
    if raw is None or not raw.strip():
        return DEFAULT_MODE
    value = raw.strip().lower()
    if value not in VALID_MODES:
        log.warning(
            "vtl: %s=%r is not one of %s; leaving lm_head in bf16",
            MODE_ENV, raw, ", ".join(VALID_MODES),
        )
        return "off"
    return value


@functools.cache
def mode() -> str:
    """THE single read of ``VTL_LM_HEAD_QUANT`` in the process. Same rule as
    ``quant_w4a8.schedule()``: one reader, so a sweep cannot half-apply."""
    return parse_mode(os.environ.get(MODE_ENV))


def _log_outcome(requested: str, ok: bool) -> None:
    """Report what the head ACTUALLY ended up as, from process_weights_after_loading.

    Deliberately after the fact. Logging the intent at get_quant_method time would print
    "quantized to int4" in exactly the world where the quantize then threw and the head stayed
    bf16 -- the one failure this file most needs to be loud about. Grepped by ``make verify``.
    """
    if ok:
        log.info("vtl: lm_head quantized to %s", requested)
    else:
        log.warning(
            "vtl: lm_head quantization (%s) FAILED; the head is serving bf16 "
            "-- that is 268 MB/step, 31%% of the decode weight budget",
            requested,
        )


def _quant_act_fp8(x_2d):
    """Per-token fp8-e4m3 activation quant, via the CUDA op -- NEVER via ``QuantFP8``.

    ``QuantFP8`` is a ``CustomOp``, and under ``-O3`` vLLM disables custom ops so inductor can
    see through them into ``forward_native`` (that is exactly what lets RMSNormQuantFusionPass
    match norm+quant in the model graph). ``compute_logits`` is a SEPARATE compile region that
    no fusion pass reaches, so here the decomposition buys nothing and costs everything: inductor
    lowers the ``.to(float8_e4m3fn)`` to ``tl.float8e4nv`` and Triton refuses it --

        ValueError("type fp8e4nv not supported in this architecture.
                    The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')")

    -- which is an InductorError during ``profile_run``, i.e. the engine never boots. That is
    the sm_80 dtype list; if the box is really sm_90 then Triton is being handed stale device
    properties, which points at the AOT-compile cache baked in by ``COPY docker/cache/``
    (Dockerfile:145) plus ``VLLM_USE_AOT_COMPILE=1``. Worth chasing separately -- a stale target
    miscompiles any NEW graph shape, not just this one -- but this path should not depend on it
    either way.

    ``ops.scaled_fp8_quant`` is a registered custom op, so inductor emits an extern call and
    never codegens fp8. Same op the weight side of this file already uses, and the same kernel
    [[dynamic_per_token_quant]] overrides for ``o_proj``. Output is identical to QuantFP8's
    PER_TOKEN dynamic mode.
    """
    from vllm import _custom_ops as ops

    return ops.scaled_fp8_quant(x_2d, scale=None, use_per_token_if_dynamic=True)


@functools.cache
def _int4_method_cls():
    """int4 rung: [[quant_w4a8]]'s linear method with an embedding-shaped create_weights.

    Everything that matters -- the RTN quantize/pack, the packed-parameter install, the
    install-or-nothing try, ``ops.cutlass_w4a8_mm``, and the inherited ``tie_weights`` -- is
    reused verbatim. ``apply`` already derives its output width from
    ``weight_chan_scale.shape[0]``, so N=65536 needs no special case.

    Cached on the FUNCTION for the same pickle reason as ``quant_w4a8._linear_method_cls``.

    LOAD-TIME PEAK ~2.5 GiB, the largest transient the process ever makes -- 4x the biggest body
    layer (w13, ~0.6 GiB), because lm_head is 4x its size. ``_quantize_and_pack`` promotes to
    fp32, and ``quantize_weights`` (quant_utils.py:610-702) then holds roughly five live
    K*N*4-byte buffers at once: the fp32 transpose, the permute/reshape copy, the ``w / w_s``
    temporary, ``w_q`` as int32, and the dequantized ``w_ref`` it builds even though we discard
    it. Freed before the KV cache is sized, so it fits on an 18 GiB MIG slice with room to
    spare and is a non-event on a full 141 GiB H200 -- but it is the first thing to look at if
    model load OOMs, and the failure is silent (the parent's try leaves the head bf16). The fp8
    rung below has no such peak: one fused op, no fp32 round trip.
    """
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    from vtl.patches.quant_w4a8 import _linear_method_cls

    class VtlW4A8LMHeadMethod(_linear_method_cls()):
        def __init__(self) -> None:
            super().__init__()
            # The inherited apply() calls `self.quant_fp8(x_2d)`. In the model graph that is
            # correct -- it is the node RMSNormQuantFusionPass matches. Here it runs in
            # compute_logits, a separate compile region no pass reaches, where the CustomOp
            # decomposition is a Triton fp8 codegen that can fail the boot outright. Rebinding
            # the attribute reuses the parent's apply() verbatim; see _quant_act_fp8.
            self.quant_fp8 = _quant_act_fp8

        def create_weights(self, layer, *args, **kwargs) -> None:
            # Called UNBOUND. VocabParallelEmbedding wants a plain Parameter with
            # input_dim=1/output_dim=0 and its own vocab weight_loader, not the
            # ModelWeightParameter the linear path allocates. The body never touches `self`,
            # so borrowing it is exact -- and re-implementing the allocation here would be one
            # more copy of a layout that has to match vLLM's loader.
            UnquantizedEmbeddingMethod.create_weights(self, layer, *args, **kwargs)

        def process_weights_after_loading(self, layer) -> None:
            super().process_weights_after_loading(layer)
            _log_outcome("int4", getattr(layer, "_vtl_w4a8_done", False))

    return VtlW4A8LMHeadMethod


@functools.cache
def _fp8_method_cls():
    """fp8 rung: per-output-channel fp8 weights + per-token fp8 activations.

    Same two ops the rest of the tree already uses -- ``ops.scaled_fp8_quant(...,
    use_per_token_if_dynamic=True)`` for the weight (as in [[quant_fp8]]'s channelwise upgrade)
    and ``ops.cutlass_scaled_mm`` for the GEMM (as in the fp8 arm of ``_vtl_out_proj_quantized``
    in the short_conv patch). No third spelling of fp8 in this repo.
    """
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    class VtlFp8LMHeadMethod(QuantizeMethodBase):
        def create_weights(self, layer, *args, **kwargs) -> None:
            UnquantizedEmbeddingMethod.create_weights(self, layer, *args, **kwargs)

        def process_weights_after_loading(self, layer) -> None:
            if getattr(layer, "_vtl_lm_head_fp8_done", False):
                return
            try:
                # use_per_token_if_dynamic on a [N, K] weight scales per ROW, i.e. per output
                # channel -> [N, 1]. Strictly better than per-tensor for a 65536-row head.
                qweight, weight_scale = ops.scaled_fp8_quant(
                    layer.weight.data, scale=None, use_per_token_if_dynamic=True
                )
                # Install-or-nothing, same discipline as the int4 rung: a half-built layer whose
                # done-flag is False but whose fp8 params exist is wrong numerics, not slow ones.
                # `.t()` (a view, not a copy) because cutlass_scaled_mm wants B column-major [K, N].
                layer.register_parameter(
                    "weight_fp8", torch.nn.Parameter(qweight.t(), requires_grad=False)
                )
                layer.register_parameter(
                    "weight_chan_scale",
                    torch.nn.Parameter(weight_scale, requires_grad=False),
                )
                # Rebinds lm_head.weight only. embed_tokens.weight is a separate attribute on a
                # separate module holding the same storage, so the embedding table survives --
                # see the module docstring.
                layer.weight = torch.nn.Parameter(
                    torch.empty(0, dtype=layer.weight.dtype, device=layer.weight.device),
                    requires_grad=False,
                )
                layer._vtl_lm_head_fp8_done = True
            except Exception as exc:
                layer._vtl_lm_head_fp8_done = False
                log.warning("vtl: lm_head fp8 quantize failed (%s)", exc)
            _log_outcome("fp8", getattr(layer, "_vtl_lm_head_fp8_done", False))

        def apply(self, layer, x, bias=None):
            if not getattr(layer, "_vtl_lm_head_fp8_done", False):
                return torch.nn.functional.linear(x, layer.weight, bias)  # bf16 fallback

            x_2d = x.reshape(-1, x.shape[-1])
            xq, x_scales = _quant_act_fp8(x_2d)
            out = ops.cutlass_scaled_mm(
                xq,
                layer.weight_fp8,
                scale_a=x_scales,
                scale_b=layer.weight_chan_scale,
                out_dtype=x.dtype,
                bias=bias,
            )
            return out.reshape(x.shape[:-1] + (out.shape[-1],))

    return VtlFp8LMHeadMethod


def lm_head_method(layer):
    """Quant method for a ``ParallelLMHead``, or ``None`` to leave it bf16.

    ``None`` is the correct "not ours" signal here: VocabParallelEmbedding maps it to
    ``UnquantizedEmbeddingMethod`` (vocab_parallel_embedding.py:279-280). That is the opposite of
    the LinearBase path, where ``None`` makes ``LinearBase.__init__`` raise -- see the except arm
    of ``VtlW4A8Config.get_quant_method``.
    """
    requested = mode()
    if requested == "off":
        return None
    try:
        # ParallelLMHead has no input_size/output_size -- those are LinearBase attributes.
        # embedding_dim is K; num_embeddings_padded (NOT num_embeddings) is the N the weight is
        # actually allocated at, and the padded value is what the GEMM sees.
        in_features = layer.embedding_dim
        out_features = layer.num_embeddings_padded

        if requested == "int4":
            # Ask the real kernel about K/N divisibility and arch rather than re-hardcoding it.
            from vtl.patches.quant_w4a8 import _can_implement

            if not _can_implement(in_features, out_features):
                log.warning(
                    "vtl: w4a8 kernel cannot implement lm_head (%dx%d); it stays bf16",
                    in_features, out_features,
                )
                return None
            return _int4_method_cls()()

        return _fp8_method_cls()()
    except Exception as exc:
        log.warning("vtl: lm_head quantization unavailable (%s); it stays bf16", exc)
        return None


def _self_check() -> None:
    # Unset/blank -> the shipped default; the three names round-trip; anything else is bf16.
    assert parse_mode(None) == "int4"
    assert parse_mode("") == "int4"
    assert parse_mode("   ") == "int4"
    assert parse_mode("int4") == "int4"
    assert parse_mode("fp8") == "fp8"
    assert parse_mode("off") == "off"
    assert parse_mode("  FP8  ") == "fp8"      # case/space tolerant, unlike VTL_W4A8_SCHEDULE
    assert parse_mode("int8") == "off"         # plausible typo must not silently quantize
    assert parse_mode("4bit") == "off"
    assert parse_mode("true") == "off"         # not a boolean knob
    assert DEFAULT_MODE in VALID_MODES

    # The traffic claim in the docstring, which is the whole reason this file exists. Anchored
    # here so a future dim change makes the arithmetic fail rather than the comment rot.
    vocab, hidden, group, scale_pack = 65536, 2048, 128, 8
    numel = vocab * hidden
    assert numel * 2 == 268435456                                    # bf16 268.4 MB
    assert numel * 1 == 134217728                                    # fp8  134.2 MB
    int4 = numel // 2 + (numel // group) * scale_pack + vocab * 4    # packed + group + channel
    assert 75_000_000 < int4 < 76_500_000, int4
    # Both dims must clear the CUTLASS W4A8 K%128/N%128 gate or the int4 rung is dead on arrival.
    assert hidden % group == 0 and vocab % group == 0

    print("lm_head_quant self-check ok")


if __name__ == "__main__":
    _self_check()
