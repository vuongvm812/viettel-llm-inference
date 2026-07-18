"""Route LFM2's GQA attention through the vendored official FlashAttention-3 kernel.

Gated, fail-closed A/B lever: ``VTL_ENABLE_FA3_ATTN=1`` monkeypatches
``FlashAttentionImpl.forward`` (vLLM's v1 flash-attn backend) to call the vendored
``flash_attn_3.flash_attn_with_kvcache`` (official Dao-AILab hopper kernel, sm_90a)
instead of vLLM's bundled ``flash_attn_varlen_func``. Default OFF -> stock runs
untouched.

HONEST CEILING (ponytail: this is an A/B experiment, not a known win): vLLM v0.25.0
already defaults to FA3 on Hopper (``fa_utils.py:get_flash_attn_version`` -> 3), and
its bundled ``vllm_flash_attn`` is the *same* Dao-AILab FA3 lineage. So the most likely
outcome is PARITY. The real risk is a TPOT regression: the optimized compose runs
``cudagraph_mode=FULL_AND_PIECEWISE`` and this forward runs eager inside piecewise
regions -- a monkeypatched call into upstream FA3 may not capture into cudagraph / may
add host overhead. MEASURE (make bench-kernel + serving A/B) before trusting it. Ships
default-OFF; every failure path degrades to stock and never crashes the server.

Data-flow facts this relies on (vLLM v0.25.0, flash_attn.py):
  * New-token KV is written to the paged cache by ``do_kv_cache_update`` in a SEPARATE
    phase BEFORE ``forward`` (flash_attn.py:1009,1033). So firing (skipping ``orig``)
    never skips a cache write; we read the full cache via page_table + cache_seqlens.
  * ``forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
    output_scale=None, output_block_scale=None)`` -- query is packed [num_tokens,
    nheads, hd]; ``kv_cache.unbind(1)`` -> paged (num_blocks, block_size, nheads_kv, hd),
    exactly FA3's paged layout. cu_seqlens_q=query_start_loc, cache_seqlens=seq_lens,
    block_table, q/k/v_descale from layer._q/k/v_scale for fp8 KV.
"""

from __future__ import annotations

import logging
import os

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vtl")

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _fa3_enabled() -> bool:
    return os.environ.get("VTL_ENABLE_FA3_ATTN", "0").strip().lower() in _TRUTHY


def _device_major() -> int | None:
    """CUDA compute-capability major, or None if undeterminable (CPU self-check)."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_capability()[0]
    except Exception:
        pass
    return None


def _should_fire(
    *,
    enabled: bool,
    device_major: int | None,
    has_metadata: bool,
    use_cascade: bool,
    is_decoder_self_attn: bool,
    dcp_world_size: int,
    causal_is_bool: bool,
    alibi_is_none: bool,
    output_scale_is_none: bool,
) -> bool:
    """Pure fire-condition predicate (no torch/vLLM), so it is unit-testable on CPU.

    FA3 with_kvcache here serves only the plain paged decoder self-attention path.
    Anything else (cascade, encoder/cross, decode-context-parallel, per-sequence
    dynamic causal, ALiBi, fused output quant) falls back to stock.
    """
    if not enabled:
        return False
    if device_major != 9:  # sm90 (H200); FA3 is sm_90a-only in our build
        return False
    if not has_metadata:
        return False
    if use_cascade:
        return False
    if not is_decoder_self_attn:
        return False
    if dcp_world_size > 1:
        return False
    if not causal_is_bool:  # per-sequence tensor causal needs FA4
        return False
    if not alibi_is_none:
        return False
    if not output_scale_is_none:  # fused output quant unsupported on this path
        return False
    return True


_shapes_dumped = False
_fired_logged = False


def _dump_shapes(**tensors) -> None:  # noqa: ANN003
    """Log the real forward arg shapes/dtypes ONCE, to verify the layout on the box."""
    global _shapes_dumped
    if _shapes_dumped:
        return
    _shapes_dumped = True

    def desc(t):  # noqa: ANN001
        if t is None:
            return "None"
        try:
            return f"{tuple(t.shape)}:{str(t.dtype).replace('torch.', '')}"
        except Exception:
            return type(t).__name__

    log.info("vtl: FA3 forward shapes -> %s", {k: desc(v) for k, v in tensors.items()})


def _install(cls) -> None:  # noqa: ANN001
    """Wrap FlashAttentionImpl.forward: fire FA3 on the supported path, else stock."""
    import torch

    if already_patched(cls, "forward"):
        return
    orig_forward = cls.forward

    # Resolve the decoder self-attn marker + fp8 helpers once (lazily, at install).
    try:
        from vllm.attention.backends.abstract import AttentionType

        _DECODER = AttentionType.DECODER
    except Exception:
        _DECODER = "decoder"
    try:
        from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
    except Exception:
        canonicalize_singleton_dim_strides = lambda t: t  # noqa: E731
    try:
        from vllm.attention.backends.abstract import is_quantized_kv_cache
    except Exception:
        try:
            from vllm.v1.attention.backends.utils import is_quantized_kv_cache
        except Exception:
            is_quantized_kv_cache = lambda _dt: False  # noqa: E731

    from flash_attn_3 import flash_attn_with_kvcache

    # Hoist process-constant gates out of the per-forward hot path. apply() only installs
    # when armed on sm90, so both are fixed for the process; recomputing them every decode
    # step is pure host overhead (vLLM warns to minimise CPU work in this method).
    _armed = _fa3_enabled()
    _major = _device_major()

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):  # noqa: ANN001
        global _fired_logged
        is_decoder = getattr(self, "attn_type", _DECODER) == _DECODER
        causal_is_bool = attn_metadata is not None and isinstance(
            getattr(attn_metadata, "causal", True), bool
        )
        fire = _should_fire(
            enabled=_armed,
            device_major=_major,
            has_metadata=attn_metadata is not None,
            use_cascade=bool(getattr(attn_metadata, "use_cascade", False))
            if attn_metadata is not None
            else False,
            is_decoder_self_attn=is_decoder,
            dcp_world_size=int(getattr(self, "dcp_world_size", 1) or 1),
            causal_is_bool=causal_is_bool,
            alibi_is_none=getattr(self, "alibi_slopes", None) is None,
            output_scale_is_none=output_scale is None and output_block_scale is None,
        )
        if not fire:
            return orig_forward(
                self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale, output_block_scale,
            )

        try:
            n = attn_metadata.num_actual_tokens
            key_cache, value_cache = kv_cache.unbind(1)
            key_cache = canonicalize_singleton_dim_strides(key_cache)
            value_cache = canonicalize_singleton_dim_strides(value_cache)

            cu_seqlens_q = attn_metadata.query_start_loc
            cache_seqlens = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            block_table = attn_metadata.block_table
            # NOTE: do NOT reuse attn_metadata.scheduler_metadata -- that blob is produced
            # by vLLM's BUNDLED get_scheduler_metadata op, a different format than upstream
            # FA3 expects. Pass None so upstream FA3 schedules internally. (No latency cost:
            # cache_seqlens grows every decode step, so a precomputed schedule can't be
            # cached across steps anyway.)

            window = (
                attn_metadata.sliding_window
                if getattr(attn_metadata, "sliding_window", None) is not None
                else getattr(self, "sliding_window", None)
            )
            window_size = tuple(window) if window is not None else (-1, -1)

            q_descale = k_descale = v_descale = None
            if is_quantized_kv_cache(self.kv_cache_dtype):
                from vllm.platforms import current_platform

                fp8 = current_platform.fp8_dtype()
                key_cache = key_cache.view(fp8)
                value_cache = value_cache.view(fp8)
                descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)
                if getattr(self, "supports_quant_query_input", False):
                    q_descale = layer._q_scale.expand(descale_shape)
                k_descale = layer._k_scale.expand(descale_shape)
                v_descale = layer._v_scale.expand(descale_shape)

            _dump_shapes(
                query=query[:n],
                key_cache=key_cache,
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
            )

            out = flash_attn_with_kvcache(
                q=query[:n],
                k_cache=key_cache,
                v_cache=value_cache,
                page_table=block_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                softmax_scale=self.scale,
                causal=bool(attn_metadata.causal),
                window_size=window_size,
                softcap=self.logits_soft_cap or 0.0,
                scheduler_metadata=None,
                # 0 = FA3's own split-count heuristic. We run EAGER (not cudagraph-captured),
                # so we don't need vLLM's fixed max_num_splits; the heuristic picks the split
                # count that best fills the H200's 132 SMs for small-batch decode -- the main
                # decode-occupancy lever for GQA 32:8 at TP=1. (If this path is ever cudagraph-
                # captured, a data-dependent grid would break capture -> revisit.)
                num_splits=0,
                pack_gqa=None,  # auto: packs the 4 q-heads/kv-head so KV is loaded once
            )
            # with_kvcache has no out= param: copy its result into vLLM's buffer.
            output[:n].copy_(out.view(output[:n].shape))
            if not _fired_logged:
                _fired_logged = True
                log.info("vtl: FA3 route fired (upstream flash_attn_with_kvcache)")
            return output
        except Exception as exc:  # any drift -> stock, never crash the model
            log.warning("vtl: FA3 route bailed (%s); stock backend", exc)
            return orig_forward(
                self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale, output_block_scale,
            )

    cls.forward = mark_patched(forward, orig_forward)


# Registry name "fa3_attn" so selection keys on VTL_ENABLE_FA3_ATTN (the same var the
# wrapper arms on) -- default OFF, so nothing is installed unless that var is truthy.
@register_patch("fa3_attn", default=False)
def apply() -> None:
    if not _fa3_enabled():
        return
    major = _device_major()
    if major is not None and major != 9:
        log.info("vtl: FA3 route needs sm90 (H200); device major=%s -> stock", major)
        return
    try:
        import flash_attn_3  # noqa: F401  -- built as flash_attn_3._C (sm_90a)
    except Exception as exc:
        log.warning("vtl: flash_attn_3 not importable (%s); FA3 route unavailable", exc)
        return
    try:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    except Exception as exc:
        log.warning("vtl: FlashAttentionImpl not importable (%s); FA3 route unchanged", exc)
        return
    _install(FlashAttentionImpl)
    log.info(
        "vtl: FA3 attention route ARMED (upstream FA3; A/B vs stock -- verify TPOT before shipping)"
    )


def _self_check() -> None:
    from vtl.registry import PATCH_REGISTRY, is_enabled

    patch = next(p for p in PATCH_REGISTRY if p.name == "fa3_attn")
    assert patch.default is False

    os.environ.pop("VTL_ENABLE_FA3_ATTN", None)
    assert _fa3_enabled() is False
    assert is_enabled(patch) is False  # default OFF: not selected -> not installed

    os.environ["VTL_ENABLE_FA3_ATTN"] = "1"
    assert _fa3_enabled() is True
    assert is_enabled(patch) is True  # armed -> selected

    base = dict(
        enabled=True,
        device_major=9,
        has_metadata=True,
        use_cascade=False,
        is_decoder_self_attn=True,
        dcp_world_size=1,
        causal_is_bool=True,
        alibi_is_none=True,
        output_scale_is_none=True,
    )
    assert _should_fire(**base) is True  # clean decoder self-attn on sm90
    assert _should_fire(**{**base, "enabled": False}) is False
    assert _should_fire(**{**base, "device_major": 10}) is False  # Blackwell -> stock
    assert _should_fire(**{**base, "device_major": None}) is False
    assert _should_fire(**{**base, "has_metadata": False}) is False  # profiling run
    assert _should_fire(**{**base, "use_cascade": True}) is False
    assert _should_fire(**{**base, "is_decoder_self_attn": False}) is False  # encoder/cross
    assert _should_fire(**{**base, "dcp_world_size": 2}) is False
    assert _should_fire(**{**base, "causal_is_bool": False}) is False  # dynamic causal
    assert _should_fire(**{**base, "alibi_is_none": False}) is False
    assert _should_fire(**{**base, "output_scale_is_none": False}) is False  # fused quant

    os.environ.pop("VTL_ENABLE_FA3_ATTN", None)
    print("fa3_attention self-check ok")


if __name__ == "__main__":
    _self_check()
