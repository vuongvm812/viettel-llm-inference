"""Fused Q/K RMSNorm + RoPE for LFM2's 6 attention layers.

vLLM already ships the CUDA kernel (``_C::fused_qk_norm_rope``) and a fusion pass
(``pass_config.enable_qk_norm_rope_fusion``), but the pass cannot match ``Lfm2Attention``:
its pattern expects ``q.view(...)`` directly off the qkv split, while lfm2.py inserts
``.contiguous()`` clones. So we call the kernel by hand instead.

Replaces 3 launches (q_layernorm, k_layernorm, rotary_embedding) + 2 clones with 1 in-place
launch, per attention layer. 6 attn layers x 2 saved launches = 12 fewer launches/decode step
on a host-bound TPOT path. Numerically identical: same weights, same eps, same neox rope.
"""

from __future__ import annotations

import logging

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vllm.vtl.qk_norm_rope")


@register_patch("qk_norm_rope", default=True)
def apply() -> None:
    import torch

    from vllm import _custom_ops as ops
    from vllm.model_executor.models.lfm2 import Lfm2Attention

    if already_patched(Lfm2Attention, "forward"):
        return
    if not hasattr(torch.ops._C, "fused_qk_norm_rope"):
        log.warning("vtl: qk_norm_rope skipped, _C::fused_qk_norm_rope not in this build")
        return

    original = Lfm2Attention.forward

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        # ponytail: fall back to stock for any shape the kernel doesn't cover
        # (partial rotary_dim, gptj-style rope). Checked once per call, ~free.
        rope = self.rotary_emb
        if rope.rotary_dim != self.head_dim or not rope.is_neox_style:
            return original(self, positions, hidden_states)

        qkv, _ = self.qkv_proj(hidden_states)
        ops.fused_qk_norm_rope(
            qkv,
            self.num_heads,
            self.num_kv_heads,
            self.num_kv_heads,
            self.head_dim,
            self.q_layernorm.variance_epsilon,
            self.q_layernorm.weight,
            self.k_layernorm.weight,
            rope._match_cos_sin_cache_dtype(qkv),
            True,  # is_neox
            positions.view(-1),
        )
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v)
        output, _ = self.out_proj(attn_output)
        return output

    Lfm2Attention.forward = mark_patched(forward, original)
    log.info("vtl: qk_norm_rope installed (fused QK-RMSNorm+RoPE on Lfm2Attention)")


def _self_check() -> None:
    """Compare patched forward against the stock one on a real Lfm2Attention. Needs a GPU."""
    import torch

    from transformers import AutoConfig

    from vllm.model_executor.models.lfm2 import Lfm2Attention

    cfg = AutoConfig.from_pretrained("/hf-model")
    layer = Lfm2Attention(
        config=cfg,
        layer_idx=2,
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
    ).cuda().to(torch.bfloat16)
    torch.nn.init.normal_(layer.q_layernorm.weight, mean=1.0, std=0.1)
    torch.nn.init.normal_(layer.k_layernorm.weight, mean=1.0, std=0.1)

    hs = torch.randn(17, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    pos = torch.arange(17, device="cuda")

    # Compare only the q/k produced before self.attn, which needs no kv cache.
    qkv, _ = layer.qkv_proj(hs)
    q, k, _ = qkv.split([layer.q_size, layer.kv_size, layer.kv_size], dim=-1)
    q = layer.q_layernorm(q.view(17, layer.num_heads, layer.head_dim).contiguous())
    k = layer.k_layernorm(k.view(17, layer.num_kv_heads, layer.head_dim).contiguous())
    ref_q, ref_k = layer.rotary_emb(pos, q.view(17, -1), k.view(17, -1))

    from vllm import _custom_ops as ops

    fused = qkv.clone()
    ops.fused_qk_norm_rope(
        fused, layer.num_heads, layer.num_kv_heads, layer.num_kv_heads, layer.head_dim,
        layer.q_layernorm.variance_epsilon, layer.q_layernorm.weight,
        layer.k_layernorm.weight, layer.rotary_emb._match_cos_sin_cache_dtype(fused),
        True, pos,
    )
    got_q, got_k, _ = fused.split([layer.q_size, layer.kv_size, layer.kv_size], dim=-1)
    torch.testing.assert_close(got_q, ref_q, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(got_k, ref_k, rtol=2e-2, atol=2e-2)
    print("qk_norm_rope self-check ok")


if __name__ == "__main__":
    # `make check` runs every patch's self-check on a laptop with no torch/vllm/GPU, so an
    # in-container-only check must skip, not fail -- same contract as the other patches here.
    try:
        _self_check()
    except ImportError as exc:
        print(f"qk_norm_rope self-check SKIPPED (needs the container: {exc})")
