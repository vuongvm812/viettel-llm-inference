#!/usr/bin/env python3
"""Generate a MOCK EAGLE-3 draft checkpoint (model.safetensors) for LFM2.5-1.2B.

Purpose: exercise the W1 surgery + W4 integration end-to-end BEFORE a real head exists.
It writes random weights with the exact tensor names/shapes/dtype vLLM's Eagle3 loader
(`vllm/model_executor/models/llama_eagle3.py`) expects, so the engine LOADS and RUNS the
draft — acceptance will be ~0 (random weights), which is the point: it proves the plumbing,
not the quality. Replace with the SpecForge `specforge export` output for the real head.

The tensor spec is derived from load_weights() in llama_eagle3.py:
  - every non-lm_head/d2t/t2d name is prefixed `model.`, then `midlayer.`->`layers.0.`;
    q/k/v and gate/up are stacked by vLLM, so the checkpoint carries them SEPARATE.
  - layer 0 concatenates [embeds, hidden] -> qkv input is 2*hidden.
  - fc: [hidden, target_hidden * num_aux(3)] = [2048, 6144].
  - lm_head + d2t use the reduced draft_vocab; t2d is skipped on load.

    python make_mock_checkpoint.py            # writes ./model.safetensors next to config.json
"""
import argparse
import json
import os

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))


def build(cfg: dict) -> dict[str, torch.Tensor]:
    h = cfg["hidden_size"]                       # 2048
    inter = cfg["intermediate_size"]             # 12288
    n_heads = cfg["num_attention_heads"]         # 32
    n_kv = cfg["num_key_value_heads"]            # 8
    hd = cfg["head_dim"]                         # 64
    vocab = cfg["vocab_size"]                    # 65536
    dvocab = cfg["draft_vocab_size"]             # 32000
    t_hidden = cfg["target_hidden_size"]         # 2048
    n_aux = len(cfg["eagle_config"]["eagle_aux_hidden_state_layer_ids"])  # 3
    q_out, kv_out = n_heads * hd, n_kv * hd      # 2048, 512
    qkv_in = 2 * h                               # layer 0: [embeds, hidden] -> 4096
    dt = torch.bfloat16

    def r(*shape):
        return (torch.randn(*shape) * 0.02).to(dt)

    w = {
        # --- the single eagle3 decoder layer (HF name `midlayer.` -> vLLM `layers.0.`) ---
        "midlayer.self_attn.q_proj.weight": r(q_out, qkv_in),
        "midlayer.self_attn.k_proj.weight": r(kv_out, qkv_in),
        "midlayer.self_attn.v_proj.weight": r(kv_out, qkv_in),
        "midlayer.self_attn.o_proj.weight": r(h, q_out),
        "midlayer.mlp.gate_proj.weight": r(inter, h),
        "midlayer.mlp.up_proj.weight": r(inter, h),
        "midlayer.mlp.down_proj.weight": r(h, inter),
        "midlayer.input_layernorm.weight": r(h),
        "midlayer.hidden_norm.weight": r(h),            # eagle3-specific
        "midlayer.post_attention_layernorm.weight": r(h),
        # --- model-level ---
        "fc.weight": r(h, t_hidden * n_aux),            # [2048, 6144]
        "norm.weight": r(h),
        "embed_tokens.weight": r(vocab, h),
        # --- head + vocab mapping (reduced draft vocab) ---
        "lm_head.weight": r(dvocab, h),
        # d2t: draft id -> (target id - draft id) offset. Mock = identity (first dvocab tokens).
        "d2t": torch.zeros(dvocab, dtype=torch.long),
        "t2d": torch.zeros(vocab, dtype=torch.bool),    # skipped on load; present for completeness
    }
    w["t2d"][:dvocab] = True
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "model.safetensors"))
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    weights = build(cfg)
    save_file(weights, args.out, metadata={"format": "pt", "mock": "true"})
    total = sum(t.numel() for t in weights.values())
    print(f"wrote {args.out}  ({len(weights)} tensors, {total/1e6:.1f}M params)")
    for k, v in weights.items():
        print(f"  {k:45s} {tuple(v.shape)} {v.dtype}")


if __name__ == "__main__":
    main()
