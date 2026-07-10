#!/usr/bin/env python3
"""Tier 2 — offline weight quantization via llm-compressor (compressed-tensors).

One tool, three schemes. Output is loaded natively by vLLM (auto-detected).
Idempotent: skips if OUT already holds a quantized checkpoint.

Env:
  SRC          source HF checkpoint (default /model)
  OUT          output dir           (default /qmodel)
  QUANT_SCHEME fp8 | gptq | awq     (default fp8)

fp8  = W8A8 FP8_DYNAMIC, no calibration data, ~lossless, best latency.
gptq/awq = INT4 W4A16, needs a small calibration set (auto-downloaded).
"""
import os
import sys

SRC = os.environ.get("SRC", "/model")
OUT = os.environ.get("OUT", "/qmodel")
SCHEME = os.environ.get("QUANT_SCHEME", "fp8").lower()

# idempotent: don't re-quantize on container restart
if os.path.exists(os.path.join(OUT, "config.json")):
    print(f"[quantize] {OUT} already populated — skipping (scheme={SCHEME})")
    sys.exit(0)

from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier, GPTQModifier
from llmcompressor.modifiers.awq import AWQModifier

print(f"[quantize] loading {SRC}  scheme={SCHEME}")
model = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype="auto", device_map="auto")
tok = AutoTokenizer.from_pretrained(SRC)

if SCHEME == "fp8":
    recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
    oneshot(model=model, recipe=recipe)
elif SCHEME in ("gptq", "awq"):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:512]")
    ds = ds.map(lambda ex: {"text": tok.apply_chat_template(ex["messages"], tokenize=False)})
    Mod = GPTQModifier if SCHEME == "gptq" else AWQModifier
    recipe = Mod(targets="Linear", scheme="W4A16", ignore=["lm_head"])
    oneshot(model=model, recipe=recipe, dataset=ds,
            max_seq_length=2048, num_calibration_samples=512)
else:
    sys.exit(f"[quantize] unknown QUANT_SCHEME={SCHEME!r} (want fp8|gptq|awq)")

model.save_pretrained(OUT, save_compressed=True)
tok.save_pretrained(OUT)
print(f"[quantize] saved -> {OUT}")
