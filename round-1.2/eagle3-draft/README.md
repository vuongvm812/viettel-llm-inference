# EAGLE-3 draft checkpoint for LFM2.5-1.2B (serving side — W3 + W4)

The vLLM-loadable draft head that pairs with the `SupportsEagle3` target (W1, `lfm2.patch`).
`config.json` is the real, retargeted serving config; `model.safetensors` is produced either by
`make_mock_checkpoint.py` (structural testing now) or `specforge export` (the real head, from W2).

## config.json — retargeted from the 8B reference to our target (`hf-model/config.json`)
`architectures: ["Eagle3LlamaForCausalLM"]` (vLLM's serving class), `hidden_size 2048`,
`intermediate_size 12288`, `num_attention_heads 32`, `num_key_value_heads 8`, `head_dim 64`,
`vocab_size 65536`, `draft_vocab_size 32000`, `target_hidden_size 2048`, `rope_theta 1e6`,
`rms_norm_eps 1e-5`, `max_position_embeddings 128000`. **`eagle_config.eagle_aux_hidden_state_layer_ids:
[2,8,13]`** pins the serve-time aux layers to what vLLM extracts (`get_eagle3_default_aux_hidden_state_layers`,
N=16 → (2,8,13)) — and MUST equal what SpecForge captured in training (the #1 acceptance failure mode).

## The mock (test the wiring before a real head exists)
```bash
python make_mock_checkpoint.py          # -> model.safetensors (~581MB, gitignored), random weights
```
Tensor names/shapes are derived from `load_weights()` in `vllm/model_executor/models/llama_eagle3.py`:
the single decoder layer as `midlayer.*` (vLLM maps `midlayer.`→`layers.0.`; q/k/v and gate/up kept
SEPARATE — vLLM stacks them), layer-0 qkv input `2*hidden=4096` (it concatenates `[embeds, hidden]`),
`fc [2048, 6144]` (target_hidden × 3 aux), reduced `lm_head [32000, 2048]` + `d2t`; `t2d` is skipped on
load. Random weights → **acceptance ≈ 0**; the mock only proves vLLM loads the draft and the W1 aux
extraction + verify loop run. It is NOT a quality check.

## W3 — packaging the real head
`specforge export` (see `../eagle3-training/README.md` step 4) emits this same layout. Verify the exported
`config.json` matches the fields above — especially `architectures`, `draft_vocab_size`, and the
`(2,8,13)` aux ids — then drop its `model.safetensors` here (or point `--speculative-config.model` at the
export dir directly).

## W4 — integrate (on the H200 box)
Stage this dir into the served image (or mount it), then in `round-1.2/docker-compose.yaml` swap the
spec-decode line (an eagle3 variant is pre-staged there, commented):
```
- '--speculative-config={"method":"eagle3","model":"/draft","num_speculative_tokens":2}'
```
- EAGLE-3 is **V2-compatible** (unlike ngram — `vllm/config/vllm.py:2072`), so you may keep
  `VLLM_USE_V2_MODEL_RUNNER=1` and A/B V1 vs V2 rather than being forced to V1.
- **Greedy-lossless gate:** spec ON vs OFF at temp 0 must be token-identical over a long generation
  (the short-conv rollback fix + chain verify guarantee it). With the MOCK, expect ~0 acceptance but still
  lossless output — that is the pass criterion for wiring.
- **Go/no-go on the MIG 1g.18gb slice** (`bench/replay.py`): only a REAL head can win TPOT, and only if
  accepted-length × host-per-step-savings beats the added GPU verify cost. Sweep `num_speculative_tokens`.

## W5 — H200 optimization (after go/no-go says yes)
Profile-driven; see the plan. Prime custom-CUDA target is `combine_hidden_states` (fc_norm×3 + concat +
fc, every draft step) reusing vtl's `rms_norm_quant.cu`/`mul_quant.cu`; plus widening the draft loop from
PIECEWISE to FULL cudagraph capture. Do NOT build these until the real head clears the go/no-go.
