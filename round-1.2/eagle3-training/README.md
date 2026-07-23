# Training an EAGLE-3 draft head for LFM2.5-1.2B-Instruct (SpecForge)

Inputs for [SpecForge](https://github.com/sgl-project/SpecForge) to train the drafter that
Workstream-1 enabled (`Lfm2ForCausalLM` is now `SupportsEagle3`; see `round-1.2/vtl/vllm_patches/
v0.25.0/lfm2.patch`). Two files here are LFM2-eagle3-specific and don't ship with SpecForge:

- `lfm2.5-1.2b-instruct-eagle3.json` → copy into SpecForge `configs/`
- `lfm2.5-1.2b-instruct-eagle3-online.yaml` → copy into SpecForge `examples/configs/`

SpecForge **already supports LFM2.5-1.2B as a target** (it ships `lfm2.5-1.2b-instruct-dflash-online.yaml`
and the `"lfm"` chat template at `specforge/data/template.py:123`), so no target adapter to write —
only the eagle3 draft config + recipe above, combining that dflash target block with the eagle3 strategy
from `llama3.1-8b-eagle3-online.yaml`.

## Draft config — how the dims were chosen (all from `hf-model/config.json`)
`hidden_size 2048`, `num_attention_heads 32`, `num_key_value_heads 8`, `head_dim 64`,
`intermediate_size 12288`, `vocab_size 65536`, `rope_theta 1e6`, `rms_norm_eps 1e-5`,
`max_position_embeddings 128000`, `bos/eos/pad = 1/7/0`. `num_hidden_layers: 1` (eagle3 is one layer),
`draft_vocab_size: 32000` (reduced; the vocab mapping picks the top-32k by corpus frequency).

## ⚠️ The one thing that will silently ruin acceptance: aux-layer consistency
vLLM extracts aux hidden states at **layers (2, 8, 13)** for our 16-layer target
(`get_eagle3_default_aux_hidden_state_layers`; N→(2, N//2, N-3)). **SpecForge must train the draft on the
SAME 3 layers**, or the draft's `fc` input is fed a different residual stream at serve time → garbage
acceptance despite a "successful" run. The draft JSON pins `eagle_aux_hidden_state_layer_ids: [2,8,13]`;
before a full run, confirm SpecForge captures exactly those (a short smoke run + its logs), and that the
**exported** vLLM config declares the same ids. This is the #1 integration failure mode — check it first.

## Workflow (run from the SpecForge repo root, on the GPU box)

**1. Data.** Start with ShareGPT for a first pass:
```bash
python ./scripts/prepare_data.py --dataset sharegpt      # -> ./cache/dataset/sharegpt_train.jsonl
```
Best practice (do before a production head): **self-distill** — regenerate responses with the *served
target* so the draft learns what THIS model says, ideally the W4A8 build you actually serve:
```bash
python ./scripts/regenerate_train_data.py ...            # see script --help; produces *_regen.jsonl
```
Then point `data.train_data_path` at the regenerated file. Add a few-k multilingual-chat slice (LFM2.5
covers en/ar/zh/fr/de/ja/ko/es/pt) and, if possible, a few hundred samples shaped like the judge trace
(multi-turn, ~4k context). Keep it small first (~15–20k conversations); scale only if acceptance is low.

**2. Shared vocab mapping** (required for online/disaggregated — producer & consumer must agree):
```bash
python ./scripts/prepare_hidden_states.py ...            # writes <out>/vocab_mapping/vocab_mapping.pt
```
Copy/point it to the `model.vocab_mapping_path` in the yaml (`cache/vocab_mapping/lfm2.5-1.2b-eagle3.pt`).
(Offline/local eagle3 auto-derives this when the path is empty; online does not.)

**3. Train** (online = live SGLang capture of the target's hidden states; disaggregated producer/consumer):
```bash
# start an SGLang capture server for the target on :30000 first (see SpecForge disaggregated docs),
# point it at the W4A8 build if you can; then:
specforge train --config examples/configs/lfm2.5-1.2b-instruct-eagle3-online.yaml
# smoke test: append  training.max_steps=200
```

**4. Export to a vLLM-loadable checkpoint** (Workstream 3):
```bash
specforge export --to sglang \
  --checkpoint ./outputs/lfm2.5-1.2b-instruct-eagle3-online/lfm2.5-1.2b-instruct-eagle3-online-latest \
  --draft-config configs/lfm2.5-1.2b-instruct-eagle3.json \
  --output-dir ./exports/lfm2.5-1.2b-eagle3
```
The runtime checkpoint holds training state and is NOT directly servable — you must export. The export's
`config.json` should end up `architectures: ["Eagle3LlamaForCausalLM"]` for vLLM (Workstream 3 verifies
the final fields: `target_hidden_size 2048`, `draft_vocab_size 32000`, `eagle_config.use_aux_hidden_state`,
and the `(2,8,13)` ids). vLLM's loader maps SpecForge's `midlayer.*`/`d2t`/`t2d` names automatically.

## Then (Workstream 4, on the H200 box)
```
--speculative-config={"method":"eagle3","model":"<exported-dir>","num_speculative_tokens":2}
```
Gate on greedy-lossless (spec ON==OFF, temp 0) and the MIG-slice go/no-go before shipping.
