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

## Prerequisites — CUDA Linux GPU box (NOT your Mac)
SpecForge pins `torch==2.11.0` + `sglang==0.5.14` (CUDA-Linux only) and needs Python 3.11. None of the
steps below — SGLang capture server, `torchrun`, `specforge train` — run on macOS/CPU. Your Mac is only
for editing/committing these config files.

Also required: the **CUDA toolkit (`nvcc` + headers)**, not just the driver. SGLang's FlashInfer backend
JIT-compiles its attention kernels at launch and fails with `Could not find nvcc / cuda_home
'/usr/local/cuda' doesn't exist` if only the driver is present. Install a toolkit inside a **version window** — `12.8 ≤ toolkit ≤ (the CUDA version nvidia-smi reports)`:
- **too new** (e.g. 13 when the driver only supports 12.x): CUDA error `804 forward compatibility was
  attempted on non supported HW` — torch can't see the GPU at all (Ampere/consumer cards don't do
  forward-compat). The `cuda-compat-*` package pulled in alongside a too-new toolkit is the trigger.

So `cuda-toolkit-12-9` (or 12-8) is the safe pick on a recent-glibc box whose driver supports ≥12.8;
purge any `cuda-compat-*`, export `CUDA_HOME=/usr/local/cuda-12.9` + `PATH=$CUDA_HOME/bin:$PATH`, clear
`~/.cache/flashinfer`, and confirm `python -c "import torch;print(torch.cuda.is_available())"` prints
True before launching SGLang. (If the driver's max CUDA < 12.8, upgrade the driver first, ≈570+.)
### glibc ≥ 2.41: `rsqrt/rsqrtf ... exception specification does not match`
Launch dies during CUDA-graph capture, in the FlashInfer JIT, with a `Capture cuda graph failed: Ninja
build failed` wrapping that error. It is **not** a memory problem and **not** fixed by a newer toolkit —
verified still broken on CUDA 13.1 + glibc 2.43. It is a bug in NVIDIA's header:
- glibc declares C23 `rsqrt`/`rsqrtf` (as `noexcept`) under `__GLIBC_USE (IEC_60559_FUNCS_EXT_C23)`,
  which `bits/libc-header-start.h` turns on for **`__USE_GNU`** — and libstdc++ always sets `_GNU_SOURCE`;
- `crt/math_functions.h:123` tests only `__GLIBC_USE_ISOC23 || __STDC_WANT_IEC_60559_FUNCS_EXT__`, misses
  `__USE_GNU`, so it thinks glibc has no `rsqrt` and declares its own at lines 629/653 — the only two
  math decls in that file lacking `__THROW`.

Fix = give those two lines the `__THROW` every neighbour already has (do **not** instead flip NVIDIA's
`__NV_GLIBC_PROVIDES_IEC_60559_FUNCS` to 1 — that also deletes the device-side `sinpi`/`cospi` bodies in
`math_functions.hpp`). Redo after any CUDA reinstall:
```bash
H=/usr/local/cuda-13/include/crt/math_functions.h
sudo cp -n $H $H.orig
sudo sed -i -E "629s/(double[[:space:]]+rsqrt\(double x\));/\1 __THROW;/;
                653s/(float[[:space:]]+rsqrtf\(float x\));/\1 __THROW;/" $H
sed -n "629p;653p" $H       # both lines must now end in __THROW;
rm -rf ~/.cache/flashinfer  # the failed build is cached
```
Non-fixes, for the record: `--attention-backend triton` (SGLang asserts `Lfm2ForCausalLM does not support
triton attention backend, as the first layer might not be an attention layer` — LFM2 layer 0 is a
short-conv); `--cuda-graph-backend-decode=disabled` (the JIT just moves to the first real decode);
swapping the `-ccbin clang` host compiler for g++ (same redeclaration, same error).

After any driver up/downgrade, **reboot** — otherwise `nvidia-smi` fails with `NVML: Driver/library
version mismatch` (new userspace libs vs the still-loaded old kernel module). A recent driver (580+)
supports CUDA 13 and lifts the ceiling so torch's bundled runtime + a 12.9/13 toolkit both fit.

On the GPU box:
```bash
git clone https://github.com/sgl-project/SpecForge.git && cd SpecForge
uv python install 3.11 && uv venv -p 3.11 --python-preference only-managed && source .venv/bin/activate
uv pip install -v . --prerelease=allow      # pulls torch 2.11.0 + sglang 0.5.14 (no separate sglang install)
# NOTE: use uv's managed Python. A pyenv-built 3.11 often lacks _bz2/_lzma/_sqlite3 (skipped at
# build time if the system dev libs were missing) -> `datasets` import fails with No module named
# '_bz2'. Fix = the managed interpreter above, or `apt-get install libbz2-dev liblzma-dev libsqlite3-dev`
# then rebuild the pyenv Python.
# then copy this dir's files in:
cp <repo>/round-1.2/eagle3-training/lfm2.5-1.2b-instruct-eagle3.json        configs/
cp <repo>/round-1.2/eagle3-training/lfm2.5-1.2b-instruct-eagle3-online.yaml examples/configs/
```

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
Best practice (do before a production head): **self-distill** — regenerate the responses with the target
so the draft learns what THIS model actually says. Serve the target under SGLang, then regenerate:
```bash
# a) serve the target for capture (bf16 is the simple path; SGLang captures via its own runtime, so an
#    exact W4A8 match isn't achievable here anyway — a small train(bf16)/serve(W4A8) gap remains).
python3 -m sglang.launch_server \
  --model LiquidAI/LFM2.5-1.2B-Instruct \
  --tp 1 --dtype bfloat16 --trust-remote-code \
  --host 0.0.0.0 --port 30000

# b) regenerate ShareGPT answers with that target. temperature 0 matches the greedy deployment
#    (the judge runs temp 0); raise it only if the head generalizes poorly.
python scripts/regenerate_train_data.py \
  --model LiquidAI/LFM2.5-1.2B-Instruct \
  --server-address localhost:30000 \
  --input-file-path ./cache/dataset/sharegpt_train.jsonl \
  --output-file-path ./cache/dataset/sharegpt_lfm2.5-1.2b_regen.jsonl \
  --temperature 0.0 \
  --max-tokens 4096 \
  --concurrency 64 \
  --reasoning none \
  --resume
```
Then set `data.train_data_path: ./cache/dataset/sharegpt_lfm2.5-1.2b_regen.jsonl` in the yaml. Add a few-k multilingual-chat slice (LFM2.5
covers en/ar/zh/fr/de/ja/ko/es/pt) and, if possible, a few hundred samples shaped like the judge trace
(multi-turn, ~4k context). Keep it small first (~15–20k conversations); scale only if acceptance is low.

**2. Shared vocab mapping** (required for online/disaggregated — producer & consumer must agree). The
eagle3 capture script derives the reduced 32k draft vocab (`d2t`/`t2d`) from the corpus and writes
`<output-path>/vocab_mapping/vocab_mapping.pt`:
```bash
torchrun --nproc_per_node=1 \
  scripts/prepare_hidden_states.py \
  --target-model-path LiquidAI/LFM2.5-1.2B-Instruct \
  --data-path ./cache/dataset/sharegpt_lfm2.5-1.2b_regen.jsonl \
  --output-path ./cache/hidden_states/lfm2.5-1.2b-eagle3 \
  --chat-template lfm \
  --max-length 4096 \
  --tp-size 1 \
  --batch-size 32 \
  --strategy eagle3 \
  --draft-model-config configs/lfm2.5-1.2b-instruct-eagle3.json
# then make it the path the yaml expects:
mkdir -p cache/vocab_mapping
cp ./cache/hidden_states/lfm2.5-1.2b-eagle3/vocab_mapping/vocab_mapping.pt \
   cache/vocab_mapping/lfm2.5-1.2b-eagle3.pt
```
(Offline/local eagle3 auto-derives this when `vocab_mapping_path` is empty; online/disaggregated does
not. This same run also produces the offline hidden-state features under `--output-path` — reuse them if
you switch to an offline recipe.)

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
