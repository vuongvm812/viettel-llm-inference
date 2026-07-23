#!/usr/bin/env bash
# Regenerate the v0.25.0 tree-spec patches from the local (gitignored) vllm/ checkout.
#
# vllm/ is gitignored INSIDE the parent repo (not its own git repo), so `git diff` can't see it.
# We diff each edited file against a pristine v0.25.0 source instead. Point V025 at any clean
# v0.25.0 tree (e.g. extracted from the vllm/vllm-openai:v0.25.0 image, or a fresh `git checkout
# v0.25.0`). The *.patch files are the committed source of truth; this script is dev convenience.
#
# Usage:  V025=/path/to/pristine/vllm-0.25.0  bash round-1.2/vtl/vllm_patches/gen.sh
set -euo pipefail

: "${V025:?set V025 to a pristine vllm-0.25.0 source root (containing vllm/)}"
REPO=$(cd "$(dirname "$0")/../../.." && pwd)   # round-1.2/..
LOCAL="$REPO/vllm"                              # the edited (gitignored) checkout
OUT="$REPO/round-1.2/vtl/vllm_patches/v0.25.0"

gen() {  # $1 = package-relative path, $2 = patch basename
  diff -u --label "a/vllm/$1" --label "b/vllm/$1" \
    "$V025/vllm/$1" "$LOCAL/vllm/$1" > "$OUT/$2.patch" || true  # diff exits 1 when files differ
  echo "wrote $2.patch ($(wc -l < "$OUT/$2.patch") lines)"
}

gen entrypoints/openai/api_server.py          api_server_rust_frontend
gen v1/sample/rejection_sampler.py            rejection_sampler
gen model_executor/layers/mamba/short_conv.py short_conv
# Paired with short_conv: the in_proj hoist only lets RMSNormQuantFusionPass fire if the
# operator_norm output also stops having a second consumer. Regenerate them together.
gen model_executor/models/lfm2.py                lfm2
gen v1/attention/backends/flash_attn.py       flash_attn
gen v1/worker/gpu_model_runner.py             gpu_model_runner
# V2 model runner (VLLM_USE_V2_MODEL_RUNNER=1): greedy argmax fast path in the V2
# sampler, and the dead-num_accepted-scatter elision in the hybrid model state.
gen v1/worker/gpu/sample/sampler.py             v2_greedy_sampler
gen v1/worker/gpu/model_states/mamba_hybrid.py  mamba_hybrid_postprocess

echo "dry-run applying all patches against pristine v0.25.0..."
TMP=$(mktemp -d); cp -r "$V025/vllm" "$TMP/vllm"
for p in "$OUT"/*.patch; do
  [ -s "$p" ] || { echo "skip empty $p"; continue; }
  patch -p1 -d "$TMP" --dry-run < "$p" >/dev/null && echo "  OK $(basename "$p")" || { echo "  FAIL $(basename "$p")"; exit 1; }
done
rm -rf "$TMP"
echo "all patches apply clean."
