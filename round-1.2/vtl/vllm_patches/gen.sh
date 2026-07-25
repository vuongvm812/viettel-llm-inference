#!/usr/bin/env bash
# Regenerate the vLLM source patches from the local (gitignored) vllm/ checkout.
#
# vllm/ is a real git clone pinned to the $VER tag, so `git diff HEAD` IS the patch set --
# no pristine second tree needed. Edit vllm/ directly, then run this. The *.patch files are
# the committed source of truth; this script is dev convenience.
#
# Note `git apply -3` STAGES what it merges, so we diff against HEAD, not the index.
#
# Upgrading to a new vLLM tag:
#   git -C vllm fetch --depth 1 origin tag vX.Y.Z && git -C vllm checkout vX.Y.Z
#   for p in round-1.2/vtl/vllm_patches/<old>/*.patch; do git -C vllm apply -3 "$p"; done
#   # resolve any conflict markers, then:
#   VER=vX.Y.Z bash round-1.2/vtl/vllm_patches/gen.sh
#
# Usage:  bash round-1.2/vtl/vllm_patches/gen.sh
set -euo pipefail

VER="${VER:-v0.26.0}"
REPO=$(cd "$(dirname "$0")/../../.." && pwd)   # round-1.2/..
LOCAL="$REPO/vllm"                             # the edited (gitignored) checkout
OUT="$REPO/round-1.2/vtl/vllm_patches/$VER"

[ -d "$LOCAL/.git" ] || { echo "$LOCAL is not a git checkout -- see the upgrade recipe above"; exit 1; }
mkdir -p "$OUT"

CAPTURED=()
gen() {  # $1 = package-relative path, $2 = patch basename
  git -C "$LOCAL" diff HEAD -- "vllm/$1" > "$OUT/$2.patch"
  CAPTURED+=("vllm/$1")
  echo "wrote $2.patch ($(wc -l < "$OUT/$2.patch") lines)"
}

gen entrypoints/openai/api_server.py            api_server_rust_frontend
gen model_executor/layers/mamba/short_conv.py   short_conv
# Paired with short_conv: the in_proj hoist only lets RMSNormQuantFusionPass fire if the
# operator_norm output also stops having a second consumer. Regenerate them together.
gen model_executor/models/lfm2.py               lfm2
# V2 model runner (VLLM_USE_V2_MODEL_RUNNER=1): greedy argmax fast path in the V2
# sampler, and the dead-num_accepted-scatter elision in the hybrid model state.
gen v1/worker/gpu/sample/sampler.py             v2_greedy_sampler
gen v1/worker/gpu/model_states/mamba_hybrid.py  mamba_hybrid_postprocess

# THE guard. A hardcoded gen() list silently drops any edit to a file not named above -- that is
# how three files' worth of spec-decode work sat unshipped in the checkout while notes claimed it
# had landed (see not-applied/README.md). The checkout knows what changed; ask it.
MODIFIED=$(git -C "$LOCAL" diff HEAD --name-only -- vllm/ | sort)
EXPECTED=$(printf '%s\n' "${CAPTURED[@]}" | sort)
if [ "$MODIFIED" != "$EXPECTED" ]; then
  echo "ERROR: modified files in $LOCAL do not match what gen.sh captures."
  comm -23 <(echo "$MODIFIED") <(echo "$EXPECTED") | sed 's/^/  NOT CAPTURED (add a gen line, or park under not-applied\/): /'
  comm -13 <(echo "$MODIFIED") <(echo "$EXPECTED") | sed 's/^/  captured but unmodified (stale gen line): /'
  exit 1
fi

echo "dry-run applying all patches against pristine $VER..."
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
git -C "$LOCAL" archive "$VER" | tar -x -C "$TMP"
for p in "$OUT"/*.patch; do
  # Empty is a FAILURE, not a skip: `git diff HEAD` goes empty the moment someone commits in the
  # checkout, which would blank every patch here and still print a green result.
  [ -s "$p" ] || { echo "  EMPTY $(basename "$p") -- nothing captured (did you commit in $LOCAL?)"; exit 1; }
  patch -p1 -d "$TMP" --dry-run -s < "$p" && echo "  OK $(basename "$p")" \
    || { echo "  FAIL $(basename "$p")"; exit 1; }
done

# Parked work is only worth keeping if it still applies. Stack it on the APPLIED tree so the
# claim in not-applied/README.md is re-checked here instead of rotting.
for p in "$OUT"/*.patch; do patch -p1 -d "$TMP" -s < "$p"; done
for p in "$(dirname "$OUT")"/not-applied/*.patch; do
  [ -e "$p" ] || continue
  patch -p1 -d "$TMP" --dry-run -s < "$p" && echo "  OK (parked) $(basename "$p")" \
    || { echo "  FAIL (parked) $(basename "$p") -- forward-port it or drop it"; exit 1; }
done
echo "all patches apply clean."
