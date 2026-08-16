#!/usr/bin/env bash
# Fetch the round-2 model (Qwen/Qwen3.5-122B-A10B-FP8) into repo-root hf-model/.
#
#   scripts/fetch-model.sh              # full download   -- or: make model-fetch
#   scripts/fetch-model.sh --meta-only  # config/tokenizer only -- or: make model-fetch-meta
#
# WHY THIS EXISTS. The full checkpoint is 127.2 GB across 39 safetensors shards, which only
# the GPU host (H200, ~1 TB disk) can hold -- a dev box cannot. But almost everything that is
# not the forward pass (trace building, tokenizer work, config-driven planning, the PGO
# frontend boot) needs only the ~25 MB of JSON/tokenizer files. So one script serves both
# hosts: full mode for the H200, --meta-only for everywhere else. `hf download` is resumable
# and content-addressed, so re-running after an interrupted transfer continues instead of
# restarting, and running --meta-only first then full later costs nothing extra.
#
# The target dir is repo-root hf-model/ (gitignored, populated per-host) because that is what
# every compose overlay bind-mounts at /model and what PGO_HFMODEL defaults to. Override with
# HF_MODEL_DIR=... for a scratch location, MODEL_ID=... for a different checkpoint.
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-122B-A10B-FP8}"
# Resolve the default target relative to this script, not the caller's cwd, so
# `make model-fetch` and a bare `scripts/fetch-model.sh` land in the same place.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_MODEL_DIR="${HF_MODEL_DIR:-$REPO_ROOT/hf-model}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-8}"
# Full checkpoint is 127.2 GB; require headroom for the transfer's in-flight temp files.
REQUIRED_GB=140

META_ONLY=0
case "${1:-}" in
  "") ;;
  --meta-only) META_ONLY=1 ;;
  *) echo "usage: $0 [--meta-only]" >&2; exit 2 ;;
esac

have() { command -v "$1" >/dev/null 2>&1; }

# --- resolve the HF downloader ------------------------------------------------------------
# Prefer the modern `hf` CLI, fall back to the legacy `huggingface-cli` name, and as a last
# resort install the package -- both names ship in huggingface_hub[cli].
resolve_cli() {
  if have hf; then echo "hf"; return; fi
  if have huggingface-cli; then echo "huggingface-cli"; return; fi
  echo ""
}

HF_CLI="$(resolve_cli)"
if [ -z "$HF_CLI" ]; then
  echo "fetch-model: no 'hf' or 'huggingface-cli' found; installing huggingface_hub[cli]..." >&2
  pip install -U 'huggingface_hub[cli]' >&2
  hash -r 2>/dev/null || true
  HF_CLI="$(resolve_cli)"
  if [ -z "$HF_CLI" ]; then
    echo "fetch-model: install succeeded but no CLI on PATH -- check 'pip show huggingface_hub'" >&2
    exit 1
  fi
fi

# --- preflight: full mode needs the disk before it needs the network ----------------------
# `hf download` fails 100+ GB in otherwise, with the partial transfer already on disk. Check
# the TARGET filesystem (the dir may be a mount), portably via POSIX df -Pk.
if [ "$META_ONLY" -eq 0 ]; then
  mkdir -p "$HF_MODEL_DIR"
  avail_kb="$(df -Pk "$HF_MODEL_DIR" | awk 'NR==2 {print $4}')"
  required_kb=$((REQUIRED_GB * 1024 * 1024))
  if [ -z "$avail_kb" ] || [ "$avail_kb" -lt "$required_kb" ]; then
    avail_gb=$((${avail_kb:-0} / 1024 / 1024))
    cat >&2 <<EOF
fetch-model: NOT ENOUGH DISK for the full checkpoint.
  model:     $MODEL_ID -- 127.2 GB in 39 safetensors shards
  needed:    ${REQUIRED_GB} GB free on the filesystem holding $HF_MODEL_DIR
  available: ${avail_gb} GB
Run the full download on the GPU host (make model-fetch), or fetch just the ~25 MB of
config/tokenizer files here with: $0 --meta-only   (make model-fetch-meta)
EOF
    exit 1
  fi
fi

# --- download -----------------------------------------------------------------------------
set -- download "$MODEL_ID" --local-dir "$HF_MODEL_DIR" --max-workers "$HF_MAX_WORKERS"
if [ "$META_ONLY" -eq 1 ]; then
  # Everything the tokenizer/config consumers need, nothing weight-sized. Covers config.json,
  # generation_config.json, tokenizer.json/tokenizer_config.json, and any vocab/merges files.
  # One --include PER pattern: the hf>=1.x CLI parses a second bare pattern as a positional
  # FILENAME and then silently ignores --include altogether ("Ignoring `--include` since
  # filenames have been explicitly set"), which drops config.json from the fetch.
  set -- "$@" --include '*.json' --include '*.txt' --include 'tokenizer*' \
              --include 'vocab*' --include 'merges*'
  echo "fetch-model: META-ONLY fetch of $MODEL_ID -> $HF_MODEL_DIR"
else
  echo "fetch-model: FULL fetch of $MODEL_ID (127.2 GB, resumable) -> $HF_MODEL_DIR"
fi

"$HF_CLI" "$@"

echo "fetch-model: done. Contents of $HF_MODEL_DIR:"
ls -lh "$HF_MODEL_DIR"
