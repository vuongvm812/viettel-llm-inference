#!/usr/bin/env bash
#
# P2 exit criterion (docs/ROADMAP.md §P2, docs/design/inference-backend/design.md):
# one request through the real runtime must byte-match a direct `llama-cli` greedy
# run on the same GGUF. Greedy (temp=0) is deterministic, so the two should agree
# token-for-token.
#
# TARGET-ONLY: needs `--features llama` (cmake + GPU toolchain), the BF16 GGUF, and
# the `llama-cli` binary. It cannot run in a CPU-only/no-model sandbox — that's why
# this lives as a script rather than a `cargo test`. Run it on the deploy/dev-GPU box.
#
# Usage:
#   GGUF=models/qwen3.5-2b-bf16.gguf LLAMA_CLI=/path/to/llama-cli ./script/verify-determinism.sh
#
# Env:
#   GGUF        path to the model (default: models/qwen3.5-2b-bf16.gguf, from the config)
#   LLAMA_CLI   path to llama-cli (default: `llama-cli` on PATH)
#   PROMPT      fixed prompt to compare (default: a short deterministic one)
#   N           tokens to generate (default: 64)
#   PORT        runtime port (default: 8099, avoids the configured 8001/vLLM 8000)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GGUF="${GGUF:-models/qwen3.5-2b-bf16.gguf}"
LLAMA_CLI="${LLAMA_CLI:-llama-cli}"
PROMPT="${PROMPT:-The capital of France is}"
N="${N:-64}"
PORT="${PORT:-8099}"

log()  { printf '\033[1;34m[verify]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*"; exit 1; }

[[ -f "$GGUF" ]] || fail "GGUF not found at $GGUF (run ./script/setup.sh, or set GGUF=...)"
command -v "$LLAMA_CLI" >/dev/null || fail "llama-cli not found (set LLAMA_CLI=/path/to/llama-cli)"

# 1. Reference: direct llama-cli greedy decode (temp 0, seed 42, no chat template).
#    --no-display-prompt so stdout is only the generated continuation.
log "llama-cli reference run"
ref="$($LLAMA_CLI -m "$GGUF" -p "$PROMPT" --temp 0 --seed 42 -n "$N" \
        --no-display-prompt -no-cnv 2>/dev/null)"

# 2. Our runtime: build with the real backend, start it, send one request.
log "building inference-runtime --features llama"
cargo build --manifest-path services/Cargo.toml -p inference-runtime --features llama --release

tmp_cfg="$(mktemp)"
cat > "$tmp_cfg" <<YAML
server: { host: "127.0.0.1", port: ${PORT} }
model: { gguf_path: "${GGUF}", n_ctx: 8192, n_threads: 1, n_gpu_layers: -1 }
runtime:
  max_inflight: 8
  ring_size: 64
  cores: { web_io: 0, text: 1, fast_loop: 2 }
YAML

log "starting runtime on :${PORT}"
./services/target/release/inference-runtime "$tmp_cfg" &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null; rm -f "$tmp_cfg"' EXIT

# Wait for readiness.
for _ in $(seq 1 60); do
  curl -fs "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
  sleep 1
done

# Send the fixed request; extract concatenated SSE delta.content (skip [DONE]).
log "requesting one completion"
body="$(printf '{"model":"qwen","messages":[{"role":"user","content":%s}],"max_tokens":%d}' \
        "$(printf '%s' "$PROMPT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" "$N")"
ours="$(curl -fsN "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'content-type: application/json' -d "$body" \
      | sed -n 's/^data: //p' | grep -v '^\[DONE\]$' \
      | python3 -c 'import json,sys; print("".join(json.loads(l)["choices"][0]["delta"].get("content","") for l in sys.stdin if l.strip()), end="")')"

# 3. Byte-compare.
if [[ "$ours" == "$ref" ]]; then
  log "PASS — runtime output byte-matches llama-cli ($(printf '%s' "$ours" | wc -c) bytes)"
else
  printf '\033[1;33m--- llama-cli ---\033[0m\n%s\n\033[1;33m--- runtime ---\033[0m\n%s\n' "$ref" "$ours"
  fail "outputs differ (see diff above). Check tokenizer/BOS/chat-template parity and sampler."
fi
