#!/usr/bin/env bash
# PGO (profile-guided optimization) build of the vllm-rs frontend binary.
#
# Runs inside the `rust-builder` stage of Dockerfile.vllm-fork when
# VLLM_RS_PGO=1. There is no GPU here: the frontend serves entirely on CPU by
# pairing `vllm-rs serve --data-parallel-size-local 0` (frontend-only, owns the
# ZMQ handshake) with `vllm-mock-engine` (fakes prefill, emits random decode
# tokens). The real CPU tokenizer + full HTTP→simd_json-parse→stream path are
# exercised by replaying the recorded round-2 trace. That training run steers the
# final `-Cprofile-use` build.
#
# Fails loudly (set -e) rather than silently shipping a non-PGO binary — matches
# the repo's "fail the build, not the judge's run" rule.
set -euo pipefail

# Tokenizer the training run boots the frontend with. Defaults to the local
# hf-model bind-mounted at /model (real token distributions, no network); the mock
# fakes the forward pass so only the tokenizer/config are read, never the weights.
# Override PGO_MODEL with a HF repo id to fetch a stand-in over the network.
export PGO_MODEL="${PGO_MODEL:-/model}"
MODEL="$PGO_MODEL"
RUST_DIR=/src/rust
PGO_RAW=/pgo
PGO_DATA=/pgo.profdata
HANDSHAKE_PORT=29550
HTTP_PORT=8000
FEAT="--features native-tls-vendored"
CPU="-Ctarget-cpu=x86-64-v3"

cd "$RUST_DIR"
PROFDATA="$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/host: //p')/bin/llvm-profdata"

echo "== [pgo 1/6] build the (uninstrumented) mock engine"
cargo build --release -p vllm-mock-engine

echo "== [pgo 2/6] instrumented build of vllm-rs"
rm -rf "$PGO_RAW"
RUSTFLAGS="-Cprofile-generate=$PGO_RAW $CPU" \
    cargo build --release --bin vllm-rs $FEAT

echo "== [pgo 3/6] prepare training trace (inject max_tokens, pin model)"
python3 - <<'PY'
import json
src = "/src/trace-round2.jsonl"
dst = "/tmp/pgo-train.jsonl"
model = __import__("os").environ.get("PGO_MODEL", "/model")
n = 0
with open(src) as f, open(dst, "w") as out:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        body = rec.setdefault("body", {})
        body.setdefault("max_tokens", 64)   # the mock stops by length
        body["model"] = model               # match the served tokenizer
        out.write(json.dumps(rec) + "\n")
        n += 1
print(f"prepared {n} training requests -> {dst}")
PY

echo "== [pgo 4/6] boot frontend (CPU-only) + mock engine"
BIN="$RUST_DIR/target/release"
"$BIN/vllm-rs" serve "$MODEL" \
    --data-parallel-size 1 --data-parallel-size-local 0 \
    --handshake-port "$HANDSHAKE_PORT" --host 127.0.0.1 --port "$HTTP_PORT" &
FRONTEND_PID=$!
"$BIN/vllm-mock-engine" --handshake-address "tcp://127.0.0.1:${HANDSHAKE_PORT}" &
MOCK_PID=$!

# Wait for the frontend to report healthy (tokenizer download + engine register).
for i in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${HTTP_PORT}/health" >/dev/null 2>&1; then
        echo "frontend healthy after ${i}s"; break
    fi
    if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo "ERROR: frontend exited before becoming healthy"; exit 1
    fi
    sleep 1
done
curl -fsS "http://127.0.0.1:${HTTP_PORT}/health" >/dev/null

echo "== [pgo 5/6] replay training corpus"
pip install --no-cache-dir -q aiohttp
( cd /src/bench && python3 replay.py \
    --target "http://127.0.0.1:${HTTP_PORT}" \
    --trace /tmp/pgo-train.jsonl \
    --closed-loop 16 \
    --out /tmp/pgo-replay.json )

# Flush counters: SIGINT lets the instrumented frontend write its .profraw on exit.
kill -INT "$FRONTEND_PID" 2>/dev/null || true
wait "$FRONTEND_PID" 2>/dev/null || true
kill "$MOCK_PID" 2>/dev/null || true

echo "== [pgo 6/6] merge profiles + final profile-use build"
ls "$PGO_RAW"/*.profraw >/dev/null   # fail loudly if the run produced nothing
"$PROFDATA" merge -o "$PGO_DATA" "$PGO_RAW"
RUSTFLAGS="-Cprofile-use=$PGO_DATA -Cllvm-args=-pgo-warn-missing-function $CPU" \
    cargo build --release --bin vllm-rs $FEAT

echo "== PGO build complete: $BIN/vllm-rs"
