#!/usr/bin/env bash
# PGO (profile-guided optimization) build of the vllm-rs frontend binary.
#
# Runs inside the `rust-builder` stage of Dockerfile.vllm-fork when
# VLLM_RS_PGO=1. There is no GPU here: the frontend serves entirely on CPU by
# pairing `vllm-rs serve --data-parallel-size-local 0` (frontend-only, owns the
# ZMQ handshake) with `vllm-mock-engine` (fakes prefill, emits random decode
# tokens). The real CPU tokenizer + full HTTP→sonic_rs-parse→stream path are
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
# Per-decode-step delay for the mock engine (ms). Set to ~the real model's TPOT so the
# frontend's streaming/detokenize/backpressure loop is profiled at production cadence
# instead of at memory speed (which biases the profile toward the wrong hot paths).
PGO_DECODE_STEP_DELAY_MS="${PGO_DECODE_STEP_DELAY_MS:-4}"
# -Ctarget-cpu (default native from the build arg): full host codegen (AVX-512 on an H200
# host). Applies to BOTH the instrumented and final builds, so build on NATIVE amd64 — a
# native/AVX2 instrumented binary crashes at startup under Rosetta/OrbStack during the training
# run. For an emulated local build, pass PGO_TARGET_CPU= (empty) to fall back to baseline
# x86-64; sonic_rs runtime-detects SIMD, so the JSON win survives either way.
CPU="${PGO_TARGET_CPU:+-Ctarget-cpu=$PGO_TARGET_CPU}"

cd "$RUST_DIR"
PROFDATA="$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/host: //p')/bin/llvm-profdata"

echo "== [pgo 1/6] build the (uninstrumented) mock engine"
cargo build --release -p vllm-mock-engine

echo "== [pgo 2/6] instrumented build of vllm-rs"
rm -rf "$PGO_RAW"
RUSTFLAGS="-Cprofile-generate=$PGO_RAW $CPU" \
    cargo build --release --bin vllm-rs $FEAT

echo "== [pgo 3/6] prepare training trace (pin model to the served tokenizer)"
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
        body["model"] = model   # match the served tokenizer; keep the trace's max_tokens
        out.write(json.dumps(rec) + "\n")
        n += 1
print(f"prepared {n} training requests -> {dst}")
PY

echo "== [pgo 4/6] boot frontend (CPU-only) + mock engine"
export RUST_BACKTRACE=full
BIN="$RUST_DIR/target/release"
FRONTEND_LOG=/tmp/pgo-frontend.log

# Self-check: does the instrumented binary even run? Isolates a PGO-instrumentation
# crash from a serve/model-specific one. `vllm-rs` needs a subcommand, so use --help
# (exits 0); the profile-generate rt writes profraw to $PGO_RAW.
echo "-- instrumented binary self-check (--help):"
LLVM_PROFILE_FILE="$PGO_RAW/selfcheck-%p.profraw" "$BIN/vllm-rs" --help >/dev/null \
    || { echo "FATAL: instrumented vllm-rs crashes before serve (PGO-instrumentation issue, not model/serve)"; exit 1; }

echo "-- model dir contents ($MODEL):"; ls -la "$MODEL" 2>&1 | head -20 || true

# Distinct filename for the SERVE profile so the merge/gate can tell it apart from the
# throwaway --help self-check profraw above (see the [pgo 6/6] gate).
LLVM_PROFILE_FILE="$PGO_RAW/serve-%p.profraw" "$BIN/vllm-rs" serve "$MODEL" \
    --data-parallel-size 1 --data-parallel-size-local 0 \
    --handshake-port "$HANDSHAKE_PORT" --host 127.0.0.1 --port "$HTTP_PORT" \
    > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
"$BIN/vllm-mock-engine" --handshake-address "tcp://127.0.0.1:${HANDSHAKE_PORT}" \
    --decode-step-delay-ms "$PGO_DECODE_STEP_DELAY_MS" &
MOCK_PID=$!

dump_frontend_log() { echo "--- frontend log (${FRONTEND_LOG}) ---"; cat "$FRONTEND_LOG" 2>/dev/null || echo "(empty)"; echo "--- end frontend log ---"; }

# Wait for the frontend to report healthy (tokenizer load + engine register).
for i in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${HTTP_PORT}/health" >/dev/null 2>&1; then
        echo "frontend healthy after ${i}s"; break
    fi
    if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo "ERROR: frontend exited before becoming healthy (exit/signal below)"
        wait "$FRONTEND_PID"; echo "frontend exit status: $?"
        dump_frontend_log; exit 1
    fi
    sleep 1
done
curl -fsS "http://127.0.0.1:${HTTP_PORT}/health" >/dev/null || { dump_frontend_log; exit 1; }

echo "== [pgo 5/6] replay training corpus"
pip install --no-cache-dir -q aiohttp
# closed-loop 4 ~= the real workload's steady-state in-flight count (~2-3 by Little's
# law: ~1.1 req/s aggregate x ~2.4s/response). 16 over-saturated queues and biased the
# profile toward backpressure paths production never hits.
( cd /src/bench && python3 replay.py \
    --target "http://127.0.0.1:${HTTP_PORT}" \
    --trace /tmp/pgo-train.jsonl \
    --closed-loop 4 \
    --out /tmp/pgo-replay.json )

# Flush counters: SIGINT lets the instrumented frontend write its .profraw on exit.
kill -INT "$FRONTEND_PID" 2>/dev/null || true
wait "$FRONTEND_PID" 2>/dev/null || true
kill "$MOCK_PID" 2>/dev/null || true

echo "== [pgo 6/6] gate on a real serve profile + merge + final profile-use build"
# Only the SERVE run's profile is useful. The old `ls *.profraw` guard also matched the
# --help self-check profraw, so a serve flush failure (SIGINT race / handler never reached)
# would silently ship a binary PGO-optimized for `--help` alone — slower than a plain -O3
# build because every real hot function is then treated as cold. Gate on a NON-EMPTY
# serve-*.profraw so that failure fails the build loudly instead.
shopt -s nullglob
serve_profiles=("$PGO_RAW"/serve-*.profraw)
shopt -u nullglob
if [ ${#serve_profiles[@]} -eq 0 ]; then
    echo "FATAL: no serve-*.profraw — the training replay never flushed its counters"
    dump_frontend_log; exit 1
fi
real=""
for p in "${serve_profiles[@]}"; do
    sz=$(wc -c < "$p")
    echo "  serve profile $p: ${sz} bytes"
    if [ "$sz" -gt 1024 ]; then real=1; fi   # explicit if: `&& real=1` can trip set -e
done
[ -n "$real" ] || { echo "FATAL: serve profile(s) present but <1KB — counters didn't flush (empty profile)"; exit 1; }
"$PROFDATA" merge -o "$PGO_DATA" "${serve_profiles[@]}"
RUSTFLAGS="-Cprofile-use=$PGO_DATA -Cllvm-args=-pgo-warn-missing-function $CPU" \
    cargo build --release --bin vllm-rs $FEAT

echo "== PGO build complete: $BIN/vllm-rs"
