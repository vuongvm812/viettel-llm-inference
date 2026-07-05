#!/usr/bin/env bash
# P6 — PGO → LTO → BOLT build optimization for the `inference-runtime` binary.
# Spec: docs/design/build-optimization/design.md. Order: LTO is baked into
# [profile.release] (services/Cargo.toml); PGO is a two-pass build trained on the
# trace replay; BOLT is a Linux-only post-link pass on the PGO+LTO binary.
#
# Optimizes OUR serving/ring/scheduler/sampling hot paths, not libllama (P7).
#
# Idempotent: wipes prior profile data on each run. Builds run from services/ so
# services/.cargo/config.toml (-Ctarget-cpu=native) applies; the PGO passes re-pass
# -Ctarget-cpu=native in RUSTFLAGS because an env RUSTFLAGS overrides that config.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICES="$REPO/services"
BIN="$SERVICES/target/release/inference-runtime"
CONFIG="$REPO/config/default-config.yaml"
TARGET_URL="http://localhost:8001"           # our server (8000 is vLLM baseline)
PGO_DIR="/tmp/pgo"
PROFDATA="$PGO_DIR/merged.profdata"
OUT_DIR="${OUT_DIR:-$REPO/target-pgo}"        # baseline/optimized json + report land here
NATIVE="-Ctarget-cpu=native"
REPLAY_LOOPS="${PGO_REPLAY_LOOPS:-3}"         # loop the 120-req trace for denser PGO coverage (design §5)
PY="$(command -v python3 || command -v python)" || { echo "python not found" >&2; exit 1; }

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }

# Never leave a server bound to 0.0.0.0:8001 if a build/replay fails mid-run (it
# would wedge the next run's bind). run_and_replay sets SERVER_PID while a server
# is up and clears it after a clean shutdown; this trap kills any survivor.
SERVER_PID=""
cleanup() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

mkdir -p "$OUT_DIR"
rm -rf "$PGO_DIR" /tmp/perf.data /tmp/bolt.fdata
mkdir -p "$PGO_DIR"

# --- locate an llvm-profdata matching rustc's LLVM (rustup llvm-tools-preview) ---
LLVM_PROFDATA="$(command -v llvm-profdata || true)"
if [ -z "$LLVM_PROFDATA" ]; then
  LLVM_PROFDATA="$(find "$(rustc --print sysroot)" -name 'llvm-profdata*' 2>/dev/null | head -n1 || true)"
fi
[ -n "$LLVM_PROFDATA" ] || die "llvm-profdata not found — run: rustup component add llvm-tools-preview"
log "llvm-profdata: $LLVM_PROFDATA"

# Build release from services/. $NATIVE goes on EVERY build (incl. baseline) so the
# baseline vs PGO comparison isolates PGO — a set-but-empty `RUSTFLAGS=""` overrides
# (does not merge with) .cargo/config.toml's build.rustflags, so target-cpu must be
# passed here, not relied upon from config, whenever RUSTFLAGS is set. $1 = extra flags.
build() { ( cd "$SERVICES" && RUSTFLAGS="$NATIVE ${1:-}" cargo build --release -p inference-runtime ); }

# Start the server, poll /health, replay the trace REPLAY_LOOPS times to drive load,
# then SIGINT so the graceful-shutdown path runs atexit → PGO flushes *.profraw.
# $1 = replay output json (last loop wins; profile accumulates over the whole run).
run_and_replay() {
  local out="$1"
  ( cd "$REPO" && "$BIN" "$CONFIG" ) & SERVER_PID=$!
  for _ in $(seq 1 100); do
    curl -fsS "$TARGET_URL/health" >/dev/null 2>&1 && break
    kill -0 "$SERVER_PID" 2>/dev/null || die "server exited before becoming ready"
    sleep 0.1
  done
  curl -fsS "$TARGET_URL/health" >/dev/null 2>&1 || die "server never became ready"
  for _ in $(seq 1 "$REPLAY_LOOPS"); do
    ( cd "$REPO" && "$PY" bench/replay.py --target "$TARGET_URL" --out "$out" --closed-loop 8 )
  done
  kill -INT "$SERVER_PID"
  # Bound the graceful drain. replay.py has already exited so no streams linger,
  # but never let a wedged shutdown hang the build — SIGKILL after ~30s.
  for _ in $(seq 1 300); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 0.1; done
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    die "server did not exit within 30s of SIGINT (PGO profile may be missing)"
  fi
  wait "$SERVER_PID" || die "server did not exit cleanly (PGO profile may be missing)"
  SERVER_PID=""   # cleanly shut down — nothing for the EXIT trap to reap
}

# 1. Baseline: plain --release (LTO + target-cpu=native, no PGO). NB the baseline
# already carries fat LTO from [profile.release], so the table isolates the PGO
# delta — LTO's own contribution vs a no-LTO build is not separately measured (a
# full design §5 stage-by-stage would add a plain-vs-LTO row).
log "baseline build (--release)"
build ""
run_and_replay "$OUT_DIR/baseline.json"

# 2. PGO pass 1: instrumented build; the replay emits *.profraw.
log "PGO pass 1: instrumented build + trace replay"
build "-Cprofile-generate=$PGO_DIR"
run_and_replay "$OUT_DIR/pgo-gen.json"
ls "$PGO_DIR"/*.profraw >/dev/null 2>&1 \
  || die "no *.profraw emitted — the atexit flush (graceful shutdown) did not run"

# 3. Merge profiles.
log "merge profiles → $PROFDATA"
"$LLVM_PROFDATA" merge -o "$PROFDATA" "$PGO_DIR"/*.profraw
[ -s "$PROFDATA" ] || die "merged.profdata is empty"

# 4. PGO pass 2: optimized build using the profile.
log "PGO pass 2: profile-use build (+ fat LTO)"
build "-Cprofile-use=$PROFDATA -Cllvm-args=-pgo-warn-missing-function"
run_and_replay "$OUT_DIR/optimized.json"

# 5. Before/after comparison.
log "compare: baseline vs PGO+LTO"
( cd "$REPO" && "$PY" bench/compare.py "$OUT_DIR/baseline.json" "$OUT_DIR/optimized.json" )

# 6. BOLT — ELF/Linux only, needs perf (LBR) + llvm-bolt. (Fallback if no perf LBR:
#    `llvm-bolt -instrument` → run trace → .fdata → re-optimize; not wired here.)
if [ "$(uname -s)" = "Linux" ] && command -v perf >/dev/null && command -v llvm-bolt >/dev/null; then
  log "BOLT: emit-relocs build + perf profile + llvm-bolt"
  build "-Cprofile-use=$PROFDATA -Clink-args=-Wl,--emit-relocs"
  # Here SERVER_PID is the *perf* pid (perf launches the server as its child).
  # SIGINT stops the perf session, which reaps the server; BOLT needs only the
  # already-written perf.data, not a graceful server exit. Linux-only, deferred —
  # exercise on the target box (see ROADMAP P6).
  ( cd "$REPO" && perf record -e cycles:u -j any,u -o /tmp/perf.data -- "$BIN" "$CONFIG" ) & SERVER_PID=$!
  for _ in $(seq 1 100); do
    curl -fsS "$TARGET_URL/health" >/dev/null 2>&1 && break
    kill -0 "$SERVER_PID" 2>/dev/null || die "perf/server exited before becoming ready"
    sleep 0.1
  done
  ( cd "$REPO" && "$PY" bench/replay.py --target "$TARGET_URL" --out "$OUT_DIR/bolt-train.json" --closed-loop 8 )
  kill -INT "$SERVER_PID"; wait "$SERVER_PID" || true; SERVER_PID=""
  perf2bolt -p /tmp/perf.data -o /tmp/bolt.fdata "$BIN"
  llvm-bolt "$BIN" -o "$BIN.bolt" -data=/tmp/bolt.fdata \
    -reorder-blocks=ext-tstate -reorder-functions=hfsort -split-functions -split-all-cold -dyno-stats
  [ -x "$BIN.bolt" ] || die "BOLT produced no binary"
  log "BOLT binary: $BIN.bolt"
else
  log "BOLT skipped (non-Linux / perf|llvm-bolt absent) — ELF-only. Final artifact: PGO+LTO $BIN"
fi

[ -x "$BIN" ] || die "final binary missing"
log "done — baseline=$OUT_DIR/baseline.json optimized=$OUT_DIR/optimized.json"
printf '\033[1;33mNOTE: the dev sandbox runs the P1 mock backend (synthetic latency/tokens), so the\n'
printf 'comparison table exercises the pipeline but is NOT a real perf signal. The tok/s win\n'
printf 'is a --features llama + Linux/GPU deliverable; BOLT is Linux-only. See ROADMAP P6.\033[0m\n'
