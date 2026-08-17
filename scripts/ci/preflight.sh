#!/usr/bin/env bash
# Refuse to produce a number nobody can trust. Runs immediately before the bench stage.
#
# The concurrency group in the workflows stops two CI jobs sharing the GPU. It does NOT stop a
# human running `make up` on the box while CI benches -- which is likely with people working the
# same machine for three days, and completely invisible in the result: the bench succeeds, every
# request returns 200, and the latency is simply wrong.
#
# So this fails the job instead. A missing bench is obviously missing; a wrong bench gets acted on.
#
#   scripts/ci/preflight.sh round-2
set -euo pipefail

ROUND="${1:-round-2}"
fail=0

note() { printf '  %s\n' "$*"; }
bad()  { printf 'FAIL %s\n' "$*"; fail=1; }
ok()   { printf 'OK   %s\n' "$*"; }

echo "=== preflight ($ROUND) ==="

# --- 1. Is anything else using the GPU? --------------------------------------------------------
# `nvidia-smi --query-compute-apps` lists processes with a CUDA context. Empty is what we want.
# Absent nvidia-smi is not a pass: it means we cannot tell, and on a bench box that is worse than
# a clear no -- so warn loudly rather than silently proceeding as if the box were idle.
if command -v nvidia-smi >/dev/null 2>&1; then
  apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true)"
  if [ -n "$apps" ]; then
    bad "GPU is not idle -- something already holds a CUDA context:"
    printf '%s\n' "$apps" | sed 's/^/       /'
    note "a bench sharing the card measures the other process as much as ours"
  else
    ok "GPU idle"
  fi
else
  bad "nvidia-smi absent -- cannot prove the GPU is idle"
fi

# --- 2. Is another compose stack already up? ---------------------------------------------------
# Our own stack is expected to be down at this point: the bench stage brings it up itself. Anything
# already running is either a leftover from a killed job or someone's manual `make up`.
if command -v docker >/dev/null 2>&1; then
  running="$(docker ps --filter 'label=com.docker.compose.project' \
                       --format '{{.Names}}\t{{.Image}}' 2>/dev/null || true)"
  if [ -n "$running" ]; then
    bad "a compose stack is already running:"
    printf '%s\n' "$running" | sed 's/^/       /'
    note "tear it down (make down / make ci-down) before benching"
  else
    ok "no compose stack running"
  fi
else
  bad "docker absent"
fi

# --- 3. Both require-flags forced on for the bench ----------------------------------------------
# vtl-sched's Kind enum covers FullAttention and Mamba only; anything else (sliding-window,
# chunked-local, cross-attention) makes it log one line and hand back to stock vLLM. round-2 added
# a second, independent refusal path in rust_runner.py with its own flag. Either one self-disabling
# looks IDENTICAL, in every latency number, to a component that ran and did not help -- which is
# the single easiest way to spend a contest day optimizing something that was never running.
#
# CHECKED IN THE COMPOSE OVERLAY, NOT THE SHELL. These reach the server through the compose `environment:`
# block; exporting them in the runner's shell does nothing at all. The submission deliberately ships
# VTL_RUST_SCHED_REQUIRE=0 (serving-but-slower beats not serving), so the bench overlay is what has
# to override it -- and that override is exactly what is easy to lose in a merge.
CI_OVERLAY="${ROUND}/docker-compose.ci-bench.yaml"
if [ ! -f "$CI_OVERLAY" ]; then
  bad "$CI_OVERLAY not found -- cannot confirm the bench forces the require-flags on"
else
  for flag in VTL_RUST_SCHED_REQUIRE VTL_RUST_RUNNER_REQUIRE; do
    if grep -qE "^[[:space:]]*${flag}:[[:space:]]*[\"']?1[\"']?[[:space:]]*$" "$CI_OVERLAY"; then
      ok "$flag=1 in $CI_OVERLAY"
    else
      bad "$flag is not forced to 1 in $CI_OVERLAY -- the component may self-disable and the bench will not say so"
    fi
  done
fi

# --- 4. Disk ------------------------------------------------------------------------------------
# Images are ~10GB base plus build layers, and a full disk fails a build halfway with an error that
# reads like a code problem.
avail_kb="$(df -Pk . | awk 'NR==2 {print $4}')"
if [ "${avail_kb:-0}" -lt 20971520 ]; then   # 20 GiB
  bad "only $(( avail_kb / 1048576 )) GiB free on the docker data-root filesystem"
else
  ok "$(( avail_kb / 1048576 )) GiB free"
fi

# --- 5. GPU clocks (report, do not gate) --------------------------------------------------------
# Clocks are the dominant variance term: on the default DVFS governor two identical arms can differ
# by more than the effect being measured. But `scripts/host-tune.sh` has only ever been run in
# `show` mode -- its nvidia-smi/sysfs writes are unexercised on a real GPU host -- so this REPORTS
# what it observes rather than asserting that host-tune worked.
if command -v nvidia-smi >/dev/null 2>&1; then
  persist="$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader 2>/dev/null | head -1 || true)"
  note "persistence mode: ${persist:-unknown}  (make host-tune locks clocks; make host-tune-show reads them)"
fi

echo
[ "$fail" -eq 0 ] || { echo "preflight FAILED -- not benching"; exit 1; }
echo "preflight OK"
