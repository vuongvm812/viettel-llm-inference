#!/usr/bin/env bash
#
# Linux deployment setup for the inference-runtime:
#   1. CPU pinning prep  — dedicate cores 0/1/2 (perf governor, turbo off, isolcpus guidance)
#   2. BOLT              — LLVM post-link optimizer, used in P6 (docs/ROADMAP.md)
#   3. jemalloc          — LD_PRELOAD allocator for lower fragmentation / faster multithread alloc
#
# Idempotent: safe to re-run. Steps that touch the system use sudo and check first.
# macOS is a dev-only target (pinning/BOLT/jemalloc are Linux) — the script exits early there.
#
# Usage:
#   ./script/setup.sh                 # all steps
#   source ./script/runtime-env.sh    # then load LD_PRELOAD into your shell (written by step 3)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/script/runtime-env.sh"

log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

if [[ "$(uname -s)" != "Linux" ]]; then
  warn "Non-Linux host ($(uname -s)). CPU pinning, BOLT and jemalloc are Linux-only;"
  warn "this box is dev-only — topology/correctness run here, the perf story is on the Linux target."
  exit 0
fi

# Cores the runtime pins to (must match config/default-config.yaml runtime.cores).
WEB_IO_CORE=0
TEXT_CORE=1
FAST_LOOP_CORE=2

# ---------------------------------------------------------------------------
# 1. CPU pinning prep
# ---------------------------------------------------------------------------
setup_cpu_pinning() {
  log "CPU pinning prep for cores ${WEB_IO_CORE}/${TEXT_CORE}/${FAST_LOOP_CORE}"

  # Lock the pinned cores to the performance governor so the busy-spin loops
  # (core1/core2) don't get down-clocked mid-flight.
  if have cpupower; then
    sudo cpupower frequency-set -g performance >/dev/null 2>&1 \
      && log "  scaling governor → performance" \
      || warn "  cpupower failed (VM / no cpufreq?) — skipping governor"
  else
    for core in "$WEB_IO_CORE" "$TEXT_CORE" "$FAST_LOOP_CORE"; do
      gov="/sys/devices/system/cpu/cpu${core}/cpufreq/scaling_governor"
      [[ -w "$gov" ]] && echo performance | sudo tee "$gov" >/dev/null || true
    done
    log "  scaling governor set where writable (install linux-tools for cpupower)"
  fi

  # Disable turbo so per-iteration latency is stable (not the max clock).
  if [[ -w /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
    echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo >/dev/null
    log "  intel turbo disabled"
  elif [[ -w /sys/devices/system/cpu/cpufreq/boost ]]; then
    echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost >/dev/null
    log "  cpufreq boost disabled"
  fi

  # isolcpus can't be set at runtime — it's a boot param. Print the recommended
  # line instead of editing GRUB (too destructive to automate).
  cat <<EOF
  [manual] To fully dedicate the busy-spin cores, isolate them from the scheduler
           at boot. Add to GRUB_CMDLINE_LINUX in /etc/default/grub, then update-grub:
             isolcpus=${TEXT_CORE},${FAST_LOOP_CORE} nohz_full=${TEXT_CORE},${FAST_LOOP_CORE} rcu_nocbs=${TEXT_CORE},${FAST_LOOP_CORE}
           Verify after reboot:  cat /sys/devices/system/cpu/isolated
EOF
}

# ---------------------------------------------------------------------------
# 2. BOLT (LLVM post-link optimizer)  — https://github.com/llvm/llvm-project/tree/main/bolt
# ---------------------------------------------------------------------------
setup_bolt() {
  log "BOLT install"
  if have llvm-bolt || compgen -G "/usr/bin/llvm-bolt*" >/dev/null; then
    log "  llvm-bolt already present ($(command -v llvm-bolt || echo /usr/bin/llvm-bolt*))"
  else
    # Prebuilt LLVM (incl. llvm-bolt + perf2bolt) from apt.llvm.org — far less work
    # than building the monorepo from source per the BOLT README.
    log "  installing prebuilt LLVM toolchain from apt.llvm.org"
    tmp="$(mktemp -d)"
    curl -fsSL https://apt.llvm.org/llvm.sh -o "${tmp}/llvm.sh"
    chmod +x "${tmp}/llvm.sh"
    sudo "${tmp}/llvm.sh" all
    rm -rf "$tmp"
    # apt.llvm.org names binaries llvm-bolt-<ver>; expose an unversioned alias.
    latest_bolt="$(ls /usr/bin/llvm-bolt-* 2>/dev/null | sort -V | tail -1 || true)"
    if [[ -n "$latest_bolt" && ! -e /usr/bin/llvm-bolt ]]; then
      sudo ln -sf "$latest_bolt" /usr/bin/llvm-bolt
      sudo ln -sf "${latest_bolt/llvm-bolt/perf2bolt}" /usr/bin/perf2bolt 2>/dev/null || true
      log "  linked $(basename "$latest_bolt") → /usr/bin/llvm-bolt"
    fi
  fi

  # BOLT's profiling stage uses `perf record`; allow it without root.
  if [[ "$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 4)" -gt 1 ]]; then
    echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid >/dev/null
    log "  kernel.perf_event_paranoid → 1 (needed for perf record in the PGO/BOLT pipeline)"
  fi
  have llvm-bolt && llvm-bolt --version | head -1 | sed 's/^/  /' || warn "  llvm-bolt not on PATH yet"
}

# ---------------------------------------------------------------------------
# 3. jemalloc (LD_PRELOAD)
# ---------------------------------------------------------------------------
setup_jemalloc() {
  log "jemalloc install"
  if ! dpkg -s libjemalloc2 >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y libjemalloc2
  else
    log "  libjemalloc2 already installed"
  fi

  # libjemalloc2 ships libjemalloc.so.2; the unversioned .so comes from -dev.
  # Prefer the exact path if present, else the newest match.
  local so="/usr/lib/x86_64-linux-gnu/libjemalloc.so"
  if [[ ! -e "$so" ]]; then
    so="$(find /usr/lib /lib -name 'libjemalloc.so*' 2>/dev/null | sort -V | tail -1 || true)"
  fi
  if [[ -z "$so" || ! -e "$so" ]]; then
    warn "  could not locate libjemalloc — skipping LD_PRELOAD"
    return
  fi

  # A child process can't mutate the parent shell, so persist the export into a
  # sourceable env file (and print it). `make run/inference` / your launcher can
  # `source script/runtime-env.sh` before starting the binary.
  cat > "$ENV_FILE" <<EOF
# Generated by script/setup.sh — source before launching inference-runtime.
export LD_PRELOAD="${so}"
EOF
  log "  LD_PRELOAD=${so}"
  log "  wrote ${ENV_FILE} — run:  source script/runtime-env.sh"
}

main() {
  setup_cpu_pinning
  setup_bolt
  setup_jemalloc
  log "done. Load the allocator into your shell with:  source script/runtime-env.sh"
}

main "$@"
