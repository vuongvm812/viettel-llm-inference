#!/usr/bin/env bash
# Host-level latency tuning for the DEV/BENCH box. NOT part of any submission.
#
#   sudo scripts/host-tune.sh apply     # or: make host-tune
#   sudo scripts/host-tune.sh reset     # or: make host-tune-reset
#   scripts/host-tune.sh show           # read-only, no root needed
#
# WHY THIS EXISTS, AND WHY IT IS NOT AN OPTIMIZATION. None of this makes the submitted
# container faster -- the judge runs `docker compose up` on their own host and none of these
# knobs are reachable from a compose file. What it does is make OUR measurements mean
# something. The dominant term is GPU clocks: left on the default DVFS governor an H200
# ramps between clock states in response to thermals and load, so two identical arms can
# differ by more than the effect being measured. Locking clocks collapses that variance,
# which is why the A/B protocol needs several boots per arm without it.
#
# Every change is recorded to a state file at apply time and restored by `reset`, because a
# box left with clocks locked and the governor pinned is a box that lies about power draw
# and thermals for everyone who uses it next.
set -uo pipefail

STATE="${VTL_TUNE_STATE:-/var/tmp/vtl-host-tune.state}"

log() { printf '  %-34s %s\n' "$1" "$2"; }

need_root() {
  if [ "$(id -u)" -ne 0 ]; then
    echo "host-tune: must run as root (try: sudo $0 $*)" >&2
    exit 1
  fi
}

have() { command -v "$1" >/dev/null 2>&1; }

# --- read current state -------------------------------------------------------------------
show() {
  echo "== GPU"
  if have nvidia-smi; then
    nvidia-smi --query-gpu=name,persistence_mode,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem \
      --format=csv 2>/dev/null | sed 's/^/  /' || echo "  (query failed)"
  else
    echo "  nvidia-smi absent"
  fi
  echo "== CPU"
  if [ -r /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    log "governor(cpu0)" "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
  else
    log "governor" "no cpufreq sysfs (VM or fixed-frequency host)"
  fi
  echo "== Memory / kernel"
  [ -r /sys/kernel/mm/transparent_hugepage/enabled ] &&
    log "transparent_hugepage" "$(cat /sys/kernel/mm/transparent_hugepage/enabled)"
  for k in kernel.numa_balancing vm.swappiness vm.zone_reclaim_mode; do
    have sysctl && log "$k" "$(sysctl -n "$k" 2>/dev/null || echo n/a)"
  done
  echo "== State"
  if [ -f "$STATE" ]; then log "saved state" "$STATE"; else log "saved state" "(none -- not applied)"; fi
}

# --- apply --------------------------------------------------------------------------------
apply() {
  need_root apply
  if [ -f "$STATE" ]; then
    echo "host-tune: already applied ($STATE). Run 'reset' first to re-apply cleanly." >&2
    exit 1
  fi
  : > "$STATE"

  echo "== GPU"
  if have nvidia-smi; then
    # Persistence mode: without it the driver tears down and re-initializes its context when
    # no client holds the GPU, and the next process pays that init inside its first request.
    prev_pm=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader 2>/dev/null | head -1)
    echo "PERSISTENCE=$prev_pm" >> "$STATE"
    nvidia-smi -pm 1 >/dev/null 2>&1 && log "persistence mode" "ENABLED (was $prev_pm)" \
      || log "persistence mode" "FAILED (not root, or unsupported)"

    # Locked clocks: THE reason this script exists. Pin SM and memory clocks to their max so
    # DVFS stops being a variable between A/B arms.
    max_sm=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits 2>/dev/null | head -1)
    max_mem=$(nvidia-smi --query-gpu=clocks.max.mem --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "${max_sm:-}" ] && [ "$max_sm" != "[N/A]" ]; then
      echo "LOCKED_CLOCKS=1" >> "$STATE"
      nvidia-smi -lgc "$max_sm,$max_sm" >/dev/null 2>&1 \
        && log "SM clock" "locked at ${max_sm} MHz" || log "SM clock" "lock FAILED"
      nvidia-smi -lmc "$max_mem,$max_mem" >/dev/null 2>&1 \
        && log "memory clock" "locked at ${max_mem} MHz" || log "memory clock" "lock unsupported"
    else
      log "clocks" "max clocks unreadable; left on DVFS (expect noisier A/B)"
    fi
  else
    log "nvidia-smi" "absent -- skipping all GPU tuning"
  fi

  echo "== CPU"
  if [ -w /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    echo "GOVERNOR=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)" >> "$STATE"
    n=0
    for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
      echo performance > "$g" 2>/dev/null && n=$((n + 1))
    done
    log "governor" "performance on $n CPUs"
  else
    log "governor" "no writable cpufreq sysfs; skipped"
  fi

  echo "== Memory / kernel"
  if [ -w /sys/kernel/mm/transparent_hugepage/enabled ]; then
    # `madvise`, NOT `always`: jemalloc is configured with metadata_thp:always and asks for
    # hugepages where it wants them. Blanket `always` hands them out everywhere and can cost
    # more in fault latency and fragmentation than it saves.
    cur=$(sed -n 's/.*\[\(.*\)\].*/\1/p' /sys/kernel/mm/transparent_hugepage/enabled)
    echo "THP=$cur" >> "$STATE"
    echo madvise > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null \
      && log "transparent_hugepage" "madvise (was $cur)" || log "transparent_hugepage" "write FAILED"
  else
    log "transparent_hugepage" "not writable; skipped"
  fi

  if have sysctl; then
    for kv in kernel.numa_balancing=0 vm.swappiness=0 vm.zone_reclaim_mode=0; do
      k=${kv%%=*}; v=${kv#*=}
      prev=$(sysctl -n "$k" 2>/dev/null)
      if [ -n "$prev" ]; then
        echo "SYSCTL:$k=$prev" >> "$STATE"
        sysctl -qw "$kv" 2>/dev/null && log "$k" "$v (was $prev)" || log "$k" "write FAILED"
      fi
    done
  fi

  # irqbalance keeps moving device interrupts between cores; on a latency run that is one
  # more source of variance. Only touched if it is actually running.
  if have systemctl && systemctl is-active --quiet irqbalance 2>/dev/null; then
    echo "IRQBALANCE=active" >> "$STATE"
    systemctl stop irqbalance && log "irqbalance" "stopped"
  fi

  if have tuned-adm; then
    prev=$(tuned-adm active 2>/dev/null | sed -n 's/.*: //p')
    [ -n "$prev" ] && echo "TUNED=$prev" >> "$STATE"
    tuned-adm profile latency-performance >/dev/null 2>&1 \
      && log "tuned profile" "latency-performance (was ${prev:-unknown})"
  fi

  echo
  echo "host-tune: applied. State saved to $STATE -- run 'reset' when done benchmarking."
}

# --- reset --------------------------------------------------------------------------------
reset() {
  need_root reset
  if [ ! -f "$STATE" ]; then
    echo "host-tune: no state file at $STATE -- nothing to reset." >&2
    exit 0
  fi

  while IFS= read -r line; do
    case "$line" in
      PERSISTENCE=*)
        want=${line#PERSISTENCE=}
        case "$want" in
          Disabled) nvidia-smi -pm 0 >/dev/null 2>&1 && log "persistence mode" "restored Disabled";;
          *) log "persistence mode" "left Enabled (was $want)";;
        esac ;;
      LOCKED_CLOCKS=1)
        nvidia-smi -rgc >/dev/null 2>&1 && log "SM clock" "reset to DVFS"
        nvidia-smi -rmc >/dev/null 2>&1 && log "memory clock" "reset to DVFS" ;;
      GOVERNOR=*)
        want=${line#GOVERNOR=}
        for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
          echo "$want" > "$g" 2>/dev/null
        done
        log "governor" "restored $want" ;;
      THP=*)
        want=${line#THP=}
        echo "$want" > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null \
          && log "transparent_hugepage" "restored $want" ;;
      SYSCTL:*)
        kv=${line#SYSCTL:}
        sysctl -qw "$kv" 2>/dev/null && log "${kv%%=*}" "restored ${kv#*=}" ;;
      IRQBALANCE=active)
        systemctl start irqbalance 2>/dev/null && log "irqbalance" "restarted" ;;
      TUNED=*)
        tuned-adm profile "${line#TUNED=}" >/dev/null 2>&1 \
          && log "tuned profile" "restored ${line#TUNED=}" ;;
    esac
  done < "$STATE"

  rm -f "$STATE"
  echo
  echo "host-tune: reset complete."
}

case "${1:-show}" in
  apply) apply ;;
  reset) reset ;;
  show)  show ;;
  *) echo "usage: $0 {apply|reset|show}" >&2; exit 2 ;;
esac
