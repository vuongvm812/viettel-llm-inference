#!/usr/bin/env bash
# Host-level latency tuning for the DEV/BENCH box. NOT part of any submission.
#
#   sudo scripts/host-tune.sh apply     # or: make host-tune
#   sudo scripts/host-tune.sh reset     # or: make host-tune-reset
#   scripts/host-tune.sh show           # read-only, no root needed
#   scripts/host-tune.sh --cpuset       # read-only: print the GPU-local `cpuset:` compose line
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

# --- NUMA / cpuset advice ------------------------------------------------------------------
# READ-ONLY host inspection: it opens sysfs and procfs and changes nothing, so it cannot
# disturb apply/reset state. It prints the `cpuset:` line to paste under the `model` service
# in round-2/docker-compose.yaml, where it ships COMMENTED.
#
# WHY: an H200 hangs off one socket's PCIe root complex. Left unpinned, the container's
# threads -- the OMP pool, the tokio runtimes, and above all the per-step pinned staging
# buffers behind every H2D/D2H copy -- get scheduled on whichever cores are free, so a copy
# can cross the interconnect on its way to the GPU. Pinning to the GPU's own node removes
# that. Pinning to the WRONG node makes every copy remote, which is worse than not pinning
# at all, and the judge box's topology is not ours to guess: hence commented-by-default.

# Collapse a newline-separated CPU list into compact ranges: 0 1 2 5 -> "0-2,5".
_ranges() {
  sort -n | awk '
    function emit() { out = out (out == "" ? "" : ",") (start == prev ? start : start "-" prev) }
    NR == 1 { start = prev = $1; next }
    $1 == prev + 1 { prev = $1; next }
    { emit(); start = prev = $1 }
    END { if (NR) { emit(); print out } }'
}

# The nvidia driver names its procfs directories after the PCI BDF (domain included), which
# is the authoritative answer and needs no tooling installed. lspci is the fallback.
_gpu_bdf() {
  local d b
  for d in /proc/driver/nvidia/gpus/*/; do
    [ -r "${d}information" ] || continue
    b=$(basename "$d")
    case "$b" in *:*:*.*) echo "$b"; return 0 ;; esac
  done
  # -D forces the domain prefix; /sys/bus/pci/devices paths are keyed on the full form.
  have lspci && lspci -D -d 10de: 2>/dev/null | awk '/VGA|3D controller/ { print $1; exit }'
}

# nvidia-smi's own CPU-affinity column, used only when lscpu and the node sysfs are absent.
# The row's first field that looks like a CPU range IS the CPU Affinity column: everything
# before it is "GPU0" and the link-type matrix ("X", "PIX", ...), and the NUMA Affinity
# column that could also match a bare number comes after it.
_topo_affinity() {
  have nvidia-smi || return 0
  nvidia-smi topo -m 2>/dev/null | awk '
    /^GPU0/ {
      for (i = 2; i <= NF; i++)
        if ($i ~ /^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$/) { print $i; exit }
    }'
}

cpuset() {
  local bdf node cores src
  echo "== GPU / NUMA affinity"

  bdf=$(_gpu_bdf)
  if [ -z "${bdf:-}" ]; then
    log "gpu pci bdf" "not found (no nvidia driver, no lspci) -- cannot advise"
    return 0
  fi
  log "gpu pci bdf" "$bdf"

  node=""
  [ -r "/sys/bus/pci/devices/$bdf/numa_node" ] && node=$(cat "/sys/bus/pci/devices/$bdf/numa_node")
  if [ -z "${node:-}" ]; then
    log "numa_node" "unreadable (/sys/bus/pci/devices/$bdf/numa_node)"
    node="-1"
  else
    log "numa_node" "$node"
  fi

  # -1 is what the kernel reports on a single-node box, and on VMs/hypervisors that do not
  # expose PCI locality at all. Either way there is no node to prefer.
  if [ "$node" = "-1" ]; then
    echo
    echo "  no NUMA pinning needed -- the GPU reports no node affinity (single-node host,"
    echo "  or a hypervisor that hides PCI locality). Leave cpuset commented out."
    return 0
  fi

  cores=""
  if have lscpu; then
    cores=$(lscpu -p=CPU,NODE 2>/dev/null | awk -F, -v n="$node" '!/^#/ && $2 == n { print $1 }' | _ranges)
    src="lscpu -p=CPU,NODE"
  fi
  if [ -z "${cores:-}" ] && [ -r "/sys/devices/system/node/node$node/cpulist" ]; then
    cores=$(cat "/sys/devices/system/node/node$node/cpulist")
    src="/sys/devices/system/node/node$node/cpulist"
  fi
  if [ -z "${cores:-}" ]; then
    cores=$(_topo_affinity)
    src="nvidia-smi topo -m (CPU Affinity)"
  fi
  if [ -z "${cores:-}" ]; then
    log "node $node cpus" "could not be resolved -- leave cpuset commented out"
    return 0
  fi
  log "node $node cpus" "$cores  [$src]"

  echo
  echo "  Paste under the \`model\` service in round-2/docker-compose.yaml (it ships commented):"
  echo
  echo "    cpuset: \"$cores\""
  echo
  echo "  DEV/BENCH BOXES ONLY. This is host-specific: a compose file carrying one host's core"
  echo "  list is a compose file that pins the judge's container to the wrong socket."
}

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
  echo "== NUMA"
  log "gpu-local cpuset" "run '$0 --cpuset' for the compose line"
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
  cpuset|--cpuset) cpuset ;;
  *) echo "usage: $0 {apply|reset|show|--cpuset}" >&2; exit 2 ;;
esac
