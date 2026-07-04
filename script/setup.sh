#!/usr/bin/env bash
#
# Setup for the inference-runtime:
#   0. llama.cpp backend — cmake + C/C++ toolchain so `cargo build --features llama`
#                          can build libllama (Metal on macOS, CUDA on Linux), plus
#                          download + wire the GGUF model. Runs on BOTH macOS and Linux.
#   1. CPU pinning prep  — dedicate cores 0/1/2 (perf governor, turbo off, isolcpus guidance)   [Linux]
#   2. BOLT              — LLVM post-link optimizer, used in P6 (docs/ROADMAP.md)               [Linux]
#   3. jemalloc          — LD_PRELOAD allocator for lower fragmentation / faster alloc          [Linux]
#
# Idempotent: safe to re-run. Steps that touch the system use sudo and check first.
# Steps 1–3 are Linux-only (the 3-core perf story is the Linux target); step 0 runs
# everywhere so Metal inference works on a macOS dev box too.
#
# Usage:
#   ./script/setup.sh                 # all steps; converts Qwen3.5-2B (HF BF16) → BF16 GGUF
#   MODEL_URL=https://.../model.gguf ./script/setup.sh   # skip convert, use a pre-built GGUF
#   source ./script/runtime-env.sh    # then load LD_PRELOAD into your shell (Linux, written by step 3)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/script/runtime-env.sh"
OS="$(uname -s)"

log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# Cores the runtime pins to (must match config/default-config.yaml runtime.cores).
WEB_IO_CORE=0
TEXT_CORE=1
FAST_LOOP_CORE=2

# Model: Qwen3.5-2B dense transformer, BF16 (unquantized). llama.cpp needs GGUF,
# and a BF16 HF model has no pre-quantized GGUF to download — so the default flow
# converts the HF safetensors to a BF16 GGUF (convert_hf_to_gguf.py --outtype bf16).
# The dest basename must match `model.gguf_path` in config/default-config.yaml.
# Override via env:
#   MODEL_URL       direct download URL for an already-built .gguf (skips convert), OR
#   MODEL_HF_REPO   a HF repo id of safetensors to snapshot + convert (default below), and
#   MODEL_HF_FILE   a single pre-built .gguf file in that repo to download instead of converting.
#   MODEL_OUTTYPE   GGUF precision for conversion (default bf16 — keep unquantized).
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/models}"
MODEL_DEST="${MODEL_DEST:-${MODEL_DIR}/qwen3.5-2b-bf16.gguf}"
MODEL_URL="${MODEL_URL:-}"
MODEL_HF_REPO="${MODEL_HF_REPO:-Qwen/Qwen3.5-2B}"
MODEL_HF_FILE="${MODEL_HF_FILE:-}"
MODEL_OUTTYPE="${MODEL_OUTTYPE:-bf16}"
# Pin the expected SHA-256 of a downloaded GGUF. Strongly recommended: a GGUF is
# native-library input at the FFI boundary, so an unverified download is untrusted
# native input. Downloads verify before install; unset → loud warning, not a hard fail.
MODEL_SHA256="${MODEL_SHA256:-}"

# ---------------------------------------------------------------------------
# 0. llama.cpp backend toolchain + model  (macOS + Linux)
# ---------------------------------------------------------------------------
# The `llama-cpp-2` crate compiles a bundled libllama with cmake at `cargo build`
# time — so we don't clone/build llama.cpp separately, we just need the toolchain
# it invokes: cmake, a C/C++ compiler, and libclang (bindgen). GPU accel is picked
# per-OS by Cargo.toml: Metal on macOS (ships with Xcode), CUDA on Linux (needs the
# CUDA toolkit — checked, not auto-installed; it's multi-GB and box-specific).
setup_llama_toolchain() {
  log "llama.cpp build toolchain"
  case "$OS" in
    Darwin)
      if ! xcode-select -p >/dev/null 2>&1; then
        warn "  Xcode Command Line Tools missing — installing (provides clang + Metal)"
        xcode-select --install || warn "  run 'xcode-select --install' manually, then re-run"
      else
        log "  Xcode CLT present (clang + Metal framework)"
      fi
      if have cmake; then
        log "  cmake present ($(cmake --version | head -1))"
      elif have brew; then
        log "  installing cmake via Homebrew"; brew install cmake
      else
        warn "  cmake missing and Homebrew not found — install Homebrew (brew.sh) then 'brew install cmake'"
      fi
      ;;
    Linux)
      local pkgs=()
      have cmake   || pkgs+=(cmake)
      have cc      || pkgs+=(build-essential)
      have clang   || pkgs+=(clang libclang-dev)
      if ((${#pkgs[@]})); then
        log "  apt install: ${pkgs[*]}"
        sudo apt-get update -qq && sudo apt-get install -y "${pkgs[@]}"
      else
        log "  cmake + compiler + clang already present"
      fi
      # CUDA is required for the `cuda` feature but too heavy/box-specific to auto-install.
      if have nvcc; then
        log "  CUDA toolkit present ($(nvcc --version | grep -o 'release [0-9.]*' | head -1))"
      else
        warn "  nvcc not found — install the CUDA toolkit to build with --features llama on Linux:"
        warn "    https://developer.nvidia.com/cuda-downloads  (then ensure nvcc is on PATH)"
      fi
      ;;
    *) warn "  unknown OS '$OS' — install cmake + a C/C++ toolchain + libclang manually" ;;
  esac
}

# Convert a Hugging Face safetensors repo → GGUF at MODEL_OUTTYPE (bf16) using
# llama.cpp's convert_hf_to_gguf.py. Downloads the HF snapshot and the convert
# script into a cache, in a throwaway venv. Best-effort: needs python3 + network.
convert_hf_to_gguf() {
  local cache="${MODEL_DIR}/.cache"
  local snap="${cache}/hf/${MODEL_HF_REPO//\//_}"
  local llama_src="${cache}/llama.cpp"
  local venv="${cache}/venv"
  mkdir -p "$cache"

  if ! have python3; then
    warn "  python3 required to convert BF16 safetensors → GGUF. Install it, or set"
    warn "  MODEL_URL to a pre-built .gguf. Skipping."
    return 1
  fi

  # One venv for huggingface_hub (download) + the convert script's deps.
  [[ -d "$venv" ]] || python3 -m venv "$venv"
  # shellcheck disable=SC1091
  source "${venv}/bin/activate"
  log "  installing convert deps (huggingface_hub + llama.cpp requirements) into venv"
  pip install -q --upgrade pip huggingface_hub >/dev/null

  # llama.cpp source only for convert_hf_to_gguf.py + its requirements.
  if [[ ! -d "$llama_src/.git" ]]; then
    log "  cloning llama.cpp (convert script)"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$llama_src"
  fi
  pip install -q -r "${llama_src}/requirements/requirements-convert_hf_to_gguf.txt" >/dev/null

  # Snapshot the HF repo (safetensors + tokenizer/config).
  log "  downloading HF snapshot ${MODEL_HF_REPO} (BF16 weights — several GB)"
  python3 - "$MODEL_HF_REPO" "$snap" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2],
                  allow_patterns=["*.safetensors","*.json","*.txt","*.model","tokenizer*"])
PY

  log "  converting → ${MODEL_OUTTYPE} GGUF"
  python3 "${llama_src}/convert_hf_to_gguf.py" "$snap" --outtype "$MODEL_OUTTYPE" --outfile "$MODEL_DEST"
  deactivate
}

# Download a URL to a temp file, verify SHA-256 (if MODEL_SHA256 set), then move
# into MODEL_DEST atomically. A partial/corrupt/tampered download never lands where
# the FFI backend would load it.
download_verified() {
  local url="$1"
  local tmp="${MODEL_DEST}.part"
  curl -fL --progress-bar "$url" -o "$tmp" || { rm -f "$tmp"; return 1; }

  if [[ -n "$MODEL_SHA256" ]]; then
    local got
    if have sha256sum; then got="$(sha256sum "$tmp" | cut -d' ' -f1)"
    elif have shasum;    then got="$(shasum -a 256 "$tmp" | cut -d' ' -f1)"
    else warn "  no sha256sum/shasum — cannot verify MODEL_SHA256"; rm -f "$tmp"; return 1; fi
    if [[ "$got" != "$MODEL_SHA256" ]]; then
      warn "  SHA-256 mismatch! expected ${MODEL_SHA256}, got ${got} — refusing to install"
      rm -f "$tmp"; return 1
    fi
    log "  SHA-256 verified"
  else
    warn "  MODEL_SHA256 not set — installing UNVERIFIED native-library input. Pin it:"
    warn "    MODEL_SHA256=$( { have sha256sum && sha256sum "$tmp" || shasum -a 256 "$tmp"; } | cut -d' ' -f1)"
  fi
  mv "$tmp" "$MODEL_DEST"
}

# Place the GGUF model where the config expects it.
setup_model() {
  log "GGUF model → ${MODEL_DEST}"
  if [[ -f "$MODEL_DEST" ]]; then
    log "  already present ($(du -h "$MODEL_DEST" | cut -f1)) — skipping"
    return
  fi
  mkdir -p "$MODEL_DIR"

  if [[ -n "$MODEL_URL" ]]; then
    log "  downloading pre-built GGUF from MODEL_URL"
    download_verified "$MODEL_URL" || return
  elif [[ -n "$MODEL_HF_FILE" ]]; then
    # A single pre-built .gguf published in the HF repo → download directly, no convert.
    log "  downloading ${MODEL_HF_REPO}/${MODEL_HF_FILE} from Hugging Face"
    download_verified "https://huggingface.co/${MODEL_HF_REPO}/resolve/main/${MODEL_HF_FILE}" || return
  elif [[ -n "$MODEL_HF_REPO" ]]; then
    # Default: Qwen3.5-2B is BF16 safetensors → convert to a BF16 GGUF.
    convert_hf_to_gguf || {
      warn "  conversion skipped/failed — provide MODEL_URL (pre-built .gguf) or convert manually:"
      warn "    git clone --depth 1 https://github.com/ggml-org/llama.cpp && \\"
      warn "    pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt && \\"
      warn "    huggingface-cli download ${MODEL_HF_REPO} --local-dir ./hf-${MODEL_HF_REPO##*/} && \\"
      warn "    python llama.cpp/convert_hf_to_gguf.py ./hf-${MODEL_HF_REPO##*/} --outtype ${MODEL_OUTTYPE} --outfile ${MODEL_DEST}"
      return
    }
  else
    warn "  no model source — set MODEL_URL or MODEL_HF_REPO, or drop a .gguf at ${MODEL_DEST}."
    return
  fi
  log "  model ready: $(du -h "$MODEL_DEST" | cut -f1)"
}

# Print how to build + run against the real backend once the toolchain + model are in place.
print_llama_usage() {
  local accel; [[ "$OS" == "Darwin" ]] && accel="Metal" || accel="CUDA"
  cat <<EOF
  [next] Build + run real inference (${accel}):
           cargo build --manifest-path services/Cargo.toml -p inference-runtime --features llama
           make run/inference        # config model.gguf_path must point at ${MODEL_DEST#${REPO_ROOT}/}
         The default build (no --features llama) keeps the mock backend for dev/tests.
EOF
}

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
  # Step 0 runs everywhere: the llama.cpp backend + Metal work on the macOS dev box.
  setup_llama_toolchain
  setup_model

  # Steps 1–3 are the Linux perf story (pinning / BOLT / jemalloc).
  if [[ "$OS" == "Linux" ]]; then
    setup_cpu_pinning
    setup_bolt
    setup_jemalloc
    log "done. Load the allocator into your shell with:  source script/runtime-env.sh"
  else
    warn "CPU pinning, BOLT and jemalloc are Linux-only — skipped on ${OS} (dev box)."
  fi

  print_llama_usage
}

main "$@"
