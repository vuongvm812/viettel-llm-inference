# Development Guide — Build & Run Reference

This document covers every component in the stack and the exact commands to build, run, and
test each one. Run all `make` commands from the **repo root**, not from a round directory.
The default round is `round-1.2`; override with `ROUND=round-2` on any target.

---

## Component Map

```
vllm-fork base image          (Rust frontend + vLLM source patches)
    └── main image            (vtl plugin + vtl._C CUDA extensions baked in)
            └── server        (vllm-rs HTTP frontend + Python engine)
                    ├── vtl._C           (CUDA kernels — loaded at server start)
                    ├── vtl plugin       (Python patches — loaded at server start)
                    ├── vtl_sched wheel  (Rust KV scheduler — loaded at server start)
                    └── vllm-rs binary   (HTTP frontend — spawned by api_server.py)
```

Build dependency order:

```
[1] vllm-fork  ──►  [2] main image  ──►  [3] up / push / warm
```

Step [1] is only needed when changing `vtl/vllm_patches/` or the Rust frontend source.
For routine Python-patch or CUDA-kernel iteration, start at step [2].

---

## 1. vtl Plugin (Python patches)

**What it is:** The `vtl` package installed into the image via `pip`. Registers itself through
the `vllm.general_plugins` entry point so vLLM calls `vtl.plugin:register()` at startup.
Contains all Python-level monkey-patches (`vtl/patches/*.py`) and the Rust scheduler wheel
(`vtl-sched/`).

### Build

Built automatically inside `make build` — there is no separate step. The `pip install` in
the Dockerfile compiles `vtl._C` (CUDA extensions) and `vtl_sched` (Rust crate) together.

To build just the Python package locally (no GPU, no Docker — for `make check`):

```bash
cd round-1.2   # or round-2
pip install --no-build-isolation --no-deps -e .
```

Note: `vtl._C` requires CUDA/nvcc and a GPU to compile. On a Mac you can install the pure
Python parts (`--no-deps`) and run `make check`, which exercises all patches off-GPU.

### Run (self-checks, no GPU needed)

```bash
make check                                   # runs every patch's self-check + bench scripts
make check ROUND=round-2

# Individual patch:
python3 round-1.2/vtl/patches/sched_policy.py
python3 round-1.2/vtl/patches/decode_fastpath.py   # no vLLM import needed
```

### Key env gates

Every patch is independently killable:

| Env var | Default | Effect |
|---|---|---|
| `VTL_DISABLE=1` | off | Bypass the whole plugin; run stock vLLM |
| `VTL_ENABLE_<PATCH>=0` | on | Kill one patch while leaving the rest active |
| `VTL_DECODE_FASTPATH_SHADOW=1` | off | Run both fast and stock paths, log divergence |
| `VTL_NSTEP_MODE=eager` | graph | Disable CUDA graph capture for the N-step burst |

---

## 2. vtl._C (CUDA Extensions)

**What it is:** A compiled `.so` (`vtl/_C*.so`) containing the fused CUDA kernels: fused
RMSNorm+FP8-quant, fused SiLU-mul+FP8-quant, the short-conv decode megakernel, and their
torch op registrations. Built by `setup.py` / `pyproject.toml` using nvcc.

### Build (inside Docker — happens automatically)

```bash
# Happens inside `make build`. Controlled by:
make build CUDA_ARCHS="9.0+PTX"    # H200-only, leaner binary
make build CUDA_ARCHS="8.0;8.6;8.9;9.0+PTX"  # default: dev + H200
make build MAX_JOBS=8               # more parallel nvcc processes (needs RAM)
```

### Build (directly on GPU box)

```bash
cd round-1.2   # or round-2
TORCH_CUDA_ARCH_LIST="9.0+PTX" MAX_JOBS=4 \
    pip install --no-cache-dir --no-build-isolation --no-deps .
```

### Verify

```bash
# Inside the image (needs a GPU):
python3 -c "import vtl._C; print('vtl._C ok')"

# Check which SM cubins were compiled in:
SO=$(python3 -c "import importlib.util as u; print(u.find_spec('vtl._C').origin)")
cuobjdump --list-elf "$SO"
cuobjdump --list-ptx "$SO"   # PTX = JIT fallback for future SM archs
```

### Run (kernel tests — needs GPU)

```bash
make test-kernel               # correctness: vtl kernel vs stock vLLM reference
make bench-kernel              # microbenchmark at trace shapes
make debug-kernel              # memory-fault isolation (VTL_KERNEL_SYNC=1)
make debug-kernel T=test_rms   # one test only
```

---

## 3. vtl-sched (Rust KV Scheduler Crate)

**What it is:** A ~9K-line Rust crate (`vtl-sched/`) that implements the KV block pool,
prefix cache, radix tree index, and schedule loop. Shipped as a `vtl_sched` Python wheel.
Used by `vtl/patches/rust_sched.py` when `VTL_ENABLE_RUST_SCHED=1`.

### Build (inside Docker — happens automatically with the main image)

The `pip install /src` step in the Dockerfile also builds the `vtl_sched` wheel via maturin.

### Build (directly on GPU box)

```bash
cd round-1.2/vtl-sched   # or round-2/vtl-sched
pip install maturin
maturin develop --release
```

Or build a wheel to install elsewhere:

```bash
maturin build --release -o /tmp/vtl-sched-dist/
pip install /tmp/vtl-sched-dist/vtl_sched-*.whl
```

### Run (self-check, no GPU needed)

```bash
cd round-1.2/vtl-sched
cargo test
```

### Key env gates

| Env var | Effect |
|---|---|
| `VTL_ENABLE_RUST_SCHED=1` | Install the patch at all |
| `VTL_RUST_SCHED=1` | Make Rust authoritative for the KV manager surface |
| `VTL_RUST_SCHED_FULL=1` | Also run the Rust `schedule()` loop (implies `RUST_SCHED`) |
| `VTL_RUST_SCHED_SHADOW=1` | Run Rust alongside Python every call and compare (safe for bench) |
| `VTL_RUST_SCHED_REQUIRE=1` | Turn a silent refusal into a boot failure (use in bench, NOT submit) |

---

## 4. vllm-rs (Rust HTTP Frontend)

**What it is:** The `vllm-rs` binary that replaces Python's uvicorn as the HTTP server when
`VLLM_USE_RUST_FRONTEND=1`. Built from the vLLM v0.25.0 Rust workspace with vtl's
optimizations applied (sonic_rs JSON parser, fat-LTO, optional PGO).

### Build (via Docker — the `vllm-fork` target)

Only needed when changing `vtl/vllm_patches/rust-frontend/*.patch` or tuning the Rust
release profile. On routine patch iteration, skip this.

```bash
make vllm-fork                   # plain optimized (fat-LTO)
make vllm-fork VLLM_RS_PGO=1    # + profile-guided optimization
make vllm-fork PUSH=1            # build + push to registry
# After pushing, pin the new digest in Makefile's VLLM_FORK_TAG
```

### Build (directly on H200 box — no Docker)

```bash
# Prerequisites
sudo apt-get install -y build-essential perl pkg-config curl ca-certificates unzip git
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --default-toolchain 1.95 --profile minimal
rustup component add llvm-tools-preview
source ~/.cargo/env

# protoc (Ubuntu 22.04's version is too old)
PROTOC_VERSION=28.3
curl -fsSL -o /tmp/protoc.zip \
  "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-x86_64.zip"
sudo unzip -o /tmp/protoc.zip -d /usr/local 'bin/protoc' 'include/*'
export PROTOC=/usr/local/bin/protoc

# Fetch the vLLM Rust workspace (sparse clone)
git clone --depth 1 --branch v0.25.0 --filter=blob:none --sparse \
  https://github.com/vllm-project/vllm /tmp/vllm-src
git -C /tmp/vllm-src sparse-checkout set rust
cp -r /tmp/vllm-src/rust /tmp/vllm-rust

# Apply vtl Rust patches
for p in round-1.2/vtl/vllm_patches/rust-frontend/*.patch; do
  patch -p1 -d /tmp/vllm-rust < "$p"
done

# Build (native = AVX-512 on H200)
cd /tmp/vllm-rust
RUSTFLAGS="-Ctarget-cpu=native" \
  cargo build --release --bin vllm-rs --features native-tls-vendored

# Install into the vLLM site-packages
SITE=$(python3 -c "import os,vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))")
install -m0755 /tmp/vllm-rust/target/release/vllm-rs "$SITE/vllm/vllm-rs"
"$SITE/vllm/vllm-rs" --help   # sanity check
```

### PGO build (on H200 — additional step after plain build)

```bash
# Instrumented build
RUSTFLAGS="-Ctarget-cpu=native -Cprofile-generate=/tmp/pgo-data" \
  cargo build --release --bin vllm-rs --features native-tls-vendored

# Training run (replays trace against the mock engine, no GPU needed)
bash round-1.2/vtl/pgo_train.sh

# Merge profiles + final PGO build
llvm-profdata merge -output=/tmp/pgo-data/merged.profdata /tmp/pgo-data/*.profraw
RUSTFLAGS="-Ctarget-cpu=native -Cprofile-use=/tmp/pgo-data/merged.profdata" \
  cargo build --release --bin vllm-rs --features native-tls-vendored
```

---

## 5. Main Image (vtl plugin + CUDA kernels baked in)

**What it is:** The Docker image submitted to the judge. Built FROM the `vllm-fork` base
image, it pip-installs the vtl package (compiling vtl._C and vtl_sched), applies the
api_server Rust-frontend patch, and bakes torch.compile/Triton/FlashInfer caches.

### Build

```bash
make build                              # loads as unseenablefuture/awesome-badger:dev
make build ROUND=round-2
make build CUDA_ARCHS="9.0+PTX"        # H200-only, submission build
make build CUDA_ARCHS="9.0+PTX" MAX_JOBS=8
```

### Push to registry (produces the digest to pin in compose)

```bash
make push                               # prints digest after push
make push ROUND=round-2
# Copy the printed sha256:... into docker-compose.yaml's image: line
```

### Post-build verify (boots the image)

```bash
make up                   # build + compose up (requires GPU)
make verify               # post-boot assertions against live container logs
make down                 # tear down
```

---

## 6. Server (full stack running)

### Via Docker Compose (recommended — identical to judge environment)

```bash
make up                   # build + docker compose up (local dev)
make up ROUND=round-2

# Wait for healthy:
docker compose -f round-1.2/docker-compose-optimized.yaml \
               -f round-1.2/docker-compose.localtest.yaml \
               -f round-1.2/docker-compose.cpucap.yaml ps

make down                 # tear down + remove volumes
```

The `docker-compose.localtest.yaml` overlay adds the `/model` bind-mount (the judge
provides the model directly; locally you need the weights).

### Via raw Python (directly on Ubuntu H200 — no Docker)

Prerequisites: jemalloc, vLLM v0.25.0, vtl installed (see components above).

```bash
sudo apt-get install -y libjemalloc2

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
MALLOC_CONF="background_thread:false,dirty_decay_ms:100,muzzy_decay_ms:100,\
tcache:true,lg_tcache_max:16,percpu_arena:percpu,narenas:3,metadata_thp:always" \
PYTHONMALLOC=malloc \
VLLM_PLUGINS=vtl \
VLLM_LOGGING_LEVEL=WARNING \
VTL_DISABLE=0 \
OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OMP_WAIT_POLICY=PASSIVE \
VTL_GIL_SWITCH_INTERVAL=0.0002 \
VLLM_USE_RUST_FRONTEND=1 \
TOKIO_WORKER_THREADS=2 VLLM_RS_ZMQ_WORKER_THREADS=1 VLLM_RS_REQUEST_WORKER_THREADS=2 \
VLLM_RS_DISABLE_HTTP_TRACE=1 \
VTL_ENABLE_FP8=1 VTL_FP8_CHANNELWISE=1 VTL_FP8_IGNORE=lm_head \
VTL_ENABLE_RMS_NORM_QUANT=1 VTL_ENABLE_DYNAMIC_PER_TOKEN_QUANT=1 \
VTL_ENABLE_SILU_MUL_QUANT=1 VTL_SILU_MUL_FUSION=1 \
VTL_ENABLE_QK_NORM_ROPE=1 VTL_ENABLE_SHORTCONV_QUANT=1 \
VTL_ENABLE_MUL_QUANT=1 VTL_ENABLE_BCX_CONV_GATE=1 \
VTL_ENABLE_KV_CACHE_MANAGER=1 VTL_ENABLE_SCHED_POLICY=1 VTL_SCHED_AGING_ALPHA=0 \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256" \
python3 -m vllm.entrypoints.openai.api_server \
  -O3 \
  --model=/path/to/model \
  --served-model-name=LFM2.5-1.2B-Instruct \
  --host=0.0.0.0 --port=8000 \
  --max-model-len=32768 \
  --gpu-memory-utilization=0.85 \
  --tensor-parallel-size=1 \
  --enable-prefix-caching \
  --mamba_cache_mode=align \
  --enable-chunked-prefill \
  --max-num-seqs=32 \
  --max-num-batched-tokens=4096 \
  --quantization=vtl_fp8 \
  --kv-cache-dtype=fp8_e4m3 \
  --disable-log-stats \
  --uvicorn-log-level=warning \
  '--compilation-config={"pass_config":{"fuse_norm_quant":true,"fuse_act_quant":true},"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,16,32]}'
```

### Startup timeline (cold boot, no compile cache)

| Stage | Time | What happens |
|---|---|---|
| 0–5s | Plugin registers, all patches apply |
| 5–30s | Model weights load, FP8 quantization |
| 30s–3min | `torch.compile` + CUDA graph capture (first boot only) |
| ~10s after `make warm` | Same as above, but caches are warm |

### Warm the compile caches (do this once per image)

```bash
make warm              # boot → replay trace → copy caches back → rebuild
make warm ROUND=round-2
```

---

## 7. Benchmarking (needs a running server)

```bash
# Open-loop + closed-loop sweep
make bench

# Single open-loop run
cd round-1.2
python3 bench/replay.py --target http://localhost:8000 \
  --trace data/input/trace-round2.jsonl --out bench-open.json

# Closed-loop at fixed concurrency
python3 bench/replay.py --target http://localhost:8000 \
  --trace data/input/trace-round2.jsonl --closed-loop 8 --out bench-closed-8.json

# Compute ERS score from results
python3 bench/metrics.py bench-open.json

# Post-boot assertions (check all patches loaded correctly)
make verify
```

---

## 8. Profiling (needs GPU + running server)

```bash
make profile           # boots with torch profiler, drives trace, prints kernel cost table
make test-kernel       # kernel correctness vs stock vLLM
make bench-kernel      # kernel microbenchmark at trace shapes
make debug-kernel      # memory-fault isolation (VTL_KERNEL_SYNC=1)
```

---

## 9. Submission Checklist

Before running `make push` and submitting:

1. `make check` passes (no GPU needed, runs off-box)
2. `make test-kernel` passes (needs GPU)
3. `make warm` was run on the target hardware → caches baked into image
4. `make verify` against a live container shows all OK lines
5. **Hard-fail flags are OFF** in the submitted compose:
   - `VTL_RUST_RUNNER_REQUIRE=0` (not `1` — a refusal is score-0)
   - `VTL_RUST_SCHED_REQUIRE` absent or `0`
6. `docker-compose.yaml` has no `${VAR}` interpolation (`grep -F '${''{' docker-compose.yaml` prints nothing)
7. Image digest in `docker-compose.yaml` matches what `make push` printed

---

## 10. A/B Protocol

When evaluating a change:

1. Run `make bench` **≥3 boots per arm** to bracket the noise floor (~0.5ms TPOT variance)
2. Use `bench/sweep_report.py` to compare arms with the noise floor included
3. Kill exactly one env var at a time — do not change multiple things between arms
4. Record the verdict as a comment in `docker-compose.yaml` (see existing comments for examples)

Any change that scores within noise should be left OFF in the submission compose; the
composition rule is "only keep what wins clearly."
