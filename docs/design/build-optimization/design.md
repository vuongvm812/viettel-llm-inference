# Design — Build Optimization (PGO → LTO → BOLT)

Squeeze the **Rust binary** using `trace-round1.jsonl` as representative training data. Order:
LTO is a build-profile setting; PGO wraps a two-pass build; BOLT is a post-link pass on the
PGO+LTO binary. Optimizes *our* code; `libllama` is optimized separately (P7).

## 1. Cargo release profile (LTO + friends)

`services/Cargo.toml` (workspace) or the `inference-runtime` crate:

```toml
[profile.release]
lto           = "fat"       # whole-program inlining across crates (incl. disruptor)
codegen-units = 1           # max optimization, no parallel-codegen boundaries
panic         = "abort"     # no unwind tables on the hot path
opt-level     = 3
strip         = false       # keep symbols; BOLT + perf need them (strip AFTER bolt if desired)
```

`.cargo/config.toml` (target-native codegen):

```toml
[build]
rustflags = ["-C", "target-cpu=native"]     # deploy box CPU; set explicit arch for portability
```

## 2. PGO (profile-guided optimization) — two-pass

The replay harness (`benchmark/design.md`) is the profiling workload.

```bash
# Pass 1 — instrumented build
RUSTFLAGS="-Cprofile-generate=/tmp/pgo -Ctarget-cpu=native" \
  cargo build --release -p inference-runtime

# Run the representative workload → emits .profraw
./target/release/inference-runtime &        # start server
python bench/replay.py --target http://localhost:PORT --trace data/input/trace-round1.jsonl
# (drive enough load to cover hot paths; the 120-req trace is the minimum — loop it if thin)

# Merge
llvm-profdata merge -o /tmp/pgo/merged.profdata /tmp/pgo/*.profraw

# Pass 2 — optimized build using the profile
RUSTFLAGS="-Cprofile-use=/tmp/pgo/merged.profdata -Cllvm-args=-pgo-warn-missing-function -Ctarget-cpu=native" \
  cargo build --release -p inference-runtime
```

Use the same `llvm-profdata` version as the Rust toolchain's LLVM (`rustc -Vv` → LLVM version;
`rustup component add llvm-tools-preview` provides a matching `llvm-profdata`).

## 3. BOLT (binary optimization & layout tuning) — Linux/ELF only

BOLT reorders basic blocks/functions using real branch profiles. **ELF-only → runs on the
Linux deploy box, not macOS.** Apply to the PGO+LTO binary.

```bash
# Build must emit relocations for BOLT
RUSTFLAGS="-Cprofile-use=/tmp/pgo/merged.profdata -Clink-args=-Wl,--emit-relocs -Ctarget-cpu=native" \
  cargo build --release -p inference-runtime

# Collect branch profile with perf while replaying the trace
perf record -e cycles:u -j any,u -o /tmp/perf.data -- ./target/release/inference-runtime &
python bench/replay.py --target http://localhost:PORT --trace data/input/trace-round1.jsonl
# stop the server; then convert perf → BOLT profile
perf2bolt -p /tmp/perf.data -o /tmp/bolt.fdata ./target/release/inference-runtime

# Optimize
llvm-bolt ./target/release/inference-runtime -o ./target/release/inference-runtime.bolt \
  -data=/tmp/bolt.fdata \
  -reorder-blocks=ext-tstate -reorder-functions=hfsort \
  -split-functions -split-all-cold -dyno-stats
```

Alternative if `perf` LBR is unavailable (VMs/containers): BOLT **instrumentation** mode
(`llvm-bolt -instrument`) → run the trace → emit `.fdata` → re-optimize. Slower but no perf.

## 4. Orchestration: `build-pgo-bolt.sh`

One script chains the stages; skips BOLT on non-Linux:

```
1. cargo build --release  (instrumented)         # PGO gen
2. run server + replay trace                       # collect .profraw
3. llvm-profdata merge                             # → merged.profdata
4. cargo build --release  (profile-use, emit-relocs)
5. if Linux: perf record + replay → perf2bolt → llvm-bolt   # else stop at PGO+LTO
6. emit final binary (…​.bolt on Linux, plain PGO+LTO elsewhere)
```

Keep it a plain shell script (no build framework). Idempotent: clean `/tmp/pgo` and
`/tmp/perf.data` at the start.

## 5. Caveats / honesty

- **Measure, don't assume.** Each stage must beat the previous on the P5 benchmark metrics or
  it's not worth the build complexity — record before/after.
- **Profile coverage.** The 120-request trace may under-cover cold/error paths; loop the trace
  or add a warmup so PGO/BOLT see steady-state decode. Note what's uncovered (silent
  truncation of coverage reads as "fully optimized" when it isn't).
- **macOS.** PGO works; BOLT does not (Mach-O). CI/local on mac stops at PGO+LTO; the deploy
  pipeline on Linux adds BOLT.
- **Scope.** These passes optimize the Rust orchestration/sampling/ring hot paths, **not**
  `libllama`'s CUDA kernels (the bulk of GPU time). Expect gains on CPU-side latency (TTFT
  scheduling overhead, per-token dispatch), not on the GPU forward pass itself.
