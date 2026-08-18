# round-2 handoff

A model-agnostic vLLM optimization workspace. **No model is baked in.** Everything here
either matches on a vLLM class/graph pattern or reads the loaded config at runtime, so the
stack applies unchanged to whatever gets mounted at `/model`.

Three things to read in order: [what is in here](#1-components), [how to prepare the
host](#2-host-os-setup), [how to plug in a model and a kernel](#3-adding-a-model) /
[kernel](#4-adding-a-custom-cuda-kernel). What the judges actually run and score is
[§6](#6-official-grading-workload--scoring-round-2-btc-spec) — read it before tuning
anything, because both the workload and the scoring band changed from round 1.

Run everything from the **repo root**, not from `round-2/`. `round-2` is the default
`ROUND`, so `make up` means `make up ROUND=round-2`.

---

## 1. Components

### 1.1 How the code gets into vLLM

There is exactly one integration seam: the `vllm.general_plugins` entry point
`vtl.plugin:register`, declared in `pyproject.toml`. vLLM calls it once per process (API
server, engine core, every worker) before it does real work.

This is why **the plugin must be pip-installed, not bind-mounted** — the entry point lives
in the `.dist-info` that only a real install produces. A mounted `vtl/` is importable and
silently never loads.

`register()` imports `vtl.patches`, which imports each module in `_MODULES` order. Each
module registers a zero-arg callable with `@register_patch(name, default=...)`, and
`apply_all()` runs the enabled ones.

Three invariants hold throughout, and they are the reason the submission is hard to kill:

- **Never raises.** A patch that throws is logged and skipped. A broken patch degrades to
  stock vLLM; it never takes the server down.
- **Idempotent.** vLLM may call `register()` more than once.
- **Import-safe without vLLM.** `python3 -c "import vtl.patches"` works on a bare host,
  which is what makes `make check` possible off-box.

`VTL_DISABLE=1` bypasses the whole plugin — the A/B control for "is any of this helping?".

### 1.2 Patch modules (`vtl/patches/`)

Apply order matters and is documented in `vtl/patches/__init__.py`. Every one is gated by
`VTL_ENABLE_<NAME>`, where `<NAME>` is the uppercased **registry** name — which is not
always the module name. The two that differ are the ones you are most likely to want:
`quant_fp8` → `VTL_ENABLE_FP8`, `quant_w4a8` → `VTL_ENABLE_W4A8`. Everywhere else they
match. `python3 vtl/registry.py` and the `@register_patch(...)` line in each module are
the authority.

| Module | Default | What it does |
|---|---|---|
| `quant_fp8` | on | fp8-e4m3 weights + activations via the vLLM quant-config seam (`vtl_fp8`) |
| `quant_w4a8` | on | int4 weights + fp8 acts (`vtl_w4a8`); falls back to fp8 per-layer on any rejected shape |
| `rms_norm_quant` | on | fused norm→fp8-quant CUDA kernel, inserted by a torch.compile fusion pass |
| `dynamic_per_token_quant` | on | standalone activation quant, for inputs no norm precedes |
| `silu_mul_quant` | on | fused SwiGLU→fp8-quant, same fusion-pass mechanism |
| `kv_cache_manager` | on | adds `plan_request`/`free_blocks` signals to the KV manager |
| `sched_policy` | on | cache-aware shortest-job-first reorder of the waiting queue |
| `rust_sched` | **off** | the Rust scheduler core — see §1.4 |
| `greedy_sampler` | on | argmax fast path for plain greedy steps |
| `step0_eos_ban` | on | masks EOS out of each request's **first** sampled token |
| `decode_fastpath` | on | V2 runner: skips the dead metadata build on repeat pure-decode steps |
| `nstep_decode` | on | N decode iterations per engine step (armed by `rust_sched`'s commit) |
| `shm_ipc` | **off** | iceoryx2 zero-copy shm data plane for frontend↔EngineCore |
| `profiler` | **off** | torch-profiler capture inside the worker |
| `l2_persist` | on | boot probe of the L2 set-aside + opt-in persisting window |
| `megakernel_probe` | on | read-only go/no-go for a cooperative-grid decode megakernel |

`step0_eos_ban` is worth knowing about: an int4 `lm_head` can argmax straight to EOS at
step 0, producing an empty stream — a request that scores zero rather than badly. It is
cheap insurance, not a micro-optimization.

**`lm_head_quant`** is *not* in that table and *not* a registered patch — it is a helper
`quant_w4a8` imports, driven by `VTL_LM_HEAD_QUANT` = `int4` (default) | `fp8` | `off`. It
covers the output head, which is not a `LinearBase` and so never reaches the quant configs
at all. On a large-vocab model that head is a big share of per-step decode weight traffic,
and every one of its failure modes leaves it silently bf16 — so `make verify` checks the
outcome explicitly.

### 1.3 CUDA kernels

| Path | Extension | Notes |
|---|---|---|
| `vtl/csrc/rms_norm_quant.cu` | `vtl._C` | fused residual-add + RMSNorm + per-token fp8 quant |
| `vtl/csrc/dynamic_per_token_quant.cu` | `vtl._C` | standalone per-token fp8 quant |
| `vtl/csrc/silu_mul_quant.cu` | `vtl._C` | fused SwiGLU + fp8 quant |
| `vtl/csrc/torch_bindings.cpp` | `vtl._C` | op schemas + dispatch registration |
| `vtl/csrc/w4a8/w4a8_mm_v2.cu` | `vtl._C_w4a8` | extra CUTLASS W4A8 schedules; **sm_90a only**, no PTX |
| `vtl/kernels/*.cu` | *(none — NVRTC)* | compiled at model load, see §4.2 |

`vtl._C_w4a8` is optional: it needs CUTLASS headers, fetched best-effort at build time. No
headers → the extension is absent → `VTL_W4A8_SCHEDULE_V2` is inert and the stock W4A8
kernel runs. That is a working server, just a less tuned one.

### 1.4 Rust

Two independent surfaces, both **model-agnostic**.

**`vtl-sched/`** — a ~9k-line Rust crate (KV block pool, prefix cache, radix index,
scheduler core, token store, update loop) shipped as a `vtl_sched` wheel. It is
**spec-driven, not model-driven**: it reads `kv_cache_config.kv_cache_groups` at runtime.

> ⚠️ **The one sharp edge in this workspace.** `Kind` has exactly two variants —
> `FullAttention` and `Mamba`. Anything else (sliding-window, chunked-local,
> cross-attention) makes it log one line and hand back to stock vLLM. Sliding-window
> attention is **common** in current models (Gemma, Mistral, GPT-OSS, Qwen3-Next), and on
> such a model this entire 9k-line component silently switches itself off — which looks
> identical, in every latency number, to a component that ran and did not help.
>
> **Set `VTL_RUST_SCHED_REQUIRE=1` in every bench and CI run.** It turns that refusal into
> a boot failure. Leave it unset in the submission, where serving-but-slower beats not
> serving. `make verify` also reports engagement.
>
> Supporting a new kind = a new `Kind` variant in `vtl-sched/src/single_type.rs` plus its
> arm in `rust_sched.py::build_config`. The affected functions are `num_skipped_tokens`,
> `remove_skipped_blocks`, `find_longest_cache_hit`.

**`vtl/vllm_patches/rust-frontend/*.patch`** — five patches into vLLM's own `vllm-rs`
frontend binary: sonic-rs body decode, shm IPC transport, per-token streaming, an HTTP
trace toggle, and PGO pacing for the mock engine. Applied by `Dockerfile.vllm-fork`.

### 1.5 vLLM source patches (`vtl/vllm_patches/v0.25.0/`)

Applied to site-packages by `Dockerfile.vllm-fork`. All five are engine plumbing — none
touches a model definition. **A patch that targets one model's module belongs in
`reference/`, not here.**

`api_server_rust_frontend` (also applied by the main Dockerfile so it lands on stock),
`hotpath_microopt`, `v2_greedy_sampler`, `rejection_sampler`, `mamba_hybrid_postprocess`.

Regenerate with `vtl/vllm_patches/gen.sh`. **Except `hotpath_microopt`** — it spans two
files and `gen()` is a single-file diff, so running it through `gen` would silently
truncate the patch.

### 1.6 Bench + tooling (`bench/`)

`replay.py` (open/closed-loop SSE replayer), `metrics.py`, `compare.py`,
`sweep_report.py` (reads the noise floor off each run's own boot spread),
`eval_quality.py` (**the accuracy gate for any quantization change**),
`trace_stats.py`, `profile_trace.py`, `build_trace_round2.py`, `build_grading_spec.py`,
`aiperf_adapter.py` (converts a `make bench-aiperf` aiperf run into the repo schema — see
[§6](#6-official-grading-workload--scoring-round-2-btc-spec)),
plus `test_*.py` kernel/parity tests run by `make test-kernel`.

### 1.7 Compose overlays

`docker-compose.yaml` is the **submission artifact** and the single source of truth for
every serve flag and env var. Overlays carry only their differences; later `-f` wins.

| Overlay | Purpose |
|---|---|
| `docker-compose-optimized.yaml` | local `:dev` image tag |
| `docker-compose.localtest.yaml` | `build:` + mounts `../hf-model` at `/model` |
| `docker-compose.cpucap.yaml` | simulates the judge's CPU/memory cap |
| `docker-compose.nvrtc-dev.yaml` | NVRTC kernel dev loop (§4.2) |
| `docker-compose.profile.yaml` | arms the torch profiler |
| `docker-compose.ci-bench.yaml` | pinned digest + model mount for CI |

> **No shell-variable interpolation in `docker-compose.yaml`, ever.** It is submitted
> as-is; an unset variable is a boot failure, i.e. score 0. Literals only.
> Check: `grep -F '$''{' round-2/docker-compose.yaml` must print nothing.

### 1.8 Environment flags

75 env vars ship in compose. Grouped by what they control:

**Kill switches** — `VTL_DISABLE` (bypass everything), `VTL_ENABLE_<NAME>` for each patch
in §1.2, `VTL_SKIP_EXT=1` (ignore `vtl._C`, run stock kernels — used by `make test-kernel`).

**Quantization** — `VTL_W4A8_IGNORE`, `VTL_FP8_IGNORE`, `VTL_LM_HEAD_QUANT`,
`VTL_W4A8_SCHEDULE`, `VTL_W4A8_SCHEDULE_V2`, `VTL_W4A8_SCHEDULE_V2_PREFILL`,
`VTL_W4A8_V2_MTHRESH`, `VTL_W4A8_V2_PREFILL_MAX`, `VTL_W4A8_SM_COUNT`,
`VTL_W4A8_EXPECT_SHAPES` (opt-in allowlist; see §3.4).

**V2 model runner** — `VLLM_USE_V2_MODEL_RUNNER`, `VTL_V2_GREEDY_FASTPATH`,
`VTL_V2_GREEDY_ARGMAX_KERNEL`, `VTL_V2_FUSED_POSTPROCESS`, `VTL_V2_FORCE_CUTLASS_FP8`.

**Rust scheduler** — `VTL_ENABLE_RUST_SCHED` plus a mode flag (`VTL_RUST_SCHED` /
`VTL_RUST_SCHED_FULL`); both are required. Sub-features:
`_RADIX`, `_TABLE`, `_SPEC`, `_R8`, `_R9`, `_TOKSTORE`, `_UFO`, `VTL_RUST_HASHER`.
Each rung used to carry a `_SHADOW` twin that kept Python authoritative and logged
divergence; those soak arms were removed once the port was proved — `bench/` and the
crate's own tests are the guard now. Plus **`VTL_RUST_SCHED_REQUIRE`** (§1.4).

**N-step decode** — `VTL_ENABLE_NSTEP_DECODE`, `VTL_NSTEP`, `VTL_NSTEP_N`,
`VTL_NSTEP_MODE`, `VTL_NSTEP_QUEUE_EMPTY_ONLY`, `VTL_NSTEP_FOLD_T1`, `VTL_NSTEP_UNROLL`,
`VTL_SAMPLE_IN_GRAPH`, `VTL_STREAM_PER_TOKEN`.

**Rust frontend** — `VLLM_USE_RUST_FRONTEND`, `TOKIO_WORKER_THREADS`,
`VLLM_RS_ZMQ_WORKER_THREADS`, `VLLM_RS_REQUEST_WORKER_THREADS`,
`VLLM_RS_DISABLE_HTTP_TRACE`, `VLLM_HTTP_TIMEOUT_KEEP_ALIVE`.

**shm IPC** — `VTL_ENABLE_SHM_IPC`, `VTL_SHM_IPC`, `VTL_SHM_IPC_RAW`,
`VTL_SHM_IPC_RUST_PUB`.

**NVRTC** — `VTL_NVRTC` (default **off**), `VTL_NVRTC_SRC`, `VTL_NVRTC_CACHE`,
`VTL_NVRTC_ARCH`.

**Warm-up** — `VTL_WARMUP_CONCURRENCY`, `VTL_DISABLE_WARMUP`, `VTL_WARMUP_MODEL`
(override; normally discovered from `/v1/models`), `VTL_WARMUP_PORT`,
`VTL_WARMUP_TRACE_FILE`, `VTL_WARMUP_SENTINEL`, `VTL_WARMUP_POST_TIMEOUT`.

**Host/CPU** — `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OMP_WAIT_POLICY`, `OMP_PROC_BIND`,
`OMP_PLACES`, plus `LD_PRELOAD`/`MALLOC_CONF` (jemalloc) set in the Dockerfile.

**Misc** — `VTL_LOG_LEVEL` (the `vllm.vtl` subtree gets its own handler, independent of
`VLLM_LOGGING_LEVEL`), `VTL_HOTPATH_MICROOPT`, `VTL_L2_PERSIST(+_BYTES,_HIT_RATIO)`,
`VTL_UVA_POOL`, `VTL_KERNEL_SYNC` (debug: sync after every launch), `VTL_ENABLE_PROFILER`,
`VTL_PROFILE_DIR`, `VTL_PROFILE_STEPS`.

Three knobs are **deliberately commented out** in compose — `CUDA_DEVICE_MAX_CONNECTIONS`,
`CUDA_MODULE_LOADING=EAGER`, `PYTHONMALLOC`. Each is plausible and none is free; see §2.3.

---

## 2. Host / OS setup

### 2.1 Prerequisites

| Need | For |
|---|---|
| Docker + Compose v2, NVIDIA Container Toolkit | everything |
| An NVIDIA GPU (H200 / sm_90 is the target) | `up`, `bench`, `test-kernel`, `warm` |
| `python3` | `make check` (no GPU, no vLLM, no torch needed) |
| Rust 1.95 + maturin | only to build `vtl-sched` outside Docker |
| `gh` CLI | the `ci-*` targets |
| Model weights at `../hf-model` | local runs; the judge mounts their own at `/model` |

### 2.2 Host tuning (dev/bench box only)

```bash
make host-tune-show     # read-only, no root
sudo -v && make host-tune
# ... benchmark ...
make host-tune-reset    # ALWAYS
```

**This is not an optimization.** The judge runs `docker compose up` on their host and none
of these knobs are reachable from a compose file. It exists to make *our* measurements
mean something.

The dominant term is **GPU clocks**. Left on the default DVFS governor, an H200 ramps
between clock states in response to thermals and load, so two identical arms can differ by
more than the effect being measured. Locking clocks collapses that variance.

`scripts/host-tune.sh apply` sets: GPU persistence mode, locked SM + memory clocks, CPU
governor `performance`, THP `madvise` (not `always` — jemalloc already asks for hugepages
where it wants them via `metadata_thp:always`), `kernel.numa_balancing=0`,
`vm.swappiness=0`, `vm.zone_reclaim_mode=0`, stops `irqbalance`, and `tuned-adm
latency-performance` when available.

Every change is recorded to `/var/tmp/vtl-host-tune.state` and restored by `reset`. Run the
reset — a box left with clocks locked lies about power and thermals for whoever uses it
next. Each step degrades independently: no `nvidia-smi`, no cpufreq sysfs, a read-only
`/sys` — all skip with a message rather than failing.

> Not yet exercised on a real GPU host: only the `show` path and the root refusal have been
> run. Read its output on first use rather than trusting it.

### 2.3 Container-side settings (these DO ship)

Already in `docker-compose.yaml`: `seccomp=unconfined` + `apparmor=unconfined` (the default
seccomp profile puts a BPF filter in front of every syscall, and the decode loop is
syscall-heavy), `ipc: host`, `ulimits` (`memlock: -1` so pinned host staging buffers do not
fall back to pageable memory and halve H2D bandwidth; `nofile`, `rtprio`), and OMP thread
pinning. In the Dockerfile: jemalloc via `LD_PRELOAD` with a latency-tuned `MALLOC_CONF`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `CUDA_MODULE_LOADING=LAZY`.

⚠️ `MALLOC_CONF` uses `dirty_decay_ms:-1`, which **never returns pages to the OS**. Host RSS
only grows. If the judge caps the container's memory, validate peak RSS during `make bench`
before submitting or the container OOM-kills, which is unscoreable.

The three commented-out knobs need A/B'ing, not faith:

- `CUDA_DEVICE_MAX_CONNECTIONS=1` — fewer HW queues cuts launch overhead, but **serializes
  independent streams**, so it fights `VTL_STREAM_PER_TOKEN=1`. A/B the pair, not each alone.
- `CUDA_MODULE_LOADING=EAGER` — moves module load off the first launch onto boot, which the
  health-gated warm-up already hides. Costs boot time and VRAM for unused modules.
- `PYTHONMALLOC=malloc` — routes CPython through jemalloc instead of pymalloc. Genuinely
  50/50 on a many-small-objects decode path.

Turn on **one**, run `make bench` with ≥3 boots per arm, keep it only if the delta clears
the noise floor that run reports.

### 2.4 Daily commands

```bash
make check          # no GPU: registry, every patch's self-check, bench self-checks
make build          # buildx --load, linux/amd64
make up / make down
make verify         # greps the boot log: plugin loaded, quant registered, layers
                    # quantized, fusion count, jemalloc, rust-scheduler engagement
make warm           # boot -> 2 replay passes -> harvest compile caches -> rebuild
make bench          # open-loop replay + closed-loop at 1/8/32/128
make test-kernel    # GPU: kernel + parity tests, ours then stock (VTL_SKIP_EXT=1)
make profile        # torch profiler capture -> bucketed summary
make push           # prints the digest to pin in docker-compose.yaml
make vllm-fork      # rebuild the forked vLLM base (then re-pin in round.mk)
```

---

## 3. Adding a model

Nothing in `vtl/` needs to change to *serve* a new model. What changes is configuration,
plus re-validating the two things that are shape-sensitive.

### 3.1 Set the four literals

`docker-compose.yaml` carries a `FILL IN BEFORE SUBMITTING` header naming these.

1. **`--served-model-name=round2-model`** — a placeholder. It must match
   `bench/build_trace_round2.py --model-name` and the `"model"` field in
   `data/input/trace-round2.jsonl`, or every replayed request 404s.
2. **`--max-model-len`** — must equal the model's native context
   (`config.json: max_position_embeddings`). A tighter cap truncates the judge's
   long-context probe and gets flagged as truncation/dual-path cheating.
3. **`--quantization`** — see §3.4.
4. **Architecture-specific serve flags** — deliberately absent. A hybrid SSM model also
   needs `--mamba_cache_mode=align`; a model that does not implement it **fails at
   startup** if you pass it. Add only what the chosen model actually needs.

The warm-up healthcheck needs no change — it reads the served name from the server's own
`/v1/models`.

### 3.2 Check vLLM registers the architecture

```bash
docker run --rm vllm/vllm-openai:v0.25.0 python3 -c \
  "from vllm.model_executor.models.registry import ModelRegistry; print(ModelRegistry.get_supported_archs())"
```

If the arch is absent, v0.25.0 cannot load it and the base image pin has to move — which
means regenerating every patch in `vtl/vllm_patches/v0.25.0/` against the new tag.

### 3.3 Regenerate the trace

```bash
cd round-2
python3 bench/build_trace_round2.py --model-name <served-name> --model ../hf-model
python3 bench/trace_stats.py --model ../hf-model     # KV geometry comes from config.json
```

### 3.4 Re-validate what is shape-sensitive

**Quantization is not portable.** `--quantization=vtl_w4a8` carries over from the previous
round but is not validated against a new architecture. Two separate checks:

- **Accuracy** — `python3 bench/eval_quality.py` compares greedy outputs between two server
  configs. RTN int4 is not free and how much it costs depends on the model. Revert path:
  `vtl_fp8`, or drop the flag.
- **Speed** — TPOT ≈ max(host, gpu). If the host term dominates, the expected TPOT delta is
  ~0 and TTFT can get *worse* (prefill re-dequantizes weight tiles fp8 never does).

**Tile schedules.** Boot once and read the log line:

```
vtl: w4a8 observed shape keys: n2048k2048,n3072k2048,...
```

Those are the real `n<out>k<in>` keys for this model. Put them in
`VTL_W4A8_SCHEDULE_V2` / `VTL_W4A8_EXPECT_SHAPES` and sweep with
`make sweep-schedule`. Until `VTL_W4A8_EXPECT_SHAPES` is set, no key is warned about —
there is nothing to compare against.

### 3.5 Check the Rust scheduler actually engages

```bash
VTL_ENABLE_RUST_SCHED=1 VTL_RUST_SCHED=1 VTL_RUST_SCHED_REQUIRE=1 make up
make verify   # want: rust_sched: AUTHORITY mode active (N groups, ...)
```

A dense model gives `1 groups` (unitary path). If it instead reports
`unported kv cache spec ...`, `REQUIRE=1` makes that a hard boot failure and §1.4's
`Kind::SlidingWindow` work becomes the next task.

### 3.6 Re-pin the fork

Round-2's patch set differs from earlier rounds, so it needs its own fork image:

```bash
make vllm-fork PUSH=1              # prints the pushed digest
$EDITOR round-2/round.mk           # set VLLM_FORK_TAG, drop the VLLM_IMAGE stock default
```

Until then `round.mk` points at **stock** vLLM — correct, just without the frontend
optimizations.

### 3.7 Ship

```bash
make check && make build && make up && make verify
make bench                          # >=3 boots per arm
make warm                           # bake compile caches into the image
make push                           # pin the printed digest in docker-compose.yaml
```

---

## 4. Adding a custom CUDA kernel

Two paths. **Start with NVRTC** — you get a working kernel in seconds instead of a
20-minute rebuild per iteration, and you can promote it to AOT later if it earns it.

|  | AOT (`vtl._C`) | NVRTC (`vtl/kernels/`) |
|---|---|---|
| Edit → running | full image rebuild, ~20 min (nvcc × 4 arches) | container restart, ~15 s |
| Shapes | runtime values | **compile-time `-D` constants** |
| Headers | full torch/ATen/c10 | CUDA only |
| Compiled | at image build | at model load, cached to disk |
| Default | on | **off** (`VTL_NVRTC=1`) |

### 4.1 The AOT path

1. Write `vtl/csrc/my_kernel.cu`. Include `fp8_common.cuh` for the shared helpers
   (`kFp8Max`, vector traits, `block_reduce`, fp8 conversion, `kernel_sync_enabled`).
2. Declare it in `namespace vtl` in `vtl/csrc/torch_bindings.cpp`, `m.def()` its schema in
   `TORCH_LIBRARY(vllm_cuda, ...)`, and `m.impl()` it in the CUDA block. Predicates with no
   tensor arguments (occupancy, scratch size) cannot be dispatched by device key — register
   those as catch-alls in `TORCH_LIBRARY` itself.
3. Add the `.cu` to `setup.py`'s sources list.
4. Write `vtl/patches/my_kernel.py` with `@register_patch("my_kernel", default=False)`, and
   add the name to `_MODULES` in `vtl/patches/__init__.py`. **Default off** so a model it
   does not fit degrades to stock instead of failing.
5. Add `bench/test_my_kernel.py`. `make test-kernel` globs `/bench/test_*.py`, so it is
   picked up with no Makefile change, and runs it against ours *and* stock
   (`VTL_SKIP_EXT=1`) — agreeing with the same oracle is what proves the port.

⚠️ **`CUDA_ARCHS` is not optional.** Without it nvcc probes the build host (which has no
GPU) and guesses. A mismatch does **not** degrade gracefully: dlopen succeeds, the override
installs, and the first launch dies with `cudaErrorNoKernelImageForDevice`. Build for every
device you intend to run on; the Dockerfile asserts the embedded cubins.

### 4.2 The NVRTC path

Copy `vtl/kernels/rms_norm_quant.cu` — it is the worked template. Three rules:

1. **No torch/ATen headers.** No `torch::Tensor`, no `c10::BFloat16`, no `AT_DISPATCH`.
   Entry points take raw pointers; the caller passes `.data_ptr()`.
2. **No host code.** Only `__global__`/`__device__`. Launch config is computed in Python.
3. **Shapes arrive as `-D` macros.** That is the entire point — `HIDDEN` is a compile-time
   constant, so loops unroll, `HIDDEN/VEC` folds to a literal, and the compiler sizes
   registers for the real case. An AOT build cannot do this, because at AOT time the model
   is unknown.

Use `extern "C"` on the entry point so the symbol name is not mangled.

From Python:

```python
from vtl import nvrtc

k = nvrtc.compile_kernel("my_kernel", {"HIDDEN": hidden, "THREADS": 256})
if k is None:
    ...  # MANDATORY: fall back to the AOT kernel. Every failure path returns None.
else:
    k(grid=(num_tokens, 1, 1), block=(256, 1, 1),
      args=nvrtc.pack_args(out.data_ptr(), scales.data_ptr(), x.data_ptr(), ("f", eps)))
```

`pack_args` takes ints as device pointers (`0` for an optional tensor) and **needs the C
type stated** for scalars — `("f", 1e-6)`, `("i", n)`, `("u", n)`, `("l", n)`. Python's
int/float do not map to a unique C width, and guessing wrong silently misaligns every
argument after it.

Dev loop:

```bash
docker compose -f docker-compose.yaml -f docker-compose-optimized.yaml \
               -f docker-compose.localtest.yaml -f docker-compose.nvrtc-dev.yaml up -d
$EDITOR round-2/vtl/kernels/my_kernel.cu
docker compose ... restart model      # ~15 s
```

Cubins cache under `/opt/vtl/cache/nvrtc`, keyed by (source, defines, arch, toolkit
version) — so a changed kernel gets a different key and there is no stale-cubin trap, and
`make warm` bakes them into the image so the judge's boot pays **zero** JIT stall.

New `.cu` files are picked up automatically: `pyproject.toml` ships `vtl/kernels/*.cu` as
package data. Without that stanza the wheel omits them and every NVRTC kernel silently
degrades to AOT.

> The NVRTC compile/launch path has **not** been exercised on a GPU yet. The pure half
> (cache keys, gating, arg packing) runs in `make check`; the compile + numeric parity half
> is `bench/test_nvrtc.py` under `make test-kernel` and runs first on the H200.

### 4.3 Getting a kernel into the graph

Two mechanisms, and which one you need depends on the call site:

- **Override an op vLLM already emits** — register your impl for the same `_C::<op>` on the
  CUDA key (as `rms_norm_quant` does). Op identity is unchanged, so `FUSED_OPS`,
  `FixFunctionalizationPass`, the meta kernel and the torch.compile cache key all still
  work. Only the kernel behind the dispatch key changes.
- **Insert a new fused op** — register `vllm_cuda::<op>` plus a fake/meta kernel in Python,
  and add a torch.compile fusion pattern that rewrites the pair into it (as
  `silu_mul_quant` does). A model whose graph has no match is a silent no-op, not an error.

Verify it actually fired — a fusion pass that matches nothing looks exactly like success:

```bash
make verify     # includes "rms_quant_fusion Replaced N patterns"; N=0 means it never ran
```

### 4.4 Rules of thumb

- **Default off while unproven.** Every *new* kernel gets `@register_patch(..., default=False)`
  and an env gate. A model it does not fit must degrade, not fail. Flip the default to `True`
  only once the degrade path is the real safety story rather than the gate — which is what
  happened on 2026-08-17 to the seven Qwen3.5 kernel patches (`gdn_kernels`,
  `gdn_prefill_backend`, `nvrtc_block_quant`, `gdn_decode_step`, `w4a8_from_fp8`,
  `moe_decode_gemv`, `greedy_argmax`): each has a per-op fallback ladder
  (NVRTC → AOT → stock), and `greedy_argmax` additionally registers nothing until a boot
  parity gate matches `torch.argmax` bit for bit. Their `VTL_ENABLE_*` lines in
  `docker-compose.yaml` now restate the code default instead of overriding it.
- **Size the prize before writing it.** A wall-clock A/B beats a profiler's call count;
  small-call profiling overstates dispatch overhead badly.
- **Numerics are part of the contract.** The fp8 kernels round twice on purpose — matching
  `c10::BFloat16`'s homogeneous `operator*`. Making it "more accurate" in fp32 shifts amax
  and therefore the per-token scale. That is a different kernel, not a better one.
- **Leave one runnable check** that fails if the logic breaks.

---

## 5. Known state

**Fixed here, still present in round-1.2** —
`test_update_step_pack_np_fold_cache_is_idempotent_with_an_explicit_call` declared 16
computed tokens but never pushed block hashes or allocated blocks, so it panicked inside
the crate before reaching any assertion.

**Untouched, pre-existing** — `make check ROUND=round-1.1` fails: `bench/sweep_report.py`
does not exist there.

**Unverified without an NVIDIA GPU** — the NVRTC compile/launch path (§4.2) and
`host-tune.sh`'s sysfs/nvidia-smi writes (§2.2).

**The real acceptance test for this workspace** — boot against a plain dense model that is
nothing like the previous round's (e.g. `Qwen/Qwen3-0.6B`) and require both `make verify`
green **and** `rust_sched: AUTHORITY mode active (1 groups, ...)`. That is the Rust port
engaging on a model it has never seen. Needs the GPU box.

**Reference material** — `reference/lfm2/` holds the previous round's model-specific
kernels, patches and tests. Frozen, never imported, excluded from the build. Copy patterns
from it; do not wire it back in.

---

## 6. Official grading workload & scoring (round-2 BTC spec)

Both halves changed from round 1 — the workload is a different kind of thing entirely, and
the scoring band moved by more than an order of magnitude. The frozen round-1.2 writeup in
`reference/lfm2/HANDOFF.md` is **historical**; nothing in it describes what round 2 grades.

### 6.1 Scoring — ERS (Effective Request Score)

Formula unchanged from round 1; per request:

```
S_request = w · s_ttft + (1 − w) · s_tpot
s_ttft = [clamp((C_ttft − TTFT)      / (C_ttft − F_ttft), 0, 1)]^γ
s_tpot = [clamp((C_tpot − TPOT_mean) / (C_tpot − F_tpot), 0, 1)]^γ
ERS    = mean of S_request over ALL requests — a failed request scores 0 but stays
         in the denominator.
```

Round-2 parameters (BTC, 2026-08):

| Param | Round 2 | (Round 1, for contrast) |
|---|---|---|
| F_ttft / C_ttft | **200 ms / 6,000 ms** | 10 / 400 ms |
| F_tpot / C_tpot | **8 ms / 100 ms** | 1 / 10 ms |
| γ | 2.0 | 2.0 |
| w | 0.5 | 0.5 |

Reference implementation: `bench/_ci_report.py` (`F_TTFT`/`C_TTFT`/`F_TPOT`/`C_TPOT`).

What the new band means for prioritization:

- **Exchange rate**: the TTFT band is 5,800 ms wide, the TPOT band 92 ms — at equal
  normalized headroom, **1 ms of TPOT ≈ 63 ms of TTFT**. TPOT still dominates, even more
  than in round 1.
- **But the absolute gradient collapsed ~10×**: dERS/dTPOT ≈ 0.011·u per ms (u =
  normalized headroom ≤ 1), vs ~0.11·u in round 1. Sub-millisecond TPOT micro-opts that
  were worth real score last round are now near-noise; re-derive any per-knob effort
  threshold before committing to kernel work (see the sizing note in `RUST-RUNNER.md` §6).
- The floors moved from "unreachable" (10 ms TTFT / 1 ms TPOT) to **plausibly reachable**
  (200 ms TTFT / 8 ms TPOT): saturating a component is now a real target, and past the
  floor further speed buys nothing — spend the headroom on the other component or on
  avoiding failures.
- Failures still cost the full request score. A request that exceeds the ceilings scores
  ~0 anyway, so under load-shedding pressure, degrading one request's latency is no worse
  than failing it — but a failed request can also invalidate the run (§6.2).

### 6.2 Workload — aiperf AgentX replay of the Weka corpus (spec §3.1)

Grading replays **real multi-turn Claude Code sessions** (the SemiAnalysis Weka corpus,
including subagent spawn/join and recorded inter-turn think time) against the submission
for **900 s**, using NVIDIA
[aiperf](https://docs.nvidia.com/aiperf/tutorials/datasets-inputs/inference-x-agent-x-mvp-benchmark)'s
locked scenario:

```
aiperf profile --scenario inferencex-agentx-mvp \
  --public-dataset semianalysis_cc_traces_weka_062126 \   # HF: semianalysisai/cc-traces-weka-062126 (pinned)
  --concurrency 5 --max-context-length 204800 \
  --benchmark-duration 900 --random-seed <hidden>
```

Confirmed BTC values: concurrency **5** (session *trees* — subagent streams inside a tree
add parallelism beyond 5 in-flight requests), context cap **204,800**, duration **900 s**,
dataset pinned to the `062126` snapshot. The seed is hidden. Scenario-locked behavior:
streaming on, `ignore_eos:true` (full-length decodes, no early stop), per-play cache-bust
marker on each trace's first turn, recorded inter-turn delays preserved with a 10 s global
idle cap, chat endpoint, server-reported token counts. aiperf stamps `submission_valid` in
its output — **false** on >1% context-overflow rate, cancellation, or rule override, and
an invalid run's numbers are not comparable to anything.

Run it locally with `make bench-aiperf` (H200 box; `pip install -r
bench/requirements-aiperf.txt` first — the dataset auto-downloads from HF on first run).
The target mirrors the BTC command except the seed (`AIPERF_SEED`, default 0 — sweep 2–3
seeds to bound seed sensitivity). `bench/aiperf_adapter.py` converts the aiperf artifacts
into the repo's run schema and `bench/_ci_report.py` prints the ERS report. Smoke:
`make bench-aiperf AIPERF_LIMIT=8 AIPERF_DURATION=120`.

### 6.3 Two bench paths — which numbers to trust

| | `make bench` (synthetic trace) | `make bench-aiperf` (grading fidelity) |
|---|---|---|
| Workload | authored 420-request trace, `data/input/trace-round2.jsonl` | Weka corpus replay via aiperf |
| Where it runs | anywhere incl. rtx3060 CI | H200 box only |
| Use for | fast iteration, CI regression signal | any decision about what to ship |

The synthetic trace's arrival process, token shape, and prefix structure **no longer match
grading** (it predates spec §3.1). Its numbers remain useful as a *relative* regression
signal, but final tuning decisions — scheduler settings, `--max-num-seqs`, speculative
decoding, anything traded against ERS — must be justified on `bench-aiperf` runs.

**`make warm` and the vllm-fork PGO/BOLT stages no longer replay the synthetic trace.**
They consume `data/input/trace-weka.jsonl` (`make trace-weka`, generated from the real
Weka corpus by `bench/build_trace_weka.py` — see `bench/README.md`), so baked compile
caches and the frontend's profile-guided layout are trained on grading-shaped traffic
(long contexts, decode-heavy, real prefix-reuse topology).

### 6.4 Serving-config implications (open items)

- **Blocking**: `docker-compose.yaml` still pins the LFM2.5 boot placeholder with
  `--max-model-len=32768`. 32,768 < 204,800 guarantees a >1% context-overflow rate →
  `submission_valid: false`. A valid `bench-aiperf` run (and any real submission) needs
  the round-2 model literals landed first (§3.1 of this handoff).
- `--max-num-seqs=32`'s comment cites "peak trace concurrency ~6" — that was the dead
  synthetic trace (whose role in warm/PGO the Weka-derived trace has since taken over).
  Re-measure concurrency under aiperf (5 trees + subagent fan-out) before trusting the
  cap or the cudagraph capture sizes.
- `ignore_eos:true` means every request decodes its full recorded length: EOS-dependent
  early-stop logic (e.g. `VTL_ENABLE_STEP0_EOS_BAN`) is inert under grading, and the
  decode:prefill ratio is far higher than the prefill-biased synthetic trace assumed.
