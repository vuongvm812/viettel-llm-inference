# Root dispatcher. The buildable project lives in per-round folders; this Makefile drives
# whichever round you select and keeps the shared model weights (hf-model/) and vendored engine
# repos at the root so they are not duplicated per round.
#
#   round-1.1/  frozen Qwen3.5-2B baseline (phase-1)
#   round-1.2/  LFM2.5-1.2B refactor       (phase-2, default)
#
# Pick the round with ROUND=... on any target:
#   make up                      # round-1.2 (default)
#   make up ROUND=round-1.1      # the baseline
#   make test-kernel ROUND=round-1.2
ROUND ?= round-1.2
ifeq ($(wildcard $(ROUND)/.),)
$(error ROUND '$(ROUND)' not found -- use ROUND=round-1.1 or ROUND=round-1.2)
endif

IMAGE ?= unseenablefuture/awesome-badger
TAG ?= dev
TARGET ?= http://localhost:8000
# The H200 box is amd64 and the vLLM base image is multi-arch. Never let the build
# host pick: an arm64 Mac would otherwise produce an image the GPU box can't run.
PLATFORM ?= linux/amd64
# SM archs baked into vtl._C. A wrong arch fails at the first kernel launch, not at import.
# Narrow to '9.0+PTX' for the submission build. See the ARG in Dockerfile.
CUDA_ARCHS ?= 8.0;8.6;8.9;9.0+PTX
# Cap parallel nvcc so the CUDA build does not OOM a small box (see ARG in Dockerfile).
# Bump on a big-RAM CI host: make build MAX_JOBS=28.
MAX_JOBS ?= 4

# Upstream stock vLLM. The forked image is built FROM this (make vllm-fork); never FROM the fork.
VLLM_STOCK ?= vllm/vllm-openai:v0.26.0
# The bare version ("0.26.0"), derived from VLLM_STOCK so there is ONE place to bump. Passed to
# the fork build as VLLM_VER, where it drives both the `vllm.__version__` assert and the
# vtl/vllm_patches/v$(VLLM_VER)/ directory -- the patch set can no longer disagree with the base.
VLLM_VER := $(patsubst v%,%,$(lastword $(subst :, ,$(VLLM_STOCK))))
# Forked vLLM base = VLLM_STOCK + vtl source patches, built by $(ROUND)/Dockerfile.vllm-fork
# (Python-only overlay -> no CUDA rebuild). Pin VLLM_FORK_TAG to the pushed digest. See make vllm-fork.
#
# The patches include the short_conv in_proj hoist + the paired lfm2 empty_like fix
# (vtl/vllm_patches/v0.26.0/{short_conv,lfm2}.patch), which are what let RMSNormQuantFusionPass
# reach the 10 short-conv layers. Any future edit to those patches needs `make vllm-fork PUSH=1`
# and a re-pin here, or `make up` silently serves the old fork -- which looks exactly like
# success. `make verify` is the check: the "fusion replaced N patterns" count drops back to its
# pre-hoist value instead of covering the conv layers.
VLLM_FORK_IMAGE ?= unseenablefuture/vllm-fork
# TAG and DIGEST are split because they have incompatible uses: `docker buildx build -t` REFUSES
# a digest reference ("refusing to create a tag with a digest reference"), so the build target can
# only ever use the bare tag, while consumption must be digest-pinned for reproducibility. Folding
# both into one variable is why the pre-upgrade tree's `make vllm-fork` could not run.
VLLM_FORK_TAG ?= v0.26.0
# ROLLBACK (vLLM v0.25.0, last known-good before the 2026-07-26 v0.26.0 upgrade):
#   VLLM_FORK_TAG=v0.25.0-tree VLLM_FORK_DIGEST=@sha256:45ee1a65d96e9e65c54296273629e7a1ac6824dbe69611e104bb7073c799f91e
# Note that is an IMAGE-level rollback only: vtl/vllm_patches/v0.25.0/ was deleted in the upgrade,
# so rebuilding a v0.25.0 fork from source needs `git revert` first. VTL_DISABLE=1 does NOT undo
# it either -- that only bypasses the plugin, never the .patch-applied source baked into the image.
#
# UNPINNED until `make vllm-fork PUSH=1` runs on the H200. Empty = resolve by mutable tag, which
# means two "identical" A/B boots can silently be different images -- the one failure the
# boot-to-boot noise methodology cannot detect. `make push` refuses to ship while it is empty.
VLLM_FORK_DIGEST ?= @sha256:bb66a9208449fb6551a79e73e18fa94a8c959d726fd84f176973b6978bd3d66d
# Base image the MAIN image builds FROM. Defaults to the fork above so build/up/warm run the
# patched vLLM. Stock build (or the round-1.1 baseline): make ... VLLM_IMAGE=$(VLLM_STOCK)
VLLM_IMAGE ?= $(VLLM_FORK_IMAGE):$(VLLM_FORK_TAG)$(VLLM_FORK_DIGEST)
# 1 = the fork's rust-builder stage does a profile-guided-optimization build of vllm-rs
# (CPU-only training run against the mock engine). 0 = plain optimized (fat-LTO) build.
VLLM_RS_PGO ?= 1
# Tokenizer the PGO training run boots the frontend with. Defaults to the local model
# (hf-model/, bind-mounted at /model in the fork's PGO stage — only tokenizer/config are
# read, the mock fakes the forward pass so the 5.9 GB of weights are never loaded).
# Override with a HF repo id to fetch a stand-in over the network, e.g. PGO_MODEL=Qwen/Qwen3-0.6B.
PGO_MODEL ?= /model
# Host path to the local model dir mounted at /model for the PGO training run.
PGO_HFMODEL ?= ../hf-model
# -Ctarget-cpu for the vllm-rs binary (plain AND PGO builds). Default `x86-64`: the portable
# baseline ISA, so the binary is independent of whatever box built it and cannot SIGILL on the
# judge host. Note the spelling — rustc/LLVM's CPU name is `x86-64`, NOT `x86_64`; the
# underscore form is unrecognised and LLVM warns and silently falls back, which looks like it
# worked. Raise it for more codegen: `x86-64-v3` (AVX2, safe on any modern server),
# `x86-64-v4` (AVX-512), or `native` (best codegen, but bakes in the BUILD box's ISA — only
# correct when building ON the H200, and a native/AVX2 instrumented binary can crash the PGO
# training replay under Rosetta/OrbStack). Empty also means baseline (RUSTFLAGS unset).
PGO_TARGET_CPU ?= x86-64

# All paths below are relative to the selected round. `IN` cd's into it so docker-compose build
# contexts, relative volume mounts, and `docker cp` cache paths all resolve inside the round.
IN := cd $(ROUND) &&
TRACE := data/input/trace-round2.jsonl
# docker-compose.yaml is the SUBMISSION artifact and the single source of truth for every
# serve flag and env var; the three overlays only carry their differences (dev image tag,
# local build+mount, judge-box resource caps). Order matters -- later -f wins.
COMPOSE_FILES := -f docker-compose.yaml -f docker-compose-optimized.yaml -f docker-compose.localtest.yaml -f docker-compose.cpucap.yaml
DC := docker compose $(COMPOSE_FILES)

# CI bench lifecycle (remote runner). No build step — image is the pinned digest from ci-build.
IMAGE_DIGEST ?=
CIBENCH_COMPOSE := -f docker-compose-optimized.yaml -f docker-compose.ci-bench.yaml
_CI_IMAGE = $(if $(IMAGE_DIGEST),$(IMAGE)@$(IMAGE_DIGEST),$(IMAGE):$(TAG))

.PHONY: check stats build up down warm push bench arm sweep-sched-tokens sweep-quant sweep-schedule profile test-kernel bench-kernel debug-kernel verify prove vllm-fork ci-build ci-digest ci-watch ci-status ci-up ci-down ci-bench ci-bootstrap

## Self-checks. Run anywhere: no GPU, no vLLM, no running server. Adapts to the round's patch set
## (round-1.1 has the GDN patches; round-1.2 does not) by globbing rather than hardcoding names.
check:
	$(IN) python3 vtl/registry.py
	$(IN) for f in vtl/patches/*.py; do \
	  case "$$f" in */__init__.py) continue;; esac; \
	  echo "-- $$f"; PYTHONPATH=. python3 "$$f" || exit 1; \
	done
	$(IN) python3 bench/trace_stats.py --self-check
	$(IN) python3 bench/metrics.py
	$(IN) python3 bench/eval_quality.py --self-check
	$(IN) python3 bench/profile_trace.py --self-check
	$(IN) python3 bench/arm_compose.py --self-check
	$(IN) [ -f bench/build_trace_round2.py ] && PYTHONPATH=. python3 bench/build_trace_round2.py --self-check || true
	$(IN) python3 -c "import vtl.patches, vtl.plugin; print('vtl imports without vLLM: ok')"

# No compose, no server, no model: the kernel tests just need the image and a GPU.
# --entrypoint bash because the vLLM base image starts the API server otherwise.
# -p no:cacheprovider because /bench is mounted read-only.
# The /bench/test_*.py glob expands IN THE CONTAINER, so each round runs exactly its own tests.
# Both targets depend on `build`: $(IMAGE):$(TAG) also names a registry repo, so without it
# docker silently pulls a stale published image and the tests run against whatever kernel
# it happens to contain.
KRUN := docker run --rm --gpus all -v $(PWD)/$(ROUND)/bench:/bench:ro --entrypoint bash $(IMAGE):$(TAG) -lc
PYTEST := pytest -q -p no:cacheprovider /bench/test_*.py

## Kernel correctness. Needs a GPU. Runs one oracle against our kernel AND against the
## stock one -- importing vtl._C overrides _C process-wide, so they cannot coexist and
## agreeing with the same reference is what proves ours matches stock.
test-kernel: build
	$(KRUN) 'pip install -q pytest && \
	  echo "--- vtl kernel"  && $(PYTEST) && \
	  echo "--- stock kernel" && VTL_SKIP_EXT=1 $(PYTEST)'

## Kernel microbenchmark at the trace's real shapes. Needs a GPU.
bench-kernel: build
	$(KRUN) 'for t in /bench/test_*.py; do \
	    echo "=== $$t (vtl)"; python3 $$t; \
	    echo "=== $$t (stock)"; VTL_SKIP_EXT=1 python3 $$t; \
	  done'

## Pinpoint a memory fault. VTL_KERNEL_SYNC makes the kernel synchronise after every launch
## and, on a fault, raise with the exact shape and path. Runs each test stopping at first raise.
##   make debug-kernel                          # whole suite, stops at first fault
##   make debug-kernel T=test_misaligned        # one test
T ?=
DBG_KRUN := docker run --rm --gpus all -e VTL_KERNEL_SYNC=1 -e CUDA_LAUNCH_BLOCKING=1 \
              -v $(PWD)/$(ROUND)/bench:/bench:ro --entrypoint bash $(IMAGE):$(TAG) -lc
debug-kernel: build
	$(DBG_KRUN) 'pip install -q pytest && \
	  python3 -m pytest -q -p no:cacheprovider -x /bench/test_*.py \
	  $(if $(T),-k $(T),)'

stats:
	$(IN) python3 bench/trace_stats.py --trace $(TRACE)

## Boot at INFO, run the assertions below, tear down. THIS is the target that can actually
## prove the artifact: `make verify` alone reads the logs of whatever is already running, and
## the shipped config logs at WARNING, where every vtl success line is dropped and the very
## first grep ("vtl: applied") fails. See docker-compose.verify.yaml for the full reasoning.
## Never measure latency from this boot -- INFO logging costs host time on a host-bound loop.
prove:
	$(IN) $(DC) -f docker-compose.verify.yaml up -d --force-recreate --wait
	@$(MAKE) verify ROUND=$(ROUND) || { $(IN) $(DC) -f docker-compose.verify.yaml down; exit 1; }
	$(IN) $(DC) -f docker-compose.verify.yaml down

## Post-boot assertions. We rely on vLLM's defaults rather than passing risky flags,
## so prove the defaults actually resolved our way. Run against a live container.
## Reads an ALREADY-RUNNING container's logs -- prefer `make prove`, which boots one at the
## log level these greps need.
verify:
	@$(IN) $(DC) logs model 2>/dev/null > /tmp/vtl-verify.log || true
	@# vLLM always logs this line once config is resolved, enabled or not. Its absence
	@# means the server died before that -- do not report it as "async disabled".
	@grep -q "Asynchronous scheduling is" /tmp/vtl-verify.log \
	  || { echo "FAIL server never reached config resolution. Last lines:"; \
	       tail -5 /tmp/vtl-verify.log; exit 1; }
	@grep -q "Asynchronous scheduling is enabled" /tmp/vtl-verify.log \
	  && echo "OK   async scheduling (zero-overhead batch scheduler) enabled" \
	  || { echo "FAIL async scheduling is DISABLED -- an incompatible option is set"; exit 1; }
	@grep -q "vtl: applied" /tmp/vtl-verify.log \
	  && echo "OK   vtl plugin loaded" \
	  || { echo "FAIL vtl plugin never ran -- you are benchmarking stock vLLM"; exit 1; }
	@grep -q "registered quantization method 'vtl_fp8'" /tmp/vtl-verify.log \
	  && echo "OK   vtl_fp8 registered" \
	  || { echo "FAIL vtl_fp8 not registered"; exit 1; }
	@# vtl_w4a8 is the shipped --quantization. Every one of its failure modes degrades to fp8
	@# rather than crashing, so without these three greps `make verify` prints all-OK in exactly
	@# the world where int4 silently never happened. Check registration, kernel availability,
	@# and how many layers actually ended up int4.
	@# The premise of the whole W4A8 patch: a MIG 1g.18gb slice (~19 SMs) is where int4 pays.
	@# On a FULL H200 (132 SMs) the same saving is ~0.09 ms against a 3.4 ms host term -- i.e.
	@# invisible, with the TTFT and accuracy costs unchanged, so vtl_fp8 is the better ship.
	@d=$$(sed -n 's/.*w4a8 device = //p' /tmp/vtl-verify.log | tail -1); \
	 if [ -z "$$d" ]; then echo "WARN GPU identity unknown -- w4a8 registration never logged"; \
	 else \
	   sm=$$(echo "$$d" | sed -n 's/.*, \([0-9]*\) SMs.*/\1/p'); \
	   if [ -n "$$sm" ] && [ "$$sm" -gt 60 ]; then \
	     echo "WARN GPU is $$d -- NOT a MIG slice. W4A8 buys ~0.09 ms here and still costs TTFT"; \
	     echo "     + accuracy: run 'make sweep-quant' before submitting."; \
	   else echo "OK   GPU is $$d"; fi; \
	 fi
	@grep -q "registered quantization method 'vtl_w4a8'" /tmp/vtl-verify.log \
	  && echo "OK   vtl_w4a8 registered" \
	  || { echo "FAIL vtl_w4a8 not registered -- the serve flag would abort at startup"; exit 1; }
	@grep -q "W4A8 CUDA ops absent" /tmp/vtl-verify.log \
	  && { echo "FAIL W4A8 kernel missing from this image (needs sm90a + CUDA>=12); serving fp8"; exit 1; } \
	  || echo "OK   W4A8 CUDA ops present"
	@grep -q "w4a8 quantized 0 layers" /tmp/vtl-verify.log \
	  && { echo "FAIL 0 layers quantized to int4 -- this is an fp8 server wearing a w4a8 flag"; exit 1; } \
	  || true
	@n=$$(sed -n 's/.*w4a8 quantized \([0-9]*\) layers.*/\1/p' /tmp/vtl-verify.log | tail -1); \
	 if [ -n "$$n" ]; then echo "OK   vtl_w4a8 quantized $$n layers to int4"; \
	 else echo "WARN w4a8 layer count unknown -- model load never finished, or logging below INFO"; fi
	@# lm_head is 268 MB/step, 31.5% of the post-w4a8 decode weight budget, and every one of its
	@# failure modes leaves it silently bf16 -- indistinguishable from success in every other
	@# signal. lm_head_quant.py logs the outcome AFTER the weights actually changed.
	@grep -q "lm_head quantization .* FAILED" /tmp/vtl-verify.log \
	  && { echo "FAIL lm_head fell back to bf16 -- 268 MB/step of decode traffic we thought was gone"; exit 1; } \
	  || true
	@m=$$(sed -n 's/.*lm_head quantized to \([a-z0-9]*\).*/\1/p' /tmp/vtl-verify.log | tail -1); \
	 if [ -n "$$m" ]; then echo "OK   lm_head quantized to $$m"; \
	 elif grep -q 'lm_head=off' /tmp/vtl-verify.log; then echo "WARN lm_head is bf16 (VTL_LM_HEAD_QUANT=off)"; \
	 else echo "WARN lm_head outcome unknown -- model load never finished, or VTL_W4A8_IGNORE names it"; fi
	@grep -q "channelwise fp8 unavailable" /tmp/vtl-verify.log \
	  && echo "WARN channelwise fp8 fell back to stock per-tensor" \
	  || echo "OK   channelwise fp8 active"
	@# The jemalloc apt step deliberately cannot fail the build (a failed build scores zero,
	@# glibc malloc only scores worse), so the check it used to do at build time lives here:
	@# a missing lib makes the loader print this once per process. WARN, not FAIL -- it serves.
	@grep -q "libjemalloc.so.2.*cannot be preloaded" /tmp/vtl-verify.log \
	  && echo "WARN jemalloc NOT preloaded -- running on glibc malloc; see the build log for 'jemalloc:'" \
	  || echo "OK   jemalloc preloaded"
	@# Expected to fail when you deliberately A/B with VTL_ENABLE_RMS_NORM_QUANT=0.
	@grep -q "fused rms_norm+fp8-quant CUDA kernel installed" /tmp/vtl-verify.log \
	  && echo "OK   vtl fused norm+quant kernel installed" \
	  || { echo "FAIL vtl kernel not installed -- stock _C kernel is running"; exit 1; }
	@# RMSNormQuantFusionPass emits only `Replaced N patterns` (rms_quant_fusion.py, DEBUG).
	@# N is how many nodes were rewritten into the op our kernel backs; N=0 means it never runs.
	@n=$$(grep "rms_quant_fusion.py.*Replaced" /tmp/vtl-verify.log | tail -1 \
	      | sed -n 's/.*Replaced \([0-9]*\) patterns.*/\1/p'); \
	 if [ -z "$$n" ]; then \
	   echo "WARN fusion match count unknown -- rerun with VLLM_LOGGING_LEVEL=DEBUG"; \
	 elif [ "$$n" -eq 0 ]; then \
	   echo "FAIL rms_norm+quant fusion replaced 0 patterns -- the kernel is never reached"; exit 1; \
	 else \
	   echo "OK   rms_norm+quant fusion replaced $$n patterns (each one calls our kernel)"; \
	 fi

# --provenance=false --sbom=false: skip the SBOM/provenance attestation manifest and the
# single-entry manifest LIST buildx would otherwise wrap a one-platform image in.
# NOCACHE=--no-cache forces a full rebuild (re-runs pip/nvcc so the CURRENT kernels are recompiled).
NOCACHE ?=
BUILDX_FLAGS := --provenance=false --sbom=false $(NOCACHE)

## Build (PUSH=1 to push) the forked vLLM base image: stock v0.26.0 + vtl/vllm_patches +
## an optimized rebuild of the Rust frontend `vllm-rs` (fat-LTO release; VLLM_RS_PGO=1 adds a
## CPU-only PGO training run via the mock engine). The rust-builder stage reads the vllm/rust/
## workspace at repo root via the named build context. Then point the main image at it, pin by digest:
##   make vllm-fork                 # plain optimized rebuild
##   make vllm-fork VLLM_RS_PGO=1   # + profile-guided optimization
##   make vllm-fork PUSH=1
##   make push VLLM_IMAGE=$(VLLM_FORK_IMAGE):$(VLLM_FORK_TAG)@sha256:<digest>
vllm-fork:
	$(IN) docker buildx build $(BUILDX_FLAGS) --platform $(PLATFORM) --build-arg VLLM_IMAGE='$(VLLM_STOCK)' --build-arg VLLM_VER='$(VLLM_VER)' --build-arg VLLM_SRC_REF='v$(VLLM_VER)' --build-arg PGO_MODEL='$(PGO_MODEL)' --build-arg PGO_TARGET_CPU='$(PGO_TARGET_CPU)' $(if $(filter 1,$(VLLM_RS_PGO)),--build-arg RUST_BUILDER=rust-builder-pgo --build-context hfmodel=$(PGO_HFMODEL)) $(if $(PUSH),--push,--load) -t $(VLLM_FORK_IMAGE):$(VLLM_FORK_TAG) -f Dockerfile.vllm-fork .
	@echo "forked vLLM base: $(VLLM_FORK_IMAGE):$(VLLM_FORK_TAG)"
	@if [ -n "$(PUSH)" ]; then $(IN) docker buildx imagetools inspect $(VLLM_FORK_IMAGE):$(VLLM_FORK_TAG) --format 'set VLLM_FORK_DIGEST=@{{.Manifest.Digest}}'; fi

build:
	$(IN) docker buildx build $(BUILDX_FLAGS) --platform $(PLATFORM) --build-arg VLLM_IMAGE='$(VLLM_IMAGE)' --build-arg CUDA_ARCHS='$(CUDA_ARCHS)' --build-arg MAX_JOBS='$(MAX_JOBS)' --load -t $(IMAGE):$(TAG) .
	@docker inspect $(IMAGE):$(TAG) --format 'built {{.Os}}/{{.Architecture}}'

# `docker compose up --build` cannot take --build-arg, so build first (which honors it) then up.
up:
	$(IN) $(DC) build --build-arg VLLM_IMAGE='$(VLLM_IMAGE)'
	$(IN) $(DC) up

down:
	$(IN) $(DC) down -v

## torch.compile needs a real GPU, so `docker build` cannot warm its cache. Boot the image, drive
## enough traffic to trigger compile + CUDA graph capture + FlashInfer autotune, then copy the
## caches back into the build context and rebuild. Two passes: open-loop warms low-concurrency
## shapes; closed-loop saturates a full batch so the multi-seq kernels compile into the cache too.
WARM_CONCURRENCY ?= 16
WARM_REQS ?= 32
warm:
	$(IN) $(DC) build --build-arg VLLM_IMAGE='$(VLLM_IMAGE)'
	$(IN) $(DC) up -d --wait   # --wait blocks until the healthcheck passes
	$(IN) python3 bench/replay.py --target $(TARGET) --trace $(TRACE) --limit 4 --out /dev/null
	$(IN) python3 bench/replay.py --target $(TARGET) --trace $(TRACE) \
	  --closed-loop $(WARM_CONCURRENCY) --limit $(WARM_REQS) --out /dev/null
	$(IN) docker cp "$$($(DC) ps -q model)":/opt/vtl/cache/. docker/cache/
	@# Never bake a startup plan (VLLM_ENABLE_STARTUP_PLAN). Its fingerprint covers VllmConfig +
	@# device + torch build but NOT the VTL_* env knobs that change resident weight bytes, and the
	@# apply gate samples free memory BEFORE weights load -- so a plan recorded under one vtl
	@# config gets replayed under another and sizes the KV cache for memory that isn't there.
	$(IN) rm -rf docker/cache/vllm/startup_plan
	$(IN) $(DC) down
	$(MAKE) build ROUND=$(ROUND)

## buildx --push writes the manifest straight to the registry, so the pushed image
## is $(PLATFORM) regardless of what this machine is.
push:
	@# A pushed image is what the judge runs, so refuse to build one on top of a mutable tag.
	@case '$(VLLM_IMAGE)' in *@sha256:*) ;; *) \
	  echo "FAIL base image '$(VLLM_IMAGE)' is not digest-pinned."; \
	  echo "     run 'make vllm-fork PUSH=1', then set VLLM_FORK_DIGEST=@sha256:<digest>."; \
	  exit 1;; esac
	$(IN) docker buildx build $(BUILDX_FLAGS) --platform $(PLATFORM) --build-arg VLLM_IMAGE='$(VLLM_IMAGE)' --build-arg CUDA_ARCHS='$(CUDA_ARCHS)' --build-arg MAX_JOBS='$(MAX_JOBS)' --push -t $(IMAGE):$(TAG) .
	@echo "pin this digest in $(ROUND)/docker-compose.yaml:"
	@docker buildx imagetools inspect $(IMAGE):$(TAG) --format '{{.Manifest.Digest}}'

## Open-loop replay (honors the trace's arrival times) + a closed-loop sweep.
bench:
	$(IN) python3 bench/replay.py --target $(TARGET) --trace $(TRACE) --out bench-open.json
	$(IN) for n in 1 8 32 128; do \
	  python3 bench/replay.py --target $(TARGET) --trace $(TRACE) \
	    --closed-loop $$n --out bench-closed-$$n.json; \
	done

# --- flag arms ------------------------------------------------------------------------------
# bench/arm_compose.py needs the same overlay chain docker sees, in its own spelling.
ARM_COMPOSE_FILES := $(patsubst -f,--compose-file,$(COMPOSE_FILES))

## Boot ONE arm: the shipped config overridden by exactly the flags you name, replay, tear down.
## Never edits docker-compose.yaml -- that file is the submission artifact, and a literal left
## un-reverted after a sweep ships a config nobody chose. See bench/arm_compose.py.
##   make arm ARM=fp8   ARM_FLAGS='--quantization=vtl_fp8'
##   make arm ARM=st1024 ARM_FLAGS='--max-num-scheduled-tokens=1024'
##   make arm ARM=noflag ARM_FLAGS='~--max-num-scheduled-tokens'   # prove the flag is neutral
## ARM_EVAL=1 also captures greedy outputs for an accuracy diff (eval_quality.py --compare).
ARM ?= arm
ARM_FLAGS ?=
ARM_EVAL ?=
arm:
	@[ -n "$(ARM_FLAGS)" ] || { echo "usage: make arm ARM=<name> ARM_FLAGS='--flag=value ...'"; exit 1; }
	$(IN) python3 bench/arm_compose.py --out /tmp/vtl-arm-$(ARM).yaml $(ARM_COMPOSE_FILES) $(ARM_FLAGS)
	$(IN) $(DC) -f /tmp/vtl-arm-$(ARM).yaml up -d --force-recreate --wait
	@# Tear down even if the replay fails, or the next arm boots on top of this one.
	$(IN) ( python3 bench/replay.py --target $(TARGET) --trace $(TRACE) --out bench-arm-$(ARM).json \
	  && { [ -z "$(ARM_EVAL)" ] || python3 bench/eval_quality.py --target $(TARGET) --trace $(TRACE) --out ref-$(ARM).json; } \
	  ); rc=$$?; $(DC) -f /tmp/vtl-arm-$(ARM).yaml down; exit $$rc

## The --max-num-scheduled-tokens bracket. This flag ships LIVE at 2048 with no measurement
## behind it: 2048 is below the longest prompt (4,281 tok), so a turn-1 prefill spans 3
## scheduler steps instead of 1 -- it trades TTFT for a flatter in-flight decode ITL tail.
## 8192 == max_num_batched_tokens == the flag-absent behaviour. Watch the TPOT tail, not the mean.
SCHED_TOKENS ?= 8192 4096 2048 1024
sweep-sched-tokens:
	@for v in $(SCHED_TOKENS); do \
	  $(MAKE) arm ROUND=$(ROUND) ARM=st$$v ARM_FLAGS="--max-num-scheduled-tokens=$$v" || exit 1; \
	done
	@echo "compare: cd $(ROUND) && python3 bench/compare.py bench-arm-st*.json"

## W4A8 vs FP8, end to end. Two questions in one sweep, and the second is the one nobody has
## asked: (1) does int4 actually pay on THIS box -- it is worth ~1.08 ms/step of GPU time on a
## MIG slice but only ~0.09 ms on a full H200, where it still costs TTFT; (2) does group-128
## symmetric RTN with no calibration and no zero-points hurt quality? quant_w4a8.py says
## ACCURACY IS UNMEASURED. ARM_EVAL=1 captures both arms' greedy output for the diff.
sweep-quant:
	@for q in vtl_w4a8 vtl_fp8; do \
	  $(MAKE) arm ROUND=$(ROUND) ARM=$$q ARM_EVAL=1 ARM_FLAGS="--quantization=$$q" || exit 1; \
	done
	@echo "latency:  cd $(ROUND) && python3 bench/compare.py bench-arm-vtl_w4a8.json bench-arm-vtl_fp8.json"
	@echo "accuracy: cd $(ROUND) && python3 bench/eval_quality.py --compare ref-vtl_fp8.json ref-vtl_w4a8.json"

## Sweep the CUTLASS W4A8 tile/cluster schedule (needs the GPU). The kernel's own heuristic
## (w4a8_mm_entry.cu:341-372) keys ONLY on M/N/K and is blind to SM count -- it was tuned on a
## full 132-SM Hopper and the judge's MIG 1g.18gb slice has ~19, so its choice is an open
## question, not a settled one. This is the one already-exposed W4A8 tunable never swept.
## `heuristic` is the sentinel for "unset", the baseline every other row must beat -- a literal
## empty word cannot survive make -> shell word splitting. The three named tiles are what the
## heuristic picks for our shapes (M<=32 decode, 8192-token prefill).
SCHEDULES ?= heuristic 128x16_1x1x1 128x32_1x1x1 128x256_1x1x1
sweep-schedule:
	@for name in $(SCHEDULES); do \
	  case "$$name" in heuristic) s="";; *) s="$$name";; esac; \
	  echo "=== VTL_W4A8_SCHEDULE=$${s:-<heuristic>}"; \
	  ( cd $(ROUND) && VTL_W4A8_SCHEDULE="$$s" $(DC) up -d --force-recreate --wait \
	    && VTL_W4A8_SCHEDULE="$$s" python3 bench/replay.py --target $(TARGET) --trace $(TRACE) \
	         --out bench-sched-$$name.json \
	    ; rc=$$?; VTL_W4A8_SCHEDULE="$$s" $(DC) down; exit $$rc ) || exit 1; \
	done
	@echo "compare: $(ROUND)/bench-sched-*.json (heuristic is the baseline)"

## Phase-0 profiler (needs the H200). Boots with vLLM's torch profiler enabled, drives a small
## closed-loop replay, and prints a ranked GPU-kernel cost table. See docs/plans/.
profile:
	$(IN) mkdir -p bench-profile
	$(IN) rm -f bench-profile/vtl-trace-*.json bench-profile/.arm
	$(IN) $(DC) -f docker-compose.profile.yaml up -d --build --force-recreate --wait
	$(IN) touch bench-profile/.arm   # arm AFTER warmup so the capture is the replay, not warmup
	$(IN) python3 bench/replay.py --target $(TARGET) --trace $(TRACE) --closed-loop 8 --limit 48 --out /dev/null
	$(IN) sleep 3   # let the worker finish export_chrome_trace
	$(IN) python3 bench/profile_trace.py --profile-dir bench-profile --out bench-profile-summary.json
	$(IN) $(DC) -f docker-compose.profile.yaml down

# --- CI (remote build on self-hosted runner) -----------------------------------------------
CI_WORKFLOW ?= build-push.yml
CI_REPO ?= $(shell git remote get-url origin | sed 's|.*github.com[:/]\(.*\)\.git|\1|')

## Trigger remote CI build, stream logs, print digest. Exits non-zero on CI failure.
##   make ci-build                           # default ROUND, archs=9.0+PTX
##   make ci-build ROUND=round-1.1           # different round
##   make ci-build CUDA_ARCHS='8.6;9.0+PTX'  # custom arch list
##   make ci-build BUILD_FORK=1              # also rebuild vLLM fork image
ci-build:
	@echo "=== Triggering CI: $(ROUND) on $(CI_REPO) ==="
	gh workflow run $(CI_WORKFLOW) -R $(CI_REPO) \
	  -f workdir=$(ROUND) \
	  $(if $(CUDA_ARCHS),-f cuda_archs='$(CUDA_ARCHS)') \
	  $(if $(BUILD_FORK),-f build_fork=true) \
	  --ref $(shell git rev-parse --abbrev-ref HEAD)
	@sleep 6
	@id=$$(gh run list -R $(CI_REPO) -w $(CI_WORKFLOW) --limit 1 --json databaseId -q '.[0].databaseId'); \
	echo "=== Run $$id ==="; \
	if gh run watch $$id -R $(CI_REPO) --exit-status; then \
	  echo ""; \
	  $(MAKE) ci-digest RUN_ID=$$id; \
	else \
	  false; \
	fi

## Print digest for a specific CI run.
##   make ci-digest RUN_ID=9876543210
ci-digest:
	@[ -n "$(RUN_ID)" ] || { echo "ERROR: RUN_ID required. Usage: make ci-digest RUN_ID=<id>"; exit 1; }
	@digest=$$(gh run view $(RUN_ID) -R $(CI_REPO) --log 2>/dev/null | grep -oP '::VTL_DIGEST::\K.*' | tail -1); \
	if [ -n "$$digest" ]; then \
	  echo "$$digest"; \
	else \
	  echo "ERROR: digest not found — run $(RUN_ID) still running, failed, or older CI version"; \
	  exit 1; \
	fi

## Stream the most recent CI run log.
ci-watch:
	@id=$$(gh run list -R $(CI_REPO) -w $(CI_WORKFLOW) --limit 1 --json databaseId -q '.[0].databaseId'); \
	gh run watch $$id -R $(CI_REPO)

## List last 5 CI runs.
ci-status:
	gh run list -R $(CI_REPO) -w $(CI_WORKFLOW) --limit 5

## Start server for CI bench — no build, pinned image + model mount. Waits for healthy.
##    make ci-up IMAGE_DIGEST=sha256:abc...
ci-up:
	$(IN) CI_IMAGE='$(_CI_IMAGE)' \
	  docker compose $(CIBENCH_COMPOSE) up -d --wait

## Stop CI bench server, remove volumes.
ci-down:
	$(IN) docker compose $(CIBENCH_COMPOSE) down -v

## Trigger remote CI bench.
##    make ci-bench                                        # bench :dev
##    make ci-bench IMAGE_DIGEST=sha256:abc...              # bench specific image
ci-bench:
	gh workflow run bench.yml -R $(CI_REPO) \
	  -f workdir=$(ROUND) \
	  $(if $(IMAGE_DIGEST),-f image_digest='$(IMAGE_DIGEST)')
	@sleep 6
	@id=$$(gh run list -R $(CI_REPO) -w bench.yml --limit 1 --json databaseId -q '.[0].databaseId'); \
	echo "=== Run $$id ==="; \
	gh run watch $$id -R $(CI_REPO) --exit-status || true

## Trigger remote bootstrap smoke test — starts server, checks health, stops.
##    make ci-bootstrap                                        # test :dev
##    make ci-bootstrap IMAGE_DIGEST=sha256:abc...              # test specific image
ci-bootstrap:
	gh workflow run bootstrap.yml -R $(CI_REPO) \
	  -f workdir=$(ROUND) \
	  $(if $(IMAGE_DIGEST),-f image_digest='$(IMAGE_DIGEST)')
	@sleep 6
	@id=$$(gh run list -R $(CI_REPO) -w bootstrap.yml --limit 1 --json databaseId -q '.[0].databaseId'); \
	echo "=== Run $$id ==="; \
	gh run watch $$id -R $(CI_REPO) --exit-status
