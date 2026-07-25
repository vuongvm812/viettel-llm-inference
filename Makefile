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
VLLM_STOCK ?= vllm/vllm-openai:v0.25.0
# Forked vLLM base = VLLM_STOCK + vtl tree-spec source patches, built by $(ROUND)/Dockerfile.vllm-fork
# (Python-only overlay -> no CUDA rebuild). Pin VLLM_FORK_TAG to the pushed digest. See make vllm-fork.
#
# Re-pinned in 66e5882 to include the short_conv in_proj hoist + the paired lfm2 empty_like fix
# (vtl/vllm_patches/v0.25.0/{short_conv,lfm2}.patch), which are what let RMSNormQuantFusionPass
# reach the 10 short-conv layers. Any future edit to those patches needs `make vllm-fork PUSH=1`
# and a re-pin here, or `make up` silently serves the old fork -- which looks exactly like
# success. `make verify` is the check: the "fusion replaced N patterns" count drops back to its
# pre-hoist value instead of covering the conv layers.
VLLM_FORK_IMAGE ?= unseenablefuture/vllm-fork
VLLM_FORK_TAG ?= v0.25.0-tree@sha256:45ee1a65d96e9e65c54296273629e7a1ac6824dbe69611e104bb7073c799f91e
# Base image the MAIN image builds FROM. Defaults to the fork above so build/up/warm run the
# tree-spec vLLM. Stock build (or the round-1.1 baseline): make ... VLLM_IMAGE=$(VLLM_STOCK)
VLLM_IMAGE ?= $(VLLM_FORK_IMAGE):$(VLLM_FORK_TAG)
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
# -Ctarget-cpu for the vllm-rs binary (plain AND PGO builds). Default native: full host
# codegen (AVX-512 on an H200 host CPU). Bakes in the BUILD box's ISA — build on the deploy
# CPU (the H200) for the full win; an older build box (Mac under Rosetta ≈ AVX2) yields a
# portable subset that still runs on H200. For an emulated build, override PGO_TARGET_CPU=
# x86-64-v3 or empty (a native/AVX2 instrumented binary can crash the PGO training replay).
PGO_TARGET_CPU ?= native

# All paths below are relative to the selected round. `IN` cd's into it so docker-compose build
# contexts, relative volume mounts, and `docker cp` cache paths all resolve inside the round.
IN := cd $(ROUND) &&
TRACE := data/input/trace-round2.jsonl
# docker-compose.yaml is the SUBMISSION artifact and the single source of truth for every
# serve flag and env var; the three overlays only carry their differences (dev image tag,
# local build+mount, judge-box resource caps). Order matters -- later -f wins.
COMPOSE_FILES := -f docker-compose.yaml -f docker-compose-optimized.yaml -f docker-compose.localtest.yaml -f docker-compose.cpucap.yaml
# DEBUG=1 stacks the verification overlay (VLLM_LOGGING_LEVEL=DEBUG). Everything `make verify`
# asserts is logged at INFO/DEBUG, and the submission compose pins WARNING -- so `make up
# DEBUG=1 && make verify DEBUG=1` is the only combination in which those greps mean anything.
COMPOSE_FILES += $(if $(DEBUG),-f docker-compose.debug.yaml)
# PYSRC=1 mounts the checkout's pure-Python vtl sources over the image's copies and SKIPS the
# build, because `COPY vtl /src/vtl` sits above the nvcc RUN and a one-line Python edit would
# otherwise recompile every kernel. See docker-compose.pysrc.yaml for what it does not cover.
COMPOSE_FILES += $(if $(PYSRC),-f docker-compose.pysrc.yaml)
DC := docker compose $(COMPOSE_FILES)
BUILD := $(if $(PYSRC),echo "PYSRC=1: skipping build; vtl python is mounted",$(DC) build --build-arg VLLM_IMAGE='$(VLLM_IMAGE)')

.PHONY: check stats build up down warm push bench sweep-schedule profile slack test-kernel bench-kernel debug-kernel verify vllm-fork

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

## Post-boot assertions. We rely on vLLM's defaults rather than passing risky flags,
## so prove the defaults actually resolved our way. Run against a live container.
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
	@# One kv cache group per spec: the merge drops one SSM metadata build per decode step but
	@# cuts num_blocks ~40% (the pool is sliced by max layers-per-group). Print the capacity it
	@# bought/cost so the trade is visible on whatever GPU this is, not just the dev box.
	@grep -q "vtl: merged .* kv cache groups" /tmp/vtl-verify.log \
	  && { sed -n 's/.*vtl: \(merged [0-9]* kv cache groups[^;]*\).*/OK   \1/p' /tmp/vtl-verify.log | tail -1; \
	       sed -n 's/.*\(GPU KV cache size: .*\)/     \1/p;s/.*\(Maximum concurrency for .*\)/     \1/p' /tmp/vtl-verify.log | tail -2; } \
	  || echo "WARN kv cache groups not merged -- stock 3-group split (VTL_ENABLE_KV_CACHE_GROUPS=0?)"
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
	     echo "     + accuracy: A/B VTL_QUANT=vtl_fp8 before submitting."; \
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

## Build (PUSH=1 to push) the forked vLLM base image: stock v0.25.0 + vtl/vllm_patches +
## an optimized rebuild of the Rust frontend `vllm-rs` (fat-LTO release; VLLM_RS_PGO=1 adds a
## CPU-only PGO training run via the mock engine). The rust-builder stage reads the vllm/rust/
## workspace at repo root via the named build context. Then point the main image at it, pin by digest:
##   make vllm-fork                 # plain optimized rebuild
##   make vllm-fork VLLM_RS_PGO=1   # + profile-guided optimization
##   make vllm-fork PUSH=1
##   make push VLLM_IMAGE=$(VLLM_FORK_IMAGE):$(VLLM_FORK_TAG)@sha256:<digest>
vllm-fork:
	$(IN) docker buildx build $(BUILDX_FLAGS) --platform $(PLATFORM) --build-arg VLLM_IMAGE='$(VLLM_STOCK)' --build-arg PGO_MODEL='$(PGO_MODEL)' --build-arg PGO_TARGET_CPU='$(PGO_TARGET_CPU)' $(if $(filter 1,$(VLLM_RS_PGO)),--build-arg RUST_BUILDER=rust-builder-pgo --build-context hfmodel=$(PGO_HFMODEL)) $(if $(PUSH),--push,--load) -t $(VLLM_FORK_IMAGE):$(VLLM_FORK_TAG) -f Dockerfile.vllm-fork .
	@echo "forked vLLM base: $(VLLM_FORK_IMAGE):$(VLLM_FORK_TAG)"
	@if [ -n "$(PUSH)" ]; then $(IN) docker buildx imagetools inspect $(VLLM_FORK_IMAGE):$(VLLM_FORK_TAG) --format 'pin this digest: {{.Manifest.Digest}}'; fi

build:
	$(IN) docker buildx build $(BUILDX_FLAGS) --platform $(PLATFORM) --build-arg VLLM_IMAGE='$(VLLM_IMAGE)' --build-arg CUDA_ARCHS='$(CUDA_ARCHS)' --build-arg MAX_JOBS='$(MAX_JOBS)' --load -t $(IMAGE):$(TAG) .
	@docker inspect $(IMAGE):$(TAG) --format 'built {{.Os}}/{{.Architecture}}'

# `docker compose up --build` cannot take --build-arg, so build first (which honors it) then up.
up:
	$(IN) $(BUILD)
	$(IN) $(DC) up

down:
	$(IN) $(DC) down -v

## torch.compile needs a real GPU, so `docker build` cannot warm its cache. Boot the image, drive
## enough traffic to trigger compile + CUDA graph capture + FlashInfer autotune, then copy the
## caches back into the build context and rebuild. Two passes: open-loop warms low-concurrency
## shapes; closed-loop saturates a full batch so the multi-seq kernels compile into the cache too.
WARM_CONCURRENCY ?= 16
WARM_REQS ?= 32

## bench/replay.py needs aiohttp, which the GPU boxes do not have (no pip either), so every
## previous iteration hand-rolled a `docker run` around it. The served image already ships
## aiohttp 3.14.1 -- run the repo's own driver in it. --network host so $(TARGET)'s localhost
## resolves; the round dir is mounted at /w so --trace/--out paths stay repo-relative.
REPLAY ?= docker run --rm --network host -v "$$PWD:/w" -w /w --entrypoint python3 $(IMAGE):$(TAG)
warm:
	$(IN) $(BUILD)
	$(IN) $(DC) up -d --wait   # --wait blocks until the healthcheck passes
	$(IN) $(REPLAY) bench/replay.py --target $(TARGET) --trace $(TRACE) --limit 4 --out /dev/null
	$(IN) $(REPLAY) bench/replay.py --target $(TARGET) --trace $(TRACE) \
	  --closed-loop $(WARM_CONCURRENCY) --limit $(WARM_REQS) --out /dev/null
	$(IN) docker cp "$$($(DC) ps -q model)":/opt/vtl/cache/. docker/cache/
	$(IN) du -sh docker/cache/*
	@# The harvest is silent when it comes back short, and a missing triton/ is invisible until
	@# the judge box JIT-compiles every Triton kernel inline (measured 2026-07-25: ~0.25s of a
	@# 0.70s decode window went to code_generator.visit + make_ptx + a forked ptxas). The image
	@# built on 2026-07-25 baked ONLY cache/vllm -- triton/ and inductor/ were never harvested.
	@# NOTE both caches are keyed on GPU arch: a harvest from a dev box (sm_86) MISSES on the
	@# judge's H200 (sm_90). `make warm` has to run on the target arch to be worth anything.
	$(IN) test -d docker/cache/triton || { echo "make warm: NOTHING harvested into docker/cache/triton -- the image would ship an empty Triton cache"; exit 1; }
	$(IN) $(DC) down
	$(MAKE) build ROUND=$(ROUND)

## buildx --push writes the manifest straight to the registry, so the pushed image
## is $(PLATFORM) regardless of what this machine is.
push:
	$(IN) docker buildx build $(BUILDX_FLAGS) --platform $(PLATFORM) --build-arg VLLM_IMAGE='$(VLLM_IMAGE)' --build-arg CUDA_ARCHS='$(CUDA_ARCHS)' --build-arg MAX_JOBS='$(MAX_JOBS)' --push -t $(IMAGE):$(TAG) .
	@echo "pin this digest in $(ROUND)/docker-compose.yaml:"
	@docker buildx imagetools inspect $(IMAGE):$(TAG) --format '{{.Manifest.Digest}}'

## Open-loop replay (honors the trace's arrival times) + a closed-loop sweep.
bench:
	$(IN) $(REPLAY) bench/replay.py --target $(TARGET) --trace $(TRACE) --out bench-open.json
	$(IN) for n in 1 8 32 128; do \
	  $(REPLAY) bench/replay.py --target $(TARGET) --trace $(TRACE) \
	    --closed-loop $$n --out bench-closed-$$n.json; \
	done

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
	    && $(REPLAY) bench/replay.py --target $(TARGET) --trace $(TRACE) \
	         --out bench-sched-$$name.json \
	    ; rc=$$?; VTL_W4A8_SCHEDULE="$$s" $(DC) down; exit $$rc ) || exit 1; \
	done
	@echo "compare: $(ROUND)/bench-sched-*.json (heuristic is the baseline)"

## Phase-0 profiler (needs the H200). Boots with vLLM's torch profiler enabled, drives a small
## closed-loop replay, and prints a ranked GPU-kernel cost table plus the host-vs-GPU split
## (gpu-busy vs idle per step -- the number the TPOT tuning rests on). See docs/plans/.
## Must match VTL_PROFILE_STEPS in docker-compose.profile.yaml; only divides totals per step.
PROFILE_STEPS ?= 20
## Host-slack probe. Answers the ONE question a profile cannot: would saving host time move
## TPOT at all? Async scheduling overlaps step N+1's host work with step N's GPU work, so the
## step period is max(host, gpu) and every host micro-optimization is worth exactly zero until
## host is the longer side. A "GPU busy 65%" figure from `make profile` is a union of kernel
## spans and reads the same either way -- it does NOT identify the critical path.
##
## This burns a known number of microseconds on the engine thread before each step
## (vtl/patches/profiler.py, VTL_STEP_DELAY_US) and sweeps it. Read the slope of TPOT vs delay:
##   slope ~= 1   host IS the critical path; 1 ms saved on the host is 1 ms of TPOT
##   flat, then slope 1 at delay D   there is D of host slack; nothing below D is worth doing
##
## One boot for the whole sweep -- boot-to-boot variance is the dominant noise here (see the
## docker-compose TPOT block), and the delay is a runtime file, not an env var. The repeated
## first arm at the end is the drift control: if it does not reproduce arm 1, the sweep
## measured warm-up rather than the delay.
## Each arm is "<overlapped>/<serial>" us. The two halves are NOT equivalent: execute_model
## runs for step N+1 while step N is still on the GPU, so it has slack; update_from_output runs
## after the engine has already blocked on step N's tokens, so it has none. Sweeping only the
## overlapped half reads "host work is free" and is how you talk yourself out of a real win.
STEP_DELAYS ?= 0/0 500/0 1000/0 2000/0 0/500 0/1000 0/0
## First N trace records, arrival times preserved. The full 420 take ~5 min per arm; the point
## is the SLOPE, and the trace's decode shape (mean 2 concurrent) is stationary after the first
## few turns, so a prefix measures the same thing in a third of the time.
SLACK_LIMIT ?= 150
slack:
	$(IN) mkdir -p bench-profile
	$(IN) $(BUILD)
	$(IN) $(DC) -f docker-compose.profile.yaml up -d --force-recreate --wait
	$(IN) i=0; for d in $(STEP_DELAYS); do \
	  i=$$((i+1)); o=$${d%%/*}; ser=$${d##*/}; \
	  echo "$$o" > bench-profile/delay_us; echo "$$ser" > bench-profile/delay_serial_us; sleep 2; \
	  echo "=== arm $$i: overlapped=$$o us serial=$$ser us"; \
	  $(REPLAY) bench/replay.py --target $(TARGET) --trace $(TRACE) \
	    --limit $(SLACK_LIMIT) --out bench-slack-$$i-$${o}o-$${ser}s.json || exit 1; \
	done
	$(IN) rm -f bench-profile/delay_us bench-profile/delay_serial_us
	$(IN) $(DC) -f docker-compose.profile.yaml down

## A/B one env var against the SCORED open-loop trace. Iterations 4, 5 and 6 each hand-rolled
## this same driver, so it lives here now, with the three things each of them had to learn:
##
##   1. DISCARD THE FIRST REP after a boot. It reads ~8.5-9 ms TPOT vs ~6.5 warm on the same
##      server, so a single post-boot run measures warm-up, not the change.
##   2. ALTERNATE THE ARMS ACROSS BOOTS, AND USE ENOUGH OF THEM. Measured 2026-07-25 with this
##      target: two boots of the SAME arm, same image, same env, differed by 0.53 ms of TPOT p50
##      (6.64 vs 6.10) while the two reps WITHIN each boot differed by 0.00-0.11 ms. So rep count
##      buys you almost nothing and boot count is the only thing that shrinks the error bar --
##      the resolution of an A/B is roughly 0.5 ms / sqrt(boots per arm). Anything measured over
##      one boot per arm, at this or any earlier iteration, resolves nothing below ~0.5 ms.
##   3. BIAS PAST THE HOST KNEE. `make slack` measured ~1 ms of host slack on the dev box, so a
##      host-side change smaller than that is absorbed and reads exactly zero here. Running both
##      arms at VTL_STEP_DELAY_US=2000 makes both host-critical, where a real saving shows at
##      slope 1. The delay-0 rows are the control: they say what this box would have measured.
##
## Needs the profile overlay (that is what installs the delay probe) but never arms the capture,
## so no trace is written. Arms are proved by inspecting the container env, because the log line
## a patch prints is INFO and the submission compose pins WARNING.
##
##   make ab AB_ENV=VTL_ENABLE_KV_CACHE_GROUPS                      # patch on vs off
##   make ab AB_ENV=VTL_QUANT AB_ARMS="vtl_w4a8 vtl_fp8" AB_DELAYS=0
##
## ~14 min per boot at the defaults (2.5 boot + 5 reps x ~2.3 min), so the default 4 boots is
## ~55 min. Resolving 0.25 ms needs ~4 boots per arm (AB_ROUNDS="1 2 3 4"), i.e. ~2 hours.
##
## Results: $(ROUND)/bench-ab-<round>-<arm>-<delay>-<rep>.json, one summary line per rep.
AB_ENV ?=
AB_ARMS ?= 1 0
AB_DELAYS ?= 2000 0
## Reps within a boot. 2 is plenty: within-boot spread is ~0.05 ms, an order of magnitude below
## the between-boot spread. Add boots (AB_ROUNDS), not reps, when you need resolution.
AB_REPS ?= 2
AB_ROUNDS ?= 1 2
AB_LIMIT ?= 150
ab:
	@test -n "$(AB_ENV)" || { echo "make ab: set AB_ENV=<env var to sweep>"; exit 1; }
	$(IN) mkdir -p bench-profile
	$(IN) $(BUILD)
	$(IN) for r in $(AB_ROUNDS); do for a in $(AB_ARMS); do \
	  printf 'services:\n  model:\n    environment:\n      %s: "%s"\n' "$(AB_ENV)" "$$a" > /tmp/vtl-ab-arm.yaml; \
	  $(DC) -f docker-compose.profile.yaml -f /tmp/vtl-ab-arm.yaml up -d --force-recreate --wait || exit 1; \
	  echo "=== round $$r arm $(AB_ENV)=$$(docker inspect $$($(DC) ps -q model) \
	    --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^$(AB_ENV)=//p')"; \
	  echo 0 > bench-profile/delay_us; \
	  $(REPLAY) bench/replay.py --target $(TARGET) --trace $(TRACE) \
	    --limit $(AB_LIMIT) --out /dev/null || exit 1; \
	  for d in $(AB_DELAYS); do echo "$$d" > bench-profile/delay_us; sleep 2; \
	    i=0; while [ $$i -lt $(AB_REPS) ]; do i=$$((i+1)); \
	      o=bench-ab-$$r-$$a-$${d}us-$$i.json; \
	      $(REPLAY) bench/replay.py --target $(TARGET) --trace $(TRACE) \
	        --limit $(AB_LIMIT) --out $$o || exit 1; \
	      python3 bench/metrics.py --summarize $$o || exit 1; \
	    done; \
	  done; \
	done; done
	$(IN) rm -f bench-profile/delay_us
	$(IN) $(DC) -f docker-compose.profile.yaml down
	@echo "compare: $(ROUND)/bench-ab-*.json -- read the $(firstword $(AB_DELAYS))us rows, the 0us rows are the control"

## Seconds into the replay at which the capture window opens. Anything that arms BEFORE the
## replay starts captures the opening prefill burst instead of steady-state decode.
PROFILE_ARM_DELAY ?= 90
profile:
	$(IN) mkdir -p bench-profile
	$(IN) rm -f bench-profile/vtl-trace-*.json bench-profile/vtl-pyprof-*.txt bench-profile/.arm
	$(IN) $(BUILD)
	$(IN) $(DC) -f docker-compose.profile.yaml up -d --force-recreate --wait
	@# Drive the REAL trace open-loop and arm mid-run. The old driver was
	@# `--closed-loop 8 --limit 48`, which saturates the server: most steps become
	@# prefill-mixed, mamba refuses a FULL cudagraph on those, and the model forward runs
	@# PIECEWISE -- so short_conv/marlin_gemm/unified_attention showed up as host time and the
	@# ranking was of a workload the judge never generates (trace concurrency is mean 2 / peak 8).
	$(IN) ( $(REPLAY) bench/replay.py --target $(TARGET) --trace $(TRACE) --out /dev/null & \
	        sleep $(PROFILE_ARM_DELAY); touch bench-profile/.arm; wait )
	$(IN) sleep 3   # let the worker finish export_chrome_trace
	$(IN) python3 bench/profile_trace.py --profile-dir bench-profile --steps $(PROFILE_STEPS) --out bench-profile-summary.json
	$(IN) $(DC) -f docker-compose.profile.yaml down
