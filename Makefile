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
DC := docker compose $(COMPOSE_FILES)

.PHONY: check stats build up down warm push bench sweep-schedule profile test-kernel bench-kernel debug-kernel verify vllm-fork

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
	$(IN) python3 bench/replay.py --target $(TARGET) --trace $(TRACE) --out bench-open.json
	$(IN) for n in 1 8 32 128; do \
	  python3 bench/replay.py --target $(TARGET) --trace $(TRACE) \
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
