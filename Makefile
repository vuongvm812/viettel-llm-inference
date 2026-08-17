# Root dispatcher. The buildable project lives in per-round folders; this Makefile drives
# whichever round you select and keeps the shared model weights (hf-model/) and vendored engine
# repos at the root so they are not duplicated per round.
#
#   round-1.1/  frozen Qwen3.5-2B baseline (phase-1)
#   round-1.2/  LFM2.5-1.2B refactor       (phase-2)
#   round-2/    model-agnostic workspace   (phase-3, default; no model baked in)
#
# Pick the round with ROUND=... on any target:
#   make up                      # round-2 (default)
#   make up ROUND=round-1.2      # the LFM2.5 phase-2 stack
#   make test-kernel ROUND=round-1.1
ROUND ?= round-2
ifeq ($(wildcard $(ROUND)/.),)
$(error ROUND '$(ROUND)' not found -- use ROUND=round-1.1, round-1.2 or round-2)
endif

# Per-round overrides, included BEFORE every default below so a plain `?=` cannot clobber them.
# This is how a round pins its OWN forked-vLLM digest: the fork is built from that round's
# vtl/vllm_patches/, so two rounds with different patch sets cannot share one VLLM_FORK_TAG.
# Optional -- a round without a round.mk just takes the defaults.
-include $(ROUND)/round.mk

# traitimbanggia, not unseenablefuture: the old account belongs to the REMOTE teammate, and
# round 2 is pushed from the venue where nobody holds those credentials. Public repo -- the
# judge pulls the submission pin anonymously. Frozen rounds keep their old-account digest pins.
IMAGE ?= traitimbanggia/yasuoadc
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
VLLM_FORK_IMAGE ?= traitimbanggia/slowleveling
# latest@a44447ac is a byte-identical mirror of the old unseenablefuture/vllm-fork
# v0.25.0-tree@a41d4237 pin (all 38 layer digests verified equal; only the manifest digest
# changed, as re-pushing re-serializes it). Account move, not a content change.
VLLM_FORK_TAG ?= latest@sha256:a44447acf529bb7c5a48ee454bd36bebfb4f727f92e13c80c25ffecb5dec7dc4
# Base image the MAIN image builds FROM. Defaults to the fork above so build/up/warm run the
# tree-spec vLLM. Stock build (or the round-1.1 baseline): make ... VLLM_IMAGE=$(VLLM_STOCK)
VLLM_IMAGE ?= $(VLLM_FORK_IMAGE):$(VLLM_FORK_TAG)
# 1 = the fork's rust-builder stage does a profile-guided-optimization build of vllm-rs
# (CPU-only training run against the mock engine). 0 = plain optimized (fat-LTO) build.
VLLM_RS_PGO ?= 1
# Tokenizer the PGO training run boots the frontend with. Defaults to the local model
# (hf-model/, bind-mounted at /model in the fork's PGO stage — only the tokenizer/config
# JSONs are read, the mock fakes the forward pass so the round-2 checkpoint's 127.2 GB of
# weights are never loaded).
# Override with a HF repo id to fetch a stand-in over the network, e.g. PGO_MODEL=Qwen/Qwen3-0.6B.
PGO_MODEL ?= /model
# Host path to the local model dir the PGO training run reads. NOT passed to buildx
# directly: the Dockerfile's `RUN --mount=type=bind,from=hfmodel` mounts the context ROOT,
# so buildx would sync the whole directory into BuildKit — on a GPU host after a full
# `make model-fetch` that ships all 127.2 GB of weights into the build cache for a stage
# that reads ~25 MB of JSONs. The pgo-hfmodel-ctx target stages the metadata subset (same
# patterns as fetch-model.sh --meta-only) into $(PGO_HFMODEL_CTX) and buildx gets THAT.
PGO_HFMODEL ?= /home/team17/Qwen3.5-122B-A10B-FP8
# Staged metadata-only build context (recreated on every PGO build; gitignored).
PGO_HFMODEL_CTX ?= ../.pgo-hfmodel-ctx
# -Ctarget-cpu for the vllm-rs binary (plain AND PGO builds). Default sapphirerapids: the
# round-2 deploy host is a Xeon Platinum 8558 (Emerald Rapids; sapphirerapids is the newest
# LLVM target that ISA fully covers — AVX-512 + AMX). Pinning the target instead of `native`
# means the same optimized binary comes out of ANY build box, including an emulated one.
# For an emulated build that must stay portable, override PGO_TARGET_CPU=x86-64-v3 or empty
# (a too-new instrumented binary can crash the PGO training replay on an older build CPU).
PGO_TARGET_CPU ?= sapphirerapids

# All paths below are relative to the selected round. `IN` cd's into it so docker-compose build
# contexts, relative volume mounts, and `docker cp` cache paths all resolve inside the round.
IN := cd $(ROUND) &&
TRACE ?= data/input/trace-round2.jsonl
# Grading-fidelity bench (round-2): NVIDIA aiperf's AgentX MVP scenario, mirroring the BTC's
# published invocation. Concurrency/context/dataset are the BTC's confirmed values; the seed is
# hidden on their side, so ours defaults to 0 and should be swept to bound seed sensitivity.
AIPERF ?= aiperf
AIPERF_SEED ?= 0
AIPERF_DURATION ?= 900
AIPERF_CONCURRENCY ?= 5
AIPERF_DATASET ?= semianalysis_cc_traces_weka_062126
# The stable alias from docker-compose.yaml's --served-model-name, so the same command works
# regardless of which real model the compose currently pins.
AIPERF_MODEL ?= round2-model
AIPERF_LIMIT ?=# set to N for a smoke run (--num-dataset-entries N; scenario docs: smoke only)
AIPERF_ART ?= bench-aiperf
# docker-compose.yaml is the SUBMISSION artifact and the single source of truth for every
# serve flag and env var; the three overlays only carry their differences (dev image tag,
# local build+mount, judge-box resource caps). Order matters -- later -f wins.
COMPOSE_FILES := -f docker-compose.yaml -f docker-compose-optimized.yaml -f docker-compose.localtest.yaml -f docker-compose.cpucap.yaml
DC := docker compose $(COMPOSE_FILES)

# CI bench lifecycle (remote runner). No build step — image is the pinned digest from ci-build,
# or :dev when the runner built it itself (Case 1: build and bench are the same box).
IMAGE_DIGEST ?=
# docker-compose.yaml MUST lead. docker-compose-optimized.yaml says so in its own header -- it is
# an overlay carrying only the :dev image tag, NOT a standalone stack. Without the base file the
# service comes up with no entrypoint, no serve flags, no env and no healthcheck (so `--wait`
# returns on a container that never became a server), AND under a different compose project name
# -- the base pins `name: viettel-llm-optimized`, while an overlay-only stack takes its name from
# the directory. `make verify` uses $(DC), which does include the base, so it would then read the
# logs of a project that does not exist and report a dead server.
# Deliberately NOT localtest/cpucap: no `build:` (the image is already here) and no resource cap.
CIBENCH_COMPOSE := -f docker-compose.yaml -f docker-compose-optimized.yaml -f docker-compose.ci-bench.yaml
_CI_IMAGE = $(if $(IMAGE_DIGEST),$(IMAGE)@$(IMAGE_DIGEST),$(IMAGE):$(TAG))

.PHONY: pgo-hfmodel-ctx check stats build up down warm push bench bench-aiperf sweep-schedule sweep-schedule-micro profile test-kernel bench-kernel debug-kernel verify vllm-fork ci-build ci-digest ci-watch ci-status ci-up ci-down ci-bench ci-bootstrap host-tune host-tune-reset host-tune-show model-fetch model-fetch-meta vllm-src

## Host-level latency tuning for the DEV/BENCH box. NOT part of any submission -- the judge
## runs `docker compose up` on their host and none of these knobs are reachable from a
## compose file. The point is measurement quality: unlocked GPU clocks make two identical
## arms differ by more than the effect being measured. Round-independent, so no ROUND=.
## ALWAYS run host-tune-reset when you are done -- a box left with clocks locked lies about
## power and thermals for whoever uses it next.
host-tune:
	sudo scripts/host-tune.sh apply
host-tune-reset:
	sudo scripts/host-tune.sh reset
host-tune-show:
	@scripts/host-tune.sh show

## Per-host setup: model weights + pristine vLLM source. Both land at repo root (gitignored:
## /hf-model/ and vllm/) so rounds share one copy; the compose overlays bind-mount hf-model/
## at /model and PGO_HFMODEL defaults to it.
##
## model-fetch is for the GPU host ONLY -- the round-2 checkpoint (Qwen3.5-122B-A10B-FP8) is
## 127.2 GB / 39 shards and the script refuses to start without 140 GB free. model-fetch-meta
## grabs just the ~25 MB of config/tokenizer JSONs, which is all that trace building, planning
## and the PGO frontend boot need -- it runs anywhere. Resumable; override MODEL_ID /
## HF_MODEL_DIR / HF_MAX_WORKERS in the environment (see scripts/fetch-model.sh).
model-fetch:
	scripts/fetch-model.sh
model-fetch-meta:
	scripts/fetch-model.sh --meta-only

## Pristine vLLM v0.25.0 source tree at repo-root vllm/ (gitignored). This is the REFERENCE
## tree that $(ROUND)/vtl/vllm_patches/gen.sh diffs against (its V025=... path) when
## regenerating patches -- never edit it, never build from it; the runnable engine is the
## $(VLLM_STOCK) image. Idempotent: re-running verifies the checkout matches the v0.25.0 tag.
vllm-src:
	@if [ -d vllm/.git ]; then \
	  tag=$$(git -C vllm describe --tags --exact-match 2>/dev/null || git -C vllm rev-parse --short HEAD); \
	  if [ "$$tag" = "v0.25.0" ]; then echo "vllm/ already present at v0.25.0 -- nothing to do"; \
	  else echo "WARN vllm/ exists but is at '$$tag', not v0.25.0 -- remove it and re-run"; exit 1; fi; \
	else \
	  git clone --depth 1 --branch v0.25.0 https://github.com/vllm-project/vllm vllm; \
	fi

## Self-checks. Run anywhere: no GPU, no vLLM, no running server. Adapts to the round's patch set
## (round-1.1 has the GDN patches; round-1.2 does not) by globbing rather than hardcoding names.
check:
	$(IN) python3 vtl/registry.py
	$(IN) for f in vtl/patches/*.py; do \
	  case "$$f" in */__init__.py) continue;; esac; \
	  echo "-- $$f"; PYTHONPATH=. python3 "$$f" || exit 1; \
	done
	@# grep-guarded: older rounds' healthcheck has no --selfcheck flag.
	$(IN) if grep -q selfcheck vtl/warmup_healthcheck.py; then python3 vtl/warmup_healthcheck.py --selfcheck; fi
	@# NVRTC harness (round-2+). The pure half only -- cache keys, gating, arg packing.
	@# The compile+numerics half is bench/test_nvrtc.py under `make test-kernel` (needs a GPU).
	$(IN) if [ -f vtl/nvrtc.py ]; then PYTHONPATH=. python3 bench/test_nvrtc.py --self-check; fi
	@# GDN gated-RMSNorm harness (rounds that ship the GDN kernels). Same split: the pure half
	@# (NVRTC define/cache-key contract, the two quant oracles' shape+eps contracts) here, the
	@# kernel parity half under `make test-kernel`.
	@# grep-guarded as well as -f guarded: round-1.1 ships this file WITHOUT a --self-check half.
	$(IN) if [ -f bench/test_gdn_gated_rmsnorm.py ] && grep -q -- --self-check bench/test_gdn_gated_rmsnorm.py; then \
	  PYTHONPATH=. python3 bench/test_gdn_gated_rmsnorm.py --self-check; fi
	@# The two NVRTC production consumers (round-2+): group-128 block quant, and the fused
	@# GDN decode step. Pure half here -- kernel<->patch define agreement, cubin cache-key
	@# distinctness, the engage predicate, and that the patch imports and stays OFF without
	@# vLLM. The parity halves need a GPU and run under `make test-kernel`.
	$(IN) if [ -f bench/test_nvrtc_block_quant.py ]; then \
	  PYTHONPATH=. python3 bench/test_nvrtc_block_quant.py --self-check; fi
	$(IN) if [ -f bench/test_gdn_decode_step.py ]; then \
	  PYTHONPATH=. python3 bench/test_gdn_decode_step.py --self-check; fi
	@# Fused greedy argmax (round-2+). Pure half here -- kernel entry <-> op name <-> the
	@# name the forked V2 sampler resolves, nstep's three call sites going through _ARGMAX,
	@# and cubin cache-key distinctness across {VOCAB, THREADS}. Bit-parity against
	@# torch.argmax needs a GPU and runs under `make test-kernel`.
	$(IN) if [ -f bench/test_greedy_argmax.py ]; then \
	  PYTHONPATH=. python3 bench/test_greedy_argmax.py --self-check; fi
	@# The int4 track (round-2+): the fp8-block -> int4 requant of the dense layers, and the
	@# MoE decode grouped-GEMV. Pure half here -- the double-quantization error bound in numpy,
	@# the (token, slot) bookkeeping, and cubin cache-key distinctness across the int4/fp8
	@# weight arms. The kernel parity halves need a GPU and run under `make test-kernel`.
	$(IN) if [ -f bench/test_w4a8_from_fp8.py ]; then \
	  PYTHONPATH=. python3 bench/test_w4a8_from_fp8.py --self-check; fi
	$(IN) if [ -f bench/test_moe_decode.py ]; then \
	  PYTHONPATH=. python3 bench/test_moe_decode.py --self-check; fi
	$(IN) python3 bench/trace_stats.py --self-check
	$(IN) python3 bench/metrics.py
	$(IN) python3 bench/sweep_report.py --selfcheck
	@# aiperf -> repo-schema converter. Parses files only, so it runs off-box without aiperf.
	$(IN) if [ -f bench/aiperf_adapter.py ]; then python3 bench/aiperf_adapter.py --selfcheck; fi
	$(IN) python3 bench/eval_quality.py --self-check
	$(IN) python3 bench/profile_trace.py --self-check
	@# if-form, not `[ -f x ] && cmd || true`: that swallows the script's OWN failure as well as
	@# its absence, i.e. a broken self-check reports green. Absent = fine, present = must pass.
	$(IN) if [ -f bench/build_trace_round2.py ]; then PYTHONPATH=. python3 bench/build_trace_round2.py --self-check; fi
	@# Formula half runs anywhere; the live-allocator half needs CUDA and skips off-box
	@# (it runs for real under `make test-kernel`, which pytest-globs bench/test_*.py).
	$(IN) if [ -f bench/test_kv_alignment.py ]; then python3 bench/test_kv_alignment.py; fi
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
	@# The Rust scheduler covers exactly two KV cache-spec kinds (full attention, mamba) and
	@# REFUSES anything else -- sliding-window, chunked-local, cross-attention -- by logging one
	@# line and handing back to stock vLLM. That is the right default for serving and the worst
	@# possible default for measurement: a whole scheduler not running is indistinguishable, in
	@# every latency number, from one that ran and did not help. So name it here. Set
	@# VTL_RUST_SCHED_REQUIRE=1 to turn the refusal into a boot failure instead.
	@if grep -q "rust_sched: NOT ENGAGED" /tmp/vtl-verify.log; then \
	   echo "WARN rust scheduler NOT ENGAGED: $$(sed -n 's/.*rust_sched: NOT ENGAGED -- //p' /tmp/vtl-verify.log | tail -1)"; \
	   echo "     you are measuring the stock scheduler. VTL_RUST_SCHED_REQUIRE=1 makes this fatal."; \
	 elif grep -q "rust_sched: AUTHORITY mode active" /tmp/vtl-verify.log; then \
	   echo "OK   rust scheduler engaged: $$(grep -oE 'rust_sched: AUTHORITY mode active \([^)]*\)' /tmp/vtl-verify.log | tail -1)"; \
	 else \
	   echo "WARN rust scheduler state unknown -- no mode selected, or logging below INFO"; \
	 fi
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
# Stage a metadata-only copy of $(PGO_HFMODEL) for the buildx named context. The PGO stage
# only reads the tokenizer/config JSONs (the frontend has no weight loader at all), but a
# `RUN --mount=type=bind,from=hfmodel` makes buildx sync the WHOLE context to BuildKit —
# with the full round-2 checkpoint in hf-model/ that is a 127.2 GB transfer plus a second
# on-disk copy in the build cache, for ~25 MB actually used. Include patterns mirror
# fetch-model.sh --meta-only (plus *.jinja for the dedicated chat template file).
pgo-hfmodel-ctx:
	$(IN) test -f "$(PGO_HFMODEL)/config.json" || { echo "FAIL: $(PGO_HFMODEL)/config.json missing -- run 'make model-fetch-meta' (~25 MB) first or point PGO_HFMODEL at a model dir"; exit 1; }
	$(IN) rm -rf "$(PGO_HFMODEL_CTX)" && mkdir -p "$(PGO_HFMODEL_CTX)" && find "$(PGO_HFMODEL)" -maxdepth 1 -type f \( -name '*.json' -o -name '*.jinja' -o -name '*.txt' -o -name 'tokenizer*' -o -name 'vocab*' -o -name 'merges*' \) -exec cp {} "$(PGO_HFMODEL_CTX)/" \;
	@$(IN) du -sh "$(PGO_HFMODEL_CTX)" | sed 's/^/staged metadata-only PGO context: /'

vllm-fork: $(if $(filter 1,$(VLLM_RS_PGO)),pgo-hfmodel-ctx)
	$(IN) docker buildx build $(BUILDX_FLAGS) --platform $(PLATFORM) --build-arg NO_PROXY=localhost,127.0.0.1 --build-arg no_proxy=localhost,127.0.0.1 --build-arg http_proxy="$(HTTP_PROXY)" --build-arg https_proxy="$(HTTPS_PROXY)" --build-arg HTTP_PROXY="$(HTTP_PROXY)" --build-arg HTTPS_PROXY="$(HTTPS_PROXY)" --build-arg VLLM_IMAGE='$(VLLM_STOCK)' --build-arg PGO_MODEL='$(PGO_MODEL)' --build-arg PGO_TARGET_CPU='$(PGO_TARGET_CPU)' $(if $(filter 1,$(VLLM_RS_PGO)),--build-arg RUST_BUILDER=rust-builder-pgo --build-context hfmodel=$(PGO_HFMODEL_CTX)) $(if $(PUSH),--push,--load) -t $(VLLM_FORK_IMAGE):$(VLLM_FORK_TAG) -f Dockerfile.vllm-fork .
	@echo "forked vLLM base: $(VLLM_FORK_IMAGE):$(VLLM_FORK_TAG)"
	@if [ -n "$(PUSH)" ]; then $(IN) docker buildx imagetools inspect $(VLLM_FORK_IMAGE):$(VLLM_FORK_TAG) --format 'pin this digest: {{.Manifest.Digest}}'; fi

build:
	$(IN) docker buildx build $(BUILDX_FLAGS) --platform $(PLATFORM) --build-arg http_proxy="$(HTTP_PROXY)" --build-arg https_proxy="$(HTTPS_PROXY)" --build-arg HTTP_PROXY="$(HTTP_PROXY)" --build-arg HTTPS_PROXY="$(HTTPS_PROXY)" --build-arg VLLM_IMAGE='$(VLLM_IMAGE)' --build-arg CUDA_ARCHS='$(CUDA_ARCHS)' --build-arg MAX_JOBS='$(MAX_JOBS)' --load -t $(IMAGE):$(TAG) .
	@docker inspect $(IMAGE):$(TAG) --format 'built {{.Os}}/{{.Architecture}}'

# `docker compose up --build` cannot take --build-arg, so build first (which honors it) then up.
up:
	$(IN) $(DC) build --build-arg http_proxy="$(HTTP_PROXY)" --build-arg https_proxy="$(HTTPS_PROXY)" --build-arg HTTP_PROXY="$(HTTP_PROXY)" --build-arg HTTPS_PROXY="$(HTTPS_PROXY)" --build-arg VLLM_IMAGE='$(VLLM_IMAGE)'
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

## Grading-fidelity bench (round-2, H200 only): replays the SemiAnalysis Weka corpus of real
## Claude Code sessions via aiperf's AgentX MVP scenario -- the workload the BTC actually
## grades with -- then converts the artifacts into the repo schema and prints the ERS report.
## Install: pip install -r round-2/bench/requirements-aiperf.txt (heavy; dataset auto-downloads
## from HF on first run). Smoke: make bench-aiperf AIPERF_LIMIT=8 AIPERF_DURATION=120.
## Two things are confirmed-at-first-run and may need adjusting: (a) the scenario preset may
## already own some of the explicit flags below (if aiperf rejects one as a duplicate/conflict,
## drop the explicit copy here and note which); (b) the artifact-dir layout -- the adapter
## searches recursively, so a flat layout also works.
bench-aiperf:
	$(IN) $(AIPERF) profile --scenario inferencex-agentx-mvp \
	  --model $(AIPERF_MODEL) --url $(TARGET) --endpoint-type chat \
	  --public-dataset $(AIPERF_DATASET) \
	  --concurrency $(AIPERF_CONCURRENCY) --max-context-length 204800 \
	  --benchmark-duration $(AIPERF_DURATION) --random-seed $(AIPERF_SEED) \
	  --streaming --extra-inputs ignore_eos:true --cache-bust first_turn_prefix \
	  --system-idle-gap-cap-seconds 10 --use-server-token-count \
	  $(if $(AIPERF_LIMIT),--num-dataset-entries $(AIPERF_LIMIT)) \
	  --artifact-dir $(AIPERF_ART)
	$(IN) python3 bench/aiperf_adapter.py --artifact-dir $(AIPERF_ART) --out bench-aiperf.json
	$(IN) python3 bench/_ci_report.py

## Sweep the CUTLASS W4A8 tile/cluster schedule (needs the GPU). The kernel's own heuristic
## (w4a8_mm_entry.cu:341-372) keys ONLY on M/N/K and is blind to SM count -- it was tuned on a
## full 132-SM Hopper and the judge's MIG 1g.18gb slice has ~19, so its choice is an open
## question, not a settled one. This is the one already-exposed W4A8 tunable never swept.
## `heuristic` is the sentinel for "unset", the baseline every other row must beat -- a literal
## empty word cannot survive make -> shell word splitting. All ten stock tiles are swept, not
## just the three the heuristic picks for our shapes: the point is that its M/N/K-only rule was
## tuned on hardware this box is not.
##
##   make sweep-schedule                    # heuristic + all 10 stock tiles, 3 boots each
##   make sweep-schedule BOOTS=5            # add BOOTS, not reps -- the floor is boot-to-boot
##   make sweep-schedule V2=1               # + the vtl._C_w4a8 Stream-K/cluster/pingpong arms
##   make sweep-schedule MICRO=1            # microbench every arm FIRST, then boot (see below)
##   make sweep-schedule SCHEDULES=128x16_1x1x1   # one arm, no baseline
##
## COMPILE COST: either schedule env changes the kernel selected for every W4A8 GEMM, which
## invalidates the AOT/inductor caches baked into /opt/vtl/cache -- the FIRST boot of each arm
## pays that compile, mostly in TTFT. That is why 3 boots is the default and why sweep_report.py
## reads the noise floor off the baseline arm's own boot spread instead of a constant.
##
## ENV CONTRACT: a stock name goes to VTL_W4A8_SCHEDULE (vtl/patches/quant_w4a8.py). Anything
## else -- *_sk, *_pp, 128x8_1x1x1, cluster variants beyond the stock ten -- is a v2 schedule
## from the vtl._C_w4a8 extension and goes to VTL_W4A8_SCHEDULE_V2 instead, with
## VTL_W4A8_SCHEDULE left EMPTY so exactly one knob steers the kernel. This target only sets the
## envs; it does not require the v2 extension to exist (an image without it ignores the var and
## the arm reads as a duplicate baseline -- which is itself the check that the .so is loaded).
##
## docker-compose.yaml pins VTL_W4A8_SCHEDULE as a literal (the submission artifact takes no
## ${VAR}) and a compose literal beats the host env, so each arm is driven by a throwaway
## overlay instead. Exporting the var on the host, as this target used to, does nothing.
STOCK_SCHEDULES := 256x128_1x1x1 256x64_1x1x1 256x32_1x1x1 256x16_1x1x1 128x256_2x1x1 \
                   128x256_1x1x1 128x128_1x1x1 128x64_1x1x1 128x32_1x1x1 128x16_1x1x1
## WS2's planned v2 instantiations. Opt-in (V2=1): they need the sm_90a vtl._C_w4a8 build, and
## sweeping them against an image that lacks it just burns boots.
V2_SCHEDULES ?= 128x16_1x1x1_sk 128x32_1x1x1_sk 128x16_2x1x1 128x8_1x1x1 \
                64x16_1x1x1_pp 64x32_1x1x1_pp \
                128x16_1x1x1_sk_nd 128x32_1x1x1_sk_nd \
                128x16_1x1x1_splitk4 128x32_1x1x1_splitk4 \
                128x16_1x1x1_s4 128x16_1x1x1_s8
## Prefill-band arms. NOT in the default sweep: they select a different env knob
## (VTL_W4A8_SCHEDULE_V2_PREFILL) and only fire when VTL_W4A8_V2_PREFILL_MAX is non-zero, so
## sweeping them alongside the decode arms would silently measure the baseline. Sweep with
## `make sweep-schedule PREFILL=1`, which sets both.
V2_PREFILL_SCHEDULES ?= 128x128_1x2x1_pf 128x128_1x1x1_pf
SCHEDULES ?= heuristic $(STOCK_SCHEDULES) $(if $(V2),$(V2_SCHEDULES)) \
             $(if $(PREFILL),$(V2_PREFILL_SCHEDULES))
BOOTS ?= 3
# Named for the knob it sweeps, NOT /tmp/vtl-sched.yaml -- that reads like the vtl-sched Rust
# crate (WS4), which has nothing to do with W4A8 tiles.
SWEEP_OVERLAY := /tmp/vtl-w4a8-sweep.yaml
# Upper bound of the prefill band while sweeping a *_pf arm. Only applied to those arms: left at
# 0 the band cannot fire at all, which is what keeps every other arm a clean decode measurement.
PREFILL_MAX ?= 1024
# One line, duplicated in the micro target rather than abstracted: stock name -> $$s, heuristic
# -> neither, a *_pf name -> $$pf (a different env knob), anything else -> $$v2.
SWEEP_CLASSIFY = s=""; v2=""; pf=""; pfmax=0; \
	  case " $(STOCK_SCHEDULES) " in *" $$name "*) s="$$name";; \
	  *) case "$$name" in heuristic) ;; \
	     *_pf) pf="$$name"; pfmax=$(PREFILL_MAX);; \
	     *) v2="$$name";; esac;; esac
sweep-schedule: $(if $(MICRO),sweep-schedule-micro)
	@for name in $(SCHEDULES); do \
	  $(SWEEP_CLASSIFY); \
	  printf 'services:\n  model:\n    environment:\n      VTL_W4A8_SCHEDULE: "%s"\n      VTL_W4A8_SCHEDULE_V2: "%s"\n      VTL_W4A8_SCHEDULE_V2_PREFILL: "%s"\n      VTL_W4A8_V2_PREFILL_MAX: "%s"\n' \
	    "$$s" "$$v2" "$$pf" "$$pfmax" > $(SWEEP_OVERLAY); \
	  b=0; while [ $$b -lt $(BOOTS) ]; do b=$$((b+1)); \
	    echo "=== $$name boot $$b/$(BOOTS) (VTL_W4A8_SCHEDULE='$$s' VTL_W4A8_SCHEDULE_V2='$$v2' PREFILL='$$pf'/$$pfmax)"; \
	    ( cd $(ROUND) && $(DC) -f $(SWEEP_OVERLAY) up -d --force-recreate --wait \
	      && python3 bench/replay.py --target $(TARGET) --trace $(TRACE) \
	           --out bench-sched-$$name-b$$b.json \
	      ; rc=$$?; $(DC) -f $(SWEEP_OVERLAY) down; exit $$rc ) || exit 1; \
	  done; \
	done
	@echo "verdict: cd $(ROUND) && python3 bench/sweep_report.py bench-sched-*.json"
	@echo "  (compare.py prints one column per BOOT and cannot group arms -- use sweep_report)"

## Executor-topology A/B: does dropping `mp` for `uni` pay? Compose ships `uni` (no backend
## flag at TP=1), which revives three shipped optimizations that a two-process split had
## silently disabled -- the N-step burst, VTL_SAMPLE_IN_GRAPH and the R9 fast hit -- at the
## cost of VTL_SCHED_SO_RING (Phase C), which requires `mp` and self-refuses under uni.
## `uni-nofast` is the attribution arm: uni WITHOUT the fast path, so the pickle-hop saving
## can be separated from the features the flip unlocks.
##
## Why the base compose is REWRITTEN per arm instead of overlaid: a compose override replaces
## `command:` wholesale, so an overlay carrying one serve flag would have to duplicate the
## whole 40-line arg list -- which then rots silently the next time a flag changes here.
TOPOLOGY_ARMS ?= mp uni uni-nofast
TOPO_BASE := /tmp/vtl-topology-base.yaml
TOPO_OVERLAY := /tmp/vtl-topology-sweep.yaml
## Same overlay stack as COMPOSE_FILES, with the SUBMISSION file swapped for the per-arm copy.
TOPO_DC := docker compose -f $(TOPO_BASE) -f docker-compose-optimized.yaml \
           -f docker-compose.localtest.yaml -f docker-compose.cpucap.yaml -f $(TOPO_OVERLAY)
sweep-topology:
	@for name in $(TOPOLOGY_ARMS); do \
	  fast=1; \
	  case "$$name" in *-nofast) fast=0;; esac; \
	  if [ "$$name" = "mp" ]; then \
	    awk '{print} /--tensor-parallel-size=1$$/{print "      - --distributed-executor-backend=mp"}' \
	      $(ROUND)/docker-compose.yaml > $(TOPO_BASE); \
	    grep -qE '^[[:space:]]*- --distributed-executor-backend=mp$$' $(TOPO_BASE) \
	      || { echo "FATAL: could not insert the mp flag -- the --tensor-parallel-size=1 anchor moved."; \
	           echo "  Without it the 'mp' arm would boot uni and the whole sweep would compare uni to uni."; \
	           exit 1; }; \
	  else \
	    cp $(ROUND)/docker-compose.yaml $(TOPO_BASE); \
	  fi; \
	  printf 'services:\n  model:\n    environment:\n      VTL_ENABLE_DECODE_FASTPATH: "%s"\n' "$$fast" > $(TOPO_OVERLAY); \
	  b=0; while [ $$b -lt $(BOOTS) ]; do b=$$((b+1)); \
	    echo "=== $$name boot $$b/$(BOOTS) (fastpath=$$fast)"; \
	    ( cd $(ROUND) && $(TOPO_DC) up -d --force-recreate --wait \
	      && python3 bench/replay.py --target $(TARGET) --trace $(TRACE) \
	           --out bench-sched-$$name-b$$b.json \
	      ; rc=$$?; $(TOPO_DC) down; exit $$rc ) || exit 1; \
	  done; \
	done
	@echo "verdict: cd $(ROUND) && python3 bench/sweep_report.py --baseline mp bench-sched-*.json"
	@echo "  mp is the baseline (today's shipped config); its own boot spread IS the noise floor."
	@echo "  Check the boot logs for 'classified N FA' and 'SO_RING refused' before trusting a delta."

## Kernel-level filter to run BEFORE committing $(BOOTS) boots x $(words $(SCHEDULES)) arms of
## trace replay to the box. Same microbench `make bench-kernel` runs (bench/test_*.py print
## CUDA-event tables), once per schedule: minutes instead of hours, and an arm that already
## loses at the kernel does not deserve a boot. No compose and no replay -- a microbench needs
## neither -- so this is also the only way to sweep on a GPU box with no model weights.
sweep-schedule-micro: build
	@for name in $(SCHEDULES); do \
	  $(SWEEP_CLASSIFY); \
	  echo "=== micro $$name (VTL_W4A8_SCHEDULE='$$s' VTL_W4A8_SCHEDULE_V2='$$v2' PREFILL='$$pf'/$$pfmax)"; \
	  $(KRUN) "export VTL_W4A8_SCHEDULE='$$s' VTL_W4A8_SCHEDULE_V2='$$v2' \
	    VTL_W4A8_SCHEDULE_V2_PREFILL='$$pf' VTL_W4A8_V2_PREFILL_MAX='$$pfmax'; \
	    for t in /bench/test_*.py; do echo \"--- \$$t\"; python3 \$$t || exit 1; done" || exit 1; \
	done

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
	gh run watch $$id -R $(CI_REPO) --exit-status
	@# No `|| true` here. It used to swallow the exit status, so a bench that failed on the
	@# runner reported success locally -- the one outcome this target exists to tell you about.

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
