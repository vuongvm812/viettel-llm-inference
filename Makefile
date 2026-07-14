IMAGE ?= unseenablefuture/awesome-badger
TAG ?= dev
TARGET ?= http://localhost:8000
# The H200 box is amd64 and the vLLM base image is multi-arch. Never let the build
# host pick: an arm64 Mac would otherwise produce an image the GPU box can't run.
PLATFORM ?= linux/amd64
# SM archs baked into vtl._C. A wrong arch fails at the first kernel launch, not at import.
# Narrow to '9.0+PTX' for the submission build. See the ARG in Dockerfile.
CUDA_ARCHS ?= 8.0;8.6;8.9;9.0+PTX
TRACE := data/input/trace-round1.jsonl
LOCAL := docker compose -f docker-compose-optimized.yaml -f docker-compose.localtest.yaml -f docker-compose.cpucap.yaml

.PHONY: check stats build up down warm push bench test-kernel bench-kernel verify

## Self-checks. Run anywhere: no GPU, no vLLM, no running server.
check:
	python3 vtl/registry.py
	PYTHONPATH=. python3 vtl/patches/quant_fp8.py
	PYTHONPATH=. python3 vtl/patches/rms_norm_quant.py
	PYTHONPATH=. python3 vtl/patches/dynamic_per_token_quant.py
	PYTHONPATH=. python3 vtl/patches/silu_mul_quant.py
	PYTHONPATH=. python3 vtl/patches/mul_sigmoid_quant.py
	PYTHONPATH=. python3 vtl/patches/gdn_kernels.py
	PYTHONPATH=. python3 vtl/patches/gdn_prefill_backend.py
	PYTHONPATH=. python3 vtl/patches/kv_cache_manager.py
	PYTHONPATH=. python3 vtl/patches/sched_policy.py
	python3 bench/trace_stats.py --self-check
	python3 bench/metrics.py
	@python3 -c "import vtl.patches, vtl.plugin; print('vtl imports without vLLM: ok')"

# No compose, no server, no model: the kernel tests just need the image and a GPU.
# --entrypoint bash because the vLLM base image starts the API server otherwise.
# -p no:cacheprovider because /bench is mounted read-only.
# Both targets depend on `build`: $(IMAGE):$(TAG) also names a registry repo, so without it
# docker silently pulls a stale published image and the tests run against whatever kernel
# it happens to contain.
KRUN := docker run --rm --gpus all -v $(PWD)/bench:/bench:ro --entrypoint bash $(IMAGE):$(TAG) -lc
KERNEL_TESTS := /bench/test_rms_norm_quant.py /bench/test_dynamic_per_token_quant.py \
                /bench/test_silu_mul_quant.py /bench/test_mul_sigmoid_quant.py \
                /bench/test_gdn_gated_rmsnorm.py /bench/test_gdn_chunk_scan.py
PYTEST := pytest -q -p no:cacheprovider $(KERNEL_TESTS)

## Kernel correctness. Needs a GPU. Runs one oracle against our kernel AND against the
## stock one -- importing vtl._C overrides _C process-wide, so they cannot coexist and
## agreeing with the same reference is what proves ours matches stock.
test-kernel: build
	$(KRUN) 'pip install -q pytest && \
	  echo "--- vtl kernel"  && $(PYTEST) && \
	  echo "--- stock kernel" && VTL_SKIP_EXT=1 $(PYTEST)'

## Kernel microbenchmark at the trace's real shapes. Needs a GPU.
bench-kernel: build
	$(KRUN) 'for t in $(KERNEL_TESTS); do \
	    echo "=== $$t (vtl)"; python3 $$t; \
	    echo "=== $$t (stock)"; VTL_SKIP_EXT=1 python3 $$t; \
	  done'

## Pinpoint a memory fault. VTL_KERNEL_SYNC makes the kernel synchronise after every launch
## and, on a fault, raise with the exact shape and path (fast/generic, dtype, stride,
## pointer alignments) -- so it is attributed to its own launch instead of cascading into a
## later test. Runs each test in its own process (-p no:randomly, --forked-ish via -x stop)
## so the first raise is the culprit. compute-sanitizer is not in the runtime image; this
## needs no extra tooling.
##   make debug-kernel                          # whole suite, stops at first fault
##   make debug-kernel T=test_misaligned        # one test
T ?=
DBG_KRUN := docker run --rm --gpus all -e VTL_KERNEL_SYNC=1 -e CUDA_LAUNCH_BLOCKING=1 \
              -v $(PWD)/bench:/bench:ro --entrypoint bash $(IMAGE):$(TAG) -lc
debug-kernel: build
	$(DBG_KRUN) 'pip install -q pytest && \
	  python3 -m pytest -q -p no:cacheprovider -x $(KERNEL_TESTS) \
	  $(if $(T),-k $(T),)'

stats:
	python3 bench/trace_stats.py

## Post-boot assertions. We rely on vLLM's defaults rather than passing risky flags,
## so prove the defaults actually resolved our way. Run against a live container.
verify:
	@$(LOCAL) logs model 2>/dev/null > /tmp/vtl-verify.log || true
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
	@grep -q "channelwise fp8 unavailable" /tmp/vtl-verify.log \
	  && echo "WARN channelwise fp8 fell back to stock per-tensor" \
	  || echo "OK   channelwise fp8 active"
	@# Expected to fail when you deliberately A/B with VTL_ENABLE_RMS_NORM_QUANT=0.
	@grep -q "fused rms_norm+fp8-quant CUDA kernel installed" /tmp/vtl-verify.log \
	  && echo "OK   vtl fused norm+quant kernel installed" \
	  || { echo "FAIL vtl kernel not installed -- stock _C kernel is running"; exit 1; }
	@# RMSNormQuantFusionPass.__call__ overrides VllmPatternMatcherPass.__call__ and never
	@# touches match_table, so it never appears in "fusion pass matches:". The only line it
	@# emits is `Replaced N patterns` (rms_quant_fusion.py:671, DEBUG). N is how many nodes
	@# were rewritten into the op our kernel backs; N=0 means the kernel never runs.
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
# single-entry manifest LIST buildx would otherwise wrap a one-platform image in -- pure export
# overhead here, and the manifest-list form makes some `docker pull`s slower. The base image
# layers still dominate a first push; that is a one-time cost (Docker Hub dedups by digest, so
# later pushes upload only the changed vtl layer). For code iteration prefer `make build`
# (local --load, no registry round-trip) and only `make push` for the submission.
# NOCACHE=--no-cache forces a full rebuild: re-runs pip/nvcc so the CURRENT kernels are
# recompiled and every COPY is redone, instead of reusing cached layers. The base FROM image
# stays cached (pulled, not built). Use when you must be sure the latest vtl code is baked in:
#   make push NOCACHE=--no-cache
NOCACHE ?=
BUILDX_FLAGS := --provenance=false --sbom=false $(NOCACHE)

build:
	docker buildx build $(BUILDX_FLAGS) --platform $(PLATFORM) --build-arg CUDA_ARCHS='$(CUDA_ARCHS)' --load -t $(IMAGE):$(TAG) .
	@docker inspect $(IMAGE):$(TAG) --format 'built {{.Os}}/{{.Architecture}}'

up:
	$(LOCAL) up --build

down:
	$(LOCAL) down -v

## torch.compile needs a real GPU, so `docker build` cannot warm its cache. Boot the
## image, drive enough traffic to trigger compile + CUDA graph capture + FlashInfer
## autotune, then copy the caches back into the build context and rebuild.
##
## Two passes on purpose. The open-loop --limit 4 warms the low-concurrency shapes.
## The closed-loop pass then SATURATES a full batch (= --max-num-seqs in
## docker-compose-optimized.yaml) so the multi-seq Triton kernels compile into the
## cache too -- notably FlashInfer's batch_memcpy_kernel and vLLM's _zero_kv_blocks_kernel,
## which only fire once a real batch forms and KV blocks churn. Without this pass they
## JIT on the judge's first saturated batch and spike latency. WARM_REQS > concurrency
## forces a second wave so blocks get freed/zeroed (triggers _zero_kv_blocks_kernel).
WARM_CONCURRENCY ?= 16
WARM_REQS ?= 32
warm:
	$(LOCAL) up -d --build
	python3 bench/replay.py --target $(TARGET) --trace $(TRACE) --limit 4 --out /dev/null
	python3 bench/replay.py --target $(TARGET) --trace $(TRACE) \
	  --closed-loop $(WARM_CONCURRENCY) --limit $(WARM_REQS) --out /dev/null
	docker cp "$$($(LOCAL) ps -q model)":/opt/vtl/cache/. docker/cache/
	$(LOCAL) down
	$(MAKE) build

## buildx --push writes the manifest straight to the registry, so the pushed image
## is $(PLATFORM) regardless of what this machine is.
push:
	docker buildx build $(BUILDX_FLAGS) --platform $(PLATFORM) --build-arg CUDA_ARCHS='$(CUDA_ARCHS)' --push -t $(IMAGE):$(TAG) .
	@echo "pin this digest in docker-compose-optimized.yaml:"
	@docker buildx imagetools inspect $(IMAGE):$(TAG) --format '{{.Manifest.Digest}}'

## Open-loop replay (honors the trace's arrival times) + a closed-loop sweep.
bench:
	python3 bench/replay.py --target $(TARGET) --trace $(TRACE) --out bench-open.json
	for n in 1 8 32 128; do \
	  python3 bench/replay.py --target $(TARGET) --trace $(TRACE) \
	    --closed-loop $$n --out bench-closed-$$n.json; \
	done
