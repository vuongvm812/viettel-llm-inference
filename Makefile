IMAGE ?= linhthuydanhbo1234/vtl-vllm
TAG ?= dev
TARGET ?= http://localhost:8000
# The H200 box is amd64 and the vLLM base image is multi-arch. Never let the build
# host pick: an arm64 Mac would otherwise produce an image the GPU box can't run.
PLATFORM ?= linux/amd64
TRACE := data/input/trace-round1.jsonl
LOCAL := docker compose -f docker-compose-optimized.yaml -f docker-compose.localtest.yaml -f docker-compose.cpucap.yaml

# Dev-box overrides for the three interpolated knobs in docker-compose-optimized.yaml.
# Unset, they resolve to the H200 submission values -- which do not fit a 12 GB card, and
# whose fp8 KV cache needs SM89+ (Ampere has no CUTLASS fp8 W8A8 either, so vtl_fp8 falls
# back to Marlin there; `make verify` prints WARN when that happens).
LOCAL_ENV := VTL_GPU_MEM_UTIL=0.90 VTL_MAX_NUM_SEQS=8 VTL_KV_DTYPE=auto

.PHONY: check stats verify build up down warm push bench

## Self-checks. Run anywhere: no GPU, no vLLM, no running server.
check:
	python3 vtl/registry.py
	PYTHONPATH=. python3 vtl/patches/quant_fp8.py
	python3 bench/trace_stats.py --self-check
	python3 bench/metrics.py
	@python3 -c "import vtl.patches, vtl.plugin; print('vtl imports without vLLM: ok')"

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
	@# The whole reason we pass method=ngram_gpu and not ngram. The CPU proposer turns
	@# async scheduling off with a warning, not an error, so only the log proves it.
	@grep -q "Async scheduling not supported with" /tmp/vtl-verify.log \
	  && { echo "FAIL spec decode disabled async scheduling -- method must be ngram_gpu"; exit 1; } \
	  || echo "OK   spec decode kept async scheduling"
	@grep -q "ngram_gpu" /tmp/vtl-verify.log \
	  && echo "OK   ngram_gpu drafter configured" \
	  || { echo "FAIL ngram_gpu absent from the resolved engine config"; exit 1; }
	@# Includes the bonus token, so a drafter that never lands reads exactly 1.00.
	@# Below ~1.10 the draft-verify work is not paying for itself: drop --speculative-config.
	@# `grep | tail` would always exit 0, so test before printing.
	@if grep -q "SpecDecoding metrics" /tmp/vtl-verify.log; then \
	  grep "SpecDecoding metrics" /tmp/vtl-verify.log | tail -1; \
	else \
	  echo "NOTE no SpecDecoding metrics yet -- drive traffic (make bench) first"; \
	fi

build:
	docker buildx build --platform $(PLATFORM) --load -t $(IMAGE):$(TAG) .
	@docker inspect $(IMAGE):$(TAG) --format 'built {{.Os}}/{{.Architecture}}'

up:
	$(LOCAL_ENV) $(LOCAL) up --build

down:
	$(LOCAL) down -v

## torch.compile needs a real GPU, so `docker build` cannot warm its cache. Boot the
## image, drive enough traffic to trigger compile + CUDA graph capture + FlashInfer
## autotune, then copy the caches back into the build context and rebuild.
##
## Deliberately NOT run under $(LOCAL_ENV): compile/Triton cache keys include the device
## capability and the cudagraph capture sizes (which follow --max-num-seqs). Warming on an
## sm86 dev card at max_num_seqs=8 bakes caches the sm90 judge box can never hit. Run this
## on the H200, with the submission's defaults. Re-run it after any change to
## --speculative-config or --compilation-config: ngram_gpu adds a @support_torch_compile
## module and fuse_attn_quant changes the graph, so the baked cache goes stale and the
## judge pays the compile stall inside the 180 s healthcheck start_period.
warm:
	$(LOCAL) up -d --build
	python3 bench/replay.py --target $(TARGET) --trace $(TRACE) --limit 4 --out /dev/null
	docker cp "$$($(LOCAL) ps -q model)":/opt/vtl/cache/. docker/cache/
	$(LOCAL) down
	$(MAKE) build

## buildx --push writes the manifest straight to the registry, so the pushed image
## is $(PLATFORM) regardless of what this machine is.
push:
	docker buildx build --platform $(PLATFORM) --push -t $(IMAGE):$(TAG) .
	@echo "pin this digest in docker-compose-optimized.yaml:"
	@docker buildx imagetools inspect $(IMAGE):$(TAG) --format '{{.Manifest.Digest}}'

## Open-loop replay (honors the trace's arrival times) + a closed-loop sweep.
bench:
	python3 bench/replay.py --target $(TARGET) --trace $(TRACE) --out bench-open.json
	for n in 1 8 32 128; do \
	  python3 bench/replay.py --target $(TARGET) --trace $(TRACE) \
	    --closed-loop $$n --out bench-closed-$$n.json; \
	done
