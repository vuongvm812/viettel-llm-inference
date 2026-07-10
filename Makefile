IMAGE ?= linhthuydanhbo1234/vtl-vllm
TAG ?= dev
TARGET ?= http://localhost:8000
# The H200 box is amd64 and the vLLM base image is multi-arch. Never let the build
# host pick: an arm64 Mac would otherwise produce an image the GPU box can't run.
PLATFORM ?= linux/amd64
TRACE := data/input/trace-round1.jsonl
LOCAL := docker compose -f docker-compose-optimized.yaml -f docker-compose.localtest.yaml

.PHONY: check stats build up down warm push bench

## Self-checks. Run anywhere: no GPU, no vLLM, no running server.
check:
	python3 vtl/registry.py
	python3 bench/trace_stats.py --self-check
	python3 bench/metrics.py

stats:
	python3 bench/trace_stats.py

build:
	docker buildx build --platform $(PLATFORM) --load -t $(IMAGE):$(TAG) .
	@docker inspect $(IMAGE):$(TAG) --format 'built {{.Os}}/{{.Architecture}}'

up:
	$(LOCAL) up --build

down:
	$(LOCAL) down -v

## torch.compile needs a real GPU, so `docker build` cannot warm its cache. Boot the
## image, drive enough traffic to trigger compile + CUDA graph capture + FlashInfer
## autotune, then copy the caches back into the build context and rebuild.
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
