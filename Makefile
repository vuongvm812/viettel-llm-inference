IMAGE ?= linhthuydanhbo1234/vtl-vllm
TAG ?= dev
TARGET ?= http://localhost:8000
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
	docker build -t $(IMAGE):$(TAG) .

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

push: build
	docker push $(IMAGE):$(TAG)
	@echo "pin this digest in docker-compose-optimized.yaml:"
	@docker inspect --format='{{index .RepoDigests 0}}' $(IMAGE):$(TAG)

## Open-loop replay (honors the trace's arrival times) + a closed-loop sweep.
bench:
	python3 bench/replay.py --target $(TARGET) --trace $(TRACE) --out bench-open.json
	for n in 1 8 32 128; do \
	  python3 bench/replay.py --target $(TARGET) --trace $(TRACE) \
	    --closed-loop $$n --out bench-closed-$$n.json; \
	done
