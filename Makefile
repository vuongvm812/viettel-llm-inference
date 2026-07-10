MANIFEST := services/Cargo.toml

# Run the inference service. Override config: `make run/inference CONFIG=path/to.yaml`
.PHONY: run/inference
run/inference:
	cargo run --manifest-path $(MANIFEST) -p inference-runtime --features llama -- $(CONFIG)

# Run the P5 benchmark against a running server and print latency / throughput /
# TTFT / TPOT. Start the server first (`make run/inference`). Override target:
# `make bench BENCH_TARGET=http://localhost:8000`. If aiohttp isn't on python3,
# use the bench venv: `make bench BENCH_PY=bench/.venv/bin/python`.
BENCH_TARGET ?= http://localhost:8000
BENCH_OUT    ?= bench-results.json
BENCH_PY     ?= python3

.PHONY: bench
bench:
	$(BENCH_PY) bench/replay.py --target $(BENCH_TARGET) --out $(BENCH_OUT)
	$(BENCH_PY) bench/compare.py $(BENCH_OUT)

# Byte-parity gate for flash attention (runtime.flash_attention). The service must
# reproduce llama-cli's greedy output byte-for-byte, so flash_attention: on ships only
# if it leaves output identical. Capture completions from each server, then diff:
#   1. start the server with flash_attention: auto  → make parity-capture PARITY_OUT=ref.json
#   2. restart with flash_attention: on             → make parity-capture PARITY_OUT=on.json
#   3. make parity-flash PARITY_A=ref.json PARITY_B=on.json   (exit 1 on any mismatch)
PARITY_OUT ?= parity.json
PARITY_A   ?= ref.json
PARITY_B   ?= on.json

.PHONY: parity-capture parity-flash
parity-capture:
	$(BENCH_PY) bench/parity.py capture --target $(BENCH_TARGET) --out $(PARITY_OUT)

parity-flash:
	$(BENCH_PY) bench/parity.py compare $(PARITY_A) $(PARITY_B)

# P6 — PGO → LTO → BOLT optimized build (docs/design/build-optimization/design.md).
.PHONY: build/pgo
build/pgo:
	./scripts/build-pgo-bolt.sh

# Build the PGO+LTO(+BOLT) binary, then run it. Prefers the BOLT binary if the
# script produced one (Linux-only), else the PGO+LTO binary.
.PHONY: run/inference-optimize
run/inference-optimize:
	./scripts/build-pgo-bolt.sh
	@BIN=services/target/release/inference-runtime; \
	[ -x "$$BIN.bolt" ] && BIN="$$BIN.bolt"; \
	echo "==> running $$BIN"; \
	"$$BIN" $(CONFIG)

# Build the optimized (llama+CUDA, PGO+LTO+BOLT, native) binary OUTSIDE Docker and
# strip it into dist/ — the only artifact the published image ships (no source).
# GPU build host only. Override the image tag: `make docker/inference-optimize IMAGE=user/name:tag`.
# Namespaced with the Docker Hub user so `docker push` works out of the box.
DOCKERHUB_USER ?= vuongvm812
IMAGE         ?= $(DOCKERHUB_USER)/inference-runtime:optimized
RELEASE_IMAGE ?= $(DOCKERHUB_USER)/inference-runtime:release

.PHONY: dist/optimized
dist/optimized:
	FEATURES=llama ./scripts/build-pgo-bolt.sh
	mkdir -p dist
	cp "$$(ls services/target/release/inference-runtime.bolt 2>/dev/null || echo services/target/release/inference-runtime)" dist/inference-runtime
	strip dist/inference-runtime

.PHONY: docker/inference-optimize
docker/inference-optimize: dist/optimized
	docker build -t $(IMAGE) .

# Same optimized profile (fat LTO + native + opt-level 3) but WITHOUT the PGO/BOLT
# passes — a plain --release build. No GPU/model/trace-replay needed. Runs from
# services/ so .cargo/config.toml's -Ctarget-cpu=native applies.
.PHONY: dist/release
dist/release:
	cd services && cargo build --release -p inference-runtime --features llama
	mkdir -p dist
	cp services/target/release/inference-runtime dist/inference-runtime
	strip dist/inference-runtime

.PHONY: docker/inference-release
docker/inference-release: dist/release
	docker build -t $(RELEASE_IMAGE) .

# Push the release image to Docker Hub. Run `docker login` first. Override the tag:
# `make docker/push RELEASE_IMAGE=youruser/inference-runtime:release`.
.PHONY: docker/push
docker/push:
	docker push $(RELEASE_IMAGE)

# Build the non-PGO optimized image and push it in one step.
.PHONY: docker/inference-release-push
docker/inference-release-push: docker/inference-release docker/push

# Full flow: build the optimized binary → build the compose image → run it.
.PHONY: up/inference-optimize
up/inference-optimize: dist/optimized
	docker compose build inference
	docker compose up
