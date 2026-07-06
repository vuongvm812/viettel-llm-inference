MANIFEST := services/Cargo.toml

# Run the inference service. Override config: `make run/inference CONFIG=path/to.yaml`
.PHONY: run/inference
run/inference:
	cargo run --manifest-path $(MANIFEST) -p inference-runtime -- $(CONFIG)

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
IMAGE ?= inference-runtime:optimized

.PHONY: dist/optimized
dist/optimized:
	FEATURES=llama ./scripts/build-pgo-bolt.sh
	mkdir -p dist
	cp "$$(ls services/target/release/inference-runtime.bolt 2>/dev/null || echo services/target/release/inference-runtime)" dist/inference-runtime
	strip dist/inference-runtime

.PHONY: docker/inference-optimize
docker/inference-optimize: dist/optimized
	docker build -t $(IMAGE) .

# Full flow: build the optimized binary → build the compose image → run it.
.PHONY: up/inference-optimize
up/inference-optimize: dist/optimized
	docker compose build inference
	docker compose up
