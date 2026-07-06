MANIFEST := services/Cargo.toml

# Run the inference service. Override config: `make run/inference CONFIG=path/to.yaml`
.PHONY: run/inference
run/inference:
	cargo run --manifest-path $(MANIFEST) -p inference-runtime -- $(CONFIG)

# P6 — PGO → LTO → BOLT optimized build (docs/design/build-optimization/design.md).
.PHONY: build/pgo
build/pgo:
	./build-pgo-bolt.sh
