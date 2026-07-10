# Runtime-only image for publishing to a public registry. Ships ONLY the pre-built,
# optimized (PGO+LTO+BOLT, --features llama, native), stripped binary + config. The
# model is bind-mounted at /model (docker-compose), NOT baked in — keeps the image
# small. No source, no scripts, no toolchain — so publishing can't leak our code.
#
# The binary is built OUTSIDE Docker on a GPU host (needs GPU + CUDA + model to train
# PGO/BOLT). Build + strip it first, then build this image:
#     make dist/optimized          # → dist/inference-runtime (stripped)
#     docker build -t <user>/inference-runtime:optimized .
#     # or both:  make docker/inference-optimize
#
# CUDA-runtime base carries cudart/cuBLAS; the driver (libcuda.so) is injected by the
# host's nvidia runtime at `docker run --gpus all`.
ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# Runtime libs the llama.cpp+CUDA binary links beyond the CUDA base: OpenMP (ggml),
# the C++ runtime (libllama), and jemalloc (LD_PRELOAD allocator — setup.sh step 3).
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 libstdc++6 libjemalloc2 \
 && rm -rf /var/lib/apt/lists/*

# Lower fragmentation / faster alloc — same allocator setup.sh wires via runtime-env.sh.
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2

WORKDIR /app
COPY dist/inference-runtime /usr/local/bin/inference-runtime
COPY config/default-config.yaml /app/config/default-config.yaml

EXPOSE 8000
# Run with GPU + model mount:  docker run --gpus all -p 8000:8000 -v ./models/qwen3.5-2b-bf16.gguf:/model:ro <image>
ENTRYPOINT ["inference-runtime", "/app/config/default-config.yaml"]
