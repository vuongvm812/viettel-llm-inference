# syntax=docker/dockerfile:1.7
# The constant --platform below is deliberate, not a mistake to be linted away:
# it is what stops `docker build .` on an arm64 Mac from producing an unrunnable image.
# check=skip=FromPlatformFlagConstDisallowed
#
# The plugin must be pip-installed, not bind-mounted: vLLM finds register() through
# the `vllm.general_plugins` entry point, which lives in the .dist-info that only a
# real install produces. A mounted `vtl/` is importable but silently never loads.
#
# vllm/vllm-openai is multi-arch. Without the explicit --platform, building on an
# arm64 machine silently yields an arm64 image that the amd64 H200 box cannot run.
# v0.25.0: first base that registers Qwen3_5ForConditionalGeneration (the served
# Qwen3.5-2B is a qwen3_5 VL hybrid — linear-attn/GDN + full-attn + MTP + vision).
# v0.22.1 could not load it. The fp8 _C ops + _C_stable_libtorch registration our
# override depends on are byte-identical in v0.25.0 (verified against tag v0.25.0).
ARG VLLM_IMAGE=vllm/vllm-openai:v0.25.0

FROM --platform=linux/amd64 ${VLLM_IMAGE} AS runtime

COPY pyproject.toml setup.py README.md /src/
COPY vtl /src/vtl
COPY docker/entrypoint.sh docker/warmup.py /opt/vtl/
RUN chmod +x /opt/vtl/entrypoint.sh

# Which SM cubins go into vtl._C. Without this, nvcc probes the build host -- which has no
# GPU -- and guesses. A mismatch does NOT degrade gracefully: dlopen succeeds, the override
# installs, and the first kernel launch dies with cudaErrorNoKernelImageForDevice. So build
# for every device you intend to run on.
#
# The judge box is an H200 (sm_90); the trailing +PTX embeds compute_90 PTX so newer
# devices (Blackwell) can JIT. Ampere/Ada entries let the same image run on a dev box. Narrow
# this to just "9.0+PTX" for a leaner submission build:
#   make build CUDA_ARCHS='9.0+PTX'
ARG CUDA_ARCHS="8.0;8.6;8.9;9.0+PTX"

# Scoped to the RUN, not ENV: it is a build input and has no business in the served image.
# --no-build-isolation so setup.py sees the image's torch.
RUN TORCH_CUDA_ARCH_LIST="${CUDA_ARCHS}" \
    pip install --no-cache-dir --no-build-isolation --no-deps /src \
    && rm -rf /src

# Fail the build, not the judge's run.
RUN python3 -c "import importlib.metadata as m; \
eps = {e.name: e.value for e in m.entry_points(group='vllm.general_plugins')}; \
assert eps.get('vtl') == 'vtl.plugin:register', f'plugin NOT registered: {eps}'; \
print('plugin entry point ok:', eps)"

# The CUDA kernel must be in the wheel. Importing it needs a driver, so only check the
# .so landed -- vtl/patches/rms_norm_quant.py imports it for real at server start.
# Also print the embedded cubins/PTX: a .so with the wrong SM builds and imports fine and
# only fails at the first kernel launch, so this is the last place to catch it.
RUN python3 -c "import importlib.util as u; \
s = u.find_spec('vtl._C'); \
assert s is not None, 'vtl._C did not build'; \
print('vtl._C built:', s.origin)" \
 && SO=$(python3 -c "import importlib.util as u; print(u.find_spec('vtl._C').origin)") \
 && echo "vtl._C device code:" \
 && cuobjdump --list-elf "$SO" | sed 's/^/  /' \
 && { cuobjdump --list-ptx "$SO" | sed 's/^/  /' || echo "  (no PTX: no JIT fallback)"; }

ENV VLLM_PLUGINS=vtl \
    VLLM_USE_AOT_COMPILE=1 \
    VLLM_USE_STANDALONE_COMPILE=1 \
    VLLM_USE_FLASHINFER_SAMPLER=1 \
    VLLM_CACHE_ROOT=/opt/vtl/cache/vllm \
    TORCHINDUCTOR_CACHE_DIR=/opt/vtl/cache/inductor \
    TRITON_CACHE_DIR=/opt/vtl/cache/triton \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
    CUDA_MODULE_LOADING=LAZY \
    OMP_NUM_THREADS=1 \
    PYTHONHASHSEED=0

# Warm torch.compile / Triton / FlashInfer caches, produced by `make warm` on a GPU
# box. Empty on a cold build -- the server still boots, it just pays the compile
# stall on the first request.
COPY docker/cache/ /opt/vtl/cache/

# No ENTRYPOINT: the compose file pins `bash /opt/vtl/entrypoint.sh`, which wraps
# `python3 -m vllm.entrypoints.openai.api_server` with a warmup preamble.
