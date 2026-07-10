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
ARG VLLM_IMAGE=vllm/vllm-openai:v0.22.1

FROM --platform=linux/amd64 ${VLLM_IMAGE} AS runtime

COPY pyproject.toml README.md /src/
COPY vtl /src/vtl
RUN pip install --no-cache-dir --no-deps /src && rm -rf /src

# Fail the build, not the judge's run.
RUN python3 -c "import importlib.metadata as m; \
eps = {e.name: e.value for e in m.entry_points(group='vllm.general_plugins')}; \
assert eps.get('vtl') == 'vtl.plugin:register', f'plugin NOT registered: {eps}'; \
print('plugin entry point ok:', eps)"

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
#
# Re-run `make warm` on the H200 after any change to --speculative-config or
# --compilation-config: ngram_gpu adds a @support_torch_compile module and
# fuse_attn_quant changes the compiled graph, so an older cache silently misses and
# the judge pays that stall inside the 180 s healthcheck start_period. Cache keys also
# include device capability, so a cache warmed on a non-Hopper card is dead weight.
COPY docker/cache/ /opt/vtl/cache/

# No ENTRYPOINT: the compose file pins
# `python3 -m vllm.entrypoints.openai.api_server` to stay diffable against baseline.
