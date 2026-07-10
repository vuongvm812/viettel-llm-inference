# syntax=docker/dockerfile:1.7
#
# The plugin must be pip-installed, not bind-mounted: vLLM finds register() through
# the `vllm.general_plugins` entry point, which lives in the .dist-info that only a
# real install produces. A mounted `vtl/` is importable but silently never loads.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.22.1

FROM ${VLLM_IMAGE} AS builder
WORKDIR /src
COPY pyproject.toml README.md ./
COPY vtl ./vtl
RUN pip install --no-cache-dir build && python3 -m build --wheel -o /dist

FROM ${VLLM_IMAGE} AS runtime
COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir --no-deps /tmp/*.whl && rm -f /tmp/*.whl

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
COPY docker/cache/ /opt/vtl/cache/

# No ENTRYPOINT: the compose file pins
# `python3 -m vllm.entrypoints.openai.api_server` to stay diffable against baseline.
