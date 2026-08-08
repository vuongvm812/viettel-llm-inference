# Per-round overrides, included by the root Makefile before its own defaults.
#
# WHY THIS FILE EXISTS: the forked-vLLM base is built FROM this round's vtl/vllm_patches/, and
# round-2's patch set is not round-2's (the three model-specific v0.25.0 patches are gone). A
# shared VLLM_FORK_TAG would mean `make up ROUND=round-2` silently serves a fork carrying patches
# this round no longer ships -- which looks exactly like success.
#
# UNPINNED ON PURPOSE. Until the fork is built for this round, default to STOCK vLLM so the round
# boots correctly rather than against another round's fork. Build and pin with:
#
#   make vllm-fork ROUND=round-2 PUSH=1       # prints the pushed digest
#   # then set VLLM_FORK_TAG below and drop the VLLM_IMAGE line
#
# The generic frontend patches (rust-frontend/*) are what the fork buys; without it,
# VLLM_USE_RUST_FRONTEND still works off the stock base via api_server_rust_frontend.patch,
# just without the sonic-rs / shm / per-token-stream work.
VLLM_IMAGE ?= vllm/vllm-openai:v0.25.0
