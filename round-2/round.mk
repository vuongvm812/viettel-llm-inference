# Per-round overrides, included by the root Makefile before its own defaults.
#
# WHY THIS FILE EXISTS: the forked-vLLM base is built FROM this round's vtl/vllm_patches/, and
# a shared VLLM_FORK_TAG would mean `make up ROUND=round-2` silently serves a fork carrying
# patches this round no longer ships -- which looks exactly like success.
#
# The round-2 fork is built and pinned in the root Makefile (VLLM_FORK_TAG, re-pinned in 66e5882),
# so VLLM_IMAGE defaults to it there. Stock baseline: make ... VLLM_IMAGE=$(VLLM_STOCK).
