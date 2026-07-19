"""Shadow radix tree over vLLM's block-hash prefix cache (SGLang RadixAttention port).

This installs an explicit ``VtlRadixCache`` (see ``vtl/radix_tree.py``) alongside vLLM's
own prefix cache and A/B-measures the two. It is a **faithful structural re-expression**,
not a functional change: vLLM's ``find_longest_cache_hit`` already walks the prefix-chained
block hashes like a radix tree, so the shadow tree can only *equal* it, never beat it (same
block hashes, same ``block_size`` sharing granularity). The deliverable is the tree itself
plus the equality/drift measurement.

Seam ("A"): a ``KVCacheManager`` subclass, ``BlockPool`` untouched.
  * ``get_computed_blocks`` -> call ``super()`` for the REAL served hit, then query the tree
    for the SAME prefix and record (coordinator vs tree) tokens. Returns ``super()``'s value
    unchanged -- the tree NEVER drives allocation.
  * ``allocate_slots`` -> after ``super()`` caches, mirror the request's full blocks
    (``block_hashes`` -> full-attn block ids) into the tree.

Advisory only. The tree is the full-attention group's index; it never touches the mamba /
short-conv group (group 1). Letting a longer full-attn tree hit drive allocation would leave
the mamba recurrence with no state for that prefix -> silent output corruption, so we don't.

Eviction is NOT mirrored: vLLM keeps freed blocks cached until the free-queue reclaims them,
and round-2 has ~43x KV headroom so eviction is rare. The tree therefore never evicts, and any
``tree_hit > coordinator_hit`` gap in the A/B log IS that (near-zero) eviction drift.
# ponytail: unmodeled eviction, tree grows to #unique-blocks-seen; fine for a default-off
# bench harness. Mirror BlockRemoved events here if a long-run A/B ever shows real drift.

Gated by ``VTL_ENABLE_RADIX_CACHE`` (default OFF -- measurement, not a serving default). Any
tree error disables the tree for the process and falls through to pure stock. Composes with the
``kv_cache_manager`` patch: subclasses whatever ``KVCacheManager`` is currently bound.
"""

from __future__ import annotations

import logging

from vtl.registry import register_patch

log = logging.getLogger("vtl")

_DEFAULT_BLOCK_SIZE = 16
_LOG_EVERY = 512  # requests between A/B summary log lines


def _mirror_args(hashes: list, ids: list) -> tuple[list, list]:
    """Full-block (hashes -> block ids) pairs to insert. ``block_hashes`` only exist for
    full blocks, so aligning to the shorter of the two drops any trailing partial block."""
    n = min(len(hashes), len(ids))
    return hashes[:n], list(ids[:n])


def _ab_step(state: dict, coord_tok: int, tree_tok: int, total_tok: int) -> dict:
    """Fold one request into the running A/B totals (all in tokens)."""
    state["coord"] += coord_tok
    state["tree"] += tree_tok
    state["total"] += total_tok
    state["drift"] += max(0, tree_tok - coord_tok)
    state["n"] += 1
    return state


def _ab_summary(state: dict) -> str:
    total = state["total"] or 1
    return (
        f"radix A/B: n={state['n']} "
        f"coord_hit={state['coord'] / total:.4f} "
        f"tree_hit={state['tree'] / total:.4f} "
        f"drift_tokens={state['drift']}"
    )


@register_patch("radix_cache", default=False)
def apply() -> None:
    import vllm.v1.core.sched.scheduler as sched_mod

    from vtl.radix_tree import VtlRadixCache

    base = sched_mod.KVCacheManager
    if getattr(base, "__vtl_radix__", False):
        return  # idempotent

    class VtlRadixKVCacheManager(base):
        """Stock manager + an advisory shadow radix tree + A/B counters."""

        __vtl_radix__ = True

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._tree = None
            self._fa_group = 0
            self._ab = {"coord": 0, "tree": 0, "total": 0, "drift": 0, "n": 0}
            try:
                groups = self.kv_cache_config.kv_cache_groups
                # The tree indexes the FULL-ATTENTION group only (never mamba/short-conv).
                # It is first in a "simple hybrid" but don't assume -- find it by spec.
                self._fa_group = next(
                    (i for i, g in enumerate(groups)
                     if "FullAttention" in type(g.kv_cache_spec).__name__),
                    0,
                )
                self._block_size = groups[self._fa_group].kv_cache_spec.block_size
                self._tree = VtlRadixCache()  # allocator=None: never frees (advisory)
                log.info(
                    "vtl: radix_cache installed (shadow tree, fa_group=%d, block_size=%d)",
                    self._fa_group,
                    self._block_size,
                )
            except Exception:
                self._block_size = _DEFAULT_BLOCK_SIZE
                self._tree = None  # degrade to pure stock
                log.exception("vtl: radix_cache init failed, tree disabled")

        def get_computed_blocks(self, request):
            blocks, num_computed = super().get_computed_blocks(request)
            if self._tree is not None:
                self._observe(request, num_computed)
            return blocks, num_computed

        def allocate_slots(self, request, *args, **kwargs):
            out = super().allocate_slots(request, *args, **kwargs)
            if self._tree is not None and out is not None:
                self._mirror(request)
            return out

        # ----- tree side, all fail-open ------------------------------------

        def _observe(self, request, coord_tokens: int) -> None:
            try:
                hashes = list(request.block_hashes)
                tree_tokens = self._tree.match_prefix_len(hashes) * self._block_size
                _ab_step(self._ab, coord_tokens, tree_tokens, request.num_tokens)
                if self._ab["n"] % _LOG_EVERY == 0:
                    log.info("vtl: %s", _ab_summary(self._ab))
            except Exception:
                log.exception("vtl: radix_cache observe failed, tree disabled")
                self._tree = None

        def _mirror(self, request) -> None:
            try:
                hashes = list(request.block_hashes)
                if not hashes:
                    return
                ids = self.get_block_ids(request.request_id)[self._fa_group]
                units, value = _mirror_args(hashes, list(ids))
                if units:
                    self._tree.insert(units, value)
            except Exception:
                log.exception("vtl: radix_cache mirror failed, tree disabled")
                self._tree = None

    sched_mod.KVCacheManager = VtlRadixKVCacheManager
    log.info("vtl: radix_cache patch applied (base=%s)", base.__name__)


def _self_check() -> None:
    """Runs with no vLLM. Uses the real (torch-less) VtlRadixCache + fake requests to
    exercise mirror/observe alignment and the A/B fold."""
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from radix_tree import VtlRadixCache

    BS = 16

    # _mirror_args drops the trailing partial block (fewer hashes than block ids).
    assert _mirror_args([1, 2, 3], [10, 20, 30, 40]) == ([1, 2, 3], [10, 20, 30])
    assert _mirror_args([], [10]) == ([], [])

    tree = VtlRadixCache()

    # Turn 1: a 3-block prompt gets cached (mirror). Nothing was in the tree before,
    # so at observe-time (which precedes mirror in the real flow) the tree is empty.
    h1 = [101, 102, 103]
    u, v = _mirror_args(h1, [1, 2, 3])
    tree.insert(u, v)

    # Turn 2: same conversation, one block longer. The tree must now find the 3-block
    # shared prefix -- the same hit vLLM's coordinator would serve.
    h2 = [101, 102, 103, 104]
    tree_blocks = tree.match_prefix_len(h2)
    assert tree_blocks == 3, tree_blocks

    state = {"coord": 0, "tree": 0, "total": 0, "drift": 0, "n": 0}
    # coordinator served the same 3 blocks -> no drift.
    _ab_step(state, coord_tok=3 * BS, tree_tok=tree_blocks * BS, total_tok=4 * BS)
    assert state["drift"] == 0, state
    assert state["coord"] == state["tree"] == 48

    # Drift case: coordinator evicted the prefix (served 0) but the never-evicting tree
    # still has it -> positive drift, recorded, never fatal.
    _ab_step(state, coord_tok=0, tree_tok=3 * BS, total_tok=4 * BS)
    assert state["drift"] == 48, state
    assert "tree_hit=" in _ab_summary(state)

    print("radix_cache self-check ok")


if __name__ == "__main__":
    _self_check()
