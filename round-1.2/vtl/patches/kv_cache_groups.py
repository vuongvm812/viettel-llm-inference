"""Merge same-spec KV cache groups so the per-step metadata build runs once per spec.

LFM2.5-1.2B is 6 GQA attention layers + 10 short-conv layers. vLLM's group sizer
(``kv_cache_utils._get_kv_cache_groups_uniform_page_size``) picks
``group_size = min(6, 10) = 6`` -- the ``max < min * 1.5`` escape hatch that would pick 10
does not fire because ``10 >= 9`` -- and then splits the conv layers into
``cdiv(10, 6) = 2`` groups of 5. So we serve THREE kv cache groups: one attention, two
short-conv.

That extra group is pure per-step cost. ``gpu/attn_utils.build_attn_metadata`` loops over
``range(len(kv_cache_groups))`` and runs the group's metadata builder inside, so the mamba
builder's ``_compute_common_metadata`` runs TWICE every decode step against ONE flashinfer
build -- measured at ~0.85 of 6.5 ms/step on the dev box, i.e. ~13% of TPOT, forever. The
two conv builds are near-identical: same requests, same query_start_loc, same
decode/prefill split; only the state-index tensor read out of the group's own block table
differs.

The fix is to hand back one group per distinct spec. vLLM already treats same-spec groups
as one unit downstream -- ``HybridKVCacheCoordinator.verify_and_split_kv_cache_groups``
buckets them by spec for exactly this reason -- and its ``> 1 attention group`` assertion
still holds at 2 (attention + conv). Groups are NOT required to hold the same number of
layers: block tables are sized per group (``model_runner`` builds ``block_sizes`` and
``max_num_blocks_per_group`` as per-group lists) and the shared KV pool is laid out by
``group_size = max(len(g.layer_names))`` with a per-slot ``shared_by`` list that already
handles short groups (``if i < len(kv_cache_groups[j].layer_names)``).

The one real cost: that ``group_size`` goes 6 -> 10, so the pool is cut into 10 slices
instead of 6 and ``num_blocks`` drops ~40%. Check the boot log's "GPU KV cache size" and
"Maximum concurrency" before and after -- the round-2 trace's KV working set is ~1.4 GB
against tens of GB available, so the headroom is there, but that is the number that
decides whether this is free.

Set ``VTL_ENABLE_KV_CACHE_GROUPS=0`` to serve vLLM's stock grouping.
"""

from __future__ import annotations

import logging

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vllm.vtl")


def merge_same_spec_groups(groups: list, spec_order: dict) -> list:
    """One group per distinct ``kv_cache_spec``, layers back in model order.

    ``spec_order`` maps layer name -> its index in the model's kv_cache_spec dict; it only
    decides which pool slot a layer lands in, so sorting by it just keeps attention layer
    *i* paired with conv layer *i* the way stock does. Returns ``groups`` unchanged when
    there is nothing to merge.
    """
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec

    by_spec: dict = {}
    for group in groups:
        by_spec.setdefault(group.kv_cache_spec, []).extend(group.layer_names)
    if len(by_spec) == len(groups):
        return groups
    return [
        KVCacheGroupSpec(sorted(names, key=lambda n: spec_order[n]), spec)
        for spec, names in by_spec.items()
    ]


@register_patch("kv_cache_groups", default=True)
def patch_kv_cache_groups() -> None:
    from vllm.v1.core import kv_cache_utils

    if already_patched(kv_cache_utils, "_get_kv_cache_groups_uniform_page_size"):
        return

    original = kv_cache_utils._get_kv_cache_groups_uniform_page_size

    # Signature-agnostic: v0.25.0 takes (kv_cache_spec) and later versions take
    # (kv_cache_spec, drafter_layers). Binding either one explicitly makes the wrapper
    # TypeError at engine start on the other, which kills the server -- register_patch
    # only catches failures at APPLY time, not at call time.
    def _merged(kv_cache_spec, *args, **kwargs):
        groups = original(kv_cache_spec, *args, **kwargs)
        drafter_layers = kwargs.get("drafter_layers", args[0] if args else None)
        if drafter_layers:
            # Drafter layers are deliberately isolated into their own group(s) upstream
            # (a mixed group trips validate_same_kv_cache_group); merging by spec would
            # fold them straight back into the target's. No spec-decode here anyway.
            return groups
        try:
            spec_order = {name: i for i, name in enumerate(kv_cache_spec)}
            merged = merge_same_spec_groups(groups, spec_order)
        except Exception:
            # Boot-time only, but it IS on the boot path: degrade to stock grouping rather
            # than take the server down. registry.apply_all only guards the apply, not this.
            log.exception("vtl: kv cache group merge failed, keeping stock grouping")
            return groups
        if merged is not groups:
            log.info(
                "vtl: merged %d kv cache groups into %d (layers per group: %s); "
                "one metadata build per spec per step",
                len(groups),
                len(merged),
                [len(g.layer_names) for g in merged],
            )
        return merged

    kv_cache_utils._get_kv_cache_groups_uniform_page_size = mark_patched(
        _merged, original
    )


def _self_check() -> None:
    class FakeSpec:
        """Hashable/eq by name, like the real frozen KVCacheSpec dataclasses."""

        def __init__(self, name: str) -> None:
            self.name = name

        def __hash__(self) -> int:
            return hash(self.name)

        def __eq__(self, other) -> bool:
            return isinstance(other, FakeSpec) and other.name == self.name

    class FakeGroup:
        def __init__(self, layer_names, spec) -> None:
            self.layer_names = list(layer_names)
            self.kv_cache_spec = spec

    import sys
    import types

    module = types.ModuleType("vllm.v1.kv_cache_interface")
    module.KVCacheGroupSpec = FakeGroup
    sys.modules.setdefault("vllm", types.ModuleType("vllm"))
    sys.modules.setdefault("vllm.v1", types.ModuleType("vllm.v1"))
    sys.modules["vllm.v1.kv_cache_interface"] = module

    attn, conv = FakeSpec("attn"), FakeSpec("conv")
    # The real LFM2.5 shape: 6 attn in one group, 10 conv dealt round-robin into two.
    convs = [f"conv.{i}" for i in range(10)]
    groups = [
        FakeGroup([f"attn.{i}" for i in range(6)], attn),
        FakeGroup(convs[0::2], conv),
        FakeGroup(convs[1::2], conv),
    ]
    order = {n: i for i, n in enumerate([f"attn.{i}" for i in range(6)] + convs)}

    merged = merge_same_spec_groups(groups, order)
    assert [len(g.layer_names) for g in merged] == [6, 10], merged
    # Model order restored, so pool slot i pairs attn.i with conv.i as stock does.
    assert merged[1].layer_names == convs, merged[1].layer_names
    # Every layer survives exactly once.
    flat = [n for g in merged for n in g.layer_names]
    assert sorted(flat) == sorted(order), flat

    # Nothing to merge -> the exact same list object, so the caller can skip logging.
    single = [FakeGroup(["a"], attn), FakeGroup(["b"], conv)]
    assert merge_same_spec_groups(single, {"a": 0, "b": 1}) is single

    print("kv_cache_groups self-check ok")


if __name__ == "__main__":
    _self_check()
