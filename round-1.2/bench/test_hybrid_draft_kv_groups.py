"""Contract test for the HYBRID separate draft model (vLLM issue #49112).

Serving LFM2.5-1.2B with LFM2.5-350M as a `method="draft_model"` drafter puts two
LFM2-shaped models in one engine. Their attention layers have an identical KV cache
spec (8 KV heads x 64 head_size, fp8_e4m3, same block size); their short-conv layers
do not (conv_dim 2048 vs 1024). That asymmetry is the whole problem, and it is also
what makes the fix small. Two claims are pinned here, both pure-Python:

  1. **The grouping needs no patch.** Equal attention specs hash equal, so the 12
     attention layers land in one group on their own, and the two conv specs -- which
     differ in `shapes` -- are already distinct keys. Upstream #35062 / #49138 add a
     drafter-layer isolation pass that this model pair does not need. If a vLLM
     upgrade changes the sizing heuristic so the drafter's layers get scattered
     across groups, this test fails and the isolation pass becomes necessary again.

  2. **`get_mamba_groups` tolerates differing state shapes.** Stock asserts full
     MambaSpec equality across mamba groups, which the two conv widths cannot satisfy
     -- it is the blocker the upstream RFC (#49138) declared unimplemented. The vtl
     patch compares every field except `shapes`, keeping `len(shapes)` so a drafter
     from a different mamba family still fails loudly.

Collected by ``make test-kernel`` (pytest, inside the image) and runnable standalone:

    python3 round-1.2/bench/test_hybrid_draft_kv_groups.py
"""

try:
    import torch

    from vllm.v1.core.kv_cache_utils import _get_kv_cache_groups_uniform_page_size
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
    )
    from vllm.v1.worker.mamba_utils import get_mamba_groups

    _SKIP = ""
except Exception as exc:  # torch / vllm not importable on this host
    _SKIP = f"vllm not importable: {exc}"

BLOCK_SIZE = 32
# LFM2.5: 16 layers, 10 conv + 6 full_attention, conv_L_cache=3.
# num_spec=3 widens the conv state to conv_L_cache - 1 + num_spec = 5.
_CONV_STATE_LEN = 5


def _attn_spec():
    # Identical for both models: num_key_value_heads=8, head_dim=64 in BOTH configs.
    return FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=8, head_size=64, dtype=torch.bfloat16
    )


def _conv_spec(conv_dim: int):
    return MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((conv_dim, _CONV_STATE_LEN),),
        dtypes=(torch.bfloat16,),
        num_speculative_blocks=1,
    )


def _lfm2_pair_spec():
    """The real {target + draft} layer -> spec map for the 350M/1.2B pair."""
    spec: dict = {}
    for prefix, conv_dim in (("model", 2048), ("draft_model", 1024)):
        for i in range(16):
            # layer_types: conv everywhere except indices 2, 5, 8, 10, 12, 14.
            is_attn = i in (2, 5, 8, 10, 12, 14)
            spec[f"{prefix}.layers.{i}.mixer"] = (
                _attn_spec() if is_attn else _conv_spec(conv_dim)
            )
    return spec


def test_attention_specs_are_shared_between_target_and_draft():
    """Equal specs must hash equal -- this is what removes the need for a patch."""
    if _SKIP:
        import pytest

        pytest.skip(_SKIP)
    assert _attn_spec() == _attn_spec()
    assert hash(_attn_spec()) == hash(_attn_spec())
    # ...and the two conv widths must NOT collapse together.
    assert _conv_spec(2048) != _conv_spec(1024)


def test_grouping_needs_no_drafter_isolation():
    """3 spec-uniform groups: [attn 12, conv_target 10, conv_draft 10]."""
    if _SKIP:
        import pytest

        pytest.skip(_SKIP)
    spec = _lfm2_pair_spec()
    groups = _get_kv_cache_groups_uniform_page_size(spec)

    assert len(groups) == 3, [g.layer_names for g in groups]
    for group in groups:
        kinds = {type(spec[n]) for n in group.layer_names if n in spec}
        assert len(kinds) == 1, (group.layer_names, kinds)
        # Every real layer in a group must carry that group's exact spec, or
        # _reshape_kv_cache_tensors allocates the wrong geometry for some layer.
        for name in group.layer_names:
            if name in spec:
                assert spec[name] == group.kv_cache_spec, name

    # The decisive property: no group mixes the drafter with the target.
    for group in groups:
        owners = {n.split(".")[0] for n in group.layer_names if n in spec}
        if any(isinstance(spec[n], MambaSpec) for n in group.layer_names if n in spec):
            assert len(owners) == 1, ("conv group spans both models", group.layer_names)


def test_get_mamba_groups_allows_differing_state_shapes():
    """The blocker upstream #49138 called unimplemented."""
    if _SKIP:
        import pytest

        pytest.skip(_SKIP)
    config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["model.layers.2.mixer"], _attn_spec()),
            KVCacheGroupSpec(["model.layers.0.mixer"], _conv_spec(2048)),
            KVCacheGroupSpec(["draft_model.layers.0.mixer"], _conv_spec(1024)),
        ],
    )
    group_ids, spec = get_mamba_groups(config)
    assert group_ids == [1, 2], group_ids
    # Only block_size / num_speculative_blocks are read off the returned spec, and
    # both are engine-wide, so returning either group's spec is correct.
    assert spec.block_size == BLOCK_SIZE
    assert spec.num_speculative_blocks == 1


def test_get_mamba_groups_still_rejects_a_different_mamba_family():
    """len(shapes) stays in the comparison key: mamba2 (2 tensors) vs short_conv (1).

    mamba_state_copy_funcs is a single tuple taken from the TARGET model class and
    zipped against each layer's kv_cache, so a mismatched arity must fail loudly
    rather than silently mis-copy.
    """
    if _SKIP:
        import pytest

        pytest.skip(_SKIP)
    two_state = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((1024, _CONV_STATE_LEN), (1024, 128)),
        dtypes=(torch.bfloat16, torch.bfloat16),
        num_speculative_blocks=1,
    )
    config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["model.layers.0.mixer"], _conv_spec(2048)),
            KVCacheGroupSpec(["draft_model.layers.0.mixer"], two_state),
        ],
    )
    try:
        get_mamba_groups(config)
    except AssertionError:
        return
    raise AssertionError("mixed mamba families must not be accepted")


if __name__ == "__main__":
    if _SKIP:
        print(f"SKIP ({_SKIP})")
    else:
        test_attention_specs_are_shared_between_target_and_draft()
        test_grouping_needs_no_drafter_isolation()
        test_get_mamba_groups_allows_differing_state_shapes()
        test_get_mamba_groups_still_rejects_a_different_mamba_family()
        print("hybrid draft kv group contract ok")
