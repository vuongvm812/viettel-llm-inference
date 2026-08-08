"""Parity for the fused mamba-align pre-copy (vtl/csrc/conv_align_fused.cu).

Ported from vLLM's own ``tests/kernels/mamba/test_precopy_mamba_align.py``: the oracle is the
TWO STOCK TRITON KERNELS the fused op replaces --
``preprocess_mamba_align_fused_kernel`` (state advance + the scratch columns) followed by
``precopy_mamba_align_fused_kernel`` -- run on cloned inputs.

Three things must match, and each fails differently:
  * the migrated CONV STATE, bit-exact (it is a byte copy in both);
  * ``state_idx``, because it drives every subsequent step's window;
  * ``num_accepted``, because the boundary reset is what makes the migrated state readable
    with a neutral bias.

    pytest bench/test_precopy_conv_align.py -q
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

DIM = 2048
STATE_LEN = 2      # LFM2.5: conv_kernel - 1
MAMBA_BLOCK = 16
NUM_LAYERS = 3     # keep the test cheap; the kernel loops states identically at 10


def _have_op() -> bool:
    try:
        import vllm._C_stable_libtorch  # noqa: F401
        import vtl._C  # noqa: F401
    except Exception:
        return False
    return hasattr(torch.ops.vllm_cuda, "conv_align_fused")


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not _have_op(), reason="vtl._C conv_align_fused unavailable"),
]


def build(num_reqs, max_reqs=16, num_blocks=32, seed=0):
    """Metadata in exactly the layout MambaSpecDecodeGPUContext hands the Triton kernels."""
    torch.manual_seed(seed)
    dev = "cuda"
    # SD conv states: (blocks, state_len, dim), so conv_width = size(1), inner = stride(1).
    states = [
        torch.randn(num_blocks, STATE_LEN, DIM, dtype=torch.bfloat16, device=dev)
        for _ in range(NUM_LAYERS)
    ]
    block_table = torch.arange(
        1, max_reqs * 8 + 1, dtype=torch.int32, device=dev
    ).reshape(max_reqs, 8) % num_blocks
    block_table = block_table.contiguous()

    meta = dict(
        block_table_ptrs=torch.tensor([block_table.data_ptr()], dtype=torch.int64, device=dev),
        block_table_stride_req=block_table.stride(0),
        state_base_addrs=torch.tensor([s.data_ptr() for s in states], dtype=torch.int64,
                                      device=dev),
        state_block_strides=torch.tensor([s.stride(0) * 2 for s in states], dtype=torch.int64,
                                         device=dev),
        state_elem_sizes=torch.full((NUM_LAYERS,), 2, dtype=torch.int32, device=dev),
        state_inner_sizes=torch.tensor([s.stride(1) for s in states], dtype=torch.int64,
                                       device=dev),
        state_conv_widths=torch.full((NUM_LAYERS,), STATE_LEN, dtype=torch.int32, device=dev),
        state_group_indices=torch.zeros(NUM_LAYERS, dtype=torch.int32, device=dev),
        state_dim_row_count=torch.zeros(NUM_LAYERS, dtype=torch.int32, device=dev),
        state_dim_row_stride=torch.zeros(NUM_LAYERS, dtype=torch.int64, device=dev),
    )
    return states, block_table, meta


def run_stock(state_idx, num_accepted, idx_mapping, num_computed, qsl, meta, num_reqs):
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.triton_utils import triton
    from vllm.v1.worker.mamba_utils import (
        precopy_mamba_align_fused_kernel,
        preprocess_mamba_align_fused_kernel,
    )

    n = state_idx.numel()
    src_col = torch.full((n,), -1, dtype=torch.int32, device="cuda")
    src_off = torch.zeros(n, dtype=torch.int32, device="cuda")
    block = 256
    preprocess_mamba_align_fused_kernel[(triton.cdiv(num_reqs, block),)](
        idx_mapping, state_idx, num_computed, qsl, num_accepted, src_col, src_off, num_reqs,
        BLOCK_SIZE=block, MAMBA_BLOCK_SIZE=MAMBA_BLOCK,
    )
    precopy_mamba_align_fused_kernel[(num_reqs, NUM_LAYERS)](
        state_idx, src_col, src_off,
        meta["block_table_ptrs"], meta["block_table_stride_req"], meta["state_base_addrs"],
        meta["state_block_strides"], meta["state_elem_sizes"], meta["state_inner_sizes"],
        meta["state_conv_widths"], meta["state_group_indices"], meta["state_dim_row_count"],
        meta["state_dim_row_stride"], idx_mapping, num_reqs,
        COPY_BLOCK_SIZE=1024, CONV_STATE_DIM_FIRST=is_conv_state_dim_first(),
    )


def run_fused(state_idx, num_accepted, idx_mapping, num_computed, qsl, meta, num_reqs):
    torch.ops.vllm_cuda.conv_align_fused(
        state_idx, num_accepted, idx_mapping, num_computed, qsl,
        meta["block_table_ptrs"], meta["block_table_stride_req"], meta["state_base_addrs"],
        meta["state_block_strides"], meta["state_elem_sizes"], meta["state_inner_sizes"],
        meta["state_conv_widths"], meta["state_group_indices"], num_reqs, MAMBA_BLOCK,
    )


CASES = {
    # name: (num_computed_before_step, initial state_idx, initial num_accepted)
    "fresh":            (16, -1, 1),   # src_col < 0 -> nothing to migrate
    "same_block":       (17, 1, 1),    # still inside the window -> no copy
    "boundary_cross":   (31, 1, 1),    # 31 + 1 = 32 -> new block column
    "cross_with_bias":  (31, 1, 2),    # accepted-token bias shifts the copied window
}


@pytest.mark.parametrize("num_reqs", [1, 2, 8])
@pytest.mark.parametrize("case", sorted(CASES))
def test_matches_stock_triton(num_reqs, case):
    num_computed_v, state_idx_v, accepted_v = CASES[case]
    max_reqs = 16

    def fresh_inputs(seed):
        states, _bt, meta = build(num_reqs, max_reqs=max_reqs, seed=seed)
        state_idx = torch.full((max_reqs,), state_idx_v, dtype=torch.int32, device="cuda")
        num_accepted = torch.full((max_reqs,), accepted_v, dtype=torch.int32, device="cuda")
        num_computed = torch.full((max_reqs,), num_computed_v, dtype=torch.int32, device="cuda")
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device="cuda")
        qsl = torch.arange(num_reqs + 1, dtype=torch.int32, device="cuda")  # 1 token/request
        return states, meta, state_idx, num_accepted, num_computed, idx_mapping, qsl

    a = fresh_inputs(seed=hash(case) % 1000)
    b = fresh_inputs(seed=hash(case) % 1000)
    run_stock(a[2], a[3], a[5], a[4], a[6], a[1], num_reqs)
    run_fused(b[2], b[3], b[5], b[4], b[6], b[1], num_reqs)
    torch.cuda.synchronize()

    assert torch.equal(a[2], b[2]), f"{case}: state_idx diverged"
    assert torch.equal(a[3], b[3]), f"{case}: num_accepted diverged"
    for i, (sa, sb) in enumerate(zip(a[0], b[0])):
        assert torch.equal(sa, sb), f"{case}: conv state of layer {i} diverged"


def test_negative_idx_mapping_rows_are_skipped():
    """A -1 slot is a filtered row. The fused kernel must not advance or copy anything for
    it -- stock's precopy skips it, and stock's preprocess only survives because real batches
    never contain one."""
    num_reqs, max_reqs = 4, 16
    states, _bt, meta = build(num_reqs, max_reqs=max_reqs, seed=5)
    state_idx = torch.full((max_reqs,), 1, dtype=torch.int32, device="cuda")
    num_accepted = torch.ones(max_reqs, dtype=torch.int32, device="cuda")
    num_computed = torch.full((max_reqs,), 31, dtype=torch.int32, device="cuda")
    idx_mapping = torch.tensor([0, -1, 2, 3], dtype=torch.int32, device="cuda")
    qsl = torch.arange(num_reqs + 1, dtype=torch.int32, device="cuda")
    before = [s.clone() for s in states]

    run_fused(state_idx, num_accepted, idx_mapping, num_computed, qsl, meta, num_reqs)
    torch.cuda.synchronize()

    assert int(state_idx[1]) == 1, "the filtered slot's state_idx must not advance"
    assert all(torch.isfinite(s.float()).all() for s in states)
    assert len(before) == len(states)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
