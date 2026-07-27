"""Correctness for vtl's fused int4 GEMV + argmax over the DRAFT lm_head (draft_argmax.cu).

The kernel replaces `compute_logits(h).argmax(-1)` on the draft path and reads its OWN plain
int4 packing rather than the CUTLASS-reordered production weights -- that independence is the
whole point, and it is what this file checks.

The failure mode here is nasty and worth stating: a wrong draft token is NOT wrong output. The
target verifies every drafted token, so a bug shows up only as a slightly worse acceptance rate,
weeks later, attributed to the model. So the oracle is exact, not approximate: `pack_...` then
dequantize in torch, take the argmax, and require the SAME ids, including torch's
lowest-index-wins tie-breaking.

    pytest bench/test_draft_argmax.py -q      # skips entirely without vtl._C
    python bench/test_draft_argmax.py         # same checks, no pytest
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import pytest
except ImportError:  # standalone run on a bare dev box

    class _Mark:
        @staticmethod
        def skipif(_cond, reason=""):  # noqa: ARG004
            return lambda fn: fn

        @staticmethod
        def parametrize(_names, _values):
            return lambda fn: fn

    class pytest:  # type: ignore[no-redef]  # noqa: N801
        mark = _Mark()


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

USE_STOCK = os.environ.get("VTL_SKIP_EXT", "").strip() in ("1", "true", "yes", "on")

try:
    import torch

    HAVE_TORCH = True
    DTYPES = [torch.bfloat16, torch.float16]
except Exception:
    HAVE_TORCH = False
    # The @parametrize decorators below run at IMPORT time, so they cannot reference torch
    # attributes directly -- without torch that is a NameError during collection rather than the
    # skip this module's docstring promises. Placeholders keep collection working; pytestmark
    # skips every case anyway.
    DTYPES = [None]

GROUP = 128
PACK = 8


def _have_op() -> bool:
    if USE_STOCK or not HAVE_TORCH:
        return False
    try:
        import vllm._C_stable_libtorch  # noqa: F401
        import vtl._C  # noqa: F401
    except Exception:
        return False
    return hasattr(torch.ops.vllm_cuda, "draft_argmax")


HAVE_OP = _have_op()

if HAVE_TORCH:
    pytestmark = [
        pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
        pytest.mark.skipif(not HAVE_OP, reason="vtl._C draft_argmax unavailable"),
    ]


def dequantize(qweight, group_scale, k):
    """Pure-torch inverse of pack_draft_argmax_weight -- the oracle's weight."""
    n = qweight.shape[0]
    shifts = torch.arange(0, 4 * PACK, 4, device=qweight.device, dtype=torch.int32)
    # Sign-extend each nibble exactly as nibble_to_int4 does: shift up to the top, arithmetic
    # shift back down.
    words = qweight.unsqueeze(-1)
    nib = (words >> shifts) & 0xF
    vals = (nib << 28) >> 28  # int32 arithmetic shift -> [-8, 7]
    vals = vals.reshape(n, k).float()
    scale = group_scale.repeat_interleave(GROUP, dim=1)
    return vals * scale


def reference_ids(hidden, qweight, group_scale, k):
    """What the kernel must reproduce, bit for bit on the ids."""
    w = dequantize(qweight, group_scale, k)
    logits = hidden.float() @ w.t()
    return logits.argmax(dim=-1)


def run(hidden, qweight, group_scale):
    n = qweight.shape[0]
    tiles = int(torch.ops.vllm_cuda.draft_argmax_tiles(n))
    rows = hidden.shape[0]
    out = torch.empty(rows, dtype=torch.int64, device=hidden.device)
    pv = torch.empty((rows, tiles), dtype=torch.float32, device=hidden.device)
    pi = torch.empty((rows, tiles), dtype=torch.int32, device=hidden.device)
    torch.ops.vllm_cuda.draft_argmax(out, hidden, qweight, group_scale, pv, pi)
    return out


def build(rows, n, k, dtype, seed=0, device="cuda"):
    from vtl.patches.lm_head_quant import pack_draft_argmax_weight

    torch.manual_seed(seed)
    hidden = torch.randn(rows, k, dtype=dtype, device=device)
    weight = torch.randn(n, k, dtype=torch.bfloat16, device=device)
    qweight, group_scale = pack_draft_argmax_weight(weight)
    return hidden, qweight, group_scale


# 16400 is deliberately NOT a multiple of kRowsPerBlock (64): it exercises stage 1's tail-tile
# `break`, which every power-of-two vocab leaves untested. draft_argmax_supported does not
# require the multiple, so this shape is reachable in production.
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows", [1, 2, 8])
@pytest.mark.parametrize("n", [16384, 8192, 4096, 16400])
def test_matches_reference(dtype, rows, n):
    k = 1024  # LFM2.5-350M hidden size
    hidden, qweight, group_scale = build(rows, n, k, dtype, seed=n + rows)
    got = run(hidden, qweight, group_scale)
    exp = reference_ids(hidden, qweight, group_scale, k)
    assert torch.equal(got, exp), (got[:8], exp[:8])


def test_non_contiguous_hidden():
    """`hidden.stride(0)` must actually be honoured.

    In production `hidden_states` arrives as a slice / advanced-index result, so a kernel that
    hardcoded the row stride to K would pass every other test in this file and be wrong here.
    """
    k, n, rows = 1024, 4096, 4
    hidden, qweight, group_scale = build(rows * 2, n, k, torch.bfloat16, seed=77)
    strided = hidden[::2]  # stride(0) == 2*k, still last-dim contiguous
    assert strided.stride(0) == 2 * k and strided.stride(1) == 1
    got = run(strided, qweight, group_scale)
    exp = reference_ids(strided, qweight, group_scale, k)
    assert torch.equal(got, exp)


def test_batch_size_changes_across_calls():
    """The production workspace is cached on the layer and sliced (`buf[0][:n_rows]`).

    Shrinking then growing is what exercises both the reuse and the realloc branch; a stale
    workspace would show up as a wrong id for the rows beyond the previous batch.
    """
    k, n = 1024, 4096
    _, qweight, group_scale = build(1, n, k, torch.bfloat16, seed=91)
    for rows in (8, 2, 8, 1, 4):
        hidden = torch.randn(rows, k, dtype=torch.bfloat16, device="cuda")
        got = run(hidden, qweight, group_scale)
        exp = reference_ids(hidden, qweight, group_scale, k)
        assert torch.equal(got, exp), rows


def test_pack_round_trip_is_exact():
    """Packing then dequantizing must be lossless w.r.t. the quantized values themselves.

    Catches a nibble-order or sign-extension mistake, which would otherwise only show up as a
    small logit perturbation that the argmax usually absorbs.
    """
    from vtl.patches.lm_head_quant import pack_draft_argmax_weight

    torch.manual_seed(5)
    n, k = 256, 1024
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    qweight, group_scale = pack_draft_argmax_weight(weight)
    deq = dequantize(qweight, group_scale, k)

    # Re-derive the intended quantized values and compare against the unpacked ones exactly.
    # This scale formula must stay identical to vLLM's quantize_weights
    # (quant_utils.py:659-662): two-sided, able to emit -8. If pack_draft_argmax_weight ever
    # drifts from the production codebook again, the fused and stock heads disagree on which
    # token to draft -- which is not wrong output, only a quietly worse acceptance rate.
    w = weight.float().view(n, k // GROUP, GROUP)
    scale = torch.maximum(
        (w.amax(dim=-1, keepdim=True) / 7.0).abs(),
        (w.amin(dim=-1, keepdim=True) / -8.0).abs(),
    )
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.clamp(torch.round(w / scale), -8, 7)
    torch.testing.assert_close(deq, (q * scale).view(n, k), rtol=0.0, atol=0.0)
    # And the codebook really does reach -8, which a one-sided /7 scale never can.
    assert int(q.min()) == -8, int(q.min())


def test_tie_breaking_matches_torch_argmax():
    """Lowest index wins. An all-equal row is the only reliable way to force the comparison."""
    from vtl.patches.lm_head_quant import pack_draft_argmax_weight

    n, k = 4096, 1024
    # Identical rows -> identical logits -> every index ties.
    weight = torch.ones(n, k, dtype=torch.bfloat16, device="cuda")
    qweight, group_scale = pack_draft_argmax_weight(weight)
    hidden = torch.randn(2, k, dtype=torch.bfloat16, device="cuda")
    got = run(hidden, qweight, group_scale)
    exp = reference_ids(hidden, qweight, group_scale, k)
    assert torch.equal(got, exp)
    # And it really is the lowest index, not just "the same as torch by luck".
    assert int(got.min()) == 0 and int(got.max()) == 0, got


def test_supported_predicate():
    supported = torch.ops.vllm_cuda.draft_argmax_supported
    assert supported(16384, 1024)  # the shipped draft shape
    assert not supported(16384, 1000)  # K not a multiple of the group
    assert not supported(0, 1024)


if __name__ == "__main__":
    if not HAVE_OP:
        raise SystemExit("vtl._C draft_argmax unavailable; nothing to check")
    for dtype in DTYPES:
        for rows in (1, 2, 8):
            for n in (16384, 8192, 4096, 16400):
                test_matches_reference(dtype, rows, n)
    test_non_contiguous_hidden()
    test_batch_size_changes_across_calls()
    test_pack_round_trip_is_exact()
    test_tie_breaking_matches_torch_argmax()
    test_supported_predicate()
    print("draft_argmax checks ok")
