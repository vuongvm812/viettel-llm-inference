"""Parity + microbench for the megakernel's W4A8 GEMV (vtl/csrc/w4a8_gemv.cuh).

The GEMV is only reachable through ``shortconv_decode_mega``, so this file exercises it via
that op with the conv phases neutralised (a null state index makes phase 3 emit zeros, which
would mask a GEMV bug) -- no. Instead it checks the two halves the GEMV owns directly:

  * ``pack_int4_nibbles`` round-trips every int4 value, in both nibble positions. A packing
    bug is silently wrong weights, which no end-to-end tolerance would catch;
  * ``build_mega_weights`` reproduces the SAME quantized values as
    ``quant_w4a8._quantize_and_pack`` -- only the storage order differs -- so the megakernel
    and the CUTLASS path cannot drift apart numerically;
  * a torch reference of the exact GEMV arithmetic (fp32 accumulate per 64-wide half-group,
    scale, sum) against ``ops.cutlass_w4a8_mm`` at M=1..8 x both LFM2 shapes, to size how far
    the two accumulation orders really are apart. This is the number that justifies the
    tolerance used in test_shortconv_decode_mega.py.

    pytest bench/test_w4a8_gemv.py -q
    python bench/test_w4a8_gemv.py --bench     # M x shape latency table, GPU only
"""

from __future__ import annotations

import sys

import pytest

torch = pytest.importorskip("torch")

DIM = 2048  # LFM2.5 hidden size
SHAPES = [(3 * DIM, DIM), (DIM, DIM)]  # in_proj, out_proj
GROUP = 128


def _mega():
    try:
        from vtl.patches.shortconv_mega import build_mega_weights, pack_int4_nibbles
    except Exception as exc:
        pytest.skip(f"vtl not importable: {exc}")
    return build_mega_weights, pack_int4_nibbles


def test_nibble_packing_round_trips_every_int4():
    _, pack = _mega()
    # Every (low, high) pair of the 16x16 int4 cross-product, twice over so both nibble
    # positions see every value.
    lo = torch.arange(-8, 8, dtype=torch.int8).repeat_interleave(16)
    hi = torch.arange(-8, 8, dtype=torch.int8).repeat(16)
    row = torch.stack([lo, hi], dim=1).reshape(1, -1)  # [1, 512], alternating lo/hi
    packed = pack(row)
    assert packed.dtype == torch.uint8 and packed.shape == (1, row.shape[1] // 2)

    got_lo = (packed & 0xF).to(torch.int16)
    got_hi = ((packed >> 4) & 0xF).to(torch.int16)
    got_lo = torch.where(got_lo > 7, got_lo - 16, got_lo)
    got_hi = torch.where(got_hi > 7, got_hi - 16, got_hi)
    assert torch.equal(got_lo.flatten(), lo.to(torch.int16))
    assert torch.equal(got_hi.flatten(), hi.to(torch.int16))


def gemv_reference(wq_packed, gs, x_fp8, x_scale):
    """The exact arithmetic w4a8_gemv_row performs, in torch. [N, K/2] x [M, K] -> [M, N]."""
    lo = (wq_packed & 0xF).to(torch.int16)
    hi = ((wq_packed >> 4) & 0xF).to(torch.int16)
    lo = torch.where(lo > 7, lo - 16, lo)
    hi = torch.where(hi > 7, hi - 16, hi)
    w = torch.stack([lo, hi], dim=2).reshape(wq_packed.shape[0], -1).float()  # [N, K]

    N, K = w.shape
    M = x_fp8.shape[0]
    a = x_fp8.float()  # [M, K]
    out = torch.zeros(M, N, device=w.device, dtype=torch.float32)
    for g in range(K // GROUP):
        sl = slice(g * GROUP, (g + 1) * GROUP)
        out += (a[:, sl] @ w[:, sl].t()) * gs[:, g].unsqueeze(0)
    return out * x_scale.reshape(M, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("N,K", SHAPES)
def test_mega_weights_match_the_cutlass_quantization(N, K):
    build, _ = _mega()
    try:
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            quantize_weights,
        )
        from vllm.scalar_type import scalar_types
    except Exception as exc:
        pytest.skip(f"vllm not importable: {exc}")

    torch.manual_seed(N)
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.02
    wq, gs = build(w)
    assert wq.shape == (N, K // 2) and wq.dtype == torch.uint8
    assert gs.shape == (N, K // GROUP) and gs.dtype == torch.float32

    # Same quantizer, same values -- only the storage order differs from _quantize_and_pack.
    _, ref_q, ref_s, _ = quantize_weights(
        w.t().contiguous().float(), scalar_types.int4, group_size=GROUP, zero_points=False
    )
    lo = (wq & 0xF).to(torch.int16)
    hi = ((wq >> 4) & 0xF).to(torch.int16)
    lo = torch.where(lo > 7, lo - 16, lo)
    hi = torch.where(hi > 7, hi - 16, hi)
    unpacked = torch.stack([lo, hi], dim=2).reshape(N, K)
    assert torch.equal(unpacked, ref_q.t().to(torch.int16)), "int4 values drifted"
    assert torch.allclose(gs, ref_s.t().float()), "group scales drifted"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("M", [1, 2, 4, 8])
@pytest.mark.parametrize("N,K", SHAPES)
def test_reference_gemv_tracks_cutlass(M, N, K):
    """How far apart are the two accumulation orders? This sets the megakernel's tolerance."""
    build, _ = _mega()
    try:
        from vllm import _custom_ops as ops

        from vtl.patches.quant_w4a8 import GROUP_SIZE, _quantize_and_pack
    except Exception as exc:
        pytest.skip(f"vllm/vtl w4a8 path unavailable: {exc}")
    assert GROUP_SIZE == GROUP

    torch.manual_seed(M * 1000 + N)
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.02
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    x_scale = x.abs().amax(dim=1, keepdim=True).float() / 448.0
    x_fp8 = (x.float() / x_scale).clamp(-448, 448).to(torch.float8_e4m3fn)

    wq, gs = build(w)
    mine = gemv_reference(wq, gs, x_fp8, x_scale)

    packed, group_scales, chan_scales = _quantize_and_pack(w)
    theirs = ops.cutlass_w4a8_mm(
        a=x_fp8, b_q=packed, b_group_scales=group_scales, b_group_size=GROUP,
        a_token_scales=x_scale, b_channel_scales=chan_scales,
    ).float()

    rel = (mine - theirs).abs().max() / theirs.abs().max().clamp(min=1e-6)
    assert rel < 2e-2, f"M={M} {N}x{K}: reference GEMV is {rel:.4f} from CUTLASS"


def bench():  # pragma: no cover -- run by hand on the GPU box
    build, _ = _mega()
    import time

    from vllm import _custom_ops as ops

    from vtl.patches.quant_w4a8 import _quantize_and_pack

    print(f"{'shape':>12} {'M':>3} {'cutlass us':>11}")
    for N, K in SHAPES:
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.02
        packed, gsc, csc = _quantize_and_pack(w)
        build(w)  # exercise the prep cost too
        for M in (1, 2, 4, 8):
            x_fp8 = torch.zeros(M, K, dtype=torch.float8_e4m3fn, device="cuda")
            xs = torch.ones(M, 1, device="cuda")
            for _ in range(10):
                ops.cutlass_w4a8_mm(a=x_fp8, b_q=packed, b_group_scales=gsc,
                                    b_group_size=GROUP, a_token_scales=xs,
                                    b_channel_scales=csc)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(200):
                ops.cutlass_w4a8_mm(a=x_fp8, b_q=packed, b_group_scales=gsc,
                                    b_group_size=GROUP, a_token_scales=xs,
                                    b_channel_scales=csc)
            torch.cuda.synchronize()
            print(f"{N}x{K:>6} {M:>3} {(time.perf_counter() - t0) / 200 * 1e6:11.2f}")


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench()
    else:
        raise SystemExit(pytest.main([__file__, "-q"]))
