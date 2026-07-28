"""Parity + micro-benchmark for ``vllm_cuda::w4a8_mm_v2`` (vtl._C_w4a8).

    pytest bench/test_w4a8_v2.py -q      # parity of every built v2 arm vs the stock kernel
    python bench/test_w4a8_v2.py         # CUDA-event table, per (schedule, shape, M)

Needs sm_90 and a build that has the extension (``CUTLASS_DIR`` was set at pip-install time).
Everything else -- a dev-box GPU, the stock-only image, ``VTL_SKIP_EXT=1`` -- skips with a
reason rather than failing, because ``make test-kernel`` runs the whole bench/ glob twice and
this file has no stock counterpart to compare under the second (stock) pass.

TWO TOLERANCE REGIMES, on purpose:

  M <= threshold  the v2 arm runs. Stream-K changes the ORDER of the K reduction, so bitwise
                  equality is the wrong bar; 1e-2 rel/abs on bf16 output is ~1 ulp of headroom.
  M >  threshold  the op forwards to `_C::cutlass_w4a8_mm` with no schedule override -- the
                  exact call the un-patched path makes. That must be BIT-EXACT: if it is not,
                  the forwarding branch is passing something different from what it claims.

The weights come out of ``vtl.patches.quant_w4a8._quantize_and_pack``, i.e. the same pack/encode
sequence the server uses, rather than a local copy of it -- a parity test against a divergent
packer proves nothing about the server.
"""

from __future__ import annotations

import os

import pytest
import torch

FP8 = torch.float8_e4m3fn

# (N, K) of the LFM2.5 linears, in the layer's own orientation: N = out features.
#   3072x2048   attn qkv          2048x2048   attn out_proj / conv out_proj
#   16384x2048  mlp w13           2048x8192   mlp w2   (the deep-wave Stream-K case)
#   6144x2048   conv in_proj      65536x2048  lm_head
SHAPES = [(3072, 2048), (2048, 2048), (16384, 2048), (2048, 8192), (6144, 2048), (65536, 2048)]
MS = [1, 4, 8, 16, 33, 512]
MTHRESH = 32


def _skip_reason() -> str | None:
    if os.environ.get("VTL_SKIP_EXT", "").strip() in ("1", "true", "yes", "on"):
        return "VTL_SKIP_EXT=1: w4a8_mm_v2 is a vtl-only op, there is no stock counterpart"
    if not torch.cuda.is_available():
        return "needs CUDA"
    if torch.cuda.get_device_capability() != (9, 0):
        return f"needs sm_90; this is sm_{'%d%d' % torch.cuda.get_device_capability()}"
    try:
        import vtl._C  # noqa: F401  -- owns TORCH_LIBRARY(vllm_cuda); must be imported first
        import vtl._C_w4a8  # noqa: F401
    except Exception as exc:
        return f"vtl._C_w4a8 not importable ({exc}); built without CUTLASS_DIR?"
    if not hasattr(torch.ops.vllm_cuda, "w4a8_mm_v2"):
        return "vtl._C_w4a8 imported but did not register w4a8_mm_v2"
    return None


SKIP = _skip_reason()
pytestmark = pytest.mark.skipif(SKIP is not None, reason=SKIP or "")

# Every name quant_w4a8 will accept. An arm compiled out (#if VTL_W4A8_ENABLE_PP=0, or a
# builder static_assert that forced it out) is discovered at runtime by _available(), not
# hardcoded here -- the point is that dropping an arm must not turn into a failing test.
ALL_SCHEDULES = [
    "128x16_1x1x1_sk",
    "128x32_1x1x1_sk",
    "128x16_2x1x1",
    "128x8_1x1x1",
    "64x16_1x1x1_pp",
    "64x32_1x1x1_pp",
]

_packed_cache: dict[tuple[int, int], tuple] = {}


def packed_weight(n: int, k: int):
    """(b_q, group_scales, channel_scales) for a random bf16 [N, K], cached.

    65536x2048 costs ~0.5 GB of transient fp32 to quantize; building it once per shape instead
    of once per (schedule, M) is the difference between a test run and an OOM.
    """
    key = (n, k)
    if key not in _packed_cache:
        from vtl.patches.quant_w4a8 import _quantize_and_pack

        torch.manual_seed(n * 31 + k)
        weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.05
        _packed_cache[key] = _quantize_and_pack(weight)
        del weight
        torch.cuda.empty_cache()
    return _packed_cache[key]


def activation(m: int, k: int):
    torch.manual_seed(m * 7 + k)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    scales = x.abs().amax(-1, keepdim=True).float() / 448.0
    return (x / scales).to(FP8), scales


def stock_mm(xq, scales, packed):
    from vllm import _custom_ops as ops

    b_q, group_scales, chan_scales = packed
    return ops.cutlass_w4a8_mm(
        a=xq, b_q=b_q, b_group_scales=group_scales, b_group_size=128,
        a_token_scales=scales, b_channel_scales=chan_scales,
    )


def v2_mm(xq, scales, packed, schedule, m_threshold=MTHRESH):
    b_q, group_scales, chan_scales = packed
    return torch.ops.vllm_cuda.w4a8_mm_v2(
        xq, b_q, group_scales, 128, chan_scales, scales, schedule, m_threshold
    )


_available_cache: dict[str, bool] = {}


def available(schedule: str) -> bool:
    """Launch the arm once at a tiny shape. An arm that was compiled out raises from the C++
    dispatcher ("unknown or not-compiled-in schedule"); a wrong-arch cubin faults at the sync."""
    if schedule not in _available_cache:
        try:
            xq, scales = activation(16, 128)
            v2_mm(xq, scales, packed_weight(128, 128), schedule, m_threshold=16)
            torch.cuda.synchronize()
            _available_cache[schedule] = True
        except Exception:
            _available_cache[schedule] = False
    return _available_cache[schedule]


@pytest.mark.parametrize("schedule", ALL_SCHEDULES)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"n{s[0]}k{s[1]}")
@pytest.mark.parametrize("m", MS)
def test_matches_stock(schedule, shape, m):
    if not available(schedule):
        pytest.skip(f"{schedule} is not compiled into this build")
    n, k = shape
    packed = packed_weight(n, k)
    xq, scales = activation(m, k)

    got = v2_mm(xq, scales, packed, schedule)
    exp = stock_mm(xq, scales, packed)

    assert got.shape == (m, n) and got.dtype == torch.bfloat16
    if m > MTHRESH:
        # Same kernel, same arguments: anything but bitwise equality means the forwarding
        # branch is not forwarding what it says it is.
        assert torch.equal(got, exp), f"M={m} > {MTHRESH} must be the stock kernel, bit for bit"
    else:
        torch.testing.assert_close(got.float(), exp.float(), rtol=1e-2, atol=1e-2)


def test_threshold_is_honoured():
    """M == threshold takes the v2 arm, M == threshold + 1 does not. Off-by-one here silently
    routes all of decode to the stock kernel and every A/B reads as "no effect"."""
    schedule = next((s for s in ALL_SCHEDULES if available(s)), None)
    if schedule is None:
        pytest.skip("no v2 arm compiled into this build")
    packed = packed_weight(2048, 2048)

    # threshold=1: M=1 is the v2 arm, M=2 is stock. Compare each against stock at its own M.
    xq1, s1 = activation(1, 2048)
    xq2, s2 = activation(2, 2048)
    assert torch.equal(v2_mm(xq2, s2, packed, schedule, m_threshold=1), stock_mm(xq2, s2, packed))
    # The M=1 arm is a different kernel, so only close -- but it must still be *right*.
    torch.testing.assert_close(
        v2_mm(xq1, s1, packed, schedule, m_threshold=1).float(),
        stock_mm(xq1, s1, packed).float(), rtol=1e-2, atol=1e-2,
    )


def test_unknown_schedule_raises():
    """The C++ dispatcher must reject a name rather than silently pick something. quant_w4a8
    validates env strings for exactly this reason; this is the backstop."""
    packed = packed_weight(2048, 2048)
    xq, scales = activation(4, 2048)
    with pytest.raises(RuntimeError, match="schedule"):
        v2_mm(xq, scales, packed, "definitely_not_a_schedule")


def _bench():
    if SKIP is not None:
        print(f"test_w4a8_v2: skipped -- {SKIP}")
        return
    import statistics

    built = [s for s in ALL_SCHEDULES if available(s)]
    dropped = [s for s in ALL_SCHEDULES if s not in built]
    print(f"device: {torch.cuda.get_device_name()}  "
          f"({torch.cuda.get_device_properties(0).multi_processor_count} SMs)")
    print(f"built arms: {', '.join(built) or 'none'}")
    if dropped:
        print(f"not built : {', '.join(dropped)}")
    print(f"\n{'shape':>12} {'M':>4} {'schedule':>18} {'us/call':>9} {'vs stock':>9}")

    def timed(fn, *args):
        for _ in range(10):
            fn(*args)
        torch.cuda.synchronize()
        samples = []
        for _ in range(5):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(50):
                fn(*args)
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) / 50 * 1e3)
        return statistics.median(samples)

    for n, k in SHAPES:
        packed = packed_weight(n, k)
        for m in (1, 4, 8, 16, 32):
            xq, scales = activation(m, k)
            base = timed(stock_mm, xq, scales, packed)
            print(f"{f'n{n}k{k}':>12} {m:>4} {'stock':>18} {base:>9.1f} {'--':>9}")
            for schedule in built:
                us = timed(lambda: v2_mm(xq, scales, packed, schedule))
                print(f"{'':>12} {'':>4} {schedule:>18} {us:>9.1f} {base / us:>8.2f}x")


if __name__ == "__main__":
    _bench()
