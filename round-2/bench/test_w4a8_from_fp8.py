"""Harness for [[w4a8_from_fp8]]: the fp8-block -> int4 requant of the DENSE linear layers.

    python3 bench/test_w4a8_from_fp8.py --self-check   # no GPU, no torch, no vLLM
    pytest -q bench/test_w4a8_from_fp8.py              # GPU: real dequant + pack + GEMM

THE CLAIM UNDER TEST. The checkpoint is already fp8 block-quant, so shipping int4 means
quantizing a second time. The claim that makes that survivable is a NESTING claim:

    an int4 group is 128 consecutive elements along K of ONE output row,
    and the checkpoint's [128,128] block scale is CONSTANT across exactly that span.

If that holds, requantizing from fp8 is *identical* to RTN-int4 of the fp8 values with the
block scale factored out, so the two quantizations do not interact: the error against the
original bf16 is (int4 error) + (fp8 error), not (1+e_int4)(1+e_fp8) compounding, and the
int4 step size never sees a scale the fp8 rounding moved. If it did NOT hold -- if the block
grid were, say, [64,256] -- a single int4 group would straddle two block scales and the
group absmax would be set by whichever half had the bigger scale, which is a real accuracy
cliff and not a subtle one.

The self-check half proves the nesting property and the resulting error bound in pure numpy
(no torch, no GPU, no vLLM -- it runs in `make check`). The pytest half proves the real
patch code path reproduces it on device, end to end through quant_w4a8's own packer.

Style follows bench/test_w4a8_v2.py (importorskip so the bench/ glob cannot collection-error
on a host with no torch) and bench/test_nvrtc.py (a --self-check main that needs nothing).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Pure-python constants, restated from the patch so the self-check does not need vLLM to
# import vtl.patches.quant_w4a8 (which reaches for QuantizationConfig at module scope).
BLOCK_N = BLOCK_K = 128
GROUP = 128
INT4_MAX, INT4_MIN = 7, -8
FP8_MAX = 448.0
FP8_MANTISSA_BITS = 3


# ---------------------------------------------------------------------------------------
# Pure-numpy model of the checkpoint's quantization and of our requant.
# ---------------------------------------------------------------------------------------


def e4m3_round(x, np):
    """Round an array to the e4m3 grid (3 mantissa bits, max 448), keeping fp32 storage.

    Not a bit-exact fp8 emulator -- it is a rounding model, which is all the error bound
    needs. Subnormals are handled by the same frexp expression: below 2^-6 the step stops
    shrinking, which is what `np.maximum` on the exponent encodes.
    """
    x = np.clip(np.asarray(x, dtype=np.float64), -FP8_MAX, FP8_MAX)
    mant, exp = np.frexp(np.abs(x))            # |x| = mant * 2^exp, mant in [0.5, 1)
    # e4m3 bias 7 -> smallest normal 2^-6; below that the step is fixed at 2^-9.
    exp = np.maximum(exp, -6)
    step = np.ldexp(1.0, exp - FP8_MANTISSA_BITS - 1)
    return np.sign(x) * np.round(np.abs(x) / step) * step


def block_quantize(w, np, block_n=BLOCK_N, block_k=BLOCK_K):
    """bf16-ish ``[N, K]`` -> ``(fp8 values as fp32, fp32 block scales [N/bn, K/bk])``.

    The checkpoint recipe: one scale per [128,128] tile, chosen so the tile's absmax lands
    on 448. Returned unpacked (as fp32 holding e4m3-representable values) because the point
    is the ARITHMETIC, not the byte layout.
    """
    n, k = w.shape
    gn, gk = -(-n // block_n), -(-k // block_k)
    scales = np.zeros((gn, gk), dtype=np.float64)
    out = np.zeros_like(w, dtype=np.float64)
    for bi in range(gn):
        for bj in range(gk):
            rs, re = bi * block_n, min((bi + 1) * block_n, n)
            cs, ce = bj * block_k, min((bj + 1) * block_k, k)
            tile = w[rs:re, cs:ce]
            amax = float(np.abs(tile).max())
            s = amax / FP8_MAX if amax > 0 else 1.0
            scales[bi, bj] = s
            out[rs:re, cs:ce] = e4m3_round(tile / s, np) * s
    return out, scales


def rtn_int4_groups(w, np, group=GROUP):
    """RTN int4 along K, group ``group`` -> ``(levels [-8,7], scales, dequantized)``.

    vLLM's own formula for a signed int4 with no zero point
    (``quant_utils.quantize_weights``): ``scale = max(|max|/7, |min|/8)``. Deliberately NOT
    ``absmax/7`` -- the int4 range is asymmetric and using the symmetric formula throws away
    a level on every negative-dominated group.
    """
    n, k = w.shape
    g = w.reshape(n, k // group, group)
    scale = np.maximum(np.abs(g.max(-1)) / INT4_MAX, np.abs(g.min(-1)) / -INT4_MIN)
    scale = np.maximum(scale, np.finfo(np.float32).tiny)
    q = np.clip(np.round(g / scale[..., None]), INT4_MIN, INT4_MAX)
    return q.reshape(n, k), scale, (q * scale[..., None]).reshape(n, k)


def block_scale_is_constant_within_int4_group(n, k, group=GROUP,
                                              block_n=BLOCK_N, block_k=BLOCK_K) -> bool:
    """The nesting property, as arithmetic rather than as prose: does every int4 group sit
    inside one [block_n, block_k] tile? True iff the group divides the tile along K."""
    del n, block_n  # rows: an int4 group is one row, so it is inside a tile by construction
    return block_k % group == 0 and k >= 0


# ---------------------------------------------------------------------------------------
# GPU half.
# ---------------------------------------------------------------------------------------

try:
    import pytest

    HAVE_PYTEST = True
except ImportError:  # pragma: no cover -- the `make check` path
    HAVE_PYTEST = False

    class _NoPytest:
        """Enough of pytest for the decorators to evaluate at import time; the bodies never
        run without pytest, so these are no-ops and not implementations."""

        class mark:
            @staticmethod
            def skipif(*a, **k):
                return lambda fn: fn

            @staticmethod
            def parametrize(*a, **k):
                return lambda fn: fn

        @staticmethod
        def importorskip(name, **k):
            raise ImportError(name)

        @staticmethod
        def skip(reason):
            raise SystemExit(reason)

    pytest = _NoPytest()


if HAVE_PYTEST:
    # importorskip, not a bare import: `make test-kernel` globs bench/test_*.py and pytest is
    # also pointed at bench/ from a host with no torch, where a module-level import would be a
    # COLLECTION error -- the whole directory failing instead of this one file skipping.
    torch = pytest.importorskip("torch", reason="needs torch + an sm_90 GPU")
else:  # pragma: no cover
    torch = None


def _skip_reason() -> str | None:
    import os

    if os.environ.get("VTL_SKIP_EXT", "").strip() in ("1", "true", "yes", "on"):
        return "VTL_SKIP_EXT=1: the w4a8 CUTLASS ops are vtl/vLLM-only, no stock counterpart"
    if torch is None or not torch.cuda.is_available():
        return "needs CUDA"
    if torch.cuda.get_device_capability() != (9, 0):
        return "needs sm_90 (the CUTLASS W4A8 kernel is sm_90a only)"
    try:
        from vtl.patches import quant_w4a8, w4a8_from_fp8  # noqa: F401
    except Exception as exc:
        return f"vtl patches not importable ({exc})"
    if not quant_w4a8.w4a8_ops_available():
        return "this vLLM build has no cutlass_w4a8_mm"
    return None


SKIP = _skip_reason() if HAVE_PYTEST else "no pytest"
requires_gpu = pytest.mark.skipif(SKIP is not None, reason=str(SKIP))


# TOLERANCES, and why.
#   The kernel path is fp8 activations x int4 weights accumulated in fp32, compared against a
#   bf16 matmul of the DEQUANTIZED int4 weights. So the only differences are (a) the fp8
#   activation quant, ~2^-3 relative per element, and (b) the accumulation order. Errors over
#   a K=2048 dot product average out as sqrt(K), so the sane bar is a RELATIVE one on the
#   whole output, not an elementwise ulp count.
#   3e-2 relative Frobenius error is ~4x the fp8 activation noise floor measured on random
#   normal inputs; anything structurally wrong (a transposed pack, a shifted group scale)
#   blows past 1e-1 immediately. This is a "did the plumbing survive" bar, NOT an accuracy
#   gate -- the accuracy gate is bench/eval_quality.py on the GPU host.
REL_TOL = 3e-2
# The requant-vs-dequant weight comparison has no activation noise in it, so it is held to
# the int4 step itself: every weight must land within half a group step of its fp8 value.
WEIGHT_STEP_TOL = 0.51

SHAPES = [(2048, 2048), (3072, 2048), (2048, 4096)]


def _fake_fp8_block_weight(n: int, k: int, seed: int = 0):
    """A plausible fp8 block-quant checkpoint tensor: ``(fp8 [N,K], fp32 scales, bf16 ref)``."""
    torch.manual_seed(seed)
    w = (torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.05)
    gn, gk = -(-n // BLOCK_N), -(-k // BLOCK_K)
    scale = torch.empty(gn, gk, device="cuda", dtype=torch.float32)
    q = torch.empty_like(w)
    for bi in range(gn):
        for bj in range(gk):
            rs, re = bi * BLOCK_N, min((bi + 1) * BLOCK_N, n)
            cs, ce = bj * BLOCK_K, min((bj + 1) * BLOCK_K, k)
            tile = w[rs:re, cs:ce]
            s = tile.abs().max().clamp_min(1e-12) / 448.0
            scale[bi, bj] = s
            q[rs:re, cs:ce] = tile / s
    wq = q.clamp(-448, 448).to(torch.float8_e4m3fn)
    return wq, scale, (wq.float() * scale.repeat_interleave(BLOCK_N, 0)[:n]
                       .repeat_interleave(BLOCK_K, 1)[:, :k]).to(torch.bfloat16)


@requires_gpu
@pytest.mark.parametrize("n,k", SHAPES)
def test_dequant_matches_the_block_definition(n, k):
    """``dequant_fp8_block`` is the ONE place the checkpoint's scale grid is interpreted; a
    transposed grid here would be a silently wrong model, not a crash."""
    from vtl.patches import w4a8_from_fp8

    wq, scale, ref = _fake_fp8_block_weight(n, k)
    got = w4a8_from_fp8.dequant_fp8_block(wq, scale)
    assert got.dtype == torch.bfloat16 and got.shape == (n, k)
    assert torch.equal(got, ref), "block dequant disagrees with the per-tile construction"


@requires_gpu
@pytest.mark.parametrize("n,k", SHAPES)
def test_requant_stays_within_one_int4_step_of_the_fp8_weight(n, k):
    """The nesting claim, on device: because the [128,128] block scale is constant across an
    int4 group, requantizing from fp8 can only cost the int4 rounding itself."""
    from vtl.patches import quant_w4a8

    wq, scale, deq = _fake_fp8_block_weight(n, k, seed=1)
    del wq, scale
    # quant_w4a8 groups along K after transposing to [K, N]; mirror that here.
    w_kn = deq.t().contiguous().float()
    g = w_kn.reshape(k // GROUP, GROUP, n)
    step = torch.maximum(g.amax(1).abs() / INT4_MAX, g.amin(1).abs() / -INT4_MIN)
    q = torch.clamp(torch.round(g / step.unsqueeze(1)), INT4_MIN, INT4_MAX)
    err = (q * step.unsqueeze(1) - g).abs() / step.unsqueeze(1).clamp_min(1e-30)
    assert float(err.max()) <= WEIGHT_STEP_TOL, float(err.max())


@requires_gpu
@pytest.mark.parametrize("n,k", SHAPES)
def test_layer_parity_against_the_dequantized_reference(n, k):
    """End to end through the REAL code: dequant with the patch's helper, pack with
    quant_w4a8's own ``_quantize_and_pack``, run ``cutlass_w4a8_mm``, and compare against a
    bf16 matmul of the dequantized weights.

    Comparing against the DEQUANTIZED weight (not the original fp32) is deliberate: it isolates
    the kernel + packing from the quantization error, so a packing bug cannot hide inside "int4
    is lossy".
    """
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

    from vtl.patches import quant_w4a8, w4a8_from_fp8

    wq, scale, _ = _fake_fp8_block_weight(n, k, seed=2)
    deq = w4a8_from_fp8.dequant_fp8_block(wq, scale)          # bf16 [N, K]
    packed, group_scales, chan_scales = quant_w4a8._quantize_and_pack(deq)

    # The int4-dequantized weight, i.e. what the kernel is actually multiplying by. Rebuilt
    # from the same grouping quant_w4a8 uses so the reference is the kernel's own operand.
    w_kn = deq.t().contiguous().float()
    g = w_kn.reshape(k // GROUP, GROUP, n)
    step = torch.maximum(g.amax(1).abs() / INT4_MAX, g.amin(1).abs() / -INT4_MIN)
    w_int4 = (torch.clamp(torch.round(g / step.unsqueeze(1)), INT4_MIN, INT4_MAX)
              * step.unsqueeze(1)).reshape(k, n).t().contiguous()

    quant_fp8 = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)
    for m in (1, 4, 16):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.5
        xq, x_scales = quant_fp8(x)
        got = ops.cutlass_w4a8_mm(
            a=xq, b_q=packed, b_group_scales=group_scales, b_group_size=GROUP,
            a_token_scales=x_scales, b_channel_scales=chan_scales, maybe_schedule=None,
        ).float()
        ref = (x.float() @ w_int4.t().float())
        rel = (got - ref).norm() / ref.norm().clamp_min(1e-12)
        assert float(rel) < REL_TOL, f"m={m} rel={float(rel):.4f} (tol {REL_TOL})"


@requires_gpu
def test_an_odd_block_grid_is_declined_rather_than_misread():
    """The fail-soft rung that matters: a checkpoint whose scale grid is not [128,128] must
    land on stock fp8, not be dequantized against the wrong scales."""
    from vtl.patches import w4a8_from_fp8

    assert w4a8_from_fp8.validate_block_layout((2048, 2048), (16, 16)) is None
    assert w4a8_from_fp8.validate_block_layout((2048, 2048), (16, 17)) is not None
    assert w4a8_from_fp8.validate_block_layout((2048, 2048), (32, 8)) is not None


# ---------------------------------------------------------------------------------------
# Self-check: the accuracy story, in numpy, with nothing installed.
# ---------------------------------------------------------------------------------------


def _self_check() -> None:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        print("test_w4a8_from_fp8 self-check skipped (numpy absent)")
        return

    rng = np.random.default_rng(0)
    n, k = 256, 512                      # 2x4 block tiles, 4 int4 groups per row
    w = rng.normal(0.0, 0.05, size=(n, k))
    # A few rows deliberately much larger, so at least one [128,128] tile has a scale far
    # from the others -- a uniform-magnitude matrix would make the nesting claim vacuous.
    w[:8] *= 300.0
    w[:, :64] *= 0.02

    # -- the geometric claim the whole accuracy story rests on --
    assert block_scale_is_constant_within_int4_group(n, k), \
        "a [128,128] block must contain a whole number of 128-wide int4 groups"

    w_fp8, bscales = block_quantize(w, np)
    fp8_err = np.abs(w_fp8 - w).max()
    assert fp8_err > 0.0, "the fp8 step must be a real step or this proves nothing"
    assert bscales.shape == (n // BLOCK_N, k // BLOCK_K)
    assert bscales.max() / bscales.min() > 100.0, "tiles must actually differ in scale"

    # -- ROUND TRIP: requant-from-fp8 vs quantizing the STORED fp8 values with the block
    #    scale factored out. If the nesting property holds, the int4 group scale is exactly
    #    (block scale) x (scale of the raw values) and the two produce the same LEVELS. --
    q_fp8, s_fp8, deq_from_fp8 = rtn_int4_groups(w_fp8, np)
    bs_full = np.repeat(np.repeat(bscales, BLOCK_N, axis=0), BLOCK_K, axis=1)[:n, :k]
    raw = w_fp8 / bs_full                                  # the stored fp8 values themselves
    q_raw, s_raw, _ = rtn_int4_groups(raw, np)
    # BLOCK_K == GROUP, so column ``g*GROUP`` names group g's block scale.
    bs_group = bs_full[:, ::GROUP]
    assert np.abs(s_fp8 - s_raw * bs_group).max() <= 1e-12 * float(s_fp8.max()), \
        "the block scale did not factor out of the group scale -- an int4 group is " \
        "straddling two block scales, and the accuracy argument does not hold"
    # The LEVELS agree except at exact ties: v/s is a ratio of dyadic rationals times 7 or 8,
    # so landing on exactly k+0.5 is common, and there a 1-ulp difference in the scale flips
    # round-half-to-even. Every mismatch must be such a tie, and must be one level, never two.
    mismatch = q_fp8 != q_raw
    assert np.abs(q_fp8 - q_raw).max() <= 1, "requant levels differ by more than one step"
    t = (raw.reshape(n, k // GROUP, GROUP) / s_raw[..., None]).reshape(n, k)
    residual = np.abs(t - np.round(t))
    assert mismatch.mean() < 0.05, mismatch.mean()
    if mismatch.any():
        assert np.abs(residual[mismatch] - 0.5).max() < 1e-9, \
            "a level disagreed away from a rounding tie -- the factoring is not exact"

    # -- ERROR BOUND: the double quantization must ADD, not compound --
    _, _, deq_direct = rtn_int4_groups(w, np)
    direct_err = np.abs(deq_direct - w).max()
    from_fp8_err = np.abs(deq_from_fp8 - w).max()
    # |requant(fp8(w)) - w| <= |requant(fp8(w)) - fp8(w)| + |fp8(w) - w|, and the first term is
    # bounded by the direct RTN error plus whatever the group absmax moved (at most fp8_err).
    # The 1.10 is headroom for that absmax drift; the assertion IS the accuracy claim.
    assert from_fp8_err <= 1.10 * direct_err + fp8_err, (from_fp8_err, direct_err, fp8_err)
    # ...and it must not be dramatically worse than direct-from-fp32 either, or "int4 from an
    # fp8 checkpoint" would be a different quality proposition from "int4 from bf16".
    assert from_fp8_err <= 2.5 * direct_err, (from_fp8_err, direct_err)

    # -- the same statement in RMS terms, which is what a perplexity delta actually tracks --
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))  # noqa: E731
    rms_direct = rms(deq_direct - w)
    rms_from_fp8 = rms(deq_from_fp8 - w)
    assert rms_from_fp8 <= 1.6 * rms_direct, (rms_from_fp8, rms_direct)
    # Relative to the weights themselves this is the number to quote in a handoff.
    rel = rms_from_fp8 / rms(w)
    assert rel < 0.12, rel

    # -- the int4 level formula: asymmetric range, and the levels are actually used --
    q, scale, _ = rtn_int4_groups(w, np)
    assert q.min() >= INT4_MIN and q.max() <= INT4_MAX
    assert (q == INT4_MIN).any() or (q == INT4_MAX).any(), "no group reached a rail"
    assert scale.shape == (n, k // GROUP)
    # A group whose min dominates must use the /8 branch, not /7.
    col = np.zeros((1, GROUP))
    col[0, 0], col[0, 1] = 1.0, -1.5
    _, s, _ = rtn_int4_groups(col, np)
    assert math.isclose(float(s[0, 0]), 1.5 / 8.0, rel_tol=1e-12), float(s[0, 0])

    # -- e4m3 rounding model sanity: exact on representable values, bounded elsewhere --
    for v in (0.0, 1.0, 2.0, 448.0, -448.0, 0.5, 1.25):
        assert float(e4m3_round(np.array([v]), np)[0]) == v, v
    x = rng.normal(0, 10, size=4096)
    x = np.clip(x, -FP8_MAX, FP8_MAX)
    r = e4m3_round(x, np)
    rel_err = np.abs(r - x) / np.maximum(np.abs(x), 1e-30)
    assert rel_err.max() <= 2.0 ** -(FP8_MANTISSA_BITS + 1) + 1e-12, rel_err.max()

    # -- and the patch's own pure half must agree with this file about the grid --
    try:
        from vtl.patches import w4a8_from_fp8
    except Exception as exc:  # pragma: no cover -- vLLM absent is fine, it is a lazy import
        print(f"test_w4a8_from_fp8 self-check ok (patch not importable here: {exc})")
        return
    assert w4a8_from_fp8.BLOCK_N == BLOCK_N and w4a8_from_fp8.BLOCK_K == BLOCK_K
    assert w4a8_from_fp8.scale_grid(n, k) == (n // BLOCK_N, k // BLOCK_K)
    assert w4a8_from_fp8.validate_block_layout((n, k), (n // BLOCK_N, k // BLOCK_K)) is None
    assert w4a8_from_fp8.runtime_enabled("0") is False

    print("test_w4a8_from_fp8 self-check ok")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    elif not HAVE_PYTEST:
        raise SystemExit("pytest not installed; run with --self-check")
    else:
        raise SystemExit(pytest.main([__file__, "-q"]))
