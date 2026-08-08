"""Shared fp8 parity helpers for the vtl kernel tests.

Every kernel is validated against a pure-torch reference under BOTH settings -- our kernel
(default) and stock vLLM (VTL_SKIP_EXT=1) -- because importing vtl._C overrides _C
process-wide, so the two implementations cannot coexist. Agreeing with the same oracle under
both is what proves ours matches stock. The tolerances below are lifted verbatim from
test_rms_norm_quant.py so every fp8 kernel is judged on the same bar.
"""

from __future__ import annotations

import torch

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
MIN_SCALE = 1.0 / (FP8_MAX * 512.0)


def mismatch_fraction(a, b):
    """Fraction of differing fp8 codes. Block-reduction order differs from torch's pairwise
    sum, so a scale can land a ulp away and flip the odd fp8 code."""
    ca = a.view(torch.uint8).to(torch.int32)
    cb = b.view(torch.uint8).to(torch.int32)
    return (ca != cb).float().mean().item()


# Relative spacing of adjacent representable values (2**-mantissa_bits).
_DTYPE_ULP = {torch.float16: 2.0**-10, torch.bfloat16: 2.0**-7, torch.float32: 2.0**-23}


def scale_rtol(dtype):
    """Our fp32 block reduction and torch's differ in summation order, so an amax element can
    cross at most one dtype rounding boundary -> scale moves by up to one dtype ULP. The 2x is
    headroom; the 1e-5 floor covers fp32 where accumulation error dominates."""
    return max(2.0 * _DTYPE_ULP[dtype], 1e-5)


def assert_fp8_equivalent(got, exp, dtype):
    """MAGNITUDE bound: every code differs from the reference by at most one e4m3 step (3-bit
    mantissa -> <=1/8 relative; smallest subnormal 2**-9). FRACTION bound: guards a systematic
    rounding bias -- reduction-order noise flips <~0.2% of codes, a wrong rule flips ~half, so
    1% sits ~5x above noise / ~50x below a bug."""
    torch.testing.assert_close(got.float(), exp.float(), rtol=0.13, atol=2.0**-9)
    frac = mismatch_fraction(got, exp)
    assert frac < 1e-2, f"{frac:.2%} of fp8 codes differ -- systematic, not reduction noise"
