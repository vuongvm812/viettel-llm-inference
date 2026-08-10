"""Correctness + micro-benchmark for vtl's fused rms_norm + per-token fp8 quant kernel.

Needs a GPU. Run inside the container:

    pytest bench/test_rms_norm_quant.py -q                  # our kernel
    VTL_SKIP_EXT=1 pytest bench/test_rms_norm_quant.py -q   # stock kernel, same oracle

Importing ``vtl._C`` overrides ``_C``'s CUDA kernel process-wide, so the two
implementations cannot coexist. Both are therefore checked against the same pure-torch
reference; passing under both settings is what proves ours matches stock.

    python bench/test_rms_norm_quant.py                     # benchmark, ours
    VTL_SKIP_EXT=1 python bench/test_rms_norm_quant.py      # benchmark, stock
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
MIN_SCALE = 1.0 / (FP8_MAX * 512.0)

USE_STOCK = os.environ.get("VTL_SKIP_EXT", "").strip() in ("1", "true", "yes", "on")


def _op():
    import vllm._C_stable_libtorch  # noqa: F401  -- defines the _C schema

    if USE_STOCK:
        return torch.ops._C.rms_norm_dynamic_per_token_quant

    try:
        import vtl._C  # noqa: F401  -- overrides _C's CUDA kernel; defines vllm_cuda::
    except ModuleNotFoundError as exc:  # pragma: no cover - environment problem
        raise RuntimeError(
            "vtl._C is missing from this image. $(IMAGE):$(TAG) also names a Docker Hub "
            "repo, so an un-rebuilt tag silently pulls a published image without the "
            "extension. Run `make build` (the Makefile targets now depend on it), or set "
            "VTL_SKIP_EXT=1 to test vLLM's stock kernel instead."
        ) from exc

    return torch.ops.vllm_cuda.rms_norm_dynamic_per_token_quant


OP = _op()


def reference(x_in, weight, eps, residual=None, scale_ub=None):
    """Independent restatement of the kernel's documented semantics."""
    x = x_in.float()
    new_residual = None
    if residual is not None:
        x = x + residual.float()
        new_residual = x.to(x_in.dtype)

    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    # Two roundings, both in x_in.dtype. c10::BFloat16's operator*(BFloat16, BFloat16)
    # computes in float but RETURNS BFloat16, so stock's
    # `static_cast<scalar_t>(x * rms) * weight[i]` narrows the weight product as well.
    # Widening before the weight multiply is more accurate but shifts amax, and therefore
    # the per-token scale, by ~6e-5 relative -- a different kernel, not a better one.
    y = ((x * rms).to(x_in.dtype) * weight).float()

    amax = y.abs().amax(-1, keepdim=True)
    if scale_ub is not None:
        amax = torch.minimum(amax, scale_ub)
    scale = torch.clamp(amax / FP8_MAX, min=MIN_SCALE)
    out = torch.clamp(y / scale, -FP8_MAX, FP8_MAX).to(FP8)
    return out, scale, new_residual


def run(x_in, weight, eps, residual=None, scale_ub=None):
    # Explicitly contiguous: x_in may be a strided view, and the kernel requires a
    # contiguous result/residual while accepting a strided input.
    out = torch.empty(x_in.shape, dtype=FP8, device=x_in.device)
    scale = torch.empty((x_in.shape[0], 1), dtype=torch.float32, device=x_in.device)
    res = residual.contiguous().clone() if residual is not None else None
    OP(out, x_in, weight, scale, eps, scale_ub, res)
    return out, scale, res


def _skip_if_it_will_not_fit(num_tokens, hidden, dtype):
    """The big shapes are sized for an H200. reference() materialises several fp32
    temporaries, so the worst case (8192 x 12288 fp32) wants ~4 GiB. Skip rather than OOM
    on a dev GPU."""
    itemsize = torch.tensor([], dtype=dtype).element_size()
    # x + residual + clone (itemsize each), fp8 out (1), ~6 fp32 temporaries in reference()
    est = num_tokens * hidden * (3 * itemsize + 1 + 6 * 4)
    free = torch.cuda.mem_get_info()[0]
    if est > 0.7 * free:
        pytest.skip(f"needs ~{est / 2**30:.1f} GiB, only {free / 2**30:.1f} GiB free")


def _mismatch_fraction(a, b):
    """Fraction of differing fp8 codes. Block-reduction order differs from torch's
    pairwise sum, so `rms` can land a ulp away and flip the odd fp8 code."""
    ca = a.view(torch.uint8).to(torch.int32)
    cb = b.view(torch.uint8).to(torch.int32)
    return (ca != cb).float().mean().item()


# Relative spacing of adjacent representable values (2**-mantissa_bits).
_DTYPE_ULP = {torch.float16: 2.0**-10, torch.bfloat16: 2.0**-7, torch.float32: 2.0**-23}


def _scale_rtol(dtype):
    """scale = amax(|y|)/448, and y is rounded to `dtype`. Our fp32 block reduction and
    torch's differ in summation order, so `rms` differs at the last fp32 bits; that can
    push the amax element across at most one `dtype` rounding boundary, moving scale by up
    to one dtype ULP. (Verified separately that the fp8 converters themselves agree
    bit-for-bit, so this is the whole story.) The 2x is headroom; the 1e-5 floor covers
    fp32, where reduction-accumulation error (~log2(H)*2**-24) dominates over the ULP."""
    return max(2.0 * _DTYPE_ULP[dtype], 1e-5)


def _assert_fp8_equivalent(got, exp, dtype):
    """The rigorous bound is on MAGNITUDE, not count: every code differs from the
    reference by at most one e4m3 step (3-bit mantissa -> <=1/8 relative; smallest
    subnormal 2**-9). A stride or indexing bug surfaces here as many-step errors.

    The fraction bound is separate and only guards against a SYSTEMATIC rounding bias:
    reduction-order noise flips a handful of near-boundary codes (<~0.2% even for bf16 on
    small samples), whereas a wrong rounding rule flips ~half. 1% sits ~5x above the noise
    and ~50x below a bug -- a separation, not a fit to make the suite pass."""
    torch.testing.assert_close(got.float(), exp.float(), rtol=0.13, atol=2.0**-9)
    frac = _mismatch_fraction(got, exp)
    assert frac < 1e-2, f"{frac:.2%} of fp8 codes differ -- systematic, not reduction noise"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("num_tokens", [1, 7, 256, 1000, 8192])
@pytest.mark.parametrize(
    "hidden",
    [
        2048,  # a common hidden size, fast path (256 threads x 8, no loop)
        2050,  # not a multiple of the vector width -> generic fallback
        12288,  # too wide to hold in registers -> generic fallback
    ],
)
@pytest.mark.parametrize("with_residual", [False, True])
def test_matches_reference(dtype, num_tokens, hidden, with_residual):
    _skip_if_it_will_not_fit(num_tokens, hidden, dtype)
    torch.manual_seed(0)
    dev = "cuda"
    eps = 1e-6
    x = torch.randn(num_tokens, hidden, dtype=dtype, device=dev)
    w = torch.randn(hidden, dtype=dtype, device=dev)
    res = torch.randn(num_tokens, hidden, dtype=dtype, device=dev) if with_residual else None

    got_out, got_scale, got_res = run(x, w, eps, res)
    exp_out, exp_scale, exp_res = reference(x, w, eps, res)

    torch.testing.assert_close(got_scale, exp_scale, rtol=_scale_rtol(dtype), atol=0)
    if with_residual:
        # residual is just (input + residual) rounded: no reduction, must be bit-exact
        torch.testing.assert_close(got_res, exp_res, rtol=0, atol=0)
    _assert_fp8_equivalent(got_out, exp_out, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("with_residual", [False, True])
def test_row_stride_larger_than_hidden(dtype, with_residual):
    """Stays on the fast path with input_stride=2056 != hidden_size=2048. This is the
    only case where confusing `in_off` (strided) with `row_off` (packed) shows up."""
    torch.manual_seed(0)
    x = torch.randn(64, 2056, dtype=dtype, device="cuda")[:, :2048]
    assert x.stride(0) == 2056 and x.stride(1) == 1
    w = torch.randn(2048, dtype=dtype, device="cuda")
    res = torch.randn(64, 2048, dtype=dtype, device="cuda") if with_residual else None

    got_out, got_scale, got_res = run(x, w, 1e-6, res)
    exp_out, exp_scale, exp_res = reference(x, w, 1e-6, res)

    torch.testing.assert_close(got_scale, exp_scale, rtol=_scale_rtol(dtype), atol=0)
    if with_residual:
        torch.testing.assert_close(got_res, exp_res, rtol=0, atol=0)
    _assert_fp8_equivalent(got_out, exp_out, dtype)


@pytest.mark.skipif(
    USE_STOCK,
    reason="stock has no alignment guard: it takes the vectorised path unconditionally "
    "and faults (cudaErrorMisalignedAddress), poisoning the context for later tests. The "
    "generic fallback is a vtl-only property, so this asserts our kernel, not stock's.",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_misaligned_input_falls_back_to_generic(dtype):
    """A one-element storage offset breaks the 16-byte alignment the vectorised loads
    need, so `fast` must be false. Nothing else in the suite reaches the generic kernel
    with an otherwise fast-path-shaped tensor."""
    torch.manual_seed(0)
    flat = torch.randn(4 * 2048 + 1, dtype=dtype, device="cuda")
    x = flat[1:].view(4, 2048)
    assert x.data_ptr() % 16 != 0, "offset did not break 16-byte alignment"
    w = torch.randn(2048, dtype=dtype, device="cuda")

    got_out, got_scale, _ = run(x, w, 1e-6, None)
    exp_out, exp_scale, _ = reference(x, w, 1e-6, None)
    torch.testing.assert_close(got_scale, exp_scale, rtol=_scale_rtol(dtype), atol=0)
    _assert_fp8_equivalent(got_out, exp_out, dtype)


def test_weight_product_is_narrowed_not_widened():
    """Pins the numerics contract the oracle encodes, independent of the ULP-scale
    tolerances above: stock forms the weight product IN scalar_t (bf16*bf16 -> bf16), it
    does not keep it in fp32. Both are ~1-ulp effects, so a scalar tolerance can't
    distinguish them -- this test does it directly.

    Construct two references, `narrow` (correct) and `wide` (the fp32-product bug); they
    differ systematically in a few percent of fp8 codes. The kernel must land on `narrow`.
    Reduction-order noise only flips <~0.2% of codes, far below that gap, so the verdict
    is unambiguous."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    x = torch.randn(1, 2048, dtype=dtype, device="cuda")
    w = torch.randn(2048, dtype=dtype, device="cuda")

    xf = x.float()
    rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
    xr = (xf * rms).to(dtype)
    y_narrow = (xr * w).float()  # scalar_t * scalar_t, as stock does
    y_wide = xr.float() * w.float()  # the bug: fp32 product

    def quant(y):
        s = torch.clamp(y.abs().amax(-1, keepdim=True) / FP8_MAX, min=MIN_SCALE)
        return torch.clamp(y / s, -FP8_MAX, FP8_MAX).to(FP8)

    o_narrow, o_wide = quant(y_narrow), quant(y_wide)
    gap = _mismatch_fraction(o_narrow, o_wide)
    assert gap > 0.01, f"crafted case too weak to distinguish narrow/wide ({gap:.2%})"

    got_out, _, _ = run(x, w, 1e-6, None)
    d_narrow = _mismatch_fraction(got_out, o_narrow)
    d_wide = _mismatch_fraction(got_out, o_wide)
    assert d_narrow < d_wide / 4, (
        f"kernel is {d_narrow:.2%} from the narrowed product but {d_wide:.2%} from the "
        f"fp32-widened one -- the weight multiply is not narrowing to {dtype}"
    )


def test_scale_ub_is_honoured():
    torch.manual_seed(0)
    x = torch.randn(64, 2048, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(2048, dtype=torch.bfloat16, device="cuda")
    ub = torch.tensor([0.5], dtype=torch.float32, device="cuda")

    _, got_scale, _ = run(x, w, 1e-6, None, ub)
    assert (got_scale <= 0.5 / FP8_MAX + 1e-9).all()


def test_tiny_input_clamps_to_min_scale():
    x = torch.zeros(4, 2048, dtype=torch.bfloat16, device="cuda")
    w = torch.ones(2048, dtype=torch.bfloat16, device="cuda")
    _, scale, _ = run(x, w, 1e-6, None, None)
    torch.testing.assert_close(scale, torch.full_like(scale, MIN_SCALE), rtol=1e-6, atol=0)


@pytest.mark.skipif(USE_STOCK, reason="asserts the override, not the kernel")
def test_override_is_installed():
    """Our kernel rejects int8 output; stock accepts it. Cheapest unambiguous probe
    that _C's CUDA kernel is really ours."""
    x = torch.randn(4, 2048, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(2048, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(4, 2048, dtype=torch.int8, device="cuda")
    scale = torch.empty((4, 1), dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="fp8_e4m3 output only"):
        torch.ops._C.rms_norm_dynamic_per_token_quant(out, x, w, scale, 1e-6, None, None)


def _bench():
    import time

    label = "stock _C" if USE_STOCK else "vtl"
    major, minor = torch.cuda.get_device_capability()
    print(f"impl: {label}   device: {torch.cuda.get_device_name()} (sm_{major}{minor})")
    if (major, minor) < (8, 9):
        print(
            "\nWARNING: sm_89 is the first arch with a hardware fp8 converter\n"
            "  (cvt.rn.satfinite.e4m3x2.f32). Below it, CUDA's header emits a software\n"
            "  conversion, so the quantise epilogue is ALU-bound rather than free and\n"
            "  these numbers do not transfer to the H200. Correctness still holds."
        )
    print(f"\n{'shape':>16} {'us/call':>10} {'GB/s':>8}")
    for num_tokens in (256, 8192):  # decode (max-num-seqs), prefill (max-num-batched)
        hidden = 2048
        # The kernel does residual := input + residual, so a full-size input would double
        # it every iteration and reach inf within the timed loop. A small input leaves the
        # memory traffic identical and the values bounded (200 iters drift it by ~0.2 sd).
        x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda") * 1e-3
        w = torch.randn(hidden, dtype=torch.bfloat16, device="cuda")
        res = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
        out = torch.empty_like(x, dtype=FP8)
        scale = torch.empty((num_tokens, 1), dtype=torch.float32, device="cuda")

        for _ in range(20):
            OP(out, x, w, scale, 1e-6, None, res)
        torch.cuda.synchronize()

        iters = 200
        t0 = time.perf_counter()
        for _ in range(iters):
            OP(out, x, w, scale, 1e-6, None, res)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / iters * 1e6

        # read input + residual (2B each), write residual (2B) + out (1B)
        moved = num_tokens * hidden * 7
        print(f"{num_tokens:>7}x{hidden:<8} {us:>10.1f} {moved / us / 1e3:>8.0f}")


if __name__ == "__main__":
    _bench()
