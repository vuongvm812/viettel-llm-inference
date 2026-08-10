"""NVRTC harness: cache-key/gating logic off-box, real compile + numeric parity on a GPU.

    python3 bench/test_nvrtc.py --self-check   # no GPU, no cuda-python (runs in `make check`)
    pytest -q bench/test_nvrtc.py              # GPU: compiles for real and checks the math

The GPU half is the one that matters. A shape-specialized kernel that is FAST and WRONG is
the failure mode this whole harness invites, so the parity check compares against a torch
reference computed with the same double-rounding the kernel is required to reproduce.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vtl import nvrtc  # noqa: E402

# `make check` runs this file's --self-check on a bare host with neither pytest nor torch.
# Importing either at module scope unconditionally would turn "no GPU tooling here" into a
# failed gate, so both are optional and only the pytest half hard-requires them.
try:
    import pytest

    HAVE_PYTEST = True
except ImportError:  # pragma: no cover -- the `make check` path
    HAVE_PYTEST = False

    class _NoPytest:
        """Just enough of pytest for the decorators below to evaluate at import time.

        The test bodies are never RUN without pytest -- only defined -- so these need to
        be no-ops, not working implementations.
        """

        class mark:
            @staticmethod
            def skipif(*a, **k):
                return lambda fn: fn

            @staticmethod
            def parametrize(*a, **k):
                return lambda fn: fn

    pytest = _NoPytest()

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

requires_gpu = pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(), reason="needs torch + a CUDA device"
)

HIDDEN = 2048
THREADS = 256
FP8_MAX = 448.0
MIN_SCALE = 1.0 / (448.0 * 512.0)


def reference(x_bf16, w_bf16, residual=None, eps=1e-6):
    """The kernel's exact arithmetic in torch, double rounding included.

    Written to MATCH, not to be ideal: `bf16(x*rms) * bf16(w)` narrows the product, which
    moves amax and therefore the per-token scale. A float32 reference would disagree with
    both this kernel and the AOT one.
    """
    x = x_bf16.float()
    if residual is not None:
        x = x + residual.float()
        new_residual = x.to(torch.bfloat16)
    else:
        new_residual = None
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    xn = (x * rms).to(torch.bfloat16)
    y = (xn * w_bf16).to(torch.bfloat16).float()
    amax = y.abs().amax(-1, keepdim=True)
    scale = torch.clamp(amax / FP8_MAX, min=MIN_SCALE)
    q = torch.clamp(y / scale, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale.squeeze(-1), new_residual


def _compile(has_residual: int):
    """Compile the shipped kernel source for one specialization, or skip with the reason."""
    src = nvrtc.load_source("rms_norm_quant")
    assert src, "vtl/kernels/rms_norm_quant.cu must ship with the package"
    built = nvrtc.build_cubin(
        src,
        "rms_norm_quant",
        {"HIDDEN": HIDDEN, "THREADS": THREADS, "HAS_RESIDUAL": has_residual},
    )
    if built is None:
        pytest.skip("NVRTC unavailable (no cuda-python / no driver) -- AOT path stands")
    return nvrtc._load_cubin(built[0], "rms_norm_quant")


@requires_gpu
@pytest.mark.parametrize("has_residual", [0, 1])
@pytest.mark.parametrize("num_tokens", [1, 7, 64])
def test_nvrtc_kernel_matches_the_reference(has_residual, num_tokens):
    from cuda.bindings import driver  # noqa: F401  (skip early if absent)

    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device=dev)
    w = torch.randn(HIDDEN, dtype=torch.bfloat16, device=dev)
    res = (
        torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device=dev)
        if has_residual
        else None
    )
    res_in = res.clone() if res is not None else None

    out = torch.empty(num_tokens, HIDDEN, dtype=torch.float8_e4m3fn, device=dev)
    scales = torch.empty(num_tokens, dtype=torch.float32, device=dev)
    # A kernel that never writes would otherwise "match" an all-zero expectation.
    out.fill_(torch.finfo(torch.float8_e4m3fn).max)
    scales.fill_(float("nan"))

    launch = _compile(has_residual)
    residual_ptr = res.data_ptr() if res is not None else 0
    launch(
        grid=(num_tokens, 1, 1),
        block=(THREADS, 1, 1),
        args=nvrtc.pack_args(
            out.data_ptr(), scales.data_ptr(), x.data_ptr(), w.data_ptr(),
            residual_ptr, ("f", 1e-6),
        ),
    )
    torch.cuda.synchronize()

    q_ref, s_ref, res_ref = reference(x, w, res_in, eps=1e-6)
    assert torch.allclose(scales, s_ref, rtol=1e-5, atol=0), (scales, s_ref)
    # fp8 is exact here: same inputs, same rounding mode -> bit-identical, not merely close.
    assert torch.equal(out.view(torch.uint8), q_ref.view(torch.uint8))
    if res is not None:
        assert torch.equal(res.view(torch.uint8), res_ref.view(torch.uint8))


@requires_gpu
def test_cubin_cache_is_reused_not_recompiled(tmp_path, monkeypatch):
    """The second compile must come off disk -- otherwise the judge's boot pays for it."""
    monkeypatch.setenv("VTL_NVRTC_CACHE", str(tmp_path))
    src = nvrtc.load_source("rms_norm_quant")
    defines = {"HIDDEN": HIDDEN, "THREADS": THREADS, "HAS_RESIDUAL": 0}
    first = nvrtc.build_cubin(src, "rms_norm_quant", defines)
    if first is None:
        pytest.skip("NVRTC unavailable")
    cubins = list(tmp_path.glob("*.cubin"))
    assert len(cubins) == 1, cubins
    stamp = cubins[0].stat().st_mtime_ns
    second = nvrtc.build_cubin(src, "rms_norm_quant", defines)
    assert second[0] == first[0]
    assert cubins[0].stat().st_mtime_ns == stamp, "cache was rewritten, i.e. recompiled"

    # A different specialization is a different cubin, not a cache hit on the first.
    nvrtc.build_cubin(src, "rms_norm_quant", {**defines, "HIDDEN": 4096})
    assert len(list(tmp_path.glob("*.cubin"))) == 2


@requires_gpu
def test_a_broken_kernel_degrades_instead_of_raising():
    """The whole safety story: a bad kernel must return None, never take the server down."""
    assert nvrtc.build_cubin("this is not CUDA", "broken", {"HIDDEN": 8}) is None


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        nvrtc._self_check()
        if torch is None:
            print("test_nvrtc self-check ok (torch absent; reference check skipped)")
        else:
            # The reference must be self-consistent on CPU even with no GPU present.
            x = torch.randn(4, 64, dtype=torch.bfloat16)
            w = torch.randn(64, dtype=torch.bfloat16)
            q, s, _ = reference(x, w)
            assert q.shape == (4, 64) and s.shape == (4,)
            assert (s >= MIN_SCALE).all(), "scale floor must hold for every token"
            # An all-zero token is the degenerate case the floor exists for.
            _, s0, _ = reference(torch.zeros(1, 64, dtype=torch.bfloat16), w)
            assert abs(float(s0[0]) - MIN_SCALE) < 1e-12, s0
            print("test_nvrtc self-check ok")
    elif not HAVE_PYTEST:
        raise SystemExit("pytest not installed; run with --self-check")
    else:
        raise SystemExit(pytest.main([__file__, "-q"]))
