"""Block-quant NVRTC harness: gating/identity off-box, real compile + parity on a GPU.

    python3 bench/test_nvrtc_block_quant.py --self-check   # no GPU, no torch, no vLLM
    pytest -q bench/test_nvrtc_block_quant.py              # GPU: compiles and checks the math

Same split and doctrine as bench/test_nvrtc.py. The GPU half is the one that matters, and
it is stricter than "close": vtl/patches/nvrtc_block_quant.py REPLACES the CUDA kernel
behind two stock ops, so anything short of the stock formula bit for bit is a silent model
change, not an optimization. The oracles below therefore reproduce the STOCK kernels'
arithmetic -- which is NOT the same as vtl's own per-token kernel:

  * ``_C::rms_norm_per_block_quant`` (csrc/libtorch_stable/quantization/fused_kernels/
    fused_layernorm_dynamic_per_token_quant.cu) narrows TWICE -- ``bf16(x*rms)`` and then
    the homogeneous ``bf16 * bf16`` weight multiply -- takes the abs-max per GROUP of 128
    rather than per token, and quantizes by TRUE DIVISION (``ScaledQuant`` with
    is_scale_inverted=false, "for exact match with FBGemm"). vtl's per-token kernel
    reciprocal-multiplies instead; using that oracle here would pass a wrong kernel.
  * ``_C::per_token_group_fp8_quant`` (csrc/libtorch_stable/quantization/w8a8/fp8/
    per_token_group_quant.cu) SEEDS the group abs-max with ``eps`` and applies no scale
    floor at all -- again unlike the rms path's ``min_scaling_factor``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vtl import nvrtc  # noqa: E402
from vtl.patches import nvrtc_block_quant as patch  # noqa: E402

# `make check` runs --self-check on a bare host with neither pytest nor torch; see the
# note in bench/test_nvrtc.py. Both stay optional and only the pytest half requires them.
try:
    import pytest

    HAVE_PYTEST = True
except ImportError:  # pragma: no cover -- the `make check` path
    HAVE_PYTEST = False

    class _NoPytest:
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

HIDDEN = 3072          # Qwen3.5-122B-A10B text hidden size
GROUP = 128
THREADS = HIDDEN // 8  # 384: one vec8 per thread, as the kernel's static_assert demands
GQ_THREADS = patch.GQ_THREADS
FP8_MAX = 448.0
FP8_MIN = -448.0
MIN_SCALE = 1.0 / (448.0 * 512.0)


# --------------------------------------------------------------------------------------
# oracles: the STOCK kernels' arithmetic, transcribed
# --------------------------------------------------------------------------------------

def _to_fp8(v):
    """``float_to_fp8``: saturate to +/-448 FIRST, then convert (quant_conversions.cuh)."""
    return torch.clamp(v, FP8_MIN, FP8_MAX).to(torch.float8_e4m3fn)


def rms_reference(x_bf16, w_bf16, residual=None, eps=1e-6, scale_ub=None, group=GROUP):
    """rms_norm_per_block_quant, written to MATCH rather than to be ideal.

    Both narrowings are load-bearing: keeping ``y`` in fp32 would move each group's
    abs-max, hence its scale, hence every fp8 code in the group.
    """
    x = x_bf16.float()
    new_residual = None
    if residual is not None:
        x = x + residual.float()
        new_residual = x.to(torch.bfloat16)
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    y = ((x * rms).to(torch.bfloat16) * w_bf16).to(torch.bfloat16).float()

    tokens = y.shape[0]
    groups = y.shape[1] // group
    g = y.view(tokens, groups, group)
    amax = g.abs().amax(-1)
    if scale_ub is not None:
        amax = torch.minimum(amax, scale_ub.float())
    scale = torch.clamp(amax / FP8_MAX, min=MIN_SCALE)      # min_scaling_factor floor
    q = _to_fp8(g / scale.unsqueeze(-1))                    # TRUE division, not 1/s
    return q.view(tokens, -1), scale, new_residual


def group_reference(x_bf16, eps=1e-10, group=GROUP):
    """per_token_group_fp8_quant: abs-max SEEDED with eps, and no scale floor."""
    x = x_bf16.reshape(-1, group).float()
    y_s = x.abs().amax(-1).clamp_min(eps) / FP8_MAX
    q = _to_fp8(x / y_s.unsqueeze(-1))
    return q, y_s


# --------------------------------------------------------------------------------------
# compile helpers
# --------------------------------------------------------------------------------------

def _source():
    src = nvrtc.load_source(patch.KERNEL)
    assert src, "vtl/kernels/rms_norm_block_quant.cu must ship with the package"
    return src


def _compile(entry: str, has_residual: int = 0, hidden: int = HIDDEN):
    """Compile the shipped source for one specialization, or skip with the reason."""
    built = nvrtc.build_cubin(
        _source(), patch.KERNEL, patch._defines(hidden, GROUP, has_residual)
    )
    if built is None:
        pytest.skip("NVRTC unavailable (no cuda-python / no driver) -- stock ops stand")
    return nvrtc._load_cubin(built[0], entry)


def _launch_rms(launch, out, scales, x, w, residual, eps, transposed=0, outer_stride=1,
                scale_ub=None):
    launch(
        grid=(x.shape[0], 1, 1),
        block=(THREADS, 1, 1),
        args=nvrtc.pack_args(
            out.data_ptr(),
            scales.data_ptr(),
            x.data_ptr(),
            w.data_ptr(),
            scale_ub.data_ptr() if scale_ub is not None else 0,
            residual.data_ptr() if residual is not None else 0,
            ("f", eps),
            ("l", x.stride(0)),
            ("i", transposed),
            ("l", outer_stride),
        ),
    )
    torch.cuda.synchronize()


# --------------------------------------------------------------------------------------
# GPU half
# --------------------------------------------------------------------------------------

@requires_gpu
@pytest.mark.parametrize("has_residual", [0, 1])
@pytest.mark.parametrize("num_tokens", [1, 7, 64])
def test_rms_entry_matches_the_stock_oracle(has_residual, num_tokens):
    from cuda.bindings import driver  # noqa: F401  (skip early if absent)

    torch.manual_seed(0)
    dev = "cuda"
    groups = HIDDEN // GROUP
    x = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device=dev)
    w = torch.randn(HIDDEN, dtype=torch.bfloat16, device=dev)
    res = (
        torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device=dev)
        if has_residual
        else None
    )
    res_in = res.clone() if res is not None else None

    out = torch.empty(num_tokens, HIDDEN, dtype=torch.float8_e4m3fn, device=dev)
    scales = torch.empty(num_tokens, groups, dtype=torch.float32, device=dev)
    # A kernel that never writes would otherwise "match" an all-zero expectation.
    out.fill_(torch.finfo(torch.float8_e4m3fn).max)
    scales.fill_(float("nan"))

    _launch_rms(_compile(patch.KERNEL, has_residual), out, scales, x, w, res, 1e-6)

    q_ref, s_ref, res_ref = rms_reference(x, w, res_in, eps=1e-6)
    assert torch.allclose(scales, s_ref, rtol=1e-5, atol=0), (scales, s_ref)
    # Same inputs, same rounding mode -> bit-identical, not merely close.
    assert torch.equal(out.view(torch.uint8), q_ref.view(torch.uint8))
    if res is not None:
        assert torch.equal(res.view(torch.uint8), res_ref.view(torch.uint8))


@requires_gpu
def test_rms_entry_scale_ub_clamps_every_group():
    """scale_ub caps the abs-max BEFORE the /448 and the floor -- stock's order."""
    from cuda.bindings import driver  # noqa: F401

    torch.manual_seed(1)
    dev, num_tokens = "cuda", 5
    groups = HIDDEN // GROUP
    x = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device=dev)
    w = torch.randn(HIDDEN, dtype=torch.bfloat16, device=dev)
    # Low enough to bind on essentially every group, so the branch is actually taken.
    ub = torch.tensor([0.25], dtype=torch.float32, device=dev)

    out = torch.empty(num_tokens, HIDDEN, dtype=torch.float8_e4m3fn, device=dev)
    scales = torch.full((num_tokens, groups), float("nan"), dtype=torch.float32, device=dev)
    _launch_rms(_compile(patch.KERNEL, 0), out, scales, x, w, None, 1e-6, scale_ub=ub)

    q_ref, s_ref, _ = rms_reference(x, w, None, eps=1e-6, scale_ub=ub)
    assert (s_ref < 0.25 / FP8_MAX + 1e-12).all(), "the ub must actually bind in this test"
    assert torch.allclose(scales, s_ref, rtol=1e-5, atol=0)
    assert torch.equal(out.view(torch.uint8), q_ref.view(torch.uint8))


@requires_gpu
@pytest.mark.parametrize("outer_stride", [1, 4])
def test_rms_entry_transposed_scale_layout(outer_stride):
    """scales[g * scale_rows + token] with scale_rows = ceil(T/outer)*outer -- verbatim.

    The padded case (outer_stride > 1) is the one worth a test: get scale_rows wrong and
    every group but the first lands on another group's row, which a non-transposed test
    cannot see.
    """
    from cuda.bindings import driver  # noqa: F401

    torch.manual_seed(2)
    dev, num_tokens = "cuda", 7
    groups = HIDDEN // GROUP
    scale_rows = -(-num_tokens // outer_stride) * outer_stride
    x = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device=dev)
    w = torch.randn(HIDDEN, dtype=torch.bfloat16, device=dev)

    out = torch.empty(num_tokens, HIDDEN, dtype=torch.float8_e4m3fn, device=dev)
    flat = torch.full((groups * scale_rows,), float("nan"), dtype=torch.float32, device=dev)
    _launch_rms(_compile(patch.KERNEL, 0), out, flat, x, w, None, 1e-6,
                transposed=1, outer_stride=outer_stride)

    _, s_ref, _ = rms_reference(x, w, None, eps=1e-6)
    got = flat.view(groups, scale_rows)[:, :num_tokens].t().contiguous()
    assert torch.allclose(got, s_ref, rtol=1e-5, atol=0), (got, s_ref)


@requires_gpu
@pytest.mark.parametrize("col_major", [False, True])
@pytest.mark.parametrize("num_groups", [32, 112, 4096])
def test_group_entry_matches_the_stock_oracle(col_major, num_groups):
    from cuda.bindings import driver  # noqa: F401

    torch.manual_seed(3)
    dev = "cuda"
    x = torch.randn(num_groups * GROUP, dtype=torch.bfloat16, device=dev)
    out_q = torch.empty(num_groups * GROUP, dtype=torch.float8_e4m3fn, device=dev)
    out_q.fill_(torch.finfo(torch.float8_e4m3fn).max)

    # Stock reads the layout off the strides: col-major == stride(0) < stride(1). Both
    # layouts address the SAME logical [rows, cols] scale matrix, indexed by group id.
    rows, cols = num_groups // 16, 16
    out_s = torch.full((rows, cols), float("nan"), dtype=torch.float32, device=dev)
    if col_major:
        out_s = out_s.t().contiguous().t()   # [rows, cols] with strides (1, rows)
    assert (out_s.stride(0) < out_s.stride(1)) is col_major, "rows must be > 1 here"

    eps = 1e-10
    launch = _compile("group_quant_fp8")
    groups_per_block = GQ_THREADS // patch.LANES_PER_GROUP
    launch(
        grid=((num_groups + groups_per_block - 1) // groups_per_block, 1, 1),
        block=(GQ_THREADS, 1, 1),
        args=nvrtc.pack_args(
            out_q.data_ptr(), out_s.data_ptr(), x.data_ptr(),
            ("f", eps), ("f", FP8_MIN), ("f", FP8_MAX),
            ("l", num_groups),
            ("i", 1 if col_major else 0),
            ("i", int(out_s.size(1))),
            ("i", int(out_s.stride(1))),
        ),
    )
    torch.cuda.synchronize()

    q_ref, s_ref = group_reference(x, eps=eps)
    assert torch.equal(out_q.view(torch.uint8), q_ref.reshape(-1).view(torch.uint8))
    assert torch.allclose(out_s, s_ref.view(rows, cols), rtol=1e-6, atol=0)


@requires_gpu
def test_group_entry_eps_seeds_the_absmax_for_an_all_zero_group():
    """An all-zero group must get scale eps/448, not 0 -- and must not produce NaNs."""
    from cuda.bindings import driver  # noqa: F401

    dev, num_groups, eps = "cuda", 16, 1e-10
    x = torch.zeros(num_groups * GROUP, dtype=torch.bfloat16, device=dev)
    out_q = torch.full((num_groups * GROUP,), 1.0, dtype=torch.float32,
                       device=dev).to(torch.float8_e4m3fn)
    out_s = torch.full((num_groups, 1), float("nan"), dtype=torch.float32, device=dev)

    launch = _compile("group_quant_fp8")
    groups_per_block = GQ_THREADS // patch.LANES_PER_GROUP
    launch(
        grid=((num_groups + groups_per_block - 1) // groups_per_block, 1, 1),
        block=(GQ_THREADS, 1, 1),
        args=nvrtc.pack_args(
            out_q.data_ptr(), out_s.data_ptr(), x.data_ptr(),
            ("f", eps), ("f", FP8_MIN), ("f", FP8_MAX),
            ("l", num_groups), ("i", 0), ("i", 1), ("i", 1),
        ),
    )
    torch.cuda.synchronize()

    assert torch.allclose(out_s, torch.full_like(out_s, eps / FP8_MAX), rtol=1e-6)
    assert (out_q.float() == 0).all(), out_q.float()


@requires_gpu
@pytest.mark.parametrize("has_residual", [0, 1])
def test_eager_rms_fallback_agrees_with_the_kernel(has_residual):
    """The eager path is what a call outside the envelope gets, and it is all that is left
    once the override evicts the stock C++ kernel -- so "slow" is allowed and "different"
    is not. Held to the KERNEL rather than to the oracle, so the two cannot drift apart."""
    from cuda.bindings import driver  # noqa: F401

    torch.manual_seed(4)
    dev, num_tokens = "cuda", 5
    groups = HIDDEN // GROUP
    x = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device=dev)
    w = torch.randn(HIDDEN, dtype=torch.bfloat16, device=dev)
    res = (torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device=dev)
           if has_residual else None)

    def _fresh():
        return (torch.empty(num_tokens, HIDDEN, dtype=torch.float8_e4m3fn, device=dev),
                torch.full((num_tokens, groups), float("nan"), dtype=torch.float32,
                           device=dev),
                res.clone() if res is not None else None)

    k_out, k_scale, k_res = _fresh()
    _launch_rms(_compile(patch.KERNEL, has_residual), k_out, k_scale, x, w, k_res, 1e-6)

    e_out, e_scale, e_res = _fresh()
    patch._eager_rms(e_out, x, w, e_scale, 1e-6, None, e_res, GROUP, False)

    assert torch.equal(e_out.view(torch.uint8), k_out.view(torch.uint8))
    assert torch.allclose(e_scale, k_scale, rtol=1e-6, atol=0)
    if res is not None:
        assert torch.equal(e_res.view(torch.uint8), k_res.view(torch.uint8))


@requires_gpu
@pytest.mark.parametrize("col_major", [False, True])
def test_eager_group_fallback_agrees_with_the_kernel(col_major):
    from cuda.bindings import driver  # noqa: F401

    torch.manual_seed(5)
    dev, num_groups, eps = "cuda", 112, 1e-10
    rows, cols = num_groups // 16, 16
    x = torch.randn(num_groups * GROUP, dtype=torch.bfloat16, device=dev)

    def _fresh():
        s = torch.full((rows, cols), float("nan"), dtype=torch.float32, device=dev)
        if col_major:
            s = s.t().contiguous().t()
        return torch.empty(num_groups * GROUP, dtype=torch.float8_e4m3fn, device=dev), s

    k_q, k_s = _fresh()
    groups_per_block = GQ_THREADS // patch.LANES_PER_GROUP
    _compile("group_quant_fp8")(
        grid=((num_groups + groups_per_block - 1) // groups_per_block, 1, 1),
        block=(GQ_THREADS, 1, 1),
        args=nvrtc.pack_args(
            k_q.data_ptr(), k_s.data_ptr(), x.data_ptr(),
            ("f", eps), ("f", FP8_MIN), ("f", FP8_MAX), ("l", num_groups),
            ("i", 1 if col_major else 0), ("i", int(k_s.size(1))), ("i", int(k_s.stride(1))),
        ),
    )
    torch.cuda.synchronize()

    e_q, e_s = _fresh()
    patch._eager_gq(x, e_q, e_s, GROUP, eps, FP8_MIN, FP8_MAX, False)

    assert torch.equal(e_q.view(torch.uint8), k_q.view(torch.uint8))
    assert torch.allclose(e_s, k_s, rtol=1e-6, atol=0)


# --------------------------------------------------------------------------------------
# no-GPU half
# --------------------------------------------------------------------------------------

def _self_check() -> None:
    import os

    src = _source()

    # -- names line up: source file <-> compile_kernel(name=...) <-> the entry symbols --
    assert patch.KERNEL == "rms_norm_block_quant"
    assert 'extern "C" __global__ void __launch_bounds__(THREADS) rms_norm_block_quant(' in src
    assert 'extern "C" __global__ void __launch_bounds__(GQ_THREADS) group_quant_fp8(' in src
    for macro in ("HIDDEN", "GROUP", "THREADS", "HAS_RESIDUAL", "GQ_THREADS"):
        assert f"#ifndef {macro}" in src, f"-D{macro} is not guarded in the source"
    # The two shape defines have no defensible default; the source must refuse them.
    assert '#error "NVRTC: -DHIDDEN=<hidden_size> is required"' in src
    assert '#error "NVRTC: -DGROUP=<group_size> is required"' in src

    # -- the patch's defines are exactly what the source expects, for the shipped model --
    d = patch._defines(HIDDEN, GROUP, 0)
    assert d == {"HIDDEN": 3072, "GROUP": 128, "THREADS": 384,
                 "HAS_RESIDUAL": 0, "GQ_THREADS": 256}, d
    assert patch._defines(HIDDEN, GROUP, 1)["HAS_RESIDUAL"] == 1

    # -- one cubin identity per specialization. A collision here means a warm cache serves
    #    the residual kernel to a no-residual call, which is silently wrong output. --
    sets = [
        patch._defines(3072, 128, 0),
        patch._defines(3072, 128, 1),
        patch._defines(4096, 128, 0),
        patch._defines(4096, 128, 1),
    ]
    keys = {nvrtc.cache_key(src, s, "90a", "12.8") for s in sets}
    assert len(keys) == len(sets), "define sets must not collide in the cubin cache"
    # ...arch and toolkit are part of the identity too, and dict order is not.
    assert nvrtc.cache_key(src, sets[0], "90", "12.8") not in keys
    assert nvrtc.cache_key(src, sets[0], "90a", "12.9") not in keys
    perm = dict(reversed(list(sets[0].items())))
    assert nvrtc.cache_key(src, perm, "90a", "12.8") == nvrtc.cache_key(src, sets[0],
                                                                        "90a", "12.8")

    # -- geometry envelope: the model this ships for passes; foreign shapes do not --
    assert patch._geometry_ok(3072, 128) is True
    assert patch._geometry_ok(3072, 64) is False
    assert patch._geometry_ok(3000, 128) is False
    assert patch._geometry_ok(16384, 128) is False
    assert patch._threads_for(3072) == THREADS

    # -- import-without-vLLM: the module loads and registers ON by default --
    # Enabled by default per project decision (2026-08-17); the compose gate now restates the
    # code default instead of overriding it. Default-on is safe because the fallback ladder,
    # not the gate, is the safety story: NVRTC -> AOT -> stock op, behind a geometry envelope.
    from vtl.registry import PATCH_REGISTRY, is_enabled

    p = next(x for x in PATCH_REGISTRY if x.name == "nvrtc_block_quant")
    assert p.default is True and is_enabled(p) is True

    # With VTL_NVRTC off, apply() must return cleanly even though vLLM is absent --
    # nothing is imported, nothing is installed, the stock ops keep the dispatch key.
    os.environ.pop("VTL_NVRTC", None)
    assert nvrtc.compile_kernel(patch.KERNEL, sets[0]) is None
    patch.apply()
    assert patch._state["installed"] is False
    assert patch._state["launchers"] is None

    # With VTL_NVRTC on but vLLM absent, apply() may raise -- registry.apply_all isolates
    # it -- but it must not leave the module half-installed.
    os.environ["VTL_NVRTC"] = "1"
    try:
        patch.apply()
    except Exception:
        pass
    finally:
        os.environ.pop("VTL_NVRTC", None)
    assert patch._state["installed"] is False
    assert patch._state["launchers"] is None

    if torch is None:
        print("test_nvrtc_block_quant self-check ok (torch absent; oracles not exercised)")
        return

    # -- the oracles must be self-consistent on CPU with no GPU present --
    x = torch.randn(4, 256, dtype=torch.bfloat16)
    w = torch.randn(256, dtype=torch.bfloat16)
    q, s, _ = rms_reference(x, w)
    assert q.shape == (4, 256) and s.shape == (4, 2)
    assert (s >= MIN_SCALE).all(), "the min_scaling_factor floor must hold per GROUP"
    _, s0, _ = rms_reference(torch.zeros(1, 256, dtype=torch.bfloat16), w)
    assert abs(float(s0[0, 0]) - MIN_SCALE) < 1e-12, s0
    # The residual is written back as bf16(x + residual), and it is what pass 2 reads.
    r = torch.randn(4, 256, dtype=torch.bfloat16)
    _, _, r_out = rms_reference(x, w, r)
    assert torch.equal(r_out, (x.float() + r.float()).to(torch.bfloat16))

    # The group entry has NO floor -- only the eps seed -- which is the whole reason it
    # needs its own oracle rather than reusing the rms one.
    _, ys = group_reference(torch.zeros(2 * GROUP, dtype=torch.bfloat16), eps=1e-10)
    assert torch.allclose(ys, torch.full_like(ys, 1e-10 / FP8_MAX))
    assert float(ys[0]) < MIN_SCALE, "if this ever floors, the oracle drifted to the rms path"

    print("test_nvrtc_block_quant self-check ok")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    elif not HAVE_PYTEST:
        raise SystemExit("pytest not installed; run with --self-check")
    else:
        raise SystemExit(pytest.main([__file__, "-q"]))
