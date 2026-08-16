"""Correctness + micro-benchmark for vtl's GDN per-head gated RMSNorm kernels: the
standalone norm, the round-1.1-compat per-TOKEN fp8 quant fusion, and the round-2
per-(token,128-GROUP) fp8 quant fusion (the out_proj input of Qwen3.5-122B-A10B's 36 GDN
layers: HV=64 heads x DV=128, group 128 == head dim).

    python3 bench/test_gdn_gated_rmsnorm.py --self-check   # no GPU/torch (runs in `make check`)
    pytest -q bench/test_gdn_gated_rmsnorm.py              # GPU parity (make test-kernel)
    python3 bench/test_gdn_gated_rmsnorm.py                # micro-benchmark

These are vtl-only ops (no stock _C counterpart), so there is no VTL_SKIP_EXT dual mode --
the GPU half is skipped entirely under stock. The oracle encodes vLLM v0.25.0's ACTUAL
composition for this model: RMSNormGated.forward_static (norm_before_gate=True,
group_size=None, fp32 math, weight upcast, ONE narrowing to the model dtype) followed by
_C::per_token_group_fp8_quant numerics (fp32 re-read of the narrowed tensor, amax floored
at 1e-10, scale = amax/448 with NO min-scale clamp, TRUE division, clamp then fp8-rne).
The NVRTC variant must BIT-match the AOT one -- same arithmetic, op for op.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# `make check` runs --self-check on a bare host with neither pytest nor torch.
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

USE_STOCK = os.environ.get("VTL_SKIP_EXT", "").strip() in ("1", "true", "yes", "on")

requires_gpu = pytest.mark.skipif(
    torch is None or not torch.cuda.is_available() or USE_STOCK,
    reason="needs torch + CUDA; gated_rmsnorm is a vtl-only op (no stock kernel)",
)

EPS = 1e-6
FP8_MAX = 448.0
MIN_SCALE = 1.0 / (FP8_MAX * 512.0)  # per-TOKEN entry only
GROUP_EPS = 1e-10                    # per-GROUP entry: amax floor, no min-scale clamp
GROUP = 128

# The served model's GDN geometry.
MODEL_H, MODEL_D = 64, 128

_ops = {}


def _op(name):
    if name not in _ops:
        import vtl._C  # noqa: F401  -- registers vllm_cuda::gated_rmsnorm(+ quant variants)

        _ops[name] = getattr(torch.ops.vllm_cuda, name)
    return _ops[name]


# ---------------------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------------------

def _act(zf, is_silu):
    return zf * torch.sigmoid(zf) if is_silu else torch.sigmoid(zf)


def reference_norm(x, gate, weight, is_silu=True, eps=EPS):
    """RMSNormGated.forward_static, norm_before_gate=True, group_size=None: all fp32,
    weight UPCAST (never narrowed -- unlike rms_norm_quant's double rounding), one
    narrowing to x.dtype at the end."""
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    out = ((xf * torch.rsqrt(var + eps)) * weight.float()) * _act(gate.float(), is_silu)
    return out.to(x.dtype)


def reference_group_quant(x, gate, weight, H, is_silu=True, eps=EPS):
    """The fused op's full contract: per-head norm -> narrow to x.dtype (the RMSNormGated
    -> quant graph boundary; this second rounding is load-bearing) -> reshape [N, H*D] ->
    per-(token,128-group) fp8 quant with stock numerics."""
    N, D = x.shape[0] // H, weight.numel()
    yq = reference_norm(x, gate, weight, is_silu, eps).float()  # re-widened narrowed values
    grouped = yq.reshape(N, (H * D) // GROUP, GROUP)
    amax = grouped.abs().amax(-1).clamp(min=GROUP_EPS)
    scale = amax / FP8_MAX  # NO min-scale clamp: stock group quant floors amax instead
    q = torch.clamp(grouped / scale.unsqueeze(-1), -FP8_MAX, FP8_MAX)
    return q.reshape(N, H * D).to(torch.float8_e4m3fn), scale


def reference_token_quant(x, gate, weight, H, is_silu=True, eps=EPS):
    """Round-1.1 compat entry: per-head norm (fp32, NOT re-narrowed -- the kernel keeps the
    normed row in registers), per-TOKEN amax, min-scale clamp."""
    normed_f32 = x.float()
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    normed_f32 = ((xf * torch.rsqrt(var + eps)) * weight.float()) * _act(gate.float(), is_silu)
    N, D = x.shape[0] // H, weight.numel()
    merged = normed_f32.reshape(N, H * D)
    amax = merged.abs().amax(-1, keepdim=True)
    scale = torch.clamp(amax / FP8_MAX, min=MIN_SCALE)
    q = torch.clamp(merged / scale, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


# ---------------------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------------------

def run_norm(x, gate, weight, is_silu=True, eps=EPS):
    out = torch.empty_like(x)
    _op("gated_rmsnorm")(out, x, gate, weight, float(eps), is_silu)
    return out


def run_group_quant(x, gate, weight, H, is_silu=True, eps=EPS, col_major=False):
    N, D = x.shape[0] // H, weight.numel()
    groups = (H * D) // GROUP
    result = torch.empty((N, H * D), device="cuda", dtype=torch.float8_e4m3fn)
    if col_major:
        scale = torch.empty((groups, N), device="cuda", dtype=torch.float32).permute(-1, -2)
    else:
        scale = torch.empty((N, groups), device="cuda", dtype=torch.float32)
    _op("gated_rmsnorm_per_group_quant")(
        result, scale, x, gate, weight, float(eps), H, GROUP, is_silu
    )
    return result, scale


def run_token_quant(x, gate, weight, H, is_silu=True, eps=EPS):
    N, D = x.shape[0] // H, weight.numel()
    result = torch.empty((N, H * D), device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.empty((N, 1), device="cuda", dtype=torch.float32)
    _op("gated_rmsnorm_dynamic_per_token_quant")(
        result, scale, x, gate, weight, float(eps), H, is_silu, None
    )
    return result, scale


def _mismatch_fraction(a, b):
    ca = a.view(torch.uint8).to(torch.int32)
    cb = b.view(torch.uint8).to(torch.int32)
    return (ca != cb).float().mean().item()


def _assert_fp8_equivalent(got, exp):
    """Magnitude bound: every code within one e4m3 step of the reference (stride/index
    bugs surface as many-step errors). Fraction bound: reduction-order noise flips only
    near-boundary codes; a wrong rounding rule flips ~half."""
    torch.testing.assert_close(got.float(), exp.float(), rtol=0.13, atol=2.0**-9)
    frac = _mismatch_fraction(got, exp)
    assert frac < 1e-2, f"{frac:.2%} of fp8 codes differ -- systematic, not reduction noise"


def _weight(D, dtype, w_dtype):
    """v0.25.0 stores the RMSNormGated weight in the MODEL dtype; fp32 kept for compat."""
    return torch.randn(D, dtype=w_dtype if w_dtype is not None else dtype, device="cuda")


# ---------------------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------------------

@requires_gpu
@pytest.mark.parametrize("is_silu", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("num_rows", [1, 16, 4096])
@pytest.mark.parametrize("D", [128, 64, 100])  # 128 = model head dim; 100 = odd width
def test_norm_matches_reference(is_silu, dtype, num_rows, D):
    torch.manual_seed(0)
    x = torch.randn(num_rows, D, dtype=dtype, device="cuda")
    gate = torch.randn(num_rows, D, dtype=dtype, device="cuda")
    weight = _weight(D, dtype, torch.float32)
    got = run_norm(x, gate, weight, is_silu)
    exp = reference_norm(x, gate, weight, is_silu)
    rtol, atol = {torch.bfloat16: (2e-2, 2e-3), torch.float16: (2e-3, 2e-4),
                  torch.float32: (1e-5, 1e-6)}[dtype]
    torch.testing.assert_close(got.float(), exp.float(), rtol=rtol, atol=atol)


@requires_gpu
@pytest.mark.parametrize("w_dtype", [torch.float32, None])  # None = model (input) dtype
def test_norm_accepts_model_dtype_weight(w_dtype):
    """v0.25.0 creates the weight in the model dtype (bf16 here), not fp32."""
    torch.manual_seed(0)
    x = torch.randn(256, MODEL_D, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(256, MODEL_D, dtype=torch.bfloat16, device="cuda")
    weight = _weight(MODEL_D, torch.bfloat16, w_dtype)
    got = run_norm(x, gate, weight)
    exp = reference_norm(x, gate, weight)
    torch.testing.assert_close(got.float(), exp.float(), rtol=2e-2, atol=2e-3)


# "both entry points" x the required token counts.
@requires_gpu
@pytest.mark.parametrize("entry", ["group", "token"])
@pytest.mark.parametrize("num_tokens", [1, 7, 64])
@pytest.mark.parametrize("is_silu", [True, False])
@pytest.mark.parametrize("H,D", [(MODEL_H, MODEL_D), (16, 128)])
def test_fused_quant_matches_reference(entry, num_tokens, is_silu, H, D):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    x = torch.randn(num_tokens * H, D, dtype=dtype, device="cuda")
    gate = torch.randn(num_tokens * H, D, dtype=dtype, device="cuda")
    weight = _weight(D, dtype, None)  # model-dtype weight, as served

    if entry == "group":
        got_q, got_s = run_group_quant(x, gate, weight, H, is_silu)
        exp_q, exp_s = reference_group_quant(x, gate, weight, H, is_silu)
    else:
        got_q, got_s = run_token_quant(x, gate, weight, H, is_silu)
        exp_q, exp_s = reference_token_quant(x, gate, weight, H, is_silu)

    # bf16 reduction-order drift moves the amax element by <=1 bf16 ULP, which moves the
    # scale by the same relative amount; the dequant/code checks are the quality gate.
    torch.testing.assert_close(got_s, exp_s, rtol=1e-2, atol=1e-8)
    _assert_fp8_equivalent(got_q, exp_q)


@requires_gpu
@pytest.mark.parametrize("num_tokens", [1, 7, 64])
def test_group_quant_dequant_roundtrip(num_tokens):
    """Dequantized product matches the pre-quant values within e4m3 rounding -- the
    end-to-end quality bound the fusion must preserve."""
    torch.manual_seed(1)
    H, D = MODEL_H, MODEL_D
    x = torch.randn(num_tokens * H, D, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(num_tokens * H, D, dtype=torch.bfloat16, device="cuda")
    weight = _weight(D, torch.bfloat16, None)
    got_q, got_s = run_group_quant(x, gate, weight, H)
    yq = reference_norm(x, gate, weight).float().reshape(num_tokens, H * D)
    deq = got_q.float().reshape(num_tokens, (H * D) // GROUP, GROUP) * got_s.unsqueeze(-1)
    torch.testing.assert_close(
        deq.reshape(num_tokens, H * D), yq, rtol=8e-2,
        atol=2 * float(got_s.max()) if got_s.numel() else 1e-6,
    )


@requires_gpu
def test_group_quant_scale_layouts_agree():
    """The kernel writes scales through their strides: the column-major layout vLLM's
    cutlass block-fp8 path allocates must carry the same values as row-major."""
    torch.manual_seed(2)
    H, D = MODEL_H, MODEL_D
    x = torch.randn(7 * H, D, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(7 * H, D, dtype=torch.bfloat16, device="cuda")
    weight = _weight(D, torch.bfloat16, None)
    q_row, s_row = run_group_quant(x, gate, weight, H, col_major=False)
    q_col, s_col = run_group_quant(x, gate, weight, H, col_major=True)
    assert s_col.stride() == (1, 7)
    assert torch.equal(q_row.view(torch.uint8), q_col.view(torch.uint8))
    assert torch.equal(s_row, s_col.contiguous())


@requires_gpu
def test_group_quant_zero_input_hits_eps_floor():
    """All-zero groups: stock floors the amax at 1e-10, so scale == 1e-10/448 -- NOT the
    per-token MIN_SCALE. Pins the 'no min-scale clamp' decision."""
    H, D = MODEL_H, MODEL_D
    x = torch.zeros(2 * H, D, dtype=torch.bfloat16, device="cuda")
    gate = torch.zeros(2 * H, D, dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(D, dtype=torch.bfloat16, device="cuda")
    _, s = run_group_quant(x, gate, weight, H)
    torch.testing.assert_close(
        s, torch.full_like(s, GROUP_EPS / FP8_MAX), rtol=1e-6, atol=0.0
    )


@requires_gpu
@pytest.mark.parametrize("num_tokens", [1, 7, 64])
@pytest.mark.parametrize("w_fp32", [False, True])
def test_nvrtc_variant_bitmatches_aot(num_tokens, w_fp32):
    """The NVRTC twin must be the SAME kernel, bit for bit: identical fp8 codes and
    identical scales. Anything less means the two files' arithmetic diverged."""
    from vtl import nvrtc
    from vtl.patches.gdn_kernels import NVRTC_THREADS, _nvrtc_defines, _nvrtc_grid

    src = nvrtc.load_source("gated_rmsnorm_group_quant")
    assert src, "vtl/kernels/gated_rmsnorm_group_quant.cu must ship with the package"
    built = nvrtc.build_cubin(
        src, "gated_rmsnorm_group_quant", _nvrtc_defines(MODEL_D, MODEL_H, True, w_fp32)
    )
    if built is None:
        pytest.skip("NVRTC unavailable (no cuda-python / no driver) -- AOT path stands")
    launch = nvrtc._load_cubin(built[0], "gated_rmsnorm_group_quant")

    torch.manual_seed(3)
    H, D = MODEL_H, MODEL_D
    x = torch.randn(num_tokens * H, D, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(num_tokens * H, D, dtype=torch.bfloat16, device="cuda")
    weight = _weight(D, torch.bfloat16, torch.float32 if w_fp32 else None)

    aot_q, aot_s = run_group_quant(x, gate, weight, H)

    got_q = torch.empty_like(aot_q)
    got_s = torch.empty_like(aot_s)
    got_q.view(torch.uint8).fill_(0x7E)  # a kernel that never writes must not "match"
    got_s.fill_(float("nan"))
    launch(
        grid=_nvrtc_grid(num_tokens * H, D),
        block=(NVRTC_THREADS, 1, 1),
        args=nvrtc.pack_args(
            got_q.data_ptr(), got_s.data_ptr(), x.data_ptr(), gate.data_ptr(),
            weight.data_ptr(), ("l", got_s.stride(0)), ("l", got_s.stride(1)),
            ("l", num_tokens * H), ("f", EPS),
        ),
    )
    torch.cuda.synchronize()

    assert torch.equal(got_s, aot_s), "NVRTC and AOT scales must be bit-identical"
    assert torch.equal(got_q.view(torch.uint8), aot_q.view(torch.uint8)), (
        "NVRTC and AOT fp8 codes must be bit-identical"
    )


@requires_gpu
def test_python_ladder_op_matches_aot():
    """vllm_cuda::gdn_gated_rmsnorm_group_quant (the op the fusion pattern inserts) must
    resolve a tier and reproduce the AOT result (bit-identical when it lands on AOT or on
    the bit-matching NVRTC kernel; VTL_NVRTC unset here, so AOT)."""
    from vtl.patches import gdn_kernels

    gdn_kernels._register_op()
    torch.manual_seed(4)
    H, D = MODEL_H, MODEL_D
    x = torch.randn(7 * H, D, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(7 * H, D, dtype=torch.bfloat16, device="cuda")
    weight = _weight(D, torch.bfloat16, None)

    aot_q, aot_s = run_group_quant(x, gate, weight, H)
    got_q = torch.empty_like(aot_q)
    got_s = torch.empty_like(aot_s)
    torch.ops.vllm_cuda.gdn_gated_rmsnorm_group_quant(
        got_q, got_s, x, gate, weight, EPS, H, True
    )
    assert torch.equal(got_s, aot_s)
    assert torch.equal(got_q.view(torch.uint8), aot_q.view(torch.uint8))


@requires_gpu
def test_matches_vllm_rmsnormgated():
    """The real gate: our standalone kernel vs vLLM's own RMSNormGated. Skips if the class
    is not importable / signature drifted -- in which case gdn_kernels stays disabled."""
    try:
        from vllm.model_executor.layers.layernorm import RMSNormGated
    except Exception:
        pytest.skip("vLLM RMSNormGated not importable in this environment")

    D = MODEL_D
    torch.manual_seed(0)
    x = torch.randn(256, D, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(256, D, dtype=torch.bfloat16, device="cuda")
    try:
        norm = RMSNormGated(
            hidden_size=D, eps=EPS, group_size=None, norm_before_gate=True, activation="silu"
        ).cuda()
        norm.weight.data = torch.randn_like(norm.weight.data)
        with torch.no_grad():
            ref = norm.forward_native(x, gate) if hasattr(norm, "forward_native") else norm(x, gate)
    except Exception as exc:
        pytest.skip(f"RMSNormGated signature differs on this vLLM: {exc}")

    got = run_norm(x, gate, norm.weight.data.contiguous(), is_silu=True)
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-3)


# ---------------------------------------------------------------------------------------
# Self-check (no GPU, no torch, no vLLM) -- the `make check` half.
# ---------------------------------------------------------------------------------------

def _self_check() -> None:
    from vtl import nvrtc
    from vtl.patches import gdn_kernels, gdn_prefill_backend

    # Patch modules import (already done above) and self-check without vllm/torch.
    gdn_kernels._self_check()
    gdn_prefill_backend._self_check()

    # The NVRTC source ships with the package and each specialization is a distinct cubin.
    src = nvrtc.load_source("gated_rmsnorm_group_quant")
    assert src, "vtl/kernels/gated_rmsnorm_group_quant.cu missing from the package"
    for guard in ("DV", "HV", "GROUP", "IS_SILU"):
        assert f"#error \"NVRTC: -D{guard}" in src, f"missing #error guard for -D{guard}"
    base = gdn_kernels._nvrtc_defines(128, 64, True, False)
    k0 = nvrtc.cache_key(src, base, "90a", "13.0")
    assert k0 == nvrtc.cache_key(src, dict(reversed(list(base.items()))), "90a", "13.0"), (
        "cache key must not depend on define order"
    )
    for variant in (
        gdn_kernels._nvrtc_defines(128, 64, False, False),   # sigmoid gate
        gdn_kernels._nvrtc_defines(128, 64, True, True),     # fp32 weight
        gdn_kernels._nvrtc_defines(128, 16, True, False),    # different head count
    ):
        assert nvrtc.cache_key(src, variant, "90a", "13.0") != k0, variant
    assert nvrtc.cache_key(src, base, "90", "13.0") != k0, "arch is part of the key"

    # Shape math: model geometry 64x128, group 128 -> scale [N, 64], 16 rows per block.
    assert gdn_kernels._scale_shape(1, 64 * 128) == (1, 64)
    assert gdn_kernels._scale_shape(7, 64 * 128) == (7, 64)
    assert gdn_kernels._nvrtc_grid(64, 128) == (4, 1, 1)
    assert gdn_kernels._nvrtc_supported(128, 64)
    assert not gdn_kernels._nvrtc_supported(96, 64), "group != head dim must be refused"

    if torch is not None:
        # The two oracles must disagree exactly where their contracts differ: an all-zero
        # token gets MIN_SCALE per-token but eps/448 per-group.
        x = torch.zeros(2 * 4, 128, dtype=torch.bfloat16)
        g = torch.zeros(2 * 4, 128, dtype=torch.bfloat16)
        w = torch.ones(128, dtype=torch.bfloat16)
        _, s_grp = reference_group_quant(x, g, w, 4)
        _, s_tok = reference_token_quant(x, g, w, 4)
        assert abs(float(s_grp[0, 0]) - GROUP_EPS / FP8_MAX) < 1e-18, s_grp[0, 0]
        assert abs(float(s_tok[0, 0]) - MIN_SCALE) < 1e-12, s_tok[0, 0]
        # And the group oracle's shape contract matches the scale-shape helper.
        q, s = reference_group_quant(
            torch.randn(7 * 4, 128, dtype=torch.bfloat16),
            torch.randn(7 * 4, 128, dtype=torch.bfloat16),
            torch.randn(128, dtype=torch.bfloat16), 4)
        assert q.shape == (7, 512) and tuple(s.shape) == gdn_kernels._scale_shape(7, 512)
        print("test_gdn_gated_rmsnorm self-check ok")
    else:
        print("test_gdn_gated_rmsnorm self-check ok (torch absent; oracle check skipped)")


# ---------------------------------------------------------------------------------------
# Micro-benchmark
# ---------------------------------------------------------------------------------------

def _bench():
    import time

    print(f"device: {torch.cuda.get_device_name()}")
    H, D = MODEL_H, MODEL_D
    print(f"\n{'op':>18} {'tokens x H x D':>18} {'us/call':>10} {'GB/s':>8}")
    for tokens in (32, 256, 8192):
        x = torch.randn(tokens * H, D, dtype=torch.bfloat16, device="cuda")
        gate = torch.randn(tokens * H, D, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(D, dtype=torch.bfloat16, device="cuda")
        for name, fn in (
            ("norm", lambda: run_norm(x, gate, weight)),
            ("norm+group-quant", lambda: run_group_quant(x, gate, weight, H)),
            ("norm+token-quant", lambda: run_token_quant(x, gate, weight, H)),
        ):
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            iters = 200
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            us = (time.perf_counter() - t0) / iters * 1e6
            moved = tokens * H * D * (2 + 2 + 1)  # read x+z (bf16), write fp8
            print(f"{name:>18} {f'{tokens}x{H}x{D}':>18} {us:>10.1f} {moved / us / 1e3:>8.0f}")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    elif torch is None or not torch.cuda.is_available():
        raise SystemExit("no CUDA device; run with --self-check")
    elif USE_STOCK:
        raise SystemExit("gated_rmsnorm is a vtl-only op; nothing to benchmark under stock")
    else:
        _bench()
