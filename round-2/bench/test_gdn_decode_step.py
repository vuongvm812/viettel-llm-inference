"""Fused GDN decode-step harness: engage logic off-box, parity vs the real Triton pair.

    python3 bench/test_gdn_decode_step.py --self-check   # no GPU, no torch, no vLLM
    pytest -q bench/test_gdn_decode_step.py              # GPU: compiles and checks the math

THE ORACLE IS THE STOCK KERNELS THEMSELVES, not a torch transcription of them. That choice
is the point of this file. ``vtl/kernels/gdn_decode_step.cu`` fuses
``causal_conv1d_update`` and ``fused_recurrent_gated_delta_rule_packed_decode``, and a
hand-written reference would only prove our reading of the Triton source is self-
consistent -- it would reproduce any misreading on both sides. Running the actual pair on
cloned inputs and diffing all three mutated tensors cannot.

WHAT IS COMPARED, AND WHY THE TOLERANCES ARE WHAT THEY ARE
  conv_state (bf16)  BIT-EXACT (``torch.equal``). The rotation [s0,s1,s2] -> [s1,s2,x] is
                     pure data movement; x is stored RAW, before the silu. Any arithmetic
                     drift here is a bug, not a rounding difference, so nothing is allowed.
  ssm_state (fp32)   rtol 1e-4 / atol 1e-5. Two divergences are structural and documented
                     in the .cu header: (1) fp32 REDUCTION ORDER -- Triton tree-reduces the
                     l2norm sums and the two length-128 dots over 128 lanes, we do a
                     per-lane 4-chain plus a 32-lane shuffle tree; (2) TRANSCENDENTAL
                     LOWERING -- expf/logf/sqrtf here vs whatever tl.exp/tl.log/tl.sqrt
                     lower to in the installed Triton. Both are ~1 ulp fp32 (1.2e-7
                     relative) and the recurrence contracts rather than amplifies them
                     (|exp(g)| <= 1). 1e-4 is ~3 decades of headroom over that, and still
                     three decades TIGHTER than any structural error would produce -- a
                     swapped q/k index, a dropped beta or a missing bf16 narrowing moves
                     results by 1e-2 and up, which this bound rejects.
  out (bf16)         rtol 1e-2 / atol 1e-2. bf16 carries 8 mantissa bits, so one ulp is
                     ~0.4% relative: 1e-2 is ~2 ulp, the smallest bound that is not
                     measuring the output dtype's own rounding.

  mixed_qkv is DELIBERATELY NOT COMPARED. The stock pair writes the conv output back over
  it in place; the fused kernel keeps that intermediate in shared memory and never writes
  it, because nothing on this path reads it afterwards. The oracle therefore runs against
  a clone -- if it did not, the second stock kernel would read our (untouched) input.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vtl import nvrtc  # noqa: E402
from vtl.patches import gdn_decode_step as patch  # noqa: E402

# `make check` runs --self-check on a bare host with neither pytest nor torch; see the
# note in bench/test_nvrtc.py.
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

        @staticmethod
        def skip(*a, **k):
            raise SystemExit("pytest not installed")

    pytest = _NoPytest()

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

requires_gpu = pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(), reason="needs torch + a CUDA device"
)

DK = 128
DV = 128
CONV = 4
NULL_BLOCK_ID = 0

# Qwen3.5-122B-A10B at TP=1, and a small stand-in that exercises the same code with a
# different HV/HK sibling count. Per-line SSM state is HV*DV*DK*4 B (4 MB at HV=64).
PROD_GEOM = (DK, DV, 16, 64, CONV)
SMALL_GEOM = (DK, DV, 2, 8, CONV)

STATE_RTOL, STATE_ATOL = 1e-4, 1e-5   # fp32 ssm state; see the module docstring
OUT_RTOL, OUT_ATOL = 1e-2, 1e-2       # bf16 output, ~2 ulp


# --------------------------------------------------------------------------------------
# fakes: just enough of the layer + metadata for the patch's own _plan() to run
# --------------------------------------------------------------------------------------

class _FakeConv1d:
    def __init__(self, weight, bias):
        self.weight = weight   # [conv_dim, 1, CONV] -- vLLM's unsqueezed conv1d weight
        self.bias = bias


class _FakeLayer:
    """Only the attributes _plan() actually reads. Wrong on purpose: nothing else exists,
    so a _plan() that starts depending on more of the layer fails here rather than in
    production."""

    def __init__(self, geom, kv_cache, conv1d, A_log, dt_bias, activation="silu"):
        dk, dv, hk, hv, conv = geom
        self.head_k_dim, self.head_v_dim = dk, dv
        self.num_k_heads, self.num_v_heads = hk, hv
        self.tp_size = 1
        self.conv_kernel_size = conv
        self.activation = activation
        self.kv_cache = kv_cache
        self.conv1d = conv1d
        self.A_log = A_log
        self.dt_bias = dt_bias


class _FakeMeta:
    def __init__(self, tokens, idx, num_decodes=None):
        self.spec_sequence_masks = None
        self.num_prefills = 0
        self.num_spec_decodes = 0
        self.num_decodes = tokens if num_decodes is None else num_decodes
        self.num_actual_tokens = tokens
        self.non_spec_state_indices_tensor = idx


def _arm_state(geom):
    """Compile the shipped kernel for `geom` and arm the patch, or skip with the reason."""
    src = nvrtc.load_source(patch.KERNEL)
    assert src, "vtl/kernels/gdn_decode_step.cu must ship with the package"
    built = nvrtc.build_cubin(src, patch.KERNEL, patch._defines(geom))
    if built is None:
        pytest.skip("NVRTC unavailable (no cuda-python / no driver) -- stock pair stands")
    patch._state.update(geom=geom, launcher=nvrtc._load_cubin(built[0], patch.KERNEL),
                        installed=True, armed=True)
    return patch._state["launcher"]


# --------------------------------------------------------------------------------------
# GPU half
# --------------------------------------------------------------------------------------

def _stock_pair():
    """The two kernels we replace, or a skip. Imported lazily: vLLM is a GPU-box thing."""
    try:
        from vllm.model_executor.layers.fla.ops import (
            fused_recurrent_gated_delta_rule_packed_decode,
        )
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"stock Triton pair unimportable ({exc!r})")
    return causal_conv1d_update, fused_recurrent_gated_delta_rule_packed_decode


def _make_inputs(geom, tokens, dev, seed, layout="DS", idx_stride=1, has_bias=False):
    dk, dv, hk, hv, conv = geom
    conv_dim = patch._conv_dim(geom)
    g = torch.Generator(device=dev).manual_seed(seed)

    def rnd(*shape, dtype=torch.bfloat16, scale=1.0):
        return (torch.randn(*shape, generator=g, device=dev, dtype=torch.float32)
                * scale).to(dtype)

    # One cache line per token plus the null block at index 0, which every decode batch
    # has (padded cudagraph rows point at it) and which both kernels must skip.
    lines = tokens + 1
    if layout == "DS":
        conv_state = rnd(lines, conv_dim, conv - 1)
    else:
        # SD: the [.., state_len, dim] checkpoint layout the call site transposes. Same
        # logical tensor, non-unit dim stride -- which is what exercises the .cu's strided
        # conv-state path rather than its happy one.
        conv_state = rnd(lines, conv - 1, conv_dim).transpose(-1, -2)

    ssm_state = (torch.randn(lines, hv, dv, dk, generator=g, device=dev,
                             dtype=torch.float32) * 0.1)
    mixed_qkv = rnd(tokens, conv_dim)
    a = rnd(tokens, hv, scale=0.5)
    b = rnd(tokens, hv)
    # A in [0.1, 1] so exp(g) lands in ~[0.5, 0.93]: a decay that neither freezes the
    # state (exp(g)~1, which would hide a broken gate) nor annihilates it (exp(g)~0,
    # which would make the state comparison vacuous).
    A_log = torch.log(torch.rand(hv, generator=g, device=dev) * 0.9 + 0.1)
    dt_bias = torch.randn(hv, generator=g, device=dev, dtype=torch.float32) * 0.1
    conv_w = rnd(conv_dim, conv)
    conv_bias = rnd(conv_dim) if has_bias else None

    # block_table_tensor[:, 0] is what the metadata builder hands both stock kernels in
    # the eager path -- a column slice, stride = the table's ROW stride, NOT 1.
    table = torch.zeros(tokens, idx_stride, dtype=torch.int32, device=dev)
    table[:, 0] = torch.arange(1, tokens + 1, dtype=torch.int32, device=dev)
    table[0, 0] = NULL_BLOCK_ID   # the padded/null row every decode batch carries
    idx = table[:, 0]
    assert idx.stride(0) == idx_stride

    return dict(conv_state=conv_state, ssm_state=ssm_state, mixed_qkv=mixed_qkv,
                a=a, b=b, A_log=A_log, dt_bias=dt_bias, conv_w=conv_w,
                conv_bias=conv_bias, idx=idx, conv_dim=conv_dim)


def _run_stock(geom, t, dev, tensors):
    """Run the real pair on CLONES and return (conv_state, ssm_state, out)."""
    conv_update, recurrent = _stock_pair()
    dk, dv, hk, hv, conv = geom

    conv_state = tensors["conv_state"].clone()
    ssm_state = tensors["ssm_state"].clone()
    # Cloned because kernel 1 overwrites it with the conv output in place.
    mixed_qkv = tensors["mixed_qkv"].clone()

    conv_out = conv_update(
        mixed_qkv,
        conv_state,
        tensors["conv_w"],
        tensors["conv_bias"],
        "silu",
        conv_state_indices=tensors["idx"],
        validate_data=False,
    )
    out = torch.zeros(t, 1, hv, dv, dtype=torch.bfloat16, device=dev)
    recurrent(
        mixed_qkv=conv_out,
        a=tensors["a"],
        b=tensors["b"],
        A_log=tensors["A_log"],
        dt_bias=tensors["dt_bias"],
        scale=float(dk) ** -0.5,
        initial_state=ssm_state,
        out=out,
        ssm_state_indices=tensors["idx"],
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    return conv_state, ssm_state, out.squeeze(1)


def _run_fused(geom, t, dev, tensors):
    """Plan + launch through the PATCH's own code path, on the original tensors."""
    dk, dv, hk, hv, conv = geom
    launcher = _arm_state(geom)

    conv_dim = tensors["conv_dim"]
    weight = tensors["conv_w"].view(conv_dim, 1, conv)   # vLLM's unsqueezed conv1d weight
    try:
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"vLLM mamba_utils unimportable ({exc!r})")
    # _plan() rebuilds the (.., dim, state_len) view from kv_cache[0] exactly as the stock
    # method does, so hand it a kv_cache[0] the current layout setting maps back to ours.
    ds_view = tensors["conv_state"]
    kv0 = ds_view if is_conv_state_dim_first() else ds_view.transpose(-1, -2)

    layer = _FakeLayer(
        geom,
        (kv0, tensors["ssm_state"]),
        _FakeConv1d(weight, tensors["conv_bias"]),
        tensors["A_log"],
        tensors["dt_bias"],
    )
    out = torch.zeros(t, hv, dv, dtype=torch.bfloat16, device=dev)
    plan = patch._plan(layer, tensors["mixed_qkv"], tensors["b"], tensors["a"], out,
                       _FakeMeta(t, tensors["idx"]))
    assert plan is not None, "the fast path must engage on a pure non-spec decode batch"
    grid, block, args = plan
    assert grid == (hk, t, 1) and block == (dk, 1, 1), (grid, block)
    launcher(grid=grid, block=block, args=args)
    torch.cuda.synchronize()
    return tensors["conv_state"], tensors["ssm_state"], out


def _compare(stock, fused, idx):
    cs_ref, ss_ref, o_ref = stock
    cs, ss, o = fused
    # Pure data movement: nothing may differ, not even in the last bf16 bit.
    assert torch.equal(cs.reshape(-1).view(torch.uint8), cs_ref.reshape(-1).view(torch.uint8)), \
        "conv state rotation diverged -- that path has no arithmetic to round"
    assert torch.allclose(ss, ss_ref, rtol=STATE_RTOL, atol=STATE_ATOL), \
        (ss - ss_ref).abs().max()
    assert torch.allclose(o.float(), o_ref.float(), rtol=OUT_RTOL, atol=OUT_ATOL), \
        (o.float() - o_ref.float()).abs().max()
    # The null row: both kernels must zero its output and touch nothing else.
    null = (idx == NULL_BLOCK_ID).nonzero().flatten()
    assert null.numel() > 0, "this fixture must contain a null block to be meaningful"
    assert (o[null].float() == 0).all()


@requires_gpu
@pytest.mark.parametrize("tokens", [1, 3, 8])
def test_fused_step_matches_the_stock_triton_pair(tokens):
    torch.manual_seed(0)
    dev = "cuda"
    t = tokens
    tensors = _make_inputs(SMALL_GEOM, t, dev, seed=t)
    stock = _run_stock(SMALL_GEOM, t, dev, tensors)
    fused = _run_fused(SMALL_GEOM, t, dev, tensors)
    _compare(stock, fused, tensors["idx"])


@requires_gpu
def test_fused_step_matches_at_the_production_geometry():
    """HK=16 HV=64 DK=DV=128 CONV=4 -- Qwen3.5-122B-A10B at TP=1, 4 MB of state per line."""
    dev, t = "cuda", 4
    tensors = _make_inputs(PROD_GEOM, t, dev, seed=11)
    stock = _run_stock(PROD_GEOM, t, dev, tensors)
    fused = _run_fused(PROD_GEOM, t, dev, tensors)
    _compare(stock, fused, tensors["idx"])


@requires_gpu
@pytest.mark.parametrize("layout", ["DS", "SD"])
def test_fused_step_handles_both_conv_state_layouts(layout):
    """DS stores (dim, state_len); SD is the transposed checkpoint layout the call site
    flips. The .cu takes both conv-state strides as arguments precisely for this."""
    dev, t = "cuda", 5
    tensors = _make_inputs(SMALL_GEOM, t, dev, seed=21, layout=layout)
    stock = _run_stock(SMALL_GEOM, t, dev, tensors)
    fused = _run_fused(SMALL_GEOM, t, dev, tensors)
    _compare(stock, fused, tensors["idx"])


@requires_gpu
@pytest.mark.parametrize("idx_stride", [1, 4])
def test_fused_step_honours_the_state_index_stride(idx_stride):
    """The eager path passes ``block_table_tensor[:, 0]``, whose stride is the table's row
    stride. Assuming stride 1 reads another request's cache line -- silently, and only off
    the cudagraph path, where the builder happens to copy into a contiguous buffer."""
    dev, t = "cuda", 6
    tensors = _make_inputs(SMALL_GEOM, t, dev, seed=31, idx_stride=idx_stride)
    stock = _run_stock(SMALL_GEOM, t, dev, tensors)
    fused = _run_fused(SMALL_GEOM, t, dev, tensors)
    _compare(stock, fused, tensors["idx"])


@requires_gpu
def test_fused_step_with_a_conv_bias():
    """Qwen3.5's conv1d has bias=False, so the biased branch is otherwise never run."""
    dev, t = "cuda", 3
    tensors = _make_inputs(SMALL_GEOM, t, dev, seed=41, has_bias=True)
    stock = _run_stock(SMALL_GEOM, t, dev, tensors)
    fused = _run_fused(SMALL_GEOM, t, dev, tensors)
    _compare(stock, fused, tensors["idx"])


@requires_gpu
@pytest.mark.parametrize("field,value", [
    ("spec_sequence_masks", object()),
    ("num_prefills", 1),
    ("num_spec_decodes", 2),
    ("num_decodes", 0),
])
def test_plan_declines_anything_that_is_not_a_pure_non_spec_decode(field, value):
    """The fall-through half of the contract, checked against REAL tensors so a decline
    cannot come from an unrelated envelope check passing by accident."""
    dev, t = "cuda", 3
    tensors = _make_inputs(SMALL_GEOM, t, dev, seed=51)
    _arm_state(SMALL_GEOM)
    dk, dv, hk, hv, conv = SMALL_GEOM

    try:
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"vLLM mamba_utils unimportable ({exc!r})")
    ds_view = tensors["conv_state"]
    kv0 = ds_view if is_conv_state_dim_first() else ds_view.transpose(-1, -2)
    layer = _FakeLayer(
        SMALL_GEOM, (kv0, tensors["ssm_state"]),
        _FakeConv1d(tensors["conv_w"].view(tensors["conv_dim"], 1, conv),
                    tensors["conv_bias"]),
        tensors["A_log"], tensors["dt_bias"],
    )
    out = torch.zeros(t, hv, dv, dtype=torch.bfloat16, device=dev)
    md = _FakeMeta(t, tensors["idx"])
    assert patch._plan(layer, tensors["mixed_qkv"], tensors["b"], tensors["a"], out, md) \
        is not None, "control: the untouched fixture must engage"

    setattr(md, field, value)
    assert patch._plan(layer, tensors["mixed_qkv"], tensors["b"], tensors["a"], out, md) \
        is None, f"{field}={value!r} must fall through to the stock pair"

    # ...and so must a geometry the kernel was not compiled for.
    md = _FakeMeta(t, tensors["idx"])
    layer.head_v_dim = dv * 2
    assert patch._plan(layer, tensors["mixed_qkv"], tensors["b"], tensors["a"], out, md) \
        is None, "a foreign geometry must fall through"


# --------------------------------------------------------------------------------------
# no-GPU half
# --------------------------------------------------------------------------------------

def _self_check() -> None:
    import os

    from vtl.registry import PATCH_REGISTRY, is_enabled

    # -- names line up: source file <-> compile_kernel(name=) <-> the entry symbol --
    assert patch.KERNEL == "gdn_decode_step"
    src = nvrtc.load_source(patch.KERNEL)
    assert src, "vtl/kernels/gdn_decode_step.cu missing from the package"
    assert f'extern "C" __global__ void __launch_bounds__(THREADS) {patch.KERNEL}(' in src
    for macro in ("DK", "DV", "HK", "HV", "CONV", "THREADS"):
        assert f"#ifndef {macro}" in src, f"-D{macro} is not guarded in the source"
    for macro in ("DK", "DV", "HK", "HV"):
        assert f'#error "NVRTC: -D{macro}=' in src, f"-D{macro} must have no default"
    # The stride the eager path needs: block_table_tensor[:, 0] is not contiguous.
    assert "indices_stride" in src

    # -- defines: exactly the set the .cu reads, at the geometry this ships for --
    assert patch._defines(PROD_GEOM) == {"DK": 128, "DV": 128, "HK": 16, "HV": 64,
                                         "CONV": 4, "THREADS": 128}
    assert patch._conv_dim(PROD_GEOM) == 12288      # 2*16*128 + 64*128
    assert patch._conv_dim(SMALL_GEOM) == 1536

    # -- one cubin identity per specialization; arch and toolkit count, dict order does not --
    sets = [patch._defines(PROD_GEOM), patch._defines(SMALL_GEOM),
            patch._defines((DK, DV, 8, 64, CONV))]
    keys = {nvrtc.cache_key(src, s, "90a", "12.8") for s in sets}
    assert len(keys) == len(sets), "define sets must not collide in the cubin cache"
    assert nvrtc.cache_key(src, sets[0], "90", "12.8") not in keys
    assert nvrtc.cache_key(src, sets[0], "90a", "12.9") not in keys
    perm = dict(reversed(list(sets[0].items())))
    assert nvrtc.cache_key(src, perm, "90a", "12.8") == nvrtc.cache_key(src, sets[0],
                                                                        "90a", "12.8")

    # -- geometry envelope --
    assert patch._geometry_ok(*PROD_GEOM) is True
    assert patch._geometry_ok(*SMALL_GEOM) is True
    assert patch._geometry_ok(256, 256, 16, 64, 4) is False, "phase 3 is 32 lanes x float4"
    assert patch._geometry_ok(128, 128, 16, 64, 3) is False, "the taps are width 4"
    assert patch._geometry_ok(128, 128, 16, 60, 4) is False, "HV must be a multiple of HK"

    # -- engage condition, against fakes. This is the whole safety story off-box: the fast
    #    path must be blind to anything that is not a pure non-spec decode batch. --
    idx = object()
    assert patch._decode_batch_ok(_FakeMeta(4, idx)) is True
    # A full-cudagraph decode batch is TOKEN-padded; its padded rows carry NULL_BLOCK_ID,
    # which the kernel skips. Rejecting it would disable the fast path where it matters.
    assert patch._decode_batch_ok(_FakeMeta(32, idx, num_decodes=4)) is True
    for field, value in (("spec_sequence_masks", object()), ("num_prefills", 1),
                         ("num_spec_decodes", 2), ("num_decodes", 0),
                         ("num_actual_tokens", 0)):
        md = _FakeMeta(4, idx)
        setattr(md, field, value)
        assert patch._decode_batch_ok(md) is False, f"{field}={value!r} must not engage"
    md = _FakeMeta(2, idx, num_decodes=4)
    assert patch._decode_batch_ok(md) is False, "fewer tokens than decodes is incoherent"
    assert patch._decode_batch_ok(object()) is False, "unknown metadata must not engage"

    # -- registration: present, OFF by default, env-overridable --
    p = next(x for x in PATCH_REGISTRY if x.name == patch.NAME)
    assert p.default is False and is_enabled(p) is False
    os.environ["VTL_ENABLE_GDN_DECODE_STEP"] = "1"
    assert is_enabled(p) is True
    os.environ.pop("VTL_ENABLE_GDN_DECODE_STEP")

    # -- import-without-vLLM: with VTL_NVRTC off, apply() is a clean no-op and nothing is
    #    armed; with it on but vLLM absent it may raise (registry.apply_all isolates it)
    #    but must not leave a half-armed module. --
    os.environ.pop("VTL_NVRTC", None)
    assert nvrtc.compile_kernel(patch.KERNEL, sets[0]) is None
    patch.apply()
    assert patch._state["launcher"] is None and patch._state["installed"] is False
    os.environ["VTL_NVRTC"] = "1"
    try:
        patch.apply()
    except Exception:
        pass
    finally:
        os.environ.pop("VTL_NVRTC", None)
    assert patch._state["launcher"] is None and patch._state["installed"] is False

    print("test_gdn_decode_step self-check ok")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    elif not HAVE_PYTEST:
        raise SystemExit("pytest not installed; run with --self-check")
    else:
        raise SystemExit(pytest.main([__file__, "-q"]))
