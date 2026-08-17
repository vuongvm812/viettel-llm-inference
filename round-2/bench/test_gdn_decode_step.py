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

THE FUSED EPILOGUE (-DFUSED_EPILOGUE=1) IS HELD TO A STRICTER STANDARD THAN THE CORE, and
the two halves are tested against different oracles on purpose:

  vs the STANDALONE KERNEL   BIT-EXACT (``torch.equal`` on the fp8 codes AND the fp32
                             scales). ``vtl/kernels/gated_rmsnorm_group_quant.cu`` is what
                             runs today, and bench/test_gdn_gated_rmsnorm.py's oracles
                             describe ITS math; if the fused epilogue drifted by even one
                             code those oracles would stop covering the production path.
                             The comparison is run on IDENTICAL inputs -- the unfused
                             build's own bf16 output, fed to the standalone kernel -- so
                             the only thing under test is the epilogue.
  vs the CHAINED ORACLE      (stock Triton pair -> RMSNormGated reference -> group-quant
                             reference) fp8 codes within one e4m3 step, and under 1% of
                             codes differing at all. Bit-exactness is NOT available here
                             and asking for it would be wrong: the core's two documented
                             fp32 divergences from Triton (reduction order, transcendental
                             lowering) move `o` by ~1 ulp fp32, which can flip a code that
                             sits on an e4m3 rounding boundary. One step bounds the flip;
                             the 1% fraction is what separates rounding noise from a
                             systematic error (a wrong scale layout or a dropped gate flips
                             tens of percent). Same two bounds bench/test_gdn_gated_rmsnorm
                             .py uses for the same reason.
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

# Fused epilogue. FP8_RTOL is one e4m3 mantissa step (3 bits -> 2**-3 = 0.125, plus slack
# for the binade edge); FP8_ATOL is one subnormal step at the quantized magnitude scale.
# FP8_MISMATCH_FRAC separates reduction-order noise from a systematic error -- both bounds
# are lifted from bench/test_gdn_gated_rmsnorm.py::_assert_fp8_equivalent.
FP8_RTOL, FP8_ATOL = 0.13, 2.0 ** -9
FP8_MISMATCH_FRAC = 1e-2
NORM_EPS = 1e-6                       # RMSNormGated eps on this model
GROUP = 128
FP8_MAX = 448.0


# --------------------------------------------------------------------------------------
# fakes: just enough of the layer + metadata for the patch's own _plan() to run
# --------------------------------------------------------------------------------------

class _FakeConv1d:
    def __init__(self, weight, bias):
        self.weight = weight   # [conv_dim, 1, CONV] -- vLLM's unsqueezed conv1d weight
        self.bias = bias


class _FakeNorm:
    """vLLM's RMSNormGated, reduced to the four attributes _norm_params() reads."""

    def __init__(self, weight, eps=NORM_EPS, activation="silu"):
        self.weight = weight
        self.eps = eps
        self.activation = activation
        self.norm_before_gate = True
        self.group_size = None


class _FakeLayer:
    """Only the attributes _plan() actually reads. Wrong on purpose: nothing else exists,
    so a _plan() that starts depending on more of the layer fails here rather than in
    production."""

    def __init__(self, geom, kv_cache, conv1d, A_log, dt_bias, activation="silu",
                 norm=None):
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
        self.norm = norm
        self.gqa_interleaved_layout = False


class _FakeMeta:
    def __init__(self, tokens, idx, num_decodes=None):
        self.spec_sequence_masks = None
        self.num_prefills = 0
        self.num_spec_decodes = 0
        self.num_decodes = tokens if num_decodes is None else num_decodes
        self.num_actual_tokens = tokens
        self.non_spec_state_indices_tensor = idx


def _compile(geom, defines):
    src = nvrtc.load_source(patch.KERNEL)
    assert src, "vtl/kernels/gdn_decode_step.cu must ship with the package"
    built = nvrtc.build_cubin(src, patch.KERNEL, defines)
    if built is None:
        pytest.skip("NVRTC unavailable (no cuda-python / no driver) -- stock pair stands")
    return nvrtc._load_cubin(built[0], patch.KERNEL)


def _arm_state(geom):
    """Compile the shipped kernel for `geom` and arm the patch, or skip with the reason.

    Leaves the FUSED launcher unarmed: the unfused tests must exercise the unfused path.
    """
    patch._claims.clear()
    patch.EPILOGUE_CONSUMERS.clear()
    patch._state.update(geom=geom, launcher=_compile(geom, patch._defines(geom)),
                        launcher_fused=None, installed=True, armed=True)
    return patch._state["launcher"]


def _arm_fused(geom, is_silu=True, w_fp32=False):
    """Arm BOTH launchers, as _arm_epilogue does on the box."""
    _arm_state(geom)
    defines = patch._fused_defines(geom, is_silu, w_fp32)
    patch._state["launcher_fused"] = _compile(geom, defines)
    return patch._state["launcher_fused"]


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


def _make_inputs(geom, tokens, dev, seed, layout="DS", idx_stride=1, has_bias=False,
                 with_gate=False, w_dtype=None):
    dk, dv, hk, hv, conv = geom
    conv_dim = patch._conv_dim(geom)
    hidden = hv * dv
    # NOT a default argument: `make check` imports this module with torch absent, and a
    # default is evaluated at import time.
    w_dtype = torch.bfloat16 if w_dtype is None else w_dtype
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
    if with_gate:
        # EXACTLY the production provenance: forward_cuda splits ONE projection buffer into
        # mixed_qkv and z, so mixed_qkv is a strided view and z is the columns after it.
        # A contiguous stand-in would never exercise _z_ptr's stride recovery.
        qkvz = rnd(tokens, conv_dim + hidden)
        mixed_qkv, z = qkvz.split([conv_dim, hidden], dim=-1)
        z[0].zero_()               # a zero gate row
        z[1].fill_(20.0)           # a saturated one
    else:
        qkvz, z = None, None
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
                conv_bias=conv_bias, idx=idx, conv_dim=conv_dim, qkvz=qkvz, z=z,
                hidden=hidden, norm_w=rnd(dv, dtype=w_dtype))


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


def _make_layer(geom, tensors, with_norm=False):
    """A _FakeLayer wired to the fixture, with kv_cache[0] in whatever layout the installed
    vLLM's ``is_conv_state_dim_first()`` says _plan() will transpose back to ours."""
    dk, dv, hk, hv, conv = geom
    conv_dim = tensors["conv_dim"]
    try:
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"vLLM mamba_utils unimportable ({exc!r})")
    ds_view = tensors["conv_state"]
    kv0 = ds_view if is_conv_state_dim_first() else ds_view.transpose(-1, -2)
    return _FakeLayer(
        geom,
        (kv0, tensors["ssm_state"]),
        _FakeConv1d(tensors["conv_w"].view(conv_dim, 1, conv), tensors["conv_bias"]),
        tensors["A_log"],
        tensors["dt_bias"],
        norm=_FakeNorm(tensors["norm_w"]) if with_norm else None,
    )


def _attach_epilogue(layer, geom, rows, tensors, consumer=True, mode=None):
    """What _arm_epilogue() attaches on the box: staging buffers + the consumer handshake.

    ``mode`` fast-forwards the three-state latch. Production always starts at None (probe),
    but a test that wants to see the bf16 write actually GONE has to reach "fused", and
    doing so by hand keeps that assertion independent of PROBE_CONSUMES.
    """
    dk, dv, hk, hv, conv = geom
    hidden = hv * dv
    dev = tensors["norm_w"].device
    layer._vtl_gdn_epilogue = {
        "mode": mode,
        "consumed": patch.PROBE_CONSUMES if mode == "fused" else 0,
        "pending": False,
        "weight_ptr": tensors["norm_w"].data_ptr(),
        "w_ptr": tensors["norm_w"].data_ptr(),
        "eps": NORM_EPS,
        "is_silu": True,
        "rows": rows,
        "hidden": hidden,
        "fp8": torch.zeros(rows, hidden, dtype=torch.float8_e4m3fn, device=dev),
        "scales": torch.zeros(rows, hv, dtype=torch.float32, device=dev),
        "views": {},
    }
    patch.EPILOGUE_CONSUMERS.clear()
    if consumer:
        patch.note_epilogue_consumer(tensors["norm_w"].data_ptr())
    return layer._vtl_gdn_epilogue


def _run_fused(geom, t, dev, tensors):
    """Plan + launch through the PATCH's own code path, on the original tensors."""
    dk, dv, hk, hv, conv = geom
    launcher = _arm_state(geom)
    layer = _make_layer(geom, tensors)
    out = torch.zeros(t, hv, dv, dtype=torch.bfloat16, device=dev)
    plan = patch._plan(layer, tensors["mixed_qkv"], tensors["b"], tensors["a"], out,
                       _FakeMeta(t, tensors["idx"]))
    assert plan is not None, "the fast path must engage on a pure non-spec decode batch"
    grid, block, args, chosen, claim = plan
    assert grid == (hk, t, 1) and block == (dk, 1, 1), (grid, block)
    assert chosen is launcher and claim is None, "no epilogue armed -> the unfused launcher"
    chosen(grid=grid, block=block, args=args)
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
# GPU half: the fused epilogue
# --------------------------------------------------------------------------------------

def _reference_epilogue(x, z, weight, is_silu=True, eps=NORM_EPS):
    """RMSNormGated.forward_static -> reshape [T, H*D] -> per_token_group_fp8_quant.

    Byte for byte the composition bench/test_gdn_gated_rmsnorm.py::reference_group_quant
    and gdn_kernels._stock_group_quant use, including the double rounding at the op
    boundary, the 1e-10 amax FLOOR (not a min-scale clamp) and the /448 scale.
    """
    xf = x.float()
    zf = z.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    y = (xf * torch.rsqrt(var + eps)) * weight.float()
    act = zf * torch.sigmoid(zf) if is_silu else torch.sigmoid(zf)
    yq = (y * act).to(x.dtype).float()
    amax = yq.abs().amax(-1).clamp(min=1e-10)
    scale = amax / FP8_MAX
    q = torch.clamp(yq / scale.unsqueeze(-1), -FP8_MAX, FP8_MAX)
    return q.reshape(x.shape[0], -1).to(torch.float8_e4m3fn), scale


def _run_fused_epilogue(geom, t, dev, tensors, rows=None, is_silu=True, mode="fused"):
    """Plan + launch the FUSED variant through the patch, and consume the claim the way
    gdn_kernels' op does. Returns (fp8, scales, out_buffer)."""
    dk, dv, hk, hv, conv = geom
    fused = _arm_fused(geom, is_silu=is_silu)
    layer = _make_layer(geom, tensors, with_norm=True)
    _attach_epilogue(layer, geom, rows or t, tensors, mode=mode)

    out = torch.zeros(t, hv, dv, dtype=torch.bfloat16, device=dev)
    plan = patch._plan(layer, tensors["mixed_qkv"], tensors["b"], tensors["a"], out,
                       _FakeMeta(t, tensors["idx"]))
    assert plan is not None, "the fused path must engage once every input is resolvable"
    grid, block, args, chosen, claim = plan
    assert chosen is fused, "the fused launcher must be the one chosen"
    assert claim is not None and claim[0] == out.data_ptr()
    chosen(grid=grid, block=block, args=args)
    patch._publish_claim(*claim)

    # ...and the consumer side, verbatim: a reshape VIEW of core_attn_out is what the op
    # is handed, so the pointer identity has to survive it.
    staged = patch.take_epilogue(out.reshape(-1, dv), hv)
    assert staged is not None, "the claim must be consumable through the reshape view"
    torch.cuda.synchronize()
    return staged[0], staged[1], out


def _assert_fp8_close(got_q, exp_q, got_s, exp_s):
    torch.testing.assert_close(got_q.float(), exp_q.float(), rtol=FP8_RTOL, atol=FP8_ATOL)
    frac = (got_q.view(torch.uint8) != exp_q.view(torch.uint8)).float().mean().item()
    assert frac < FP8_MISMATCH_FRAC, \
        f"{frac:.2%} of fp8 codes differ -- systematic, not reduction noise"
    torch.testing.assert_close(got_s, exp_s, rtol=1e-3, atol=0.0)


@requires_gpu
@pytest.mark.parametrize("tokens", [1, 3, 8])
def test_fused_epilogue_bitmatches_the_standalone_norm_quant_kernel(tokens):
    """THE parity requirement: identical inputs, identical bytes.

    The unfused build's bf16 output is fed to ``vtl/kernels/gated_rmsnorm_group_quant.cu``
    -- the kernel that runs today -- and the fused build must reproduce its fp8 codes and
    fp32 scales exactly. That is what keeps bench/test_gdn_gated_rmsnorm.py's oracles a
    description of the production path after the fusion.
    """
    dev, t = "cuda", tokens
    geom = SMALL_GEOM
    dk, dv, hk, hv, conv = geom
    tensors = _make_inputs(geom, t, dev, seed=60 + t, with_gate=True)

    # unfused core -> bf16 out. Same inputs, CLONED state, so the fused run below starts
    # from the same place rather than from this run's leftovers.
    plain = dict(tensors)
    plain["conv_state"] = tensors["conv_state"].clone()
    plain["ssm_state"] = tensors["ssm_state"].clone()
    _arm_state(geom)
    layer = _make_layer(geom, plain)
    out = torch.zeros(t, hv, dv, dtype=torch.bfloat16, device=dev)
    grid, block, args, chosen, claim = patch._plan(
        layer, plain["mixed_qkv"], plain["b"], plain["a"], out, _FakeMeta(t, plain["idx"]))
    chosen(grid=grid, block=block, args=args)
    torch.cuda.synchronize()

    exp_q, exp_s = _run_standalone_norm_quant(out, tensors["z"].reshape(t, hv, dv),
                                              tensors["norm_w"], hv)
    got_q, got_s, _ = _run_fused_epilogue(geom, t, dev, tensors, mode="fused")

    assert torch.equal(got_q.view(torch.uint8), exp_q.view(torch.uint8)), \
        "the fused epilogue must be BIT-IDENTICAL to the standalone kernel it replaces"
    assert torch.equal(got_s, exp_s), "group scales must be bit-identical too"


def _run_standalone_norm_quant(x, z, weight, hv):
    """vtl/kernels/gated_rmsnorm_group_quant.cu on [T, HV, DV] inputs, via its own patch's
    define/grid helpers -- so a drift between the two kernels' defines shows up here."""
    from vtl.patches import gdn_kernels

    dv = weight.numel()
    t = x.shape[0]
    rows = t * hv
    defines = gdn_kernels._nvrtc_defines(dv, hv, True, weight.dtype == torch.float32)
    built = nvrtc.build_cubin(nvrtc.load_source("gated_rmsnorm_group_quant"),
                              "gated_rmsnorm_group_quant", defines)
    if built is None:  # pragma: no cover
        pytest.skip("NVRTC unavailable -- cannot build the standalone twin")
    launch = nvrtc._load_cubin(built[0], "gated_rmsnorm_group_quant")
    q = torch.zeros(t, hv * dv, dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.zeros(t, hv, dtype=torch.float32, device=x.device)
    launch(
        grid=gdn_kernels._nvrtc_grid(rows, dv),
        block=(gdn_kernels.NVRTC_THREADS, 1, 1),
        args=nvrtc.pack_args(
            q.data_ptr(), s.data_ptr(), x.contiguous().data_ptr(), z.contiguous().data_ptr(),
            weight.data_ptr(), ("l", s.stride(0)), ("l", s.stride(1)), ("l", rows),
            ("f", NORM_EPS),
        ),
    )
    torch.cuda.synchronize()
    return q, s


@requires_gpu
@pytest.mark.parametrize("geom", [SMALL_GEOM, PROD_GEOM])
def test_fused_epilogue_matches_the_chained_stock_oracle(geom):
    """stock Triton pair -> RMSNormGated reference -> group-quant reference, end to end.

    The tolerance (not bit-exactness) is the core's, not the epilogue's -- see the module
    docstring. The previous test is what pins the epilogue itself.
    """
    dev, t = "cuda", 4
    dk, dv, hk, hv, conv = geom
    tensors = _make_inputs(geom, t, dev, seed=71, with_gate=True)
    ref_cs, ref_ss, ref_out = _run_stock(geom, t, dev, tensors)
    exp_q, exp_s = _reference_epilogue(ref_out, tensors["z"].reshape(t, hv, dv),
                                       tensors["norm_w"])
    got_q, got_s, out = _run_fused_epilogue(geom, t, dev, tensors, mode="fused")

    _assert_fp8_close(got_q, exp_q, got_s, exp_s)
    # The state is still the core's business and still bounded the same way.
    assert torch.equal(tensors["conv_state"].view(torch.uint8), ref_cs.view(torch.uint8))
    assert torch.allclose(tensors["ssm_state"], ref_ss, rtol=STATE_RTOL, atol=STATE_ATOL)
    # ...and the bf16 core output is NOT written by the fused build. That is the traffic the
    # fusion removes, so leaving it written would mean the fusion did not happen.
    assert (out.float() == 0).all(), "the fused build must not write core_attn_out"


@requires_gpu
def test_fused_epilogue_quantizes_null_rows_like_the_stock_chain():
    """A padded cudagraph row (NULL_BLOCK_ID) has a zero core output, and the stock chain
    still norms and quantizes it: amax floors at 1e-10, the scale is 1e-10/448 and every
    code is 0. Short-circuiting the row would write whatever the buffer held."""
    dev, t = "cuda", 3
    geom = SMALL_GEOM
    dk, dv, hk, hv, conv = geom
    tensors = _make_inputs(geom, t, dev, seed=81, with_gate=True)
    got_q, got_s, out = _run_fused_epilogue(geom, t, dev, tensors, mode=None)
    null = (tensors["idx"] == NULL_BLOCK_ID).nonzero().flatten()
    # probe mode zeroes the bf16 row too, exactly as the unfused build does
    assert (out[null].float() == 0).all()
    assert null.numel() > 0
    assert (got_q[null].float() == 0).all(), "a null row's codes must all be zero"
    torch.testing.assert_close(
        got_s[null], torch.full_like(got_s[null], 1e-10 / FP8_MAX), rtol=1e-6, atol=0.0)


@requires_gpu
def test_probe_mode_is_a_strict_superset_of_the_unfused_launch():
    """Before the handoff has been observed working, the fused build must ALSO write the
    bf16 core output -- bit-identically to the unfused build. That is what makes the very
    first serving steps safe no matter which way the downstream actually goes, and it is
    the only reason dropping the write later is not a leap of faith."""
    dev, t = "cuda", 4
    geom = SMALL_GEOM
    dk, dv, hk, hv, conv = geom
    tensors = _make_inputs(geom, t, dev, seed=111, with_gate=True)

    plain = dict(tensors)
    plain["conv_state"] = tensors["conv_state"].clone()
    plain["ssm_state"] = tensors["ssm_state"].clone()
    _arm_state(geom)
    layer = _make_layer(geom, plain)
    ref = torch.zeros(t, hv, dv, dtype=torch.bfloat16, device=dev)
    grid, block, args, chosen, _ = patch._plan(
        layer, plain["mixed_qkv"], plain["b"], plain["a"], ref, _FakeMeta(t, plain["idx"]))
    chosen(grid=grid, block=block, args=args)
    torch.cuda.synchronize()

    q, s, out = _run_fused_epilogue(geom, t, dev, tensors, mode=None)
    assert torch.equal(out.view(torch.uint8), ref.view(torch.uint8)), \
        "probe mode must write the SAME bf16 bytes the unfused build writes"
    # ...and the fp8 it produced alongside is still the real thing.
    exp_q, exp_s = _reference_epilogue(ref, tensors["z"].reshape(t, hv, dv),
                                       tensors["norm_w"])
    assert torch.equal(q.view(torch.uint8), exp_q.view(torch.uint8))
    assert torch.equal(s, exp_s)


@requires_gpu
def test_probe_latches_to_fused_only_after_observed_handoffs():
    """The state machine, on the real launcher: probe until PROBE_CONSUMES claims are
    actually consumed, then fused. And a claim nobody consumes latches to plain forever."""
    dev, t = "cuda", 3
    geom = SMALL_GEOM
    dk, dv, hk, hv, conv = geom
    tensors = _make_inputs(geom, t, dev, seed=121, with_gate=True)
    _arm_fused(geom)
    layer = _make_layer(geom, tensors, with_norm=True)
    epi = _attach_epilogue(layer, geom, t, tensors)
    out = torch.zeros(t, hv, dv, dtype=torch.bfloat16, device=dev)
    md = _FakeMeta(t, tensors["idx"])

    for _ in range(patch.PROBE_CONSUMES):
        _, _, _, _, claim = patch._plan(layer, tensors["mixed_qkv"], tensors["b"],
                                        tensors["a"], out, md)
        assert epi["mode"] == "probe" and epi["pending"] is True
        patch._publish_claim(*claim)
        assert patch.take_epilogue(out.reshape(-1, dv), hv) is not None
    patch._plan(layer, tensors["mixed_qkv"], tensors["b"], tensors["a"], out, md)
    assert epi["mode"] == "fused", "observed handoffs must promote the layer"

    # the negative arm, from a fresh layer
    layer2 = _make_layer(geom, tensors, with_norm=True)
    epi2 = _attach_epilogue(layer2, geom, t, tensors)
    patch._plan(layer2, tensors["mixed_qkv"], tensors["b"], tensors["a"], out, md)
    patch._claims.clear()          # nobody consumed it
    assert patch._plan(layer2, tensors["mixed_qkv"], tensors["b"], tensors["a"], out,
                       md)[4] is None
    assert epi2["mode"] == "plain", "an unconsumed claim must latch the layer off"


@requires_gpu
def test_fused_epilogue_declines_when_the_gate_is_not_recoverable():
    """A contiguous ``mixed_qkv`` is not the split view ``forward_cuda`` produces, so there
    is no z behind it. The patch must fall back to the unfused launcher rather than read
    whatever follows the buffer -- and having done so, must stay latched there."""
    dev, t = "cuda", 3
    geom = SMALL_GEOM
    dk, dv, hk, hv, conv = geom
    tensors = _make_inputs(geom, t, dev, seed=91, with_gate=False)
    _arm_fused(geom)
    layer = _make_layer(geom, tensors, with_norm=True)
    epi = _attach_epilogue(layer, geom, t, tensors)
    out = torch.zeros(t, hv, dv, dtype=torch.bfloat16, device=dev)
    grid, block, args, chosen, claim = patch._plan(
        layer, tensors["mixed_qkv"], tensors["b"], tensors["a"], out,
        _FakeMeta(t, tensors["idx"]))
    assert chosen is patch._state["launcher"] and claim is None
    assert epi["mode"] == "plain", "the decision must latch, not be re-taken every step"
    # ...and the unfused launcher it fell back to still writes the bf16 output.
    chosen(grid=grid, block=block, args=args)
    torch.cuda.synchronize()
    assert not (out.float() == 0).all()


@requires_gpu
@pytest.mark.parametrize("case", ["no-consumer", "staging-rows", "out-rows"])
def test_fused_epilogue_claim_conditions(case):
    """Each reason the bf16 write may NOT be dropped, checked against real tensors."""
    dev, t = "cuda", 3
    geom = SMALL_GEOM
    dk, dv, hk, hv, conv = geom
    tensors = _make_inputs(geom, t, dev, seed=101, with_gate=True)
    _arm_fused(geom)
    layer = _make_layer(geom, tensors, with_norm=True)
    rows = t
    consumer = True
    if case == "no-consumer":
        consumer = False           # gdn_kernels' op never ran for this layer
    elif case == "staging-rows":
        rows = t - 1               # the batch is wider than the staging buffers
    _attach_epilogue(layer, geom, rows, tensors, consumer=consumer)

    n_out = t + 1 if case == "out-rows" else t   # padded rows the consumer would also norm
    out = torch.zeros(n_out, hv, dv, dtype=torch.bfloat16, device=dev)
    plan = patch._plan(layer, tensors["mixed_qkv"], tensors["b"], tensors["a"], out,
                       _FakeMeta(t, tensors["idx"]))
    assert plan is not None, "control: the unfused fast path must still engage"
    assert plan[3] is patch._state["launcher"], f"{case} must not take the fused launcher"
    assert plan[4] is None and not patch._claims, f"{case} must publish no claim"


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
    for macro in ("DK", "DV", "HK", "HV", "CONV", "THREADS", "FUSED_EPILOGUE"):
        assert f"#ifndef {macro}" in src, f"-D{macro} is not guarded in the source"
    for macro in ("DK", "DV", "HK", "HV"):
        assert f'#error "NVRTC: -D{macro}=' in src, f"-D{macro} must have no default"
    # The stride the eager path needs: block_table_tensor[:, 0] is not contiguous.
    assert "indices_stride" in src

    # -- the epilogue is COMPILED OUT of the unfused build, not branched around, and it
    #    carries the same constants (and the same hardware converter) as its standalone
    #    twin. A drift in either would silently end the bit-parity the tests above assert. --
    assert "#if FUSED_EPILOGUE" in src and "gdn_norm_quant_epilogue" in src
    twin = nvrtc.load_source("gated_rmsnorm_group_quant")
    assert twin, "vtl/kernels/gated_rmsnorm_group_quant.cu missing from the package"
    for const in ("kFp8Max = 448.0f", "kGroupQuantEps = 1e-10f",
                  "__nv_cvt_float2_to_fp8x2", "__NV_SATFINITE, __NV_E4M3"):
        assert const in src and const in twin, f"epilogue/twin drift on {const!r}"

    # -- defines: exactly the set the .cu reads, at the geometry this ships for --
    assert patch._defines(PROD_GEOM) == {"DK": 128, "DV": 128, "HK": 16, "HV": 64,
                                         "CONV": 4, "THREADS": 128, "FUSED_EPILOGUE": 0}
    assert patch._fused_defines(PROD_GEOM, True, False) == {
        "DK": 128, "DV": 128, "HK": 16, "HV": 64, "CONV": 4, "THREADS": 128,
        "FUSED_EPILOGUE": 1, "GROUP": 128, "IS_SILU": 1, "W_FP32": 0}
    assert patch.GROUP == GROUP, "the quant group must be the checkpoint's 128"
    assert patch._conv_dim(PROD_GEOM) == 12288      # 2*16*128 + 64*128
    assert patch._conv_dim(SMALL_GEOM) == 1536

    # 8192 / 128 == 64 == HV: one head is exactly one quant group. That identity is the
    # whole reason the epilogue needs no cross-block communication -- assert it, do not
    # assume it.
    assert (PROD_GEOM[3] * PROD_GEOM[1]) // GROUP == PROD_GEOM[3]
    assert PROD_GEOM[1] == GROUP

    # -- one cubin identity per specialization; arch and toolkit count, dict order does not.
    #    The fused and unfused builds are DIFFERENT PROGRAMS and must never share a cubin. --
    sets = [patch._defines(PROD_GEOM), patch._defines(SMALL_GEOM),
            patch._defines((DK, DV, 8, 64, CONV)),
            patch._fused_defines(PROD_GEOM, True, False),
            patch._fused_defines(PROD_GEOM, False, False),
            patch._fused_defines(PROD_GEOM, True, True),
            patch._fused_defines(SMALL_GEOM, True, False)]
    keys = {nvrtc.cache_key(src, s, "90a", "12.8") for s in sets}
    assert len(keys) == len(sets), "define sets must not collide in the cubin cache"
    assert nvrtc.cache_key(src, sets[0], "90", "12.8") not in keys
    assert nvrtc.cache_key(src, sets[0], "90a", "12.9") not in keys
    perm = dict(reversed(list(sets[0].items())))
    assert nvrtc.cache_key(src, perm, "90a", "12.8") == nvrtc.cache_key(src, sets[0],
                                                                        "90a", "12.8")

    # -- the fused geometry envelope, on top of the core one --
    assert patch._fused_geometry_ok(PROD_GEOM) is True
    assert patch._fused_geometry_ok(SMALL_GEOM) is True
    assert patch._fused_geometry_ok((DK, DV, 4, 64, CONV)) is False, "GPB=16 -> 256 lanes"
    assert patch._fused_geometry_ok((DK, 64, 16, 64, CONV)) is False, "DV must BE the group"

    # -- the claim book and the consumer handshake, with fakes. This is the whole safety
    #    story for dropping the bf16 write: no consumer -> no claim -> no dropped write. --
    class _Ptr:
        def __init__(self, ptr, shape):
            self._p, self.shape = ptr, shape

        def data_ptr(self):
            return self._p

    patch._claims.clear()
    patch.EPILOGUE_CONSUMERS.clear()
    assert patch.take_epilogue(_Ptr(0x40, (128, 128)), 64) is None
    patch._publish_claim(0x40, 2, 8192, "Q", "S")
    assert patch.take_epilogue(_Ptr(0x80, (128, 128)), 64) is None, "pointer identity"
    assert patch.take_epilogue(_Ptr(0x40, (128, 128)), 64) == ("Q", "S")
    assert patch.take_epilogue(_Ptr(0x40, (128, 128)), 64) is None, "consumed exactly once"
    patch.note_epilogue_consumer(0x1234)
    assert 0x1234 in patch.EPILOGUE_CONSUMERS
    patch.EPILOGUE_CONSUMERS.clear()
    patch._claims.clear()

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

    # -- registration: present, ON by default, env-overridable --
    # Enabled by default per project decision (2026-08-17); the compose gate now restates the
    # code default. Default-on is safe because the ladders, not the gate, are the safety
    # story: NVRTC -> stock conv1d + delta rule, the geometry envelope, the `_decode_batch_ok`
    # engage predicate, and the launch-failure latch.
    p = next(x for x in PATCH_REGISTRY if x.name == patch.NAME)
    assert p.default is True and is_enabled(p) is True
    os.environ["VTL_ENABLE_GDN_DECODE_STEP"] = "0"
    assert is_enabled(p) is False
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
    assert patch._state["launcher_fused"] is None, "no fused launcher may survive a bad arm"

    # -- gdn_kernels' side of the handshake must exist and must be import-safe without
    #    torch: the decode step can only drop the bf16 write because that op calls back. --
    from vtl.patches import gdn_kernels

    assert hasattr(gdn_kernels, "_decode_step_epilogue")
    assert gdn_kernels.GROUP == GROUP, "the two patches must agree on the quant group"

    print("test_gdn_decode_step self-check ok")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    elif not HAVE_PYTEST:
        raise SystemExit("pytest not installed; run with --self-check")
    else:
        raise SystemExit(pytest.main([__file__, "-q"]))
