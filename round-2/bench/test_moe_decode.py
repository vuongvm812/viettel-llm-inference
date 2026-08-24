"""Harness for the MoE decode grouped-GEMV ([[moe_decode_gemv]] + kernels/moe_decode_gemv.cu).

    python3 bench/test_moe_decode.py --self-check   # no GPU, no torch, no vLLM
    pytest -q bench/test_moe_decode.py              # GPU: compile + numeric parity
    python3 bench/test_moe_decode.py                # GPU: CUDA-event table vs stock

WHAT THE GPU HALF IS FOR. A hand-written GEMV that reads packed int4 out of a 3-D expert
stack has three places to be silently wrong, and all three produce plausible-looking numbers:

  1. the nibble order (element 8j+i vs i*L/8+j) -- wrong by a permutation, right by magnitude
  2. the scale indexing (per-row group scale vs the checkpoint's [128,128] block scale row)
  3. the (token, slot) -> block mapping, which transposes tokens and experts

So the oracle is a torch reference built from the DEQUANTIZED weights, not from the original
fp32: quantization error is then not in the comparison at all, and any of the three bugs
above shows up as a gross mismatch instead of as "int4 is lossy". The int4 arm dequantizes
through the patch's OWN packer output, so a packing bug cannot cancel itself out.

TOLERANCES.
  REL_TOL = 1.5e-2 on the relative Frobenius norm of the whole [M, K] output. The kernel and
  the reference do the same arithmetic in a different ORDER (the kernel folds the group scales
  in per 32-element chunk and reduces across a warp; the reference does one fp32 dot), so
  bitwise equality is the wrong bar. The floor is the bf16 output cast, ~4e-3 relative; the
  headroom above that covers the rare case where an fp32 ordering difference pushes an
  intermediate across an fp8 quantization boundary in the group-128 requant between the two
  GEMVs, which costs ~2^-3 relative on that one element.
  COS_TOL = 0.9985 on the flattened cosine similarity. A pure scale error (a group scale off
  by a constant factor) can hide inside a norm ratio; it cannot hide here.
  The stock-fused_moe comparison is looser (STOCK_REL_TOL) because stock's Triton kernel
  quantizes activations per its own recipe and accumulates in a different tiling entirely --
  it is a "same answer" check, not a "same arithmetic" check.

Style: bench/test_w4a8_v2.py (importorskip, so the bench/ glob cannot collection-error on a
host with no torch) + bench/test_nvrtc.py (a --self-check main that needs nothing at all).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vtl import nvrtc  # noqa: E402
from vtl.patches import moe_decode_gemv as moe  # noqa: E402

GROUP = moe.GROUP
FP8_MAX = 448.0
MIN_SCALE = 1.0 / (448.0 * 512.0)

# Small but STRUCTURALLY faithful: K and N multiples of 128 (so the block-scale grid has more
# than one row and column and a transposed index cannot pass), more experts than TOPK, and an
# expert count that is not a multiple of TOPK.
E, K, N, TOPK = 6, 256, 128, 3
MS = (1, 2, 6, 16)
THREADS = 256
REL_TOL = 1.5e-2
COS_TOL = 0.9985
STOCK_REL_TOL = 6e-2

try:
    import pytest

    HAVE_PYTEST = True
except ImportError:  # pragma: no cover -- the `make check` path
    HAVE_PYTEST = False

    class _NoPytest:
        """Enough of pytest for the decorators to evaluate at import time. The bodies never
        run without pytest, so no-ops are correct here, not a stub implementation."""

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
    torch = pytest.importorskip("torch", reason="the MoE GEMV parity half needs torch + CUDA")
else:  # pragma: no cover
    torch = None


def _skip_reason() -> str | None:
    if torch is None or not torch.cuda.is_available():
        return "needs torch + a CUDA device"
    try:
        from cuda.bindings import driver  # noqa: F401
    except Exception:
        try:
            from cuda import driver  # noqa: F401
        except Exception as exc:
            return f"cuda-python not available ({exc}); NVRTC path cannot run"
    return None


SKIP = _skip_reason() if HAVE_PYTEST else "no pytest"
requires_gpu = pytest.mark.skipif(SKIP is not None, reason=str(SKIP))


# ---------------------------------------------------------------------------------------
# Compilation. build_cubin + _load_cubin rather than compile_kernel, for the same reason
# test_nvrtc.py does it: compile_kernel is gated on VTL_NVRTC and this test is ABOUT the
# kernel, so the gate would turn "the kernel is broken" into "the test was skipped".
# ---------------------------------------------------------------------------------------


def _kernels(weights: str, k: int = K, n: int = N, topk: int = TOPK,
             extra: tuple[str, ...] = ()):
    """``{entry: launchable}`` for one arm, or skip with the reason NVRTC gave."""
    src = nvrtc.load_source(moe.KERNEL)
    assert src, "vtl/kernels/moe_decode_gemv.cu must ship with the package"
    defines = moe.kernel_defines(k, n, topk, weights, THREADS)
    built = nvrtc.build_cubin(src, moe.KERNEL, defines)
    if built is None:
        pytest.skip(f"NVRTC could not build the {weights} arm (no driver? bad arch?)")
    cubin, _key = built
    return {e: nvrtc._load_cubin(cubin, e) for e in moe.ENTRIES + extra}


# ---------------------------------------------------------------------------------------
# Oracles. Everything below mirrors the kernel's arithmetic contract exactly; the only
# licensed difference is the ORDER of the fp32 reduction.
# ---------------------------------------------------------------------------------------


def quant_groups(v, group: int = GROUP):
    """fp32 ``[..., L]`` -> ``(fp8 [..., L], fp32 scales [..., L/group])``.

    ``scale = max(amax/448, 1/(448*512))``, the same floor as rms_norm_quant.cu: an all-zero
    group must still get a finite scale or the whole row becomes NaN.
    """
    lead = tuple(v.shape[:-1])
    length = v.shape[-1]
    g = v.reshape(*lead, length // group, group)
    scale = torch.clamp(g.abs().amax(-1) / FP8_MAX, min=MIN_SCALE)
    q = torch.clamp(g / scale.unsqueeze(-1), -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.reshape(*lead, length), scale.to(torch.float32)


def dequant_groups(q, scale, group: int = GROUP):
    lead = tuple(q.shape[:-1])
    length = q.shape[-1]
    return (q.float().reshape(*lead, length // group, group)
            * scale.unsqueeze(-1)).reshape(*lead, length)


def reference_moe(x, w13_deq, w2_deq, topk_ids, topk_weights):
    """The kernel's contract in torch: quantize x, per (token, slot) do
    ``silu(w1.x) * (w3.x)``, requantize that to fp8 group-128, then ``w2.h``, then the
    routing-weighted sum. Loops on purpose -- it is an oracle, not a kernel."""
    m, k = x.shape
    n = w2_deq.shape[2]
    xq, xs = quant_groups(x.float())
    xd = dequant_groups(xq, xs)
    out = torch.zeros(m, k, dtype=torch.float32, device=x.device)
    for t in range(m):
        for s in range(topk_ids.shape[1]):
            e = int(topk_ids[t, s])
            if e < 0 or e >= w13_deq.shape[0]:
                continue
            gate = w13_deq[e, :n] @ xd[t]
            up = w13_deq[e, n:] @ xd[t]
            h = torch.nn.functional.silu(gate) * up
            hq, hs = quant_groups(h.unsqueeze(0))
            hd = dequant_groups(hq, hs).squeeze(0)
            out[t] += (w2_deq[e] @ hd) * float(topk_weights[t, s])
    return out.to(torch.bfloat16)


def unpack_int4_last_dim(packed):
    """int32 ``[..., L/8]`` -> int8 ``[..., L]``, the .cu's unpack expression in torch.

    ``((int)(word << (28 - 4*i))) >> 28`` in the kernel is a shift-left-then-arithmetic-shift-
    right sign extension of nibble i; here it is mask-then-subtract-16, which is the same map.
    Writing it a DIFFERENT way on purpose: two transcriptions of one expression agreeing is
    evidence, two copies of one transcription is not.
    """
    shifts = torch.arange(moe.PACK_FACTOR, device=packed.device, dtype=torch.int32) * 4
    nib = (packed.unsqueeze(-1) >> shifts) & 0xF
    q = torch.where(nib >= 8, nib - 16, nib)
    lead = tuple(packed.shape[:-1])
    return q.reshape(*lead, packed.shape[-1] * moe.PACK_FACTOR).to(torch.int8)


def _fp8_block_experts(seed: int = 0):
    """``(w13 fp8 [E,2N,K], w13_bs [E,2N/128,K/128], w2 fp8 [E,K,N], w2_bs [E,K/128,N/128])``
    plus the fp32 dequantized views, i.e. the checkpoint layout this patch claims."""
    torch.manual_seed(seed)
    made = []
    for shape in ((E, 2 * N, K), (E, K, N)):
        e, rows, cols = shape
        w = torch.randn(shape, device="cuda", dtype=torch.float32) * 0.05
        # Give the blocks genuinely different scales, or a transposed scale index would pass.
        w = w * (1.0 + 8.0 * torch.rand(e, rows // GROUP, 1, cols // GROUP, 1, device="cuda")
                 ).reshape(e, rows // GROUP, 1, cols // GROUP, 1).expand(
                     e, rows // GROUP, GROUP, cols // GROUP, GROUP
                 ).reshape(e, rows, cols)
        tiles = w.reshape(e, rows // GROUP, GROUP, cols // GROUP, GROUP)
        amax = tiles.abs().amax(dim=(2, 4))
        bs = (amax / FP8_MAX).clamp_min(1e-12)
        q = (tiles / bs[:, :, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        q = q.reshape(e, rows, cols).contiguous()
        deq = (q.float().reshape(e, rows // GROUP, GROUP, cols // GROUP, GROUP)
               * bs[:, :, None, :, None]).reshape(e, rows, cols).contiguous()
        made.append((q, bs.contiguous(), deq))
    (w13, w13_bs, w13_deq), (w2, w2_bs, w2_deq) = made
    return w13, w13_bs, w13_deq, w2, w2_bs, w2_deq


def _routing(m: int, seed: int = 1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(m)])
    w = torch.rand(m, TOPK, generator=g)
    w = w / w.sum(-1, keepdim=True)
    return ids.to("cuda", torch.int32).contiguous(), w.to("cuda", torch.float32).contiguous()


def _armed(kernels, weights, w13, w13_s, w2, w2_s, chunk: int = 1024):
    return moe._Armed(
        num_experts=E, k=K, n=N, topk=TOPK, weights=weights, threads=THREADS,
        max_m=10 ** 9, chunk=chunk, kernels=kernels,
        w13=w13, w13_scale=w13_s, w2=w2, w2_scale=w2_s,
    )


def _rel_and_cos(got, ref):
    g, r = got.float().flatten(), ref.float().flatten()
    rel = float((g - r).norm() / r.norm().clamp_min(1e-12))
    cos = float(torch.dot(g, r) / (g.norm() * r.norm()).clamp_min(1e-12))
    return rel, cos


# ---------------------------------------------------------------------------------------
# GPU tests.
# ---------------------------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("m", MS)
def test_fp8_arm_matches_the_dequantized_reference(m):
    """WEIGHT_FP8=1: the kernel reads the checkpoint tensors straight, so the reference is the
    block-dequantized weight. This arm is the one that ships first -- zero extra memory."""
    w13, w13_bs, w13_deq, w2, w2_bs, w2_deq = _fp8_block_experts()
    ids, wts = _routing(m)
    x = (torch.randn(m, K, device="cuda", dtype=torch.bfloat16) * 0.5)

    armed = _armed(_kernels("fp8"), "fp8", w13, w13_bs, w2, w2_bs)
    got = moe._run(armed, x, wts, ids)
    ref = reference_moe(x, w13_deq, w2_deq, ids, wts)

    rel, cos = _rel_and_cos(got, ref)
    assert rel < REL_TOL, f"m={m} rel={rel:.4g}"
    assert cos > COS_TOL, f"m={m} cos={cos:.6f}"


@requires_gpu
@pytest.mark.parametrize("m", MS)
def test_int4_arm_matches_its_own_dequantized_weights(m):
    """WEIGHT_FP8=0. The reference is built by UNPACKING the patch's own packed int32 and
    dequantizing it, so the comparison contains no quantization error at all: a wrong nibble
    order or a wrong group-scale row shows up as a gross mismatch, not as extra noise."""
    w13, w13_bs, _, w2, w2_bs, _ = _fp8_block_experts()
    ids, wts = _routing(m)
    x = (torch.randn(m, K, device="cuda", dtype=torch.bfloat16) * 0.5)

    w13_q, w13_gs = moe.requantize_experts(w13, w13_bs, 2)
    w2_q, w2_gs = moe.requantize_experts(w2, w2_bs, 2)
    assert w13_q.shape == (E, 2 * N, K // moe.PACK_FACTOR) and w13_q.dtype == torch.int32
    assert w13_gs.shape == (E, 2 * N, K // GROUP)
    assert w2_q.shape == (E, K, N // moe.PACK_FACTOR)
    assert w2_gs.shape == (E, K, N // GROUP)

    w13_deq = dequant_groups(unpack_int4_last_dim(w13_q), w13_gs)
    w2_deq = dequant_groups(unpack_int4_last_dim(w2_q), w2_gs)

    armed = _armed(_kernels("int4"), "int4", w13_q, w13_gs, w2_q, w2_gs)
    got = moe._run(armed, x, wts, ids)
    ref = reference_moe(x, w13_deq, w2_deq, ids, wts)

    rel, cos = _rel_and_cos(got, ref)
    assert rel < REL_TOL, f"m={m} rel={rel:.4g}"
    assert cos > COS_TOL, f"m={m} cos={cos:.6f}"


@requires_gpu
def test_pack_unpack_round_trips_through_the_real_packer():
    """``quantize_int4_group`` -> ``pack_int4_last_dim`` -> unpack must be the identity on the
    levels. This is the contract the .cu's unpack is written against."""
    torch.manual_seed(3)
    w = torch.randn(E, 2 * N, K, device="cuda") * 0.05
    q, s = moe.quantize_int4_group(w)
    assert int(q.min()) >= moe.INT4_MIN and int(q.max()) <= moe.INT4_MAX
    assert (q == moe.INT4_MIN).any() or (q == moe.INT4_MAX).any(), "no group reached a rail"
    packed = moe.pack_int4_last_dim(q)
    assert torch.equal(unpack_int4_last_dim(packed), q)
    # ...and the scale is the asymmetric-range formula, not absmax/7.
    g = w.reshape(E, 2 * N, K // GROUP, GROUP)
    want = torch.maximum(g.amax(-1).abs() / moe.INT4_MAX, g.amin(-1).abs() / -moe.INT4_MIN)
    assert torch.allclose(s, want, rtol=1e-6, atol=0)


@requires_gpu
def test_token_chunking_gives_the_same_answer_as_one_shot():
    """The FREE_FP8 arm sends whole prefills through this kernel in chunks. Chunking must be
    invisible: each chunk is an independent set of tokens, so any difference is a bug in the
    pointer arithmetic, not in the numerics."""
    w13, w13_bs, _, w2, w2_bs, _ = _fp8_block_experts()
    m = 13  # deliberately not a multiple of the chunk
    ids, wts = _routing(m)
    x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16) * 0.5
    kernels = _kernels("fp8")
    whole = moe._run(_armed(kernels, "fp8", w13, w13_bs, w2, w2_bs, chunk=1024), x, wts, ids)
    for chunk in (1, 4, 5):
        part = moe._run(_armed(kernels, "fp8", w13, w13_bs, w2, w2_bs, chunk=chunk), x, wts, ids)
        assert torch.equal(whole, part), f"chunk={chunk} changed the answer"


@requires_gpu
def test_a_masked_slot_contributes_nothing():
    """topk_ids may carry -1 for a masked slot. The kernel must treat it as a zero
    contribution -- not read expert -1, and not leave the intermediate uninitialized."""
    w13, w13_bs, w13_deq, w2, w2_bs, w2_deq = _fp8_block_experts()
    ids, wts = _routing(4)
    ids[1, 0] = -1
    ids[3, TOPK - 1] = -1
    x = torch.randn(4, K, device="cuda", dtype=torch.bfloat16) * 0.5
    got = moe._run(_armed(_kernels("fp8"), "fp8", w13, w13_bs, w2, w2_bs), x, wts, ids)
    ref = reference_moe(x, w13_deq, w2_deq, ids, wts)
    assert torch.isfinite(got.float()).all(), "a masked slot left NaNs behind"
    rel, _ = _rel_and_cos(got, ref)
    assert rel < REL_TOL, rel


@requires_gpu
def test_moe_int4_to_fp8_reexpansion():
    """The RESERVED fifth entry: int4 levels x group scale / block scale, stored as fp8, so
    that ``fp8_value * block_scale`` reproduces the int4-dequantized weight. Not wired into
    the patch yet (see the kernel header); tested so it cannot rot before it is."""
    w13, w13_bs, _, _, _, _ = _fp8_block_experts()
    w13_q, w13_gs = moe.requantize_experts(w13, w13_bs, 2)
    kernels = _kernels("int4", extra=("moe_int4_to_fp8",))
    rows, length = 2 * N, K
    dst = torch.empty((E, rows, length), dtype=torch.float8_e4m3fn, device="cuda")
    kernels["moe_int4_to_fp8"](
        grid=(E * rows, 1, 1), block=(THREADS, 1, 1),
        args=nvrtc.pack_args(dst.data_ptr(), w13_q.data_ptr(), w13_gs.data_ptr(),
                             w13_bs.data_ptr(), ("i", rows), ("i", length)),
    )
    torch.cuda.synchronize()
    got = (dst.float().reshape(E, rows // GROUP, GROUP, length // GROUP, GROUP)
           * w13_bs[:, :, None, :, None]).reshape(E, rows, length)
    ref = dequant_groups(unpack_int4_last_dim(w13_q), w13_gs)
    rel, cos = _rel_and_cos(got, ref)
    # One extra fp8 rounding on top of the int4 value, hence a looser bar than the GEMV arms.
    assert rel < 3e-2, rel
    assert cos > 0.999, cos


@requires_gpu
@pytest.mark.parametrize("m", (1, 6, 16))
def test_against_stock_fused_experts(m):
    """The stock oracle: vLLM's own Triton block-fp8 fused_moe on the same weights. Looser
    tolerance -- stock quantizes activations and tiles the reduction its own way, so this
    proves 'same answer', not 'same arithmetic'."""
    try:
        import vllm.model_executor.layers.fused_moe.fused_moe  # noqa: F401
        stock = torch.ops.vllm.fused_experts
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"stock fused_experts unavailable ({exc})")

    w13, w13_bs, w13_deq, w2, w2_bs, w2_deq = _fp8_block_experts()
    ids, wts = _routing(m)
    x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16) * 0.5
    try:
        ref = stock(
            x, w13, w2, wts, ids, activation="silu", use_fp8_w8a8=True,
            global_num_experts=E, w1_scale=w13_bs, w2_scale=w2_bs,
            block_shape=[GROUP, GROUP],
        )
    except Exception as exc:  # pragma: no cover -- API drift is a skip, not a failure
        pytest.skip(f"stock fused_experts rejected these shapes ({exc})")

    got = moe._run(_armed(_kernels("fp8"), "fp8", w13, w13_bs, w2, w2_bs), x, wts, ids)
    rel, cos = _rel_and_cos(got, ref)
    assert rel < STOCK_REL_TOL, f"m={m} rel={rel:.4g}"
    assert cos > 0.99, f"m={m} cos={cos:.6f}"


# ---------------------------------------------------------------------------------------
# Self-check: slot bookkeeping and cache-key distinctness, with fake tensors.
# ---------------------------------------------------------------------------------------


class _FakeTensor:
    """Just enough of a tensor for the shape-only helpers. Deliberately NOT a numpy array:
    the point is that ``validate_geometry`` reads nothing but ``.shape``, so a shape mismatch
    is caught before anything touches CUDA."""

    def __init__(self, *shape, dtype: str = "float8_e4m3fn") -> None:
        self.shape = tuple(shape)
        self.dtype = dtype

    def __repr__(self) -> str:  # pragma: no cover -- assertion messages only
        return f"_FakeTensor{self.shape}:{self.dtype}"


def _self_check() -> None:
    # -- the shipped model's real geometry, through fake tensors --
    e, k, n, topk = 256, 3072, 1024, 8
    w13 = _FakeTensor(e, 2 * n, k)
    w2 = _FakeTensor(e, k, n)
    w13_s = _FakeTensor(e, 2 * n // 128, k // 128, dtype="float32")
    w2_s = _FakeTensor(e, k // 128, n // 128, dtype="float32")
    assert moe.validate_geometry(w13.shape, w2.shape, w13_s.shape, w2_s.shape) == (e, k, n)
    # The layouts the checkpoint really has, spelled out so a config change is visible here.
    assert w13.shape == (256, 2048, 3072) and w2.shape == (256, 3072, 1024)
    assert w13_s.shape == (256, 16, 24) and w2_s.shape == (256, 24, 8)

    # A backend that shuffled the weights (DeepGEMM/Marlin/FlashInfer) must be declined, not
    # misread: the weights still look fine, the scale grid does not.
    shuffled = _FakeTensor(e, k // 128, 2 * n // 128, dtype="float32")
    assert isinstance(moe.validate_geometry(w13.shape, w2.shape, shuffled.shape, w2_s.shape), str)

    # -- SLOT BOOKKEEPING. The kernel gets one blockIdx.x per (token, slot) and recovers both
    #    by /TOPK and %TOPK. Drive that against a fake row-major [M, TOPK] topk_ids and check
    #    every slot reads the expert the host thinks it wrote. --
    m = 5
    fake_ids = [[(t * topk + s) % e for s in range(topk)] for t in range(m)]
    flat = [v for row in fake_ids for v in row]          # row-major, as the kernel sees it
    assert len(flat) == m * topk
    rows = set()
    for t in range(m):
        for s in range(topk):
            ts = moe.slot_row(t, s, topk)
            rows.add(ts)
            assert flat[ts] == fake_ids[t][s], (t, s, ts)
            assert ts // topk == t and ts % topk == s
    assert rows == set(range(m * topk)), "the (token, slot) map must be a bijection"
    # The transposed convention (slot-major) must NOT accidentally agree -- if it did, the
    # test above would prove nothing about which way round the kernel indexes.
    assert moe.slot_row(1, 0, topk) != (0 * m + 1)

    # Staging is private per (token, slot): that is what makes the reduction atomic-free and
    # bit-reproducible across runs.
    assert moe.staging_elems(m, topk, k) == m * topk * k
    assert moe.staging_elems(1, topk, k) * 4 == topk * k * 4

    # -- launch config agrees with the slot map: gridDim.x counts (token, slot) pairs --
    grids = moe.launch_grids(m, topk, k, n)
    assert grids["moe_gemv_up"][0] == m * topk == grids["moe_gemv_down"][0]
    assert grids["moe_finalize"][0] == m
    assert grids["moe_x_quant"][0] == m * (k // GROUP)

    # -- CACHE-KEY DISTINCTNESS across define sets. Two arms that share a cubin is the worst
    #    failure this cache can have: it is fast, it is silent, and it runs the wrong code. --
    src = nvrtc.load_source(moe.KERNEL)
    assert src, "vtl/kernels/moe_decode_gemv.cu missing from the package"
    arch, toolkit = "90a", "12.4"
    sets = {
        "int4": moe.kernel_defines(k, n, topk, "int4", 256),
        "fp8": moe.kernel_defines(k, n, topk, "fp8", 256),
        "int4/512t": moe.kernel_defines(k, n, topk, "int4", 512),
        "int4/topk4": moe.kernel_defines(k, n, 4, "int4", 256),
        "int4/k2048": moe.kernel_defines(2048, n, topk, "int4", 256),
        "int4/n512": moe.kernel_defines(k, 512, topk, "int4", 256),
    }
    keys = {name: nvrtc.cache_key(src, d, arch, toolkit) for name, d in sets.items()}
    assert len(set(keys.values())) == len(keys), keys
    for key in keys.values():
        assert len(key) == 32
    # Stable across calls and independent of dict order, or a warm cache misses half the time.
    reordered = dict(reversed(list(sets["int4"].items())))
    assert nvrtc.cache_key(src, reordered, arch, toolkit) == keys["int4"]
    # The axes nvrtc owns still bite.
    assert nvrtc.cache_key(src, sets["int4"], "80", toolkit) != keys["int4"]
    assert nvrtc.cache_key(src, sets["int4"], arch, "12.5") != keys["int4"]
    assert nvrtc.cache_key(src + "\n", sets["int4"], arch, toolkit) != keys["int4"], \
        "editing the .cu must produce a different cubin -- no stale-cubin trap"

    # -- the source declares every entry the patch launches, and the reserved one --
    for entry in moe.ENTRIES:
        assert f"void __launch_bounds__(THREADS) {entry}(" in src, entry
    assert "moe_int4_to_fp8" in src
    assert "moe_int4_to_fp8" not in moe.ENTRIES, "the reserved entry must not be launched"

    # -- the nibble convention, restated here and checked against the .cu's own text --
    assert "(28 - 4 * j)" in src or "(28 - 4 * i)" in src, \
        "the sign-extending unpack expression changed shape; re-derive unpack_int4_last_dim"
    assert moe.PACK_FACTOR == 8 and moe.GROUP == 128
    assert moe.INT4_MIN == -8 and moe.INT4_MAX == 7

    # -- and the patch's own knobs behave (a wrong default here silently ships the wrong arm) --
    assert moe.parse_weights(None) == "fp8"
    assert moe.parse_weights("int4") == "int4"
    assert moe.DEFAULT_MAX_M == 16
    assert moe.token_chunks(16, 64) == [(0, 16)]

    print("test_moe_decode self-check ok")


def _bench():  # pragma: no cover -- GPU-only, run by hand
    if SKIP is not None:
        print(f"test_moe_decode: skipped -- {SKIP}")
        return
    import statistics

    w13, w13_bs, _, w2, w2_bs, _ = _fp8_block_experts()
    w13_q, w13_gs = moe.requantize_experts(w13, w13_bs, 2)
    w2_q, w2_gs = moe.requantize_experts(w2, w2_bs, 2)
    arms = {
        "fp8": _armed(_kernels("fp8"), "fp8", w13, w13_bs, w2, w2_bs),
        "int4": _armed(_kernels("int4"), "int4", w13_q, w13_gs, w2_q, w2_gs),
    }
    print(f"device: {torch.cuda.get_device_name()}  E={E} K={K} N={N} TOPK={TOPK}")
    print(f"\n{'M':>4} {'arm':>6} {'us/call':>9}")

    def timed(fn):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(5):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(50):
                fn()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) / 50 * 1e3)
        return statistics.median(samples)

    for m in MS:
        ids, wts = _routing(m)
        x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16) * 0.5
        for name, armed in arms.items():
            us = timed(lambda a=armed: moe._run(a, x, wts, ids))
            print(f"{m:>4} {name:>6} {us:>9.1f}")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    elif os.environ.get("VTL_BENCH", "").strip() in ("1", "true", "yes", "on"):
        _bench()
    elif not HAVE_PYTEST:
        raise SystemExit("pytest not installed; run with --self-check")
    else:
        raise SystemExit(pytest.main([__file__, "-q"]))
