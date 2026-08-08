"""Arm ``vllm_cuda::shortconv_decode_mega`` -- the fused short-conv decode block.

The kernel is ``vtl/csrc/shortconv_decode_mega.cu``; the CALL SITE is the fork's
``ShortConv.forward_cuda`` decode branch (``vllm_patches/v0.25.0/short_conv.patch``). This
module owns the three things the kernel cannot own itself:

1. **The persistent scratch.** A FULL cudagraph bakes pointers at capture, so the kernel's
   inter-phase buffers (x_fp8, x_scale, bxc, y_fp8, y_scale, and the barrier words) must have
   one address for the life of the process. Allocated once, on first arming, sized by the
   kernel's own ``shortconv_decode_mega_scratch_bytes``.

2. **A plain view of the W4A8 weights.** ``quant_w4a8`` stores ``weight_packed`` after
   ``ops.cutlass_encode_and_reorder_int4b`` and ``weight_group_scale`` after
   ``ops.cutlass_pack_scale_fp8`` -- both are CUTLASS's internal mixed-input tile orders and
   are NOT addressable from hand-written code. THIS IS THE PLAN'S ONE REAL SURPRISE: the
   megakernel cannot read the shipped tensors at all. So we re-derive, once per layer at load
   time, a layout the GEMV can stream: ``uint8[N, K/2]`` two-nibble packed int4 plus
   ``float32[N, K/128]`` group scales with the per-channel residual folded in. Cost is a
   second copy of the conv projections' weights (~84 MB across LFM2.5's 10 layers) and one
   re-quantization per layer at boot. Not bit-exact against CUTLASS (different accumulation
   order inside the MMA); ``bench/test_w4a8_gemv.py`` checks it to a tolerance.

3. **The co-residency gate.** The kernel's device-wide barrier HANGS THE DEVICE if any block
   is not resident, so the grid is not a tuning constant: it is
   ``cudaOccupancyMaxActiveBlocksPerMultiprocessor`` on the real kernel (registers and smem
   included) times the SM count, queried through ``shortconv_decode_mega_grid``. The boot
   probe in ``megakernel_probe`` answers the DEVICE ceiling; this answers THIS KERNEL's, which
   can only be lower. Below the kernel's own floor it returns 0 and we stay disarmed --
   the fork's ``getattr`` then keeps the stock four-op chain.

Gates: ``VTL_ENABLE_SHORTCONV_MEGA=0`` skips arming entirely (registry); ``VTL_SHORTCONV_MEGA=0``
is the in-fork kill-switch. Either is a complete revert to the existing chain.

ON-BOX VALIDATION RECIPE (nothing here blocks on it; run it whenever)::

    make vllm-fork PUSH=1        # rebuilds the fork with mamba_align_precopy + short_conv
    # re-pin VLLM_FORK_TAG in the Makefile to the new digest
    make build                   # compiles the two new .cu into vtl._C
    make test-kernel             # bench/test_{w4a8_gemv,shortconv_decode_mega}.py + the rest
    # then, in the container, the two artifacts that size the prize:
    python /bench/test_shortconv_decode_mega.py --bench     # chain vs mega, M = 1/2/4/8
    python /bench/test_w4a8_gemv.py --bench                 # what CUTLASS costs at those M
    # quality + latency, 3 boots per arm, flipping ONLY VTL_SHORTCONV_MEGA:
    make eval && python bench/replay.py ... && python bench/ab_summary.py

Read ``-Xptxas -v`` in the build log for ``shortconv_decode_mega_kernel``: a register spill
that drops it below 2 blocks/SM does not break anything (``_grid`` re-measures and the gate
refuses), but it silently disables the feature, and the log is where that shows up.
"""

from __future__ import annotations

import logging
import os

from vtl.registry import register_patch

log = logging.getLogger("vllm.vtl.shortconv_mega")

_OP = "shortconv_decode_mega"
_GROUP = 128  # must match quant_w4a8.GROUP_SIZE and w4a8_gemv.cuh's kW4A8Group
_MAX_M = 8    # must match shortconv_decode_mega.cu's kMegaMaxM

_fake_registered = False

#: Filled by ``arm()``: {"scratch": Tensor, "grid": int} or None while disarmed. The fork
#: reads it through ``mega_state()``; a None here is the fail-closed answer.
_STATE: dict | None = None


def mega_state() -> dict | None:
    """What the fork's decode branch needs to fire the op. None = not armed, use the chain."""
    return _STATE


def enabled() -> bool:
    return os.environ.get("VTL_SHORTCONV_MEGA", "1").strip().lower() not in (
        "0", "false", "no", "off", "",
    )


def pack_int4_nibbles(q):
    """int8 ``[N, K]`` in [-8, 7] -> uint8 ``[N, K/2]``: k(2j) low nibble, k(2j+1) high.

    Deliberately NOT ``quant_w4a8.pack_int4_rows``: that one packs 8 values per int32 with a
    strided row order chosen for CUTLASS. This is the layout ``w4a8_gemv.cuh`` reads, and the
    two must never be confused -- a mismatch is silently wrong weights, not an error.
    """
    import torch

    lo = (q[:, 0::2].to(torch.int16) & 0xF).to(torch.uint8)
    hi = (q[:, 1::2].to(torch.int16) & 0xF).to(torch.uint8)
    return lo | (hi << 4)


def build_mega_weights(weight):
    """bf16 ``[N, K]`` -> ``(uint8[N, K/2], float32[N, K/128])`` for the megakernel GEMV.

    Same quantizer as ``quant_w4a8._quantize_and_pack`` (``quantize_weights`` with
    ``scalar_types.int4``, group 128, no zero points, fp32 math) so the megakernel and the
    CUTLASS path see the SAME numbers -- only the storage order differs. The per-channel
    residual is folded into the group scale here rather than applied in an epilogue, because
    the GEMV's epilogue is already one multiply by the activation scale.
    """
    import torch
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        GroupShape,
        quantize_weights,
    )
    from vllm.scalar_type import scalar_types

    assert GroupShape is not None  # imported for parity with quant_w4a8's import set
    w_kn = weight.t().contiguous().float()  # [N, K] -> [K, N], grouped along dim 0
    _, w_q, w_s, _ = quantize_weights(
        w_kn, scalar_types.int4, group_size=_GROUP, zero_points=False
    )
    del w_kn
    # w_q is [K, N] int8 in [-8, 7]; w_s is [K/128, N] float.
    wq = w_q.t().contiguous().to(torch.int8)          # [N, K]
    gs = w_s.t().contiguous().to(torch.float32)       # [N, K/128]
    return pack_int4_nibbles(wq), gs


def attach_mega_weights(layer) -> None:  # noqa: ANN001
    """Stash the megakernel's plain int4 view on ``layer``, from its still-live bf16 weight.

    Called from ``quant_w4a8.process_weights_after_loading`` at the last moment the bf16
    weight exists. Only the short-conv projections are worth the extra copy, and only when
    the megakernel is actually enabled -- everything else would be ~2 GB of dead tensors.
    """
    if not enabled():
        return
    prefix = getattr(layer, "prefix", "") or ""
    if not (prefix.endswith(".in_proj") or prefix.endswith(".out_proj")):
        return
    if "short_conv" not in prefix:
        return
    wq, gs = build_mega_weights(layer.weight.data)
    layer._vtl_mega_wq = wq
    layer._vtl_mega_gs = gs
    log.info("vtl: mega weight view attached for %s (%s -> %s)", prefix,
             tuple(layer.weight.data.shape), tuple(wq.shape))


def _register_fake() -> None:
    global _fake_registered
    if _fake_registered:
        return
    import torch

    @torch.library.register_fake(f"vllm_cuda::{_OP}")
    def _fake(out, x, norm_weight, epsilon, in_wq, in_gs, conv_weight, conv_bias, conv_state,
              state_indices, null_block_id, out_wq, out_gs, scratch, num_blocks):  # noqa: ANN001
        return None

    _fake_registered = True


def arm(dim: int, device) -> dict | None:  # noqa: ANN001
    """Allocate the scratch and settle the grid. Idempotent; None = stay on the stock chain."""
    global _STATE
    if _STATE is not None:
        return _STATE
    import torch

    grid = int(torch.ops.vllm_cuda.shortconv_decode_mega_grid(dim))
    if grid <= 0:
        log.info(
            "vtl: shortconv_mega DISARMED -- this build's occupancy leaves too few "
            "co-resident blocks for the device barrier (dim=%d)", dim,
        )
        return None
    nbytes = int(torch.ops.vllm_cuda.shortconv_decode_mega_scratch_bytes(dim))
    _STATE = {
        "scratch": torch.zeros(nbytes, dtype=torch.uint8, device=device),
        "grid": grid,
        "dim": dim,
    }
    log.info("vtl: shortconv_mega armed -- grid=%d blocks, scratch=%d B, dim=%d",
             grid, nbytes, dim)
    return _STATE


@register_patch("shortconv_mega", default=True)
def apply() -> None:
    import torch

    import vllm._C_stable_libtorch  # noqa: F401  -- keep import order uniform
    import vtl._C  # noqa: F401  -- dlopen registers the CUDA kernel + the op schema

    if not enabled():
        log.info("vtl: shortconv_mega off (VTL_SHORTCONV_MEGA=0)")
        return
    if not hasattr(torch.ops.vllm_cuda, _OP):
        log.warning("vtl: %s op not registered after importing vtl._C; disabled", _OP)
        return
    _register_fake()
    # Scratch + grid are settled lazily, on the first decode step that reaches the fork's
    # branch: `dim` is a property of the layer, and CUDA occupancy must be queried on a live
    # context, not at plugin-import time.
    log.info("vtl: shortconv_mega op registered (arms lazily on the first decode step)")


def _self_check() -> None:
    """No torch: the nibble packing and the layout contracts are what can silently rot."""
    assert _OP == "shortconv_decode_mega" and _GROUP == 128 and _MAX_M == 8

    from vtl.registry import PATCH_REGISTRY

    assert {p.name: p.default for p in PATCH_REGISTRY}.get("shortconv_mega") is True

    # Pure-Python model of pack_int4_nibbles over the full int4 range, including the two
    # cases that a naive `& 0xF` on a signed tensor gets wrong (-8 and -1).
    def ref_pack(row):
        return [(row[2 * j] & 0xF) | ((row[2 * j + 1] & 0xF) << 4) for j in range(len(row) // 2)]

    row = [-8, 7, -1, 0, 3, -5, 6, -2]
    packed = ref_pack(row)
    assert packed == [0x78, 0x0F, 0xB3, 0xE6], [hex(p) for p in packed]

    def unpack(b):
        lo, hi = b & 0xF, (b >> 4) & 0xF
        return (lo - 16 if lo > 7 else lo, hi - 16 if hi > 7 else hi)

    out = [v for b in packed for v in unpack(b)]
    assert out == row, out

    # The kernel and the fork patch must agree on the op names and the kill-switch.
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    cu = (root / "csrc" / "shortconv_decode_mega.cu").read_text(encoding="utf-8")
    assert f"void {_OP}(" in cu
    assert "constexpr int kMegaMaxM = 8;" in cu, "kMegaMaxM drifted from _MAX_M"
    assert "constexpr int kW4A8Group = 128;" in (
        root / "csrc" / "w4a8_gemv.cuh"
    ).read_text(encoding="utf-8"), "kW4A8Group drifted from _GROUP"

    patch = (root / "vllm_patches" / "v0.25.0" / "short_conv.patch").read_text(encoding="utf-8")
    assert f"torch.ops.vllm_cuda.{_OP}" in patch, "fork patch does not call the mega op"
    assert "VTL_SHORTCONV_MEGA" in patch, "fork patch lost its kill-switch"

    # `enabled()` is the kill-switch the fork mirrors; both directions must work.
    saved = os.environ.get("VTL_SHORTCONV_MEGA")
    try:
        os.environ.pop("VTL_SHORTCONV_MEGA", None)
        assert enabled() is True
        os.environ["VTL_SHORTCONV_MEGA"] = "0"
        assert enabled() is False
    finally:
        if saved is None:
            os.environ.pop("VTL_SHORTCONV_MEGA", None)
        else:
            os.environ["VTL_SHORTCONV_MEGA"] = saved

    assert mega_state() is None, "must start disarmed"
    print("shortconv_mega self-check ok")


if __name__ == "__main__":
    _self_check()
