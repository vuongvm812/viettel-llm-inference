"""Feasibility gate for a persistent (cooperative-grid) decode megakernel.

WHY THIS EXISTS, AND WHY IT SHIPS BEFORE THE KERNEL. The proposal is a resident grid that stays
warm on the SMs and pulls work from a device-side ring buffer, replacing the per-step launch of
the short-conv block chain (rms_norm_quant -> in_proj -> conv+gate+quant -> out_proj). Every
design in that family needs a device-wide barrier between phases -- out_proj is a full GEMM over
D=2048 and cannot start until the whole conv output exists -- and a device-wide barrier means
``cooperative_groups::grid.sync()``, which means ``cudaLaunchCooperativeKernel``, which requires
**the entire grid to be co-resident on the device at once**.

That last requirement is the whole risk, and it is knowable before a line of the kernel is
written. On a MIG 1g.18gb slice (~16 SMs) the ceiling is

    max_grid_blocks = maxBlocksPerMultiprocessor(kernel) * sm_count

and if the kernel's register/smem footprint drives occupancy to 1 block/SM, the entire megakernel
gets **16 blocks, total**, for a model whose per-token work is 2048 channels wide. That is not a
tuning problem, it is a different kernel design -- and finding it out after two weeks of WGMMA
plumbing would be the expensive way to learn it.

So: this module answers the three questions that can kill the design, on the box, at boot, in
microseconds, and it does so today. It does NOT implement the megakernel.

    1. Does the device support cooperative launch at all?   (cudaDevAttrCooperativeLaunch)
    2. How many blocks could ever be co-resident?           (maxBlocksPerMultiProcessor x SMs)
    3. Is MPS in use?                                       (the one documented disqualifier --
       NVIDIA's CUDA Graphs guide permits cooperative launches "so long as MPS is not in use",
       and MIG + MPS is a supported combination that some serving stacks enable)

CUDA-GRAPH COMPATIBILITY IS NOT ONE OF THEM. It was the obvious worry -- this server runs
``cudagraph_mode: FULL_AND_PIECEWISE`` -- but cooperative launches are absent from the CUDA
Programming Guide's list of stream-capture prohibitions, and are capturable. MPS is the real
constraint, which is why it is probed and graph mode is not.

DEFAULT ON, deliberately, and for the same reason ``l2_persist``'s probe is: these are read-only
device attribute queries that change no state, and they exist to answer a question that gating
them off would guarantee never gets answered. Nothing here launches a kernel, allocates, or
touches the model. ``VTL_ENABLE_MEGAKERNEL_PROBE=0`` turns it off.

Reuses ``l2_persist``'s hardened cudart handle rather than opening a second one -- it already
prefers torch's own copy of the runtime over a bare soname (a different libcudart reports a
different device state) and already clears the sticky per-thread error that would otherwise make
torch's next launch raise.
"""

from __future__ import annotations

import ctypes
import logging
import os

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vllm.vtl.megakernel")

# cudaDeviceAttr values. Spelled out rather than imported: there is no Python binding for these
# in the runtime image, and the numbers are ABI-stable across CUDA 11/12/13.
# (cudaDeviceAttr, in enum order: ...81 MaxSharedMemoryPerMultiprocessor, 82
# MaxRegistersPerMultiprocessor, ... 95 CooperativeLaunch, ... 106 MaxBlocksPerMultiprocessor,
# 108 MaxPersistingL2CacheSize -- which l2_persist.py already uses, corroborating the numbering.)
_ATTR_MULTIPROCESSOR_COUNT = 16
_ATTR_MAX_SHMEM_PER_MP = 81
_ATTR_MAX_REGS_PER_MP = 82
_ATTR_COOPERATIVE_LAUNCH = 95
_ATTR_MAX_BLOCKS_PER_MP = 106

# The block shape the megakernel would use: 256 threads is what every existing vtl elementwise
# kernel launches at D=2048 (one 16-byte vector per thread), so it is the honest starting point
# for an occupancy estimate. Overridable so the number can be re-read for a different design
# without a rebuild.
DEFAULT_BLOCK_THREADS = 256


def _attr(lib, attr: int, device: int) -> int | None:
    """One cudaDeviceGetAttribute. None on any failure, with the sticky error cleared."""
    from vtl.patches.l2_persist import _err

    value = ctypes.c_int(0)
    rc = lib.cudaDeviceGetAttribute(ctypes.byref(value), ctypes.c_int(attr), ctypes.c_int(device))
    if rc != 0:
        log.debug("vtl: cudaDeviceGetAttribute(%d) failed: %s", attr, _err(lib, rc))
        return None
    return value.value


def _mps_in_use() -> bool:
    """MPS is the one documented disqualifier for cooperative launch.

    Detected from the environment rather than a CUDA API because there is no runtime query for
    "am I a client of an MPS server": the control daemon advertises itself through
    ``CUDA_MPS_PIPE_DIRECTORY`` (and the client honours ``CUDA_MPS_ACTIVE_THREAD_PERCENTAGE``),
    and those are what a stack that turns MPS on actually sets.
    """
    return any(
        os.environ.get(name)
        for name in ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
    )


def probe(device: int = 0, block_threads: int = DEFAULT_BLOCK_THREADS) -> dict | None:
    """Everything needed to decide go/no-go on a cooperative megakernel. None if unreadable.

    ``blocks_per_mp`` here is the DEVICE ceiling, not this-kernel's occupancy: the real number
    comes from cudaOccupancyMaxActiveBlocksPerMultiprocessor once a kernel exists, and can only
    be lower. So a no-go from this probe is conclusive; a go is permission to write the kernel
    and measure again, not a guarantee.
    """
    from vtl.patches.l2_persist import _load_cudart

    lib = _load_cudart()
    if lib is None:
        return None

    sm_count = _attr(lib, _ATTR_MULTIPROCESSOR_COUNT, device)
    blocks_per_mp = _attr(lib, _ATTR_MAX_BLOCKS_PER_MP, device)
    coop = _attr(lib, _ATTR_COOPERATIVE_LAUNCH, device)
    if sm_count is None or coop is None:
        return None
    # Pre-Volta reports no such attribute; treat a missing ceiling as the conservative 1.
    blocks_per_mp = blocks_per_mp or 1

    max_grid = sm_count * blocks_per_mp
    return {
        "sm_count": sm_count,
        "blocks_per_mp": blocks_per_mp,
        "max_grid_blocks": max_grid,
        "block_threads": block_threads,
        "max_threads": max_grid * block_threads,
        "cooperative_launch": bool(coop),
        "mps": _mps_in_use(),
        "smem_per_mp": _attr(lib, _ATTR_MAX_SHMEM_PER_MP, device),
        "regs_per_mp": _attr(lib, _ATTR_MAX_REGS_PER_MP, device),
    }


def verdict(info: dict, hidden_size: int = 2048) -> tuple[bool, str]:
    """(feasible, why). The threshold is one thread per hidden channel.

    LFM2.5's short-conv block is 2048 channels wide and every existing vtl kernel in that chain
    assigns one thread per channel (vectorized 8-wide, so 256 threads cover it). If a co-resident
    grid cannot field 2048 threads, the megakernel would have to serialize channels inside the
    resident blocks -- at which point it is strictly worse than the four separate launches it
    replaces, because those get the whole device per phase.
    """
    if not info["cooperative_launch"]:
        return False, "device reports no cooperative-launch support"
    if info["mps"]:
        return False, "MPS is in use; CUDA documents cooperative launches as unsupported under it"
    if info["max_threads"] < hidden_size:
        return False, (
            f"a co-resident grid tops out at {info['max_grid_blocks']} blocks x "
            f"{info['block_threads']} threads = {info['max_threads']}, below the {hidden_size} "
            "channels one token needs; grid.sync would serialize what four launches parallelize"
        )
    return True, (
        f"{info['max_grid_blocks']} blocks co-resident "
        f"({info['blocks_per_mp']}/SM x {info['sm_count']} SMs)"
    )


def _log_probe() -> None:
    info = probe()
    if info is None:
        log.info("vtl: megakernel probe skipped (no readable CUDA runtime)")
        return
    ok, why = verdict(info)
    log.info(
        "vtl: megakernel feasibility = %s -- %s (cooperative=%s, mps=%s, %d SMs, "
        "%d blocks/SM, smem/SM=%s, regs/SM=%s)",
        "GO" if ok else "NO-GO", why, info["cooperative_launch"], info["mps"],
        info["sm_count"], info["blocks_per_mp"], info["smem_per_mp"], info["regs_per_mp"],
    )


@register_patch("megakernel_probe", default=True)
def apply() -> None:
    """Log the verdict once, after the model is loaded (so CUDA is up and the device is real).

    Hooks the same ``BaseModelLoader.load_model`` seam as ``l2_persist`` and ``quant_w4a8``, and
    passes ``patch=`` for the same reason: the unnamed already-patched check would make whichever
    of the three applies last skip itself silently.
    """
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader

    if already_patched(BaseModelLoader, "load_model", patch="megakernel_probe"):
        return
    original = BaseModelLoader.load_model

    def load_model(self, *args, **kwargs):
        model = original(self, *args, **kwargs)
        try:
            _log_probe()
        except Exception as exc:  # a diagnostic may never cost a boot
            log.warning("vtl: megakernel probe failed (%s)", exc)
        return model

    mark_patched(load_model, original, patch="megakernel_probe")
    BaseModelLoader.load_model = load_model


def _self_check() -> None:
    # verdict() is the whole decision, and it is pure -- so it is checkable off-box, which is the
    # only place this can be checked at all right now.
    base = {
        "sm_count": 16, "blocks_per_mp": 2, "max_grid_blocks": 32, "block_threads": 256,
        "max_threads": 8192, "cooperative_launch": True, "mps": False,
        "smem_per_mp": 232448, "regs_per_mp": 65536,
    }
    ok, why = verdict(base)
    assert ok, why

    # The three kill conditions, each on its own.
    assert not verdict({**base, "cooperative_launch": False})[0]
    assert not verdict({**base, "mps": True})[0]
    # 1 block/SM on 16 SMs at 256 threads = 4096 threads: still above 2048, so feasible...
    assert verdict({**base, "blocks_per_mp": 1, "max_grid_blocks": 16, "max_threads": 4096})[0]
    # ...but half that is not, and this is the case the whole probe exists to catch.
    tight = {**base, "blocks_per_mp": 1, "max_grid_blocks": 8, "max_threads": 2048}
    assert not verdict(tight, hidden_size=4096)[0]
    assert "serialize" in verdict(tight, hidden_size=4096)[1]

    # MPS detection reads the env the daemon actually sets.
    assert not _mps_in_use()
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/nvidia-mps"
    try:
        assert _mps_in_use()
    finally:
        del os.environ["CUDA_MPS_PIPE_DIRECTORY"]

    print("megakernel_probe self-check ok")


if __name__ == "__main__":
    _self_check()
