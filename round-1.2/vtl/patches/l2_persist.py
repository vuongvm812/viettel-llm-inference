"""L2 persisting-cache probe, plus an optional access-policy window over the hottest weights.

Two things live here, and only the first one is on by default:

1. **Probe (always on, zero risk).** CUDA's docs say the L2 set-aside for *persisting*
   accesses is **disabled in MIG mode**, and the judge box is an H200 MIG ``1g.18gb``
   slice. That is a documentation claim about someone else's hardware, so we read the
   two numbers off the actual device at boot instead of believing it:
   ``cudaDeviceGetAttribute(cudaDevAttrMaxPersistingL2CacheSize)`` and
   ``cudaDeviceGetLimit(cudaLimitPersistingL2CacheSize)``. Expected ``0`` on the slice.
   Nonzero means the docs (or our assumption that it is really MIG) are wrong and item 2
   is worth an A/B.

2. **Window (``VTL_L2_PERSIST=1``, default off).** Only useful if the probe reports
   nonzero. Sets ``cudaLimitPersistingL2CacheSize`` and attaches a
   ``cudaAccessPolicyWindow`` to the decode stream over the largest int4-packed weight
   buffer, so its lines are marked *persisting* in L2 and everything else *streaming*.
   On LFM2 the candidates are the per-layer MLP ``w2`` (~8 MB packed) and ``w13``
   (~16 MB packed).

Both run from one hook: a wrapper around ``BaseModelLoader.load_model``. That seam is
already the vtl convention for "after every weight exists, still outside the compiled
graph" (see ``quant_w4a8._install_load_summary``), it runs exactly once and only in the
worker process -- so the probe never spins up a CUDA context in the API-server process
just to log a number -- and by the time it returns ``weight_packed`` is populated on
every quantized layer.

Reading the probe with the shipped compose: ``VLLM_LOGGING_LEVEL`` is ``WARNING``, which
filters our INFO line (vLLM's handler is level-gated too, so raising only the logger's
level would not help). Either boot once with ``VLLM_LOGGING_LEVEL=INFO``, or just ask
the container directly -- the probe is the module's ``__main__``::

    docker compose exec model python3 -m vtl.patches.l2_persist

Env:
  VTL_ENABLE_L2_PERSIST=1   registry gate for the probe (default on)
  VTL_L2_PERSIST=0          1 = also set the limit + window (default off)
  VTL_L2_PERSIST_BYTES      set-aside request in bytes, clamped to the device max (24 MiB)
  VTL_L2_HIT_RATIO          fraction of window lines to keep persisting (0.6)

**Limitations** (documented, not bugs):
* CUDA allows **one** access-policy window per stream and the window is a single
  contiguous ``(base_ptr, num_bytes)`` range. The per-layer packed weights are separate
  allocations, so their union is not a range -- we cover the single largest buffer only.
  Covering more would mean an arena allocator for weights, which is a much bigger change
  than the expected payoff on a slice where the feature is probably disabled outright.
* The window is set on the stream that is current at load time (the default stream that
  eager decode runs on). Whether a stream access policy applies to kernels replayed from
  a captured CUDA graph is not documented; the stack runs FULL_AND_PIECEWISE cudagraphs,
  so treat any measured win as needing the usual 3-boot A/B before it is believed.
"""

from __future__ import annotations

import ctypes
import glob
import logging
import os

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vllm.vtl.l2")

# --- CUDA runtime constants (cuda 12 driver_types.h / cuda_runtime_api.h) ----------------
CUDA_SUCCESS = 0
# enum cudaDeviceAttr
DEV_ATTR_MAX_PERSISTING_L2_CACHE_SIZE = 108
DEV_ATTR_MAX_ACCESS_POLICY_WINDOW_SIZE = 109
# enum cudaLimit
LIMIT_PERSISTING_L2_CACHE_SIZE = 0x05
# enum cudaLaunchAttributeID (== cudaStreamAttrID in cuda 12)
STREAM_ATTR_ACCESS_POLICY_WINDOW = 1
# enum cudaAccessProperty
ACCESS_PROPERTY_NORMAL = 0
ACCESS_PROPERTY_STREAMING = 1
ACCESS_PROPERTY_PERSISTING = 2

DEFAULT_PERSIST_BYTES = 24 * 1024 * 1024
DEFAULT_HIT_RATIO = 0.6


class AccessPolicyWindow(ctypes.Structure):
    """``struct cudaAccessPolicyWindow`` (driver_types.h).

    ::

        void                 *base_ptr;   // 8B, align 8
        size_t                num_bytes;  // 8B
        float                 hitRatio;   // 4B
        enum cudaAccessProperty hitProp;  // 4B (enum == int)
        enum cudaAccessProperty missProp; // 4B  -> 28B used, tail-padded to 32
    """

    _fields_ = [
        ("base_ptr", ctypes.c_void_p),
        ("num_bytes", ctypes.c_size_t),
        ("hitRatio", ctypes.c_float),
        ("hitProp", ctypes.c_int),
        ("missProp", ctypes.c_int),
    ]


class StreamAttrValue(ctypes.Union):
    """``union cudaStreamAttrValue`` -- in CUDA 12 a typedef of ``cudaLaunchAttributeValue``,
    whose first member is ``char pad[64]``. Carrying the pad makes the object the right
    size under both CUDA 11 and 12 layouts; ``accessPolicyWindow`` is at offset 0 either way.
    """

    _fields_ = [
        ("pad", ctypes.c_char * 64),
        ("accessPolicyWindow", AccessPolicyWindow),
    ]


# --- env ---------------------------------------------------------------------------------
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


def env_num(name: str, default, cast):
    """``cast(os.environ[name])``, falling back to ``default`` on absent/garbage/non-positive."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = cast(raw)
    except ValueError:
        log.warning("vtl: %s=%r is not a number; using %s", name, raw, default)
        return default
    if value <= 0:
        log.warning("vtl: %s=%r must be > 0; using %s", name, raw, default)
        return default
    return value


# --- libcudart ---------------------------------------------------------------------------
_cudart = None
_cudart_tried = False


def _load_cudart():
    """Best-effort handle on the CUDA runtime. ``None`` off-box (dev Mac, CPU CI)."""
    global _cudart, _cudart_tried
    if _cudart_tried:
        return _cudart
    _cudart_tried = True

    # torch's OWN copy first (torch/lib, or the nvidia-cuda-runtime-cu12 wheel). It is the
    # runtime this process is already running on, and it is what the bare sonames below can get
    # wrong: `libcudart.so.13` is tried first by version order but torch here bundles 12.x, so
    # the generic name can hand us a second, unrelated runtime whose device state is not the one
    # the model was loaded into. torch may also have dlopen'd its copy RTLD_LOCAL, which is why
    # the soname fallbacks stay -- they just go last.
    names: list[str] = []
    try:
        import torch

        root = os.path.dirname(os.path.dirname(os.path.abspath(torch.__file__)))
        for pattern in ("torch/lib/libcudart.so*", "nvidia/*/lib/libcudart.so*"):
            names.extend(sorted(glob.glob(os.path.join(root, pattern))))
    except Exception:
        pass
    names += ["libcudart.so.13", "libcudart.so.12", "libcudart.so"]

    for name in names:
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        lib.cudaGetErrorString.restype = ctypes.c_char_p
        _cudart = lib
        return _cudart
    log.info("vtl: libcudart not loadable here; L2 probe skipped")
    return None


def _err(lib, rc: int) -> str:
    # A failed cudart call sets the sticky per-thread last-error; torch's next kernel
    # launch would inherit it and raise (boot-killing). Every failure path formats its
    # message here, so clear it here.
    try:
        lib.cudaGetLastError()
    except Exception:
        pass
    try:
        return lib.cudaGetErrorString(ctypes.c_int(rc)).decode()
    except Exception:
        return f"cudaError {rc}"


def probe(device: int = 0) -> tuple[int, int] | None:
    """``(max_persisting_l2_bytes, current_limit_bytes)``, or ``None`` if unreadable."""
    lib = _load_cudart()
    if lib is None:
        return None
    try:
        attr = ctypes.c_int(0)
        rc = lib.cudaDeviceGetAttribute(
            ctypes.byref(attr),
            ctypes.c_int(DEV_ATTR_MAX_PERSISTING_L2_CACHE_SIZE),
            ctypes.c_int(device),
        )
        if rc != CUDA_SUCCESS:
            log.info("vtl: cudaDeviceGetAttribute(maxPersistingL2) failed: %s", _err(lib, rc))
            return None
        limit = ctypes.c_size_t(0)
        rc = lib.cudaDeviceGetLimit(
            ctypes.byref(limit), ctypes.c_int(LIMIT_PERSISTING_L2_CACHE_SIZE)
        )
        if rc != CUDA_SUCCESS:
            log.info("vtl: cudaDeviceGetLimit(persistingL2) failed: %s", _err(lib, rc))
            return None
        return attr.value, limit.value
    except Exception:
        log.exception("vtl: L2 probe raised")
        return None


def _max_window_bytes(lib, device: int) -> int:
    attr = ctypes.c_int(0)
    rc = lib.cudaDeviceGetAttribute(
        ctypes.byref(attr),
        ctypes.c_int(DEV_ATTR_MAX_ACCESS_POLICY_WINDOW_SIZE),
        ctypes.c_int(device),
    )
    return attr.value if rc == CUDA_SUCCESS else 0


def largest_packed_weight(model):
    """``(name, data_ptr, nbytes)`` of the biggest ``weight_packed`` buffer, or ``None``.

    ``weight_packed`` is what ``quant_w4a8``'s ``process_weights_after_loading`` registers
    (LFM2: ``w13`` ~16 MB and ``w2`` ~8 MB dominate). One window per stream, so the union
    of all of them is not expressible -- take the single largest.
    """
    best = None
    for name, module in model.named_modules():
        packed = getattr(module, "weight_packed", None)
        if packed is None:
            continue
        nbytes = packed.numel() * packed.element_size()
        if best is None or nbytes > best[2]:
            best = (name, packed.data_ptr(), nbytes)
    return best


def _install_window(model, device: int = 0) -> None:
    """Set the L2 set-aside limit and pin the largest packed weight on the current stream."""
    import torch

    lib = _load_cudart()
    if lib is None:
        return

    probed = probe(device)
    if not probed or probed[0] <= 0:
        log.warning(
            "vtl: VTL_L2_PERSIST=1 but maxPersistingL2CacheSize=%s -- the device has no "
            "persisting set-aside (expected under MIG); leaving L2 alone",
            None if probed is None else probed[0],
        )
        return
    max_persist = probed[0]

    want = int(env_num("VTL_L2_PERSIST_BYTES", DEFAULT_PERSIST_BYTES, int))
    set_aside = min(max_persist, want)
    rc = lib.cudaDeviceSetLimit(
        ctypes.c_int(LIMIT_PERSISTING_L2_CACHE_SIZE), ctypes.c_size_t(set_aside)
    )
    if rc != CUDA_SUCCESS:
        log.warning("vtl: cudaDeviceSetLimit(persistingL2, %d) failed: %s", set_aside, _err(lib, rc))
        return

    target = largest_packed_weight(model)
    if target is None:
        # Roll the set-aside back, same reason as the cudaStreamSetAttribute failure below:
        # L2 reserved for persisting lines we then never mark is L2 taken away from everything
        # else. Leaving it set here would make "no int4 layers" strictly worse than not running.
        lib.cudaDeviceSetLimit(ctypes.c_int(LIMIT_PERSISTING_L2_CACHE_SIZE), ctypes.c_size_t(0))
        lib.cudaGetLastError()  # rc unchecked; don't let a failed rollback poison torch
        log.warning("vtl: no weight_packed buffers found; L2 window not attached, "
                    "set-aside rolled back to 0")
        return
    name, base_ptr, nbytes = target

    # The window may not exceed the device's max access-policy window, and pinning more
    # than the set-aside itself is pointless.
    window_bytes = min(nbytes, set_aside)
    hw_max = _max_window_bytes(lib, device)
    if hw_max > 0:
        window_bytes = min(window_bytes, hw_max)

    value = StreamAttrValue()
    ctypes.memset(ctypes.byref(value), 0, ctypes.sizeof(value))
    value.accessPolicyWindow = AccessPolicyWindow(
        base_ptr=ctypes.c_void_p(base_ptr),
        num_bytes=ctypes.c_size_t(window_bytes),
        hitRatio=ctypes.c_float(float(env_num("VTL_L2_HIT_RATIO", DEFAULT_HIT_RATIO, float))),
        hitProp=ACCESS_PROPERTY_PERSISTING,
        missProp=ACCESS_PROPERTY_STREAMING,
    )
    stream = torch.cuda.current_stream().cuda_stream
    rc = lib.cudaStreamSetAttribute(
        ctypes.c_void_p(stream),
        ctypes.c_int(STREAM_ATTR_ACCESS_POLICY_WINDOW),
        ctypes.byref(value),
    )
    if rc != CUDA_SUCCESS:
        # Undo the set-aside: reserving L2 we then never mark persisting is pure loss.
        lib.cudaDeviceSetLimit(ctypes.c_int(LIMIT_PERSISTING_L2_CACHE_SIZE), ctypes.c_size_t(0))
        log.warning("vtl: cudaStreamSetAttribute(accessPolicyWindow) failed: %s", _err(lib, rc))
        return
    log.info(
        "vtl: L2 persisting window on %s (%.1f MiB of %.1f MiB buffer), set-aside %.1f MiB "
        "of %.1f MiB max, hitRatio=%s",
        name,
        window_bytes / 1024**2,
        nbytes / 1024**2,
        set_aside / 1024**2,
        max_persist / 1024**2,
        env_num("VTL_L2_HIT_RATIO", DEFAULT_HIT_RATIO, float),
    )


@register_patch("l2_persist", default=True)
def apply() -> None:
    """Hook ``BaseModelLoader.load_model`` to probe (always) and pin (opt-in) after load.

    ``patch="l2_persist"``, not the bare ``already_patched(BaseModelLoader, "load_model")``:
    ``quant_w4a8`` wraps this same attribute, and the unnamed check is per-attribute, so it
    would see w4a8's wrapper and silently skip us (or vice versa, depending on the order of
    ``_MODULES``). The named check asks about US. The two wrappers stack fine.
    """
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader

    if already_patched(BaseModelLoader, "load_model", patch="l2_persist"):
        return

    original = BaseModelLoader.load_model

    def load_model(self, *args, **kwargs):
        model = original(self, *args, **kwargs)
        try:
            probed = probe()
            if probed is not None:
                log.info(
                    "vtl: L2 maxPersistingL2CacheSize=%d B, cudaLimitPersistingL2CacheSize=%d B "
                    "(0 = set-aside unavailable, expected under MIG)",
                    probed[0],
                    probed[1],
                )
            if env_flag("VTL_L2_PERSIST"):
                _install_window(model)
        except Exception:
            log.exception("vtl: L2 persist hook failed; serving unaffected")
        return model

    BaseModelLoader.load_model = mark_patched(load_model, original, patch="l2_persist")


def _self_check() -> None:
    # Struct layout is the risky part: a wrong offset silently corrupts the CUDA call.
    assert ctypes.sizeof(AccessPolicyWindow) == 32, ctypes.sizeof(AccessPolicyWindow)
    off = {f: getattr(AccessPolicyWindow, f).offset for f, _ in AccessPolicyWindow._fields_}
    assert off == {"base_ptr": 0, "num_bytes": 8, "hitRatio": 16, "hitProp": 20, "missProp": 24}, off
    # cuda 12's union is 64B (char pad[64]); the window aliases it at offset 0.
    assert ctypes.sizeof(StreamAttrValue) == 64, ctypes.sizeof(StreamAttrValue)
    assert StreamAttrValue.accessPolicyWindow.offset == 0

    # Round-trip a populated window through the union: what we hand CUDA is what we set.
    v = StreamAttrValue()
    ctypes.memset(ctypes.byref(v), 0, ctypes.sizeof(v))
    v.accessPolicyWindow = AccessPolicyWindow(
        base_ptr=ctypes.c_void_p(0x1000),
        num_bytes=ctypes.c_size_t(1 << 23),
        hitRatio=ctypes.c_float(0.6),
        hitProp=ACCESS_PROPERTY_PERSISTING,
        missProp=ACCESS_PROPERTY_STREAMING,
    )
    assert v.accessPolicyWindow.base_ptr == 0x1000
    assert v.accessPolicyWindow.num_bytes == 1 << 23
    assert abs(v.accessPolicyWindow.hitRatio - 0.6) < 1e-6
    assert (v.accessPolicyWindow.hitProp, v.accessPolicyWindow.missProp) == (2, 1)

    # env parsing: absent/blank/garbage/non-positive all fall back, good values pass through.
    for key in ("VTL_L2_PERSIST", "VTL_L2_PERSIST_BYTES", "VTL_L2_HIT_RATIO"):
        os.environ.pop(key, None)
    assert env_flag("VTL_L2_PERSIST") is False
    os.environ["VTL_L2_PERSIST"] = "on"
    assert env_flag("VTL_L2_PERSIST") is True
    os.environ["VTL_L2_PERSIST"] = "0"
    assert env_flag("VTL_L2_PERSIST") is False
    assert env_num("VTL_L2_PERSIST_BYTES", DEFAULT_PERSIST_BYTES, int) == DEFAULT_PERSIST_BYTES
    for bad in ("", "   ", "banana", "0", "-1"):
        os.environ["VTL_L2_PERSIST_BYTES"] = bad
        assert env_num("VTL_L2_PERSIST_BYTES", 7, int) == 7, bad
    os.environ["VTL_L2_PERSIST_BYTES"] = "8388608"
    assert env_num("VTL_L2_PERSIST_BYTES", 7, int) == 8388608
    assert env_num("VTL_L2_HIT_RATIO", DEFAULT_HIT_RATIO, float) == DEFAULT_HIT_RATIO
    for key in ("VTL_L2_PERSIST", "VTL_L2_PERSIST_BYTES"):
        os.environ.pop(key, None)

    # "biggest weight_packed wins", checked without torch.
    class _T:
        def __init__(self, n, ptr):
            self._n, self._p = n, ptr

        def numel(self):
            return self._n

        def element_size(self):
            return 4

        def data_ptr(self):
            return self._p

    class _M:
        def __init__(self, mods):
            self._mods = mods

        def named_modules(self):
            return iter(self._mods)

    class _Layer:
        def __init__(self, packed=None):
            if packed is not None:
                self.weight_packed = packed

    assert largest_packed_weight(_M([("a", _Layer())])) is None
    got = largest_packed_weight(
        _M(
            [
                ("norm", _Layer()),
                ("w2", _Layer(_T(2 << 20, 0xA000))),
                ("w13", _Layer(_T(4 << 20, 0xB000))),
            ]
        )
    )
    assert got == ("w13", 0xB000, 4 * (4 << 20)), got

    # The probe itself: real numbers on-box, a clean skip off-box. Never asserts a value --
    # 0 (MIG) and nonzero (not MIG) are both legitimate answers, which is the whole point.
    probed = probe()
    if probed is None:
        print("l2_persist self-check ok (no CUDA runtime here; probe skipped)")
    else:
        print(
            f"l2_persist self-check ok -- maxPersistingL2CacheSize={probed[0]} B, "
            f"cudaLimitPersistingL2CacheSize={probed[1]} B"
        )


if __name__ == "__main__":
    _self_check()
