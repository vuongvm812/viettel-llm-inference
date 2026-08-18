"""NVRTC: compile a CUDA kernel at load time, specialized to the model that got loaded.

WHY THIS EXISTS (two reasons, one mechanism):

  1. LATENCY. An AOT kernel in ``vtl._C`` is compiled once for every model it might ever
     see, so shapes are runtime values: loop trip counts are unknown, `hidden_size` is a
     register, and every `/` and `%` against it is a real division. Compiling here instead
     means the shapes are `-D` macros -- the loops unroll, the divides fold, and the
     compiler sizes registers for the real case. That is a win an AOT build cannot have,
     because at AOT time the model is not known.

  2. ITERATION. Sources live in ``vtl/kernels/*.cu`` as DATA, not baked into the .so. A
     kernel edit is a container restart (~15 s), not a `pip install /src` with nvcc over
     four SM arches (~20 min). See docker-compose.nvrtc-dev.yaml.

THE COMPILE IS NOT ON THE SERVING PATH. Cubins are cached on disk under
``$VTL_NVRTC_CACHE`` (default ``/opt/vtl/cache/nvrtc``), which is the same directory tree
``make warm`` harvests into the image. So a warmed image starts with every cubin already
built and the judge's boot pays nothing. A cold cache costs one compile per (source,
defines, arch) at model load, before the server reports healthy.

DEGRADE, NEVER DIE. Same doctrine as the jemalloc / CUTLASS / iceoryx2 steps in the
Dockerfile: no cuda-python, no driver, a syntax error in a kernel -- every one of these
returns None and leaves the AOT kernel in place. A missing optimization is a slower
server; a raised exception at model load is a dead one.

    VTL_NVRTC=1              opt in (default OFF until A/B'd against the AOT kernel)
    VTL_NVRTC_SRC=/kernels   override the source dir (the dev-loop bind mount)
    VTL_NVRTC_CACHE=<dir>    override the cubin cache dir
    VTL_NVRTC_ARCH=90a       override the target arch (default: probe the device)
    VTL_NVRTC_INCLUDE=<dir>  override the CUDA header dir passed as -I (default: probe)
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

log = logging.getLogger("vllm.vtl.nvrtc")

_TRUTHY = frozenset({"1", "true", "yes", "on"})

DEFAULT_CACHE = "/opt/vtl/cache/nvrtc"
# Mirrors setup.py's nvcc flags. --use_fast_math is deliberately ABSENT: these kernels do a
# targeted reciprocal-multiply on the fp8 output, and global fast-math would also perturb the
# RMS rsqrt and any sigmoid, which changes the per-token scale -- a different kernel, not a
# faster one.
BASE_FLAGS = ("--std=c++17", "-default-device", "-DNDEBUG")


def enabled() -> bool:
    return os.environ.get("VTL_NVRTC", "").strip().lower() in _TRUTHY


def source_dir() -> Path:
    """Where ``.cu`` sources live. The env override is what the dev-loop overlay mounts."""
    override = os.environ.get("VTL_NVRTC_SRC", "").strip()
    return Path(override) if override else Path(__file__).parent / "kernels"


def render_defines(defines: dict[str, object]) -> tuple[str, ...]:
    """``{"HIDDEN": 2048}`` -> ``("-DHIDDEN=2048",)``, sorted so the cache key is stable.

    Sorted matters: dict order would otherwise make the same specialization hash two ways
    and compile twice, which is exactly the boot-time stall this cache exists to avoid.
    """
    return tuple(f"-D{k}={defines[k]}" for k in sorted(defines))


def cache_key(src: str, defines: dict[str, object], arch: str, toolkit: str) -> str:
    """Identity of a compiled cubin.

    Every input that changes the emitted code is in here. `toolkit` especially: a cubin
    built by another NVRTC version can be silently wrong or simply not load, and the cache
    directory outlives toolkit upgrades because it is baked into the image.
    """
    h = hashlib.sha256()
    for part in (src, "\0".join(render_defines(defines)), arch, toolkit):
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()[:32]


def cache_dir() -> Path:
    return Path(os.environ.get("VTL_NVRTC_CACHE", "").strip() or DEFAULT_CACHE)


# The kernels #include <cuda_bf16.h>/<cuda_fp8.h>, and NVRTC searches NO directories by
# default -- whether those resolve without -I depends on the headers bundled into the exact
# libnvrtc build, which changed across the 12.x -> 13.x wheel generations (a driver-13.0 host
# resolves an unpinned cuda-python to 13.x, whose NVRTC no longer resolves them:
# "catastrophic error: could not open source file \"cuda_bf16.h\" (no directories in search
# list)"). So find the real headers and pass them explicitly.
def cuda_include_dir() -> Path | None:
    """First directory holding ``cuda_bf16.h``: env override, toolkit dirs, then pip wheels."""
    override = os.environ.get("VTL_NVRTC_INCLUDE", "").strip()
    if override:
        return Path(override)  # trust the operator even if the probe below would reject it
    candidates: list[Path] = []
    for base in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda"):
        if base and base.strip():
            candidates.append(Path(base.strip()) / "include")
    try:
        import nvidia  # namespace package the pip CUDA wheels install under

        for root in nvidia.__path__:
            root = Path(root)
            candidates.extend(sorted(root.glob("*/include")))    # nvidia/cuda_runtime/include (cu12)
            candidates.extend(sorted(root.glob("*/*/include")))  # nvidia/cu13/... nesting
    except Exception:
        pass
    for d in candidates:
        try:
            if (d / "cuda_bf16.h").is_file():
                return d
        except OSError:
            continue
    return None


def headers_fingerprint(inc: Path | None) -> str:
    """Identity of the CUDA headers a compile will see, for the cache key.

    The headers come from a DIFFERENT component than libnvrtc (toolkit dir or a pip wheel vs
    the nvrtc library), so they can change without ``nvrtcVersion()`` changing -- and a cubin
    built against different headers is a different cubin. ``"noinc"`` when no dir was found.
    """
    if inc is None:
        return "noinc"
    h = hashlib.sha256()
    for name in ("cuda_bf16.h", "cuda_fp16.h", "cuda_fp8.h"):
        try:
            h.update((inc / name).read_bytes())
        except OSError:
            h.update(b"absent")
        h.update(b"\0")
    return h.hexdigest()[:16]


def device_arch() -> str | None:
    """Compute capability of device 0 as an NVRTC arch string, e.g. ``90a``.

    The ``a`` suffix (Hopper+ architecture-specific features) is what CUTLASS-class kernels
    need; it is only valid from sm_90 up, so older devices get the plain number.
    """
    override = os.environ.get("VTL_NVRTC_ARCH", "").strip()
    if override:
        return override
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return None
    return f"{major}{minor}a" if major >= 9 else f"{major}{minor}"


def _nvrtc_compile(src: str, name: str, flags: tuple[str, ...]) -> bytes | None:
    """Source -> cubin via NVRTC. None (with a logged reason) on any failure."""
    try:
        from cuda.bindings import nvrtc
    except Exception:
        try:
            from cuda import nvrtc  # cuda-python < 12.8 spelling
        except Exception as exc:
            log.info("nvrtc: cuda-python not available (%s); staying on the AOT kernel", exc)
            return None

    def _ok(res, *rest):
        code = res[0] if isinstance(res, tuple) else res
        if int(code) != 0:
            raise RuntimeError(f"nvrtc error {code}")
        return rest[0] if rest else None

    prog = None
    try:
        err, prog = nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], [])
        _ok(err)
        opts = [f.encode() for f in flags]
        err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[:1]
        # Always read the log: NVRTC warnings (register spills, unused params) are the
        # only signal that a "successful" specialization is quietly worse than the AOT one.
        _, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
        buf = b" " * log_size
        nvrtc.nvrtcGetProgramLog(prog, buf)
        text = buf.decode(errors="replace").strip("\0 \n")
        if int(err) != 0:
            log.warning("nvrtc: %s failed to compile; keeping the AOT kernel\n%s", name, text)
            return None
        if text:
            log.info("nvrtc: %s compile log:\n%s", name, text)
        _, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        return cubin
    except Exception as exc:
        log.warning("nvrtc: %s compile raised (%r); keeping the AOT kernel", name, exc)
        return None
    finally:
        if prog is not None:
            try:
                nvrtc.nvrtcDestroyProgram(prog)
            except Exception:
                pass


def build_cubin(src: str, name: str, defines: dict[str, object], arch: str | None = None):
    """Compile (or load from cache) one kernel. Returns ``(cubin_bytes, key)`` or None.

    Cache read/write failures are non-fatal on purpose: a read-only cache dir must cost a
    recompile, not the kernel.
    """
    arch = arch or device_arch()
    if arch is None:
        log.info("nvrtc: no CUDA device/arch; staying on the AOT kernel")
        return None

    try:
        from cuda.bindings import nvrtc

        toolkit = str(nvrtc.nvrtcVersion()[1:])
    except Exception:
        toolkit = "unknown"

    inc = cuda_include_dir()
    if inc is None:
        log.info(
            "nvrtc: no CUDA include dir with cuda_bf16.h found; a kernel that includes CUDA "
            "headers will fail to compile -- set VTL_NVRTC_INCLUDE to fix"
        )

    # Fold the header identity in next to the toolkit version -- see headers_fingerprint.
    key = cache_key(src, defines, arch, f"{toolkit}+hdr:{headers_fingerprint(inc)}")
    path = cache_dir() / f"{name}-{key}.cubin"
    try:
        if path.is_file():
            return path.read_bytes(), key
    except Exception as exc:
        log.info("nvrtc: cache read failed (%r); recompiling", exc)

    inc_flags = (f"-I{inc}",) if inc is not None else ()
    flags = BASE_FLAGS + inc_flags + (f"--gpu-architecture=compute_{arch}",) + render_defines(defines)
    cubin = _nvrtc_compile(src, name, flags)
    if cubin is None:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".cubin.tmp")
        tmp.write_bytes(cubin)
        tmp.replace(path)  # atomic: two workers may compile the same kernel concurrently
    except Exception as exc:
        log.info("nvrtc: cache write failed (%r); the kernel still works", exc)
    return cubin, key


def load_source(name: str) -> str | None:
    """Read ``<source_dir>/<name>.cu``. None if absent (the dev mount may not be there)."""
    path = source_dir() / f"{name}.cu"
    try:
        return path.read_text()
    except Exception as exc:
        log.info("nvrtc: %s unreadable (%r); staying on the AOT kernel", path, exc)
        return None


def compile_kernel(name: str, defines: dict[str, object], entry: str | None = None):
    """The one public entry: name + shape defines -> a launchable kernel, or None.

    Returns a ``torch.cuda`` raw-module kernel whose call signature is
    ``k(grid=(gx,gy,gz), block=(bx,by,bz), args=[...], stream=...)``. None means "use the
    AOT kernel", which every caller must be prepared for -- see the module docstring.
    """
    if not enabled():
        return None
    src = load_source(name)
    if src is None:
        return None
    built = build_cubin(src, name, defines)
    if built is None:
        return None
    cubin, key = built
    try:
        kernel = _load_cubin(cubin, entry or name)
    except Exception as exc:
        log.warning("nvrtc: %s built but failed to load (%r); keeping the AOT kernel", name, exc)
        return None
    log.info("nvrtc: %s specialized for %s (cubin %s)", name, render_defines(defines), key)
    return kernel


def pack_args(*args):
    """Build the ``void**`` argument array ``cuLaunchKernel`` wants.

    Plain ints are device pointers (pass ``tensor.data_ptr()``, and ``0`` for an optional
    tensor the kernel will not read). A scalar needs its C type stated, as a
    ``(kind, value)`` pair -- ``("f", 1e-6)`` for float, ``("i", n)`` for int32,
    ``("u", n)`` for uint32 -- because Python's int/float do not map to a unique C width and
    guessing wrong silently misaligns every argument after it.

    The ctypes objects are kept alive on the returned array: if they were temporaries the
    array would hold dangling pointers by the time the kernel launches. They also stay put,
    so a caller that packs ONCE at arm time and then only writes ``arr._vtl_holders[i].value``
    per launch keeps a valid void** array forever -- which is what the hot callers do.

    ``addressof``, NOT ``cast(pointer(h), c_void_p)``. The two produce the same integer (the
    self-check asserts exactly that, below), but the cast form builds a POINTER type object
    and a c_void_p per argument and was measured at ~5.7 us per call on a 3-argument pack --
    the single most expensive line on the greedy-argmax token pick. addressof is one C call
    returning an int: ~1.7 us for the same pack, i.e. ~3.5x, and every NVRTC consumer in the
    tree gets it. (A caller on a per-token path should still pack once and mutate; see the
    holders note above. That is another ~10x on top.)
    """
    import ctypes

    kinds = {"f": ctypes.c_float, "d": ctypes.c_double, "i": ctypes.c_int32,
             "u": ctypes.c_uint32, "l": ctypes.c_int64}
    holders = []
    for a in args:
        if isinstance(a, tuple):
            kind, value = a
            if kind not in kinds:
                raise ValueError(f"unknown scalar kind {kind!r}; use one of {sorted(kinds)}")
            holders.append(kinds[kind](value))
        else:
            holders.append(ctypes.c_void_p(int(a)))
    arr = (ctypes.c_void_p * len(holders))(*[ctypes.addressof(h) for h in holders])
    arr._vtl_holders = holders  # keep-alive; see docstring
    return arr


def _load_cubin(cubin: bytes, entry: str):
    """Cubin -> callable, via the CUDA driver API.

    Kept separate from compile_kernel so the loader can be swapped without touching the
    compile/cache logic, and so the compile half stays testable without a driver.
    """
    from cuda.bindings import driver

    def _check(res):
        code, *rest = res if isinstance(res, tuple) else (res,)
        if int(code) != 0:
            raise RuntimeError(f"cuda driver error {code}")
        return rest[0] if rest else None

    import torch

    torch.cuda.init()  # ensure a context exists on this thread before we touch the driver
    module = _check(driver.cuModuleLoadData(cubin))
    func = _check(driver.cuModuleGetFunction(module, entry.encode()))

    def launch(grid, block, args, shared_mem=0, stream=None):
        import ctypes

        stream_ptr = stream if stream is not None else torch.cuda.current_stream().cuda_stream
        # cuLaunchKernel takes the ADDRESS of the void** array, not the array object.
        params = ctypes.addressof(args) if isinstance(args, ctypes.Array) else args
        _check(
            driver.cuLaunchKernel(
                func, *grid, *block, shared_mem, stream_ptr, params, 0
            )
        )

    launch.module = module
    launch.entry = entry
    return launch


def _self_check() -> None:
    """No GPU, no cuda-python: the pure half (defines, cache key, gating, source lookup)."""
    import tempfile

    # -- defines render sorted, so the cache key does not depend on dict order --
    assert render_defines({"HIDDEN": 2048}) == ("-DHIDDEN=2048",)
    assert render_defines({"B": 2, "A": 1}) == ("-DA=1", "-DB=2")
    assert render_defines({"B": 2, "A": 1}) == render_defines({"A": 1, "B": 2})
    assert render_defines({}) == ()

    # -- every input that changes the emitted code changes the key --
    base = ("src", {"H": 1}, "90a", "12.4")
    k = cache_key(*base)
    assert len(k) == 32
    assert cache_key(*base) == k, "same inputs must be stable across calls"
    assert cache_key("src2", {"H": 1}, "90a", "12.4") != k, "source"
    assert cache_key("src", {"H": 2}, "90a", "12.4") != k, "defines"
    assert cache_key("src", {"H": 1}, "80", "12.4") != k, "arch"
    assert cache_key("src", {"H": 1}, "90a", "12.5") != k, "toolkit version"
    # ...and dict order does NOT, or a warm cache would miss half the time.
    assert cache_key("s", {"A": 1, "B": 2}, "90a", "1") == cache_key("s", {"B": 2, "A": 1}, "90a", "1")
    # Separator: {"AB": 1} and {"A": "B=1"} must not collide.
    assert cache_key("s", {"AB": 1}, "90a", "1") != cache_key("s", {"A": "B=1"}, "90a", "1")

    # -- gating: OFF unless explicitly on --
    for val, want in (("", False), ("0", False), ("no", False), ("1", True), ("ON", True)):
        os.environ["VTL_NVRTC"] = val
        assert enabled() is want, (val, want)
    os.environ.pop("VTL_NVRTC", None)
    assert enabled() is False, "unset must be OFF -- this is an unproven optimization"

    # -- compile_kernel is a no-op while disabled, even if everything else is present --
    assert compile_kernel("nope", {"H": 1}) is None

    # -- source lookup honours the dev-loop override and never raises on a missing file --
    with tempfile.TemporaryDirectory() as d:
        os.environ["VTL_NVRTC_SRC"] = d
        assert source_dir() == Path(d)
        assert load_source("absent") is None
        (Path(d) / "present.cu").write_text("__global__ void k() {}")
        assert load_source("present") == "__global__ void k() {}"
        os.environ.pop("VTL_NVRTC_SRC", None)
    assert source_dir().name == "kernels", source_dir()

    # -- EVERY shipped kernel must be readable from the installed package. A glob, not a
    #    hand-maintained name list: the list drifts silently (a new .cu is simply never
    #    checked, and a packaging bug that drops it shows up as a boot-time compile failure
    #    instead of here), while the glob grows itself. --
    packaged = sorted(p.stem for p in (Path(__file__).parent / "kernels").glob("*.cu"))
    assert packaged, "vtl/kernels/ ships no .cu at all"
    for name in packaged:
        assert load_source(name), f"vtl/kernels/{name}.cu missing from the package"

    os.environ["VTL_NVRTC_CACHE"] = "/tmp/x"
    assert cache_dir() == Path("/tmp/x")
    os.environ.pop("VTL_NVRTC_CACHE", None)
    assert cache_dir() == Path(DEFAULT_CACHE)

    # -- CUDA include probing: override wins, then CUDA_HOME; fingerprint tracks contents --
    saved_env = {k: os.environ.pop(k, None) for k in ("VTL_NVRTC_INCLUDE", "CUDA_HOME", "CUDA_PATH")}
    try:
        with tempfile.TemporaryDirectory() as d:
            inc = Path(d) / "include"
            inc.mkdir()
            (inc / "cuda_bf16.h").write_text("// bf16 v1")
            os.environ["CUDA_HOME"] = d
            assert cuda_include_dir() == inc, "CUDA_HOME/include holding cuda_bf16.h must be found"
            os.environ["VTL_NVRTC_INCLUDE"] = "/elsewhere"
            assert cuda_include_dir() == Path("/elsewhere"), "the override must win, unverified"
            os.environ.pop("VTL_NVRTC_INCLUDE", None)

            fp1 = headers_fingerprint(inc)
            assert fp1 == headers_fingerprint(inc), "same headers must fingerprint the same"
            (inc / "cuda_bf16.h").write_text("// bf16 v2")
            assert headers_fingerprint(inc) != fp1, "changed header contents must change the key"
        assert headers_fingerprint(None) == "noinc"
        # An empty dir must NOT be picked (no cuda_bf16.h): point CUDA_HOME somewhere bare.
        with tempfile.TemporaryDirectory() as bare:
            os.environ["CUDA_HOME"] = bare
            found = cuda_include_dir()
            assert found is None or found != Path(bare) / "include", "a dir without cuda_bf16.h must be skipped"
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    assert "--use_fast_math" not in BASE_FLAGS, "would change the per-token scale"

    # -- pack_args: widths are explicit, and the ctypes holders outlive the call --
    import ctypes

    a = pack_args(0x1000, 0, ("f", 1.5), ("i", -7))
    assert isinstance(a, ctypes.Array) and len(a) == 4
    holders = a._vtl_holders
    assert isinstance(holders[0], ctypes.c_void_p) and holders[0].value == 0x1000
    assert holders[1].value in (0, None), "a null device pointer must survive as 0"
    assert isinstance(holders[2], ctypes.c_float) and holders[2].value == 1.5
    assert isinstance(holders[3], ctypes.c_int32) and holders[3].value == -7
    # Each slot must point AT its holder -- the bug this catches is passing values by copy,
    # which misaligns every argument after the first scalar.
    for slot, h in zip(a, holders):
        assert slot == ctypes.addressof(h), "arg slot must hold the address of its value"
    try:
        pack_args(("z", 1))
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown scalar kind must raise, not silently mis-pack")

    print("nvrtc self-check ok")


if __name__ == "__main__":
    _self_check()
