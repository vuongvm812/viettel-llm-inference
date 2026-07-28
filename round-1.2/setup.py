"""Build ``vtl._C`` and (optionally) ``vtl._C_w4a8``. Package metadata stays in pyproject.toml.

Both extensions are optional. Without nvcc (the dev Mac, a doc build) we produce a
pure-Python wheel; ``vtl/patches/rms_norm_quant.py`` then logs and leaves vLLM's
stock kernel in place. Requires ``--no-build-isolation`` so setup.py sees the
image's torch instead of an empty build env.
"""

import os

from setuptools import setup


def _cuda_ext():
    try:
        from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
    except ImportError:
        return None, {}
    if CUDA_HOME is None:
        return None, {}

    ext = CUDAExtension(
        name="vtl._C",
        sources=[
            "vtl/csrc/rms_norm_quant.cu",
            "vtl/csrc/dynamic_per_token_quant.cu",
            "vtl/csrc/silu_mul_quant.cu",
            "vtl/csrc/mul_quant.cu",
            "vtl/csrc/bcx_conv_gate_quant.cu",
            "vtl/csrc/torch_bindings.cpp",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            # No --use_fast_math: the kernels do a TARGETED reciprocal-multiply on the fp8
            # output (one true `1.0f / scale` per token); global fast-math would also perturb
            # the SiLU sigmoid and the RMS rsqrt more broadly than intended.
            #
            # No device LTO (-rdc=true -dlto) either: torch's BuildExtension auto-appends
            # `-gencode ...,code=sm_90/compute_90`, which nvcc rejects alongside -dlto ("use
            # code=lto_<arch>"), and a working device-LTO build would need a separate nvcc
            # device-link step BuildExtension does not do -- for ~0 gain on these memory-bound
            # kernels (each already fully inlined within its own TU via fp8_common.cuh).
            "nvcc": [
                "-O3",
                "-std=c++17",
                "-lineinfo",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
            ],
        },
    )
    return ext, {"build_ext": BuildExtension}


def _w4a8_ext():
    """``vtl._C_w4a8`` -- extra CUTLASS W4A8 schedules (Stream-K / cluster / small tile).

    SKIPPED SILENTLY when ``CUTLASS_DIR`` is unset, which is the normal state everywhere
    except the Dockerfile's pip-install line. CUTLASS is a header-only dependency that vLLM
    pulls in via FetchContent at ITS build time -- the headers are in neither the wheel nor the
    runtime image -- so an env var is the honest gate: no headers, no extension, and
    ``VTL_W4A8_SCHEDULE_V2`` then finds no ops and stays off. Same optional-extension shape as
    ``_cuda_ext`` above; a failure here must never cost us the five kernels in ``vtl._C``.
    """
    cutlass = os.environ.get("CUTLASS_DIR", "").strip()
    if not cutlass:
        return None
    try:
        from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension
    except ImportError:
        return None
    if CUDA_HOME is None:
        return None

    here = os.path.dirname(os.path.abspath(__file__))
    return CUDAExtension(
        name="vtl._C_w4a8",
        sources=["vtl/csrc/w4a8/w4a8_mm_v2.cu"],
        include_dirs=[
            os.path.join(cutlass, "include"),
            os.path.join(cutlass, "tools", "util", "include"),
            # So the vendored `cutlass_extensions/epilogue/...` includes resolve with the same
            # spelling vLLM uses -- the three headers are byte-identical copies and staying
            # diffable against them is worth one include path.
            os.path.join(here, "vtl", "csrc", "w4a8"),
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            # -gencode sm_90a ONLY, and NO PTX. Two deliberate choices:
            #   * the `arch` substring makes torch's BuildExtension skip its own
            #     TORCH_CUDA_ARCH_LIST expansion, so this .so does not get built for the four
            #     dev-box arches that cannot run a Hopper-only kernel anyway (each CUTLASS
            #     instantiation is ~a minute of nvcc);
            #   * no +PTX, so a wrong-arch build cannot JIT its way to looking healthy -- the
            #     Dockerfile's cuobjdump assert and the boot probe both want a hard failure.
            # Split an arm out with e.g. CUTLASS_NVCC_EXTRA='-DVTL_W4A8_ENABLE_PP=0'.
            "nvcc": [
                "-O3",
                "-std=c++17",
                "--expt-relaxed-constexpr",
                "-DNDEBUG",
                "-gencode",
                "arch=compute_90a,code=sm_90a",
            ]
            + os.environ.get("CUTLASS_NVCC_EXTRA", "").split(),
        },
    )


_ext, _cmdclass = _cuda_ext()
_exts = [e for e in (_ext, _w4a8_ext() if _ext is not None else None) if e is not None]
setup(ext_modules=_exts, cmdclass=_cmdclass)
