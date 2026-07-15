"""Build ``vtl._C``. Package metadata stays in pyproject.toml.

The extension is optional. Without nvcc (the dev Mac, a doc build) we produce a
pure-Python wheel; ``vtl/patches/rms_norm_quant.py`` then logs and leaves vLLM's
stock kernel in place. Requires ``--no-build-isolation`` so setup.py sees the
image's torch instead of an empty build env.

``mtp_fc_gemm.cu`` additionally needs cuBLASDx headers (the ``nvidia-mathdx``
wheel, which also bundles CUTLASS). If those headers are not found the file is
dropped from the build and ``vtl/patches/mtp_kernels.py`` falls back to stock --
the other kernels still build. Mirrors sglang's mathdx include resolution
(sglang/python/sglang/jit_kernel/utils.py get_mathdx_include_paths).
"""

import glob
import os

from setuptools import setup


def _mathdx_includes():
    """cuBLASDx + bundled CUTLASS include dirs, or [] if the wheel/env is absent.

    Resolution order matches sglang: $MATHDX_HOME first, then the nvidia-mathdx
    pip wheel (the ``nvidia.mathdx`` namespace package). A dir counts only if it
    actually contains ``cublasdx.hpp`` -- a wrong path that silently compiles
    without the header would fail deep in nvcc instead of here.
    """
    roots = []
    if os.environ.get("MATHDX_HOME"):
        roots.append(os.environ["MATHDX_HOME"])
    try:
        import nvidia.mathdx as _mathdx  # type: ignore

        roots.extend(_mathdx.__path__)
    except Exception:
        pass

    for root in roots:
        for inc in glob.glob(os.path.join(root, "**", "cublasdx.hpp"), recursive=True):
            base = os.path.dirname(inc)  # <root>/.../include (holds cublasdx.hpp)
            cutlass = os.path.join(os.path.dirname(base), "external", "cutlass", "include")
            dirs = [base]
            if os.path.isdir(cutlass):
                dirs.append(cutlass)
            return dirs
    return []


def _cuda_ext():
    try:
        from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
    except ImportError:
        return None, {}
    if CUDA_HOME is None:
        return None, {}

    sources = [
        "vtl/csrc/rms_norm_quant.cu",
        "vtl/csrc/dynamic_per_token_quant.cu",
        "vtl/csrc/silu_mul_quant.cu",
        "vtl/csrc/mul_sigmoid_quant.cu",
        "vtl/csrc/gdn_gated_rmsnorm.cu",
        "vtl/csrc/gdn/chunk_scan.cu",
    ]
    nvcc = [
        "-O3",
        "-std=c++17",
        "-lineinfo",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ]
    include_dirs = []

    mathdx = _mathdx_includes()
    if mathdx:
        sources.append("vtl/csrc/mtp_fc_gemm.cu")
        include_dirs.extend(mathdx)
        # cuBLASDx tensor-core GEMM targets Hopper; emit sm_90 explicitly. The
        # image's TORCH_CUDA_ARCH_LIST still governs the other kernels' cubins.
        nvcc.append("-gencode=arch=compute_90,code=sm_90")
    else:
        print("vtl: cuBLASDx headers not found (nvidia-mathdx); skipping mtp_fc_gemm.cu")

    # torch_bindings.cpp registers whatever ops built; it guards the mtp op with
    # a VTL_WITH_MTP_FC_GEMM define so the binding matches the compiled sources.
    cxx = ["-O3", "-std=c++17"]
    if mathdx:
        cxx.append("-DVTL_WITH_MTP_FC_GEMM")
        nvcc.append("-DVTL_WITH_MTP_FC_GEMM")
    sources.append("vtl/csrc/torch_bindings.cpp")

    ext = CUDAExtension(
        name="vtl._C",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={
            # No --use_fast_math: it would turn `y / scale` into a reciprocal
            # approximation and break element-for-element parity with stock.
            "cxx": cxx,
            "nvcc": nvcc,
        },
    )
    return ext, {"build_ext": BuildExtension}


_ext, _cmdclass = _cuda_ext()

setup(ext_modules=[_ext] if _ext else [], cmdclass=_cmdclass)
