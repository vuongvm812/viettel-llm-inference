"""Build ``vtl._C``. Package metadata stays in pyproject.toml.

The extension is optional. Without nvcc (the dev Mac, a doc build) we produce a
pure-Python wheel; ``vtl/patches/rms_norm_quant.py`` then logs and leaves vLLM's
stock kernel in place. Requires ``--no-build-isolation`` so setup.py sees the
image's torch instead of an empty build env.
"""

import os

from setuptools import setup


def _ngram_ext():
    """Opt-in C++ tree-ngram drafter (vtl_ngram). OFF unless VTL_BUILD_NGRAM=1 so the standard
    image build is byte-identical; vtl/ngram_tree.py always has a pure-Python fallback. Needs
    only torch's pybind11 (CppExtension), no CUDA."""
    if os.environ.get("VTL_BUILD_NGRAM", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    try:
        from torch.utils.cpp_extension import CppExtension
    except ImportError:
        return None
    return CppExtension(
        name="vtl_ngram",
        sources=["vtl/csrc/ngram_tree/ngram_tree.cpp"],
        # -flto: single-TU host ext, so LTO only helps the linker prune/inline into
        # torch's runtime — marginal, but free. No PGO: device code can't, host drafter
        # isn't hot enough to justify a profile step.
        extra_compile_args={"cxx": ["-O3", "-std=c++17", "-flto"]},
        extra_link_args=["-flto"],
    )


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
            "vtl/csrc/torch_bindings.cpp",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            # No --use_fast_math: it would turn `y / scale` into a reciprocal
            # approximation and break element-for-element parity with stock.
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


_ext, _cmdclass = _cuda_ext()
_ngram = _ngram_ext()
if _ngram is not None and "build_ext" not in _cmdclass:
    # vtl._C was skipped (no CUDA_HOME) but we still need BuildExtension for the C++ ext.
    try:
        from torch.utils.cpp_extension import BuildExtension

        _cmdclass = {"build_ext": BuildExtension}
    except ImportError:
        _ngram = None

_exts = [e for e in (_ext, _ngram) if e is not None]
setup(ext_modules=_exts, cmdclass=_cmdclass)
