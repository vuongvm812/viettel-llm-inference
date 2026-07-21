"""Build ``vtl._C``. Package metadata stays in pyproject.toml.

The extension is optional. Without nvcc (the dev Mac, a doc build) we produce a
pure-Python wheel; ``vtl/patches/rms_norm_quant.py`` then logs and leaves vLLM's
stock kernel in place. Requires ``--no-build-isolation`` so setup.py sees the
image's torch instead of an empty build env.
"""

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


_ext, _cmdclass = _cuda_ext()
_exts = [e for e in (_ext,) if e is not None]
setup(ext_modules=_exts, cmdclass=_cmdclass)
