"""Build ``vtl._C``. Package metadata stays in pyproject.toml.

The extension is optional. Without nvcc (the dev Mac, a doc build) we produce a
pure-Python wheel; ``vtl/patches/rms_norm_quant.py`` then logs and leaves vLLM's
stock kernel in place. Requires ``--no-build-isolation`` so setup.py sees the
image's torch instead of an empty build env.
"""

import os

from setuptools import setup

# Experimental link-time optimization, OFF by default so the normal (scored) build is untouched.
# `VTL_LTO=1 make build` turns it on for an on-box A/B. Two reasons it is gated, not default:
#   - These kernels are memory-bandwidth bound and each is already fully inlined within its own
#     TU (fp8_common.cuh is header-included), so the only cross-TU boundary is cold host glue --
#     expected hot-path gain is ~0.
#   - `-dlto` is DEVICE LTO and needs `-rdc=true` (relocatable device code), which CHANGES device
#     codegen (can disable some per-kernel opts) and adds a device-link step. Same codegen-risk
#     class as --use_fast_math above. Validate with Nsight + a TPOT A/B + the gpqa eval before
#     trusting it; if the image's toolchain rejects the LTO objects at link, just leave VTL_LTO off.
_LTO = os.environ.get("VTL_LTO", "").strip().lower() in {"1", "true", "yes", "on"}


def _cuda_ext():
    try:
        from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
    except ImportError:
        return None, {}
    if CUDA_HOME is None:
        return None, {}

    cxx = ["-O3", "-std=c++17"]
    nvcc = [
        "-O3",
        "-std=c++17",
        "-lineinfo",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ]
    link = []
    if _LTO:
        cxx += ["-flto"]
        # -rdc=true is required for -dlto; both the compile and the device link must carry -dlto.
        nvcc += ["-rdc=true", "-dlto"]
        link += ["-flto"]

    ext = CUDAExtension(
        name="vtl._C",
        sources=[
            "vtl/csrc/rms_norm_quant.cu",
            "vtl/csrc/dynamic_per_token_quant.cu",
            "vtl/csrc/silu_mul_quant.cu",
            "vtl/csrc/mul_quant.cu",
            "vtl/csrc/torch_bindings.cpp",
        ],
        extra_compile_args={
            # No --use_fast_math: the kernels do a TARGETED reciprocal-multiply on the fp8
            # output (one true `1.0f / scale` per token); global fast-math would also perturb
            # the SiLU sigmoid and the RMS rsqrt more broadly than intended.
            "cxx": cxx,
            "nvcc": nvcc,
        },
        extra_link_args=link,
    )
    return ext, {"build_ext": BuildExtension}


_ext, _cmdclass = _cuda_ext()
_exts = [e for e in (_ext,) if e is not None]
setup(ext_modules=_exts, cmdclass=_cmdclass)
