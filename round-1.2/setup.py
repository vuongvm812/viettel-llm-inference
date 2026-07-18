"""Build ``vtl._C`` and the vendored ``flash_attn_3._C`` (official FA3, sm_90a).

Both extensions are optional. Without nvcc (the dev Mac, a doc build) we produce a
pure-Python wheel; the patches then log and leave vLLM's stock kernels in place.
Requires ``--no-build-isolation`` so setup.py sees the image's torch instead of an
empty build env.

``flash_attn_3._C`` is the Dao-AILab Hopper FlashAttention-3 kernel, vendored under
``flash_attn_3/csrc/`` and pruned via ``-DFLASHATTENTION_DISABLE_*`` to exactly what
LFM2.5-1.2B needs (head_dim=64, bf16 + fp8/e4m3, causal, GQA, paged, split; no
backward / sm8x / fp16 / softcap / local). The explicit ``-gencode ...sm_90a`` in the
nvcc flags both selects the ``a`` ISA that WGMMA/TMA require and suppresses torch's
``TORCH_CUDA_ARCH_LIST`` auto-arch expansion for this extension only, so it stays
sm_90a while ``vtl._C`` keeps the multi-arch list. Keep the DISABLE set in sync with
``flash_attn_3/flash_attn_config.py``.
"""

import itertools
from pathlib import Path

from setuptools import setup

_ROOT = Path(__file__).parent

# LFM2 prune set. Mirror in flash_attn_3/flash_attn_config.py.
_FA3_DISABLE = [
    "FLASHATTENTION_DISABLE_BACKWARD",
    "FLASHATTENTION_DISABLE_LOCAL",
    "FLASHATTENTION_DISABLE_SOFTCAP",
    "FLASHATTENTION_DISABLE_FP16",
    "FLASHATTENTION_DISABLE_HDIM96",
    "FLASHATTENTION_DISABLE_HDIM128",
    "FLASHATTENTION_DISABLE_HDIM192",
    "FLASHATTENTION_DISABLE_HDIM256",
    "FLASHATTENTION_DISABLE_SM8x",
    "FLASH_ATTENTION_DISABLE_HDIMDIFF64",
    "FLASH_ATTENTION_DISABLE_HDIMDIFF192",
]


def _cuda_ext():
    try:
        from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension
    except ImportError:
        return None
    if CUDA_HOME is None:
        return None
    return CUDAExtension(
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


def _fa3_ext():
    """Vendored official FA3 (hopper), forward-only, sm_90a, pruned to LFM2 shapes."""
    try:
        import torch
        from packaging.version import parse
        from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension
    except ImportError:
        return None
    if CUDA_HOME is None:
        return None

    csrc = _ROOT / "flash_attn_3" / "csrc"
    cutlass_inc = csrc / "cutlass" / "include"
    if not (csrc.exists() and cutlass_inc.exists()):
        return None  # sources not vendored (e.g. slim checkout) -> skip, stay stock

    # Forward sm90 instantiations: hdim64, {bf16,e4m3}, paged/split/packgqa, no softcap.
    # PackGQA is hard-coded on when paged or split, so those combos are excluded.
    hdim, dtypes = [64], ["bf16", "e4m3"]
    split, paged, softcap, packgqa = ["", "_split"], ["", "_paged"], [""], ["", "_packgqa"]
    fwd = [
        f"instantiations/flash_fwd_hdim{h}_{d}{p}{s}{sc}{pg}_sm90.cu"
        for h, d, s, p, sc, pg in itertools.product(
            hdim, dtypes, split, paged, softcap, packgqa
        )
        if not (pg and (p or s))
    ]

    # flash_api_stable.cpp (torch >= 2.9) uses the stable ABI; older torch uses flash_api.cpp.
    if parse(torch.__version__.split("+")[0]) >= parse("2.9.0.dev20250830"):
        api, stable = "flash_api_stable.cpp", ["-DTORCH_TARGET_VERSION=0x0209000000000000"]
    else:
        api, stable = "flash_api.cpp", []

    rel_sources = [api] + fwd + ["flash_fwd_combine.cu", "flash_prepare_scheduler.cu"]
    sources = [str(csrc / s) for s in rel_sources]

    feature_args = [f"-D{flag}" for flag in _FA3_DISABLE]
    nvcc = [
        "-O3",
        "-std=c++17",
        "--use_fast_math",
        "-lineinfo",
        "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",  # WGMMA shapes FA3 uses
        "-DCUTLASS_ENABLE_GDC_FOR_SM90",  # PDL
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
        "-DNDEBUG",  # critical: perf collapses without it
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        # Explicit gencode: selects the sm_90a ISA (WGMMA/TMA) AND suppresses torch's
        # TORCH_CUDA_ARCH_LIST auto-arch expansion for this extension only.
        "-gencode",
        "arch=compute_90a,code=sm_90a",
    ] + feature_args
    return CUDAExtension(
        name="flash_attn_3._C",
        sources=sources,
        include_dirs=[str(csrc), str(cutlass_inc)],
        extra_compile_args={"cxx": ["-O3", "-std=c++17"] + stable + feature_args, "nvcc": nvcc},
    )


_exts = [e for e in (_cuda_ext(), _fa3_ext()) if e is not None]
if _exts:
    from torch.utils.cpp_extension import BuildExtension

    setup(ext_modules=_exts, cmdclass={"build_ext": BuildExtension})
else:
    setup(ext_modules=[])
