# Static build-flag config for the vendored FA3 kernels.
# Mirrors what hopper/setup.py generates, frozen to the LFM2.5-1.2B prune set
# (head_dim=64, bf16 + fp8/e4m3, causal, GQA, paged, split; no backward, no sm8x,
# no fp16, no softcap, no local/sliding-window). Keep in sync with the
# -DFLASHATTENTION_DISABLE_* set in round-1.2/setup.py.
CONFIG = {
    "build_flags": {
        "FLASHATTENTION_DISABLE_BACKWARD": True,
        "FLASHATTENTION_DISABLE_SPLIT": False,
        "FLASHATTENTION_DISABLE_PAGEDKV": False,
        "FLASHATTENTION_DISABLE_APPENDKV": False,
        "FLASHATTENTION_DISABLE_LOCAL": True,
        "FLASHATTENTION_DISABLE_SOFTCAP": True,
        "FLASHATTENTION_DISABLE_PACKGQA": False,
        "FLASHATTENTION_DISABLE_FP16": True,
        "FLASHATTENTION_DISABLE_FP8": False,
        "FLASHATTENTION_DISABLE_VARLEN": False,
        "FLASHATTENTION_DISABLE_CLUSTER": False,
        "FLASHATTENTION_DISABLE_HDIM64": False,
        "FLASHATTENTION_DISABLE_HDIM96": True,
        "FLASHATTENTION_DISABLE_HDIM128": True,
        "FLASHATTENTION_DISABLE_HDIM192": True,
        "FLASHATTENTION_DISABLE_HDIM256": True,
        "FLASHATTENTION_DISABLE_SM8x": True,
        "FLASHATTENTION_ENABLE_VCOLMAJOR": False,
        "FLASH_ATTENTION_DISABLE_HDIMDIFF64": True,
        "FLASH_ATTENTION_DISABLE_HDIMDIFF192": True,
    }
}
