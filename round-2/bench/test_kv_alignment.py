"""KV-cache page sizes are 128B-aligned and the flat buffers are 256B-aligned.

This is the regression guard for "L2 128B alignment of KV blocks", which the round-2
plan records as *already satisfied* by stock vLLM rather than something we implemented:

* ``_allocate_kv_cache_tensors`` allocates ONE flat ``torch.zeros(size, dtype=int8,
  device=cuda)`` per KV-cache tensor, and the torch caching allocator hands back 256B-aligned
  base pointers.
* ``_reshape_kv_cache_tensors`` then views that buffer as blocks exactly ``page_size_bytes``
  apart, so every block start is ``base + i * page_size_bytes``.

So block alignment reduces to: base % 256 == 0 (allocator) and page_size_bytes % 128 == 0
(model geometry). Only the second can regress from a config change -- flip ``--block-size``,
``--kv-cache-dtype`` or the conv dtype and it silently stops holding, costing an extra L2
sector fetch on every KV access. Hence this test.

The page-size formulas are vendored (small, and importing vLLM to get them would make the
whole file GPU-box-only). References into the served v0.25.0 tree:

  vllm/v1/kv_cache_interface.py  AttentionSpec.real_page_size_bytes  (2*block*heads*dim*dtype)
  vllm/v1/kv_cache_interface.py  MambaSpec.page_size_bytes           (sum of prod(shape)*dtype)
  vllm/model_executor/layers/mamba/mamba_utils.py  the conv-state shape helper
      -> (conv_dim, conv_L_cache - 1 + num_spec)
  vllm/v1/worker/gpu_model_runner.py  _allocate_kv_cache_tensors / _reshape_kv_cache_tensors

Runs anywhere. The formula asserts need nothing; the allocator assert needs CUDA and skips
without it (so it is real only inside the container, via ``make test-kernel``).

    python3 round-2/bench/test_kv_alignment.py
"""

from math import prod

# --- EXAMPLE geometry ---------------------------------------------------------------------
# Replace these with the served model's real numbers (config.json + docker-compose.yaml) once
# it is chosen. They are only inputs: what this file actually asserts is a PROPERTY of the
# page-size formulas (128B alignment), and the property is checked across a sweep of block
# sizes, dtypes and spec widths below -- not just at the one geometry pinned here.
#
# CONV_* apply only to a hybrid model with an SSM/conv state. For a plain dense model set
# NUM_CONV_LAYERS = 0 and the conv half of each check drops out.
BLOCK_SIZE = 16  # vLLM default; compose does not pass --block-size
NUM_KV_HEADS = 8  # num_key_value_heads
HEAD_SIZE = 64  # hidden_size / num_attention_heads
KV_DTYPE_BYTES = 1  # --kv-cache-dtype=fp8_e4m3
NUM_ATTN_LAYERS = 6  # count of full_attention entries in layer_types
CONV_DIM = 2048  # conv_dim        (hybrid only)
CONV_L_CACHE = 3  # conv_L_cache    (hybrid only)
CONV_DTYPE_BYTES = 2  # conv state dtype; defaults to the model dtype
NUM_CONV_LAYERS = 10  # count of recurrent entries in layer_types (0 for a dense model)

L2_SECTOR = 128  # bytes; the alignment that matters for the L2 access granularity
ALLOC_ALIGN = 256  # torch caching allocator base alignment


def attn_page_size_bytes(block_size, num_kv_heads, head_size, dtype_bytes):
    """``AttentionSpec.real_page_size_bytes`` -- K and V, hence the leading 2."""
    return 2 * block_size * num_kv_heads * head_size * dtype_bytes


def conv_page_size_bytes(conv_dim, conv_l_cache, dtype_bytes, num_spec=0):
    """``MambaSpec.page_size_bytes`` for a single conv-state tensor."""
    shapes = (conv_state_shape(conv_dim, conv_l_cache, num_spec),)
    return sum(prod(shape) * dtype_bytes for shape in shapes)


def conv_state_shape(conv_dim, conv_l_cache, num_spec=0):
    return (conv_dim, conv_l_cache - 1 + num_spec)


def test_page_sizes_are_l2_aligned():
    attn = attn_page_size_bytes(BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, KV_DTYPE_BYTES)
    conv = conv_page_size_bytes(CONV_DIM, CONV_L_CACHE, CONV_DTYPE_BYTES)
    assert attn == 16384, attn  # 16 tok x 2 x 8 heads x 64 dim x 1B
    assert conv == 8192, conv  # 2048 x 2 x 2B
    for name, page in (("attention", attn), ("conv-state", conv)):
        assert page % L2_SECTOR == 0, f"{name} page_size_bytes={page} is not {L2_SECTOR}B-aligned"

    # Spec-decode widens the conv state; alignment must survive it (the drafter work in
    # flight makes num_spec > 0 a live possibility, not a hypothetical).
    for num_spec in range(0, 5):
        page = conv_page_size_bytes(CONV_DIM, CONV_L_CACHE, CONV_DTYPE_BYTES, num_spec)
        assert page % L2_SECTOR == 0, (num_spec, page)

    # ...and so must every block size / KV dtype we might plausibly flip to.
    for block_size in (1, 8, 16, 32, 64, 128):
        for dtype_bytes in (1, 2):
            page = attn_page_size_bytes(block_size, NUM_KV_HEADS, HEAD_SIZE, dtype_bytes)
            assert page % L2_SECTOR == 0, (block_size, dtype_bytes, page)

    # Negative control: the assertion above has teeth only if some geometry fails it.
    # (1 head x 40 dim x 1 token x fp8 = 80B -- what a non-power-of-2 head_size would do.)
    assert attn_page_size_bytes(1, 1, 40, 1) % L2_SECTOR != 0


def test_kv_buffer_base_and_block_offsets_are_aligned():
    """The live allocator half: needs CUDA, skips without it."""
    try:
        import torch
    except Exception as exc:
        print(f"  (skip live-allocator check: torch not importable: {exc})")
        return
    if not torch.cuda.is_available():
        print("  (skip live-allocator check: no CUDA device)")
        return

    num_blocks = 64
    for name, page, num_layers in (
        ("attention", attn_page_size_bytes(BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, KV_DTYPE_BYTES), NUM_ATTN_LAYERS),
        ("conv-state", conv_page_size_bytes(CONV_DIM, CONV_L_CACHE, CONV_DTYPE_BYTES), NUM_CONV_LAYERS),
    ):
        # Exactly what _allocate_kv_cache_tensors does: one flat int8 buffer per layer.
        for layer in range(num_layers):
            buf = torch.zeros(num_blocks * page, dtype=torch.int8, device="cuda")
            base = buf.data_ptr()
            assert base % ALLOC_ALIGN == 0, f"{name} layer {layer} base {base:#x}"
            # _reshape_kv_cache_tensors views it as num_blocks pages, page bytes apart.
            for block in (0, 1, num_blocks // 2, num_blocks - 1):
                assert (base + block * page) % L2_SECTOR == 0, (name, layer, block)
            del buf


def main() -> None:
    test_page_sizes_are_l2_aligned()
    test_kv_buffer_base_and_block_offsets_are_aligned()
    print("kv alignment self-check ok")


if __name__ == "__main__":
    main()
