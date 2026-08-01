"""End-to-end parity + microbench for the fused short-conv decode megakernel.

The oracle is THE REAL CHAIN the kernel replaces, run op by op on cloned inputs:

    per-token fp8 quant  ->  ops.cutlass_w4a8_mm (in_proj)
      ->  vllm_cuda::bcx_conv_gate_quant       ->  ops.cutlass_w4a8_mm (out_proj)

Both the OUTPUT and the POST-KERNEL CONV STATE are compared: the ring-buffer rotation is the
half that fails silently -- a wrong state produces plausible tokens that drift over a whole
generation, not a NaN.

Tolerance is relative and generous ON PURPOSE. The megakernel accumulates fp8 x int4 in fp32
one 64-wide half-group at a time; CUTLASS accumulates inside an MMA in its own tile order.
`bench/test_w4a8_gemv.py::test_reference_gemv_tracks_cutlass` measures that gap directly --
if THIS file's tolerance ever has to grow, check that one first, because a real bug and a
reordering both look like "slightly off".

The barrier gets its own test: a graph-capture smoke (the hand-rolled barrier must survive
`torch.cuda.graph`, which is the entire reason it is not `cg::grid.sync()`), and a
repeat-replay check, because a barrier that leaks its generation counter across launches
fails on the SECOND replay, never the first.

    pytest bench/test_shortconv_decode_mega.py -q
    python bench/test_shortconv_decode_mega.py --bench    # chain vs mega, graph-replayed
"""

from __future__ import annotations

import sys

import pytest

torch = pytest.importorskip("torch")

DIM = 2048
WIDTH, STATE_LEN = 3, 2
NULL_BLOCK_ID = 0
GROUP = 128


def _have_op() -> bool:
    try:
        import vllm._C_stable_libtorch  # noqa: F401
        import vtl._C  # noqa: F401
    except Exception:
        return False
    return hasattr(torch.ops.vllm_cuda, "shortconv_decode_mega")


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not _have_op(), reason="vtl._C shortconv_decode_mega unavailable"),
]


def make_case(M, seed=0, num_blocks=16):
    from vtl.patches.shortconv_mega import build_mega_weights

    torch.manual_seed(seed)
    dev = "cuda"
    x = torch.randn(M, DIM, dtype=torch.bfloat16, device=dev)
    in_w = torch.randn(3 * DIM, DIM, dtype=torch.bfloat16, device=dev) * 0.02
    out_w = torch.randn(DIM, DIM, dtype=torch.bfloat16, device=dev) * 0.02
    conv_w = torch.randn(DIM, WIDTH, dtype=torch.bfloat16, device=dev) * 0.1
    conv_b = torch.randn(DIM, dtype=torch.bfloat16, device=dev) * 0.1
    # SD layout: allocated (blocks, state_len, dim) then transposed, so stride(1) == 1.
    state = torch.randn(num_blocks, STATE_LEN, DIM, dtype=torch.bfloat16, device=dev)
    state = state.transpose(-1, -2)
    idx = torch.arange(1, M + 1, dtype=torch.int32, device=dev).reshape(M, 1)
    return dict(
        x=x, in_w=in_w, out_w=out_w, conv_w=conv_w, conv_b=conv_b, state=state, idx=idx,
        in_mega=build_mega_weights(in_w), out_mega=build_mega_weights(out_w),
    )


def quant_per_token(t):
    scale = (t.float().abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1.0 / (448 * 512))
    return (t.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale


def run_chain(c):
    """The four stock ops, on a cloned state so the mega run starts from the same place."""
    from vllm import _custom_ops as ops

    from vtl.patches.quant_w4a8 import _quantize_and_pack

    x_fp8, x_scale = quant_per_token(c["x"])
    in_p, in_gs, in_cs = _quantize_and_pack(c["in_w"])
    bcx = ops.cutlass_w4a8_mm(
        a=x_fp8, b_q=in_p, b_group_scales=in_gs, b_group_size=GROUP,
        a_token_scales=x_scale, b_channel_scales=in_cs,
    )
    state = c["state"].clone()
    M = c["x"].shape[0]
    y_fp8 = torch.empty(M, DIM, dtype=torch.float8_e4m3fn, device="cuda")
    y_scale = torch.empty(M, 1, dtype=torch.float32, device="cuda")
    torch.ops.vllm_cuda.bcx_conv_gate_quant(
        y_fp8, y_scale, state, bcx, c["conv_w"], c["conv_b"], c["idx"], NULL_BLOCK_ID, None
    )
    out_p, out_gs, out_cs = _quantize_and_pack(c["out_w"])
    out = ops.cutlass_w4a8_mm(
        a=y_fp8, b_q=out_p, b_group_scales=out_gs, b_group_size=GROUP,
        a_token_scales=y_scale, b_channel_scales=out_cs,
    )
    return out, state


def run_mega(c, state=None):
    from vtl.patches.shortconv_mega import arm

    M = c["x"].shape[0]
    st = arm(DIM, torch.device("cuda"))
    if st is None:
        pytest.skip("megakernel refused to arm on this device (occupancy too low)")
    state = c["state"].clone() if state is None else state
    out = torch.zeros(M, DIM, dtype=torch.bfloat16, device="cuda")
    torch.ops.vllm_cuda.shortconv_decode_mega(
        out, c["x"], None, 0.0, c["in_mega"][0], c["in_mega"][1], c["conv_w"], c["conv_b"],
        state, c["idx"], NULL_BLOCK_ID, c["out_mega"][0], c["out_mega"][1],
        st["scratch"], st["grid"],
    )
    return out, state


@pytest.mark.parametrize("M", [1, 2, 4, 8])
def test_matches_the_stock_chain(M):
    c = make_case(M, seed=M)
    want, want_state = run_chain(c)
    got, got_state = run_mega(c)

    rel = (got.float() - want.float()).abs().max() / want.float().abs().max().clamp(min=1e-6)
    assert rel < 5e-2, f"M={M}: output {rel:.4f} from the chain"
    # The conv state must be BIT-exact: the rotation is the same code in both, and a
    # tolerance here would hide a ring-buffer bug that only shows up after 200 decode steps.
    assert torch.equal(got_state, want_state), f"M={M}: conv state diverged"


def test_null_blocks_leave_state_untouched():
    c = make_case(4, seed=99)
    c["idx"][1, 0] = NULL_BLOCK_ID  # a cudagraph padding row
    before = c["state"].clone()
    out, state = run_mega(c)
    # Row 1 is padding: its output is defined zeros and NO block's state may have moved
    # except the three live rows'.
    assert torch.equal(state[NULL_BLOCK_ID], before[NULL_BLOCK_ID])
    assert torch.count_nonzero(out[1]) == 0


def test_survives_cudagraph_capture_and_repeat_replay():
    """The barrier's whole justification. A generation counter that leaks across launches
    passes the first replay and hangs or corrupts on the second, so replay twice."""
    from vtl.patches.shortconv_mega import arm

    c = make_case(4, seed=7)
    st = arm(DIM, torch.device("cuda"))
    if st is None:
        pytest.skip("megakernel refused to arm on this device")
    state = c["state"].clone()
    out = torch.zeros(4, DIM, dtype=torch.bfloat16, device="cuda")

    def fire():
        torch.ops.vllm_cuda.shortconv_decode_mega(
            out, c["x"], None, 0.0, c["in_mega"][0], c["in_mega"][1], c["conv_w"],
            c["conv_b"], state, c["idx"], NULL_BLOCK_ID, c["out_mega"][0],
            c["out_mega"][1], st["scratch"], st["grid"],
        )

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fire()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fire()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()


def test_refuses_an_oversized_batch():
    c = make_case(8, seed=3)
    assert not torch.ops.vllm_cuda.shortconv_decode_mega_supported(DIM, WIDTH, STATE_LEN, 16)
    assert torch.ops.vllm_cuda.shortconv_decode_mega_supported(DIM, WIDTH, STATE_LEN, 8)
    # Wrong conv geometry is refused too -- the phase-3 tap select is width-3 only.
    assert not torch.ops.vllm_cuda.shortconv_decode_mega_supported(DIM, 4, 3, 4)
    del c


def bench():  # pragma: no cover -- run by hand on the GPU box
    import time

    print(f"{'M':>3} {'chain us':>10} {'mega us':>10} {'speedup':>8}")
    for M in (1, 2, 4, 8):
        c = make_case(M, seed=M)
        for fn in (run_chain, run_mega):
            for _ in range(5):
                fn(c)
        torch.cuda.synchronize()
        times = []
        for fn in (run_chain, run_mega):
            t0 = time.perf_counter()
            for _ in range(100):
                fn(c)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) / 100 * 1e6)
        print(f"{M:>3} {times[0]:10.2f} {times[1]:10.2f} {times[0] / times[1]:8.2f}x")


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench()
    else:
        raise SystemExit(pytest.main([__file__, "-q"]))
