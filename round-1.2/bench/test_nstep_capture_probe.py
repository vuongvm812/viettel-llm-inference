"""C0 -- the three on-box probes the N-step burst's GRAPH mode depends on.

These are go/no-go questions about CUDA graph capture that cannot be answered off-box and
that gate a runtime DEFAULT, never the implementation: ``vtl/patches/nstep_decode.py`` ships
both modes and drops to eager on its own if a capture throws. The probes exist so the answer
is a test result instead of a boot-time surprise, and they stay as regressions.

    pytest bench/test_nstep_capture_probe.py -q      # on the H200, inside the image
    make test-kernel                                  # runs them with everything else

WHAT SHIPS NOW (2026-07-30 follow-up), so the probes are read against the real code:
``nstep_decode`` no longer has just ONE graph per decode size. The in-graph ladder is

    unrolled burst graph (prologue + N-1 bodies + the accumulator store, ONE replay)
      -> prologue graph (``compute_logits`` + ``argmax`` off the persistent
         ``cudagraph_manager.hidden_states`` buffer) + N-1 per-iteration body replays
      -> per-iteration body graphs with token 1 host-launched
      -> eager body

and the PROLOGUE graph is also what ``VTL_SAMPLE_IN_GRAPH=1`` replays on a plain greedy
1-token decode step, in place of the V2 sampler. So probe A is no longer only about the burst:
it is the go/no-go for the prologue graph and therefore for in-graph N=1 sampling too, which
is the one rung that pays when bursts stay dormant. Every rung is captured separately at boot
and every failure demotes to the rung below (``_Burst.demote``), so a FAIL here is still a
performance answer, never a correctness one.

PROBE A -- lm_head + argmax inside a graph, under the vtl W4A8 head.
    The stock FULL decode graph ENDS at hidden states (cudagraph_utils.py:540-556); the
    burst graph has to carry ``compute_logits`` and ``argmax`` too, and ``lm_head_quant``
    replaces that head with a CUTLASS W4A8 GEMM whose capturability nothing else proves.
    FAIL => ``VTL_NSTEP_MODE=eager`` permanently; the P1b full-graph phase is dropped, not
    debugged, and ``VTL_SAMPLE_IN_GRAPH`` is inert with it.

    ``test_probe_a_reads_a_persistent_buffer_not_its_argument`` is the fold-in's specific
    premise: the prologue graph reads a buffer it does NOT own, written by the stock FULL
    replay that ran earlier in the same step. A graph that baked the VALUES rather than the
    address would pass the plain probe A and silently emit last step's tokens.

PROBE B -- FA3 ``get_scheduler_metadata`` inside a graph, with ``max_seqlen_k`` baked.
    The burst body recomputes the AOT schedule every iteration
    (``decode_fastpath._fa_write``). Inside a capture it may only launch kernels: any host
    read of device memory, any host branch on a device value, and the capture dies.
    ``max_seqlen_k`` is baked at ``max_model_len`` because a graph cannot take a host
    scalar. PASS => capture the call directly (what ``_capture_burst_graphs`` does).
    FAIL => slab fallback (host precomputes the N metadata slots before the launch --
    the seq_lens are deterministic, ``seq + i`` -- and the graph does an ``index_select``
    by a device step counter).

PROBE C -- replay-loop overhead.
    Prices the architecture choice: N replays of one body vs one N-unrolled capture.
    Expected ~0.05-0.1 ms for the three extra replays at N=4, i.e. well under the ~1.5 ms
    the burst saves. The unroll is now BUILT (``VTL_NSTEP_UNROLL=1``), so this probe no
    longer gates it -- it sizes what turning it off costs. Reported, never asserted.
"""

from __future__ import annotations

import sys

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

VOCAB = 65536
HIDDEN = 2048
MAX_MODEL_LEN = 32768


def _fa():
    try:
        from vllm.v1.attention.backends.flash_attn import get_scheduler_metadata
    except Exception:
        pytest.skip("FlashAttention varlen scheduler not built into this image")
    return get_scheduler_metadata


# ---------------------------------------------------------------------------------------
# Probe A -- compute_logits + argmax in a graph.
# ---------------------------------------------------------------------------------------


def _head(quantized: bool):
    """The served lm_head shape: W4A8 (what `lm_head_quant` installs) or plain bf16.

    Built from the same primitives the patch's ``apply_weights`` uses -- the per-token fp8
    activation quant plus ``cutlass_w4a8_mm`` -- rather than by instantiating a
    ``ParallelLMHead``, which would drag in the whole model loader. The question the probe
    asks is about the KERNELS' capturability, and those are the kernels.
    """
    if not quantized:
        w = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.02
        return lambda h: torch.nn.functional.linear(h, w)
    try:
        import vllm._C_stable_libtorch  # noqa: F401
        import vtl._C  # noqa: F401
        from vllm import _custom_ops as ops

        from vtl.patches.quant_w4a8 import _quantize_and_pack
    except Exception:
        pytest.skip("vtl W4A8 kernels not available in this build")
    if not hasattr(ops, "cutlass_w4a8_mm"):
        pytest.skip("cutlass_w4a8_mm not built into this image")
    from vtl.patches.lm_head_quant import _quant_act_fp8
    from vtl.patches.quant_w4a8 import GROUP_SIZE

    w = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.02
    packed, group_scales, chan_scales = _quantize_and_pack(w)

    def head(h):
        q, s = _quant_act_fp8(h)
        return ops.cutlass_w4a8_mm(
            a=q,
            b_q=packed,
            b_group_scales=group_scales,
            b_group_size=GROUP_SIZE,
            a_token_scales=s,
            b_channel_scales=chan_scales,
        )

    return head


@pytest.mark.parametrize("m", [1, 8])
@pytest.mark.parametrize("quantized", [False, True])
def test_probe_a_logits_argmax_capture_is_bit_equal_to_eager(m, quantized):
    head = _head(quantized)
    hidden = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros(m, dtype=torch.int64, device="cuda")

    def body():
        torch.argmax(head(hidden), dim=-1, out=out)

    body()  # warm up allocations / autotuning outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()

    # Replay four times with DIFFERENT inputs -- a graph that baked the result rather than
    # the computation passes a single replay and fails here.
    for seed in range(4):
        torch.manual_seed(seed)
        hidden.copy_(torch.randn_like(hidden))
        want = torch.argmax(head(hidden), dim=-1)
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, want), f"replay {seed} diverged from eager"


@pytest.mark.parametrize("m", [1, 8])
def test_probe_a_reads_a_persistent_buffer_not_its_argument(m):
    """The token-1 fold-in / in-graph N=1 sampling premise, isolated.

    ``_prologue`` captures ``argmax(compute_logits(cudagraph_manager.hidden_states[:rows]))``
    -- a buffer written by the stock FULL replay EARLIER IN THE SAME STEP, not by anything
    inside our graph. So the graph must have baked the ADDRESS. Overwriting the buffer between
    replays and checking the tokens move is the whole test; ``nstep_decode``'s boot-time
    ``_graph_matches_eager`` shadow is the runtime version of it.
    """
    head = _head(quantized=False)
    # Stand-in for `ModelCudaGraphManager.hidden_states`: allocated once, sliced per step.
    persistent = torch.zeros(16, HIDDEN, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros(m, dtype=torch.int64, device="cuda")

    def prologue():
        torch.argmax(head(persistent[:m]), dim=-1, out=out)

    prologue()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prologue()

    seen = set()
    for seed in range(4):
        torch.manual_seed(seed)
        persistent[:m].copy_(torch.randn(m, HIDDEN, dtype=torch.bfloat16, device="cuda"))
        want = torch.argmax(head(persistent[:m]), dim=-1)
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, want), (
            "PROBE A2 FAILED: the prologue graph did not track the persistent buffer; "
            "VTL_NSTEP_FOLD_T1 and VTL_SAMPLE_IN_GRAPH must stay 0"
        )
        seen.add(tuple(out.tolist()))
    assert len(seen) > 1, "the probe's inputs did not actually change the argmax"


# ---------------------------------------------------------------------------------------
# Probe B -- FA3 AOT schedule in a graph.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("m", [1, 8])
def test_probe_b_scheduler_metadata_is_capturable(m):
    get_scheduler_metadata = _fa()
    seq_lens = torch.full((m,), 128, dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.arange(m + 1, dtype=torch.int32, device="cuda")
    kw = dict(
        batch_size=m,
        max_seqlen_q=1,
        # BAKED: a graph cannot take a host scalar, so the schedule is built for the
        # worst case. Same trick mamba_hybrid.py:247-251 uses for capture.
        max_seqlen_k=MAX_MODEL_LEN,
        num_heads_q=32,
        num_heads_kv=8,
        headdim=64,
        cache_seqlens=seq_lens,
        qkv_dtype=torch.bfloat16,
        cu_seqlens_q=cu_seqlens_q,
        page_size=16,
        causal=True,
        window_size=(-1, -1),
    )
    fresh = get_scheduler_metadata(**kw)
    if fresh is None:
        pytest.skip("AOT scheduler returned None for this shape")
    dst = torch.zeros_like(fresh)

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            dst.copy_(get_scheduler_metadata(**kw))
    except Exception as exc:  # noqa: BLE001 -- this IS the probe's answer
        pytest.fail(
            "PROBE B FAILED: get_scheduler_metadata is not capturable "
            f"({exc!r}). nstep graph mode needs the slab fallback; run with "
            "VTL_NSTEP_MODE=eager until it exists."
        )

    # Mutate seq_lens ON DEVICE and replay: the schedule must track them, not the capture.
    seq_lens.fill_(4096)
    want = get_scheduler_metadata(**kw)
    dst.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(dst, want), (
        "PROBE B FAILED: the captured schedule ignored a device-side seq_lens change"
    )


# ---------------------------------------------------------------------------------------
# Probe C -- replay-loop overhead.
# ---------------------------------------------------------------------------------------


def test_probe_c_replay_loop_overhead_is_reported():
    """3 extra replays vs 1, at a decode-sized body. Reported, not asserted."""
    a = torch.randn(8, HIDDEN, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(HIDDEN, HIDDEN, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(a)

    def body():
        torch.mm(a, b, out=out)

    body()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()

    def timed(replays, reps=200):
        for _ in range(20):
            for _ in range(replays):
                graph.replay()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(reps):
            for _ in range(replays):
                graph.replay()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / reps

    one, four = timed(1), timed(4)
    per_extra = (four - one) / 3
    print(
        f"\nPROBE C: 1 replay {one * 1000:.1f} us, 4 replays {four * 1000:.1f} us "
        f"-> {per_extra * 1000:.1f} us per extra replay "
        f"({(four - one):.3f} ms for the N=4 loop)"
    )
    assert four > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
