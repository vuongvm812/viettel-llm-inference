"""Fused greedy-argmax harness: gating/identity off-box, real compile + parity on a GPU.

    python3 bench/test_greedy_argmax.py --self-check   # no GPU, no torch, no vLLM
    pytest -q bench/test_greedy_argmax.py              # GPU: compiles and checks the tokens

Same split and doctrine as bench/test_nvrtc_block_quant.py. The GPU half is the one that
matters, and "close" is not a grade here: the kernel picks the token that gets STREAMED, and
nstep_decode's boot-time ``_graph_matches_eager`` bit-compares the burst graph's tokens
against an eager step and demotes the whole burst on ONE differing token. So every GPU test
below asserts ``torch.equal`` against ``torch.argmax``, and the adversarial rows are exactly
the places a hand-written reduction and torch can legally disagree:

  * a tie -- torch.argmax returns the FIRST maximal index, and an int4 lm_head produces
    exact bf16 ties far more often than fp32 intuition suggests (bf16 has 8 mantissa bits);
  * NaN -- torch treats it as the MAXIMUM, not as "skip";
  * an all-``-inf`` row, and the ``step0_eos_ban`` pattern (-inf written into the EOS
    column IN PLACE, before the sampler ever sees the tensor);
  * the ends of the row, where an off-by-one in the vector loop hides.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vtl import nvrtc  # noqa: E402
from vtl.patches import greedy_argmax as patch  # noqa: E402

# `make check` runs --self-check on a bare host with neither pytest nor torch; see the
# note in bench/test_nvrtc.py. Both stay optional and only the pytest half requires them.
try:
    import pytest

    HAVE_PYTEST = True
except ImportError:  # pragma: no cover -- the `make check` path
    HAVE_PYTEST = False

    class _NoPytest:
        class mark:
            @staticmethod
            def skipif(*a, **k):
                return lambda fn: fn

            @staticmethod
            def parametrize(*a, **k):
                return lambda fn: fn

    pytest = _NoPytest()

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

requires_gpu = pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(), reason="needs torch + a CUDA device"
)

VOCAB = 248320         # Qwen3.5 vocab_size (hf-model/config.json); 8 * 31040, so vectorized
THREADS = patch.DEFAULT_THREADS
EOS_ID = 248044        # VTL_STEP0_EOS_ID in docker-compose.yaml


# --------------------------------------------------------------------------------------
# compile helpers
# --------------------------------------------------------------------------------------

def _source():
    src = nvrtc.load_source(patch.KERNEL)
    assert src, "vtl/kernels/greedy_argmax.cu must ship with the package"
    return src


def _compile(vocab: int = VOCAB, threads: int = THREADS):
    """Compile the shipped source for one specialization, or skip with the reason."""
    built = nvrtc.build_cubin(_source(), patch.KERNEL, patch._defines(vocab, threads))
    if built is None:
        pytest.skip("NVRTC unavailable (no cuda-python / no driver) -- torch.argmax stands")
    return nvrtc._load_cubin(built[0], patch.ENTRY)


def _run(launch, logits, out=None, threads: int = THREADS, rows: int | None = None):
    """Launch the kernel over ``logits`` and return the int64 tokens."""
    rows = int(logits.shape[0]) if rows is None else rows
    if out is None:
        # Poisoned, not empty: a kernel that never writes must not "match" a zeroed buffer.
        out = torch.full((int(logits.shape[0]),), -1, dtype=torch.int64, device="cuda")
    launch(
        grid=(rows, 1, 1),
        block=(threads, 1, 1),
        args=nvrtc.pack_args(logits.data_ptr(), out.data_ptr(), ("i", rows)),
    )
    torch.cuda.synchronize()
    return out


def _rows(n: int, vocab: int = VOCAB, seed: int = 0):
    torch.manual_seed(seed)
    return torch.randn(n, vocab, dtype=torch.bfloat16, device="cuda")


# --------------------------------------------------------------------------------------
# GPU half
# --------------------------------------------------------------------------------------

@requires_gpu
@pytest.mark.parametrize("rows", [1, 2, 4, 5, 8, 16, 32])
def test_matches_torch_argmax_bit_for_bit(rows):
    """The served batch sizes: <=5 eager (max-num-seqs), {1,2,4,8,16,32} captured."""
    from cuda.bindings import driver  # noqa: F401  (skip early if absent)

    logits = _rows(rows, seed=rows)
    got = _run(_compile(), logits)
    assert torch.equal(got, torch.argmax(logits, dim=-1)), (rows, got)


@requires_gpu
@pytest.mark.parametrize("threads", [256, 512, 1024])
def test_block_width_does_not_change_the_token(threads):
    """THREADS is a tuning knob, not a semantic one: the cross-warp tree must be exact for
    every legal block width (8, 16 and 32 warps -- i.e. the whole `kWarps` range)."""
    from cuda.bindings import driver  # noqa: F401

    logits = _rows(5, seed=7)
    got = _run(_compile(threads=threads), logits, threads=threads)
    assert torch.equal(got, torch.argmax(logits, dim=-1)), threads


@requires_gpu
@pytest.mark.parametrize("pos", [0, 1, 7, 8, VOCAB // 2, VOCAB - 2, VOCAB - 1])
def test_the_max_is_found_at_every_edge_of_the_row(pos):
    """One clear winner, walked across the vector-loop boundaries: index 0 and VOCAB-1 are
    where a stride/tail bug lives, 7 and 8 straddle the first 16-byte load."""
    from cuda.bindings import driver  # noqa: F401

    logits = torch.full((3, VOCAB), -1.0, dtype=torch.bfloat16, device="cuda")
    logits[:, pos] = 1.0
    got = _run(_compile(), logits)
    assert torch.equal(got, torch.full((3,), pos, dtype=torch.int64, device="cuda"))
    assert torch.equal(got, torch.argmax(logits, dim=-1))


@requires_gpu
def test_an_all_equal_row_breaks_the_tie_to_the_lowest_index():
    """Every element identical: torch returns 0, and so must every warp of ours. This is
    the test that catches a reduction whose tie-break depends on lane order."""
    from cuda.bindings import driver  # noqa: F401

    for fill in (0.0, 1.0, -1.0, float("-inf")):
        logits = torch.full((4, VOCAB), fill, dtype=torch.bfloat16, device="cuda")
        got = _run(_compile(), logits)
        assert torch.equal(got, torch.zeros(4, dtype=torch.int64, device="cuda")), fill
        assert torch.equal(got, torch.argmax(logits, dim=-1)), fill


@requires_gpu
def test_duplicate_maxima_resolve_to_the_first():
    """Ties AT the max, spread across warps and across the block's stride, so the winner
    can only be right if the tie-break survives both reduction levels."""
    from cuda.bindings import driver  # noqa: F401

    logits = torch.full((1, VOCAB), -3.0, dtype=torch.bfloat16, device="cuda")
    for i in (137, 4096, 65536, VOCAB - 1):
        logits[0, i] = 2.5
    got = _run(_compile(), logits)
    assert int(got[0]) == 137, got
    assert torch.equal(got, torch.argmax(logits, dim=-1))


@requires_gpu
def test_infinities_and_nan_follow_torch():
    """+inf beats everything finite; NaN beats +inf (torch treats NaN as the maximum);
    -inf everywhere else must not become a winner."""
    from cuda.bindings import driver  # noqa: F401

    launch = _compile()

    pinf = torch.full((2, VOCAB), float("-inf"), dtype=torch.bfloat16, device="cuda")
    pinf[0, 900] = float("inf")
    pinf[1, VOCAB - 1] = float("inf")
    got = _run(launch, pinf)
    assert torch.equal(got, torch.argmax(pinf, dim=-1)), got
    assert got.tolist() == [900, VOCAB - 1], got

    nan = torch.randn(3, VOCAB, dtype=torch.bfloat16, device="cuda")
    nan[0, 12345] = float("nan")
    nan[1, 0] = float("nan")
    nan[1, VOCAB - 1] = float("inf")       # NaN must still win
    nan[2, VOCAB - 3] = float("nan")
    got = _run(launch, nan)
    assert torch.equal(got, torch.argmax(nan, dim=-1)), got
    assert got.tolist() == [12345, 0, VOCAB - 3], got

    # Several NaN in one row: a tie at the top, so the FIRST one wins.
    many = torch.randn(1, VOCAB, dtype=torch.bfloat16, device="cuda")
    for i in (31, 8192, VOCAB - 9):
        many[0, i] = float("nan")
    got = _run(launch, many)
    assert torch.equal(got, torch.argmax(many, dim=-1)), got


@requires_gpu
def test_the_step0_eos_ban_pattern():
    """step0_eos_ban writes -inf into the EOS column IN PLACE, before the sampler runs, so
    the kernel just reads a mutated tensor -- nothing is folded in. What it must never do
    is pick the banned column back."""
    from cuda.bindings import driver  # noqa: F401

    logits = _rows(5, seed=11)
    logits[:, EOS_ID] = 100.0                      # make EOS the argmax...
    assert torch.equal(torch.argmax(logits, dim=-1),
                       torch.full((5,), EOS_ID, dtype=torch.int64, device="cuda"))
    logits[:, EOS_ID] = float("-inf")              # ...then ban it, the way the patch does
    got = _run(_compile(), logits)
    assert torch.equal(got, torch.argmax(logits, dim=-1))
    assert (got != EOS_ID).all(), got


@requires_gpu
def test_out_variant_writes_a_prepoisoned_buffer_in_place():
    """The nstep shape: a persistent int64 buffer, sliced, written without allocating. The
    slice beyond ``rows`` must be left exactly as it was -- nstep keeps live state there."""
    from cuda.bindings import driver  # noqa: F401

    launch = _compile()
    rows, cap = 5, 8
    logits = _rows(rows, seed=13)
    buf = torch.full((cap,), -7, dtype=torch.int64, device="cuda")
    ptr = buf.data_ptr()

    _run(launch, logits, out=buf[:rows], rows=rows)

    assert buf.data_ptr() == ptr, "the out variant must not reallocate"
    assert torch.equal(buf[:rows], torch.argmax(logits, dim=-1))
    assert torch.equal(buf[rows:], torch.full((cap - rows,), -7, dtype=torch.int64,
                                              device="cuda")), buf


@requires_gpu
def test_rows_argument_bounds_a_fixed_grid():
    """A caller may launch a grid wider than the real batch (a captured graph baked the
    padded width). Rows past ``rows`` must not be written."""
    from cuda.bindings import driver  # noqa: F401

    logits = _rows(8, seed=17)
    out = torch.full((8,), -1, dtype=torch.int64, device="cuda")
    _run(_compile(), logits, out=out, rows=3)      # grid is 3, buffer is 8
    assert torch.equal(out[:3], torch.argmax(logits[:3], dim=-1))
    assert (out[3:] == -1).all(), out


@requires_gpu
def test_capturable_into_a_cuda_graph_and_replayable_on_new_data():
    """The whole point of the OUT variant: nstep captures it inside the burst graph, and
    the Rust runner replays that same exec. Capture once, then rewrite the INPUT buffer in
    place and replay -- the tokens must track the new data, not the captured one."""
    from cuda.bindings import driver  # noqa: F401

    launch = _compile()
    rows = 4
    logits = _rows(rows, seed=19)                  # the persistent input buffer
    out = torch.full((rows,), -1, dtype=torch.int64, device="cuda")

    # Warm-up on a side stream, the way torch's own graph capture requires.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _run(launch, logits, out=out, rows=rows)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(
            grid=(rows, 1, 1),
            block=(THREADS, 1, 1),
            args=nvrtc.pack_args(logits.data_ptr(), out.data_ptr(), ("i", rows)),
        )

    for seed in (21, 23):
        logits.copy_(_rows(rows, seed=seed))
        out.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, torch.argmax(logits, dim=-1)), seed


@requires_gpu
def test_a_vocab_the_vector_path_cannot_serve_still_works():
    """VOCAB % 8 != 0 takes the kernel's scalar path -- slower, still exact. It is not
    reachable for this checkpoint; it is what keeps the kernel honest for another one."""
    from cuda.bindings import driver  # noqa: F401

    odd = 1237
    logits = torch.randn(3, odd, dtype=torch.bfloat16, device="cuda")
    got = _run(_compile(vocab=odd), logits)
    assert torch.equal(got, torch.argmax(logits, dim=-1)), got


@requires_gpu
def test_op_impls_agree_with_the_kernel_and_fall_back_outside_the_envelope():
    """The two registered ops are what the sampler and nstep actually call. Held to
    torch.argmax directly, including on the shapes that must DEGRADE rather than launch."""
    from cuda.bindings import driver  # noqa: F401

    launch = _compile()
    patch._state.update(vocab=VOCAB, threads=THREADS, launch=launch)
    try:
        logits = _rows(4, seed=29)
        assert torch.equal(patch._argmax_impl(logits), torch.argmax(logits, dim=-1))

        out = torch.full((4,), -1, dtype=torch.int64, device="cuda")
        patch._argmax_out_impl(logits, out)
        torch.cuda.synchronize()
        assert torch.equal(out, torch.argmax(logits, dim=-1))

        # Empty batch: no launch (grid 0 is illegal), no crash, right shape.
        empty = torch.empty(0, VOCAB, dtype=torch.bfloat16, device="cuda")
        assert patch._argmax_impl(empty).shape == (0,)

        # Outside the envelope -> torch.argmax inline, not a wrong token and not a raise.
        assert patch._logits_reason(logits.float()) == "dtype"
        assert torch.equal(patch._argmax_impl(logits.float()),
                           torch.argmax(logits.float(), dim=-1))
        narrow = _rows(4, seed=31)[:, : VOCAB - 8]
        assert patch._logits_reason(narrow) == "stride"
        assert torch.equal(patch._argmax_impl(narrow), torch.argmax(narrow, dim=-1))
        wide = torch.randn(2, VOCAB // 2, dtype=torch.bfloat16, device="cuda")
        assert patch._logits_reason(wide) == "vocab"
        assert torch.equal(patch._argmax_impl(wide), torch.argmax(wide, dim=-1))

        bad_out = torch.full((4,), -1, dtype=torch.int32, device="cuda")
        assert patch._out_reason(bad_out, 4) == "out-dtype"
    finally:
        patch._state.update(vocab=0, threads=0, launch=None)


# --------------------------------------------------------------------------------------
# no-GPU half
# --------------------------------------------------------------------------------------

def _self_check() -> None:
    import os

    src = _source()

    # -- names line up: source file <-> compile_kernel(name=...) <-> the entry symbol <->
    #    the op the forked sampler resolves <-> the op nstep's rebind calls --
    assert patch.KERNEL == "greedy_argmax"
    assert patch.ENTRY == "greedy_argmax_i64"
    assert f'extern "C" __global__ void __launch_bounds__(THREADS) {patch.ENTRY}(' in src
    assert patch.OP == patch.ENTRY, "the op and the cubin entry share one name on purpose"
    assert patch.OP_OUT == patch.OP + "_out"
    assert patch.NS == "vllm_cuda"

    # The forked V2 sampler resolves this exact attribute path (v2_greedy_sampler.patch);
    # if either half of the name drifts, the resolver silently keeps torch.argmax forever.
    fork = Path(__file__).resolve().parent.parent / (
        "vtl/vllm_patches/v0.25.0/v2_greedy_sampler.patch")
    if fork.is_file():
        text = fork.read_text()
        assert f"torch.ops.{patch.NS}.{patch.OP}" in text, "the fork resolves another name"
        assert "VTL_V2_GREEDY_ARGMAX_KERNEL" in text

    # nstep's indirection must exist and default to plain torch.argmax, or the rebind has
    # nothing to bind and the burst graphs capture the wrong thing.
    nstep = Path(__file__).resolve().parent.parent / "vtl/patches/nstep_decode.py"
    ns_src = nstep.read_text()
    assert "_ARGMAX = _argmax_out" in ns_src
    assert ns_src.count("_ARGMAX(") == 3, "all three token picks must go through _ARGMAX"
    assert "torch.argmax(logits, dim=-1, out=out)" in ns_src, "the default must be torch's"

    # -- VOCAB has no defensible default; the source must refuse it. THREADS does. --
    assert '#error "NVRTC: -DVOCAB=<vocab_size> is required"' in src
    for macro in ("VOCAB", "THREADS"):
        assert f"#ifndef {macro}" in src, f"-D{macro} is not guarded in the source"
    assert f"#define THREADS {patch.DEFAULT_THREADS}" in src, "kernel/patch default drift"

    # -- the kernel must not be compiled with fast math, and must not pull torch headers --
    assert "--use_fast_math" not in nvrtc.BASE_FLAGS
    assert "torch" not in src.split("#include")[1].split("\n")[0]

    # -- one cubin identity per specialization. A collision means a warm cache serves a
    #    cubin built for ANOTHER vocab, which indexes off the end of the row. --
    sets = [
        patch._defines(VOCAB, 256),
        patch._defines(VOCAB, 512),
        patch._defines(VOCAB, 1024),
        patch._defines(151936, 512),
        patch._defines(1237, 512),
    ]
    keys = {nvrtc.cache_key(src, s, "90a", "12.8") for s in sets}
    assert len(keys) == len(sets), "define sets must not collide in the cubin cache"
    # ...arch and toolkit are part of the identity too, and dict order is not.
    assert nvrtc.cache_key(src, sets[1], "90", "12.8") not in keys
    assert nvrtc.cache_key(src, sets[1], "90a", "12.9") not in keys
    perm = dict(reversed(list(sets[1].items())))
    assert nvrtc.cache_key(src, perm, "90a", "12.8") == nvrtc.cache_key(
        src, sets[1], "90a", "12.8")

    # -- the shipped vocab takes the vectorized path; the guarded one still compiles --
    assert VOCAB % patch.VEC == 0, "the served checkpoint must reach the 16-byte loads"
    assert patch._vocab_ok(VOCAB) and patch._vocab_ok(1237)
    assert patch._vocab_ok(0) is False

    # -- block width: the default ships; nonsense never reaches NVRTC --
    os.environ.pop("VTL_GREEDY_ARGMAX_THREADS", None)
    assert patch._threads_from_env() == THREADS == 512
    os.environ["VTL_GREEDY_ARGMAX_THREADS"] = "1024"
    assert patch._threads_from_env() == 1024
    os.environ["VTL_GREEDY_ARGMAX_THREADS"] = "48"       # not a whole number of warps
    assert patch._threads_from_env() == THREADS
    os.environ.pop("VTL_GREEDY_ARGMAX_THREADS")

    # -- import-without-vLLM: the module loads, registers, and stays OFF by default --
    from vtl.registry import PATCH_REGISTRY, is_enabled

    p = next(x for x in PATCH_REGISTRY if x.name == "greedy_argmax")
    assert p.default is False and is_enabled(p) is False

    # With VTL_NVRTC off, apply() must return cleanly even though vLLM is absent -- nothing
    # compiled, no op registered, so the fork resolver and nstep both keep torch.argmax.
    os.environ.pop("VTL_NVRTC", None)
    assert nvrtc.compile_kernel(patch.KERNEL, sets[1], entry=patch.ENTRY) is None
    patch.apply()
    assert patch._state["installed"] is False
    assert patch._state["launch"] is None

    # With VTL_NVRTC on but vLLM absent, apply() may raise -- registry.apply_all isolates
    # it -- but it must not leave the module half-installed.
    os.environ["VTL_NVRTC"] = "1"
    try:
        patch.apply()
    except Exception:
        pass
    finally:
        os.environ.pop("VTL_NVRTC", None)
    assert patch._state["installed"] is False
    assert patch._state["launch"] is None

    # -- the nstep rebind is import-safe: with nstep_decode unimported (it needs vLLM) this
    #    must report "nothing to rebind" rather than dragging the import failure in here --
    if "vtl.patches.nstep_decode" not in sys.modules:
        assert patch._rebind_nstep() is False

    if torch is None:
        print("test_greedy_argmax self-check ok (torch absent; parity not exercised)")
        return

    # -- the semantics the GPU half asserts, stated once on CPU so a torch upgrade that
    #    changed them would be caught here rather than as a mystery burst demotion --
    ties = torch.tensor([[1.0, 2.0, 2.0, 0.0]], dtype=torch.bfloat16)
    assert int(torch.argmax(ties, dim=-1)) == 1, "ties must go to the FIRST maximum"
    nan = torch.tensor([[1.0, float("nan"), float("inf")]], dtype=torch.bfloat16)
    assert int(torch.argmax(nan, dim=-1)) == 1, "NaN must outrank +inf"
    allneg = torch.full((1, 8), float("-inf"), dtype=torch.bfloat16)
    assert int(torch.argmax(allneg, dim=-1)) == 0, "a fully masked row must pick index 0"

    print("test_greedy_argmax self-check ok")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    elif not HAVE_PYTEST:
        raise SystemExit("pytest not installed; run with --self-check")
    else:
        raise SystemExit(pytest.main([__file__, "-q"]))
