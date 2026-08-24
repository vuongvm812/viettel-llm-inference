"""Fused greedy-argmax harness: gating/identity off-box, real compile + parity on a GPU.

    python3 bench/test_greedy_argmax.py --self-check   # no GPU, no torch, no vLLM
    pytest -q bench/test_greedy_argmax.py              # GPU: compiles and checks the tokens
    VTL_BENCH=1 python3 bench/test_greedy_argmax.py    # GPU: us/call vs torch, eager + graph

Same split and doctrine as bench/test_nvrtc_block_quant.py. The GPU half is the one that
matters, and "close" is not a grade here: the kernel picks the token that gets STREAMED.

THIS FILE AND THE BOOT PARITY GATE (``greedy_argmax._boot_parity_ok``) ARE THE PARITY
ORACLE -- between them, nothing else checks the kernel against torch. In particular
nstep_decode's ``_graph_matches_eager`` does NOT: once ``_ARGMAX`` is rebound to the fused
op, both of its arms run this kernel, so it compares fused against fused and is a
pointer-staleness smoke test for graph capture only. So every GPU test below asserts
``torch.equal`` against ``torch.argmax``, and the adversarial rows are exactly the places a
hand-written reduction and torch can legally disagree:

  * a tie -- torch.argmax returns the FIRST maximal index, and an int4 lm_head produces
    exact bf16 ties far more often than fp32 intuition suggests (bf16 has 8 mantissa bits);
  * NaN -- torch treats it as the MAXIMUM, not as "skip", and NaN-vs-NaN is a tie;
  * ``-0.0`` against ``+0.0`` -- IEEE says they are EQUAL, so torch takes the lower index
    while the raw bit patterns do not;
  * an all-``-inf`` row, and the ``step0_eos_ban`` pattern (-inf written into the EOS
    column IN PLACE, before the sampler ever sees the tensor);
  * the ends of the row, where an off-by-one in the vector loop or a split-slice boundary
    hides.
"""

from __future__ import annotations

import os
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

        @staticmethod
        def fixture(*a, **k):
            return (lambda fn: fn) if not a else a[0]

        @staticmethod
        def skip(*a, **k):
            raise SystemExit("pytest not installed")

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
SPLIT = patch.DEFAULT_SPLIT
EDGE = patch.EDGE_THREADS
EOS_ID = 248044        # VTL_STEP0_EOS_ID in docker-compose.yaml


# --------------------------------------------------------------------------------------
# fixtures: keep the GPU half hermetic and the cubin cache out of the image's cache dir
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _nvrtc_cache_in_tmp(tmp_path_factory):
    """Point the cubin cache at a throwaway dir for the session.

    The default is ``/opt/vtl/cache/nvrtc``, which `make warm` harvests INTO THE IMAGE. A
    test run that compiles a dozen throwaway specializations there ships them all.
    """
    prev = os.environ.get("VTL_NVRTC_CACHE")
    os.environ["VTL_NVRTC_CACHE"] = str(tmp_path_factory.mktemp("nvrtc-cache"))
    yield
    if prev is None:
        os.environ.pop("VTL_NVRTC_CACHE", None)
    else:
        os.environ["VTL_NVRTC_CACHE"] = prev


@pytest.fixture(autouse=True)
def _clean_module_state():
    """Per-test teardown. The module has three pieces of process-global state a test can
    leave dirty: the warn-once set (a later test's warning would be swallowed), the launch
    latch, and whatever graphs/allocations the test made."""
    yield
    patch._warned.clear()
    patch._state["disabled"] = False
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# compile helpers
# --------------------------------------------------------------------------------------

def _source():
    src = nvrtc.load_source(patch.KERNEL)
    assert src, "vtl/kernels/greedy_argmax.cu must ship with the package"
    return src


# Module-scoped so a parametrized sweep compiles each specialization ONCE. Keyed by exactly
# what the cubin identity is keyed by.
_LAUNCHERS: dict = {}


def _compile(vocab: int = VOCAB, threads: int = THREADS, split: int = SPLIT):
    """Compile the shipped source for one specialization, or skip with the reason.

    Returns the three launchers in launch order: (reset, partial, finalize).
    """
    key = (vocab, threads, split)
    if key not in _LAUNCHERS:
        built = nvrtc.build_cubin(_source(), patch.KERNEL, patch._defines(vocab, threads, split))
        if built is None:
            pytest.skip("NVRTC unavailable (no cuda-python / no driver) -- torch.argmax stands")
        _LAUNCHERS[key] = tuple(nvrtc._load_cubin(built[0], e) for e in patch.ENTRIES)
    return _LAUNCHERS[key]


def _ws(cap: int = 64):
    """The persistent u64 workspace. Poisoned, not zeroed: the reset kernel must do it."""
    return torch.full((cap,), -1, dtype=torch.int64, device="cuda")


def _run(launchers, logits, out=None, threads: int = THREADS, split: int = SPLIT,
         rows: int | None = None, grid_rows: int | None = None, ws=None):
    """reset -> partial -> finalize over ``logits``; returns the int64 tokens.

    ``grid_rows`` launches the partial kernel's grid WIDER than ``rows``, which is what a
    captured graph does for a padded batch; the kernel's own ``row >= rows`` guard is the
    only thing keeping those blocks off the workspace.
    """
    reset, partial, finalize = launchers
    rows = int(logits.shape[0]) if rows is None else rows
    if out is None:
        # Poisoned, not empty: a kernel that never writes must not "match" a zeroed buffer.
        out = torch.full((int(logits.shape[0]),), -1, dtype=torch.int64, device="cuda")
    if ws is None:
        ws = _ws(max(rows, grid_rows or rows) + 8)
    reset(grid=(1, 1, 1), block=(EDGE, 1, 1),
          args=nvrtc.pack_args(ws.data_ptr(), ("i", rows)))
    partial(grid=(split, grid_rows or rows, 1), block=(threads, 1, 1),
            args=nvrtc.pack_args(logits.data_ptr(), ws.data_ptr(), ("i", rows)))
    finalize(grid=(1, 1, 1), block=(EDGE, 1, 1),
             args=nvrtc.pack_args(ws.data_ptr(), out.data_ptr(), ("i", rows)))
    torch.cuda.synchronize()
    return out


def _rows(n: int, vocab: int = VOCAB, seed: int = 0):
    torch.manual_seed(seed)
    return torch.randn(n, vocab, dtype=torch.bfloat16, device="cuda")


def _arm_state(launchers, vocab: int = VOCAB, threads: int = THREADS, split: int = SPLIT):
    """Put the module into the state ``_arm`` would leave, without vLLM or a model config.

    Returns a callable that puts it back. Everything ``_launch`` reads is built here through
    the module's OWN helper, so a test exercises the real cached-args path rather than a
    second copy of it.
    """
    saved = dict(patch._state)
    patch._build_launch_state(vocab, threads, split, launchers)
    patch._state.update(vocab=vocab)

    def restore():
        patch._state.clear()
        patch._state.update(saved)

    return restore


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
@pytest.mark.parametrize("split", [1, 2, 3, 16, 64])
def test_split_does_not_change_the_token(split):
    """SPLIT is the other tuning knob, and the riskier one: it changes how the row is cut
    into slices, so a boundary bug or a non-associative combine shows up as a different
    token. 3 and 64 are there because they do NOT divide NVEC evenly."""
    from cuda.bindings import driver  # noqa: F401

    logits = _rows(4, seed=split)
    got = _run(_compile(split=split), logits, split=split)
    assert torch.equal(got, torch.argmax(logits, dim=-1)), split


@requires_gpu
@pytest.mark.parametrize("pos", [0, 1, 7, 8, VOCAB // SPLIT, VOCAB // 2, VOCAB - 2, VOCAB - 1])
def test_the_max_is_found_at_every_edge_of_the_row(pos):
    """One clear winner, walked across the loop boundaries: index 0 and VOCAB-1 are where a
    stride/tail bug lives, 7 and 8 straddle the first 16-byte load, and VOCAB//SPLIT is the
    first slice seam."""
    from cuda.bindings import driver  # noqa: F401

    logits = torch.full((3, VOCAB), -1.0, dtype=torch.bfloat16, device="cuda")
    logits[:, pos] = 1.0
    got = _run(_compile(), logits)
    assert torch.equal(got, torch.full((3,), pos, dtype=torch.int64, device="cuda"))
    assert torch.equal(got, torch.argmax(logits, dim=-1))


@requires_gpu
def test_an_all_equal_row_breaks_the_tie_to_the_lowest_index():
    """Every element identical: torch returns 0, and so must every warp of ours. This is
    the test that catches a reduction whose tie-break depends on lane order -- and, for the
    ``-inf`` fill, the one that catches a scan whose identity IS ``-inf`` and therefore
    never records the real ``-inf`` it read."""
    from cuda.bindings import driver  # noqa: F401

    for fill in (0.0, 1.0, -1.0, float("-inf"), float("inf")):
        logits = torch.full((4, VOCAB), fill, dtype=torch.bfloat16, device="cuda")
        got = _run(_compile(), logits)
        assert torch.equal(got, torch.zeros(4, dtype=torch.int64, device="cuda")), fill
        assert torch.equal(got, torch.argmax(logits, dim=-1)), fill


@requires_gpu
def test_signed_zero_is_a_tie_not_an_ordering():
    """IEEE says ``-0.0 == +0.0``, so torch calls it a TIE and takes the lower index. The
    raw bit patterns disagree (0x8000 > 0x0000 as an unsigned), which is exactly why the
    kernel's key map canonicalizes ``-0.0`` before it packs. Both orders, plus an
    alternating row, so a canonicalization in the wrong direction fails too."""
    from cuda.bindings import driver  # noqa: F401

    launch = _compile()
    for first, second in ((-0.0, 0.0), (0.0, -0.0)):
        logits = torch.full((1, VOCAB), -1.0, dtype=torch.bfloat16, device="cuda")
        logits[0, 3] = first
        logits[0, VOCAB // 2] = second
        logits[0, VOCAB - 1] = first
        got = _run(launch, logits)
        assert int(got[0]) == 3, (first, second, got)
        assert torch.equal(got, torch.argmax(logits, dim=-1))

    alt = torch.zeros(1, VOCAB, dtype=torch.bfloat16, device="cuda")
    alt[0, 1::2] = -0.0
    got = _run(launch, alt)
    assert torch.equal(got, torch.argmax(alt, dim=-1)), got
    assert int(got[0]) == 0, got

    allneg0 = torch.full((2, VOCAB), -0.0, dtype=torch.bfloat16, device="cuda")
    got = _run(launch, allneg0)
    assert torch.equal(got, torch.argmax(allneg0, dim=-1)), got


@requires_gpu
def test_duplicate_maxima_resolve_to_the_first():
    """Ties AT the max, spread across warps, across the block's stride AND across the split
    slices, so the winner can only be right if the tie-break survives all three levels."""
    from cuda.bindings import driver  # noqa: F401

    logits = torch.full((1, VOCAB), -3.0, dtype=torch.bfloat16, device="cuda")
    for i in (0, 137, 4096, 65536, VOCAB // 2, VOCAB - 1):
        logits[0, i] = 2.5
    got = _run(_compile(), logits)
    assert int(got[0]) == 0, got
    assert torch.equal(got, torch.argmax(logits, dim=-1))

    logits[0, 0] = -3.0
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

    nan = _rows(3, seed=41)
    nan[0, 12345] = float("nan")
    nan[1, 0] = float("nan")
    nan[1, VOCAB - 1] = float("inf")       # NaN must still win
    nan[2, VOCAB - 3] = float("nan")
    got = _run(launch, nan)
    assert torch.equal(got, torch.argmax(nan, dim=-1)), got
    assert got.tolist() == [12345, 0, VOCAB - 3], got

    # Several NaN in one row: a tie at the top, so the FIRST one wins. Seeded through
    # `_rows` so a rerun sees the same row and a failure is reproducible.
    many = _rows(1, seed=43)
    for i in (31, 8192, VOCAB - 9):
        many[0, i] = float("nan")
    got = _run(launch, many)
    assert torch.equal(got, torch.argmax(many, dim=-1)), got
    assert int(got[0]) == 31, got

    # A whole row of NaN: still a tie, still index 0.
    allnan = torch.full((2, VOCAB), float("nan"), dtype=torch.bfloat16, device="cuda")
    got = _run(launch, allnan)
    assert torch.equal(got, torch.argmax(allnan, dim=-1)), got


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
    """A caller may launch a grid WIDER than the real batch (a captured graph baked the
    padded width). The partial kernel's ``row >= rows`` guard is the only thing stopping
    those blocks, so the grid here is 8 while ``rows`` is 3 -- with an equal grid the guard
    never executes and the test proves nothing."""
    from cuda.bindings import driver  # noqa: F401

    logits = _rows(8, seed=17)
    out = torch.full((8,), -1, dtype=torch.int64, device="cuda")
    ws = _ws(16)
    _run(_compile(), logits, out=out, rows=3, grid_rows=8, ws=ws)
    assert torch.equal(out[:3], torch.argmax(logits[:3], dim=-1))
    assert (out[3:] == -1).all(), out
    # The guarded blocks must not have touched the workspace either -- the poison stands.
    assert (ws[3:] == -1).all(), ws


@requires_gpu
def test_capturable_into_a_cuda_graph_and_replayable_on_new_data():
    """The whole point of the OUT variant: nstep captures it inside the burst graph, and
    the Rust runner replays that same exec. All THREE launches must be capture-legal (the
    reset is what makes the atomicMax accumulator reusable across replays). Capture once,
    then rewrite the INPUT buffer in place and replay -- the tokens must track the new data,
    not the captured one."""
    from cuda.bindings import driver  # noqa: F401

    reset, partial, finalize = _compile()
    rows = 4
    logits = _rows(rows, seed=19)                  # the persistent input buffer
    out = torch.full((rows,), -1, dtype=torch.int64, device="cuda")
    ws = _ws(16)
    a_reset = nvrtc.pack_args(ws.data_ptr(), ("i", rows))
    a_partial = nvrtc.pack_args(logits.data_ptr(), ws.data_ptr(), ("i", rows))
    a_final = nvrtc.pack_args(ws.data_ptr(), out.data_ptr(), ("i", rows))

    def work():
        reset(grid=(1, 1, 1), block=(EDGE, 1, 1), args=a_reset)
        partial(grid=(SPLIT, rows, 1), block=(THREADS, 1, 1), args=a_partial)
        finalize(grid=(1, 1, 1), block=(EDGE, 1, 1), args=a_final)

    # Warm-up on a side stream, the way torch's own graph capture requires.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        work()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        work()

    for seed in (21, 23, 21):
        logits.copy_(_rows(rows, seed=seed))
        out.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, torch.argmax(logits, dim=-1)), seed
    del graph


@requires_gpu
def test_a_vocab_the_vector_path_cannot_serve_still_works():
    """VOCAB % 8 != 0 takes the kernel's scalar path -- slower, still exact. It is not
    reachable for this checkpoint; it is what keeps the kernel honest for another one."""
    from cuda.bindings import driver  # noqa: F401

    odd = 1237
    logits = torch.randn(3, odd, dtype=torch.bfloat16, device="cuda")
    got = _run(_compile(vocab=odd), logits)
    assert torch.equal(got, torch.argmax(logits, dim=-1)), got
    # ...including the fully-masked row, where the scalar path has its own -inf fixup.
    masked = torch.full((2, odd), float("-inf"), dtype=torch.bfloat16, device="cuda")
    assert torch.equal(_run(_compile(vocab=odd), masked),
                       torch.zeros(2, dtype=torch.int64, device="cuda"))


@requires_gpu
def test_the_compiler_refuses_the_specializations_it_should():
    """Negative compiles. A vocab-less build would index off the end of the row, and a
    block width that is not a whole number of warps would drop the tail warp -- both must
    fail at NVRTC, not at the first launch."""
    from cuda.bindings import driver  # noqa: F401

    src = _source()
    if nvrtc.build_cubin(src, patch.KERNEL, patch._defines(VOCAB, THREADS, SPLIT)) is None:
        pytest.skip("NVRTC unavailable -- nothing to refuse")
    assert nvrtc.build_cubin(src, patch.KERNEL, {"THREADS": THREADS, "SPLIT": SPLIT}) is None
    assert nvrtc.build_cubin(src, patch.KERNEL, patch._defines(VOCAB, 48, SPLIT)) is None
    assert nvrtc.build_cubin(src, patch.KERNEL, patch._defines(VOCAB, THREADS, 0)) is None
    assert nvrtc.build_cubin(src, patch.KERNEL, patch._defines(0, THREADS, SPLIT)) is None


@requires_gpu
def test_op_impls_agree_with_the_kernel_and_fall_back_outside_the_envelope():
    """The two op impls are what the sampler and nstep actually call. Held to torch.argmax
    directly, including on the shapes that must DEGRADE rather than launch."""
    from cuda.bindings import driver  # noqa: F401

    restore = _arm_state(_compile())
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
        # A batch wider than the persistent workspace: the atomicMax target is fixed-size,
        # so this must be refused rather than scribbled past.
        assert patch._logits_reason(
            torch.empty(patch.MAX_WS_ROWS + 1, VOCAB, dtype=torch.bfloat16,
                        device="cuda")) == "rows"
        # A 0-d tensor: `_logits_reason` must answer BEFORE anyone reads shape[0].
        assert patch._logits_reason(torch.tensor(1.0, device="cuda")) == "dtype"
        scalar_bf16 = torch.tensor(1.0, dtype=torch.bfloat16, device="cuda")
        assert patch._logits_reason(scalar_bf16) == "rank"
        patch._argmax_out_impl(scalar_bf16, torch.zeros(1, dtype=torch.int64, device="cuda"))
    finally:
        restore()


@requires_gpu
def test_a_misaligned_view_is_refused_on_the_vector_path():
    """A 16-byte load off a 2-byte-aligned base FAULTS -- it is not a slow path. So a view
    whose data pointer is not 16-byte aligned has to be caught in Python. Only on the vector
    specialization: a scalar one needs bf16's own 2 bytes and nothing more."""
    from cuda.bindings import driver  # noqa: F401

    flat = torch.randn(4 * VOCAB + 8, dtype=torch.bfloat16, device="cuda")
    view = flat[1: 1 + 4 * VOCAB].view(4, VOCAB)
    assert view.data_ptr() % patch.ALIGN, "the test needs an actually-misaligned base"

    restore = _arm_state(_compile())
    try:
        assert patch._logits_reason(view) == "align"
        assert torch.equal(patch._argmax_impl(view), torch.argmax(view, dim=-1))
    finally:
        restore()

    # The scalar specialization does not care: `vec` is False, so the check is skipped.
    odd = 1237
    restore = _arm_state(_compile(vocab=odd), vocab=odd)
    try:
        assert patch._state["vec"] is False
        oflat = torch.randn(3 * odd + 8, dtype=torch.bfloat16, device="cuda")
        oview = oflat[1: 1 + 3 * odd].view(3, odd)
        assert patch._logits_reason(oview) is None
        assert torch.equal(patch._argmax_impl(oview), torch.argmax(oview, dim=-1))
    finally:
        restore()


@requires_gpu
def test_the_out_fallback_branches_write_the_right_answer_without_raising():
    """The two ``out`` shapes that send us to torch are exactly the two that make
    ``torch.argmax(..., out=out)`` RAISE -- an int32 out (dtype refused) and a length
    mismatch (resize-of-a-view refused). The degrade must produce the right tokens anyway."""
    from cuda.bindings import driver  # noqa: F401

    restore = _arm_state(_compile())
    try:
        logits = _rows(4, seed=37)
        want = torch.argmax(logits, dim=-1)

        i32 = torch.full((4,), -1, dtype=torch.int32, device="cuda")
        assert patch._out_reason(i32, 4) == "out-dtype"
        patch._argmax_out_impl(logits, i32)
        torch.cuda.synchronize()
        assert torch.equal(i32, want.to(torch.int32)), i32

        longer = torch.full((7,), -1, dtype=torch.int64, device="cuda")
        assert patch._out_reason(longer, 4) == "out-shape"
        patch._argmax_out_impl(logits, longer)
        torch.cuda.synchronize()
        assert torch.equal(longer[:4], want), longer
        assert (longer[4:] == -1).all(), "the degrade must not scribble past the answer"

        strided = torch.full((8,), -1, dtype=torch.int64, device="cuda")[::2]
        assert patch._out_reason(strided, 4) == "out-stride"
        patch._argmax_out_impl(logits, strided)
        torch.cuda.synchronize()
        assert torch.equal(strided, want), strided
    finally:
        restore()


@requires_gpu
def test_a_launch_failure_latches_the_kernel_off_instead_of_crashing_every_step():
    """Both consumers resolve the impl ONCE and call it bare, so an impl that raises kills
    the engine step -- and then the next one, forever. A driver error must degrade to
    torch.argmax for the rest of the run and still return this step's answer."""
    from cuda.bindings import driver  # noqa: F401

    restore = _arm_state(_compile())
    try:
        def boom(*a, **k):
            raise RuntimeError("cuda driver error 700")

        patch._state["partial"] = boom
        logits = _rows(3, seed=47)
        want = torch.argmax(logits, dim=-1)

        assert torch.equal(patch._argmax_impl(logits), want)
        assert patch._state["disabled"] is True

        out = torch.full((3,), -1, dtype=torch.int64, device="cuda")
        patch._argmax_out_impl(logits, out)     # already latched: straight to torch
        torch.cuda.synchronize()
        assert torch.equal(out, want)
    finally:
        restore()
        patch._state["disabled"] = False


@requires_gpu
def test_through_the_dispatcher():
    """What the forked V2 sampler resolves and what an external caller sees: the REGISTERED
    ops, not the Python impls. Registration must be idempotent (a re-apply must not drop the
    Library and deregister the live ops), the ops must resolve by name, produce torch's
    tokens, and the out variant must still capture."""
    from cuda.bindings import driver  # noqa: F401

    restore = _arm_state(_compile())
    try:
        assert patch._register_ops() is True
        assert patch._register_ops() is True, "a second call must be a no-op, not a rebuild"
        ops = getattr(torch.ops, patch.NS)
        op = getattr(ops, patch.OP)
        op_out = getattr(ops, patch.OP_OUT)
        assert patch._state["op_out"] is op_out.default, "the overload is resolved once"

        logits = _rows(5, seed=53)
        want = torch.argmax(logits, dim=-1)
        assert torch.equal(op(logits), want)

        out = torch.full((5,), -1, dtype=torch.int64, device="cuda")
        op_out(logits, out)
        torch.cuda.synchronize()
        assert torch.equal(out, want)

        # ...and through the op, inside a graph.
        rows = 4
        glogits = _rows(rows, seed=59)
        gout = torch.full((rows,), -1, dtype=torch.int64, device="cuda")
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            op_out(glogits, gout)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op_out(glogits, gout)
        for seed in (61, 67):
            glogits.copy_(_rows(rows, seed=seed))
            gout.fill_(-1)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(gout, torch.argmax(glogits, dim=-1)), seed
        del graph
    finally:
        restore()


@requires_gpu
def test_the_boot_parity_gate_installs_on_a_good_kernel_and_refuses_a_bad_one():
    """The gate is the only thing between a subtly-wrong kernel and every streamed token,
    so it gets its own test in both directions."""
    from cuda.bindings import driver  # noqa: F401

    _compile()   # skip early if NVRTC is unavailable

    class _Config:
        def get_vocab_size(self):
            return VOCAB

    saved = dict(patch._state)
    # `_arm` rebinds nstep's `_ARGMAX` if nstep_decode is imported in this process, and that
    # would outlive the test. Put it back.
    nstep = sys.modules.get("vtl.patches.nstep_decode")
    saved_argmax = getattr(nstep, "_ARGMAX", None)
    try:
        # -- the good direction: a real compile, the gate passes, both ops appear --
        patch._state.update(armed=False, installed=False, disabled=False)
        os.environ["VTL_NVRTC"] = "1"
        os.environ["VTL_GREEDY_ARGMAX_SPLIT"] = str(SPLIT)
        os.environ["VTL_GREEDY_ARGMAX_THREADS"] = str(THREADS)
        patch._arm(_Config())
        assert patch._state["installed"] is True, "a correct kernel must arm"
        assert patch._state["vocab"] == VOCAB
        ops = getattr(torch.ops, patch.NS)
        assert hasattr(ops, patch.OP) and hasattr(ops, patch.OP_OUT)

        # -- the bad direction: the same arming path with a POISONED partial kernel (one
        #    that launches nothing), so the workspace keeps the reset identity and finalize
        #    unpacks 0xffffffff for every row. Nothing may be published. --
        patch._state.update(armed=False, installed=False, vocab=0)

        def _no_op_partial(**kwargs):
            return None

        real = nvrtc.compile_kernel

        def poisoned(name, defines, entry=None):
            k = real(name, defines, entry=entry)
            return _no_op_partial if entry == patch.ENTRY_PARTIAL else k

        nvrtc.compile_kernel = poisoned
        try:
            patch._arm(_Config())
        finally:
            nvrtc.compile_kernel = real
        assert patch._state["installed"] is False, "a wrong kernel must NOT arm"
        assert patch._state["vocab"] == 0, "the envelope must be shut"
        assert patch._state["partial"] is None
    finally:
        os.environ.pop("VTL_NVRTC", None)
        os.environ.pop("VTL_GREEDY_ARGMAX_SPLIT", None)
        os.environ.pop("VTL_GREEDY_ARGMAX_THREADS", None)
        patch._state.clear()
        patch._state.update(saved)
        if nstep is not None and saved_argmax is not None:
            nstep._ARGMAX = saved_argmax


# --------------------------------------------------------------------------------------
# no-GPU half
# --------------------------------------------------------------------------------------

def _self_check() -> None:
    import types

    src = _source()

    # -- names line up: source file <-> compile_kernel(name=...) <-> the entry symbols <->
    #    the op the forked sampler resolves <-> the op nstep's rebind calls --
    assert patch.KERNEL == "greedy_argmax"
    assert patch.ENTRIES == (patch.ENTRY_RESET, patch.ENTRY_PARTIAL, patch.ENTRY_FINALIZE)
    for entry in patch.ENTRIES:
        assert f" {entry}(" in src, entry
    assert f'extern "C" __global__ void __launch_bounds__(THREADS) {patch.ENTRY_PARTIAL}(' in src
    assert patch.OP == "greedy_argmax_i64"
    assert patch.OP_OUT == patch.OP + "_out"
    assert patch.NS == "vllm_cuda"

    # The forked V2 sampler resolves this exact attribute path (v2_greedy_sampler.patch);
    # if either half of the name drifts, the resolver silently keeps torch.argmax forever.
    root = Path(nvrtc.__file__).resolve().parent
    fork = root / "vllm_patches/v0.25.0/v2_greedy_sampler.patch"
    if fork.is_file():
        text = fork.read_text()
        assert f"torch.ops.{patch.NS}.{patch.OP}" in text, "the fork resolves another name"
        assert "VTL_V2_GREEDY_ARGMAX_KERNEL" in text

    # nstep's indirection must exist and default to plain torch.argmax, or the rebind has
    # nothing to bind and the burst graphs capture the wrong thing.
    nstep = root / "patches/nstep_decode.py"
    if nstep.is_file():
        ns_src = nstep.read_text()
        assert "_ARGMAX = _argmax_out" in ns_src
        assert ns_src.count("_ARGMAX(") == 3, "all three token picks must go through _ARGMAX"
        assert "torch.argmax(logits, dim=-1, out=out)" in ns_src, "the default must be torch's"
        # ...and exactly ONE place calls it. A second call site would be a token pick that
        # bypasses `_ARGMAX` and therefore never gets the fused kernel (or, worse, keeps
        # torch's while the rest of the burst uses ours). The CALL form on purpose: the
        # module also names `torch.argmax` in prose, and prose is not a call site.
        assert ns_src.count("torch.argmax(") == 1, "one torch.argmax call site, behind _ARGMAX"

    # -- VOCAB has no defensible default; the source must refuse it. THREADS/SPLIT do. --
    assert '#error "NVRTC: -DVOCAB=<vocab_size> is required"' in src
    for macro in ("VOCAB", "THREADS", "SPLIT"):
        assert f"#ifndef {macro}" in src, f"-D{macro} is not guarded in the source"
    assert f"#define THREADS {patch.DEFAULT_THREADS}" in src, "kernel/patch THREADS drift"
    assert f"#define SPLIT {patch.DEFAULT_SPLIT}" in src, "kernel/patch SPLIT drift"

    # -- the kernel must not be compiled with fast math (which would license dropping the
    #    `v + 0.0f` that canonicalizes -0.0), and must not pull torch headers. EVERY
    #    #include, not just the first: a torch header three lines down is still fatal. --
    assert "--use_fast_math" not in nvrtc.BASE_FLAGS
    assert "v + 0.0f" in src, "the -0.0 canonicalization is load-bearing; see the key proof"
    includes = [ln.strip() for ln in src.splitlines() if ln.strip().startswith("#include")]
    assert includes, "the kernel includes nothing at all?"
    for line in includes:
        assert "torch" not in line and "ATen" not in line, line

    # -- one cubin identity per specialization. A collision means a warm cache serves a
    #    cubin built for ANOTHER vocab, which indexes off the end of the row. --
    sets = [
        patch._defines(VOCAB, 256, 16),
        patch._defines(VOCAB, 512, 16),
        patch._defines(VOCAB, 1024, 16),
        patch._defines(VOCAB, 256, 1),
        patch._defines(151936, 512, 16),
        patch._defines(1237, 512, 16),
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

    # -- block width / split: the defaults ship; nonsense never reaches NVRTC --
    os.environ.pop("VTL_GREEDY_ARGMAX_THREADS", None)
    assert patch._threads_from_env() == THREADS == 256
    os.environ["VTL_GREEDY_ARGMAX_THREADS"] = "1024"
    assert patch._threads_from_env() == 1024
    os.environ["VTL_GREEDY_ARGMAX_THREADS"] = "48"       # not a whole number of warps
    assert patch._threads_from_env() == THREADS
    os.environ.pop("VTL_GREEDY_ARGMAX_THREADS")
    os.environ.pop("VTL_GREEDY_ARGMAX_SPLIT", None)
    assert patch._split_from_env() == SPLIT == 16
    os.environ["VTL_GREEDY_ARGMAX_SPLIT"] = "64"
    assert patch._split_from_env() == 64
    os.environ["VTL_GREEDY_ARGMAX_SPLIT"] = "0"
    assert patch._split_from_env() == SPLIT
    os.environ.pop("VTL_GREEDY_ARGMAX_SPLIT")

    # -- the vocab reader survives all three config shapes it meets in the wild --
    class _Works:
        def get_vocab_size(self):
            return VOCAB

    class _RaisesWithFallback:
        hf_text_config = types.SimpleNamespace(vocab_size=151936)

        def get_vocab_size(self):
            raise AttributeError("moved")

    class _Neither:
        pass

    assert patch._vocab_from(_Works()) == VOCAB
    assert patch._vocab_from(_RaisesWithFallback()) == 151936
    assert patch._vocab_from(_Neither()) == 0
    assert patch._vocab_ok(patch._vocab_from(_Neither())) is False

    # -- import-without-vLLM: the module loads and registers ON by default --
    # Enabled by default per project decision (2026-08-17); the compose gate now restates the
    # code default. Default-on is safe because the boot parity gate, not the registry gate, is
    # the safety story: the op is registered only after it matches torch.argmax bit for bit,
    # and every consumer resolves through _ARGMAX with torch.argmax as the fallback rung.
    from vtl.registry import PATCH_REGISTRY, is_enabled

    p = next(x for x in PATCH_REGISTRY if x.name == "greedy_argmax")
    assert p.default is True and is_enabled(p) is True

    from vtl import patches as _patches

    assert "greedy_argmax" in _patches._MODULES, "the patch must be in the apply list"

    # With VTL_NVRTC off, apply() must return cleanly even though vLLM is absent -- nothing
    # compiled, no op registered, so the fork resolver and nstep both keep torch.argmax.
    os.environ.pop("VTL_NVRTC", None)
    assert nvrtc.compile_kernel(patch.KERNEL, sets[1], entry=patch.ENTRY_PARTIAL) is None
    patch.apply()
    assert patch._state["installed"] is False
    assert patch._state["partial"] is None

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
    assert patch._state["partial"] is None
    assert patch._state["vocab"] == 0, "a half-arm must leave the envelope shut"
    assert patch._state["disabled"] is False

    # -- the nstep rebind is import-safe: with nstep_decode unimported (it needs vLLM) this
    #    must report "nothing to rebind" rather than dragging the import failure in here --
    if "vtl.patches.nstep_decode" not in sys.modules:
        assert patch._rebind_nstep() is False

    # ...and it lands on the module global, as the IMPL function rather than the dispatcher.
    stub = types.ModuleType("vtl.patches.nstep_decode")
    stub._ARGMAX = object()
    real = sys.modules.get("vtl.patches.nstep_decode")
    sys.modules["vtl.patches.nstep_decode"] = stub
    try:
        assert patch._rebind_nstep() is True
        assert stub._ARGMAX is patch._argmax_out_impl
    finally:
        if real is None:
            del sys.modules["vtl.patches.nstep_decode"]
        else:
            sys.modules["vtl.patches.nstep_decode"] = real

    if torch is None:
        print("test_greedy_argmax self-check ok (torch absent; parity not exercised)")
        return

    # -- the semantics the GPU half asserts, stated once on CPU so a torch upgrade that
    #    changed them would be caught here rather than as a mystery burst demotion --
    ties = torch.tensor([[1.0, 2.0, 2.0, 0.0]], dtype=torch.bfloat16)
    assert int(torch.argmax(ties, dim=-1)) == 1, "ties must go to the FIRST maximum"
    nan = torch.tensor([[1.0, float("nan"), float("inf")]], dtype=torch.bfloat16)
    assert int(torch.argmax(nan, dim=-1)) == 1, "NaN must outrank +inf"
    nans = torch.tensor([[1.0, float("nan"), float("nan")]], dtype=torch.bfloat16)
    assert int(torch.argmax(nans, dim=-1)) == 1, "NaN vs NaN is a tie -> the lower index"
    allneg = torch.full((1, 8), float("-inf"), dtype=torch.bfloat16)
    assert int(torch.argmax(allneg, dim=-1)) == 0, "a fully masked row must pick index 0"
    zeros = torch.tensor([[-1.0, -0.0, 0.0, -0.0]], dtype=torch.bfloat16)
    assert int(torch.argmax(zeros, dim=-1)) == 1, "-0.0 == +0.0, so the tie goes to index 1"

    print("test_greedy_argmax self-check ok")


# --------------------------------------------------------------------------------------
# bench arm
# --------------------------------------------------------------------------------------

def _bench():  # pragma: no cover -- GPU-only, run by hand
    """us/call against torch.argmax, EAGER and inside a replayed CUDA graph.

    Both arms matter and they say different things. Eager is where the Python/dispatch
    envelope shows up, which is most of what this patch buys. In a graph the host cost is
    zero for both, so the graph column is a pure kernel-time comparison -- and it is the one
    that says whether SPLIT is set sensibly, because that is the number nstep's burst pays.
    """
    if torch is None or not torch.cuda.is_available():
        print("test_greedy_argmax: skipped -- needs torch + a CUDA device")
        return
    import statistics

    try:
        launchers = _compile()
    except SystemExit:
        print("test_greedy_argmax: skipped -- NVRTC unavailable")
        return
    reset, partial, finalize = launchers

    def timed(fn):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(5):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(50):
                fn()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) / 50 * 1e3)
        return statistics.median(samples)

    print(f"device: {torch.cuda.get_device_name()}  VOCAB={VOCAB} "
          f"THREADS={THREADS} SPLIT={SPLIT}")
    print(f"\n{'rows':>5} {'fused-eager':>12} {'torch-eager':>12} "
          f"{'fused-graph':>12} {'torch-graph':>12}   (us/call)")

    for rows in (1, 2, 4, 5, 8, 16, 32):
        logits = _rows(rows, seed=rows)
        out = torch.empty(rows, dtype=torch.int64, device="cuda")
        ws = torch.zeros(rows + 8, dtype=torch.int64, device="cuda")
        a_reset = nvrtc.pack_args(ws.data_ptr(), ("i", rows))
        a_partial = nvrtc.pack_args(logits.data_ptr(), ws.data_ptr(), ("i", rows))
        a_final = nvrtc.pack_args(ws.data_ptr(), out.data_ptr(), ("i", rows))

        def fused():
            reset(grid=(1, 1, 1), block=(EDGE, 1, 1), args=a_reset)
            partial(grid=(SPLIT, rows, 1), block=(THREADS, 1, 1), args=a_partial)
            finalize(grid=(1, 1, 1), block=(EDGE, 1, 1), args=a_final)

        def stock():
            torch.argmax(logits, dim=-1, out=out)

        us_fused, us_stock = timed(fused), timed(stock)

        graphs = []
        for work in (fused, stock):
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                work()
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                work()
            graphs.append(g)
        gf, gs = (timed(g.replay) for g in graphs)

        print(f"{rows:>5} {us_fused:>12.1f} {us_stock:>12.1f} {gf:>12.1f} {gs:>12.1f}")
        del graphs
        torch.cuda.synchronize()


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    elif os.environ.get("VTL_BENCH", "").strip().lower() in ("1", "true", "yes", "on"):
        _bench()
    elif not HAVE_PYTEST:
        raise SystemExit("pytest not installed; run with --self-check")
    else:
        raise SystemExit(pytest.main([__file__, "-q"]))
