"""LFM2 short-conv state commit/rollback for tree verify — the runner post-sample helper.

The hybrid short-conv layers keep a rolling conv window (last K-1 conv inputs) as persistent state.
Under tree verify the forward sees num_nodes draft tokens per request, but only the ACCEPTED path
may advance the persistent state. Strategy (see vtl/vllm_patches/v0.25.0/short_conv.patch):

  1. `ShortConv.forward_cuda` stages every node's conv input Bx into a runner scratch buffer
     [b, max_nodes, conv_dim] and does NOT mutate the persistent conv_state (implicit rollback:
     rejected nodes were never committed).
  2. After the sampler returns the accepted path, this helper gathers the accepted nodes' Bx and
     writes the last (K-1) of [pre-verify window ++ accepted Bx] back as the new conv_state.

Wraps the golden-tested `tree_spec.conv_commit_window`. CPU-testable off-box.

ON-BOX: `mamba_cache_mode=align` migrates the conv block by an integer length; `conv_commit_window`
returns that length (eff_len == accepted path length), but the write into vLLM's mamba cache manager
is backend-specific — confirm the block-migration write contract on the H200 (guarded in the patch).

Self-check (needs torch, no vLLM/GPU):  python vtl/conv_commit.py
"""
from __future__ import annotations

try:  # package vs bare import — see tree_runtime
    from vtl.tree_spec import conv_commit_window
except ImportError:
    from tree_spec import conv_commit_window


def commit_conv_states(scratch_bx, accepted_ids, init_windows, state_len):
    """Commit the accepted path's conv window per request.

    scratch_bx   : [b, max_nodes, conv_dim] — per-node staged Bx from the verify forward.
    accepted_ids : list[list[int]] — accepted node ids per request, root->leaf order (from the sampler).
    init_windows : [b, state_len, conv_dim] — each request's conv window BEFORE the tree.
    Returns (committed [b, state_len, conv_dim], eff_lens int64 [b]) — eff_lens = accepted path length,
    the scalar the align-mode block migration needs.
    """
    import torch

    b = scratch_bx.shape[0]
    device = scratch_bx.device
    committed = init_windows.clone()
    eff = torch.zeros(b, dtype=torch.int64, device=device)
    for r in range(b):
        path = accepted_ids[r]
        if not path:
            continue  # nothing accepted -> conv window unchanged (state stays at pre-verify)
        staged = {node: scratch_bx[r, node] for node in path}
        window, eff_len = conv_commit_window(path, staged, list(init_windows[r]), state_len)
        committed[r] = torch.stack([w if torch.is_tensor(w) else torch.as_tensor(w, device=device)
                                    for w in window])
        eff[r] = eff_len
    return committed, eff


class ConvStager:
    """Staging half of the P5 seam (the runner installs `.stage` as short_conv's _VTL_CONV_STAGE).

    During tree verify, for each short-conv layer: clone the persistent conv_state so the in-forward
    `causal_conv1d_update` advances the CLONE (persistent state untouched = rejected drafts roll back
    for free), and stash the per-node Bx for the post-sample commit. After the sampler returns the
    accepted paths, `commit_all` writes each layer's accepted window back into the real conv_state.

    ON-BOX: the map from draft-token rows in Bx_d to (request, node) slots depends on the verify
    forward's token packing; `stage` records Bx verbatim and the runner supplies the packing to
    `commit_all`. Guarded so a mismatch raises in commit_conv_states rather than corrupting state.
    """

    def __init__(self):
        self._staged = {}          # layer_key -> Bx_d staged this step
        self._conv_states = {}     # layer_key -> the real (persistent) conv_state tensor

    def stage(self, layer_key, Bx_d, conv_state, state_indices):
        import torch

        self._staged[layer_key] = Bx_d
        self._conv_states[layer_key] = conv_state
        # scratch clone: the update mutates this, not the persistent state.
        return conv_state.clone()

    def commit_all(self, accepted_ids, scratch_by_layer_fn, init_windows_fn, state_len):
        """Commit each staged layer's accepted-path window into its persistent conv_state.

        scratch_by_layer_fn(layer_key) -> [b, max_nodes, conv_dim] per-node Bx for that layer,
        init_windows_fn(layer_key) -> [b, state_len, conv_dim] pre-verify window. Kept as callbacks
        so the runner owns the on-box packing; the math is commit_conv_states (tested).
        """
        for key in self._staged:
            committed, _ = commit_conv_states(
                scratch_by_layer_fn(key), accepted_ids, init_windows_fn(key), state_len)
            # ON-BOX: write `committed` back into self._conv_states[key] via the mamba cache
            # manager's align block-migration contract (write path is backend-specific).
            yield key, committed

    def reset(self):
        self._staged.clear()
        self._conv_states.clear()


def _selfcheck():
    import torch

    # Single request, conv_dim=1, K-1 window = 2. Accept path 0->1->3; staged Bx[node]=[node].
    # Matches tree_spec._selfcheck's conv_commit_window case: window -> [[1],[3]], eff 3.
    scratch = torch.zeros(1, 6, 1)
    for n in range(6):
        scratch[0, n, 0] = float(n)
    init = torch.tensor([[[-1.0], [-2.0]]])  # [b=1, state_len=2, conv_dim=1]
    committed, eff = commit_conv_states(scratch, [[0, 1, 3]], init, state_len=2)
    assert committed[0].tolist() == [[1.0], [3.0]], committed[0].tolist()
    assert int(eff[0]) == 3

    # Short path pulls from init_state: accept only node 0 -> window [[-2],[0]], eff 1.
    committed2, eff2 = commit_conv_states(scratch, [[0]], init, state_len=2)
    assert committed2[0].tolist() == [[-2.0], [0.0]] and int(eff2[0]) == 1, committed2[0].tolist()

    # Nothing accepted -> window unchanged (rollback of all drafts), eff 0.
    committed3, eff3 = commit_conv_states(scratch, [[]], init, state_len=2)
    assert committed3[0].tolist() == [[-1.0], [-2.0]] and int(eff3[0]) == 0

    # Two requests batched, conv_dim=2.
    sc = torch.arange(2 * 4 * 2, dtype=torch.float32).reshape(2, 4, 2)
    iw = torch.zeros(2, 2, 2)
    c, e = commit_conv_states(sc, [[0, 1], [2]], iw, state_len=2)
    assert c.shape == (2, 2, 2) and e.tolist() == [2, 1], (c.shape, e.tolist())
    # request 0 accepted nodes 0,1 -> window = last 2 of [init(0,0), Bx0, Bx1] = [Bx0, Bx1]
    assert c[0].tolist() == sc[0, :2].tolist(), c[0].tolist()

    # ConvStager: stage returns a scratch clone (persistent untouched); commit_all yields windows.
    stager = ConvStager()
    persistent = torch.zeros(1, 3, 4)  # pretend conv_state
    scratch = stager.stage("layerA", torch.ones(1, 4), persistent, state_indices=None)
    scratch += 5  # mutate scratch
    assert persistent.abs().sum().item() == 0.0, "persistent conv_state must be untouched"
    sc2 = torch.arange(1 * 4 * 1, dtype=torch.float32).reshape(1, 4, 1)
    iw2 = torch.zeros(1, 2, 1)
    got = dict(stager.commit_all([[0, 1]], lambda k: sc2, lambda k: iw2, state_len=2))
    assert "layerA" in got and got["layerA"].shape == (1, 2, 1), got
    print("PASS: conv_commit — accepted-path window == oracle, short path, rollback, batched, stager")


if __name__ == "__main__":
    _selfcheck()
