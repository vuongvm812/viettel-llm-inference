"""Tree speculative decoding — vLLM integration (Phases 2-5 of the SGLang ngram-v2 port).

This patch wires the tree-spec CORE LOGIC (vtl/tree_spec.py, all unit-tested against the
golden oracles) into vLLM's spec-decode path. It is the integration surface; the algorithms
themselves live in vtl.tree_spec / vtl.ngram_tree / vtl.tree_verify_ref.

Default OFF (`VTL_ENABLE_TREE_SPEC` unset -> False). It is EXPERIMENTAL and, unlike the other
vtl patches, has NOT been validated on hardware: it monkeypatches private vLLM V1 methods whose
exact signatures/buffers must be confirmed on the H200 before trust. Each hook degrades to stock
behavior (logs + returns) on any mismatch so it can never crash serving or corrupt the working
ngram/suffix chain path. Turn it on only alongside `--speculative-config method=custom_class
model=vtl.ngram_tree.TreeNgramProposer`.

The five hooks (vLLM function -> what we route through), from the fork map:
  1. proposer            : TreeNgramProposer stashes the per-request tree; runner reads it.
  2. _calc_spec_decode_metadata (gpu_model_runner) : emit tree positions + attach tree_mask
                           via tree_spec.flatten_batch_trees / reconstruct_indices.
  3. FlashAttentionImpl.forward (verify step)      : cascade(prefix, tree)+LSE merge
                           (tree_spec.cascade_tree_attention; prod path uses paged FA varlen).
  4. rejection_sample (rejection_sampler)          : tree_verify_ref.tree_greedy_accept.
  5. short-conv commit (mamba postprocess)         : stage Bx per node + conv_commit_window.

`run_tree_spec_step` below composes the tested pieces into one reference decode step with NO
vLLM dependency — it is the executable spec for hooks 1/2/4/5 and runs in the self-check.
"""
from __future__ import annotations

import logging

from vtl.registry import already_patched, mark_patched, register_patch

log = logging.getLogger("vtl")


# --------------------------------------------------------------- reference orchestration

def run_tree_spec_step(context_tokens, target_fn, drafter, *, max_nodes, state_len,
                       staged_bx_fn, init_conv_state, seq_len):
    """One tree-spec decode step over the tested pieces — the executable spec for the hooks.

    context_tokens : the request's tokens so far (drives the drafter).
    target_fn(node_tokens, parent, seq_len) -> (target_pred, root_pred): stands in for the
        target model's greedy argmax after each tree node and at the anchor.
    staged_bx_fn(node_id) -> Bx vector: the short-conv input staged during the (mocked) verify.
    Returns dict with accepted tokens, bonus, committed conv window, effective path length.
    """
    from vtl.tree_spec import reconstruct_indices, tree_greedy_accept_indices, conv_commit_window

    node_tokens, parent = drafter.build_tree(context_tokens)
    node_tokens, parent = node_tokens[:max_nodes], parent[:max_nodes]
    if not node_tokens:
        return {"accepted": [], "bonus": None, "conv_window": init_conv_state, "eff_len": 0}

    idx = reconstruct_indices(parent, seq_len)
    target_pred, root_pred = target_fn(node_tokens, parent, seq_len)
    root_children = [i for i, p in enumerate(parent) if p == -1]

    accepted, bonus = tree_greedy_accept_indices(
        node_tokens, idx["retrieve_next_token"], idx["retrieve_next_sibling"],
        target_pred, root_pred, root_children,
    )
    staged = {n: staged_bx_fn(n) for n in accepted}
    conv_window, eff = conv_commit_window(accepted, staged, init_conv_state, state_len)
    return {
        "accepted": [node_tokens[i] for i in accepted],
        "bonus": bonus,
        "conv_window": conv_window,
        "eff_len": eff,
        "accepted_ids": accepted,
    }


# ------------------------------------------------------------------------- vLLM hooks

def _install_metadata_hook() -> bool:
    """Hook 2: emit tree positions + tree_mask in the verify metadata.

    Wraps `GPUModelRunner._calc_spec_decode_metadata`. Confirm the arg/return shape on-box
    (fork map: gpu_model_runner.py:2798). Returns False (degrade to stock) if the symbol
    or signature is not what we expect.
    """
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception:
        log.warning("vtl tree_spec: GPUModelRunner import failed; metadata hook skipped")
        return False
    if already_patched(GPUModelRunner, "_calc_spec_decode_metadata"):
        return True
    orig = getattr(GPUModelRunner, "_calc_spec_decode_metadata", None)
    if orig is None:
        log.warning("vtl tree_spec: _calc_spec_decode_metadata missing; metadata hook skipped")
        return False

    def wrapper(self, *args, **kwargs):
        md = orig(self, *args, **kwargs)
        # Attach tree topology captured by the proposer (hook 1, TreeNgramProposer.propose
        # stashes it on the drafter) so the sampler hook can read it. Absence / any mismatch
        # -> plain chain metadata (stock behavior).
        drafter = getattr(self, "drafter", None)
        trees = getattr(drafter, "_vtl_pending_trees", None)
        # Guard: the stash is from the previous step's propose; only attach if it still lines
        # up with this verify's batch (num_draft_tokens is [batch_size]). Mismatch -> skip.
        if trees and len(trees) == len(getattr(md, "num_draft_tokens", [])):
            try:
                from vtl.tree_runtime import build_carrier
                seq_lens = getattr(drafter, "_vtl_seq_lens", None) or [0] * len(trees)
                device = getattr(self, "device", "cpu")
                # Device-tensor carrier (static [b, max_nodes]) consumed by the attention/sampler
                # forks. For a width-1 (chain) batch every consumer routes to stock, so attaching it
                # is behavior-neutral — the P2 regression gate (bit-identical to the no-carrier run).
                setattr(md, "vtl_tree", build_carrier(
                    trees, seq_lens=seq_lens,
                    max_nodes=getattr(drafter, "_vtl_max_nodes", 16), device=device))
            except Exception:
                log.exception("vtl tree_spec: tree metadata build failed; using chain md")
        return md

    setattr(GPUModelRunner, "_calc_spec_decode_metadata", mark_patched(wrapper, orig))
    return True


def _install_sampler_hook() -> bool:
    """Hook 4: tree acceptance in the rejection sampler.

    Wraps `vllm.v1.sample.rejection_sampler.rejection_sample` (fork map:349,394). When the
    metadata carries `vtl_tree`, accept along the best tree path via the oracle; otherwise
    defer to the stock chain sampler.
    """
    try:
        import vllm.v1.sample.rejection_sampler as rs
    except Exception:
        log.warning("vtl tree_spec: rejection_sampler import failed; sampler hook skipped")
        return False
    if already_patched(rs, "rejection_sample"):
        return True
    orig = getattr(rs, "rejection_sample", None)
    if orig is None:
        return False

    warned = [False]

    def wrapper(*args, **kwargs):
        md = kwargs.get("metadata") or (args[-1] if args else None)
        tree = getattr(md, "vtl_tree", None) if md is not None else None
        if tree is None:
            return orig(*args, **kwargs)  # no tree metadata -> stock chain path
        # A width-1 tree IS a chain: greedy tree-accept over a linear tree is exactly stock
        # rejection sampling, so stock is CORRECT here (not a fallback) and we run it directly.
        # A branching tree needs the tree kernel + tree attention (Phase-3 core edit); until
        # that lands we emit chains only, so this degrade path should never trigger in serving.
        if _is_chain(tree):
            return orig(*args, **kwargs)
        if not warned[0]:
            warned[0] = True
            log.warning("vtl tree_spec: branching tree reached the sampler but tree attention "
                        "is not installed (Phase-3 core edit); accepting the chain via stock. "
                        "Keep the drafter emitting chains until the backend path lands.")
        return orig(*args, **kwargs)

    setattr(rs, "rejection_sample", mark_patched(wrapper, orig))
    return True


def _is_chain(tree) -> bool:
    """True iff every per-request tree in the flattened metadata is linear (no branching).

    Chain <=> no node has a next-sibling. flatten_batch_trees emits retrieve_next_sibling as
    [batch, max_nodes] with -1 for "none"; any other value means a fork.
    """
    sib = tree.get("retrieve_next_sibling") if isinstance(tree, dict) else None
    if sib is None:
        return True  # unknown shape -> treat as chain (stock is safe)
    return all(s == -1 for row in sib for s in row)


def _install_attention_hook() -> bool:
    """Hook 3: cascade tree attention for the verify step (FA3 on H200, no custom mask).

    For the chain milestone (width-1 tree) stock causal attention is EXACTLY the tree
    attention, so nothing needs patching here and the verify is correct. The width>1 path
    builds two `flash_attn_varlen_func` calls (prefix + expand with the page-index permutation)
    over a block_size=1 draft KV region and LSE-merges them; the merge math is validated
    (bench/spike_fa3_tree_cascade.py). That edit touches the FA metadata builder + forward
    (fork map: flash_attn.py) with paged token-granular indices and cannot be a safe monkeypatch
    without the on-box buffer layout -> it is the one remaining core edit, validated on the H200.
    """
    log.info("vtl tree_spec: attention uses stock causal (correct for the width-1 chain "
             "milestone). Width>1 cascade attention is the remaining core edit; see "
             "vtl.tree_spec.cascade_tree_attention for the validated merge.")
    return True


def _install_conv_commit_hook() -> bool:
    """Hook 5: short-conv Bx staging + accepted-path gather-commit.

    For the width-1 chain milestone the accepted path is linear, so the stock scalar-prefix
    conv commit is already correct and nothing is patched here. The width>1 path wraps LFM2
    ShortConv.forward to stage per-node Bx and replaces the commit with
    `tree_spec.conv_commit_window` (validated: bench/spike_tree_conv_commit.py), plus a
    multi-token verify path in ShortConv.forward_cuda (fork map: short_conv.py:282). Those are
    model-layer edits that go in together with the tree attention edit and are validated on-box.
    """
    log.info("vtl tree_spec: conv uses stock scalar-prefix commit (correct for the width-1 "
             "chain milestone). Width>1 gather-commit ships with the tree attention edit; see "
             "vtl.tree_spec.conv_commit_window for the validated commit.")
    return True


@register_patch("tree_spec", default=False)
def apply() -> None:
    """Install the tree-spec integration hooks. Experimental; default OFF; never crashes serving."""
    ok_meta = _install_metadata_hook()
    ok_samp = _install_sampler_hook()
    ok_attn = _install_attention_hook()
    ok_conv = _install_conv_commit_hook()
    log.info("vtl tree_spec: hooks metadata=%s sampler=%s attention=%s conv=%s "
             "(width-1 chain flow is complete + correct; width>1 tree attention/conv is the "
             "one remaining core edit, validated on-box)",
             ok_meta, ok_samp, ok_attn, ok_conv)


# ------------------------------------------------------------------------------- checks

def _selfcheck() -> None:
    from vtl.ngram_tree import TreeNgramDrafter

    # A repetitive context so the drafter builds a real tree.
    ctx = [1, 2, 3, 9, 1, 2, 3, 9, 1, 2, 3]
    drafter = TreeNgramDrafter(min_n=2, max_n=4, max_nodes=8, max_depth=4)

    # target_fn accepts the drafter's own top chain (so acceptance is non-trivial),
    # then diverges (bonus) — mimics a high-acceptance step.
    def target_fn(node_tokens, parent, seq_len):
        # accept the first-level node, then follow parent->child along node order
        target_pred = [0] * len(node_tokens)
        # make the target agree with each node's first child token
        from vtl.tree_spec import reconstruct_indices
        idx = reconstruct_indices(parent, seq_len)
        for i in range(len(node_tokens)):
            nt = idx["retrieve_next_token"][i]
            target_pred[i] = node_tokens[nt] if nt != -1 else -999  # diverge at a leaf -> bonus
        root_pred = node_tokens[0]  # agree with the top root draft
        return target_pred, root_pred

    out = run_tree_spec_step(
        ctx, target_fn, drafter,
        max_nodes=8, state_len=2,
        staged_bx_fn=lambda n: [float(n)],
        init_conv_state=[[-1.0], [-2.0]],
        seq_len=len(ctx),
    )
    assert out["accepted"], out                       # accepted a non-empty path
    assert out["bonus"] == -999, out                  # diverged into the bonus at the leaf
    assert len(out["conv_window"]) == 2, out          # committed window == state_len
    assert out["eff_len"] == len(out["accepted_ids"]) # align-consistent path length
    print(f"PASS: tree_spec patch pipeline — accepted {out['accepted']} bonus {out['bonus']} "
          f"conv_window {out['conv_window']} eff_len {out['eff_len']}")


if __name__ == "__main__":
    _selfcheck()
