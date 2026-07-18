"""
vtl tree-ngram speculative drafter — Phase 1 of the SGLang ngram-v2 -> vLLM fork.

Pure-Python reference drafter + a vLLM `custom_class` proposer. This is BOTH the
Phase-1 milestone (emit the best root->leaf chain, verified by vLLM's existing
chain path) AND the correctness reference the later optimizations must match:
  * the C++ suffix-automaton/trie corpus (port of sglang cpp_ngram) is a drop-in
    speed replacement for TreeNgramDrafter and must reproduce its tokens/tree,
  * `reconstructIndicesFromTreeMask` (CUDA) consumes tree_mask_from_parent(),
  * the vLLM-core tree verify (attention/sampler/conv-commit, Phases 2-5) consumes
    the full tree; until it lands, the proposer emits the best CHAIN only.

Invoke (chain milestone — no core fork needed):
  --speculative-config={"method":"custom_class",
      "model":"vtl.ngram_tree.TreeNgramProposer",
      "num_speculative_tokens":4,"prompt_lookup_min":3,"prompt_lookup_max":8}

Self-check (no torch/vllm/numpy):  python vtl/ngram_tree.py
"""
from __future__ import annotations

from collections import defaultdict

try:  # optional C++ fast path (vtl/csrc/ngram_tree/ngram_tree.cpp); pure-Python fallback below
    import vtl_ngram as _cpp
except Exception:  # not built (e.g. off-box) -> use the reference Python implementation
    _cpp = None


class TreeNgramDrafter:
    """Suffix-match tree drafter over a token context (intra-sequence corpus).

    Algorithmic mirror of sglang cpp_ngram: match the longest recent suffix of the
    context against earlier occurrences, expand a frequency-ranked tree of
    continuations (branching on alternative next-tokens), cap at `max_nodes`.
    """

    def __init__(self, min_n: int = 3, max_n: int = 8, max_nodes: int = 16, max_depth: int = 8):
        self.min_n = min_n
        self.max_n = max_n
        self.max_nodes = max_nodes  # draft_token_num (tree-size cap)
        self.max_depth = max_depth
        self._count: list[int] = []

    def _continuations(self, tokens: list[int]) -> list[list[int]]:
        """Continuation sequences following the longest recent suffix match.

        Anchors on the longest suffix length in [min_n, max_n] that recurs earlier
        (KMP-optimal anchor); returns every earlier occurrence's continuation, most
        recent first, so frequency ranking downstream favors the modal next-token.
        """
        n = len(tokens)
        for L in range(min(self.max_n, n - 1), self.min_n - 1, -1):
            suffix = tokens[n - L :]
            conts = []
            for p in range(n - L - 1, -1, -1):  # most-recent occurrence first
                if tokens[p : p + L] == suffix:
                    cont = tokens[p + L : p + L + self.max_depth]
                    if cont:
                        conts.append(cont)
            if conts:
                return conts
        return []

    def build_tree(self, tokens: list[int]) -> tuple[list[int], list[int]]:
        """Merge continuations into a draft trie. Returns (node_tokens, parent).

        parent[i] is the node index of i's parent, or -1 for a first-level draft
        token. Node order is insertion order (root-first within each branch).
        """
        if _cpp is not None:  # C++ mirror; identical semantics to the Python below
            r = _cpp.build_tree(list(tokens), self.min_n, self.max_n, self.max_nodes, self.max_depth)
            self._count = list(r.count)
            return list(r.node_tokens), list(r.parent)
        node_tokens: list[int] = []
        parent: list[int] = []
        count: list[int] = []
        children: dict[int, dict[int, int]] = defaultdict(dict)  # node -> {token: child}
        ROOT = -1
        for cont in self._continuations(tokens):
            cur = ROOT
            for tok in cont:
                if tok in children[cur]:
                    ci = children[cur][tok]
                    count[ci] += 1
                else:
                    if len(node_tokens) >= self.max_nodes:
                        break
                    ci = len(node_tokens)
                    node_tokens.append(tok)
                    parent.append(-1 if cur == ROOT else cur)
                    count.append(1)
                    children[cur][tok] = ci
                cur = ci
        self._count = count
        return node_tokens, parent

    def best_chain(self, node_tokens: list[int], parent: list[int], k: int) -> list[int]:
        """Highest-frequency root->leaf path, truncated to k tokens (chain milestone)."""
        if not node_tokens or k <= 0:
            return []
        count = self._count
        by_parent: dict[int, list[int]] = defaultdict(list)
        for i, p in enumerate(parent):
            by_parent[p].append(i)
        roots = by_parent.get(-1, [])
        if not roots:
            return []
        cur = max(roots, key=lambda i: count[i])
        chain = [node_tokens[cur]]
        while len(chain) < k:
            kids = by_parent.get(cur, [])
            if not kids:
                break
            cur = max(kids, key=lambda i: count[i])
            chain.append(node_tokens[cur])
        return chain

    def propose_chain(self, tokens: list[int], k: int) -> list[int]:
        node_tokens, parent = self.build_tree(tokens)
        return self.best_chain(node_tokens, parent, k)


def tree_mask_from_parent(parent: list[int]) -> list[list[bool]]:
    """n x n: mask[i][j] = True iff j is on i's root-path (ancestor incl. self).

    Same semantics as sglang Result.mask and the GPU tree_mask consumed by
    reconstructIndicesFromTreeMask.
    """
    n = len(parent)
    mask = [[False] * n for _ in range(n)]
    for i in range(n):
        j = i
        while j != -1:
            mask[i][j] = True
            j = parent[j]
    return mask


class TreeNgramProposer:
    """vLLM `custom_class` proposer. Phase-1 milestone: best chain per request.

    `.propose` mirrors NgramProposer.propose
    (vllm/v1/spec_decode/ngram_proposer.py): returns list[list[int]], one draft
    chain per request, [] to skip. vLLM is imported lazily so this module stays
    self-testable without vLLM installed.
    """

    def __init__(self, vllm_config):
        spec = vllm_config.speculative_config
        self.k = spec.num_speculative_tokens
        self.drafter = TreeNgramDrafter(
            min_n=spec.prompt_lookup_min or 3,
            max_n=spec.prompt_lookup_max or 8,
            max_nodes=max(self.k, 16),
            max_depth=self.k,
        )
        self.max_model_len = vllm_config.model_config.max_model_len

    def propose(
        self,
        num_speculative_tokens,
        sampled_token_ids,
        num_tokens_no_spec,
        token_ids_cpu,
        slot_mappings=None,  # unused (chain milestone)
    ):
        drafts: list[list[int]] = []
        for i, sampled in enumerate(sampled_token_ids):
            if not sampled:
                drafts.append([])
                continue
            n = int(num_tokens_no_spec[i])
            if n >= self.max_model_len:
                drafts.append([])
                continue
            row = token_ids_cpu[i]
            ctx = row[:n].tolist() if hasattr(row, "tolist") else list(row[:n])
            k = min(num_speculative_tokens, self.max_model_len - n)
            drafts.append(self.drafter.propose_chain(ctx, k))
        return drafts

    def load_model(self, *args, **kwargs):
        # No model to load — the drafter is a pure-Python matcher.
        pass


def _selfcheck():
    # Pattern 1,2,3 recurs; after the final "...9,1,2,3" the modal continuation is 9,1,2,3.
    d = TreeNgramDrafter(min_n=2, max_n=4, max_nodes=16, max_depth=4)
    toks = [5, 1, 2, 3, 9, 1, 2, 3, 9, 1, 2, 3]
    assert d.propose_chain(toks, 2) == [9, 1], d.propose_chain(toks, 2)
    assert d.propose_chain(toks, 4) == [9, 1, 2, 3], d.propose_chain(toks, 4)

    # No recurrence -> no draft.
    assert TreeNgramDrafter(min_n=3, max_n=5).propose_chain([1, 2, 3, 4, 5], 3) == []

    # Branching tree: two continuations after the suffix -> both branches present,
    # best_chain follows the more frequent one.
    d2 = TreeNgramDrafter(min_n=1, max_n=1, max_nodes=16, max_depth=3)
    #   suffix "7": followed by 8 (twice) and by 4 (once) -> best chain starts with 8.
    seq = [7, 8, 0, 7, 8, 0, 7, 4, 0, 7]
    nt, par = d2.build_tree(seq)
    assert 8 in nt and 4 in nt, (nt, par)
    assert d2.best_chain(nt, par, 1) == [8], (nt, par, d2._count)

    # tree_mask ancestry: 0 -> {1,2}; 1 -> {3}
    parent = [-1, 0, 0, 1]
    m = tree_mask_from_parent(parent)
    assert m[3] == [True, True, False, True]  # node 3 sees root 0, parent 1, self 3
    assert m[2] == [True, False, True, False]  # node 2 sees root 0, self 2
    assert m[0] == [True, False, False, False]

    print("PASS: TreeNgramDrafter — suffix match, frequency chain, branching, tree mask")


if __name__ == "__main__":
    _selfcheck()
