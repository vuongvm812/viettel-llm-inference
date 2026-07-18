// C++ port of vtl/ngram_tree.py TreeNgramDrafter (Phase-1 speed path of the tree-spec fork).
//
// Correct-by-construction mirror of the tested Python reference: _continuations() +
// build_tree() have identical semantics, so vtl/ngram_tree.py's self-check defines this
// module's behavior. TreeNgramDrafter imports `vtl_ngram` when built and falls back to the
// pure-Python path otherwise, so serving never depends on this being compiled.
//
// NOTE (unbuilt off-box): the matcher is not the decode bottleneck (~us vs ~4ms/token), so this
// is a low-ROI optimization; it exists to match the plan's Phase-1 "C++ corpus" item. A full
// suffix automaton (sglang cpp_ngram) is a further step; this intra-context trie matches the
// Python reference and is enough to measure acceptance.
//
// Build (on the box): add to setup.py as a torch CppExtension (torch ships pybind11):
//   CppExtension("vtl_ngram", ["vtl/csrc/ngram_tree/ngram_tree.cpp"])
// Torch's cpp_extension provides <pybind11/*>; no extra build dep.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace {

// Mirror of TreeNgramDrafter._continuations: continuations following the longest recent
// suffix (length in [min_n, max_n]) that recurs earlier, most-recent occurrence first.
std::vector<std::vector<int64_t>>
continuations(const std::vector<int64_t> &toks, int min_n, int max_n, int max_depth) {
  const int n = static_cast<int>(toks.size());
  const int hi = std::min(max_n, n - 1);
  for (int L = hi; L >= min_n; --L) {
    const std::vector<int64_t> suffix(toks.end() - L, toks.end());
    std::vector<std::vector<int64_t>> conts;
    for (int p = n - L - 1; p >= 0; --p) {  // most-recent occurrence first
      bool match = true;
      for (int t = 0; t < L; ++t) {
        if (toks[p + t] != suffix[t]) { match = false; break; }
      }
      if (match) {
        const int e = std::min(p + L + max_depth, n);
        if (p + L < e) conts.emplace_back(toks.begin() + p + L, toks.begin() + e);
      }
    }
    if (!conts.empty()) return conts;
  }
  return {};
}

struct TreeResult {
  std::vector<int64_t> node_tokens;
  std::vector<int64_t> parent;  // -1 for a first-level draft token
  std::vector<int64_t> count;   // branch frequency (for best_chain)
};

// Mirror of TreeNgramDrafter.build_tree: merge continuations into a frequency-counted trie,
// insertion order = root-first within each branch, capped at max_nodes.
TreeResult build_tree(const std::vector<int64_t> &toks, int min_n, int max_n, int max_nodes,
                      int max_depth) {
  TreeResult r;
  const auto conts = continuations(toks, min_n, max_n, max_depth);
  std::unordered_map<int, std::unordered_map<int64_t, int>> children;  // node(-1=root) -> {tok: child}
  for (const auto &cont : conts) {
    int cur = -1;
    for (const int64_t tok : cont) {
      auto &m = children[cur];
      const auto it = m.find(tok);
      if (it != m.end()) {
        r.count[it->second] += 1;
        cur = it->second;
      } else {
        if (static_cast<int>(r.node_tokens.size()) >= max_nodes) break;
        const int ci = static_cast<int>(r.node_tokens.size());
        r.node_tokens.push_back(tok);
        r.parent.push_back(cur == -1 ? -1 : cur);
        r.count.push_back(1);
        m[tok] = ci;
        cur = ci;
      }
    }
  }
  return r;
}

}  // namespace

PYBIND11_MODULE(vtl_ngram, m) {
  m.doc() = "vtl tree-ngram drafter (C++ mirror of vtl/ngram_tree.py TreeNgramDrafter)";
  py::class_<TreeResult>(m, "TreeResult")
      .def_readonly("node_tokens", &TreeResult::node_tokens)
      .def_readonly("parent", &TreeResult::parent)
      .def_readonly("count", &TreeResult::count);
  m.def("build_tree", &build_tree, py::arg("tokens"), py::arg("min_n"), py::arg("max_n"),
        py::arg("max_nodes"), py::arg("max_depth"),
        "Build a frequency-counted draft trie; returns TreeResult(node_tokens, parent, count).");
}
