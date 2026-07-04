# SUMMARY template

Copy this skeleton and fill it from the paper. It lives at
`docs/research/<category>/<research-topic>.md`; the source PDF is moved to
`docs/research/reference/<research-topic>.pdf` and linked via `../reference/`. Keep the
house style: `# <Category> — <Title>`, `## Overview`, numbered `## N.` sections, fenced
formula blocks, symbol/parameter tables.

**Layered output:** `## TL;DR` is skim-level; everything below it is the full
logic-complete distillation. The TL;DR summarizes; it never introduces a fact that the
body does not also carry.

Delete the `<!-- … -->` authoring notes before writing the final file. Keep any
"not specified by the paper" lines — they are content, not placeholders.

---

```markdown
# <Category> — <Paper Title>

> **Source:** <title> · <authors> · <venue/year>
> **PDF:** ../reference/<research-topic>.pdf   <!-- the source PDF, moved to docs/research/reference/ -->
> **URL:** <arXiv/DOI link, or "local file" if there was none>
> **Summarized:** <YYYY-MM-DD>  ·  **Fidelity:** faithful distillation (no info added/removed)

## TL;DR
<!-- 3–6 bullets. Core idea, the method in one line, the headline result with its number,
     and where/when it applies. Skim-level only; no fact here that the body lacks. -->
- ...

## Overview
<!-- 1–2 paragraphs: the problem the paper addresses and its core contribution,
     stated faithfully in the paper's own framing. -->

## 1. Problem & Setup
<!-- The model/market setup, assumptions, notation, and the objective function exactly as
     the paper defines them. Use the paper's symbols. -->

## 2. Method / Model
<!-- The algorithm or model, every step. If the paper gives a derivation, preserve its
     logical steps (you may condense prose between steps, not the steps themselves). -->

## 3. Key Equations
<!-- Fenced blocks for each important equation, in the paper's notation, followed by a
     symbol table. Do not rename symbols silently; if you must, record the mapping in
     Source Fidelity Notes. -->

```
<equation as written in the paper>
```

| Symbol | Meaning | Value / range (per paper) |
|--------|---------|---------------------------|
| `...`  | ...     | ... or "not specified"    |

## 4. Theoretical Results
<!-- Theorems, propositions, lemmas, closed-form results, asymptotics — as stated.
     Quote conditions and bounds exactly. Omit long proofs but point to the paper's
     section/theorem number; note the omission in Source Fidelity Notes. -->

## 5. Empirical Results
<!-- Datasets, instruments, sample period, metrics, and the EXACT numbers/tables the
     paper reports. Do not round beyond the paper, do not editorialize the significance. -->

## 6. Parameters
<!-- Every parameter the paper names, with the value/range/calibration the paper gives.
     If the paper does not give a value, write "not specified by the paper". Never
     supply a default of your own. -->

| Parameter | Role | Value / range (per paper) |
|-----------|------|---------------------------|
| `...`     | ...  | ... or "not specified by the paper" |

## 7. Assumptions & Limitations
<!-- The paper's own stated assumptions and limitations. Do not add limitations you infer
     unless you label them clearly as your inference in Source Fidelity Notes. -->

## 8. Takeaways
<!-- Only conclusions the paper itself supports. Practical relevance to this project is
     fine ONLY where the paper's results justify it; otherwise leave it out. -->

## Source Fidelity Notes
<!-- The integrity ledger for this summary:
     - Ambiguities in the paper and how you handled them.
     - Sections distilled vs. omitted, with paper §/figure/theorem references.
     - Any symbol renames, mapped back to the original notation.
     - Anything you could not extract (e.g. image-only figure, scanned page). -->
```

---

## Out of scope — do not fabricate

**Codebase-specific implementation** (Rust structs, core pinning, queue wiring, config
blocks) is NOT derivable from a paper and MUST NOT be invented here. A faithful summary
stops at the paper's content. Leave integration/implementation to a separate,
human-authored pass — optionally noted as a `<!-- TODO: implementation (out of scope) -->`
stub, never as fabricated code.
