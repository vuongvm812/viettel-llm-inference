# Fidelity rules — the integrity guard

This is the heart of the skill. A summary that loses or invents information is wrong, no
matter how readable it is. Read this fully before writing any summary.

## The rule

**Preserve every claim, equation, number, definition, assumption, and logical step.
Add nothing. Remove no information.**

- You MAY compress *wording* — cut repetition, motivation prose, and rhetorical framing.
- You MAY NOT cut a fact: a hypothesis, a parameter, a result number, a boundary
  condition, a step of a derivation.
- You MAY NOT add a fact the paper does not state.

**Violating the letter of this rule is violating the spirit of it.** "I kept the gist"
is not the standard; "every fact is preserved or explicitly marked omitted" is.

## Notation discipline

Use the paper's own symbols and names. If you must rename a symbol (e.g. a clash within
the doc), record the original→new mapping in **Source Fidelity Notes**. Never let a
reader think a renamed symbol is the paper's.

## Uncertainty discipline

If the paper does not specify something — a parameter value, a dataset detail, a
condition — write **"not specified by the paper"**. Never fill the gap with a guess, a
typical value, or a default. Absence of information is itself information; record it.

## Red flags — STOP

If you think any of these, you are about to break the rule:

| Excuse (you might think) | Reality |
|--------------------------|---------|
| "I'll add a sensible default value." | A default the paper didn't state is fabrication. Write "not specified by the paper". |
| "The paper probably means X." | Probably ≠ stated. Record it as ambiguous in Source Fidelity Notes; don't assert X. |
| "This is how it'd be implemented (Rust struct / core layout)." | Implementation is out of scope. Inventing it corrupts the summary. Leave a TODO stub. |
| "This detail is minor, I'll drop it." | Minor to you ≠ minor to the reader. Compress wording, keep the fact. |
| "I'll round this number for readability." | The paper's precision is the fact. Keep it as reported. |
| "I'll smooth the contradiction." | If the paper is inconsistent, report the inconsistency; don't resolve it for the author. |
| "The conclusion is obvious, I'll strengthen it." | Only state conclusions the paper supports, at the strength the paper states them. |

## Verification checklist (Workflow step 4 — mandatory before writing)

Run BOTH passes. Do not write the file until both pass.

**Forward pass — completeness (paper → summary):**
- [ ] Every section of the paper is represented, OR consciously omitted with a pointer
      (paper §/figure/theorem) in Source Fidelity Notes.
- [ ] Every equation that carries logic appears (or is noted omitted with a pointer).
- [ ] Every reported result number, dataset, and parameter is present, or marked
      "not specified by the paper" where the paper is silent.
- [ ] Every stated assumption and limitation is captured.

**Reverse pass — faithfulness (summary → paper):**
- [ ] Every sentence in the summary traces to a specific place in the paper.
- [ ] No equation, number, parameter value, or claim exists that the paper does not
      state (anything inferred is labeled as inference in Source Fidelity Notes).
- [ ] No codebase-specific implementation has been invented.
- [ ] Symbols match the paper, or renames are mapped in Source Fidelity Notes.

**Provenance:**
- [ ] The `PDF:` line links to the source PDF you moved into
      `docs/research/reference/<research-topic>.pdf` (i.e. `../reference/<research-topic>.pdf`
      from the summary), and that file is actually there.
- [ ] The `URL:` line holds a stable link — `https://arxiv.org/abs/<id>` or the DOI —
      or `local file` when the input was a local PDF with no URL. NOT a temp download path.

**Spot re-check:**
- [ ] Re-open the paper and re-verify 3–5 equations and all headline result numbers
      character-for-character against the summary.

If any box fails, fix the summary and re-run the failed pass.
