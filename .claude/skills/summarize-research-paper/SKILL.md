---
name: summarize-research-paper
description: Use when summarizing a research paper or academic PDF (a local file or an arXiv link) into the repo under docs/research, or when asked to faithfully distill a paper without losing or adding information. Keywords: research paper, arxiv, PDF, summarize paper, paper summary, research doc, docs/research, faithful summary, distill paper.
---

# Summarize Research Paper

## Overview

Take a paper the user points to — a **local PDF** (primary) or an **arXiv link/id** — move
the source PDF into `docs/research/reference/`, and write a faithful summary that links to
it under `docs/research/<category>/<research-topic>.md`. The `<category>` and
`<research-topic>` are derived from your summary (e.g. `strategies`, `compiler`,
`latency`, `networking`).

**Core principle: faithful, logic-complete distillation — add nothing, remove nothing.**
You may compress *wording*. You may never drop a claim, equation, number, assumption, or
logical step, and you may never invent one.

## The Iron Rule

Every sentence, equation, and number in the summary MUST trace to the paper. Anything the
paper does not state MUST NOT appear in the summary — unless explicitly written as
"not specified by the paper". Guesses, "reasonable defaults", and "this is how it'd be
implemented" are correctness violations, not helpfulness.

**REQUIRED before writing:** read `references/fidelity-rules.md`. It defines the integrity
rules, the red-flag rationalizations, and the verification checklist you must pass.

## Workflow

1. **Ingest** → get a local PDF path to read:
   - **Local file (primary):** the user gives a path; use it directly. Note the original
     path — you will move this file in step 4.
   - **arXiv:** `bash scripts/fetch_paper.sh <arxiv-id|arxiv-url|pdf-url>` prints one local
     path on stdout; `Read` it. Record the canonical `https://arxiv.org/abs/<id>` for the
     `URL:` line.

   `Read` the PDF (read natively). → verify: full text accessible, including equations and
   tables.
2. **Distill** → fill `references/summary-template.md` section by section. Keep the paper's
   own notation; condense prose only, never facts. From the content, decide:
   - `<category>` — the research area: `strategies`, `compiler`, `latency`, `networking`, …
   - `<research-topic>` — a kebab-case slug naming the paper's subject.
3. **Verify (mandatory)** → run the checklist in `references/fidelity-rules.md`: forward
   pass (every paper claim represented or consciously omitted-with-pointer) and reverse
   pass (every summary sentence traces to a paper location); re-check equations/numbers
   against the source. → verify: must pass before step 4.
4. **Place** → `mkdir -p docs/research/reference docs/research/<category>`, then:
   - **Move** the source PDF into the shared reference folder:
     `mv "<source.pdf>" docs/research/reference/<research-topic>.pdf`
     (this relocates the user's file — say so).
   - **Write** the summary to `docs/research/<category>/<research-topic>.md`, with its
     `PDF:` header line linking across to `../reference/<research-topic>.pdf`.

   If the category is ambiguous or either target file already exists, confirm with the
   user before moving/overwriting — never silently clobber. Report both final paths.

## Quick Reference

| Input | How to ingest |
|-------|---------------|
| Local PDF (primary) | use the path directly; moved into the repo in step 4 |
| arXiv id (`2301.01234`) | `bash scripts/fetch_paper.sh 2301.01234` |
| arXiv abs/pdf URL | `bash scripts/fetch_paper.sh <url>` |

## Red Flags — STOP

If you catch yourself thinking any of these, you are about to violate the Iron Rule:

- "I'll add a sensible default value." → The paper didn't state it. Write "not specified".
- "The paper probably means…" → Probably ≠ stated. Mark it as ambiguous instead.
- "This is how it'd be implemented in our codebase." → Out of scope. Do not invent structs,
  core layout, or wiring. Faithful paper content only.

Full rationalization table and verification steps: `references/fidelity-rules.md`.
