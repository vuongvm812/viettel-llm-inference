# Slide design spec — "Inside LLM Inference" tech-sharing deck

Design system extracted from the approved HTML prototype so the deck can be
regenerated once the content docs are refined. Target: a single self-contained
HTML file (no build step, no external deps beyond Google Fonts), ~23 slides,
30 minutes.

## 1. Theme — vllm.ai dark

Palette taken from vllm.ai's own stylesheet (their `--color-vllm-blue` and logo
gold), backgrounds matched to their dark theme:

```css
:root {
    --bg:     #060d1e;   /* deep navy page background */
    --card:   #0f1930;   /* raised surface (cards, terminal blocks, pills) */
    --border: #1e2b47;   /* 1px card borders */
    --fg:     #f4f7fc;   /* primary text */
    --muted:  #92a0ba;   /* secondary text, labels */
    --blue:   #30a2ff;   /* vLLM blue — THE accent: kickers, highlights, fills */
    --gold:   #fdb517;   /* vLLM gold — sparing second accent: climaxes, warnings-adjacent emphasis */
    --red:    #ff6b6b;   /* rejected/banned/negative items only */
    --green:  #4ade80;   /* rare: per-request "own" cells in block diagrams */
}
```

Typography — vllm.ai's own pairing (Google Fonts):

- **Inter** (400/500/600/700/800) — all prose and headings. Headings weight 800,
  letter-spacing −0.02…−0.03em.
- **JetBrains Mono** (400/500/700) — code, flags, metrics, kickers, diagram
  labels, slide chrome. Mono NEVER appears inside a heading.

Rules learned from review rounds:

- **One font, one weight, one accent per title.** No mixed styling inside a
  heading; at most one `<span class="accent">` (blue). Gold in a title only for
  the single climactic word (e.g. "the host **IS** the TPOT").
- Gold is reserved for: final/best scores, the top ladder rung, the "remember
  this" card, burst/new cells in block visuals. Everything else accents blue.

## 2. Slide style (layout & chrome)

### Viewport fitting — non-negotiable

- Every `.slide`: `width: 100vw; height: 100vh; height: 100dvh; overflow: hidden;
  scroll-snap-align: start;` on an `html { scroll-snap-type: y mandatory }` scroller.
- All sizes in `clamp()`; height breakpoints at 700/600/500px shrink padding and
  type; `prefers-reduced-motion` support.
- **Root scaling** (the fix for "empty on big displays"): everything is rem-based and

  ```css
  html { font-size: clamp(16px, min(1.1vw, 2.2vh), 27px); }
  ```

  16px at ≤1455px wide (the verified 720p layout), growing to 27px on large
  displays; the `vh` term keeps short ultrawide windows from overflowing.
- Content column: `.slide-content { max-width: 88rem; margin: 0 auto; }` —
  centered on ultrawide, never hugging the left.
- **No pixel caps on visuals** — container widths in rem (`min(90vw, 56–62rem)`)
  so diagrams grow with the root font.
- Vertical gaps scale with viewport HEIGHT so tall screens spread content:
  `--content-gap: clamp(0.6rem, 4vh, 4rem)`, `--element-gap: clamp(0.25rem, 1.6vh, 1.6rem)`,
  h2 margin `clamp(1rem, 4.5vh, 3.2rem)`, fact-row padding `clamp(0.45rem, 2.2vh, 1.6rem)`,
  lesson gaps `clamp(0.9rem, 4.5vh, 3rem)`, line-height 1.6–1.65 on prose.
- A `.tight` slide variant (reduced gaps) for the 2–3 densest slides; a
  `.grid.mini` variant (fixed 3-column, compact cards) for the concept-map slide.

### Per-slide chrome

- Slim mono header inside each slide: left `NN · section name` (number in blue),
  right = section minute budget.
- Fixed chrome: 3px top progress bar (blue→gold gradient), right-edge nav dots,
  bottom-right `n / N` counter, bottom-left `← → · space` hint.
- Title + closing slides: centered layout, radial blue halo glow at top, mono
  pill tag above the h1, stat pills below.

### Interaction (JS, ~90 lines, one class)

- IntersectionObserver (threshold 0.55) adds `.visible` → staggered `.reveal`
  rise-in animations (translateY 24px, 0.6s expo ease, 0.1s stagger).
- Keyboard: arrows / space / PageUp-Down / Home / End. Touch: vertical swipe.
  Nav dots clickable. Score-bar fills animate to `--w` on `.visible`.
- No inline-edit code (explicitly declined).

## 3. Presentation style (structure & pacing)

**Framing decision:** teach-first for software engineers new to LLMs. The
competition is the hook and the case study, NOT the subject.
**Hard rule: never mix contest material into the knowledge section** — §2 must
contain zero contest references; contest numbers live only in §1/§3/closing.

| § | Section | Time | Slides |
|---|---------|------|--------|
| 1 | Why this talk (the hackathon hook: ship a server → judge hammers it → fastest wins; score pills) | 2 min | 2 |
| 2 | How inference works (the knowledge share) | 14 min | 10 |
| 3 | The race (the case study: 3 rounds, 3 rungs, results) | 9 min | 7 |
| 4 | Lessons + takeaway + closing | 5 min | 4 |

§2 opens with **"The map: nine ideas, one each"** — a 3×3 numbered grid listing
every fundamental before teaching them, in teaching order:

1. The next-token loop — generation is a loop, not a call
2. Prefill vs decode — TTFT and TPOT, the two numbers
3. Decode is memory-bound — every token reads every weight
4. The KV cache — never re-read the past
5. Paged & shared prefixes — cache blocks, reused across requests
6. Continuous batching — requests join & leave every step (+ chunked prefill)
7. The serving engine — vLLM puts it all together (diagram credits "Anatomy of vLLM")
8. Quantization — smaller numbers, faster tokens
9. The host overhead — the CPU between the steps (+ CUDA graphs)

§2 grounding: the vLLM team's ["Anatomy of vLLM"](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)
— use its terminology (waiting/running queues, "step", engine loop, "flattened
into one long super sequence") and cite it on the architecture slide as the
recommended deep-dive.

§3 arc: bridge slide ("you now know enough to see everything we did") → three
rounds as three different bottlenecks → rung 1 flags (incl. the 82.4%
prefix-cache bar) → measurement discipline → rung 2 patch-the-engine → rung 3
Rust → results bars + honest footnote.

§4: 6 lessons as big one-liners (3 performance + 3 systems), a "remember three
things" recap mirroring the fundamentals, closing slide reviving the line
"From docker compose up to a Rust CUDA-graph runner".

## 4. Content style

- **Visual-first, minimal text.** No paragraph bullets; the spoken talk and the
  repo docs carry the detail. Every concept slide = one visual + ≤1 short lede
  + ≤2 pills/cards.
- Preferred visual per content shape:
  - proportions → `.hbar` horizontal segment bar (e.g. 82.4% cache hits; 1 ms GPU / 2 ms host)
  - phases over time → `.timeline` (wide prefill segment + repeated decode cells, TTFT/TPOT labels)
  - token/block state → `.kvblock` cell strip (blue done / gold new-burst / green per-request; row tags for multi-request)
  - pipelines → `.flow` node-arrow-node rows
  - layered systems → `.stack` (overlay-on-stock diagram)
  - escalation → `.ladder` ascending rungs, gold top rung
  - comparative magnitude → `.score-bars` animated fills, gold for the best
  - bit-widths → `.qrow` width-proportional bars (fp16/fp8/int4)
  - engine anatomy → `.arch` boxes + dashed engine-loop container
  - knob → verdict facts → `.fact-row` (mono key, bold verdict; `.bad` red for rejected)
  - big claims → `.big-quote` (≤2 lines, one accent word)
  - stats → `.pill` (mono, big colored number + tiny label)
- Density limits: ≤6 fact-rows, ≤4 cards, ≤5 ladder rungs, ≤4 lessons per slide.
  Content exceeds limits → split the slide, never shrink below the type scale.
- Numbers stated on slides must come verbatim from the source docs
  (`docs/tech-talk-hackathon-rounds.md`, `docs/vllm-architecture-primer.md`) —
  no invented stats.
- Tone: plain language for the audience ("bring friends", "never re-read the
  past"); jargon only after it's been defined.

## 5. Regeneration & verification workflow

1. Refine the content docs, then regenerate `docs/hackathon-talk-slides.html`
   from this spec (single file, all CSS/JS inline).
2. Verify with headless Chromium (playwright) before delivering: for each of
   1280×720, 1920×1080, 2560×1440, 2560×1080 assert per-slide
   `scrollHeight − clientHeight ≤ 1` and `scrollWidth − clientWidth ≤ 1` on both
   `.slide` and `.slide-content`. Zero overflow is the acceptance bar.
3. Screenshot spot-checks of the diagram-heavy slides (map, timeline,
   architecture, ladder, kv-block, results) at 720p and 1440p.
