# vLLM Fundamentals — 15-minute talk, slide content spec

- **Audience:** software engineers, little/no LLM-inference background
- **Duration:** 15 min (11 slides, ~14 min content + 1 min buffer)
- **Goal:** after the talk, the audience can explain why serving LLMs is a memory problem and what vLLM's two core tricks (PagedAttention, continuous batching) do about it
- **Style:** minimal on-slide text, one big visual per slide; the speaker notes carry the words
- **Primary source:** [Anatomy of vLLM](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) (vLLM team, Sep 2025)
- **Visuals policy:** reuse a blog figure only when it is simple enough to read from the back of the room in ~5 seconds. The blog's worked-example figures (block metadata, hash maps, slot mappings) are great for reading, too dense for a slide — for those, build a simple-stupid diagram of our own instead. Credit "Figure: Anatomy of vLLM, vllm.ai" on every reused image.

## Reused figure assets (simple ones only)

All under `https://vllm.ai/blog-assets/figures/2025-vllm-anatomy/`:

| File | Shows | Used on slide |
|------|-------|---------------|
| `latency_diagram.png` | query → token timeline with TTFT / ITL / e2e brackets | 3 |
| `roofline.png` | roofline: memory-bandwidth-bound vs compute-bound zones | 4 |
| `engine_loop.png` | waiting queue → schedule / forward pass / postprocess ring | 10 |

Rejected as too dense for slides (fine as speaker prep / "further reading"): `kv_cache_blocks.png`, `fwd_pass.png`, `prefix_pt1-3.png`, `engine_constructor.png` — all worked examples with heavy annotation text; slides 7–10 use simple custom diagrams instead.

Download once for offline slides:

```bash
mkdir -p assets && cd assets
for f in latency_diagram roofline engine_loop; do
  curl -sLO "https://vllm.ai/blog-assets/figures/2025-vllm-anatomy/$f.png"
done
```

Note: figures are drawn light-background; on a dark deck place them on a white rounded card.

## Agenda / time budget

| # | Slide | Time |
|---|-------|------|
| 1 | What happens after you POST? | 0:30 |
| 2 | One token at a time | 1:30 |
| 3 | Two phases: prefill vs decode | 1:30 |
| 4 | Decode is memory-bound | 1:30 |
| 5 | The KV cache | 1:30 |
| 6 | The naive serving problem | 1:00 |
| 7 | PagedAttention | 2:00 |
| 8 | Continuous batching | 2:00 |
| 9 | Free win: prefix caching | 1:00 |
| 10 | vLLM in one picture | 1:30 |
| 11 | What to measure & takeaways | 1:30 |
| | **Total** | **14:30** |

---

## Slide 1 — What happens after you POST? (0:30)

**Key message:** This talk opens the black box between an HTTP request and streamed tokens.

**On-slide text:**
- `POST /v1/chat/completions`
- 15 minutes, 5 ideas

**Visual:** Three-panel strip. Left: a short `curl` snippet with a chat request. Middle: a large dark box labeled "?" with a vLLM logo hint. Right: tokens streaming out one by one (`"The" → " quick" → " brown" → …`). Arrow flows left to right.

**Speaker notes:** Everyone here has called an OpenAI-style API. Between that POST and the tokens streaming back sits an inference engine, and vLLM is the most widely used open-source one. In 15 minutes we'll open this box. You need zero ML background — this is a systems talk: it's about memory, scheduling, and batching, which is home turf for software engineers.

---

## Slide 2 — One token at a time (1:30)

**Key message:** An LLM is a function you call in a loop: each call reads everything so far and appends exactly one token.

**On-slide text:**
- one forward pass → one token
- output feeds back as input

**Visual:** Loop diagram. A row of prompt tokens (`["Why", " is", " the", " sky"]`) enters a box labeled "model (one forward pass)". The box emits one token (`" blue"`), which is appended to the row, with a curved arrow feeding the extended row back into the box. A step counter (`step 1, 2, 3…`) advances; ideally animated per keystroke/click so the sequence visibly grows one token per step. End state shows generation stopping at an `<end>` token.

**Speaker notes:** Strip away everything else and an LLM is a next-token predictor. One forward pass through the model consumes the whole sequence so far and produces a probability distribution over the next token; we sample one token, append it, and call the model again. A 500-token answer means 500 sequential forward passes — you cannot produce token 400 before token 399. This loop is autoregressive generation, and its serial nature is the root cause of everything else in this talk.

---

## Slide 3 — Two phases: prefill vs decode (1:30)

**Key message:** Serving one request has two very different phases — prefill (process the whole prompt at once) and decode (one token per step) — and each has its own latency metric.

**On-slide text:**
- prefill: all prompt tokens, one pass → TTFT
- decode: one token per pass → ITL

**Visual:** **Reuse `latency_diagram.png`** — user at the bottom, vLLM server at the top, query going up, `token 1 … token n` coming down, with TTFT, ITL, and e2e-latency brackets already drawn. Add only two overlay labels of our own: "prefill" over the query→token-1 span and "decode" over the token-1→token-n span, in two distinct colors (these two colors recur on slide 8's caption and slide 10's tags).

**Speaker notes:** The prompt tokens all exist up front, so the engine can process them in a single big parallel pass — that's prefill, and it ends when the first output token appears, which the user experiences as time-to-first-token. Then the loop from the previous slide takes over: decode, one token per pass, and the gap between tokens is the inter-token latency users feel as "typing speed". These two phases stress the hardware in opposite ways — prefill is a big parallel computation, decode is thousands of tiny sequential ones — and that asymmetry drives the engine design we'll see next.

---

## Slide 4 — Decode is memory-bound (1:30)

**Key message:** Each decode step must stream all model weights from GPU memory to compute one token — the bottleneck is memory bandwidth, not compute.

**On-slide text:**
- every step: read ALL weights
- GPU compute sits idle at batch = 1
- fix: share the read across a batch

**Visual:** **Reuse `roofline.png`** — the blog's roofline chart (perf vs arithmetic intensity, with "mem bw bound" and "compute bound" zones). Add two overlay dots of our own: `batch = 1` low on the bandwidth-bound slope, `batch = 32` up near the knee, with an arrow between them labeled "same weight read, 32× the tokens". Optional small side sketch (custom, only if time to build): HBM cylinder → "memory bandwidth" pipe → compute grid, to make "reading all the weights" concrete for the audience before showing the chart.

**Speaker notes:** Here's the systems insight most people miss. To compute one decode token, the GPU must read every model weight from memory — tens of gigabytes — to do a comparatively tiny amount of math. So decode speed is set by memory bandwidth, and the expensive compute units mostly idle. The escape hatch: if 32 requests are decoding together, the engine reads the weights once and produces 32 tokens for the same memory traffic. Batching is nearly free throughput — which is why everything in vLLM is built to keep batches as full as possible.

---

## Slide 5 — The KV cache (1:30)

**Key message:** To avoid re-processing the whole sequence every step, the engine caches per-token attention state (keys/values) — and that cache grows with every token and eats GPU memory.

**On-slide text:**
- remember the past, don't recompute it
- grows every token, per request
- GPU memory = weights + KV cache

**Visual:** Left: the decode loop from slide 2, but the "re-read everything" arrow is replaced by a box labeled "KV cache" that the model reads from and appends one entry to per step — show the cache bar growing tick by tick. Right: a fixed-height container labeled "GPU memory": bottom section "model weights (fixed)", above it several per-request KV bars of different lengths growing upward toward a "full" line. Callout: "MB per token per request → GBs at scale".

**Speaker notes:** Attention lets each new token look back at every previous token. Recomputing that state for the whole sequence on every step would be quadratically wasteful, so engines cache each token's keys and values — the KV cache — and each step only computes the new token's entry. The price is memory: every request carries a cache that grows linearly with its length, at roughly megabytes per token for mid-size models, so a handful of long conversations claims gigabytes. Remember slide 4: throughput comes from big batches, and batch size is now limited by how many KV caches fit in GPU memory. KV memory management *is* the serving problem.

---

## Slide 6 — The naive serving problem (1:00)

**Key message:** Pre-allocating contiguous KV memory per request wastes most of it, and static batching makes finished requests wait for the slowest one.

**On-slide text:**
- reserve for max length → mostly unused
- static batch → everyone waits for the longest

**Visual:** Two stacked panels. Top — "memory": a horizontal GPU-memory strip divided into equal contiguous reservations per request; within each, only a small colored head is "actually used", the long grey tail labeled "reserved, never used"; a rejected request bounces off the full strip. Bottom — "time": a static batch of 4 request rows starting together; three finish early and turn into hatched "idle" bars while the fourth still runs; new requests queue outside the batch boundary.

**Speaker notes:** The pre-vLLM approach: since a request's final length is unknown, reserve contiguous memory for the maximum possible length. Most requests stop far short, so the majority of that reservation is dead — real deployments wasted well over half of KV memory to fragmentation and over-reservation, directly shrinking the batch. And with static batching, the GPU runs a fixed group to completion: short requests finish and their slots sit idle while everyone queued outside waits for the longest member. Two classic systems problems — memory fragmentation and head-of-line blocking. vLLM's two signature ideas answer them one-to-one.

---

## Slide 7 — PagedAttention (2:00)

**Key message:** vLLM manages KV cache like an OS manages RAM — fixed-size blocks, allocated on demand, with a per-request block table mapping logical positions to scattered physical blocks.

**On-slide text:**
- KV memory in fixed blocks (16 tokens)
- allocate on demand, zero over-reservation
- block table: logical → physical

**Visual:** Simple custom, side-by-side analogy with mirrored layout. Left ("your OS"): a process's contiguous virtual address space → page table → scattered physical RAM frames. Right ("vLLM"): a request's logical token sequence chunked into 16-token blocks → block table → scattered physical KV blocks in GPU memory, plus a "free block pool" bucket blocks are grabbed from on demand and returned to when the request ends. Identical arrow styles on both sides so the analogy lands visually. No metadata, no hashes, no code — max ~12 boxes total. Small footer contrast: slide-6's wasteful strip vs this tightly packed grid.

**Speaker notes:** This is the idea vLLM is named for. Instead of one contiguous slab per request, KV memory is carved into fixed-size blocks — 16 tokens each — and a request grabs a new block from a shared free pool only when it actually fills one. A per-request block table maps logical token positions to physical blocks, exactly like a page table maps virtual to physical memory, so physical blocks can live anywhere. Over-reservation disappears: waste is bounded by one partially-filled block per request. Freed blocks return to the pool instantly for anyone else. If you've ever taken an OS course, you already understood PagedAttention — it's virtual memory for the KV cache, and it's what lets vLLM pack far more concurrent requests into the same GPU.

---

## Slide 8 — Continuous batching (2:00)

**Key message:** vLLM rebuilds the batch every single step, so requests join and leave mid-flight and the GPU never idles waiting for stragglers.

**On-slide text:**
- new batch every step
- join instantly, leave instantly
- prefill + decode mixed in one pass

**Visual:** Simple custom grid (inspired by the blog's Figure 4, radically simplified). Columns = engine steps (t1…t10), rows = requests R1…R5. Cells colored with the slide-3 palette: prefill color, decode color, empty when absent. R1 and R2 start at t1; R3 arrives at t3 (prefill cell appears mid-grid); R2 finishes at t5 and its row goes empty; R4 takes the freed capacity at t6. Bottom annotation row: each column's cells flatten into a single bar labeled "one forward pass". No token ids, no slot mappings — just colored cells. Caption contrast vs slide 6: "no idle slots, no waiting for the batch".

**Speaker notes:** Second signature idea. Before each forward pass — each column here — the scheduler re-decides the batch: finished requests exit immediately and free their blocks, and waiting requests are pulled in immediately, running their prefill in the same pass where others decode; vLLM flattens all their tokens into one sequence for the GPU. Compare with static batching: no idle hatched bars, no queue stuck at a batch boundary. This is what keeps the batch full — which slide 4 told us is where all the throughput lives. PagedAttention supplies flexible memory; continuous batching supplies flexible scheduling; each is what makes the other work at scale.

---

## Slide 9 — Free win: prefix caching (1:00)

**Key message:** Because KV lives in content-hashed blocks, identical prompt prefixes — like a shared system prompt — are computed once and reused by later requests.

**On-slide text:**
- same prefix → same blocks
- system prompt computed once
- skip prefill, cut TTFT

**Visual:** Simple custom, two request rows top-down. Request A: blocks `[sys][sys][sys][userA…]`, all in "computed" color. Request B below: its first three blocks are dashed outlines with arrows pointing up to A's physical blocks, labeled "reused (hash match)"; only `[userB…]` blocks are in computed color. One small `hash(block contents) → block` tag on a single shared block — no hash-map internals. Timeline inset: B's prefill bar visibly shorter, TTFT bracket shrunk.

**Speaker notes:** Paging has a bonus. Since blocks are fixed-size chunks of token content, vLLM hashes each block's contents — the KV cache becomes content-addressable. When a new request starts with token blocks the engine has already computed — the app's system prompt, shared few-shot examples, earlier turns of the same chat — it just points its block table at the existing physical blocks and skips that part of prefill entirely. In production, where nearly every request shares a long system prompt, this routinely eliminates most prefill work and slashes time-to-first-token. Notice it costs nothing extra: it falls out of the block design from slide 7.

---

## Slide 10 — vLLM in one picture (1:30)

**Key message:** Everything so far assembles into one loop: schedule → forward pass → postprocess, running around the paged KV cache.

**On-slide text:**
- schedule → forward → postprocess
- repeat every ~10 ms

**Visual:** Split layout. Left: simple custom architecture map, ≤6 boxes — "API server (OpenAI-compatible)" on top, below it an "Engine core" ring containing **Scheduler** (waiting/running queues), **Model executor / GPU forward pass**, **postprocess: sample & stream**, with the **KV cache manager + block pool** in the center. Right: **reuse `engine_loop.png`** — the blog's clean drawing of a request entering the waiting queue and the schedule → forward pass → postprocess ring. Overlay small colored tags mapping boxes back to earlier slides: scheduler → "slide 8", model executor → "slides 2–4", KV cache manager → "slides 5/7/9", each echoing that slide's accent color so the map reads as a recap.

**Speaker notes:** Here's the whole machine, and you already know every box. Requests land on an OpenAI-compatible API server, get tokenized, and enter the scheduler's waiting queue. Then the engine loops: the scheduler builds this step's batch — admitting prefills, continuing decodes, asking the KV manager for blocks; the model executor runs one flattened forward pass on the GPU; postprocessing samples each request's next token and streams it back. Then the loop runs again, milliseconds later, on a freshly re-formed batch. That's it — the black box from slide 1 is a tight scheduler-plus-allocator loop wrapped around a GPU.

---

## Slide 11 — What to measure & takeaways (1:30)

**Key message:** Judge a serving stack by TTFT, ITL, and throughput — knowing they trade off — and remember five ideas.

**On-slide text:**
- TTFT · ITL · throughput — pick your tradeoff
- vllm.ai/blog — "Anatomy of vLLM"

**Visual:** Left: latency-vs-throughput curve — x-axis "batch size / load", left y-axis "tokens/sec" rising then flattening at a marked **saturation knee**, right y-axis "per-token latency" flat then rising past the knee; two zone labels: "latency-friendly" before the knee, "throughput territory" after. Right: recap strip of five icons with two-word labels — token loop (2) · prefill/decode (3) · KV cache (5) · PagedAttention (7) · continuous batching (8). Footer: QR code / link to the Anatomy of vLLM post.

**Speaker notes:** When you deploy or benchmark this, three numbers matter: time to first token, inter-token latency, and total token throughput. They fight each other: growing the batch is nearly free throughput up to the saturation point — that's slide 4's bandwidth story — but past the knee, each request's tokens arrive slower. Chat UIs care about the left of this curve; batch pipelines care about the right; vLLM exposes the knobs to pick your point. If you keep five things: models emit one token at a time; prefill and decode are different workloads; the KV cache is the scarce resource; PagedAttention manages it like virtual memory; continuous batching keeps the GPU full. Everything deeper — speculative decoding, quantization, multi-GPU serving — is in the Anatomy of vLLM post, which this talk is built on. Questions?
