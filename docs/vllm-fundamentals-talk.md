# vLLM Fundamentals — 15-minute talk, slide content spec

- **Audience:** software engineers, little/no LLM-inference background
- **Duration:** 18 min (15 slides). For a hard 15-minute slot, the designated cut is the advanced trio (slides 12–14) — dropping them restores exactly 15:00 without touching the core arc.
- **Goal:** after the talk, the audience can explain why serving LLMs is a memory problem and what vLLM's two core tricks (PagedAttention, continuous batching) do about it
- **Style:** one big visual per slide, plus two concise lede bullets (each fits one line — never wrapped mid-sentence) and a row of fact pills with concrete numbers — enough that the slides teach on their own; the speaker notes carry the narrative
- **Primary source:** [Anatomy of vLLM](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) (vLLM team, Sep 2025)
- **Visuals policy:** reuse a blog figure only when it is simple enough to read from the back of the room in ~5 seconds. The blog's worked-example figures (block metadata, hash maps, slot mappings) are great for reading, too dense for a slide — for those, build a simple-stupid diagram of our own instead. Credit "Figure: Anatomy of vLLM, vllm.ai" on every reused image.

## Reused figure assets (simple ones only)

All under `https://vllm.ai/blog-assets/figures/2025-vllm-anatomy/`:

| File | Shows | Used on slide |
|------|-------|---------------|
| `engine_constructor.png` (**top panel only, cropped**) | requests in → processor → engine core (scheduler, KV cache manager, model executor) → output processor → result out | 2 |
| `latency_diagram.png` | query → token timeline with TTFT / ITL / e2e brackets | 4 |
| `roofline.png` | roofline: memory-bandwidth-bound vs compute-bound zones | 5 |
| `engine_loop.png` | waiting queue → schedule / forward pass / postprocess ring | 11 |

Crop note for `engine_constructor.png`: use only the top architecture panel — cut everything from the green "indexing structure" label downward. The panel alone passes the 5-second rule; the full image does not.

Rejected as too dense for slides (fine as speaker prep / "further reading"): `kv_cache_blocks.png`, `fwd_pass.png`, `prefix_pt1-3.png`, and the lower panels of `engine_constructor.png` — all worked examples with heavy annotation text; the concept slides use simple custom diagrams instead.

Download once for offline slides:

```bash
mkdir -p assets && cd assets
for f in latency_diagram roofline engine_loop engine_constructor; do
  curl -sLO "https://vllm.ai/blog-assets/figures/2025-vllm-anatomy/$f.png"
done
```

Note: figures are drawn light-background; on a dark deck place them on a white rounded card.

## Numbers used on slides (sources)

| Number | Slide | Source / derivation |
|--------|-------|---------------------|
| 7B @ fp16 ≈ 14 GB of weights | 5 | arithmetic: 7e9 params × 2 bytes |
| A100 HBM ≈ 2 TB/s → ~140 tok/s ceiling at batch = 1 | 5 | A100 80GB spec (2.0 TB/s); 2 TB/s ÷ 14 GB ≈ 143/s — labeled *back-of-envelope* on the slide |
| ≈ 0.5 MB KV per token (7B-class, fp16) | 6 | arithmetic: 2 (K,V) × 32 layers × 4096 hidden × 2 bytes = 512 KB |
| 4k-token chat ≈ 2 GB per request | 6 | 4096 × 0.5 MB |
| 60–80% of KV memory wasted pre-vLLM | 7, 8 | PagedAttention paper (Kwon et al., SOSP 2023) |
| < 4% KV waste with PagedAttention | 8 | same paper |
| block size = 16 tokens | 8 | vLLM default |
| prefix caching on by default | 10 | vLLM v1 default |
| 14 GB → 7 GB (fp8) → 3.5 GB (int4) | 12 | arithmetic: same 7e9 params at 2 / 1 / 0.5 bytes |

## Agenda / time budget

Slide titles are the pure concept being introduced (the hook slide 1 excepted); the lede sentence under the title carries the explanation.

Bookend structure: the architecture map is shown FIRST (slide 2) as orientation — "here's the whole machine, we'll open each box" — and revisited near the end (slide 11) once every box is understood.

| # | Slide | Time |
|---|-------|------|
| 1 | What happens after you POST? (hook) | 0:30 |
| 2 | vLLM architecture — the map | 1:00 |
| 3 | The token loop | 1:30 |
| 4 | Prefill & decode | 1:30 |
| 5 | Memory-bound decode | 1:30 |
| 6 | The KV cache | 1:30 |
| 7 | Fragmentation & head-of-line blocking | 1:00 |
| 8 | PagedAttention | 2:00 |
| 9 | Continuous batching | 2:00 |
| 10 | Prefix caching | 1:00 |
| 11 | vLLM architecture — revisited | 1:00 |
| 12 | Quantization | 1:00 |
| 13 | Speculative decoding | 1:00 |
| 14 | CUDA graphs & host overhead | 1:00 |
| 15 | The latency–throughput tradeoff | 1:30 |
| | **Total** | **18:00** |

## Scope: what this talk deliberately skips

The advanced levers (quantization, speculative decoding, CUDA graphs & host overhead) are taught as short slides 12–14, placed after the architecture revisit closes the core arc; chunked prefill gets one pill on slide 9. Still skipped, named on slide 15's "not covered today" strip:

- **Multi-GPU serving** (tensor/pipeline parallelism, disaggregated prefill/decode) — scale-out once one GPU isn't enough; a different problem class from the single-GPU mental model this talk builds.

Practical follow-up topic for a second talk: the operator knobs (`--max-num-seqs`, `--max-model-len`, `gpu_memory_utilization`) — each maps directly onto a concept slide here.

---

## Slide 1 — What happens after you POST? (0:30)

**Key message:** This talk opens the black box between an HTTP request and streamed tokens.

**On-slide text:**
- `POST /v1/chat/completions`
- 15 minutes · 5 ideas · zero ML background needed
- roadmap strip: token loop → prefill/decode → KV cache → PagedAttention → continuous batching

**Visual:** Three-panel strip. Left: a short `curl` snippet with a chat request. Middle: a large dark box labeled "?" with a vLLM logo hint. Right: tokens streaming out one by one (`"The" → " quick" → " brown" → …`). Arrow flows left to right.

**Speaker notes:** Everyone here has called an OpenAI-style API. Between that POST and the tokens streaming back sits an inference engine, and vLLM is the most widely used open-source one. In 15 minutes we'll open this box. You need zero ML background — this is a systems talk: it's about memory, scheduling, and batching, which is home turf for software engineers.

---

## Slide 2 — vLLM architecture — the map (1:00)

**Key message:** Here is the whole machine; the rest of the talk opens one box at a time.

**On-slide text:**
- lede bullets: `request → processor → engine core → output processor → response` (mono) / engine core = scheduler + KV cache manager + model executor / every box gets its own slide
- pill: scheduler → slide 9 · model executor → slides 3–5 · KV cache manager → slides 6/8/10

**Visual:** **Reuse the top panel of `engine_constructor.png`, cropped** (cut everything from the green "indexing structure" label downward) on a white rounded card, standard credit line. Overlay small forward-pointing slide tags on three boxes: scheduler → "9", model executor → "3–5", KV cache manager → "6/8/10", using the accent colors those slides will use.

**Speaker notes:** Before any theory, here's the map of what we're opening. A request comes in, a processor tokenizes it, and it enters the engine core — the heart of vLLM — which has three moving parts: a scheduler with waiting and running queues, a KV cache manager, and a model executor that drives the GPU. Results flow back out through an output processor. Don't worry about what any of these do yet — every box on this map gets its own slide, and we'll return to this exact picture at the end when you can read it fluently.

---

## Slide 3 — The token loop (1:30)

Title uses the plain term; the formal name "autoregressive generation" is introduced on-slide as a labeled pill, not assumed.

**Key message:** An LLM is a function you call in a loop: each call reads everything so far and appends exactly one token.

**On-slide text:**
- lede bullets: one forward pass reads everything so far — and predicts just the next token / append it, feed it back, repeat until `<end>`
- sub-line under the lede (small mono): a token ≈ a word piece — "inference" is two of them
- one pass → one token
- 500-token answer = 500 sequential passes
- no token 400 before token 399 — strictly serial
- formal name: "autoregressive generation" — the output feeds back as input

**Visual:** Loop diagram. A row of prompt tokens (`["Why", " is", " the", " sky"]`) enters a box labeled "model (one forward pass)". The box emits one token (`" blue"`), which is appended to the row, with a curved arrow feeding the extended row back into the box. A step counter (`step 1, 2, 3…`) advances; ideally animated per keystroke/click so the sequence visibly grows one token per step. End state shows generation stopping at an `<end>` token.

**Speaker notes:** Strip away everything else and an LLM is a next-token predictor. One forward pass through the model consumes the whole sequence so far and produces a probability distribution over the next token; we sample one token, append it, and call the model again. A 500-token answer means 500 sequential forward passes — you cannot produce token 400 before token 399. This loop is autoregressive generation, and its serial nature is the root cause of everything else in this talk.

---

## Slide 4 — Prefill & decode (1:30)

**Key message:** Serving one request has two very different phases — prefill (process the whole prompt at once) and decode (one token per step) — and each has its own latency metric.

**On-slide text:**
- lede bullets: prompt tokens all exist up front → one big parallel pass (prefill) / after that: one pass per new token, forever (decode)
- prefill: whole prompt, one pass → TTFT = how long until the answer starts
- decode: one token per pass → TPOT/TBT = "typing speed"
- credit line carries the bridge: the figure's "ITL" label = TPOT/TBT, same metric
- prefill = compute-heavy, parallel · decode = thousands of tiny sequential steps

**Visual:** **Reuse `latency_diagram.png`** — user at the bottom, vLLM server at the top, query going up, `token 1 … token n` coming down, with TTFT, ITL, and e2e-latency brackets already drawn. Add only two overlay labels of our own: "prefill" over the query→token-1 span and "decode" over the token-1→token-n span, in two distinct colors (these two colors recur on slide 9's caption and slide 11's tags).

**Speaker notes:** The prompt tokens all exist up front, so the engine can process them in a single big parallel pass — that's prefill, and it ends when the first output token appears, which the user experiences as time-to-first-token. Then the loop from the previous slide takes over: decode, one token per pass, and the gap between tokens is the time per output token — TPOT, also called TBT, and labeled ITL on the figure — which users feel as "typing speed". These two phases stress the hardware in opposite ways — prefill is a big parallel computation, decode is thousands of tiny sequential ones — and that asymmetry drives the engine design we'll see next.

---

## Slide 5 — Memory-bound decode (1:30)

**Key message:** Each decode step must stream all model weights from GPU memory to compute one token — the bottleneck is memory bandwidth, not compute.

**On-slide text:**
- lede bullets: one decode token = streaming EVERY model weight from GPU memory / bandwidth sets the speed, not FLOPs
- note card: 7B @ fp16 ≈ 14 GB of weights, read every step · A100 HBM ≈ 2 TB/s → ~140 tok/s ceiling at batch = 1 (back-of-envelope) · batch = 1 low on the slope, compute idle · batch = 32 GPU nearly fully busy: same weight read, 32× the tokens · x-axis "arithmetic intensity" = math done per byte moved — decode is far left
- pills: decode speed ≈ bandwidth ÷ bytes moved · batching is nearly free throughput

**Visual:** **Reuse `roofline.png`** — the blog's roofline chart (perf vs arithmetic intensity, with "mem bw bound" and "compute bound" zones). Add two overlay dots of our own: `batch = 1` low on the bandwidth-bound slope, `batch = 32` up near the knee, with an arrow between them labeled "same weight read, 32× the tokens". Optional small side sketch (custom, only if time to build): HBM cylinder → "memory bandwidth" pipe → compute grid, to make "reading all the weights" concrete for the audience before showing the chart.

**Speaker notes:** Here's the systems insight most people miss. To compute one decode token, the GPU must read every model weight from memory — tens of gigabytes — to do a comparatively tiny amount of math. So decode speed is set by memory bandwidth, and the expensive compute units mostly idle. The escape hatch: if 32 requests are decoding together, the engine reads the weights once and produces 32 tokens for the same memory traffic. Batching is nearly free throughput — which is why everything in vLLM is built to keep batches as full as possible.

---

## Slide 6 — The KV cache (1:30)

**Key message:** To avoid re-processing the whole sequence every step, the engine caches per-token attention state (keys/values) — and that cache grows with every token and eats GPU memory.

**On-slide text:**
- lede bullets: attention: each new token is compared against every previous token / cache each token's comparison data (keys/values) — compute once, reuse
- ≈ 0.5 MB per token (7B-class, fp16)
- 4k-token chat ≈ 2 GB — per request
- batch size is now capped by KV memory

**Visual:** Left: the decode loop from slide 3, but the "re-read everything" arrow is replaced by a box labeled "KV cache" that the model reads from and appends one entry to per step — show the cache bar growing tick by tick. Right: a fixed-height container labeled "GPU memory": bottom section "model weights (fixed)", above it several per-request KV bars of different lengths growing upward toward a "full" line. Callout: "MB per token per request → GBs at scale".

**Speaker notes:** Attention lets each new token look back at every previous token. Recomputing that state for the whole sequence on every step would be quadratically wasteful, so engines cache each token's keys and values — the KV cache — and each step only computes the new token's entry. The price is memory: every request carries a cache that grows linearly with its length, at roughly megabytes per token for mid-size models, so a handful of long conversations claims gigabytes. Remember slide 5: throughput comes from big batches, and batch size is now limited by how many KV caches fit in GPU memory. KV memory management *is* the serving problem.

---

## Slide 7 — Fragmentation & head-of-line blocking (1:00)

**Key message:** Pre-allocating contiguous KV memory per request wastes most of it, and static batching makes finished requests wait for the slowest one.

**On-slide text:**
- lede bullets: output length unknown → pre-reserve worst-case contiguous memory / fixed batch → stragglers hold the whole GPU
- 60–80% of KV memory wasted in pre-vLLM systems (vLLM paper)
- two classic systems problems → two vLLM ideas, next

**Visual:** Two stacked panels. Top — "memory": a horizontal GPU-memory strip divided into equal contiguous reservations per request; within each, only a small colored head is "actually used", the long grey tail labeled "reserved, never used"; a rejected request bounces off the full strip. Bottom — "time": a static batch of 4 request rows starting together; three finish early and turn into hatched "idle" bars while the fourth still runs; new requests queue outside the batch boundary.

**Speaker notes:** The pre-vLLM approach: since a request's final length is unknown, reserve contiguous memory for the maximum possible length. Most requests stop far short, so the majority of that reservation is dead — real deployments wasted well over half of KV memory to fragmentation and over-reservation, directly shrinking the batch. And with static batching, the GPU runs a fixed group to completion: short requests finish and their slots sit idle while everyone queued outside waits for the longest member. Two classic systems problems — memory fragmentation and head-of-line blocking. vLLM's two signature ideas answer them one-to-one.

---

## Slide 8 — PagedAttention (2:00)

**Key message:** vLLM manages KV cache like an OS manages RAM — fixed-size blocks, allocated on demand, with a per-request block table mapping logical positions to scattered physical blocks.

**On-slide text:**
- lede bullets: carve KV memory into fixed 16-token blocks / a per-request block table maps logical position → any free physical block
- sub-line under the lede (small mono): the "v" in vLLM = virtual memory — this idea named the whole project
- equivalence chips: virtual page ≡ 16-token block · page table ≡ block table · RAM frame ≡ KV block
- waste ≤ one partial block per request
- measured waste: < 4%, vs 60–80% before
- (dropped "freed blocks return instantly" pill — the free-pool note inside the column already says it)

**Visual:** Simple custom, side-by-side analogy with mirrored layout. Left ("your OS"): a process's contiguous virtual address space → page table → scattered physical RAM frames. Right ("vLLM"): a request's logical token sequence chunked into 16-token blocks → block table → scattered physical KV blocks in GPU memory, plus a "free block pool" bucket blocks are grabbed from on demand and returned to when the request ends. Identical arrow styles on both sides so the analogy lands visually; both physical grids carry mirrored captions ("physical RAM (scattered)" / "physical KV memory on GPU (scattered)"). Below the columns, an explicit equivalence strip of three mono chips: virtual page ≡ 16-token block · page table ≡ block table · RAM frame ≡ KV block. No metadata, no hashes, no code — max ~12 boxes total.

**Speaker notes:** This is the idea vLLM is named for. Instead of one contiguous slab per request, KV memory is carved into fixed-size blocks — 16 tokens each — and a request grabs a new block from a shared free pool only when it actually fills one. A per-request block table maps logical token positions to physical blocks, exactly like a page table maps virtual to physical memory, so physical blocks can live anywhere. Over-reservation disappears: waste is bounded by one partially-filled block per request. Freed blocks return to the pool instantly for anyone else. If you've ever taken an OS course, you already understood PagedAttention — it's virtual memory for the KV cache, and it's what lets vLLM pack far more concurrent requests into the same GPU.

---

## Slide 9 — Continuous batching (2:00)

**Key message:** vLLM rebuilds the batch every single step, so requests join and leave mid-flight and the GPU never idles waiting for stragglers.

**On-slide text:**
- lede bullets: before every pass the scheduler rebuilds the batch / finished leave, waiting join — prefill and decode run together
- join instantly, leave instantly
- all tokens flattened into one GPU pass
- GPU never waits for the slowest request
- long prompt? chunked prefill splits it across steps — no one stalls

**Visual:** Simple custom grid (inspired by the blog's Figure 4, radically simplified). Columns = engine steps (t1…t10), rows = requests R1…R5. Cells colored with the slide-4 palette: prefill color, decode color, empty when absent. R1 and R2 start at t1; R3 arrives at t3 (prefill cell appears mid-grid); R2 finishes at t5 and its row goes empty; R4 takes the freed capacity at t6. Bottom annotation row: each column's cells flatten into a single bar labeled "one forward pass". No token ids, no slot mappings — just colored cells. Caption contrast vs slide 7: "no idle slots, no waiting for the batch".

**Speaker notes:** Second signature idea. Before each forward pass — each column here — the scheduler re-decides the batch: finished requests exit immediately and free their blocks, and waiting requests are pulled in immediately, running their prefill in the same pass where others decode; vLLM flattens all their tokens into one sequence for the GPU. Compare with static batching: no idle hatched bars, no queue stuck at a batch boundary. This is what keeps the batch full — which slide 5 told us is where all the throughput lives. PagedAttention supplies flexible memory; continuous batching supplies flexible scheduling; each is what makes the other work at scale. And if someone asks "doesn't a huge prompt stall everyone?" — no: chunked prefill caps how many prompt tokens one request may contribute per step, spreading a long prefill across several passes.

---

## Slide 10 — Prefix caching (1:00)

**Key message:** Because KV lives in content-hashed blocks, identical prompt prefixes — like a shared system prompt — are computed once and reused by later requests.

**On-slide text:**
- lede bullets: blocks are content-hashed → the KV cache is content-addressable / same prefix = same blocks = that prefill is skipped
- shared system prompt / few-shot / chat history → computed once
- cuts TTFT on nearly every production request
- on by default in vLLM v1

**Visual:** Simple custom, two request rows top-down. Request A: blocks `[sys][sys][sys][userA…]`, all in "computed" color. Request B below: its first three blocks are dashed outlines with arrows pointing up to A's physical blocks, labeled "reused (hash match)"; only `[userB…]` blocks are in computed color. One small `hash(block contents) → block` tag on a single shared block — no hash-map internals. Timeline inset: B's prefill bar visibly shorter, TTFT bracket shrunk.

**Speaker notes:** Paging has a bonus. Since blocks are fixed-size chunks of token content, vLLM hashes each block's contents — the KV cache becomes content-addressable. When a new request starts with token blocks the engine has already computed — the app's system prompt, shared few-shot examples, earlier turns of the same chat — it just points its block table at the existing physical blocks and skips that part of prefill entirely. In production, where nearly every request shares a long system prompt, this routinely eliminates most prefill work and slashes time-to-first-token. Notice it costs nothing extra: it falls out of the block design from slide 8.

---

## Slide 11 — vLLM architecture — revisited (1:00)

**Key message:** The same map from slide 2 — but now the audience knows every box; the whole machine is a schedule → forward → postprocess loop around the paged KV cache.

**On-slide text:**
- lede bullets: the map from slide 2 — now you can read every box / `tokenize → queue → batch → one GPU pass → sample → stream` — repeat
- engine core loops every ~10 ms
- slide tags on the boxes: scheduler → 9 · model executor → 3–5 · KV manager → 6/8/10

**Visual:** Split layout. Left: simple custom architecture map, ≤6 boxes — "API server (OpenAI-compatible)" on top, below it an "Engine core" ring containing **Scheduler** (waiting/running queues), **Model executor / GPU forward pass**, **postprocess: sample & stream**, with the **KV cache manager + block pool** in the center. Right: **reuse `engine_loop.png`** — the blog's clean drawing of a request entering the waiting queue and the schedule → forward pass → postprocess ring. Overlay small colored tags mapping boxes back to earlier slides: scheduler → "slide 9", model executor → "slides 3–5", KV cache manager → "slides 6/8/10", each echoing that slide's accent color so the map reads as a recap.

**Speaker notes:** Remember the map from the start? You now know every box. Requests land on the API server, get tokenized, and enter the scheduler's waiting queue; every ~10 ms the loop runs — the scheduler builds a fresh batch, the KV manager hands out blocks, the model executor does one flattened GPU pass, postprocessing samples and streams each token. The black box from slide 1 is just a tight scheduler-plus-allocator loop wrapped around a GPU. (Keep this quick — it's a victory lap, not new material.)

---

## Slide 12 — Quantization (1:00)

**Key message:** Shrink the bytes: storing weights in 8 or 4 bits instead of 16 directly raises the memory-bound speed ceiling from slide 5.

**On-slide text:**
- lede bullets: decode speed ≈ bandwidth ÷ bytes moved (slide 5) / store weights in 8 or 4 bits instead of 16 — fewer bytes to move
- fp8 = half the bytes → ~2× the bandwidth ceiling
- accuracy cost is real — measure it, don't assume it
- the KV cache can be quantized too

**Visual:** Bit-width bars, one row per precision: fp16 full-width bar labeled "14 GB", fp8 half-width "7 GB", int4 quarter-width "3.5 GB" (the same 7B example from slide 5). Caption: "same model, fewer bytes per weight".

**Speaker notes:** Slide 5 said decode speed is bandwidth divided by bytes moved — so the most direct lever is moving fewer bytes. Quantization stores each weight in fewer bits: fp8 halves the weight traffic, int4 quarters it, and modern GPUs execute low-precision math natively. It is not free: the model loses a little precision, so quality must be measured on your workload, not assumed. Bonus: the KV cache can be quantized too, which both speeds decode and lets more requests fit — attacking slide 6's memory cap from the other side.

---

## Slide 13 — Speculative decoding (1:00)

**Key message:** Attack the serial token loop: a cheap draft proposes several tokens and the big model verifies them all in one pass — with output guaranteed identical.

**On-slide text:**
- lede bullets: the token loop is serial (slide 3) / a tiny draft guesses k tokens; the big model verifies all in ONE pass
- output provably identical to the big model alone
- wins only when the draft guesses well
- vLLM drafts: n-gram, EAGLE, Medusa

**Visual:** Flow: small "draft model" box emits 4 dashed gold token cells → arrow into a "big model — one verify pass" box → the 4 cells re-emerge as ✓✓✓ (green) and one ✗ (red, "rejected → resampled"). Caption: "3 tokens for the price of 1 big pass".

**Speaker notes:** The token loop is serial per slide 3 — but verification doesn't have to be. If a cheap draft guesses the next few tokens, the big model can check all of them in one prefill-shaped parallel pass, which slide 4 told us is cheap. Accept left-to-right until the first mismatch, resample there, repeat. The accept/reject rule is designed so the output distribution is exactly the big model's — this is a pure latency trick, not an approximation. It pays off on predictable text (code, boilerplate, common phrasing) and does nothing on high-entropy output where the draft keeps missing.

---

## Slide 14 — CUDA graphs & host overhead (1:00)

**Key message:** Between GPU passes the Python host schedules, samples, and launches kernels — at small batch those gaps dominate TPOT, and CUDA graphs shrink them by replaying a pre-recorded step.

**On-slide text:**
- lede bullets: each step: ~ms of GPU work, but Python schedules & launches in between / those host gaps add straight to TPOT
- thousands of kernel launches → one graph replay
- matters most at small batch / short steps
- vLLM captures graphs at startup

**Visual:** Two horizontal timelines. Top "eager": alternating segments — wide grey "host" gaps between blue "GPU" blocks. Bottom "CUDA graphs": thin host slivers, GPU blocks packed nearly back-to-back. Caption: "record once, replay every step".

**Speaker notes:** The GPU never launches its own work — the CPU does, kernel by kernel, with Python scheduling and sampling in between. Each decode step is only a few milliseconds of GPU time, so even a millisecond of host work between steps lands directly on the time between tokens. CUDA graphs fix the launch half: record the step's entire kernel sequence once, then replay it as a single unit — that's what vLLM's capture phase at startup is doing. Classic Amdahl: at big batch the long GPU step hides host time; at small batch, host overhead is the bottleneck.

---

## Slide 15 — The latency–throughput tradeoff (1:30)

**Key message:** Judge a serving stack by TTFT, TPOT, and throughput — knowing they trade off — and remember five ideas.

**On-slide text:**
- lede bullets: bigger batches buy throughput until the GPU is fully busy / past that point, every request's tokens arrive slower
- TTFT · TPOT · throughput — pick your tradeoff
- recap pills: one token at a time · prefill ≠ decode · KV cache = the scarce resource · PagedAttention = virtual memory · continuous batching keeps the GPU full
- strip: not covered today: multi-GPU serving — tensor/pipeline parallelism, disaggregated prefill/decode
- vllm.ai/blog — "Anatomy of vLLM"

**Visual:** Left: latency-vs-throughput curve — x-axis "batch size / load", left y-axis "tokens/sec" rising then flattening where the GPU becomes **fully busy** (chart label: "GPU fully busy"), right y-axis "per-token latency" flat then rising past that point; two zone labels: "latency-friendly" before it, "throughput territory" after. Right: recap strip of five icons with two-word labels — token loop (3) · prefill/decode (4) · KV cache (6) · PagedAttention (8) · continuous batching (9). Footer: QR code / link to the Anatomy of vLLM post.

**Speaker notes:** When you deploy or benchmark this, three numbers matter: time to first token, time per output token, and total token throughput. They fight each other: growing the batch is nearly free throughput until the GPU is fully busy — that's slide 5's bandwidth story (benchmarking docs call this the "saturation point") — but past it, each request's tokens arrive slower. If someone asks how both curves can rise at once: throughput is the aggregate across all requests (batch ÷ step time), latency is per-user (one token per step) — a bigger batch slows the step slightly while multiplying tokens per step, like a bigger bus: more passengers per hour, each trip a little slower. Chat UIs care about the left of this curve; batch pipelines care about the right; vLLM exposes the knobs to pick your point. If you keep five things: models emit one token at a time; prefill and decode are different workloads; the KV cache is the scarce resource; PagedAttention manages it like virtual memory; continuous batching keeps the GPU full. Everything deeper — multi-GPU serving and the rest — is in the Anatomy of vLLM post, which this talk is built on. Questions?
