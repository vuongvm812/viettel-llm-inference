#!/usr/bin/env python3
"""Regenerate ``data/input/trace-round2.jsonl`` as realistic multi-turn conversations that follow
the phase-2 grading spec ``data/input/trace_grading_public.jsonl``.

Each grading-spec row is one turn (`conv_id, turn_idx, timestamp_ms, think_ms, in_tokens_est,
out_tokens_max`, no bodies). We emit a replayable trace (`request_id, timestamp_ms, workload_type,
body`) where every conversation is a genuine human<->assistant dialogue:

  * A coherent **system prompt** = an assistant persona + a realistic reference document (a
    "briefing" assembled from a bank of self-contained notes, sized so the turn-0 prompt hits the
    spec's `in_tokens_est` ~4k). The document is fixed for a conversation -> it is a prefix-cache
    hit on every later turn, and it is unique per conversation (a per-session header + a per-conv
    shuffle) so there are no accidental cross-conversation hits.
  * **Accumulating turns** -- turn N's request carries the whole thread so far
    `[system, user_1, assistant_1, ..., user_N]`, exactly as a real chat client re-sends history.
    Real conversations grow, so later turns run a little longer than the spec's flat estimate; that
    is the realistic behaviour, and the growth is just the cached prefix plus one new (user,
    assistant) pair.
  * **Random think-pauses** -- a human pauses to read/think between turns, so each turn's arrival is
    `prev_arrival + response_time + a random pause` (seeded log-normal, ~a few seconds to a minute)
    rather than a fixed cadence. Seeded, so reruns are byte-identical.

Self-contained: all text is authored below (no external corpus), so it never breaks when a trace
file moves.

    python bench/build_trace_round2.py                 # from round-2/: writes data/input/trace-round2.jsonl
    python bench/build_trace_round2.py --self-check    # no model/IO; checks the assembly logic
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# The trace's per-request "model" field. It MUST equal --served-model-name in
# docker-compose.yaml or every replayed request 404s -- so the default here is the same
# placeholder that file carries, and both change together. Override with --model-name.
MODEL_NAME = "round2-model"
SEED = 42
# Nominal time the server takes to stream out_tokens_max, added between a turn and the next arrival
# (open-loop pacing only; does not affect prefill/cache behaviour).
RESPONSE_MS_PER_TOKEN = 8
# Human think-pause between turns: log-normal, median ~e^8.5 ~= 4.9s, clamped to [0.8s, 180s].
THINK_LOG_MU = 8.5
THINK_LOG_SIGMA = 0.8
THINK_MIN_MS, THINK_MAX_MS = 800, 180_000

# --- Reference material: a bank of short, self-contained notes an assistant might be grounded on.
# Assembled (shuffled, per conversation) into a "briefing" that gives each prompt its bulk while
# staying readable. Topics are deliberately generic so the analytical Q&A below fits any selection.
KB_NOTES = [
    "Onboarding: new hires get a buddy for their first two weeks, a checklist in the shared drive, and a 30-minute setup call on day one. Accounts should be provisioned the day before they start.",
    "Support SLAs: first response within 4 business hours for standard tickets and 1 hour for outages. Anything tagged 'billing' is routed to finance, not the general queue.",
    "Release process: changes ship on Tuesdays and Thursdays. A release needs one code review, a green CI run, and a rollback note. Friday deploys are avoided so nobody debugs over the weekend.",
    "Meeting hygiene: every recurring meeting needs an agenda in the invite or it gets declined. Notes and action items go in the shared doc within a day, with an owner and a due date per item.",
    "Expense policy: anything under 50 is auto-approved; 50-500 needs a manager sign-off; above 500 needs a director. Receipts are required within a week or the claim is voided.",
    "Data retention: raw logs are kept 30 days, aggregated metrics 13 months, and anything with personal data is purged on account deletion. Backups follow the same windows.",
    "Customer research: we run five interviews per feature before building, recruit from active users, and never ask leading questions. Findings are summarised as 'what we heard' vs 'what we'll do'.",
    "Incident review: blameless post-mortems within three business days, focused on the system and the process, not the person. Each one produces at most three concrete follow-up actions.",
    "Hiring loop: a screen, a take-home or pairing session, and two panel interviews. Feedback is submitted independently before the debrief so nobody anchors on the first opinion.",
    "Content style: short sentences, active voice, no jargon without a definition, and one idea per paragraph. Headings should say what the section is about, not just label it.",
    "Budget planning: departments submit a draft in Q3, defend it in Q4, and get a locked number by year-end. Anything unspent does not roll over, so sandbagging is discouraged.",
    "Security basics: hardware keys for admin access, no shared passwords, quarterly access reviews, and encryption at rest and in transit. Phishing drills run twice a year.",
    "Roadmap rules: at most three big bets per quarter, each with a single directly-responsible owner and a measurable outcome. Everything else is explicitly parked, not silently dropped.",
    "Vendor selection: shortlist three, run the same evaluation for each, check two references, and pilot before committing. Exit clauses and data-portability terms are non-negotiable.",
    "Performance reviews: twice a year, based on a rubric shared in advance, with examples not adjectives. Ratings are calibrated across managers before they are shared with anyone.",
    "Documentation: every service has a one-page README covering what it does, how to run it, who owns it, and where the dashboards are. Stale docs are treated as bugs.",
    "Pricing experiments: change one variable at a time, run for a full billing cycle, and pre-register the metric that decides success so results are not reinterpreted after the fact.",
    "Remote norms: default to written and async, keep meetings for decisions and relationships, and overlap at least three hours across time zones. Cameras optional, good notes mandatory.",
    "Quality gates: no feature ships without tests for the happy path and the top two failure modes, an accessibility pass, and a check on the slowest realistic input.",
    "Communication: escalate early and in writing, assume good intent, and separate the decision from the person who has to make it. Disagree in the room, commit outside it.",
    "Analytics hygiene: define each metric once in a shared glossary, instrument events at the source, and never let two dashboards compute the same number two different ways.",
    "Legal review: contracts over a year or 25k get a legal pass; anything touching user data gets a privacy pass. Turnaround is three business days if the template is used.",
    "Capacity planning: size for the 95th-percentile load, not the average, keep one failure domain of headroom, and load-test before the season, not during it.",
    "Feedback culture: give it close to the event, specific and kind, and tie it to impact rather than personality. Ask before giving unsolicited feedback outside a review.",
    "Prioritisation: score work by reach, impact, confidence, and effort; revisit the list weekly; and protect a fixed slice for maintenance so the interest on technical debt gets paid.",
    "Accessibility: sufficient colour contrast, full keyboard navigation, labelled form fields, and captions on video. Test with a screen reader before, not after, launch.",
]

# --- Conversation scenarios: an assistant persona plus a coherent six-turn dialogue whose questions
# work against any briefing. `answers` are the (synthetic) prior replies that make up the history;
# the live server generates the answer to the *last* user turn of each request.
SCENARIOS = [
    {
        "persona": "You are a sharp operations analyst. You have been given a set of internal notes and asked to help make sense of them. Be concise, structured, and practical, and ground every answer in the material provided.",
        "label": "Internal notes",
        "qa": [
            ("Can you give me a quick, plain-English overview of what these notes cover?",
             "Sure. At a high level the notes describe how the team runs day to day: how work ships, how support and incidents are handled, how people are hired and reviewed, and the guardrails around money, data, and security. It reads like an operating handbook rather than a single project."),
            ("Which items look most time-sensitive to act on first?",
             "The ones with hard deadlines or outage risk: the support and incident response windows, the release and quality gates, and anything touching security and data retention. Those fail loudly and quickly, so they earn priority over the planning and review cadences."),
            ("The parts about planning and budgeting are a bit dense — can you simplify them?",
             "In short: teams draft budgets mid-year, defend them, and get a locked number they can't roll over; and the roadmap caps big bets per quarter, each with one owner and a measurable outcome. The theme is 'few commitments, clear owners, no silent drift'."),
            ("Are there any gaps or risks you'd flag from what's here?",
             "A couple. The policies assume the templates and checklists are actually used, but nothing here enforces that. And several processes depend on a single owner, which is clean but fragile if that person is out. I'd add a light audit and a documented backup owner."),
            ("Can you turn the key points into a short action list I can share?",
             "Yes: 1) confirm SLAs and on-call are staffed; 2) verify release gates (review, CI, rollback) are enforced, not optional; 3) check access reviews and data-retention jobs are running; 4) name a backup owner for each single-owner process; 5) schedule the next planning cycle."),
            ("Thanks. What would you suggest we do next after that list?",
             "Once the list is done, pick the two items that are both high-impact and currently unowned and assign them this week. Then set a recurring monthly check to review the same notes against reality, so the handbook stays a description of what actually happens, not an aspiration."),
        ],
    },
    {
        "persona": "You are a patient study coach. You help someone understand a body of material by explaining it clearly and checking their understanding. Keep answers friendly, concrete, and free of jargon.",
        "label": "Study material",
        "qa": [
            ("I have to get across all of this material — where should I even start?",
             "Start by grouping it into themes rather than reading top to bottom. Most of it clusters into 'how work ships', 'how people are managed', and 'how risk is controlled'. Learn one cluster at a time and you'll retain far more than a single linear pass."),
            ("Can you explain the 'how work ships' part in simple terms?",
             "Think of it as a set of small promises: changes go out on set days, each one needs a review and a passing test and a way to undo it, and risky days are avoided. The goal is that shipping is boring and reversible, so a mistake is annoying rather than catastrophic."),
            ("How would you quiz me to check I've understood it?",
             "I'd ask you to explain, without looking, why Friday deploys are discouraged, what three things a release needs, and what happens to an expense over 500. If you can answer those in your own words, you've understood the shape, not just memorised the lines."),
            ("I keep mixing up the money rules — can you make them stick?",
             "Anchor them to one ladder: under 50 is automatic, up to 500 needs your manager, above that needs a director — and receipts within a week or it's void. It's the same 'bigger number, higher approver' pattern as most of the other policies, so it should feel familiar."),
            ("Can you summarise the whole thing as a few bullet points for revision?",
             "1) Ship small, reviewed, reversible changes on set days. 2) Manage people with rubrics, calibration, and timely feedback. 3) Control risk with SLAs, access reviews, and retention limits. 4) Plan few things, each with one owner. 5) Write everything down and keep it current."),
            ("Great — what's a good way to test myself tomorrow?",
             "Close the notes and try to reconstruct those five bullets from memory, then expand each into two sentences. Anywhere you stall is exactly the section to reread. That active recall will do more in ten minutes than an hour of passive rereading."),
        ],
    },
    {
        "persona": "You are a meticulous editor and communications advisor. You help turn rough material into something clear and usable. Be direct about what works, what doesn't, and how to fix it.",
        "label": "Draft material",
        "qa": [
            ("Can you tell me what's working and what isn't in this material?",
             "What works: it's specific, each item has a rule and a reason, and it avoids vague aspiration. What's weaker: it's a flat list with no priority, so a reader can't tell the load-bearing policies from the nice-to-haves, and some items assume context a newcomer won't have."),
            ("How would you reorganise it to be easier to follow?",
             "Group it under three or four headings — shipping, people, risk, planning — and within each, lead with the one rule that matters most. A short 'if you only remember one thing' line per section would carry most of the value for a busy reader."),
            ("Some of it feels wordy. Can you show me a tighter version of one part?",
             "Take the meeting rule: instead of a long sentence, 'No agenda, no meeting. Notes and owners within a day.' Same content, half the words, and it's memorable. Most items here can lose their qualifiers without losing meaning."),
            ("Is anything unclear or likely to be misread?",
             "The retention windows could be misread as applying uniformly; it's worth stating plainly that personal data is the exception and is purged on deletion. And 'auto-approved' should say who is still accountable, so it isn't read as 'unmonitored'."),
            ("Can you give me a cleaned-up summary I could put at the top?",
             "Something like: 'This handbook covers how we ship, manage people, control risk, and plan. Each policy states a rule and its reason. When in doubt, prefer the smaller, reversible, well-owned option, and write down what you decided.'"),
            ("Perfect. Any final suggestions before I send it round?",
             "Two: add a last-updated date and an owner so people trust it's current, and link each section to the tool or dashboard it refers to. A handbook people can act from beats a handbook people merely read."),
        ],
    },
    {
        "persona": "You are a pragmatic consultant. You read the material and give balanced, decision-oriented advice, always separating what's certain from what's a judgement call.",
        "label": "Briefing document",
        "qa": [
            ("Give me your honest read on this — what's the overall picture?",
             "It's a mature, sensible operating model: small reversible changes, clear approval ladders, and blameless learning. Nothing here is exotic, which is a good sign — the risk in setups like this is usually enforcement and drift, not the policies themselves."),
            ("If you had to pick the biggest risk, what would it be?",
             "Single points of ownership. Naming one directly-responsible person is great for clarity but brittle when that person is unavailable. I'd treat 'who's the backup?' as the first question for every process that names exactly one owner."),
            ("What would you recommend we prioritise this quarter?",
             "Enforcement over new policy. Verify the release gates, access reviews, and retention jobs actually run, and that SLAs are staffed. You'll get more value from making the existing rules real than from writing new ones."),
            ("Are there any quick wins we could do this week?",
             "Yes: add last-updated dates and owners to each section, name backup owners, and turn the SLAs into an on-call schedule someone actually holds. All low-effort, all reduce real risk, none require a big project."),
            ("Can you frame this as a short recommendation I can take to leadership?",
             "'The operating model is sound; the gap is enforcement and resilience. Recommend a one-quarter effort to verify controls are running, remove single points of ownership, and keep the handbook current — low cost, meaningful risk reduction, no new headcount.'"),
            ("Understood. What should we measure to know it worked?",
             "Track control coverage (percentage of processes with a live owner and backup), SLA attainment, and how current the docs are. If those three move in the right direction over the quarter, the effort is paying off."),
        ],
    },
    {
        "persona": "You are a friendly onboarding guide. Someone just joined and is trying to understand how things work here. Explain things warmly and walk them through what matters, one step at a time.",
        "label": "Team handbook",
        "qa": [
            ("I just joined and this is a lot — what do I actually need to know first?",
             "Welcome! For week one, focus on three things: how we ship changes, how we handle support and incidents, and the basic money and security rules. The rest — planning, reviews, hiring — you'll pick up as it comes around. You don't need it all today."),
            ("How does shipping work here — I don't want to break anything?",
             "You won't easily. Changes go out on set days, and each one needs a review, a passing test, and a rollback note, so there's always a safety net. When in doubt, ship something small and reversible and ask your buddy to look first."),
            ("What do I do if something breaks or a customer is stuck?",
             "For customers, respond within the SLA and route billing issues to finance. For outages, it's a fast first response and then a blameless review afterwards focused on the system, not you. Nobody here gets blamed for surfacing a problem early."),
            ("Are there money or security things I should be careful about from day one?",
             "A few easy ones: small expenses are auto-approved but still need receipts within a week, use your hardware key for anything sensitive, and never share passwords. If you're unsure whether something's allowed, ask first — that's always the safe move."),
            ("Can you give me a simple checklist for my first two weeks?",
             "Sure: meet your buddy, get your accounts set up, read the one-page README for your main service, ship one tiny reversible change, sit in on a support rotation, and skim the handbook sections you haven't hit yet. That's a solid first fortnight."),
            ("This helps a lot, thank you. Who do I ask if I'm still stuck?",
             "Your buddy first — that's exactly what they're there for in weeks one and two — then your manager. And escalate early and in writing rather than sitting on something; here that's seen as responsible, not as a bother."),
        ],
    },
    {
        "persona": "You are a thoughtful peer thinking through some shared notes with a colleague. Keep it conversational and collaborative, offering opinions while staying grounded in what the notes actually say.",
        "label": "Shared notes",
        "qa": [
            ("I've been staring at these notes for a while — what jumps out to you?",
             "Honestly, how consistent the underlying pattern is: almost every rule is 'keep it small, give it one owner, write it down, make it reversible'. Once you see that pattern, most of the individual policies stop needing to be memorised — they're the same idea applied to different areas."),
            ("Do you think anything in here is over-engineered?",
             "Maybe the number of approval tiers, depending on team size — three levels for expenses can be a lot of overhead for a small group. But it's easy to collapse later, and erring toward clarity early is usually the cheaper mistake, so I wouldn't fight it now."),
            ("What about the people side — reviews, hiring, feedback?",
             "That part's actually the strongest. Calibrating ratings before sharing them and submitting interview feedback independently both fight the same bias — anchoring on the loudest first opinion. Whoever wrote it clearly cares about fairness, not just process."),
            ("If we had to trim this down for a smaller team, what would you cut?",
             "I'd keep the shipping, incident, and security rules untouched, and lighten the planning and approval machinery. A small team can plan more loosely without much risk, but it can't afford to be sloppy about outages, data, or access."),
            ("Can you boil down what we agree on into a few lines?",
             "We agree the core pattern is sound — small, owned, written, reversible — the people practices are the highlight, and the main thing to tune for team size is the planning and approval overhead. Everything risk-related stays as is."),
            ("Nice. Want to sketch what we'd tell the rest of the team?",
             "Let's tell them: the handbook is in good shape, here's the one pattern that explains most of it, here are the parts we'd keep no matter what, and here's the overhead we can safely dial down as the team grows. Short, honest, and actionable."),
        ],
    },
]


# Fixed system guidelines shared VERBATIM by every conversation. Placed first in every system
# message so its tokens are a GLOBAL prefix-cache hit (the grading spec's shared_system_prefix).
# Padded to shared_system_prefix_tokens at build time from the note bank (see sized_text).
SHARED_SYSTEM_INTRO = (
    "You are a careful, grounded assistant embedded in an internal knowledge tool. Answer only "
    "from the reference material provided in this conversation; if the material does not cover "
    "something, say so plainly rather than guessing. Prefer short, structured answers: lead with "
    "the direct answer, then the few supporting points that matter. Keep a neutral, professional "
    "tone, define any term a new colleague might not know, and never invent policies, numbers, "
    "names, or dates that are not in the material. When a question is ambiguous, state the "
    "assumption you are making before answering, and cite the relevant part of the material in "
    "your own words rather than quoting it at length. The standing reference for every "
    "conversation in this tool follows:"
)


def encode_len(tok, text: str) -> int:
    return len(tok.encode(text).ids)


def sized_text(tok, base_text: str, note_ids: list[list[int]], target_tokens: int, seed: int) -> str:
    """Real-English text of EXACTLY ``target_tokens``: ``base_text`` then whole notes (shuffled by
    ``seed``) appended until the target is reached, then decoded back. Deterministic per seed, so
    a fixed seed yields identical text (a shared prefix) and a per-conv seed yields a distinct one.
    Whole notes keep it readable; the cycle just revisits themes, which reads as a longer digest.
    """
    ids: list[int] = list(tok.encode(base_text).ids)
    order = list(range(len(note_ids)))
    random.Random(seed).shuffle(order)
    i = 0
    while len(ids) < target_tokens:
        ids.extend(note_ids[order[i % len(order)]])
        i += 1
        if i > len(order) * 16:  # safety: never loop unbounded
            break
    return tok.decode(ids[:target_tokens])


def think_pause_ms(rng: random.Random) -> int:
    return int(min(max(rng.lognormvariate(THINK_LOG_MU, THINK_LOG_SIGMA), THINK_MIN_MS), THINK_MAX_MS))


def build_records(spec: list[dict], tok, sizes: dict) -> list[dict]:
    """Turn the per-turn grading spec into replayable bodies matching the workload structure.

    ``sizes`` carries the grading spec's token blocks: ``shared`` (global system prefix),
    ``conv`` (per-conversation prefix), ``user`` (new user tokens/turn), ``output`` (pinned
    output/turn). The system message is ``shared_prefix + conv_block`` so the first ``shared``
    tokens are identical across every conversation (a global cache hit) and the next ``conv``
    are unique per conversation; each turn adds a ``user``-sized question, and history answers
    are ``output``-sized, so the accumulating prompt grows exactly as the spec's in_tokens_est.
    """
    from collections import defaultdict

    note_ids = [tok.encode(n).ids for n in KB_NOTES] if tok is not None else None

    # Global shared prefix: fixed seed -> byte-identical for every conversation.
    shared_prefix = sized_text(tok, SHARED_SYSTEM_INTRO, note_ids, sizes["shared"], seed=0)

    by_conv: dict[int, list[dict]] = defaultdict(list)
    for row in spec:
        by_conv[row["conv_id"]].append(row)

    rng = random.Random(SEED)  # drives the think-pauses, in conv order for determinism
    staged: list[tuple[int, dict]] = []

    for conv_id in sorted(by_conv):
        turns = sorted(by_conv[conv_id], key=lambda r: r["turn_idx"])
        scenario = SCENARIOS[conv_id % len(SCENARIOS)]
        n_turns = min(len(turns), len(scenario["qa"]))

        # Per-conversation prefix: persona + a unique header + reference briefing, sized to `conv`
        # tokens and uniquely shuffled per conv so it is a distinct prefix (no cross-conv hits).
        header = (f"[Session {conv_id:04d}] Reference material for this conversation "
                  f"({scenario['label']}):\n\n")
        conv_base = f"{scenario['persona']}\n\n{header}"
        conv_block = sized_text(tok, conv_base, note_ids, sizes["conv"], seed=1000 + conv_id)
        system_content = f"{shared_prefix}\n\n{conv_block}"

        # Size each turn's question to `user` tokens (question + the excerpt the user is asking
        # about) and each historical answer to `output` tokens (a thorough grounded reply), so the
        # thread accumulates ~ (user + output) per turn.
        user_texts, asst_texts = [], []
        for i in range(n_turns):
            q, a = scenario["qa"][i]
            user_texts.append(sized_text(
                tok, f"{q}\n\nThe part I'm looking at:\n", note_ids, sizes["user"], seed=7000 + conv_id * 10 + i))
            asst_texts.append(sized_text(
                tok, f"{a} ", note_ids, sizes["output"], seed=9000 + conv_id * 10 + i))

        arrival = turns[0]["timestamp_ms"]
        for t in range(n_turns):
            # Accumulating thread: system + completed (user, assistant) pairs + the new user turn.
            messages = [{"role": "system", "content": system_content}]
            for i in range(t):
                messages.append({"role": "user", "content": user_texts[i]})
                messages.append({"role": "assistant", "content": asst_texts[i]})
            messages.append({"role": "user", "content": user_texts[t]})

            out_max = turns[t]["out_tokens_max"]
            staged.append((arrival, {
                "timestamp_ms": arrival,
                "workload_type": "conversation",
                "body": {
                    "model": MODEL_NAME,
                    "messages": messages,
                    "max_tokens": out_max,
                    "temperature": 0,
                    "seed": 42,
                },
            }))
            # Next turn arrives after the server responds + the human reads and thinks.
            arrival += out_max * RESPONSE_MS_PER_TOKEN + think_pause_ms(rng)

    staged.sort(key=lambda x: x[0])
    return [{"request_id": i, **rec} for i, (_, rec) in enumerate(staged)]


def load_sizes(path: str) -> dict:
    """Token blocks from the grading workload spec (data/input/grading-workload-spec.json)."""
    s = json.load(open(path))
    return {
        "shared": s["shared_system_prefix_tokens"],
        "conv": s["per_conversation_prefix_tokens"],
        "user": s["new_user_tokens_per_turn"],
        "output": s["output_tokens_per_turn_pinned"],
    }


def main() -> None:
    global MODEL_NAME  # --model-name rebinds it; the assertions below read the module global
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="data/input/trace_grading_public.jsonl")
    ap.add_argument("--workload", default="data/input/grading-workload-spec.json")
    ap.add_argument("--model", default="../hf-model")
    ap.add_argument("--out", default="data/input/trace-round2.jsonl")
    ap.add_argument("--model-name", default=MODEL_NAME,
                    help="value of each record's body.model; must match --served-model-name")
    args = ap.parse_args()
    MODEL_NAME = args.model_name

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    sizes = load_sizes(args.workload)
    spec = [json.loads(line) for line in open(args.spec) if line.strip()]
    records = build_records(spec, tok, sizes)

    with open(args.out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    lens = sorted(sum(encode_len(tok, m["content"]) for m in r["body"]["messages"]) for r in records)
    n_turns = [len(r["body"]["messages"]) for r in records]
    # Verify the prefix-cache structure the benchmark depends on: the shared block is byte-identical
    # across all conversations, the per-conv block differs, and prompts grow with the thread.
    systems = {r["body"]["messages"][0]["content"] for r in records}
    shared_len = encode_len(tok, sized_text(tok, SHARED_SYSTEM_INTRO,
                                            [tok.encode(n).ids for n in KB_NOTES], sizes["shared"], seed=0))
    assert all(s[:200] == next(iter(systems))[:200] for s in systems), "shared prefix must be identical"
    assert all(r["body"]["model"] == MODEL_NAME for r in records)
    assert all(r["body"]["messages"][-1]["role"] == "user" for r in records)
    assert len(records) == len(spec)
    print(
        f"wrote {len(records)} records to {args.out}\n"
        f"  shared prefix   {shared_len} tok, identical across all {len(systems)} conversations (global cache hit)\n"
        f"  prompt tokens   min={lens[0]:,}  p50={lens[len(lens) // 2]:,}  max={lens[-1]:,}  (grows per turn)\n"
        f"  messages/req    min={min(n_turns)}  max={max(n_turns)}  (system + accumulating history)\n"
        f"  out_tokens      pinned {sizes['output']} per turn"
    )


def _self_check() -> None:
    class FakeEncoding:
        def __init__(self, ids):
            self.ids = ids

    class FakeTok:
        def encode(self, text):
            return FakeEncoding([ord(c) % 256 for c in text])

        def decode(self, ids):
            return "".join(chr(65 + (i % 26)) for i in ids)

    tok = FakeTok()
    sizes = {"shared": 60, "conv": 40, "user": 20, "output": 30}
    spec = []
    for conv_id in range(3):
        for turn_idx in range(6):
            spec.append({
                "conv_id": conv_id, "turn_idx": turn_idx,
                "timestamp_ms": conv_id * 100, "think_ms": 3000,
                "in_tokens_est": 2150 + turn_idx * 450, "out_tokens_max": sizes["output"],
            })
    recs = build_records(spec, tok, sizes)

    assert len(recs) == 18, len(recs)
    assert [r["request_id"] for r in recs] == list(range(18)), "sequential ids after sort"
    assert all(r["timestamp_ms"] <= recs[i + 1]["timestamp_ms"] for i, r in enumerate(recs[:-1])), "arrivals sorted"
    # Accumulating structure: within a conversation, turn t has system + t*(user,asst) + 1 user
    # = 2t + 2 messages. Find one conversation's records by its identical system message.
    sys0 = recs[0]["body"]["messages"][0]["content"]
    conv0 = [r for r in recs if r["body"]["messages"][0]["content"] == sys0]
    counts = sorted(len(r["body"]["messages"]) for r in conv0)
    assert counts == [2, 4, 6, 8, 10, 12], counts
    # System prompt is identical across a conversation's turns (prefix reuse) ...
    assert all(r["body"]["messages"][0]["content"] == sys0 for r in conv0)
    # ... full system content is distinct across conversations (per-conv block differs) ...
    systems = {r["body"]["messages"][0]["content"] for r in recs}
    assert len(systems) == 3, len(systems)
    # ... but the SHARED prefix (first `shared` tokens) is byte-identical across every conversation
    # (the global cache hit); the per-conv block only starts after it.
    assert len({s[:sizes["shared"]] for s in systems}) == 1, "shared prefix must be identical across convs"
    # User/assistant turns are sized to the spec blocks so the thread grows ~(user+output)/turn.
    assert encode_len(tok, recs[0]["body"]["messages"][-1]["content"]) == sizes["user"], "user turn sized"
    a_msgs = [m for m in conv0[-1]["body"]["messages"] if m["role"] == "assistant"]
    assert a_msgs and all(encode_len(tok, m["content"]) == sizes["output"] for m in a_msgs), "history answers sized"
    # Last message is always a user turn; assistant turns only appear in history.
    assert all(r["body"]["messages"][-1]["role"] == "user" for r in recs)
    # Think-pauses vary (not a constant cadence).
    gaps = [recs[i + 1]["timestamp_ms"] - r["timestamp_ms"] for i, r in enumerate(recs[:-1])]
    assert len(set(gaps)) > 1, "arrivals should have varied (random) gaps"
    assert all(r["body"]["model"] == MODEL_NAME for r in recs)
    print("build_trace_round2 self-check ok")


if __name__ == "__main__":
    import sys

    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
