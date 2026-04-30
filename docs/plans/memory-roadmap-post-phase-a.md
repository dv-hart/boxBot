# Memory roadmap — work after Phase A

Phase A (commit `5de8c6f`, 2026-04-30) shipped the real batch-driven
extraction pipeline + transcript recovery. This doc captures every
follow-up agreed during the design discussion so we can pick any item
up later without re-litigating the design.

The order here roughly follows expected delivery sequence, but the
items are independent enough to be reordered or split into separate
PRs.

## Glossary

- **Phase A** (done): durable extraction pipeline. `pending_extractions`
  + `cost_log` tables, `BatchPoller`, real prompt + structured output,
  transcript recovery via `search_memory(mode="transcript")`,
  `accessed_memory_ids` plumbed through injection.
- **Day pipeline**: per-conversation extraction. Bias toward
  encoding (just create memories), no apply-time dedup. Phase A.
- **Dream phase**: nightly consolidation. Where the messy work lives —
  dedup, schema formation, staleness, system memory promotion,
  optional self-improvement reflection. Phase B.
- **Person record**: today, the person's *name string* is the soft join
  key across `perception.persons`, `auth.users`, and
  `memory.memories.person`. No central registry above these. A future
  refactor unifies them.

---

## 1. Real Haiku reranking for `search_memory`

**Status:** `rerank_stub` in `src/boxbot/memory/search.py:439` returns
candidates as-is with `f"Matched query (score: {x:.2f})"` as the
relevance string. Top-K hybrid score wins, no model judgment.

**Goal:** small-model reranking step that scores relevance, generates a
contextual title, and writes a relevance reason — exactly as described
in `docs/memory.md` "Lookup Mode" pipeline (step 2).

### Design

Replace `rerank_stub` with `rerank_with_haiku`. Pipeline:

1. After `hybrid_search` returns ~30 fact candidates + ~10 conversation
   candidates, batch into groups of ~6.
2. Fire 5–6 parallel Haiku calls (one per group) using
   `client.messages.create` with structured output (tool use).
3. Each call returns, per candidate: `relevant: bool`, `title: str`
   (≤80 chars, contextual to the query), `summary: str` (one
   sentence), `relevance_reason: str` (one sentence).
4. Drop candidates where `relevant=False`.
5. Re-sort by hybrid score; return top 10 facts + top 3 conversations.

### Tool schema

```python
RERANK_TOOL = {
    "name": "rerank_candidates",
    "input_schema": {
        "type": "object",
        "required": ["judgments"],
        "properties": {
            "judgments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["candidate_id", "relevant", "title",
                                 "summary", "relevance_reason"],
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "relevant": {"type": "boolean"},
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "relevance_reason": {"type": "string"},
                    },
                },
            },
        },
    },
}
```

### Performance + cost

- Synchronous (must return in seconds; this is the tool/injection path,
  not batch).
- Haiku 4.5 standard pricing: $1/$5 per MTok. With ~30 candidates of
  ~50 tokens each + system prompt ~500 tokens, per-call cost is
  trivial (~$0.0005).
- Use 5-min cache TTL on the rerank system prompt (it's stable across
  calls within a session).
- Target end-to-end rerank latency: <2 seconds for the parallel batch.

### Interaction with dream phase

Reranking is hot-path search. Dream phase doesn't replace it.

### Tests

- Unit: mock parallel Haiku calls, verify ordering + filtering.
- Integration: gated on `BOXBOT_TEST_LIVE`, single search against a
  seeded memory set, verify the top result actually answers the query.

### Cost log purpose: `"rerank"`

---

## 2. Summary mode synthesis

**Status:** `_search_summary` in `search.py` likely returns candidates
without synthesizing an answer (the docstring promises `{answer,
sources}` but the implementation doesn't appear to call a model).
Audit and complete.

**Goal:** match `docs/memory.md` "Summary Mode" — return a natural-
language answer with source citations.

### Design

After hybrid search + parallel filtering (which is the same Haiku step
as reranking, with a different prompt — "find snippets relevant to
this question"), one final Haiku synthesis call:

- Input: question + concatenated relevant snippets + their IDs.
- Output (structured): `{answer: str, source_ids: list[str]}`.
- Cite IDs that materially contributed to the answer.

### Implementation note

The filter step shares 90% of the prompt with reranking. Build a
shared `filter_candidates(query, intent="rank|summarize")` helper that
both modes use, then a separate `synthesize_answer` call only for
summary mode.

### Cost log purpose: `"summary"`

### Tests

- Unit: mock filter + synthesize calls, verify answer + sources shape.
- Integration: gated on `BOXBOT_TEST_LIVE`, "what does Erik like to
  eat?" against a seeded set should return a coherent answer citing
  the right memories.

---

## 3. Dream phase (Phase B) — nightly consolidation

This is the big one. Read this section carefully — every design
decision was deliberated and locked.

### Trigger

Recurring scheduler trigger fires at **3 AM local** every night. Seed
the trigger from config on first boot; runtime DB authoritative
afterwards. Existing `boxbot.core.scheduler` cron triggers handle this
shape.

### Cognitive model (the "why")

Day extraction = encoding. Just capture what happened, one
conversation at a time, all `action=create`. Don't try to dedupe.

Night extraction = consolidation. Cross-conversation cleanup, schema
formation, staleness handling, system memory promotion. Maps to
human NREM (deterministic, mechanical replay/integration) + REM
(associative, abstract, sometimes generative).

This split simplifies the day path (no apply-time Haiku check, no
dedup logic at all) at the cost of slightly noisier memory state
during the day. For a single household that's a fine trade.

### Cadence + scope per night

- **3 AM nightly** (skip if there's nothing to consolidate).
- **All new memories from today** (created or modified in the last 24h).
- **Plus 6 revisit memories** drawn from three pools:
  - **3 from "used today"**: any memory injected into a conversation
    today. Heavy exploit signal — BB itself surfaced these recently.
  - **2 from age-decayed random**: weight ∝ `exp(-age_days / 30)`
    (~21-day half-life). Recent more likely than old, but old not
    excluded.
  - **1 from pure uniform random**: explore. Catches the truly
    forgotten.

Why 6: small enough to be cheap, big enough to surface patterns. Track
performance for a few weeks before tuning.

**Explicitly rejected:** temporal-anchor flagging (parsing memory
content for "next Saturday" etc to detect staleness). Too brittle.
Decay handles it implicitly — stale memories stop being injected and
stop being randomly sampled, and the next time they ARE sampled the
dream model can question their relevance.

### Phase 1 — deterministic (no model)

Before any LLM call, do the cheap mechanical work:

1. **Cluster by embedding** (cosine ≥ 0.7 threshold) across the
   candidate set. Clusters of size ≥3 flagged for schema formation.
2. **Pair near-duplicates** (cosine ≥ 0.85) where one was not in the
   other's accessed_memory_ids at creation time → flagged for dedup.
3. **Run existing maintenance**: `boxbot.memory.maintenance.run_maintenance`
   does archival + storage cap + FTS rebuild. Fold this into the
   nightly job (currently it's not scheduled anywhere I can see —
   verify and wire if needed).

### Phase 2 — model-driven (one Anthropic batch, many requests)

Build a multi-request batch (one batch, several `requests` entries with
distinct `custom_id`s):

- One **dedup request** per near-duplicate pair, including up to ~3
  conversation summaries that cite either memory as evidence.
- One **schema request** per cluster of ≥3 related memories.
- Optional: one **revisit request** per revisit-sampled memory the
  dream phase wants to interrogate ("is this still relevant? worth
  archiving?").
- Optional: one **self-improvement request** per tool-using turn from
  the day's conversations (see §4).

Model: **Sonnet 4.6 batch** throughout. Cost log purpose: `"dream"`.

### Output schema (per request)

Different request types have different tool schemas. Common fields
across all:

- `evidence: list[str]` — memory IDs / conversation IDs the model is
  citing. Required for any change. Apply step rejects writes whose
  evidence list is empty.
- `confidence: float ∈ [0, 1]` — model's stated confidence.
- `notes: str` — optional rationale, logged for audit.

#### Dedup request output

```json
{
  "decision": "merge_into_a | merge_into_b | distinct | unsure",
  "merged_content": "...",   // required if merging
  "merged_summary": "...",
  "evidence": ["mem-aaa", "mem-bbb"],
  "confidence": 0.0-1.0,
  "notes": "..."
}
```

#### Schema request output

```json
{
  "form_schema": true|false,
  "schema": {                  // present if form_schema=true
    "type": "person | household | methodology",
    "person": "...",
    "content": "...",          // the abstracted statement
    "summary": "...",
    "tags": [...]
  },
  "supporting_ids": ["mem-aaa", "mem-bbb", "mem-ccc"],
  "confidence": 0.0-1.0,
  "notes": "..."
}
```

#### Revisit request output

```json
{
  "still_relevant": true|false,
  "should_invalidate": true|false,
  "reason": "...",
  "evidence": [...],
  "confidence": 0.0-1.0
}
```

### Apply step (next morning, after batch ends)

- **Dedup**: if confidence ≥ 0.8, apply merge (update one, archive
  the other with `superseded_by`). Below: log only, do nothing.
- **Schema**: if confidence ≥ 0.8 and ≥3 supporting memories: create
  the schema memory, mark the supporting memories with new
  `supporting_for=<schema_id>` field (status stays `active`, but
  injection prefers the schema). Below: log only.
- **Revisit**: if `should_invalidate=true` and confidence ≥ 0.9:
  invalidate. Else if `still_relevant=false`: archive. Below: no-op.

Confidence thresholds are **higher than day extraction** because the
dream phase is operating on inference, not direct user statements.

### System memory proposals

The dream phase emits proposed system-memory updates as **to-dos for
the admin** (not auto-applied):

- Admin sees a to-do at next conversation: "Want me to add 'X' to
  standing instructions? I noticed it three nights running."
- Yes → applies via existing `update_system_memory` path with
  `updated_by="dream:<batch_id>"`.
- No → discarded.

Admin confirmation is the default for the first month after Phase B
ships. After that we can consider auto-apply on triple-recurrence.

**In-conversation immediate add still works as before**: if Jacob says
"always check the calendar before suggesting dinner times" the agent
calls `update_system_memory` directly in that turn. Dream phase only
catches what was MISSED.

### What qualifies for system memory promotion

The dream model is allowed to propose system memory updates ONLY in
these categories:

1. **Standing instructions** ("always do X" / "never do Y")
2. **Standard operating procedures** (recurring how-tos)
3. **Safety / medical** (allergies, medication, accessibility)
4. **Hard privacy / sharing boundaries**
5. **Identity / role / permissions** (roster facts, admin access)
6. **Stable household terminology** ("'the office' = Pearl District")
7. **Stable, impactful interaction-style preferences** ("Carina prefers
   shorter responses")

Common test: would the assistant fail if it didn't know this every
single time? If no → person memory, not system memory.

Erik's school name and grade go in **person memories**, not system
memory. System memory shrinks to: roster + standing instructions +
operational notes. That's it.

### Dream-generated to-dos

Tightly scoped to **dropped continuity** — never new initiative. The
allowed situations:

1. **Missed promise from BB** — BB said "I'll let you know when X" or
   "I'll get back to you on Y" and didn't.
2. **Stalled to-do** — sat untouched ≥7 days; gentle nudge.
3. **Unresolved decision thread** — conversation ended with explicit
   ambiguity ("we'll figure that out later") and never resumed.
4. **User-flagged "I'll think about it"** — same as above but
   user-initiated.
5. **Calendar-anchored dropped intent** — "I'll bring it up at dinner
   with Carina"; dinner happened (calendar event passed); did the
   topic come up?

Bar: did the user (or BB) leave something open? If yes → can fire.
Anything else (pattern observations, "wouldn't it be cool if...") →
NOT a to-do.

To-dos created by dream phase carry `source="dream"` and prompt
guidance instructs the agent: surface only when the user opens the
relevant topic, framed as a question, never proactively. So if Jacob
mentions Carina, BB might ask "did you end up talking to her about
the dining number?" — but won't volunteer it on a Tuesday morning.

### Operational rollup

Weekly. **Sunday night's dream phase rolls up the prior week's
operational entries.** Cluster `type=operational` memories created
in the last 7 days by tag/topic, write a single rollup memory ("Updated
slideshow 12 times this week, mostly Carina vacation photos"), archive
the originals. Keeps activity-log searches sharp.

Operational memories never get deduped at the day level (append-only
log). Only the weekly rollup compresses them.

### Self-improvement reflection (optional, observable)

**Scope:** anchored tightly to **tool-using turns only**. After each
day, for each `execute_script` call:

- What was the task?
- What did BB do?
- Did it succeed?
- Could it have been done better, faster, more cleanly?

Output: a separate "what-ifs" log. **Not memories.** Lives in
`workspace/notes/system/reflection/<YYYY-MM-DD>.md`.

After a week or two of running, **the human reviews the log** and
decides whether the reflections are valuable. If yes → consider
promoting some entries to `methodology` memories (manual, not
automatic). If no → disable the reflection step entirely. The whole
feature is built to be turn-off-able.

This is the most failure-prone part of the dream phase. Bound it
hard:

- Does NOT create memories directly.
- Does NOT modify code.
- Output is plain text in workspace.
- Capped at N reflections per night (start: 5).
- Cost log purpose: `"reflection"`.

### Storage additions

```sql
CREATE TABLE pending_dreams (
    batch_id        TEXT PRIMARY KEY,
    submitted_at    TEXT NOT NULL,
    candidate_ids   TEXT NOT NULL,    -- JSON: which memories were in scope
    request_types   TEXT NOT NULL,    -- JSON: counts per type {dedup: 4, schema: 2, ...}
    status          TEXT NOT NULL,    -- submitted | applied | failed
    completed_at    TEXT,
    summary         TEXT              -- what the dream did (audit trail)
);

ALTER TABLE memories ADD COLUMN consolidated_by TEXT;
-- batch_id of the most recent dream that touched this memory; null if untouched.

ALTER TABLE memories ADD COLUMN supporting_for TEXT;
-- if this memory was demoted to "supporting evidence" for a schema,
-- points to the schema memory id. injection prefers the schema.
```

### Risks + mitigations (do not skip)

1. **Confabulation** — model invents connections not in evidence.
   - Every dream-created memory must cite ≥1 source memory or
     conversation in `evidence`. Apply step rejects creates without it.
   - Schema memories list constituent memory IDs; auditable.
   - Every dream change has a `dreamed_by=<batch_id>` audit field.
2. **Aggressive invalidation** — confidence threshold ≥0.9 for
   invalidations (vs day extraction's direct contradiction rule).
3. **Soft-delete only.** Dream phase NEVER `DELETE`s. Only `archive`,
   `invalidate`, or `supporting_for`. Recoverable.
4. **Daily change budget**. Cap dream-phase modifications at **30
   memories per night**. Above the cap, halt and apply only the
   highest-confidence subset; defer the rest.
5. **System memory creep** — dream proposes, admin confirms via to-do.
   No auto-apply for the first month minimum.
6. **Compute creep** — bound batch size. Per-night soft cap: $1 in
   cost (logged via `cost_log`). Hard cap: skip lowest-confidence
   requests if a single batch's submission would exceed it.

### Audit + observability

- **Workspace log per night**: `workspace/notes/system/dream-log/<YYYY-MM-DD>.md`
  with: candidates considered, decisions made, evidence cited, total
  cost, status of each request type.
- **Morning brief** (already discussed, agreed): one-paragraph summary
  via WhatsApp at ~7am: "Last night I consolidated 3 memories,
  formed 1 schema (Jacob's expense focus), flagged 2 stalled to-dos."
  Cheap to add (existing morning wake-up trigger; one extra structured
  output field).
- **Undo script**: `scripts/undo_last_dream.py` walks `pending_dreams`
  for the most recent batch, reverts every change keyed to that
  `batch_id` (un-archive, un-invalidate, drop schemas, restore
  supporting_for=NULL).

### Cost expectation

Sonnet 4.6 batch. Per night:
- Dedup: 0–10 requests × ~2K tokens each
- Schema: 0–5 requests × ~5K tokens each
- Revisit: 6 requests × ~3K tokens each
- Reflection: 0–5 requests × ~10K tokens each (longer context)
- System prompt cached @ 1h TTL, mostly hits

Order-of-magnitude budget: $0.30/night = $9/month. Target soft cap of
$1/night gives plenty of headroom for outliers.

### Tests

- Unit: each dream prompt + result-parsing path with mocked client.
- Unit: confidence-threshold gate (low-confidence proposals not
  applied).
- Unit: daily-change-budget enforcement.
- Unit: undo script idempotency.
- Integration (gated): single dream cycle on a seeded memory set;
  verify expected schemas + dedups land.

### Phasing within Phase B

Build in this order to minimize blast radius:

1. **Dedup** only (lowest risk, highest leverage). Soft-launch for a
   week. Audit log only — no auto-apply.
2. **Schema formation**. Same audit-only soft-launch.
3. **Maintenance integration** (archival + rollup). Mostly
   deterministic, low risk.
4. **Revisit sampling + invalidation**. Higher risk; gate on threshold.
5. **System memory proposals → admin to-dos**. Test the to-do prompt
   carefully.
6. **Dream-generated to-dos for missed continuity**. Test prompts for
   each of the 5 situations.
7. **Self-improvement reflection** (optional, observable). Add last;
   easy to disable.
8. **Morning brief**. Cosmetic; add when phase B is solid.

---

## 4. Person-record unification (separate refactor)

**Not part of Phase B.** Standalone refactor whenever it bubbles up.

### Today's state

Three identity systems, name-string as the soft join:

| Layer | DB | Identity key | Notes |
|---|---|---|---|
| Perception | `data/perception/perception.db` `persons` | uuid + name | linked to voice_embeddings, visual_embeddings, centroids |
| Auth | `data/auth/auth.db` `users` | phone (PK) + name | WhatsApp authorization only |
| Memory | `data/memory/memory.db` `memories` | `person` text field | no FK |

This works because names are usually consistent across systems. It
breaks down when:

- Same physical person shows up under different name spellings
  ("Jake" vs "Jacob") in different layers.
- A person's name changes (rare, but).
- Two people share a name.
- We want to display "this is Jacob from his WhatsApp identity" with
  full identity provenance.

### Proposal

Add a canonical `Person` record at the application layer:

```sql
CREATE TABLE persons (
    id              TEXT PRIMARY KEY,        -- canonical uuid
    primary_name    TEXT NOT NULL,           -- display name
    aliases         TEXT NOT NULL,           -- JSON: alternate names
    role            TEXT,                    -- admin | member | guest
    relationship    TEXT,                    -- 'son of Jacob', 'partner of Jacob' (free text)
    perception_id   TEXT,                    -- FK → perception.persons.id (UUID; cross-DB soft FK)
    auth_phone      TEXT,                    -- FK → auth.users.phone
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
```

Lives in `data/identity/persons.db` (new), or fold into `memory.db`.
The latter is simpler — memory.db is already the application-layer
crossroads.

### Migration

- One-time script reads all three existing tables, picks the union of
  unique names, creates Person records, links via
  `perception_id`/`auth_phone`. Manual disambiguation for collisions.
- `memories.person` (text) stays as-is for backwards compat. Add an
  optional `memories.person_id` FK that's populated for new writes
  AND backfilled for old rows that have an unambiguous name match.
  Searches use `person_id` when available, fall back to text otherwise.

### Why not now

- Memory is functional with name-as-join. The pain is theoretical
  until we hit a name collision or a rename.
- Touching three DBs is a multi-PR refactor with real migration risk
  on a live system.
- Phase B (dream phase) doesn't depend on it. Schema formation can
  cluster by name string.

Revisit when:
- A second household tries the system (likely name collisions).
- We want to display rich identity provenance in a UI.
- The dream phase wants to merge "Jake" and "Jacob" memories — at
  that point, having stable IDs makes the merge unambiguous.

---

## 5. Open considerations (not phases, but worth tracking)

### Redaction of secrets in transcripts

Today: transcripts go to Anthropic verbatim. If a user speaks an SSN,
credit card, or API key aloud, it'll appear in the transcript. Not a
huge risk on a household device but worth a regex strip before batch
submission. Use the same `SECRET_PATTERNS` from `store.py`. **Low
priority; user's call to ship before or after dream phase.**

### Cost monitoring view

`cost_log` is being populated by Phase A. A simple display
(`workspace/notes/system/cost-summary.md` regenerated nightly) showing
last-7-days and last-30-days totals by purpose would be useful.
Trivial — 20 lines of code.

### Workspace search wiring

Memory operational entries reference workspace files
("Saved report to workspace/notes/jacob/q1_summary.md"). Confirm the
agent can grep workspace via the existing `bb.workspace` SDK. If not,
add `bb.workspace.search(query)` returning matching paths + snippets.
Required for the memory→workspace handoff to actually work.

### Backfill of the 173 stub conversations

Stubs from before Phase A have no transcripts. **Cannot be
re-extracted.** Leave as-is. Phase B's dream phase should skip
conversations whose summary starts with `[Extraction stub]` — they're
useless for clustering and would just add noise.

### `extraction_version` column

If we ever do re-extraction (we decided NOT to in Phase A — agent uses
`mode=transcript` instead), we'd need this. Skip until needed.

---

## How to pick this up later

A future session can implement any of these in isolation:

- **§1 reranking**: ~1 day. Self-contained in `search.py`.
- **§2 summary synthesis**: ~0.5 day. Same file, shared with reranking.
- **§3 dream phase**: phased over multiple PRs (1–4 above), each ~1
  day. Implement deterministic + dedup first (1 PR), validate for a
  week, then continue.
- **§4 person unification**: ~2 days including migration, but only
  pick up when motivated by an actual user-visible problem.

All of §1, §2, §3 use the same `cost_log` infrastructure already
built. All of §3 reuses `BatchPoller`'s polling patterns (a separate
poller for dream batches, but the shape is identical — could even
consolidate to one poller that handles multiple purposes).

The Phase A foundation is deliberately broad enough to support the
rest without further plumbing changes.
