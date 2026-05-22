# What-If Reflection Cycle — Implementation Plan

**Status:** Draft (2026-05-20). Net-new feature. Roadmap §4
("Self-improvement reflection") was deferred from Phase B PR1 and never
built; this plan supersedes its sketch with a concrete design grounded
in how conversations are actually persisted today.

## 1. Goal

Once a day, look back at the *complex* things boxBot did and ask: **how
did it do, and could it have done better?** Output is a plain-text
"what-ifs" log a human reviews — never memories, never code changes.

This is the dream phase's highest-variance, most failure-prone idea (an
LLM grading its own work), so it ships **off by default**, hard-bounded,
and observable — same posture we just used for dedup `audit_only`.

### Divergence from roadmap §4 (intentional)

The roadmap said "one reflection **per `execute_script` call**." We
reject that granularity:

- It misses the *arc*. The interesting failures are cross-turn (retry
  loops, wrong approach pursued for 20 turns, server-vs-client confusion)
  — invisible when you grade calls in isolation.
- It's noisy and expensive (the May-15 marathon had 29 calls).

Instead: **one reflection job per qualifying conversation**, given the
whole task arc, free to emit multiple what-if entries. This matches the
user's framing ("when it builds a new display, how did it do?").

## 2. Feasibility — data availability (RESOLVED)

The blocking question was: are the day's `execute_script` calls + results
recoverable at 3 AM? **Yes, for WhatsApp.**

- `data/conversations/conversations.db` → `conversation_turns.content_json`
  stores **the raw Anthropic message dict** per turn, including assistant
  `tool_use` blocks (the `execute_script` `script` body) and the
  `tool_result` blocks (stdout / `status:error` / stderr). Verified live:
  38 conversations, 890 turns.
- `ConversationStore.mark_extracted()` sets state but **does not delete
  turns** (`delete()` is a separate, unused-by-lifecycle call), so threads
  survive past memory extraction and are readable nightly.

**Out of scope v1: voice and trigger conversations.** They are transient
by design (`CLAUDE.md` §10) — in-memory thread, extraction fires
synchronously on `ConversationEnded`, nothing persisted. Reflecting on
them would require persisting their threads first. In practice the
complex authoring/debugging tasks happen over WhatsApp anyway, so this is
an acceptable v1 boundary. Note it explicitly in the log header.

## 3. Trigger — what counts as "complex"

Per-conversation complexity is computed straight from the persisted
thread. Signals (all derivable from `conversation_turns`):

| Signal | Source |
|--------|--------|
| `turn_count` | `COUNT(*)` of turns |
| `script_count` | `tool_use` blocks with `name="execute_script"` |
| `error_count` | `tool_result` blocks whose content contains an error/`status:error` |

**Gate (config-driven, conservative defaults):**

```
qualifies = script_count >= reflection_min_scripts (8)
            AND turn_count >= reflection_min_turns (25)
```

Empirically over the live corpus (38 conversations, ~3 weeks) this
selects ~6–8 conversations — the genuine marathons (136/82/80/63/59/56…
turn threads). The long tail of short Q&A threads is excluded. **Most
days select zero**, which is the intended behaviour.

`error_count` is logged and available for a future weighted score, but
the v1 gate keeps it simple (count thresholds, AND logic — mirrors the
trigger style in `boxbot.core.scheduler`).

**Candidate window + idempotency.** Select conversations where
`last_activity_at_iso` falls in the last ~26h AND `state` is terminal
(`extracted`/`expired`), excluding any already reflected. Track with a new
nullable `reflected_at_iso` column on `conversations` (cheap migration,
same shape as `extracted_at_iso`).

## 4. Architecture — mirror the extraction pipeline

New module `src/boxbot/memory/reflection.py`, structurally a sibling of
`src/boxbot/memory/extraction.py` (which is the proven template for
"post-conversation, batch, structured-output, record cost"):

```
select_reflection_candidates(conv_store, *, now, cfg) -> list[ConvRecord]
distill_thread(turns) -> str               # token-bounded transcript
build_reflection_request(conv, transcript, *, model) -> dict
submit_reflection_batch(client, conv_store, candidates, *, model) -> batch_id
process_reflection_result(batch, conv_store) -> ReflectionResult
write_reflection_log(entries, *, now)       # workspace markdown
```

Reuse verbatim from extraction/dedup:
- Cached system prompt block (`cache_control: ephemeral, 1h`) so repeated
  nightly runs share the prefix.
- `tool_choice: {type:tool, name:"emit_reflection"}` for structured output.
- `client.messages.batches.create(...)` — batch API, **−50% cost**, no
  latency concern at 3 AM.
- `boxbot.cost.record` with `purpose="reflection"` (the tag the cost
  schema already reserves; `cost_log.purpose` is free-text).

### 4a. Thread distillation (the cost lever)

A 136-turn thread is too big to send raw. `distill_thread` keeps what
matters for judging quality and drops noise:

- **Keep:** every user turn (text); each assistant `execute_script` —
  the first ~15 lines of the `script` + which `bb` modules it touched;
  each `tool_result`'s `status` and, if error, the error message
  (truncated). Assistant natural-language turns kept (short).
- **Drop / truncate:** long *successful* stdout dumps, base64 image
  blocks, repeated doc-fetch payloads.

Target ≤ ~12K tokens per conversation. Emit the realized token estimate
into the log so we can tune.

### 4b. Output schema (`emit_reflection` tool)

```jsonc
{
  "task_summary": "string — what the human asked BB to accomplish",
  "outcome": "succeeded | partial | failed",
  "entries": [{
     "observation": "what BB did, concretely (cite turn range)",
     "what_if":     "the better/faster/cleaner path",
     "category":    "retry_loop | wrong_approach | missing_lookup |
                     hallucination | inefficiency | good_practice",
     "severity":    "low | med | high",
     "turns_wasted_estimate": 0
  }],
  "confidence": 0.0
}
```

`category="good_practice"` is allowed on purpose — the log should also
capture what went *right*, so promotion to `methodology` memory has
positive exemplars, not just scolding.

## 5. Integration into the dream cycle

`run_dream_cycle` gains a `conversation_store` arg and reflection config,
and a **Phase 4** after maintenance:

```
Phase 1  gather / cluster / find_near_duplicates      (existing)
Phase 2  submit dedup batch                           (existing)
Phase 3  run_maintenance                              (existing)
Phase 4  reflection: select → distill → submit batch  (NEW)
```

`agent._run_dream_cycle_for_trigger` already holds `self._conversation_store`
and `self._client`; thread both through plus the new config fields.

**Batch + poller.** Reflection submits its **own** batch (different system
prompt + tool) and records a `pending_dreams` row with
`request_types={"reflection": N}`. Extend `DreamPoller._apply_completed`
to dispatch by `request_types`: dedup pairs → `apply_dream_result`
(mutates memory); reflection → `process_reflection_result` (writes log
only, **no mutation**). This reuses the existing crash-resume + cost
recording path with one branch, rather than a parallel poller.

## 6. Output location & format

`data/workspace/notes/system/reflection/<YYYY-MM-DD>.md` (the path the
roadmap reserved; sibling of the existing `dream-log/`). One section per
conversation:

```markdown
## conv_b7f530eac013 — weekly_glance display + calendar binding
- channel: whatsapp · turns: 136 · scripts: 29 · errors: 18
- outcome: succeeded (over 136 turns; ~30–40% estimated reducible)

### What-ifs
- [high · retry_loop] Turns 15–31: hammered the same `import boxbot_sdk`
  script six times against an identical `[sandbox-bootstrap] seccomp`
  failure before one succeeded. → After two identical bootstrap errors,
  stop and read the error (or `load_skill bb`) instead of retrying blind.
  (~6 turns wasted)
- [high · hallucination] Turn 93: addressed a message to "Jarvis" —
  rejected ("valid recipients: Jacob, Carina"). → Validate recipient
  against the known household before sending; never invent a name.
- [med · missing_lookup] Turns 57–113: bound the calendar integration to
  the display by trial-and-error, hitting repeated `status:error`, before
  consulting `bb.integrations` docs. → Read the integration module doc
  first; distinguish server-side failures (Jacob fixed the `when` field /
  TZ / staging gap) from client-side so retries aren't redundant.
- [low · good_practice] Turns 124–131: cleanly de-duplicated calendar
  entries on request without over-deleting.
```

## 7. Config (`MemoryConfig`, alongside the `dream_*` fields)

```python
reflection_enabled: bool = False         # observable rollout, like audit_only
reflection_min_turns: int = 25
reflection_min_scripts: int = 8
reflection_max_per_night: int = 3        # hard cap on batch size
reflection_model: str | None = None      # None → DEFAULT_EXTRACTION_MODEL (Sonnet, batch)
```

## 8. Cost

Per reflection: distilled thread (~5–12K in) + ~1–2K out. Sonnet 4.6
**batch** (−50%): **~$0.02–0.08 each**. Capped at 3/night → **< $0.25 on
a busy day, ~$0 on most days**. Negligible against the $166 conversation
spend. Recorded under `purpose="reflection"` for attribution.

## 9. Safety / bounds (same contract as roadmap §4)

- Creates **no** memories. Modifies **no** code. Output is plain text in
  the workspace.
- Off by default; capped per night; fully turn-off-able by config.
- Human reviews the log; promotion of any entry to a `methodology` memory
  is **manual**, never automatic.

## 10. Testing

- `distill_thread`: keeps tool_use/errors, drops big stdout, respects
  token budget (fixture: the 136-turn thread shape).
- `select_reflection_candidates`: threshold + window + idempotency
  (`reflected_at_iso`) logic.
- Schema parse + `write_reflection_log` formatting.
- Poller dispatch: a `request_types={"reflection":N}` batch routes to
  `process_reflection_result`, never to memory mutation.
- All hardware/Anthropic mocked, per repo testing norms.

## 11. Phasing

- **PR1:** module + candidate selection + distillation + batch submit +
  poller dispatch + log writer + config + migration + tests. Ship with
  `reflection_enabled=False`.
- **Soft launch:** enable on the Pi for 1–2 weeks. It writes logs, costs
  pennies, mutates nothing.
- **Review gate:** Jacob reads the logs. Valuable → keep, hand-promote
  best entries to methodology memories, maybe surface in morning brief.
  Not valuable → flip `reflection_enabled=False` and the feature is inert.

## 12. Open questions

1. **Granularity within a conversation.** A 136-turn thread can span
   several tasks. v1 lets the model emit multiple `entries`; if that's
   too coarse, a later version could segment by user-turn boundaries.
2. **Voice/trigger reflection.** Deferred until/unless transient threads
   are persisted. Worth it only if voice produces complex tool-use arcs.
3. **Combined vs separate batch.** Plan keeps reflection in its own batch
   (cleaner prompts, independent failure). Could co-submit with dedup to
   save one API round-trip — not worth the coupling in v1.
