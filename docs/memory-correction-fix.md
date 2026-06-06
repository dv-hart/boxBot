# Memory-correction failure chain — diagnosis & fix

## The incident

Jacob's preschool-graduation event (Mon Jun 8) was repeatedly reported as
**Zara's** when it is **Erik's**. Jacob corrected BB twice (2026-06-04 and
2026-06-05); both corrections appeared to succeed ("Corrected." replies)
yet BB kept saying Zara.

## Root causes (three independent failures, all required a fix)

### A. In-conversation `memory.delete` / `invalidate` silently no-ops
- Memories are injected into context with an **8-char id prefix**
  (`retrieval.py` → `#{c.id[:8]}`). That prefix is the only handle BB ever
  sees.
- `MemoryStore.invalidate_memory` matched on the **full UUID**
  (`WHERE id = ?`). Passing the prefix `fe98abdb` matched **0 rows**.
- A 0-row `UPDATE` is not a SQLite error, and the `memory.delete` handler
  returned `{"status": "ok"}` **unconditionally** — never checking
  rowcount. The SDK saw success, BB told Jacob "Corrected", and the wrong
  memory stayed `active`.
- The SDK had no `invalidate()` at all (only `save`/`search`/`delete`), so
  BB's first instinct (`bb.memory.invalidate(...)`) raised `AttributeError`
  before it fell back to `delete`.

### B. The dream cycle (the intended resolution mechanism) can't resolve contradictions
- `dream_audit_only: true` — dry-run, mutates nothing.
- It is a **deduplicator**, not a contradiction-resolver. Its decision
  space is `merge_into_a / merge_into_b / distinct / unsure`. For
  "Zara's graduation" vs "Erik's graduation" the correct *dedup* answer is
  `distinct` (different person → keep both), so both survive forever.
- Candidate window blind spot: `build_candidate_set` selects
  `new_today = created since midnight UTC`, but the cycle runs at 10:00 UTC.
  Only memories created 00:00–10:00 UTC get the load-bearing Pass-2
  nearest-neighbour scan; the other 14h are never `new_today`. Every
  Erik-graduation memory was created in that blind spot → "6 candidates,
  0 pairs" every night.

### C. Trigger/proactive output is not the user's conversation
- The 7:00 AM briefing ran as a **transient `trigger:` conversation**
  (`conv_640d6db31a6e`), pulled the calendar (event `193ptomq…` titled
  "Zara Preschool Graduation"), delivered WhatsApp text to Jacob, then was
  summarised **with no extraction** and discarded.
- Jacob's 7:21 AM reply created a **fresh `whatsapp:` conversation** with no
  calendar, no injected memory, no briefing context. So BB couldn't fix the
  calendar (no handle), extraction couldn't supersede the memory (empty
  `injected_memories_block`), and BB had to guess what "that" referred to.

There is also a live **data** issue: the Google Calendar contains a
duplicate event literally titled "Zara Preschool Graduation…" alongside the
correct "Preschool Graduation…". (Owner: Jacob, handled calendar-side.)

## Fixes

### Fix A — honest, prefix-tolerant, self-reporting memory writes
- `store.resolve_memory_id(id_or_prefix)`: exact match first, else prefix
  match against **active** memories. Returns 0 / 1 / many.
- `store.invalidate_memory(...)` returns rows affected.
- `memory.delete` handler: resolve → error on no-match / ambiguous →
  invalidate → **return the invalidated record** `{id, person, summary}`.
  No more silent `ok`.
- SDK: `delete()` returns the record dict; add `invalidate(memory_id,
  reason=None)` as the explicit-correction entry point.
- Document id-prefix acceptance + "delete = soft-invalidate" in
  `skills/bb/modules/memory.md`.

### Fix B — dream cycle resolves clear contradictions
- Add a **contradiction** judgment alongside dedup: same subject/predicate,
  incompatible value → choose survivor (recency + presence of an explicit
  correction memory) and invalidate the other with `superseded_by`.
- **Authority (decided):** the cycle MAY autonomously invalidate when a
  memory is *clearly* invalidated (e.g. "it's Erik's, not Zara's"). The
  prompt gives instructions and leaves flag-vs-invalidate to the agent's
  discretion — invalidate when unambiguous, flag when uncertain.
- Fix the candidate window: select new memories via a **persisted
  watermark** (created since last dream run) so every new memory gets the
  Pass-2 scan exactly once, regardless of creation time.
- Validate in audit mode, then set `dream_audit_only: false`.

### Fix C — proactive/trigger output threads into the user's conversation
- When a trigger's output is user-facing text on a threaded channel
  (WhatsApp), deliver through and record on that user's persistent
  `whatsapp:{phone}` conversation instead of the throwaway `trigger:` one.
- **Content (decided):** thread the **full trigger reasoning** into the
  persistent conversation (KISS — no separate "context note" path).
- Effect: the reply rehydrates with the briefing + calendar pull + injected
  memory, so Fix A and the extraction safety net both have their handles.

## Sequencing
1. Fix A — smallest; stops corrections silently failing.
2. Fix C — restores context so A and extraction work as designed.
3. Fix B — autonomous backstop; validate in audit mode, then enable.
