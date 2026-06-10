# bb.memory — save, search, invalidate durable facts

`bb.memory` is the fact store: small, durable, retrieval-ranked records
("Zara is 2", "Erik attends preschool"). It shares one backend with the
`search_memory` tool and with conversation-start injection, so a write
here is visible to a later lookup and vice-versa.

Use it for facts that should *"ring a bell"* later. For detail you'll
"look up" (long lists, CSVs, drafts), use `bb.workspace` instead.

## API

```python
import boxbot_sdk as bb

# Save — returns the new id (a UUID).
mid = bb.memory.save(
    "Erik's preschool graduation: Mon Jun 8 2026, 8:45-10 AM.",
    memory_type="person",          # person | household | methodology
    person="Erik",
    people=["Erik"],               # optional; defaults to [person]
    tags=["school"],               # optional
)

# Search — ranked records.
for m in bb.memory.search("preschool graduation", people=["Erik"]):
    print(m.id, m.content)

# Invalidate (soft delete) — the explicit move when a fact is wrong.
bb.memory.invalidate("fe98abdb", reason="Jacob: it's Erik's, not Zara's")
# → {"id": "fe98abdb-…", "person": "Zara",
#    "summary": "Zara's preschool graduation: …", "status": "invalidated"}
```

`delete()` is the same operation as `invalidate()` (deletion is a soft
invalidate); use `invalidate()` when you're acting on a correction.

All writes (`save`, `delete`/`invalidate`) raise `bb.ActionError` when
the main process rejects them — a memory that didn't persist never
looks like it did.

## Ids: the prefix you see IS a valid handle

Injected memories appear with an **8-char id prefix**, e.g.
`#fe98abdb (person/Zara): Zara's preschool graduation…`. You can pass that
prefix straight to `invalidate()` / `delete()` — the main process resolves
it to the full id.

- **No match** → raises `bb.ActionError("no active memory matches id '…'")`.
- **Ambiguous prefix** (matches >1 active memory) → raises with the list of
  candidates; pass a longer prefix or the full id.
- **Success** → returns the invalidated record, so you can confirm *what*
  you removed. **Do not** run a separate `search` to verify — the return
  value already tells you.

This means an `invalidate()` that finds nothing now FAILS LOUDLY instead of
looking like it worked. If you tell the user "corrected", make sure the
call returned a record.

## Handling a correction (the right pattern)

When a user corrects a stored fact:

1. Find the offending record — use the injected `#prefix` if it's in
   context, otherwise `bb.memory.search(...)` to get its id.
2. `bb.memory.invalidate(<id>, reason="<who said what>")` — and check it
   returned a record.
3. `bb.memory.save(<corrected fact>)`.

If the wrong fact came from an external source you can see in context
(e.g. a calendar event the briefing pulled), fixing the memory is not
enough — that source will regenerate the fact. Offer to fix the source too.

## When NOT to use it

- Long or bulky content → `bb.workspace`. Memory records stay short.
- Transient conversation state → it isn't a durable fact.
- Anything you can read live from an integration (calendar, weather) → read
  it live; don't memorialize a copy that can drift from the source.
