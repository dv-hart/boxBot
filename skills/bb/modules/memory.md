# bb.memory — durable facts

Small, durable, retrieval-ranked records ("Zara is 2", "Erik attends
preschool"). Same backend as the `search_memory` tool and
conversation-start injection — a write here shows up in a later
lookup.

For facts that should *ring a bell* later. For detail you'll *look up*
(long lists, CSVs, drafts) use `bb.workspace`.

## API

```python
import boxbot_sdk as bb

mid = bb.memory.save(                  # → new id (UUID)
    "Erik's preschool graduation: Mon Jun 8 2026, 8:45-10 AM.",
    memory_type="person",              # person | household | methodology
    person="Erik",
    people=["Erik"],                   # optional; defaults to [person]
    tags=["school"],                   # optional
)

for m in bb.memory.search("preschool graduation", people=["Erik"]):
    print(m.id, m.content)

bb.memory.invalidate("fe98abdb", reason="Jacob: it's Erik's, not Zara's")
# → {"id": "fe98abdb-…", "person": "Zara",
#    "summary": "Zara's preschool graduation: …", "status": "invalidated"}
```

`delete()` is the same operation as `invalidate()` — deletion is a soft
invalidate. Use `invalidate()` when acting on a correction.

`save` and `delete`/`invalidate` raise `bb.ActionError` on rejection. A
memory that did not persist never looks like it did.

## Ids: the 8-char prefix is a valid handle

Injected memories carry a prefix, e.g.
`#fe98abdb (person/Zara): Zara's preschool graduation…`. Pass it
straight to `invalidate()` / `delete()`.

- No match → raises `bb.ActionError("no active memory matches id '…'")`.
- Ambiguous prefix → raises, listing candidates. Pass more characters.
- Success → returns the invalidated record. **Do not** run a `search`
  to verify; the return value already told you.

An `invalidate()` that finds nothing FAILS LOUDLY. Before you tell the
user "corrected", confirm the call returned a record.

## Corrections

1. Find the record — the injected `#prefix` if it's in context, else
   `bb.memory.search(...)`.
2. `bb.memory.invalidate(<id>, reason="<who said what>")`. Check it
   returned a record.
3. `bb.memory.save(<corrected fact>)`.

If the wrong fact came from an external source you can see (a calendar
event a briefing pulled), fixing memory is not enough — the source
regenerates it. Offer to fix the source too.

## Not this

- Long or bulky content → `bb.workspace`. Records stay short.
- Transient conversation state → not a durable fact.
- Anything readable live from an integration (calendar, weather) →
  read it live. A memorized copy drifts.
