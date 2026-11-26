# bb.photos — the photo library

Durable images worth remembering: WhatsApp sends, saved camera
captures, uploads. Not: scratch/debug images → `bb.workspace`;
perception crops (ephemeral, never surface here).

"What's in that photo?" → `view()` first — speak from pixels, not
metadata.

## Find — search(query, tags, people, date_from, date_to, source, limit, show) / get(photo_id)

```python
bb.photos.search(query="Zara", show=True)        # "show us photos of Zara" — one call
photos = bb.photos.search(query="Zara at the beach")
photos = bb.photos.search(limit=5)               # no query = newest first
p = bb.photos.get("photo_abc123def456")          # full record; PhotosError if missing
```

Person's name goes in `query`. A known name (roster or existing label)
in `query` or `people=[...]` = hard filter: only that person's photos
(label, or whole name in tags/description); other words rank within
them. Whole names only: "Ann" ≠ "Ann-Marie".
Several names = photos with ALL of them; none → `[]` + hint with per-name counts.
Name not on the roster and never labelled = plain keyword, no filter
→ pass `people=[...]` or `set_roster` first.
`tags` = hard filter (AND). Filler-only query ("show me photos") =
newest first. Same backend + defaults as the `search_photos` tool
(limit 10).

Rows: `id, description (short), tags, people, created, source, sender,
score`. `print(p)` shows people with confidence.

| `people[].confidence` | Means | Say |
|---|---|---|
| `confirmed` | confirmed by a person, as reported by the agent (you, earlier) | plainly |
| `captioned` | caption passed at ingest names them — not checked against pixels | plainly; `view()` first if it matters |
| `likely` | inferred from appearance vs roster | "looks like Zara" |
| `p.unsure` (names) | weak guess, not a person, never matched | ask |

Nothing found → `[]` + `photos.hint` (which filter to drop, or
rephrase). Results are never broadened for you — do NOT show unrelated
photos; retry per the hint and say what you dropped. `show=True` on an
empty result shows nothing (`photos.shown == 0`).

## People — roster / set_roster / unconfirmed / confirm_people / set_people

The small model names household people at ingest from the caption +
the roster. Roster = who may be named, one line each on how to
recognise them. Local to the box; roster + caption go only into the
existing small-model tagging call.

```python
bb.photos.roster()                                   # [{name, note, person_id}]
bb.photos.set_roster("Zara", note="<age, hair, glasses, usual context>")  # add / update
bb.photos.remove_from_roster("Zara")

for p in bb.photos.unconfirmed():                    # likely/unsure, unreviewed
    print(p.id, p.people, p.unsure)                  # items carry `basis`
bb.photos.confirm_people(photo_id)                   # a person agreed with labels as-is
bb.photos.set_people(photo_id, names=["Zara", "Erik"])   # a person gave the names
```

- `set_people` REPLACES all names (add / remove / correct); `names=[]`
  = nobody from the household. Works with no detected slot.
- `confirm_people` / `set_people` only after a person told you. They
  mark the photo `confirmed` — stored as reported by you
  (`source: agent_reported` + this conversation and channel), not as
  the person's own act; the tagger never changes it again.
- When it matters who is in a photo (sending it on, a keepsake, a
  `likely` you are about to assert), `view()` it and ask a human.
- Empty roster = nobody gets named. New household member, or a child
  grew/changed → update the note.
- Names: letters, digits, space, `. ' -`, max 40. Note max 200.

## View — pixels attach to the tool result

```python
bb.photos.view("abc123def456")   # → {id, filename, kind: "image", attached: True}
bb.photos.view_path("<path>")    # same, for files not yet in the library
```

Inbound image: message starts `[image attached at <path>]` — pass that
exact path to `view_path`. Allowlisted roots only (sandbox tmp,
workspace, photos, perception crops).

## Ingest — ingest(path, source, sender, caption) into the library

```python
photo_id = bb.photos.ingest(path,
                            source="whatsapp",        # mandatory
                            sender="Erik",            # optional
                            caption="my new pokémon") # optional; pass it verbatim
```

Copies bytes into `data/photos/`, tags (description, tags, roster
people), indexes, deletes the original. Pass the sender's caption
as-is: a caption naming a roster person → `captioned` ("not Zara" →
`likely`). Names the caption did not give are kept out of
description/tags. Ingest = worth keeping (family moments, "remember
this"). Memes/throwaway = view, respond, don't ingest — the inbound
janitor reaps them (7-day TTL).

## Show on the 7" screen — search(..., show=True) / show_on_screen(photo_ids)

```python
bb.photos.search(query="Zara", show=True)             # default recipe: find + show
bb.photos.show_on_screen([p.id for p in photos[:5]])  # ids you already hold
```

Dispatches the `picture` display — for humans in the room, does NOT
attach to the tool result; pair with `view()` to see what you're
showing. Headless → nothing shown (`shown == 0` / `{dispatched: False,
reason: "display manager not running"}`). `picture` with no ids =
slideshow over the `add_to_slideshow` set; empty set = "No photos yet."

## Metadata — update / set_tags (replaces) / merge_tags / rename_tag / delete_tag

```python
bb.photos.update(photo_id, description="Dad's 60th, Apr 2026")  # re-embeds for search
bb.photos.set_tags(photo_id, tags=["family", "party"])          # REPLACES whole list
```

Add one tag = read-modify-write:
`set_tags(id, tags=sorted(set(p.tags) | {"birthday"}))`.
Who is in a photo → `set_people`, not a name tag.
(`set_person(photo_id, person_index=, name=)` renames one existing
slot; prefer `set_people`.)

Tag vocabulary is flat, shared across all photos:
`merge_tags("kids", into="children")` ·
`rename_tag("xmas", to="christmas")` · `delete_tag("blurry")`.
Return nothing; affected-photo counts land in the tool result's
`sdk_actions`.

## Slideshow & lifecycle — add_to_slideshow / remove_from_slideshow / delete (soft) / restore

`add_to_slideshow(id)` · `remove_from_slideshow(id)` · `delete(id)`
(soft, 30-day retention) · `restore(id)`
