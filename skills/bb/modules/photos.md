# bb.photos — the photo library

Durable images worth remembering: WhatsApp sends, saved camera
captures, uploads. Not: scratch/debug images → `bb.workspace`;
perception crops (ephemeral, never surface here).

"What's in that photo?" → `view()` first — speak from pixels, not
metadata.

## Call signatures — search(query, tags, people, limit) / get(photo_id)

```python
photos = bb.photos.search(query="snorlax plushie on kitchen table",
                          tags=["indoor"], people=["Erik"], limit=10)
p = bb.photos.get("abc123def456")   # full record; PhotosError if missing
```

`search` → `PhotoRecord`s, hybrid-ranked (vector + BM25). Filters AND.
No query = newest first.

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
                            caption="my new pokémon") # optional; seeds description
```

Copies bytes into `data/photos/`, runs detection + tagging, indexes,
deletes the original. Ingest = worth keeping (family moments, "remember
this"). Memes/throwaway = view, respond, don't ingest — the inbound
janitor reaps them (7-day TTL).

## Show on the 7" screen — show_on_screen(photo_ids)

```python
bb.photos.show_on_screen([p.id for p in results[:5]])
```

Dispatches the `picture` display — for humans in the room, does NOT
attach to the tool result; pair with `view()` to see what you're
showing. Headless → `{dispatched: False, reason: "display manager not
running"}`. `picture` with no ids = slideshow over the
`add_to_slideshow` set; empty set = "No photos yet."

## Metadata — update / set_tags (replaces) / set_person / merge_tags / rename_tag / delete_tag

```python
bb.photos.update(photo_id, description="Dad's 60th, Apr 2026")  # re-embeds for search
bb.photos.set_tags(photo_id, tags=["family", "party"])          # REPLACES whole list
bb.photos.set_person(photo_id, person_index=0, name="Erik")
```

Add one tag = read-modify-write:
`set_tags(id, tags=sorted(set(p.tags) | {"birthday"}))`.
`set_person` labels an already-detected face slot; no faces detected =
no slots = error → fall back to a name tag or the description.

Tag vocabulary is flat, shared across all photos:
`merge_tags("kids", into="children")` ·
`rename_tag("xmas", to="christmas")` · `delete_tag("blurry")`.
Return nothing; affected-photo counts land in the tool result's
`sdk_actions`.

## Slideshow & lifecycle — add_to_slideshow / remove_from_slideshow / delete (soft) / restore

`add_to_slideshow(id)` · `remove_from_slideshow(id)` · `delete(id)`
(soft, 30-day retention) · `restore(id)`
