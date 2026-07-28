# bb.photos — the photo library

Durable store of images worth remembering: WhatsApp sends, camera
captures you saved, user uploads. Separate from the workspace (your
notebook) and from perception crops (ephemeral).

**Use for:** finding a photo the user asked for; pulling photos tagged
with a person; curating the slideshow; `view()` before describing a
photo so you speak from pixels, not metadata.

**Not for:** scratch images, visualizations, debug snapshots →
`bb.workspace`. Raw perception state — never surfaces as a library
photo.

## API

### Search and get

```python
photos = bb.photos.search(
    query="snorlax plushie on kitchen table",
    tags=["indoor", "kitchen"],     # AND
    people=["Erik"],                # AND
    limit=10,
)
for p in photos:
    print(p.id, p.description, p.tags)

p = bb.photos.get("abc123def456")   # full record; PhotosError if missing
```

`search` returns `PhotoRecord`s ranked by hybrid retrieval (vector +
BM25). Filters AND together. No query = newest first.

### View — see the pixels

```python
bb.photos.view("abc123def456")
bb.photos.view_path("/var/lib/boxbot-sandbox/tmp/inbound/whatsapp/wamid.HBg.jpg")
```

`view()` attaches the JPEG to the tool result as an image block — same
mechanism as `bb.workspace.view()` and `bb.camera.capture()`. Returns
`{id, filename, kind: "image", attached: True}`. Use it before
answering "what's in that photo?".

`view_path()` does the same for files not yet in the library — usually
an inbound image. The user's message starts with
`[image attached at <path>]`; pass that exact path. Only allowlisted
roots work (sandbox tmp, workspace, photos, perception crops).

### Ingest

```python
photo_id = bb.photos.ingest(
    "/var/lib/boxbot-sandbox/tmp/inbound/whatsapp/wamid.HBg.jpg",
    source="whatsapp",          # mandatory; used for filtering later
    sender="Erik",              # optional
    caption="my new pokémon",   # optional; seeds the description
)
```

Copies bytes into `data/photos/`, runs detection + tagging, indexes for
search, deletes the original on success.

Ingest what is worth keeping: family moments, things the user asked you
to remember. Skip memes and throwaway shares — view, respond, and let
the inbound janitor reap them (7-day TTL).

### Show on the 7" screen

```python
bb.photos.show_on_screen(["abc123def456"])

results = bb.photos.search(query="Emily birthday")
bb.photos.show_on_screen([p.id for p in results[:5]])
```

Dispatches to the `picture` display. For humans in the room — it does
NOT attach to the tool result. Pair with `view()` to see what you are
showing. Returns `{dispatched: False, reason: "display manager not
running"}` when headless.

### Metadata

```python
bb.photos.update(photo_id, description="Dad's 60th, Apr 2026")
bb.photos.set_tags(photo_id, tags=["family", "party"])
bb.photos.set_person(photo_id, person_index=0, name="Erik")
```

`update()` re-embeds the description so hybrid search stays consistent.

`set_tags()` **replaces** the whole tag list. To add one tag, read
first:

```python
p = bb.photos.get(photo_id)
bb.photos.set_tags(photo_id, tags=sorted(set(p.tags) | {"birthday"}))
```

`set_person(person_index=…)` labels an already-detected face slot. If
intake found no faces there are no slots — it errors. Fall back to a
name tag or the description.

### Tag vocabulary

Flat and shared across all photos. Keep it tidy:

```python
bb.photos.merge_tags("kids", into="children")
bb.photos.rename_tag("xmas", to="christmas")
bb.photos.delete_tag("blurry")
```

These return nothing. Affected-photo counts land in the tool result's
`sdk_actions` entry.

### Slideshow and lifecycle

```python
bb.photos.add_to_slideshow(photo_id)
bb.photos.remove_from_slideshow(photo_id)
bb.photos.delete(photo_id)    # soft delete, 30-day retention
bb.photos.restore(photo_id)
```

## Patterns

"Show me that photo Emily sent last week":

```python
results = bb.photos.search(query="recent photos from Emily",
                           people=["Emily"], limit=1)
if results:
    bb.photos.view(results[0].id)               # see it
    bb.photos.show_on_screen([results[0].id])   # and put it on screen
else:
    print("no matching photo")
```

Inbound image — view, then decide:

```python
path = "/var/lib/boxbot-sandbox/tmp/inbound/whatsapp/wamid.HBg.jpg"
bb.photos.view_path(path)          # see it this turn
bb.photos.ingest(path, source="whatsapp", sender="Erik",
                 caption="check this out")
```

Not keeping it? Don't call `ingest()`. The janitor handles it.

Curate the idle slideshow:

```python
for tag in ("family", "vacation"):
    for p in bb.photos.search(tags=[tag], limit=20):
        bb.photos.add_to_slideshow(p.id)
```

## Display behavior

`show_on_screen(ids)` renders those photos full-screen on `picture`.
Switching to `picture` with no ids runs slideshow mode over the
`add_to_slideshow` set; an empty set shows "No photos yet."
