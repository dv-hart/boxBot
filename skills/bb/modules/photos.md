# bb.photos — the photo library

The photo library is the durable store of images the agent should
remember: WhatsApp sends, camera captures the agent explicitly saved,
user uploads. It's separate from the workspace (the agent's notebook)
and from the perception pipeline's crops (ephemeral).

## When to use it

- User asks for a photo ("show me the beach photos from last summer").
- You want to pull up something tagged with a person ("any recent
  photos with Emily?").
- Curating the slideshow rotation.
- Attaching a specific photo to a reply — call `view()` so you can
  look at it and describe it accurately.

## When NOT to use it

- For ephemeral scratch images (agent-authored visualizations, debug
  snapshots). Those belong in `bb.workspace`.
- For raw perception state. Those live in perception/crops and never
  surface as library photos.

## API

### Search

```python
photos = bb.photos.search(
    query="snorlax plushie on kitchen table",   # free-text
    tags=["indoor", "kitchen"],                  # AND-filter
    people=["Erik"],                             # AND-filter
    limit=10,
)
for p in photos:
    print(p.id, p.description, p.tags)
```

Returns a list of `PhotoRecord` objects ordered by hybrid-retrieval
score (vector similarity + BM25). Filters are AND-combined. If no
query is given, results come back newest-first.

### Get one by id

```python
p = bb.photos.get("abc123def456")
print(p.description, p.tags, p.people)
```

Returns the full record: id, description, tags, people, dimensions,
source, created_at, etc. Raises `bb.photos.PhotosError` if missing.

### View — see the pixels

```python
bb.photos.view("abc123def456")
```

Attaches the photo's JPEG to the tool result as an image content block.
Same mechanism as `bb.workspace.view()` and `bb.camera.capture()` —
the agent sees the image this turn. Returns `{id, filename, kind:
"image", attached: True}`.

Use this before responding to "what's in that photo?" or when you need
the actual content, not just the description.

### Show on the 7" screen

```python
# A single photo
bb.photos.show_on_screen(["abc123def456"])

# A rotating pick
results = bb.photos.search(query="Emily birthday")
bb.photos.show_on_screen([p.id for p in results[:5]])
```

Dispatches to the `picture` display on the physical screen. This is
for humans in the room — it does NOT attach to the tool result. Pair
with `view()` if you want to see what you're showing.

Currently stubs through when the display manager isn't wired up yet
(the call returns `{dispatched: False, reason: "display manager not
running"}`).

### Metadata updates

```python
bb.photos.update(photo_id, description="Dad's 60th, Apr 2026")
bb.photos.set_tags(photo_id, tags=["family", "party"])
bb.photos.set_person(photo_id, person_index=0, name="Erik")
```

### Slideshow

```python
bb.photos.add_to_slideshow(photo_id)
bb.photos.remove_from_slideshow(photo_id)
```

### Lifecycle

```python
bb.photos.delete(photo_id)   # soft delete, 30-day retention
bb.photos.restore(photo_id)
```

## Patterns

### "Show me that photo Emily sent last week"

```python
results = bb.photos.search(
    query="recent photos from Emily",
    people=["Emily"],
    limit=1,
)
if results:
    bb.photos.view(results[0].id)          # see it
    bb.photos.show_on_screen([results[0].id])  # and put it on the screen
else:
    print("no matching photo")
```

### Describing a photo the user is asking about

```python
p = bb.photos.get(photo_id)         # metadata
bb.photos.view(photo_id)            # pixels (attaches to tool result)
# Now you have the description + can cross-reference visual details.
```

### Curate the idle slideshow

```python
for tag in ("family", "vacation"):
    for p in bb.photos.search(tags=[tag], limit=20):
        bb.photos.add_to_slideshow(p.id)
```

## Known gaps

- `show_on_screen` is wired to the display manager but the `picture`
  display itself isn't fully implemented yet — until it lands, the
  call returns `dispatched: False`. `view()` (pixels to tool result)
  works now.
- Metadata mutation (`set_tags`, `delete`, etc.) currently acks the
  action but the main-process handlers are stubs. Read/view works;
  write doesn't yet.
