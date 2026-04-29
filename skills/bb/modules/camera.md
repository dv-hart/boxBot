# bb.camera — capture stills

The Pi camera is a shared resource. These calls go through the HAL and
serialize with the perception pipeline. Every capture is attached to
the tool result as an image content block, so you actually see the
pixels back — no need for a separate "view" step.

## When to use it

- Someone new is in front of boxBot and you want to note what they
  look like (pair with `identify_person` + a person memory).
- The user points at something and asks what it is.
- You want to save a moment to the photo library or the workspace.
- A skill needs a visual snapshot for analysis.

## When NOT to use it

- For continuous monitoring — that's what the perception pipeline is
  for. Don't call `capture()` in a tight loop.
- Before you have any reason to. Captures take ~30 ms (preview) to
  ~200 ms (full sensor), and compete with perception for camera time.
- To capture audio (use `identify_person` / voice pipeline for that).

## API

### Simple capture

```python
result = bb.camera.capture()
# {
#   "ref": "camera_abc123.jpg",    # filename
#   "path": "tmp/camera_abc123.jpg",  # relative to sandbox tmp
#   "width": 1280, "height": 720,
#   "saved": False,                 # ephemeral
#   "attached": True,               # you see it this turn
#   "fallback": False,              # True if test-pattern stand-in
# }
```

The image is attached to the tool result so you can look at it right
away. If you don't save it, it gets cleaned up with the sandbox tmp.

### Full-sensor capture (slower, higher quality)

```python
bb.camera.capture(full_res=True)
```

Uses the 12 MP still configuration. Slower (~200 ms for the mode
switch) but the output is suitable for the photo library or anything
that needs detail.

### Save to workspace

```python
bb.camera.capture(save_to="captures/erik_2026-04-24.jpg")
```

Writes directly to the workspace (goes through `bb.workspace` path
safety + quota). Returns the workspace-relative path. Still attaches
the image to the tool result.

### Cropped capture

```python
bb.camera.capture_cropped(
    bbox={"x": 420, "y": 80, "w": 180, "h": 240},
    save_to="notes/people/erik/headshot.jpg",
)
```

`bbox` uses main-stream coordinates (top-left origin). Values are
clamped to image bounds, so slightly oversized boxes don't fail — they
just get trimmed. Useful for the person-bootstrap flow: take a crop of
the speaker's bbox from perception and note what they look like.

Set `full_res=True` to treat the bbox as full-sensor coordinates; you
get higher-detail crops at the cost of latency.

## Patterns

### First-meeting bootstrap

```python
# You've just met someone new; perception gave you their bbox.
bbox = {"x": 420, "y": 80, "w": 180, "h": 240}
cap = bb.camera.capture_cropped(bbox=bbox, save_to=f"notes/people/{name}/headshot.jpg")
# Image is attached — you see their face now. Write appearance notes.
bb.workspace.write(
    f"notes/people/{name}.md",
    "# {name}\n- medium build, dark curly hair, round glasses\n- sounded ~30s\n",
)
bb.memory.save(
    kind="person",
    subject=name,
    content=f"{name} appearance notes are at notes/people/{name}.md; headshot alongside.",
)
```

### Snapshot to the photo library (full-res)

```python
r = bb.camera.capture(full_res=True, save_to="captures/moment.jpg")
# Then move/copy it into the photo library via bb.photos (module-specific API).
```

### "What is this?" question

```python
# User points at something unusual; take a quick look
bb.camera.capture()
# The image attaches; use the visual context when composing your reply.
```

## Limits and gotchas

- Max 8 images attached per `execute_script` call. Past that, captures
  still happen but aren't inlined to the tool result.
- Test-pattern fallback: if the camera HAL isn't running (dev/test),
  `capture()` returns a teal-with-diagonal-stripe test frame and sets
  `"fallback": True` in the result. Useful for dev; on-device it should
  never appear — if it does, the camera is down.
- `capture_cropped` with `w=0` or `h=0` will silently clamp to 1 px.
  Don't assume a zero-sized bbox produces an error.
