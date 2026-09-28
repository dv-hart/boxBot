# bb.camera — stills from the box's own lens (property cameras are the `home` skill)

Shared resource. Calls go through the HAL and serialize with the
perception pipeline. Every capture attaches to the tool result as an
image block — you see the pixels immediately. No separate view step.

**Use for:** noting what someone new looks like (pair with
`identify_person` + a person memory); "what is this?" questions; saving
a moment; a visual snapshot a skill needs.

**Not for:** the property's security cameras — garage, doorbell,
porch — those are the `home` skill (`home_assistant` integration,
`camera_snapshot`). Continuous monitoring — that's the perception
pipeline; never loop on `capture()`. Captures cost ~30 ms (preview) to ~200 ms
(full sensor) and compete with perception for camera time. Audio —
that's the voice pipeline.

## Call signatures — capture() / capture_cropped(bbox, save_to)

```python
result = bb.camera.capture()
# {
#   "ref": "camera_abc123.jpg",
#   "path": "tmp/camera_abc123.jpg",   # relative to sandbox tmp
#   "width": 1280, "height": 720,
#   "saved": False,                     # ephemeral
#   "attached": True,                   # you see it this turn
#   "fallback": False,                  # True = test-pattern stand-in
# }

bb.camera.capture(full_res=True)        # 12 MP still config, ~200 ms slower
bb.camera.capture(save_to="captures/erik_2026-04-24.jpg")   # → workspace

bb.camera.capture_cropped(
    bbox={"x": 420, "y": 80, "w": 180, "h": 240},
    save_to="notes/people/erik/headshot.jpg",
)
```

Unsaved captures die with the sandbox tmp dir. `save_to` writes through
`bb.workspace` (path safety + quota) and returns the
workspace-relative path; the image still attaches.

`bbox` uses main-stream coordinates, top-left origin. Values clamp to
image bounds — an oversized box gets trimmed, not rejected. Pass
`full_res=True` to treat the bbox as full-sensor coordinates.

## Patterns — headshot crop to workspace on first meeting

First-meeting bootstrap:

```python
bbox = {"x": 420, "y": 80, "w": 180, "h": 240}   # from perception
bb.camera.capture_cropped(bbox=bbox, save_to=f"notes/people/{name}/headshot.jpg")
# Image attached — you see their face. Now write appearance notes.
bb.workspace.write(
    f"notes/people/{name}.md",
    f"# {name}\n- medium build, dark curly hair, round glasses\n- sounded ~30s\n",
)
bb.memory.save(
    content=f"{name} appearance notes are at notes/people/{name}.md; headshot alongside.",
    memory_type="person",
    person=name,
    summary=f"{name} headshot + notes saved",
)
```

"What is this?" — `bb.camera.capture()`, then use the attached image
when composing your reply.

## Gotchas — 8-image attach cap, no-camera error, test-frame fallback, zero-size bbox clamps

- Max 8 images per `execute_script` call. Beyond that, captures still
  happen but do not inline.
- No camera → `{"status": "error", "error": "No camera on this device."}`,
  nothing attached. Say so; do not describe an image you were not given.
- `"fallback": True` = teal test frame from the dev-only
  `camera.test_pattern_without_camera` flag, not a real capture.
- `capture_cropped` with `w=0` or `h=0` clamps to 1 px. It does not
  error.
