# Cameras — snapshot, then choose how to look

`camera_snapshot` is deliberately a single-responsibility action: it
fetches the latest still from HA, writes a JPEG to the workspace, and
returns the path. No interpretation, no tagging, no inference. Those
are the *next* step — and the right next step depends on the question.

## The three interpretation paths

```python
import boxbot_sdk as bb

snap = bb.integrations.get(
    "home_assistant",
    action="camera_snapshot",
    entity_id="camera.front_door",
)
path = snap["output"]["image_path"]   # e.g. "tmp/ha/camera_front_door_20260517T143052Z.jpg"
```

### Path A — agent sees pixels (rich visual reasoning)

When the question wants full visual judgment ("who's at the door?",
"what is she holding?", "is the mail truck out there?"), pull the
pixels into *your* context:

```python
bb.workspace.view(path)
# Image attaches to the tool result. The full Claude vision stack —
# faces, objects, text, scene — is available to you for the rest of
# the turn.
```

Cost: vision tokens, one image worth of context. Speed: one extra
attachment round-trip. Use when reasoning matters.

### Path B — small-model tag (cheap binary/classification)

When the question is binary or low-stakes ("is anyone at the porch?",
"is there a package?", "is the gate open?"), defer to the photo
tagging path the photo intake already uses. Don't burn the main
agent's vision tokens on yes/no questions.

The exact small-model entry point depends on what's wired in the
sandbox; check `bb.photos` or the in-process photo tagger. If a
small-model surface for ad-hoc images doesn't exist yet, fall back
to Path A — *don't* invent a half-built classifier inline.

### Path C — screen only (no interpretation)

When the user just wants to *see* the porch, neither path A nor B —
put it on the display.

```python
# As part of a "picture" display or any display that accepts an image:
bb.display.show("picture", args={"image_paths": [path]})
```

The user looks. The agent doesn't burn tokens on pixels it didn't
need to interpret. If the user then asks a question about what's on
screen, switch to Path A.

## Which path, when?

The user's question is the deciding signal. A rough triage:

| User says… | Path |
|------------|------|
| "Show me the porch." | C — screen only |
| "Who's at the door?" | A — pixels to agent |
| "What is on the porch?" | A |
| "Is anyone outside?" | B if available, else A |
| "Is the mail here?" | B if available, else A |
| "Describe what you see." | A |
| "Watch the porch for the next hour and tell me if a package arrives." | trigger + B in a loop; A is too expensive |

Don't pre-commit. The integration returns a path; the path supports
all three. Optionality is the design.

## Cleanup

Snapshots accumulate in `tmp/ha/`. They're in a `tmp/` folder
intentionally — workspace tmp paths are fair game to delete. If you
care about the image long-term (the agent identified the delivery
person, you want to remember the face), copy it to a non-tmp path:

```python
bb.workspace.write("captures/porch/delivery_2026-05-17.jpg",
                   bb.workspace.read(path, binary=True))  # if binary read is wired
```

Or save the relevant facts to memory and leave the image transient.

## Failure modes

- **`"camera not found or no snapshot available"`** — HA can't reach
  the camera (ADC cloud hiccup, network), or the entity exists but
  has no still buffered. Retry after a few seconds; if persistent,
  surface to the user.
- **`"empty snapshot returned"`** — HA returned 200 with zero bytes.
  Same triage as above.
- **`"camera_snapshot expects a camera.* entity"`** — the agent
  passed a non-camera entity_id. Likely a fuzzy-match miss; recheck
  with `get_states(domain="camera")`.

## Why not stream?

There is no live-video block in the display system today. The
display data-source manager re-fetches on a refresh schedule — fast
enough for ~1 fps "is the porch still active" effects, way too slow
for actual video. If real-time matters, add a `camera_stream` block
(out of scope for V1) or accept the 1-second cadence.

For "watch the porch and ping me on motion," the right shape is a
trigger that polls `binary_sensor.front_door_motion` (or whatever
HA's motion entity is for that camera) — not video.
