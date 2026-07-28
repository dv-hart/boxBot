# Cameras — snapshot, then choose how to look

`camera_snapshot` fetches the latest still from HA, writes a JPEG to
the workspace, returns the path. No interpretation, no tagging, no
inference. That is the *next* step, and it depends on the question.

```python
import boxbot_sdk as bb

snap = bb.integrations.get(
    "home_assistant",
    action="camera_snapshot",
    entity_id="camera.front_door",
)
path = snap["output"]["image_path"]   # "tmp/ha/camera_front_door_20260517T143052Z.jpg"
```

## Three paths

**A — pixels to you.** Full visual judgment: "who's at the door?",
"what is she holding?", "is the mail truck out there?"

```python
bb.workspace.view(path)
# Attaches to the tool result. Faces, objects, text, scene — all
# available to you for the rest of the turn.
```

Costs one image of vision tokens and an attachment round-trip. Use when
reasoning matters.

**B — small-model tag.** Binary or low-stakes: "is anyone at the
porch?", "is there a package?", "is the gate open?" Do not burn the
main agent's vision tokens on yes/no.

The entry point depends on what's wired in the sandbox — check
`bb.photos` or the in-process photo tagger. If no ad-hoc small-model
surface exists yet, fall back to Path A. Do **not** invent a half-built
classifier inline.

**C — screen only.** The user wants to *see* the porch:

```python
bb.display.show("picture", args={"image_paths": [path]})
```

They look; you spend nothing on pixels you did not need to interpret.
A follow-up question about what's on screen moves you to Path A.

## Triage

| User says | Path |
|-----------|------|
| "Show me the porch." | C |
| "Who's at the door?" | A |
| "What is on the porch?" | A |
| "Is anyone outside?" | B if available, else A |
| "Is the mail here?" | B if available, else A |
| "Describe what you see." | A |
| "Watch the porch for an hour, tell me if a package arrives." | entity trigger, snapshot on fire |
| "Tell me when someone approaches the front door." | entity trigger on `binary_sensor.<camera>_person` |

Do not pre-commit. One path supports all three. Optionality is the
design.

## Watching — entity triggers, no polling

Alarm.com cameras expose per-camera detection sensors that flip `on`
for ~10 s when video analytics fires: `binary_sensor.<camera>_person`,
`_vehicle`, `_animal`, `_package` (e.g.
`binary_sensor.front_door_person`). The HA events bridge streams these
in ~1-2 s.

```python
bb.tasks.create_trigger(
    description="Person at the front door",
    instructions=(
        "Someone approached the front door. Grab a snapshot of "
        "camera.front_door, look at it, and tell Jacob who/what it is."
    ),
    entity="binary_sensor.front_door_person",
    for_person="Jacob",
)
```

On fire you wake with the instructions — usually `camera_snapshot` →
`bb.workspace.view` (Path A) → `message`.

Conditions AND together: adding `person="Jacob"` means "only while
Jacob is in the room" (BB's own camera); `fire_after="2h"` means "not
before two hours from now". "Watch for the next hour only" = pass
`expires` with an ISO datetime an hour out. One-shot by default;
expires in ~7 days if never fired. Find sensor ids with
`get_states(domain="binary_sensor")`.

## Cleanup

Snapshots pile up in `tmp/ha/`. Workspace tmp paths are fair game to
delete. To keep one (you identified the delivery person and want to
remember the face), copy it out of tmp:

```python
bb.workspace.write("captures/porch/delivery_2026-05-17.jpg",
                   bb.workspace.read(path, binary=True))
```

Or save the facts to memory and let the image go.

## Failures

- `"camera not found or no snapshot available"` — HA can't reach the
  camera (ADC cloud hiccup, network), or no still is buffered. Retry in
  a few seconds; if it persists, tell the user.
- `"empty snapshot returned"` — HA returned 200 with zero bytes. Same
  triage.
- `"camera_snapshot expects a camera.* entity"` — non-camera
  entity_id, likely a fuzzy-match miss. Recheck with
  `get_states(domain="camera")`.

## No streaming

There is no live-video block. The display data-source manager
re-fetches on a refresh schedule — fine for ~1 fps "is the porch still
active", useless for video. Real-time would need a `camera_stream`
block (out of scope for V1).
