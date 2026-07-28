---
name: home
description: Control the household through Home Assistant — lights, scenes, climate, media, smart plugs — and read live state like alarm armed/disarmed, doors, sensors, presence. The home_assistant integration is the single pipe; this skill is the map. Loads when the user asks BB to do anything touching a physical device, set a mood, check the house, or peek at a camera.
when_to_use: |
  Load when the user:
    - Turns something on/off, changes brightness/color, runs a scene ("set the
      mood", "movie time", "wind down"), plays media on a speaker.
    - Asks about house state — alarm armed, front door locked, anyone home,
      thermostat, lights on.
    - Asks to show a camera ("show me the porch", "who's at the door").
    - Adds/replaces HA credentials, or asks why a smart-home action failed.
  Not for: BB's own onboard camera (`bb.camera`), the photo library
  (`bb.photos`), or playing audio files (`bb.audio`).
---

# home — household control via Home Assistant

Everything physical in this house — Alarm.com panel, sensors, cameras,
Sengled Z-Wave bulbs, whatever lands later — flows through one Home
Assistant instance as a unified entity graph. BB talks to HA through
the **`home_assistant` integration**. No direct Alarm.com connector, no
direct Hue/LIFX. HA is the abstraction.

```python
import boxbot_sdk as bb

bb.integrations.get("home_assistant", action="get_states", domain="light")
bb.integrations.get("home_assistant", action="call_service",
                    domain="light", service="turn_on",
                    entity_id="light.living_room",
                    service_data={"rgb_color": [180, 30, 90], "brightness": 80})
```

That is the whole surface. `action` picks a verb; everything else is
parameters.

## The five actions

| Action | Does | Use |
|--------|------|-----|
| `get_states` | All entities, optional `domain` filter. Trimmed to entity_id, state, friendly_name, last_changed. | "What lights are on?" |
| `get_state` | Full state + attributes for one entity. | "Is the alarm armed?" |
| `call_service` | Any HA service. `domain` + `service` (e.g. `light.turn_on`), optional `entity_id` and `service_data`. | "Turn off the bedroom." |
| `camera_snapshot` | Latest JPEG from a `camera.*` entity → `tmp/ha/<entity>_<ts>.jpg` in the workspace. Returns the path. | "Show me the porch." |
| `list_services` | Available services, optional `domain` filter. | "What can I do with this thermostat?" |

Every call returns `{"status": "ok", "output": {...}}`, or
`{"status": "error", "error": "..."}` / `"timeout"`. `output` fields
vary by action — see the sub-docs.

## Watching state: use entity triggers, not polling

"Tell me when X happens" — someone at the door, a door opening, a
package — is a trigger, not a poll:

```python
bb.tasks.create_trigger(..., entity="binary_sensor.front_door_person")
```

Fires within seconds; the HA events bridge streams state over a
WebSocket. Camera-watch pattern: [modules/cameras.md](modules/cameras.md).
Trigger semantics: the `bb` tasks module.

## Setup — once per device

```python
bb.secrets.store("HOME_ASSISTANT_URL",   "http://192.168.0.5:8123")
bb.secrets.store("HOME_ASSISTANT_TOKEN", "<long-lived access token>")
```

Token: HA web UI → profile (bottom-left avatar) → "Long-Lived Access
Tokens" → "Create Token". They do not expire. URL is wherever HA is
reachable on the LAN, usually a Docker host.

Either missing → every action returns an `error` naming the secret to
set.

## Camera snapshots do NOT interpret

`camera_snapshot` is deliberately dumb: fetch the JPEG, save it, return
the path. **What happens next is your call**, and it depends on the
question.

```python
snap = bb.integrations.get("home_assistant",
                           action="camera_snapshot",
                           entity_id="camera.front_door")
path = snap["output"]["image_path"]
```

| Question | Do |
|----------|-----|
| "Who's at the door?" / anything needing visual reasoning | `bb.workspace.view(path)` — attaches pixels so YOU see it |
| "Is anyone on the porch?" / cheap classification | Small-model tag — see `modules/cameras.md` |
| "Show me the porch on the screen" | `bb.display.show`, or a picture display |

Do not pre-commit to one path. One fetch supports all three. That
flexibility is the design.

## Mutation policy (V1)

State-changing service calls are **blocked** in three domains until the
confirmation gate ships:

- `alarm_control_panel.*` — arm/disarm
- `lock.*` — lock/unlock
- `cover.*` — garage doors, gates (conservatively also blinds;
  per-entity in V2)

Reads still work — "is the alarm armed?" answers fine. Only
`call_service` is gated. A blocked call returns:

> service calls in domain 'alarm_control_panel' are blocked in V1 …

When asked, say BB can see the state but not change it yet, and offer
to relay through HA's app. Full list and V2 plan:
`modules/mutation_policy.md`.

## Patterns

Set the mood — color, brightness, and a song, composed in one script:

```python
bb.integrations.get("home_assistant", action="call_service",
                    domain="light", service="turn_on",
                    entity_id="light.living_room",
                    service_data={"rgb_color": [180, 30, 90], "brightness": 80})
bb.audio.play("music/mood/lets_get_it_on.mp3")
```

Walk the house:

```python
states = bb.integrations.get("home_assistant", action="get_states",
                             domain="light")["output"]["entities"]
on = [e["friendly_name"] for e in states if e["state"] == "on"]
```

Check the alarm without changing it:

```python
panel = bb.integrations.get("home_assistant", action="get_state",
                            entity_id="alarm_control_panel.home")["output"]
# panel["state"] ∈ {"armed_home", "armed_away", "disarmed", ...}
```

Discover what an unfamiliar entity supports:

```python
bb.integrations.get("home_assistant", action="list_services", domain="climate")
```

## Deeper docs

`load_skill(name="home", subpath="modules/<file>.md")`

- `modules/entities.md` — HA's entity model, common domains, finding
  what's in this house.
- `modules/services.md` — service-call reference for lights, scenes,
  climate, media_player, switches, fans.
- `modules/cameras.md` — snapshot pattern, three interpretation paths,
  small-model-tag recipe.
- `modules/mutation_policy.md` — what's gated in V1 and why.

## Related

`bb.integrations` (the runner this rides on) · `bb.audio` (mood
workflows) · `bb.workspace` (where snapshots land) · `bb.display`
(snapshots on screen, no interpretation).
