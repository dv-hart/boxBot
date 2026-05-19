---
name: home
description: Control the household through Home Assistant — lights, scenes, climate, media, smart plugs — and read live state like alarm armed/disarmed, doors, sensors, presence. The home_assistant integration is the single pipe; this skill is the map. Loads when the user asks BB to do anything that touches a physical device, set a mood, check the house, or peek at a camera.
when_to_use: |
  Load this when the user:
    - Asks BB to turn something on/off, change brightness/color, run a scene
      ("set the mood", "movie time", "wind down"), play media on a speaker.
    - Asks about house state — is the alarm armed, is the front door locked,
      is anyone home, what's the thermostat, are the lights on.
    - Asks BB to show a camera ("show me the porch", "who's at the door").
    - Asks BB to add/replace HA credentials, or asks why a smart-home action
      failed.
  Do NOT load this for: BB's own onboard camera (that's `bb.camera`), photo
  library (`bb.photos`), or speaker output of audio files (`bb.audio`).
---

# home — household control via Home Assistant

Everything physical in this house — Alarm.com panel + sensors + cameras, the
Sengled Z-Wave bulbs, and whatever else gets added later — flows through one
Home Assistant instance as a unified entity graph. BB talks to HA via the
**`home_assistant` integration**. There is no direct connector to Alarm.com,
no direct Hue/LIFX/etc.; HA is the abstraction.

```python
import boxbot_sdk as bb

bb.integrations.get("home_assistant", action="get_states", domain="light")
bb.integrations.get("home_assistant", action="call_service",
                    domain="light", service="turn_on",
                    entity_id="light.living_room",
                    service_data={"rgb_color": [180, 30, 90], "brightness": 80})
```

That's the whole API surface. The `action` argument picks one of five
verbs; everything else is parameters to it.

## The five actions

| Action | What it does | Common use |
|--------|--------------|------------|
| `get_states` | Lists all entities (optionally filtered by `domain`). Trimmed to entity_id, state, friendly_name, last_changed. | "What lights are on?" "What's in this house?" |
| `get_state` | Returns full state + attributes for one entity. | "Is the alarm armed?" "Is the front door locked?" |
| `call_service` | Calls any HA service. `domain` + `service` (e.g. `light.turn_on`), plus optional `entity_id` and `service_data`. | "Turn off the bedroom." "Set the thermostat to 68." |
| `camera_snapshot` | Pulls latest JPEG from a `camera.*` entity, writes it to `tmp/ha/<entity>_<timestamp>.jpg` in the workspace, returns the path. | "Show me the porch." "Who's at the door?" |
| `list_services` | Lists available services (optionally filtered by `domain`). Useful when you don't know what's callable. | "What can I do with this thermostat?" |

Output shape: every call returns `{"status": "ok", "output": {...}}` on
success, or `{"status": "error", "error": "..."}` (or `"timeout"`). Inside
`output`, the fields depend on the action — see the per-action sub-docs.

## Setup — one-time per device

Before BB can talk to your HA instance the user has to provide two things:

```python
bb.secrets.store("HOME_ASSISTANT_URL",   "http://192.168.0.5:8123")
bb.secrets.store("HOME_ASSISTANT_TOKEN", "<long-lived access token>")
```

Long-lived access tokens are generated in HA's web UI: profile (bottom-left
avatar) → "Long-Lived Access Tokens" → "Create Token". They don't expire.
The URL is whatever address HA is reachable at on the LAN — usually a
Docker host on the home network.

If either is missing, every action returns an `error` field with a clear
hint about which secret to set.

## Camera snapshots: the integration does NOT interpret

`camera_snapshot` is deliberately dumb. It fetches the JPEG, saves it,
returns the path. **What happens next is the caller's choice** — and the
choice depends on the question being asked.

```python
snap = bb.integrations.get("home_assistant",
                           action="camera_snapshot",
                           entity_id="camera.front_door")
path = snap["output"]["image_path"]
```

From here:

| If the question is… | Do this |
|---------------------|---------|
| "Who's at the door?" / "What is that?" / anything that wants visual reasoning | `bb.workspace.view(path)` — attaches pixels to your tool result so YOU see the image |
| "Is anyone at the porch?" / "Is a package there?" / cheap classification | Defer to a small-model tag (see `modules/cameras.md` for the pattern) |
| "Show me the porch on the screen" — no interpretation needed | `bb.display.show` with the saved image, or render via a picture display |

Do NOT pre-commit to one path. The same fetch supports all three downstream
choices; that flexibility is the design.

## Mutation policy (V1)

Three domains have their state-changing service calls **blocked** until the
confirmation gate ships:

- `alarm_control_panel.*` — arming and disarming the panel
- `lock.*` — locking and unlocking
- `cover.*` — garage doors, gates (conservatively also covers blinds; will
  refine to per-entity in V2)

State reads on these entities work fine — "is the alarm armed?" returns
the answer. Only `call_service` is gated. If a blocked call happens, the
integration returns an `error` like:

> service calls in domain 'alarm_control_panel' are blocked in V1 …

When the user asks for one of these, explain that BB can see the state
but can't change it yet, and offer to relay through HA's app instead.
See `modules/mutation_policy.md` for the full list and the V2 plan.

## Common patterns

**Set the mood** — color + brightness on a known bulb, then a song. The
agent composes both in one script; this isn't a primitive.

```python
bb.integrations.get("home_assistant", action="call_service",
                    domain="light", service="turn_on",
                    entity_id="light.living_room",
                    service_data={"rgb_color": [180, 30, 90], "brightness": 80})
bb.audio.play("music/mood/lets_get_it_on.mp3")
```

**Walk the house** — get every light's state.

```python
states = bb.integrations.get("home_assistant", action="get_states",
                             domain="light")["output"]["entities"]
on = [e["friendly_name"] for e in states if e["state"] == "on"]
```

**Check the alarm without changing it.**

```python
panel = bb.integrations.get("home_assistant", action="get_state",
                            entity_id="alarm_control_panel.home")["output"]
# panel["state"] ∈ {"armed_home", "armed_away", "disarmed", ...}
```

**Discover what's possible on an entity you've never touched.**

```python
bb.integrations.get("home_assistant", action="list_services", domain="climate")
```

## Progressive disclosure

This file (Level 2) is the overview. Load these for depth:

- `modules/entities.md` — HA's entity model, common domains, how to find
  what's in this house.
- `modules/services.md` — service-call reference for the things you'll
  actually call (lights, scenes, climate, media_player, switches, fans).
- `modules/cameras.md` — the snapshot pattern, the three interpretation
  paths, and the small-model-tag recipe.
- `modules/mutation_policy.md` — what's gated in V1, why, and how V2 will
  unblock it.

Load via `load_skill(name="home", subpath="modules/<file>.md")`.

## Related skills + APIs

- `bb` skill / `bb.integrations` — the integration runner this skill rides on.
- `bb.audio` — pairs naturally with `home` for "set the mood" workflows.
- `bb.workspace` — where camera snapshots land; where mood audio files live.
- `bb.display` — for putting snapshots on screen without agent interpretation.
