# HA entity model — domains, naming, discovery

HA unifies every device into one namespace: `<domain>.<object_id>`.
Domain says what kind of thing it is. `object_id` is HA's handle — the
friendly_name slugified at import, then editable.

## Domains in this house

| Domain | Holds | States |
|--------|-------|--------|
| `light` | Smart bulbs, switches that report on/off. Sengled Z-Wave lands here. | `"on"` / `"off"` |
| `switch` | Smart plugs, generic relays. | `"on"` / `"off"` |
| `binary_sensor` | Door/window, motion, presence. Read-only. | `"on"` (open/detected) / `"off"` |
| `sensor` | Numeric or textual readings — temperature, humidity, battery. Read-only. | varies (`"72.3"`) |
| `alarm_control_panel` | The ADC panel. | `"armed_home"`, `"armed_away"`, `"armed_night"`, `"disarmed"`, `"pending"`, `"triggered"` |
| `lock` | Smart locks. | `"locked"`, `"unlocked"`, `"locking"`, `"unlocking"` |
| `cover` | Garage doors, gates, motorized blinds. | `"open"`, `"closed"`, `"opening"`, `"closing"` |
| `camera` | Every camera HA knows, ADC included. | `"idle"`, `"recording"`, `"streaming"` |
| `climate` | Thermostats. | `"heat"`, `"cool"`, `"off"`, `"auto"` |
| `media_player` | Speakers, TVs, casts. | `"playing"`, `"paused"`, `"idle"`, `"off"` |
| `person` | Combined presence from device_tracker + zones. | `"home"`, `"not_home"`, zone name |
| `scene` | Pre-saved entity bundles. You *activate* them. | (meaningless) |
| `script` / `automation` | HA-side procedures. | `"on"` while running |

## Discovery

You do not know the inventory ahead of time. Pull it.

```python
import boxbot_sdk as bb

all_states = bb.integrations.get("home_assistant",
                                 action="get_states")["output"]["entities"]

lights = bb.integrations.get("home_assistant", action="get_states",
                             domain="light")["output"]["entities"]
# [{"entity_id": "light.living_room", "state": "on",
#   "friendly_name": "Living Room", "last_changed": "2026-05-17T14:22:01Z"}, ...]
```

Speak the `friendly_name` back to the user, never the raw `entity_id`.

## Full attributes

`get_states` trims to four fields per entity for context efficiency.
For the rich payload — bulb color, thermostat target, camera attrs —
call `get_state` on one entity:

```python
bulb = bb.integrations.get("home_assistant", action="get_state",
                           entity_id="light.living_room")["output"]
# {"state": "on",
#  "attributes": {"brightness": 200, "rgb_color": [255, 180, 90],
#                 "color_mode": "rgb",
#                 "supported_color_modes": ["rgb", "color_temp"],
#                 "friendly_name": "Living Room", ...},
#  "last_changed": "2026-05-17T14:22:01Z"}
```

`attributes` is whatever HA exposes for that integration: color and
brightness for Z-Wave bulbs, supported armed modes for the panel, a
`friendly_name` and sometimes a `model` for cameras.

## Do not poll

Entity lists rarely change. Fetched `get_states` this turn? Trust it
for the rest of the turn. Each call is an HTTP round-trip and shows up
in `bb.integrations.logs("home_assistant")`.

## Fuzzy naming

The user's name for a device may not be HA's. Don't fail loudly —
match against `friendly_name`:

```python
lights = bb.integrations.get("home_assistant", action="get_states",
                             domain="light")["output"]["entities"]
target = next(
    (e for e in lights
     if "living" in e["friendly_name"].lower()
        or "living" in e["entity_id"].lower()),
    None,
)
```

Can't disambiguate? Ask. Don't guess between "living room" and "living
room accent."
