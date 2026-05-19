# HA entity model — domains, naming, discovery

Home Assistant unifies every device into one **entity** namespace:
`<domain>.<object_id>`. The domain says what *kind* of thing it is; the
object_id is HA's internal handle (it's the friendly_name slugified at
import time, then editable).

## Common domains you'll see in this house

| Domain | What lives there | Typical state values |
|--------|------------------|----------------------|
| `light` | Smart bulbs, light switches that report on/off. The Sengled Z-Wave bulbs surface here. | `"on"` / `"off"` |
| `switch` | Smart plugs, generic on/off relays. | `"on"` / `"off"` |
| `binary_sensor` | Door/window sensors, motion sensors, presence detectors. Read-only. | `"on"` (open/detected) / `"off"` (closed/clear) |
| `sensor` | Numeric/textual readings — temperature, humidity, battery levels. Read-only. | varies (e.g. `"72.3"`) |
| `alarm_control_panel` | The ADC panel itself. | `"armed_home"`, `"armed_away"`, `"armed_night"`, `"disarmed"`, `"pending"`, `"triggered"` |
| `lock` | Smart locks. | `"locked"`, `"unlocked"`, `"locking"`, `"unlocking"` |
| `cover` | Garage doors, gates, motorized blinds. | `"open"`, `"closed"`, `"opening"`, `"closing"` |
| `camera` | All cameras HA knows about — ADC cameras land here. | `"idle"`, `"recording"`, `"streaming"` |
| `climate` | Thermostats. | `"heat"`, `"cool"`, `"off"`, `"auto"` |
| `media_player` | Speakers, TVs, casts. | `"playing"`, `"paused"`, `"idle"`, `"off"` |
| `person` | A household member's combined presence (from device_tracker + zones). | `"home"`, `"not_home"`, zone name |
| `scene` | Pre-saved entity bundles. State is meaningless; you *activate* them. | (n/a) |
| `script` / `automation` | HA-side procedures. | `"on"` when running |

## Discovery — what's actually in this house?

The agent does not know the user's entity inventory ahead of time. Pull it.

```python
import boxbot_sdk as bb

# Everything
all_states = bb.integrations.get("home_assistant", action="get_states")["output"]["entities"]

# Just one domain
lights = bb.integrations.get("home_assistant",
                             action="get_states",
                             domain="light")["output"]["entities"]
# [{"entity_id": "light.living_room", "state": "on",
#   "friendly_name": "Living Room", "last_changed": "2026-05-17T14:22:01Z"}, ...]
```

Each entry includes the `friendly_name` HA renders in its UI — that's the
right thing to speak back to the user, not the raw `entity_id`.

## Full attributes — when get_states isn't enough

`get_states` trims to four fields per entity for context-efficiency. When
you actually need the rich payload — bulb color, thermostat target temp,
camera attributes — call `get_state` for the single entity:

```python
bulb = bb.integrations.get("home_assistant",
                           action="get_state",
                           entity_id="light.living_room")["output"]
# {
#   "state": "on",
#   "attributes": {
#     "brightness": 200,
#     "rgb_color": [255, 180, 90],
#     "color_mode": "rgb",
#     "supported_color_modes": ["rgb", "color_temp"],
#     "friendly_name": "Living Room",
#     ...
#   },
#   "last_changed": "2026-05-17T14:22:01Z"
# }
```

The `attributes` dict is whatever HA exposes for that integration. For
Z-Wave bulbs you get color/brightness; for the alarm panel you get the
list of armed modes it supports; for cameras you get a `friendly_name`
and (sometimes) a `model`.

## Caching — don't poll in a loop

Entity lists rarely change. If you've fetched `get_states` this turn,
trust it for the rest of the turn rather than fetching again. The
integration is cheap but not free — each call is one HTTP round-trip
and shows up in `bb.integrations.logs("home_assistant")`.

## Naming tips for the user

When the user refers to a device by a name HA doesn't recognize, do not
fail loudly — do a fuzzy match against `friendly_name`:

```python
lights = bb.integrations.get("home_assistant",
                             action="get_states",
                             domain="light")["output"]["entities"]
target = next(
    (e for e in lights
     if "living" in e["friendly_name"].lower()
        or "living" in e["entity_id"].lower()),
    None,
)
```

If you can't disambiguate, ask the user which one — don't guess between
"living room" and "living room accent."
