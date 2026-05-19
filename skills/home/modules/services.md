# Service-call reference — what to do in each domain

Every action that *changes* something in HA is a service call:
`<domain>.<service>` plus a body. The body usually targets one or more
entities and may include extra parameters (color, temperature, volume).

```python
bb.integrations.get("home_assistant", action="call_service",
                    domain=..., service=...,
                    entity_id="...",        # optional — which entity to act on
                    service_data={...})     # optional — params for the service
```

`entity_id` and any keys in `service_data` are merged into the request body
HA receives. `entity_id` may be a single ID or a list of IDs.

If you don't know which service to call, `list_services` with the same
`domain` returns the full menu HA exposes — including services from custom
integrations the user installed.

## Lights

```python
# On / off
{"domain": "light", "service": "turn_on",  "entity_id": "light.living_room"}
{"domain": "light", "service": "turn_off", "entity_id": "light.living_room"}
{"domain": "light", "service": "toggle",   "entity_id": "light.living_room"}

# Brightness 0–255
{"domain": "light", "service": "turn_on",
 "entity_id": "light.living_room",
 "service_data": {"brightness": 80}}

# RGB color (only on bulbs whose supported_color_modes include "rgb")
{"domain": "light", "service": "turn_on",
 "entity_id": "light.living_room",
 "service_data": {"rgb_color": [180, 30, 90], "brightness": 120}}

# Color temperature (warm 153 ⇢ cool 500 mireds, varies by bulb)
{"domain": "light", "service": "turn_on",
 "entity_id": "light.living_room",
 "service_data": {"color_temp": 350}}

# Smooth transitions
{"domain": "light", "service": "turn_on",
 "entity_id": "light.living_room",
 "service_data": {"brightness": 30, "transition": 4}}   # seconds
```

Sengled Z-Wave color bulbs (the two paired through the ADC panel) support
`rgb_color`, `brightness`, `color_temp`, and `transition`. Confirm with
`get_state` → `attributes.supported_color_modes` before assuming.

## Switches and smart plugs

```python
{"domain": "switch", "service": "turn_on",  "entity_id": "switch.basement_fan"}
{"domain": "switch", "service": "turn_off", "entity_id": "switch.basement_fan"}
{"domain": "switch", "service": "toggle",   "entity_id": "switch.basement_fan"}
```

Switches are dumb — no brightness, no color, just on/off. If a switch
controls something dangerous (a heater, a garage door wired as a switch),
treat the user's "turn it on" with care.

## Scenes and scripts — the canonical "mood"

A **scene** is a saved snapshot of one or more entities ("Movie Night" =
TV on, lights at 10% warm). A **script** is a procedure HA runs ("Goodnight"
= dim lights over 30s, lock doors, arm alarm).

```python
# Activate a scene
{"domain": "scene", "service": "turn_on", "entity_id": "scene.movie_night"}

# Run a script
{"domain": "script", "service": "turn_on", "entity_id": "script.goodnight"}
# or, equivalently:
{"domain": "script", "service": "goodnight"}
```

If the user defined a "set the mood" scene in HA, prefer calling it over
reconstructing one in code — the scene is editable in HA's UI without
touching BB.

## Climate

```python
# Set target temperature
{"domain": "climate", "service": "set_temperature",
 "entity_id": "climate.living_room",
 "service_data": {"temperature": 68}}

# Set mode (heat / cool / off / auto / heat_cool)
{"domain": "climate", "service": "set_hvac_mode",
 "entity_id": "climate.living_room",
 "service_data": {"hvac_mode": "cool"}}

# Set fan mode (auto / on / low / medium / high — varies by thermostat)
{"domain": "climate", "service": "set_fan_mode",
 "entity_id": "climate.living_room",
 "service_data": {"fan_mode": "auto"}}
```

Check the entity's `attributes.hvac_modes` and `attributes.fan_modes`
before guessing what values are valid.

## Media players

```python
# Basic transport
{"domain": "media_player", "service": "media_play",  "entity_id": "media_player.kitchen"}
{"domain": "media_player", "service": "media_pause", "entity_id": "media_player.kitchen"}
{"domain": "media_player", "service": "media_stop",  "entity_id": "media_player.kitchen"}
{"domain": "media_player", "service": "media_next_track", "entity_id": "media_player.kitchen"}

# Volume (0.0 – 1.0)
{"domain": "media_player", "service": "volume_set",
 "entity_id": "media_player.kitchen",
 "service_data": {"volume_level": 0.4}}

# Mute
{"domain": "media_player", "service": "volume_mute",
 "entity_id": "media_player.kitchen",
 "service_data": {"is_volume_muted": true}}
```

For *playing audio on the box itself*, use `bb.audio.play` — not HA. HA
media players are external speakers the user has paired (Sonos, casts,
TVs).

## Fans / humidifiers / vacuums / etc.

Same pattern: `domain.turn_on`, `domain.turn_off`, plus domain-specific
services exposed via `list_services`. When in doubt, list and read the
descriptions HA ships.

## Notifications

```python
{"domain": "notify", "service": "mobile_app_<device>",
 "service_data": {"message": "Front door opened.", "title": "boxBot"}}
```

The exact service name depends on which HA companion apps are paired.
List `notify.*` services to see what's available. This is rarely the
right tool for BB — its native WhatsApp + voice channels are usually
better — but it exists.

## Blocked in V1

```python
{"domain": "alarm_control_panel", "service": "alarm_arm_home", ...}   # ✗
{"domain": "alarm_control_panel", "service": "alarm_disarm",   ...}   # ✗
{"domain": "lock",  "service": "lock",   ...}                          # ✗
{"domain": "lock",  "service": "unlock", ...}                          # ✗
{"domain": "cover", "service": "open_cover",  ...}                     # ✗
{"domain": "cover", "service": "close_cover", ...}                     # ✗
```

The integration returns an error with the domain name. See
`mutation_policy.md` for the rationale and the V2 plan.
