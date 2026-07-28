# Service-call reference

Every change in HA is a service call: `<domain>.<service>` plus a body
targeting entities, optionally with parameters.

```python
bb.integrations.get("home_assistant", action="call_service",
                    domain=..., service=...,
                    entity_id="...",        # optional; one id or a list
                    service_data={...})     # optional; params
```

`entity_id` and `service_data` keys merge into the request body.

Don't know the service? `list_services` on the same `domain` returns
HA's full menu, custom integrations included.

## Lights

```python
{"domain": "light", "service": "turn_on",  "entity_id": "light.living_room"}
{"domain": "light", "service": "turn_off", "entity_id": "light.living_room"}
{"domain": "light", "service": "toggle",   "entity_id": "light.living_room"}

# Brightness 0–255
{"domain": "light", "service": "turn_on",
 "entity_id": "light.living_room",
 "service_data": {"brightness": 80}}

# RGB — only when supported_color_modes includes "rgb"
{"domain": "light", "service": "turn_on",
 "entity_id": "light.living_room",
 "service_data": {"rgb_color": [180, 30, 90], "brightness": 120}}

# Color temp, warm 153 ⇢ cool 500 mireds (varies by bulb)
{"domain": "light", "service": "turn_on",
 "entity_id": "light.living_room",
 "service_data": {"color_temp": 350}}

# Transition, seconds
{"domain": "light", "service": "turn_on",
 "entity_id": "light.living_room",
 "service_data": {"brightness": 30, "transition": 4}}
```

The Sengled Z-Wave color bulbs (paired through the ADC panel) support
`rgb_color`, `brightness`, `color_temp`, `transition`. Confirm via
`get_state` → `attributes.supported_color_modes` before assuming.

## Switches and plugs

```python
{"domain": "switch", "service": "turn_on",  "entity_id": "switch.basement_fan"}
{"domain": "switch", "service": "turn_off", "entity_id": "switch.basement_fan"}
{"domain": "switch", "service": "toggle",   "entity_id": "switch.basement_fan"}
```

On/off only. If a switch drives something dangerous (a heater, a garage
door wired as a switch), treat "turn it on" with care.

## Scenes and scripts — the canonical mood

A **scene** is a saved entity snapshot ("Movie Night" = TV on, lights
10% warm). A **script** is a procedure ("Goodnight" = dim over 30s,
lock doors, arm alarm).

```python
{"domain": "scene",  "service": "turn_on", "entity_id": "scene.movie_night"}
{"domain": "script", "service": "turn_on", "entity_id": "script.goodnight"}
{"domain": "script", "service": "goodnight"}     # equivalent
```

If the user already defined a mood scene in HA, call it rather than
rebuilding one in code — theirs stays editable in HA's UI.

## Climate

```python
{"domain": "climate", "service": "set_temperature",
 "entity_id": "climate.living_room",
 "service_data": {"temperature": 68}}

{"domain": "climate", "service": "set_hvac_mode",
 "entity_id": "climate.living_room",
 "service_data": {"hvac_mode": "cool"}}     # heat/cool/off/auto/heat_cool

{"domain": "climate", "service": "set_fan_mode",
 "entity_id": "climate.living_room",
 "service_data": {"fan_mode": "auto"}}
```

Check `attributes.hvac_modes` and `attributes.fan_modes` before
guessing valid values.

## Media players

```python
{"domain": "media_player", "service": "media_play",  "entity_id": "media_player.kitchen"}
{"domain": "media_player", "service": "media_pause", "entity_id": "media_player.kitchen"}
{"domain": "media_player", "service": "media_stop",  "entity_id": "media_player.kitchen"}
{"domain": "media_player", "service": "media_next_track", "entity_id": "media_player.kitchen"}

# Volume 0.0–1.0
{"domain": "media_player", "service": "volume_set",
 "entity_id": "media_player.kitchen",
 "service_data": {"volume_level": 0.4}}

{"domain": "media_player", "service": "volume_mute",
 "entity_id": "media_player.kitchen",
 "service_data": {"is_volume_muted": true}}
```

Audio on the box itself is `bb.audio.play`, not HA. HA media players
are external speakers the user paired (Sonos, casts, TVs).

## Fans, humidifiers, vacuums, the rest

Same shape: `domain.turn_on` / `domain.turn_off` plus domain-specific
services. `list_services` and read HA's own descriptions.

## Notifications

```python
{"domain": "notify", "service": "mobile_app_<device>",
 "service_data": {"message": "Front door opened.", "title": "boxBot"}}
```

Service name depends on which companion apps are paired; list `notify.*`
to see. Rarely right for BB — voice and text channels are usually
better — but it exists.

## Blocked in V1

```python
{"domain": "alarm_control_panel", "service": "alarm_arm_home", ...}   # ✗
{"domain": "alarm_control_panel", "service": "alarm_disarm",   ...}   # ✗
{"domain": "lock",  "service": "lock",   ...}                         # ✗
{"domain": "lock",  "service": "unlock", ...}                         # ✗
{"domain": "cover", "service": "open_cover",  ...}                    # ✗
{"domain": "cover", "service": "close_cover", ...}                    # ✗
```

The integration returns an error naming the domain. Rationale and V2
plan: `mutation_policy.md`.
