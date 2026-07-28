# bb.integrations — data pipes

An integration is a manifest+script bundle that pulls data from an
external service or computes outputs from declared inputs. Stateful
(credentials, OAuth tokens, its own caches), called by many consumers:
you, displays, scheduled briefings.

**Skills are nouns you read. Integrations are verbs that run when
called.** Longer version:
[skills/skill_authoring/SKILL.md](../../skill_authoring/SKILL.md).

**Use for:** fresh data from an external service when you want a
registered, cached, observable pipe instead of a one-off `requests`
call; anything you're about to build twice; a display data source that
needs a refresh schedule.

**Not for:** step-by-step workflow instructions → a **skill**. A
one-off transformation → inline in `execute_script`. Persistent notes →
`bb.workspace`.

## Read

```python
import boxbot_sdk as bb

bb.integrations.list()
# → {"status": "ok", "integrations": [
#       {"name": "weather", "description": "...", "inputs": {...},
#        "outputs": {...}, "secrets": [...], "timeout": 20}, ...]}

bb.integrations.get("weather", lat=45.5, lon=-122.7, forecast_days=5)
# → {"status": "ok",      "output": {"temp": "62", ...}}
# → {"status": "error",   "error": "..."}   # crashed / bad input
# → {"status": "timeout", "error": "..."}   # exceeded manifest timeout

bb.integrations.logs("weather", limit=5)
# → {"status": "ok", "runs": [
#       {"started_at": …, "finished_at": …, "duration_ms": 312,
#        "status": "ok", "inputs": {...}, "output": {...}},
#       {"status": "error", "error": "401 Unauthorized", ...}, ...]}
```

`logs` is the self-debugging primitive. Five consecutive auth errors
usually means a secret needs refreshing — read the logs, ask the user
for a new key, store it, retry.

## Author

```python
i = bb.integrations.create("solar")
i.description = (
    "Solar production forecast for the household array via Forecast.Solar. "
    "Use when the user asks about solar output, power generation, or panel performance."
)
i.add_input("date", type="string", required=True,
            description="ISO date — the day to forecast.")
i.add_output("kwh", type="float", description="Estimated kWh for the day.")
i.add_secret("FORECAST_SOLAR_API_KEY")
i.timeout = 20
i.script = '''
from boxbot_sdk.integration import inputs, return_output
import os, httpx

api_key = os.environ.get("BOXBOT_SECRET_FORECAST_SOLAR_API_KEY", "")
date = inputs()["date"]
# … fetch from Forecast.Solar API …
return_output({"kwh": kwh})
'''
i.save()
# → {"status": "ok", "name": "solar", "path": ".../integrations/solar"}
```

On disk: `integrations/solar/manifest.yaml` + `script.py`, owned
`boxbot:boxbot` mode `0644`. The sandbox reads but cannot modify them
after save.

## Update and delete

```python
bb.integrations.update("solar", script="…revised script…")   # wholesale
bb.integrations.update("solar", manifest={"timeout": 60})    # field merge
bb.integrations.delete("solar")
```

`update(manifest=…)` is a **field-level merge**, not a replace. Omitted
fields are preserved; a sent field replaces that whole section
(`{"secrets": []}` clears the list). `name` is immutable — delete and
recreate to rename. `get_source(name)` shows the current manifest
before you patch.

`update` / `delete` raise `bb.ActionError` on an unknown name; they
never auto-promote to a create.

## Inside script.py

Same security profile as `execute_script`: separate user, seccomp,
read-only site-packages, full `bb.*` available.

```python
from boxbot_sdk.integration import inputs, return_output

args = inputs()              # dict the runner passed in, defaults filled
return_output({"...": ...})  # this call's result; LAST CALL WINS
```

**`return_output()` is last-call-wins.** An early error return that
keeps executing gets silently overwritten. Always exit after one:

```python
if not token:
    return_output({"error": "GOOGLE_CALENDAR_TOKEN_JSON not stored"})
    sys.exit(0)   # REQUIRED — without this, later code clobbers the error
```

No subprocesses — seccomp blocks `execve`/`fork`. Use
`httpx`/`requests`, in-process libraries, stdlib.

Manifest-declared secrets arrive as `BOXBOT_SECRET_<NAME>` env vars.
Read with `os.environ.get(...)`.

## Timeouts — 300s ceiling

`timeout` caps one call before the runner kills the subprocess
(`status: "timeout"`). The validator rejects anything over **300
seconds**. Work that legitimately needs longer belongs in a `bb.tasks`
trigger that stages it, or in an integration that fetches
incrementally and caches.

## Concurrency — calls are NOT serialized

Every call spawns a fresh subprocess. Two consumers (you plus a display
refresh) run concurrently. No per-integration lock.

Pure reads (weather, quotes) are naturally safe. Anything that
**mutates state** — OAuth refresh writing back via
`bb.secrets.store(...)`, counters, caches — must be idempotent and
survive a parallel run doing the same thing. The calendar pattern:
refresh on 401, persist the rotated token, retry once. Safe because the
long-lived refresh token stays valid when two runs refresh at once —
last write wins, both succeed.

## No internal schedule

Integrations never run on their own. From the consumer's side they are
pure functions: call, cache if you want. There is no `schedule` or
`cron` field.

Recurring fetch → a `bb.tasks` trigger that calls it. Display →
declare an `integration` data source and the data-source manager
handles cadence:

```json
{"name": "solar", "type": "integration",
 "inputs": {"date": "2026-05-15"}, "refresh": 3600}
```

The manager calls `bb.integrations.get(<name>, **inputs)` per tick and
binds the output dict to the source name. Full spec:
[display.md](display.md).

Back-compat: an old `{"type": "builtin", "name": "weather"}` source (or
`"calendar"`, or any name that isn't clock/tasks/people/agent_status)
resolves to the integration of the same name. Weather and calendar
*are* integrations now.

## Built-ins and their setup

**calendar** — Google Calendar v3.
Secret `GOOGLE_CALENDAR_TOKEN_JSON`: the OAuth token JSON from
`scripts/calendar_auth.py` (installed-app flow; same shape as
`google.oauth2.credentials.Credentials.to_json()`; must include
`refresh_token`, `client_id`, `client_secret`). Auto-refreshes on 401
and persists the rotated token.
Actions: `list_upcoming_events`, `create_event`, `update_event`,
`delete_event`.
`bb.integrations.get("calendar", action="list_upcoming_events", max_results=5)`

**home_assistant** — HA REST API.
Secrets `HOME_ASSISTANT_URL` (e.g. `http://homeassistant.local:8123`)
and `HOME_ASSISTANT_TOKEN` (long-lived token from the HA profile page).
Actions: `get_states`, `get_state`, `call_service`, `camera_snapshot`,
`list_services`.
`bb.integrations.get("home_assistant", action="get_state", entity_id="light.living_room")`

**weather** — NOAA (api.weather.gov), US lat/lon only.
No secrets. `lat`/`lon` required but fall back to
`BOXBOT_WEATHER_LAT` / `BOXBOT_WEATHER_LON` via `default_env`, so a
configured device can omit them.
`bb.integrations.get("weather", forecast_days=5)`

## Device config: `default_env`

For per-device inputs (location, zip, household kW capacity), declare
`default_env`. The runner reads that env var when the caller supplied
nothing, so every consumer picks up the same default without threading
it through call sites.

```yaml
inputs:
  lat:
    type: float
    required: true
    default_env: BOXBOT_WEATHER_LAT
    description: Latitude. Falls back to BOXBOT_WEATHER_LAT env var.
```

Read in the main process at validation time, before the sandbox spawn —
so non-secret env flows through without touching the sandbox's safe-env
allowlist. Actually-secret values go in `bb.secrets`.

## Lifecycle and discovery

States: **registered** or not. No active/paused/scheduled.

The loader scans `integrations/` at startup and on every read call. An
integration you `create().save()` is runnable on the very next `get()`
— no deploy, no setup re-run. Same for `update()`.

## Failure modes

Reads return a `status`; branch on it.

- `ok` — script ran, returned a value.
- `error` — script crashed, exited non-zero, or never called
  `return_output()`. `error` carries stderr. Usually a bug, a
  missing/expired secret (often a `401`/`403`), or a bad input. Read
  `logs(name)`: repeated identical auth errors mean a secret needs
  refreshing.
- `timeout` — ran past the manifest `timeout`. Make it faster, raise
  the timeout via `update(name, manifest={"timeout": …})` (300s
  ceiling), or move the work to a scheduled trigger.

Writes (`save`, `update`, `delete`) **raise `bb.ActionError`** instead
of returning a status. The message says why: name taken (`save`), name
not registered (`update`/`delete` — use `create`), or manifest
validation failed (non-lowercase name, a secret that isn't
`SCREAMING_SNAKE_CASE`, `timeout` over 300).

Cannot: rename in place (delete + recreate), schedule from inside the
manifest (use a `bb.tasks` trigger), spawn subprocesses (seccomp).
