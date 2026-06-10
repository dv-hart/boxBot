# bb.integrations — list, call, and author data-pipe integrations

An **integration** is a manifest+script bundle that pulls data from
an external service or computes outputs from declared inputs. Unlike
skills, integrations are stateful (credentials, OAuth tokens, caches
they may maintain themselves) and are designed to be called by many
consumers (the agent, displays, scheduled briefings).

For the conceptual difference, see
[skills/skill_authoring/SKILL.md](../../skill_authoring/SKILL.md). One-liner:
**skills are nouns the agent reads; integrations are verbs that run
when called.**

## When to use it

- The user asks for fresh data from an external service (weather,
  stocks, RSS, calendar, etc.) and you want a registered, cached,
  observable pipe rather than a one-off `requests` call.
- You're about to build the same fetcher twice — make it an
  integration so future-you and other consumers can reuse it.
- A display data source needs server-side data on a refresh
  schedule. Wire it in as ``{"type": "integration", "inputs": {...}}``
  in the display spec — same path whether the integration was
  pre-seeded or you just authored it.

## When NOT to use it

- You need step-by-step instructions for a workflow → that's a
  **skill**.
- You need a one-off transformation in this conversation → just do
  it inline in `execute_script`.
- You need persistent agent-owned notes → `bb.workspace`.

## API

### Read

```python
import boxbot_sdk as bb

# Discover what's registered
bb.integrations.list()
# → {"status": "ok", "integrations": [
#       {"name": "weather", "description": "...",
#        "inputs": {...}, "outputs": {...},
#        "secrets": [...], "timeout": 20},
#       ...
#    ]}

# Call one
bb.integrations.get("weather", lat=45.5, lon=-122.7, forecast_days=5)
# → {"status": "ok", "output": {"temp": "62", "condition": "Cloudy", ...}}

# Failure shape
# → {"status": "error",   "error": "..."}      # script crashed / bad input
# → {"status": "timeout", "error": "..."}      # exceeded manifest timeout

# Inspect prior runs (debugging)
bb.integrations.logs("weather", limit=5)
# → {"status": "ok", "runs": [
#       {"started_at": ..., "finished_at": ..., "duration_ms": 312,
#        "status": "ok", "inputs": {...}, "output": {...}},
#       {"status": "error", "error": "401 Unauthorized", ...},
#       ...
#    ]}
```

`logs` is the load-bearing self-debugging primitive. Five consecutive
auth errors usually mean a secret needs refreshing — read the logs,
ask the user for a new key, store it, retry.

### Write — author a new integration

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
# → {"status": "exists", ...}  if a "solar" integration already exists
```

### Update / delete

```python
bb.integrations.update("solar", script="…revised script…")
bb.integrations.update("solar", manifest={"description": "…"})
# Errors with status:"missing" if the name doesn't exist — never
# auto-promotes. Use create() if you need a new one.

bb.integrations.delete("solar")
# Errors with status:"missing" if the name doesn't exist.
```

## What lands on disk

For the example above:

```
integrations/solar/
  manifest.yaml      # name, description, inputs, outputs, secrets, timeout
  script.py          # your script
```

Files are owned `boxbot:boxbot` mode `0644`. The sandbox can read
but not modify them after save.

## Inside an integration's script.py

The script runs in the sandbox with the same security profile as
`execute_script`: separate user, seccomp filter, read-only
site-packages, full `bb.*` SDK available.

Two helpers in `boxbot_sdk.integration`:

```python
from boxbot_sdk.integration import inputs, return_output

args = inputs()             # dict the runner passed in (after default fill)
return_output({"...": ...}) # set this call's result; last call wins
```

**`return_output()` is last-call-wins.** If you return an error early
and keep executing, a later `return_output({...})` silently overwrites
the error. Always exit immediately after an early error return:

```python
if not token:
    return_output({"error": "GOOGLE_CALENDAR_TOKEN_JSON not stored"})
    sys.exit(0)   # REQUIRED — without this, later code can clobber the error
```

Subprocess invocation does **not** work — seccomp blocks
`execve`/`fork` in the sandbox. Use `httpx`/`requests`, in-process
libraries, and stdlib only.

Secrets declared in the manifest are injected as `BOXBOT_SECRET_<NAME>`
env vars at runtime. Read them with `os.environ.get(...)`.

## Timeouts — 300s hard ceiling

The manifest's `timeout` field caps how long one call may run before
the runner kills the subprocess (`status: "timeout"`). The validator
rejects anything over **300 seconds (5 minutes)** — longer-running
work belongs in a scheduled trigger, not an integration call.
Integrations are request/response pipes; if a job legitimately needs
more than 5 minutes, register a trigger (`bb.tasks`) that wakes you to
do the work in stages, or restructure the integration to fetch
incrementally and cache.

## Concurrency — calls are NOT serialized

Every call spawns a fresh sandbox subprocess. Two consumers calling
the same integration at the same time (e.g. the agent plus a display
refresh) run **concurrently** — there is no per-integration lock.
Write your script so concurrent runs can't corrupt shared state:

- Pure reads (weather, stock quotes) are naturally safe.
- Anything that **mutates state** — OAuth token refresh that writes
  back via `bb.secrets.store(...)` (the calendar integration does
  this), counters, caches — must be idempotent and tolerate another
  run doing the same mutation in parallel. The calendar pattern:
  refresh on 401, persist the rotated token, retry once. Safe under
  concurrency because the long-lived refresh token stays valid when
  two runs refresh at once — last write wins, both runs succeed.

## Pipe model — no internal schedule

Integrations don't run on their own. They're pure functions from the
consumer's perspective: call them when you want data; cache the
result if you need to. There is no `schedule` or `cron` field in the
manifest.

If you want recurring fetches, register a **trigger** (`bb.tasks`)
that fires on a cadence and calls the integration. If you're a
display, declare it as an ``integration`` data source and the
data-source manager handles refresh cadence:

```json
{"name": "solar", "type": "integration",
 "inputs": {"date": "2026-05-15"}, "refresh": 3600}
```

The manager calls ``bb.integrations.get(<name>, **inputs)`` on each
tick and binds the output dict to the source name. See
[display.md](display.md) for the full spec form.

Backward compat: an old display source spec of
``{"type": "builtin", "name": "weather"}`` (or ``"calendar"``, or any
name that isn't clock/tasks/people/agent_status) resolves to the
integration of the same name — weather and calendar *are*
integrations now, not built-ins.

## Built-in integrations — setup

Three integrations ship with boxBot. What each needs before it works:

**calendar** — Google Calendar v3.
- Secret: `GOOGLE_CALENDAR_TOKEN_JSON` — the OAuth token JSON produced
  by `scripts/calendar_auth.py` (the installed-app flow; same shape as
  `google.oauth2.credentials.Credentials.to_json()`, must include
  `refresh_token`, `client_id`, `client_secret`). The script
  auto-refreshes on 401 and persists the rotated token back to the
  secret store.
- Actions: `list_upcoming_events`, `create_event`, `update_event`,
  `delete_event`.
- Example:
  `bb.integrations.get("calendar", action="list_upcoming_events", max_results=5)`

**home_assistant** — Home Assistant REST API.
- Secrets: `HOME_ASSISTANT_URL` (e.g. `http://homeassistant.local:8123`)
  and `HOME_ASSISTANT_TOKEN` (a long-lived access token from the HA
  user profile page).
- Actions: `get_states`, `get_state`, `call_service`,
  `camera_snapshot`, `list_services`.
- Example:
  `bb.integrations.get("home_assistant", action="get_state", entity_id="light.living_room")`

**weather** — NOAA forecasts (api.weather.gov), US lat/lon only.
- No secrets. `lat`/`lon` are required but fall back to the
  `BOXBOT_WEATHER_LAT` / `BOXBOT_WEATHER_LON` env vars via
  `default_env`, so on a configured device you can omit them.
- Example: `bb.integrations.get("weather", forecast_days=5)`

## Device-level config: ``default_env``

For inputs that are per-device rather than per-call (location, home
zip code, household kW capacity), declare a ``default_env`` on the
input. The runner reads ``os.environ[default_env]`` when the caller
didn't supply a value, so every consumer — agent, display, scheduled
trigger — picks up the same device default without threading it
through each call site.

```yaml
inputs:
  lat:
    type: float
    required: true
    default_env: BOXBOT_WEATHER_LAT
    description: Latitude. Falls back to BOXBOT_WEATHER_LAT env var.
```

The env var is read in the main process at validation time, before
the sandbox spawn — so non-secret env (lat/lon, household ids) flows
through without being added to the sandbox's safe-env allowlist.
Use ``bb.secrets`` for actually-secret values; this is for
configuration that's neither secret nor per-call.

## Conflict and lifecycle

- States: **registered** or not. There is no active/paused/scheduled.
- `create` refuses if the name is taken (`status: "exists"`); never
  silently overwrites a community integration.
- `update` errors with `status: "missing"` if the name is unknown.
- `delete` errors with `status: "missing"` similarly.

## Discovery and timing

The loader scans `integrations/` at startup and on every read call,
so a freshly-saved integration is reachable from
`bb.integrations.list()` and `bb.integrations.get()` immediately —
no restart needed.
