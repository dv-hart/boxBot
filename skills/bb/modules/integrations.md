# bb.integrations — data pipes

Integration = manifest+script bundle pulling data from an external
service (or computing outputs from inputs). Stateful (credentials,
OAuth tokens, caches); called by many consumers — you, displays,
briefings. **Skills = nouns you read; integrations = verbs that run.**

Use for: a registered/cached/observable external fetch; anything you'd
build twice; a display source needing a refresh schedule. Not for:
workflow instructions → skill; one-off transform → inline
`execute_script`; notes → `bb.workspace`.

## Reading — list() / get(name, **inputs) / get_source(name) / logs(name, limit)

```python
bb.integrations.list()
bb.integrations.get_source("weather")  # → {"status","manifest","script"}
# → {"status": "ok", "integrations": [
#       {"name", "description", "inputs", "outputs", "secrets",
#        "timeout"}, ...]}

bb.integrations.get("weather", lat=45.5, lon=-122.7, forecast_days=5)
# status "ok"      → {"output": {...}}
#        "error"   → crashed / bad input / missing secret / never called
#                    return_output(); error carries stderr
#        "timeout" → exceeded manifest timeout

bb.integrations.logs("weather", limit=5)
# → {"status": "ok", "runs": [{started_at, finished_at, duration_ms,
#     status, inputs, output|error}, ...]}
```

Branch on `status`. Repeated identical auth errors in `logs` = stale
secret → ask user for a new key, `bb.secrets.store`, retry.

## Authoring — create(name) builder: description/add_input/add_output/add_secret/timeout/script → save()

```python
i = bb.integrations.create("solar")
i.description = (
    "Solar production forecast via Forecast.Solar. Use when the user "
    "asks about solar output or panel performance."
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
# … fetch …
return_output({"kwh": kwh})
'''
i.save()
# → {"status": "ok", "name": "solar", "path": ".../integrations/solar"}
```

On disk: `integrations/<name>/manifest.yaml` + `script.py`,
`boxbot:boxbot` mode `0644` — sandbox reads, cannot modify after save.
Loader rescans on every read call: `save()` is runnable on the next
`get()`, no deploy.

## Update / delete — update(name, script|manifest field-merge) / delete(name)

```python
bb.integrations.update("solar", script="…revised…")        # wholesale
bb.integrations.update("solar", manifest={"timeout": 60})  # FIELD MERGE
bb.integrations.delete("solar")
```

`manifest=` merges per field — omitted fields preserved, a sent field
replaces its whole section (`{"secrets": []}` clears the list). `name`
immutable → delete + recreate. Read `get_source(name)` before patching.
Unknown name → `bb.ActionError`, never auto-creates.

## script.py contract

Same sandbox profile as `execute_script` (separate user, seccomp — no
`execve`/`fork`, read-only site-packages); full `bb.*` available. HTTP:
`httpx`/`requests`/stdlib.

```python
from boxbot_sdk.integration import inputs, return_output
args = inputs()               # runner-passed dict, defaults filled
return_output({...})          # LAST CALL WINS
```

Error-return then keep executing = error silently overwritten. Always:

```python
if not token:
    return_output({"error": "GOOGLE_CALENDAR_TOKEN_JSON not stored"})
    sys.exit(0)   # REQUIRED — later code clobbers the error otherwise
```

Manifest-declared secrets arrive as `BOXBOT_SECRET_<NAME>` env vars —
`os.environ.get(...)`.

### Fired by a trigger

`bb.tasks.create_trigger(..., run_integration="<name>")` runs a script
with no model call ([tasks.md](tasks.md)). Same contract, plus:

- `return_output({"escalate": "<why>"})` = reserved key, wakes me with
  the trigger's `instructions` + your output. The only way a script
  reaches a person — `message` is agent-gated.
- Denied to every integration script: all `tasks.*` (returns
  `status: error`).
- Silent otherwise. Put time windows / hold durations / OR-logic here;
  trigger conditions are a plain AND.

## Rules — timeout ≤300s · parallel calls, fresh subprocess · no cron field · writes raise ActionError

- `timeout` ≤ **300s** (validator rejects more; over → runner kills →
  `status: "timeout"`). Longer work → `bb.tasks` trigger or incremental
  fetch + cache.
- Calls NOT serialized — fresh subprocess each, consumers overlap, no
  per-integration lock. Mutating scripts (OAuth refresh via
  `bb.secrets.store`, counters, caches) must be idempotent under a
  parallel run. Calendar pattern: refresh on 401, persist rotated
  token, retry once — safe because the refresh token stays valid; last
  write wins.
- No internal schedule — no `cron` field; from the consumer's side a
  pure function. Recurring fetch → `bb.tasks` trigger. Display →
  `{"type": "integration", "refresh": N}` source; the manager calls
  `get(<name>, **inputs)` per tick (spec: [display.md](display.md)).
  Back-compat: `{"type": "builtin"}` with any name outside the true
  built-in sources (`clock`, `tasks`, `people`, `agent_status`)
  resolves to the same-name integration (e.g. weather, calendar).
- Writes (`save`/`update`/`delete`) raise `bb.ActionError` — name taken
  (save), not registered (update/delete → use create), manifest invalid
  (non-lowercase name, secret not `SCREAMING_SNAKE_CASE`, timeout >
  300). Reads return `status` instead.

## Built-ins and their setup

**calendar** — Google Calendar v3.
Secret `GOOGLE_CALENDAR_TOKEN_JSON`: OAuth token JSON from
`scripts/calendar_auth.py` (installed-app flow; shape of
`google.oauth2.credentials.Credentials.to_json()`; must include
`refresh_token`, `client_id`, `client_secret`). Auto-refreshes on 401,
persists the rotated token.
Actions: `list_upcoming_events`, `create_event`, `update_event`,
`delete_event`.
`bb.integrations.get("calendar", action="list_upcoming_events", max_results=5)`

**home_assistant** — HA REST API.
Secrets `HOME_ASSISTANT_URL` (e.g. `http://homeassistant.local:8123`)
and `HOME_ASSISTANT_TOKEN` (long-lived token, HA profile page).
Actions: `get_states`, `get_state`, `call_service`, `camera_snapshot`,
`list_services`.
`bb.integrations.get("home_assistant", action="get_state", entity_id="light.living_room")`

**weather** — NOAA (api.weather.gov), US lat/lon only. No secrets.
`lat`/`lon` required but fall back to `BOXBOT_WEATHER_LAT` /
`BOXBOT_WEATHER_LON` via `default_env`.
`bb.integrations.get("weather", forecast_days=5)`

## Device config: `default_env`

Per-device inputs (location, zip, kW capacity) → declare `default_env`;
the runner reads that env var when the caller omits the input. Resolved
in the main process pre-sandbox, so non-secret env skips the sandbox
safe-env allowlist. Actually-secret values → `bb.secrets`.

```yaml
inputs:
  lat:
    type: float
    required: true
    default_env: BOXBOT_WEATHER_LAT
```
