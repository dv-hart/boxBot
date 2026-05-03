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
  schedule (the data-source manager calls integrations).

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

Subprocess invocation does **not** work — seccomp blocks
`execve`/`fork` in the sandbox. Use `httpx`/`requests`, in-process
libraries, and stdlib only.

Secrets declared in the manifest are injected as `BOXBOT_SECRET_<NAME>`
env vars at runtime. Read them with `os.environ.get(...)`.

## Pipe model — no internal schedule

Integrations don't run on their own. They're pure functions from the
consumer's perspective: call them when you want data; cache the
result if you need to. There is no `schedule` or `cron` field in the
manifest.

If you want recurring fetches, register a **trigger** (`bb.tasks`)
that fires on a cadence and calls the integration. If you're a
display, the data-source manager already handles refresh cadence.

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
