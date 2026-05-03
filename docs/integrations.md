# Integrations

Integrations are sandbox-runnable data pipes: a manifest plus a
Python script that turns inputs into outputs. They live next to
skills as a peer extension system — but where skills are **structured
prompt data** the agent reads, integrations are **callable functions**
that fetch data from external services or compute something on
demand.

## Skill vs. integration

| | **Skill** | **Integration** |
|---|---|---|
| Purpose | Teach BB *how* to do something | *Provide* data or access to a service |
| Form | SKILL.md + optional bundled scripts | Python script + manifest declaring inputs/outputs/secrets/timeout |
| Stateful? | No | Yes (creds, OAuth tokens, caches the script chooses to maintain) |
| Runs on its own? | Never | Only when called |
| Consumers | One agent in one conversation | Many: agent, displays, scheduled briefings, future triggers |
| Lives at | `skills/<name>/` | `integrations/<name>/` |
| Loads via | `load_skill` (Level 2 on demand) | Always discoverable via `bb.integrations.list()` |

Mnemonic: **skills are nouns the agent reads; integrations are verbs
that run when called.**

## The pipe model — no internal schedule

Integrations don't run on their own. Every call spawns a fresh
sandbox subprocess, runs the script, captures the output, and
returns. The integration manifest does **not** declare a `cron` or
`schedule`.

This is deliberate. Different consumers want different freshness:
the display might want weather every 30 min, the agent during a
conversation wants fresh-now, a morning briefing wants one fetch at
7am. Putting the schedule on the integration would force one
cadence on all consumers. Letting consumers decide keeps it simple.

If you want recurring fetches, register a **trigger** (`bb.tasks`)
that fires on a cadence and calls the integration. If you're a
display, the data-source manager already handles refresh cadence.

## On-disk layout

```
integrations/<name>/
  manifest.yaml      # contract: name, description, inputs, outputs, secrets, timeout
  script.py          # sandbox-runnable; reads inputs(), calls return_output()
```

Files are owned `boxbot:boxbot` mode `0644`. The sandbox user can
read but not modify them after save, so a later sandbox script
can't tamper with what an earlier one wrote.

### Manifest schema

```yaml
name: weather                          # ≤64 chars, lowercase, [a-z0-9_-]+
description: >-                        # ≤1024 chars, no XML
  Get NOAA weather forecasts for a US lat/lon. Returns current
  conditions and a multi-day outlook with display-ready icon names.
inputs:                                # informational; runner applies defaults + checks required
  lat:
    type: float
    required: true
  lon:
    type: float
    required: true
  forecast_days:
    type: int
    default: 5
outputs:                               # informational; the script's return value, declared
  temp: {type: string}
  forecast: {type: array}
secrets:                               # SCREAMING_SNAKE_CASE; injected as BOXBOT_SECRET_<NAME>
  - WEATHER_API_KEY
timeout: 20                            # seconds; runner kills the script after this
```

Inputs are applied with defaults and missing required fields are
rejected before the script runs. Outputs are descriptive in v1 — no
runtime schema enforcement.

The runner refuses timeouts above 5 minutes — anything that needs
longer is background work that belongs in a scheduled trigger, not
an interactive integration call.

## Lifecycle — two states only

An integration is **registered** (its directory exists with both
files) or it isn't. There's no active/paused/scheduled state, no
health flag, no version pin.

CRUD via the SDK:
- `bb.integrations.create(name)` → builder → `save()`. Refuses if the
  name is taken (`status: "exists"`).
- `bb.integrations.update(name, manifest=…, script=…)`. Errors with
  `status: "missing"` if the name isn't registered. Never auto-creates.
- `bb.integrations.delete(name)`. Errors with `status: "missing"` if
  the name isn't there.
- `bb.integrations.list()` → metadata for everything registered.
- `bb.integrations.get(name, **inputs)` → executes and returns the output.
- `bb.integrations.logs(name, limit=20)` → recent runs for self-debugging.

See [skills/bb/modules/integrations.md](../skills/bb/modules/integrations.md)
for the full SDK reference.

## Runner protocol

The runner (`src/boxbot/integrations/runner.py`) is the single
entrypoint. It validates inputs against the manifest, spawns a
sandbox subprocess running the script, demuxes any `bb.*` SDK
actions the script makes, captures the script's structured output,
and logs the run.

The script reads inputs and returns output through two helpers:

```python
from boxbot_sdk.integration import inputs, return_output

args = inputs()                    # dict the runner provided
return_output({"value": ...})      # the script's result; last call wins
```

Behind the scenes:
- `BOXBOT_INTEGRATION_INPUTS_PATH` — file the runner wrote with the
  validated inputs as JSON.
- `BOXBOT_INTEGRATION_OUTPUT_PATH` — file `return_output` writes to.
  The runner reads it after the script exits.

Output is filed rather than streamed because the script's stdout is
already used by the `__BOXBOT_SDK_ACTION__:` markers from any
`bb.*` calls the script makes (e.g. `bb.workspace.read`).

## Failure modes

Every call comes back with one of:
- `{"status": "ok", "output": ...}` — script exited 0, returned a value.
- `{"status": "error", "error": ..., "exit_code": ...}` — script
  exited non-zero, didn't call `return_output`, returned invalid
  JSON, or raised before completing.
- `{"status": "timeout", "error": ...}` — script exceeded the
  manifest's timeout. The runner SIGKILLed it.

The dispatcher returns the same shape from `bb.integrations.get`.
Display data sources, scheduler triggers, and any other consumer
should branch on `status` and fall back gracefully (e.g. the
WeatherSource returns a placeholder dict).

## Logs

Every invocation is recorded in `data/integrations/runs.db` (SQLite),
regardless of outcome. Schema:

```
runs(id, name, started_at, finished_at, duration_ms,
     status, inputs_json, output_json, error)
```

Retention: last **100 runs per integration**, pruned on every
insert. Disk impact is negligible (≤~5MB total across all
integrations) but the window is long enough to span a typical
debugging session.

The agent reads the log via `bb.integrations.logs(name)` to
self-debug failures: five consecutive auth errors usually means a
secret needs refreshing.

## Permissions

- `integrations/` directory: group-writable to `boxbot`, group-readable
  to `boxbot-sandbox` (sandbox user is in the `boxbot` group).
- Files inside: `boxbot:boxbot` mode `0644`. Sandbox can read, not
  modify.
- The runner spawns the sandbox subprocess via the same path
  `execute_script` uses: `sudo -u boxbot-sandbox` + the seccomp
  filter from `scripts/sandbox_bootstrap.py`.
- Secrets declared in the manifest are injected as `BOXBOT_SECRET_<NAME>`
  env vars at script-launch time.

## Built-in integrations

| Name | Description |
|---|---|
| `weather` | NOAA `api.weather.gov` forecasts for a US lat/lon. No API key required. |
| `calendar` | Google Calendar v3 — `list_upcoming_events`, `create_event`, `update_event`, `delete_event`. Reads OAuth token from secret `GOOGLE_CALENDAR_TOKEN_JSON`; auto-refreshes and persists the rotated token back. |

The display system's `WeatherSource` calls `weather` via the runner;
`CalendarSource` calls `calendar` the same way.

## Calendar migration runbook (one-time, per device)

The calendar surface lives in `integrations/calendar/`, backed by an
OAuth token stored as the secret `GOOGLE_CALENDAR_TOKEN_JSON`. There
is no `bb.calendar`, no `src/boxbot/integrations/google_calendar.py`,
no calendar-specific dispatcher path — it's just a regular integration.

Both helper scripts below need the project venv (`anthropic` and
`google-auth-oauthlib` aren't available to the system Python). Use
`.venv/bin/python3` everywhere, not bare `python3`.

### Path A — You already have a working `data/credentials/google_calendar_token.json`

```bash
ssh "$BOXBOT_DEPLOY_TARGET"
cd software/boxBot
.venv/bin/python3 scripts/migrate_calendar_secret.py
# Reads the existing token file, stores it as GOOGLE_CALENDAR_TOKEN_JSON,
# renames the source to .migrated. Idempotent — safe to re-run.
scripts/restart-boxbot.sh
```

If Google then comes back with `invalid_grant` (the migrated refresh
token expired or was revoked), fall through to Path C.

### Path B — Local desktop with a browser

```bash
.venv/bin/python3 scripts/calendar_auth.py
# Opens a browser, runs a local server on port 8765, completes the flow.
```

### Path C — Headless device (Pi over SSH, no keyboard/mouse)

Two-phase flow. Phase 1 runs on the device, prints the consent URL,
saves OAuth state (including the PKCE code verifier) to
`/tmp/boxbot_oauth_state.json` mode 0600. Phase 2 takes the redirect
URL and completes the flow — both phases are non-interactive.

```bash
# Phase 1 — print the auth URL
ssh "$BOXBOT_DEPLOY_TARGET" \
  'cd software/boxBot && .venv/bin/python3 scripts/calendar_auth.py --print-url'
```

Open the printed URL in any browser, sign in, grant calendar access.
The "App not verified" warning appears for testing-mode OAuth clients
— click **Advanced → Go to <project> (unsafe)**. After consent the
browser redirects to a `localhost:8765/?state=…&code=…` URL that
fails to load — that's expected. Copy the entire redirect URL.

```bash
# Phase 2 — complete the flow
ssh "$BOXBOT_DEPLOY_TARGET" \
  "cd software/boxBot && .venv/bin/python3 scripts/calendar_auth.py \
   --redirect-url '<paste full URL here>'"
# → Stored GOOGLE_CALENDAR_TOKEN_JSON in the secret store (replaced).
# → Calendar integration is ready.
```

Phase 2 cleans up `/tmp/boxbot_oauth_state.json` on success. Both
phases must run within the same boot — `/tmp` is wiped on reboot.

### After any path

The `data/credentials/google_client_secrets.json` file stays on disk:
only the runtime-rotating token moved into the secret store; the
static client credentials are still needed for any future
`calendar_auth.py` run.

To verify end-to-end:

```bash
ssh "$BOXBOT_DEPLOY_TARGET" \
  'cd software/boxBot && .venv/bin/python3 -c "
import asyncio
from boxbot.integrations.runner import run
print(asyncio.run(run(\"calendar\", {\"action\": \"list_upcoming_events\", \"max_results\": 3})))
"'
```

Expected: `{"status": "ok", "output": {"events": [...], "count": N}}`.

### When the refresh token expires

Google revokes refresh tokens for OAuth clients in **testing** status
after seven days of inactivity (sometimes sooner). If
`bb.integrations.get("calendar", ...)` starts returning
`{"output": {"error": "calendar API 400: ... invalid_grant ..."}}`,
re-run Path C.

The durable fix is to publish the OAuth client (Google Cloud Console
→ APIs & Services → OAuth consent screen → **Publish app**). For
personal/internal use this requires no review and changes the refresh
token lifetime to ~6 months of inactivity.

## Open follow-ups

These are explicitly deferred to keep the v1 scope contained:

- **Output deduplication.** If three consumers call the same
  integration in the same minute, that's three network hits. Acceptable
  today; revisit with a per-call `max_age_seconds` parameter if it
  becomes a problem.
- **Output schema versioning.** Manifest declares output shape
  descriptively; no version pin yet. Consumers couple to whatever
  shape they read.
- **Schedule wiring.** Triggers (`bb.tasks`) can call
  `bb.integrations.get` from inside a trigger script today, but
  there's no first-class "every 30 min, refresh integration X"
  primitive. Add when there's a concrete need.
