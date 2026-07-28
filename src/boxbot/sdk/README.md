# sdk/

The boxBot SDK — a constrained, immutable API that agent-written
scripts import to reach boxBot. The **only** interface sandbox scripts
have to the system.

> **API reference lives in [`skills/bb/`](../../../skills/bb/).** That
> is the agent-facing doc: `SKILL.md` for the module index,
> `modules/<name>.md` per module. This README is the developer-facing
> architecture and security contract. Don't duplicate the API here.

## Why an SDK

The agent's primary tool is `execute_script`. Rather than one tool per
operation, the agent writes Python that imports this SDK:

1. **Slim tool list** — 10 always-loaded tools, not 25.
2. **Constrained access** — safe, validated operations only; no path to
   core code.
3. **Composability** — one script tags a photo AND adds it to the
   slideshow AND saves a memory about it.
4. **Immutability** — pre-installed in the sandbox venv; agent scripts
   cannot modify it.
5. **Extensibility** — new capabilities need no new tools.

## Installation

Part of the repo (`src/boxbot/sdk/`) but installed **independently**
into the sandbox venv. No dependency on boxBot internals — stdlib plus
what's already in the sandbox.

```bash
# scripts/setup-sandbox.sh does this:
<runtime_dir>/venv/bin/pip install src/boxbot/sdk/
```

Not editable — bump the SDK version when the public surface changes so
deploys pick it up.

## How it works

Agent scripts call the Python API. The SDK emits structured JSON on
stdout; `execute_script` in the main process reads action lines,
dispatches to per-module handlers, writes JSON replies back, and
collects image attachments into a multimodal tool result.

```
execute_script runs:
┌──────────────────────────────────────────┐
│  from boxbot_sdk import display          │
│  spec = {"name": "weather_board", ...}   │
│  display.save(spec)                      │
└─────────────────┬────────────────────────┘
                  ▼
SDK → stdout:
┌──────────────────────────────────────────┐
│  {"_sdk": "display.save",                │
│   "spec": { ... full spec dict ... }}    │
└─────────────────┬────────────────────────┘
                  ▼
Main process:
┌──────────────────────────────────────────┐
│  Validate spec against block schemas     │
│  Write data/displays/weather_board.json  │
│  Register with the display manager       │
│  Reply on the sandbox's stdin            │
└──────────────────────────────────────────┘
```

The agent never writes render code. Specs are declarative block trees;
the renderer draws from a fixed, validated registry.

## Modules

| Module | Surface |
|---|---|
| `workspace` | Notes, CSVs, images. Path-safe, quota-capped, grep-searchable. |
| `camera` | `capture` / `capture_cropped`. Images attach to the tool result. |
| `photos` | Library search, get, view, ingest, tags, slideshow, soft-delete. |
| `audio` | Play workspace audio through the speaker. Blocks until drained or interrupted. |
| `display` | Spec dicts in, spec dicts out. `list`, `load`, `save`, `preview`, `delete`, `describe_source`, `schema`, `update_data`. No builder. |
| `memory` | `save` / `search` / `invalidate`. Shares the backend with `search_memory`. |
| `tasks` | Triggers + to-dos. Shares the backend with `manage_tasks`. |
| `auth` | Registered users, registration codes, admin notify. |
| `skill` | Create skills at runtime (SKILL.md + resources + scripts). |
| `integrations` | List / call / create / update / delete data pipes; read run logs. |
| `packages` | Request a PyPI install. Human approves out-of-band. |
| `secrets` | Write-only credential store. |
| `integration` | In-script helpers for integration authors: `inputs()`, `return_output()`. |

## Error contract

Mutating calls (`save`, `write`, `delete`, `create`) raise
`bb.ActionError` — or a subclass such as `WorkspaceError`,
`MemoryError`, `AudioError` — when the main process rejects them. A
write that failed never looks like it succeeded.

Read calls return response dicts. A failing *run* (an integration
returning `{"status": "error"}`) is data to inspect, not an exception.

## Security properties

1. **Immutable** — installed in the sandbox venv's site-packages,
   read-only to scripts.
2. **Validated** — every input is schema-checked before an action is
   emitted. Invalid specs are rejected with explicit errors.
3. **Declarative displays** — the agent describes what to show. No raw
   render code ever runs in the main process.
4. **Approval gates** — `packages.request()` only *queues*. The install
   follows out-of-band human approval (admin messaging reply). There is
   no SDK action meaning "approve." Display saves do **not** gate: a
   declarative spec has no executable path to police.
5. **No core access** — cannot import `boxbot.*` (different venv).
   Communicates only through structured JSON actions.
6. **Write-only secrets** — stored values never return through any SDK
   call, only as `BOXBOT_SECRET_<NAME>` env vars in a launched
   subprocess.
7. **Auditable** — every action is logged with timestamp and context,
   and surfaces in the tool result's `sdk_actions`.

## Files

Public modules: `workspace.py`, `camera.py`, `photos.py`, `audio.py`,
`display.py`, `memory.py`, `tasks.py`, `auth.py`, `skill.py`,
`integrations.py`, `packages.py`, `secrets.py`, `integration.py`.

Internal:

- `_transport.py` — structured JSON on stdout, replies on stdin. Not
  public API.
- `_validators.py` — input validation schemas. Every action is
  well-formed before it leaves the sandbox.
