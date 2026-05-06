---
name: hal-sandbox-ref
description: Using the boxbot_sdk from sandboxed scripts — display, memory, photos, secrets, tasks, skills, packages.
when_to_use: Any time you're writing a script via execute_script that needs to interact with boxBot beyond pure computation — displays, memory, photos, secrets, scheduling, skill authoring, or package installs.
---

# boxBot SDK reference (sandbox)

The sandbox ships with an immutable, preinstalled `boxbot_sdk` package. It
is the **only** way a script invoked via `execute_script` can affect the
boxBot system. Core imports from `boxbot.*` are not available in the
sandbox venv — you cannot reach into internals, only declare intent via
the SDK.

Under the hood, the SDK does not act directly. It emits structured JSON
actions on stdout. The `execute_script` tool in the main process parses
these, validates them, and applies them safely. You don't need to think
about the transport — just import and call.

For the full reference (all builder methods, validator rules, long
examples), see `src/boxbot/sdk/README.md` in the repo.

## Modules at a glance

| Module                | What it does                                                |
| --------------------- | ----------------------------------------------------------- |
| `boxbot_sdk.display`  | Declarative display builder (blocks + data binding)         |
| `boxbot_sdk.skill`    | Create new skills (auto-activate, sandboxed)                |
| `boxbot_sdk.packages` | Request package install (blocks on out-of-band approval)    |
| `boxbot_sdk.memory`   | Save / search / delete memories                             |
| `boxbot_sdk.secrets`  | Write-only credential store (you can't read back values)    |
| `boxbot_sdk.photos`   | Photo library: search, tag, slideshow, soft-delete          |
| `boxbot_sdk.tasks`    | Triggers (wake conditions) and the to-do list               |

## display — declarative displays

Compose displays from blocks. You describe **what** to show; the main
process renders. Never write raw render code.

```python
from boxbot_sdk import display

d = display.create("weather_board")
d.set_theme("boxbot")
d.data("weather")                          # built-in data source

header = d.row(padding=24, align="center")
header.icon("{weather.icon}", size="xl")
header.text("{weather.temp}°F", size="title")

d.preview()   # renders to a PNG; view it multimodally and iterate
d.save()      # queues for user approval
```

Layout containers: `row`, `column`/`stack`, `columns`, `card`, `spacer`,
`divider`, `repeat`. Content blocks include `text`, `metric`, `icon`,
`image`, `chart`, `clock`, `countdown`. Data binding uses `{source.field}`
placeholders. For the block reference and themes, load
`docs/display-system.md` or ask for `display-authoring` guidance.

Displays require **user confirmation** before they go live.

## memory — persistent facts

```python
from boxbot_sdk import memory

mem_id = memory.save(
    content="Jacob is allergic to peanuts",
    memory_type="person",       # person | household | methodology | operational
    person="Jacob",
    summary="Jacob — peanut allergy",
    tags=["health", "allergy"],
)

hits = memory.search("allergies", people=["Jacob"])
for m in hits:
    print(m.content)
```

Shares the backend with the `search_memory` tool. `memory_type` defaults
to `household` if omitted; `summary` is auto-derived from `content` when
not provided. Writes raise `MemoryError` if the main process rejects the
call — a successful return means the row is on disk.

## photos — library + slideshow

```python
from boxbot_sdk import photos

results = photos.search(query="beach sunset", people=["Jacob"], limit=10)
photos.set_tags("photo_123", tags=["family", "beach"])
photos.add_to_slideshow("photo_123")
photos.delete("photo_456")   # soft delete, 30-day restore window
```

Shares the backend with the `search_photos` tool. Photos do **not** write
to person embedding clouds — matching is read-only.

## secrets — write-only credential store

```python
from boxbot_sdk import secrets

secrets.store("GMAIL_APP_PASSWORD", value=user_provided)
# Later, in a skill script:
pw = secrets.use("GMAIL_APP_PASSWORD")   # injected into env for this call only
```

You cannot read a secret back as plain text after storage. Use it via
`secrets.use()` which scopes the value to a single call.

## tasks — triggers and to-dos

Two subsystems. **Triggers** wake the agent when conditions are met (AND
logic across all specified conditions). **To-dos** are persistent action
items the agent tracks.

```python
from boxbot_sdk import tasks

# Timer + person (AND): both must fire before the agent wakes
trigger_id = tasks.create_trigger(
    description="Remind Jacob the Amazon package arrived",
    instructions="Remind Jacob the Amazon package arrived",
    fire_after="30m",
    person="Jacob",
    for_person="Jacob",
)

todo_id = tasks.create_todo(
    description="Return library books",
    notes="Books on kitchen counter. Due Saturday.",
    for_person="Jacob",
    due_date="2026-02-22",
)

for t in tasks.list_triggers(status="active"):
    print(t.description)
```

Trigger kinds: `fire_at` (ISO timestamp), `fire_after` (duration, max 24h),
`cron` (recurring), `person` (on detection). Combine for compound wake
conditions. `create_trigger` and `create_todo` return the new id; writes
raise `RuntimeError` if the main process rejects them.

## skill — author new skills

```python
from boxbot_sdk import skill

s = skill.create("check_gmail")
s.description = "Check for unread emails and return summaries"
s.add_parameter("max_results", type="integer", default=10)
s.add_env_var("GMAIL_USER", secret=True)
s.add_env_var("GMAIL_APP_PASSWORD", secret=True)
s.set_script("""
import imaplib, os, json
mail = imaplib.IMAP4_SSL("imap.gmail.com")
mail.login(os.environ["GMAIL_USER"], os.environ["GMAIL_APP_PASSWORD"])
# ...
""")
s.save()   # auto-activates — skill logic runs sandboxed
```

Skills auto-activate because their runtime lives in the sandbox. (Displays
do not — they run in the main process and require approval.)

## packages — request installs

```python
from boxbot_sdk import packages

result = packages.request(
    "google-api-python-client",
    reason="Needed for Gmail integration",
)
if result.approved:
    import googleapiclient
else:
    print(f"Denied: {result.reason}")
```

Blocks until the admin taps the screen or replies YES via WhatsApp.
There is no way to spoof approval from inside the sandbox.

## What you cannot do

- Import from `boxbot.*` — those packages are not in the sandbox venv.
- Spawn subprocesses — seccomp blocks `execve` and `fork`.
- Read `.env` or other secrets from disk — mode 0600, different owner.
- Install packages without approval — `site-packages` is read-only; use
  `packages.request()`.
- Read a stored secret back — `secrets` is write-only by design.
- Write raw display render code — use the declarative builder only.

## Further reading

- Full SDK reference: `src/boxbot/sdk/README.md`
- Sandbox security model: `docs/sandbox.md`
- Display block catalog and themes: `docs/display-system.md`
- Memory architecture: `docs/memory.md`
- Photo system: `src/boxbot/photos/README.md`
