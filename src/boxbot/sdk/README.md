# sdk/

The boxBot SDK — a constrained, immutable API that agent-written scripts
import to interact with boxBot internals. This is the **only** interface
sandbox scripts have to the system.

## Why an SDK?

The agent's primary tool is `execute_script`. Instead of bloating the tool
list with a dedicated tool for every operation (create display, manage photos,
manage memory, install packages...), the agent writes Python scripts that
import from this SDK. This gives us:

1. **Slim tool list** — only 9 always-loaded tools instead of 14+
2. **Constrained access** — the SDK exposes safe, validated operations;
   the agent can't bypass them to touch core code
3. **Composability** — a single script can combine multiple SDK operations
   (tag a photo AND add it to the slideshow AND save a memory about it)
4. **Immutability** — the SDK is pre-installed in the sandbox venv;
   agent scripts cannot modify it
5. **Extensibility** — new SDK capabilities don't require new tools

## Installation

The SDK is part of the boxBot repo (`src/boxbot/sdk/`) but is installed
**independently** into the sandbox venv during setup. It has no dependency
on boxBot internals — only stdlib and packages already in the sandbox.

```bash
# Done automatically by scripts/setup.sh:
data/sandbox/venv/bin/pip install -e src/boxbot/sdk/
```

## How It Works

Agent scripts interact with the SDK's Python API. Under the hood, the SDK
communicates results back to the main process through structured JSON on
stdout. The `execute_script` tool separates SDK actions from regular script
output and applies them.

```
Agent calls execute_script with:
┌──────────────────────────────────────────┐
│  from boxbot_sdk import display          │
│                                          │
│  spec = {                                │
│    "name": "weather_board",              │
│    "theme": "boxbot",                    │
│    "data_sources": [{"name": "weather"}],│
│    "layout": { ... block tree ... },     │
│  }                                       │
│  display.preview(spec)                   │
│  display.save(spec)                      │
└─────────────────┬────────────────────────┘
                  │
                  ▼
SDK emits structured JSON to stdout:
┌──────────────────────────────────────────┐
│  {"_sdk": "display.save",                │
│   "spec": { ... full spec dict ... }}    │
└─────────────────┬────────────────────────┘
                  │
                  ▼
execute_script tool (main process) applies:
┌──────────────────────────────────────────┐
│  Validates spec against block schemas    │
│  Writes data/displays/weather_board.json │
│  Registers with the display manager      │
│  Returns confirmation to the agent       │
└──────────────────────────────────────────┘
```

The agent doesn't write raw render code — the spec is purely
declarative. The rendering engine draws each block from a fixed,
validated registry; the agent describes **what** to show.

## Modules

### `display` — Display specs as JSON dicts

The display SDK is a thin pair of CRUD calls over JSON dicts. There
is no builder, no fluent API, no block classes — the agent reads,
mutates, and writes the spec dict directly.

For the complete block reference and data binding system, see
[../../skills/bb/modules/display.md](../../skills/bb/modules/display.md).

```python
from boxbot_sdk import display

# Author from scratch
spec = {
    "name": "weather_board",
    "theme": "boxbot",
    "data_sources": [{"name": "weather"}],
    "layout": {
        "type": "column",
        "padding": 24,
        "gap": 16,
        "children": [
            {"type": "row", "padding": 24, "align": "center", "children": [
                {"type": "icon", "name": "{weather.icon}", "size": "xl"},
                {"type": "metric", "value": "{weather.temp}°F",
                 "label": "{weather.condition}"},
            ]},
            {"type": "row", "gap": 16, "padding": [0, 24], "align": "spread",
             "children": [
                {"type": "repeat", "source": "{weather.forecast}", "max": 5,
                 "children": [{"type": "column", "align": "center", "children": [
                    {"type": "text", "content": "{.day}", "size": "caption",
                     "color": "muted"},
                    {"type": "icon", "name": "{.icon}", "size": "sm"},
                    {"type": "text", "content": "{.high}°/{.low}°",
                     "size": "small"},
                 ]}]},
            ]},
        ],
    },
}

display.preview(spec)    # render PNG, attach to the tool result
display.save(spec)       # validate, write, register live

# Edit an existing display
spec = display.load("weather_board")
spec["theme"] = "midnight"
display.save(spec)
```

**SDK calls (8 total):** `list`, `load`, `save`, `preview` (with
optional ``data=`` override for http_json testing), `delete`,
`describe_source`, `schema`, `update_data`.

**Layout containers (7):** `row`, `column`/`stack`, `columns`,
`card`, `spacer`, `divider`, `repeat`.

**Content blocks (13):** `text`, `metric`, `badge`, `list`, `table`,
`key_value`, `icon`, `emoji`, `image`, `chart`, `progress`, `clock`,
`countdown`.

**Data sources:** Built-in (`weather`, `calendar`, `tasks`, `people`,
`agent_status`, `clock`); custom (`http_json`, `http_text`, `static`,
`memory_query`).

**Themes:** `boxbot`, `midnight`, `daylight`, `classic`.

**Validation + warnings:** `save` and `preview` validate the spec and
raise `RuntimeError` listing every problem on a bad input. Successful
calls return `warnings` — bindings that didn't resolve at render time
(usually a typo, sometimes a not-yet-fetched http_json source). The
agent fixes warnings by editing the dict and previewing again.

**Discovery:** `display.schema()` returns the full block reference as
a dict — every field, default, and valid-values list. Use it to
introspect what's available without re-reading the doc.

### `skill` — Skill Builder

Create new skills. A skill is **structured prompt data** — a SKILL.md
(YAML frontmatter + markdown body) the agent reads on demand later,
optionally bundled with helper scripts under `scripts/` and Level 3
sub-docs at the skill root.

Skills are not callable functions — there are no parameters, env vars,
or scheduling. If you need any of those, what you actually want is an
**integration** (`src/boxbot/integrations/`), not a skill. See
`skills/skill_authoring/SKILL.md` for the full authoring guide.

```python
import boxbot_sdk as bb

s = bb.skill.create("weather")
s.description = (
    "Get NOAA weather forecasts for the configured location. "
    "Use when the user asks about weather, temperature, or conditions."
)
s.body = """
# Weather

Use `bb.weather.forecast(days=N)` to get an N-day forecast.
For hourly precipitation detail, see HOURLY.md.
"""

# Optional Level 3 sub-doc — referenced from the body, loaded on demand
s.add_resource("HOURLY.md", "# Hourly forecast\n\n…")

# Optional bundled helper script — importable as
# `from skills.weather.scripts import nws_raw` inside execute_script.
# The writer auto-stamps a sibling __init__.py so the import resolves.
s.add_script("nws_raw.py", "import requests\n…")

s.save()  # loader picks it up on next discovery scan
```

**Validation** (per Anthropic's Agent Skills spec):
- `name`: ≤64 chars, lowercase, `[a-z0-9_-]+`, not `anthropic`/`claude`
- `description`: ≤1024 chars, non-empty, no XML brackets

**Conflict policy:** if `skills/<name>/` already exists, save fails
with `status: "exists"`. Skills are never silently overwritten.

### `packages` — Package Manager

Request package installation. Triggers user approval flow.

```python
from boxbot_sdk import packages

# Blocks until user approves or denies
result = packages.request("google-api-python-client",
                          reason="Needed for Gmail integration")
if result.approved:
    import googleapiclient  # now available
else:
    print(f"Denied: {result.reason}")
```

### `memory` — Memory Store

Read and write memories.

```python
from boxbot_sdk import memory

# Save a memory
memory.save(
    content="Jacob is allergic to peanuts",
    people=["Jacob"],
    tags=["health", "allergy"],
    importance=0.9
)

# Search memories
results = memory.search("allergies", people=["Jacob"])
for m in results:
    print(f"{m.content} (importance: {m.importance})")

# Delete a memory
memory.delete(memory_id="abc-123")
```

### `photos` — Photo Manager

Manage the photo library, tags, slideshow, and soft-delete lifecycle.
Photo **search** is also available as a direct agent tool
(`search_photos`) — both use the same backend.

```python
from boxbot_sdk import photos

# Search (shared backend with search_photos tool)
results = photos.search(query="beach sunset", tags=["vacation"],
                        people=["Jacob"], limit=10)
for p in results:
    print(f"{p.id}: {p.description} [{', '.join(p.tags)}]")

# Get full details for a photo
info = photos.get("photo_123")
print(info.description, info.tags, info.people, info.file_path)

# View an unsaved file by path (e.g. inbound WhatsApp image staged at
# tmp/inbound/whatsapp/<wamid>.jpg). Path must be under an allowlisted
# root: sandbox tmp, workspace, photos, perception crops.
photos.view_path("/var/lib/boxbot-sandbox/tmp/inbound/whatsapp/wamid.HBg.jpg")

# Save a local image into the library — pipeline copies bytes, runs
# detection + tagging, and deletes the source on success. Use this
# when an inbound photo is worth keeping; otherwise let the inbound
# janitor reap it (7-day TTL on tmp/inbound/).
photo_id = photos.ingest(
    "/var/lib/boxbot-sandbox/tmp/inbound/whatsapp/wamid.HBg.jpg",
    source="whatsapp",
    sender="Erik",
    caption="my new pokémon",
)

# Update metadata
photos.update("photo_123",
              description="Updated description of this photo")

# Set tags (replaces existing tags on the photo)
photos.set_tags("photo_123", tags=["family", "beach", "summer"])

# Tag people in photos (for unknown persons identified later)
# person_index refers to the order in photo_people for this photo
photos.set_person("photo_123", person_index=0, name="Sarah")

# Slideshow management
photos.add_to_slideshow("photo_123")
photos.remove_from_slideshow("photo_456")

# Tag library management
photos.merge_tags("sunsets", into="sunset")   # consolidate synonyms
photos.rename_tag("vacaction", to="vacation") # fix typos
photos.delete_tag("obsolete_tag")             # remove unused tag

# Soft delete / restore (30-day retention, configurable)
photos.delete("photo_123")          # soft delete
photos.restore("photo_123")         # restore before retention expires
deleted = photos.list_deleted()     # see soft-deleted photos

# Storage quota info
info = photos.storage_info()
print(f"{info.used_gb:.1f} GB / {info.quota_gb:.1f} GB "
      f"({info.used_percent:.0f}%)")
```

### `tasks` — Trigger & To-Do Manager

Create and manage triggers (wake conditions) and to-do items. The
`manage_tasks` tool handles common conversational operations directly.
This SDK module is for complex multi-step task management within scripts
— batch operations, conditional logic, or combining task management
with other SDK calls.

```python
from boxbot_sdk import tasks

# Point-in-time trigger
tasks.create_trigger(
    description="Dentist reminder for Jacob",
    instructions="Remind Jacob about his dentist appointment at 4pm",
    fire_at="2026-02-21T15:30:00",
    for_person="Jacob"
)

# Timer trigger (max 24h)
tasks.create_trigger(
    description="Check for package delivery",
    instructions="Look outside and check if a package was delivered",
    fire_after="2h"
)

# Person trigger
tasks.create_trigger(
    description="Tell Jacob about dinner",
    instructions="Tell Jacob that dinner is at 7pm. Carina asked.",
    person="Jacob",
    for_person="Jacob"
)

# Compound trigger (timer + person, AND logic)
tasks.create_trigger(
    description="Remind Jacob about package",
    instructions="Remind Jacob the Amazon package arrived",
    fire_after="30m",
    person="Jacob",
    for_person="Jacob"
)

# Recurring trigger
tasks.create_trigger(
    description="Morning briefing",
    instructions="Check weather, review to-do list, update displays",
    cron="0 7 * * *"
)

# Create a to-do item
tasks.create_todo(
    description="Return library books",
    notes="Jacob mentioned during breakfast. Books on kitchen counter. "
          "Due Saturday. Check if the Poe collection has a hold.",
    for_person="Jacob",
    due_date="2026-02-22"
)

# List triggers and to-dos
for trigger in tasks.list_triggers(status="active"):
    print(f"Trigger: {trigger.description} [{trigger.status}]")

for todo in tasks.list_todos(status="pending"):
    print(f"To-do: {todo.description}")

# Get full details (includes notes)
item = tasks.get("todo_abc123")
print(item.notes)  # detailed context loaded on demand

# Complete a to-do
tasks.complete("todo_abc123")

# Cancel a trigger or to-do
tasks.cancel("trigger_xyz789")
```

## Security Properties

1. **Immutable** — the SDK is installed in the sandbox venv's site-packages.
   Agent scripts cannot modify it (site-packages is read-only to scripts)
2. **Validated** — all inputs are schema-validated before emitting actions.
   Invalid specs are rejected with clear error messages
3. **Declarative displays** — the agent describes what to show using
   building blocks, never writes raw render code that runs in main process
4. **Approval gates** — `packages.request()` blocks for out-of-band
   human approval (screen tap or admin WhatsApp YES). Display saves do
   *not* gate: the render spec is declarative, so there is no executable
   code path to police
5. **No core access** — the SDK cannot import from `boxbot.*` (different
   venv). It communicates only through structured JSON actions
6. **Auditable** — every SDK action is logged with timestamp and context

## Files

### `__init__.py`
Package init — exports top-level modules.

### `display.py`
`DisplayBuilder` class with block-based declarative builder methods.

### `skill.py`
`SkillBuilder` class for defining skills declaratively.

### `packages.py`
Package installation request flow.

### `memory.py`
Memory CRUD operations.

### `photos.py`
Photo management operations.

### `tasks.py`
Trigger and to-do management operations.

### `_transport.py`
Internal module that handles structured JSON output to stdout.
Not part of the public API.

### `_validators.py`
Input validation schemas. Ensures all SDK actions are well-formed
before emitting.
