# skills/

The skills framework — boxBot's modular, optional capabilities. Each skill
adds a domain-specific ability the agent can invoke during conversations.

Skills are distinct from **tools** (`src/boxbot/tools/`). Tools are always
loaded and provide core system operations (switch display, execute script,
send message). Skills are modular extensions — added, removed, or
customized without touching core code. Most skill logic lives in embedded
Python scripts that run via the `execute_script` tool, keeping skill
definitions slim and the core repo focused.

## Architecture

Skills are registered as additional tools with the Claude Agent SDK. The
framework handles discovery, loading, and schema generation. Only relevant
skills are injected per conversation to conserve the agent's context window.

## Files

### `base.py`
The `Skill` base class that all skills must extend:
- `name` — unique identifier
- `description` — natural language description for the agent's context
- `parameters` — JSON schema defining the skill's input parameters
- `execute()` — async method that performs the skill's action
- `setup()` / `teardown()` — optional lifecycle hooks

### `registry.py`
Skill discovery and registration:
- Scans `src/boxbot/skills/builtins/` for built-in skills
- Scans `skills/` (project root) for user-installed skills
- Validates skill schemas and resolves conflicts
- Provides `get_skills()` to return all registered skills as agent tools

### `builtins/`
Built-in skills that ship with boxBot:

#### `reminders.py`
Create, list, and manage reminders. Wraps the `manage_tasks` tool with
reminder-specific logic — translates natural-language reminder patterns
into the appropriate triggers and to-do items. Supports compound triggers
like "tell Jacob when you see him" (person trigger) and "in 30 minutes,
remind Jacob when he gets home" (timer + person compound trigger).

#### `weather.py`
Fetch current weather and forecasts via API. Used by the weather display
and proactive morning briefings. The skill's logic is a Python script that
calls a weather API and formats the response.

#### `home.py`
Placeholder for home automation integrations. Designed as an interface that
users customize for their specific smart home setup (Home Assistant, etc.).

Note: Script execution, photo management, and memory management are **core
tools** (see `src/boxbot/tools/`), not skills. They are always available
to the agent regardless of which skills are loaded.

## User-Installed Skills

User skills live in `skills/` at the project root (not in `src/`). Each skill
is a directory containing:
```
skills/
  my_skill/
    __init__.py       # exports the Skill subclass
    skill.yaml        # metadata: name, description, version, author
    README.md         # documentation
    requirements.txt  # additional dependencies (optional)
```

The registry auto-discovers skills in this directory on startup.
