# bb.skill — create new skills at runtime

Skills are **structured prompt data** — markdown instructions you read
later, optionally bundled with helper scripts. Use this module to teach
yourself a new recurring workflow so the next time you face the same
problem, the recipe is already on disk.

For the full authoring guide (when to write a skill, frontmatter rules,
length guidance, skill-vs-integration), load
`skills/skill_authoring/SKILL.md`. This page is just the API reference.

## When to use it

- You just figured out how to do something useful and want to keep the
  recipe (e.g. "how I draft a morning brief", "how I diagnose a
  misbehaving photo tag").
- The user asks you to remember a workflow ("when I say 'plan dinner',
  do X then Y").
- You hit the same problem twice in a row and want the third time to
  be cheap.

## When NOT to use it

- For data pipelines, scheduled fetchers, or service connectors with
  credentials. Those belong in **integrations**
  (`src/boxbot/integrations/`), not skills.
- For one-off scratch work. Use `bb.workspace` instead.
- For secrets. Use `bb.secrets`.

## API

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
s.add_resource("HOURLY.md", "# Hourly forecast\n\n…")
s.add_script("nws_raw.py", "import requests\n…")
s.save()
```

### Method reference

| Call | Required | Notes |
|---|---|---|
| `bb.skill.create(name)` | yes | Returns a builder. `name` ≤64 chars, lowercase, `[a-z0-9_-]+`, not `anthropic`/`claude`. |
| `s.description = "…"` | yes | ≤1024 chars, non-empty, no XML brackets. Should answer *what* and *when*. |
| `s.body = "…"` | yes | SKILL.md markdown body. Keep ≤5KB; split overflow into Level 3 sub-docs via `add_resource`. |
| `s.add_script(filename, content)` | optional, repeatable | `filename` must be a bare basename ending in `.py`. Bundled under `scripts/<filename>`. The writer also stamps a `scripts/__init__.py` so you can `from skills.<name>.scripts import <module>` later. |
| `s.add_resource(filename, content)` | optional, repeatable | Bare basename. Lives at the skill root. Conventionally `.md`. Cannot be named `SKILL.md`. |
| `s.save()` | terminal | Emits `skill.save`. Refuses with `status: "exists"` if `skills/<name>/` already exists. |

### What lands on disk

For the example above:

```
skills/weather/
  SKILL.md          # frontmatter (name, description) + body
  HOURLY.md         # the resource you added
  scripts/
    __init__.py     # auto-written
    nws_raw.py      # your script
```

Files are owned `boxbot:boxbot` mode `0644`. The sandbox can read but
not modify them after save — your future self won't be able to
overwrite this skill from a sandbox script. Use `skill.delete` (when
implemented) and re-create instead.

### Conflict policy

If a skill named `<name>` already exists, `save()` fails fast with
`status: "exists"` and writes nothing. There is no overwrite. This
prevents accidental clobbers of community skills.

### Activation

The loader picks new skills up on its next discovery scan — typically
the next conversation. There is no live in-conversation registration.
If you need the skill *right now*, you do not — the markdown body
already exists in the conversation that just wrote it.

### Importing bundled scripts

Inside `execute_script`, the project's `skills/` directory is on
`sys.path`. Bundled scripts are importable directly:

```python
from skills.weather.scripts import nws_raw
data = nws_raw.fetch(lat=45.5, lon=-122.7, days=5)
```

Subprocess invocation does not work — seccomp blocks `execve`/`fork`
in the sandbox. Always import.
