# bb.skill — create skills at runtime

Skills are **structured prompt data**: markdown you read later,
optionally bundled with helper scripts. Use this to keep a recipe so
the next time costs nothing.

Full authoring guide (when to write one, frontmatter, length,
skill-vs-integration): `skills/skill_authoring/SKILL.md`. This is the
API reference.

**Use when:** you just worked out something reusable ("how I draft a
morning brief"); the user asks you to remember a workflow ("when I say
'plan dinner', do X then Y"); you hit the same problem twice.

**Not for:** data pipelines, scheduled fetchers, credentialed service
connectors → **integrations**. One-off scratch work → `bb.workspace`.
Credentials → `bb.secrets`.

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

Use `bb.weather.forecast(days=N)` for an N-day forecast.
Hourly precipitation detail: HOURLY.md.
"""
s.add_resource("HOURLY.md", "# Hourly forecast\n\n…")
s.add_script("nws_raw.py", "import requests\n…")
s.save()
```

| Call | Required | Notes |
|---|---|---|
| `bb.skill.create(name)` | yes | Returns a builder. `name` ≤64 chars, lowercase `[a-z0-9_-]+`, not `anthropic`/`claude`. |
| `s.description = "…"` | yes | ≤1024 chars, non-empty, no XML brackets. Answer *what* and *when*. |
| `s.body = "…"` | yes | SKILL.md markdown. ≤5 KB; overflow goes to Level 3 sub-docs via `add_resource`. |
| `s.add_script(f, content)` | optional, repeatable | Bare basename ending `.py`. Lands at `scripts/<f>`. A `scripts/__init__.py` is stamped for you. |
| `s.add_resource(f, content)` | optional, repeatable | Bare basename, skill root, conventionally `.md`. Cannot be `SKILL.md`. |
| `s.save()` | terminal | Emits `skill.save`. Fails `status: "exists"` if `skills/<name>/` is taken. |

## On disk

```
skills/weather/
  SKILL.md          # frontmatter (name, description) + body
  HOURLY.md
  scripts/
    __init__.py     # auto-written
    nws_raw.py
```

Owned `boxbot:boxbot`, mode `0644`. The sandbox reads but cannot modify
after save — you cannot overwrite a skill from a sandbox script.

**No overwrite.** An existing `<name>` makes `save()` fail fast with
`status: "exists"` and write nothing. This protects community skills.

## Activation

The loader picks new skills up on its next discovery scan, typically
the next conversation. No live registration — and you do not need it:
the markdown body already exists in the conversation that wrote it.

## Importing bundled scripts

`skills/` is on `sys.path` inside `execute_script`:

```python
from skills.weather.scripts import nws_raw
data = nws_raw.fetch(lat=45.5, lon=-122.7, days=5)
```

Import only. Seccomp blocks `execve`/`fork`, so subprocess invocation
never works.
