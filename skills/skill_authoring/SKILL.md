---
name: skill_authoring
description: How to create your own skills. A skill is structured prompt data — a markdown SKILL.md you read on demand, optionally bundled with helper scripts. Load before calling bb.skill.create(), or when you catch yourself solving the same kind of problem twice.
when_to_use: You're about to create a skill, edit one, or you just found a recurring workflow that should outlive this conversation.
---

# Authoring skills

A skill is a folder under `skills/` with a `SKILL.md`. The body teaches
*future you* how to do something. Optional scripts under `scripts/` give
you deterministic helpers.

Skills are not callable. No parameters, no env vars, no schedule. Need
any of those? You want an **integration**
(`integrations/<name>/`) — see [Skill vs. integration](#skill-vs-integration).

## Write one when

All three hold:

- You'd benefit from re-reading it later. (Once is enough →
  `bb.workspace`.)
- The instructions are general, not tied to this conversation's state.
- No other consumer (display, scheduler, briefing) needs the same
  logic. (They do → integration.)

When in doubt, write it. Skills are cheap: ~100 tokens of metadata in
the system prompt until triggered.

## Write it terse

You are the reader. You do not need hand-holding prose.

- Fragments over sentences. Drop articles, hedges, and pleasantries.
- Lead with the rule, then the qualifier.
- Tables and code over paragraphs.
- Keep code, identifiers, paths, and error strings **byte-for-byte
  exact** — compress prose, never payload.
- Keep full clarity for anything destructive, irreversible, or
  security-relevant. Terse is not ambiguous.

"Use `bb.weather.forecast(days=N)`" beats "You'll probably want to
reach for the forecast helper, which takes a number of days."

## SKILL.md format

```markdown
---
name: weather
description: Get NOAA weather forecasts for the configured location. Use when the user asks about weather, temperature, rain, or what to wear.
when_to_use: User mentions weather, temperature, rain, snow, sun, what to wear, what the day looks like outside.
---

# Weather

`bb.weather.forecast(days=N)` for an N-day forecast.
Hourly precipitation: HOURLY.md.
```

| Field | Required | Constraint |
|---|---|---|
| `name` | yes | ≤64 chars; `^[a-z0-9-]+$`; not `anthropic` or `claude`; no XML |
| `description` | yes | ≤1024 chars; non-empty; no XML; must answer *what* AND *when* |
| `when_to_use` | recommended | One sentence of trigger conditions. Helps the loader rank. |

Bad: `"Weather skill"`. Good: `"Get NOAA weather forecasts for the
configured location. Use when the user asks about weather, temperature,
rain, or what to wear."`

The description sits in the system prompt at Level 1, always. It is how
future-you decides relevance. Be specific.

## Length and progressive disclosure

The body is **Level 2**, loaded on trigger. Keep it **under 5 KB**.
Overflow goes to Level 3:

```
skills/<name>/
  SKILL.md        # Level 2 — overview, when to use
  REFERENCE.md    # Level 3 — full API table
  EXAMPLES.md     # Level 3 — worked examples
  scripts/
    helper.py     # importable in execute_script
```

Link Level 3 docs by filename (`see REFERENCE.md`) so future-you knows
they exist without paying for them.

## Bundled scripts

`skills/` is on the sandbox `sys.path`:

```python
from skills.weather.scripts import nws_raw
data = nws_raw.fetch(lat=45.5, lon=-122.7, days=5)

from skills.weather.scripts.nws_raw import format_for_voice
```

Import only. `subprocess.run(["python3", ...])` never works — seccomp
blocks `execve`/`fork`.

`bb.skill.add_script(filename, content)` stamps `scripts/__init__.py`
for you.

Bundle when the logic is too long to inline every time, when you want a
stable interface other skills reuse, or when determinism matters.
Otherwise just describe the steps.

## Creating

```python
import boxbot_sdk as bb

s = bb.skill.create("weather")
s.description = (
    "Get NOAA weather forecasts. Use when the user asks about weather, "
    "temperature, rain, or what to wear."
)
s.body = """
# Weather

`bb.weather.forecast(days=N)` for the forecast.
Hourly precipitation: HOURLY.md.
"""
s.add_resource("HOURLY.md", "# Hourly forecast\n\n…")
s.add_script("nws_raw.py", "import requests\n\ndef fetch(...):\n    …")
s.save()
```

`save()` returns immediately. The loader picks it up on the next
discovery scan, typically next conversation. An existing name fails with
`status: "exists"` — delete first or pick another name. No overwrite.

## Iterating on your own skill

**Read your script.** Skills live at `<repo>/skills/<name>/`, which the
sandbox cannot browse. Pull a file in with `load_skill` — a tool call,
not `execute_script`:

```
load_skill(name="refresh_weekly_glance_agenda", subpath="scripts/refresh.py")
```

**Replace via delete + create:**

```python
import boxbot_sdk as bb

bb.skill.delete("refresh_weekly_glance_agenda")
s = bb.skill.create("refresh_weekly_glance_agenda")
s.description = "..."
s.body = "..."
s.add_script("refresh.py", fixed_source)
s.save()
```

`delete` refuses built-ins (`bb`, `skill_authoring`, `onboarding`,
`hal-sandbox-ref`, anything shipped in git) — it needs the
`.agent-authored` marker that `save` stamps. `status: "forbidden"` means
you hit a built-in; pick a different name.

## Skill vs. integration

| | Skill | Integration |
|---|---|---|
| Form | SKILL.md + optional scripts | `integrations/<name>/{manifest.yaml, script.py}` |
| Stateful | No | Yes — creds, caches, refresh tokens |
| Runs on its own | No, inert until triggered | Yes, on schedule or events |
| Consumers | You, in one conversation | Displays, scheduler, SDK, briefings |
| Lifecycle | Lives until removed | Owns refresh cadence + error handling |

Wanting `s.add_parameter(...)`, `s.add_env_var(...)`, or "fetch this on
a schedule" means you want an integration. **Skills are nouns you read.
Integrations are verbs that run whether you're awake or not.**

## Self-check

- [ ] `name` ≤64 chars, lowercase + hyphens, not "anthropic"/"claude"
- [ ] `description` says *what* AND *when*
- [ ] Body under ~5 KB, or split to Level 3
- [ ] Terse — fragments, tables, code; no filler prose
- [ ] No parameters, env vars, or scheduling (those are integrations)
- [ ] Scripts are importable, not subprocess-invoked
- [ ] You'd be glad to find this in a future conversation
