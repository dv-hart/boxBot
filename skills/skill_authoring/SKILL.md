---
name: skill_authoring
description: How to create your own skills. A skill is structured prompt data — a markdown SKILL.md you'll read on demand later, optionally bundled with helper scripts. Load this whenever you're about to call bb.skill.create(), or whenever you find yourself solving the same kind of problem twice and want to teach yourself the recipe so the next time is easier.
when_to_use: You're about to create a new skill, edit an existing one, or you just discovered a recurring workflow that should outlive this conversation.
---

# Authoring skills

A skill is a folder under `skills/` containing a `SKILL.md` file. The
markdown body teaches *future you* how to do something — instructions,
when to use them, gotchas, examples. Optional bundled scripts under
`scripts/` give you deterministic helpers to call.

Skills are not callable functions. They have no parameters, no env vars,
no schedule. If you need any of those, what you actually want is an
**integration** (`src/boxbot/integrations/`), not a skill — see
[Skill vs. integration](#skill-vs-integration) below.

## When to write a skill

Write one when **all** of these are true:

- You'd benefit from re-reading the instructions later. (If once is
  enough, jot it in `bb.workspace` instead.)
- The instructions are general — they don't depend on this specific
  conversation's state.
- Other consumers (the display, the scheduler, a routine briefing) do
  not need the same logic. (If they do → integration.)

Write a skill **when in doubt** for repeatable agent-side workflows:
"how I draft a morning briefing", "how I onboard a new household
member", "how I diagnose a misbehaving photo tag". Skills are cheap to
add and only enter the system prompt as ~100 tokens of metadata until
triggered.

## SKILL.md format

```markdown
---
name: weather
description: Get NOAA weather forecasts for the configured location. Use when the user asks about weather, temperature, rain, or what to wear.
when_to_use: User mentions weather, temperature, rain, snow, sun, what to wear, what the day looks like outside.
---

# Weather

Use `bb.weather.forecast(days=N)` to get an N-day forecast.
For hourly precipitation, see HOURLY.md.
```

### Frontmatter rules (Anthropic Agent Skills spec)

| Field | Required | Constraint |
|---|---|---|
| `name` | yes | ≤64 chars; `^[a-z0-9-]+$`; cannot be `anthropic` or `claude`; no XML |
| `description` | yes | ≤1024 chars; non-empty; no XML; should answer *what it does* AND *when to use it* |
| `when_to_use` | no, but recommended | one sentence on the trigger conditions; helps the loader rank skills |

A bad description: `"Weather skill"`. A good one:
`"Get NOAA weather forecasts for the configured location. Use when the user asks about weather, temperature, rain, or what to wear."`

The description goes into the system prompt at Level 1 (always). It is
how *you, in some future conversation* decide whether this skill is
relevant. Be specific.

## Body length and progressive disclosure

The body is **Level 2** — loaded when the skill is triggered. Keep it
**under 5 KB**. If your guidance grows past that, split into Level 3
sub-docs:

```
skills/<name>/
  SKILL.md        # Level 2 — overview, when to use, top of the funnel
  REFERENCE.md    # Level 3 — full API table, only loaded if needed
  EXAMPLES.md     # Level 3 — worked examples
  scripts/
    helper.py     # bundled helper, importable in execute_script
```

In SKILL.md, *link* to the Level 3 docs by filename (`see REFERENCE.md`)
so future-you knows they exist without paying for them upfront.

## Bundled scripts

A bundled script is a Python file under `scripts/` you ship alongside
the SKILL.md. Two ways to use it inside `execute_script`:

```python
# Import and call (works because skills/ is on the sandbox sys.path)
from skills.weather.scripts import nws_raw
data = nws_raw.fetch(lat=45.5, lon=-122.7, days=5)
```

```python
# Reuse the script's logic from another bundled script in another skill
from skills.weather.scripts.nws_raw import format_for_voice
```

Subprocess execution (`subprocess.run(["python3", "..."])`) does **not**
work — seccomp blocks `execve`/`fork` in the sandbox. Always import.

When you call `bb.skill.add_script(filename, content)`, the writer also
stamps a `scripts/__init__.py` so the import path resolves. You don't
have to write it yourself.

Bundle a script when:
- Its logic is too long or fiddly to inline in every conversation.
- You want a stable interface other skills can reuse.
- Determinism matters — you'd rather call tested code than re-derive it.

Otherwise, just describe the steps in the SKILL.md body.

## Creating a skill

```python
import boxbot_sdk as bb

s = bb.skill.create("weather")
s.description = (
    "Get NOAA weather forecasts. Use when the user asks about weather, "
    "temperature, rain, or what to wear."
)
s.body = """
# Weather

Use `bb.weather.forecast(days=N)` for the forecast. For hourly
precipitation, see HOURLY.md.
"""
s.add_resource("HOURLY.md", "# Hourly forecast\n\n…")
s.add_script("nws_raw.py", "import requests\n\ndef fetch(...):\n    …")
s.save()
```

`save()` returns immediately. The loader picks the skill up on the next
discovery scan (typically next conversation). If a skill named `weather`
already exists, save fails with `status: "exists"` — delete it first
(or pick a different name) instead of overwriting.

## Skill vs. integration

| | Skill | Integration |
|---|---|---|
| Form | SKILL.md + optional bundled scripts | Python module under `src/boxbot/integrations/` |
| Stateful? | No | Yes (creds, cached values, refresh tokens) |
| Runs on its own? | No, inert until triggered | Yes, on a schedule or events |
| Consumers | The agent, in one conversation | Many: displays, scheduler, agent SDK, briefings |
| Lifecycle | Lives until manually removed | Owns refresh cadence + error handling |

If you find yourself wanting `s.add_parameter(...)`, `s.add_env_var(...)`,
or "fetch this on a schedule", what you want is an integration. Skills
are nouns the agent reads; integrations are verbs that run whether the
agent is awake or not.

## Self-check before saving

- [ ] `name` is ≤64 chars, lowercase + hyphens, not "anthropic" or "claude"
- [ ] `description` is specific about *what* and *when*, not just *what*
- [ ] Body is under ~5 KB or split into Level 3 sub-docs
- [ ] No invocation parameters, no env vars, no scheduling — those are integration concerns
- [ ] Bundled scripts (if any) are importable, not subprocess-invoked
- [ ] You'd be glad to find this skill in a future conversation
