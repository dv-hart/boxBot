# Skills Development Guide

## Overview

Skills are boxBot's extensible capabilities. Each skill becomes a tool
the Claude agent can invoke during conversations. The framework is designed
so you can add new skills without modifying any core code.

## Quick Start

1. Create a directory in `skills/` at the project root:
   ```
   skills/my_skill/
   ```

2. Create the skill manifest (`skill.yaml`):
   ```yaml
   name: my_skill
   description: "Brief description of what this skill does"
   version: "0.1.0"
   author: "Your Name"
   ```

3. Create `__init__.py` exporting your Skill subclass:
   ```python
   from boxbot.skills.base import Skill

   class MySkill(Skill):
       name = "my_skill"
       description = "What this skill does — this text is shown to the agent"
       parameters = {
           "type": "object",
           "properties": {
               "query": {
                   "type": "string",
                   "description": "The search query"
               }
           },
           "required": ["query"]
       }

       async def execute(self, **kwargs) -> str:
           query = kwargs["query"]
           # Do the thing
           result = await self.do_something(query)
           return result
   ```

4. boxBot discovers and loads the skill on next startup.

## Skill Base Class

Every skill extends `boxbot.skills.base.Skill`:

| Attribute/Method | Required | Description |
|-----------------|----------|-------------|
| `name` | Yes | Unique identifier (snake_case) |
| `description` | Yes | Natural language description for the agent |
| `parameters` | Yes | JSON Schema for input parameters |
| `execute(**kwargs)` | Yes | Async method — performs the skill's action |
| `setup()` | No | Called once on load (init resources) |
| `teardown()` | No | Called on shutdown (cleanup resources) |

`execute()` returns a string that becomes the tool response shown to the
agent. Keep responses concise and structured.

## Accessing boxBot Services

Skills receive a `context` object in `setup()` that provides access to
boxBot services:

```python
async def setup(self, context):
    self.config = context.config      # App configuration
    self.memory = context.memory      # Memory store
    self.scheduler = context.scheduler # Task scheduler
    self.events = context.events      # Event bus
```

## Best Practices

- **Keep skills focused** — one skill, one capability
- **Concise descriptions** — the agent reads these; be clear and brief
- **Structured output** — return well-formatted strings the agent can
  parse and relay to the user
- **Handle errors gracefully** — return error messages as strings, don't
  raise exceptions (the agent needs to communicate failures)
- **Include a README** — document what the skill does and any setup needed
- **Add a test** — `tests/skills/test_my_skill.py`

## Testing

Skills can be tested independently by mocking the context:

```python
import pytest
from skills.my_skill import MySkill

@pytest.fixture
def skill():
    s = MySkill()
    # Mock setup if needed
    return s

@pytest.mark.asyncio
async def test_execute(skill):
    result = await skill.execute(query="test")
    assert "expected" in result
```

## Example Skills

See `skills/example_skill/` for a complete, annotated reference
implementation.
