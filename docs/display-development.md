# Display Development Guide

## Overview

Displays are swappable screen layouts for boxBot's 7" LCD (1024x600).
The display manager rotates through displays during idle time, and the
agent can switch to a specific display contextually.

For the full display system design — block library, data binding, themes,
animation, and architecture — see [display-system.md](display-system.md).

## Three Ways to Create Displays

### 1. Agent-Created (Block System)

The agent uses `boxbot_sdk.display` to compose displays from the block
library. See [display-system.md](display-system.md) for the complete
block reference and use case walkthroughs.

```python
from boxbot_sdk import display

d = display.create("morning_briefing")
d.set_theme("boxbot")
d.data("weather")

header = d.row(padding=24, align="center")
header.icon("{weather.icon}", size="xl")
header.text("{weather.temp}°F", size="title", animation="count_up")

d.preview()   # render to PNG, agent views it
d.save()      # emit spec, queue for user approval
```

### 2. User-Contributed (Drop-in Directories)

Community displays go in `displays/` at the project root. They can
use the block system (write a `display.json` spec) or implement a
Display subclass directly in Python for full pygame access.

#### Using the Block System

```
displays/
  my_display/
    display.json       # display spec (same format as agent-created)
    display.yaml       # metadata: name, description, version, author
    README.md          # documentation
```

The spec format is the same JSON that `d.save()` emits. Write it by
hand or generate it with the SDK.

#### Using Python Directly

For cases that need capabilities beyond the block library:

```
displays/
  my_display/
    __init__.py        # exports the Display subclass
    display.yaml       # metadata
    assets/            # fonts, images, icons
    README.md          # documentation
```

```python
from boxbot.displays.base import Display

class MyDisplay(Display):
    name = "my_display"
    description = "What this display shows — used by the agent"
    update_interval = 60

    async def setup(self, context):
        # Load fonts, images, etc.
        pass

    def render(self, surface, context):
        # Full pygame access — draw onto the 1024x600 surface
        surface.fill((25, 23, 20))
        # ... render your content
```

### 3. Built-in Displays

Shipped with boxBot in `src/boxbot/displays/builtins/`. These are
developer-authored and can use either the block system or raw pygame.

## Display Base Class

Every display (Python-based) extends `boxbot.displays.base.Display`:

| Attribute/Method | Required | Description |
|-----------------|----------|-------------|
| `name` | Yes | Unique identifier (snake_case) |
| `description` | Yes | Natural language description for the agent |
| `update_interval` | Yes | Seconds between re-renders |
| `render(surface, context)` | Yes | Draw content onto pygame surface |
| `setup(context)` | No | Called once on load |
| `teardown()` | No | Called on shutdown |

## Render Context

The `context` dict passed to `render()` includes:

```python
{
    "time": datetime,           # Current time
    "config": AppConfig,        # App configuration
    "weather": WeatherData,     # Latest weather (if available)
    "tasks": list[Task],        # Pending tasks
    "people_present": list,     # Currently detected people
    "agent_state": str,         # "sleeping", "listening", "thinking", etc.
    "memories": list[Memory],   # Recent relevant memories
    "args": dict,               # Display-specific args from switch_display
}
```

Not all fields are always populated — check before using.

### Display Args

The `args` dict contains display-specific arguments passed through from
the `switch_display` tool. Each display defines what args it accepts.
The display manager passes args through without interpreting them.

Example: the `picture` display uses args to control its mode:
- `args={}` or no args — slideshow mode
- `args={"image_ids": ["abc", "def"]}` — cycle through specific photos

Displays should handle missing or empty args gracefully by falling back
to a sensible default mode.

## Design Guidelines

- **Resolution:** 1024 x 600 pixels (landscape)
- **Readability:** Text legible from 1-2 meters. Minimum font size ~18px
  for body text, ~28px for headers
- **Dark backgrounds:** The display sits in a living space. Use the theme
  system — default theme uses warm dark backgrounds
- **Performance:** `render()` should complete in <50ms. Pre-compute heavy
  data in `setup()` or background tasks, not in the render loop
- **Assets:** Store fonts, icons, and images in an `assets/` subdirectory
- **Themes:** Use theme tokens for colors and fonts. Don't hardcode hex
  values. See [display-system.md](display-system.md#theme-system)

## Testing

Test displays by rendering to an off-screen surface:

```python
import pygame
import pytest

@pytest.fixture
def surface():
    pygame.init()
    return pygame.Surface((1024, 600))

def test_render(surface):
    from displays.my_display import MyDisplay
    display = MyDisplay()
    context = {"time": datetime.now(), "agent_state": "sleeping"}
    display.render(surface, context)
```

## Example

See `displays/example_display/` for a directory structure template.
See [display-system.md](display-system.md#use-case-walkthroughs) for
complete block-based display examples.
