# displays/

The display system — manages the 7" screen with swappable layouts that the
agent selects contextually or rotates through on a timer.

For the full design — block library, data binding, themes, animation, and
architecture — see [../../docs/display-system.md](../../docs/display-system.md).

## Architecture

Displays are composed from a block library (layout containers + content
blocks) and rendered via pygame on the HDMI output. Data flows through
displays automatically via declared data sources — the agent creates a
display once and it stays current without agent involvement.

```
Display Spec (JSON)
  ├── data_sources: declared fetch schedules (weather, HTTP, etc.)
  ├── layout: tree of containers and content blocks with data bindings
  └── theme: named theme reference

Display Manager (manager.py)
  ├── Data Source Manager — schedules async fetches, caches results
  ├── Layout Engine — resolves block tree → positioned render commands
  └── Renderer — pygame surface, animations, dirty rectangling
```

## Files

### `base.py`
The `Display` base class for Python-based displays:
- `name` — unique identifier
- `description` — what this display shows (used by the agent to decide
  which display to activate)
- `render(surface, context)` — draw content onto the pygame surface
  (1024x600). Must complete in <50ms
- `update_interval` — seconds between re-renders
- `setup()` / `teardown()` — optional lifecycle hooks

### `manager.py`
Display lifecycle, data source management, and rendering:
- Loads and registers all display modules (built-in + user-installed)
- Manages data sources: schedules async fetches per source, caches
  results, normalizes data (including icon/color mapping)
- Runs the layout engine: resolves block trees into positioned render
  commands, handles text wrapping/truncation, applies theme tokens
- Renders to pygame surface with animation support (fade, count_up,
  slide transitions)
- Handles display switching (agent-triggered or timed rotation)
- Passes `args` from `switch_display` calls through to display context
- Frame rate management: 30fps animating → 1fps live blocks → 0fps idle

### `blocks/`
Block implementations. Each block:
- Declares parameters and types
- Implements `measure(available_width, available_height)` → needed size
- Implements `render(surface, rect, theme, data)` → draws itself

Layout containers (7): `row`, `column`/`stack`, `columns`, `card`,
`spacer`, `divider`, `repeat`

Content blocks (13): `text`, `metric`, `badge`, `list`, `table`,
`key_value`, `icon`, `emoji`, `image`, `chart`, `progress`, `clock`,
`countdown`

Composite widgets (2): `weather_widget`, `calendar_widget`

Meta blocks (2): `rotate`, `page_dots`

### `sources/`
Data source implementations:
- Built-in: `clock`, `weather`, `calendar`, `tasks`, `people`,
  `agent_status`
- Custom: `http_json`, `http_text`, `memory_query`, `static`

Built-in sources deliver display-ready data including resolved icon
names and semantic color tokens. Custom sources support `fields` with
`map` transforms for declarative value mapping.

### `themes/`
Built-in theme definitions: `boxbot` (default), `midnight`, `daylight`,
`classic`. Community themes go in `themes/` at the project root.

### `builtins/`
Built-in displays shipped with boxBot:

#### `calendar_display.py`
Today's date, upcoming events, and agenda.

#### `daily_highlights.py`
Morning briefing — weather summary, top tasks, reminders, overnight
messages. Designed for the first wake-up cycle.

#### `weather_display.py`
Current conditions and forecast with temperature, conditions icon,
and multi-day outlook.

#### `agent_status.py`
Agent state display: Sleeping, Listening, Thinking, Working. Shows
recent activity log and next scheduled wake time.

#### `picture.py`
Default sleep-state display. Two modes via `args`:
- **Slideshow** (`args={}`) — cycles through slideshow-enabled photos
- **Focused** (`args={"image_ids": [...]}`) — specific photos only

## User-Installed Displays

User displays live in `displays/` at the project root. Each display is
a directory containing either a `display.json` spec (block-based) or a
Python `__init__.py` (direct pygame). Auto-discovered on startup.

See [../../docs/display-development.md](../../docs/display-development.md)
for the full development guide.

## Design Constraints

- **Resolution:** 1024x600 pixels (7" IPS, landscape)
- **Readability:** Text legible from ~1-2 meters (minimum ~18px body,
  ~28px headers)
- **Theme:** Use theme tokens for all colors and fonts. Default theme
  uses warm dark backgrounds suited for a living space
- **Performance:** Block renders complete in <50ms. Layout engine caches
  computed layouts and only recomputes on data shape changes
