# Display System

## Overview

boxBot's 7" LCD (1024x600, IPS) is the primary visual output. The display
system renders swappable layouts — dashboards, clocks, photo slideshows,
weather boards — using pygame on the HDMI output.

Displays are authored three ways:
1. **Built-in** — shipped with boxBot, developer-authored in pygame
2. **User-contributed** — community displays dropped into `displays/`
3. **Agent-created** — built through the `boxbot_sdk.display` builder,
   using a declarative block system

All three result in the same thing: a display spec that the rendering
engine draws to screen. Agent-created displays use the block vocabulary
described in this document. Developer-authored displays can use the same
blocks or write pygame directly.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Agent (via boxbot_sdk.display)                      │
│  "I want a weather board with a clock and forecast"  │
└────────────────────┬────────────────────────────────┘
                     │ display spec (JSON)
                     ▼
┌─────────────────────────────────────────────────────┐
│  Display Manager (src/boxbot/displays/manager.py)    │
│                                                      │
│  ┌────────────────┐  ┌────────────────────────────┐ │
│  │ Data Source Mgr │  │ Layout Engine              │ │
│  │                 │  │                            │ │
│  │ Schedules fetch │  │ Resolves block tree into   │ │
│  │ per source      │  │ positioned render commands │ │
│  │ Caches results  │  │ Text wrapping, truncation  │ │
│  │ Maps & normals  │  │ Applies theme tokens       │ │
│  └───────┬─────────┘  └──────────┬─────────────────┘ │
│          │ data                   │ layout             │
│          ▼                       ▼                    │
│  ┌────────────────────────────────────────────────┐  │
│  │ Renderer (pygame)                              │  │
│  │                                                │  │
│  │ Two update paths:                              │  │
│  │   LIVE blocks (clock, countdown) → self-tick   │  │
│  │   DATA blocks → re-render on source refresh    │  │
│  │                                                │  │
│  │ Animation engine (fade, count_up, slide)       │  │
│  │ Dirty rectangling for efficient partial redraws│  │
│  │ Frame rate: 30fps animating, 1fps live, 0 idle │  │
│  └────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Block Library

Displays are composed from a fixed set of blocks. Layout containers
arrange children. Content blocks display data. The agent never specifies
pixel coordinates — containers handle all positioning.

### Layout Containers (7)

| Block | Parameters | Purpose |
|-------|-----------|---------|
| `row` | `gap`, `align`, `padding` | Horizontal flow — children side by side |
| `column` / `stack` | `gap`, `align`, `padding` | Vertical flow — children stacked |
| `columns` | `ratios`, `gap`, `padding` | Multi-column layout with weight ratios |
| `card` | `color`, `radius`, `padding` | Surface with background, rounded corners, shadow |
| `spacer` | `size` | Fixed or flexible space between elements |
| `divider` | `color`, `thickness`, `orientation` | Separator line (horizontal or vertical) |
| `repeat` | `source`, `max` | Iterate over data array, stamp template per item |

#### Column Ratios

`columns` accepts a list of integer ratios, not fixed pixel widths:

```python
d.columns([2, 1])        # 2/3 + 1/3
d.columns([1, 1])         # equal halves
d.columns([1, 2, 1])      # 25% + 50% + 25%
d.columns([3, 1])          # 75% + 25%
```

The layout engine converts ratios to pixel widths based on available
space minus gaps and padding. The agent never thinks in pixels.

#### Alignment

All containers support `align`:
- `start` — pack children to the start (left for row, top for column)
- `center` — center children
- `end` — pack to end
- `spread` — distribute evenly with space between

#### Padding

Padding accepts a single value or `[vertical, horizontal]` or
`[top, right, bottom, left]`:

```python
d.row(padding=24)              # 24px all sides
d.row(padding=[16, 24])        # 16px top/bottom, 24px left/right
d.row(padding=[16, 24, 24, 24]) # explicit per-side
```

#### The `repeat` Block

`repeat` iterates over a bound data array and stamps a template for
each item. Fields on the current item use `{.field}` syntax:

```python
forecast = weather_card.row(gap=16, align="spread")
day = forecast.repeat("{weather.forecast}", max=5)
day_col = day.column(align="center", padding=[8, 12])
day_col.text("{.day}", size="caption", color="muted")
day_col.icon("{.icon}", size="sm")
day_col.text("{.high}°/{.low}°", size="small")
```

When used with `rotate` (see Meta Blocks), `repeat` supports
`highlight_active=True` — the active item receives the theme's
active styling automatically.

### Content Blocks (13)

#### Text

```python
d.text(content, size, color, weight, align, max_lines, animation)
```

| Param | Values | Default |
|-------|--------|---------|
| `size` | `title`, `heading`, `subtitle`, `body`, `caption`, `small` | `body` |
| `color` | `default`, `muted`, `dim`, `accent`, `success`, `warning`, `error` | `default` |
| `weight` | `normal`, `medium`, `semibold`, `bold` | from theme |
| `align` | `left`, `center`, `right` | `left` |
| `max_lines` | integer or `None` | `None` (unlimited) |
| `animation` | `none`, `fade`, `typewriter` | `none` |
| `min_width` | integer (pixels) | `None` |

Text blocks handle overflow automatically. When `max_lines` is set,
text truncates with an ellipsis. When unset, text wraps within its
container. The agent never needs to worry about collision.

#### Metric

```python
d.metric(value, label, icon, change, change_color, animation)
```

The "big number with context" pattern — temperature, stock price,
battery level, etc.

| Param | Values | Default |
|-------|--------|---------|
| `value` | string (can be data-bound) | required |
| `label` | string | `None` |
| `icon` | Lucide icon name | `None` |
| `change` | string, e.g. `"+2.3%"` | `None` |
| `change_color` | `auto`, or any color token | `auto` |
| `animation` | `none`, `fade`, `count_up` | `none` |

When `change_color="auto"`, the metric block applies `success` for
positive values and `error` for negative. No conditional logic needed.

#### Badge

```python
d.badge(text, color)
```

Small colored label — "Active", "3 new", "Live". Uses the theme's
badge styling (background tint + text color).

#### List

```python
d.list(items, style, icon, max_items)
```

| Param | Values | Default |
|-------|--------|---------|
| `items` | list of strings or data binding | required |
| `style` | `bullet`, `number`, `check`, `none` | `bullet` |
| `icon` | Lucide icon name (replaces bullet) | `None` |
| `max_items` | integer | `None` |

When `max_items` is set and the list is longer, displays "+N more"
at the bottom.

#### Table

```python
d.table(headers, rows, striped, max_rows)
```

Data table with automatic column sizing. Columns auto-size based on
content width. Text within cells truncates with ellipsis if needed.

#### Key-Value

```python
d.key_value(data)
```

Two-column label/value pairs — "Humidity: 65%", "Wind: 12 mph".
Clean aligned layout without a full table.

#### Icon

```python
d.icon(name, size, color)
```

From the Lucide icon library (1400+ icons, MIT licensed). Categories
include weather, home, people, time, status, media, nature, tech, and
navigation.

| Size | Approximate Pixels |
|------|-------------------|
| `sm` | 16px |
| `md` | 24px |
| `lg` | 32px |
| `xl` | 48px |

Icons can be data-bound: `d.icon("{weather.icon}", size="xl")` — the
data source provides the icon name (see Data Binding below).

#### Emoji

```python
d.emoji(name, size)
```

From the Twemoji set (CC-BY licensed), rendered as images for visual
consistency regardless of system font. Size options: `md`, `lg`, `xl`.

#### Image

```python
d.image(source, fit, radius)
```

| Param | Values | Default |
|-------|--------|---------|
| `source` | `photo:id`, `url:https://...`, `asset:filename` | required |
| `fit` | `cover`, `contain`, `fill` | `cover` |
| `radius` | integer (corner radius) | from theme |

#### Chart

```python
d.chart(data, type, color, height, x_labels, show_grid, fill_opacity)
```

Simple charts rendered directly in pygame — lines, rectangles, fills.
No charting library dependency.

**Single series:**
```python
d.chart(data="{stocks.history}", type="line", color="accent", height=250)
```

**Multi-series comparison:**
```python
d.chart(
    series=[
        {"data": "{energy.solar}", "color": "success", "label": "Solar"},
        {"data": "{energy.usage}", "color": "warning", "label": "Usage"},
    ],
    type="area",
    height=250,
    show_legend=True,
    fill_opacity=0.15
)
```

| Param | Values | Default |
|-------|--------|---------|
| `type` | `line`, `bar`, `area` | `line` |
| `color` | theme color token | `accent` |
| `height` | integer (pixels) | `200` |
| `x_labels` | data binding or list | auto |
| `show_grid` | boolean | `True` |
| `show_legend` | boolean | `False` |
| `fill_opacity` | 0.0 - 1.0 (for area charts) | `0.15` |
| `show_dots` | boolean (data points on line) | `False` |

Data format: array of numbers (x-axis is implicit index), or array of
`{x, y}` objects for explicit x-values.

#### Progress

```python
d.progress(value, label, color)
```

Progress/capacity bar. `value` is 0.0 to 1.0 or a data binding.
When `color="auto"`, color shifts based on fill level (green → yellow →
red as it approaches full).

#### Clock (Live Block)

```python
d.clock(format, show_date, show_seconds, size)
```

Self-updating — the renderer ticks it directly, no data source needed.

| Param | Values | Default |
|-------|--------|---------|
| `format` | `12h`, `24h` | `12h` |
| `show_date` | boolean | `True` |
| `show_seconds` | boolean | `False` |
| `size` | `md`, `lg`, `xl` | `lg` |

#### Countdown (Live Block)

```python
d.countdown(target, label)
```

Live countdown to a target datetime. Self-updating like `clock`.

### Composite Widgets (2)

Pre-built, opinionated combinations of primitives for common patterns.
Internally composed from the same blocks, but the agent doesn't need
to build them manually.

| Widget | Purpose |
|--------|---------|
| `weather_widget(data_source)` | Full weather card: icon, temp, condition, forecast row |
| `calendar_widget(data_source)` | Today's agenda with time slots and event details |

### Meta Blocks (2)

#### Rotate

```python
d.rotate(source, key, interval)
```

Cycles through items in a data array on a timer. Makes `{current}`
available as a binding prefix for the active item.

```python
d.rotate(source="portfolio", key="tickers", interval=20)
d.text("{current.symbol}", size="title")
d.chart(data="{current.history}", type="line")
```

Every `interval` seconds, `{current}` advances to the next item.
The display re-renders with the new values (using the configured
data transition animation).

#### Page Dots

```python
d.page_dots(color)
```

Visual indicator showing which item in a `rotate` sequence is active.
Like iOS page dots — small circles, active one highlighted.

## Data Binding

Displays are living templates. Data flows through them automatically —
the agent creates the display once, and it stays current without agent
involvement.

### Binding Syntax

Curly braces reference data source fields:

```python
d.text("{weather.temp}°F")                  # scalar value + literal text
d.icon("{weather.icon}")                     # icon name from data
d.list(items="{tasks.descriptions}")         # array binding
d.chart(data="{stocks.history}")             # array of numbers
```

Nested fields: `{weather.forecast[0].high}`

Current item in `repeat`: `{.field}` (short for current iteration item)

Current item in `rotate`: `{current.field}`

### Data Sources

Every display declares its data sources. The display manager fetches
and caches data on each source's refresh schedule. The agent does NOT
write fetch scripts — data fetching runs in the main process.

#### Built-in Sources (Zero Configuration)

These are always available. They tap into boxBot's existing subsystems:

| Source | Fields | Default Refresh |
|--------|--------|----------------|
| `clock` | `hour`, `minute`, `second`, `display`, `date`, `day_of_week` | Live (renderer ticks directly) |
| `weather` | `temp`, `condition`, `icon`, `humidity`, `wind`, `forecast[]` | 3600s |
| `calendar` | `events[]{time, title, duration, location}` | 300s |
| `tasks` | `items[]{description, due_date, status}`, `count` | 60s |
| `people` | `present[]{name, since}`, `count` | Live (event-driven) |
| `agent_status` | `state`, `last_active`, `next_wake` | Live (event-driven) |

Built-in sources deliver **display-ready data** — including resolved
icon names and semantic color tokens. The weather source maps API
condition codes to Lucide icon names internally (e.g. "Partly Cloudy"
→ `cloud-sun`, "Thunderstorm" → `cloud-lightning`). The display never
sees raw API data.

```python
# Zero config — just declare and bind
d.data("weather")
d.icon("{weather.icon}", size="xl")   # already a Lucide icon name
d.text("{weather.temp}°F", size="title")
```

#### Custom Sources

For agent-created displays that fetch external data:

**`http_json`** — Fetch JSON from a URL:
```python
d.data("stocks", type="http_json",
       url="https://api.polygon.io/v2/snapshot/...",
       params={"tickers": "AAPL,GOOGL,MSFT"},
       secret="polygon_api_key",      # references boxbot_sdk.secrets
       refresh=60)
```

**`http_text`** — Fetch text/HTML from a URL (for scraping or RSS).

**`static`** — Hardcoded values the agent can update later without
rebuilding the display:
```python
d.data("greeting", type="static",
       value={"text": "Have a great day!", "name": "Jacob"})

# Later, in a different conversation:
display.update_data("morning_board", "greeting",
                    value={"text": "Stay warm today!"})
```

**`memory_query`** — Run a memory search:
```python
d.data("reminders", type="memory_query",
       query="household reminders and standing instructions",
       refresh=300)
```

#### Field Extraction and Mapping

Custom HTTP sources often return data in API-specific formats. The
`fields` parameter extracts and normalizes values before they reach
the display:

```python
d.data("stocks", type="http_json",
       url="https://api.polygon.io/v2/snapshot/...",
       secret="polygon_key",
       refresh=60,
       fields={
           # Simple extraction: rename API fields
           "symbol": "ticker",
           "price": "lastPrice",
           "change_pct": "changePercent",

           # Map transform: discrete value → icon name or color
           "trend_icon": {
               "from": "direction",
               "map": {"up": "trending-up", "down": "trending-down", "flat": "minus"}
           },
           "trend_color": {
               "from": "direction",
               "map": {"up": "success", "down": "error", "flat": "muted"}
           }
       })
```

`map` is a static lookup table — no code, no conditionals. The data
source resolves values before they reach the display, so blocks can
bind directly:

```python
d.icon("{current.trend_icon}", color="{current.trend_color}")
```

#### How Icons and Colors Work Without Conditionals

Three layers handle the mapping, none in the display spec:

| Layer | Handles | Example |
|-------|---------|---------|
| Built-in data source | Normalizes API → display-ready data internally | `"Cloudy"` → icon `"cloud"` |
| Custom `fields.map` | Static lookup table in source declaration | `"up"` → icon `"trending-up"` |
| Block semantic defaults | `color="auto"` uses block-specific logic | Positive change = green |

The display spec is purely declarative. All conditional mapping lives
either in the data source implementation or in the block rendering
logic.

### Refresh Architecture

The display manager runs data fetching as async tasks in the main
process. No scripts, no cron, no external schedulers.

```
Display Manager
  ├── weather source    → async fetch every 3600s
  ├── calendar source   → async query every 300s
  ├── stocks source     → async fetch every 60s
  └── clock             → renderer ticks directly
```

When a source refreshes, only blocks bound to that source re-render.
The layout engine tracks which blocks depend on which sources.

Different blocks on the same display can update at completely different
rates:

```
t=0:00  Clock ticks → re-render clock area only
t=0:01  Clock ticks → re-render clock area only
...
t=1:00  Stocks refresh → re-render stock blocks (with count_up animation)
        Clock ticks → re-render clock area
...
t=5:00  Calendar refreshes → re-render calendar column
...
t=60:00 Weather refreshes → re-render weather card
```

#### Three Block Update Categories

| Category | Examples | Update Trigger |
|----------|----------|---------------|
| **Live** | `clock`, `countdown` | Renderer ticks directly (every second) |
| **Data-bound** | `text("{weather.temp}")`, `chart`, `list` | Data source refresh |
| **Static** | `text("Portfolio")`, `icon("sun")` | Never (rendered once) |

#### Error Handling

When a data source fetch fails, the display shows the last successful
data with a subtle staleness indicator (dimmed refresh icon or muted
color shift). The display never goes blank because of a network error.

## Theme System

Themes define colors, typography, spacing, and shape. Every block reads
from the active theme — the agent picks semantic tokens (`accent`,
`muted`, `success`), never hex colors.

### Theme Schema

```yaml
name: string                    # unique identifier
description: string             # human-readable

colors:
  background: color             # main background
  surface: color                # card/container background
  surface_alt: color            # nested card background (subtle depth)
  text: color                   # primary text
  muted: color                  # secondary text
  dim: color                    # tertiary text (timestamps, dots)
  accent: color                 # primary accent (highlights, active items)
  accent_soft: color            # accent at low opacity (tinted backgrounds)
  secondary: color              # secondary accent
  success: color                # positive states
  warning: color                # caution states
  error: color                  # negative states

fonts:
  family: string                # font family name (must be bundled)
  title:    { size: int, weight: int, tracking: float }
  heading:  { size: int, weight: int, tracking: float }
  subtitle: { size: int, weight: int }
  body:     { size: int, weight: int }
  caption:  { size: int, weight: int }
  small:    { size: int, weight: int }

spacing:
  xs: int     # 4
  sm: int     # 8
  md: int     # 16
  lg: int     # 24
  xl: int     # 32

radius: int                     # default corner radius for cards
shadow: boolean                 # subtle drop shadows on cards
icon_style: outline | filled    # icon variant
transition: string              # default display switch animation
```

### Built-in Themes

#### `boxbot` (Default)

The signature theme. Warm, minimal, designed for a wooden enclosure.
Inspired by the warmth of a vintage radio, the clean lines of Apple
products, and Anthropic's warm color palette.

```yaml
name: boxbot
description: "Warm minimal — designed for the wooden enclosure"

colors:
  background: "#191714"         # deep warm charcoal (not cold, not pure black)
  surface: "#252018"            # warm dark card backgrounds
  surface_alt: "#302a20"        # nested depth
  text: "#ede8e0"               # warm off-white cream
  muted: "#8a8078"              # warm gray
  dim: "#5a5550"                # very subtle
  accent: "#d4845a"             # warm amber-coral (Anthropic-inspired)
  accent_soft: "#d4845a22"      # accent at low opacity
  secondary: "#c4a46c"          # soft gold (tube glow warmth)
  success: "#7a9e6c"            # earthy sage green
  warning: "#d4a043"            # warm amber
  error: "#c45c5c"              # muted red

fonts:
  family: "Inter"
  title:    { size: 42, weight: 700, tracking: -0.02 }
  heading:  { size: 28, weight: 600, tracking: -0.01 }
  subtitle: { size: 22, weight: 500 }
  body:     { size: 18, weight: 400 }
  caption:  { size: 15, weight: 400 }
  small:    { size: 13, weight: 400 }

spacing:
  xs: 4
  sm: 8
  md: 16
  lg: 24
  xl: 32

radius: 14
shadow: true
icon_style: outline
transition: crossfade
```

#### `midnight`

Near-black for nighttime. Only essentials visible. Paired with HAL
brightness dimming.

```yaml
name: midnight
description: "Near-black for nighttime — only essentials visible"

colors:
  background: "#0c0b0a"
  surface: "#141210"
  surface_alt: "#1a1816"
  text: "#6a6560"
  muted: "#3a3835"
  dim: "#2a2825"
  accent: "#8a6040"
  accent_soft: "#8a604015"
  secondary: "#7a6840"
  success: "#4a6a44"
  warning: "#8a7030"
  error: "#7a3a3a"

# Same font family, same sizes, same spacing
radius: 14
shadow: false                   # no shadows in near-dark
icon_style: outline
transition: crossfade
```

#### `daylight`

Warm cream backgrounds for well-lit rooms. High contrast for
readability in bright conditions.

```yaml
name: daylight
description: "Warm cream for bright rooms — high contrast readability"

colors:
  background: "#f5f0e6"         # warm linen
  surface: "#ffffff"
  surface_alt: "#ece7dd"
  text: "#2a2218"               # dark warm brown
  muted: "#7a7068"
  dim: "#a09890"
  accent: "#c46a3c"             # deeper amber on light backgrounds
  accent_soft: "#c46a3c18"
  secondary: "#8a7040"
  success: "#4a7a40"
  warning: "#b08020"
  error: "#b04040"

radius: 14
shadow: true
icon_style: outline
transition: crossfade
```

#### `classic`

Leans into the vintage aesthetic. Amber-tinted text recalls old
instruments and nixie tube displays.

```yaml
name: classic
description: "Vintage radio — amber on dark, warm and nostalgic"

colors:
  background: "#141008"
  surface: "#1e1810"
  surface_alt: "#282018"
  text: "#d4b88c"               # amber-tinted text
  muted: "#8a7860"
  dim: "#5a5040"
  accent: "#e0943c"             # strong amber
  accent_soft: "#e0943c20"
  secondary: "#c49030"          # gold
  success: "#6a9050"
  warning: "#c48830"
  error: "#a84040"

radius: 10                      # slightly less rounded
shadow: true
icon_style: outline
transition: crossfade
```

### Community Themes

Community-contributed themes are YAML files dropped into `themes/`
at the project root. The theme must bundle any non-standard fonts in
`themes/{name}/fonts/`. Standard font: Inter (always available).

```
themes/
  my_theme.yaml              # simple: uses Inter
  retro_terminal/
    theme.yaml               # custom font
    fonts/
      VT323-Regular.ttf      # bundled font
```

The agent sets themes per display:
```python
d.set_theme("boxbot")        # use the default
d.set_theme("classic")       # use the vintage theme
```

If unset, the system default (from `config.yaml`) applies.

## Animation

Subtle motion makes the display feel alive without being distracting.
All animations are predefined — the agent picks from a menu, never
defines keyframes or timing curves.

### Data Transition Animations

Applied to content blocks when their bound data changes:

| Animation | Effect | Good For |
|-----------|--------|----------|
| `fade` | Crossfade old → new value | Text, icons, any value |
| `count_up` | Number rolls from old to new | Temperatures, prices, percentages |
| `slide_up` | New value slides in from below | Status changes, notifications |

Usage:
```python
d.metric(value="{weather.temp}°F", animation="count_up")
d.text("{stocks.symbol}", animation="fade")
```

### Display Switch Transitions

Applied when the display manager switches between displays:

| Transition | Effect |
|------------|--------|
| `crossfade` | Smooth opacity blend (default) |
| `slide_left` | Current slides out left, new slides in from right |
| `slide_right` | Current slides out right, new slides in from left |
| `none` | Instant switch |

Set per-theme (default) or per-display:
```python
d.set_transition("slide_left")
```

### Frame Rate Management

Rendering adapts to what's happening:

| State | Frame Rate | Trigger |
|-------|-----------|---------|
| Animating | 30 fps | Data transition or display switch in progress |
| Live blocks active | 1 fps | Clock with seconds showing |
| Clock without seconds | 1/60 fps | Clock updates once per minute |
| Static display | 0 fps | Sleep until next data source refresh |
| Screen off | No rendering | Night mode or manual off |

## Preview Workflow

The agent cannot see what it creates. The preview system lets it
render a display to a PNG, view the image (multimodal), and iterate.

### How It Works

```python
d = display.create("morning_dashboard")
# ... build layout ...

preview = d.preview()    # SDK action: render to PNG, return path
```

1. `d.preview()` emits an SDK action to stdout
2. `execute_script` tool renders the spec to a 1024x600 PNG
3. For data-bound blocks, the preview uses current data source values
   (or placeholder values if sources aren't configured yet)
4. The image path is returned in the tool response
5. The agent sees the image and evaluates the result
6. If changes needed → modify and preview again
7. When satisfied → `d.save()`

Preview is optional but recommended for agent-created displays. Each
preview costs one additional tool call. For displays created rarely
(not every conversation), this is a worthwhile tradeoff for visual
quality.

### Placeholder Data

When previewing a display before data sources are live, the renderer
fills bindings with plausible sample data:

- `{weather.temp}` → `"72"`
- `{weather.condition}` → `"Partly Cloudy"`
- `{weather.icon}` → `"cloud-sun"`
- `{calendar.events}` → 3 sample events
- `{tasks.items}` → 3 sample tasks

This ensures the agent sees a realistic layout, not empty boxes.

## Display Lifecycle

### Creation (Agent)

```
1. Agent loads display_authoring skill (documentation + examples)
2. Agent writes SDK script using display builder
3. Agent calls d.preview() → views PNG → iterates
4. Agent calls d.save() → SDK emits display spec as JSON
5. execute_script tool writes spec to displays/{name}/display.json
6. Display queued for user approval (auto_activate_displays: false)
7. Agent informs user and asks if they want to activate it
8. User confirms → display enters rotation or becomes active
```

### Activation & Rotation

The display manager handles two modes:

**Active display** — explicitly selected by the agent via
`switch_display`. Stays active until the agent switches again or
idle timeout triggers rotation.

**Idle rotation** — configured in `config.yaml`. The display manager
cycles through a list of displays during idle/sleep state:

```yaml
display:
  rotation_interval: 30
  idle_displays:
    - "clock"
    - "weather_simple"
  brightness: 0.8
  night_mode:
    enabled: true
    start: "22:00"
    end: "07:00"
    brightness: 0.3
```

### Data Source Lifecycle

Data sources are tied to their display:
- Display activated → sources start fetching
- Display deactivated → sources pause (stop fetching)
- Display in idle rotation → sources stay warm if in rotation list
- Display deleted → sources cleaned up

No orphaned fetchers. No background scripts to manage.

### Updating Data Without Rebuilding

The agent can update a display's static data sources without
rebuilding the layout:

```python
display.update_data("morning_board", "greeting",
                    value={"text": "Stay warm today!"})
```

This changes what flows through the pipe without touching the layout.
No preview needed, no approval needed.

## Display Authoring Skill

The full block library documentation lives in a skill, not the
always-loaded system prompt. This keeps the agent's context slim
(9 tools) while providing comprehensive guidance when needed.

```yaml
name: display_authoring
description: "Create, modify, and preview displays for boxBot's screen"
triggers:
  - "create a display"
  - "make a screen"
  - "new display"
  - "show me"
  - "put on the screen"
```

The skill document includes:
- Complete block reference with all parameters
- Layout examples for common patterns
- Data binding documentation with examples
- Icon library reference (categorized by domain)
- Theme reference
- The preview workflow
- Common recipes: weather board, task dashboard, morning briefing,
  stock tracker, energy dashboard, greeting card, photo wall

## Extension Points

### New Blocks

Community-contributed blocks go in `src/boxbot/displays/blocks/`.
Each block is a Python class that:
1. Declares its parameters and their types
2. Implements `measure(available_width, available_height)` → returns
   needed size
3. Implements `render(surface, rect, theme, data)` → draws itself

### New Themes

YAML files in `themes/`. Must follow the theme schema. Can bundle
custom fonts.

### New Data Source Types

Implement the data source protocol:
1. `fetch(config) → dict` — fetch and return normalized data
2. `get_refresh_interval(config) → int` — seconds between fetches

### New Composite Widgets

Combine existing blocks into reusable patterns. Registered in the
block library so the agent can use them like built-in blocks.

## Use Case Walkthroughs

### Morning Dashboard

A clock, weather, and calendar on one screen. Left column (2/3 width)
houses the clock and weather. Right column (1/3 width) shows the day's
schedule.

```python
d = display.create("morning_dashboard")
d.set_theme("boxbot")
d.data("weather")
d.data("calendar")

header = d.row(padding=[16, 24], align="center")
header.emoji("sunrise")
header.text("Good Morning", size="title")
header.spacer()
header.clock(format="12h", show_date=False, size="subtitle")

body = d.columns([2, 1], gap=24, padding=[0, 24, 24, 24])

# Left — clock and weather
left = body.column(gap=20)

time_card = left.card()
time_card.clock(format="12h", show_date=True, show_seconds=False, size="xl")

weather_card = left.card()
top = weather_card.row(gap=16, align="center")
top.icon("{weather.icon}", size="xl")
info = top.column()
info.text("{weather.temp}°F", size="title", animation="count_up")
info.text("{weather.condition}", size="body", color="muted")
forecast = weather_card.row(gap=0, align="spread")
day = forecast.repeat("{weather.forecast}", max=5)
day_col = day.column(align="center", padding=[8, 12])
day_col.text("{.day}", size="caption", color="muted")
day_col.icon("{.icon}", size="sm")
day_col.text("{.high}°/{.low}°", size="small")

# Right — today's schedule
right = body.column()
right.text("Today", size="heading")
right.spacer(8)
right.divider()
slot = right.repeat("{calendar.events}", max=8)
slot_row = slot.row(gap=12, padding=[10, 0])
slot_row.text("{.time}", size="caption", color="muted", min_width=50)
slot_row.divider(orientation="vertical", color="accent")
slot_body = slot_row.column()
slot_body.text("{.title}", size="body", max_lines=1)
slot_body.text("{.duration}", size="small", color="muted")
```

### Stock Tracker with Rotation

Rotates through stocks every 20 seconds. Each shows a price chart.
Bottom ticker shows all symbols with the active one highlighted.

```python
d = display.create("stock_tracker")
d.set_theme("boxbot")

d.data("portfolio", type="http_json",
       url="https://api.polygon.io/v2/snapshot/...",
       params={"tickers": "AAPL,GOOGL,MSFT,AMZN"},
       secret="polygon_api_key",
       refresh=60,
       fields={
           "symbol": "ticker",
           "price": "lastPrice",
           "change_pct": "changePercent",
           "trend_icon": {
               "from": "direction",
               "map": {"up": "trending-up", "down": "trending-down", "flat": "minus"}
           },
           "trend_color": {
               "from": "direction",
               "map": {"up": "success", "down": "error", "flat": "muted"}
           }
       })

d.data("intraday", type="http_json",
       url="https://api.polygon.io/v2/aggs/ticker/{current.symbol}/...",
       secret="polygon_api_key",
       refresh=300)

d.rotate(source="portfolio", key="tickers", interval=20)

header = d.row(padding=[16, 24], align="center")
header.text("{current.symbol}", size="title", weight="bold")
header.text("{current.name}", size="subtitle", color="muted")
header.spacer()
header.metric(value="${current.price}",
              change="{current.change_pct}%",
              animation="count_up")

d.chart(data="{intraday.results}",
        type="area",
        color="accent",
        height=320,
        padding=[0, 24],
        show_grid=True,
        fill_opacity=0.15)

footer = d.row(padding=[16, 24], gap=0, align="spread")
stock = footer.repeat("{portfolio.tickers}", highlight_active=True)
chip = stock.column(align="center", padding=[8, 16])
chip.text("{.symbol}", size="caption", weight="bold")
chip.text("{.change_pct}%", size="small")

d.page_dots(color="accent")
```

### Home Energy Dashboard

Real-time solar production vs usage with live metrics and a
comparison chart. No rotation — everything on one screen.

```python
d = display.create("energy_dashboard")
d.set_theme("boxbot")

d.data("energy", type="http_json",
       url="https://my-enphase.local/api/stats",
       secret="enphase_key",
       refresh=120)

header = d.row(padding=[16, 24], align="center")
header.icon("zap", size="lg", color="accent")
header.text("Home Energy", size="title")

body = d.columns([1, 1], gap=24, padding=[0, 24, 24, 24])

left = body.column(gap=16)
left.text("Right Now", size="heading")
left.metric(value="{energy.solar_kw} kW", label="Solar",
            icon="sun", color="success", animation="count_up")
left.metric(value="{energy.usage_kw} kW", label="Usage",
            icon="home", color="warning", animation="count_up")
left.metric(value="{energy.net_kw} kW", label="Net Export",
            icon="arrow-up-right", color="accent", animation="count_up")
left.spacer()
left.progress(value="{energy.battery_pct}",
              label="Battery {energy.battery_pct}%",
              color="success")

right = body.column(gap=12)
right.text("Today", size="heading")
right.chart(
    series=[
        {"data": "{energy.solar_hourly}", "color": "success", "label": "Solar"},
        {"data": "{energy.usage_hourly}", "color": "warning", "label": "Usage"},
    ],
    type="area",
    height=280,
    x_labels="{energy.hours}",
    show_legend=True,
    fill_opacity=0.1
)
```
