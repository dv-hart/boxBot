# bb.display — the 7" screen

Displays are what the humans in the room see. The screen cycles through
idle displays (clock, weather, slideshow) by default; calling
`bb.display.show()` or using the `switch_display` tool takes control.

## When to use it

- User asks to see something ("put the weather up", "show me tonight's
  schedule").
- Responding to a prompt visually instead of (or alongside) speaking.
- Authoring a new display layout.

## When NOT to use it

- For output the *agent* needs to see (that goes through the tool
  result as image content blocks — use `bb.workspace.view`,
  `bb.camera.capture`, or `bb.photos.view`).
- For rapid-fire changes. Display switches involve teardown/setup of
  data sources; don't cycle them faster than ~1/sec.

## Current built-in displays

- `clock` — time + date, full-screen
- `weather_simple` — current conditions
- `picture` — shows photos from the library; takes
  `args.image_ids = [<photo_id>, …]`

## API

### Switch to a named display

The primary entry point is the built-in `switch_display` tool (always
available, no `bb` import needed):

```python
# From inside execute_script:
import boxbot_sdk as bb
bb.display.show("weather_simple")
bb.display.show("picture", args={"image_ids": ["abc123..."]})
```

Or use the tool directly for the common case:

```
switch_display("clock")
```

Both paths route through the same display manager, so `args` flows
through and gets exposed to the display's bindings as the `args` data
source (e.g. `{args.image_ids[0]}`).

### Preview a layout before committing

```python
img = bb.display.preview(spec_dict)  # returns PNG bytes
bb.workspace.write("drafts/weather_v2.png", img)
bb.workspace.view("drafts/weather_v2.png")   # look at it
```

This renders a display spec without putting it on-screen, so you can
iterate before users see it.

### Author a new display

```python
spec = bb.display.build(
    name="daily_summary",
    theme="boxbot",
    layout=bb.display.column(
        gap=12,
        children=[
            bb.display.clock_block(format="12h", show_date=True),
            bb.display.text("You have {tasks.open} open items", size="body"),
        ],
    ),
    data_sources=[
        bb.display.source("tasks", builtin=True, refresh=60),
    ],
)
bb.display.save(spec)   # requires user confirmation
```

User-authored displays need a confirmation on the screen or via
WhatsApp admin before they activate — same approval gate as package
installs.

## Data binding

Any string param on any block can include `{source.field}` bindings.
Resolved at render time from:

- `args.<field>` — the `args={}` passed to `switch_display`.
- `<source_name>.<field>` — data sources declared on the display spec
  (built-ins: `weather`, `calendar`, `tasks`, `people`, `agent_status`;
  or `http_json` / `static` / `memory_query` custom sources).

Array indexing works: `{args.image_ids[0]}`, `{weather.forecast[2].high}`.

## Patterns

### Put a specific photo on screen

```python
results = bb.photos.search(query="Emily birthday")
if results:
    bb.display.show("picture", args={"image_ids": [results[0].id]})
```

This is what `bb.photos.show_on_screen(ids)` does under the hood.

### Back to idle

The display manager auto-returns to the idle rotation when you don't
specify a display for a while; no explicit "unswitch" call is needed.

## Known gaps

- Display authoring SDK (`bb.display.build/preview/save`) surface is
  partially wired. `show()` works; `preview()` and `build()` actions
  are acknowledged but not fully implemented yet.
- When the display manager isn't running (dev, startup race), `show()`
  returns `{status: "error", error: "display manager not running"}`
  rather than silently pretending success.
