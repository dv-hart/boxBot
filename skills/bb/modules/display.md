# bb.display — the 7" screen

A display is a JSON document. Read it, edit the dict, write it back. No
builder, no fluent API — spec dicts plus the SDK calls that load,
preview, save, and list them.

**Use for:** showing something the user asked for (`switch_display` if
it already exists); authoring a layout that doesn't exist yet (build a
dict, `preview()`, fix warnings, `save()`); editing an existing one
(`load()` returns a dict you mutate).

**Not for:** output *you* need to see → `bb.workspace.view`,
`bb.camera.capture`, `bb.photos.view`. Rapid cycling — switches tear
down and set up data sources; stay under ~1/sec.

## Built-ins

- `clock` — full-screen time + date.
- `weather_simple` — current conditions.
- `picture` — photo viewer. One id = static, several = slideshow.
  `args.image_ids: [str, ...]` (required), `args.interval: int`
  (seconds, default 8, used with 2+ ids).
- `notice` — short centered card (`args.title` + `args.lines`).
- Plus everything in `displays/` and `data/displays/`.

`bb.display.list()` enumerates all of them.

## SDK surface

| Call | Returns | Purpose |
| --- | --- | --- |
| `bb.display.list()` | `[{name, source}, …]` | enumerate |
| `bb.display.get_active()` | `{name, args, theme, pinned, rotation}` | what's on screen + pin/rotation state |
| `bb.display.unpin()` | `{pinned, rotation}` | release pin, resume rotation |
| `bb.display.set_rotation(displays=, interval=)` | `{pinned, rotation}` | configure idle rotation (clears pin) |
| `bb.display.screenshot()` | `{path, attached, name}` | live screen → PNG, attached |
| `bb.display.load(name)` | `dict` | read a spec for editing |
| `bb.display.save(spec)` | `{path, registered, warnings}` | validate + write + register live |
| `bb.display.preview(spec, data=None)` | `{path, attached, warnings}` | render PNG with placeholder data |
| `bb.display.delete(name)` | — | remove an agent-saved display |
| `bb.display.describe_source(name)` | `{fields, example}` | data source field schema |
| `bb.display.schema()` | `{blocks, themes, …}` | full block reference |
| `bb.display.update_data(display, source, value=)` | `{}` | push values into a `static` source |

`save` and `preview` validate first. On error they raise `RuntimeError`
listing every problem. On success they return `warnings`:

- A binding (`{source.field}`) that didn't resolve. Usually a typo;
  sometimes legitimate (an `http_json` source whose first fetch hasn't
  happened — clears once live data lands).
- An icon `name` outside the bundled Lucide subset (renders as a
  circled-letter placeholder). Enumerate with
  `bb.display.schema()["icons"]`.

**Warnings check placeholder data, not real data.** A binding that
resolves against a placeholder may render empty when a real fetch
returns a sparser shape (a calendar event with no `location`). Sanity-
check live renders after `switch_display`.

## Switching

The always-loaded tool — no script needed:

```
switch_display("morning_brief")
switch_display("picture", args={"image_ids": ["abc123..."]})
```

`args` binds as `{args.<field>}`.

### Pin and rotation

`switch_display` **pins by default**. The display holds and idle
rotation pauses until you replace it or unpin. No auto-revert.

```python
state = bb.display.get_active()
# {"name": "picture", "args": {"image_ids": [...]},
#  "theme": "boxbot", "pinned": True,
#  "rotation": {"active": False, "displays": [...],
#               "interval": 30, "next_in_sec": None}}

bb.display.unpin()                                        # resume rotation
bb.display.set_rotation(displays=["picture"], interval=120)
switch_display("weather_simple", pin=False)               # preview without taking control
```

A daily rhythm, one trigger per phase:

| Time | Agent calls | Effect |
| --- | --- | --- |
| 07:00 | `switch_display("morning_brief")` | pinned digest |
| 09:00 | `bb.display.unpin()` | rotation resumes |
| 22:00 | `switch_display("picture", args={"image_ids": [...], "interval": 10})` | pinned slideshow |

Slideshow: pass 2+ ids in `image_ids`; the manager rotates every
`interval` seconds (default 8). One id renders static. For "all
slideshow-tagged photos," gather ids with `bb.photos.search(...)` first.

Do **not** build a slideshow with the `rotate` block — the renderer
only draws its first child. The picture display's built-in cycle is the
supported path.

## Spec shape

```json
{
  "name": "morning_glance",
  "theme": "boxbot",
  "transition": "crossfade",
  "data_sources": [
    {"name": "weather", "type": "integration", "refresh": 3600},
    {"name": "calendar", "type": "integration",
     "inputs": {"action": "list_upcoming_events", "max_results": 5}},
    {"name": "tasks"}
  ],
  "layout": {
    "type": "column",
    "padding": 24,
    "gap": 16,
    "children": [
      {"type": "row", "align": "spread", "children": [
        {"type": "clock", "format": "12h", "show_date": false, "size": "lg"},
        {"type": "text", "content": "{clock.day_of_week}, {clock.date}",
         "size": "caption", "color": "muted"}
      ]},
      {"type": "card", "color": "muted", "padding": 18, "children": [
        {"type": "row", "gap": 18, "children": [
          {"type": "icon", "name": "{weather.icon}", "size": "xl",
           "color": "accent"},
          {"type": "metric", "value": "{weather.temp}°",
           "label": "{weather.condition}"}
        ]}
      ]},
      {"type": "card", "color": "muted", "padding": 18, "children": [
        {"type": "column", "gap": 8, "children": [
          {"type": "text", "content": "NEXT 2", "size": "small",
           "weight": "semibold"},
          {"type": "repeat", "source": "{calendar.events}", "max": 2,
           "children": [{"type": "row", "gap": 12, "children": [
             {"type": "text", "content": "{.time}", "color": "accent",
              "weight": "bold"},
             {"type": "text", "content": "{.title}"}
           ]}]}
        ]}
      ]}
    ]
  }
}
```

| Key | Type | Notes |
| --- | --- | --- |
| `name` | str | unique, alphanumeric + `_-` |
| `theme` | str | `boxbot` (default) / `midnight` / `daylight` / `classic` |
| `transition` | str | optional. `crossfade` (default), `slide_left`, `slide_right`, `none` |
| `data_sources` | list[dict] | declared feeds, below |
| `layout` | dict | the block tree |

`layout` is **one** block. Use `column` or `row` to host several
children.

## Blocks

Every block has `"type"` plus a flat bag of config fields. Containers
also take `"children"`.

`bb.display.schema()` returns every field, default, and valid-values
list for every block. Use it instead of guessing.

### Containers

| `type` | Fields | Notes |
| --- | --- | --- |
| `row` | `gap`, `align`, `padding` | horizontal. `align`: start/center/end/spread |
| `column` / `stack` | `gap`, `align`, `padding` | vertical |
| `columns` | `ratios` (list[int]), `gap`, `padding` | weighted. `ratios=[2,1]` = 2/3 + 1/3 |
| `card` | `color`, `radius`, `padding` | **invisible without `color=`**. `"muted"` = subtle surface |
| `spacer` | `size` (int, or omit for flexible) | fixed or stretchy gap |
| `divider` | `color`, `thickness`, `orientation` | `orientation`: horizontal/vertical |
| `repeat` | `source` (binding), `max`, `highlight_active` | iterate an array. Single child = template; bind item fields with `{.field}` |

### Content

| `type` | Required | Optional | Notes |
| --- | --- | --- | --- |
| `text` | `content` | `size`, `color`, `weight`, `align`, `max_lines`, `animation`, `min_width` | `size`: title/heading/subtitle/body/caption/small. `color`: default/muted/dim/accent/success/warning/error. `weight`: normal/medium/semibold/bold. `align`: left/center/right |
| `metric` | `value` | `label`, `icon`, `change`, `change_color`, `animation` | intrinsically big — **no `size=`**. Value uses theme `text`; only `change_color` is configurable |
| `badge` | `text` | `color` | small colored label |
| `list` | `items` (list or binding) | `style`, `icon`, `max_items` | `style`: bullet/number/check/none |
| `table` | `headers`, `rows` | `striped`, `max_rows` | both can be bindings |
| `key_value` | `data` (dict or binding) | — | label/value pairs |
| `icon` | `name` | `size`, `color` | Lucide names. `size`: sm/md/lg/xl |
| `emoji` | `name` | `size` | Twemoji. `size`: md/lg/xl |
| `image` | `source` | `fit`, `radius` | `source`: `"photo:<id>"` / `"url:..."` / `"asset:..."` |
| `chart` | `data` *or* `series` | `type`, `color`, `height`, `x_labels`, `show_grid`, `show_legend`, `fill_opacity`, `show_dots`, `padding` | `type`: line/bar/area |
| `progress` | `value` (0..1 or binding) | `label`, `color` | `color="auto"` shifts green→yellow→red |
| `clock` | — | `format`, `show_date`, `show_seconds`, `size` | live. `size`: md/lg/xl |
| `countdown` | `target` | `label` | live, counts down to a datetime string |

### Three `size` taxonomies — do not mix

| Block | `size` values |
| --- | --- |
| `text` | semantic: title, heading, subtitle, body, caption, small |
| `icon` | t-shirt: sm, md, lg, xl |
| `emoji` | t-shirt: md, lg, xl |
| `clock` | t-shirt: md, lg, xl |
| `metric` | none — already large |

`text(size="title")` is **bigger** than `clock(size="xl")`. The scales
are not comparable. Pick visually.

## Data sources

Declare once at the top level, bind anywhere with `{source.field}`.

### Built-ins, zero config

```json
{"name": "tasks"}
{"name": "people"}
{"name": "agent_status"}
{"name": "clock"}
```

These read live in-process state — the scheduler's to-do list, present
people from perception, agent state, the clock. They cannot be
integrations; the data never leaves the main process.

`bb.display.describe_source("tasks")` lists the fields
(`items[].description`, `count`, …). The doc rots; the schema doesn't.

### External: `integration`

Everything that talks to the outside world flows through one type:

```json
{"name": "weather", "type": "integration", "refresh": 3600}
{"name": "calendar", "type": "integration",
 "inputs": {"action": "list_upcoming_events", "max_results": 5},
 "refresh": 600}
{"name": "solar", "type": "integration",
 "inputs": {"date": "2026-05-15"}, "refresh": 3600}
```

The manager calls `bb.integrations.get(<name>, **inputs)` on cadence
and binds the output dict to the source name, so `{weather.temp}`,
`{calendar.events[0].title}`, `{solar.kwh}` work.

- `integration` — override which integration to call (defaults to
  `name`). Lets one integration appear under several bindings.
- `inputs` — passed verbatim. Manifests declare defaults and
  `default_env` fallbacks for device config like `lat`/`lon`
  (`BOXBOT_WEATHER_LAT` / `BOXBOT_WEATHER_LON`), so you rarely repeat
  them per display.
- `refresh` — seconds between fetches. Default 300.

Pre-seeded integrations and ones you author share this path. No
privileged track. `bb.integrations.list()` shows what exists;
`bb.display.describe_source(name)` reads the manifest `outputs` for you.

### `http_json`

```json
{
  "name": "stocks", "type": "http_json",
  "url": "https://api.example.com/quote",
  "params": {"symbol": "AAPL"},
  "secret": "STOCKS_API_KEY",
  "refresh": 300,
  "fields": {
    "price": "data.current.price",
    "trend": {"from": "data.change",
              "map": {"+": "trending-up", "-": "trending-down"}}
  }
}
```

`secret` is the **name** of a secret, never the value. The manager
looks it up via `bb.secrets` at fetch time and sends it as a Bearer
token. Store the key once with
`bb.secrets.store("STOCKS_API_KEY", "…")`.

Bind `{stocks.price}` and `{stocks.trend}` (the latter as an icon
`name`).

To verify `fields` without a real fetch, pass a fixture:

```python
bb.display.preview(spec, data={
    "stocks": {"data": {"current": {"price": "184.20"}, "change": "+"}}
})
```

The override layers onto normal data assembly, so `fields` runs against
your fixture and warnings reflect real behavior.

### `http_text`

```json
{"name": "page", "type": "http_text", "url": "https://example.com"}
```

Bind `{page.text}` — the whole body.

### `static`

Hardcoded values you can change later via `bb.display.update_data(...)`:

```json
{"name": "session", "type": "static",
 "value": {"task": "writing", "minutes": 0, "progress": 0.0}}
```

### `memory_query`

Re-runs a memory search on every refresh — a standing "household
reminders" board. Hybrid vector + keyword only (no model reranking, no
conversation summaries), so refreshing is free:

```json
{"name": "recent", "type": "memory_query",
 "query": "kitchen renovation", "refresh": 600, "limit": 5}
```

`limit` default 5 — screen space is small. Output: `results` (array of
`{text, type, age}`; `text` is the summary, `type` is
person/household/methodology, `age` is `"3d"` / `"2w"`), `count`,
`query`.

```json
{"type": "repeat", "source": "{recent.results}",
 "children": [{"type": "row", "gap": 12, "children": [
   {"type": "text", "content": "{.text}"},
   {"type": "text", "content": "{.age}", "color": "muted"}
 ]}]}
```

## Bindings

Any string in any block can contain `{source.field}`, resolved at
render time:

- `args.<field>` — the `args={}` from `switch_display`.
- `<source>.<field>` — a declared source. Indexing works:
  `{calendar.events[0].title}`, `{weather.forecast[2].high}`.
- `{.field}` — current item inside a `repeat`.
- `{current.field}` — active item inside a `rotate`.

A string that is *entirely* one binding passes the raw value through
(so arrays reach `list` / `table` intact). A mixed string
(`"{weather.temp}°F"`) stringifies.

## Themes

| Theme | Mood | Background |
| --- | --- | --- |
| `boxbot` | warm amber-coral, wooden enclosure | dark warm |
| `midnight` | cool indigo, low ambient light | dark cool |
| `daylight` | bright daytime contrast | light |
| `classic` | high-contrast neutral | light |

`boxbot` and `midnight` both read dark — preview after a theme change,
don't trust the JSON. **Never put `color="muted"` text inside a
`card(color="muted")`** — same surface tone, the text vanishes.

## Authoring

```python
import boxbot_sdk as bb

spec = {
    "name": "morning_glance",
    "theme": "boxbot",
    "data_sources": [
        {"name": "calendar", "type": "integration",
         "inputs": {"action": "list_upcoming_events", "max_results": 5}},
        {"name": "weather", "type": "integration"},
    ],
    "layout": {
        "type": "column", "padding": 24, "gap": 16,
        "children": [
            {"type": "row", "align": "spread", "children": [
                {"type": "clock", "format": "12h", "show_date": False,
                 "size": "lg"},
                {"type": "text",
                 "content": "{clock.day_of_week}, {clock.date}",
                 "size": "caption", "color": "muted"},
            ]},
            {"type": "card", "color": "muted", "padding": 18, "children": [
                {"type": "row", "gap": 18, "children": [
                    {"type": "icon", "name": "{weather.icon}", "size": "xl",
                     "color": "accent"},
                    {"type": "metric", "value": "{weather.temp}°",
                     "label": "{weather.condition}"},
                ]},
            ]},
        ],
    },
}

result = bb.display.preview(spec)   # check result["warnings"], PNG auto-attaches
bb.display.save(spec)               # validate + write + register live
```

## Editing

```python
spec = bb.display.load("morning_glance")   # → dict
spec["theme"] = "midnight"

spec["layout"]["children"][2] = {
    "type": "card", "color": "muted", "padding": 18, "children": [
        {"type": "column", "gap": 4, "children": [
            {"type": "text", "content": "UP NEXT", "size": "small",
             "color": "muted", "weight": "semibold"},
            {"type": "row", "gap": 12, "children": [
                {"type": "text", "content": "{calendar.events[0].time}",
                 "size": "subtitle", "color": "accent"},
                {"type": "text", "content": "{calendar.events[0].title}"},
            ]},
        ]},
    ],
}

bb.display.preview(spec)
bb.display.save(spec)
```

Plain dict mutation. `children` is a list — `replace`, `pop`, `insert`,
`append` all work.

## Live values: `static` + `update_data`

```python
spec = {
    "name": "focus",
    "theme": "boxbot",
    "data_sources": [
        {"name": "session", "type": "static",
         "value": {"task": "writing", "minutes": 0, "progress": 0.0}},
    ],
    "layout": {"type": "column", "padding": 32, "gap": 16, "children": [
        {"type": "text", "content": "Focus: {session.task}", "size": "title"},
        {"type": "progress", "value": "{session.progress}"},
        {"type": "metric", "value": "{session.minutes}", "label": "min"},
    ]},
}
bb.display.save(spec)

switch_display("focus")
bb.display.update_data("focus", "session",
                       value={"task": "writing", "minutes": 12,
                              "progress": 0.48})
```

`update_data` works only while the display is active, and only on
`static` sources.

## Seeing the screen

- `bb.display.get_active()` — structural read, cheap, no render.
  Confirms a `switch_display` took effect; tells you what you'd replace.
- `bb.display.screenshot()` — pixel read of the live 1024x600 surface,
  attached as an image. The only way to verify a layout against **real**
  data; `preview()` sees placeholders.

## Attachment cap

Each `execute_script` call attaches at most **8 images**
(`MAX_IMAGES_PER_CALL`). `preview()` counts as one, alongside
`bb.workspace.view` / `bb.camera.capture`. Past the cap `preview()`
still returns the PNG path (view it later) but `attached` comes back
`False`. Two or three previews per script is fine. Ten is not.

## Patterns

```python
results = bb.photos.search(query="Emily birthday")
if results:
    bb.photos.show_on_screen([results[0].id])
```

```python
schema = bb.display.describe_source("weather")
print(schema["fields"])    # {"temp": "...", "icon": "...", "forecast": "..."}
print(schema["example"])   # plausible sample, exactly the live shape

ref = bb.display.schema()
print(ref["blocks"]["chart"]["fields"])
```
