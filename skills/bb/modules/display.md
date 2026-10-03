# bb.display — the 7" screen

A display = a JSON spec. Workflow: build dict or `load(name)` → mutate →
`preview(spec)` → fix warnings → `save(spec)`. Reference spec: `load()`
any name from `list()` — real and current, don't invent from scratch.
Field reference: `bb.display.schema()`; source fields:
`describe_source(name)`. Never guess either.

Existing display → `switch_display` (always-loaded tool, no script).
Output *you* need to see → `bb.workspace.view` / `bb.camera.capture` /
`bb.photos.view`, not a display. Switches tear down data sources — max
~1/sec.

## Built-in displays — clock / weather_simple / notice / picture + their args

`clock` · `weather_simple` · `notice` (`args.title` + `args.lines`) ·
`picture` (`args.image_ids: [str, ...]` required; 1 id = static, 2+ =
slideshow every `args.interval` sec, default 8) · everything in
`displays/` and `data/displays/`. Enumerate: `bb.display.list()`.

## Call signatures — every bb.display call (list, screenshot, load/save/preview, …)

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

`save`/`preview` validate first. Invalid → `bb.ActionError` listing
every problem. Valid → `warnings`:

- unresolved binding `{source.field}` — typo, or an `http_json` source
  pre-first-fetch (clears once live data lands).
- icon `name` outside the bundled Lucide subset (renders as
  circled-letter placeholder). Valid set: `bb.display.schema()["icons"]`.

Warnings check **placeholder** data — sparser real data can still
render a binding empty. Verify live: `screenshot()` after
`switch_display`.

## Switching displays — switch_display(name, args, pin)

```
switch_display("morning_brief")
switch_display("picture", args={"image_ids": ["abc123..."]})
```

`args` binds as `{args.<field>}`. **Pins by default**: display holds,
rotation pauses, no auto-revert. `pin=False` = show without taking
control. Resume rotation = `bb.display.unpin()`.

Slideshow = 2+ `image_ids` on `picture` — NEVER the `rotate` block (the
renderer draws only its first child). "All slideshow-tagged photos" →
gather ids with `bb.photos.search(...)` first.

## Display spec shape — name / theme / data_sources / root block tree

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
    "type": "column", "padding": 24, "gap": 16,
    "children": [
      {"type": "row", "align": "spread", "children": [
        {"type": "clock", "format": "12h", "size": "lg"},
        {"type": "metric", "value": "{weather.temp}°",
         "label": "{weather.condition}", "icon": "{weather.icon}"}
      ]},
      {"type": "card", "color": "muted", "padding": 18, "children": [
        {"type": "repeat", "source": "{calendar.events}", "max": 2,
         "children": [{"type": "row", "gap": 12, "children": [
           {"type": "text", "content": "{.time}", "color": "accent"},
           {"type": "text", "content": "{.title}"}
         ]}]}
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
| `layout` | dict | the block tree — **one** block; host children in `column`/`row` |

## Blocks — container + content block types, their fields, and the three size taxonomies

Every block = `"type"` + flat config fields; containers add
`"children"`. Full field/default/valid-values reference:
`bb.display.schema()`.

### Containers

| `type` | Fields | Notes |
| --- | --- | --- |
| `row` | `gap`, `align`, `valign`, `padding` | horizontal. `align`: start/center/end/spread. `valign`: center (default)/top/bottom |
| `column` / `stack` | `gap`, `align`, `item_align`, `padding` | vertical. `align` = vertical packing: start/center/end. `item_align` = horizontal per-child: stretch (default)/start/center/end |
| `columns` | `ratios` (list[int]), `gap`, `padding` | weighted. `ratios=[2,1]` = 2/3 + 1/3 |
| `card` | `color`, `radius`, `padding`, `align` | **invisible without `color=`**. `"muted"` = subtle surface |
| `spacer` | `size` (int, or omit for flexible) | fixed or stretchy gap. Sizeless = flex: absorbs leftover space — center heroes, anchor footers |
| `divider` | `color`, `thickness`, `orientation` | `orientation`: horizontal/vertical |
| `repeat` | `source` (binding), `max`, `highlight_active` | iterate an array. Single child = template; bind item fields with `{.field}` |

Any child of a vertical flow accepts `grow: true` — absorbs leftover
height. Full-canvas rule: compose with flex spacers / `align` / `grow`,
never a top-pinned stack over a dead bottom half. Style rules:
docs/display-style-guide.md.

### Content

| `type` | Required | Optional | Notes |
| --- | --- | --- | --- |
| `text` | `content` | `size`, `color`, `weight`, `align`, `max_lines`, `animation`, `min_width` | `size`: title/heading/subtitle/body/caption/small, or a pixel number (8–240) for a hero readout. `color`: default/muted/dim/accent/success/warning/error. `weight`: normal/medium/semibold/bold. `align`: left/center/right |
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
| `text` | semantic: title (42px) → small (13px), **or** a pixel number, 8–240 |
| `icon` | t-shirt: sm, md, lg, xl |
| `emoji` | t-shirt: md, lg, xl |
| `clock` | t-shirt: md (40px), lg (64px), xl (112px) |
| `metric` | none — already large |

Scales are not comparable — pick visually. `text(size="title")` <
`clock(size="xl")`; hero temperature = pixel number:
`{"type": "text", "content": "{climate.temp}°", "size": 140}`.
Out-of-range numbers clamp, never break the render.

## Data source types — built-ins (tasks/people/agent_status/clock) · integration · http_json · http_text · static · memory_query

Declare once in `data_sources`, bind anywhere as `{source.field}`.

### Built-ins, zero config

`{"name": "tasks"}` · `{"name": "people"}` · `{"name": "agent_status"}`
· `{"name": "clock"}` — live in-process state (scheduler to-dos,
present people, agent state, time); cannot be integrations. Fields:
`bb.display.describe_source(name)` — the doc rots, the schema doesn't.

### `integration`

```json
{"name": "weather", "type": "integration", "refresh": 3600}
{"name": "calendar", "type": "integration",
 "inputs": {"action": "list_upcoming_events", "max_results": 5},
 "refresh": 600}
```

Manager calls `bb.integrations.get(<name>, **inputs)` on cadence, binds
the output dict to the source name → `{weather.temp}`,
`{calendar.events[0].title}`.

- `integration` — override which integration to call (defaults to
  `name`); one integration under several bindings.
- `inputs` — passed verbatim. Manifests carry defaults + `default_env`
  fallbacks (`BOXBOT_WEATHER_LAT` / `BOXBOT_WEATHER_LON`) — rarely
  repeat per display.
- `refresh` — seconds between fetches. Default 300.

Agent-authored and pre-seeded integrations share this path — no
privileged track. What exists: `bb.integrations.list()`; output shape:
`bb.display.describe_source(name)`.

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

`secret` = a secret's **name**, never the value — looked up via
`bb.secrets` at fetch time, sent as Bearer token. Store once:
`bb.secrets.store("STOCKS_API_KEY", "…")`. Bind `{stocks.price}`,
`{stocks.trend}`.

Verify `fields` without a real fetch — fixture layers onto normal data
assembly, so warnings reflect real behavior:

```python
bb.display.preview(spec, data={
    "stocks": {"data": {"current": {"price": "184.20"}, "change": "+"}}
})
```

### `http_text`

`{"name": "page", "type": "http_text", "url": "https://example.com"}`
→ bind `{page.text}` (whole body).

### `static`

```json
{"name": "session", "type": "static",
 "value": {"task": "writing", "minutes": 0, "progress": 0.0}}
```

Push new values later:
`bb.display.update_data("focus", "session", value={...})` — works only
while the display is active, `static` sources only.

### `memory_query`

```json
{"name": "recent", "type": "memory_query",
 "query": "kitchen renovation", "refresh": 600, "limit": 5}
```

Re-runs a memory search per refresh (hybrid vector + keyword only — no
reranking, no summaries — so refreshing is free). `limit` default 5.
Output: `results` = `[{text, type, age}]` (`type`:
person/household/methodology; `age`: `"3d"` / `"2w"`), `count`,
`query`. Bind rows via `repeat`.

## Bindings — {source.field} resolution in any block string

Any string in any block can contain `{source.field}`, resolved at
render:

- `args.<field>` — the `args={}` from `switch_display`.
- `<source>.<field>` — a declared source. Indexing works:
  `{calendar.events[0].title}`, `{weather.forecast[2].high}`.
- `{.field}` — current item inside a `repeat`.
- `{current.field}` — active item inside a `rotate`.

String = entirely one binding → raw value passes through (arrays reach
`list`/`table` intact). Mixed string (`"{weather.temp}°F"`) →
stringifies.

## Themes — boxbot / midnight / daylight / classic; mood + background tone

| Theme | Mood | Background |
| --- | --- | --- |
| `boxbot` | warm amber-coral, wooden enclosure | dark warm |
| `midnight` | cool indigo, low ambient light | dark cool |
| `daylight` | bright daytime contrast | light |
| `classic` | high-contrast neutral | light |

`boxbot` and `midnight` both read dark — preview after a theme change.
**Never `color="muted"` text inside `card(color="muted")`** — same
surface tone, text vanishes.

## Editing a display — load → mutate dict → preview → save

`load(name)` → plain dict. `children` is a list — replace/pop/insert/
append all work. Then `preview` → `save`.

## Seeing the screen — get_active (structural) vs screenshot (pixels)

- `bb.display.get_active()` — structural, cheap, no render. Confirms a
  switch; tells you what you'd replace.
- `bb.display.screenshot()` — pixels, attached. The ONLY verify against
  **real** data (`preview` sees placeholders). Surface size is
  config-driven — 1024x600 on the 7" LCD by default; the screenshot
  reports its own size, never assume.

## Attachment cap — ≤8 images per execute_script call

≤ **8** images per `execute_script` call (`MAX_IMAGES_PER_CALL`);
`preview` counts alongside `workspace.view` / `camera.capture`. Over
cap: PNG path still returned, `attached: False`. 2–3 previews per
script fine; 10 not.
