# bb.display — the 7" screen

A display is a JSON document. You read it, edit the dict, write it
back. There is no builder, no fluent API — just spec dicts and the
seven SDK calls that load, preview, save, and list them.

## When to use it

- User asks to see something ("put the weather up", "show the next
  meeting"). Use ``switch_display`` for an existing display.
- User asks for a layout that doesn't exist. Author it: build a dict,
  ``preview()`` it, fix anything the warnings flag, ``save()``.
- Editing an existing display ("a bit more contrast on morning_brief").
  ``load()`` returns a dict you mutate freely.

## When NOT to use it

- Output the *agent* needs to see — that's ``bb.workspace.view``,
  ``bb.camera.capture``, or ``bb.photos.view``.
- Rapid-fire changes — display switches involve teardown/setup of
  data sources; don't cycle them faster than ~1/sec.

## Built-in displays

- ``clock`` — full-screen time + date.
- ``weather_simple`` — current conditions.
- ``picture`` — one photo, parameterized by ``args.image_ids``.
- ``notice`` — short centered card (``args.title`` + ``args.lines``).
- Plus anything in ``displays/`` and any agent-saved displays in
  ``data/displays/``.

Call ``bb.display.list()`` to enumerate everything.

## SDK surface

| Call | Returns | Purpose |
| --- | --- | --- |
| ``bb.display.list()`` | ``[{name, source}, …]`` | enumerate displays |
| ``bb.display.get_active()`` | ``{name, args, theme}`` | what's on screen right now |
| ``bb.display.screenshot()`` | ``{path, attached, name}`` | live screen → PNG (attached to tool result) |
| ``bb.display.load(name)`` | ``dict`` | read a spec for editing |
| ``bb.display.save(spec)`` | ``{path, registered, warnings}`` | validate + write + register live |
| ``bb.display.preview(spec, data=None)`` | ``{path, attached, warnings}`` | render PNG with placeholder data |
| ``bb.display.delete(name)`` | — | remove an agent-saved display |
| ``bb.display.describe_source(name)`` | ``{fields, example}`` | data source field schema |
| ``bb.display.schema()`` | ``{blocks, themes, …}`` | full block reference |
| ``bb.display.update_data(display, source, value=)`` | ``{}`` | push live values into a ``static`` source |

``save`` and ``preview`` validate the spec before doing anything. On
error they raise ``RuntimeError`` with every problem listed. On success
they return ``warnings`` — typed lint hints:

- A binding (``{source.field}``) that didn't resolve against the
  data the renderer would use at preview/save time. Usually a typo;
  sometimes legitimate (an ``http_json`` source whose first fetch
  hasn't happened yet — those clear once live data lands).
- An icon ``name`` that isn't in the bundled Lucide subset (renders
  as a circled-letter placeholder otherwise). Use
  ``bb.display.schema()["icons"]`` to enumerate what's available.

**Limits of the warning system.** Warnings fire against the data the
*preview* sees: live cache + each ``static`` source's declared
``value=`` + placeholders for built-ins. If a binding resolves cleanly
against placeholder data but a real fetch returns a sparser shape
(e.g. a calendar event without a ``location`` field), no warning will
fire and the affected text will render empty. Sanity-check live
renders after switch_display, especially for fields placeholders
populate that real data may not.

## Switching to an existing display

The always-loaded ``switch_display`` tool — no script needed:

```
switch_display("morning_brief")
switch_display("picture", args={"image_ids": ["abc123..."]})
```

The ``args`` dict shows up in bindings as ``{args.<field>}``.

## The spec shape

```json
{
  "name": "morning_glance",
  "theme": "boxbot",
  "transition": "crossfade",
  "data_sources": [
    {"name": "weather", "type": "builtin", "refresh": 3600},
    {"name": "calendar", "type": "builtin"},
    {"name": "tasks", "type": "builtin"}
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

Five top-level keys:

| Key | Type | Notes |
| --- | --- | --- |
| ``name`` | str | unique, alphanumeric + ``_-`` |
| ``theme`` | str | ``boxbot`` (default) / ``midnight`` / ``daylight`` / ``classic`` |
| ``transition`` | str | optional. ``crossfade`` (default), ``slide_left``, ``slide_right``, ``none`` |
| ``data_sources`` | list[dict] | declared data feeds — see below |
| ``layout`` | dict | the block tree — a single block, optionally with ``children`` |

The layout is **one block** at the top level. Use a ``column`` or
``row`` to host multiple top-level children.

## Block reference

Every block has ``"type"`` and a flat bag of config fields. Containers
also have ``"children"`` (a list of blocks).

Get the full machine-readable reference any time with
``bb.display.schema()`` — it returns every field, default, and
valid-values list for every block. Use it instead of guessing.

### Layout containers

| ``type`` | Fields | Notes |
| --- | --- | --- |
| ``row`` | ``gap``, ``align``, ``padding`` | horizontal flow. ``align``: start/center/end/spread |
| ``column`` / ``stack`` | ``gap``, ``align``, ``padding`` | vertical flow |
| ``columns`` | ``ratios`` (list[int]), ``gap``, ``padding`` | weighted multi-column. ``ratios=[2,1]`` = 2/3 + 1/3 |
| ``card`` | ``color``, ``radius``, ``padding`` | **invisible without ``color=``**. Pass ``"muted"`` for a subtle surface |
| ``spacer`` | ``size`` (int or omit for flexible) | fixed or stretchy gap |
| ``divider`` | ``color``, ``thickness``, ``orientation`` | line. ``orientation``: horizontal/vertical |
| ``repeat`` | ``source`` (binding), ``max``, ``highlight_active`` | iterate over an array. Single child = template; bind item fields with ``{.field}`` |

### Content blocks

| ``type`` | Required | Optional | Notes |
| --- | --- | --- | --- |
| ``text`` | ``content`` | ``size``, ``color``, ``weight``, ``align``, ``max_lines``, ``animation``, ``min_width`` | ``size``: title/heading/subtitle/body/caption/small. ``color``: default/muted/dim/accent/success/warning/error. ``weight``: normal/medium/semibold/bold. ``align``: left/center/right |
| ``metric`` | ``value`` | ``label``, ``icon``, ``change``, ``change_color``, ``animation`` | intrinsically big — **no ``size=``**. Value uses theme ``text``; only ``change_color`` is configurable |
| ``badge`` | ``text`` | ``color`` | small colored label |
| ``list`` | ``items`` (list or binding) | ``style``, ``icon``, ``max_items`` | ``style``: bullet/number/check/none |
| ``table`` | ``headers``, ``rows`` | ``striped``, ``max_rows`` | both can be bindings |
| ``key_value`` | ``data`` (dict or binding) | — | label/value pairs |
| ``icon`` | ``name`` | ``size``, ``color`` | Lucide icon names. ``size``: sm/md/lg/xl |
| ``emoji`` | ``name`` | ``size`` | Twemoji. ``size``: md/lg/xl |
| ``image`` | ``source`` | ``fit``, ``radius`` | ``source``: ``"photo:<id>"`` / ``"url:..."`` / ``"asset:..."`` |
| ``chart`` | ``data`` *or* ``series`` | ``type``, ``color``, ``height``, ``x_labels``, ``show_grid``, ``show_legend``, ``fill_opacity``, ``show_dots``, ``padding`` | ``type``: line/bar/area |
| ``progress`` | ``value`` (0..1 or binding) | ``label``, ``color`` | ``color="auto"`` shifts green→yellow→red |
| ``clock`` | — | ``format``, ``show_date``, ``show_seconds``, ``size`` | live block. ``size``: md/lg/xl |
| ``countdown`` | ``target`` | ``label`` | live, counts down to a datetime string |

### Three different ``size`` taxonomies — don't mix them

| Block | ``size`` values |
| --- | --- |
| ``text`` | semantic: title, heading, subtitle, body, caption, small |
| ``icon`` | t-shirt: sm, md, lg, xl |
| ``emoji`` | t-shirt: md, lg, xl |
| ``clock`` | t-shirt: md, lg, xl |
| ``metric`` | (no ``size`` — already large) |

``text(size="title")`` is **bigger** than ``clock(size="xl")``. The
scales aren't comparable; pick visually, not by name.

## Data sources

Every binding in your layout pulls from a declared data source.
Declare them once at the top level, bind anywhere with
``{source.field}``.

### Built-ins (zero config)

```json
{"name": "weather"}
{"name": "calendar", "refresh": 600}
{"name": "tasks"}
{"name": "people"}
{"name": "agent_status"}
{"name": "clock"}
```

Use ``bb.display.describe_source("weather")`` to see what fields each
exposes (``temp``, ``icon``, ``forecast[].high``, etc.). The doc rots;
the schema doesn't.

### Custom: ``http_json``

Fetch JSON from any URL on a refresh interval. Use ``fields`` to pull
nested values out and (optionally) map them to icon names or color
tokens.

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

The ``secret`` field is the **name** of a secret in the boxBot secret
store, not the value. The data-source manager looks it up via
``bb.secrets`` at fetch time and sends it as a Bearer token. Store
the actual key once with ``bb.secrets.store("STOCKS_API_KEY", "…")``;
the spec only ever references the name.

Then bind ``{stocks.price}`` and ``{stocks.trend}`` (the latter as an
icon ``name``).

To verify the ``fields`` shape without waiting for a real fetch, pass
the JSON the API would return as a ``data=`` override on preview:

```python
spec = {...}
bb.display.preview(spec, data={
    "stocks": {"data": {"current": {"price": "184.20"}, "change": "+"}}
})
```

The override layers on top of the manager's normal data assembly, so
``fields`` runs against your fixture and the warnings list reflects
what would happen with real data.

### Custom: ``http_text``

```json
{"name": "page", "type": "http_text", "url": "https://example.com"}
```

Bind as ``{page.text}`` — the whole response body.

### Custom: ``static``

Hardcoded values the agent can later change with
``bb.display.update_data(...)``:

```json
{"name": "session", "type": "static",
 "value": {"task": "writing", "minutes": 0, "progress": 0.0}}
```

### Custom: ``memory_query``

Run a memory search on every refresh:

```json
{"name": "recent", "type": "memory_query",
 "query": "kitchen renovation", "refresh": 600}
```

## Bindings

Any string in any block can include ``{source.field}``. Resolved at
render time:

- ``args.<field>`` — the ``args={}`` passed to ``switch_display``.
- ``<source>.<field>`` — declared data source. Array indexing works:
  ``{calendar.events[0].title}``, ``{weather.forecast[2].high}``.
- ``{.field}`` — current item *inside* a ``repeat`` block.
- ``{current.field}`` — active item *inside* a ``rotate`` block.

If the entire string is one binding, the raw value is passed through
(useful for arrays into ``list`` / ``table``). If the string is mixed
(``"{weather.temp}°F"``), the value is stringified.

## Themes

| Theme | Mood | Background |
| --- | --- | --- |
| ``boxbot`` | warm amber-coral, wooden enclosure | dark warm |
| ``midnight`` | cool indigo, low ambient light | dark cool |
| ``daylight`` | bright daytime contrast | light |
| ``classic`` | high-contrast neutral | light |

``boxbot`` and ``midnight`` both look dark — verify by previewing
after a theme change, not by trusting the JSON alone. **Don't put
``color="muted"`` text inside a ``card(color="muted")``** — they're
the same surface tone, the text disappears.

## Authoring workflow

```python
import boxbot_sdk as bb

# Build the dict however feels natural — literal, programmatic, mixed.
spec = {
    "name": "morning_glance",
    "theme": "boxbot",
    "data_sources": [
        {"name": "calendar"},
        {"name": "weather"},
    ],
    "layout": {
        "type": "column",
        "padding": 24,
        "gap": 16,
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

result = bb.display.preview(spec)
# inspect result["warnings"], view result["path"] (auto-attached)

bb.display.save(spec)        # validates + writes + registers live
```

## Editing an existing display

```python
spec = bb.display.load("morning_glance")  # → dict
spec["theme"] = "midnight"

# Swap the second card's children for a single "UP NEXT" line
new_card = {
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
spec["layout"]["children"][2] = new_card

bb.display.preview(spec)
bb.display.save(spec)
```

Plain dict mutation — ``children`` is a list, ``replace``, ``pop``,
``insert``, ``append`` all work.

## Live data: ``static`` source + ``update_data``

For agent-controlled values (sport scores, focus timer, anything
computed):

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

``update_data`` only works while the display is active and only on
``static`` sources.

## See what's on screen right now

The system prompt tells you which display is active in the
``Current Context`` block, but for the actual pixels and live data
state you have two calls:

- ``bb.display.get_active()`` — structural read: ``{name, args, theme}``.
  Cheap, no rendering. Use this to confirm a ``switch_display`` took
  effect, or to know what you'd be replacing before authoring.
- ``bb.display.screenshot()`` — pixel read: captures the live 1024x600
  surface and attaches it as an image. Use this to verify a layout
  looks right with **real** data flowing — ``preview()`` only sees
  placeholders.

```python
state = bb.display.get_active()
# {"name": "morning_brief", "args": {}, "theme": "boxbot"}

bb.display.screenshot()
# attaches the live frame so you literally see what the household sees
```

## Preview attachment cap

Each ``execute_script`` call attaches at most **8 images** to its
result (``MAX_IMAGES_PER_CALL``). ``preview()`` counts as one
attachment, alongside ``bb.workspace.view`` / ``bb.camera.capture``.
If you blow through the cap, ``preview()`` still returns the PNG
path (you can ``bb.workspace.view`` it later) but ``attached`` will
come back ``False``. Two or three previews per script is fine; ten
is not.

## Patterns

### Put a specific photo on screen

```python
results = bb.photos.search(query="Emily birthday")
if results:
    bb.photos.show_on_screen([results[0].id])
```

### Discover what fields a data source has

```python
schema = bb.display.describe_source("weather")
print(schema["fields"])   # {"temp": "...", "icon": "...", "forecast": "..."}
print(schema["example"])  # plausible sample, exactly the live shape
```

### Inspect every block type

```python
ref = bb.display.schema()
print(ref["blocks"]["chart"]["fields"])
# {"data": {"type": "...", "default": null},
#  "type": {"type": "str", "default": "line",
#           "valid_values": ["area", "bar", "line"]}, ...}
```

### Back to idle

The display manager auto-returns to the idle rotation when nothing
takes control for a while; no explicit "unswitch" call is needed.
