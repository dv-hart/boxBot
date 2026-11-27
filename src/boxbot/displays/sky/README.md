# displays/sky — the Living Sky page

The HTML renderer's page (`display.renderer: html`). One local Chromium window shows `index.html`;
the display manager pushes state into it over the DevTools pipe. Nothing here talks to the network.

The agent never reads pixels. It knows what is on screen from the spec, the data and the atmosphere —
the same JSON this page draws from. Anything a block shows must come from its params or from
`state.atmosphere`; a block never invents content.

## Files

| File | Holds |
|---|---|
| `index.html` | Shell, CSP, layer markup, load order |
| `core.css` | Reset, layers, type classes, `.slab`, `.ground-guard`, person classes |
| `scene.js` | `scene(o)` → `{svg, P}`, `palette(phase, weather)`, colour helpers (`rgba`, `tune`, `mix`) |
| `icons.js` | `ICONS` registry, `ico(name, style)`; accepts Lucide names from data sources |
| `core.js` | `BB`: state in, atmosphere tokens, block registry, cross-fades, ticks |
| `blocks/base.{js,css}` | The original block vocabulary (row, column, card, text, metric, clock, …) |
| `blocks/home.{js,css}` | `zones`, `zone`, `sky_clock`, `now`, `next_event`, `house`; also `BB.live` and `BB.cal` (used by calendar.js) |
| `blocks/calendar.{js,css}` | `surface`, `say`, `agenda_timeline`, `todo_list`, `forecast_strip`, `cal_header`, `legend`, `week_grid`, `day_grid` |
| `blocks/inbox.{js,css}` | `inbox` |
| `blocks/picture.{js,css}` | `picture`, `photo_caption`, `capture`, `sky_chip` — full-bleed layouts paint their own opaque frame; `side` uses the scene |
| `blocks/presence.{js,css}` | `BB.presence` (voice-state dock) and `status_pill` |
| `specs/<name>.json` | Display specs registered under the html renderer (`home`, `brief`, `week`, `day`, `inbox`, `picture`, `capture`) |
| `fixtures/<name>.js` | `BB.fixture = {state}` — dev and test states |
| `fonts/` | Fraunces + Figtree (OFL), bundled |

## State (Python → page)

`BB.apply(state)`; the whole state every time, the page diffs.

```js
{
  v: 1,
  display: "home",                 // active display name
  transition: "crossfade",         // "none" = replace without fading
  now: "2026-11-24T16:02:00-08:00",// wall clock; the page ticks on from it (optional: defaults to Date)
  people: ["Jacob", "Carina", "Erik", "Mia"],   // household, stable order → colour slots p1..p4
  atmosphere: { phase: "golden",   // dawn noon afternoon golden dusk night
                weather: "partly", // clear partly fog overcast rain snow
                time: "16:02",     // drives ferry position, lit windows
                dim: false,        // night mode: veil + lamp-cream ink
                sun: { next: "sunset", at: "4:24 PM" } },   // or null: no location configured
  scene: { horizon: "high", scale: null, vista: "kerry", seed: 7, hidden: false },   // from the spec; unset keys absent
  layout: { type: "...", ...params, children: [...] },   // block tree, data bindings already resolved
  data: { clock: {...}, weather: {...}, ... },           // raw source data (blocks should prefer params)
  args: {},                        // switch_display args
  status: "Searching the web…",    // or null — agent status pill
  presence: { state: "listening", who: "Carina", text: "…" },   // or null — voice-state dock (below)
  frame: null,                     // data: URL of a frame drawn elsewhere, shown full-bleed over the view, or null:
                                   //   camera live view, and displays only PIL can draw (products, cart)
  images: ["/abs/vetted/a.jpg"]    // every absolute image path the page may load for this state
}
```

Bindings (`{weather.temp}`) are resolved in Python before the push: a block sees values, not templates.
A binding that resolved to nothing arrives as `null` or `""` — render nothing for it, never "undefined".

`presence` fields: `state` (`listening | thinking | speaking | timer | notice | greeting`), `who`
(household name), `text` (the one serif line), `detail` (quiet words after the kicker), `label`
(replaces the state word), and for `timer`: `progress` (0..1 remaining) or `until` (ISO) + `duration`
(seconds), which the page then ticks itself. Python sends the first three states today.

## Writing a block

```js
(function () {
  ICONS.ferry = '<path d="…"/>';                       // optional, 24×24 outline
  BB.define('next_event', (b, ctx) => {
    const e = b.event; if (!e) return '';
    return `<div class="next ${BB.person(e.person)}">
      <div class="label">${BB.esc(b.label || 'Next')}</div>
      <div class="ttl d">${BB.esc(e.title)}</div></div>`;
  });
  BB.onTick((date, root) => { /* once a second: update countdowns inside root */ });
})();
```

- `(block, ctx) => html string`. `block` = the block's JSON (`type`, params, `children`).
  `ctx` = `{state, data, args, atmosphere, P, now, display}`; `P` is the palette (`P.ink` = `dark|light`).
- **Escape everything from data with `BB.esc()`**, including attribute values. Titles, subjects and
  captions come from outside the house.
- `BB.children(block, ctx)`, `BB.render(childBlock, ctx)` for containers.
- `BB.person(name)` → `p-p1…p-p4 | p-family` (sets `--p`). Never hard-code a household name.
- `BB.now()`, `BB.fmtTime(date, {meridiem, h24})`, `BB.until(date)` ("in 1 h 28 min").
- `BB.onTick(fn)` for anything that changes with the clock; `BB.onMount(fn)` after a pane is inserted.
- `BB.live(type, fn)` (home.js) instead of `BB.define` for a block whose output depends on the clock:
  `fn(block, ctx, mem)` is re-run every tick with `ctx.now` and the DOM is swapped only when the HTML
  changes. An element marked `data-fit` that overflows makes the block re-render with `mem.trim` raised
  by one until it fits.
- Missing or empty data is normal (nobody home, no events, weather fetch failed). Every block has an
  honest empty state.

## Tokens

On `#stage`: `--text --muted --accent --hair --hair2 --chip --guard --ground --ground-muted`, people
`--p1 --p2 --p3 --p4 --family`. Classes `ph-<phase> wx-<weather> ink-<dark|light>`, `dim` at night.

Slab (the only surface for dense content; class `.slab` + `.left | .right | .bottom`):
`--glass --glass-deep --sheen --rim --slab-solid --slab-fade --raise --st --sm --sh --sh2 --sa --paper --on-accent`,
light-set people `--l1…--l4 --lf`. Inside a `.slab`, `--text/--muted/--accent/--hair/--p*` are remapped,
so the same block markup works on the sky and on a slab.

`blocks/base.css` adds `--bb-ok --bb-warn --bb-err --bb-sec --bb-dim --bb-line --bb-on --bb-halo` on
`#stage` and resets them inside `.bb-card` / `.slab`.

Type: Fraunces (`.d`) for numerals and titles, Figtree for the rest. Floor 16px. Sky-zone text takes
`.sky-ink`; ground-zone text takes `.ground` over a `.ground-guard`.

Safe zones at `horizon: high`: quiet sky x 40–460, y 30–290 · landmarks y 150–372, x 290–1024 ·
ground y 470–600. Sun or moon sits top-right (x 840–950, y 80–130).

## Render budget (Pi 5)

No animated `filter`, `mask` or `box-shadow`. Motion is `transform` and `opacity` only. Blur inside the
scene is baked into the SVG once, never animated.

`backdrop-filter` is used in one place: the glass recipe in `core.css` (`.slab`, `.bb-card`, `.bb-plate`,
`.legend`). It stays cheap because the scene under it is still. Two rules keep it that way:

- Nothing animates underneath a glass surface.
- No ancestor of a glass surface takes `opacity`, `filter` or `mask` — that cuts the surface off from the
  scene and the blur disappears. Panes therefore cross-fade piece by piece (`markFade` in `core.js`). A new
  glass class must be added to `GLASS` there.

## Dev

```bash
scripts/sky-shot.sh /tmp/out "home|fixture=home" "home-rain|fixture=home&phase=afternoon&weather=rain"
```

Query overrides for fixtures: `phase weather time vista dim now status presence who text`.

A new block type also needs an entry in `../sky_blocks.py` (its params, for spec validation and the
agent's `bb.display.schema()`); `tests/test_sky_page.py` fails if the page and Python disagree on the
vocabulary. Image params reach a block as vetted absolute paths: specs write `"photo:<id or path>"` and
Python resolves it, or drops it. **Turn a path into an `<img>` source only with `BB.img(path)`**: it
returns `''` for an absolute path that is not in `state.images`. Name an image param `source`, `src`,
`thumb`, `path`, `image` or `photo` (lists: `photos`, `images`, `thumbs`) so Python vets it
(`manager._IMAGE_KEYS`).

A display whose tree contains a block that is neither in `BLOCK_REGISTRY` nor in `sky_blocks.py` is not
drawn by this page at all: the manager renders it with PIL and sends it as `frame`.

Tests: `tests/test_sky_renderer.py` (Python side, fake browser) and `tests/test_sky_page.py` (this page in
a real Chromium: every fixture and spec draws cleanly, hostile text stays text, no network).
