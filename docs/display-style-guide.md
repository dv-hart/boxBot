# boxBot Display Style Guide

> **Status:** Proposal — synthesizes existing rules from
> `display-system.md` and `display-development.md` with the project's
> physical and brand context. Review before treating as canonical.

## 1. Premise

> *What if Anthropic made a smart speaker?*

boxBot is the answer. The visual language of the display extends a
physical object — a black walnut box with 3/4" roundovers — into pixels.
Mid-century modern furniture meets Apple curves. A vintage wooden radio
crossed with an iPhone. The screen is the glowing dial of that radio,
not a generic tablet UI.

Every visual choice should pass one test: **does this feel like it
belongs in the wooden box?**

## 2. Inspirations

| Source | What we borrow |
|---|---|
| **Anthropic brand** | Warm coral-amber accent, restrained palette, calm tone, plain-spoken typography |
| **Vintage radios & nixie tubes** | Amber-on-dark glow, soft warmth, physicality of materials |
| **Mid-century modern furniture** | Honest materials, generous proportions, low-contrast confidence |
| **Apple industrial design** | Tight type, soft roundovers, content-first restraint |
| **Black walnut enclosure** | Backgrounds are warm charcoal, never cold black; nothing on screen should clash with the wood it sits inside |

What we are **not**: dashboard-y, neon, gamer, glassmorphic, Material
Design, generic dark-mode-grey.

## 3. Color

### 3.1 Principle

The display should harmonize with walnut, not fight it. Backgrounds are
**warm dark** (slight brown/amber bias), accents are **warm metals**
(amber-coral, soft gold), and pure white / pure black are forbidden.
Saturated primaries (Material blue, alert red) are forbidden.

### 3.2 Token system

Agents and contributors use **semantic tokens**, never hex literals.
The token names are stable across themes — only the values change.

```
background     surface     surface_alt
text           muted       dim
accent         accent_soft secondary
success        warning     error
```

See `display-system.md §Theme Schema` for the canonical schema.

**Rule:** if you find yourself writing `#` in display code, you're
doing it wrong. Add a theme token instead.

### 3.3 Default palette — `boxbot` theme

The signature theme. Everything else is a variant.

| Token | Value | Role |
|---|---|---|
| background | `#191714` | Deep warm charcoal — reads as walnut shadow |
| surface | `#252018` | Card background, slightly lifted |
| surface_alt | `#302a20` | Nested depth |
| text | `#ede8e0` | Warm cream — like the inside of a lampshade |
| muted | `#8a8078` | Warm gray |
| dim | `#5a5550` | Timestamps, dots, subtle metadata |
| **accent** | **`#d4845a`** | **Amber-coral — Anthropic-inspired, the BB voice** |
| accent_soft | `#d4845a22` | Tinted backgrounds, badges |
| secondary | `#c4a46c` | Soft gold — tube-glow warmth |
| success | `#7a9e6c` | Earthy sage, never a bright green |
| warning | `#d4a043` | Warm amber |
| error | `#c45c5c` | Muted red, never fire-engine |

The accent (`#d4845a`) is boxBot's **signature color** — it's the voice
of the system on screen. Use it sparingly: one or two highlights per
view, not as decoration.

### 3.4 Theme variants

| Theme | When | Mood |
|---|---|---|
| `boxbot` | Default, daytime, ambient | Warm minimal — the canonical look |
| `midnight` | Late hours, paired with HAL dimming | Near-black, only essentials visible, no shadows |
| `daylight` | Bright rooms, well-lit kitchens | Warm linen background, deeper amber accent for contrast |
| `classic` | Vintage / radio mode | Amber-tinted text on dark, nixie-tube energy |

All four use the same token names and the same Inter type scale. Themes
are interchangeable per display.

## 4. Typography

### 4.1 Family

**Inter** is the standard, always available, used in all built-in
themes. Community themes can bundle a different font in
`themes/{name}/fonts/`, but Inter is the default and the fallback.

Why Inter: it carries the Apple-grade restraint without being SF, sits
quietly on a warm background, and reads well at 1–2 m.

### 4.2 Scale

A single 6-step scale — never invent custom sizes.

| Token | Size | Weight | Use |
|---|---|---|---|
| `title` | 42 | 700 | Once per display, top-of-hierarchy (clock, headline number) |
| `heading` | 28 | 600 | Section headers, item titles |
| `subtitle` | 22 | 500 | Supporting copy under a heading |
| `body` | 18 | 400 | Default paragraph and list text |
| `caption` | 15 | 400 | Metadata, labels, axis ticks |
| `small` | 13 | 400 | Footnotes, source attributions |

**Floor:** 18 px for any prose. The display is read from across a
living room — body text smaller than that is decorative, not legible.

### 4.3 Tracking

Negative tracking on display sizes (`title: -0.02em`, `heading:
-0.01em`) for an Apple-tight feel. Body and below: default tracking.

## 5. Shape & spacing

### 5.1 Roundovers

The physical box has a **3/4" roundover**. The screen echoes that — every
card and surface uses generous corner radius. Sharp corners are reserved
for dividers and chart grid lines.

| Element | Radius |
|---|---|
| `card` | `14` (default) — matches enclosure feel |
| `classic` theme override | `10` — slightly tighter, vintage |
| Chips, badges | `8` |
| Pills | half height |
| Divider lines | `0` (sharp) |

### 5.2 Spacing scale

Use the theme spacing tokens. No magic numbers.

| Token | Pixels | Use |
|---|---|---|
| `xs` | 4 | Tight inline gaps |
| `sm` | 8 | Within a card, between label and value |
| `md` | 16 | Between cards, section padding |
| `lg` | 24 | Section breaks, card outer padding |
| `xl` | 32 | Top-level layout breathing room |

### 5.3 Layout

- **Resolution:** 1024 × 600, landscape, fixed.
- **Layout is flow-based** (`row`, `column`, `columns([2,1])`). Agents
  and contributors **never** write pixel coordinates.
- **One title per display.** If you find yourself reaching for a second
  `title`, it's probably a `heading`.
- **Generous whitespace.** Mid-century furniture proportions —
  air around content, not edge-to-edge density.
- **Content-first.** Chrome (borders, backgrounds, dividers) earns its
  place by clarifying hierarchy. Decorative chrome is removed.

## 6. Iconography

- Style: **outline** in `boxbot`, `midnight`, `daylight`, `classic`. Set
  via `icon_style: outline` in the theme.
- Stroke weight matches Inter's regular weight — icons should feel like
  letterforms, not stickers.
- Color: same tokens as text (`text`, `muted`, `accent`).
- **Filled icons** are reserved for "on/active" states (e.g. a filled
  bell for an active alert).

## 7. Motion

- Default transition: **`crossfade`** between displays. Slow enough to
  feel intentional, fast enough not to annoy (~250 ms target).
- Live blocks (clock, countdown) tick at 1 fps. No animations on idle.
- Data blocks fade in on refresh — never pop or slide aggressively.
- The `midnight` theme disables shadows; consider also softening
  transitions there if it ever feels jarring at night.
- **No bouncing, no parallax, no spinners-as-decoration.** A still
  display is the default state.

## 8. Composition principles

1. **Calm over busy.** A boxBot display is glanced at, not studied.
   If it has more than ~3 information densities, simplify.
2. **One accent per view.** The amber-coral is a signal, not a
   pattern. Two accents = no accent.
3. **No conditional logic in the spec.** Mapping (value → icon, value
   → color) belongs in the data source, not in the layout. Keeps
   displays declarative and reviewable.
4. **Theme tokens or nothing.** No hardcoded colors, no hardcoded font
   sizes. If a token is missing for a real need, propose adding one.
5. **Performance budget: render in <50 ms.** Heavy work in `setup()`
   or background tasks, not the render loop.
6. **Wood-first contrast check.** Imagine the rendered display set into
   the walnut box. If it would visually clash with the wood (cold
   blues, neon greens, pure white panels), revise.

## 9. Quick checklist

Before shipping a display, confirm:

- [ ] Background uses a warm dark token (not cold grey, not pure black)
- [ ] At most one accent-colored element drives the eye
- [ ] All colors are theme tokens — no hex in the spec
- [ ] Body text ≥ 18 px; one `title` maximum
- [ ] Cards use the theme `radius`; corners feel rounded, not square
- [ ] Layout is flow-based — no pixel coordinates
- [ ] Renders cleanly on `boxbot`, `midnight`, and `daylight`
- [ ] No spinners, bounces, or parallax
- [ ] Reads from 1–2 m away
- [ ] Looks like it belongs in a wooden box

## 10. References

- `docs/display-system.md` — block library, theme schema, full theme
  definitions, data binding
- `docs/display-development.md` — author workflow, performance rules
- `themes/` — community theme drop-in directory
- Anthropic brand color reference — the `#d4845a` accent traces back
  here
