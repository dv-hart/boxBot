# boxBot Display Style Guide

> **Status:** Canonical. The `boxbot` default theme, the renderer's
> layout rules, and every shipped display spec follow this guide.
> Verified on real hardware at 1024×600 (7" Pi display) and 1280×800.

## 1. Premise

boxBot's screen is an **ambient panel**, not an app. It is glanced at
from across a room, mounted flush in a household — a wooden box on a
shelf or a screen on a wall. The visual language is
**Alarm.com, modernized**: the confidence of a security brand — deep
slate, one signal orange, disciplined type — with the calm of a
well-made ambient display. Fresh, not corporate; quiet, not empty.

Every choice should pass one test: **does this read instantly from
two meters, and would it look at home next to the Alarm.com wordmark?**

## 2. Design pillars

| Pillar | In practice |
|---|---|
| **Slate, not black** | Backgrounds are deep blue-gray slate (`#141a21`), straight from Alarm.com's ink color family. Never pure black, never brown, never cold neutral gray. |
| **One signal** | Alarm.com orange (`#f25e0f`) is *the* signal. One or two orange elements per screen. Orange means "this is the thing" — never decoration. |
| **Type carries hierarchy** | Inter, six sizes, tight tracking on display sizes. Size + weight + tone (text/muted/dim) do the structuring; boxes and lines don't. |
| **Soft, flat, precise** | Surfaces are flat slate cards with generous radius and a whisper of shadow. No gradients-as-decoration, no glassmorphism, no neon. |
| **Composed, not stacked** | Content is deliberately placed on the full canvas — centered heroes, anchored footers, balanced grids. Never a top-pinned stack with a dead bottom half. |

What we are **not**: dashboard-dense, Material Design, gamer RGB,
skeuomorphic, or a web page.

## 3. Color

### 3.1 Token system

Displays use **semantic tokens only** — never hex literals in a spec.
Token names are stable across themes; only values change.

```
background     surface     surface_alt
text           muted       dim
accent         accent_soft secondary
success        warning     error
```

**Rule:** if you write `#` in a display spec, you're doing it wrong.

### 3.2 Default palette — `boxbot` theme

Derived directly from Alarm.com's brand palette (slate ink `#1d252d`,
signal orange `#e35205`/`#f25e0f`, slate grays `#505a6a`/`#8a989e`),
tuned for a self-luminous dark panel.

| Token | Value | Role |
|---|---|---|
| background | `#141a21` | Deep slate — Alarm.com ink, deepened for ambient use |
| surface | `#1d252d` | Card background — Alarm.com's exact brand dark |
| surface_alt | `#28323c` | Nested depth, chart tracks, stripes |
| text | `#f0f2f3` | Primary text — Alarm.com light gray |
| muted | `#8a989e` | Secondary text — Alarm.com slate gray |
| dim | `#505a6a` | Tertiary — timestamps, grid lines, inactive dots |
| **accent** | **`#f25e0f`** | **Signal orange — Alarm.com brand. THE highlight.** |
| accent_soft | `#f25e0f29` | Tinted fills behind accent content (badges, pills) |
| secondary | `#ff9965` | Orange tint — chart lines, secondary emphasis |
| success | `#3fae5f` | Calm green (from Alarm.com `#0caa41`, desaturated for dark) |
| warning | `#e8a33d` | Amber |
| error | `#e2574b` | Alert red — reserved for true alarm states |

### 3.3 Theme variants

| Theme | When | Mood |
|---|---|---|
| `boxbot` | Default, day and evening | Slate + signal orange — the canonical look |
| `midnight` | Late hours, paired with HAL dimming | Embers: near-black slate, dimmed text, no orange louder than a coal |
| `daylight` | Bright rooms | Alarm.com web look: `#f4f4f4` field, white cards, ink text, `#e35205` accent |
| `classic` | Heritage / vintage radio mode | The original warm amber-on-walnut boxBot palette, kept as a variant |

All variants share token names, the Inter scale, and spacing. Specs
are theme-portable by construction.

### 3.4 Usage rules

- **Orange budget: 1–2 elements per screen.** The active state chip,
  the next event time, the alarm badge. Two accents = no accent.
- `secondary` (`#ff9965`) for charts and quieter emphasis so the
  full-strength orange keeps its authority.
- `error` appears only for genuinely wrong states (offline, alarm,
  failure) — never as a warm decoration.
- Eyebrow labels (`OUTSIDE`, `NEXT UP`) are `muted`, values are `text`,
  footnotes are `dim`. Three tones, consistently.

## 4. Typography

### 4.1 Family

**Inter**, bundled, all themes. Weights: Regular 400, Medium 500,
SemiBold 600, Bold 700, ExtraBold 800.

### 4.2 Scale

| Token | Size | Weight | Tracking | Use |
|---|---|---|---|---|
| `title` | 42 | 700 | −0.02em | Hero number/headline — once per display |
| `heading` | 28 | 600 | −0.01em | Section headers, greeting |
| `subtitle` | 22 | 500 | 0 | Supporting line under a hero |
| `body` | 18 | 400 | 0 | Default text |
| `caption` | 15 | 400 | 0 | Metadata, axis ticks |
| `small` | 13 | 500 | +0.06em | Eyebrow labels (uppercase), footnotes |

**Floor:** 18px for prose. Smaller sizes are labels/metadata only.

**Eyebrow pattern:** uppercase `small` + `semibold` + `muted` +
positive tracking. This is the standard section label
(`WEATHER`, `TO-DO`, `POWER DRAW`).

### 4.3 Clock numerals

Clocks are the most-seen pixels boxBot renders. Dedicated sizes:
`md` 40 / `lg` 64 / `xl` 112, weight 600, tracking −0.03em. The date
line sits below at `subtitle`/`muted` with a fixed 12px gap, and the
time+date group centers **as a unit** in its container — the renderer
guarantees they never collide.

## 5. Shape, space, composition

### 5.1 Surfaces

- Card radius **16** (all themes except `classic` at 10).
- Shadow: a soft, blurred, low-alpha drop (renderer-composited) —
  never a hard offset edge. `midnight` disables shadows entirely.
- Card interiors: 18–24px padding. Content never touches a card edge.

### 5.2 Spacing scale

`xs 4 · sm 8 · md 16 · lg 24 · xl 32` — no magic numbers between.
Screen margins: ≥32px on the panel canvas. Cards breathe: ≥14px gaps.

### 5.3 Composition (the anti-dead-space rules)

- **Own the full canvas.** Layouts must look composed at both
  1024×600 and 1280×800. Columns support main-axis `align:
  start|center|end` and **flex spacers** (`spacer` with no size)
  that absorb leftover height — use them. A top-pinned stack with an
  empty bottom half is a defect.
- **Ambient screens center their hero** (clock, person name) slightly
  above geometric center, metadata anchored to an edge.
- **Dashboards balance the grid** — cards share row heights; a short
  card gets its content vertically centered, not top-pinned over a void.
- **Rows center-align vertically** (`valign: center` is the default):
  icons sit on the text's optical centerline, never hanging from the
  top edge.
- One `title`-sized element per display. Generous negative space is a
  feature — density is capped at ~3 information groups per screen.

## 6. Iconography

- **Lucide outline**, stroke color from the same three text tones.
- Icon size pairs with the text it accompanies (`sm` 16 ↔ caption,
  `md` 24 ↔ body/subtitle, `lg` 32 ↔ heading, `xl` 48 ↔ hero).
- Icons are labels, not decoration — an icon without adjacent text
  must be universally readable (weather glyphs, mic, wifi).
- Accent-colored icons count against the orange budget.

## 7. Motion

- Display switches: **crossfade ~250ms**. Nothing else moves at idle.
- Live blocks (clock, countdown) tick at 1fps with no transition.
- Data refreshes swap in place — no slides, no pops, no spinners.

## 8. Checklist

Before shipping a display:

- [ ] Composed at 1280×800 **and** 1024×600 — no dead bottom half
- [ ] ≤2 orange elements; eyebrows muted; footnotes dim
- [ ] All colors are tokens; all sizes from the scale
- [ ] Rows valign-centered; no icon hanging above its label's centerline
- [ ] Text never collides — verify with the preview renderer, then on-device
- [ ] Renders cleanly on `boxbot`, `midnight`, `daylight`
- [ ] Reads from 2m: hero legible, eyebrows discernible, footnotes ignorable

## 9. References

- `docs/display-system.md` — block library, theme schema, data binding
- `docs/display-development.md` — authoring workflow
- `src/boxbot/displays/themes.py` — canonical theme values
- Alarm.com brand palette (extracted from alarm.com production CSS):
  orange `#e35205` / `#f25e0f` / `#bf4600` / `#ff9965`, slate ink
  `#1d252d`, slate grays `#505a6a` / `#8a989e` / `#d7dee0` / `#f4f4f4`
