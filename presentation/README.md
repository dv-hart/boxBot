# boxBot — Smart Home 2.0 (deck for Alarm.com CX PMs)

**File:** `boxBot_SmartHome_2.0.pptx` — 16:9, 14 slides.

Minimal slides built to accompany a spoken talking track. Each slide carries a
suggested talking track in the **speaker-notes pane** (View → Notes) — edit to
taste. Visual language matches boxBot's own display theme (warm amber on
near-black).

## Where your assets go

- **BB device photo** — slide 1 (title) and slide 8 (Meet boxBot) have dashed
  `▶ BB PHOTO` placeholder rectangles. Drop your photo in, delete the box.
- **Demo clip** — slide 8 has a `▶ DEMO CLIP ~45s` placeholder. Suggested clip:
  one spoken request ("BB, what's on today?") → voice reply + the screen
  updating. Insert → Video, size it into the box.
- The three screens on **slide 10 ("Displays, not widgets")** are real boxBot
  screenshots pulled from `data/previews/` — already embedded.

## Slide flow

1. Title — Smart Home 2.0
2. The thesis — *the assistant, with the home built around it*
3. Where we are today — every capability is a surface to configure
4. A concrete example — "text my wife when I add to the calendar"
5. The old way — open settings, build the rule
6. The new way — she just tells boxBot (one line in a skill file)
7. What just happened — old model vs. agent model
8. Meet boxBot — the device + **[demo clip]**
9. The design philosophy — Tools / Skills / Displays
10. Displays, not widgets — real boxBot screens
11. Displays on demand — "add solar to my dashboard"
12. Agent-first architecture — the dark-mode story
13. Why this matters — stickier / cheaper / trust
14. Close — *the UI becomes whatever the user wants*

## Rebuilding

Edit and re-run the generator (requires `python-pptx`, `Pillow`):

```bash
python3 build_deck.py
```
