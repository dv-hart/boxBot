---
name: bb
description: The bb Python package — your hands inside the sandbox. Import it from execute_script for camera stills, photo search, the 7" display, a persistent notes/CSV workspace, memory, triggers and to-dos, audio playback, new skills, secrets, and data-pipe integrations (calendar lives here). Load when the user mentions photos, the camera, the screen, notes, lists tracked over time, reminders, or anything worth composing in one turn.
when_to_use: User mentions photos, camera, display/screen, notes, keeping a list, tracking something over time, reminders, or asks for a multi-step action that would otherwise cost several tool calls.
---

# bb — the boxBot SDK

Inside `execute_script`: `import bb` or `import boxbot_sdk as bb` (same
module). Runs in the sandbox — separate venv, `boxbot-sandbox` user,
seccomp, read-only site-packages. Talks to the main process over JSON
on stdout.

**`bb` is for composition.** One script can search memory, pull photos,
write a note, and change the display. Cheaper and clearer than four
tool calls.

**Errors:** mutating calls (save, write, delete, create) raise
`bb.ActionError` when the main process rejects them. A failed write
never looks like a success. Read calls return response dicts. Wrap
risky writes in `try: … except bb.ActionError as e:` when you want to
handle the failure instead of dying.

## Modules

| Module | Does | Reach for it when |
|--------|------|-------------------|
| `bb.workspace` | Read, write, view, search your notes and CSVs. | You need *content* back, not just a fact — a list, draft, table, image. |
| `bb.camera` | Capture stills and crops. Images attach to the tool result — that *is* the vision API. | You want to see the room: an unrecognized speaker, an object someone points at. |
| `bb.photos` | Search, view, manage the photo library. | User wants a photo found, shown, or organized. |
| `bb.audio` | Play workspace audio (wav/flac/ogg/mp3) through the speaker. Mic detaches; wake word interrupts cleanly. | User wants a song, sound effect, recorded clip, chime. |
| `bb.display` | Create and update 7" screen displays. | Change or preview the screen, or author a new layout. |
| `bb.memory` | Save, search, invalidate memories. | Persist a durable fact, or dig deeper than `search_memory` went. |
| `bb.tasks` | Triggers (wake conditions) and to-dos. | Batching task edits with other SDK calls. One-off edits: use `manage_tasks`. |
| `bb.auth` | Registration codes, registered-user list, message all admins. | Admin adds a user (`generate_registration_code`), first boot (`generate_bootstrap_code`). Full flow: `onboarding` skill. |
| `bb.skill` | Create skills at runtime. | Teach yourself a recurring workflow. |
| `bb.integrations` | List, call, create, update, delete data pipes. Read execution logs. | Fresh external data, or registering a new pipe. **Calendar lives here**: `bb.integrations.get("calendar", action="list_upcoming_events", …)`. |
| `bb.packages` | Request a PyPI install; check status. | A script needs a new dependency. Admin approves out-of-band; returns `pending` immediately — check back with `status()`. |
| `bb.secrets` | Store credentials (write-only), list names, pass to scripts/integrations as env vars. | User pasted an API key; an integration declared a secret; a script needs a credential. |

## Deeper docs

Load one only when you need it:
`load_skill(name="bb", subpath="modules/<x>.md")` — from your main
loop, not from inside the sandbox.

`workspace` · `memory` · `tasks` · `audio` · `camera` · `photos` ·
`display` · `integrations` · `secrets` · `auth` · `packages` · `skill`

## Memory vs workspace

Memory = "rings a bell." Workspace = "now look it up."

Memory is small, searchable, auto-injected at conversation start — it
is how you *recognize* something is relevant. Workspace is
filesystem-backed — it is how you *retrieve* the content.

Pattern: Erik lists fifteen favorite Pokémon. Write the list to
`workspace/notes/people/erik/pokemon.md`. Save a one-line memory:
"Erik keeps a top-15 Pokémon list at `notes/people/erik/pokemon.md`."
Next time Erik comes up, the memory surfaces; you open the file.

Do NOT put the full list in memory — it dilutes retrieval and bloats
context. Do NOT hide facts in the workspace with no memory pointer —
you will never go looking.

## Quickstart

```python
import boxbot_sdk as bb

bb.workspace.write(
    "notes/people/erik.md",
    "- favorite pokemon: snorlax, pikachu, eevee\n- tea over coffee\n",
)

hits = bb.workspace.search("pokemon")
for h in hits:
    print(h["path"], h["line"], h["text"])

bb.workspace.csv_write("data/chores.csv", [
    {"task": "dishes", "assigned": "Emily", "done": False},
    {"task": "trash",  "assigned": "Jacob", "done": True},
])

bb.workspace.view("captures/erik_2026-04-24.jpg")   # pixels attach
```
