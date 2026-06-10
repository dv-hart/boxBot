---
name: bb
description: The bb Python package — the agent's hands inside the sandbox. Import it from execute_script to capture photos from the camera, search and view the photo library, show things on the 7" display, take notes and maintain CSVs in a persistent workspace, search and save memories, manage triggers and to-dos, create new skills and displays, store secrets, and read/write the Google Calendar. Load this skill whenever the user asks about photos, the camera, what's on screen, note-taking, lists tracked over time, or anything that benefits from composing multiple operations in one turn.
when_to_use: User mentions photos, the camera, the display/screen, taking notes, keeping a list, tracking something over time, setting reminders, or asks for a multi-step action that would otherwise require several tool calls.
---

# bb — the boxBot SDK, in one package

Inside `execute_script` the package is importable both as `import bb`
and as `import boxbot_sdk` (same module); the examples below use
`import boxbot_sdk as bb`, but a bare `import bb` works identically.
Every call runs in the sandbox (separate venv, `boxbot-sandbox` user,
seccomp, read-only site-packages) and communicates with the main
process via structured JSON on stdout.

The key thing to internalise: **`bb` is about composition.** A single
script can search memory, pull a few photos, write a note, and change the
display. That is usually cheaper and clearer than four separate tool
calls.

**Error rule:** mutating calls (saves, writes, deletes, creates) raise
`bb.ActionError` when the main process rejects them — a write that
failed never looks like it succeeded. Read calls return raw response
dicts with documented shapes. Wrap risky writes in
`try: … except bb.ActionError as e:` when you want to handle the
failure instead of letting the script die.

## Modules

| Module | What it does | When to reach for it |
|--------|--------------|----------------------|
| `bb.workspace` | Read, write, view, search your own notes and CSVs. | You need to remember *content* (not just facts) — a list, a draft, a table, an image the user gave you. |
| `bb.camera` | Capture stills and cropped stills from the camera. Captured images attach to the tool result so you see the pixels directly — that *is* the vision API. | You want to *see* what's in the room — a speaker you don't recognise yet, an object someone points at. |
| `bb.photos` | Search, view, and manage the photo library. | User asks for a photo, wants one shown on the display, or wants to organise the library. |
| `bb.audio` | Play audio files (wav/flac/ogg/mp3) stored in the workspace through the speaker. The mic detaches during playback; the wake word interrupts cleanly. | User asks to hear a song, sound effect, recorded clip, or chime. |
| `bb.display` | Create and update displays on the 7" screen. | You want to change or preview what's on the screen, or author a new layout. |
| `bb.memory` | Save, search, invalidate memories. | Lets you persist a durable fact or dig into prior recall. The `search_memory` tool is fine for most lookups. |
| `bb.tasks` | Manage triggers (wake conditions) and to-do items. | Setting a reminder, scheduling a check-in, marking something done. The `manage_tasks` tool is fine for most edits. |
| `bb.auth` | Mint registration codes for new users, list registered users, message every admin. | Admin asks to add a new user (`generate_registration_code`), first-boot bootstrap (`generate_bootstrap_code`), or you need to know who's registered. See the `onboarding` skill for the full flow. |
| `bb.skill` | Create new skills at runtime. | You want to teach yourself a new recurring workflow. |
| `bb.integrations` | List, call, create, update, delete data-pipe integrations. Read execution logs. | You need fresh data from an external service (weather, calendar, stocks, …) or you want to register a new such pipe. **Calendar lives here** — call `bb.integrations.get("calendar", action="list_upcoming_events", …)`. |
| `bb.packages` | Request package installation; check request status. | A script needs a new PyPI dependency. An admin approves out-of-band; the request returns `pending` immediately — check back with `status()`. |
| `bb.secrets` | Store credentials (write-only); list names; hand them to scripts/integrations as env vars. | User pasted an API key; an integration declared a secret it needs; a one-off script needs a credential. |

## Progressive disclosure

This file is a map. For each module, there is a deeper doc under
`modules/`:

- `modules/workspace.md` — full API and patterns for the notebook
- `modules/memory.md` — `bb.memory` API: save/search/invalidate, id-prefix handling, the correction pattern
- `modules/tasks.md` — `bb.tasks` API for triggers (wake conditions) and to-do items
- `modules/audio.md` — `bb.audio` API for playing workspace audio files
- `modules/skill.md` — `bb.skill` API for creating new skills at runtime
- `modules/integrations.md` — `bb.integrations` API for calling and authoring data-pipe integrations
- `modules/secrets.md` — `bb.secrets` API for storing credentials and handing them to scripts/integrations
- `modules/auth.md` — `bb.auth` API for user/admin state, registration codes, and admin notifications
- `modules/packages.md` — `bb.packages` API for requesting PyPI installs (human-approved, async)
- (further modules land here as they become stable; check `ls modules/`)

Load a module doc only when you need it. The sandbox path to do so:

```python
# not from inside the sandbox — in your main orchestration loop,
# call load_skill(name="bb", subpath="modules/workspace.md").
```

## Memory vs workspace — pick the right one

Memory = "rings a bell". Workspace = "now look it up."

- **Memory** is small, searchable, and auto-injected into conversation
  context. It is how you *recognise* that something is relevant.
- **Workspace** is filesystem-backed. It is how you *retrieve* the
  content once memory said there was something to look up.

A good pattern:

1. User tells you Erik's favorite Pokémon. There are fifteen of them.
2. You save the list to `workspace/notes/people/erik/pokemon.md`.
3. You save a short memory: "Erik keeps a top-15 Pokémon list; it lives
   at `notes/people/erik/pokemon.md`."
4. Next time Erik comes up, the memory surfaces; you open the workspace
   file to get the list itself.

Do NOT stuff the full list into memory — it dilutes retrieval and bloats
context. Do NOT put every passing fact in the workspace — if memory
never mentions it, you will never go looking.

## Quickstart

```python
import boxbot_sdk as bb

# Take a note
bb.workspace.write(
    "notes/people/erik.md",
    "- favorite pokemon: snorlax, pikachu, eevee\n- tea over coffee\n",
)

# Search across notes
hits = bb.workspace.search("pokemon")
for h in hits:
    print(h["path"], h["line"], h["text"])

# Keep a CSV that powers a display
bb.workspace.csv_write("data/chores.csv", [
    {"task": "dishes", "assigned": "Emily", "done": False},
    {"task": "trash",  "assigned": "Jacob", "done": True},
])

# View an image (the pixels are attached to the tool result)
bb.workspace.view("captures/erik_2026-04-24.jpg")
```

For everything else, load the relevant module doc.
