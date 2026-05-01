---
name: bb
description: The bb Python package — the agent's hands inside the sandbox. Import it from execute_script to capture photos from the camera, search and view the photo library, show things on the 7" display, take notes and maintain CSVs in a persistent workspace, search and save memories, manage triggers and to-dos, create new skills and displays, store secrets, and read/write the Google Calendar. Load this skill whenever the user asks about photos, the camera, what's on screen, note-taking, lists tracked over time, or anything that benefits from composing multiple operations in one turn.
when_to_use: User mentions photos, the camera, the display/screen, taking notes, keeping a list, tracking something over time, setting reminders, or asks for a multi-step action that would otherwise require several tool calls.
---

# bb — the boxBot SDK, in one package

`bb` is importable as `boxbot_sdk` inside `execute_script`; the examples
below use `import boxbot_sdk as bb` for brevity. Every call runs in the
sandbox (separate venv, `boxbot-sandbox` user, seccomp, read-only
site-packages) and communicates with the main process via structured
JSON on stdout.

The key thing to internalise: **`bb` is about composition.** A single
script can search memory, pull a few photos, write a note, and change the
display. That is usually cheaper and clearer than four separate tool
calls.

## Modules

| Module | What it does | When to reach for it |
|--------|--------------|----------------------|
| `bb.workspace` | Read, write, view, search your own notes and CSVs. | You need to remember *content* (not just facts) — a list, a draft, a table, an image the user gave you. |
| `bb.camera` | Capture stills and cropped stills from the camera. Captured images attach to the tool result so you see the pixels directly — that *is* the vision API. | You want to *see* what's in the room — a speaker you don't recognise yet, an object someone points at. |
| `bb.photos` | Search, view, and manage the photo library. | User asks for a photo, wants one shown on the display, or wants to organise the library. |
| `bb.display` | Create and update displays on the 7" screen. | You want to change or preview what's on the screen, or author a new layout. |
| `bb.memory` | Save, search, invalidate memories. | Lets you persist a durable fact or dig into prior recall. The `search_memory` tool is fine for most lookups. |
| `bb.tasks` | Manage triggers (wake conditions) and to-do items. | Setting a reminder, scheduling a check-in, marking something done. The `manage_tasks` tool is fine for most edits. |
| `bb.auth` | Mint registration codes for new users, list registered users, message every admin. | Admin asks to add a new user (`generate_registration_code`), first-boot bootstrap (`generate_bootstrap_code`), or you need to know who's registered. See the `onboarding` skill for the full flow. |
| `bb.skill` | Create new skills at runtime. | You want to teach yourself a new recurring workflow. |
| `bb.packages` | Request package installation. | A script needs a new PyPI dependency. Requires human approval. |
| `bb.secrets` | Store and use credentials (write-only after store). | Before calling a third-party API on behalf of the user. |
| `bb.calendar` | Read and write Google Calendar. | Any calendar operation. |

## Progressive disclosure

This file is a map. For each module, there is a deeper doc under
`modules/`:

- `modules/workspace.md` — full API and patterns for the notebook
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
