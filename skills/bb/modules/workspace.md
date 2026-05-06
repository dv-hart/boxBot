# bb.workspace — the agent's notebook

A filesystem-backed scratch space under `data/workspace/` that you own.
Read, write, view, search, and organise files however you like.

## When to use it

- Keeping a running note about a person, project, or decision.
- Maintaining a list or table that is too big or structured to put in
  memory (chores, reading list, budget, Pokémon roster).
- Drafting a reply or a plan before committing it.
- Saving an image you want to refer back to (a speaker crop, a whiteboard
  photo someone took).
- Building a CSV that a display binds to.

## When NOT to use it

- For durable *facts* the agent should recognise later without being
  told where to look. Those belong in memory (via `search_memory` tool
  or `bb.memory.save`).
- For secrets. Those belong in `bb.secrets`.
- For anything that fits naturally in the photo library — use
  `bb.photos` instead; the workspace is not a dumping ground for camera
  captures. The workspace is for things the *agent* wrote or curated.

## Layout convention

You organise the workspace however helps you. A sensible default:

```
workspace/
  notes/
    people/<name>.md
    projects/<slug>.md
    daily/2026-04-24.md
  data/
    chores.csv
    reading_list.csv
  captures/
    <ts>_<subject>.jpg
  drafts/
    response_to_<name>.md
```

The main process creates the root; subdirectories are created on demand
when you `write()`.

## API

All paths are **relative** to the workspace root. Absolute paths, `..`
segments, symlink escapes, and null bytes are rejected.

### Write / append / read

```python
bb.workspace.write(path, content)       # str or bytes; creates/overwrites
bb.workspace.append(path, text)         # text only; creates if missing
bb.workspace.read(path)                 # → {path, size, kind, content}
bb.workspace.read(path, binary=True)    # → {path, size, kind, binary: True}
                                        #   (bytes are not transported;
                                        #   use view() for images)
```

`write()` returns `{path, size, kind}`. The `kind` field is one of
`"text" | "image" | "csv" | "json" | "binary"`.

### Inspect

```python
bb.workspace.ls()                       # list the root
bb.workspace.ls("notes/people")         # list a subdir
bb.workspace.exists("notes/erik.md")    # → bool
```

Each `ls()` entry: `{path, size, modified, is_dir, kind}`.

### Search

A `grep`-style scan over all text/CSV/JSON files. Returns up to `limit`
hits; each hit is `{path, line, text}`. The query is a regex when valid,
otherwise a literal substring.

```python
hits = bb.workspace.search("pokemon")
hits = bb.workspace.search("TODO", path="notes/")
hits = bb.workspace.search(r"^\s*-\s+\[ \]", case_insensitive=False)
```

### View — the one that attaches images

```python
result = bb.workspace.view(path)
```

- Text / CSV / JSON → returns `{path, kind, content}` directly.
- Image (`.jpg`, `.png`, `.gif`, `.webp`) → the file is attached to the
  tool result as an image content block so you actually see the pixels.
  The result dict carries `{path, kind: "image", attached: True}`.
- Other binary → returns `{path, kind, message}` (no content).

Max 8 images attached per `execute_script` call; each image must be
≤4 MB.

### Delete

```python
bb.workspace.delete("drafts/old.md")    # file
bb.workspace.delete("drafts/")          # empty directory only
```

Refuses to remove a non-empty directory. There is no soft-delete today;
be deliberate.

### CSV helpers

```python
bb.workspace.csv_write("data/chores.csv", [
    {"task": "dishes", "assigned": "Emily", "done": False},
])
bb.workspace.csv_append("data/chores.csv", {"task": "vacuum", "assigned": "Erik", "done": False})
rows = bb.workspace.csv_read("data/chores.csv")   # → list[dict]
```

`csv_write` infers column order from the first row unless you pass
`fieldnames=[...]`. `csv_append` uses the existing header if the file is
there; otherwise it writes one from the row's keys.

## Quota

The workspace has a soft cap (default 100 MB). Writes that would exceed
the cap raise an error. Prune with `delete()` when you notice it filling
up — images saved here count; images in the photo library do not.

## Patterns

### Short note tied to a memory

```python
bb.workspace.write(
    "notes/people/erik/pokemon.md",
    "- snorlax\n- pikachu\n- eevee\n- gengar\n...",
)
bb.memory.save(
    content="Erik keeps a top-15 Pokémon list at notes/people/erik/pokemon.md — read the file for the current lineup.",
    memory_type="person",
    person="Erik",
    summary="Erik's Pokémon list lives at notes/people/erik/pokemon.md",
)
```

### Maintain a display-backing CSV

```python
bb.workspace.csv_append("data/chores.csv",
    {"task": "water plants", "assigned": "Jacob", "done": False})
# Later, a display bound to workspace:data/chores.csv re-renders.
```

### Review what you've written

```python
for entry in bb.workspace.ls("notes/people"):
    if not entry["is_dir"]:
        print(entry["path"], entry["size"])
```

### Grep for open TODOs

```python
for hit in bb.workspace.search(r"\[ \]"):
    print(hit["path"], hit["line"], hit["text"])
```
