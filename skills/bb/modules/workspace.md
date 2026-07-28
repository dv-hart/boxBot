# bb.workspace — your notebook

Filesystem scratch space under `data/workspace/` that you own. Read,
write, view, search, organise.

**Use for:** running notes on a person or project; lists and tables too
big or structured for memory (chores, reading list, budget); drafts;
images you want to revisit (speaker crop, whiteboard photo); CSVs that
back a display.

**Not for:** durable *facts* you should recognize later without being
told where to look → memory. Credentials → `bb.secrets`. Camera
captures and household photos → `bb.photos`. The workspace holds what
*you* wrote or curated.

## Layout

Yours to organize. Sensible default:

```
workspace/
  notes/people/<name>.md
  notes/projects/<slug>.md
  notes/daily/2026-04-24.md
  data/chores.csv
  captures/<ts>_<subject>.jpg
  drafts/response_to_<name>.md
```

Root exists already. Subdirectories are created on `write()`.

## API

Paths are **relative** to the workspace root. Absolute paths, `..`
segments, symlink escapes, and null bytes are rejected.

### Write, append, read

```python
bb.workspace.write(path, content)       # str or bytes; creates/overwrites
bb.workspace.append(path, text)         # text only; creates if missing
bb.workspace.read(path)                 # → {path, size, kind, content}
bb.workspace.read(path, binary=True)    # → {path, size, kind, binary: True}
                                        #   bytes not transported; use view()
```

`write()` returns `{path, size, kind}`. `kind` is
`"text" | "image" | "csv" | "json" | "binary"`.

### Inspect

```python
bb.workspace.ls()                       # root
bb.workspace.ls("notes/people")         # subdir
bb.workspace.exists("notes/erik.md")    # → bool
```

`ls()` entry: `{path, size, modified, is_dir, kind}`.

### Search

`grep`-style over text/CSV/JSON files. Returns up to `limit` hits as
`{path, line, text}`. Query is a regex when valid, else a literal
substring.

```python
hits = bb.workspace.search("pokemon")
hits = bb.workspace.search("TODO", path="notes/")
hits = bb.workspace.search(r"^\s*-\s+\[ \]", case_insensitive=False)
```

### View — this is what attaches images

```python
result = bb.workspace.view(path)
```

- Text / CSV / JSON → `{path, kind, content}`.
- Image (`.jpg`, `.png`, `.gif`, `.webp`) → attached to the tool result
  as an image block; you see the pixels. Returns
  `{path, kind: "image", attached: True}`.
- Other binary → `{path, kind, message}`, no content.

Max 8 images per `execute_script` call, ≤4 MB each.

### Delete

```python
bb.workspace.delete("drafts/old.md")    # file
bb.workspace.delete("drafts/")          # empty directory only
```

Refuses non-empty directories. No soft-delete. Be deliberate.

### CSV

```python
bb.workspace.csv_write("data/chores.csv", [
    {"task": "dishes", "assigned": "Emily", "done": False},
])
bb.workspace.csv_append("data/chores.csv",
    {"task": "vacuum", "assigned": "Erik", "done": False})
rows = bb.workspace.csv_read("data/chores.csv")   # → list[dict]
```

`csv_write` infers column order from the first row unless you pass
`fieldnames=[...]`. `csv_append` reuses the existing header, or writes
one from the row's keys.

## Quota and errors

Soft cap, default 100 MB. Over-cap writes raise. Prune with
`delete()`. Images here count against it; photo-library images do not.

Every failure raises `WorkspaceError` (bad path, missing file, quota).
It subclasses `bb.ActionError`.

## Patterns

Note plus memory pointer:

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

Feed a display:

```python
bb.workspace.csv_append("data/chores.csv",
    {"task": "water plants", "assigned": "Jacob", "done": False})
# A display bound to workspace:data/chores.csv re-renders.
```

Grep open TODOs:

```python
for hit in bb.workspace.search(r"\[ \]"):
    print(hit["path"], hit["line"], hit["text"])
```
