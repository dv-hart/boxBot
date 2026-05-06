"""Example: take a note in the workspace + pair it with a memory.

The note holds the detail; the memory is the breadcrumb that reminds you
the note exists. Next time Erik comes up, the memory surfaces, you open
the file, and the full list is right there.
"""

import boxbot_sdk as bb

path = "notes/people/erik/pokemon.md"

bb.workspace.write(
    path,
    (
        "# Erik's Pokémon\n"
        "Updated 2026-04-24.\n\n"
        "- snorlax (favorite)\n"
        "- pikachu\n"
        "- eevee\n"
        "- gengar\n"
        "- dragonite\n"
    ),
)

memory_id = bb.memory.save(
    content=(
        f"Erik keeps a list of his top Pokémon at {path}. "
        "Open the file whenever the subject comes up — the list changes."
    ),
    memory_type="person",
    person="Erik",
    summary=f"Erik's top-Pokémon list lives at {path}",
)

print(f"wrote {path} and saved memory {memory_id} pointing to it")
