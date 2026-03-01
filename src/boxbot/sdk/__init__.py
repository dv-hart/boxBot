"""boxBot SDK — constrained, immutable API for agent-written sandbox scripts.

This SDK is the ONLY interface sandbox scripts have to boxBot internals.
It communicates with the main process through structured JSON on stdout.

Modules:
    display   — Declarative block-based display builder
    skill     — Skill builder for creating agent skills
    packages  — Package installation with user approval
    memory    — Memory store operations (save, search, delete)
    photos    — Photo library management
    tasks     — Trigger and to-do management
    secrets   — Write-only secret storage
"""

from . import display, skill, packages, memory, photos, tasks, secrets

__all__ = [
    "display",
    "skill",
    "packages",
    "memory",
    "photos",
    "tasks",
    "secrets",
]
