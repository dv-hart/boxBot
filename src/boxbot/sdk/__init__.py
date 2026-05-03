"""boxBot SDK — constrained, immutable API for agent-written sandbox scripts.

This SDK is the ONLY interface sandbox scripts have to boxBot internals.
It communicates with the main process through structured JSON on stdout.

Modules:
    auth         — User/admin state and registration code minting (no
                   raw secrets — RPC façade onto AuthManager)
    camera       — Capture stills from the Pi camera; images attach to
                   the tool result so the agent sees pixels directly
    display      — Declarative block-based display builder
    skill        — Skill builder for creating agent skills
    integrations — Call, list, create, update, delete data-pipe
                   integrations (weather, calendar, custom services)
    packages     — Package installation with user approval
    memory       — Memory store operations (save, search, delete)
    photos       — Photo library management
    tasks        — Trigger and to-do management
    secrets      — Write-only secret storage
    workspace    — Filesystem-backed notebook (read, write, view, search,
                   CSVs). The counterpart to memory: memory recognizes,
                   workspace holds content.

The calendar surface lives behind the ``calendar`` integration —
reach it via ``bb.integrations.get("calendar", action="...", ...)``.
"""

from . import (
    auth,
    camera,
    display,
    integrations,
    memory,
    packages,
    photos,
    secrets,
    skill,
    tasks,
    workspace,
)

__all__ = [
    "auth",
    "camera",
    "display",
    "integrations",
    "memory",
    "packages",
    "photos",
    "secrets",
    "skill",
    "tasks",
    "workspace",
]
