"""bb.workspace — the agent's persistent notebook.

The workspace is a filesystem-backed scratch space the agent owns. Use it
for anything you want to remember the *content* of — as opposed to memory,
which is for recognizing that something is *relevant*. A memory entry might
say "Erik's favorite Pokémon list is in notes/people/erik/pokemon.md"; the
workspace is where the actual list lives.

Common patterns:

    # Take a note
    bb.workspace.write("notes/people/erik.md", "- favorite pokemon: Snorlax\\n- prefers tea over coffee\\n")

    # Read it back later
    text = bb.workspace.read("notes/people/erik.md")["content"]

    # Search across notes
    hits = bb.workspace.search("pokemon")
    for h in hits:
        print(h["path"], h["line"], h["text"])

    # Keep a CSV that powers a display
    bb.workspace.csv_write("data/chores.csv", [
        {"task": "dishes", "assigned": "Emily", "done": False},
        {"task": "trash", "assigned": "Jacob", "done": True},
    ])

    # Append one row
    bb.workspace.csv_append("data/chores.csv", {"task": "vacuum", "assigned": "Erik", "done": False})

    # View an image (attaches it to the tool result so you can see it)
    bb.workspace.view("captures/erik_2026-04-24.jpg")

Paths are relative to the workspace root. No absolute paths, no ``..``.
"""

from __future__ import annotations

from typing import Any

from . import _transport, _validators as v


_TIMEOUT = 15


class WorkspaceError(Exception):
    """Raised on workspace operation failures (bad path, quota, etc)."""


def _check(resp: dict[str, Any]) -> dict[str, Any]:
    """Raise WorkspaceError if the main process returned an error."""
    if isinstance(resp, dict) and resp.get("status") == "error":
        raise WorkspaceError(resp.get("error", "unknown workspace error"))
    return resp


# ---------------------------------------------------------------------------
# Write / append / read
# ---------------------------------------------------------------------------

def write(path: str, content: str | bytes) -> dict[str, Any]:
    """Write (or overwrite) a file in the workspace.

    Args:
        path: Workspace-relative path, e.g. ``notes/people/erik.md``.
        content: Text (``str``) or binary (``bytes``) data.

    Returns:
        ``{path, size, kind}`` on success.
    """
    v.require_str(path, "path")
    if isinstance(content, (bytes, bytearray)):
        # Binary: base64-encode for transport
        import base64
        payload = {
            "path": path,
            "b64": base64.b64encode(bytes(content)).decode("ascii"),
        }
    else:
        v.require_str(content, "content", allow_empty=True)
        payload = {"path": path, "content": content}
    return _check(_transport.request("workspace.write", payload, timeout=_TIMEOUT))


def append(path: str, content: str) -> dict[str, Any]:
    """Append text to a file. Creates the file if it doesn't exist."""
    v.require_str(path, "path")
    v.require_str(content, "content", allow_empty=True)
    return _check(_transport.request(
        "workspace.append",
        {"path": path, "content": content},
        timeout=_TIMEOUT,
    ))


def read(path: str, *, binary: bool = False) -> dict[str, Any]:
    """Read a file's contents.

    For text files, returns ``{path, size, kind, content: str}``.
    For binary files (or when ``binary=True``), returns ``{path, size, kind,
    binary: True}`` — the bytes themselves are not transported; use
    :func:`view` to surface images.
    """
    v.require_str(path, "path")
    return _check(_transport.request(
        "workspace.read",
        {"path": path, "binary": bool(binary)},
        timeout=_TIMEOUT,
    ))


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def ls(path: str | None = None) -> list[dict[str, Any]]:
    """List entries in a workspace directory.

    Each entry: ``{path, size, modified, is_dir, kind}``.
    """
    payload: dict[str, Any] = {}
    if path is not None:
        v.require_str(path, "path", allow_empty=True)
        payload["path"] = path
    resp = _check(_transport.request("workspace.ls", payload, timeout=_TIMEOUT))
    return resp.get("entries", [])


def exists(path: str) -> bool:
    """Return True if the file or directory exists."""
    v.require_str(path, "path")
    resp = _check(_transport.request(
        "workspace.exists", {"path": path}, timeout=_TIMEOUT
    ))
    return bool(resp.get("exists", False))


def search(
    query: str,
    *,
    path: str | None = None,
    limit: int = 50,
    case_insensitive: bool = True,
) -> list[dict[str, Any]]:
    """Grep across text files in the workspace.

    Returns a list of ``{path, line, text}`` hits.
    """
    v.require_str(query, "query")
    v.require_int(limit, "limit", min_val=1, max_val=500)
    payload: dict[str, Any] = {
        "query": query,
        "limit": limit,
        "case_insensitive": bool(case_insensitive),
    }
    if path is not None:
        v.require_str(path, "path", allow_empty=True)
        payload["path"] = path
    resp = _check(_transport.request("workspace.search", payload, timeout=_TIMEOUT))
    return resp.get("hits", [])


# ---------------------------------------------------------------------------
# View — text returns inline, images attach to tool result
# ---------------------------------------------------------------------------

def view(path: str) -> dict[str, Any]:
    """Render a file for the agent.

    - Text / CSV / JSON: returns ``{path, kind, content}``.
    - Image: returns ``{path, kind: "image"}`` and attaches the image
      to the tool result so the agent sees the pixels.
    - Other binary: returns ``{path, kind, message}`` (no content).
    """
    v.require_str(path, "path")
    return _check(_transport.request(
        "workspace.view", {"path": path}, timeout=_TIMEOUT
    ))


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

def delete(path: str) -> dict[str, Any]:
    """Delete a file or an empty directory."""
    v.require_str(path, "path")
    return _check(_transport.request(
        "workspace.delete", {"path": path}, timeout=_TIMEOUT
    ))


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def csv_write(
    path: str,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str] | None = None,
) -> dict[str, Any]:
    """Write rows to a CSV, replacing any existing file.

    Args:
        path: Workspace-relative path, e.g. ``data/chores.csv``.
        rows: List of dicts (one per row).
        fieldnames: Column order. If omitted, inferred from the first row's
            keys in order of appearance across all rows.
    """
    v.require_str(path, "path")
    v.require_list(rows, "rows")
    payload: dict[str, Any] = {"path": path, "rows": rows}
    if fieldnames is not None:
        v.require_list(fieldnames, "fieldnames")
        payload["fieldnames"] = fieldnames
    return _check(_transport.request(
        "workspace.csv_write", payload, timeout=_TIMEOUT
    ))


def csv_append(path: str, row: dict[str, Any]) -> dict[str, Any]:
    """Append a single row to an existing CSV (creates it if missing)."""
    v.require_str(path, "path")
    v.require_dict(row, "row")
    return _check(_transport.request(
        "workspace.csv_append",
        {"path": path, "row": row},
        timeout=_TIMEOUT,
    ))


def csv_read(path: str) -> list[dict[str, Any]]:
    """Read a CSV as a list of dicts (uses the first row as header)."""
    v.require_str(path, "path")
    resp = _check(_transport.request(
        "workspace.csv_read", {"path": path}, timeout=_TIMEOUT
    ))
    return resp.get("rows", [])
