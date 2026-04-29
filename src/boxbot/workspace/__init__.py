"""Agent workspace — filesystem-backed notebook for the Claude agent.

The workspace is the agent's persistent scratch space: markdown notes,
CSVs that drive displays, JSON state, images saved from conversations.
Unlike memory (which is about recognition — "rings a bell"), the workspace
is about content (which is about lookup — "now go read the file").

Layout:
    data/workspace/
      notes/            — freeform markdown (agent decides substructure)
      data/             — CSVs/JSON referenced by displays or other workflows
      captures/         — images the agent captured or saved

Security model:
- The sandbox user owns the directory and can read/write within it.
- All operations go through ``store.Workspace`` which validates paths
  (no ``..``, no absolute paths, no symlinks) and enforces a size quota.
- The ``bb.workspace`` SDK module is the sandbox-side interface; this
  package is the main-process backend that services those actions.
"""

from __future__ import annotations

from boxbot.workspace.store import Workspace, WorkspaceError

__all__ = ["Workspace", "WorkspaceError"]
