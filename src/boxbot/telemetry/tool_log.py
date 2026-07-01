"""Append a ToolInvocation to the ``tool_invocations`` table.

One row per tool call the agent makes, across every channel (voice
included). This is the ground-truth signal the prefetch layer is tuned
and measured against: which searches/loads happened, in what turn, and
whether a repeat could have been avoided.

Mirrors ``boxbot.cost.record``: a loose Protocol over any object
exposing an aiosqlite ``.db`` connection, so this module doesn't depend
on ``boxbot.memory.store`` and the table can move DBs later without
churning callers. The timestamp is stamped and the tool input is
redacted at write time, so call sites stay trivial.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class _HasDB(Protocol):
    db: Any  # aiosqlite.Connection — kept loose to avoid an import cycle


@dataclass(slots=True)
class ToolInvocation:
    """One tool call. ``tool_input`` is redacted+truncated at write time."""

    tool_name: str
    conversation_id: str | None = None
    channel: str | None = None
    turn_number: int | None = None
    tool_input: dict | None = None
    result_status: str = "ok"  # ok | error | unknown_tool | dispatched
    latency_ms: int | None = None
    # Filled in later (active-mode matcher / offline analysis):
    # 'hit' | 'miss' | 'satisfiable'. Null while shadow-only.
    prefetch_attribution: str | None = None
    metadata: dict | None = field(default=None)


# Keys whose values are dropped from the stored input — belt-and-braces
# even though prefetch/tool inputs shouldn't carry raw secrets.
_SECRET_HINTS = ("secret", "token", "password", "credential", "api_key", "apikey")

_MAX_INPUT_CHARS = 200

_INSERT_SQL = """
INSERT INTO tool_invocations (
    timestamp,
    conversation_id,
    channel,
    turn_number,
    tool_name,
    tool_input_redacted,
    result_status,
    latency_ms,
    prefetch_attribution,
    metadata
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _redact(tool_input: dict | None) -> str | None:
    """Strip secret-looking keys and truncate to a bounded string.

    Returns a JSON-ish string (never raw secrets, never unbounded). On
    any serialization failure returns a short placeholder rather than
    raising — telemetry must not throw.
    """
    if not tool_input:
        return None
    try:
        cleaned = {
            k: ("<redacted>"
                if any(h in str(k).lower() for h in _SECRET_HINTS)
                else v)
            for k, v in tool_input.items()
        }
        s = json.dumps(cleaned, default=str, ensure_ascii=False)
    except Exception:
        return "<unserializable>"
    if len(s) > _MAX_INPUT_CHARS:
        s = s[:_MAX_INPUT_CHARS] + "…"
    return s


async def record_tool_invocation(store: _HasDB, inv: ToolInvocation) -> None:
    """Append one row to ``tool_invocations``.

    Not internally guarded — callers wrap this in try/except so the
    failure surface stays visible at the call site (mirrors how
    ``record_cost`` is invoked). Kept unguarded so tests can assert on
    write behavior directly.
    """
    await store.db.execute(
        _INSERT_SQL,
        (
            datetime.utcnow().isoformat(),
            inv.conversation_id,
            inv.channel,
            inv.turn_number,
            inv.tool_name,
            _redact(inv.tool_input),
            inv.result_status,
            inv.latency_ms,
            inv.prefetch_attribution,
            json.dumps(inv.metadata) if inv.metadata else None,
        ),
    )
    await store.db.commit()
