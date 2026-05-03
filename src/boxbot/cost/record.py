"""Append a CostEvent to the cost_log table.

The cost log currently lives inside the memory database. We accept a
loose Protocol here (any object exposing an aiosqlite ``.db``
connection) so that this module does not depend on
``boxbot.memory.store`` and can move to its own DB later without
churning callers.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Protocol

from boxbot.cost.event import CostEvent

logger = logging.getLogger(__name__)


class _HasDB(Protocol):
    db: Any  # aiosqlite.Connection — runtime-checked, kept loose to avoid imports


_INSERT_SQL = """
INSERT INTO cost_log (
    timestamp,
    purpose,
    provider,
    model,
    input_tokens,
    output_tokens,
    cache_read_tokens,
    cache_write_5m_tokens,
    cache_write_1h_tokens,
    cache_write_tokens,
    is_batch,
    character_count,
    audio_seconds,
    iterations,
    correlation_id,
    cost_usd,
    metadata
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


async def record(store: _HasDB, event: CostEvent) -> None:
    """Append one row to ``cost_log``.

    ``cache_write_tokens`` (the legacy column) is dual-written from
    ``cache_write_5m_tokens + cache_write_1h_tokens`` so older SQL
    queries against that column continue to return correct totals.
    """
    legacy_cache_write = event.cache_write_5m_tokens + event.cache_write_1h_tokens
    await store.db.execute(
        _INSERT_SQL,
        (
            datetime.utcnow().isoformat(),
            event.purpose,
            event.provider,
            event.model,
            event.input_tokens,
            event.output_tokens,
            event.cache_read_tokens,
            event.cache_write_5m_tokens,
            event.cache_write_1h_tokens,
            legacy_cache_write,
            1 if event.is_batch else 0,
            event.character_count,
            event.audio_seconds,
            event.iterations,
            event.correlation_id,
            event.cost_usd,
            json.dumps(event.metadata) if event.metadata else None,
        ),
    )
    await store.db.commit()
