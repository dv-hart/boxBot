"""Daily background maintenance for the memory system.

Handles:
- Archive memories past their retention window
- Storage cap enforcement (evict oldest archived, then oldest active)
- FTS5 index rebuild

Usage:
    from boxbot.memory.maintenance import run_maintenance

    await run_maintenance(store)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from boxbot.memory.store import MemoryStore

logger = logging.getLogger(__name__)

# Retention windows per memory type (in days)
RETENTION_DAYS: dict[str, int] = {
    "person": 180,       # 6 months
    "household": 180,    # 6 months
    "methodology": 90,   # 3 months
    "operational": 60,   # 2 months
    "conversation": 60,  # 2 months
}

# Storage cap defaults (bytes)
DEFAULT_STORAGE_CAP = 50 * 1024 * 1024  # 50 MB
EVICTION_TRIGGER_RATIO = 0.99           # Start evicting at 99%
EVICTION_TARGET_RATIO = 0.70            # Evict down to 70%


async def run_maintenance(
    store: MemoryStore,
    *,
    storage_cap: int = DEFAULT_STORAGE_CAP,
) -> dict[str, int]:
    """Run all maintenance tasks.

    Args:
        store: The MemoryStore instance.
        storage_cap: Maximum database size in bytes.

    Returns:
        Dict with counts of actions taken:
        - archived_memories: memories moved to archived status
        - archived_conversations: conversations past retention
        - evicted_archived: permanently deleted archived records
        - evicted_active: permanently deleted active records (emergency)
        - fts_rebuilt: 1 if FTS was rebuilt, 0 otherwise
    """
    stats = {
        "archived_memories": 0,
        "archived_conversations": 0,
        "evicted_archived": 0,
        "evicted_active": 0,
        "fts_rebuilt": 0,
    }

    # 1. Archive memories past retention window
    stats["archived_memories"] = await _archive_stale_memories(store)

    # 2. Archive old conversations
    stats["archived_conversations"] = await _archive_old_conversations(store)

    # 3. Storage cap enforcement
    evicted = await _enforce_storage_cap(store, storage_cap)
    stats["evicted_archived"] = evicted["archived"]
    stats["evicted_active"] = evicted["active"]

    # 4. Rebuild FTS indexes (optimize)
    await _rebuild_fts(store)
    stats["fts_rebuilt"] = 1

    logger.info(
        "Maintenance complete: archived=%d memories + %d conversations, "
        "evicted=%d archived + %d active",
        stats["archived_memories"],
        stats["archived_conversations"],
        stats["evicted_archived"],
        stats["evicted_active"],
    )

    return stats


async def _archive_stale_memories(store: MemoryStore) -> int:
    """Archive active memories whose last_relevant_at exceeds their retention window."""
    now = datetime.utcnow()
    archived_count = 0

    for memory_type, retention_days in RETENTION_DAYS.items():
        if memory_type == "conversation":
            continue  # Handled separately

        cutoff = (now - timedelta(days=retention_days)).isoformat()

        cursor = await store.db.execute(
            """UPDATE memories
               SET status = 'archived'
               WHERE status = 'active'
                 AND type = ?
                 AND last_relevant_at < ?""",
            (memory_type, cutoff),
        )
        archived_count += cursor.rowcount

    await store.db.commit()
    return archived_count


async def _archive_old_conversations(store: MemoryStore) -> int:
    """Delete conversations past the 2-month retention window.

    Conversations are simply deleted (not archived) since the facts
    extracted from them persist independently via source_conversation FK.
    """
    cutoff_days = RETENTION_DAYS["conversation"]
    cutoff = (datetime.utcnow() - timedelta(days=cutoff_days)).isoformat()

    cursor = await store.db.execute(
        "SELECT id FROM conversations WHERE started_at < ?",
        (cutoff,),
    )
    rows = await cursor.fetchall()

    for row in rows:
        await store.delete_conversation(row["id"])

    return len(rows)


async def _enforce_storage_cap(
    store: MemoryStore,
    cap: int,
) -> dict[str, int]:
    """Enforce database size cap by evicting records.

    Strategy:
    1. If DB > 99% cap → evict archived memories oldest-first down to 70%
    2. If still > 70% after clearing archives → evict oldest active memories

    Returns:
        Dict with counts: {"archived": int, "active": int}
    """
    result = {"archived": 0, "active": 0}

    current_size = await store.get_db_size_bytes()
    trigger_size = int(cap * EVICTION_TRIGGER_RATIO)
    target_size = int(cap * EVICTION_TARGET_RATIO)

    if current_size < trigger_size:
        return result

    logger.warning(
        "Storage at %d bytes (%.1f%% of %d cap), evicting...",
        current_size, (current_size / cap) * 100, cap,
    )

    # Phase 1: Evict archived memories
    cursor = await store.db.execute(
        """SELECT id FROM memories
           WHERE status = 'archived'
           ORDER BY last_relevant_at ASC"""
    )
    archived_rows = await cursor.fetchall()

    for row in archived_rows:
        if await store.get_db_size_bytes() <= target_size:
            break
        await store.delete_memory(row["id"])
        result["archived"] += 1

    # Phase 2: If still over target, evict oldest active memories
    if await store.get_db_size_bytes() > target_size:
        cursor = await store.db.execute(
            """SELECT id FROM memories
               WHERE status = 'active'
               ORDER BY last_relevant_at ASC"""
        )
        active_rows = await cursor.fetchall()

        for row in active_rows:
            if await store.get_db_size_bytes() <= target_size:
                break
            await store.delete_memory(row["id"])
            result["active"] += 1

    logger.info(
        "Eviction complete: %d archived + %d active deleted, "
        "size now %d bytes",
        result["archived"], result["active"],
        await store.get_db_size_bytes(),
    )

    return result


async def _rebuild_fts(store: MemoryStore) -> None:
    """Rebuild FTS5 indexes for optimal search performance."""
    try:
        await store.db.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
        await store.db.execute("INSERT INTO conversations_fts(conversations_fts) VALUES('rebuild')")
        await store.db.commit()
    except Exception as e:
        logger.warning("FTS rebuild failed (non-fatal): %s", e)
