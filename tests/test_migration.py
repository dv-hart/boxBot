"""Verify that initialising MemoryStore on a pre-existing DB (with the
older schema) cleanly adds the new pending_extractions and cost_log
tables without harming existing data."""

from __future__ import annotations

import sqlite3

import pytest


@pytest.mark.asyncio
async def test_migration_preserves_old_data(tmp_path):
    from boxbot.memory.store import MemoryStore

    path = tmp_path / "m.db"
    # Pre-create with the OLD schema only
    with sqlite3.connect(str(path)) as c:
        c.executescript("""
            CREATE TABLE memories (
                id TEXT PRIMARY KEY, type TEXT, content TEXT,
                summary TEXT, person TEXT, people TEXT, tags TEXT,
                source_conversation TEXT, created_at TEXT, last_relevant_at TEXT,
                status TEXT DEFAULT 'active', invalidated_by TEXT,
                superseded_by TEXT, embedding BLOB
            );
            CREATE TABLE conversations (
                id TEXT PRIMARY KEY, channel TEXT, participants TEXT,
                started_at TEXT, summary TEXT, topics TEXT,
                accessed_memories TEXT, embedding BLOB
            );
            CREATE TABLE system_memory_versions (
                version INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT,
                updated_at TEXT, updated_by TEXT, change_summary TEXT
            );
            INSERT INTO memories VALUES (
                'm1','person','test','t',NULL,'[]','[]',NULL,
                '2026-04-29','2026-04-29','active',NULL,NULL,NULL
            );
        """)

    store = MemoryStore(db_path=path)
    await store.initialize()

    # Old data intact
    cur = await store.db.execute("SELECT id FROM memories")
    rows = await cur.fetchall()
    assert [r[0] for r in rows] == ["m1"]

    # New tables present
    cur = await store.db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = {r[0] for r in await cur.fetchall()}
    assert "pending_extractions" in tables
    assert "cost_log" in tables

    # New CRUD works
    await store.create_pending_extraction(
        conversation_id="x", transcript="t", accessed_memory_ids=[],
        channel="voice", participants=["Jacob"],
        started_at="2026-04-29T10:00:00",
    )
    row = await store.get_pending_extraction("x")
    assert row is not None
    assert row.status == "queued"

    await store.close()
