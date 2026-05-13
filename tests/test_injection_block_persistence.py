"""Tests for the persisted [Active Memories] block (lifecycle step 4).

The extraction model invalidates memories against a [Active Memories]
block. Before step 4 we only had memory IDs; the block was placeholders
that read "(content not preserved at submit time)". After step 4 the
agent captures the rendered block on the Conversation and forwards it
to the extraction batch via ``pending_extractions``.
"""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_pending_extraction_round_trips_block(tmp_path, monkeypatch):
    """Block stored on create_pending_extraction is returned on read."""
    from pathlib import Path as _Path
    from boxbot.memory.store import MemoryStore

    store = MemoryStore(db_path=_Path(tmp_path) / "mem.db")
    await store.initialize()
    try:
        block = (
            "[Active Memories]\n"
            "#abc12345 (person/Jacob): Jacob is allergic to shellfish.\n"
        )
        await store.create_pending_extraction(
            conversation_id="conv_test",
            transcript="some transcript",
            accessed_memory_ids=["abc12345"],
            channel="voice",
            participants=["boxBot", "Jacob"],
            started_at="2026-05-13T09:00:00",
            injected_memories_block=block,
        )
        row = await store.get_pending_extraction("conv_test")
        assert row is not None
        assert row.injected_memories_block == block
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_pending_extraction_defaults_block_to_empty(tmp_path, monkeypatch):
    """Backward-compat: callers that don't pass a block get empty
    string, not a crash."""
    from pathlib import Path as _Path
    from boxbot.memory.store import MemoryStore

    store = MemoryStore(db_path=_Path(tmp_path) / "mem.db")
    await store.initialize()
    try:
        await store.create_pending_extraction(
            conversation_id="conv_legacy",
            transcript="x",
            accessed_memory_ids=[],
            channel="voice",
            participants=["boxBot"],
            started_at="2026-05-13T09:00:00",
        )
        row = await store.get_pending_extraction("conv_legacy")
        assert row is not None
        assert row.injected_memories_block == ""
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_batch_poller_prefers_persisted_block_over_id_fallback(
    tmp_path, monkeypatch,
):
    """If the row carries a real injection block, the poller's batch
    submission uses it verbatim — not the minimal-IDs fallback."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from pathlib import Path as _Path
    from boxbot.memory.store import MemoryStore
    from boxbot.memory.batch_poller import BatchPoller

    store = MemoryStore(db_path=_Path(tmp_path) / "mem.db")
    await store.initialize()
    try:
        block = (
            "[Active Memories]\n"
            "#abc12345 (methodology): Use bb.integrations.get('calendar').\n"
        )
        await store.create_pending_extraction(
            conversation_id="conv_step4",
            transcript="real transcript here",
            accessed_memory_ids=["abc12345"],
            channel="whatsapp",
            participants=["boxBot", "Jacob"],
            started_at="2026-05-13T09:00:00",
            injected_memories_block=block,
        )
        row = await store.get_pending_extraction("conv_step4")
        assert row is not None

        captured: dict = {}

        async def fake_submit(
            client, *, injected_memories_block, **kwargs,
        ):
            captured["block"] = injected_memories_block
            captured["kwargs"] = kwargs
            return "batch_xyz"

        poller = BatchPoller(
            store=store,
            client=MagicMock(),
            model="claude-sonnet-4-6",
        )
        with patch(
            "boxbot.memory.batch_poller.submit_extraction_batch",
            new=fake_submit,
        ):
            await poller.submit(row)

        assert captured["block"] == block
        # And the ID-only fallback text is NOT present
        assert "content not preserved at submit time" not in captured["block"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_batch_poller_falls_back_to_ids_for_legacy_rows(
    tmp_path, monkeypatch,
):
    """Rows that predate step 4 (no block stored) get the legacy
    minimal-IDs rendering, so we don't break in-flight extractions
    across a deploy."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from pathlib import Path as _Path
    from boxbot.memory.store import MemoryStore
    from boxbot.memory.batch_poller import BatchPoller

    store = MemoryStore(db_path=_Path(tmp_path) / "mem.db")
    await store.initialize()
    try:
        await store.create_pending_extraction(
            conversation_id="conv_legacy2",
            transcript="x",
            accessed_memory_ids=["mem_old1", "mem_old2"],
            channel="voice",
            participants=["boxBot"],
            started_at="2026-05-13T09:00:00",
            # No block — simulates a pre-step-4 row.
        )
        row = await store.get_pending_extraction("conv_legacy2")
        assert row.injected_memories_block == ""

        captured: dict = {}

        async def fake_submit(
            client, *, injected_memories_block, **kwargs,
        ):
            captured["block"] = injected_memories_block
            return "batch_legacy"

        poller = BatchPoller(
            store=store,
            client=MagicMock(),
            model="claude-sonnet-4-6",
        )
        with patch(
            "boxbot.memory.batch_poller.submit_extraction_batch",
            new=fake_submit,
        ):
            await poller.submit(row)

        # Legacy fallback should mention the IDs and signal that
        # content wasn't preserved.
        assert "mem_old1" in captured["block"]
        assert "content not preserved" in captured["block"]
    finally:
        await store.close()
