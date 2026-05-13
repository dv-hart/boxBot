"""Tests for accessed_memory_ids wiring (lifecycle step 6).

Before the fix, ``_apply_extraction_result`` hardcoded
``accessed_memories=[]`` when stamping the conversation log row, so
the dream phase's co-injection index (``dream._build_co_injection_index``)
saw empty sets for every conversation and couldn't tell which memory
pairs had already been considered together by daytime extraction.

After the fix, the IDs flow:
    Conversation.accessed_memory_ids
    → _post_conversation
    → create_pending_extraction (already correct)
    → batch_poller picks up row.accessed_memory_ids
    → process_extraction_result writes them onto the conversations row
"""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_process_extraction_writes_accessed_memories_to_conv_row(
    tmp_path, monkeypatch,
):
    from pathlib import Path as _Path

    from boxbot.memory.store import MemoryStore
    from boxbot.memory.extraction import (
        ExtractionResult,
        ConversationSummary,
        process_extraction_result,
    )

    store = MemoryStore(db_path=_Path(tmp_path) / "mem.db")
    await store.initialize()
    try:
        # Stub conversation row, as the live agent does at conversation start.
        await store.create_conversation_stub(
            conversation_id="conv_step6",
            channel="voice",
            participants=["Jacob"],
        )

        # Synthesize a minimal extraction result.
        result = ExtractionResult(
            conversation_summary=ConversationSummary(
                channel="voice",
                participants=["boxBot", "Jacob"],
                summary="Jacob asked about dinner.",
                topics=["dinner"],
            ),
            extracted_memories=[],
            invalidations=[],
            system_memory_updates=[],
        )

        accessed_ids = ["mem_a1b2c3", "mem_d4e5f6"]
        await process_extraction_result(
            store, result, "conv_step6",
            accessed_memory_ids=accessed_ids,
        )

        # Verify the conversations row carries the accessed memories.
        conv = await store.get_conversation("conv_step6")
        assert conv is not None
        assert conv.accessed_memories == accessed_ids
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_process_extraction_works_without_accessed_memory_ids(
    tmp_path, monkeypatch,
):
    """Backward-compat — callers that don't pass the new kwarg still work."""
    from pathlib import Path as _Path

    from boxbot.memory.store import MemoryStore
    from boxbot.memory.extraction import (
        ExtractionResult,
        ConversationSummary,
        process_extraction_result,
    )

    store = MemoryStore(db_path=_Path(tmp_path) / "mem.db")
    await store.initialize()
    try:
        await store.create_conversation_stub(
            conversation_id="conv_step6_legacy",
            channel="voice",
            participants=[],
        )
        result = ExtractionResult(
            conversation_summary=ConversationSummary(
                channel="voice",
                participants=["boxBot"],
                summary="x",
                topics=[],
            ),
            extracted_memories=[],
            invalidations=[],
            system_memory_updates=[],
        )
        # No accessed_memory_ids kwarg — must default cleanly to [].
        await process_extraction_result(
            store, result, "conv_step6_legacy",
        )
        conv = await store.get_conversation("conv_step6_legacy")
        assert conv is not None
        assert conv.accessed_memories == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_co_injection_index_sees_real_pairs(tmp_path, monkeypatch):
    """End-to-end: two memories that were co-injected into a single
    conversation end up in each other's co-injection sets, which is
    what the dream phase relies on to skip "this pair was already
    considered" comparisons."""
    from pathlib import Path as _Path

    from boxbot.memory.store import MemoryStore
    from boxbot.memory.extraction import (
        ExtractionResult,
        ConversationSummary,
        process_extraction_result,
    )
    from boxbot.memory.dream import _build_co_injection_index

    store = MemoryStore(db_path=_Path(tmp_path) / "mem.db")
    await store.initialize()
    try:
        # Create two real memories
        m1 = await store.create_memory(
            type="methodology",
            content="Use bb.integrations.get('calendar').",
            summary="calendar via integrations.get",
            person=None, people=[], tags=["calendar"],
        )
        m2 = await store.create_memory(
            type="household",
            content="bb.integrations has weather + calendar pipes.",
            summary="integrations: weather + calendar",
            person=None, people=[], tags=["calendar"],
        )

        # Simulate a conversation that surfaced both
        await store.create_conversation_stub(
            conversation_id="conv_coinj",
            channel="whatsapp",
            participants=["Jacob"],
        )
        result = ExtractionResult(
            conversation_summary=ConversationSummary(
                channel="whatsapp",
                participants=["boxBot", "Jacob"],
                summary="discussed calendar setup",
                topics=["calendar"],
            ),
            extracted_memories=[],
            invalidations=[],
            system_memory_updates=[],
        )
        await process_extraction_result(
            store, result, "conv_coinj",
            accessed_memory_ids=[m1, m2],
        )

        # Now ask the dream phase's co-injection helper what it sees
        m1_row = (await store.list_memories(status="active"))
        # Fetch as the dream module would
        memories = [m for m in m1_row if m.id in (m1, m2)]
        idx = await _build_co_injection_index(store, memories)
        assert "conv_coinj" in idx[m1]
        assert "conv_coinj" in idx[m2]
    finally:
        await store.close()
