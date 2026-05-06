"""Tests for the persistent conversation store.

Covers:
    * CRUD round-trips (create, append turns, rehydrate, mark_extracted).
    * Rolling-window queries (get_active, list_active, list_extractable)
      using monkeypatched activity timestamps.
    * Atomicity of mark_extracted under simulated double-sweep.
    * Bulk append preserves order and bumps last_activity_at.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from boxbot.conversations.store import ConversationStore


@pytest.fixture
async def store(tmp_path: Path):
    s = ConversationStore(db_path=tmp_path / "conv.db")
    await s.initialize()
    try:
        yield s
    finally:
        await s.close()


def _shift_last_activity(
    store: ConversationStore,
    conversation_id: str,
    *,
    seconds_ago: float,
) -> str:
    """Helper — directly UPDATE last_activity_at_iso to the past."""
    target = (
        datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    ).isoformat()

    async def _do() -> str:
        db = store._require_db()
        await db.execute(
            "UPDATE conversations SET last_activity_at_iso = ? "
            "WHERE conversation_id = ?",
            (target, conversation_id),
        )
        await db.commit()
        return target

    return _do()  # returns coroutine; test awaits it


@pytest.mark.asyncio
async def test_create_and_get_roundtrip(store):
    rec = await store.create(
        channel="whatsapp",
        channel_key="whatsapp:+15551111111",
        participants={"Jacob"},
    )
    assert rec.state == "active"
    assert rec.participants == ["Jacob"]
    assert rec.summary is None

    fetched = await store.get(rec.conversation_id)
    assert fetched is not None
    assert fetched.conversation_id == rec.conversation_id
    assert fetched.channel_key == "whatsapp:+15551111111"


@pytest.mark.asyncio
async def test_append_turn_assigns_sequential_indexes(store):
    rec = await store.create(
        channel="whatsapp",
        channel_key="whatsapp:+15551111111",
    )
    idx_0 = await store.append_turn(
        rec.conversation_id, role="user",
        content={"role": "user", "content": "hello"},
    )
    idx_1 = await store.append_turn(
        rec.conversation_id, role="assistant",
        content={"role": "assistant", "content": "hi"},
    )
    idx_2 = await store.append_turn(
        rec.conversation_id, role="user",
        content={"role": "user", "content": "?"},
    )
    assert (idx_0, idx_1, idx_2) == (0, 1, 2)

    turns = await store.get_turns(rec.conversation_id)
    assert [t.turn_index for t in turns] == [0, 1, 2]
    assert [t.role for t in turns] == ["user", "assistant", "user"]


@pytest.mark.asyncio
async def test_get_thread_returns_anthropic_dicts_in_order(store):
    rec = await store.create(
        channel="whatsapp", channel_key="whatsapp:+15551111111",
    )
    await store.append_turn(
        rec.conversation_id, role="user",
        content={"role": "user", "content": "first"},
    )
    await store.append_turns(rec.conversation_id, [
        {"role": "assistant", "content": "second"},
        {"role": "assistant", "content": [{"type": "text", "text": "third"}]},
    ])
    thread = await store.get_thread(rec.conversation_id)
    assert thread == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "assistant", "content": [{"type": "text", "text": "third"}]},
    ]


@pytest.mark.asyncio
async def test_get_active_inside_window(store):
    rec = await store.create(
        channel="whatsapp", channel_key="whatsapp:+15551111111",
    )
    await store.append_turn(
        rec.conversation_id, role="user",
        content={"role": "user", "content": "hi"},
    )
    # Just appended — well inside any reasonable window.
    found = await store.get_active(
        "whatsapp:+15551111111", max_inactive_seconds=3600,
    )
    assert found is not None
    assert found.conversation_id == rec.conversation_id


@pytest.mark.asyncio
async def test_get_active_outside_window_returns_none(store):
    rec = await store.create(
        channel="whatsapp", channel_key="whatsapp:+15551111111",
    )
    # Force last_activity into the past.
    await _shift_last_activity(store, rec.conversation_id, seconds_ago=7200)
    found = await store.get_active(
        "whatsapp:+15551111111", max_inactive_seconds=3600,
    )
    assert found is None


@pytest.mark.asyncio
async def test_list_extractable_filters_by_window_and_state(store):
    fresh = await store.create(
        channel="whatsapp", channel_key="whatsapp:fresh",
    )
    expired = await store.create(
        channel="whatsapp", channel_key="whatsapp:expired",
    )
    other_channel = await store.create(
        channel="voice", channel_key="voice:room",
    )
    await _shift_last_activity(store, expired.conversation_id, seconds_ago=7200)
    await _shift_last_activity(store, other_channel.conversation_id, seconds_ago=7200)

    rows = await store.list_extractable(
        max_inactive_seconds=3600, channel="whatsapp",
    )
    ids = {r.conversation_id for r in rows}
    assert ids == {expired.conversation_id}
    # Without channel filter the voice row also shows up.
    rows_all = await store.list_extractable(max_inactive_seconds=3600)
    assert {r.conversation_id for r in rows_all} == {
        expired.conversation_id, other_channel.conversation_id,
    }


@pytest.mark.asyncio
async def test_mark_extracted_is_atomic(store):
    rec = await store.create(
        channel="whatsapp", channel_key="whatsapp:+15551111111",
    )
    flipped_first = await store.mark_extracted(
        rec.conversation_id, summary="done",
    )
    flipped_second = await store.mark_extracted(
        rec.conversation_id, summary="again",
    )
    assert flipped_first is True
    assert flipped_second is False
    fetched = await store.get(rec.conversation_id)
    assert fetched.state == "extracted"
    assert fetched.summary == "done"
    assert fetched.extracted_at_iso is not None


@pytest.mark.asyncio
async def test_list_active_warm_load_filters_by_channel(store):
    a = await store.create(channel="whatsapp", channel_key="whatsapp:a")
    b = await store.create(channel="whatsapp", channel_key="whatsapp:b")
    voice = await store.create(channel="voice", channel_key="voice:room")
    rows = await store.list_active(
        channel="whatsapp", max_inactive_seconds=3600,
    )
    ids = {r.conversation_id for r in rows}
    assert ids == {a.conversation_id, b.conversation_id}
    assert voice.conversation_id not in ids


@pytest.mark.asyncio
async def test_update_participants_round_trip(store):
    rec = await store.create(
        channel="whatsapp", channel_key="whatsapp:+15551111111",
        participants={"Jacob"},
    )
    await store.update_participants(
        rec.conversation_id, {"Jacob", "Carina"},
    )
    refreshed = await store.get(rec.conversation_id)
    assert sorted(refreshed.participants) == ["Carina", "Jacob"]


@pytest.mark.asyncio
async def test_append_turn_bumps_last_activity(store):
    rec = await store.create(
        channel="whatsapp", channel_key="whatsapp:+15551111111",
    )
    # Force last_activity into the past, then write a turn — it should
    # snap forward to "now".
    await _shift_last_activity(store, rec.conversation_id, seconds_ago=7200)
    before = await store.get(rec.conversation_id)
    assert before.last_activity_at_iso < datetime.now(
        timezone.utc
    ).isoformat()

    await store.append_turn(
        rec.conversation_id, role="user",
        content={"role": "user", "content": "still here"},
    )
    after = await store.get(rec.conversation_id)
    # Now within 60s of present.
    last_dt = datetime.fromisoformat(after.last_activity_at_iso)
    assert (
        datetime.now(timezone.utc) - last_dt
    ) < timedelta(seconds=60)


@pytest.mark.asyncio
async def test_delete_cascades_turns(store):
    rec = await store.create(
        channel="whatsapp", channel_key="whatsapp:+15551111111",
    )
    await store.append_turn(
        rec.conversation_id, role="user",
        content={"role": "user", "content": "hi"},
    )
    await store.delete(rec.conversation_id)
    assert await store.get(rec.conversation_id) is None
    turns = await store.get_turns(rec.conversation_id)
    assert turns == []
