"""Tests for prefetch persistence: event log + trigger cache."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from boxbot.prefetch.bundle import PrefetchBundle
from boxbot.prefetch.store import (
    cache_get,
    cache_has_fresh,
    cache_put,
    cache_stamp_conversation,
    record_prefetch_event,
)


def _bundle():
    return PrefetchBundle(
        memories=[("m1", "s1"), ("m2", "s2")],
        skill_bodies={"weather": "body"},
        workspace_excerpts=[("notes/a.md", "hi")],
        pulled_data=[{"source": "calendar", "action": "list_upcoming_events",
                      "payload": {"n": 1}, "pulled_at": "2026-07-01T10:00:00Z"}],
        likely_next_note="remind Jacob",
    )


async def _store(tmp_path):
    from boxbot.memory.store import MemoryStore

    store = MemoryStore(db_path=tmp_path / "m.db")
    await store.initialize()
    return store


class TestEventLog:
    async def test_records_predicted_sets(self, tmp_path):
        store = await _store(tmp_path)
        try:
            b = _bundle()
            b.render(token_budget=1500)
            await record_prefetch_event(
                store, key="conv-1", key_kind="conversation",
                channel="whatsapp", mode="shadow", bundle=b,
                latency_ms=120, cost_usd=0.0004,
            )
            cur = await store.db.execute(
                """SELECT key, mode, predicted_memory_ids, predicted_skills,
                          predicted_integration_calls, note
                   FROM prefetch_events"""
            )
            row = (await cur.fetchall())[0]
            assert row[0] == "conv-1"
            assert row[1] == "shadow"
            assert "m1" in row[2] and "m2" in row[2]
            assert "weather" in row[3]
            assert "calendar" in row[4]
            assert row[5] == "remind Jacob"
        finally:
            await store.close()


class TestCache:
    async def test_put_get_roundtrip(self, tmp_path):
        store = await _store(tmp_path)
        try:
            expires = datetime.now(timezone.utc) + timedelta(minutes=20)
            await cache_put(
                store, trigger_id="t-1", bundle=_bundle(), expires_at=expires,
            )
            assert await cache_has_fresh(store, "t-1")
            got = await cache_get(store, "t-1")
            assert got is not None
            assert got.likely_next_note == "remind Jacob"
            assert got.predicted_memory_ids() == ["m1", "m2"]
        finally:
            await store.close()

    async def test_expired_returns_none(self, tmp_path):
        store = await _store(tmp_path)
        try:
            expires = datetime.now(timezone.utc) - timedelta(minutes=1)
            await cache_put(
                store, trigger_id="t-2", bundle=_bundle(), expires_at=expires,
            )
            assert await cache_get(store, "t-2") is None
            assert await cache_has_fresh(store, "t-2") is False
        finally:
            await store.close()

    async def test_missing_returns_none(self, tmp_path):
        store = await _store(tmp_path)
        try:
            assert await cache_get(store, "nope") is None
        finally:
            await store.close()

    async def test_stamp_conversation(self, tmp_path):
        store = await _store(tmp_path)
        try:
            expires = datetime.now(timezone.utc) + timedelta(minutes=20)
            await cache_put(
                store, trigger_id="t-3", bundle=_bundle(), expires_at=expires,
            )
            await cache_stamp_conversation(store, "t-3", "conv-xyz")
            cur = await store.db.execute(
                "SELECT conversation_id FROM prefetch_cache WHERE trigger_id='t-3'"
            )
            assert (await cur.fetchone())[0] == "conv-xyz"
        finally:
            await store.close()
