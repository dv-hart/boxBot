"""Tests for tool-invocation telemetry (``boxbot.telemetry.tool_log``).

Covers the ``tool_invocations`` table/schema, the ``record_tool_invocation``
writer, and the ``_redact`` helper (secret stripping + truncation).
Uses a real (temp-file) MemoryStore, mirroring ``tests/test_cost.py``.
"""

from __future__ import annotations

import json

from boxbot.telemetry.tool_log import (
    ToolInvocation,
    _redact,
    record_tool_invocation,
)


class TestRedact:
    def test_none_and_empty_return_none(self):
        assert _redact(None) is None
        assert _redact({}) is None

    def test_secret_like_keys_are_stripped(self):
        out = _redact({"api_key": "sk-abc", "token": "t", "query": "hi"})
        parsed = json.loads(out)
        assert parsed["api_key"] == "<redacted>"
        assert parsed["token"] == "<redacted>"
        assert parsed["query"] == "hi"

    def test_truncates_to_bounded_length(self):
        out = _redact({"query": "x" * 500})
        # Bounded (200 chars + ellipsis), not the full 500-char payload.
        assert len(out) < 260
        assert out.endswith("…")

    def test_unserializable_input_does_not_raise(self):
        out = _redact({"obj": object()})
        # default=str keeps it serializable; either way, never raises.
        assert isinstance(out, str)


class TestSchema:
    async def test_fresh_db_has_tool_invocations_table(self, tmp_path):
        from boxbot.memory.store import MemoryStore

        store = MemoryStore(db_path=tmp_path / "m.db")
        await store.initialize()
        try:
            cur = await store.db.execute("PRAGMA table_info(tool_invocations)")
            cols = {row[1] for row in await cur.fetchall()}
            for expected in (
                "timestamp",
                "conversation_id",
                "channel",
                "turn_number",
                "tool_name",
                "tool_input_redacted",
                "result_status",
                "latency_ms",
                "prefetch_attribution",
                "metadata",
            ):
                assert expected in cols, f"missing column: {expected}"
        finally:
            await store.close()


class TestRecord:
    async def test_writes_row_with_all_fields(self, tmp_path):
        from boxbot.memory.store import MemoryStore

        store = MemoryStore(db_path=tmp_path / "m.db")
        await store.initialize()
        try:
            await record_tool_invocation(
                store,
                ToolInvocation(
                    tool_name="search_memory",
                    conversation_id="conv-1",
                    channel="voice",
                    turn_number=3,
                    tool_input={"query": "erik pokemon", "api_key": "sk-x"},
                    result_status="ok",
                    latency_ms=42,
                    metadata={"backend": "raw"},
                ),
            )
            cur = await store.db.execute(
                """SELECT conversation_id, channel, turn_number, tool_name,
                          tool_input_redacted, result_status, latency_ms,
                          prefetch_attribution, metadata
                   FROM tool_invocations"""
            )
            rows = await cur.fetchall()
            assert len(rows) == 1
            row = rows[0]
            assert row[0] == "conv-1"
            assert row[1] == "voice"
            assert row[2] == 3
            assert row[3] == "search_memory"
            # secret stripped in the stored input
            assert "<redacted>" in row[4]
            assert "sk-x" not in row[4]
            assert row[5] == "ok"
            assert row[6] == 42
            assert row[7] is None  # prefetch_attribution unset in shadow phase
            assert json.loads(row[8]) == {"backend": "raw"}
        finally:
            await store.close()

    async def test_status_variants_persist(self, tmp_path):
        from boxbot.memory.store import MemoryStore

        store = MemoryStore(db_path=tmp_path / "m.db")
        await store.initialize()
        try:
            for status in ("ok", "error", "unknown_tool", "dispatched"):
                await record_tool_invocation(
                    store,
                    ToolInvocation(tool_name="t", result_status=status),
                )
            cur = await store.db.execute(
                "SELECT result_status FROM tool_invocations ORDER BY id"
            )
            got = [r[0] for r in await cur.fetchall()]
            assert got == ["ok", "error", "unknown_tool", "dispatched"]
        finally:
            await store.close()

    async def test_null_input_stores_null(self, tmp_path):
        from boxbot.memory.store import MemoryStore

        store = MemoryStore(db_path=tmp_path / "m.db")
        await store.initialize()
        try:
            await record_tool_invocation(
                store, ToolInvocation(tool_name="mute_mic", tool_input=None)
            )
            cur = await store.db.execute(
                "SELECT tool_input_redacted, metadata FROM tool_invocations"
            )
            row = (await cur.fetchall())[0]
            assert row[0] is None
            assert row[1] is None
        finally:
            await store.close()
