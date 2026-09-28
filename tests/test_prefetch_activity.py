"""Recent-activity log (prefetch/activity.py).

Contract under test: the gather is deterministic (no model calls),
newest-first across the three sources (conversation log → pending
extractions → open text threads), deduped by conversation id, bounded
by window/limit/exclusions; lines never persist in a cached bundle;
the briefing carries a compressed, id-stripped copy; and every
rendered id resolves through search_memory(mode="transcript"),
including still-open text threads via the ConversationStore
fall-through.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from boxbot.prefetch.activity import gather_recent_activity
from boxbot.prefetch.bundle import PrefetchBundle
from boxbot.prefetch.request import PrefetchRequest


def _iso(minutes_ago: float) -> str:
    return (
        datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    ).isoformat()


@pytest_asyncio.fixture
async def conv_store(tmp_path):
    from boxbot.conversations.store import ConversationStore

    store = ConversationStore(db_path=tmp_path / "conversations.db")
    await store.initialize()
    yield store
    await store.close()


async def _seed_applied(memory_store, *, minutes_ago=40.0):
    return await memory_store.create_conversation(
        channel="voice",
        participants=["boxBot", "Jacob"],
        summary="Jacob asked how to include a z-wave lock",
        started_at=_iso(minutes_ago),
    )


async def _seed_pending(memory_store, *, cid="pend-1", minutes_ago=10.0):
    await memory_store.create_pending_extraction(
        conversation_id=cid,
        transcript=(
            "[Jacob]: how do I pair the new lock?\n"
            "[boxBot thought]: private notes\n"
            "[boxBot → Jacob via voice]: press the pairing button and say done"
        ),
        accessed_memory_ids=[],
        channel="voice",
        participants=["boxBot", "Jacob"],
        started_at=_iso(minutes_ago),
    )
    return cid


async def _seed_open_thread(conv_store):
    rec = await conv_store.create(
        channel="whatsapp",
        channel_key="whatsapp:+15551234567",
        participants={"Carina"},
    )
    await conv_store.append_turn(
        rec.conversation_id,
        role="user",
        content={"role": "user", "content": "[Carina]: did the plumber come?"},
    )
    await conv_store.append_turn(
        rec.conversation_id,
        role="assistant",
        content={
            "role": "assistant",
            "content": [
                {"type": "text", "text": "{\"thought\": \"private\"}"},
                {
                    "type": "tool_use",
                    "id": "tu1",
                    "name": "message",
                    "input": {
                        "channel": "text",
                        "to": "Carina",
                        "content": "Yes — he came at 2pm and fixed the leak.",
                    },
                },
            ],
        },
    )
    return rec.conversation_id


class TestGather:
    @pytest.mark.asyncio
    async def test_three_sources_newest_first_with_ids(
        self, memory_store, conv_store
    ):
        applied = await _seed_applied(memory_store, minutes_ago=40)
        pending = await _seed_pending(memory_store, minutes_ago=10)
        open_id = await _seed_open_thread(conv_store)

        lines = await gather_recent_activity(
            memory_store=memory_store, conversation_store=conv_store,
        )
        assert len(lines) == 3
        # Open thread wrote last (create/append use now()) — newest.
        assert f"[id {open_id}]" in lines[0]
        assert "still open" in lines[0]
        assert "did the plumber come?" in lines[0]
        assert "Carina" in lines[0]
        # Pending: first-utterance snippet stands in for the summary.
        assert f"[id {pending}]" in lines[1]
        assert "how do I pair the new lock?" in lines[1]
        # Applied: the extraction receipt.
        assert f"[id {applied}]" in lines[2]
        assert "z-wave lock" in lines[2]

    @pytest.mark.asyncio
    async def test_last_reply_on_newest_item_only(
        self, memory_store, conv_store
    ):
        await _seed_pending(memory_store, minutes_ago=10)
        await _seed_open_thread(conv_store)
        lines = await gather_recent_activity(
            memory_store=memory_store, conversation_store=conv_store,
        )
        assert "last reply:" in lines[0]
        assert "fixed the leak" in lines[0]
        assert "last reply:" not in lines[1]
        # Private internal notes never surface.
        assert "private" not in "\n".join(lines)

    @pytest.mark.asyncio
    async def test_exclude_window_and_limit(self, memory_store, conv_store):
        applied = await _seed_applied(memory_store, minutes_ago=40)
        pending = await _seed_pending(memory_store, minutes_ago=10)
        open_id = await _seed_open_thread(conv_store)

        # Exclude the current conversation.
        lines = await gather_recent_activity(
            memory_store=memory_store, conversation_store=conv_store,
            exclude_ids={open_id},
        )
        assert all(open_id not in l for l in lines)

        # Window: only the open thread (written just now) survives 5min.
        lines = await gather_recent_activity(
            memory_store=memory_store, conversation_store=conv_store,
            window_hours=5 / 60,
        )
        assert [l for l in lines if pending in l or applied in l] == []

        # Limit.
        lines = await gather_recent_activity(
            memory_store=memory_store, conversation_store=conv_store,
            limit=1,
        )
        assert len(lines) == 1

    @pytest.mark.asyncio
    async def test_pending_row_for_applied_conversation_dedups(
        self, memory_store
    ):
        applied = await _seed_applied(memory_store, minutes_ago=40)
        # Same conversation still has its (applied-status-lagging) row.
        await _seed_pending(memory_store, cid=applied, minutes_ago=40)
        lines = await gather_recent_activity(memory_store=memory_store)
        assert len(lines) == 1
        assert "z-wave lock" in lines[0]  # receipt wins over snippet

    @pytest.mark.asyncio
    async def test_summary_item_pulls_last_reply_from_transcript(
        self, memory_store
    ):
        applied = await _seed_applied(memory_store, minutes_ago=40)
        await _seed_pending(memory_store, cid=applied, minutes_ago=40)
        lines = await gather_recent_activity(memory_store=memory_store)
        assert 'last reply: "press the pairing button' in lines[0]

    @pytest.mark.asyncio
    async def test_missing_stores_and_empty_world(self, memory_store):
        assert await gather_recent_activity(
            memory_store=None, conversation_store=None,
        ) == []
        assert await gather_recent_activity(
            memory_store=memory_store, conversation_store=None,
        ) == []


class TestBundleRide:
    def test_render_is_empty_and_never_cached(self):
        bundle = PrefetchBundle()
        assert bundle.is_empty()
        bundle.recent_activity = ["- 10m ago · voice with Jacob: x [id abc]"]
        assert not bundle.is_empty()
        rendered = bundle.render(token_budget=20000)
        assert "Recent conversations" in rendered
        assert "[id abc]" in rendered
        # Never persisted: cached bundles must not serve a stale
        # recency index (same rule as live_context).
        d = bundle.to_dict()
        assert "recent_activity" not in d
        assert PrefetchBundle.from_dict(d).recent_activity == []


class TestBriefingRide:
    def test_briefing_compresses_and_strips_ids(self):
        req = PrefetchRequest(
            key="c1", key_kind="conversation", channel="voice",
            person="Jacob", text="I did it, what's next?",
            recent_activity=[
                "- 12m ago · voice with Jacob: asked how to include a "
                'z-wave lock [id abc-123]\n  last reply: "press the button"',
                "- 3h ago · whatsapp with Carina: plumber visit [id def-456]",
                "- 5h ago · voice with Erik: pokemon list [id ghi-789]",
            ],
        )
        briefing = req.briefing()
        assert "Recent prior conversations:" in briefing
        assert "z-wave lock" in briefing
        assert "plumber visit" in briefing
        # Max 2 lines, ids and continuation lines stripped.
        assert "pokemon" not in briefing
        assert "abc-123" not in briefing
        assert "last reply" not in briefing

    def test_briefing_unchanged_without_activity(self):
        req = PrefetchRequest(
            key="c1", key_kind="conversation", channel="voice", text="hi",
        )
        assert "Recent prior conversations" not in req.briefing()


class TestTranscriptFallThrough:
    @pytest.mark.asyncio
    async def test_open_thread_resolves_via_conversation_store(
        self, memory_store, conv_store, tmp_path, monkeypatch
    ):
        import boxbot.conversations.store as conv_store_mod
        from boxbot.memory.search import search_memories

        open_id = await _seed_open_thread(conv_store)
        # The fall-through opens its own read connection at the default
        # path; point it at this test's DB.
        monkeypatch.setattr(
            conv_store_mod, "DB_PATH", tmp_path / "conversations.db"
        )

        result = await search_memories(
            memory_store, mode="transcript", conversation_id=open_id,
        )
        assert result.get("error") is None
        assert result["channel"] == "whatsapp"
        assert result["thread_state"] == "active"
        assert "[Carina]: did the plumber come?" in result["transcript"]
        assert "fixed the leak" in result["transcript"]
        # Private internal-notes text blocks stay out.
        assert "thought" not in result["transcript"]

    @pytest.mark.asyncio
    async def test_unknown_id_still_errors(self, memory_store, monkeypatch):
        import boxbot.conversations.store as conv_store_mod
        from boxbot.memory.search import search_memories

        monkeypatch.setattr(
            conv_store_mod, "DB_PATH",
            memory_store._db_path.parent / "conversations.db",
        )
        result = await search_memories(
            memory_store, mode="transcript", conversation_id="nope",
        )
        assert "error" in result
