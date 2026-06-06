"""Tests for dispatch-as-bridge (PR 2.2).

When a trigger conversation delivers a text, that delivery is also
recorded into the recipient's own real conversation — so a reply
continues a thread that contains the briefing, instead of landing in
a disconnected fresh conversation.

Covers:
  * _delivered_text_messages_from_thread — parsing message tool calls
  * Conversation.ingest_trigger_delivery — IDLE/LISTENING immediate
    fold-in, THINKING deferred fold-in (the mid-generation race),
    ENDED rejection
  * build_trigger_context_turns — the shared turn shape (full reasoning)
  * tool-name normalization (base_tool_name) so the SDK backend's
    mcp__boxbot_tools__message deliveries are seen by the scanner/receipt
"""
from __future__ import annotations

import pytest

from boxbot.core.agent import (
    _delivered_text_messages_from_thread,
    _summarize_trigger_thread,
)
from boxbot.core.agent_sdk_adapter import base_tool_name, mcp_tool_name
from boxbot.core.conversation import Conversation, ConversationState


# ---------------------------------------------------------------------------
# _delivered_text_messages_from_thread
# ---------------------------------------------------------------------------


def _assistant_message(
    to: str, channel: str, content: str, *, name: str = "message",
) -> dict:
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_x",
                "name": name,
                "input": {"to": to, "channel": channel, "content": content},
            }
        ],
    }


def _ctx_turns(
    recipient: str = "Jacob",
    delivered: str = "Morning briefing — Thu.",
    transcript: str = "[boxBot → Jacob via text]: Morning briefing — Thu.",
) -> list[dict]:
    return Conversation.build_trigger_context_turns(
        description="Morning briefing",
        transcript=transcript,
        recipient=recipient,
        delivered_texts=[delivered],
    )


class TestDeliveredTextParser:
    def test_extracts_text_deliveries(self):
        msgs = [
            {"role": "user", "content": "[trigger] [Trigger fired: Morning briefing]"},
            _assistant_message("Jacob", "text", "Morning briefing — Thu."),
            _assistant_message("Carina", "text", "Morning briefing — Thu."),
        ]
        out = _delivered_text_messages_from_thread(msgs)
        assert ("Jacob", "Morning briefing — Thu.") in out
        assert ("Carina", "Morning briefing — Thu.") in out

    def test_skips_voice_deliveries(self):
        """voice:room is transient — a spoken reply already lands there,
        so voice deliveries are not bridged."""
        msgs = [_assistant_message("room", "speak", "Good morning.")]
        assert _delivered_text_messages_from_thread(msgs) == []

    def test_skips_unresolved_recipients(self):
        """A wake-cycle trigger has no current_speaker; a real delivery
        always names a registered user explicitly."""
        msgs = [
            _assistant_message("current_speaker", "text", "hi"),
            _assistant_message("unknown", "text", "hi"),
            _assistant_message("room", "text", "hi"),
        ]
        assert _delivered_text_messages_from_thread(msgs) == []

    def test_ignores_non_message_tool_calls(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "switch_display",
                        "input": {"display_name": "weather"},
                    }
                ],
            }
        ]
        assert _delivered_text_messages_from_thread(msgs) == []

    def test_extracts_mcp_namespaced_deliveries(self):
        """Regression: the SDK backend records the delivery as
        mcp__boxbot_tools__message. The scanner missed it before, so the
        bridge bridged nothing and replies lost the briefing."""
        msgs = [
            {"role": "user", "content": "[Trigger fired: Morning briefing]"},
            _assistant_message(
                "Jacob", "text", "Morning briefing — Fri.",
                name=mcp_tool_name("message"),
            ),
        ]
        out = _delivered_text_messages_from_thread(msgs)
        assert ("Jacob", "Morning briefing — Fri.") in out


class TestTriggerReceipt:
    def test_receipt_recognizes_mcp_namespaced_delivery(self):
        """The receipt summarizer must also see SDK-namespaced messages,
        or it reports 'nothing delivered' for a briefing that went out."""
        msgs = [
            {"role": "user", "content": "[Trigger fired: Morning briefing]"},
            _assistant_message(
                "Jacob", "text", "hi",
                name=mcp_tool_name("message"),
            ),
            _assistant_message(
                "Carina", "text", "hi",
                name=mcp_tool_name("message"),
            ),
        ]
        receipt = _summarize_trigger_thread(msgs, "2026-06-05T14:00:00")
        assert "Delivered" in receipt
        assert "Jacob" in receipt and "Carina" in receipt
        assert "nothing delivered" not in receipt


def test_base_tool_name_strips_prefix_and_passes_bare_through():
    assert base_tool_name(mcp_tool_name("message")) == "message"
    assert base_tool_name("message") == "message"
    assert base_tool_name("mcp__boxbot_tools__execute_script") == (
        "execute_script"
    )


# ---------------------------------------------------------------------------
# build_trigger_context_turns
# ---------------------------------------------------------------------------


def test_build_context_turns_shape():
    turns = Conversation.build_trigger_context_turns(
        description="Morning briefing",
        transcript="[boxBot used tool: execute_script]\n"
        "[Tool Result]: {'id': '193ptomq', 'title': 'Zara Preschool "
        "Graduation'}\n[boxBot → Jacob via text]: Morning briefing — Thu.",
        recipient="Jacob",
        delivered_texts=["Morning briefing — Thu."],
    )
    assert len(turns) == 2
    # Framing turn: user role, [trigger] prefix (so it's not mistaken
    # for a human reply and role-alternation holds).
    assert turns[0]["role"] == "user"
    assert turns[0]["content"].startswith("[trigger]")
    assert "Jacob" in turns[0]["content"]
    # Full reasoning is threaded — the calendar event id is present, so a
    # reply can fix the upstream source, not just a memory.
    assert "193ptomq" in turns[0]["content"]
    # The delivered text is an assistant turn — BB said it. Ends on
    # assistant so the next inbound user reply alternates cleanly.
    assert turns[1]["role"] == "assistant"
    assert turns[1]["content"] == "Morning briefing — Thu."


def test_build_context_turns_truncates_huge_transcript():
    turns = Conversation.build_trigger_context_turns(
        description="x",
        transcript="A" * 50_000,
        recipient="Jacob",
        delivered_texts=["done"],
        max_transcript_chars=1_000,
    )
    assert "transcript truncated" in turns[0]["content"]
    assert len(turns[0]["content"]) < 2_000


# ---------------------------------------------------------------------------
# Conversation.ingest_trigger_delivery
# ---------------------------------------------------------------------------


def _make_conversation(state: ConversationState) -> Conversation:
    """A minimal Conversation in a chosen state, no store, no generation."""
    conv = object.__new__(Conversation)
    import asyncio
    conv.conversation_id = "conv_recipient"
    conv.channel = "whatsapp"
    conv._thread = [{"role": "user", "content": "earlier message"}]
    conv._state = state
    conv._store = None
    conv._lock = asyncio.Lock()
    conv._pending_external_turns = []
    conv._last_activity_monotonic = 0.0
    # _reset_silence_timer is a no-op when there's no timer task wired;
    # stub it so the minimal object doesn't need the full timer setup.
    conv._reset_silence_timer = lambda: None
    return conv


@pytest.mark.asyncio
async def test_ingest_when_listening_folds_in_immediately():
    conv = _make_conversation(ConversationState.LISTENING)
    ok = await conv.ingest_trigger_delivery(
        recipient="Jacob", turns=_ctx_turns(),
    )
    assert ok is True
    # Turns are in the live thread right away.
    assert conv._thread[-1]["role"] == "assistant"
    assert conv._thread[-1]["content"] == "Morning briefing — Thu."
    assert conv._thread[-2]["content"].startswith("[trigger]")
    # Nothing deferred.
    assert conv._pending_external_turns == []


@pytest.mark.asyncio
async def test_ingest_when_thinking_defers_fold_in():
    """The mid-generation race: appending to _thread under an in-flight
    API call would corrupt it. THINKING deliveries are deferred to
    _pending_external_turns instead."""
    conv = _make_conversation(ConversationState.THINKING)
    thread_len_before = len(conv._thread)
    ok = await conv.ingest_trigger_delivery(
        recipient="Jacob", turns=_ctx_turns(),
    )
    assert ok is True
    # The live thread is untouched mid-generation...
    assert len(conv._thread) == thread_len_before
    # ...the turns wait in the deferred queue.
    assert len(conv._pending_external_turns) == 2
    assert conv._pending_external_turns[-1]["content"] == "Morning briefing — Thu."


@pytest.mark.asyncio
async def test_ingest_when_ended_returns_false():
    """An ENDED conversation can't accept turns — caller falls back to
    a fresh store-backed thread."""
    conv = _make_conversation(ConversationState.ENDED)
    ok = await conv.ingest_trigger_delivery(
        recipient="Jacob", turns=_ctx_turns(),
    )
    assert ok is False
    assert conv._pending_external_turns == []


@pytest.mark.asyncio
async def test_deferred_turns_fold_in_preserve_order():
    """Two deliveries during THINKING both land, in order, when folded."""
    conv = _make_conversation(ConversationState.THINKING)
    await conv.ingest_trigger_delivery(
        recipient="Jacob", turns=_ctx_turns(delivered="first"),
    )
    await conv.ingest_trigger_delivery(
        recipient="Jacob", turns=_ctx_turns(delivered="second"),
    )
    # 2 turns per delivery
    assert len(conv._pending_external_turns) == 4
    assert conv._pending_external_turns[1]["content"] == "first"
    assert conv._pending_external_turns[3]["content"] == "second"


# ---------------------------------------------------------------------------
# Agent._bridge_trigger_delivery — the two routing branches
# ---------------------------------------------------------------------------


class _FakeUser:
    def __init__(self, name: str, phone: str) -> None:
        self.name = name
        self.phone = phone


@pytest.mark.asyncio
async def test_bridge_routes_into_live_conversation(monkeypatch, mock_config):
    """If the recipient has a live in-memory conversation, the delivery
    folds into it — not a new store row."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from boxbot.core import agent as agent_mod

    agent = object.__new__(agent_mod.BoxBotAgent)
    agent._index_lock = asyncio.Lock()

    # A live recipient conversation, LISTENING.
    live = _make_conversation(ConversationState.LISTENING)
    live.conversation_id = "conv_jacob_live"
    agent._conversations = {"conv_jacob_live": live}
    agent._conversation_by_key = {"whatsapp:+1555": "conv_jacob_live"}

    store = MagicMock()
    store.get_or_create_active = AsyncMock()
    store.append_turns = AsyncMock()
    agent._conversation_store = store

    fake_auth = MagicMock()
    fake_auth.list_users = AsyncMock(return_value=[_FakeUser("Jacob", "+1555")])
    monkeypatch.setattr(
        "boxbot.communication.auth.get_auth_manager", lambda: fake_auth,
    )

    await agent._bridge_trigger_delivery("Jacob", _ctx_turns())

    # Folded into the live conversation...
    assert live._thread[-1]["content"] == "Morning briefing — Thu."
    # ...and the store-only path was NOT taken.
    store.get_or_create_active.assert_not_awaited()
    store.append_turns.assert_not_awaited()


@pytest.mark.asyncio
async def test_bridge_routes_to_store_when_no_live_conversation(
    monkeypatch, mock_config,
):
    """No live conversation for the recipient → the delivery goes to
    the store (get_or_create_active + append_turns), so the next
    inbound rehydrates a thread containing the briefing."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from boxbot.core import agent as agent_mod

    agent = object.__new__(agent_mod.BoxBotAgent)
    agent._index_lock = asyncio.Lock()
    agent._conversations = {}
    agent._conversation_by_key = {}

    record = MagicMock()
    record.conversation_id = "conv_carina_stored"
    store = MagicMock()
    store.get_or_create_active = AsyncMock(return_value=(record, True))
    store.append_turns = AsyncMock()
    agent._conversation_store = store

    fake_auth = MagicMock()
    fake_auth.list_users = AsyncMock(
        return_value=[_FakeUser("Carina", "+1777")]
    )
    monkeypatch.setattr(
        "boxbot.communication.auth.get_auth_manager", lambda: fake_auth,
    )

    await agent._bridge_trigger_delivery(
        "Carina",
        _ctx_turns(
            recipient="Carina",
            transcript="[boxBot used tool: execute_script]\n[Tool Result]: "
            "{'id': '193ptomq', 'title': 'Zara Preschool Graduation'}",
        ),
    )

    store.get_or_create_active.assert_awaited_once()
    assert (
        store.get_or_create_active.call_args.kwargs["channel_key"]
        == "whatsapp:+1777"
    )
    store.append_turns.assert_awaited_once()
    cid, turns = store.append_turns.call_args.args
    assert cid == "conv_carina_stored"
    assert turns[-1]["content"] == "Morning briefing — Thu."
    # Full reasoning (the calendar event id) rode along into the thread.
    assert "193ptomq" in turns[0]["content"]


@pytest.mark.asyncio
async def test_bridge_drops_unregistered_recipient(monkeypatch, mock_config):
    """An unresolvable recipient is dropped, not crashed — and never
    touches the store."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from boxbot.core import agent as agent_mod

    agent = object.__new__(agent_mod.BoxBotAgent)
    agent._index_lock = asyncio.Lock()
    agent._conversations = {}
    agent._conversation_by_key = {}
    store = MagicMock()
    store.get_or_create_active = AsyncMock()
    store.append_turns = AsyncMock()
    agent._conversation_store = store

    fake_auth = MagicMock()
    fake_auth.list_users = AsyncMock(return_value=[_FakeUser("Jacob", "+1555")])
    monkeypatch.setattr(
        "boxbot.communication.auth.get_auth_manager", lambda: fake_auth,
    )

    # "Stranger" isn't a registered user.
    await agent._bridge_trigger_delivery(
        "Stranger", _ctx_turns(recipient="Stranger"),
    )

    store.get_or_create_active.assert_not_awaited()
    store.append_turns.assert_not_awaited()
