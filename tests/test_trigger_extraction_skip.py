"""Tests for trigger-conversation extraction skip in :mod:`boxbot.core.agent`.

Routine trigger-fired conversations (morning brief, midday check,
evening review, etc.) used to extract a memory like "Morning briefing
sent to Jacob & Carina on 5/12; calendar feed down" every firing. Those
operational memories crowded the injection pool with near-duplicate
log entries that re-asserted stale state at every later trigger.

The fix: when a trigger conversation has no human reply, write a
deterministic conversation-log summary and skip the extraction batch
entirely.
"""
from __future__ import annotations

import pytest

from boxbot.core.agent import (
    _has_human_reply,
    _summarize_trigger_thread,
)


def _trigger_initial_msg(description: str = "Morning briefing") -> dict:
    return {
        "role": "user",
        "content": f"[Trigger fired: {description}]\nInstructions: do stuff",
    }


def _tool_result_user_msg() -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "x", "content": "ok"},
        ],
    }


def _assistant_message_tool_use(to: str, content: str = "hi") -> dict:
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "message",
                "input": {"to": to, "channel": "text", "content": content},
            }
        ],
    }


class TestHasHumanReply:
    def test_pure_trigger_thread_has_no_human_reply(self) -> None:
        msgs = [
            _trigger_initial_msg(),
            _assistant_message_tool_use("Jacob"),
            _tool_result_user_msg(),
        ]
        assert _has_human_reply(msgs) is False

    def test_user_text_reply_counts_as_human(self) -> None:
        msgs = [
            _trigger_initial_msg(),
            _assistant_message_tool_use("Jacob"),
            _tool_result_user_msg(),
            {"role": "user", "content": "actually wait, what about Carina?"},
        ]
        assert _has_human_reply(msgs) is True

    def test_inbound_message_with_attachment_counts_as_human(self) -> None:
        """WhatsApp image-message arrives as a string body too (the
        inbound handler prefixes with [image attached at ...])."""
        msgs = [
            _trigger_initial_msg(),
            {
                "role": "user",
                "content": "[image attached at /tmp/x.jpg] thanks",
            },
        ]
        assert _has_human_reply(msgs) is True

    def test_only_tool_results_after_trigger_init_is_not_human(self) -> None:
        msgs = [
            _trigger_initial_msg(),
            _assistant_message_tool_use("Jacob"),
            _tool_result_user_msg(),
            _assistant_message_tool_use("Carina"),
            _tool_result_user_msg(),
        ]
        assert _has_human_reply(msgs) is False

    def test_empty_thread_is_not_human(self) -> None:
        assert _has_human_reply([]) is False


class TestSummarizeTriggerThread:
    def test_extracts_description_and_recipients(self) -> None:
        msgs = [
            _trigger_initial_msg("Morning briefing"),
            _assistant_message_tool_use("Jacob"),
            _tool_result_user_msg(),
            _assistant_message_tool_use("Carina"),
            _tool_result_user_msg(),
        ]
        summary = _summarize_trigger_thread(msgs)
        assert "Morning briefing" in summary
        assert "Jacob" in summary
        assert "Carina" in summary

    def test_no_outbound_messages(self) -> None:
        msgs = [_trigger_initial_msg("Midday check")]
        summary = _summarize_trigger_thread(msgs)
        assert "Midday check" in summary
        assert "No outbound messages" in summary

    def test_dedupes_repeated_recipient(self) -> None:
        msgs = [
            _trigger_initial_msg("Evening review"),
            _assistant_message_tool_use("Jacob", "first line"),
            _tool_result_user_msg(),
            _assistant_message_tool_use("Jacob", "second line"),
            _tool_result_user_msg(),
        ]
        summary = _summarize_trigger_thread(msgs)
        # Recipient should appear exactly once
        assert summary.count("Jacob") == 1

    def test_falls_back_to_generic_label_when_no_prefix(self) -> None:
        """If somehow we end up summarising a non-trigger thread (e.g.
        a future caller misuses this), don't crash — just label it
        generically."""
        msgs = [
            {"role": "user", "content": "Hey BB, status?"},
            _assistant_message_tool_use("Jacob"),
        ]
        summary = _summarize_trigger_thread(msgs)
        assert "Trigger fired" in summary


class TestPostConversationTriggerSkip:
    """Integration-ish: _post_conversation should write a summary and
    NOT call create_pending_extraction / batch poller for routine
    trigger conversations."""

    @pytest.mark.asyncio
    async def test_trigger_thread_skips_extraction_writes_summary(
        self, mock_config,
    ) -> None:
        from unittest.mock import AsyncMock, MagicMock

        # Build the minimal agent surface we need to call _post_conversation.
        # We don't construct a full BoxBotAgent — too much wiring. Instead
        # we instantiate one with fakes for the two collaborators it
        # actually touches in this code path: _memory_store and
        # _batch_poller. Use object.__new__ to bypass __init__.
        from boxbot.core import agent as agent_mod

        agent = object.__new__(agent_mod.BoxBotAgent)
        agent._memory_store = MagicMock()
        agent._memory_store.create_pending_extraction = AsyncMock()
        agent._memory_store.get_pending_extraction = AsyncMock(return_value=None)
        agent._memory_store.get_conversation = AsyncMock(return_value=None)
        agent._memory_store.create_conversation = AsyncMock()
        agent._memory_store.update_conversation = AsyncMock()
        agent._batch_poller = MagicMock()
        agent._batch_poller.submit = AsyncMock()

        messages = [
            _trigger_initial_msg("Morning briefing"),
            _assistant_message_tool_use("Jacob"),
            _tool_result_user_msg(),
        ]

        await agent._post_conversation(
            conversation_id="conv_test1",
            channel="trigger",
            person_name="Jacob",
            messages=messages,
            accessed_memory_ids=[],
            started_at="2026-05-13T07:00:00",
        )

        # No extraction was queued
        agent._memory_store.create_pending_extraction.assert_not_awaited()
        agent._batch_poller.submit.assert_not_awaited()
        # A conversation summary was written
        agent._memory_store.create_conversation.assert_awaited_once()
        kwargs = agent._memory_store.create_conversation.call_args.kwargs
        assert kwargs["conversation_id"] == "conv_test1"
        assert kwargs["channel"] == "trigger"
        assert "Morning briefing" in kwargs["summary"]
        assert "Jacob" in kwargs["summary"]

    @pytest.mark.asyncio
    async def test_trigger_thread_with_human_reply_goes_through_extraction(
        self, mock_config,
    ) -> None:
        """If a human chimes in mid-trigger, we want the full extraction
        path so any genuinely novel content gets captured."""
        from unittest.mock import AsyncMock, MagicMock

        from boxbot.core import agent as agent_mod

        agent = object.__new__(agent_mod.BoxBotAgent)
        agent._memory_store = MagicMock()
        agent._memory_store.create_pending_extraction = AsyncMock()
        agent._memory_store.get_pending_extraction = AsyncMock(
            return_value="row"
        )
        agent._batch_poller = MagicMock()
        agent._batch_poller.submit = AsyncMock()

        messages = [
            _trigger_initial_msg("Morning briefing"),
            _assistant_message_tool_use("Jacob"),
            _tool_result_user_msg(),
            {"role": "user", "content": "wait, also tell Carina"},
        ]

        await agent._post_conversation(
            conversation_id="conv_test2",
            channel="trigger",
            person_name="Jacob",
            messages=messages,
            accessed_memory_ids=[],
            started_at="2026-05-13T07:00:00",
        )

        agent._memory_store.create_pending_extraction.assert_awaited_once()
        agent._batch_poller.submit.assert_awaited_once()
