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
    # Mirror the EXACT wire format produced by
    # Conversation._format_user_message for source="trigger":
    #   {"role": "user", "content": f"[trigger] {text}"}
    # where ``text`` is the "[Trigger fired: ...]\nInstructions: ..."
    # string built in agent._on_trigger_fired. Earlier this fixture
    # used the un-prefixed form, which is why the _has_human_reply
    # bug (matching "[Trigger fired:" instead of "[trigger]") slipped
    # through — the test didn't reproduce the real thread content.
    return {
        "role": "user",
        "content": f"[trigger] [Trigger fired: {description}]\nInstructions: do stuff",
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
    def test_recognises_real_format_user_message_output(self) -> None:
        """Regression guard: exercise the ACTUAL
        Conversation._format_user_message so the fixture can never
        drift from the real wire format again. The original
        _has_human_reply bug was exactly this drift — it matched
        "[Trigger fired:" while the thread stores "[trigger] ...".
        """
        from boxbot.core.conversation import Conversation

        # _format_user_message is a pure method; we only need an
        # instance, not a fully wired conversation.
        conv = object.__new__(Conversation)
        trigger_msg = conv._format_user_message(
            "[Trigger fired: Morning briefing]\nInstructions: do stuff",
            speaker_name="Jacob",
            source="trigger",
        )
        # A thread of just the synthetic trigger message + tool traffic
        # must NOT count as a human reply.
        msgs = [
            trigger_msg,
            _assistant_message_tool_use("Jacob"),
            _tool_result_user_msg(),
        ]
        assert _has_human_reply(msgs) is False

        # And a real human voice line through the same formatter MUST.
        human_msg = conv._format_user_message(
            "what about Carina?", speaker_name="Jacob", source="user",
        )
        assert _has_human_reply([trigger_msg, human_msg]) is True

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


def _tool_result_with_sdk_actions(actions: list) -> dict:
    """A user-role tool_result whose JSON body carries sdk_actions —
    mirrors what execute_script returns."""
    import json as _json
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "x",
                "content": _json.dumps(
                    {"status": "success", "sdk_actions": actions}
                ),
            }
        ],
    }


class TestSummarizeTriggerThread:
    def test_receipt_has_description_recipients_and_date(self) -> None:
        msgs = [
            _trigger_initial_msg("Morning briefing"),
            _assistant_message_tool_use("Jacob"),
            _tool_result_user_msg(),
            _assistant_message_tool_use("Carina"),
            _tool_result_user_msg(),
        ]
        summary = _summarize_trigger_thread(
            msgs, started_at="2026-05-14T14:00:00+00:00",
        )
        assert "Delivered" in summary
        assert "Morning briefing" in summary
        assert "Jacob" in summary
        assert "Carina" in summary
        assert "5/14" in summary

    def test_receipt_is_a_receipt_not_content(self) -> None:
        """The receipt must NOT carry weather/calendar/todo content —
        that's the earworm vector. Only: what ran, when, where it went."""
        msgs = [
            _trigger_initial_msg("Morning briefing"),
            _assistant_message_tool_use(
                "Jacob",
                "Weather: rain, high 65. Calendar down pending reauth.",
            ),
            _tool_result_user_msg(),
        ]
        summary = _summarize_trigger_thread(
            msgs, started_at="2026-05-14T14:00:00+00:00",
        )
        # The delivered message BODY must not leak into the receipt.
        assert "rain" not in summary.lower()
        assert "reauth" not in summary.lower()
        assert "calendar" not in summary.lower()

    def test_no_outbound_messages_is_ran_nothing_delivered(self) -> None:
        msgs = [_trigger_initial_msg("Midday check")]
        summary = _summarize_trigger_thread(
            msgs, started_at="2026-05-14T19:00:00+00:00",
        )
        assert "Ran" in summary
        assert "Midday check" in summary
        assert "nothing delivered" in summary

    def test_receipt_carries_workspace_pointer(self) -> None:
        """A trigger that wrote a work-product to the workspace gets a
        pointer in the receipt — that's how 'how did the job review go?'
        stays answerable via deliberate lookup."""
        msgs = [
            _trigger_initial_msg("Job listing review"),
            _assistant_message_tool_use("Jacob"),
            _tool_result_with_sdk_actions([
                {
                    "action": "workspace.write",
                    "status": "ok",
                    "path": "notes/job-review/2026-05-14.md",
                }
            ]),
        ]
        summary = _summarize_trigger_thread(
            msgs, started_at="2026-05-14T14:00:00+00:00",
        )
        assert "notes/job-review/2026-05-14.md" in summary

    def test_dedupes_repeated_recipient(self) -> None:
        msgs = [
            _trigger_initial_msg("Evening review"),
            _assistant_message_tool_use("Jacob", "first line"),
            _tool_result_user_msg(),
            _assistant_message_tool_use("Jacob", "second line"),
            _tool_result_user_msg(),
        ]
        summary = _summarize_trigger_thread(
            msgs, started_at="2026-05-14T03:00:00+00:00",
        )
        assert summary.count("Jacob") == 1

    def test_handles_missing_started_at(self) -> None:
        """started_at is optional — no date clause, no crash."""
        msgs = [
            _trigger_initial_msg("Morning briefing"),
            _assistant_message_tool_use("Jacob"),
            _tool_result_user_msg(),
        ]
        summary = _summarize_trigger_thread(msgs)
        assert "Morning briefing" in summary
        assert "Jacob" in summary


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
