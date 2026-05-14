"""Tests for [Recent Conversations] injection filtering (PR 1.1).

Trigger conversations are the agent talking to itself. Injecting their
receipts back into the next run is an earworm vector — no human in the
loop to dampen a wrong assertion. So ambient injection EXCLUDES
trigger conversations. The receipt is still in the conversations table
and still searchable via search_memory (deliberate lookup) — searchable,
not injected.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from boxbot.memory.search import SearchCandidate


def _mk_conversation(
    cid: str, channel: str, summary: str = "discussed something"
) -> SearchCandidate:
    return SearchCandidate(
        id=cid,
        source="conversation",
        type="conversation",
        person=None,
        content=summary,
        summary=summary,
        vector_score=0.9,
        bm25_score=0.5,
        combined_score=0.8,
        metadata={
            "channel": channel,
            "started_at": "2026-05-14T07:00:00",
            "participants": ["Jacob"],
        },
    )


class _FakeStore:
    async def update_memory_relevance(self, _id):
        pass


@pytest.mark.asyncio
async def test_trigger_conversations_excluded_from_injection():
    """A trigger receipt that ranks high on hybrid search must NOT
    appear in the [Recent Conversations] block."""
    from boxbot.memory.retrieval import inject_memories

    candidates = [
        _mk_conversation(
            "conv_trigger1", "trigger",
            "Delivered morning briefing for 5/14 -> Jacob, Carina",
        ),
        _mk_conversation(
            "conv_human1", "whatsapp",
            "discussed date-night reservation options",
        ),
    ]

    async def fake_search(*args, **kwargs):
        return candidates

    with patch("boxbot.memory.retrieval.hybrid_search", new=fake_search):
        block, ids = await inject_memories(
            _FakeStore(), person="Jacob", utterance="what did we discuss",
        )

    assert "conv_human1" in ids
    assert "conv_trigger1" not in ids
    assert "Delivered morning briefing" not in block
    assert "date-night reservation" in block


@pytest.mark.asyncio
async def test_voice_and_whatsapp_conversations_still_injected():
    """Only `trigger` is excluded — real human channels stay."""
    from boxbot.memory.retrieval import inject_memories

    candidates = [
        _mk_conversation("conv_v", "voice", "talked about the weekend"),
        _mk_conversation("conv_w", "whatsapp", "planned the grocery run"),
    ]

    async def fake_search(*args, **kwargs):
        return candidates

    with patch("boxbot.memory.retrieval.hybrid_search", new=fake_search):
        block, ids = await inject_memories(
            _FakeStore(), person="Jacob", utterance="catch me up",
        )

    assert "conv_v" in ids
    assert "conv_w" in ids


@pytest.mark.asyncio
async def test_all_trigger_pool_yields_empty_recent_block():
    """If every recent conversation is a trigger, the [Recent
    Conversations] block is simply absent — not a crash, not a
    trigger receipt leaking through."""
    from boxbot.memory.retrieval import inject_memories

    candidates = [
        _mk_conversation("conv_t1", "trigger", "Delivered morning briefing"),
        _mk_conversation("conv_t2", "trigger", "Delivered evening review"),
        _mk_conversation("conv_t3", "trigger", "Delivered midday check"),
    ]

    async def fake_search(*args, **kwargs):
        return candidates

    with patch("boxbot.memory.retrieval.hybrid_search", new=fake_search):
        block, ids = await inject_memories(
            _FakeStore(), person="Jacob", utterance="anything?",
        )

    assert ids == []
    assert "[Recent Conversations]" not in block


@pytest.mark.asyncio
async def test_over_fetch_absorbs_trigger_filtering():
    """conversation_limit is over-fetched (4x) so trigger rows getting
    dropped post-search doesn't starve the human-conversation slots.
    Verify inject_memories asks hybrid_search for more than
    max_conversations."""
    from boxbot.memory.retrieval import inject_memories, MAX_CONVERSATIONS

    seen_kwargs = {}

    async def fake_search(*args, **kwargs):
        seen_kwargs.update(kwargs)
        return []

    with patch("boxbot.memory.retrieval.hybrid_search", new=fake_search):
        await inject_memories(
            _FakeStore(), person="Jacob", utterance="hi",
        )

    assert seen_kwargs["conversation_limit"] > MAX_CONVERSATIONS


def test_static_prompt_frames_injected_context_as_non_authoritative():
    """The static system prompt must tell the agent that injected
    [Recent Conversations] are receipts, not ground truth, and that
    its own past output is never authoritative. Without this, the
    agent reads a topic-pointer summary and treats it as a fact."""
    from boxbot.core.agent import _prompt_capabilities

    text = _prompt_capabilities().lower()
    assert "receipt" in text
    # Steers to live sources for current state
    assert "live source" in text
    # Explicitly: own past output is not evidence
    assert "past words are never authoritative" in text or (
        "past output" in text and "authoritative" in text
    )
