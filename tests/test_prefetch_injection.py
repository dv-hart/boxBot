"""Regression tests for prefetch-bundle injection.

The bundle renders ONCE (agent._bundle_to_context) and rides inside the
user turn (Conversation.handle_input) so it persists in the thread —
the opus-review CRITICALs on the stage-A PR showed the old per-turn
system-prompt render vanished on the next turn (making cross-turn dedup
suppress content the model no longer had) and was silently dropped when
a barge-in hit the SPEAKING/THINKING queue paths.

Also locks the 8b9702c fixes: a bundle without memory-lane output must
not cost the turn its legacy recall, and ``injected_memories_block``
keeps the ``[Active Memories]`` header extraction's invalidation keys
on.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from boxbot.core.agent import BoxBotAgent
from boxbot.prefetch.bundle import PrefetchBundle

LEGACY_BLOCK = (
    "[Active Memories]\n#deadbeef (person/Jacob): Jacob is vegetarian.\n"
)


class _Conv:
    def __init__(self):
        self.conversation_id = "c1"
        self.accessed_memory_ids: list[str] = []
        self.injected_memories_block = ""


@pytest.fixture
def agent(mock_config):
    a = BoxBotAgent.__new__(BoxBotAgent)
    a._read_system_memory = AsyncMock(return_value="")
    a._inject_memories = AsyncMock(
        return_value=(LEGACY_BLOCK, ["deadbeef-full-id"])
    )
    a._latest_speaker_identities = {}
    a._prefetch_injected = {}
    return a


async def _dynamic(agent, ctx) -> str:
    return await BoxBotAgent._prompt_dynamic_context(
        agent, "Jacob", "signal", ctx, "what's on the garage camera", _Conv(),
    )


class TestBundleToContext:
    def test_renders_to_thread_borne_text_and_tracks(self, agent):
        conv = _Conv()
        bundle = PrefetchBundle(skill_bodies={"home-control": "body"})
        ctx = agent._bundle_to_context(conv, bundle)
        assert "home-control" in ctx["prefetch_text"]
        assert ctx["prefetch_memories"] is False
        # Tracking happens here — the moment the claim becomes true.
        assert "home-control" in agent._prefetch_injected["c1"]

    def test_memory_bookkeeping_and_active_memories_header(self, agent):
        conv = _Conv()
        bundle = PrefetchBundle(
            memories=[("abc12345deadbeef", "Jacob is vegetarian")],
        )
        ctx = agent._bundle_to_context(conv, bundle)
        assert ctx["prefetch_memories"] is True
        assert "Relevant memories" in ctx["prefetch_text"]
        # Extraction invalidation keys on this exact header.
        assert conv.injected_memories_block.startswith("[Active Memories]\n")
        assert "#abc12345: Jacob is vegetarian" in conv.injected_memories_block
        assert "abc12345deadbeef" in conv.accessed_memory_ids

    def test_empty_render_returns_nothing_and_tracks_nothing(self, agent):
        ctx = agent._bundle_to_context(_Conv(), PrefetchBundle())
        assert ctx == {}
        assert agent._prefetch_injected == {}


class TestDynamicContextLegacyRecall:
    @pytest.mark.asyncio
    async def test_no_bundle_keeps_legacy_injection(self, agent):
        out = await _dynamic(agent, {})
        assert agent._inject_memories.await_count == 1
        assert "[Active Memories]" in out

    @pytest.mark.asyncio
    async def test_bundle_without_memories_still_gets_legacy_recall(
        self, agent
    ):
        """A skill-only bundle must not suppress the memory block."""
        out = await _dynamic(agent, {"prefetch_memories": False})
        assert agent._inject_memories.await_count == 1
        assert "[Active Memories]" in out

    @pytest.mark.asyncio
    async def test_bundle_with_memories_supersedes_legacy(self, agent):
        await _dynamic(agent, {"prefetch_memories": True})
        assert agent._inject_memories.await_count == 0  # dedup win preserved

    @pytest.mark.asyncio
    async def test_bundle_never_renders_into_the_system_prompt(self, agent):
        """The regression this file exists for: system-prompt-borne
        bundles vanish on the next turn while dedup remembers them."""
        bundle = PrefetchBundle(skill_bodies={"home-control": "BODY-MARKER"})
        out = await _dynamic(agent, {"prefetch_bundle": bundle})
        assert "BODY-MARKER" not in out


class TestThreadBorneInjection:
    """The bundle rides as turn METADATA (`prefetch_text` key), never
    inside `content` — thread consumers that read content as human
    speech (memory-search query off thread[-1], extraction transcript)
    must see pure utterance; the wire payload merges via
    _materialize_history / _materialize_turn_text."""

    @pytest.mark.asyncio
    async def test_queued_input_carries_its_bundle_as_metadata(self):
        """Barge-in regression: a bundle arriving while the conversation
        is SPEAKING/THINKING must ride the queued message, not vanish
        with the discarded context dict — and must NOT pollute content."""
        from boxbot.core.conversation import Conversation, ConversationState

        conv = Conversation(
            conversation_id="c1", channel="voice", channel_key="voice:room",
            generate_fn=AsyncMock(),
        )
        conv._state = ConversationState.SPEAKING
        await conv.handle_input(
            "and the garage too",
            speaker_name="Jacob",
            context={"prefetch_text": "## Prefetched context\nlocks doc"},
        )
        (queued,) = conv.drain_pending_inputs()
        assert queued["content"] == "[Jacob]: and the garage too"
        assert queued["prefetch_text"].startswith("## Prefetched context")

    def test_materialize_merges_and_strips_for_the_wire(self):
        turn = {
            "role": "user",
            "content": "[Jacob]: Lock the door",
            "prefetch_text": "## Prefetched context\nlocks doc",
        }
        assert BoxBotAgent._materialize_turn_text(turn) == (
            "## Prefetched context\nlocks doc\n\n[Jacob]: Lock the door"
        )
        (wire,) = BoxBotAgent._materialize_history([turn])
        assert "prefetch_text" not in wire
        assert wire["content"].startswith("## Prefetched context")
        assert wire["role"] == "user"
        # Plain turns pass through untouched (same object, no copy).
        plain = {"role": "assistant", "content": [{"type": "text"}]}
        assert BoxBotAgent._materialize_history([plain]) == [plain]

    def test_clean_content_feeds_search_and_transcripts(self):
        """N1/N2 regression: content alone is the utterance."""
        turn = {
            "role": "user",
            "content": "[Jacob]: Lock the door",
            "prefetch_text": "## Prefetched context\n1800 tokens of docs",
        }
        assert str(turn.get("content")) == "[Jacob]: Lock the door"


class TestShadowModeAndQueuedFlags:
    def test_materialize_strips_all_metadata_keys(self):
        turn = {
            "role": "user", "content": "[Jacob]: hi",
            "prefetch_text": "docs", "prefetch_memories": True,
        }
        (wire,) = BoxBotAgent._materialize_history([turn])
        assert "prefetch_text" not in wire
        assert "prefetch_memories" not in wire
        assert wire["content"] == "docs\n\n[Jacob]: hi"

    @pytest.mark.asyncio
    async def test_queued_turn_carries_the_recall_flag(self):
        """N3 regression: the legacy-recall suppression flag must ride
        the queued message — the drain site adopts it, so a fresh
        generation never inherits the PREVIOUS turn's flag."""
        from boxbot.core.conversation import Conversation, ConversationState

        conv = Conversation(
            conversation_id="c1", channel="voice", channel_key="voice:room",
            generate_fn=AsyncMock(),
        )
        conv._state = ConversationState.SPEAKING
        await conv.handle_input(
            "what about the garage",
            speaker_name="Jacob",
            context={"prefetch_text": "docs", "prefetch_memories": True},
        )
        (queued,) = conv._pending_inputs
        assert queued["prefetch_memories"] is True


def test_compaction_counts_prefetch_metadata_tokens():
    """N5 regression: the compaction budget must count bundle metadata
    — it ships on every wire call even though it lives outside content."""
    from boxbot.core.compaction import estimate_tokens

    bare = [{"role": "user", "content": "[Jacob]: hi"}]
    loaded = [{
        "role": "user", "content": "[Jacob]: hi",
        "prefetch_text": "x" * 4000,
    }]
    assert estimate_tokens(loaded) >= estimate_tokens(bare) + 900
