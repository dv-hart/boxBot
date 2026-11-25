"""Who the agent is told it is talking to.

Two defects, one session:

1. The "People in this session" legend printed thresholds from the
   retired single-centroid matcher (0.85 / 0.70 / 0.60) and a ``low``
   tier cloud matching cannot produce, next to scores such as
   ``voice: high (0.69)``.
2. "Speaking with: <name>" came from the perception layer's
   most-recently-seen person, which is never pruned — an unrecognised
   speaker in a new session was presented as "Speaking with: Jacob".
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

import boxbot.core.agent as agent_module
from boxbot.core.agent import BoxBotAgent, _render_identity_section
from boxbot.core.conversation import Conversation, GenerationResult
from boxbot.core.events import TranscriptReady, VoiceSessionEnded

UNKNOWN = {
    "person_id": None, "person_name": None,
    "voice_tier": "unknown", "voice_score": 0.21, "source": "unknown",
}
JACOB_HIGH = {
    "person_id": "p1", "person_name": "Jacob",
    "voice_tier": "high", "voice_score": 0.69, "source": "voice_reid_match",
}
JACOB_MEDIUM = {**JACOB_HIGH, "voice_tier": "medium", "voice_score": 0.48}
ERIK_PINNED = {
    "person_id": "p2", "person_name": "Erik",
    "voice_tier": "unknown", "voice_score": 0.0, "source": "agent_identify",
}


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------


def test_legend_uses_the_live_cloud_thresholds(mock_config):
    out = _render_identity_section({"Jacob": JACOB_HIGH})

    confirmed = mock_config.perception.voice_confirmed_threshold
    maybe = mock_config.perception.voice_maybe_threshold
    assert f"high (voice ≥{confirmed:.2f})" in out
    assert f"medium ({maybe:.2f}–{confirmed:.2f})" in out
    assert f"unknown (<{maybe:.2f})" in out
    # The score shown must sit inside the range the legend gives its tier.
    assert "voice: high (0.69)" in out
    assert 0.69 >= confirmed


def test_legend_follows_a_config_change(mock_config):
    mock_config.perception.voice_confirmed_threshold = 0.61
    mock_config.perception.voice_maybe_threshold = 0.5

    out = _render_identity_section({"Jacob": JACOB_HIGH})

    assert "high (voice ≥0.61)" in out
    assert "medium (0.50–0.61)" in out
    assert "unknown (<0.50)" in out


def test_legend_has_no_retired_thresholds_or_dead_tier(mock_config):
    out = _render_identity_section({"Speaker A": UNKNOWN})

    for stale in ("0.85", "0.70", "0.60", "- low"):
        assert stale not in out
    assert "do NOT guess a name" in out


def test_headlines_per_tier(mock_config):
    out = _render_identity_section({
        "Jacob": JACOB_HIGH,
        "Speaker A": JACOB_MEDIUM,
        "Speaker B": UNKNOWN,
        "Erik": ERIK_PINNED,
    })
    assert "- Jacob  → likely Jacob" in out
    assert "- Speaker A  → possibly Jacob" in out
    assert "- Speaker B  → not recognized" in out
    assert "- Erik  → confirmed as Erik" in out


def test_empty_block_renders_nothing():
    assert _render_identity_section({}) == ""
    assert _render_identity_section(None) == ""


# ---------------------------------------------------------------------------
# Speaker of one utterance
# ---------------------------------------------------------------------------


def _bare_agent() -> BoxBotAgent:
    a = BoxBotAgent.__new__(BoxBotAgent)
    a._present_people = {}
    a._speaker_identities = {}
    a._latest_speaker_identities = {}
    return a


def test_unrecognised_speaker_is_not_named_after_whoever_was_seen_last():
    a = _bare_agent()
    a._present_people = {"Jacob": datetime.now()}

    assert a._voice_speaker_name({"Speaker A": UNKNOWN}) is None


@pytest.mark.parametrize("block,expected", [
    ({"Jacob": JACOB_HIGH}, "Jacob"),
    ({"Erik": ERIK_PINNED}, "Erik"),
    # Medium is a "possibly" — the agent verifies; the header stays blank.
    ({"Speaker A": JACOB_MEDIUM}, None),
    # First speaker heard is the one the turn is with.
    ({"Speaker A": UNKNOWN, "Jacob": JACOB_HIGH}, None),
    ({"Jacob": JACOB_HIGH, "Speaker A": UNKNOWN}, "Jacob"),
])
def test_speaker_name_comes_from_the_utterance_identity_block(block, expected):
    a = _bare_agent()
    a._present_people = {"Carina": datetime.now()}

    assert a._voice_speaker_name(block) == expected


def test_no_identity_block_falls_back_to_recent_presence_only():
    a = _bare_agent()
    a._present_people = {"Jacob": datetime.now() - timedelta(seconds=30)}
    assert a._voice_speaker_name({}) == "Jacob"

    a._present_people = {"Jacob": datetime.now() - timedelta(minutes=13)}
    assert a._voice_speaker_name({}) is None
    assert a._get_most_recent_person() is None


def test_most_recent_person_picks_the_freshest_sighting():
    a = _bare_agent()
    now = datetime.now()
    a._present_people = {
        "Jacob": now - timedelta(minutes=20),
        "Carina": now - timedelta(seconds=5),
    }
    assert a._get_most_recent_person() == "Carina"


# ---------------------------------------------------------------------------
# Conversation carries the attribution of its latest human input
# ---------------------------------------------------------------------------


def _conv(generate_fn=None, **kw) -> Conversation:
    kw.setdefault("conversation_id", "c1")
    kw.setdefault("channel", "voice")
    kw.setdefault("channel_key", "voice:room")
    return Conversation(
        generate_fn=generate_fn or AsyncMock(return_value=GenerationResult()),
        silence_timeout=0, **kw,
    )


@pytest.mark.asyncio
async def test_conversation_tracks_the_latest_human_speaker():
    conv = _conv()
    assert conv.speaker_resolved is False

    await conv.handle_input("[Jacob]: hi", speaker_name="Jacob", human=True)
    await conv._generation_task
    assert (conv.speaker_resolved, conv.current_speaker) == (True, "Jacob")

    # An unrecognised voice follows: the name must not stick.
    await conv.handle_input("[Speaker A]: hello?", human=True)
    await conv._generation_task
    assert (conv.speaker_resolved, conv.current_speaker) == (True, None)

    # Machine-written input (presence update) changes nothing.
    await conv.handle_input("[Presence update: Jacob arrived]", source="user")
    await conv._generation_task
    assert conv.current_speaker is None
    await conv.end(reason="test")


@pytest.mark.asyncio
async def test_speaker_for_conversation_prefers_attribution_over_presence():
    a = _bare_agent()
    a._present_people = {"Jacob": datetime.now()}

    trigger = _conv(channel="trigger", channel_key="trigger:t:1")
    # No human input yet (trigger run): presence guess stands.
    assert a._speaker_for_conversation(trigger) == "Jacob"

    voice = _conv()
    await voice.handle_input("[Speaker A]: hello?", human=True)
    await voice._generation_task
    assert a._speaker_for_conversation(voice) is None

    text = _conv(channel="signal", channel_key="signal:+1")
    await text.handle_input("hi", speaker_name="Carina", human=True)
    await text._generation_task
    # A text from Carina is with Carina, whoever is at the box.
    assert a._speaker_for_conversation(text) == "Carina"
    for c in (voice, text):
        await c.end(reason="test")


# ---------------------------------------------------------------------------
# End to end through the transcript handler
# ---------------------------------------------------------------------------


@pytest.fixture
def routed_agent(monkeypatch, mock_config):
    mem = MagicMock()
    mem.read_system_memory = MagicMock(return_value="")
    mem.create_conversation_stub = AsyncMock()
    a = BoxBotAgent(memory_store=mem)
    a._running = True
    a._inject_memories = AsyncMock(return_value=("", []))
    a._ingest_pending_relay = AsyncMock()
    a._prefetch_context_for_text = AsyncMock(return_value={})
    monkeypatch.setattr(agent_module, "get_presence_snapshot", lambda: None)
    monkeypatch.setattr(
        agent_module, "get_status_line", AsyncMock(return_value=""),
    )

    seen: list[dict] = []

    async def _generate(conv):
        person = a._speaker_for_conversation(conv)
        seen.append({
            "person": person,
            "context": await a._prompt_dynamic_context(
                person_name=person, channel=conv.channel,
                context=conv.current_context,
                initial_message=str(conv.thread[-1]["content"]), conv=conv,
            ),
            "utterance": conv.thread[-1]["content"],
        })
        return GenerationResult()

    a._generate_for_conversation = _generate
    a._seen = seen
    return a


async def _hear(agent, session: str, transcript: str, identities: dict):
    await agent._on_transcript_ready(TranscriptReady(
        conversation_id=session, transcript=transcript,
        raw_text=transcript.split("]: ", 1)[-1],
        speaker_identities=identities,
    ))
    conv = agent._get_voice_room_conversation()
    await conv._generation_task
    return conv


@pytest.mark.asyncio
async def test_unknown_speaker_in_a_new_session_is_not_presented_as_jacob(
    routed_agent,
):
    """The session-1 speaker (Jacob) must not label session 2's stranger."""
    a = routed_agent
    a._present_people = {"Jacob": datetime.now()}

    await _hear(a, "voice_1", "[Jacob]: play megamind", {"Jacob": JACOB_HIGH})
    first = a._seen[-1]
    assert first["person"] == "Jacob"
    assert "- Speaking with: Jacob" in first["context"]

    await a._on_voice_session_ended(VoiceSessionEnded(conversation_id="voice_1"))

    conv = await _hear(
        a, "voice_2", "[Speaker A]: can you search for Titanic books?",
        {"Speaker A": UNKNOWN},
    )
    second = a._seen[-1]
    assert second["person"] is None
    assert "Speaking with" not in second["context"]
    assert "likely Jacob" not in second["context"]
    assert "- Speaker A  → not recognized" in second["context"]
    # The thread line carries the speaker's own label only.
    assert second["utterance"] == (
        "[Speaker A]: can you search for Titanic books?"
    )
    assert conv.participants == set()
    await conv.end(reason="test")


@pytest.mark.asyncio
async def test_identity_block_does_not_leak_into_the_next_session(
    routed_agent,
):
    """An utterance that resolves no identities used to keep showing the
    previous session's block under "People in this session"."""
    a = routed_agent

    conv = await _hear(a, "voice_1", "[Jacob]: hi", {"Jacob": JACOB_HIGH})
    assert "likely Jacob" in a._seen[-1]["context"]
    await conv.end(reason="test")

    # No VoiceSessionEnded delivered — the session id change alone
    # must clear it.
    conv = await _hear(a, "voice_2", "hello?", {})
    assert a._latest_speaker_identities == {}
    assert "People in this session" not in a._seen[-1]["context"]
    await conv.end(reason="test")


@pytest.mark.asyncio
async def test_same_session_followup_from_a_stranger_drops_the_name(
    routed_agent,
):
    a = routed_agent

    await _hear(a, "voice_1", "[Jacob]: hi", {"Jacob": JACOB_HIGH})
    conv = await _hear(
        a, "voice_1", "[Speaker A]: what about me", {"Speaker A": UNKNOWN},
    )

    assert a._seen[-1]["person"] is None
    assert "Speaking with" not in a._seen[-1]["context"]
    await conv.end(reason="test")
