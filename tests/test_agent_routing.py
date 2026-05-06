"""Tests for BoxBotAgent event routing to Conversation instances.

These tests exercise the new AgentCore routing layer introduced in M2:
inbound events (TranscriptReady, WhatsAppMessage, TriggerFired,
VoiceSessionEnded) resolve to the correct Conversation in the index,
with voice keyed as ``voice:room``, whatsapp keyed per-sender, and
trigger keyed per-firing. The Anthropic agent loop is not exercised —
we replace ``_generate_for_conversation`` with a stub.

Focus areas:
- Conversation creation on first inbound event per channel key.
- Reuse of an existing conversation on subsequent events to the same key.
- Cross-channel parallelism (voice + whatsapp don't serialize).
- Voice-session-ended ends the room conversation.
- Multiple WhatsApp senders get independent conversations.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from boxbot.core.agent import BoxBotAgent
from boxbot.core.conversation import (
    Conversation,
    ConversationState,
    GenerationResult,
    SpokenSegment,
)
from boxbot.core.events import (
    AgentSpeaking,
    AgentSpeakingDone,
    ConversationEnded,
    TranscriptReady,
    TriggerFired,
    VoiceSessionEnded,
    WhatsAppMessage,
    get_event_bus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stub_generate(reply: str = "ok", delay: float = 0.0):
    """Return an async generate_fn that produces one assistant turn."""
    async def _gen(conv: Conversation) -> GenerationResult:
        if delay:
            await asyncio.sleep(delay)
        # Record the spoken segment so the conversation's interrupt
        # machinery has something to fold if we're cancelled.
        conv.record_segment(SpokenSegment(
            channel="voice", to="room", content=reply,
        ))
        return GenerationResult(
            thread_additions=[{"role": "assistant", "content": reply}],
            turn_count=1,
            summary=reply,
        )
    return _gen


@pytest.fixture
async def agent():
    """Minimal AgentCore with stubbed generate_fn and no real Anthropic client."""
    mem = MagicMock()
    mem.read_system_memory = MagicMock(return_value="")
    agent = BoxBotAgent(memory_store=mem)
    # Bypass start() because it validates ANTHROPIC_API_KEY. Install
    # the event subscriptions by hand.
    agent._client = MagicMock()
    agent._running = True
    bus = get_event_bus()
    bus.subscribe(WhatsAppMessage, agent._on_whatsapp_message)
    bus.subscribe(TriggerFired, agent._on_trigger_fired)
    bus.subscribe(TranscriptReady, agent._on_transcript_ready)
    bus.subscribe(VoiceSessionEnded, agent._on_voice_session_ended)
    bus.subscribe(ConversationEnded, agent._on_conversation_ended)
    bus.subscribe(AgentSpeaking, agent._on_agent_speaking)
    bus.subscribe(AgentSpeakingDone, agent._on_agent_speaking_done)
    # Replace the Anthropic loop with a stub.
    agent._generate_for_conversation = _make_stub_generate()
    try:
        yield agent
    finally:
        bus.unsubscribe(WhatsAppMessage, agent._on_whatsapp_message)
        bus.unsubscribe(TriggerFired, agent._on_trigger_fired)
        bus.unsubscribe(TranscriptReady, agent._on_transcript_ready)
        bus.unsubscribe(VoiceSessionEnded, agent._on_voice_session_ended)
        bus.unsubscribe(ConversationEnded, agent._on_conversation_ended)
        bus.unsubscribe(AgentSpeaking, agent._on_agent_speaking)
        bus.unsubscribe(AgentSpeakingDone, agent._on_agent_speaking_done)


async def _drain_active(agent) -> None:
    """Await all in-flight generation tasks on active conversations."""
    for conv in list(agent._conversations.values()):
        if conv._generation_task is not None:
            try:
                await conv._generation_task
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Voice routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transcript_creates_room_conversation(agent):
    await agent._on_transcript_ready(TranscriptReady(
        conversation_id="vs_001",
        transcript="[Jacob]: hello",
        speaker_identities={},
        source="voice",
    ))
    await _drain_active(agent)

    assert "voice:room" in agent._conversation_by_key
    conv_id = agent._conversation_by_key["voice:room"]
    conv = agent._conversations[conv_id]
    assert conv.channel == "voice"
    # Thread contains the user input + the stubbed assistant reply.
    assert len(conv.thread) == 2
    assert conv.thread[-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_two_transcripts_same_session_reuse_conversation(agent):
    await agent._on_transcript_ready(TranscriptReady(
        conversation_id="vs_001",
        transcript="[Jacob]: one",
        source="voice",
    ))
    await _drain_active(agent)
    first_conv_id = agent._conversation_by_key["voice:room"]

    await agent._on_transcript_ready(TranscriptReady(
        conversation_id="vs_001",  # same session id
        transcript="[Jacob]: two",
        source="voice",
    ))
    await _drain_active(agent)

    # Same conversation reused.
    assert agent._conversation_by_key["voice:room"] == first_conv_id
    conv = agent._conversations[first_conv_id]
    # Thread grows: user(one), assistant(ok), user(two), assistant(ok).
    assert len(conv.thread) == 4
    assert conv.thread[0]["content"].endswith("one")
    assert conv.thread[2]["content"].endswith("two")


@pytest.mark.asyncio
async def test_new_voice_session_ends_old_room_conversation(agent):
    # First session produces one conversation.
    await agent._on_transcript_ready(TranscriptReady(
        conversation_id="vs_001",
        transcript="[Jacob]: one",
        source="voice",
    ))
    await _drain_active(agent)
    first_conv_id = agent._conversation_by_key["voice:room"]

    # New voice session id arrives — previous room conv must be ended
    # and a fresh one started.
    await agent._on_transcript_ready(TranscriptReady(
        conversation_id="vs_002",
        transcript="[Carina]: hi again",
        source="voice",
    ))
    await _drain_active(agent)

    second_conv_id = agent._conversation_by_key["voice:room"]
    assert second_conv_id != first_conv_id
    # Old conversation has been dropped from the index.
    assert first_conv_id not in agent._conversations


@pytest.mark.asyncio
async def test_voice_session_ended_ends_room_conversation(agent):
    await agent._on_transcript_ready(TranscriptReady(
        conversation_id="vs_001",
        transcript="[Jacob]: hi",
        source="voice",
    ))
    await _drain_active(agent)
    assert "voice:room" in agent._conversation_by_key

    await agent._on_voice_session_ended(VoiceSessionEnded(
        conversation_id="vs_001",
    ))

    # Let the ConversationEnded handler run.
    await asyncio.sleep(0)
    assert "voice:room" not in agent._conversation_by_key
    assert agent._current_voice_session_id is None


# ---------------------------------------------------------------------------
# WhatsApp routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_whatsapp_message_creates_per_sender_conversation(agent):
    await agent._on_whatsapp_message(WhatsAppMessage(
        sender_name="Jacob",
        sender_phone="+15551111111",
        text="what's the weather?",
    ))
    await _drain_active(agent)
    key = "whatsapp:+15551111111"
    assert key in agent._conversation_by_key
    conv = agent._conversations[agent._conversation_by_key[key]]
    assert conv.channel == "whatsapp"
    assert "Jacob" in conv.participants


@pytest.mark.asyncio
async def test_two_whatsapp_senders_get_independent_conversations(agent):
    await agent._on_whatsapp_message(WhatsAppMessage(
        sender_name="Jacob",
        sender_phone="+15551111111",
        text="hi from jacob",
    ))
    await agent._on_whatsapp_message(WhatsAppMessage(
        sender_name="Carina",
        sender_phone="+15552222222",
        text="hi from carina",
    ))
    await _drain_active(agent)

    assert "whatsapp:+15551111111" in agent._conversation_by_key
    assert "whatsapp:+15552222222" in agent._conversation_by_key
    a_id = agent._conversation_by_key["whatsapp:+15551111111"]
    b_id = agent._conversation_by_key["whatsapp:+15552222222"]
    assert a_id != b_id
    assert "Jacob" in agent._conversations[a_id].participants
    assert "Carina" in agent._conversations[b_id].participants


@pytest.mark.asyncio
async def test_voice_and_whatsapp_conversations_run_concurrently(agent):
    """The same agent handles a voice conversation and a WhatsApp
    conversation in parallel; each runs its own generation without
    serializing behind a global lock."""
    started = {"voice": asyncio.Event(), "wa": asyncio.Event()}
    release = asyncio.Event()

    async def gen(conv):
        if conv.channel == "voice":
            started["voice"].set()
        else:
            started["wa"].set()
        # Each generation waits for the same release, so if the agent
        # were serializing across conversations we'd deadlock.
        await release.wait()
        return GenerationResult(
            thread_additions=[
                {"role": "assistant", "content": f"{conv.channel} ok"}
            ],
        )

    agent._generate_for_conversation = gen

    # Fire both inbound events close together.
    task_a = asyncio.create_task(
        agent._on_transcript_ready(TranscriptReady(
            conversation_id="vs_001",
            transcript="[Jacob]: hi",
            source="voice",
        ))
    )
    task_b = asyncio.create_task(
        agent._on_whatsapp_message(WhatsAppMessage(
            sender_name="Carina",
            sender_phone="+15553333333",
            text="hi over whatsapp",
        ))
    )
    await task_a
    await task_b

    # Both generations must be in flight simultaneously — without the
    # global lock. If one blocked the other we'd time out here.
    await asyncio.wait_for(started["voice"].wait(), timeout=1.0)
    await asyncio.wait_for(started["wa"].wait(), timeout=1.0)

    # Release both and let them complete.
    release.set()
    await _drain_active(agent)

    voice_conv = agent._conversations[agent._conversation_by_key["voice:room"]]
    wa_conv = agent._conversations[
        agent._conversation_by_key["whatsapp:+15553333333"]
    ]
    assert voice_conv.thread[-1]["content"] == "voice ok"
    assert wa_conv.thread[-1]["content"] == "whatsapp ok"


# ---------------------------------------------------------------------------
# Trigger routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trigger_fired_creates_unique_conversation(agent):
    await agent._on_trigger_fired(TriggerFired(
        trigger_id="trig_abc",
        description="morning brief",
        instructions="Tell Jacob his first meeting is at 9",
        for_person="Jacob",
    ))
    await _drain_active(agent)

    # One trigger-keyed conversation in the index.
    keys = [k for k in agent._conversation_by_key if k.startswith("trigger:")]
    assert len(keys) == 1
    conv = agent._conversations[agent._conversation_by_key[keys[0]]]
    assert conv.channel == "trigger"
    # Trigger text was tagged.
    assert conv.thread[0]["content"].startswith("[trigger]")

# ---------------------------------------------------------------------------
# Speaking-state coordination via AgentSpeaking / AgentSpeakingDone events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_speaking_event_flips_room_conv_to_speaking(agent):
    """AgentSpeaking on the bus flips the room conversation into the
    SPEAKING state so subsequent transcripts get queued instead of
    cancelling the in-flight generation."""
    # Hold the generation open so we can observe THINKING → SPEAKING.
    release = asyncio.Event()

    async def _stuck_gen(conv):
        # Initial state set by the conversation: THINKING.
        await release.wait()
        return GenerationResult(
            thread_additions=[{"role": "assistant", "content": "ok"}],
        )

    agent._generate_for_conversation = _stuck_gen

    await agent._on_transcript_ready(TranscriptReady(
        conversation_id="vs_001",
        transcript="[Jacob]: hi",
        source="voice",
    ))
    # Let the gen task start.
    await asyncio.sleep(0)
    conv = agent._conversations[agent._conversation_by_key["voice:room"]]
    assert conv.state is ConversationState.THINKING

    # Voice layer publishes AgentSpeaking when speak() begins.
    await agent._on_agent_speaking(AgentSpeaking(
        conversation_id="vs_001",
        text="hi",
    ))
    assert conv.state is ConversationState.SPEAKING

    # Stale AgentSpeaking after an interrupt must NOT flip back.
    conv.set_state(ConversationState.LISTENING)
    await agent._on_agent_speaking(AgentSpeaking(
        conversation_id="vs_001",
        text="stale",
    ))
    assert conv.state is ConversationState.LISTENING

    release.set()
    await _drain_active(agent)


@pytest.mark.asyncio
async def test_agent_speaking_done_interrupted_calls_conv_interrupt(agent):
    """AgentSpeakingDone(interrupted=True) — published by the wake-word
    handler when it stops mid-TTS playback — drives the room
    conversation through interrupt(): partial spoken segment folds in,
    queued utterances drop, state lands on LISTENING."""
    started = asyncio.Event()

    async def _gen(conv):
        if conv.thread[-1]["content"].endswith("hi"):
            conv.set_state(ConversationState.SPEAKING)
            conv.record_segment(SpokenSegment(
                channel="voice", to="room",
                content="weather looks fine today",
            ))
            started.set()
            await asyncio.sleep(10)
            return GenerationResult()
        return GenerationResult(
            thread_additions=[{"role": "assistant", "content": "new turn"}],
        )

    agent._generate_for_conversation = _gen

    await agent._on_transcript_ready(TranscriptReady(
        conversation_id="vs_001",
        transcript="[Jacob]: hi",
        source="voice",
    ))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    conv = agent._conversations[agent._conversation_by_key["voice:room"]]
    assert conv.state is ConversationState.SPEAKING

    # Background utterance queues during SPEAKING.
    await conv.handle_input("kid babble", speaker_name="Kid")

    # Wake-word handler publishes the interrupt event.
    await agent._on_agent_speaking_done(AgentSpeakingDone(
        conversation_id="vs_001",
        interrupted=True,
    ))

    assert conv.state is ConversationState.LISTENING
    assert conv.is_generating is False
    roles = [m["role"] for m in conv.thread]
    # user(hi), assistant(folded interrupted partial). Queued kid
    # babble was dropped.
    assert roles == ["user", "assistant"]
    assert "weather" in conv.thread[1]["content"]
    assert "interrupted" in conv.thread[1]["content"].lower()


@pytest.mark.asyncio
async def test_agent_speaking_done_natural_completion_is_noop(agent):
    """AgentSpeakingDone(interrupted=False) is the natural-completion
    signal — _run_generation handles state transitions itself; the
    agent's handler must NOT call interrupt() in this case."""
    interrupt_calls = 0

    await agent._on_transcript_ready(TranscriptReady(
        conversation_id="vs_001",
        transcript="[Jacob]: hi",
        source="voice",
    ))
    await _drain_active(agent)
    conv = agent._conversations[agent._conversation_by_key["voice:room"]]

    original_interrupt = conv.interrupt

    async def _spy_interrupt():
        nonlocal interrupt_calls
        interrupt_calls += 1
        return await original_interrupt()

    conv.interrupt = _spy_interrupt  # type: ignore[method-assign]

    await agent._on_agent_speaking_done(AgentSpeakingDone(
        conversation_id="vs_001",
        interrupted=False,
    ))

    assert interrupt_calls == 0


# ---------------------------------------------------------------------------
# WhatsApp persistent-mode (ConversationStore wired)
# ---------------------------------------------------------------------------


@pytest.fixture
async def persistent_agent(tmp_path):
    """Like ``agent`` but with a real ConversationStore wired in.

    Loads minimal config so ``get_config().whatsapp.*`` resolves with
    the production defaults. Uses a temp DB so tests are isolated.
    """
    from boxbot.conversations.store import ConversationStore
    from boxbot.core import config as config_module

    # Load defaults so get_config() works inside _on_whatsapp_message.
    config_module._config = config_module.BoxBotConfig()

    store = ConversationStore(db_path=tmp_path / "conv.db")
    await store.initialize()

    mem = MagicMock()
    mem.read_system_memory = MagicMock(return_value="")
    agent = BoxBotAgent(memory_store=mem, conversation_store=store)
    agent._client = MagicMock()
    agent._running = True
    bus = get_event_bus()
    bus.subscribe(WhatsAppMessage, agent._on_whatsapp_message)
    bus.subscribe(ConversationEnded, agent._on_conversation_ended)
    agent._generate_for_conversation = _make_stub_generate(reply="ack")
    try:
        yield agent, store
    finally:
        bus.unsubscribe(WhatsAppMessage, agent._on_whatsapp_message)
        bus.unsubscribe(ConversationEnded, agent._on_conversation_ended)
        await store.close()
        config_module._config = None


@pytest.mark.asyncio
async def test_persistent_whatsapp_writes_thread_to_store(persistent_agent):
    agent, store = persistent_agent
    await agent._on_whatsapp_message(WhatsAppMessage(
        sender_name="Jacob",
        sender_phone="+15551111111",
        text="hi BB",
    ))
    await _drain_active(agent)

    rec = await store.get_active(
        "whatsapp:+15551111111", max_inactive_seconds=3600,
    )
    assert rec is not None
    thread = await store.get_thread(rec.conversation_id)
    # User turn + stubbed assistant reply both persisted.
    assert len(thread) == 2
    assert thread[0]["role"] == "user"
    assert thread[1]["role"] == "assistant"
    assert thread[1]["content"] == "ack"


@pytest.mark.asyncio
async def test_persistent_whatsapp_rehydrates_after_restart(
    persistent_agent, tmp_path,
):
    """Same channel_key on a fresh agent + same DB → rehydrated thread."""
    agent, store = persistent_agent
    await agent._on_whatsapp_message(WhatsAppMessage(
        sender_name="Jacob",
        sender_phone="+15551111111",
        text="round one",
    ))
    await _drain_active(agent)
    first_id = agent._conversation_by_key["whatsapp:+15551111111"]
    first_thread_len = len(agent._conversations[first_id].thread)

    # Simulate restart: drop in-memory state, build a fresh agent that
    # shares the same store.
    bus = get_event_bus()
    bus.unsubscribe(WhatsAppMessage, agent._on_whatsapp_message)
    bus.unsubscribe(ConversationEnded, agent._on_conversation_ended)

    mem = MagicMock()
    mem.read_system_memory = MagicMock(return_value="")
    agent2 = BoxBotAgent(memory_store=mem, conversation_store=store)
    agent2._client = MagicMock()
    agent2._running = True
    bus.subscribe(WhatsAppMessage, agent2._on_whatsapp_message)
    bus.subscribe(ConversationEnded, agent2._on_conversation_ended)
    agent2._generate_for_conversation = _make_stub_generate(reply="ack2")

    try:
        await agent2._on_whatsapp_message(WhatsAppMessage(
            sender_name="Jacob",
            sender_phone="+15551111111",
            text="round two",
        ))
        await _drain_active(agent2)

        # Same conversation id reused (rehydrated), thread grew.
        assert agent2._conversation_by_key["whatsapp:+15551111111"] == first_id
        conv = agent2._conversations[first_id]
        assert len(conv.thread) == first_thread_len + 2
        # Order: user(round one), assistant(ack), user(round two), assistant(ack2)
        # User messages get a "[Jacob]: " prefix from _format_user_message.
        assert conv.thread[-2]["content"].endswith("round two")
        assert conv.thread[-1]["content"] == "ack2"
        # Earlier turns survived the rehydration.
        assert conv.thread[0]["content"].endswith("round one")
        assert conv.thread[1]["content"] == "ack"
    finally:
        bus.unsubscribe(WhatsAppMessage, agent2._on_whatsapp_message)
        bus.unsubscribe(ConversationEnded, agent2._on_conversation_ended)


@pytest.mark.asyncio
async def test_persistent_whatsapp_starts_fresh_after_window(
    persistent_agent,
):
    """A message arriving past the window mints a fresh conversation."""
    agent, store = persistent_agent
    await agent._on_whatsapp_message(WhatsAppMessage(
        sender_name="Jacob",
        sender_phone="+15551111111",
        text="morning",
    ))
    await _drain_active(agent)
    first_id = agent._conversation_by_key["whatsapp:+15551111111"]

    # Force the existing row's last_activity into the past, beyond
    # the configured window (default 14400s).
    db = store._require_db()
    far_past = (
        datetime.now(timezone.utc) - timedelta(seconds=20000)
    ).isoformat()
    await db.execute(
        "UPDATE conversations SET last_activity_at_iso = ? "
        "WHERE conversation_id = ?",
        (far_past, first_id),
    )
    await db.commit()
    # Drop in-memory entry to mimic the row going stale.
    agent._conversations.pop(first_id, None)
    agent._conversation_by_key.pop("whatsapp:+15551111111", None)

    await agent._on_whatsapp_message(WhatsAppMessage(
        sender_name="Jacob",
        sender_phone="+15551111111",
        text="evening, different topic",
    ))
    await _drain_active(agent)
    second_id = agent._conversation_by_key["whatsapp:+15551111111"]
    assert second_id != first_id


@pytest.mark.asyncio
async def test_extraction_sweep_marks_expired_active(persistent_agent):
    """Sweep iteration flips expired rows to 'extracted'."""
    agent, store = persistent_agent
    # Patch _post_conversation so the test doesn't depend on real
    # extraction infrastructure.
    called = []

    async def _fake_post(**kwargs):
        called.append(kwargs["conversation_id"])

    agent._post_conversation = _fake_post  # type: ignore[method-assign]

    await agent._on_whatsapp_message(WhatsAppMessage(
        sender_name="Jacob",
        sender_phone="+15551111111",
        text="hi",
    ))
    await _drain_active(agent)
    conv_id = agent._conversation_by_key["whatsapp:+15551111111"]

    # Force expiry.
    db = store._require_db()
    far_past = (
        datetime.now(timezone.utc) - timedelta(seconds=20000)
    ).isoformat()
    await db.execute(
        "UPDATE conversations SET last_activity_at_iso = ? "
        "WHERE conversation_id = ?",
        (far_past, conv_id),
    )
    await db.commit()

    # Run a single sweep iteration directly (bypass the loop).
    await agent._run_extraction_sweep(store, window=14400.0)
    # Let the create_task'd extraction kick off.
    await asyncio.sleep(0)

    fetched = await store.get(conv_id)
    assert fetched.state == "extracted"
    assert conv_id in called


@pytest.mark.asyncio
async def test_extraction_sweep_idempotent_on_already_extracted(
    persistent_agent,
):
    """A second sweep over the same expired row does not double-extract."""
    agent, store = persistent_agent
    called = []

    async def _fake_post(**kwargs):
        called.append(kwargs["conversation_id"])

    agent._post_conversation = _fake_post  # type: ignore[method-assign]

    await agent._on_whatsapp_message(WhatsAppMessage(
        sender_name="Jacob",
        sender_phone="+15551111111",
        text="hi",
    ))
    await _drain_active(agent)
    conv_id = agent._conversation_by_key["whatsapp:+15551111111"]

    db = store._require_db()
    far_past = (
        datetime.now(timezone.utc) - timedelta(seconds=20000)
    ).isoformat()
    await db.execute(
        "UPDATE conversations SET last_activity_at_iso = ? "
        "WHERE conversation_id = ?",
        (far_past, conv_id),
    )
    await db.commit()

    await agent._run_extraction_sweep(store, window=14400.0)
    await asyncio.sleep(0)
    await agent._run_extraction_sweep(store, window=14400.0)
    await asyncio.sleep(0)

    assert called == [conv_id]

