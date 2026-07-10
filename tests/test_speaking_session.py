"""The speaking ceremony must leave the box as it found it.

Regression coverage for the 2026-07-08 incident: a Signal message asked
BB to speak a question into the room. ``_speaking_session`` lit the ring
green and detached mic capture unconditionally, but restored both only
when a voice session happened to be ACTIVE. Coming from a text channel
there was no voice session, so the ring stayed green for two days while
nothing was attached to the microphone — the box advertised that it was
listening when it was not, and the person it spoke to answered into a
void.

These tests pin the invariant that broke: speech restores the LED and
capture state it found, and speech dispatched on a human's behalf opens
the mic and seeds the room conversation with what was asked.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from boxbot.communication.audio_capture import AudioCapture
from boxbot.communication.voice import VoiceSession, VoiceSessionState
from boxbot.core.conversation import Conversation, RelayContext


class _FakeMic:
    """Microphone stand-in that tracks the ring pattern like the real HAL."""

    def __init__(self) -> None:
        self._current_pattern = "off"
        self._next_handle = 0
        self.consumers: dict[int, object] = {}

    @property
    def current_led_pattern(self) -> str:
        return self._current_pattern

    async def set_led_pattern(self, pattern: str, params=None) -> None:
        self._current_pattern = pattern

    def add_consumer(self, callback, name: str = "") -> int:
        self._next_handle += 1
        self.consumers[self._next_handle] = callback
        return self._next_handle

    def remove_consumer(self, handle: int) -> None:
        self.consumers.pop(handle, None)


def _make_session():
    from boxbot.core.config import VoiceConfig

    config = VoiceConfig()
    mic = _FakeMic()
    speaker = MagicMock()
    speaker.is_playing = False
    speaker.stop_playback = AsyncMock()

    session = VoiceSession(mic, speaker, config)
    session._audio_capture = AudioCapture(MagicMock(), config.turn_detection)
    return session, mic


class TestSpeakingSessionRestoresState:
    """``_speaking_session`` is symmetric: what it takes, it gives back."""

    @pytest.mark.asyncio
    async def test_idle_speech_leaves_ring_dark_and_mic_detached(self):
        """The 2026-07-08 bug: green ring, no capture, forever."""
        session, mic = _make_session()
        session._state = VoiceSessionState.IDLE

        async with session._speaking_session(label="hello"):
            assert mic.current_led_pattern == "speaking"

        assert mic.current_led_pattern == "off", (
            "ring left lit after speaking from an idle session — the box "
            "is claiming to listen"
        )
        assert not session._audio_capture.is_running
        assert mic.consumers == {}

    @pytest.mark.asyncio
    async def test_active_speech_returns_to_listening_and_rearms_capture(self):
        """The pre-existing ACTIVE behaviour must not regress."""
        session, mic = _make_session()
        session._state = VoiceSessionState.ACTIVE
        await session._audio_capture.start(mic)
        await mic.set_led_pattern("thinking")

        async with session._speaking_session(label="hello"):
            assert mic.current_led_pattern == "speaking"
            assert not session._audio_capture.is_running

        assert mic.current_led_pattern == "listening"
        assert session._audio_capture.is_running

    @pytest.mark.asyncio
    async def test_muted_ring_survives_idle_speech(self):
        """A deliberate mute_mic state is not clobbered by speaking."""
        session, mic = _make_session()
        session._state = VoiceSessionState.IDLE
        await mic.set_led_pattern("muted")

        async with session._speaking_session(label="hello"):
            pass

        assert mic.current_led_pattern == "muted"

    @pytest.mark.asyncio
    async def test_session_ending_mid_speech_does_not_resurrect_ring(self):
        """If the session dies while we speak, don't repaint its ring."""
        session, mic = _make_session()
        session._state = VoiceSessionState.ACTIVE
        await mic.set_led_pattern("listening")

        async with session._speaking_session(label="hello"):
            session._state = VoiceSessionState.IDLE

        assert mic.current_led_pattern == "off"
        assert not session._audio_capture.is_running


class TestSpeakAndListen:
    """Speaking into the room on someone's behalf opens the mic."""

    @pytest.mark.asyncio
    async def test_opens_capture_from_idle(self):
        session, mic = _make_session()
        session._state = VoiceSessionState.IDLE
        session.speak = AsyncMock(wraps=_noop_speak(session))

        await session.speak_and_listen("Carina — books tonight?")

        assert session.state is VoiceSessionState.ACTIVE
        assert session._audio_capture.is_running
        assert mic.current_led_pattern == "listening"

    @pytest.mark.asyncio
    async def test_relay_context_is_stashed_and_consumed_once(self):
        session, _mic = _make_session()
        session._state = VoiceSessionState.IDLE
        session.speak = AsyncMock(wraps=_noop_speak(session))
        relay = _relay()

        await session.speak_and_listen("Books tonight?", relay=relay)

        assert session.consume_relay_context() is relay
        assert session.consume_relay_context() is None, (
            "a follow-up transcript is an ordinary reply, not a new relay"
        )

    @pytest.mark.asyncio
    async def test_no_relay_pending_by_default(self):
        session, _mic = _make_session()
        assert session.consume_relay_context() is None

    @pytest.mark.asyncio
    async def test_unanswered_relay_is_discarded_when_session_ends(self):
        """Nobody answered. Don't seed this into tomorrow's conversation."""
        session, _mic = _make_session()
        session._state = VoiceSessionState.ACTIVE
        session._pending_relay = _relay()

        await session._deactivate_session(reason="conversation_ended")

        assert session.consume_relay_context() is None

    @pytest.mark.asyncio
    async def test_unanswered_relay_is_discarded_going_dormant(self):
        session, _mic = _make_session()
        session._state = VoiceSessionState.ACTIVE
        session._pending_relay = _relay()

        await session._enter_dormant()

        assert session.consume_relay_context() is None

    @pytest.mark.asyncio
    async def test_relay_arms_a_grace_timer_so_the_mic_cannot_stay_hot(self):
        """No answer must eventually close the mic.

        ``_on_agent_speaking`` cancels the grace timer the moment TTS
        begins, and the post-response idle timer that normally takes over
        only arms on a voice-channel AgentTurnEnded — which a relay never
        produces, because the turn belongs to the text conversation.
        Without a re-arm after speech the mic stays hot forever.

        The speak stub drives ``_on_agent_speaking`` directly, standing in
        for the event bus that ``start()`` would have wired up.
        """
        from boxbot.core.events import AgentSpeaking

        session, _mic = _make_session()
        session._state = VoiceSessionState.IDLE

        async def _speak(text: str, priority: str = "normal") -> None:
            await session._on_agent_speaking(
                AgentSpeaking(conversation_id=session._conversation_id, text=text)
            )
            assert session._active_timeout_task is None, (
                "precondition: AgentSpeaking should have cancelled the grace "
                "timer armed by _activate_session"
            )

        session.speak = _speak

        await session.speak_and_listen("Books tonight?", relay=_relay())

        assert session._active_timeout_task is not None, (
            "unanswered relay left the mic open with no timeout armed"
        )
        assert not session._active_timeout_task.done()
        session._active_timeout_task.cancel()


class TestRelayContextTurns:
    """The room conversation must learn what was asked and who is waiting."""

    def test_turns_alternate_and_name_the_asker(self):
        turns = Conversation.build_relay_context_turns(_relay())

        assert [t["role"] for t in turns] == ["user", "assistant"]
        framing, spoken = turns[0]["content"], turns[1]["content"]
        assert framing.startswith("[relay]")
        assert "Jacob" in framing and "Carina" in framing
        assert "signal" in framing
        assert 'channel="text"' in framing
        assert spoken == "Books tonight?"


class TestAgentSeedsRoomConversation:
    """The room conversation learns the relay before it sees the answer."""

    @pytest.mark.asyncio
    async def test_relay_is_ingested_into_the_room_thread(self, monkeypatch):
        from boxbot.core.agent import BoxBotAgent
        import boxbot.communication.voice as voice_mod

        session = MagicMock()
        session.consume_relay_context = MagicMock(return_value=_relay())
        monkeypatch.setattr(
            voice_mod, "get_voice_session", lambda: session, raising=False
        )

        conv = MagicMock()
        conv.conversation_id = "conv_room"
        conv.ingest_context_turns = AsyncMock(return_value=True)

        # Unbound: the helper touches only the conversation and session.
        await BoxBotAgent._ingest_pending_relay(MagicMock(), conv)

        conv.ingest_context_turns.assert_awaited_once()
        turns = conv.ingest_context_turns.await_args.kwargs["turns"]
        assert [t["role"] for t in turns] == ["user", "assistant"]
        assert "Jacob" in turns[0]["content"]

    @pytest.mark.asyncio
    async def test_no_relay_pending_is_a_noop(self, monkeypatch):
        from boxbot.core.agent import BoxBotAgent
        import boxbot.communication.voice as voice_mod

        session = MagicMock()
        session.consume_relay_context = MagicMock(return_value=None)
        monkeypatch.setattr(
            voice_mod, "get_voice_session", lambda: session, raising=False
        )

        conv = MagicMock()
        conv.ingest_context_turns = AsyncMock()

        await BoxBotAgent._ingest_pending_relay(MagicMock(), conv)

        conv.ingest_context_turns.assert_not_awaited()


def _relay() -> RelayContext:
    return RelayContext(
        origin_conversation_id="conv_934cf4780ab5",
        origin_channel="signal",
        origin_person="Jacob",
        addressee="Carina",
        spoken_text="Books tonight?",
    )


def _noop_speak(session):
    """Stand in for TTS: run the ceremony without touching ElevenLabs."""

    async def _speak(text: str, priority: str = "normal") -> None:
        async with session._speaking_session(label=text):
            pass

    return _speak
