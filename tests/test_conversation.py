"""Tests for ``boxbot.core.conversation.Conversation``.

Covers the core state machine, cancel-and-regenerate on new input, the
per-conversation silence timeout, event publishing, and basic error
recovery. The Conversation takes an injected ``generate_fn`` so tests
can simulate short, hung, or cancelled generations deterministically.
"""

from __future__ import annotations

import asyncio

import pytest

from boxbot.core.conversation import (
    Conversation,
    ConversationState,
    GenerationResult,
    SpokenSegment,
)
from boxbot.core.events import (
    ConversationEnded,
    ConversationStarted,
    get_event_bus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _EventSink:
    """Accumulates events from the global bus for a single test."""

    def __init__(self) -> None:
        self.started: list[ConversationStarted] = []
        self.ended: list[ConversationEnded] = []

    async def on_started(self, event: ConversationStarted) -> None:
        self.started.append(event)

    async def on_ended(self, event: ConversationEnded) -> None:
        self.ended.append(event)


@pytest.fixture
def event_sink():
    """Subscribe to conversation lifecycle events; unsubscribe on teardown."""
    bus = get_event_bus()
    sink = _EventSink()
    bus.subscribe(ConversationStarted, sink.on_started)
    bus.subscribe(ConversationEnded, sink.on_ended)
    try:
        yield sink
    finally:
        bus.unsubscribe(ConversationStarted, sink.on_started)
        bus.unsubscribe(ConversationEnded, sink.on_ended)


def _make_conv(generate_fn, *, silence_timeout: float | None = None) -> Conversation:
    return Conversation(
        conversation_id="conv_test_01",
        channel="voice",
        channel_key="voice:room",
        generate_fn=generate_fn,
        silence_timeout=silence_timeout,
    )


# ---------------------------------------------------------------------------
# Construction + basic state
# ---------------------------------------------------------------------------


def test_new_conversation_starts_idle():
    async def gen(_conv):
        return GenerationResult()
    conv = _make_conv(gen)
    assert conv.state is ConversationState.IDLE
    assert conv.thread == []
    assert conv.is_generating is False
    assert conv.is_ended is False


# ---------------------------------------------------------------------------
# Happy path: single turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_turn_happy_path(event_sink):
    async def gen(conv):
        # Simulate a short Claude call: advance through THINKING→
        # SPEAKING, deliver one segment, then return.
        conv.set_state(ConversationState.SPEAKING)
        conv.record_segment(SpokenSegment(
            channel="voice", to="room", content="Hello there.",
        ))
        return GenerationResult(
            thread_additions=[{"role": "assistant", "content": "Hello there."}],
            turn_count=1,
            summary="Hello there.",
        )

    conv = _make_conv(gen, silence_timeout=0)
    await conv.handle_input("hi there", speaker_name="Jacob")

    # Wait for the generation task to finish.
    await conv._generation_task  # type: ignore[arg-type]

    assert conv.state is ConversationState.LISTENING
    # Thread contains [user, assistant].
    assert len(conv.thread) == 2
    assert conv.thread[0]["role"] == "user"
    assert "hi there" in conv.thread[0]["content"]
    assert conv.thread[1]["role"] == "assistant"
    # ConversationStarted fired exactly once; ended not yet.
    assert len(event_sink.started) == 1
    assert event_sink.started[0].conversation_id == "conv_test_01"
    assert event_sink.started[0].channel == "voice"
    assert len(event_sink.ended) == 0


@pytest.mark.asyncio
async def test_participants_tracked_from_speaker():
    async def gen(_conv):
        return GenerationResult()
    conv = _make_conv(gen, silence_timeout=0)
    await conv.handle_input("hi", speaker_name="Jacob")
    await conv._generation_task  # type: ignore[arg-type]
    await conv.handle_input("also me", speaker_name="Carina")
    await conv._generation_task  # type: ignore[arg-type]
    assert conv.participants == {"Jacob", "Carina"}


# ---------------------------------------------------------------------------
# Interrupt-and-regenerate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_new_input_cancels_in_flight_generation():
    """Core requirement: while the agent is thinking, a new utterance
    cancels the current generation and starts a fresh one."""
    gen_started = asyncio.Event()
    gen_cancelled = asyncio.Event()

    async def gen(conv):
        if conv.thread[-1]["content"].endswith("first"):
            gen_started.set()
            try:
                # Pretend to generate forever until cancelled.
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                gen_cancelled.set()
                raise
            return GenerationResult()  # unreachable
        # Second call — return immediately.
        return GenerationResult(
            thread_additions=[{"role": "assistant", "content": "second ack"}],
        )

    conv = _make_conv(gen, silence_timeout=0)
    await conv.handle_input("first")
    await asyncio.wait_for(gen_started.wait(), timeout=1.0)

    # Interrupt with a new input.
    await conv.handle_input("second")

    assert gen_cancelled.is_set()
    await conv._generation_task  # type: ignore[arg-type]
    assert conv.state is ConversationState.LISTENING
    # Thread order: user(first), user(second), assistant(second ack).
    roles = [m["role"] for m in conv.thread]
    assert roles == ["user", "user", "assistant"]
    assert "first" in conv.thread[0]["content"]
    assert "second" in conv.thread[1]["content"]
    assert conv.thread[2]["content"] == "second ack"


@pytest.mark.asyncio
async def test_thinking_interrupt_folds_partial_into_thread():
    """A new utterance during THINKING (before any TTS has started)
    cancels the in-flight generation and folds whatever segments the
    generator had recorded into the thread as an interrupted assistant
    turn so the next cycle sees what the user actually heard."""
    started = asyncio.Event()

    async def gen(conv):
        if conv.thread[-1]["content"].endswith("first"):
            # Stay in THINKING but record a segment to simulate work
            # already done in the partial-delivery sense.
            conv.record_segment(SpokenSegment(
                channel="voice", to="room",
                content="I was about to say something long",
            ))
            started.set()
            await asyncio.sleep(10)  # gets cancelled
            return GenerationResult()
        return GenerationResult(
            thread_additions=[{"role": "assistant", "content": "ok"}],
        )

    conv = _make_conv(gen, silence_timeout=0)
    await conv.handle_input("first")
    await asyncio.wait_for(started.wait(), timeout=1.0)
    await conv.handle_input("stop")
    await conv._generation_task  # type: ignore[arg-type]

    roles = [m["role"] for m in conv.thread]
    # Expect: user(first), assistant(interrupted partial),
    # user(stop), assistant(ok).
    assert roles == ["user", "assistant", "user", "assistant"]
    assert "about to say something long" in conv.thread[1]["content"]
    assert "interrupted" in conv.thread[1]["content"].lower()


@pytest.mark.asyncio
async def test_speaking_state_queues_input_without_cancelling():
    """During SPEAKING, a new utterance is queued — not cancelled.
    The generator finishes its TTS uninterrupted; once it returns,
    the queued message is drained into the thread and a fresh
    generation runs."""
    speaking = asyncio.Event()
    speak_release = asyncio.Event()
    gen_calls = 0

    async def gen(conv):
        nonlocal gen_calls
        gen_calls += 1
        # First turn: simulate speaking, then wait for release.
        if conv.thread[-1]["content"].endswith("first"):
            conv.set_state(ConversationState.SPEAKING)
            conv.record_segment(SpokenSegment(
                channel="voice", to="room",
                content="here is the answer to your first question",
            ))
            speaking.set()
            await speak_release.wait()
            return GenerationResult(
                thread_additions=[
                    {"role": "assistant", "content": "first response"},
                ],
            )
        # Second turn (drained pending): respond to whatever piled up.
        return GenerationResult(
            thread_additions=[
                {"role": "assistant", "content": "ack overheard"},
            ],
        )

    conv = _make_conv(gen, silence_timeout=0)
    await conv.handle_input("first", speaker_name="Jacob")
    await asyncio.wait_for(speaking.wait(), timeout=1.0)
    assert conv.state is ConversationState.SPEAKING

    # Queue a stray utterance during SPEAKING. It must NOT cancel.
    await conv.handle_input("ben 10 alien watch", speaker_name="Kid")
    assert conv.state is ConversationState.SPEAKING
    assert conv.is_generating is True
    # Thread still only has the original "first" — overhead is queued.
    assert len(conv.thread) == 1

    # Let TTS complete.
    speak_release.set()
    await conv._generation_task  # type: ignore[arg-type]

    assert conv.state is ConversationState.LISTENING
    assert gen_calls == 2  # original + drain pass
    roles = [m["role"] for m in conv.thread]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert conv.thread[0]["content"].endswith("first")
    assert conv.thread[1]["content"] == "first response"
    assert "ben 10" in conv.thread[2]["content"]
    assert conv.thread[3]["content"] == "ack overheard"


@pytest.mark.asyncio
async def test_wake_word_interrupt_folds_partial_and_drops_queue():
    """``conv.interrupt()`` (the wake-word path) cancels the in-flight
    generation, folds any partial spoken segments into the thread as an
    interrupted assistant turn, and drops any utterances that had been
    queued during SPEAKING. The user has explicitly taken back the
    floor — overheard chatter from before the interrupt is irrelevant."""
    speaking = asyncio.Event()

    async def gen(conv):
        conv.set_state(ConversationState.SPEAKING)
        conv.record_segment(SpokenSegment(
            channel="voice", to="room",
            content="I was about to tell you about the weather",
        ))
        speaking.set()
        await asyncio.sleep(10)  # interrupt() will cancel this
        return GenerationResult()

    conv = _make_conv(gen, silence_timeout=0)
    await conv.handle_input("first", speaker_name="Jacob")
    await asyncio.wait_for(speaking.wait(), timeout=1.0)

    # Background utterance queues during SPEAKING.
    await conv.handle_input("kid babble", speaker_name="Kid")

    # Wake word fires → interrupt().
    await conv.interrupt()

    assert conv.state is ConversationState.LISTENING
    assert conv.is_generating is False

    roles = [m["role"] for m in conv.thread]
    # Expect: user(first), assistant(interrupted partial). The queued
    # "kid babble" was dropped; no draining.
    assert roles == ["user", "assistant"]
    assert "weather" in conv.thread[1]["content"]
    assert "interrupted" in conv.thread[1]["content"].lower()


@pytest.mark.asyncio
async def test_interrupt_is_noop_when_idle():
    """interrupt() with no in-flight generation and no queue must
    be a clean no-op — must not change state or publish events."""
    async def gen(_conv):
        return GenerationResult(
            thread_additions=[{"role": "assistant", "content": "x"}],
        )
    conv = _make_conv(gen, silence_timeout=0)
    await conv.handle_input("hi")
    await conv._generation_task  # type: ignore[arg-type]
    assert conv.state is ConversationState.LISTENING

    await conv.interrupt()
    assert conv.state is ConversationState.LISTENING


@pytest.mark.asyncio
async def test_rapid_successive_interrupts_do_not_wedge_state():
    """Three interrupts in quick succession. Final generation must
    still produce a clean LISTENING state."""
    gen_count = 0

    async def gen(conv):
        nonlocal gen_count
        gen_count += 1
        # Keep every generation but the last running "forever".
        if conv.thread[-1]["content"].endswith("final"):
            return GenerationResult(
                thread_additions=[{"role": "assistant", "content": "done"}],
            )
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            raise
        return GenerationResult()

    conv = _make_conv(gen, silence_timeout=0)
    await conv.handle_input("one")
    await asyncio.sleep(0)  # let gen start
    await conv.handle_input("two")
    await asyncio.sleep(0)
    await conv.handle_input("three")
    await asyncio.sleep(0)
    await conv.handle_input("final")
    await conv._generation_task  # type: ignore[arg-type]

    assert conv.state is ConversationState.LISTENING
    assert gen_count == 4
    # Last thread message is the "done" assistant turn.
    assert conv.thread[-1] == {"role": "assistant", "content": "done"}


# ---------------------------------------------------------------------------
# Silence timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_silence_timeout_ends_conversation(event_sink):
    async def gen(_conv):
        return GenerationResult(
            thread_additions=[{"role": "assistant", "content": "hi"}],
        )
    # Very short timeout for the test.
    conv = _make_conv(gen, silence_timeout=0.05)
    await conv.handle_input("ping")
    await conv._generation_task  # type: ignore[arg-type]
    # Let the silence timer fire.
    await asyncio.sleep(0.1)
    assert conv.state is ConversationState.ENDED
    assert len(event_sink.ended) == 1
    assert event_sink.ended[0].conversation_id == "conv_test_01"


@pytest.mark.asyncio
async def test_new_input_resets_silence_timer():
    async def gen(_conv):
        return GenerationResult(
            thread_additions=[{"role": "assistant", "content": "ok"}],
        )
    conv = _make_conv(gen, silence_timeout=0.1)
    await conv.handle_input("one")
    await conv._generation_task  # type: ignore[arg-type]
    # Right before timeout, speak again — should reset.
    await asyncio.sleep(0.05)
    await conv.handle_input("two")
    await conv._generation_task  # type: ignore[arg-type]
    # If the timer had NOT been reset, the conversation would already
    # be ENDED here (0.1s elapsed since the first input).
    assert conv.state is ConversationState.LISTENING


# ---------------------------------------------------------------------------
# Explicit end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_is_idempotent(event_sink):
    async def gen(_conv):
        return GenerationResult()
    conv = _make_conv(gen, silence_timeout=0)
    await conv.handle_input("hi")
    await conv._generation_task  # type: ignore[arg-type]
    await conv.end(reason="explicit")
    await conv.end(reason="again")  # second call no-ops
    assert conv.state is ConversationState.ENDED
    assert len(event_sink.ended) == 1  # only one ended event


@pytest.mark.asyncio
async def test_end_cancels_in_flight_generation():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def gen(_conv):
        started.set()
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return GenerationResult()

    conv = _make_conv(gen, silence_timeout=0)
    await conv.handle_input("hang")
    await asyncio.wait_for(started.wait(), timeout=1.0)
    await conv.end(reason="explicit")
    assert cancelled.is_set()
    assert conv.state is ConversationState.ENDED


@pytest.mark.asyncio
async def test_input_after_end_is_ignored():
    async def gen(_conv):
        return GenerationResult()
    conv = _make_conv(gen, silence_timeout=0)
    await conv.end(reason="test")
    await conv.handle_input("hello?")
    # Thread is empty — input was refused.
    assert conv.thread == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generator_exception_returns_to_listening():
    """A broken generator shouldn't wedge the conversation. Next input
    should start a fresh cycle."""
    calls = []

    async def gen(conv):
        calls.append(len(conv.thread))
        if len(conv.thread) == 1:
            raise RuntimeError("boom")
        return GenerationResult(
            thread_additions=[{"role": "assistant", "content": "recovered"}],
        )

    conv = _make_conv(gen, silence_timeout=0)
    await conv.handle_input("first")
    await conv._generation_task  # type: ignore[arg-type]
    # State should be LISTENING even after the exception.
    assert conv.state is ConversationState.LISTENING
    await conv.handle_input("second")
    await conv._generation_task  # type: ignore[arg-type]
    assert conv.state is ConversationState.LISTENING
    assert conv.thread[-1] == {"role": "assistant", "content": "recovered"}
    # First call saw 1 msg (the user input). Broken generator added
    # nothing. Second input appended user msg #2, so the second gen
    # call saw 2 messages at the time it was invoked.
    assert calls == [1, 2]


# ---------------------------------------------------------------------------
# Empty / whitespace input
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_input_is_dropped():
    async def gen(_conv):
        return GenerationResult(
            thread_additions=[{"role": "assistant", "content": "x"}],
        )
    conv = _make_conv(gen, silence_timeout=0)
    await conv.handle_input("")
    await conv.handle_input("   \n\t")
    assert conv.thread == []
    assert conv.state is ConversationState.IDLE


# ---------------------------------------------------------------------------
# Trigger input format
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trigger_input_is_tagged():
    async def gen(_conv):
        return GenerationResult(
            thread_additions=[{"role": "assistant", "content": "ack"}],
        )
    conv = _make_conv(gen, silence_timeout=0)
    await conv.handle_input(
        "Remind Jacob to check the thermostat",
        source="trigger",
    )
    await conv._generation_task  # type: ignore[arg-type]
    assert conv.thread[0]["content"].startswith("[trigger] ")


# ---------------------------------------------------------------------------
# Context passthrough
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_is_available_to_generator():
    observed = {}

    async def gen(conv):
        observed.update(conv.current_context)
        return GenerationResult()

    conv = _make_conv(gen, silence_timeout=0)
    await conv.handle_input(
        "hi",
        context={"voice_session_id": "v123", "speaker_tier": "high"},
    )
    await conv._generation_task  # type: ignore[arg-type]
    assert observed == {"voice_session_id": "v123", "speaker_tier": "high"}
