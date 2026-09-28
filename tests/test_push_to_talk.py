"""Tests for push-to-talk capture: the finger is the turn boundary.

The contract is small and entirely about state: a hold is open between
``begin_hold()`` and ``ButtonPressed(action="release")``, chunks are
buffered iff a hold is open, and exactly one Utterance leaves per hold.
Everything interesting here is a degenerate case — a lost release frame,
a replayed press, a touch-up nobody asked for — because on the panel
those arrive over a socket and cannot be assumed away.

The two edges enter by different doors on purpose. ``voice.py`` calls
``begin_hold()`` once it has stopped playback; only the release rides the
bus. See ``test_voice.py`` for the ordering that buys.

No VAD, no wake word, no torch: that is the point of the workstream, and
these tests run without any of them.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import numpy as np
import pytest
import pytest_asyncio

from boxbot.communication.push_to_talk import PushToTalkCapture
from boxbot.core.config import TurnDetectionConfig
from boxbot.core.events import ButtonPressed
from boxbot.hardware.base import AudioChunk

FRAMES = 160  # 10 ms at 16 kHz
CHUNK_BYTES = FRAMES * 2  # int16 mono


class _FakeMic:
    """Microphone stand-in that records consumer registration."""

    def __init__(self) -> None:
        self._next_handle = 0
        self.consumers: dict[int, object] = {}

    def add_consumer(self, callback, name: str = "") -> int:
        self._next_handle += 1
        self.consumers[self._next_handle] = callback
        return self._next_handle

    def remove_consumer(self, handle: int) -> None:
        self.consumers.pop(handle, None)

    async def feed(self, *chunks: AudioChunk) -> None:
        """Deliver chunks to whoever is currently registered."""
        for chunk in chunks:
            for callback in list(self.consumers.values()):
                await callback(chunk)


def _chunk(timestamp: float | None = None, value: int = 10_000) -> AudioChunk:
    data = (np.ones(FRAMES, dtype=np.int16) * value).tobytes()
    return AudioChunk(
        data=data,
        timestamp=time.monotonic() if timestamp is None else timestamp,
        sample_rate=16000,
        channels=1,
        frames=FRAMES,
    )


def _press(pt: PushToTalkCapture) -> None:
    """The press edge — a direct call, the way voice.py delivers it."""
    pt.begin_hold()


async def _release(bus, button_id: str = "screen") -> None:
    await bus.publish(ButtonPressed(button_id=button_id, action="release"))


@pytest_asyncio.fixture
async def capture(event_bus):
    """A started capture, its fake mic, and its utterance callback."""
    mic = _FakeMic()
    pt = PushToTalkCapture(TurnDetectionConfig())
    callback = AsyncMock()
    pt.set_utterance_callback(callback)
    await pt.start(mic)
    yield pt, mic, callback
    await pt.stop()


# ---------------------------------------------------------------------------
# The normal turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_press_speak_release_fires_one_utterance(capture, event_bus):
    pt, mic, callback = capture

    _press(pt)
    await mic.feed(_chunk(), _chunk(), _chunk())
    await _release(event_bus)

    callback.assert_awaited_once()
    utterance = callback.await_args.args[0]
    assert len(utterance.audio) == 3 * CHUNK_BYTES
    assert utterance.sample_rate == 16000
    # timestamp_end is release, on the monotonic clock latency.begin()
    # expects, and duration is the hold it closes.
    assert utterance.timestamp_end > utterance.timestamp_start
    assert utterance.duration == pytest.approx(
        utterance.timestamp_end - utterance.timestamp_start
    )
    assert not pt.is_holding


@pytest.mark.asyncio
async def test_chunks_outside_a_hold_are_dropped(capture, event_bus):
    """A continuously-fanning backend must not leak room audio."""
    pt, mic, callback = capture

    await mic.feed(_chunk(), _chunk())
    _press(pt)
    await mic.feed(_chunk())
    await _release(event_bus)
    await mic.feed(_chunk(), _chunk())

    assert len(callback.await_args.args[0].audio) == CHUNK_BYTES
    callback.assert_awaited_once()


@pytest.mark.asyncio
async def test_silence_is_captured(capture, event_bus):
    """No VAD gate: what the finger held is what gets sent, silent or not."""
    pt, mic, callback = capture

    _press(pt)
    await mic.feed(_chunk(value=0), _chunk(value=0))
    await _release(event_bus)

    assert len(callback.await_args.args[0].audio) == 2 * CHUNK_BYTES


@pytest.mark.asyncio
async def test_releases_from_other_buttons_are_ignored(capture, event_bus):
    """A future KB2040 knob must not end a hold the screen started."""
    pt, mic, callback = capture

    _press(pt)
    await mic.feed(_chunk())
    await _release(event_bus, button_id="knob")

    assert pt.is_holding
    callback.assert_not_awaited()

    await _release(event_bus)
    callback.assert_awaited_once()


@pytest.mark.asyncio
async def test_long_press_does_not_move_the_state_machine(capture, event_bus):
    """Only press and release are boundaries; long_press is neither."""
    pt, mic, callback = capture

    await event_bus.publish(ButtonPressed(button_id="screen", action="long_press"))
    assert not pt.is_holding

    _press(pt)
    await event_bus.publish(ButtonPressed(button_id="screen", action="long_press"))
    await mic.feed(_chunk())
    assert pt.is_holding
    await _release(event_bus)

    callback.assert_awaited_once()


# ---------------------------------------------------------------------------
# Degenerate cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_release_without_press_is_a_no_op(capture, event_bus):
    pt, mic, callback = capture

    await _release(event_bus)

    assert not pt.is_holding
    callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_hold_fires_nothing(capture, event_bus):
    """No audio arrived — firing would buy an STT call on silence."""
    pt, mic, callback = capture

    _press(pt)
    await _release(event_bus)

    callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_press_without_release_keeps_holding_and_stop_drops_it(
    capture, event_bus
):
    pt, mic, callback = capture

    _press(pt)
    await mic.feed(_chunk(), _chunk())
    assert pt.is_holding
    callback.assert_not_awaited()

    await pt.stop()

    # Half a hold nobody released is not a turn.
    assert not pt.is_holding
    callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_second_press_while_holding_keeps_the_audio(capture, event_bus):
    """A replayed touch-down must not discard what was already said."""
    pt, mic, callback = capture

    _press(pt)
    await mic.feed(_chunk())
    _press(pt)
    await mic.feed(_chunk())
    await _release(event_bus)

    callback.assert_awaited_once()
    assert len(callback.await_args.args[0].audio) == 2 * CHUNK_BYTES


@pytest.mark.asyncio
async def test_release_after_cap_finalized_is_a_no_op(event_bus):
    """The cap closed the hold; the late touch-up has nothing to send."""
    mic = _FakeMic()
    pt = PushToTalkCapture(TurnDetectionConfig(max_utterance_duration=1))
    callback = AsyncMock()
    pt.set_utterance_callback(callback)
    await pt.start(mic)

    _press(pt)
    await mic.feed(_chunk(), _chunk(timestamp=pt._press_time + 1.5))
    callback.assert_awaited_once()

    await _release(event_bus)

    assert callback.await_count == 1
    await pt.stop()


# ---------------------------------------------------------------------------
# Duration cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cap_auto_finalizes_and_drops_the_rest_of_the_hold(event_bus):
    """A finger stuck down (or a lost release frame) must not deadlock a turn."""
    mic = _FakeMic()
    pt = PushToTalkCapture(TurnDetectionConfig(max_utterance_duration=1))
    callback = AsyncMock()
    pt.set_utterance_callback(callback)
    await pt.start(mic)

    _press(pt)
    press_time = pt._press_time
    await mic.feed(
        _chunk(timestamp=press_time + 0.5),
        _chunk(timestamp=press_time + 1.0),  # hits the cap
    )

    callback.assert_awaited_once()
    utterance = callback.await_args.args[0]
    assert len(utterance.audio) == 2 * CHUNK_BYTES
    assert utterance.duration == pytest.approx(1.0)
    assert not pt.is_holding

    # Everything after the cap is dropped until the next press.
    await mic.feed(_chunk(timestamp=press_time + 1.5))
    assert callback.await_count == 1

    _press(pt)
    await mic.feed(_chunk())
    await _release(event_bus)
    assert callback.await_count == 2
    assert len(callback.await_args.args[0].audio) == CHUNK_BYTES

    await pt.stop()


@pytest.mark.asyncio
async def test_default_cap_bounds_the_buffer(capture):
    """60 s at 32 KB/s — the documented ~1.9 MB ceiling."""
    pt, _mic, _callback = capture
    cap = pt._config.max_utterance_duration
    assert cap * 16000 * 2 == pytest.approx(1_920_000)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consumer_added_on_start_and_removed_on_stop(event_bus):
    mic = _FakeMic()
    pt = PushToTalkCapture(TurnDetectionConfig())

    assert not pt.is_running
    await pt.start(mic)
    assert pt.is_running
    assert len(mic.consumers) == 1

    await pt.stop()
    assert not pt.is_running
    assert mic.consumers == {}


@pytest.mark.asyncio
async def test_start_is_idempotent(event_bus):
    mic = _FakeMic()
    pt = PushToTalkCapture(TurnDetectionConfig())

    await pt.start(mic)
    handle = pt._consumer_handle
    await pt.start(mic)

    assert pt._consumer_handle == handle
    assert len(mic.consumers) == 1
    await pt.stop()


@pytest.mark.asyncio
async def test_stop_unsubscribes_from_touch(event_bus):
    """A stopped capture is detached from mic and bus alike.

    ``stop()`` is the full teardown — it is what ``VoiceSession.stop()``
    calls, and the only thing that may drop the bus subscription. The
    hold still opens (``begin_hold`` is a direct call and does not know
    the capture was stopped), but no audio reaches it and the release
    never arrives, so nothing is ever sent.
    """
    mic = _FakeMic()
    pt = PushToTalkCapture(TurnDetectionConfig())
    callback = AsyncMock()
    pt.set_utterance_callback(callback)
    await pt.start(mic)
    await pt.stop()

    _press(pt)
    await mic.feed(_chunk())
    await _release(event_bus)

    assert not pt.is_running
    callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_stop_on_a_stopped_capture_is_a_no_op(event_bus):
    pt = PushToTalkCapture(TurnDetectionConfig())
    await pt.stop()
    assert not pt.is_running


# ---------------------------------------------------------------------------
# Mute
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_muted_capture_ignores_presses(capture, event_bus):
    """begin_hold is the last line of defence; voice.py also gates."""
    pt, mic, callback = capture

    pt.mute()
    assert pt.is_muted
    _press(pt)
    await mic.feed(_chunk())
    await _release(event_bus)

    assert not pt.is_holding
    callback.assert_not_awaited()

    pt.unmute()
    _press(pt)
    await mic.feed(_chunk())
    await _release(event_bus)

    callback.assert_awaited_once()


@pytest.mark.asyncio
async def test_mute_drops_an_open_hold(capture, event_bus):
    pt, mic, callback = capture

    _press(pt)
    await mic.feed(_chunk())
    pt.mute()

    assert not pt.is_holding
    await _release(event_bus)
    callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_mute_survives_stop_start(capture, event_bus):
    """Muting is a deliberate agent choice; a detach/reattach can't clear it."""
    pt, mic, _callback = capture

    pt.mute()
    await pt.stop()
    await pt.start(mic)

    assert pt.is_muted


# ---------------------------------------------------------------------------
# Holds that produce nothing still have to be reported
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_hold_reports_discarded(capture, event_bus):
    """The press opened a socket; somebody has to be told to close it."""
    pt, _mic, callback = capture
    discarded = AsyncMock()
    pt.set_hold_discarded_callback(discarded)

    _press(pt)
    await _release(event_bus)

    callback.assert_not_awaited()
    discarded.assert_awaited_once()


@pytest.mark.asyncio
async def test_release_with_no_hold_reports_discarded(capture, event_bus):
    """A muted press, or one that landed while the session tore down."""
    pt, _mic, callback = capture
    discarded = AsyncMock()
    pt.set_hold_discarded_callback(discarded)

    await _release(event_bus)

    callback.assert_not_awaited()
    discarded.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_real_utterance_does_not_report_discarded(capture, event_bus):
    """The utterance path owns the socket when there is an utterance."""
    pt, mic, callback = capture
    discarded = AsyncMock()
    pt.set_hold_discarded_callback(discarded)

    _press(pt)
    await mic.feed(_chunk())
    await _release(event_bus)

    callback.assert_awaited_once()
    discarded.assert_not_awaited()
