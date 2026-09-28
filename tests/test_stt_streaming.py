"""Tests for ScribeRealtimeSTT — the prebuffer, and every way a session ends.

No network and no ``elevenlabs`` package: the provider takes a ``connect``
seam returning a :class:`FakeConnection`, and ``stt._RealtimeEvents`` is
monkeypatched with :class:`FakeEvents` (the real symbol is ``None`` on any
machine without the SDK, which includes this venv and CI).

The prebuffer is the subject. Its contract is that capture never waits on
the handshake, so most of these tests hold the connect open deliberately
and assert on what happens to audio fed in the meantime.
"""

from __future__ import annotations

import asyncio
import base64

import pytest

from boxbot.communication import stt as stt_mod
from boxbot.communication.stt import (
    STTProvider,
    ScribeRealtimeSTT,
    StreamingSTTProvider,
    StreamingSTTUnavailableError,
)
from boxbot.core import latency

# 8192 B = one flush slice = 256 ms at 16 kHz mono s16le.
SLICE = 8192


class FakeEvents:
    """Stand-in for ``elevenlabs.realtime.connection.RealtimeEvents``.

    Attribute name == value, so a test can emit by the same string the
    provider dispatches on.
    """

    COMMITTED_TRANSCRIPT = "COMMITTED_TRANSCRIPT"
    PARTIAL_TRANSCRIPT = "PARTIAL_TRANSCRIPT"
    INSUFFICIENT_AUDIO_ACTIVITY = "INSUFFICIENT_AUDIO_ACTIVITY"
    CLOSE = "CLOSE"
    ERROR = "ERROR"
    AUTH_ERROR = "AUTH_ERROR"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    RATE_LIMITED = "RATE_LIMITED"
    CHUNK_SIZE_EXCEEDED = "CHUNK_SIZE_EXCEEDED"
    SESSION_TIME_LIMIT_EXCEEDED = "SESSION_TIME_LIMIT_EXCEEDED"


class FakeConnection:
    """Records what reached the wire; lets a test emit server events.

    ``emit`` calls handlers **synchronously** and swallows exceptions,
    mirroring the SDK's ``_emit`` — a handler that blocks or raises here
    would break the socket read loop in production, so the fake must be
    just as unforgiving.
    """

    def __init__(self) -> None:
        self.handlers: dict[str, list] = {}
        self.sent: list[bytes] = []
        self.commits = 0
        self.closed = 0
        self.send_error: Exception | None = None

    def on(self, event: str, callback) -> None:
        self.handlers.setdefault(event, []).append(callback)

    async def send(self, data: dict) -> None:
        if self.send_error is not None:
            raise self.send_error
        # The SDK reads ``audio_base_64``; anything else is silently empty
        # audio, so assert on the exact key rather than on ``data``.
        self.sent.append(base64.b64decode(data["audio_base_64"]))

    async def commit(self) -> None:
        self.commits += 1

    async def close(self) -> None:
        self.closed += 1

    def emit(self, event: str, payload: dict | None = None) -> None:
        for cb in self.handlers.get(event, []):
            try:
                cb(payload or {})
            except Exception:
                pass


@pytest.fixture(autouse=True)
def realtime_env(monkeypatch):
    """Supply the SDK enums and keep cost writes away from sqlite."""
    monkeypatch.setattr(stt_mod, "_RealtimeEvents", FakeEvents)
    recorded: list[dict] = []

    async def _fake_record(**kwargs):
        recorded.append(kwargs)

    monkeypatch.setattr(stt_mod, "_record_stt_cost", _fake_record)
    return recorded


class Session:
    """A provider plus its fake connection, with the handshake on a leash.

    ``release()`` completes the pending connect. Nothing connects until a
    test says so, which is what makes "capture never waits on the
    handshake" testable rather than a race.
    """

    def __init__(self, *, fail: Exception | None = None) -> None:
        self.conn = FakeConnection()
        self.gate = asyncio.Event()
        self.fail = fail
        self.stt = ScribeRealtimeSTT(
            api_key="k", model="scribe_v2_realtime", connect=self._connect
        )

    async def _connect(self, language: str) -> FakeConnection:
        await self.gate.wait()
        if self.fail is not None:
            raise self.fail
        return self.conn

    def release(self) -> None:
        self.gate.set()


async def settle(predicate, timeout: float = 1.0) -> None:
    """Yield to the loop until ``predicate`` holds (or fail the test)."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition never became true")
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# The prebuffer
# ---------------------------------------------------------------------------


async def test_feed_does_not_wait_on_the_handshake():
    """open() returns with no socket, and feed() still accepts audio."""
    s = Session()
    s.stt.open(16000)

    assert s.stt._conn is None
    s.stt.feed(b"\x01" * 4096)
    assert s.stt.buffered_audio == b"\x01" * 4096
    assert s.conn.sent == []

    await s.stt.close()


async def test_prebuffered_audio_flushes_in_order_once_connected():
    """Three slices fed pre-handshake arrive in order, exactly once each."""
    s = Session()
    s.stt.open(16000)
    for marker in (b"\xaa", b"\xbb", b"\xcc"):
        s.stt.feed(marker * SLICE)

    assert s.conn.sent == []
    s.release()
    await settle(lambda: len(s.conn.sent) == 3)

    assert s.conn.sent == [b"\xaa" * SLICE, b"\xbb" * SLICE, b"\xcc" * SLICE]
    await s.stt.close()


async def test_flush_uses_8192_byte_slices_with_remainder_on_commit():
    """Whole slices go out live; the partial tail waits for the commit."""
    s = Session()
    s.stt.open(16000)
    s.release()
    # Feed in 1 KB pieces — the provider coalesces, it does not forward
    # the mic's chunk size.
    for _ in range(20):
        s.stt.feed(b"\x07" * 1024)
    await settle(lambda: len(s.conn.sent) == 2)
    assert [len(c) for c in s.conn.sent] == [SLICE, SLICE]

    harvest = asyncio.create_task(s.stt.harvest(timeout=1.0))
    await settle(lambda: s.conn.commits == 1)
    assert [len(c) for c in s.conn.sent] == [SLICE, SLICE, 20 * 1024 - 2 * SLICE]

    s.conn.emit("COMMITTED_TRANSCRIPT", {"text": "hello"})
    assert (await harvest).text == "hello"


async def test_release_before_connect_loses_no_audio():
    """A press shorter than the handshake still transcribes in full."""
    s = Session()
    s.stt.open(16000)
    s.stt.feed(b"\x11" * 3000)  # under one slice: nothing to flush yet

    harvest = asyncio.create_task(s.stt.harvest(timeout=1.0))
    await asyncio.sleep(0)  # harvest is now blocked on the handshake
    assert s.conn.sent == []

    s.release()
    await settle(lambda: s.conn.commits == 1)
    assert s.conn.sent == [b"\x11" * 3000]

    s.conn.emit("COMMITTED_TRANSCRIPT", {"text": "quick"})
    assert (await harvest).text == "quick"


async def test_connect_failure_leaves_the_audio_retrievable():
    """Fallback contract: legible failure, buffer intact, nothing sent."""
    s = Session(fail=OSError("dns"))
    s.stt.open(16000)
    s.stt.feed(b"\x22" * (SLICE * 2))
    s.release()

    with pytest.raises(StreamingSTTUnavailableError):
        await s.stt.harvest(timeout=1.0)

    assert s.stt.buffered_audio == b"\x22" * (SLICE * 2)
    assert s.conn.sent == []
    # close() must survive a session that never connected, and must not
    # bill for a socket that never opened.
    await s.stt.close()
    assert s.stt.buffered_audio == b"\x22" * (SLICE * 2)


async def test_send_failure_leaves_the_audio_retrievable():
    """A socket that dies mid-utterance degrades the same way."""
    s = Session()
    s.conn.send_error = ConnectionResetError("gone")
    s.stt.open(16000)
    s.stt.feed(b"\x33" * SLICE)
    s.release()

    with pytest.raises(StreamingSTTUnavailableError):
        await s.stt.harvest(timeout=1.0)
    assert s.stt.buffered_audio == b"\x33" * SLICE
    await s.stt.close()


async def test_buffer_is_capped_at_sixty_seconds():
    """32 KB/s × 60 s. Past the cap the tail is dropped, not the head."""
    s = Session()
    s.stt.open(16000)
    cap = 60 * 16000 * 2
    s.stt.feed(b"\x44" * (cap + 5000))

    buffered = s.stt.buffered_audio
    assert len(buffered) == cap
    assert buffered[:1] == b"\x44"
    s.stt.feed(b"\x55" * 1024)
    assert len(s.stt.buffered_audio) == cap

    await s.stt.close()


async def test_reopen_on_a_live_session_keeps_the_buffer():
    """A replayed touch frame must not discard what was already said."""
    s = Session()
    s.stt.open(16000)
    s.stt.feed(b"\x66" * 512)
    s.stt.open(16000)
    assert s.stt.buffered_audio == b"\x66" * 512
    await s.stt.close()


# ---------------------------------------------------------------------------
# How sessions end
# ---------------------------------------------------------------------------


async def test_insufficient_audio_activity_is_an_empty_transcript():
    """The streaming silence case — same outcome as annotation-only batch."""
    s = Session()
    s.stt.open(16000)
    s.release()
    s.stt.feed(b"\x00" * 1024)

    harvest = asyncio.create_task(s.stt.harvest(timeout=1.0))
    await settle(lambda: s.conn.commits == 1)
    s.conn.emit("INSUFFICIENT_AUDIO_ACTIVITY", {})

    result = await harvest
    assert result.text == ""
    await s.stt.close()


async def test_annotation_only_transcript_is_dropped():
    s = Session()
    s.stt.open(16000)
    s.release()
    s.stt.feed(b"\x00" * 1024)

    harvest = asyncio.create_task(s.stt.harvest(timeout=1.0))
    await settle(lambda: s.conn.commits == 1)
    s.conn.emit("COMMITTED_TRANSCRIPT", {"text": "[background noise]"})

    assert (await harvest).text == ""
    await s.stt.close()


async def test_partials_are_waited_past():
    """MANUAL commit strategy: only COMMITTED_TRANSCRIPT is final."""
    s = Session()
    s.stt.open(16000)
    s.release()
    s.stt.feed(b"\x00" * 1024)

    harvest = asyncio.create_task(s.stt.harvest(timeout=1.0))
    await settle(lambda: s.conn.commits == 1)
    s.conn.emit("PARTIAL_TRANSCRIPT", {"text": "turn on the"})
    await asyncio.sleep(0)
    assert not harvest.done()

    s.conn.emit("COMMITTED_TRANSCRIPT", {"text": "turn on the lights"})
    assert (await harvest).text == "turn on the lights"
    await s.stt.close()


async def test_bounded_wait_expires_rather_than_hanging():
    """No transcript ever arrives; the turn must still end."""
    s = Session()
    s.stt.open(16000)
    s.release()
    s.stt.feed(b"\x00" * 1024)

    with pytest.raises(StreamingSTTUnavailableError, match="within"):
        await s.stt.harvest(timeout=0.05)
    assert s.conn.commits == 1
    await s.stt.close()


@pytest.mark.parametrize(
    "event",
    ["ERROR", "AUTH_ERROR", "QUOTA_EXCEEDED", "RATE_LIMITED",
     "CHUNK_SIZE_EXCEEDED", "SESSION_TIME_LIMIT_EXCEEDED", "CLOSE"],
)
async def test_terminal_events_fall_back(event):
    """Every error frame, and a bare hangup, means "use batch".

    CLOSE carries no payload and no error — the SDK emits it from the
    ``finally`` of its message handler. Without it the socket going away
    mid-utterance would cost the full harvest timeout of dead air before
    the batch fallback started.
    """
    s = Session()
    s.stt.open(16000)
    s.release()
    s.stt.feed(b"\x77" * 1024)

    harvest = asyncio.create_task(s.stt.harvest(timeout=1.0))
    await settle(lambda: s.conn.commits == 1)
    s.conn.emit(event, {"message": "nope"})

    with pytest.raises(StreamingSTTUnavailableError, match=event):
        await harvest
    assert s.stt.buffered_audio == b"\x77" * 1024
    await s.stt.close()


# ---------------------------------------------------------------------------
# Cost, latency, and the STTProvider adapter
# ---------------------------------------------------------------------------


async def test_close_bills_connection_seconds_not_speech(realtime_env):
    """Realtime meters the socket, so the row must say so."""
    s = Session()
    s.stt.open(16000, conversation_id="voice_1")
    s.release()
    s.stt.feed(b"\x00" * 1024)
    await settle(lambda: s.stt._conn is not None)
    await s.stt.close()

    assert len(realtime_env) == 1
    row = realtime_env[0]
    assert row["model"] == "scribe_v2_realtime"
    assert row["metadata"] == {"billed_unit": "connection_seconds"}
    assert row["conversation_id"] == "voice_1"
    # A 1 KB feed is 32 ms of speech; the socket was open for less than a
    # second. The point is only that the two numbers are unrelated.
    assert row["audio_seconds"] >= 0.0
    assert s.conn.closed == 1


async def test_never_connected_session_is_not_billed(realtime_env):
    s = Session(fail=OSError("dns"))
    s.stt.open(16000)
    s.release()
    await settle(lambda: s.stt._failure is not None)
    await s.stt.close()
    assert realtime_env == []


async def test_latency_records_stt_and_handshake_spans():
    s = Session()
    s.stt.open(16000, conversation_id="voice_2")
    s.release()
    s.stt.feed(b"\x00" * 1024)
    # The tracker opens at speech-end, after the handshake — which is why
    # the handshake is measured during the hold and reported at harvest.
    latency.begin("voice_2")

    harvest = asyncio.create_task(s.stt.harvest(timeout=1.0))
    await settle(lambda: s.conn.commits == 1)
    s.conn.emit("COMMITTED_TRANSCRIPT", {"text": "hi"})
    await harvest

    tracker = latency.get("voice_2")
    assert tracker is not None
    assert "stt" in tracker.spans
    assert "stt_connect" in tracker.spans
    latency.discard("voice_2")
    await s.stt.close()


async def test_transcribe_harvests_and_closes_the_open_session():
    """The STTProvider adapter that keeps voice.py::_on_utterance unchanged."""
    s = Session()
    s.stt.open(16000)
    s.release()
    s.stt.feed(b"\x88" * 1024)

    # ``audio`` is the caller's fallback payload, not an input — the same
    # PCM already went out during the hold.
    task = asyncio.create_task(s.stt.transcribe(b"\x88" * 1024, 16000))
    await settle(lambda: s.conn.commits == 1)
    s.conn.emit("COMMITTED_TRANSCRIPT", {"text": "done"})

    assert (await task).text == "done"
    assert s.conn.closed == 1
    assert not s.stt.is_live


async def test_transcribe_without_a_session_is_legible():
    s = Session()
    with pytest.raises(StreamingSTTUnavailableError, match="push_to_talk"):
        await s.stt.transcribe(b"\x00" * 1024, 16000)


# ---------------------------------------------------------------------------
# Registry and protocol conformance
# ---------------------------------------------------------------------------


def test_registered_under_its_own_provider_name():
    assert "elevenlabs_realtime" in stt_mod._STT_PROVIDERS
    key_field, build = stt_mod._STT_PROVIDERS["elevenlabs_realtime"]
    assert key_field == "elevenlabs"


def test_batch_model_name_is_rejected_at_construction():
    """stt.model defaults to scribe_v2, which realtime rejects."""
    with pytest.raises(ValueError, match="scribe_v2_realtime"):
        ScribeRealtimeSTT(api_key="k", model="scribe_v2", connect=lambda lang: None)


def test_non_16k_sample_rate_is_rejected():
    s = Session()
    with pytest.raises(ValueError, match="16000"):
        s.stt.open(44100)


def test_satisfies_both_protocols():
    s = Session()
    assert isinstance(s.stt, StreamingSTTProvider)
    assert isinstance(s.stt, STTProvider)


async def test_close_event_arrives_without_a_payload():
    """``_emit(CLOSE)`` passes no args; the sink must survive that."""
    s = Session()
    s.stt.open(16000)
    s.release()
    s.stt.feed(b"\x99" * 1024)
    await settle(lambda: s.stt._conn is not None)

    for cb in s.conn.handlers.get("CLOSE", []):
        cb()  # no payload, exactly as the SDK does it

    with pytest.raises(StreamingSTTUnavailableError, match="CLOSE"):
        await s.stt.harvest(timeout=1.0)
    assert s.stt.buffered_audio == b"\x99" * 1024
    await s.stt.close()


async def test_one_client_is_built_per_provider_not_per_press(monkeypatch):
    """A press used to construct (and leak) a fresh AsyncElevenLabs."""
    built = []

    class _Client:
        def __init__(self, api_key=None):
            built.append(api_key)

    monkeypatch.setattr(stt_mod, "AsyncElevenLabs", _Client)
    stt = ScribeRealtimeSTT(api_key="k", model="scribe_v2_realtime")

    assert built == ["k"]
    assert stt._client is not None
