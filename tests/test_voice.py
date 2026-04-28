"""Tests for the voice pipeline — wake word, VAD, audio capture, STT, TTS, diarization, barge-in, voice session."""

from __future__ import annotations

import asyncio
import io
import struct
import sys
import time
import wave
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import numpy as np
import pytest
import pytest_asyncio

from boxbot.core.events import (
    AgentSpeaking,
    AgentSpeakingDone,
    TranscriptReady,
    WakeWordHeard,
    get_event_bus,
)
from boxbot.hardware.base import AudioChunk, HealthStatus

# ---------------------------------------------------------------------------
# Pre-mock unavailable third-party modules so imports succeed on dev machines
# ---------------------------------------------------------------------------

_VOICE_MOCK_MODULES = [
    "sounddevice",
    "usb",
    "usb.core",
    "usb.util",
    "openwakeword",
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.hub",
    "pyannote",
    "pyannote.audio",
    "elevenlabs",
]

for _mod_name in _VOICE_MOCK_MODULES:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chunk(
    frames: int = 1024,
    sample_rate: int = 16000,
    timestamp: float | None = None,
) -> AudioChunk:
    """Create a test AudioChunk with zeroed PCM data."""
    data = np.zeros(frames, dtype=np.int16).tobytes()
    return AudioChunk(
        data=data,
        timestamp=timestamp if timestamp is not None else time.monotonic(),
        sample_rate=sample_rate,
        channels=1,
        frames=frames,
    )


def make_speech_chunk(
    frames: int = 1024,
    sample_rate: int = 16000,
    amplitude: int = 10000,
    timestamp: float | None = None,
) -> AudioChunk:
    """Create an AudioChunk with non-silent audio (simulating speech)."""
    data = (np.ones(frames, dtype=np.int16) * amplitude).tobytes()
    return AudioChunk(
        data=data,
        timestamp=timestamp if timestamp is not None else time.monotonic(),
        sample_rate=sample_rate,
        channels=1,
        frames=frames,
    )


# ---------------------------------------------------------------------------
# TestAudioChunk
# ---------------------------------------------------------------------------


class TestAudioChunk:
    """Basic AudioChunk dataclass creation and field access."""

    def test_create_audio_chunk(self):
        data = b"\x00" * 2048
        chunk = AudioChunk(
            data=data, timestamp=1.0, sample_rate=16000, channels=1, frames=1024
        )
        assert chunk.data == data
        assert chunk.timestamp == 1.0
        assert chunk.sample_rate == 16000
        assert chunk.channels == 1
        assert chunk.frames == 1024

    def test_make_chunk_helper(self):
        chunk = make_chunk(frames=512, sample_rate=16000)
        assert len(chunk.data) == 512 * 2  # int16 = 2 bytes per frame
        assert chunk.sample_rate == 16000
        assert chunk.channels == 1
        assert chunk.frames == 512


# ---------------------------------------------------------------------------
# TestMicrophone
# ---------------------------------------------------------------------------


class TestMicrophone:
    """Tests for the Microphone HAL module with mocked sounddevice + pyusb."""

    def _make_mic(self, **kwargs):
        from boxbot.hardware.microphone import Microphone

        defaults = dict(
            device_name="TestRespeaker",
            sample_rate=16000,
            capture_channels=6,
            output_channel=0,
            chunk_duration_ms=64,
            doa_enabled=True,
            led_brightness=0.5,
        )
        defaults.update(kwargs)
        return Microphone(**defaults)

    def _make_mic_from_config(self):
        from boxbot.core.config import HardwareMicrophoneConfig
        from boxbot.hardware.microphone import Microphone

        config = HardwareMicrophoneConfig(device_name="TestRespeaker")
        return Microphone(config=config)

    def test_construction_with_kwargs(self):
        mic = self._make_mic(sample_rate=44100, led_brightness=0.8)
        assert mic._sample_rate == 44100
        assert mic._led_brightness == 0.8
        assert mic.name == "microphone"

    def test_construction_with_config(self):
        mic = self._make_mic_from_config()
        assert mic._device_name == "TestRespeaker"
        assert mic._sample_rate == 16000

    @pytest.mark.asyncio
    @patch("boxbot.hardware.microphone.sd")
    @patch("boxbot.hardware.microphone._HAS_SOUNDDEVICE", True)
    async def test_start_opens_stream_and_initializes_usb(self, mock_sd):
        mic = self._make_mic()
        # Mock device discovery
        mock_sd.query_devices.return_value = [
            {"name": "TestRespeaker Array", "max_input_channels": 6}
        ]
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        await mic.start()

        assert mic._started is True
        mock_sd.InputStream.assert_called_once()
        mock_stream.start.assert_called_once()

        await mic.stop()

    @pytest.mark.asyncio
    @patch("boxbot.hardware.microphone.sd")
    @patch("boxbot.hardware.microphone._HAS_SOUNDDEVICE", True)
    async def test_stop_closes_stream_and_releases_resources(self, mock_sd):
        mic = self._make_mic()
        mock_sd.query_devices.return_value = [
            {"name": "TestRespeaker Array", "max_input_channels": 6}
        ]
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        await mic.start()
        await mic.stop()

        assert mic._started is False
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        assert mic._stream is None

    def test_add_consumer_returns_handle(self):
        mic = self._make_mic()
        cb = AsyncMock()
        handle = mic.add_consumer(cb, "test")
        assert isinstance(handle, int) and handle > 0
        assert mic.consumer_count == 1

    def test_add_consumer_same_callback_gets_distinct_handles(self):
        # Bound methods accessed twice are not `is`-identical; the API
        # MUST issue a new handle each time even for the same callable.
        mic = self._make_mic()
        cb = AsyncMock()
        h1 = mic.add_consumer(cb, "test")
        h2 = mic.add_consumer(cb, "test")
        assert h1 != h2
        assert mic.consumer_count == 2

    def test_remove_consumer_by_handle(self):
        mic = self._make_mic()
        cb = AsyncMock()
        handle = mic.add_consumer(cb, "test")
        assert mic.remove_consumer(handle) is True
        assert mic.consumer_count == 0

    def test_remove_consumer_unknown_handle_returns_false(self):
        mic = self._make_mic()
        assert mic.remove_consumer(99999) is False
        assert mic.consumer_count == 0

    def test_remove_consumer_is_idempotent_per_handle(self):
        mic = self._make_mic()
        cb = AsyncMock()
        handle = mic.add_consumer(cb, "test")
        assert mic.remove_consumer(handle) is True
        # Second remove with the same handle is a no-op (returns False).
        assert mic.remove_consumer(handle) is False

    def test_bound_method_survives_register_remove_cycle(self):
        # Regression for the stuck-state bug observed 2026-04-24: bound
        # methods like self._on_audio_chunk are not identity-stable
        # across accesses. Handle-based removal must still work.
        class _Consumer:
            async def on_chunk(self, chunk):  # pragma: no cover - smoke
                pass

        mic = self._make_mic()
        c = _Consumer()
        # Each access of c.on_chunk yields a fresh bound method, so `is`
        # would have failed. Handles must not rely on identity.
        handle = mic.add_consumer(c.on_chunk, "consumer")
        assert mic.remove_consumer(handle) is True
        assert mic.consumer_count == 0

    @pytest.mark.asyncio
    async def test_dispatch_chunk_delivers_to_all_consumers(self):
        mic = self._make_mic()
        cb1 = AsyncMock()
        cb2 = AsyncMock()
        mic.add_consumer(cb1, "c1")
        mic.add_consumer(cb2, "c2")

        chunk = make_chunk()
        await mic._dispatch_chunk(chunk)

        cb1.assert_awaited_once_with(chunk)
        cb2.assert_awaited_once_with(chunk)

    @pytest.mark.asyncio
    async def test_dispatch_chunk_error_in_one_does_not_block_others(self):
        mic = self._make_mic()
        cb1 = AsyncMock(side_effect=RuntimeError("boom"))
        cb2 = AsyncMock()
        mic.add_consumer(cb1, "bad")
        mic.add_consumer(cb2, "good")

        chunk = make_chunk()
        await mic._dispatch_chunk(chunk)

        cb2.assert_awaited_once_with(chunk)

    @pytest.mark.asyncio
    async def test_watchdog_loop_detects_stall_and_reopens(self):
        """Regression: sounddevice can silently stall with no exception;
        the watchdog must detect the stall and force a reopen."""
        import boxbot.hardware.microphone as mic_mod

        with patch.object(mic_mod, "sd") as mock_sd, \
                patch.object(mic_mod, "_HAS_SOUNDDEVICE", True):
            mock_sd.query_devices.return_value = [
                {"name": "TestRespeaker Array", "max_input_channels": 6}
            ]
            streams = []
            def _make_stream(*a, **k):
                s = MagicMock()
                streams.append(s)
                return s
            mock_sd.InputStream.side_effect = _make_stream

            mic = self._make_mic()
            # Simulate a started state without spinning up the real
            # watchdog — we drive one iteration of the loop by hand.
            mic._loop = asyncio.get_event_loop()
            mic._device_index = 0
            mic._started = True
            mic._open_stream()
            # Force staleness.
            mic._last_chunk_monotonic = time.monotonic() - 60.0

            # Shorten poll interval by patching asyncio.sleep.
            sleeps = []
            orig_sleep = asyncio.sleep

            async def _fast_sleep(seconds):
                sleeps.append(seconds)
                if len(sleeps) > 1:
                    # After the watchdog has gone once through the
                    # stall-handling path, stop the loop.
                    mic._started = False
                await orig_sleep(0)

            with patch.object(mic_mod.asyncio, "sleep", _fast_sleep):
                await mic._watchdog_loop()

            # Original + restart: at least two InputStreams were built.
            assert mock_sd.InputStream.call_count >= 2
            streams[0].stop.assert_called()
            streams[0].close.assert_called()
            streams[-1].start.assert_called()
            assert mic._stream_restart_count >= 1

    @pytest.mark.asyncio
    async def test_set_led_pattern_changes_current_pattern(self):
        mic = self._make_mic()
        await mic.set_led_pattern("listening")
        assert mic._current_pattern == "listening"
        await mic.set_led_pattern("thinking")
        assert mic._current_pattern == "thinking"

    @pytest.mark.asyncio
    async def test_set_led_pattern_unknown_ignored(self):
        mic = self._make_mic()
        await mic.set_led_pattern("nonexistent")
        # Should stay at default "off"
        assert mic._current_pattern == "off"

    def test_get_doa_returns_none_when_usb_not_available(self):
        mic = self._make_mic()
        # No USB device initialized
        assert mic.get_doa() is None

    def test_get_doa_returns_none_when_disabled(self):
        mic = self._make_mic(doa_enabled=False)
        assert mic.get_doa() is None

    def test_get_doa_reads_from_usb(self):
        mic = self._make_mic()
        mock_usb = MagicMock()
        # The pixel_ring parameter protocol returns int32 LE value +
        # int32 LE max (8 bytes total); get_doa() returns the first.
        mock_usb.ctrl_transfer.return_value = struct.pack("<ii", 90, 360)
        mic._usb_device = mock_usb
        assert mic.get_doa() == 90

    @pytest.mark.asyncio
    async def test_health_check_stopped(self):
        mic = self._make_mic()
        assert await mic.health_check() == HealthStatus.STOPPED

    @pytest.mark.asyncio
    async def test_health_check_ok_when_running(self):
        mic = self._make_mic()
        mic._started = True
        mock_stream = MagicMock()
        mock_stream.active = True
        mic._stream = mock_stream
        mic._usb_device = MagicMock()  # USB is present
        assert await mic.health_check() == HealthStatus.OK

    @pytest.mark.asyncio
    async def test_health_check_error_when_stream_inactive(self):
        mic = self._make_mic()
        mic._started = True
        mock_stream = MagicMock()
        mock_stream.active = False
        mic._stream = mock_stream
        assert await mic.health_check() == HealthStatus.ERROR

    @patch("boxbot.hardware.microphone._HAS_SOUNDDEVICE", False)
    def test_is_available_false_without_sounddevice(self):
        mic = self._make_mic()
        assert mic.is_available is False


# ---------------------------------------------------------------------------
# TestSpeaker
# ---------------------------------------------------------------------------


class TestSpeaker:
    """Tests for the Speaker HAL module with mocked sounddevice."""

    def _make_speaker(self, **kwargs):
        from boxbot.hardware.speaker import Speaker

        defaults = dict(
            device_name="test_speaker",
            sample_rate=24000,
            default_volume=0.7,
        )
        defaults.update(kwargs)
        return Speaker(**defaults)

    def _make_speaker_from_config(self):
        from boxbot.core.config import HardwareSpeakerConfig
        from boxbot.hardware.speaker import Speaker

        config = HardwareSpeakerConfig(device_name="test_speaker")
        return Speaker(config=config)

    def test_construction_with_kwargs(self):
        spk = self._make_speaker(default_volume=0.5)
        assert spk._volume == 0.5
        assert spk.name == "speaker"

    def test_construction_with_config(self):
        spk = self._make_speaker_from_config()
        assert spk._device_name == "test_speaker"

    def test_volume_clamped(self):
        spk = self._make_speaker(default_volume=1.5)
        assert spk._volume == 1.0
        spk2 = self._make_speaker(default_volume=-0.5)
        assert spk2._volume == 0.0

    @pytest.mark.asyncio
    @patch("boxbot.hardware.speaker.sd")
    @patch("boxbot.hardware.speaker._SD_AVAILABLE", True)
    async def test_start_finds_device(self, mock_sd):
        spk = self._make_speaker()
        mock_sd.query_devices.return_value = [
            {"name": "test_speaker hdmi", "max_output_channels": 2}
        ]

        await spk.start()
        assert spk._started is True
        assert spk._device_index is not None

        await spk.stop()

    @pytest.mark.asyncio
    @patch("boxbot.hardware.speaker.sd")
    @patch("boxbot.hardware.speaker._SD_AVAILABLE", True)
    async def test_stop_resets_state(self, mock_sd):
        spk = self._make_speaker()
        mock_sd.query_devices.return_value = [
            {"name": "test_speaker hdmi", "max_output_channels": 2}
        ]

        await spk.start()
        await spk.stop()
        assert spk._started is False
        assert spk._device_index is None

    @pytest.mark.asyncio
    @patch("boxbot.hardware.speaker.sd")
    @patch("boxbot.hardware.speaker._SD_AVAILABLE", True)
    async def test_play_with_volume_scaling(self, mock_sd):
        spk = self._make_speaker(default_volume=0.5)
        mock_sd.query_devices.return_value = [
            {"name": "test_speaker hdmi", "max_output_channels": 2}
        ]
        mock_sd.play = MagicMock()
        mock_stream_obj = MagicMock()
        mock_stream_obj.active = False
        mock_sd.get_stream.return_value = mock_stream_obj

        await spk.start()
        audio = np.ones(1000, dtype=np.int16).tobytes()
        await spk.play(audio, sample_rate=24000)

        mock_sd.play.assert_called_once()
        # Verify volume was applied — scaled samples should be different
        played_data = mock_sd.play.call_args[0][0]
        assert np.all(played_data <= 1)  # 1 * 0.5 = 0

    @pytest.mark.asyncio
    async def test_stop_playback_sets_stop_event(self):
        spk = self._make_speaker()
        spk._started = True
        with patch("boxbot.hardware.speaker._SD_AVAILABLE", True), \
             patch("boxbot.hardware.speaker.sd") as mock_sd:
            await spk.stop_playback()
            assert spk._stop_event.is_set()

    def test_set_volume(self):
        spk = self._make_speaker()
        spk.set_volume(0.3)
        assert spk.get_volume() == pytest.approx(0.3)

    def test_set_volume_clamped(self):
        spk = self._make_speaker()
        spk.set_volume(1.5)
        assert spk.get_volume() == 1.0
        spk.set_volume(-0.1)
        assert spk.get_volume() == 0.0

    def test_get_volume(self):
        spk = self._make_speaker(default_volume=0.7)
        assert spk.get_volume() == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_fade_volume_instant_when_zero_duration(self):
        spk = self._make_speaker(default_volume=0.7)
        await spk.fade_volume(0.3, 0)
        assert spk.get_volume() == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_fade_volume_smooth_interpolation(self):
        spk = self._make_speaker(default_volume=1.0)
        # Use a short duration to keep test fast
        await spk.fade_volume(0.0, 50)
        assert spk.get_volume() == pytest.approx(0.0, abs=0.02)

    def test_is_playing_property(self):
        spk = self._make_speaker()
        assert spk.is_playing is False
        spk._playing = True
        assert spk.is_playing is True

    @patch("boxbot.hardware.speaker._SD_AVAILABLE", False)
    def test_is_available_false_without_sounddevice(self):
        spk = self._make_speaker()
        assert spk.is_available is False

    @pytest.mark.asyncio
    async def test_health_check_stopped(self):
        spk = self._make_speaker()
        assert await spk.health_check() == HealthStatus.STOPPED

    def test_apply_volume_at_full(self):
        spk = self._make_speaker(default_volume=1.0)
        samples = np.array([100, 200, 300], dtype=np.int16)
        result = spk._apply_volume(samples)
        np.testing.assert_array_equal(result, samples)

    def test_apply_volume_at_zero(self):
        spk = self._make_speaker(default_volume=0.0)
        samples = np.array([100, 200, 300], dtype=np.int16)
        result = spk._apply_volume(samples)
        np.testing.assert_array_equal(result, np.zeros(3, dtype=np.int16))

    def test_apply_volume_at_half(self):
        spk = self._make_speaker(default_volume=0.5)
        samples = np.array([1000, 2000, -1000], dtype=np.int16)
        result = spk._apply_volume(samples)
        assert result[0] == 500
        assert result[1] == 1000
        assert result[2] == -500


# ---------------------------------------------------------------------------
# TestWakeWordDetector
# ---------------------------------------------------------------------------


class TestWakeWordDetector:
    """Tests for wake word detection with mocked openwakeword."""

    def _make_config(self, **kwargs):
        from boxbot.core.config import WakeWordConfig

        defaults = dict(
            engine="openwakeword",
            word="hey_jarvis",
            confidence_threshold=0.7,
            model_path=None,
        )
        defaults.update(kwargs)
        return WakeWordConfig(**defaults)

    @pytest.mark.asyncio
    @patch("boxbot.communication.wake_word._resolve_builtin_model",
           return_value="/fake/hey_jarvis_v0.1.onnx")
    @patch("boxbot.communication.wake_word.openwakeword")
    async def test_start_loads_model_and_registers_consumer(
        self, mock_oww, _mock_resolve
    ):
        from boxbot.communication.wake_word import WakeWordDetector

        config = self._make_config()
        detector = WakeWordDetector(config)
        mic = MagicMock()

        await detector.start(mic)

        mock_oww.Model.assert_called_once_with(
            wakeword_model_paths=["/fake/hey_jarvis_v0.1.onnx"]
        )
        mic.add_consumer.assert_called_once()

    @pytest.mark.asyncio
    @patch("boxbot.communication.wake_word._resolve_builtin_model",
           return_value="/fake/hey_jarvis_v0.1.onnx")
    @patch("boxbot.communication.wake_word.openwakeword")
    async def test_stop_unregisters_consumer(self, mock_oww, _mock_resolve):
        from boxbot.communication.wake_word import WakeWordDetector

        config = self._make_config()
        detector = WakeWordDetector(config)
        mic = MagicMock()

        await detector.start(mic)
        await detector.stop()

        mic.remove_consumer.assert_called_once()
        assert detector._model is None

    @pytest.mark.asyncio
    @patch("boxbot.communication.wake_word._resolve_builtin_model",
           return_value="/fake/hey_jarvis_v0.1.onnx")
    @patch("boxbot.communication.wake_word.openwakeword")
    async def test_on_audio_chunk_publishes_when_above_threshold(
        self, mock_oww, _mock_resolve
    ):
        from boxbot.communication.wake_word import WakeWordDetector

        config = self._make_config(confidence_threshold=0.7)
        detector = WakeWordDetector(config)
        mic = MagicMock()
        await detector.start(mic)

        # Configure model to return high confidence
        detector._model.predict.return_value = {"hey_jarvis": 0.95}

        events_received: list = []
        bus = get_event_bus()

        async def handler(event: WakeWordHeard):
            events_received.append(event)

        bus.subscribe(WakeWordHeard, handler)

        chunk = make_chunk()
        await detector._on_audio_chunk(chunk)

        assert len(events_received) == 1
        assert events_received[0].confidence == 0.95
        detector._model.reset.assert_called_once()

    @pytest.mark.asyncio
    @patch("boxbot.communication.wake_word._resolve_builtin_model",
           return_value="/fake/hey_jarvis_v0.1.onnx")
    @patch("boxbot.communication.wake_word.openwakeword")
    async def test_debounce_suppresses_immediate_repeat_detection(
        self, mock_oww, _mock_resolve
    ):
        """Regression: one spoken wake word can produce 2–3 detections in
        under a second as the model's prediction decays. Debounce must
        suppress further publishes within the debounce window."""
        from boxbot.communication.wake_word import WakeWordDetector

        config = self._make_config(confidence_threshold=0.7)
        detector = WakeWordDetector(config)
        mic = MagicMock()
        await detector.start(mic)

        # First detection above threshold.
        detector._model.predict.return_value = {"hey_jarvis": 0.95}

        events_received: list = []
        bus = get_event_bus()
        async def handler(event: WakeWordHeard):
            events_received.append(event)
        bus.subscribe(WakeWordHeard, handler)

        # Three quick detections — only the first should publish.
        await detector._on_audio_chunk(make_chunk())
        detector._model.predict.return_value = {"hey_jarvis": 0.86}
        await detector._on_audio_chunk(make_chunk())
        detector._model.predict.return_value = {"hey_jarvis": 0.72}
        await detector._on_audio_chunk(make_chunk())

        assert len(events_received) == 1
        assert events_received[0].confidence == 0.95

        # After the debounce window, the detector is armed again.
        detector._suppress_until = 0.0
        detector._model.predict.return_value = {"hey_jarvis": 0.9}
        await detector._on_audio_chunk(make_chunk())
        assert len(events_received) == 2

    @pytest.mark.asyncio
    @patch("boxbot.communication.wake_word._resolve_builtin_model",
           return_value="/fake/hey_jarvis_v0.1.onnx")
    @patch("boxbot.communication.wake_word.openwakeword")
    async def test_on_audio_chunk_does_not_publish_below_threshold(
        self, mock_oww, _mock_resolve
    ):
        from boxbot.communication.wake_word import WakeWordDetector

        config = self._make_config(confidence_threshold=0.7)
        detector = WakeWordDetector(config)
        mic = MagicMock()
        await detector.start(mic)

        # Configure model to return low confidence
        detector._model.predict.return_value = {"hey_jarvis": 0.3}

        events_received: list = []
        bus = get_event_bus()
        bus.subscribe(WakeWordHeard, lambda e: events_received.append(e))

        chunk = make_chunk()
        await detector._on_audio_chunk(chunk)

        assert len(events_received) == 0

    @pytest.mark.asyncio
    @patch("boxbot.communication.wake_word.openwakeword")
    async def test_uses_custom_model_path(self, mock_oww):
        from boxbot.communication.wake_word import WakeWordDetector

        config = self._make_config(model_path="/path/to/custom.onnx")
        detector = WakeWordDetector(config)
        mic = MagicMock()

        await detector.start(mic)

        mock_oww.Model.assert_called_once_with(
            wakeword_model_paths=["/path/to/custom.onnx"]
        )


# ---------------------------------------------------------------------------
# TestVoiceActivityDetector
# ---------------------------------------------------------------------------


class TestVoiceActivityDetector:
    """Tests for VAD with mocked torch."""

    def _make_config(self, **kwargs):
        from boxbot.core.config import VADConfig

        defaults = dict(threshold=0.5, min_speech_duration=250, min_silence_duration=100)
        defaults.update(kwargs)
        return VADConfig(**defaults)

    @pytest.mark.asyncio
    @patch("boxbot.communication.vad.torch")
    async def test_start_loads_model(self, mock_torch):
        from boxbot.communication.vad import VoiceActivityDetector

        mock_model = MagicMock()
        mock_torch.hub.load.return_value = (mock_model, None)

        config = self._make_config()
        vad = VoiceActivityDetector(config)
        await vad.start()

        mock_torch.hub.load.assert_called_once_with(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        assert vad._model is mock_model

    @pytest.mark.asyncio
    async def test_process_chunk_returns_zero_when_model_not_loaded(self):
        from boxbot.communication.vad import VoiceActivityDetector

        config = self._make_config()
        vad = VoiceActivityDetector(config)
        # Model is None (not loaded)
        result = await vad.process_chunk(make_chunk())
        assert result == 0.0

    @pytest.mark.asyncio
    @patch("boxbot.communication.vad.torch")
    async def test_process_chunk_returns_speech_probability(self, mock_torch):
        from boxbot.communication.vad import VoiceActivityDetector

        mock_model = MagicMock()
        mock_torch.hub.load.return_value = (mock_model, None)
        # Mock the model call: return a tensor-like with .item() = 0.8
        mock_result = MagicMock()
        mock_result.item.return_value = 0.8
        mock_model.return_value = mock_result
        # Mock torch operations
        mock_torch.from_numpy.return_value = MagicMock(__len__=lambda s: 1024)
        tensor_mock = MagicMock()
        tensor_mock.__len__ = lambda s: 1024
        tensor_mock.__getitem__ = lambda s, k: tensor_mock
        mock_torch.from_numpy.return_value = tensor_mock
        mock_torch.nn.functional.pad.return_value = tensor_mock

        config = self._make_config()
        vad = VoiceActivityDetector(config)
        await vad.start()

        result = await vad.process_chunk(make_chunk())
        assert result == 0.8

    def test_reset_calls_model_reset_states(self):
        from boxbot.communication.vad import VoiceActivityDetector

        config = self._make_config()
        vad = VoiceActivityDetector(config)
        mock_model = MagicMock()
        vad._model = mock_model

        vad.reset()
        mock_model.reset_states.assert_called_once()


# ---------------------------------------------------------------------------
# TestAudioCapture
# ---------------------------------------------------------------------------


class TestAudioCapture:
    """Tests for audio capture with VAD-driven utterance detection."""

    def _make_config(self, **kwargs):
        from boxbot.core.config import TurnDetectionConfig

        defaults = dict(
            silence_threshold=800,
            max_utterance_duration=60,
            inter_utterance_gap=300,
        )
        defaults.update(kwargs)
        return TurnDetectionConfig(**defaults)

    def _make_vad_mock(self, speech_prob: float = 0.0):
        """Create a mock VAD that returns a fixed speech probability."""
        from boxbot.core.config import VADConfig

        vad = MagicMock()
        vad._config = VADConfig(threshold=0.5)
        vad.process_chunk = AsyncMock(return_value=speech_prob)
        return vad

    def _make_capture(self, vad=None, **config_kwargs):
        from boxbot.communication.audio_capture import AudioCapture

        if vad is None:
            vad = self._make_vad_mock()
        config = self._make_config(**config_kwargs)
        return AudioCapture(vad, config)

    @pytest.mark.asyncio
    async def test_start_registers_consumer(self):
        capture = self._make_capture()
        mic = MagicMock()
        await capture.start(mic)
        mic.add_consumer.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_unregisters_consumer(self):
        capture = self._make_capture()
        mic = MagicMock()
        await capture.start(mic)
        await capture.stop()
        mic.remove_consumer.assert_called_once()

    @pytest.mark.asyncio
    async def test_speech_accumulation_into_buffer(self):
        vad = self._make_vad_mock(speech_prob=0.8)  # above 0.5 threshold
        capture = self._make_capture(vad=vad)

        # Feed a speech chunk
        chunk = make_speech_chunk(frames=1024, timestamp=1.0)
        await capture._on_audio_chunk(chunk)

        assert capture._is_speaking is True
        assert len(capture._buffer) == len(chunk.data)

    @pytest.mark.asyncio
    async def test_silence_persistence_triggers_finalization(self):
        from boxbot.communication.audio_capture import Utterance

        utterances: list[Utterance] = []
        callback = AsyncMock(side_effect=lambda u: utterances.append(u))

        vad = self._make_vad_mock()
        capture = self._make_capture(vad=vad, silence_threshold=100)
        capture.set_utterance_callback(callback)

        # Simulate speech followed by silence
        # First: speech chunk
        vad.process_chunk.return_value = 0.8
        chunk_speech = make_speech_chunk(frames=1024, timestamp=1.0)
        await capture._on_audio_chunk(chunk_speech)

        # Then: silence chunks (each 64ms at 16kHz with 1024 frames)
        vad.process_chunk.return_value = 0.1
        chunk_duration_ms = (1024 / 16000) * 1000  # 64ms
        ts = 1.1
        # Need enough silence to exceed 100ms threshold
        for _ in range(3):
            chunk_silence = make_chunk(frames=1024, timestamp=ts)
            await capture._on_audio_chunk(chunk_silence)
            ts += 0.064

        assert len(utterances) == 1
        assert len(utterances[0].audio) > 0

    @pytest.mark.asyncio
    async def test_utterance_callback_called_with_correct_data(self):
        from boxbot.communication.audio_capture import Utterance

        utterances: list[Utterance] = []
        callback = AsyncMock(side_effect=lambda u: utterances.append(u))

        vad = self._make_vad_mock(speech_prob=0.8)
        capture = self._make_capture(vad=vad, silence_threshold=50)
        capture.set_utterance_callback(callback)

        # Speech chunk
        chunk1 = make_speech_chunk(frames=1024, timestamp=1.0)
        await capture._on_audio_chunk(chunk1)

        # Silence to trigger finalization
        vad.process_chunk.return_value = 0.1
        chunk2 = make_chunk(frames=1024, timestamp=1.1)
        await capture._on_audio_chunk(chunk2)

        assert len(utterances) == 1
        utt = utterances[0]
        assert utt.sample_rate == 16000
        assert utt.timestamp_start == 1.0

    @pytest.mark.asyncio
    async def test_max_duration_cutoff_forces_finalization(self):
        from boxbot.communication.audio_capture import Utterance

        utterances: list[Utterance] = []
        callback = AsyncMock(side_effect=lambda u: utterances.append(u))

        vad = self._make_vad_mock(speech_prob=0.8)
        # max_utterance_duration=0 => immediate force finalization after any speech
        capture = self._make_capture(vad=vad, max_utterance_duration=0)
        capture.set_utterance_callback(callback)

        chunk = make_speech_chunk(frames=1024, timestamp=1.0)
        await capture._on_audio_chunk(chunk)
        # Second chunk to trigger the max duration check (elapsed > 0)
        chunk2 = make_speech_chunk(frames=1024, timestamp=1.1)
        await capture._on_audio_chunk(chunk2)

        assert len(utterances) >= 1

    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        vad = self._make_vad_mock(speech_prob=0.8)
        capture = self._make_capture(vad=vad)

        chunk = make_speech_chunk(frames=1024, timestamp=1.0)
        await capture._on_audio_chunk(chunk)

        assert capture._is_speaking is True
        capture.reset()
        assert capture._is_speaking is False
        assert len(capture._buffer) == 0

    @pytest.mark.asyncio
    async def test_no_finalization_for_silence_only(self):
        utterances: list = []
        callback = AsyncMock(side_effect=lambda u: utterances.append(u))

        vad = self._make_vad_mock(speech_prob=0.1)  # always silence
        capture = self._make_capture(vad=vad, silence_threshold=50)
        capture.set_utterance_callback(callback)

        # Feed multiple silence chunks
        for i in range(10):
            chunk = make_chunk(frames=1024, timestamp=float(i))
            await capture._on_audio_chunk(chunk)

        assert len(utterances) == 0


# ---------------------------------------------------------------------------
# TestSTT
# ---------------------------------------------------------------------------


class TestSTT:
    """Tests for STT (speech-to-text) with mocked elevenlabs."""

    def test_pcm_to_wav_produces_valid_wav(self):
        from boxbot.communication.stt import pcm_to_wav

        pcm = np.zeros(1600, dtype=np.int16).tobytes()
        wav = pcm_to_wav(pcm, sample_rate=16000)

        # Verify WAV header (RIFF)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"

        # Parse with wave module to verify structure
        buf = io.BytesIO(wav)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 1600

    @pytest.mark.asyncio
    @patch("boxbot.communication.stt.AsyncElevenLabs")
    async def test_elevenlabs_stt_transcribe(self, mock_client_cls):
        from boxbot.communication.stt import ElevenLabsSTT

        # Mock the client and its response
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_result = MagicMock()
        mock_result.text = "Hello world"
        mock_result.words = [
            MagicMock(text="Hello", start=0.0, end=0.5, confidence=0.95),
            MagicMock(text="world", start=0.5, end=1.0, confidence=0.90),
        ]
        mock_result.language_code = "en"
        mock_client.speech_to_text.convert = AsyncMock(return_value=mock_result)

        stt = ElevenLabsSTT(api_key="test-key", model="scribe_v2")
        pcm = np.zeros(16000, dtype=np.int16).tobytes()  # 1 second
        result = await stt.transcribe(pcm, sample_rate=16000)

        assert result.text == "Hello world"
        assert len(result.words) == 2
        assert result.words[0].word == "Hello"
        assert result.words[1].confidence == 0.90
        mock_client.speech_to_text.convert.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("boxbot.communication.stt.AsyncElevenLabs")
    async def test_stt_result_parsing(self, mock_client_cls):
        from boxbot.communication.stt import ElevenLabsSTT, STTResult, WordInfo

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_result = MagicMock()
        mock_result.text = "Testing"
        mock_result.words = []
        mock_result.language_code = "en"
        mock_client.speech_to_text.convert = AsyncMock(return_value=mock_result)

        stt = ElevenLabsSTT(api_key="test-key")
        result = await stt.transcribe(b"\x00" * 3200, sample_rate=16000)

        assert isinstance(result, STTResult)
        assert result.text == "Testing"
        assert result.language == "en"
        assert result.words == []


# ---------------------------------------------------------------------------
# TestTTS
# ---------------------------------------------------------------------------


class TestTTS:
    """Tests for TTS (text-to-speech) with mocked elevenlabs."""

    @pytest.mark.asyncio
    @patch("boxbot.communication.tts.AsyncElevenLabs")
    @patch("boxbot.communication.tts.VoiceSettings")
    async def test_elevenlabs_tts_synthesize(self, mock_vs, mock_client_cls):
        from boxbot.communication.tts import ElevenLabsTTS

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Simulate async iterator response
        async def mock_chunks():
            yield b"\x00" * 1000
            yield b"\x01" * 500

        mock_client.text_to_speech.convert = AsyncMock(
            return_value=mock_chunks()
        )

        tts = ElevenLabsTTS(api_key="key", voice_id="vid")
        result = await tts.synthesize("Hello")

        assert len(result) == 1500
        mock_client.text_to_speech.convert.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("boxbot.communication.tts.AsyncElevenLabs")
    @patch("boxbot.communication.tts.VoiceSettings")
    async def test_elevenlabs_tts_synthesize_stream(self, mock_vs, mock_client_cls):
        from boxbot.communication.tts import ElevenLabsTTS

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        async def mock_stream_response():
            yield b"\x00" * 500
            yield b"\x01" * 500

        # ElevenLabs SDK 4.x: stream() is a sync method returning an
        # async iterator (not a coroutine). MagicMock matches that shape.
        mock_client.text_to_speech.stream = MagicMock(
            return_value=mock_stream_response()
        )

        tts = ElevenLabsTTS(api_key="key", voice_id="vid")
        chunks = []
        async for chunk in tts.synthesize_stream("Hello"):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0] == b"\x00" * 500

    @pytest.mark.asyncio
    async def test_tts_stream_speak_calls_play_stream(self):
        from boxbot.communication.tts import TTSStream

        tts = MagicMock()
        speaker = MagicMock()
        speaker.play_stream = AsyncMock()

        async def mock_stream():
            yield b"\x00" * 100

        tts.synthesize_stream.return_value = mock_stream()
        stream = TTSStream(tts, speaker)
        await stream.speak("Hello")

        speaker.play_stream.assert_awaited_once()
        assert stream.is_playing is False  # cleaned up in finally

    @pytest.mark.asyncio
    async def test_tts_stream_stop_calls_stop_playback(self):
        from boxbot.communication.tts import TTSStream

        tts = MagicMock()
        speaker = MagicMock()
        speaker.stop_playback = AsyncMock()

        stream = TTSStream(tts, speaker)
        await stream.stop()

        speaker.stop_playback.assert_awaited_once()
        assert stream.is_playing is False


# ---------------------------------------------------------------------------
# TestDiarization
# ---------------------------------------------------------------------------


class TestDiarization:
    """Tests for speaker diarization with mocked pyannote."""

    def _make_config(self, **kwargs):
        from boxbot.core.config import DiarizationConfig

        defaults = dict(
            engine="pyannote",
            model="pyannote/speaker-diarization-3.1",
            embedding_model="pyannote/wespeaker-voxceleb-resnet34-LM",
            min_speakers=1,
            max_speakers=6,
            match_threshold=0.65,
        )
        defaults.update(kwargs)
        return DiarizationConfig(**defaults)

    @pytest.mark.asyncio
    @patch("boxbot.communication.diarization.Inference")
    @patch("boxbot.communication.diarization.Model")
    @patch("boxbot.communication.diarization.Pipeline")
    @patch("boxbot.communication.diarization.torch")
    async def test_start_loads_pipeline_and_embedding_model(
        self, mock_torch, mock_pipeline, mock_model, mock_inference
    ):
        from boxbot.communication.diarization import SpeakerDiarizer

        mock_pipeline.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()

        config = self._make_config()
        diarizer = SpeakerDiarizer(config)
        await diarizer.start()

        mock_pipeline.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_called_once()
        assert diarizer._pipeline is not None

    @pytest.mark.asyncio
    @patch("boxbot.communication.diarization.Inference")
    @patch("boxbot.communication.diarization.Model")
    @patch("boxbot.communication.diarization.Pipeline")
    @patch("boxbot.communication.diarization.torch")
    async def test_diarize_returns_segments(
        self, mock_torch, mock_pipeline, mock_model, mock_inference
    ):
        from boxbot.communication.diarization import SpeakerDiarizer

        # Set up mock pipeline
        mock_pipeline_instance = MagicMock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance
        mock_model.from_pretrained.return_value = MagicMock()

        # Mock diarization output
        mock_segment1 = MagicMock()
        mock_segment1.start = 0.0
        mock_segment1.end = 2.0
        mock_segment2 = MagicMock()
        mock_segment2.start = 2.0
        mock_segment2.end = 4.0

        mock_diarization = MagicMock()
        mock_diarization.itertracks.return_value = [
            (mock_segment1, "track1", "SPEAKER_00"),
            (mock_segment2, "track2", "SPEAKER_01"),
        ]
        # pyannote.audio 4.x returns DiarizeOutput — emulate the unwrap
        # by pointing speaker_diarization at the same mock.
        mock_diarization.speaker_diarization = mock_diarization
        mock_pipeline_instance.return_value = mock_diarization

        # Mock torch tensor creation
        mock_torch.tensor.return_value = MagicMock(
            unsqueeze=MagicMock(return_value=MagicMock())
        )

        # Mock embedding extraction
        mock_inference_instance = MagicMock()
        mock_inference_instance.return_value = np.random.randn(192).astype(np.float32)
        mock_inference.return_value = mock_inference_instance

        config = self._make_config()
        diarizer = SpeakerDiarizer(config)
        await diarizer.start()

        # Diarize 4 seconds of audio at 16kHz
        audio = np.zeros(64000, dtype=np.int16).tobytes()
        result = await diarizer.diarize(audio, sample_rate=16000)

        assert result.num_speakers == 2
        assert len(result.segments) == 2
        assert result.segments[0].speaker_label == "SPEAKER_00"
        assert result.segments[1].speaker_label == "SPEAKER_01"

    @pytest.mark.asyncio
    @patch("boxbot.communication.diarization.Inference")
    @patch("boxbot.communication.diarization.Model")
    @patch("boxbot.communication.diarization.Pipeline")
    @patch("boxbot.communication.diarization.torch")
    async def test_embedding_extraction_for_segments(
        self, mock_torch, mock_pipeline, mock_model, mock_inference
    ):
        from boxbot.communication.diarization import SpeakerDiarizer

        mock_pipeline_instance = MagicMock()
        mock_pipeline.from_pretrained.return_value = mock_pipeline_instance
        mock_model.from_pretrained.return_value = MagicMock()

        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0

        mock_diarization = MagicMock()
        mock_diarization.itertracks.return_value = [
            (mock_segment, "track1", "SPEAKER_00"),
        ]
        mock_diarization.speaker_diarization = mock_diarization
        mock_pipeline_instance.return_value = mock_diarization

        mock_torch.tensor.return_value = MagicMock(
            unsqueeze=MagicMock(return_value=MagicMock())
        )

        # Mock embedding inference to return a 192-dim vector
        mock_emb = np.random.randn(192).astype(np.float32)
        mock_inference_instance = MagicMock(return_value=mock_emb)
        mock_inference.return_value = mock_inference_instance

        config = self._make_config()
        diarizer = SpeakerDiarizer(config)
        await diarizer.start()

        audio = np.zeros(16000, dtype=np.int16).tobytes()
        result = await diarizer.diarize(audio, sample_rate=16000)

        assert len(result.segments) == 1
        assert result.segments[0].embedding is not None


# ---------------------------------------------------------------------------
# TestVoiceSession
# ---------------------------------------------------------------------------


class TestVoiceSession:
    """Tests for the voice session state machine with all components mocked."""

    def _make_config(self, **kwargs):
        from boxbot.core.config import VoiceConfig

        return VoiceConfig(**kwargs)

    def _make_session(self, config=None):
        from boxbot.communication.voice import VoiceSession

        mic = MagicMock()
        mic.add_consumer = MagicMock()
        mic.remove_consumer = MagicMock()
        mic.set_led_pattern = AsyncMock()

        speaker = MagicMock()
        speaker.is_playing = False
        speaker.stop_playback = AsyncMock()
        speaker.play_stream = AsyncMock()

        if config is None:
            config = self._make_config()

        session = VoiceSession(mic, speaker, config)
        return session, mic, speaker

    @pytest.mark.asyncio
    @patch("boxbot.communication.voice.WakeWordDetector")
    @patch("boxbot.communication.voice.VoiceActivityDetector")
    @patch("boxbot.communication.voice.ElevenLabsSTT", None)
    @patch("boxbot.communication.voice.ElevenLabsTTS", None)
    async def test_start_enters_idle_state(self, mock_vad_cls, mock_ww_cls):
        from boxbot.communication.voice import VoiceSession, VoiceSessionState

        session, mic, speaker = self._make_session()

        mock_ww = MagicMock()
        mock_ww.start = AsyncMock()
        mock_ww.stop = AsyncMock()
        mock_ww_cls.return_value = mock_ww

        mock_vad = MagicMock()
        mock_vad.start = AsyncMock()
        mock_vad.stop = AsyncMock()
        mock_vad_cls.return_value = mock_vad

        # Patch to avoid needing config singleton
        with patch("boxbot.communication.voice.get_event_bus") as mock_bus_fn:
            mock_bus = MagicMock()
            mock_bus_fn.return_value = mock_bus
            # Patch SpeakerDiarizer import to raise so it's skipped
            with patch(
                "boxbot.communication.voice.SpeakerDiarizer",
                side_effect=ImportError,
                create=True,
            ):
                with patch("boxbot.core.config.get_config") as mock_get_config:
                    mock_cfg = MagicMock()
                    mock_cfg.api_keys.elevenlabs = None
                    mock_get_config.return_value = mock_cfg
                    await session.start()

        assert session.state == VoiceSessionState.IDLE

        await session.stop()

    @pytest.mark.asyncio
    async def test_state_transitions_idle_to_active(self):
        from boxbot.communication.voice import VoiceSession, VoiceSessionState

        session, mic, speaker = self._make_session()
        session._state = VoiceSessionState.IDLE
        session._vad = MagicMock()
        session._audio_capture = MagicMock()
        session._audio_capture.start = AsyncMock()

        await session._activate_session()

        assert session.state == VoiceSessionState.ACTIVE
        assert session._conversation_id.startswith("voice_")

    @pytest.mark.asyncio
    async def test_deactivate_stops_audio_capture_and_returns_to_idle(self):
        """Adapter-model replacement for the old suspend/end flow: one
        transition, ACTIVE → IDLE, triggered by ConversationEnded or
        the post-wake-word grace timeout."""
        from boxbot.communication.voice import VoiceSession, VoiceSessionState

        session, mic, speaker = self._make_session()
        session._state = VoiceSessionState.ACTIVE
        session._audio_capture = MagicMock()
        session._audio_capture.stop = AsyncMock()
        session._audio_capture.reset = MagicMock()
        session._vad = MagicMock()
        session._conversation_id = "voice_test123"

        await session._deactivate_session(reason="test")

        assert session.state == VoiceSessionState.IDLE
        assert session._conversation_id == ""
        session._audio_capture.stop.assert_awaited_once()
        session._vad.reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_activate_session_starts_audio_capture(self):
        from boxbot.communication.voice import VoiceSession, VoiceSessionState

        session, mic, speaker = self._make_session()
        session._state = VoiceSessionState.IDLE
        session._vad = MagicMock()
        session._audio_capture = MagicMock()
        session._audio_capture.start = AsyncMock()

        await session._activate_session()

        session._audio_capture.start.assert_awaited_once_with(mic)

    @pytest.mark.asyncio
    async def test_conversation_ended_voice_deactivates_adapter(self):
        """The adapter subscribes to ConversationEnded and deactivates
        capture/LED when the room conversation ends (silence or agent-
        initiated). This replaces the old independent suspend timer."""
        from boxbot.communication.voice import VoiceSession, VoiceSessionState
        from boxbot.core.events import ConversationEnded

        session, mic, speaker = self._make_session()
        session._state = VoiceSessionState.ACTIVE
        session._conversation_id = "voice_t"
        session._audio_capture = MagicMock()
        session._audio_capture.stop = AsyncMock()
        session._audio_capture.reset = MagicMock()
        session._vad = MagicMock()

        await session._on_conversation_ended(ConversationEnded(
            conversation_id="conv_abc",
            channel="voice",
        ))

        assert session.state == VoiceSessionState.IDLE
        session._audio_capture.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_conversation_ended_non_voice_does_not_deactivate(self):
        """WhatsApp/trigger conversation ends must not touch the mic."""
        from boxbot.communication.voice import VoiceSession, VoiceSessionState
        from boxbot.core.events import ConversationEnded

        session, mic, speaker = self._make_session()
        session._state = VoiceSessionState.ACTIVE
        session._conversation_id = "voice_t"
        session._audio_capture = MagicMock()
        session._audio_capture.stop = AsyncMock()

        await session._on_conversation_ended(ConversationEnded(
            conversation_id="conv_wa",
            channel="whatsapp",
        ))

        assert session.state == VoiceSessionState.ACTIVE
        session._audio_capture.stop.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_wake_word_triggers_activation(self):
        from boxbot.communication.voice import VoiceSession, VoiceSessionState

        session, mic, speaker = self._make_session()
        session._state = VoiceSessionState.IDLE
        session._vad = MagicMock()
        session._audio_capture = MagicMock()
        session._audio_capture.start = AsyncMock()

        event = WakeWordHeard(confidence=0.9)
        await session._on_wake_word(event)

        assert session.state == VoiceSessionState.ACTIVE

    @pytest.mark.asyncio
    async def test_wake_word_during_active_extends_grace(self):
        """A second wake word while ACTIVE refreshes the post-wake-word
        grace timer and re-attaches audio_capture (idempotent in the
        underlying ``AudioCapture.start``). In ``wake_word`` barge-in
        mode the latter is essential — STT detaches during BB's reply
        and the wake word is the only path back."""
        from boxbot.communication.voice import VoiceSession, VoiceSessionState

        session, mic, speaker = self._make_session()
        session._state = VoiceSessionState.ACTIVE
        session._conversation_id = "voice_existing"
        session._audio_capture = MagicMock()
        session._audio_capture.start = AsyncMock()

        # Pre-existing grace task (any task will do for the mock).
        import asyncio as _aio
        prior = _aio.create_task(_aio.sleep(60))
        session._active_timeout_task = prior

        event = WakeWordHeard(confidence=0.85)
        await session._on_wake_word(event)

        # Give the cancelled task one scheduler tick to settle.
        try:
            await prior
        except _aio.CancelledError:
            pass

        assert session.state == VoiceSessionState.ACTIVE
        assert session._conversation_id == "voice_existing"
        assert prior.cancelled() or prior.done()
        # New grace task replaced the prior one — NOT the same object.
        assert session._active_timeout_task is not prior
        # audio_capture.start IS called (idempotent at the AudioCapture
        # level) so wake-word mode can re-attach STT after BB's reply.
        session._audio_capture.start.assert_awaited()

        # Clean up the new grace task so it doesn't leak into other tests.
        if session._active_timeout_task is not None:
            session._active_timeout_task.cancel()
            try:
                await session._active_timeout_task
            except _aio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_wake_word_ignored_when_already_active(self):
        from boxbot.communication.voice import VoiceSession, VoiceSessionState

        session, mic, speaker = self._make_session()
        session._state = VoiceSessionState.ACTIVE
        session._conversation_id = "voice_already"
        session._vad = MagicMock()
        session._audio_capture = MagicMock()
        session._audio_capture.start = AsyncMock()

        event = WakeWordHeard(confidence=0.9)
        await session._on_wake_word(event)

        # Should remain active with same conversation
        assert session.state == VoiceSessionState.ACTIVE
        assert session._conversation_id == "voice_already"

    @pytest.mark.asyncio
    async def test_on_utterance_runs_stt(self):
        from boxbot.communication.audio_capture import Utterance
        from boxbot.communication.stt import STTResult, WordInfo
        from boxbot.communication.voice import VoiceSession, VoiceSessionState

        session, mic, speaker = self._make_session()
        session._state = VoiceSessionState.ACTIVE
        session._conversation_id = "voice_test"

        # Mock STT
        mock_stt = MagicMock()
        stt_result = STTResult(
            text="Hello boxbot",
            language="en",
            words=[
                WordInfo(word="Hello", start=0.0, end=0.5),
                WordInfo(word="boxbot", start=0.5, end=1.0),
            ],
        )
        mock_stt.transcribe = AsyncMock(return_value=stt_result)
        session._stt = mock_stt
        session._diarizer = None

        events_received: list = []
        bus = get_event_bus()
        bus.subscribe(TranscriptReady, lambda e: events_received.append(e))

        utterance = Utterance(
            audio=b"\x00" * 32000,
            duration=1.0,
            sample_rate=16000,
            timestamp_start=1.0,
            timestamp_end=2.0,
        )
        await session._on_utterance(utterance)

        mock_stt.transcribe.assert_awaited_once()
        assert len(events_received) == 1
        assert events_received[0].transcript == "Hello boxbot"

    @pytest.mark.asyncio
    async def test_speak_detaches_audio_capture_during_tts(self):
        """``speak()`` detaches STT during TTS so chatter and echo
        cannot reach the transcript pipeline. After TTS completes
        naturally, STT is re-attached so the user can continue without
        re-saying the wake word; only mid-TTS interruption requires
        the wake word."""
        from boxbot.communication.voice import VoiceSession, VoiceSessionState

        session, mic, speaker = self._make_session()
        session._state = VoiceSessionState.ACTIVE
        session._conversation_id = "voice_test"

        # Mock audio_capture and TTS
        mock_capture = MagicMock()
        mock_capture.stop = AsyncMock()
        mock_capture.start = AsyncMock()
        session._audio_capture = mock_capture

        mock_tts_stream = MagicMock()
        mock_tts_stream.speak = AsyncMock()
        session._tts_stream = mock_tts_stream

        await session.speak("Hello there")

        mock_tts_stream.speak.assert_awaited_once_with("Hello there")
        mock_capture.stop.assert_awaited_once()
        # STT is re-attached on natural completion so the user can
        # continue the conversation.
        mock_capture.start.assert_awaited_once_with(mic)

    @pytest.mark.asyncio
    async def test_wake_word_during_tts_stops_playback(self):
        """In wake_word mode, a wake-word detection mid-TTS interrupts
        playback, marks the turn as interrupted, and re-attaches STT."""
        from boxbot.communication.voice import VoiceSession, VoiceSessionState

        session, mic, speaker = self._make_session()
        session._state = VoiceSessionState.ACTIVE
        session._conversation_id = "voice_speaking"
        session._audio_capture = MagicMock()
        session._audio_capture.start = AsyncMock()

        # Speaker reports it's currently playing a TTS chunk.
        speaker.is_playing = True
        speaker.stop_playback = AsyncMock()

        events: list = []

        async def _capture_speaking_done(event):
            events.append(event)

        bus = get_event_bus()
        from boxbot.core.events import AgentSpeakingDone
        bus.subscribe(AgentSpeakingDone, _capture_speaking_done)
        try:
            await session._on_wake_word(WakeWordHeard(confidence=0.9))
        finally:
            bus.unsubscribe(AgentSpeakingDone, _capture_speaking_done)

        speaker.stop_playback.assert_awaited_once()
        assert session._tts_interrupted is True
        # STT re-attached so the user's next utterance is captured.
        session._audio_capture.start.assert_awaited()
        assert any(e.interrupted for e in events), (
            f"expected an interrupted=True AgentSpeakingDone, got {events!r}"
        )

    @pytest.mark.asyncio
    async def test_initiate_conversation_activates_and_speaks(self):
        from boxbot.communication.voice import VoiceSession, VoiceSessionState

        session, mic, speaker = self._make_session()
        session._state = VoiceSessionState.IDLE
        session._vad = MagicMock()
        session._audio_capture = MagicMock()
        session._audio_capture.start = AsyncMock()
        # speak() detaches audio_capture for the duration of TTS
        # playback and re-attaches it when TTS completes.
        session._audio_capture.stop = AsyncMock()

        # Mock TTS stream
        mock_tts_stream = MagicMock()
        mock_tts_stream.speak = AsyncMock()
        session._tts_stream = mock_tts_stream

        await session.initiate_conversation("Hello Jacob!", person_name="Jacob")

        assert session.state == VoiceSessionState.ACTIVE
        mock_tts_stream.speak.assert_awaited_once_with("Hello Jacob!")

    @pytest.mark.asyncio
    async def test_initiate_conversation_noop_when_not_idle(self):
        from boxbot.communication.voice import VoiceSession, VoiceSessionState

        session, mic, speaker = self._make_session()
        session._state = VoiceSessionState.ACTIVE

        # Should not re-activate
        mock_tts_stream = MagicMock()
        mock_tts_stream.speak = AsyncMock()
        session._tts_stream = mock_tts_stream

        await session.initiate_conversation("Hello")
        mock_tts_stream.speak.assert_not_awaited()

    def test_get_and_set_voice_session_singleton(self):
        from boxbot.communication.voice import (
            get_voice_session,
            set_voice_session,
        )

        original = get_voice_session()

        session, _, _ = self._make_session()
        set_voice_session(session)
        assert get_voice_session() is session

        set_voice_session(None)
        assert get_voice_session() is None

        # Restore original
        set_voice_session(original)

    @pytest.mark.asyncio
    async def test_wake_word_grace_deactivates_if_no_utterance(self):
        """If the user says the wake word but never speaks, the adapter
        must deactivate after the grace window so we don't keep the mic
        hot forever."""
        from boxbot.communication.voice import VoiceSession, VoiceSessionState

        session, mic, speaker = self._make_session()
        session._state = VoiceSessionState.ACTIVE
        session._conversation_id = "voice_t"
        session._audio_capture = MagicMock()
        session._audio_capture.stop = AsyncMock()
        session._audio_capture.reset = MagicMock()
        session._vad = MagicMock()

        # Shorten the grace window for the test.
        session._WAKE_WORD_GRACE_SECONDS = 0.05
        # Drive the loop directly rather than via the timer scheduler.
        await session._wake_word_grace_loop()

        assert session.state == VoiceSessionState.IDLE
        session._audio_capture.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_utterance_cancels_wake_word_grace(self):
        """Once a real utterance arrives the Conversation's silence
        timeout takes over lifecycle — the adapter-side grace is
        cancelled so it doesn't race the conversation."""
        from boxbot.communication.audio_capture import Utterance
        from boxbot.communication.stt import STTResult
        from boxbot.communication.voice import VoiceSession, VoiceSessionState

        session, mic, speaker = self._make_session()
        session._state = VoiceSessionState.ACTIVE
        session._conversation_id = "voice_t"
        session._stt = MagicMock()
        session._stt.transcribe = AsyncMock(
            return_value=STTResult(text="hello", language="en"),
        )
        session._diarizer = None

        import asyncio as _aio
        grace = _aio.create_task(_aio.sleep(5))
        session._active_timeout_task = grace

        utterance = Utterance(
            audio=b"\x00" * 32000,
            duration=1.0,
            sample_rate=16000,
            timestamp_start=1.0,
            timestamp_end=2.0,
        )
        await session._on_utterance(utterance)

        assert grace.cancelled() or grace.done()
        assert session._active_timeout_task is None

    def test_build_attributed_transcript_no_diarization(self):
        from boxbot.communication.stt import STTResult
        from boxbot.communication.voice import VoiceSession

        stt_result = STTResult(text="Hello world", language="en")
        transcript = VoiceSession._build_attributed_transcript(stt_result, None)
        assert transcript == "Hello world"

    def test_build_attributed_transcript_with_diarization(self):
        from boxbot.communication.diarization import DiarizationResult, SpeakerSegment
        from boxbot.communication.stt import STTResult, WordInfo
        from boxbot.communication.voice import VoiceSession

        stt_result = STTResult(
            text="Hello world how are you",
            language="en",
            words=[
                WordInfo(word="Hello", start=0.0, end=0.3),
                WordInfo(word="world", start=0.3, end=0.6),
                WordInfo(word="how", start=1.0, end=1.2),
                WordInfo(word="are", start=1.2, end=1.4),
                WordInfo(word="you", start=1.4, end=1.6),
            ],
        )
        diar_result = DiarizationResult(
            segments=[
                SpeakerSegment(speaker_label="SPEAKER_00", start=0.0, end=0.8),
                SpeakerSegment(speaker_label="SPEAKER_01", start=0.9, end=2.0),
            ],
            num_speakers=2,
        )

        transcript = VoiceSession._build_attributed_transcript(
            stt_result, diar_result
        )
        assert "[SPEAKER_00]" in transcript
        assert "[SPEAKER_01]" in transcript
        assert "Hello" in transcript
        assert "you" in transcript


# ---------------------------------------------------------------------------
# Config integration for voice pipeline
# ---------------------------------------------------------------------------


class TestVoiceConfig:
    """Verify voice config sections load correctly."""

    def test_default_voice_config(self):
        from boxbot.core.config import VoiceConfig

        cfg = VoiceConfig()
        assert cfg.wake_word.engine == "openwakeword"
        assert cfg.vad.threshold == 0.5
        assert cfg.turn_detection.silence_threshold == 800
        assert cfg.stt.provider == "elevenlabs"
        assert cfg.tts.provider == "elevenlabs"
        assert cfg.session.active_timeout == 30
        assert cfg.session.suspend_timeout == 180

    def test_boxbot_config_includes_voice(self):
        from boxbot.core.config import BoxBotConfig

        cfg = BoxBotConfig()
        assert hasattr(cfg, "voice")
        assert cfg.voice.wake_word.word == "hey_jarvis"
