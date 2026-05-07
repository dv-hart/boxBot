"""ElevenLabs cost-tracking integration for the voice adapters.

Verifies that ``ElevenLabsTTS`` and ``ElevenLabsSTT`` write exactly one
``cost_log`` row per successful call, with the right billed unit:

* TTS: ``character_count`` taken from the ``x-character-count`` HTTP
  response header (the authoritative billed unit). Fallback to
  ``len(text)`` only when the header is missing — and a WARNING is
  emitted in that case.
* STT: ``audio_seconds`` measured locally from the input PCM (Scribe
  does not return billed duration in its response).

The ``elevenlabs`` SDK is not installed in this environment; we
inline-mock it via ``sys.modules`` so the adapter modules import
cleanly.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Inline-mock the elevenlabs SDK (and other heavy boxBot deps) before any
# boxbot.* import so the adapter modules can be loaded without the real
# package on PYTHONPATH. Done at import time so pytest's collection
# phase already sees the mocks.
# ---------------------------------------------------------------------------


for _mod in ("anthropic", "elevenlabs"):
    sys.modules.setdefault(_mod, MagicMock())

# elevenlabs.AsyncElevenLabs and VoiceSettings are pulled by name from
# the mocked module; ensure ``from elevenlabs import AsyncElevenLabs``
# resolves to a MagicMock (the default) rather than raising.
_eleven_mock = sys.modules["elevenlabs"]
if not hasattr(_eleven_mock, "AsyncElevenLabs"):
    _eleven_mock.AsyncElevenLabs = MagicMock()
if not hasattr(_eleven_mock, "VoiceSettings"):
    _eleven_mock.VoiceSettings = MagicMock()

# ``boxbot.core.__init__`` eagerly imports ``BoxBotAgent``, which pulls
# in dream_poller, claude_agent_sdk, and the rest of the runtime stack.
# Importing ``boxbot.memory.store`` indirectly triggers that chain via
# ``boxbot.memory.dream`` -> ``boxbot.core.paths`` -> ``boxbot.core``.
# Stub the agent module before any boxbot import so the chain short-
# circuits cleanly. Same pattern as ``tests/test_cost_web_search.py``.
if "boxbot.core.agent" not in sys.modules:
    _stub_agent = MagicMock()
    _stub_agent.BoxBotAgent = MagicMock()
    sys.modules["boxbot.core.agent"] = _stub_agent


# ---------------------------------------------------------------------------
# Pricing fixture — same pattern as ``tests/test_cost.py``: a small,
# deterministic YAML. ``BOXBOT_PRICING_CONFIG`` overrides the path used
# by ``boxbot.cost.pricing.get_pricing()``.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _pricing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    yaml_text = """
anthropic:
  source_url: https://example.test/anthropic
  verified_on: 2026-05-02
  models:
    claude-haiku-4-5:
      input_per_mtok: 1.00
      output_per_mtok: 5.00
elevenlabs:
  source_url: https://example.test/elevenlabs
  verified_on: 2026-05-02
  tts:
    eleven_turbo_v2_5:
      dollars_per_char: 0.0000250
  stt:
    scribe_v2:
      dollars_per_minute: 0.003600
"""
    path = tmp_path / "pricing.yaml"
    path.write_text(yaml_text)
    monkeypatch.setenv("BOXBOT_PRICING_CONFIG", str(path))

    from boxbot.cost import reload_pricing

    reload_pricing(path)
    yield
    # Restore canonical pricing so later test files don't read this
    # file's reduced model list out of the cached singleton.
    monkeypatch.delenv("BOXBOT_PRICING_CONFIG", raising=False)
    reload_pricing()


# ---------------------------------------------------------------------------
# Shared helpers: in-memory MemoryStore wired into the adapter's
# singleton hook so we can assert directly on cost_log rows.
# ---------------------------------------------------------------------------


@pytest.fixture
async def store(tmp_path: Path):
    """Return an initialised MemoryStore against a temp DB."""
    from boxbot.memory.store import MemoryStore

    s = MemoryStore(db_path=tmp_path / "memory.db")
    await s.initialize()
    yield s
    await s.close()


async def _select_cost_rows(store_obj) -> list[dict]:
    """Read all cost_log rows back as plain dicts, ordered by id."""
    cur = await store_obj.db.execute(
        """SELECT purpose, provider, model, cost_usd,
                  character_count, audio_seconds, correlation_id
           FROM cost_log ORDER BY id"""
    )
    rows = await cur.fetchall()
    return [
        {
            "purpose": r[0],
            "provider": r[1],
            "model": r[2],
            "cost_usd": r[3],
            "character_count": r[4],
            "audio_seconds": r[5],
            "correlation_id": r[6],
        }
        for r in rows
    ]


def _patch_cost_store(monkeypatch: pytest.MonkeyPatch, module, store_obj) -> None:
    """Replace the module-level ``_get_cost_store`` async with one that
    returns ``store_obj``. Both ``tts`` and ``stt`` adapters expose the
    same hook name."""

    async def _fake() -> object:
        return store_obj

    monkeypatch.setattr(module, "_get_cost_store", _fake)


# ---------------------------------------------------------------------------
# Async-context-manager response stubs for the raw client
# ---------------------------------------------------------------------------


class _AsyncCM:
    """Minimal async context manager that yields a single response object.

    Mirrors the shape of ``client.text_to_speech.with_raw_response.stream``
    / ``...convert``: ``async with`` yields an object with ``headers`` and
    ``data`` (an async iterator of bytes).
    """

    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


async def _async_iter_bytes(chunks):
    for c in chunks:
        yield c


def _make_raw_response(headers: dict, chunks: list[bytes]):
    return SimpleNamespace(
        headers=headers,
        data=_async_iter_bytes(chunks),
    )


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestElevenLabsTTSCost:
    async def test_synthesize_records_billed_chars_from_header(
        self, store, monkeypatch
    ):
        """Header carries the authoritative billed count; we use it
        verbatim and never substitute len(text)."""
        from boxbot.communication import tts as tts_module

        _patch_cost_store(monkeypatch, tts_module, store)

        adapter = tts_module.ElevenLabsTTS.__new__(tts_module.ElevenLabsTTS)
        adapter._client = MagicMock()
        adapter._voice_id = "v1"
        adapter._model = "eleven_turbo_v2_5"
        adapter._stability = 0.5
        adapter._similarity_boost = 0.75
        adapter._optimize_streaming_latency = 3

        raw = _make_raw_response(
            headers={"x-character-count": "1000"},
            chunks=[b"\x00" * 200, b"\x01" * 200],
        )
        adapter._client.text_to_speech.with_raw_response.convert = MagicMock(
            return_value=_AsyncCM(raw)
        )

        out = await adapter.synthesize("hi there", conversation_id="conv-abc")
        assert out == b"\x00" * 200 + b"\x01" * 200

        rows = await _select_cost_rows(store)
        assert len(rows) == 1
        row = rows[0]
        assert row["purpose"] == "tts"
        assert row["provider"] == "elevenlabs"
        assert row["model"] == "eleven_turbo_v2_5"
        assert row["character_count"] == 1000
        # 1000 chars × $0.0000250 = $0.025
        assert row["cost_usd"] == pytest.approx(0.025)
        assert row["correlation_id"] == "conv-abc"

    async def test_synthesize_stream_records_after_consumption(
        self, store, monkeypatch
    ):
        """The streaming path records exactly once when the consumer
        finishes iterating — not before, not on partial drains.
        """
        from boxbot.communication import tts as tts_module

        _patch_cost_store(monkeypatch, tts_module, store)

        adapter = tts_module.ElevenLabsTTS.__new__(tts_module.ElevenLabsTTS)
        adapter._client = MagicMock()
        adapter._voice_id = "v1"
        adapter._model = "eleven_turbo_v2_5"
        adapter._stability = 0.5
        adapter._similarity_boost = 0.75
        adapter._optimize_streaming_latency = 3

        raw = _make_raw_response(
            headers={"x-character-count": "1000"},
            chunks=[b"\x00" * 100, b"\x01" * 100, b"\x02" * 100],
        )
        adapter._client.text_to_speech.with_raw_response.stream = MagicMock(
            return_value=_AsyncCM(raw)
        )

        # Drive the generator to completion (TTSStream does this via
        # speaker.play_stream(); here we iterate directly).
        chunks = []
        async for chunk in adapter.synthesize_stream(
            "hello world", conversation_id="conv-stream"
        ):
            chunks.append(chunk)

        assert len(chunks) == 3
        rows = await _select_cost_rows(store)
        assert len(rows) == 1
        assert rows[0]["purpose"] == "tts"
        assert rows[0]["character_count"] == 1000
        assert rows[0]["cost_usd"] == pytest.approx(0.025)
        assert rows[0]["correlation_id"] == "conv-stream"

    async def test_missing_header_falls_back_with_warning(
        self, store, monkeypatch, caplog
    ):
        """If the SDK strips the header for some reason we still record
        a row, but using ``len(text)`` and a WARNING — never silently."""
        from boxbot.communication import tts as tts_module

        _patch_cost_store(monkeypatch, tts_module, store)

        adapter = tts_module.ElevenLabsTTS.__new__(tts_module.ElevenLabsTTS)
        adapter._client = MagicMock()
        adapter._voice_id = "v1"
        adapter._model = "eleven_turbo_v2_5"
        adapter._stability = 0.5
        adapter._similarity_boost = 0.75
        adapter._optimize_streaming_latency = 3

        text = "twelve chars"  # len == 12
        raw = _make_raw_response(headers={}, chunks=[b"\x00" * 100])
        adapter._client.text_to_speech.with_raw_response.convert = MagicMock(
            return_value=_AsyncCM(raw)
        )

        with caplog.at_level(logging.WARNING, logger="boxbot.communication.tts"):
            await adapter.synthesize(text, conversation_id="conv-fallback")

        rows = await _select_cost_rows(store)
        assert len(rows) == 1
        assert rows[0]["character_count"] == len(text) == 12
        # 12 × $0.0000250 = $0.0003
        assert rows[0]["cost_usd"] == pytest.approx(12 * 0.0000250)

        warned = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "x-character-count" in r.getMessage()
        ]
        assert warned, "expected a WARNING about the missing header"


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestElevenLabsSTTCost:
    async def test_transcribe_records_audio_seconds_from_input(
        self, store, monkeypatch
    ):
        """Scribe does not return billed duration; we measure it from
        the input PCM (mono int16 at the given sample rate)."""
        from boxbot.communication import stt as stt_module

        _patch_cost_store(monkeypatch, stt_module, store)

        # 30 seconds of mono int16 audio @ 16 kHz: 16000 samples/s × 2 B
        # × 30 s = 960_000 bytes.
        sample_rate = 16_000
        seconds = 30.0
        pcm = b"\x00" * int(sample_rate * 2 * seconds)

        # Mock the SDK call. ``transcribe`` calls
        # ``self._client.speech_to_text.convert`` and only reads .text /
        # .words / .language_code from the result.
        result_obj = SimpleNamespace(text="hello", words=[], language_code="en")

        async def _fake_convert(**_):
            return result_obj

        adapter = stt_module.ElevenLabsSTT.__new__(stt_module.ElevenLabsSTT)
        adapter._client = MagicMock()
        adapter._client.speech_to_text.convert = _fake_convert
        adapter._model = "scribe_v2"

        out = await adapter.transcribe(
            pcm, sample_rate=sample_rate, conversation_id="conv-stt"
        )
        assert out.text == "hello"

        rows = await _select_cost_rows(store)
        assert len(rows) == 1
        row = rows[0]
        assert row["purpose"] == "stt"
        assert row["provider"] == "elevenlabs"
        assert row["model"] == "scribe_v2"
        assert row["audio_seconds"] == pytest.approx(30.0)
        # 30 s = 0.5 min × $0.003600 = $0.0018
        assert row["cost_usd"] == pytest.approx(0.5 * 0.003600)
        assert row["correlation_id"] == "conv-stt"
