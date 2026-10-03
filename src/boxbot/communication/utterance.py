"""The unit every capture path produces and STT consumes.

Lives on its own so a capture backend can produce one without importing
its siblings: another capture front-end has no business pulling in
``communication/vad.py`` (and, through it, torch) just to name the type
it hands to ``voice.py::_on_utterance``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Utterance:
    """A finalized utterance ready for STT and diarization."""

    audio: bytes  # complete PCM audio for this utterance (int16 LE mono)
    duration: float  # seconds
    sample_rate: int
    timestamp_start: float
    timestamp_end: float
