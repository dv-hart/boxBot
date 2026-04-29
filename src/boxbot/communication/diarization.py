"""Speaker diarization using pyannote.audio.

Identifies "who spoke when" in an audio segment, providing speaker labels
and optional speaker embeddings for downstream identity matching.

Usage:
    from boxbot.core.config import get_config
    from boxbot.communication.diarization import SpeakerDiarizer

    diarizer = SpeakerDiarizer(get_config().voice.diarization)
    await diarizer.start()
    result = await diarizer.diarize(pcm_bytes, sample_rate=16000)
    for seg in result.segments:
        print(f"{seg.speaker_label}: {seg.start:.2f}s - {seg.end:.2f}s")
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from boxbot.core.config import DiarizationConfig

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from pyannote.audio import Inference, Model, Pipeline
except ImportError:
    Pipeline = None  # type: ignore[assignment, misc]
    Model = None  # type: ignore[assignment, misc]
    Inference = None  # type: ignore[assignment, misc]


@dataclass
class SpeakerSegment:
    """A segment of audio attributed to a specific speaker."""

    speaker_label: str  # "SPEAKER_00", "SPEAKER_01", etc.
    start: float  # seconds from utterance start
    end: float  # seconds from utterance end
    embedding: np.ndarray | None = None  # 192-dim speaker embedding


@dataclass
class DiarizationResult:
    """Result of speaker diarization on an utterance."""

    segments: list[SpeakerSegment] = field(default_factory=list)
    num_speakers: int = 0


class SpeakerDiarizer:
    """Speaker diarization using pyannote.audio.

    Loads the pyannote diarization pipeline and an embedding model for
    extracting per-segment speaker embeddings. All heavy computation
    runs in a thread executor to avoid blocking the event loop.
    """

    def __init__(self, config: DiarizationConfig) -> None:
        self._config = config
        self._pipeline: Any = None
        self._embedding_model: Any = None
        self._embedding_inference: Any = None

    async def start(self) -> None:
        """Load pyannote pipeline and embedding model.

        Runs model loading in a thread executor since it is CPU-heavy
        and involves downloading model weights on first run.
        """
        if Pipeline is None:
            raise ImportError(
                "pyannote.audio is required for SpeakerDiarizer. "
                "Install it with: pip install pyannote.audio"
            )
        if torch is None:
            raise ImportError(
                "torch is required for SpeakerDiarizer. "
                "Install it with: pip install torch"
            )

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )

        loop = asyncio.get_event_loop()

        logger.info("Loading diarization pipeline: %s", self._config.model)
        self._pipeline = await loop.run_in_executor(
            None,
            lambda: Pipeline.from_pretrained(
                self._config.model, token=hf_token
            ),
        )

        logger.info("Loading embedding model: %s", self._config.embedding_model)
        self._embedding_model = await loop.run_in_executor(
            None,
            lambda: Model.from_pretrained(
                self._config.embedding_model, token=hf_token
            ),
        )
        self._embedding_inference = Inference(
            self._embedding_model, window="whole"
        )

        logger.info("Diarization models loaded successfully")

    async def stop(self) -> None:
        """Release models and free memory."""
        self._pipeline = None
        self._embedding_model = None
        self._embedding_inference = None
        logger.info("Diarization models released")

    async def diarize(
        self, audio: bytes, sample_rate: int
    ) -> DiarizationResult:
        """Diarize audio to identify speaker segments.

        Args:
            audio: Raw PCM int16 mono audio bytes.
            sample_rate: Sample rate of the audio (e.g. 16000).

        Returns:
            DiarizationResult with speaker segments and optional embeddings.
        """
        if self._pipeline is None:
            raise RuntimeError(
                "Diarizer not started. Call start() before diarize()."
            )

        # Convert PCM int16 bytes to float32 numpy array
        audio_int16 = np.frombuffer(audio, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Create pyannote-compatible input
        waveform = torch.tensor(audio_float).unsqueeze(0)  # (1, samples)
        input_data = {"waveform": waveform, "sample_rate": sample_rate}

        # Run diarization pipeline in executor (CPU-bound)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._pipeline(
                input_data,
                min_speakers=self._config.min_speakers,
                max_speakers=self._config.max_speakers,
            ),
        )

        # pyannote.audio 4.x returns DiarizeOutput — unwrap to the
        # classic Annotation object for .itertracks. Older versions
        # returned an Annotation directly.
        diarization = getattr(result, "speaker_diarization", result)

        # Build segments from pipeline output
        segments: list[SpeakerSegment] = []
        for segment, _track, speaker_label in diarization.itertracks(
            yield_label=True
        ):
            embedding = await self._extract_embedding(
                audio_float, sample_rate, segment.start, segment.end
            )
            segments.append(
                SpeakerSegment(
                    speaker_label=speaker_label,
                    start=segment.start,
                    end=segment.end,
                    embedding=embedding,
                )
            )

        # Count unique speakers
        unique_speakers = len({s.speaker_label for s in segments})

        logger.debug(
            "Diarization complete: %d segments, %d speakers",
            len(segments),
            unique_speakers,
        )

        return DiarizationResult(
            segments=segments, num_speakers=unique_speakers
        )

    async def _extract_embedding(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        start: float,
        end: float,
    ) -> np.ndarray | None:
        """Extract speaker embedding for a segment.

        Args:
            audio_array: Full utterance as float32 numpy array.
            sample_rate: Sample rate of the audio.
            start: Segment start time in seconds.
            end: Segment end time in seconds.

        Returns:
            192-dim speaker embedding vector, or None on error.
        """
        if self._embedding_inference is None:
            return None

        try:
            # Crop audio to segment boundaries
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            segment_audio = audio_array[start_sample:end_sample]

            if len(segment_audio) == 0:
                return None

            # Create pyannote-compatible input for the segment
            waveform = torch.tensor(segment_audio).unsqueeze(0)  # (1, samples)
            input_data = {"waveform": waveform, "sample_rate": sample_rate}

            # Run embedding extraction in executor
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: self._embedding_inference(input_data),
            )

            arr = np.array(embedding)
            # pyannote occasionally returns NaN/Inf vectors for very
            # short or near-silent segments. A single NaN propagated
            # into a stored centroid silently breaks all downstream
            # cosine similarities (NaN poisons the mean). Drop it here
            # before any downstream code sees it.
            if not np.all(np.isfinite(arr)):
                logger.warning(
                    "Embedding for segment %.2f-%.2f contained NaN/Inf "
                    "(%d non-finite of %d) — discarding",
                    start, end,
                    int((~np.isfinite(arr)).sum()), arr.size,
                )
                return None
            return arr
        except Exception:
            logger.warning(
                "Failed to extract embedding for segment %.2f-%.2f",
                start,
                end,
                exc_info=True,
            )
            return None
