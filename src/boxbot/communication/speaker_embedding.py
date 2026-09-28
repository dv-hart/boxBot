"""ONNX speaker embedding — the small-footprint voice fingerprint.

The pyannote path needs torch plus a few hundred MB of models. On a
small aarch64 host (4 cores, ~1 GB free) that is a bad trade, and
push-to-talk makes it unnecessary: one person holds the button, so every
utterance is single-speaker by construction and segmentation is dead
weight. All that is wanted is a whole-utterance embedding.

This module runs a wespeaker ONNX model (~25 MB) on ``onnxruntime`` —
already a declared dependency — behind the same ``start`` / ``stop`` /
``embed_utterance`` surface :class:`SpeakerDiarizer` exposes, so the
downstream voice-ReID path does not change. It cannot diarize; the
engine selection refuses that combination rather than pretending.

The model consumes 80-bin Kaldi filterbank features, not raw audio, so
the frontend is reimplemented here in numpy. It is a port of
``torchaudio.compliance.kaldi.fbank`` restricted to the parameters
wespeaker was trained with, and ``tests/test_speaker_embedding.py``
asserts it matches that reference to within float32 accumulation noise
(~1e-4 on log-mel values of order 1-20) wherever torchaudio is
installed. Keeping the frontend honest matters: a subtly wrong window or
mel bank still yields plausible vectors, and the failure would surface
only as quietly degraded recognition.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import numpy as np

from boxbot.core.config import DiarizationConfig

logger = logging.getLogger(__name__)

# wespeaker's training frontend. These are not tunable: they must match
# what the checkpoint saw, so they are constants rather than config.
_SAMPLE_RATE = 16000
_NUM_MEL_BINS = 80
_FRAME_LENGTH_MS = 25.0
_FRAME_SHIFT_MS = 10.0
_PREEMPHASIS = 0.97
_LOW_FREQ = 20.0
_HAMMING_ALPHA = 0.54

# torch.finfo(torch.float32).eps — the floor Kaldi applies before log.
_EPSILON = float(np.finfo(np.float32).eps)


def _mel_scale(freq: np.ndarray | float) -> np.ndarray | float:
    """Kaldi's mel scale (natural log, 1127 scaling)."""
    return 1127.0 * np.log(1.0 + np.asarray(freq, dtype=np.float64) / 700.0)


def _mel_banks(num_bins: int, padded_window_size: int, sample_rate: int) -> np.ndarray:
    """Triangular mel filterbank, matching ``kaldi.get_mel_banks``.

    Returns ``(num_bins, padded_window_size // 2)``. The caller pads one
    zero column to line up with the ``rfft`` output length.
    """
    num_fft_bins = padded_window_size // 2
    nyquist = 0.5 * sample_rate
    # high_freq=0.0 in Kaldi means "nyquist".
    mel_low = _mel_scale(_LOW_FREQ)
    mel_high = _mel_scale(nyquist)
    mel_delta = (mel_high - mel_low) / (num_bins + 1)

    bins = np.arange(num_bins, dtype=np.float64).reshape(-1, 1)
    left_mel = mel_low + bins * mel_delta
    center_mel = mel_low + (bins + 1.0) * mel_delta
    right_mel = mel_low + (bins + 2.0) * mel_delta

    fft_bin_width = sample_rate / float(padded_window_size)
    mel = _mel_scale(fft_bin_width * np.arange(num_fft_bins, dtype=np.float64))
    mel = np.asarray(mel).reshape(1, -1)

    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)
    return np.maximum(0.0, np.minimum(up_slope, down_slope))


def _hamming_window(size: int) -> np.ndarray:
    """``torch.hamming_window(size, periodic=False, alpha=.54, beta=.46)``."""
    n = np.arange(size, dtype=np.float64)
    beta = 1.0 - _HAMMING_ALPHA
    return _HAMMING_ALPHA - beta * np.cos(2.0 * np.pi * n / (size - 1))


def compute_fbank(waveform: np.ndarray, sample_rate: int = _SAMPLE_RATE) -> np.ndarray:
    """80-bin log-mel filterbank features with mean normalisation.

    Args:
        waveform: Mono samples on the **int16 value scale** (i.e. roughly
            -32768..32767) as float. wespeaker trained on Kaldi features
            computed from int16-scaled audio; passing [-1, 1] audio here
            shifts every log-energy by a constant and quietly degrades
            the embedding.
        sample_rate: Must be 16 kHz — the model's training rate.

    Returns:
        ``(num_frames, 80)`` float32, cepstral-mean-normalised over time.
        Empty ``(0, 80)`` when the utterance is shorter than one frame.
    """
    if sample_rate != _SAMPLE_RATE:
        raise ValueError(
            f"speaker embedding expects {_SAMPLE_RATE} Hz audio, got {sample_rate}"
        )

    waveform = np.asarray(waveform, dtype=np.float64).reshape(-1)
    window_size = int(sample_rate * _FRAME_LENGTH_MS / 1000.0)  # 400
    window_shift = int(sample_rate * _FRAME_SHIFT_MS / 1000.0)  # 160

    if waveform.size < window_size:
        return np.zeros((0, _NUM_MEL_BINS), dtype=np.float32)

    # round_to_power_of_two=True → 400 becomes 512.
    padded_window_size = 1
    while padded_window_size < window_size:
        padded_window_size *= 2

    # snip_edges=True: only whole frames, no padding at the tail.
    num_frames = 1 + (waveform.size - window_size) // window_shift
    idx = np.arange(window_size)[None, :] + (
        np.arange(num_frames)[:, None] * window_shift
    )
    frames = waveform[idx]

    # remove_dc_offset
    frames = frames - frames.mean(axis=1, keepdims=True)

    # Pre-emphasis with replicate padding, so frame[0] uses itself.
    shifted = np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
    frames = frames - _PREEMPHASIS * shifted

    frames = frames * _hamming_window(window_size)[None, :]

    if padded_window_size > window_size:
        frames = np.pad(
            frames, ((0, 0), (0, padded_window_size - window_size)), mode="constant"
        )

    # use_power=True → magnitude squared.
    spectrum = np.abs(np.fft.rfft(frames, n=padded_window_size)) ** 2

    banks = _mel_banks(_NUM_MEL_BINS, padded_window_size, sample_rate)
    banks = np.pad(banks, ((0, 0), (0, 1)), mode="constant")  # match rfft length

    mel_energies = spectrum @ banks.T
    feats = np.log(np.maximum(mel_energies, _EPSILON))

    # Cepstral mean normalisation over the utterance, as wespeaker does.
    feats = feats - feats.mean(axis=0, keepdims=True)
    return feats.astype(np.float32)


class OnnxSpeakerEmbedder:
    """Whole-utterance speaker embedding via a wespeaker ONNX model.

    Mirrors the subset of :class:`SpeakerDiarizer` the single-speaker
    path uses. Deliberately has no ``diarize`` method — this model
    cannot segment, and the engine selection rejects that combination up
    front rather than failing per-utterance.
    """

    def __init__(self, config: DiarizationConfig) -> None:
        self._config = config
        # ``embedding_model`` names the model for whichever engine is
        # selected: a HuggingFace id for pyannote, a filesystem path here.
        self._model_path = Path(config.embedding_model)
        self._session: Any = None

    async def start(self) -> None:
        """Load the ONNX session.

        Raises ImportError when onnxruntime is missing and FileNotFoundError
        when the model is absent — both permanent, so the voice path drops
        the backend after one attempt instead of retrying every utterance.
        """
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover - exercised via message
            raise ImportError(
                "onnxruntime is required for the 'onnx' speaker embedding "
                f"engine. Install it with: pip install onnxruntime ({e})"
            ) from e

        if not self._model_path.is_file():
            raise FileNotFoundError(
                f"speaker embedding model not found at {self._model_path} — "
                "set voice.diarization.embedding_model to the .onnx file"
            )

        # Thread count is left to onnxruntime (defaults to the core
        # count). Pinning it lower was measured slower on a 4-core A53
        # host — 0.36 s vs 0.45 s for a 4 s utterance — with no RSS
        # saving, and embedding is brief and bursty enough that it does
        # not meaningfully contend with the rest of the process.
        loop = asyncio.get_event_loop()
        self._session = await loop.run_in_executor(
            None,
            lambda: ort.InferenceSession(
                str(self._model_path),
                providers=["CPUExecutionProvider"],
            ),
        )
        logger.info(
            "Speaker embedding model loaded: %s (onnx, single-speaker)",
            self._model_path.name,
        )

    async def stop(self) -> None:
        """Release the session."""
        self._session = None
        logger.info("Speaker embedding model released")

    async def embed_utterance(
        self, audio: bytes, sample_rate: int
    ) -> np.ndarray | None:
        """Embed a whole utterance as one speaker.

        Returns the embedding, or None when the utterance is empty, too
        short for a single frame, or the model produced a non-finite
        vector. None is the same "no embedding" signal the pyannote path
        returns, and callers already handle it.
        """
        if self._session is None:
            raise RuntimeError(
                "Embedder not started. Call start() before embed_utterance()."
            )

        audio_int16 = np.frombuffer(audio, dtype=np.int16)
        if audio_int16.size == 0:
            return None

        # Kept on the int16 value scale — see compute_fbank.
        waveform = audio_int16.astype(np.float64)

        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, self._embed_sync, waveform, sample_rate
            )
        except Exception:
            logger.warning(
                "Failed to extract speaker embedding (%.2fs of audio)",
                audio_int16.size / float(sample_rate or _SAMPLE_RATE),
                exc_info=True,
            )
            return None

    def _embed_sync(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray | None:
        feats = compute_fbank(waveform, sample_rate)
        if feats.shape[0] == 0:
            return None

        outputs = self._session.run(None, {"feats": feats[None, :, :]})
        embedding = np.asarray(outputs[0], dtype=np.float32).reshape(-1)

        # Same guard the pyannote path applies: one NaN poisons a stored
        # centroid and silently disables every later cosine check.
        if not np.all(np.isfinite(embedding)):
            logger.warning(
                "Speaker embedding contained NaN/Inf (%d of %d) — discarding",
                int((~np.isfinite(embedding)).sum()),
                embedding.size,
            )
            return None
        return embedding
