"""Tests for the ONNX speaker embedding backend.

The frontend tests matter most: a subtly wrong filterbank still produces
plausible-looking vectors, so the failure would show up only as quietly
degraded recognition. Where torchaudio is installed (dev machines, not
the panel) the numpy implementation is checked against Kaldi directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from boxbot.communication.diarization import create_speaker_backend
from boxbot.communication.speaker_embedding import (
    OnnxSpeakerEmbedder,
    compute_fbank,
)
from boxbot.core.config import DiarizationConfig

MODEL_PATH = Path("data/models/voxceleb_ECAPA512.onnx")

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _noise(n: int, seed: int = 0, scale: float = 3000.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n) * scale


class TestComputeFbank:
    def test_shape_follows_kaldi_snip_edges(self):
        # 400-sample window, 160-sample shift, whole frames only.
        feats = compute_fbank(_noise(16000), 16000)
        assert feats.shape == (1 + (16000 - 400) // 160, 80)
        assert feats.dtype == np.float32

    def test_utterance_shorter_than_one_frame_is_empty(self):
        assert compute_fbank(_noise(399), 16000).shape == (0, 80)
        # Exactly one frame's worth still yields a frame.
        assert compute_fbank(_noise(400), 16000).shape == (1, 80)

    def test_output_is_mean_normalised(self):
        feats = compute_fbank(_noise(16000), 16000)
        assert np.allclose(feats.mean(axis=0), 0.0, atol=1e-4)

    def test_rejects_non_16k_audio(self):
        # The model is rate-specific; resampling silently would be worse.
        with pytest.raises(ValueError, match="16000"):
            compute_fbank(_noise(16000), 8000)

    def test_finite_on_digital_silence(self):
        # log(0) without the epsilon floor would be -inf and poison a centroid.
        feats = compute_fbank(np.zeros(16000), 16000)
        assert np.all(np.isfinite(feats))

    def test_matches_torchaudio_kaldi_reference(self):
        """The numpy frontend must equal torchaudio's Kaldi fbank."""
        torch = pytest.importorskip("torch")
        kaldi = pytest.importorskip("torchaudio.compliance.kaldi")

        for seed, n in ((1, 16000), (2, 32000), (3, 5000)):
            wav = _noise(n, seed=seed)
            mine = compute_fbank(wav, 16000)
            ref = kaldi.fbank(
                torch.tensor(wav, dtype=torch.float32).unsqueeze(0),
                num_mel_bins=80,
                frame_length=25,
                frame_shift=10,
                dither=0.0,
                sample_frequency=16000,
                window_type="hamming",
                use_energy=False,
            )
            ref = (ref - ref.mean(dim=0)).numpy()

            assert mine.shape == ref.shape
            # float32 accumulation noise only; log-mel values are O(1-20).
            assert np.abs(mine - ref).max() < 1e-3


class TestEngineSelection:
    def test_factory_returns_onnx_backend(self):
        cfg = DiarizationConfig(engine="onnx", embedding_model=str(MODEL_PATH))
        assert isinstance(create_speaker_backend(cfg), OnnxSpeakerEmbedder)

    def test_onnx_backend_cannot_diarize(self):
        # The absent method is the point: no silent single-speaker
        # pretence when someone asks for segmentation.
        cfg = DiarizationConfig(engine="onnx", embedding_model=str(MODEL_PATH))
        assert not hasattr(create_speaker_backend(cfg), "diarize")

    def test_config_rejects_diarization_with_onnx(self):
        with pytest.raises(ValueError, match="cannot diarize"):
            DiarizationConfig(engine="onnx", enabled=True)

    def test_unknown_engine_rejected(self):
        with pytest.raises(ValueError):
            DiarizationConfig(engine="nope")


class TestOnnxSpeakerEmbedder:
    @pytest.mark.asyncio
    async def test_missing_model_raises_file_not_found(self, tmp_path):
        """Permanent failure — voice.py drops the backend on this."""
        cfg = DiarizationConfig(
            engine="onnx", embedding_model=str(tmp_path / "absent.onnx")
        )
        with pytest.raises(FileNotFoundError, match="absent.onnx"):
            await OnnxSpeakerEmbedder(cfg).start()

    @pytest.mark.asyncio
    async def test_embed_before_start_raises(self):
        cfg = DiarizationConfig(engine="onnx", embedding_model=str(MODEL_PATH))
        with pytest.raises(RuntimeError, match="not started"):
            await OnnxSpeakerEmbedder(cfg).embed_utterance(b"\x00\x00", 16000)


@pytest.mark.skipif(
    not MODEL_PATH.is_file(),
    reason=f"{MODEL_PATH} not present (see docs/voice-pipeline.md)",
)
class TestOnnxSpeakerEmbedderWithModel:
    """Runs only where the ~25 MB model has been fetched."""

    @pytest.fixture
    async def embedder(self):
        cfg = DiarizationConfig(engine="onnx", embedding_model=str(MODEL_PATH))
        emb = OnnxSpeakerEmbedder(cfg)
        await emb.start()
        yield emb
        await emb.stop()

    @staticmethod
    def _tone(f0: float, seed: int, dur: float = 2.0) -> bytes:
        rng = np.random.default_rng(seed)
        t = np.arange(int(16000 * dur)) / 16000
        sig = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
        sig = sig + 0.05 * rng.standard_normal(t.size)
        return (sig / np.abs(sig).max() * 8000).astype(np.int16).tobytes()

    @pytest.mark.asyncio
    async def test_returns_finite_embedding_vector(self, embedder):
        emb = await embedder.embed_utterance(self._tone(110, seed=1), 16000)
        assert emb.shape == (192,)  # ECAPA512; resnet34_LM would be 256
        assert emb.dtype == np.float32
        assert np.all(np.isfinite(emb))

    @pytest.mark.asyncio
    async def test_deterministic(self, embedder):
        audio = self._tone(110, seed=1)
        first = await embedder.embed_utterance(audio, 16000)
        second = await embedder.embed_utterance(audio, 16000)
        assert np.allclose(first, second)

    @pytest.mark.asyncio
    async def test_same_source_scores_above_different_source(self, embedder):
        def cos(x, y):
            return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))

        a1 = await embedder.embed_utterance(self._tone(110, seed=1), 16000)
        a2 = await embedder.embed_utterance(self._tone(110, seed=2), 16000)
        b1 = await embedder.embed_utterance(self._tone(210, seed=3), 16000)
        # Synthetic tones are not speech, so only the ordering is
        # meaningful here — but a broken frontend inverts it.
        assert cos(a1, a2) > cos(a1, b1)

    @pytest.mark.asyncio
    async def test_empty_and_subframe_audio_return_none(self, embedder):
        assert await embedder.embed_utterance(b"", 16000) is None
        short = np.zeros(100, dtype=np.int16).tobytes()
        assert await embedder.embed_utterance(short, 16000) is None
