"""Tests for Phase B: Voice-Visual Identity Fusion components.

Covers VoiceReID, DOATracker, IdentityFusion, CropManager, state machine
conversation states, CloudStore voice centroids, and transcript attribution.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vec(dim: int = 192, seed: int | None = None) -> np.ndarray:
    """Generate a random L2-normalized float32 vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def _similar_vec(base: np.ndarray, noise: float = 0.01) -> np.ndarray:
    """Return a vector very close to *base* (high cosine similarity)."""
    v = base + np.random.randn(*base.shape).astype(np.float32) * noise
    v /= np.linalg.norm(v)
    return v


# ---------------------------------------------------------------------------
# VoiceReID
# ---------------------------------------------------------------------------


class TestVoiceReID:
    """Tests for voice embedding matching."""

    def _make(self, **kwargs):
        from boxbot.perception.voice_reid import VoiceReID

        return VoiceReID(**kwargs)

    def test_normalize_embedding(self):
        vr = self._make()
        raw = np.random.randn(1, 192).astype(np.float32)
        emb = vr.normalize_embedding(raw)
        assert emb.shape == (192,)
        assert emb.dtype == np.float32
        assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-5)

    def test_normalize_zero_vector(self):
        vr = self._make()
        emb = vr.normalize_embedding(np.zeros(192, dtype=np.float32))
        assert np.all(emb == 0)

    def test_match_high_confidence(self):
        vr = self._make(threshold=0.60)
        emb = _unit_vec(192, seed=42)
        centroid = _similar_vec(emb, noise=0.01)
        centroids = {"p1": ("Jacob", centroid)}
        result = vr.match(emb, centroids)
        assert result.tier == "high"
        assert result.person_id == "p1"
        assert result.person_name == "Jacob"
        assert result.confidence > 0.60

    def test_match_unknown_below_threshold(self):
        vr = self._make(threshold=0.60)
        emb = _unit_vec(192, seed=1)
        centroid = -emb  # opposite direction
        centroids = {"p1": ("Jacob", centroid)}
        result = vr.match(emb, centroids)
        assert result.tier == "unknown"
        assert result.person_id is None
        assert result.person_name is None

    def test_match_empty_centroids(self):
        vr = self._make()
        emb = _unit_vec(192)
        result = vr.match(emb, {})
        assert result.tier == "unknown"
        assert result.confidence == 0.0

    def test_match_selects_best_of_multiple(self):
        vr = self._make(threshold=0.60)
        emb = _unit_vec(192, seed=10)
        # One close, one far
        close = _similar_vec(emb, noise=0.01)
        far = -emb
        centroids = {
            "p1": ("Alice", far),
            "p2": ("Jacob", close),
        }
        result = vr.match(emb, centroids)
        assert result.person_name == "Jacob"
        assert result.person_id == "p2"

    def test_two_tier_only(self):
        """Voice uses only 'high' and 'unknown' — no 'medium' tier."""
        vr = self._make(threshold=0.60)
        emb = _unit_vec(192, seed=5)
        # Create a centroid that produces a moderate score (~0.55)
        centroid = emb + np.random.randn(192).astype(np.float32) * 0.4
        centroid /= np.linalg.norm(centroid)
        centroids = {"p1": ("Jacob", centroid)}
        result = vr.match(emb, centroids)
        assert result.tier in ("high", "unknown")
        assert result.tier != "medium"


# ---------------------------------------------------------------------------
# DOATracker
# ---------------------------------------------------------------------------


class TestDOATracker:
    """Tests for ReSpeaker DOA angle mapping and speaker-detection association."""

    def _make(self, **kwargs):
        from boxbot.perception.doa import DOATracker

        return DOATracker(**kwargs)

    def _detection(self, x1, y1, x2, y2, confidence=0.9):
        from boxbot.perception.person_detector import Detection

        return Detection(bbox=(x1, y1, x2, y2), confidence=confidence, class_id=0)

    # -- angle_to_camera_x --

    def test_forward_angle_maps_to_center(self):
        tracker = self._make(forward_angle=0, camera_hfov=120)
        assert tracker.angle_to_camera_x(0) == pytest.approx(0.0)

    def test_right_edge_of_fov(self):
        tracker = self._make(forward_angle=0, camera_hfov=120)
        # 60 degrees right = edge of 120° FOV
        assert tracker.angle_to_camera_x(60) == pytest.approx(1.0)

    def test_left_edge_of_fov(self):
        tracker = self._make(forward_angle=0, camera_hfov=120)
        # 300 degrees = -60 degrees (left edge)
        assert tracker.angle_to_camera_x(300) == pytest.approx(-1.0)

    def test_out_of_fov_value_exceeds_one(self):
        tracker = self._make(forward_angle=0, camera_hfov=120)
        x = tracker.angle_to_camera_x(90)  # 90° right
        assert abs(x) > 1.0

    def test_custom_forward_angle(self):
        tracker = self._make(forward_angle=90, camera_hfov=120)
        assert tracker.angle_to_camera_x(90) == pytest.approx(0.0)
        assert tracker.angle_to_camera_x(150) == pytest.approx(1.0)

    # -- is_in_fov --

    def test_center_is_in_fov(self):
        tracker = self._make(forward_angle=0, camera_hfov=120)
        assert tracker.is_in_fov(0) is True

    def test_edge_is_in_fov(self):
        tracker = self._make(forward_angle=0, camera_hfov=120)
        assert tracker.is_in_fov(60) is True

    def test_beyond_edge_is_out_of_fov(self):
        tracker = self._make(forward_angle=0, camera_hfov=120)
        assert tracker.is_in_fov(90) is False

    def test_behind_is_out_of_fov(self):
        tracker = self._make(forward_angle=0, camera_hfov=120)
        assert tracker.is_in_fov(180) is False

    def test_wraparound_in_fov(self):
        """Forward=350, check that 10° (20° away) is in 120° FOV."""
        tracker = self._make(forward_angle=350, camera_hfov=120)
        assert tracker.is_in_fov(10) is True

    # -- associate_speaker_to_detection --

    def test_associate_center_detection(self):
        tracker = self._make(forward_angle=0, camera_hfov=120)
        # Detection centered in frame (x center = 640 of 1280)
        det = self._detection(540, 100, 740, 500)
        result = tracker.associate_speaker_to_detection(0, [det])
        assert result is det

    def test_associate_picks_nearest(self):
        tracker = self._make(forward_angle=0, camera_hfov=120)
        # Left detection (center ~160px = -0.75 normalized)
        left = self._detection(60, 100, 260, 500)
        # Right detection (center ~1120px = +0.75 normalized)
        right = self._detection(1020, 100, 1220, 500)

        # DOA angle 50° right ≈ camera_x 0.83
        result = tracker.associate_speaker_to_detection(50, [left, right])
        assert result is right

    def test_associate_returns_none_out_of_fov(self):
        tracker = self._make(forward_angle=0, camera_hfov=120)
        det = self._detection(540, 100, 740, 500)
        result = tracker.associate_speaker_to_detection(180, [det])
        assert result is None

    def test_associate_returns_none_no_detections(self):
        tracker = self._make(forward_angle=0, camera_hfov=120)
        result = tracker.associate_speaker_to_detection(0, [])
        assert result is None


# ---------------------------------------------------------------------------
# IdentityFusion
# ---------------------------------------------------------------------------


class TestIdentityFusion:
    """Tests for multi-modal voice-visual fusion."""

    def _make_fusion(self, voice_threshold=0.60, forward_angle=0, hfov=120):
        from boxbot.core.config import PerceptionConfig
        from boxbot.perception.clouds import CloudStore
        from boxbot.perception.doa import DOATracker
        from boxbot.perception.fusion import IdentityFusion
        from boxbot.perception.voice_reid import VoiceReID

        cloud_store = AsyncMock(spec=CloudStore)
        voice_reid = VoiceReID(threshold=voice_threshold)
        doa = DOATracker(forward_angle=forward_angle, camera_hfov=hfov)
        config = PerceptionConfig()

        fusion = IdentityFusion(cloud_store, voice_reid, doa, config)
        return fusion, cloud_store

    def _active_person(self, ref, person_id=None, person_name=None, confidence=0.0, bbox=None):
        from boxbot.perception.state_machine import ActivePerson
        from boxbot.perception.visual_reid import MatchResult

        match = None
        if person_id:
            match = MatchResult(person_id, person_name, confidence, "high")
        return ActivePerson(
            ref=ref,
            match_result=match,
            last_detected=datetime.now(timezone.utc),
            bbox=bbox,
        )

    @pytest.mark.asyncio
    async def test_voice_match_no_visual(self):
        """Voice match, no visual association → voice-only identity."""
        fusion, cloud = self._make_fusion()
        emb = _unit_vec(192, seed=42)
        centroid = _similar_vec(emb, noise=0.01)
        cloud.get_voice_centroids = AsyncMock(
            return_value={"p1": ("Jacob", centroid)}
        )

        result = await fusion.fuse_speaker(
            speaker_label="SPEAKER_00",
            speaker_embedding=emb,
            active_persons={},
            doa_angle=None,
        )
        assert result.person_name == "Jacob"
        assert result.person_id == "p1"
        assert result.source == "voice"
        assert result.voice_confirmed is True

    @pytest.mark.asyncio
    async def test_voice_visual_agree(self):
        """Voice and visual both identify same person → fused."""
        fusion, cloud = self._make_fusion()
        emb = _unit_vec(192, seed=42)
        centroid = _similar_vec(emb, noise=0.01)
        cloud.get_voice_centroids = AsyncMock(
            return_value={"p1": ("Jacob", centroid)}
        )

        person = self._active_person(
            "Person A", person_id="p1", person_name="Jacob",
            confidence=0.9, bbox=(540, 100, 740, 500),
        )

        result = await fusion.fuse_speaker(
            speaker_label="SPEAKER_00",
            speaker_embedding=emb,
            active_persons={"Person A": person},
            doa_angle=0,  # straight ahead → center detection
        )
        assert result.source == "fused"
        assert result.voice_confirmed is True
        assert result.person_name == "Jacob"
        assert result.visual_ref == "Person A"

    @pytest.mark.asyncio
    async def test_voice_visual_disagree_in_fov(self):
        """Voice and visual disagree (both in FOV) → trust voice."""
        fusion, cloud = self._make_fusion()
        emb = _unit_vec(192, seed=42)
        centroid = _similar_vec(emb, noise=0.01)
        cloud.get_voice_centroids = AsyncMock(
            return_value={"p1": ("Jacob", centroid)}
        )

        person = self._active_person(
            "Person A", person_id="p2", person_name="Alice",
            confidence=0.9, bbox=(540, 100, 740, 500),
        )

        result = await fusion.fuse_speaker(
            speaker_label="SPEAKER_00",
            speaker_embedding=emb,
            active_persons={"Person A": person},
            doa_angle=0,
        )
        assert result.source == "voice"
        assert result.person_name == "Jacob"  # trusts voice
        assert result.voice_confirmed is True

    @pytest.mark.asyncio
    async def test_voice_match_out_of_fov(self):
        """Voice match, speaker out of FOV → voice-only, no visual conflict."""
        fusion, cloud = self._make_fusion()
        emb = _unit_vec(192, seed=42)
        centroid = _similar_vec(emb, noise=0.01)
        cloud.get_voice_centroids = AsyncMock(
            return_value={"p1": ("Jacob", centroid)}
        )

        person = self._active_person(
            "Person A", person_id="p2", person_name="Alice",
            confidence=0.9, bbox=(540, 100, 740, 500),
        )

        result = await fusion.fuse_speaker(
            speaker_label="SPEAKER_00",
            speaker_embedding=emb,
            active_persons={"Person A": person},
            doa_angle=180,  # behind camera → out of FOV
        )
        assert result.person_name == "Jacob"
        assert result.source == "voice"
        assert result.voice_confirmed is True
        assert result.visual_ref is None  # no visual association

    @pytest.mark.asyncio
    async def test_unknown_speaker(self):
        """No voice match → unknown speaker, not confirmed."""
        fusion, cloud = self._make_fusion()
        emb = _unit_vec(192, seed=1)
        centroid = -emb  # opposite
        cloud.get_voice_centroids = AsyncMock(
            return_value={"p1": ("Jacob", centroid)}
        )

        result = await fusion.fuse_speaker(
            speaker_label="SPEAKER_00",
            speaker_embedding=emb,
            active_persons={},
        )
        assert result.person_id is None
        assert result.voice_confirmed is False

    @pytest.mark.asyncio
    async def test_no_centroids_returns_unknown(self):
        """No voice centroids in database → unknown speaker."""
        fusion, cloud = self._make_fusion()
        cloud.get_voice_centroids = AsyncMock(return_value={})

        result = await fusion.fuse_speaker(
            speaker_label="SPEAKER_00",
            speaker_embedding=_unit_vec(192),
            active_persons={},
        )
        assert result.person_id is None
        assert result.voice_confirmed is False

    @pytest.mark.asyncio
    async def test_confirm_session_embeddings_writes(self):
        """Post-conversation confirmation writes confirmed embeddings."""
        from boxbot.perception.fusion import FusionResult

        fusion, cloud = self._make_fusion()
        cloud.add_visual_embedding = AsyncMock()
        cloud.add_voice_embedding = AsyncMock()
        cloud.recompute_centroid = AsyncMock()
        cloud.recompute_voice_centroid = AsyncMock()

        session_data = {
            "SPEAKER_00": {
                "fusion_result": FusionResult(
                    person_id="p1",
                    person_name="Jacob",
                    confidence=0.95,
                    source="fused",
                    voice_confirmed=True,
                    speaker_label="SPEAKER_00",
                    visual_ref="Person A",
                ),
                "voice_embeddings": [_unit_vec(192)],
                "visual_embeddings": [_unit_vec(512)],
            },
        }

        results = await fusion.confirm_session_embeddings(session_data, cloud)
        assert len(results) == 1
        assert results[0]["person_id"] == "p1"
        assert results[0]["visual_added"] == 1
        assert results[0]["voice_added"] == 1
        cloud.add_visual_embedding.assert_called_once()
        cloud.add_voice_embedding.assert_called_once()
        cloud.recompute_centroid.assert_called_once()
        cloud.recompute_voice_centroid.assert_called_once()

    @pytest.mark.asyncio
    async def test_confirm_skips_unconfirmed(self):
        """Unconfirmed speakers have nothing written."""
        from boxbot.perception.fusion import FusionResult

        fusion, cloud = self._make_fusion()
        cloud.add_visual_embedding = AsyncMock()
        cloud.add_voice_embedding = AsyncMock()

        session_data = {
            "SPEAKER_00": {
                "fusion_result": FusionResult(
                    person_id=None,
                    person_name=None,
                    confidence=0.3,
                    source="voice",
                    voice_confirmed=False,
                    speaker_label="SPEAKER_00",
                    visual_ref=None,
                ),
                "voice_embeddings": [_unit_vec(192)],
                "visual_embeddings": [_unit_vec(512)],
            },
        }

        results = await fusion.confirm_session_embeddings(session_data, cloud)
        assert len(results) == 0
        cloud.add_visual_embedding.assert_not_called()


# ---------------------------------------------------------------------------
# CropManager
# ---------------------------------------------------------------------------


class TestCropManager:
    """Tests for crop image retention and metadata."""

    def _make(self, tmp_path, **kwargs):
        from boxbot.perception.crops import CropManager

        return CropManager(base_path=str(tmp_path / "crops"), **kwargs)

    def test_save_crop_creates_files(self, tmp_path):
        mgr = self._make(tmp_path)
        image = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
        path = mgr.save_crop(
            image, ref="Person A", embedding_id="emb-123",
            label="Jacob", confidence=0.92, voice_confirmed=True,
        )
        assert os.path.exists(path)
        # Check sidecar JSON
        meta_path = Path(path).with_suffix(".json")
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["label"] == "Jacob"
        assert meta["confidence"] == 0.92
        assert meta["voice_confirmed"] is True
        assert meta["ref"] == "Person A"

    def test_get_crop_metadata(self, tmp_path):
        mgr = self._make(tmp_path)
        image = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
        path = mgr.save_crop(
            image, "Person A", "emb-1", "Jacob", 0.9, True,
        )
        meta = mgr.get_crop_metadata(path)
        assert meta is not None
        assert meta["embedding_id"] == "emb-1"

    def test_get_crop_metadata_missing(self, tmp_path):
        mgr = self._make(tmp_path)
        assert mgr.get_crop_metadata("/nonexistent/path.jpg") is None

    def test_prune_expired_no_crops(self, tmp_path):
        mgr = self._make(tmp_path)
        assert mgr.prune_expired() == 0

    def test_prune_keeps_recent(self, tmp_path):
        mgr = self._make(tmp_path, retention_days=30)
        image = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
        mgr.save_crop(image, "A", "e1", "Jacob", 0.9, True)
        deleted = mgr.prune_expired()
        assert deleted == 0

    def test_date_directory_structure(self, tmp_path):
        mgr = self._make(tmp_path)
        image = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
        path = mgr.save_crop(image, "A", "e1", "Jacob", 0.9, True)
        # Path should contain YYYY-MM-DD directory
        parts = Path(path).parts
        # Find the date-like directory
        date_dir = [p for p in parts if len(p) == 10 and p[4] == "-"]
        assert len(date_dir) == 1


# ---------------------------------------------------------------------------
# State Machine — Conversation States
# ---------------------------------------------------------------------------


class TestStateMachineConversation:
    """Tests for CONVERSATION and POST_CONVERSATION state transitions."""

    def _make(self, **kwargs):
        from boxbot.perception.state_machine import PerceptionStateMachine

        return PerceptionStateMachine(**kwargs)

    def _detection(self, **kwargs):
        from boxbot.perception.person_detector import Detection

        defaults = {"bbox": (100, 100, 300, 500), "confidence": 0.9, "class_id": 0}
        defaults.update(kwargs)
        return Detection(**defaults)

    def _to_detected(self, sm):
        """Drive state machine to DETECTED."""
        sm.on_motion(15.0, 12.0)
        sm.on_person_detected([self._detection()])
        return sm

    def test_detected_to_conversation(self):
        from boxbot.perception.state_machine import PerceptionState

        sm = self._to_detected(self._make())
        state = sm.on_conversation_started()
        assert state == PerceptionState.CONVERSATION

    def test_conversation_from_dormant_noop(self):
        from boxbot.perception.state_machine import PerceptionState

        sm = self._make()
        state = sm.on_conversation_started()
        # Should stay DORMANT — conversation only starts from DETECTED
        assert state == PerceptionState.DORMANT

    def test_conversation_to_post_conversation(self):
        from boxbot.perception.state_machine import PerceptionState

        sm = self._to_detected(self._make())
        sm.on_conversation_started()
        state = sm.on_conversation_ended()
        assert state == PerceptionState.POST_CONVERSATION

    def test_post_conversation_to_dormant(self):
        from boxbot.perception.state_machine import PerceptionState

        sm = self._to_detected(self._make())
        sm.on_conversation_started()
        sm.on_conversation_ended()
        state = sm.on_post_conversation_done(people_still_present=False)
        assert state == PerceptionState.DORMANT
        assert len(sm.active_persons) == 0

    def test_post_conversation_to_detected(self):
        from boxbot.perception.state_machine import PerceptionState

        sm = self._to_detected(self._make())
        sm.on_conversation_started()
        sm.on_conversation_ended()
        state = sm.on_post_conversation_done(people_still_present=True)
        assert state == PerceptionState.DETECTED

    def test_heartbeat_not_in_conversation(self):
        sm = self._to_detected(self._make(heartbeat_interval=0.0))
        sm.on_conversation_started()
        # Should NOT heartbeat in CONVERSATION state
        assert sm.should_heartbeat() is False

    def test_timeout_not_in_conversation(self):
        from boxbot.perception.state_machine import PerceptionState

        sm = self._to_detected(self._make(presence_timeout=0.0))
        sm.on_conversation_started()
        # Timeout only applies in DETECTED, not CONVERSATION
        state = sm.check_timeout()
        assert state == PerceptionState.CONVERSATION

    def test_full_lifecycle(self):
        """DORMANT→CHECKING→DETECTED→CONVERSATION→POST_CONVERSATION→DORMANT."""
        from boxbot.perception.state_machine import PerceptionState

        sm = self._make()

        assert sm.state == PerceptionState.DORMANT
        sm.on_motion(15.0, 12.0)
        assert sm.state == PerceptionState.CHECKING
        sm.on_person_detected([self._detection()])
        assert sm.state == PerceptionState.DETECTED
        sm.on_conversation_started()
        assert sm.state == PerceptionState.CONVERSATION
        sm.on_conversation_ended()
        assert sm.state == PerceptionState.POST_CONVERSATION
        sm.on_post_conversation_done(people_still_present=False)
        assert sm.state == PerceptionState.DORMANT


# ---------------------------------------------------------------------------
# CloudStore — Voice Centroids
# ---------------------------------------------------------------------------


class TestCloudStoreVoiceCentroids:
    """Tests for voice centroid storage and retrieval."""

    @pytest_asyncio.fixture
    async def store(self, tmp_path):
        from boxbot.perception.clouds import CloudStore

        s = CloudStore(db_path=tmp_path / "voice_test.db")
        await s.initialize()
        yield s
        await s.close()

    @pytest.mark.asyncio
    async def test_add_and_get_voice_embeddings(self, store):
        pid = await store.create_person("Jacob")
        emb = _unit_vec(192)
        emb_id = await store.add_voice_embedding(pid, emb)
        assert emb_id

        embeddings = await store.get_voice_embeddings(pid)
        assert len(embeddings) == 1
        np.testing.assert_array_almost_equal(embeddings[0][1], emb)

    @pytest.mark.asyncio
    async def test_recompute_voice_centroid(self, store):
        pid = await store.create_person("Jacob")
        for seed in range(5):
            await store.add_voice_embedding(pid, _unit_vec(192, seed=seed))

        centroid = await store.recompute_voice_centroid(pid)
        assert centroid is not None
        assert centroid.shape == (192,)
        assert np.linalg.norm(centroid) == pytest.approx(1.0, abs=1e-5)

    @pytest.mark.asyncio
    async def test_get_voice_centroids(self, store):
        pid = await store.create_person("Jacob")
        for seed in range(3):
            await store.add_voice_embedding(pid, _unit_vec(192, seed=seed))
        await store.recompute_voice_centroid(pid)

        centroids = await store.get_voice_centroids()
        assert pid in centroids
        name, vec = centroids[pid]
        assert name == "Jacob"
        assert vec.shape == (192,)

    @pytest.mark.asyncio
    async def test_get_voice_centroids_empty(self, store):
        """No persons → empty dict."""
        centroids = await store.get_voice_centroids()
        assert centroids == {}

    @pytest.mark.asyncio
    async def test_voice_embedding_cap_enforcement(self, store):
        pid = await store.create_person("Jacob")
        cap = 5
        for seed in range(cap + 3):
            await store.add_voice_embedding(pid, _unit_vec(192, seed=seed))
            await store._enforce_voice_cap(pid, max_count=cap)

        embeddings = await store.get_voice_embeddings(pid)
        assert len(embeddings) <= cap

    @pytest.mark.asyncio
    async def test_visual_embedding_with_crop_path(self, store):
        pid = await store.create_person("Jacob")
        emb = _unit_vec(512)
        emb_id = await store.add_visual_embedding(
            pid, emb, voice_confirmed=True, crop_path="/data/crops/test.jpg"
        )
        assert emb_id


# ---------------------------------------------------------------------------
# Transcript Attribution
# ---------------------------------------------------------------------------


class TestTranscriptAttribution:
    """Tests for VoiceSession._build_attributed_transcript with speaker identities."""

    def _make_stt_result(self, text, words=None):
        from boxbot.communication.stt import STTResult, WordInfo

        word_infos = None
        if words:
            word_infos = [
                WordInfo(word=w, start=s, end=e) for w, s, e in words
            ]
        return STTResult(text=text, language="en", words=word_infos or [])

    def _make_diarization_result(self, segments):
        """Create a mock diarization result with segments."""
        result = MagicMock()
        seg_objects = []
        for seg_dict in segments:
            seg = MagicMock()
            seg.speaker_label = seg_dict["speaker"]
            seg.start = seg_dict["start"]
            seg.end = seg_dict["end"]
            seg.embedding = seg_dict.get("embedding")
            seg_objects.append(seg)
        result.segments = seg_objects
        return result

    def test_no_diarization_returns_plain(self):
        from boxbot.communication.voice import VoiceSession

        stt = self._make_stt_result("Hello there")
        transcript = VoiceSession._build_attributed_transcript(stt, None)
        assert transcript == "Hello there"

    def test_with_diarization_and_words(self):
        from boxbot.communication.voice import VoiceSession

        stt = self._make_stt_result(
            "Hello there how are you",
            words=[
                ("Hello", 0.0, 0.5),
                ("there", 0.6, 1.0),
                ("how", 2.0, 2.3),
                ("are", 2.4, 2.6),
                ("you", 2.7, 3.0),
            ],
        )
        diar = self._make_diarization_result([
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.5},
            {"speaker": "SPEAKER_01", "start": 1.8, "end": 3.5},
        ])

        transcript = VoiceSession._build_attributed_transcript(stt, diar)
        assert "[SPEAKER_00]:" in transcript
        assert "[SPEAKER_01]:" in transcript
        assert "Hello" in transcript

    def test_speaker_identity_replacement(self):
        from boxbot.communication.voice import VoiceSession

        stt = self._make_stt_result(
            "Hello there",
            words=[("Hello", 0.0, 0.5), ("there", 0.6, 1.0)],
        )
        diar = self._make_diarization_result([
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.5},
        ])

        identities = {"SPEAKER_00": "Jacob"}
        transcript = VoiceSession._build_attributed_transcript(
            stt, diar, speaker_identities=identities,
        )
        assert "[Jacob]:" in transcript
        assert "[SPEAKER_00]" not in transcript

    def test_partial_identity_replacement(self):
        """Only identified speakers get replaced; unknown keep their label."""
        from boxbot.communication.voice import VoiceSession

        stt = self._make_stt_result(
            "Hi yo",
            words=[("Hi", 0.0, 0.5), ("yo", 2.0, 2.5)],
        )
        diar = self._make_diarization_result([
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
            {"speaker": "SPEAKER_01", "start": 1.5, "end": 3.0},
        ])

        identities = {"SPEAKER_00": "Jacob"}  # SPEAKER_01 unknown
        transcript = VoiceSession._build_attributed_transcript(
            stt, diar, speaker_identities=identities,
        )
        assert "[Jacob]:" in transcript
        assert "[SPEAKER_01]:" in transcript

    def test_no_words_with_identity(self):
        """No word-level timing — prefix with first speaker's identity."""
        from boxbot.communication.voice import VoiceSession

        stt = self._make_stt_result("Hello there")
        stt_words_empty = self._make_stt_result("Hello there", words=[])
        # STTResult with no words
        diar = self._make_diarization_result([
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
        ])

        identities = {"SPEAKER_00": "Jacob"}
        transcript = VoiceSession._build_attributed_transcript(
            stt_words_empty, diar, speaker_identities=identities,
        )
        assert "[Jacob]:" in transcript


# ---------------------------------------------------------------------------
# Pipeline — Conversation Event Handlers
# ---------------------------------------------------------------------------


class TestPipelineConversationHandlers:
    """Tests for pipeline event handling during conversation lifecycle."""

    @pytest_asyncio.fixture
    async def pipeline_setup(self, tmp_path):
        from boxbot.perception.clouds import CloudStore
        from boxbot.perception.pipeline import PerceptionPipeline

        store = CloudStore(db_path=tmp_path / "pipeline_conv.db")
        await store.initialize()

        camera = MagicMock()
        camera.get_lores_frame = AsyncMock(
            return_value=np.zeros((240, 320), dtype=np.uint8)
        )
        camera.capture_frame = AsyncMock(
            return_value=np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        )

        hailo = MagicMock()
        hailo.infer = AsyncMock(
            return_value={"output": np.zeros((2, 5, 80), dtype=np.float32)}
        )

        p = PerceptionPipeline(
            camera=camera, hailo=hailo, cloud_store=store, scan_fps=10,
        )
        await p.start()
        yield p, store
        await p.stop()
        await store.close()

    @pytest.mark.asyncio
    async def test_conversation_started_transitions_state(self, pipeline_setup):
        from boxbot.core.events import ConversationStarted
        from boxbot.perception.state_machine import PerceptionState

        pipeline, store = pipeline_setup

        # Drive to DETECTED first
        pipeline._state_machine.on_motion(15.0, 12.0)
        from boxbot.perception.person_detector import Detection
        det = Detection(bbox=(100, 100, 300, 500), confidence=0.9, class_id=0)
        pipeline._state_machine.on_person_detected([det])
        assert pipeline.state == PerceptionState.DETECTED

        # Fire conversation started
        event = ConversationStarted(conversation_id="test-conv-1", channel="voice")
        await pipeline._on_conversation_started(event)
        assert pipeline.state == PerceptionState.CONVERSATION
        assert pipeline._conversation_active is True

    @pytest.mark.asyncio
    async def test_conversation_ended_triggers_post(self, pipeline_setup):
        from boxbot.core.events import ConversationEnded, ConversationStarted
        from boxbot.perception.state_machine import PerceptionState

        pipeline, store = pipeline_setup

        # Drive to DETECTED → CONVERSATION
        pipeline._state_machine.on_motion(15.0, 12.0)
        from boxbot.perception.person_detector import Detection
        det = Detection(bbox=(100, 100, 300, 500), confidence=0.9, class_id=0)
        pipeline._state_machine.on_person_detected([det])
        await pipeline._on_conversation_started(
            ConversationStarted(conversation_id="c1", channel="voice")
        )

        # End conversation
        await pipeline._on_conversation_ended(
            ConversationEnded(conversation_id="c1", channel="voice")
        )
        assert pipeline._conversation_active is False
        # Post-conversation done → should be DETECTED or DORMANT
        assert pipeline.state in (
            PerceptionState.DETECTED, PerceptionState.DORMANT,
        )

    @pytest.mark.asyncio
    async def test_transcript_ready_with_speaker_embedding(self, pipeline_setup):
        from boxbot.core.events import ConversationStarted, SpeakerIdentified, TranscriptReady
        from boxbot.perception.state_machine import PerceptionState

        pipeline, store = pipeline_setup

        # Create a person with voice centroid
        pid = await store.create_person("Jacob")
        voice_emb = _unit_vec(192, seed=42)
        await store.add_voice_embedding(pid, voice_emb)
        await store.recompute_voice_centroid(pid)

        # Drive to CONVERSATION
        pipeline._state_machine.on_motion(15.0, 12.0)
        from boxbot.perception.person_detector import Detection
        det = Detection(bbox=(100, 100, 300, 500), confidence=0.9, class_id=0)
        pipeline._state_machine.on_person_detected([det])
        await pipeline._on_conversation_started(
            ConversationStarted(conversation_id="c1", channel="voice")
        )

        # Capture published events
        published = []
        from boxbot.core.events import get_event_bus
        bus = get_event_bus()

        async def capture_speaker(event):
            published.append(event)

        bus.subscribe(SpeakerIdentified, capture_speaker)

        try:
            # Process transcript with speaker embedding
            speaker_emb = _similar_vec(voice_emb, noise=0.01)
            event = TranscriptReady(
                conversation_id="c1",
                transcript="[SPEAKER_00]: Hello",
                speaker_segments=[
                    {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0,
                     "embedding": speaker_emb},
                ],
            )
            await pipeline._on_transcript_ready(event)

            # Should have published SpeakerIdentified
            assert len(published) == 1
            assert published[0].person_name == "Jacob"
            assert published[0].person_id == pid
            assert published[0].speaker_label == "SPEAKER_00"
        finally:
            bus.unsubscribe(SpeakerIdentified, capture_speaker)


# ---------------------------------------------------------------------------
# SpeakerIdentified Event Dataclass
# ---------------------------------------------------------------------------


class TestSpeakerIdentifiedEvent:
    """Tests for the SpeakerIdentified event."""

    def test_defaults(self):
        from boxbot.core.events import SpeakerIdentified

        event = SpeakerIdentified()
        assert event.speaker_label == ""
        assert event.person_id == ""
        assert event.person_name == ""
        assert event.confidence == 0.0
        assert event.source == "voice"

    def test_with_values(self):
        from boxbot.core.events import SpeakerIdentified

        event = SpeakerIdentified(
            speaker_label="SPEAKER_00",
            person_id="p1",
            person_name="Jacob",
            confidence=0.92,
        )
        assert event.speaker_label == "SPEAKER_00"
        assert event.person_name == "Jacob"
        assert event.confidence == 0.92

    def test_frozen(self):
        from boxbot.core.events import SpeakerIdentified

        event = SpeakerIdentified()
        with pytest.raises(AttributeError):
            event.person_name = "test"


# ---------------------------------------------------------------------------
# FusionResult Dataclass
# ---------------------------------------------------------------------------


class TestFusionResult:
    """Tests for FusionResult dataclass."""

    def test_creation(self):
        from boxbot.perception.fusion import FusionResult

        result = FusionResult(
            person_id="p1",
            person_name="Jacob",
            confidence=0.95,
            source="fused",
            voice_confirmed=True,
            speaker_label="SPEAKER_00",
            visual_ref="Person A",
        )
        assert result.person_id == "p1"
        assert result.person_name == "Jacob"
        assert result.voice_confirmed is True
        assert result.source == "fused"

    def test_unknown_result(self):
        from boxbot.perception.fusion import FusionResult

        result = FusionResult(
            person_id=None,
            person_name=None,
            confidence=0.2,
            source="voice",
            voice_confirmed=False,
            speaker_label="SPEAKER_01",
            visual_ref=None,
        )
        assert result.person_id is None
        assert result.voice_confirmed is False


# ---------------------------------------------------------------------------
# Config — New Phase B Fields
# ---------------------------------------------------------------------------


class TestPhaseB_Config:
    """Verify new PerceptionConfig fields have correct defaults."""

    def test_doa_fields(self):
        from boxbot.core.config import PerceptionConfig

        cfg = PerceptionConfig()
        assert cfg.doa_forward_angle == 0
        assert cfg.camera_hfov == 120
        assert cfg.voice_match_threshold == 0.60
