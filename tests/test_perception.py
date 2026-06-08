"""Tests for the perception pipeline — motion, detection, ReID, clouds, enrollment, state machine, and pipeline."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# Motion Detector
# ---------------------------------------------------------------------------


class TestMotionDetector:
    """Tests for CPU motion detection (frame differencing)."""

    def _make_detector(self, **kwargs):
        from boxbot.perception.motion import MotionDetector

        return MotionDetector(**kwargs)

    def test_first_frame_returns_zero(self):
        det = self._make_detector()
        frame = np.zeros((240, 320), dtype=np.uint8)
        assert det.detect(frame) == 0.0

    def test_identical_frames_return_zero(self):
        det = self._make_detector()
        frame = np.ones((240, 320), dtype=np.uint8) * 128
        det.detect(frame)  # seed
        assert det.detect(frame.copy()) == 0.0

    def test_different_frames_return_positive(self):
        det = self._make_detector()
        frame1 = np.zeros((240, 320), dtype=np.uint8)
        frame2 = np.ones((240, 320), dtype=np.uint8) * 100
        det.detect(frame1)
        score = det.detect(frame2)
        assert score > 0.0

    def test_threshold_property(self):
        det = self._make_detector(threshold=20.0)
        assert det.threshold == 20.0
        det.threshold = 15.0
        assert det.threshold == 15.0

    def test_reset_clears_previous(self):
        det = self._make_detector()
        frame = np.ones((240, 320), dtype=np.uint8) * 128
        det.detect(frame)
        det.reset()
        # After reset, next frame should return 0.0 (no previous)
        assert det.detect(frame) == 0.0

    def test_subtle_change_below_threshold(self):
        det = self._make_detector(threshold=50.0)
        frame1 = np.ones((240, 320), dtype=np.uint8) * 100
        frame2 = np.ones((240, 320), dtype=np.uint8) * 105
        det.detect(frame1)
        score = det.detect(frame2)
        assert score < det.threshold

    def test_large_change_above_threshold(self):
        det = self._make_detector(threshold=10.0)
        frame1 = np.zeros((240, 320), dtype=np.uint8)
        frame2 = np.ones((240, 320), dtype=np.uint8) * 200
        det.detect(frame1)
        score = det.detect(frame2)
        assert score > det.threshold


# ---------------------------------------------------------------------------
# Person Detector (pre/post-processing)
# ---------------------------------------------------------------------------


class TestPersonDetector:
    """Tests for YOLO pre/post-processing."""

    def _make_detector(self, **kwargs):
        from boxbot.perception.person_detector import PersonDetector

        return PersonDetector(**kwargs)

    def test_preprocess_output_shape(self):
        det = self._make_detector()
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        preprocessed, params = det.preprocess(frame)
        assert preprocessed.shape == (1, 640, 640, 3)
        assert preprocessed.dtype == np.float32
        # HEF quant is scale=1.0 zp=0; float32 must stay in native uint8 range.
        assert preprocessed.min() >= 0.0
        assert preprocessed.max() <= 255.0
        assert preprocessed.max() > 1.0  # guard against the /255 regression

    def test_preprocess_letterbox_params(self):
        det = self._make_detector()
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        _, params = det.preprocess(frame)
        assert params.scale > 0
        assert params.pad_x >= 0
        assert params.pad_y >= 0

    # Layout per the on-device diagnostic dump (yolov5s_personface_h8l):
    # outputs[<key>] = [batch_dim_len_1[per_class_len_2[ndarray(N,5)]]]
    # Each row: [y_min, x_min, y_max, x_max, score], normalized [0, 1]
    # in the model's 640×640 letterboxed input frame.

    @staticmethod
    def _hailo_outputs(
        person_rows: list[list[float]] | None = None,
        face_rows: list[list[float]] | None = None,
    ) -> dict[str, list]:
        """Build a Hailo-shaped output dict from per-class detection rows."""
        person = np.array(
            person_rows or [], dtype=np.float32
        ).reshape(-1, 5)
        face = np.array(
            face_rows or [], dtype=np.float32
        ).reshape(-1, 5)
        return {"yolov5s_personface/yolov5_nms_postprocess": [[person, face]]}

    def test_postprocess_empty_output(self):
        det = self._make_detector()
        _, params = det.preprocess(np.zeros((720, 1280, 3), dtype=np.uint8))
        outputs = self._hailo_outputs()  # both classes empty
        detections = det.postprocess(outputs, params, (720, 1280))
        assert detections == []

    def test_postprocess_with_person_detection(self):
        det = self._make_detector(confidence_threshold=0.3)
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        _, params = det.preprocess(frame)

        # Person filling left half of frame: y in [0.1, 0.9], x in [0.1, 0.5]
        outputs = self._hailo_outputs(
            person_rows=[[0.1, 0.1, 0.9, 0.5, 0.85]],
        )

        detections = det.postprocess(outputs, params, (720, 1280), frame)
        assert len(detections) == 1
        assert detections[0].confidence == pytest.approx(0.85)
        assert detections[0].class_id == 0  # person
        # Sanity: bbox should be inside the frame
        x1, y1, x2, y2 = detections[0].bbox
        assert 0 <= x1 < x2 <= 1280
        assert 0 <= y1 < y2 <= 720

    def test_postprocess_filters_low_confidence(self):
        det = self._make_detector(confidence_threshold=0.5)
        _, params = det.preprocess(np.zeros((720, 1280, 3), dtype=np.uint8))

        outputs = self._hailo_outputs(
            person_rows=[[0.1, 0.1, 0.5, 0.4, 0.3]],  # below threshold
        )
        detections = det.postprocess(outputs, params, (720, 1280))
        assert len(detections) == 0

    def test_postprocess_filters_face_class(self):
        det = self._make_detector(confidence_threshold=0.3)
        _, params = det.preprocess(np.zeros((720, 1280, 3), dtype=np.uint8))

        outputs = self._hailo_outputs(
            face_rows=[[0.1, 0.1, 0.3, 0.3, 0.9]],  # high-confidence face
        )
        detections = det.postprocess(outputs, params, (720, 1280))
        # postprocess returns person-class only
        assert len(detections) == 0

    def test_extract_reid_crops(self):
        from boxbot.perception.person_detector import Detection

        det = self._make_detector()
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        detections = [
            Detection(bbox=(100, 100, 400, 600), confidence=0.9, class_id=0)
        ]
        crops = det.extract_reid_crops(frame, detections)
        assert len(crops) == 1
        assert crops[0].shape == (1, 128, 256, 3)
        assert crops[0].dtype == np.float32


# ---------------------------------------------------------------------------
# Visual ReID
# ---------------------------------------------------------------------------


class TestVisualReID:
    """Tests for embedding normalization and matching."""

    def _make_reid(self, **kwargs):
        from boxbot.perception.visual_reid import VisualReID

        return VisualReID(**kwargs)

    def test_normalize_embedding(self):
        reid = self._make_reid()
        raw = np.random.randn(1, 512).astype(np.float32)
        emb = reid.normalize_embedding(raw)
        assert emb.shape == (512,)
        assert emb.dtype == np.float32
        assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-5)

    def test_normalize_zero_vector(self):
        reid = self._make_reid()
        raw = np.zeros(512, dtype=np.float32)
        emb = reid.normalize_embedding(raw)
        assert np.all(emb == 0)

    def test_cosine_similarity_identical(self):
        from boxbot.perception.visual_reid import VisualReID

        vec = np.random.randn(512).astype(np.float32)
        vec /= np.linalg.norm(vec)
        assert VisualReID.cosine_similarity(vec, vec) == pytest.approx(1.0, abs=1e-5)

    def test_cosine_similarity_orthogonal(self):
        from boxbot.perception.visual_reid import VisualReID

        a = np.zeros(512, dtype=np.float32)
        b = np.zeros(512, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        assert VisualReID.cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_match_high_confidence(self):
        reid = self._make_reid()
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        # Use almost-identical centroid
        centroid = emb + np.random.randn(512).astype(np.float32) * 0.01
        centroid /= np.linalg.norm(centroid)

        centroids = {"p1": ("Jacob", centroid)}
        result = reid.match(emb, centroids)
        assert result.tier == "high"
        assert result.person_name == "Jacob"
        assert result.person_id == "p1"

    def test_match_unknown(self):
        reid = self._make_reid()
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        # Use very different centroid
        centroid = -emb  # opposite direction
        centroids = {"p1": ("Jacob", centroid)}
        result = reid.match(emb, centroids)
        assert result.tier == "unknown"
        assert result.person_id is None

    def test_match_empty_centroids(self):
        reid = self._make_reid()
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        result = reid.match(emb, {})
        assert result.tier == "unknown"

    def test_compute_centroid(self):
        from boxbot.perception.visual_reid import VisualReID

        embeddings = [
            np.random.randn(512).astype(np.float32) for _ in range(10)
        ]
        for i in range(len(embeddings)):
            embeddings[i] /= np.linalg.norm(embeddings[i])

        centroid = VisualReID.compute_centroid(embeddings)
        assert centroid.shape == (512,)
        assert np.linalg.norm(centroid) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Cloud Store
# ---------------------------------------------------------------------------


class TestCloudStore:
    """Tests for embedding persistence in SQLite."""

    @pytest_asyncio.fixture
    async def store(self, tmp_path):
        from boxbot.perception.clouds import CloudStore

        db_path = tmp_path / "test_perception.db"
        store = CloudStore(db_path=db_path)
        await store.initialize()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_create_and_get_person(self, store):
        person_id = await store.create_person("Jacob")
        assert person_id is not None

        person = await store.get_person(person_id)
        assert person is not None
        assert person["name"] == "Jacob"
        assert person["is_user"] is False

    @pytest.mark.asyncio
    async def test_get_person_by_name(self, store):
        await store.create_person("Sarah")
        person = await store.get_person_by_name("Sarah")
        assert person is not None
        assert person["name"] == "Sarah"

    @pytest.mark.asyncio
    async def test_get_nonexistent_person(self, store):
        assert await store.get_person_by_name("Nobody") is None

    @pytest.mark.asyncio
    async def test_list_persons(self, store):
        await store.create_person("Alice")
        await store.create_person("Bob")
        persons = await store.list_persons()
        assert len(persons) == 2

    @pytest.mark.asyncio
    async def test_add_and_get_visual_embedding(self, store):
        pid = await store.create_person("Jacob")
        emb = np.random.randn(512).astype(np.float32)
        emb_id = await store.add_visual_embedding(pid, emb)
        assert emb_id is not None

        embeddings = await store.get_visual_embeddings(pid)
        assert len(embeddings) == 1
        assert embeddings[0][0] == emb_id
        np.testing.assert_array_almost_equal(embeddings[0][1], emb)

    @pytest.mark.asyncio
    async def test_visual_embedding_cap_enforcement(self, store):
        pid = await store.create_person("Jacob")
        # Add more than cap (use small cap for test speed)
        cap = 5
        for i in range(cap + 3):
            emb = np.random.randn(512).astype(np.float32)
            await store.add_visual_embedding(pid, emb)
            await store._enforce_visual_cap(pid, max_count=cap)

        count = await store.count_visual_embeddings(pid)
        assert count <= cap

    @pytest.mark.asyncio
    async def test_centroids(self, store):
        pid = await store.create_person("Jacob")
        emb1 = np.random.randn(512).astype(np.float32)
        emb2 = np.random.randn(512).astype(np.float32)
        await store.add_visual_embedding(pid, emb1)
        await store.add_visual_embedding(pid, emb2)

        centroid = await store.recompute_centroid(pid)
        assert centroid is not None
        assert centroid.shape == (512,)
        assert np.linalg.norm(centroid) == pytest.approx(1.0, abs=1e-5)

        all_centroids = await store.get_centroids()
        assert pid in all_centroids
        assert all_centroids[pid][0] == "Jacob"

    @pytest.mark.asyncio
    async def test_voice_embeddings(self, store):
        pid = await store.create_person("Jacob")
        emb = np.random.randn(192).astype(np.float32)
        emb_id = await store.add_voice_embedding(pid, emb)
        assert emb_id is not None

        embeddings = await store.get_voice_embeddings(pid)
        assert len(embeddings) == 1
        np.testing.assert_array_almost_equal(embeddings[0][1], emb)

    @pytest.mark.asyncio
    async def test_update_last_seen(self, store):
        pid = await store.create_person("Jacob")
        person_before = await store.get_person(pid)
        await store.update_last_seen(pid)
        person_after = await store.get_person(pid)
        # last_seen should be updated (at least not earlier)
        assert person_after["last_seen"] >= person_before["last_seen"]


# ---------------------------------------------------------------------------
# Enrollment Manager
# ---------------------------------------------------------------------------


class TestEnrollmentManager:
    """Tests for session-based enrollment."""

    @pytest_asyncio.fixture
    async def cloud_store(self, tmp_path):
        from boxbot.perception.clouds import CloudStore

        store = CloudStore(db_path=tmp_path / "enroll.db")
        await store.initialize()
        yield store
        await store.close()

    @pytest_asyncio.fixture
    async def manager(self, cloud_store):
        from boxbot.perception.enrollment import EnrollmentManager

        return EnrollmentManager(cloud_store)

    def test_buffer_embedding(self, manager):
        emb = np.random.randn(512).astype(np.float32)
        manager.buffer_embedding("Person A", emb)
        assert "Person A" in manager.get_session_refs()

    def test_buffer_multiple_embeddings(self, manager):
        for _ in range(5):
            emb = np.random.randn(512).astype(np.float32)
            manager.buffer_embedding("Person A", emb)
        session = manager.get_session_person("Person A")
        assert session is not None
        assert len(session.visual_embeddings) == 5

    @pytest.mark.asyncio
    async def test_identify_creates_new_person(self, manager, cloud_store):
        emb = np.random.randn(512).astype(np.float32)
        manager.buffer_embedding("Person A", emb)

        result = await manager.identify("Jacob", "Person A")
        assert result["status"] == "ok"
        assert result["outcome"] == "create"
        assert result["name"] == "Jacob"
        assert result["embeddings_buffered"] == 1

        # Verify person exists in store
        person = await cloud_store.get_person_by_name("Jacob")
        assert person is not None

    @pytest.mark.asyncio
    async def test_identify_links_existing_person(self, manager, cloud_store):
        await cloud_store.create_person("Jacob")
        emb = np.random.randn(512).astype(np.float32)
        manager.buffer_embedding("Person B", emb)

        result = await manager.identify("Jacob", "Person B")
        assert result["status"] == "ok"
        assert result["outcome"] == "confirm"

    @pytest.mark.asyncio
    async def test_identify_unknown_ref_raises(self, manager):
        with pytest.raises(ValueError, match="Session ref"):
            await manager.identify("Jacob", "Person Z")

    def test_clear_session(self, manager):
        manager.buffer_embedding("Person A", np.zeros(512, dtype=np.float32))
        manager.clear_session()
        assert manager.get_session_refs() == []


# ---------------------------------------------------------------------------
# State Machine
# ---------------------------------------------------------------------------


class TestPerceptionStateMachine:
    """Tests for perception state transitions."""

    def _make_sm(self, **kwargs):
        from boxbot.perception.state_machine import PerceptionStateMachine

        return PerceptionStateMachine(**kwargs)

    def _make_detection(self, **kwargs):
        from boxbot.perception.person_detector import Detection

        defaults = {"bbox": (100, 100, 300, 500), "confidence": 0.9, "class_id": 0}
        defaults.update(kwargs)
        return Detection(**defaults)

    def test_initial_state_dormant(self):
        from boxbot.perception.state_machine import PerceptionState

        sm = self._make_sm()
        assert sm.state == PerceptionState.DORMANT

    def test_motion_below_threshold_stays_dormant(self):
        from boxbot.perception.state_machine import PerceptionState

        sm = self._make_sm()
        new_state = sm.on_motion(5.0, 12.0)
        assert new_state == PerceptionState.DORMANT

    def test_motion_above_threshold_to_checking(self):
        from boxbot.perception.state_machine import PerceptionState

        sm = self._make_sm()
        new_state = sm.on_motion(15.0, 12.0)
        assert new_state == PerceptionState.CHECKING

    def test_no_person_returns_to_dormant(self):
        from boxbot.perception.state_machine import PerceptionState

        sm = self._make_sm()
        sm.on_motion(15.0, 12.0)
        assert sm.state == PerceptionState.CHECKING
        new_state = sm.on_person_detected([])
        assert new_state == PerceptionState.DORMANT

    def test_person_detected_transitions_to_detected(self):
        from boxbot.perception.state_machine import PerceptionState

        sm = self._make_sm()
        sm.on_motion(15.0, 12.0)
        det = self._make_detection()
        new_state = sm.on_person_detected([det])
        assert new_state == PerceptionState.DETECTED

    def test_detected_state_tracks_active_persons(self):
        sm = self._make_sm()
        sm.on_motion(15.0, 12.0)
        det = self._make_detection()
        sm.on_person_detected([det])
        assert len(sm.active_persons) == 1

    def test_get_present_people(self):
        from boxbot.perception.visual_reid import MatchResult

        sm = self._make_sm()
        sm.on_motion(15.0, 12.0)
        det = self._make_detection()
        sm.on_person_detected([det])

        ref = list(sm.active_persons.keys())[0]
        sm.on_identification(
            ref, MatchResult("p1", "Jacob", 0.92, "high")
        )

        people = sm.get_present_people()
        assert len(people) == 1
        assert people[0]["name"] == "Jacob"
        assert people[0]["confidence"] == 0.92

    def test_heartbeat_scheduling(self):
        sm = self._make_sm(heartbeat_interval=0.0)
        sm.on_motion(15.0, 12.0)
        sm.on_person_detected([self._make_detection()])
        # Should need heartbeat immediately with 0 interval
        assert sm.should_heartbeat()
        sm.record_heartbeat()

    def test_timeout_returns_to_dormant(self):
        from boxbot.perception.state_machine import PerceptionState

        sm = self._make_sm(presence_timeout=0.0)
        sm.on_motion(15.0, 12.0)
        sm.on_person_detected([self._make_detection()])
        assert sm.state == PerceptionState.DETECTED

        # With timeout=0, should immediately timeout
        new_state = sm.check_timeout()
        assert new_state == PerceptionState.DORMANT

    def test_reset(self):
        from boxbot.perception.state_machine import PerceptionState

        sm = self._make_sm()
        sm.on_motion(15.0, 12.0)
        sm.on_person_detected([self._make_detection()])
        sm.reset()
        assert sm.state == PerceptionState.DORMANT
        assert len(sm.active_persons) == 0


# ---------------------------------------------------------------------------
# Pipeline (with mocked HAL)
# ---------------------------------------------------------------------------


class TestPerceptionPipeline:
    """Tests for the pipeline orchestrator with mocked HAL modules."""

    def _make_mock_camera(self):
        camera = MagicMock()
        camera.get_lores_frame = AsyncMock(
            return_value=np.zeros((240, 320), dtype=np.uint8)
        )
        camera.capture_frame = AsyncMock(
            return_value=np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        )
        return camera

    def _make_mock_hailo(self):
        hailo = MagicMock()
        # YOLO output: no detections
        hailo.infer = AsyncMock(
            return_value={"output": np.zeros((2, 5, 80), dtype=np.float32)}
        )
        return hailo

    @pytest_asyncio.fixture
    async def pipeline(self, tmp_path):
        from boxbot.perception.pipeline import PerceptionPipeline
        from boxbot.perception.clouds import CloudStore

        store = CloudStore(db_path=tmp_path / "pipeline.db")
        await store.initialize()

        camera = self._make_mock_camera()
        hailo = self._make_mock_hailo()

        p = PerceptionPipeline(
            camera=camera,
            hailo=hailo,
            cloud_store=store,
            scan_fps=10,
        )
        await p.start()
        yield p
        await p.stop()
        await store.close()

    @pytest.mark.asyncio
    async def test_pipeline_starts_and_stops(self, pipeline):
        from boxbot.perception.state_machine import PerceptionState

        assert pipeline.state == PerceptionState.DORMANT

    @pytest.mark.asyncio
    async def test_pipeline_get_present_people_empty(self, pipeline):
        assert pipeline.get_present_people() == []

    @pytest.mark.asyncio
    async def test_pipeline_enrollment_available(self, pipeline):
        assert pipeline.enrollment is not None

    @pytest.mark.asyncio
    async def test_pipeline_cloud_store_available(self, pipeline):
        assert pipeline.cloud_store is not None

    @pytest.mark.asyncio
    async def test_get_pipeline_singleton(self, pipeline):
        from boxbot.perception.pipeline import get_pipeline

        assert get_pipeline() is pipeline

    @pytest.mark.asyncio
    async def test_get_pipeline_raises_when_stopped(self, tmp_path):
        from boxbot.perception.pipeline import PerceptionPipeline, get_pipeline
        from boxbot.perception.clouds import CloudStore
        import boxbot.perception.pipeline as pipeline_module

        # Ensure singleton is cleared
        pipeline_module._pipeline_instance = None
        with pytest.raises(RuntimeError, match="not started"):
            get_pipeline()


# ---------------------------------------------------------------------------
# Config integration (verify new config fields load)
# ---------------------------------------------------------------------------


class TestPerceptionConfig:
    """Verify the expanded PerceptionConfig and HardwareConfig load correctly."""

    def test_default_perception_config(self):
        from boxbot.core.config import PerceptionConfig

        cfg = PerceptionConfig()
        assert cfg.motion_threshold == 12.0
        assert cfg.reid_high_threshold == 0.85
        assert cfg.reid_low_threshold == 0.60
        assert cfg.speaker_threshold == 0.75
        assert cfg.presence_timeout == 30
        assert cfg.heartbeat_interval == 5
        assert cfg.max_visual_embeddings == 200
        assert cfg.max_voice_embeddings == 50
        assert cfg.crop_retention_days == 1
        assert cfg.crop_retention_days_debug == 7
        # Phase B fields
        assert cfg.doa_forward_angle == 0
        assert cfg.camera_hfov == 120
        assert cfg.voice_match_threshold == 0.60

    def test_default_hardware_config(self):
        from boxbot.core.config import HardwareConfig

        cfg = HardwareConfig()
        assert cfg.camera.rotation == 180
        assert cfg.camera.lores_resolution == [320, 240]
        assert cfg.hailo.preload_models is True
        assert "yolo" in cfg.hailo.models
        assert "reid" in cfg.hailo.models

    def test_boxbot_config_includes_hardware(self):
        from boxbot.core.config import BoxBotConfig

        cfg = BoxBotConfig()
        assert hasattr(cfg, "hardware")
        assert cfg.hardware.camera.rotation == 180


# ---------------------------------------------------------------------------
# Visual cloud matching (person-id-overhaul: N1)
# ---------------------------------------------------------------------------


def _unit(dim: int = 512, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestVisualCloudMatching:
    """get_visual_clouds + VisualReID.match_cloud (provenance-weighted)."""

    @pytest_asyncio.fixture
    async def store(self, tmp_path):
        from boxbot.perception.clouds import CloudStore

        s = CloudStore(db_path=tmp_path / "vc.db")
        await s.initialize()
        yield s
        await s.close()

    @pytest.mark.asyncio
    async def test_get_visual_clouds_shape_and_weights(self, store):
        from boxbot.perception.clouds import (
            PROVENANCE_AGENT_IDENTIFY,
            PROVENANCE_VISUAL_REID,
            provenance_weight,
        )

        pid = await store.create_person("Jacob")
        await store.add_visual_embedding(
            pid, _unit(seed=1), provenance=PROVENANCE_AGENT_IDENTIFY
        )
        await store.add_visual_embedding(
            pid, _unit(seed=2), provenance=PROVENANCE_VISUAL_REID
        )

        clouds = await store.get_visual_clouds()
        assert pid in clouds
        name, embs, weights = clouds[pid]
        assert name == "Jacob"
        assert embs.shape == (2, 512)
        assert np.linalg.norm(embs[0]) == pytest.approx(1.0, abs=1e-5)
        assert sorted(weights.tolist()) == sorted([
            provenance_weight(PROVENANCE_AGENT_IDENTIFY),
            provenance_weight(PROVENANCE_VISUAL_REID),
        ])

    def test_match_cloud_high_tier_on_self(self):
        from boxbot.perception.visual_reid import VisualReID

        reid = VisualReID(confirmed_threshold=0.7, maybe_threshold=0.55, topk=3)
        v = _unit(seed=3)
        clouds = {"p1": ("Jacob", np.stack([v, v]), np.array([1.0, 1.0], np.float32))}
        r = reid.match_cloud(v, clouds)
        assert r.tier == "high"
        assert r.person_id == "p1"
        assert r.confidence == pytest.approx(1.0, abs=1e-4)

    def test_match_cloud_unknown_when_orthogonal(self):
        from boxbot.perception.visual_reid import VisualReID

        reid = VisualReID(confirmed_threshold=0.7, maybe_threshold=0.55, topk=3)
        a = np.zeros(512, np.float32); a[0] = 1.0
        b = np.zeros(512, np.float32); b[1] = 1.0
        clouds = {"p1": ("Jacob", np.stack([b]), np.array([1.0], np.float32))}
        r = reid.match_cloud(a, clouds)
        assert r.tier == "unknown"
        assert r.person_id is None

    def test_match_cloud_empty_is_unknown(self):
        from boxbot.perception.visual_reid import VisualReID

        r = VisualReID().match_cloud(_unit(seed=4), {})
        assert r.tier == "unknown"

    def test_match_cloud_provenance_weighting_raises_score(self):
        # A close point and a far point. Up-weighting the close point pulls the
        # weighted top-k mean up (0.7 → 0.8), showing provenance influence.
        from boxbot.perception.visual_reid import VisualReID

        reid = VisualReID(confirmed_threshold=0.7, maybe_threshold=0.55, topk=2)
        q = np.zeros(512, np.float32); q[0] = 1.0
        close = np.zeros(512, np.float32)
        close[0] = 0.9; close[1] = float(np.sqrt(1 - 0.81))   # cos 0.9
        far = np.zeros(512, np.float32)
        far[0] = 0.5; far[1] = float(np.sqrt(1 - 0.25))       # cos 0.5
        embs = np.stack([close, far])

        equal = reid.match_cloud(
            q, {"p1": ("J", embs, np.array([1.0, 1.0], np.float32))}
        )
        weighted = reid.match_cloud(
            q, {"p1": ("J", embs, np.array([3.0, 1.0], np.float32))}
        )
        assert equal.confidence == pytest.approx(0.7, abs=1e-3)
        assert weighted.confidence == pytest.approx(0.8, abs=1e-3)


class TestVisualEvictionProvenance:
    """_enforce_visual_cap: isolation+age eviction protects anchors."""

    @pytest_asyncio.fixture
    async def store(self, tmp_path):
        from boxbot.perception.clouds import CloudStore

        s = CloudStore(db_path=tmp_path / "ev.db")
        await s.initialize()
        yield s
        await s.close()

    @pytest.mark.asyncio
    async def test_anchors_never_evicted(self, store):
        from boxbot.perception.clouds import (
            PROVENANCE_AGENT_IDENTIFY,
            PROVENANCE_VOICE_DOA,
        )

        pid = await store.create_person("Jacob")
        for _ in range(2):
            await store.add_visual_embedding(
                pid, _unit(), provenance=PROVENANCE_AGENT_IDENTIFY
            )
        for _ in range(8):
            await store.add_visual_embedding(
                pid, _unit(), provenance=PROVENANCE_VOICE_DOA
            )

        await store._enforce_visual_cap(pid, max_count=5)

        db = store._ensure_db()
        async with db.execute(
            "SELECT provenance, COUNT(*) c FROM visual_embeddings "
            "WHERE person_id = ? GROUP BY provenance",
            (pid,),
        ) as cur:
            counts = {r["provenance"]: r["c"] async for r in cur}
        # Both anchors survive; total trimmed toward the low-water mark.
        assert counts.get(PROVENANCE_AGENT_IDENTIFY) == 2
        assert sum(counts.values()) <= 5


class TestEnrollmentAdmission:
    """commit_session provenance gating + mark_admission precedence."""

    @pytest_asyncio.fixture
    async def cloud_store(self, tmp_path):
        from boxbot.perception.clouds import CloudStore

        s = CloudStore(db_path=tmp_path / "adm.db")
        await s.initialize()
        yield s
        await s.close()

    @pytest_asyncio.fixture
    async def manager(self, cloud_store):
        from boxbot.perception.enrollment import EnrollmentManager

        return EnrollmentManager(cloud_store)

    @pytest.mark.asyncio
    async def test_voice_doa_admits_visual_not_voice(self, manager, cloud_store):
        pid = await cloud_store.create_person("Jacob")
        manager.buffer_visual_embedding("Person A", _unit(), crop_path="/c.jpg")
        manager.buffer_voice_embedding("Person A", _unit(192))
        manager.mark_admission(
            "Person A", person_id=pid, person_name="Jacob",
            source="voice_doa", score=0.7,
        )
        await manager.commit_session()

        assert await cloud_store.count_visual_embeddings(pid) == 1
        assert len(await cloud_store.get_voice_embeddings(pid)) == 0

    @pytest.mark.asyncio
    async def test_voice_visual_agree_admits_both(self, manager, cloud_store):
        pid = await cloud_store.create_person("Jacob")
        manager.buffer_visual_embedding("Person A", _unit())
        manager.buffer_voice_embedding("Person A", _unit(192))
        manager.mark_admission(
            "Person A", person_id=pid, person_name="Jacob",
            source="voice_visual_agree", score=0.8,
        )
        await manager.commit_session()

        assert await cloud_store.count_visual_embeddings(pid) == 1
        assert len(await cloud_store.get_voice_embeddings(pid)) == 1

    @pytest.mark.asyncio
    async def test_reid_only_claim_held_back(self, manager, cloud_store):
        pid = await cloud_store.create_person("Jacob")
        manager.buffer_visual_embedding("Person A", _unit())
        manager.on_reid_match(
            "Person A", "visual", person_id=pid, person_name="Jacob",
            tier="medium", score=0.6,
        )
        await manager.commit_session()
        # Passive reid match must not grow the cloud.
        assert await cloud_store.count_visual_embeddings(pid) == 0

    @pytest.mark.asyncio
    async def test_admitted_visual_tagged_with_provenance(self, manager, cloud_store):
        pid = await cloud_store.create_person("Jacob")
        manager.buffer_visual_embedding("Person A", _unit())
        manager.mark_admission(
            "Person A", person_id=pid, person_name="Jacob",
            source="voice_doa", score=0.7,
        )
        await manager.commit_session()

        db = cloud_store._ensure_db()
        async with db.execute(
            "SELECT provenance FROM visual_embeddings WHERE person_id = ?",
            (pid,),
        ) as cur:
            rows = [r["provenance"] async for r in cur]
        assert rows == ["voice_doa"]

    def test_mark_admission_never_overrides_agent_identify(self, manager):
        manager._claims["Person A"] = __import__(
            "boxbot.perception.enrollment", fromlist=["SessionClaim"]
        ).SessionClaim(
            person_id="p_jacob", name="Jacob", source="agent_identify",
        )
        manager.mark_admission(
            "Person A", person_id="p_other", person_name="Carina",
            source="voice_visual_agree", score=0.95,
        )
        claim = manager.get_claim("Person A")
        assert claim.source == "agent_identify"
        assert claim.name == "Jacob"


# ---------------------------------------------------------------------------
# Identity-cloud reconciliation (person-id-overhaul: NEXT)
# ---------------------------------------------------------------------------


def _basis(idx: int, dim: int = 512) -> np.ndarray:
    v = np.zeros(dim, np.float32)
    v[idx] = 1.0
    return v


def _near(base: np.ndarray, eps: float = 0.03, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = base + eps * rng.standard_normal(base.shape).astype(np.float32)
    return v / np.linalg.norm(v)


class TestIdReconcile:
    """Deterministic dream-cycle identity hygiene (audit-only by default)."""

    @pytest_asyncio.fixture
    async def store(self, tmp_path):
        from boxbot.perception.clouds import CloudStore

        s = CloudStore(db_path=tmp_path / "rec.db")
        await s.initialize()
        yield s
        await s.close()

    def test_levenshtein_eric_erik(self):
        from boxbot.perception.reconcile import _levenshtein

        assert _levenshtein("Eric", "Erik") == 1
        assert _levenshtein("Jacob", "Jacob") == 0
        assert _levenshtein("Jacob", "Carina") >= 3

    @pytest.mark.asyncio
    async def test_outlier_detection_flags_contamination(self, store):
        from boxbot.perception.clouds import PROVENANCE_VOICE_DOA
        from boxbot.perception.reconcile import run_id_reconcile

        pid = await store.create_person("Jacob")
        base = _basis(0)
        for i in range(4):
            await store.add_visual_embedding(
                pid, _near(base, seed=i), provenance=PROVENANCE_VOICE_DOA
            )
        # A contaminating face from a different direction.
        outlier_id = await store.add_visual_embedding(
            pid, _basis(1), provenance=PROVENANCE_VOICE_DOA
        )

        report = await run_id_reconcile(cloud_store=store, audit_only=True)
        flagged = {o["embedding_id"] for o in report["outliers"]}
        assert outlier_id in flagged

    @pytest.mark.asyncio
    async def test_anchor_not_flagged_as_outlier(self, store):
        from boxbot.perception.clouds import (
            PROVENANCE_AGENT_IDENTIFY,
            PROVENANCE_VOICE_DOA,
        )
        from boxbot.perception.reconcile import run_id_reconcile

        pid = await store.create_person("Jacob")
        base = _basis(0)
        for i in range(4):
            await store.add_visual_embedding(
                pid, _near(base, seed=i), provenance=PROVENANCE_VOICE_DOA
            )
        # An isolated anchor must never be flagged for eviction.
        anchor_id = await store.add_visual_embedding(
            pid, _basis(1), provenance=PROVENANCE_AGENT_IDENTIFY
        )
        report = await run_id_reconcile(cloud_store=store, audit_only=True)
        assert anchor_id not in {o["embedding_id"] for o in report["outliers"]}

    @pytest.mark.asyncio
    async def test_duplicate_person_by_name(self, store):
        from boxbot.perception.clouds import PROVENANCE_VOICE_DOA
        from boxbot.perception.reconcile import run_id_reconcile

        eric = await store.create_person("Eric")
        erik = await store.create_person("Erik")
        await store.add_visual_embedding(
            eric, _basis(0), provenance=PROVENANCE_VOICE_DOA
        )
        await store.add_visual_embedding(
            erik, _basis(5), provenance=PROVENANCE_VOICE_DOA
        )
        report = await run_id_reconcile(cloud_store=store, audit_only=True)
        names = {frozenset((d["a"], d["b"])) for d in report["duplicate_persons"]}
        assert frozenset(("Eric", "Erik")) in names

    @pytest.mark.asyncio
    async def test_duplicate_person_by_face(self, store):
        from boxbot.perception.clouds import PROVENANCE_VOICE_DOA
        from boxbot.perception.reconcile import run_id_reconcile

        a = await store.create_person("Alice")
        b = await store.create_person("Bob")
        base = _basis(0)
        for i in range(3):
            await store.add_visual_embedding(
                a, _near(base, seed=i), provenance=PROVENANCE_VOICE_DOA
            )
            await store.add_visual_embedding(
                b, _near(base, seed=100 + i), provenance=PROVENANCE_VOICE_DOA
            )
        report = await run_id_reconcile(cloud_store=store, audit_only=True)
        pairs = {frozenset((d["a"], d["b"])) for d in report["duplicate_persons"]}
        assert frozenset(("Alice", "Bob")) in pairs

    @pytest.mark.asyncio
    async def test_mislabel_against_anchor(self, store):
        from boxbot.perception.clouds import (
            PROVENANCE_AGENT_IDENTIFY,
            PROVENANCE_VOICE_DOA,
        )
        from boxbot.perception.reconcile import run_id_reconcile

        jacob = await store.create_person("Jacob")
        carina = await store.create_person("Carina")
        await store.add_visual_embedding(
            jacob, _basis(0), provenance=PROVENANCE_AGENT_IDENTIFY
        )
        await store.add_visual_embedding(
            carina, _basis(5), provenance=PROVENANCE_AGENT_IDENTIFY
        )
        # A point filed under Carina that is actually Jacob's face.
        stray = await store.add_visual_embedding(
            carina, _basis(0), provenance=PROVENANCE_VOICE_DOA
        )
        report = await run_id_reconcile(cloud_store=store, audit_only=True)
        hits = {
            m["embedding_id"]: m["suggested_person"]
            for m in report["mislabels"]
        }
        assert hits.get(stray) == "Jacob"

    @pytest.mark.asyncio
    async def test_audit_only_does_not_mutate(self, store):
        from boxbot.perception.clouds import PROVENANCE_VOICE_DOA
        from boxbot.perception.reconcile import run_id_reconcile

        pid = await store.create_person("Jacob")
        base = _basis(0)
        for i in range(4):
            await store.add_visual_embedding(
                pid, _near(base, seed=i), provenance=PROVENANCE_VOICE_DOA
            )
        await store.add_visual_embedding(
            pid, _basis(1), provenance=PROVENANCE_VOICE_DOA
        )
        before = await store.count_visual_embeddings(pid)
        await run_id_reconcile(cloud_store=store, audit_only=True)
        assert await store.count_visual_embeddings(pid) == before

    @pytest.mark.asyncio
    async def test_apply_evicts_outliers(self, store):
        from boxbot.perception.clouds import PROVENANCE_VOICE_DOA
        from boxbot.perception.reconcile import run_id_reconcile

        pid = await store.create_person("Jacob")
        base = _basis(0)
        for i in range(4):
            await store.add_visual_embedding(
                pid, _near(base, seed=i), provenance=PROVENANCE_VOICE_DOA
            )
        await store.add_visual_embedding(
            pid, _basis(1), provenance=PROVENANCE_VOICE_DOA
        )
        report = await run_id_reconcile(cloud_store=store, audit_only=False)
        assert report["applied"]["evicted"] >= 1
        assert await store.count_visual_embeddings(pid) == 4
