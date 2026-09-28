"""Tests for the camera-free identity core (perception/identity.py).

The service must run standalone (panel: no camera, no Hailo, no cv2)
and composed by the visual pipeline (Pi) — with exactly one
VoiceSessionEnded → commit_session subscription either way.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import pytest_asyncio

from boxbot.core.events import VoiceSessionEnded, get_event_bus


def _unit_vec(dim: int = 192, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest_asyncio.fixture
async def store(tmp_path):
    from boxbot.perception.clouds import CloudStore

    s = CloudStore(db_path=tmp_path / "identity.db")
    await s.initialize()
    yield s
    await s.close()


@pytest_asyncio.fixture
async def service(store):
    from boxbot.perception.identity import IdentityService

    svc = IdentityService(cloud_store=store)
    await svc.start()
    yield svc
    await svc.stop()


class TestRegistry:
    @pytest.mark.asyncio
    async def test_get_identity_raises_before_start(self):
        from boxbot.perception.identity import get_identity

        with pytest.raises(RuntimeError, match="not started"):
            get_identity()

    @pytest.mark.asyncio
    async def test_get_identity_returns_running_instance(self, service):
        from boxbot.perception.identity import get_identity

        assert get_identity() is service

    @pytest.mark.asyncio
    async def test_registry_cleared_on_stop(self, store):
        from boxbot.perception.identity import IdentityService, get_identity

        svc = IdentityService(cloud_store=store)
        await svc.start()
        await svc.stop()
        with pytest.raises(RuntimeError):
            get_identity()


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_is_idempotent_single_commit(self, service):
        """Double start must not double the VoiceSessionEnded handler."""
        service._enrollment = MagicMock()
        service._enrollment.commit_session = AsyncMock(return_value={})

        await service.start()  # second start: no-op
        await get_event_bus().publish(
            VoiceSessionEnded(conversation_id="voice_x")
        )

        service._enrollment.commit_session.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_injected_store_not_closed_on_stop(self, store):
        from boxbot.perception.identity import IdentityService

        svc = IdentityService(cloud_store=store)
        await svc.start()
        await svc.stop()
        # Still usable — stop() must not have closed the injected store.
        assert await store.list_persons() == []

    @pytest.mark.asyncio
    async def test_owned_store_closed_on_stop(self, monkeypatch, tmp_path):
        from boxbot.perception import identity as identity_module

        fake_store = MagicMock()
        fake_store.initialize = AsyncMock()
        fake_store.close = AsyncMock()
        monkeypatch.setattr(
            identity_module, "CloudStore", MagicMock(return_value=fake_store)
        )

        svc = identity_module.IdentityService()
        await svc.start()
        await svc.stop()
        fake_store.close.assert_awaited_once()


class TestCommitOnSessionEnd:
    @pytest.mark.asyncio
    async def test_end_to_end_commit(self, service, store):
        """The live-incident flow: buffer → identify → session end → rows."""
        enrollment = service.enrollment
        enrollment.buffer_voice_embedding("Speaker A", _unit_vec(seed=1))

        result = await enrollment.identify("Jacob", "Speaker A")
        assert result["outcome"] == "create"

        await get_event_bus().publish(
            VoiceSessionEnded(conversation_id="voice_e2e")
        )

        person = await store.get_person_by_name("Jacob")
        assert person is not None
        embeddings = await store.get_voice_embeddings(person["id"])
        assert len(embeddings) == 1
        # Session buffers cleared after commit.
        assert enrollment.get_session_refs() == []

    @pytest.mark.asyncio
    async def test_commit_failure_is_contained(self, service):
        service._enrollment = MagicMock()
        service._enrollment.commit_session = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        # Must not raise out of the bus dispatch.
        await get_event_bus().publish(
            VoiceSessionEnded(conversation_id="voice_err")
        )


class TestPipelineComposition:
    """The Pi path: pipeline composes the service, same objects, one commit."""

    def _mock_camera(self):
        camera = MagicMock()
        camera.get_lores_frame = AsyncMock(
            return_value=np.zeros((240, 320), dtype=np.uint8)
        )
        camera.capture_frame = AsyncMock(
            return_value=np.zeros((720, 1280, 3), dtype=np.uint8)
        )
        return camera

    def _mock_hailo(self):
        hailo = MagicMock()
        hailo.infer = AsyncMock(
            return_value={"output": np.zeros((2, 5, 80), dtype=np.float32)}
        )
        return hailo

    @pytest.mark.asyncio
    async def test_pipeline_shares_service_objects(self, service):
        from boxbot.perception.identity import get_identity
        from boxbot.perception.pipeline import PerceptionPipeline

        pipeline = PerceptionPipeline(
            camera=self._mock_camera(),
            hailo=self._mock_hailo(),
            identity=service,
            scan_fps=10,
        )
        await pipeline.start()
        try:
            assert pipeline.enrollment is service.enrollment
            assert pipeline.cloud_store is service.cloud_store

            # One commit per session end — the pipeline must not have
            # its own VoiceSessionEnded subscription.
            service._enrollment = MagicMock()
            service._enrollment.commit_session = AsyncMock(return_value={})
            await get_event_bus().publish(
                VoiceSessionEnded(conversation_id="voice_pi")
            )
            service._enrollment.commit_session.assert_awaited_once()
        finally:
            await pipeline.stop()

        # Injected service outlives the pipeline.
        assert get_identity() is service

    @pytest.mark.asyncio
    async def test_pipeline_owns_service_when_not_injected(self, tmp_path):
        """Legacy construction (cloud_store only) still works end to end."""
        from boxbot.perception.clouds import CloudStore
        from boxbot.perception.identity import get_identity
        from boxbot.perception.pipeline import PerceptionPipeline

        store = CloudStore(db_path=tmp_path / "legacy.db")
        await store.initialize()

        pipeline = PerceptionPipeline(
            camera=self._mock_camera(),
            hailo=self._mock_hailo(),
            cloud_store=store,
            scan_fps=10,
        )
        await pipeline.start()
        assert pipeline.enrollment is not None
        assert pipeline.cloud_store is store
        assert get_identity() is pipeline._identity

        await pipeline.stop()
        with pytest.raises(RuntimeError):
            get_identity()
        # Injected store is never closed by the owned service.
        assert await store.list_persons() == []
        await store.close()
