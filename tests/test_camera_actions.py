"""Tests for the sandbox ``camera.*`` action handlers.

The contract that matters here is honesty: on a device with no camera
the agent must get an error, not pixels it will believe it saw.
"""

from __future__ import annotations

import pytest

import boxbot.tools._sandbox_actions as actions
from boxbot.tools._sandbox_actions import ActionContext, process_action


async def _capture(ctx: ActionContext, **payload):
    return await process_action({"_sdk": "camera.capture", **payload}, ctx)


@pytest.fixture
def no_camera(monkeypatch):
    """No Camera HAL, dev test pattern off (the default)."""
    import boxbot.hardware.camera as camera_mod

    monkeypatch.setattr(camera_mod, "get_camera", lambda: None)
    monkeypatch.setattr(actions, "_test_pattern_enabled", lambda: False)


@pytest.mark.asyncio
class TestCaptureWithoutCamera:
    async def test_capture_errors(self, no_camera):
        ctx = ActionContext()
        result = await _capture(ctx)
        assert result == {
            "status": "error", "error": "No camera on this device.",
        }

    async def test_nothing_is_attached(self, no_camera):
        ctx = ActionContext()
        await _capture(ctx)
        assert ctx.image_attachments == []

    async def test_capture_cropped_errors(self, no_camera):
        ctx = ActionContext()
        result = await process_action(
            {
                "_sdk": "camera.capture_cropped",
                "bbox": {"x": 0, "y": 0, "w": 10, "h": 10},
            },
            ctx,
        )
        assert result["status"] == "error"
        assert result["error"] == "No camera on this device."

    async def test_dev_flag_restores_the_test_pattern(
        self, no_camera, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(actions, "_test_pattern_enabled", lambda: True)
        monkeypatch.setattr(actions, "_tmp_capture_dir", lambda: tmp_path)
        ctx = ActionContext()
        result = await _capture(ctx)
        assert result["status"] == "ok"
        assert result["fallback"] is True
        assert len(ctx.image_attachments) == 1


class TestTestPatternFlag:
    def test_off_by_default(self, monkeypatch):
        from boxbot.core.config import BoxBotConfig

        cfg = BoxBotConfig()
        monkeypatch.setattr("boxbot.core.config.get_config", lambda: cfg)
        assert actions._test_pattern_enabled() is False

    def test_reads_config(self, monkeypatch):
        from boxbot.core.config import BoxBotConfig

        cfg = BoxBotConfig()
        cfg.camera.test_pattern_without_camera = True
        monkeypatch.setattr("boxbot.core.config.get_config", lambda: cfg)
        assert actions._test_pattern_enabled() is True

    def test_unloaded_config_is_off(self, monkeypatch):
        def _boom():
            raise RuntimeError("Configuration not loaded.")

        monkeypatch.setattr("boxbot.core.config.get_config", _boom)
        assert actions._test_pattern_enabled() is False
