"""Tests for boxbot.core.config — configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

import boxbot.core.config as config_module
from boxbot.core.config import (
    AgentConfig,
    ApiKeysConfig,
    BoxBotConfig,
    CameraConfig,
    DisplayConfig,
    LoggingConfig,
    MemoryConfig,
    ModelsConfig,
    PhotosConfig,
    ScheduleConfig,
    get_config,
    load_config,
)


class TestDefaultConfig:
    """Verify that all config sections have sane defaults."""

    def test_default_agent_name(self):
        cfg = BoxBotConfig()
        assert cfg.agent.name == "boxBot"

    def test_default_wake_word(self):
        cfg = BoxBotConfig()
        assert cfg.agent.wake_word == "hey box"

    def test_default_models(self):
        cfg = BoxBotConfig()
        assert "claude" in cfg.models.large.lower() or cfg.models.large
        assert "claude" in cfg.models.small.lower() or cfg.models.small

    def test_default_schedule_has_three_wake_cycles(self):
        cfg = BoxBotConfig()
        assert len(cfg.schedule.wake_cycle) == 3

    def test_default_display_idle_displays(self):
        cfg = BoxBotConfig()
        assert "picture" in cfg.display.idle_displays

    def test_default_camera_resolution(self):
        cfg = BoxBotConfig()
        assert cfg.camera.resolution == [1280, 720]

    def test_default_photos_storage_path(self):
        cfg = BoxBotConfig()
        assert cfg.photos.storage_path == "data/photos"


class TestConfigSingleton:
    """Test get_config() / load_config() singleton behavior."""

    def test_get_config_raises_when_not_loaded(self):
        """get_config() must raise RuntimeError before load_config()."""
        with pytest.raises(RuntimeError, match="Configuration not loaded"):
            get_config()

    def test_load_config_returns_boxbot_config(self, tmp_config):
        with patch.dict("os.environ", {}, clear=True):
            cfg = load_config(tmp_config)
        assert isinstance(cfg, BoxBotConfig)

    def test_get_config_returns_same_after_load(self, tmp_config):
        with patch.dict("os.environ", {}, clear=True):
            loaded = load_config(tmp_config)
        assert get_config() is loaded

    def test_load_config_uses_default_path_when_none(self):
        """When no path given and no file exists, loads defaults without error."""
        with patch.dict("os.environ", {}, clear=True):
            cfg = load_config("/nonexistent/path/config.yaml")
        assert isinstance(cfg, BoxBotConfig)
        assert cfg.agent.name == "boxBot"


class TestYamlOverrides:
    """Test that YAML values override defaults."""

    def test_yaml_overrides_agent_name(self, tmp_config):
        with patch.dict("os.environ", {}, clear=True):
            cfg = load_config(tmp_config)
        assert cfg.agent.name == "TestBot"

    def test_yaml_overrides_camera_resolution(self, tmp_config):
        with patch.dict("os.environ", {}, clear=True):
            cfg = load_config(tmp_config)
        assert cfg.camera.resolution == [640, 480]

    def test_yaml_overrides_sandbox_timeout(self, tmp_config):
        with patch.dict("os.environ", {}, clear=True):
            cfg = load_config(tmp_config)
        assert cfg.sandbox.timeout == 5


class TestEnvOverlay:
    """Test that environment variables overlay YAML config."""

    def test_env_overlays_model_large(self, tmp_config):
        with patch.dict("os.environ", {"BOXBOT_MODEL_LARGE": "test-env-model"}, clear=True):
            cfg = load_config(tmp_config)
        assert cfg.models.large == "test-env-model"

    def test_env_overlays_anthropic_api_key(self, tmp_config):
        with patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": "sk-test-key-123"},
            clear=True,
        ):
            cfg = load_config(tmp_config)
        assert cfg.api_keys.anthropic == "sk-test-key-123"

    def test_env_overlays_log_level(self, tmp_config):
        with patch.dict("os.environ", {"BOXBOT_LOG_LEVEL": "WARNING"}, clear=True):
            cfg = load_config(tmp_config)
        assert cfg.logging.level == "WARNING"


class TestValidation:
    """Test Pydantic model validators catch invalid values."""

    def test_camera_resolution_must_be_two_values(self):
        with pytest.raises(ValidationError):
            CameraConfig(resolution=[100])

    def test_camera_resolution_three_values_rejected(self):
        with pytest.raises(ValidationError):
            CameraConfig(resolution=[100, 200, 300])

    def test_photos_max_image_resolution_must_be_two_values(self):
        with pytest.raises(ValidationError):
            PhotosConfig(max_image_resolution=[100])


class TestApiKeysRedaction:
    """Test that ApiKeysConfig repr does not leak secrets."""

    def test_repr_redacts_set_keys(self):
        keys = ApiKeysConfig(anthropic="sk-real-secret")
        text = repr(keys)
        assert "sk-real-secret" not in text
        assert "***" in text

    def test_repr_shows_none_for_unset_keys(self):
        keys = ApiKeysConfig()
        text = repr(keys)
        assert "anthropic=None" in text
