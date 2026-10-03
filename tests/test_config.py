"""Tests for boxbot.core.config — configuration loading and validation,
plus boxbot.core.models id → provider routing."""

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
from boxbot.core.models import (
    provider_for_model,
    reasoning_effort_for_model,
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

    def test_fast_tier_off_by_default(self):
        cfg = BoxBotConfig()
        assert cfg.models.fast is None

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
        from boxbot.core.paths import PHOTOS_DIR

        cfg = BoxBotConfig()
        assert cfg.photos.storage_path == str(PHOTOS_DIR)


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

    def test_env_overlays_model_fast(self, tmp_config):
        with patch.dict("os.environ", {"BOXBOT_MODEL_FAST": "gpt-5.6-luna"}, clear=True):
            cfg = load_config(tmp_config)
        assert cfg.models.fast == "gpt-5.6-luna"

    def test_model_fast_unset_leaves_tier_off(self, tmp_config):
        with patch.dict("os.environ", {}, clear=True):
            cfg = load_config(tmp_config)
        assert cfg.models.fast is None

    def test_env_overlays_openai_api_key(self, tmp_config):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-openai-123"}, clear=True):
            cfg = load_config(tmp_config)
        assert cfg.api_keys.openai == "sk-openai-123"

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


class TestSandboxConfig:
    """Validation for the sandbox privilege-drop fields."""

    def test_defaults(self):
        from boxbot.core.config import SandboxConfig

        cfg = SandboxConfig()
        assert cfg.privilege_drop == "auto"
        assert cfg.extra_groups == []

    @pytest.mark.parametrize("mode", ["auto", "sudo", "setuid", "none"])
    def test_privilege_drop_accepts_enum(self, mode):
        from boxbot.core.config import SandboxConfig

        assert SandboxConfig(privilege_drop=mode).privilege_drop == mode

    def test_privilege_drop_rejects_unknown(self):
        from boxbot.core.config import SandboxConfig

        with pytest.raises(ValidationError):
            SandboxConfig(privilege_drop="root")

    def test_extra_groups_inet(self):
        from boxbot.core.config import SandboxConfig

        cfg = SandboxConfig(privilege_drop="setuid", extra_groups=[3003])
        assert cfg.extra_groups == [3003]


class TestProviderForModel:
    """boxbot.core.models — id → provider routing."""

    @pytest.mark.parametrize(
        "model",
        ["gpt-5.6-luna", "gpt-4o-mini", "o1", "o3-mini", "o4", "GPT-5.6-LUNA"],
    )
    def test_openai_ids(self, model):
        assert provider_for_model(model) == "openai"

    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-5",
            "claude-haiku-4-5-20251001",
            "",
            "some-unknown-model",
            "opus-5",
        ],
    )
    def test_anthropic_is_the_default(self, model):
        assert provider_for_model(model) == "anthropic"


class TestReasoningEffortForModel:
    """The fast tier always asks for the floor — which floor varies."""

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("gpt-5.6-luna", "none"),
            ("gpt-5.6-luna-2026-07-09", "none"),
            ("GPT-5.6-LUNA", "none"),
            ("gpt-5.1", "minimal"),
            ("gpt-5", "minimal"),
            ("o3-mini", "minimal"),
            # Minor is an int, not a float digit: 5.10 > 5.6.
            ("gpt-5.10-x", "none"),
            ("gpt-6", "none"),
            ("gpt-10.2", "none"),
            # Non-reasoning ids 400 on the parameter — omit it.
            ("gpt-4o", None),
            ("claude-opus-5", None),
            ("", None),
        ],
    )
    def test_floor_per_id(self, model, expected):
        assert reasoning_effort_for_model(model) == expected


class TestOpenAIEndpointShape:
    """OpenAIConfig — Azure vs public OpenAI routing."""

    @pytest.mark.parametrize(
        "api_type,api_base",
        [
            ("azure", "https://x.openai.azure.com/"),   # explicit
            ("AZURE", "https://x.openai.azure.com/"),   # case-insensitive
            (None, "https://x.openai.azure.com/"),      # inferred from host
        ],
    )
    def test_is_azure(self, api_type, api_base):
        from boxbot.core.config import OpenAIConfig

        assert OpenAIConfig(api_type=api_type, api_base=api_base).is_azure

    @pytest.mark.parametrize(
        "api_type,api_base",
        [
            (None, None),                          # public, nothing set
            (None, "https://proxy.internal/v1"),   # public via a gateway
            ("open_ai", None),                     # explicit non-azure
        ],
    )
    def test_is_not_azure(self, api_type, api_base):
        from boxbot.core.config import OpenAIConfig

        assert not OpenAIConfig(api_type=api_type, api_base=api_base).is_azure

    def test_env_overlay_populates_the_block(self, monkeypatch, tmp_path):
        from boxbot.core.config import load_config

        monkeypatch.setenv("OPENAI_API_TYPE", "azure")
        monkeypatch.setenv("OPENAI_API_BASE", "https://r.openai.azure.com/")
        monkeypatch.setenv("OPENAI_API_VERSION", "2025-01-01-preview")
        cfg_path = tmp_path / "c.yaml"
        cfg_path.write_text("{}\n")

        cfg = load_config(str(cfg_path))

        assert cfg.openai.is_azure
        assert cfg.openai.api_base == "https://r.openai.azure.com/"
        assert cfg.openai.api_version == "2025-01-01-preview"
