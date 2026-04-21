"""Configuration loading and validation.

Reads from config/config.yaml and environment variables (.env).
All configuration is centralized here — no module reads config files directly.

Usage:
    from boxbot.core.config import get_config, load_config

    # Load at startup (once)
    config = load_config("config/config.yaml")

    # Access anywhere after loading
    config = get_config()
    print(config.agent.name)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config: BoxBotConfig | None = None


def get_config() -> BoxBotConfig:
    """Return the loaded configuration singleton.

    Raises RuntimeError if load_config() has not been called yet.
    """
    if _config is None:
        raise RuntimeError(
            "Configuration not loaded. Call load_config() at startup."
        )
    return _config


def load_config(path: str | Path | None = None) -> BoxBotConfig:
    """Load configuration from a YAML file and environment variables.

    Args:
        path: Path to config YAML. Defaults to config/config.yaml relative
              to the project root, or the BOXBOT_CONFIG env var.

    Returns:
        The validated BoxBotConfig singleton.
    """
    global _config

    if path is None:
        path = os.environ.get("BOXBOT_CONFIG", "config/config.yaml")

    config_path = Path(path)
    data: dict[str, Any] = {}

    if config_path.exists():
        logger.info("Loading config from %s", config_path)
        with open(config_path) as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                data = loaded
    else:
        logger.warning(
            "Config file %s not found, using defaults + env vars", config_path
        )

    # Overlay environment variables for secrets and model selection
    _overlay_env(data)

    _config = BoxBotConfig.model_validate(data)
    logger.info("Configuration loaded successfully")
    return _config


def _overlay_env(data: dict[str, Any]) -> None:
    """Overlay environment variables onto the config dict.

    Environment variables take precedence over YAML values for security-
    sensitive settings and model selection.
    """
    # Models — always from env, never from YAML
    if "models" not in data:
        data["models"] = {}

    if val := os.environ.get("BOXBOT_MODEL_LARGE"):
        data["models"]["large"] = val
    if val := os.environ.get("BOXBOT_MODEL_SMALL"):
        data["models"]["small"] = val

    # API keys — env only, never in YAML
    if "api_keys" not in data:
        data["api_keys"] = {}

    env_key_map = {
        "ANTHROPIC_API_KEY": "anthropic",
        "OPENAI_API_KEY": "openai",
        "DEEPGRAM_API_KEY": "deepgram",
        "ELEVENLABS_API_KEY": "elevenlabs",
        "WHATSAPP_ACCESS_TOKEN": "whatsapp_access_token",
        "WHATSAPP_PHONE_NUMBER_ID": "whatsapp_phone_number_id",
        "WHATSAPP_VERIFY_TOKEN": "whatsapp_verify_token",
        "WHATSAPP_APP_SECRET": "whatsapp_app_secret",
        "AWS_ACCESS_KEY_ID": "aws_access_key_id",
        "AWS_SECRET_ACCESS_KEY": "aws_secret_access_key",
        "AWS_S3_BUCKET": "aws_s3_bucket",
    }
    for env_var, key_name in env_key_map.items():
        if val := os.environ.get(env_var):
            data["api_keys"][key_name] = val

    # Logging level override
    if val := os.environ.get("BOXBOT_LOG_LEVEL"):
        if "logging" not in data:
            data["logging"] = {}
        data["logging"]["level"] = val


# ---------------------------------------------------------------------------
# Config section models
# ---------------------------------------------------------------------------


class AgentConfig(BaseModel):
    """Agent identity and conversation settings."""

    name: str = "boxBot"
    wake_word: str = "hey box"
    max_turns: int = 20


class WakeCycleEntry(BaseModel):
    """A single recurring trigger seeded from config."""

    cron: str
    description: str
    instructions: str


class ScheduleConfig(BaseModel):
    """Agent scheduler / task management settings."""

    wake_cycle: list[WakeCycleEntry] = Field(default_factory=lambda: [
        WakeCycleEntry(
            cron="0 7 * * *",
            description="Morning briefing",
            instructions=(
                "Check weather forecast, review to-do list, update displays, "
                "prepare daily briefing for household members"
            ),
        ),
        WakeCycleEntry(
            cron="0 12 * * *",
            description="Midday check",
            instructions=(
                "Review to-do list for actionable items, check for pending "
                "triggers, update displays if data is stale"
            ),
        ),
        WakeCycleEntry(
            cron="0 20 * * *",
            description="Evening review",
            instructions=(
                "Review to-do list, summarize the day's activity, set triggers "
                "for tomorrow's tasks, update displays"
            ),
        ),
    ])
    idle_timeout: int = 300
    person_trigger_expiry_days: int = 7


class NightModeConfig(BaseModel):
    """Display night mode settings."""

    enabled: bool = True
    start: str = "22:00"
    end: str = "07:00"
    brightness: float = 0.3


class DisplayConfig(BaseModel):
    """Display manager settings."""

    rotation_interval: int = 30
    idle_displays: list[str] = Field(
        default_factory=lambda: ["picture", "calendar", "weather"]
    )
    brightness: float = 0.8
    night_mode: NightModeConfig = Field(default_factory=NightModeConfig)


class CameraConfig(BaseModel):
    """Camera hardware settings."""

    resolution: list[int] = Field(default_factory=lambda: [1280, 720])
    scan_fps: int = 5
    active_fps: int = 15

    @model_validator(mode="after")
    def validate_resolution(self) -> CameraConfig:
        if len(self.resolution) != 2:
            raise ValueError("resolution must be [width, height]")
        return self


class AudioConfig(BaseModel):
    """Audio input/output settings."""

    stt_provider: str = "whisper_api"
    tts_provider: str = "elevenlabs"
    tts_voice: str = "default"
    volume: float = 0.7


class PerceptionConfig(BaseModel):
    """Perception pipeline thresholds."""

    reid_threshold: float = 0.7
    speaker_threshold: float = 0.75
    max_embeddings_per_person: int = 20


class MemoryConfig(BaseModel):
    """Memory system settings."""

    max_context_memories: int = 15
    decay_rate: float = 0.98
    archive_threshold: float = 0.1


class SlideshowConfig(BaseModel):
    """Photo slideshow display settings."""

    seconds_per_photo: int = 15
    transition: str = "fade"
    strategy: str = "random"


class PhotoBackupConfig(BaseModel):
    """Photo cloud backup settings."""

    enabled: bool = False
    provider: str | None = None
    bucket: str | None = None
    region: str | None = None


class PhotosConfig(BaseModel):
    """Photo system settings."""

    storage_path: str = "data/photos"
    max_storage_percent: int = 50
    max_image_resolution: list[int] = Field(
        default_factory=lambda: [1920, 1080]
    )
    soft_delete_retention_days: int = 30
    slideshow: SlideshowConfig = Field(default_factory=SlideshowConfig)
    backup: PhotoBackupConfig = Field(default_factory=PhotoBackupConfig)

    @model_validator(mode="after")
    def validate_max_image_resolution(self) -> PhotosConfig:
        if len(self.max_image_resolution) != 2:
            raise ValueError("max_image_resolution must be [width, height]")
        return self


class SandboxConfig(BaseModel):
    """Sandbox execution settings."""

    venv_path: str = "data/sandbox/venv"
    user: str = "boxbot-sandbox"
    timeout: int = 30
    memory_limit_mb: int = 256
    allow_network: bool = True
    enforce: bool = True
    seccomp_profile: str = "config/seccomp-sandbox.json"
    install_approval_timeout: int = 300
    install_approval_channels: list[str] = Field(
        default_factory=lambda: ["display", "whatsapp"]
    )


class SDKConfig(BaseModel):
    """SDK authoring controls."""

    auto_activate_skills: bool = True
    auto_activate_displays: bool = False


class ModelsConfig(BaseModel):
    """Claude model selection (populated from env vars)."""

    large: str = "claude-sonnet-4-20250514"
    small: str = "claude-haiku-4-5-20251001"


class ApiKeysConfig(BaseModel):
    """API keys and secrets (populated from env vars, never from YAML).

    All fields are optional — a missing key means the corresponding
    service is not configured.
    """

    anthropic: str | None = None
    openai: str | None = None
    deepgram: str | None = None
    elevenlabs: str | None = None
    whatsapp_access_token: str | None = None
    whatsapp_phone_number_id: str | None = None
    whatsapp_verify_token: str | None = None
    whatsapp_app_secret: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_s3_bucket: str | None = None

    def __repr__(self) -> str:
        """Redact secrets in repr output."""
        fields = []
        for name in self.model_fields:
            val = getattr(self, name)
            if val is not None:
                fields.append(f"{name}='***'")
            else:
                fields.append(f"{name}=None")
        return f"ApiKeysConfig({', '.join(fields)})"


class LoggingConfig(BaseModel):
    """Logging settings."""

    level: str = "INFO"
    file: str = "logs/boxbot.log"
    max_size_mb: int = 50
    backup_count: int = 5


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class BoxBotConfig(BaseModel):
    """Root configuration model for boxBot.

    Loaded from config/config.yaml with env var overlays for secrets
    and model selection.
    """

    agent: AgentConfig = Field(default_factory=AgentConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    photos: PhotosConfig = Field(default_factory=PhotosConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    sdk: SDKConfig = Field(default_factory=SDKConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    api_keys: ApiKeysConfig = Field(default_factory=ApiKeysConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def validate_production_requirements(self) -> BoxBotConfig:
        """Validate that production-critical secrets are set when sandbox is enforced.

        When sandbox.enforce is True (the default), the system is assumed to
        be running in production. Certain secrets must be configured:
        - WHATSAPP_APP_SECRET for webhook signature validation
        """
        if self.sandbox.enforce:
            if self.api_keys.whatsapp_access_token and not self.api_keys.whatsapp_app_secret:
                logger.warning(
                    "WHATSAPP_APP_SECRET is not set but sandbox.enforce is true. "
                    "Webhook signature validation will be skipped — set "
                    "WHATSAPP_APP_SECRET in .env for production use."
                )
        return self
