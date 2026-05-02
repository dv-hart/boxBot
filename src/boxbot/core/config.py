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

from boxbot.core.paths import PERCEPTION_MODELS_DIR, PHOTOS_DIR

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
        "AWS_REGION": "aws_region",
        "BOXBOT_SQS_QUEUE_URL": "whatsapp_sqs_queue_url",
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


class HardwareCameraConfig(BaseModel):
    """Camera hardware-level settings (rotation, stream resolutions, colour)."""

    rotation: int = 180
    lores_resolution: list[int] = Field(default_factory=lambda: [320, 240])
    photo_resolution: list[int] = Field(default_factory=lambda: [4608, 2592])

    # Colour correction. The IMX708 NoIR has no IR-cut filter, so daylight
    # captures pick up an IR cast. When both fields are set, AWB is disabled
    # and these fixed values are applied via picamera2 set_controls.
    # Calibrate per install — see scripts/calibrate_colour.py.
    colour_gains: list[float] | None = None  # [red_gain, blue_gain]
    colour_correction_matrix: list[float] | None = None  # 9 values, row-major 3x3

    # Saturation multiplier. 1.0 = normal colour, 0.0 = monochrome. Useful
    # stopgap on NoIR sensors with severe IR cast — produces honest greyscale
    # rather than colour-shifted nonsense. Restore to 1.0 once an IR-cut
    # filter is installed.
    saturation: float = 1.0


class HardwareHailoConfig(BaseModel):
    """Hailo NPU configuration (model paths, preloading)."""

    models: dict[str, str] = Field(default_factory=lambda: {
        "yolo": "/usr/share/hailo-models/yolov5s_personface_h8l.hef",
        "reid": str(PERCEPTION_MODELS_DIR / "repvgg_a0_person_reid_512.hef"),
    })
    preload_models: bool = True


class HardwareMicrophoneConfig(BaseModel):
    """Microphone hardware settings (ReSpeaker 4-Mic Array)."""

    device_name: str = "ReSpeaker"
    sample_rate: int = 16000
    capture_channels: int = 6
    output_channel: int = 0  # ch 0 = processed/beamformed
    chunk_duration_ms: int = 64  # ~1024 frames at 16kHz
    doa_enabled: bool = True
    led_brightness: float = 0.5


class HardwareSpeakerConfig(BaseModel):
    """Speaker hardware settings."""

    device_name: str = "boxbot_speaker"
    sample_rate: int = 24000
    default_volume: float = 1.0
    # Soft-limited pre-output gain in dB. ElevenLabs TTS output peaks near
    # 0 dBFS but has ~20 dB crest factor, so raw RMS is well below what the
    # amp wants. +6 dB with tanh ceiling lifts perceived loudness without
    # clipping peaks.
    gain_db: float = 6.0
    # AEC reference path: a second OutputStream pointed at the ReSpeaker
    # USB playback device, fed the same TTS audio resampled to 16 kHz.
    # The XMOS XVF3000 chip uses this as its echo-cancellation reference
    # and subtracts it from the mic capture, so BB doesn't hear its own
    # voice. Set to ``null`` to disable (BB will hear itself).
    aec_reference_device: str | None = "ReSpeaker"
    # If true (default), failure to open the AEC reference path is fatal
    # at boot. Without AEC, BB transcribes its own TTS and feeds it back
    # as fake user input — conversations derail every time. Refusing to
    # start is far better than silently degrading. Set to false only on
    # hardware where you know AEC is handled elsewhere (e.g. an external
    # echo canceller, or a dev box without a ReSpeaker).
    aec_required: bool = True
    # AEC device-discovery retry: USB enumeration on cold boot is
    # occasionally slow enough that the ReSpeaker isn't in PortAudio's
    # device list on the first query. Retry a few times with a small
    # delay before giving up.
    aec_discovery_retries: int = 5
    aec_discovery_retry_delay: float = 0.2


class HardwareConfig(BaseModel):
    """Hardware abstraction layer configuration."""

    camera: HardwareCameraConfig = Field(default_factory=HardwareCameraConfig)
    hailo: HardwareHailoConfig = Field(default_factory=HardwareHailoConfig)
    microphone: HardwareMicrophoneConfig = Field(default_factory=HardwareMicrophoneConfig)
    speaker: HardwareSpeakerConfig = Field(default_factory=HardwareSpeakerConfig)


class WakeWordConfig(BaseModel):
    """Wake word detection settings."""

    engine: str = "openwakeword"
    word: str = "hey_jarvis"  # built-in model; swap to "bb" when custom model ready
    confidence_threshold: float = 0.7
    model_path: str | None = None  # None = use built-in model


class VADConfig(BaseModel):
    """Voice activity detection settings."""

    threshold: float = 0.5
    min_speech_duration: int = 250  # ms
    min_silence_duration: int = 100  # ms


class TurnDetectionConfig(BaseModel):
    """Turn detection / utterance boundary settings."""

    silence_threshold: int = 800  # ms before finalizing utterance
    max_utterance_duration: int = 60  # seconds hard cap
    inter_utterance_gap: int = 300  # ms between utterances


class DiarizationConfig(BaseModel):
    """Speaker diarization settings."""

    engine: str = "pyannote"
    model: str = "pyannote/speaker-diarization-3.1"
    embedding_model: str = "pyannote/wespeaker-voxceleb-resnet34-LM"
    min_speakers: int = 1
    max_speakers: int = 6
    match_threshold: float = 0.65


class STTConfig(BaseModel):
    """Speech-to-text provider settings."""

    provider: str = "elevenlabs"
    model: str = "scribe_v2"
    language: str = "en"


class TTSConfig(BaseModel):
    """Text-to-speech provider settings."""

    provider: str = "elevenlabs"
    voice_id: str = ""  # must be configured
    model: str = "eleven_turbo_v2_5"
    stability: float = 0.5
    similarity_boost: float = 0.75
    optimize_streaming_latency: int = 3


class SessionConfig(BaseModel):
    """Voice session lifecycle settings."""

    active_timeout: int = 30  # seconds before suspending
    suspend_timeout: int = 180  # seconds before ending
    max_session_duration: int = 600


class VoiceConfig(BaseModel):
    """Voice pipeline settings."""

    wake_word: WakeWordConfig = Field(default_factory=WakeWordConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    turn_detection: TurnDetectionConfig = Field(default_factory=TurnDetectionConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)


class PerceptionConfig(BaseModel):
    """Perception pipeline thresholds and settings."""

    motion_threshold: float = 12.0
    # YOLOv5s-personface confidence threshold for person detections.
    # Empirically the model returns ~0.20 for a desk-mounted camera
    # showing torso + face only (typical "BB sat on a desk" view), so
    # the default has to be quite low. Confirmed on hardware:
    # confidence=0.204 covering ~the full frame for the user at desk
    # distance, repeated across captures. 0.15 leaves enough margin
    # below that to catch the signal reliably without dipping into
    # the noise floor we saw in idle frames (no detections at all,
    # not low-confidence background hits). Tighter thresholds make
    # sense for floor-mounted units that see full bodies.
    person_confidence_threshold: float = 0.15
    reid_high_threshold: float = 0.85
    reid_low_threshold: float = 0.60
    speaker_threshold: float = 0.75
    presence_timeout: int = 30
    heartbeat_interval: int = 5
    max_visual_embeddings: int = 200
    max_voice_embeddings: int = 50
    crop_retention_days: int = 1
    crop_retention_days_debug: int = 7
    doa_forward_angle: int = 0  # ReSpeaker angle that maps to camera center
    camera_hfov: int = 120  # Pi Camera Module 3 Wide horizontal FOV
    voice_match_threshold: float = 0.60  # cosine similarity for voice ID


class MemoryConfig(BaseModel):
    """Memory system settings."""

    max_context_memories: int = 15
    decay_rate: float = 0.98
    archive_threshold: float = 0.1

    # Dream phase (PR1: deterministic clustering + dedup batch).
    # Default ON for the audit-only soft-launch — dream cycles run,
    # candidates and decisions are logged, but NO mutations are applied
    # to the memory store. Flip to ``False`` after a soft-launch period
    # to let the dream phase actually consolidate memories.
    dream_audit_only: bool = True
    # Cron expression for the nightly dream cycle. Defaults to 3 AM
    # every day (server-local interpretation of CronExpr — UTC by
    # default unless the host is on a different TZ).
    dream_cron: str = "0 3 * * *"
    # Hard ceiling on dedup pairs sent to the model in a single cycle.
    # Above this, lowest-confidence pairs are dropped before submission.
    dream_max_dedup_pairs: int = 30


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

    storage_path: str = str(PHOTOS_DIR)
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
    """Sandbox execution settings.

    Sandbox state lives under ``runtime_dir`` — outside the project tree
    on purpose. Putting it under ``/var/lib`` (or a similar system path)
    means:

    - The boxbot-sandbox system user doesn't need traverse permission on
      the operator's home directory just to reach the venv.
    - The repository can be cloned anywhere by anyone (this is an
      open-source project) without baking install-specific paths into
      the codebase.

    Override ``runtime_dir`` in ``config.yaml`` to use a different
    location (e.g. ``/opt/boxbot-sandbox`` on systems where ``/var/lib``
    is locked down). Setup script ``scripts/setup-sandbox.sh`` reads the
    same default and creates the directory tree.
    """

    runtime_dir: str = "/var/lib/boxbot-sandbox"
    user: str = "boxbot-sandbox"
    timeout: int = 30
    memory_limit_mb: int = 256
    allow_network: bool = True
    install_approval_timeout: int = 300
    install_approval_channels: list[str] = Field(
        default_factory=lambda: ["display", "whatsapp"]
    )
    # Seccomp filter mode. Three values:
    #   "disabled" — no syscall filter (current behaviour)
    #   "log"      — install filter, kernel logs forbidden syscalls but
    #                does NOT kill the process. Use as a soak period to
    #                verify the rule set against real workloads.
    #   "enforce"  — install filter; first forbidden syscall kills the
    #                process with SIGSYS.
    # Default is "log" so deployments that have python3-seccomp
    # installed start gathering audit data immediately. The bootstrap
    # gracefully degrades to no-filter if the library isn't present.
    # The kill-switch ``BOXBOT_SECCOMP_DISABLE=1`` env var bypasses
    # this entirely — for emergencies when the filter breaks something.
    seccomp_mode: str = "log"

    @model_validator(mode="after")
    def validate_seccomp_mode(self) -> SandboxConfig:
        if self.seccomp_mode not in {"disabled", "log", "enforce"}:
            raise ValueError(
                f"seccomp_mode must be one of disabled/log/enforce, "
                f"got {self.seccomp_mode!r}"
            )
        return self

    @property
    def venv_path(self) -> str:
        return f"{self.runtime_dir.rstrip('/')}/venv"

    @property
    def scripts_dir(self) -> str:
        return f"{self.runtime_dir.rstrip('/')}/scripts"

    @property
    def output_dir(self) -> str:
        return f"{self.runtime_dir.rstrip('/')}/output"

    @property
    def tmp_dir(self) -> str:
        return f"{self.runtime_dir.rstrip('/')}/tmp"


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
    aws_region: str = "us-west-2"
    whatsapp_sqs_queue_url: str | None = None

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
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    photos: PhotosConfig = Field(default_factory=PhotosConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    sdk: SDKConfig = Field(default_factory=SDKConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    api_keys: ApiKeysConfig = Field(default_factory=ApiKeysConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
