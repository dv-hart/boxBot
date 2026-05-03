"""boxBot application entry point.

Initialises all subsystems, starts the agent loop, and handles graceful
shutdown on SIGTERM/SIGINT signals. This module ties together configuration,
logging, memory, photos, scheduling, displays, communication, and the
agent itself.

The ``main()`` function is the CLI entry point defined in pyproject.toml:

    [project.scripts]
    boxbot = "boxbot.core.main:main"

Usage:
    # Run directly
    python3 -m boxbot.core.main

    # Or via the installed entry point
    boxbot
    boxbot --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import logging.handlers
import os
import signal
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("boxbot")


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------


def _load_dotenv(path: str | Path = ".env") -> None:
    """Load environment variables from a .env file.

    Tries python-dotenv first (if installed), otherwise falls back to
    simple manual parsing. Lines starting with ``#`` are comments.
    ``KEY=VALUE`` pairs are loaded into ``os.environ`` without overwriting
    existing values.

    Args:
        path: Path to the .env file. Defaults to ``.env`` in the CWD.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(path, override=False)
        return
    except ImportError:
        pass

    # Manual fallback
    env_path = Path(path)
    if not env_path.exists():
        return

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Remove surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            # Only set if not already in environment
            if key and key not in os.environ:
                os.environ[key] = value


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(
    level: str = "INFO",
    log_file: str = "logs/boxbot.log",
    max_size_mb: int = 50,
    backup_count: int = 5,
) -> None:
    """Configure logging with rotating file handler and console output.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to the log file.
        max_size_mb: Maximum log file size in MB before rotation.
        backup_count: Number of rotated log files to keep.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Root logger
    root = logging.getLogger()
    root.setLevel(log_level)

    # Clear any existing handlers (prevents duplicate output in tests)
    root.handlers.clear()

    # Format
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(log_level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # Rotating file handler
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_path),
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except OSError as e:
        # Log file might not be writable (e.g., in tests or CI)
        logging.warning("Could not create log file %s: %s", log_path, e)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="boxbot",
        description="boxBot — an open-source Claude agent in a box.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file (default: config/config.yaml or BOXBOT_CONFIG env var)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=".env",
        help="Path to .env file (default: .env)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Override log level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Subsystem initialisation
# ---------------------------------------------------------------------------


async def _init_hal(config: Any) -> dict[str, Any]:
    """Initialise Hardware Abstraction Layer modules.

    Starts system monitor, Hailo NPU, and camera in dependency order.
    Returns dict of module name to instance. Failures in individual
    modules are logged but do not prevent startup (graceful degradation).
    """
    modules: dict[str, Any] = {}

    # Soft-fail handlers below catch this. Anything else (including the
    # fatal HardwareInitFatal from the speaker's AEC path) propagates out
    # of _init_hal and aborts boot — by design.
    from boxbot.hardware.base import HardwareUnavailableError

    # System monitor (always available)
    from boxbot.hardware.system import System

    system = System()
    await system.start()
    modules["system"] = system
    logger.info("System monitor started")

    # Hailo NPU
    try:
        from boxbot.hardware.hailo import Hailo

        hailo = Hailo(models=config.hardware.hailo.models)
        await hailo.start()
        system.register_module(hailo)
        system.set_hailo_ref(hailo)
        modules["hailo"] = hailo
        logger.info("Hailo NPU started")
    except Exception:
        logger.warning("Hailo NPU not available — perception will be disabled", exc_info=True)

    # Camera
    try:
        from boxbot.hardware.camera import Camera, set_camera

        cam_cfg = config.hardware.camera
        camera = Camera(
            rotation=cam_cfg.rotation,
            main_resolution=tuple(config.camera.resolution),
            lores_resolution=tuple(cam_cfg.lores_resolution),
            scan_fps=config.camera.scan_fps,
            colour_gains=(
                tuple(cam_cfg.colour_gains)  # type: ignore[arg-type]
                if cam_cfg.colour_gains is not None
                else None
            ),
            colour_correction_matrix=(
                tuple(cam_cfg.colour_correction_matrix)
                if cam_cfg.colour_correction_matrix is not None
                else None
            ),
            saturation=cam_cfg.saturation,
        )
        await camera.start()
        system.register_module(camera)
        modules["camera"] = camera
        # Publish for sandbox action handlers (bb.camera.capture etc.)
        set_camera(camera)
        logger.info("Camera started")
    except Exception:
        logger.warning("Camera not available — perception will be disabled", exc_info=True)

    # Speaker BEFORE microphone. The speaker opens two streams: the
    # HDMI audible output and the AEC reference path on the ReSpeaker
    # USB playback channel. Once the microphone has claimed the
    # ReSpeaker capture side, PortAudio sometimes hides the same
    # device's output side from query_devices() (USB shared resource
    # race). Opening the AEC reference *first* avoids that. The speaker
    # also raises HardwareInitFatal — not a HardwareUnavailableError —
    # if the AEC path cannot come up and aec_required is true; that
    # propagates out of this try/except and aborts boot, which is what
    # we want (BB hearing itself derails every conversation).
    try:
        from boxbot.hardware.speaker import Speaker

        speaker = Speaker(config=config.hardware.speaker)
        await speaker.start()
        system.register_module(speaker)
        modules["speaker"] = speaker
        logger.info("Speaker started")
    except HardwareUnavailableError:
        logger.warning(
            "Speaker not available — TTS will be disabled", exc_info=True
        )

    # Microphone (ReSpeaker 4-Mic Array)
    try:
        from boxbot.hardware.microphone import Microphone

        microphone = Microphone(config=config.hardware.microphone)
        await microphone.start()
        system.register_module(microphone)
        modules["microphone"] = microphone
        logger.info("Microphone started")
    except HardwareUnavailableError:
        logger.warning(
            "Microphone not available — voice pipeline will be disabled",
            exc_info=True,
        )

    return modules


async def _stop_hal(modules: dict[str, Any]) -> None:
    """Stop HAL modules in reverse startup order."""
    for name in ("screen", "speaker", "microphone", "camera", "hailo", "system"):
        module = modules.get(name)
        if module is None:
            continue
        try:
            await module.stop()
            logger.info("Stopped HAL module: %s", name)
        except Exception:
            logger.exception("Error stopping HAL module: %s", name)


async def _init_perception(hal_modules: dict[str, Any], config: Any) -> Any | None:
    """Initialise the perception pipeline if camera and Hailo are available."""
    camera = hal_modules.get("camera")
    hailo = hal_modules.get("hailo")
    microphone = hal_modules.get("microphone")

    if camera is None or hailo is None:
        logger.warning(
            "Perception pipeline disabled (camera=%s, hailo=%s)",
            "ok" if camera else "missing",
            "ok" if hailo else "missing",
        )
        return None

    from boxbot.perception.pipeline import PerceptionPipeline

    pipeline = PerceptionPipeline(
        camera=camera,
        hailo=hailo,
        microphone=microphone,
        motion_threshold=config.perception.motion_threshold,
        reid_high_threshold=config.perception.reid_high_threshold,
        reid_low_threshold=config.perception.reid_low_threshold,
        presence_timeout=config.perception.presence_timeout,
        heartbeat_interval=config.perception.heartbeat_interval,
        scan_fps=config.camera.scan_fps,
        voice_match_threshold=config.perception.voice_match_threshold,
        doa_forward_angle=config.perception.doa_forward_angle,
        camera_hfov=config.perception.camera_hfov,
        crop_retention_days=config.perception.crop_retention_days,
        crop_retention_days_debug=config.perception.crop_retention_days_debug,
        person_confidence_threshold=(
            config.perception.person_confidence_threshold
        ),
    )
    await pipeline.start()
    logger.info("Perception pipeline started")
    return pipeline


async def _init_memory_store() -> Any:
    """Initialise and return the MemoryStore."""
    from boxbot.memory.store import MemoryStore

    store = MemoryStore()
    await store.initialize()
    logger.info("Memory store initialised")
    return store


async def _init_photo_store() -> Any:
    """Initialise and return the PhotoStore."""
    from boxbot.photos.store import PhotoStore

    store = PhotoStore()
    await store.initialize()
    logger.info("Photo store initialised")
    return store


async def _init_scheduler() -> Any:
    """Initialise and start the Scheduler."""
    from boxbot.core.scheduler import Scheduler

    scheduler = Scheduler()
    await scheduler.start()
    logger.info("Scheduler started")
    return scheduler


async def _init_display_manager() -> Any:
    """Initialise and start the DisplayManager."""
    from boxbot.displays.manager import DisplayManager, set_display_manager

    manager = DisplayManager()
    await manager.start()
    # Publish for sandbox action handlers (bb.photos.show_on_screen, …)
    # and the switch_display tool.
    set_display_manager(manager)
    logger.info("Display manager started")
    return manager


async def _init_photo_intake(photo_store: Any) -> Any:
    """Initialise and start the photo intake pipeline."""
    from boxbot.photos.intake import IntakePipeline, set_intake_pipeline

    pipeline = IntakePipeline(photo_store)
    await pipeline.start()
    # Publish so sandbox action handlers (bb.photos.ingest) can reach
    # the live pipeline without DI plumbing.
    set_intake_pipeline(pipeline)
    logger.info("Photo intake pipeline started")
    return pipeline


async def _init_voice(hal_modules: dict[str, Any], config: Any) -> Any | None:
    """Initialise the voice pipeline if microphone is available.

    Returns the VoiceSession if initialised, otherwise None.
    """
    microphone = hal_modules.get("microphone")
    speaker = hal_modules.get("speaker")

    if microphone is None:
        logger.warning("Voice pipeline disabled (microphone not available)")
        return None

    if speaker is None:
        logger.warning("Voice pipeline degraded (speaker not available, no TTS)")

    try:
        from boxbot.communication.voice import VoiceSession

        session = VoiceSession(
            microphone=microphone,
            speaker=speaker,
            config=config.voice,
        )
        await session.start()
        logger.info("Voice pipeline started")
        return session
    except Exception:
        logger.warning("Voice pipeline failed to start", exc_info=True)
        return None


async def _init_communication() -> Any | None:
    """Initialise the communication layer (WhatsApp + message router).

    Returns the MessageRouter if WhatsApp credentials are configured,
    otherwise None.
    """
    from boxbot.communication.auth import AuthManager
    from boxbot.communication.router import MessageRouter
    from boxbot.communication.whatsapp import WhatsAppClient
    from boxbot.core.config import get_config

    config = get_config()

    # WhatsApp requires an access token and phone number ID
    if not config.api_keys.whatsapp_access_token or not config.api_keys.whatsapp_phone_number_id:
        logger.warning(
            "WhatsApp credentials not configured. "
            "Communication layer will be limited to voice."
        )
        return None

    whatsapp = WhatsAppClient(
        access_token=config.api_keys.whatsapp_access_token,
        phone_number_id=config.api_keys.whatsapp_phone_number_id,
    )

    auth = AuthManager()
    await auth.init_db()

    # Register singletons so the agent's output dispatcher can reach them
    # without being passed a direct reference.
    from boxbot.communication.auth import set_auth_manager
    from boxbot.communication.whatsapp import set_whatsapp_client
    set_whatsapp_client(whatsapp)
    set_auth_manager(auth)

    router = MessageRouter(auth=auth, whatsapp=whatsapp)
    logger.info("Communication layer initialised (WhatsApp + router)")

    # First-run setup: if no admins are registered yet AND we haven't
    # already seeded setup todos, drop a small ordered backlog. The
    # agent picks these up on its next wake cycle and runs the
    # onboarding skill against them. Idempotent: re-runs see the
    # `setup:bootstrap` todo and bail.
    await _maybe_seed_setup_todos(auth)

    return router


async def _maybe_seed_setup_todos(auth: Any) -> None:
    """Seed first-run setup todos if no admin exists and none were seeded.

    The skill is the *how*; these todos are the *what*. Each is a
    discrete, observable, recoverable unit of work. If any todo
    tagged ``setup:`` already exists, this is a no-op even if the
    admin record was somehow deleted later — we don't want to re-seed
    a partially completed run.
    """
    from boxbot.core import scheduler

    if any(u.role == "admin" for u in await auth.list_users()):
        return

    existing = await scheduler.list_todos()
    if any(
        (t.get("description") or "").startswith("setup:") for t in existing
    ):
        return

    seeds = [
        (
            "setup:bootstrap — register the first admin",
            (
                "Mint a single-use bootstrap code with "
                "bb.auth.generate_bootstrap_code(), then surface it on "
                "the HDMI screen with switch_display(\"notice\", "
                "args={\"title\": \"Welcome to boxBot!\", \"lines\": ["
                "\"Text this code to BB's WhatsApp number:\", "
                "\"Code: <CODE>\", \"Expires in 10 minutes\"]}). "
                "Wait for the UserRegistered event (role=admin). Mark "
                "this todo complete after registration succeeds."
            ),
        ),
        (
            "setup:greet_admin — welcome the new admin and learn their name",
            (
                "On UserRegistered (role=admin), reply via WhatsApp "
                "with a warm welcome and ask what they'd like you to "
                "call them. When they answer, save their preferred "
                "name to system memory."
            ),
        ),
        (
            "setup:voice_fingerprint — link the admin to a voice profile",
            (
                "Send a WhatsApp message inviting the admin to come "
                "say hi at the box. When they next address you in a "
                "voice conversation, run the voice-onboarding "
                "procedure (identify_person) so you recognize their "
                "voice on future visits. Mark complete on a successful "
                "create/confirm outcome."
            ),
        ),
        (
            "setup:household — note household basics",
            (
                "Ask the admin (via WhatsApp or voice — whichever they "
                "prefer) about other household members worth knowing, "
                "the city to use for weather, and any house "
                "preferences. Save anything durable to system memory. "
                "This todo is optional — mark complete or cancel based "
                "on what they share."
            ),
        ),
    ]

    for description, notes in seeds:
        await scheduler.create_todo(
            description=description,
            notes=notes,
            source="setup",
        )

    logger.info("Seeded %d first-run setup todos", len(seeds))


async def _init_whatsapp_inbound(router: Any) -> Any | None:
    """Start the SQS-backed WhatsApp webhook poller, if configured.

    Returns the poller if started, otherwise None. Required env:
    BOXBOT_SQS_QUEUE_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY.
    """
    from boxbot.communication.whatsapp_inbound import WhatsAppInboundPoller
    from boxbot.core.config import get_config

    config = get_config()
    queue_url = config.api_keys.whatsapp_sqs_queue_url
    key = config.api_keys.aws_access_key_id
    secret = config.api_keys.aws_secret_access_key
    if not (queue_url and key and secret):
        logger.info(
            "WhatsApp SQS poller not configured "
            "(set BOXBOT_SQS_QUEUE_URL + AWS credentials to enable)"
        )
        return None

    poller = WhatsAppInboundPoller(
        router=router,
        queue_url=queue_url,
        region=config.api_keys.aws_region,
        access_key_id=key,
        secret_access_key=secret,
    )
    await poller.start()
    return poller


async def _init_agent(memory_store: Any) -> Any:
    """Initialise and start the BoxBotAgent."""
    from boxbot.core.agent import BoxBotAgent
    from boxbot.core.agent_state import get_agent_state_tracker

    # Start the state tracker before the agent so it sees the first events
    await get_agent_state_tracker().start()

    agent = BoxBotAgent(memory_store=memory_store)
    await agent.start()
    logger.info("Agent started")
    return agent


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


async def _shutdown(
    subsystems: dict[str, Any],
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Graceful shutdown: stop subsystems in reverse initialisation order.

    Args:
        subsystems: Dict of subsystem name to instance, in init order.
        loop: The running event loop.
    """
    logger.info("Shutting down boxBot...")

    # Shutdown order is reverse of initialisation.
    # Perception stops before HAL; HAL stops last.
    shutdown_order = [
        "agent",
        "voice",
        "whatsapp_inbound",
        "communication",
        "photo_intake",
        "perception",
        "display_manager",
        "scheduler",
        "photo_store",
        "memory_store",
        "hal",
    ]

    for name in shutdown_order:
        instance = subsystems.get(name)
        if instance is None:
            continue

        try:
            if name == "agent":
                from boxbot.core.agent_state import get_agent_state_tracker
                await get_agent_state_tracker().stop()
                await instance.stop()
            elif name == "voice":
                await instance.stop()
            elif name == "whatsapp_inbound":
                await instance.stop()
            elif name == "communication":
                # MessageRouter does not have a stop method currently
                pass
            elif name == "photo_intake":
                await instance.stop()
            elif name == "perception":
                await instance.stop()
            elif name == "display_manager":
                await instance.stop()
            elif name == "scheduler":
                await instance.stop()
            elif name == "photo_store":
                await instance.close()
            elif name == "memory_store":
                await instance.close()
            elif name == "hal":
                await _stop_hal(instance)
            logger.info("Stopped %s", name)
        except Exception:
            logger.exception("Error stopping %s", name)

    logger.info("boxBot shutdown complete")


# ---------------------------------------------------------------------------
# Async main
# ---------------------------------------------------------------------------


async def _async_main() -> None:
    """Async main — initialise all subsystems, run the agent, and handle shutdown.

    This is the core async entry point. It:
    1. Loads .env and configuration
    2. Sets up logging
    3. Initialises subsystems in dependency order
    4. Registers signal handlers for graceful shutdown
    5. Runs until a shutdown signal is received
    6. Tears down subsystems in reverse order
    """
    args = _parse_args()

    # 1. Load environment variables from .env
    _load_dotenv(args.env)

    # 2. Load configuration
    from boxbot.core.config import load_config

    config = load_config(args.config)

    # 3. Set up logging (CLI --log-level overrides config)
    log_level = args.log_level or config.logging.level
    _setup_logging(
        level=log_level,
        log_file=config.logging.file,
        max_size_mb=config.logging.max_size_mb,
        backup_count=config.logging.backup_count,
    )

    logger.info("=" * 60)
    logger.info("boxBot starting up")
    logger.info("=" * 60)
    logger.info(
        "Models: large=%s, small=%s",
        config.models.large,
        config.models.small,
    )

    # Track subsystems for shutdown
    subsystems: dict[str, Any] = {}

    # Event to signal shutdown
    shutdown_event = asyncio.Event()

    # 4. Register signal handlers
    loop = asyncio.get_running_loop()

    def _signal_handler(sig: signal.Signals) -> None:
        logger.info("Received signal %s, initiating shutdown...", sig.name)
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _signal_handler, sig)
        except NotImplementedError:
            # Windows does not support add_signal_handler
            signal.signal(sig, lambda s, f: shutdown_event.set())

    # 5. Initialise subsystems in dependency order
    try:
        # HAL — hardware abstraction layer (system, hailo, camera)
        hal_modules = await _init_hal(config)
        subsystems["hal"] = hal_modules

        # Memory store — foundation for the agent and many other subsystems
        memory_store = await _init_memory_store()
        subsystems["memory_store"] = memory_store

        # Photo store — independent of memory, needed by photo intake
        photo_store = await _init_photo_store()
        subsystems["photo_store"] = photo_store

        # Scheduler — manages triggers and to-do items, background loop
        scheduler = await _init_scheduler()
        subsystems["scheduler"] = scheduler

        # Display manager — manages screen output
        display_manager = await _init_display_manager()
        subsystems["display_manager"] = display_manager

        # Start idle display rotation
        display_manager.start_rotation()

        # Screen HAL — renders display frames to HDMI via pygame
        try:
            from boxbot.hardware.screen import Screen

            screen = Screen(
                display_manager=display_manager,
                brightness=config.display.brightness,
            )
            await screen.start()
            hal_modules.get("system") and hal_modules["system"].register_module(screen)
            subsystems["screen"] = screen
            logger.info("Screen started")
        except Exception:
            logger.warning("Screen not available — display will not render to HDMI", exc_info=True)

        # Perception pipeline — visual detection and re-identification
        perception = await _init_perception(hal_modules, config)
        if perception is not None:
            subsystems["perception"] = perception

        # Photo intake — background processing pipeline
        photo_intake = await _init_photo_intake(photo_store)
        subsystems["photo_intake"] = photo_intake

        # Voice pipeline — wake word, VAD, STT, TTS, diarization
        voice_session = await _init_voice(hal_modules, config)
        if voice_session is not None:
            subsystems["voice"] = voice_session

        # Communication — WhatsApp client and message routing
        comm_router = await _init_communication()
        if comm_router is not None:
            subsystems["communication"] = comm_router
            inbound = await _init_whatsapp_inbound(comm_router)
            if inbound is not None:
                subsystems["whatsapp_inbound"] = inbound

        # Agent — the brain, subscribes to events from all other subsystems
        agent = await _init_agent(memory_store)
        subsystems["agent"] = agent

        logger.info("All subsystems initialised, boxBot is running")

    except Exception:
        logger.exception("Fatal error during subsystem initialisation")
        await _shutdown(subsystems, loop)
        return

    # 6. Run until shutdown signal
    try:
        await shutdown_event.wait()
    except asyncio.CancelledError:
        pass

    # 7. Graceful shutdown
    await _shutdown(subsystems, loop)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """boxBot application entry point.

    Initialises all subsystems, starts the agent loop, and handles
    graceful shutdown on signals. This is the function referenced by
    the ``boxbot`` console script in pyproject.toml.
    """
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        # asyncio.run already handles this, but just in case
        pass


if __name__ == "__main__":
    main()
