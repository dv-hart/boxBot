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
    from boxbot.displays.manager import DisplayManager

    manager = DisplayManager()
    await manager.start()
    logger.info("Display manager started")
    return manager


async def _init_photo_intake(photo_store: Any) -> Any:
    """Initialise and start the photo intake pipeline."""
    from boxbot.photos.intake import IntakePipeline

    pipeline = IntakePipeline(photo_store)
    await pipeline.start()
    logger.info("Photo intake pipeline started")
    return pipeline


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

    router = MessageRouter(auth=auth, whatsapp=whatsapp)
    logger.info("Communication layer initialised (WhatsApp + router)")
    return router


async def _init_agent(memory_store: Any) -> Any:
    """Initialise and start the BoxBotAgent."""
    from boxbot.core.agent import BoxBotAgent

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

    # Shutdown order is reverse of initialisation
    shutdown_order = [
        "agent",
        "communication",
        "photo_intake",
        "display_manager",
        "scheduler",
        "photo_store",
        "memory_store",
    ]

    for name in shutdown_order:
        instance = subsystems.get(name)
        if instance is None:
            continue

        try:
            if name == "agent":
                await instance.stop()
            elif name == "communication":
                # MessageRouter does not have a stop method currently
                pass
            elif name == "photo_intake":
                await instance.stop()
            elif name == "display_manager":
                await instance.stop()
            elif name == "scheduler":
                await instance.stop()
            elif name == "photo_store":
                await instance.close()
            elif name == "memory_store":
                await instance.close()
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

        # Photo intake — background processing pipeline
        photo_intake = await _init_photo_intake(photo_store)
        subsystems["photo_intake"] = photo_intake

        # Communication — WhatsApp client and message routing
        comm_router = await _init_communication()
        if comm_router is not None:
            subsystems["communication"] = comm_router

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
