"""HAL base classes and shared types.

All hardware modules implement the HardwareModule ABC. Shared dataclasses
(ModelInfo, SystemHealth) and enums (HealthStatus) live here so they can
be imported without pulling in hardware-specific libraries.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from boxbot.core.events import Event, get_event_bus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Events (published by HAL modules to the internal event bus)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HardwareHealthChanged(Event):
    """A hardware module's health status changed.

    Source: Any HAL module
    Consumers: System monitor, Display, Agent
    """

    module: str = ""
    status: str = ""  # HealthStatus.value
    detail: str = ""


@dataclass(frozen=True)
class ThermalWarning(Event):
    """SoC or Hailo temperature crossed a warning threshold.

    Source: system.py
    Consumers: Perception (reduce scan FPS), Photo intake (pause)
    """

    source: str = ""  # "soc" or "hailo"
    temperature: float = 0.0
    threshold: str = ""  # "warning", "throttle", "critical"


@dataclass(frozen=True)
class ShutdownRequested(Event):
    """System shutdown was requested (SIGTERM, SIGINT, or manual).

    Source: system.py
    Consumers: Agent, Memory, Scheduler, Photos
    """

    reason: str = ""  # "signal", "thermal", "manual"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class HealthStatus(Enum):
    """Health status for a hardware module."""

    OK = "ok"
    DEGRADED = "degraded"
    ERROR = "error"
    STOPPED = "stopped"


# ---------------------------------------------------------------------------
# Shared dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AudioChunk:
    """A chunk of raw PCM audio data from the microphone.

    Distributed to all registered consumers by the Microphone HAL.
    Data is mono int16 little-endian PCM for the selected output channel.
    """

    data: bytes  # raw PCM bytes (int16 LE)
    timestamp: float  # time.monotonic() at capture
    sample_rate: int  # e.g. 16000
    channels: int  # always 1 (mono, post channel extraction)
    frames: int  # number of audio frames in this chunk


@dataclass
class ModelInfo:
    """Metadata for a loaded Hailo model."""

    name: str
    path: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


@dataclass
class SystemHealth:
    """Aggregate system health snapshot."""

    soc_temp: float | None
    hailo_temp: float | None
    memory_used_pct: float
    disk_used_pct: float
    module_health: dict[str, HealthStatus] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class HardwareUnavailableError(Exception):
    """Raised when required hardware is not found or not responsive.

    Soft failure: ``main._init_hal`` catches this per-module and lets the
    rest of boxbot run in a degraded state. Use this for hardware that
    can be missing without compromising the agent's correctness — e.g.
    no camera means no perception, no screen means no display, but BB
    can still hold a voice conversation.
    """


class HardwareInitFatal(RuntimeError):
    """Raised when a HAL component fails in a way that cannot be papered
    over. Deliberately *not* a subclass of :class:`HardwareUnavailableError`
    so the soft-fail handlers in ``main._init_hal`` cannot swallow it.

    Current trigger: the speaker can't open the AEC reference path, and
    ``speaker.aec_required`` is true. Without an AEC reference, BB hears
    its own TTS, transcribes it, and feeds it back as a fake user turn —
    every conversation derails. That is worse than not booting; refuse
    to start up rather than silently degrade.
    """


class HardwareModule(ABC):
    """Abstract base class for all HAL modules.

    Subclasses must define a ``name`` class attribute and implement
    ``start()``, ``stop()``, and ``is_available``.
    """

    name: str  # set by subclass as a class attribute

    def __init__(self) -> None:
        self._started: bool = False
        self._last_health: HealthStatus = HealthStatus.STOPPED

    @abstractmethod
    async def start(self) -> None:
        """Initialize the hardware.

        Called during system startup. May raise ``HardwareUnavailableError``
        if the device is not found.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Release hardware resources.

        Must be safe to call even if ``start()`` was never called or
        previously failed.
        """

    async def health_check(self) -> HealthStatus:
        """Return current health status.

        Default implementation returns OK if started, STOPPED otherwise.
        Override for device-specific checks (temperature, USB connected).
        """
        return HealthStatus.OK if self._started else HealthStatus.STOPPED

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether the hardware is connected and responsive."""

    @property
    def is_started(self) -> bool:
        """Whether ``start()`` has completed successfully."""
        return self._started

    async def _emit_health(self, status: HealthStatus, detail: str = "") -> None:
        """Publish a health change event to the event bus.

        Only publishes if the status actually changed from the last
        emitted value, to avoid event spam.
        """
        if status == self._last_health:
            return
        self._last_health = status
        logger.info(
            "Hardware %s health: %s%s",
            self.name,
            status.value,
            f" ({detail})" if detail else "",
        )
        bus = get_event_bus()
        await bus.publish(
            HardwareHealthChanged(
                module=self.name,
                status=status.value,
                detail=detail,
            )
        )


# Consumer callback type: async callable receiving AudioChunk
AudioConsumer = Callable[[AudioChunk], Awaitable[None]]


class AudioFanout(HardwareModule):
    """A HardwareModule that fans one PCM stream out to N async consumers.

    Shared by every microphone backend — today the ReSpeaker array
    (``hardware/microphone.py``). Capture lands on a non-async thread
    (PortAudio's audio callback), so
    ``deliver_chunk`` hops onto the event loop before awaiting anything.
    Subclasses set ``self._loop`` in ``start()``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._loop: asyncio.AbstractEventLoop | None = None
        # Consumers are keyed by a stable integer handle returned from
        # add_consumer(). This avoids the bound-method identity pitfall:
        # ``obj.method is obj.method`` is False, so using the callable
        # itself as the key silently breaks remove_consumer().
        self._consumers: list[tuple[int, AudioConsumer, str]] = []
        self._next_consumer_id: int = 1

    def add_consumer(self, callback: AudioConsumer, name: str = "") -> int:
        """Register an async callback to receive audio chunks.

        Args:
            callback: Async callable that receives AudioChunk.
            name: Human-readable name for logging.

        Returns:
            A handle id. Pass this to ``remove_consumer`` to unregister.
            Callers MUST store this id — bound methods are not
            identity-stable across accesses, so the callable itself is
            not a reliable key.
        """
        handle = self._next_consumer_id
        self._next_consumer_id += 1
        display = name or repr(callback)
        self._consumers.append((handle, callback, display))
        logger.debug(
            "Audio consumer added: %s [id=%d] (total: %d)",
            display, handle, len(self._consumers),
        )
        return handle

    def remove_consumer(self, handle: int) -> bool:
        """Remove a previously registered consumer by handle.

        Args:
            handle: The integer handle returned from ``add_consumer``.

        Returns:
            True if a consumer was removed; False if the handle was
            unknown (caller logic bug — should never happen if handles
            are stored correctly).
        """
        for i, (h, _cb, name) in enumerate(self._consumers):
            if h == handle:
                self._consumers.pop(i)
                logger.debug(
                    "Audio consumer removed: %s [id=%d] (total: %d)",
                    name, handle, len(self._consumers),
                )
                return True
        logger.warning(
            "remove_consumer called with unknown handle %d — consumer "
            "list unchanged (total: %d)",
            handle, len(self._consumers),
        )
        return False

    @property
    def consumer_count(self) -> int:
        """Number of registered audio consumers."""
        return len(self._consumers)

    def deliver_chunk(self, chunk: AudioChunk) -> None:
        """Dispatch a chunk to the consumers from a capture thread."""
        if not self._consumers or self._loop is None:
            return
        self._loop.call_soon_threadsafe(
            self._loop.create_task, self._dispatch_chunk(chunk)
        )

    async def _dispatch_chunk(self, chunk: AudioChunk) -> None:
        """Distribute an audio chunk to all registered consumers.

        Each consumer is called concurrently. Slow or failing consumers
        do not block others.
        """
        if not self._consumers:
            return

        async def _safe_deliver(
            callback: AudioConsumer, name: str, chunk: AudioChunk
        ) -> None:
            try:
                await callback(chunk)
            except Exception:
                logger.exception("Error in audio consumer %s", name)

        # Snapshot the consumer list: a consumer that unregisters itself
        # during delivery must not mutate the iterable we're awaiting on.
        snapshot = list(self._consumers)
        await asyncio.gather(
            *(_safe_deliver(cb, name, chunk) for _h, cb, name in snapshot)
        )
