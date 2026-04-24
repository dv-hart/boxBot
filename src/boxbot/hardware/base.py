"""HAL base classes and shared types.

All hardware modules implement the HardwareModule ABC. Shared dataclasses
(ModelInfo, SystemHealth) and enums (HealthStatus) live here so they can
be imported without pulling in hardware-specific libraries.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
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
    """Raised when required hardware is not found or not responsive."""


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
