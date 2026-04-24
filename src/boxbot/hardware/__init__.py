"""Hardware Abstraction Layer (HAL).

All hardware access goes through this module — no other part of boxBot
imports picamera2, hailo_platform, or touches sysfs directly.

Exports base types and enums only.  Hardware module classes (Camera,
Hailo, System) are NOT imported here because they pull in hardware
libraries that may not be available on development machines.  Import
them directly where needed::

    from boxbot.hardware.camera import Camera
    from boxbot.hardware.hailo import Hailo
    from boxbot.hardware.system import System
"""

from boxbot.hardware.base import (
    AudioChunk,
    HardwareHealthChanged,
    HardwareModule,
    HardwareUnavailableError,
    HealthStatus,
    ModelInfo,
    ShutdownRequested,
    SystemHealth,
    ThermalWarning,
)

__all__ = [
    "AudioChunk",
    "HardwareHealthChanged",
    "HardwareModule",
    "HardwareUnavailableError",
    "HealthStatus",
    "ModelInfo",
    "ShutdownRequested",
    "SystemHealth",
    "ThermalWarning",
]
