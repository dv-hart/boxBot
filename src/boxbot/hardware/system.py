"""System-level monitoring: thermal, memory/disk, and health aggregation.

Cross-cutting hardware concerns that don't belong to any single device
module. Monitors SoC temperature via sysfs and aggregates per-module
health. Process signals (SIGTERM/SIGINT) are owned by
:mod:`boxbot.core.main`.

Hardware: Pi 5 SoC, PMIC, thermals
Interface: sysfs
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from typing import Any

from boxbot.hardware.base import (
    HardwareModule,
    HealthStatus,
    SystemHealth,
    ThermalWarning,
)

logger = logging.getLogger(__name__)


class System(HardwareModule):
    """Thermal monitoring and health aggregation.

    Always available — does not depend on external hardware beyond the
    Pi 5 SoC itself.
    """

    name = "system"

    _SOC_TEMP_PATH = "/sys/class/thermal/thermal_zone0/temp"

    def __init__(
        self,
        warning_temp: float = 70.0,
        throttle_temp: float = 80.0,
        critical_temp: float = 85.0,
        poll_interval: float = 30.0,
    ) -> None:
        super().__init__()
        self._warning_temp = warning_temp
        self._throttle_temp = throttle_temp
        self._critical_temp = critical_temp
        self._poll_interval = poll_interval

        self._monitor_task: asyncio.Task[None] | None = None
        self._hal_modules: dict[str, HardwareModule] = {}
        self._hailo_ref: Any = None  # Optional reference to Hailo module for temperature

        # Track thermal state to emit events only on threshold crossings
        self._last_thermal_level: str = "normal"

    # ── Lifecycle ──────────────────────────────────────────────────

    async def start(self) -> None:
        """Start thermal monitoring.

        Process-level signal handling lives in :mod:`boxbot.core.main` —
        this module used to register its own SIGTERM/SIGINT handlers,
        but that silently overwrote main's handler and the shutdown
        event was never set, leaving the process running after a
        SIGTERM. Signals are owned by main now.
        """
        # Start thermal monitoring background task
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self._started = True
        await self._emit_health(HealthStatus.OK)
        logger.info("System monitor started (poll=%.0fs)", self._poll_interval)

    async def stop(self) -> None:
        """Stop thermal monitoring."""
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        self._started = False
        await self._emit_health(HealthStatus.STOPPED)

    # ── Module registration ────────────────────────────────────────

    def register_module(self, module: HardwareModule) -> None:
        """Register a HAL module for health aggregation.

        Called during startup to track per-module health in
        ``get_system_health()``.
        """
        self._hal_modules[module.name] = module

    def set_hailo_ref(self, hailo: Any) -> None:
        """Store a reference to the Hailo module for temperature reads."""
        self._hailo_ref = hailo

    # ── Temperature ────────────────────────────────────────────────

    @property
    def soc_temperature(self) -> float | None:
        """SoC temperature in Celsius, read from sysfs.

        Returns None if the sysfs file is not available (e.g. not on
        a Raspberry Pi).
        """
        try:
            with open(self._SOC_TEMP_PATH) as f:
                # sysfs reports in millidegrees
                return int(f.read().strip()) / 1000.0
        except (FileNotFoundError, ValueError, OSError):
            return None

    @property
    def hailo_temperature(self) -> float | None:
        """Hailo die temperature in Celsius, or None if unavailable."""
        if self._hailo_ref is not None:
            try:
                return self._hailo_ref.temperature
            except Exception:
                pass
        return None

    # ── Health aggregation ─────────────────────────────────────────

    def get_system_health(self) -> SystemHealth:
        """Aggregate system health snapshot.

        Includes SoC and Hailo temperatures, memory and disk usage,
        and per-module health status.
        """
        # Memory usage from /proc/meminfo (avoids psutil dependency)
        memory_pct = self._read_memory_percent()

        # Disk usage for the data partition
        disk = shutil.disk_usage("/")
        disk_pct = (disk.used / disk.total) * 100.0

        # Per-module health (synchronous snapshot — uses last known status)
        module_health: dict[str, HealthStatus] = {}
        for name, module in self._hal_modules.items():
            module_health[name] = module._last_health

        return SystemHealth(
            soc_temp=self.soc_temperature,
            hailo_temp=self.hailo_temperature,
            memory_used_pct=memory_pct,
            disk_used_pct=disk_pct,
            module_health=module_health,
        )

    @staticmethod
    def _read_memory_percent() -> float:
        """Read memory usage percentage from /proc/meminfo."""
        try:
            info: dict[str, int] = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2 and parts[0].endswith(":"):
                        key = parts[0][:-1]
                        info[key] = int(parts[1])
            total = info.get("MemTotal", 0)
            available = info.get("MemAvailable", 0)
            if total > 0:
                return ((total - available) / total) * 100.0
        except (FileNotFoundError, ValueError, OSError):
            pass
        return 0.0

    # ── Thermal monitoring ─────────────────────────────────────────

    async def _monitor_loop(self) -> None:
        """Background task: poll temperatures and emit events on crossings."""
        while True:
            try:
                await self._check_thermals()
            except Exception:
                logger.exception("Error in thermal monitoring loop")
            await asyncio.sleep(self._poll_interval)

    async def _check_thermals(self) -> None:
        """Check SoC temperature and emit events on threshold crossings."""
        temp = self.soc_temperature
        if temp is None:
            return

        # Determine current thermal level
        if temp >= self._critical_temp:
            level = "critical"
        elif temp >= self._throttle_temp:
            level = "throttle"
        elif temp >= self._warning_temp:
            level = "warning"
        else:
            level = "normal"

        # Only emit events on level transitions
        if level != self._last_thermal_level:
            old_level = self._last_thermal_level
            self._last_thermal_level = level

            if level == "normal":
                logger.info(
                    "SoC temperature back to normal: %.1f°C (was %s)",
                    temp,
                    old_level,
                )
            else:
                log_fn = logger.critical if level == "critical" else logger.warning
                log_fn(
                    "SoC thermal %s: %.1f°C (threshold=%.0f°C)",
                    level,
                    temp,
                    getattr(self, f"_{level}_temp"),
                )
                bus = get_event_bus()
                await bus.publish(
                    ThermalWarning(
                        source="soc",
                        temperature=temp,
                        threshold=level,
                    )
                )

    # ── Properties ─────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """Always available on a Raspberry Pi."""
        return True

    async def health_check(self) -> HealthStatus:
        """Check system health based on thermal state."""
        if not self._started:
            return HealthStatus.STOPPED
        if self._last_thermal_level == "critical":
            return HealthStatus.ERROR
        if self._last_thermal_level in ("warning", "throttle"):
            return HealthStatus.DEGRADED
        return HealthStatus.OK
