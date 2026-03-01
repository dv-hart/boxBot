"""Data source management for boxBot displays.

Data sources fetch, cache, and normalize external data for display blocks.
The display manager schedules async fetches per source on each source's
refresh interval. Blocks bind to source fields using {source.field} syntax.

Built-in sources tap into boxBot subsystems (weather, calendar, tasks, etc.)
and deliver display-ready data including resolved icon names and semantic
color tokens.

Custom sources (http_json, http_text, static, memory_query) support field
extraction and value mapping via the 'fields' + 'map' transform system.

Usage:
    from boxbot.displays.data_sources import DataSourceManager

    mgr = DataSourceManager()
    mgr.register("weather", WeatherSource())
    await mgr.start_all()
    data = mgr.get_data("weather")
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class DataSource(ABC):
    """Base class for display data sources."""

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        self.name = name
        self.config = config or {}
        self._cached_data: dict[str, Any] = {}
        self._last_fetch: float = 0.0
        self._fetch_error: str | None = None

    @abstractmethod
    async def fetch(self) -> dict[str, Any]:
        """Fetch fresh data from the source.

        Returns:
            Dict of field names to values, ready for display binding.
        """
        ...

    @property
    def refresh_interval(self) -> int:
        """Seconds between fetches. Override or set via config."""
        return self.config.get("refresh", 60)

    @property
    def cached_data(self) -> dict[str, Any]:
        """Return the last successfully fetched data."""
        return self._cached_data

    @property
    def is_stale(self) -> bool:
        """True if last fetch failed or data is older than 2x refresh interval."""
        if self._fetch_error:
            return True
        if self._last_fetch == 0.0:
            return True
        return (time.monotonic() - self._last_fetch) > (self.refresh_interval * 2)

    async def do_fetch(self) -> dict[str, Any]:
        """Fetch data, apply field transforms, and cache the result."""
        try:
            raw = await self.fetch()
            transformed = _apply_field_transforms(raw, self.config.get("fields", {}))
            self._cached_data = transformed
            self._last_fetch = time.monotonic()
            self._fetch_error = None
            return transformed
        except Exception as e:
            self._fetch_error = str(e)
            logger.warning("Fetch failed for source '%s': %s", self.name, e)
            # Return stale data rather than empty
            return self._cached_data


# ---------------------------------------------------------------------------
# Built-in sources (stubs — real implementations connect to subsystems)
# ---------------------------------------------------------------------------


class ClockSource(DataSource):
    """Live clock data. The renderer ticks this directly rather than fetching."""

    def __init__(self) -> None:
        super().__init__("clock")

    @property
    def refresh_interval(self) -> int:
        return 1  # Ticked by renderer

    async def fetch(self) -> dict[str, Any]:
        from datetime import datetime
        now = datetime.now()
        hour_12 = now.hour % 12 or 12
        return {
            "hour": now.hour,
            "minute": now.minute,
            "second": now.second,
            "display": f"{hour_12}:{now.minute:02d}",
            "date": now.strftime("%B %d, %Y"),
            "day_of_week": now.strftime("%A"),
        }


class WeatherSource(DataSource):
    """Weather data from the weather skill/API.

    Delivers display-ready data with Lucide icon names.
    Stub implementation returns placeholder data.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__("weather", config)

    @property
    def refresh_interval(self) -> int:
        return self.config.get("refresh", 3600)

    async def fetch(self) -> dict[str, Any]:
        # Stub: return placeholder data until weather skill is connected
        return _weather_placeholder()


class CalendarSource(DataSource):
    """Calendar events from the calendar skill.

    Stub implementation returns placeholder data.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__("calendar", config)

    @property
    def refresh_interval(self) -> int:
        return self.config.get("refresh", 300)

    async def fetch(self) -> dict[str, Any]:
        return _calendar_placeholder()


class TasksSource(DataSource):
    """To-do items from the scheduler.

    Stub implementation returns placeholder data.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__("tasks", config)

    @property
    def refresh_interval(self) -> int:
        return self.config.get("refresh", 60)

    async def fetch(self) -> dict[str, Any]:
        return _tasks_placeholder()


class PeopleSource(DataSource):
    """Currently detected people from the perception pipeline.

    Event-driven — refreshes on person detection events.
    Stub implementation returns empty.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__("people", config)

    @property
    def refresh_interval(self) -> int:
        return 5  # Low interval; real impl is event-driven

    async def fetch(self) -> dict[str, Any]:
        return {"present": [], "count": 0}


class AgentStatusSource(DataSource):
    """Agent state (sleeping, listening, thinking, working).

    Event-driven.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__("agent_status", config)

    @property
    def refresh_interval(self) -> int:
        return 5

    async def fetch(self) -> dict[str, Any]:
        return {
            "state": "sleeping",
            "last_active": None,
            "next_wake": None,
        }


# ---------------------------------------------------------------------------
# Custom sources
# ---------------------------------------------------------------------------


class HttpJsonSource(DataSource):
    """Fetch JSON from a URL with optional API key from secrets."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        super().__init__(name, config)

    async def fetch(self) -> dict[str, Any]:
        url = self.config.get("url", "")
        if not url:
            return {}

        try:
            import aiohttp
            headers: dict[str, str] = {}
            secret = self.config.get("secret")
            if secret:
                # In production, resolve secret from boxbot_sdk.secrets
                headers["Authorization"] = f"Bearer {secret}"

            params = self.config.get("params", {})
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers,
                                       timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    logger.warning("HTTP %d from %s", resp.status, url)
                    return {}
        except ImportError:
            logger.warning("aiohttp not available; HttpJsonSource cannot fetch")
            return {}
        except Exception as e:
            logger.warning("HTTP fetch failed for '%s': %s", self.name, e)
            return {}


class HttpTextSource(DataSource):
    """Fetch text/HTML from a URL."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        super().__init__(name, config)

    async def fetch(self) -> dict[str, Any]:
        url = self.config.get("url", "")
        if not url:
            return {}

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        return {"text": text}
                    return {}
        except ImportError:
            logger.warning("aiohttp not available; HttpTextSource cannot fetch")
            return {}
        except Exception as e:
            logger.warning("HTTP text fetch failed for '%s': %s", self.name, e)
            return {}


class StaticSource(DataSource):
    """Hardcoded values the agent can update later."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        super().__init__(name, config)
        self._cached_data = config.get("value", {})
        if not isinstance(self._cached_data, dict):
            self._cached_data = {"value": self._cached_data}

    @property
    def refresh_interval(self) -> int:
        return 999999  # Never auto-refreshes

    async def fetch(self) -> dict[str, Any]:
        return self._cached_data

    def update(self, value: dict[str, Any]) -> None:
        """Update the static data without rebuilding the display."""
        if isinstance(value, dict):
            self._cached_data = value
        else:
            self._cached_data = {"value": value}


class MemoryQuerySource(DataSource):
    """Run a memory search query and return results."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        super().__init__(name, config)

    @property
    def refresh_interval(self) -> int:
        return self.config.get("refresh", 300)

    async def fetch(self) -> dict[str, Any]:
        query = self.config.get("query", "")
        if not query:
            return {"results": []}
        # Stub: real implementation calls memory search backend
        return {"results": [], "query": query}


# ---------------------------------------------------------------------------
# Source registry and factory
# ---------------------------------------------------------------------------

_BUILTIN_SOURCES = {
    "clock", "weather", "calendar", "tasks", "people", "agent_status",
}

_CUSTOM_SOURCE_TYPES: dict[str, type[DataSource]] = {
    "http_json": HttpJsonSource,
    "http_text": HttpTextSource,
    "static": StaticSource,
    "memory_query": MemoryQuerySource,
}


def create_source(name: str, source_type: str = "builtin",
                  config: dict[str, Any] | None = None) -> DataSource:
    """Create a data source by name and type.

    Args:
        name: Source name (e.g. 'weather', 'stocks').
        source_type: 'builtin' or one of the custom types.
        config: Source-specific configuration.

    Returns:
        A DataSource instance.
    """
    config = config or {}

    if source_type == "builtin" or name in _BUILTIN_SOURCES:
        return _create_builtin_source(name, config)

    cls = _CUSTOM_SOURCE_TYPES.get(source_type)
    if cls is None:
        raise ValueError(
            f"Unknown data source type '{source_type}'. "
            f"Available: builtin, {', '.join(_CUSTOM_SOURCE_TYPES.keys())}"
        )
    return cls(name, config)


def _create_builtin_source(name: str, config: dict[str, Any]) -> DataSource:
    """Create a built-in data source by name."""
    sources: dict[str, type[DataSource]] = {
        "clock": ClockSource,
        "weather": WeatherSource,
        "calendar": CalendarSource,
        "tasks": TasksSource,
        "people": PeopleSource,
        "agent_status": AgentStatusSource,
    }
    cls = sources.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown built-in source '{name}'. "
            f"Available: {', '.join(sources.keys())}"
        )
    if cls == ClockSource:
        return cls()
    return cls(config)


# ---------------------------------------------------------------------------
# Data Source Manager
# ---------------------------------------------------------------------------


class DataSourceManager:
    """Manages data source lifecycle for the active display.

    Registers sources, starts/stops async fetch loops, and provides
    unified data access for the renderer.
    """

    def __init__(self) -> None:
        self._sources: dict[str, DataSource] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._running = False

    def register(self, source: DataSource) -> None:
        """Register a data source."""
        self._sources[source.name] = source

    def unregister(self, name: str) -> None:
        """Remove a data source and stop its fetch loop if running."""
        self._stop_source(name)
        self._sources.pop(name, None)

    def get_source(self, name: str) -> DataSource | None:
        """Get a registered source by name."""
        return self._sources.get(name)

    def get_data(self, name: str) -> dict[str, Any]:
        """Get the cached data for a source."""
        source = self._sources.get(name)
        if source is None:
            return {}
        return source.cached_data

    def get_all_data(self) -> dict[str, Any]:
        """Get all cached data as a dict mapping source names to their data."""
        return {name: src.cached_data for name, src in self._sources.items()}

    async def start_all(self) -> None:
        """Start async fetch loops for all registered sources."""
        self._running = True
        # Do an initial fetch for all sources
        fetch_tasks = [src.do_fetch() for src in self._sources.values()]
        if fetch_tasks:
            await asyncio.gather(*fetch_tasks, return_exceptions=True)

        # Start periodic fetch loops
        for name, source in self._sources.items():
            if name not in self._tasks:
                self._tasks[name] = asyncio.create_task(
                    self._fetch_loop(source),
                    name=f"datasource-{name}",
                )

    async def stop_all(self) -> None:
        """Stop all fetch loops."""
        self._running = False
        for name in list(self._tasks.keys()):
            self._stop_source(name)
        # Give tasks time to cancel
        if self._tasks:
            await asyncio.sleep(0.1)

    def _stop_source(self, name: str) -> None:
        """Stop a single source's fetch loop."""
        task = self._tasks.pop(name, None)
        if task and not task.done():
            task.cancel()

    async def _fetch_loop(self, source: DataSource) -> None:
        """Periodically fetch data for a source."""
        try:
            while self._running:
                await asyncio.sleep(source.refresh_interval)
                if not self._running:
                    break
                await source.do_fetch()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Fetch loop error for source '%s'", source.name)

    def clear(self) -> None:
        """Remove all sources. For testing."""
        for name in list(self._tasks.keys()):
            self._stop_source(name)
        self._sources.clear()
        self._tasks.clear()


# ---------------------------------------------------------------------------
# Field transforms
# ---------------------------------------------------------------------------


def _apply_field_transforms(data: dict[str, Any],
                            fields: dict[str, Any]) -> dict[str, Any]:
    """Apply field extraction and mapping transforms to raw data.

    The 'fields' config maps output field names to either:
    - A string: simple field rename (extract from raw data)
    - A dict with 'from' and 'map': value mapping transform

    Args:
        data: Raw fetched data.
        fields: Field extraction/mapping config.

    Returns:
        Transformed data dict.
    """
    if not fields:
        return data

    result = dict(data)  # Start with all raw fields

    for output_name, spec in fields.items():
        if isinstance(spec, str):
            # Simple rename: output_name = data[spec]
            result[output_name] = data.get(spec)
        elif isinstance(spec, dict):
            # Map transform
            from_field = spec.get("from", "")
            mapping = spec.get("map", {})
            raw_value = data.get(from_field)
            if raw_value is not None and mapping:
                result[output_name] = mapping.get(str(raw_value), raw_value)
            else:
                result[output_name] = raw_value

    return result


# ---------------------------------------------------------------------------
# Placeholder data for preview
# ---------------------------------------------------------------------------


def get_placeholder_data(source_name: str) -> dict[str, Any]:
    """Get plausible sample data for preview rendering.

    Used when data sources aren't live yet (e.g. during agent preview workflow).

    Args:
        source_name: The source name to generate placeholders for.

    Returns:
        Dict of sample data.
    """
    placeholders: dict[str, dict[str, Any]] = {
        "weather": _weather_placeholder(),
        "calendar": _calendar_placeholder(),
        "tasks": _tasks_placeholder(),
        "clock": _clock_placeholder(),
        "people": {"present": [{"name": "Jacob", "since": "2 min ago"}], "count": 1},
        "agent_status": {"state": "listening", "last_active": "just now", "next_wake": "7:00 AM"},
    }
    return placeholders.get(source_name, {})


def _weather_placeholder() -> dict[str, Any]:
    return {
        "temp": "72",
        "condition": "Partly Cloudy",
        "icon": "cloud-sun",
        "humidity": "65",
        "wind": "12 mph",
        "forecast": [
            {"day": "Mon", "icon": "sun", "high": "75", "low": "58"},
            {"day": "Tue", "icon": "cloud", "high": "68", "low": "55"},
            {"day": "Wed", "icon": "cloud-rain", "high": "62", "low": "50"},
            {"day": "Thu", "icon": "sun", "high": "70", "low": "54"},
            {"day": "Fri", "icon": "cloud-sun", "high": "73", "low": "57"},
        ],
    }


def _calendar_placeholder() -> dict[str, Any]:
    return {
        "events": [
            {"time": "9:00 AM", "title": "Team Standup", "duration": "30m", "location": "Zoom"},
            {"time": "11:00 AM", "title": "Design Review", "duration": "1h", "location": "Conference Room"},
            {"time": "2:00 PM", "title": "Dentist Appointment", "duration": "1h", "location": "Downtown"},
        ],
    }


def _tasks_placeholder() -> dict[str, Any]:
    return {
        "items": [
            {"description": "Buy groceries", "due_date": "Today", "status": "pending"},
            {"description": "Review budget spreadsheet", "due_date": "Tomorrow", "status": "pending"},
            {"description": "Call plumber about leak", "due_date": None, "status": "pending"},
        ],
        "count": 3,
    }


def _clock_placeholder() -> dict[str, Any]:
    from datetime import datetime
    now = datetime.now()
    hour_12 = now.hour % 12 or 12
    return {
        "hour": now.hour,
        "minute": now.minute,
        "second": now.second,
        "display": f"{hour_12}:{now.minute:02d}",
        "date": now.strftime("%B %d, %Y"),
        "day_of_week": now.strftime("%A"),
    }
