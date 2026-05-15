"""Data source management for boxBot displays.

Data sources fetch, cache, and normalize external data for display blocks.
The display manager schedules async fetches per source on each source's
refresh interval. Blocks bind to source fields using {source.field} syntax.

Built-in sources tap into in-process subsystems that aren't integrations:
the clock, the scheduler's to-do list, the perception pipeline's present
people, and the agent state tracker.

External data — weather, calendar, anything an agent authors — flows
through ``IntegrationSource``: a generic wrapper that calls
:func:`boxbot.integrations.runner.run` on each refresh and binds the
integration's ``output`` to the source name. Pre-seeded integrations
(weather, calendar) and agent-authored ones share this single path.

The other custom sources (``http_json``, ``http_text``, ``static``,
``memory_query``) cover narrower cases: agent-pushed values via
``static`` + ``update_data``, and quick HTTP/memory probes that don't
warrant a full integration.

Usage:
    from boxbot.displays.data_sources import DataSourceManager, create_source

    mgr = DataSourceManager()
    mgr.register(create_source("weather", "integration",
                               {"inputs": {"lat": 45.5, "lon": -122.7}}))
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


class IntegrationSource(DataSource):
    """Generic source backed by a registered integration.

    The display's data-source manager calls
    :func:`boxbot.integrations.runner.run` on this source's refresh
    interval and binds the integration's ``output`` dict to the source
    name. Inputs are passed through verbatim — the integration manifest
    declares its own defaults (including ``default_env`` for
    device-level config like lat/lon).

    Spec form::

        {"name": "weather", "type": "integration",
         "inputs": {"forecast_days": 7}, "refresh": 3600}

        {"name": "solar", "type": "integration",
         "integration": "solar",   # optional; defaults to ``name``
         "inputs": {"date": "2026-05-15"}, "refresh": 3600}

    On error (unknown integration, non-``ok`` status, runner exception)
    the source returns an empty dict so the renderer falls through to
    its placeholder paths. The error is logged once per fetch.
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        super().__init__(name, config)
        self._integration_name: str = (
            (config or {}).get("integration") or name
        )
        self._inputs: dict[str, Any] = dict((config or {}).get("inputs") or {})

    @property
    def refresh_interval(self) -> int:
        return self.config.get("refresh", 300)

    async def fetch(self) -> dict[str, Any]:
        from boxbot.integrations.runner import IntegrationRunError, run

        try:
            result = await run(self._integration_name, self._inputs)
        except IntegrationRunError as e:
            logger.warning(
                "Integration source '%s' (-> '%s') not registered: %s",
                self.name, self._integration_name, e,
            )
            return {}
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Integration source '%s' (-> '%s') call failed: %s",
                self.name, self._integration_name, e,
            )
            return {}

        if result.get("status") != "ok":
            logger.debug(
                "Integration source '%s' (-> '%s') returned %s: %s",
                self.name, self._integration_name,
                result.get("status"),
                result.get("error", "no error message"),
            )
            return {}

        output = result.get("output") or {}
        if isinstance(output, dict) and output.get("error"):
            logger.debug(
                "Integration source '%s' (-> '%s') script reported error: %s",
                self.name, self._integration_name, output["error"],
            )
            return {}
        return output if isinstance(output, dict) else {"value": output}


class TasksSource(DataSource):
    """To-do items from the scheduler.

    Reads pending to-dos from the scheduler DB. Falls back to empty list
    if the scheduler is unavailable.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__("tasks", config)

    @property
    def refresh_interval(self) -> int:
        return self.config.get("refresh", 60)

    async def fetch(self) -> dict[str, Any]:
        try:
            from boxbot.core.scheduler import list_todos
            todos = await list_todos(status="pending")
        except Exception as e:
            logger.warning("Failed to fetch todos: %s", e)
            return {"items": [], "count": 0}

        items = [
            {
                "id": t.get("id"),
                "description": t.get("description", ""),
                "due_date": t.get("due_date"),
                "for_person": t.get("for_person"),
                "status": t.get("status", "pending"),
            }
            for t in todos
        ]
        return {"items": items, "count": len(items)}


class PeopleSource(DataSource):
    """Currently detected people from the perception pipeline.

    Polls the pipeline's state machine for present people. Falls back
    to empty data when the perception pipeline is not running.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__("people", config)

    @property
    def refresh_interval(self) -> int:
        return 5  # Low interval; supplements event-driven updates

    async def fetch(self) -> dict[str, Any]:
        try:
            from boxbot.perception.pipeline import get_pipeline

            pipeline = get_pipeline()
            present = pipeline.get_present_people()
            return {"present": present, "count": len(present)}
        except RuntimeError:
            # Pipeline not running
            return {"present": [], "count": 0}


class AgentStatusSource(DataSource):
    """Agent state (sleeping, listening, thinking, speaking).

    Reads from the AgentStateTracker singleton, which subscribes to
    lifecycle events on the event bus. Falls back to defaults if the
    tracker has not been started.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__("agent_status", config)

    @property
    def refresh_interval(self) -> int:
        return 5

    async def fetch(self) -> dict[str, Any]:
        try:
            from boxbot.core.agent_state import get_agent_state_tracker
            return await get_agent_state_tracker().snapshot()
        except Exception as e:
            logger.warning("Failed to fetch agent state: %s", e)
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
            secret_name = self.config.get("secret")
            if secret_name:
                # The spec stores a *name* (e.g. "STOCKS_API_KEY"); the
                # SecretStore is the only place that resolves it to a
                # value. Display data sources run in the main process,
                # so we read directly — no sandbox hop.
                from boxbot.secrets import get_secret_store

                value = get_secret_store().load(secret_name)
                if value is None:
                    logger.warning(
                        "data source '%s' references secret '%s' but it "
                        "is not stored — request will go out without auth",
                        self.name, secret_name,
                    )
                else:
                    headers["Authorization"] = f"Bearer {value}"

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

# Built-ins are the four sources that read live in-process state and
# can't be expressed as integrations: the clock, the scheduler's todo
# list, the perception pipeline's present-people view, and the agent's
# state machine. Everything else — weather, calendar, anything the
# agent authors — flows through ``IntegrationSource``.
_BUILTIN_SOURCES = {"clock", "tasks", "people", "agent_status"}

_CUSTOM_SOURCE_TYPES: dict[str, type[DataSource]] = {
    "integration": IntegrationSource,
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
        source_type: 'builtin', 'integration', or one of the other
            custom types.
        config: Source-specific configuration.

    Returns:
        A DataSource instance.

    Backward compat: a bare ``{"name": "weather"}`` (no ``type``) or
    ``{"name": "weather", "type": "builtin"}`` is silently promoted to
    an ``IntegrationSource`` pointing at the integration of the same
    name. Drops the bifurcation between pre-seeded and agent-authored
    integrations without breaking existing display specs in the wild.
    """
    config = config or {}

    if source_type == "builtin" or (source_type == "" and name in _BUILTIN_SOURCES):
        if name in _BUILTIN_SOURCES:
            return _create_builtin_source(name, config)
        # Unknown built-in name → assume it's a registered integration.
        # This keeps `{"name": "weather", "type": "builtin"}` working
        # after the weather/calendar collapse.
        return IntegrationSource(name, config)

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
    - A string: dotted path into raw data (e.g. "main.temp"). Single
      keys still work as before; nested API responses no longer require
      separate post-processing.
    - A dict with 'from' (dotted path) and 'map' (value→value): value
      mapping transform.

    Args:
        data: Raw fetched data.
        fields: Field extraction/mapping config.

    Returns:
        Transformed data dict — original keys plus the extracted ones.
    """
    if not fields:
        return data

    result = dict(data) if isinstance(data, dict) else {}

    for output_name, spec in fields.items():
        if isinstance(spec, str):
            result[output_name] = _dotted_get(data, spec)
        elif isinstance(spec, dict):
            from_field = spec.get("from", "")
            mapping = spec.get("map", {})
            raw_value = _dotted_get(data, from_field)
            if raw_value is not None and mapping:
                result[output_name] = mapping.get(str(raw_value), raw_value)
            else:
                result[output_name] = raw_value

    return result


def _dotted_get(data: Any, path: str) -> Any:
    """Walk a dotted path into nested dicts/lists.

    Examples: ``"temp"``, ``"main.temp"``, ``"items.0.name"``. Returns
    ``None`` if any segment is missing — never raises, since fetch-time
    transforms must not crash a display refresh on malformed payloads.
    """
    if not path:
        return data
    current: Any = data
    for segment in path.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(segment)
        elif isinstance(current, list):
            try:
                current = current[int(segment)]
            except (ValueError, IndexError):
                return None
        else:
            current = getattr(current, segment, None)
    return current


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
