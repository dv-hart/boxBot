"""NOAA weather client (api.weather.gov).

Free, no API key required. Two-step lookup: lat/lon → grid point → forecast.
Returns display-ready dicts with Lucide icon names mapped from NOAA's
``shortForecast`` strings.

Coverage: continental US, Alaska, Hawaii, US territories. International
locations get an empty result and a warning.

Usage:
    from boxbot.integrations import noaa_weather as wx

    data = await wx.fetch_weather(lat=47.6062, lon=-122.3321)
    # {"temp": "72", "condition": "Partly Cloudy", "icon": "cloud-sun",
    #  "forecast": [{"day": "Mon", "icon": "sun", "high": "75"}, ...]}
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)


_BASE = "https://api.weather.gov"
_USER_AGENT = "boxBot/0.1 (https://github.com/boxBot, contact@example.com)"
_HEADERS = {"User-Agent": _USER_AGENT, "Accept": "application/geo+json"}
_TIMEOUT = httpx.Timeout(15.0)

# Cache the lat/lon → grid point mapping; it never changes.
_GRID_CACHE: dict[tuple[float, float], dict[str, str]] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def fetch_weather(
    *, lat: float, lon: float, forecast_days: int = 5
) -> dict[str, Any]:
    """Fetch current conditions and a multi-day forecast for one location.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees (negative for west).
        forecast_days: Number of forecast entries to return (max 7).

    Returns:
        Display-ready dict with keys: temp, condition, icon, humidity,
        wind, forecast (list of {day, icon, high, low}).
    """
    try:
        grid = await _resolve_grid(lat, lon)
    except Exception as e:
        logger.warning("NOAA grid lookup failed for %s,%s: %s", lat, lon, e)
        return {}

    forecast_url = grid.get("forecast")
    if not forecast_url:
        return {}

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS) as c:
            resp = await c.get(forecast_url)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning("NOAA forecast fetch failed: %s", e)
        return {}

    periods = data.get("properties", {}).get("periods", [])
    if not periods:
        return {}

    # First period = "now" (could be today or tonight depending on time)
    current = periods[0]
    forecast = _build_forecast(periods[1:], forecast_days)

    return {
        "temp": str(current.get("temperature", "")),
        "condition": current.get("shortForecast", ""),
        "icon": _map_icon(current.get("shortForecast", ""), current),
        "humidity": _extract_humidity(current),
        "wind": _format_wind(
            current.get("windSpeed"), current.get("windDirection")
        ),
        "forecast": forecast,
        "updated_at": data.get("properties", {}).get("updated", ""),
    }


# ---------------------------------------------------------------------------
# Grid resolution
# ---------------------------------------------------------------------------


async def _resolve_grid(lat: float, lon: float) -> dict[str, str]:
    """Resolve lat/lon to NOAA grid endpoints (cached forever)."""
    key = (round(lat, 4), round(lon, 4))
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]

    url = f"{_BASE}/points/{lat:.4f},{lon:.4f}"
    async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS) as c:
        resp = await c.get(url)
        if resp.status_code == 404:
            raise ValueError(
                f"NOAA does not cover {lat},{lon} "
                "(US-only — try a different location)"
            )
        resp.raise_for_status()
        props = resp.json().get("properties", {})

    grid = {
        "forecast": props.get("forecast", ""),
        "forecast_hourly": props.get("forecastHourly", ""),
        "office": props.get("gridId", ""),
    }
    _GRID_CACHE[key] = grid
    return grid


# ---------------------------------------------------------------------------
# Forecast building
# ---------------------------------------------------------------------------


def _build_forecast(
    periods: list[dict[str, Any]], days: int
) -> list[dict[str, Any]]:
    """Collapse NOAA day/night periods into one entry per day.

    NOAA returns alternating "day" and "night" periods. We pair each
    daytime period (high) with the following nighttime period (low) and
    surface ``forecast_days`` entries.
    """
    result: list[dict[str, Any]] = []
    seen_dates: set[str] = set()

    i = 0
    while i < len(periods) and len(result) < days:
        period = periods[i]
        is_daytime = period.get("isDaytime", True)

        if is_daytime:
            start = _parse_iso(period.get("startTime"))
            date_key = start.strftime("%Y-%m-%d") if start else None
            if date_key and date_key in seen_dates:
                i += 1
                continue
            if date_key:
                seen_dates.add(date_key)

            high = period.get("temperature")
            short = period.get("shortForecast", "")
            day_label = (
                start.strftime("%a") if start else period.get("name", "")
            )
            entry = {
                "day": day_label,
                "icon": _map_icon(short, period),
                "high": str(high) if high is not None else "",
                "low": "",
                "condition": short,
            }
            # Look ahead for the matching nighttime period
            if i + 1 < len(periods):
                nxt = periods[i + 1]
                if not nxt.get("isDaytime", False):
                    entry["low"] = str(nxt.get("temperature", ""))
                    i += 2
                    result.append(entry)
                    continue
            i += 1
            result.append(entry)
        else:
            # Skip nighttime periods that aren't paired with a day
            i += 1

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_wind(speed: str | None, direction: str | None) -> str:
    if not speed:
        return ""
    if direction:
        return f"{speed} {direction}"
    return speed


def _extract_humidity(period: dict[str, Any]) -> str:
    """NOAA puts humidity in ``relativeHumidity.value`` for hourly only."""
    rh = period.get("relativeHumidity")
    if isinstance(rh, dict):
        val = rh.get("value")
        if isinstance(val, (int, float)):
            return str(int(val))
    return ""


# ---------------------------------------------------------------------------
# Icon mapping
# ---------------------------------------------------------------------------

# NOAA shortForecast strings → Lucide icon names. Patterns are case-insensitive
# substring matches checked in order; first hit wins.
_ICON_RULES: list[tuple[str, str]] = [
    # Storms first (most specific)
    ("thunderstorm", "cloud-lightning"),
    ("tornado", "tornado"),
    ("hurricane", "wind"),
    # Snow / wintry mix
    ("blizzard", "snowflake"),
    ("snow shower", "cloud-snow"),
    ("snow", "snowflake"),
    ("sleet", "cloud-snow"),
    ("ice", "snowflake"),
    ("wintry", "cloud-snow"),
    # Rain
    ("freezing rain", "cloud-snow"),
    ("rain shower", "cloud-rain"),
    ("light rain", "cloud-drizzle"),
    ("drizzle", "cloud-drizzle"),
    ("heavy rain", "cloud-rain"),
    ("rain", "cloud-rain"),
    ("showers", "cloud-rain"),
    # Fog / haze
    ("fog", "cloud-fog"),
    ("haze", "cloud-fog"),
    ("smoke", "cloud-fog"),
    ("mist", "cloud-fog"),
    # Clouds
    ("partly sunny", "cloud-sun"),
    ("partly cloudy", "cloud-sun"),
    ("mostly cloudy", "cloud"),
    ("cloudy", "cloud"),
    ("overcast", "cloud"),
    # Wind
    ("windy", "wind"),
    ("breezy", "wind"),
    # Clear
    ("mostly sunny", "sun"),
    ("mostly clear", "moon"),
    ("sunny", "sun"),
    ("clear", "moon"),
    ("fair", "sun"),
]


def _map_icon(short_forecast: str, period: dict[str, Any]) -> str:
    """Return the best Lucide icon name for a NOAA shortForecast string."""
    text = (short_forecast or "").lower()
    is_daytime = period.get("isDaytime", True)

    for keyword, icon in _ICON_RULES:
        if keyword in text:
            # Swap day/night variants where it matters
            if not is_daytime:
                if icon == "sun":
                    return "moon"
                if icon == "cloud-sun":
                    return "cloud-moon"
            return icon

    # Default fallback
    return "cloud" if "cloud" in text else "sun"
