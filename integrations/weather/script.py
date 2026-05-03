"""NOAA weather integration — sandbox-runnable.

Fetches the forecast from api.weather.gov via two HTTP requests
(grid resolution + forecast lookup) and returns a display-ready dict
with mapped Lucide icon names.

Free, no API key required. Coverage: continental US, Alaska, Hawaii,
US territories. International locations get an empty result and a
warning written to stderr.

Migrated from ``src/boxbot/integrations/noaa_weather.py``. The grid-
endpoint cache that lived in module memory is dropped here — every
sandbox invocation does its own grid lookup. NWS allows it and the
extra request is cheap; consumers cache at their own cadence.
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Any

import httpx

from boxbot_sdk.integration import inputs as get_inputs, return_output


_BASE = "https://api.weather.gov"
_USER_AGENT = "boxBot/0.1 (https://github.com/boxBot, contact@example.com)"
_HEADERS = {"User-Agent": _USER_AGENT, "Accept": "application/geo+json"}
_TIMEOUT = httpx.Timeout(15.0)


# ---------------------------------------------------------------------------
# Icon mapping — NOAA shortForecast → Lucide icon name. First substring hit
# wins. Day/night swaps for the few icons where it matters.
# ---------------------------------------------------------------------------

_ICON_RULES: list[tuple[str, str]] = [
    ("thunderstorm", "cloud-lightning"),
    ("tornado", "tornado"),
    ("hurricane", "wind"),
    ("blizzard", "snowflake"),
    ("snow shower", "cloud-snow"),
    ("snow", "snowflake"),
    ("sleet", "cloud-snow"),
    ("ice", "snowflake"),
    ("wintry", "cloud-snow"),
    ("freezing rain", "cloud-snow"),
    ("rain shower", "cloud-rain"),
    ("light rain", "cloud-drizzle"),
    ("drizzle", "cloud-drizzle"),
    ("heavy rain", "cloud-rain"),
    ("rain", "cloud-rain"),
    ("showers", "cloud-rain"),
    ("fog", "cloud-fog"),
    ("haze", "cloud-fog"),
    ("smoke", "cloud-fog"),
    ("mist", "cloud-fog"),
    ("partly sunny", "cloud-sun"),
    ("partly cloudy", "cloud-sun"),
    ("mostly cloudy", "cloud"),
    ("cloudy", "cloud"),
    ("overcast", "cloud"),
    ("windy", "wind"),
    ("breezy", "wind"),
    ("mostly sunny", "sun"),
    ("mostly clear", "moon"),
    ("sunny", "sun"),
    ("clear", "moon"),
    ("fair", "sun"),
]


def _map_icon(short_forecast: str, period: dict[str, Any]) -> str:
    text = (short_forecast or "").lower()
    is_daytime = period.get("isDaytime", True)
    for keyword, icon in _ICON_RULES:
        if keyword in text:
            if not is_daytime:
                if icon == "sun":
                    return "moon"
                if icon == "cloud-sun":
                    return "cloud-moon"
            return icon
    return "cloud" if "cloud" in text else "sun"


# ---------------------------------------------------------------------------
# Fetch + format
# ---------------------------------------------------------------------------


def _resolve_grid(client: httpx.Client, lat: float, lon: float) -> dict[str, str]:
    url = f"{_BASE}/points/{lat:.4f},{lon:.4f}"
    resp = client.get(url)
    if resp.status_code == 404:
        raise ValueError(
            f"NOAA does not cover {lat},{lon} "
            "(US-only — try a different location)"
        )
    resp.raise_for_status()
    props = resp.json().get("properties", {})
    return {
        "forecast": props.get("forecast", ""),
        "forecast_hourly": props.get("forecastHourly", ""),
        "office": props.get("gridId", ""),
    }


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
    rh = period.get("relativeHumidity")
    if isinstance(rh, dict):
        val = rh.get("value")
        if isinstance(val, (int, float)):
            return str(int(val))
    return ""


def _build_forecast(periods: list[dict[str, Any]], days: int) -> list[dict[str, Any]]:
    """Collapse NOAA's day/night periods into one entry per day."""
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
            day_label = start.strftime("%a") if start else period.get("name", "")
            entry = {
                "day": day_label,
                "icon": _map_icon(short, period),
                "high": str(high) if high is not None else "",
                "low": "",
                "condition": short,
            }
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
            i += 1
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = get_inputs()
    lat = float(args["lat"])
    lon = float(args["lon"])
    days = int(args.get("forecast_days", 5))

    try:
        with httpx.Client(timeout=_TIMEOUT, headers=_HEADERS) as client:
            grid = _resolve_grid(client, lat, lon)
            forecast_url = grid.get("forecast")
            if not forecast_url:
                return_output({})
                return
            resp = client.get(forecast_url)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"NOAA fetch failed for {lat},{lon}: {e}\n")
        return_output({})
        return

    periods = data.get("properties", {}).get("periods", [])
    if not periods:
        return_output({})
        return

    current = periods[0]
    forecast = _build_forecast(periods[1:], days)
    return_output(
        {
            "temp": str(current.get("temperature", "")),
            "condition": current.get("shortForecast", ""),
            "icon": _map_icon(current.get("shortForecast", ""), current),
            "humidity": _extract_humidity(current),
            "wind": _format_wind(current.get("windSpeed"), current.get("windDirection")),
            "forecast": forecast,
            "updated_at": data.get("properties", {}).get("updated", ""),
        }
    )


if __name__ == "__main__":
    main()
