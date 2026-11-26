"""Atmosphere — what the Living Sky scene should look like right now.

The HTML renderer draws one generative scene behind every display. The
scene is a pure function of a few named values, computed here so the
agent can read them: it knows the sky is "golden, partly cloudy" the
same way it knows the temperature, without looking at pixels.

    phase    dawn | noon | afternoon | golden | dusk | night   (from the sun)
    weather  clear | partly | fog | overcast | rain | snow     (from the forecast text)

Sunrise and sunset come from the standard NOAA solar approximation
(good to a minute or two, far inside what a painted sky needs), so the
phase is right in a Seattle November (sunset 4:24 PM) without a network
call. With no location configured, fixed clock hours stand in.
"""

from __future__ import annotations

import math
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any

PHASES = ("dawn", "noon", "afternoon", "golden", "dusk", "night")
WEATHERS = ("clear", "partly", "fog", "overcast", "rain", "snow")

# Shown when there is no forecast to go on. Says the least: some cloud,
# the mountain neither promised nor denied.
UNKNOWN_WEATHER = "partly"

# Checked in order: the first group with a matching word wins, so
# "Rain And Snow" is snow and "Patchy Fog then Sunny" is fog.
_WEATHER_WORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("snow", ("snow", "sleet", "flurr", "blizzard", "ice", "freezing", "wintry")),
    ("rain", ("rain", "shower", "drizzle", "thunder", "storm")),
    ("fog", ("fog", "mist", "haze", "smoke")),
    ("partly", ("partly", "mostly sunny", "mostly clear", "sunbreak", "few clouds")),
    ("overcast", ("overcast", "cloudy", "clouds")),
    ("clear", ("sunny", "clear", "fair")),
)


def weather_class(condition: str | None) -> str | None:
    """Map forecast text ("Chance Rain Showers") to a scene weather.

    Returns ``None`` when the text says nothing recognisable.
    """
    text = (condition or "").lower()
    if not text.strip():
        return None
    # "Slight Chance Rain" is a grey day, not a wet one.
    slight = "slight chance" in text
    for name, words in _WEATHER_WORDS:
        if any(w in text for w in words):
            if slight and name in ("rain", "snow"):
                return "overcast"
            return name
    return None


def sun_times(day: date, lat: float, lon: float) -> tuple[datetime, datetime] | None:
    """Sunrise and sunset on ``day`` as timezone-aware UTC datetimes.

    ``None`` during polar day or night, when the sun does neither.
    """
    n = day.timetuple().tm_yday
    gamma = 2 * math.pi / 365 * (n - 1)
    eqtime = 229.18 * (
        0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma)
    )
    decl = (
        0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma)
    )
    lat_r = math.radians(lat)
    # 90.833 deg: the sun's upper limb on the horizon, with refraction.
    cos_ha = (
        math.cos(math.radians(90.833)) / (math.cos(lat_r) * math.cos(decl))
        - math.tan(lat_r) * math.tan(decl)
    )
    if not -1.0 <= cos_ha <= 1.0:
        return None
    ha = math.degrees(math.acos(cos_ha))
    midnight = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    sunrise = midnight + timedelta(minutes=720 - 4 * (lon + ha) - eqtime)
    sunset = midnight + timedelta(minutes=720 - 4 * (lon - ha) - eqtime)
    return sunrise, sunset


def _location() -> tuple[float, float] | None:
    """The box's lat/lon — the same pair the weather integration uses."""
    try:
        return (
            float(os.environ["BOXBOT_WEATHER_LAT"]),
            float(os.environ["BOXBOT_WEATHER_LON"]),
        )
    except (KeyError, ValueError):
        return None


def local_sun_times(
    now: datetime, location: tuple[float, float] | None = None,
) -> tuple[datetime, datetime] | None:
    """Today's sunrise and sunset in ``now``'s timezone.

    ``None`` with no location configured, or where the sun does not
    rise or set today.
    """
    now = now if now.tzinfo else now.astimezone()
    location = location if location is not None else _location()
    if location is not None:
        # The solar day is anchored to the local calendar date; a UTC
        # date would pick tomorrow's sun every evening west of Greenwich.
        times = sun_times(now.date(), *location)
        if times is not None:
            rise, sets = (t.astimezone(now.tzinfo) for t in times)
            # West of Greenwich the UTC arithmetic can land a day off.
            shift = now.date() - rise.date()
            return rise + shift, sets + shift
    return None


def _nominal_sun_times(now: datetime) -> tuple[datetime, datetime]:
    """7:00 and 19:00 — enough to pick a sky when the real times are
    unknown. Never shown to anyone as a time."""
    base = now.replace(minute=0, second=0, microsecond=0)
    return base.replace(hour=7), base.replace(hour=19)


def phase_for(now: datetime, sunrise: datetime, sunset: datetime) -> str:
    """Which of the six painted skies ``now`` falls in."""
    minute = timedelta(minutes=1)
    if now < sunrise - 40 * minute or now >= sunset + 45 * minute:
        return "night"
    if now < sunrise + 75 * minute:
        return "dawn"
    if now >= sunset + 5 * minute:
        return "dusk"
    if now >= sunset - 75 * minute:
        return "golden"
    midday = sunrise + (sunset - sunrise) / 2
    return "noon" if now < midday + 90 * minute else "afternoon"


def _clock_label(moment: datetime) -> str:
    return f"{moment.hour % 12 or 12}:{moment.minute:02d} {'AM' if moment.hour < 12 else 'PM'}"


def build_atmosphere(
    now: datetime | None = None,
    weather: dict[str, Any] | None = None,
    *,
    dim: bool = False,
    location: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """The atmosphere block of the page state (and of ``get_active``).

    ``weather`` is the weather source's payload (``condition`` is all
    that is read). ``weather_known`` is False when the sky is a guess.
    ``sun`` (the next sunrise or sunset) is ``None`` without a location:
    the phase then follows nominal clock hours, and no time is claimed.
    """
    now = (now or datetime.now()).astimezone()
    scene_weather = weather_class((weather or {}).get("condition"))
    times = local_sun_times(now, location)
    sun: dict[str, str] | None = None
    if times is None:
        sunrise, sunset = _nominal_sun_times(now)
    else:
        sunrise, sunset = times
        if now < sunrise:
            sun = {"next": "sunrise", "at": _clock_label(sunrise)}
        elif now < sunset:
            sun = {"next": "sunset", "at": _clock_label(sunset)}
        else:
            tomorrow = local_sun_times(now + timedelta(days=1), location)
            if tomorrow is not None:
                sun = {"next": "sunrise", "at": _clock_label(tomorrow[0])}
    return {
        "phase": phase_for(now, sunrise, sunset),
        "weather": scene_weather or UNKNOWN_WEATHER,
        "weather_known": scene_weather is not None,
        "time": f"{now.hour:02d}:{now.minute:02d}",
        "dim": bool(dim),
        "sun": sun,
    }
