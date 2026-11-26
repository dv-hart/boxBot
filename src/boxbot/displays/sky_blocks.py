"""Blocks only the HTML renderer (Living Sky) draws.

The original vocabulary (``row``, ``text``, ``clock`` …) is typed in
:mod:`boxbot.displays.blocks` and drawn by both renderers. These are
drawn by the page alone (``displays/sky/blocks/*.js``), so on the
Python side each is just a name and a description of its params: enough
to validate a spec, and to tell the agent what the block takes
(``bb.display.schema()``).

A field is ``{"type", "describe", "required"?, "default"?, "values"?}``.
``values`` is a closed set; a value that is a ``{binding}`` is not
checked, since it only has a value at render time.
"""

from __future__ import annotations

from typing import Any

def _f(type_: str, describe: str, **extra: Any) -> dict[str, Any]:
    return {"type": type_, "describe": describe, **extra}


_EVENTS = (
    "calendar events; each needs 'start' (ISO) and may carry title, end, "
    "location, all_day, person (name, 'family' or 'A+B'), short_title "
    "(<=14 chars), leave_by, done"
)
_WEEK_START = ("rolling", "monday", "sunday")
_UI = ("full", "minimal")

SKY_BLOCKS: dict[str, dict[str, Any]] = {
    # -- placement ------------------------------------------------------
    "zones": {
        "children": True,
        "describe": "Root of a Living Sky layout; children are zone and surface blocks.",
        "fields": {},
    },
    "zone": {
        "children": True,
        "describe": "One placement area over the scene.",
        "fields": {
            "place": _f("str", "sky = top-left column, sky-right = top-right "
                        "corner, ground = bottom band", default="sky",
                        values=("sky", "sky-right", "ground")),
        },
    },
    "surface": {
        "children": True,
        "describe": "The slab: the one dark surface for dense content, docked to an edge.",
        "fields": {
            "dock": _f("str", "left = 412px full-height column, bottom = full "
                       "width, height by content", default="left",
                       values=("left", "bottom")),
        },
    },
    # -- home -----------------------------------------------------------
    "sky_clock": {
        "describe": "Date label and large serif time; ticks by itself.",
        "fields": {
            "size": _f("str", "hero (home) or compact", default="hero",
                       values=("hero", "compact")),
            "date": _f("bool", "show the date label", default=True),
            "h24": _f("bool", "24-hour time", default=False),
            "weather": _f("dict", "compact only: {temp, condition, icon} beside the time"),
        },
    },
    "now": {
        "describe": "Current weather on one line: temperature, condition, one outlook sentence.",
        "fields": {
            "weather": _f("dict", "the whole weather source"),
            "style": _f("str", "line adds the outlook sentence", default="line",
                        values=("line", "compact")),
            "temp": _f("str", "overrides weather.temp"),
            "condition": _f("str", "overrides weather.condition"),
            "icon": _f("str", "overrides weather.icon"),
            "outlook": _f("str", "one sentence, e.g. 'Rain back by 10'; falls "
                          "back to today's forecast row"),
        },
    },
    "next_event": {
        "describe": "The running or next timed event with a live countdown.",
        "fields": {
            "events": _f("list", _EVENTS, required=True),
            "label": _f("str", "kicker word", default="Next"),
            "empty": _f("str", "shown with nothing upcoming", default="Nothing coming up"),
        },
    },
    "house": {
        "describe": "Who is home, then quiet counts. Zero counts are omitted.",
        "fields": {
            "present": _f("list", "people home: names or {name}"),
            "replies": _f("int|list", "messages needing a reply"),
            "todos": _f("int|list", "open to-dos (a list is counted)"),
            "sun": _f("dict", "{next: sunrise|sunset, at}; defaults to the atmosphere's"),
        },
    },
    # -- calendar -------------------------------------------------------
    "say": {
        "describe": "The agent's sentence in serif, greeting as its lead-in.",
        "fields": {
            "greeting": _f("str", "e.g. 'Good morning, Jacob'"),
            "text": _f("str", "the sentence; without it a plain one is built from events"),
            "events": _f("list", "used only for the fallback sentence"),
            "who": _f("str", "name for the fallback greeting"),
        },
    },
    "agenda_timeline": {
        "describe": "Today as an ordered rail: now marker, past collapsed, next event raised.",
        "fields": {
            "events": _f("list", _EVENTS, required=True),
            "mode": _f("str", "evening shows tonight then tomorrow; auto switches "
                       "at 17:00", default="auto", values=("auto", "morning", "evening")),
            "forecast": _f("list", "weather forecast rows, for the Tomorrow divider"),
            "gap_min": _f("int", "a gap this long opens the rail", default=90),
            "show": _f("list", "subset of person, place, countdown, leave_by"),
            "title": _f("str", "header label; derived when unset"),
            "sections": _f("list", "replaces mode: [{day, label?, remaining?, raise?, now?}]"),
        },
    },
    "todo_list": {
        "describe": "Open to-dos with a short due tag.",
        "fields": {
            "items": _f("list", "{description, due_date?, status?} or strings", required=True),
            "max": _f("int", "rows shown (1-6)", default=3),
            "label": _f("str", "header", default="To-do"),
        },
    },
    "forecast_strip": {
        "describe": "Day, icon, high for the coming days.",
        "fields": {
            "days": _f("list", "{day, icon, high}", required=True),
            "count": _f("int", "days shown (1-7)", default=5),
            "skip_today": _f("bool", "drop today's row", default=True),
        },
    },
    "cal_header": {
        "describe": "Serif title, date range and event count, optional sentence.",
        "fields": {
            "range": _f("str", "what the header covers", default="week", values=("week", "day")),
            "events": _f("list", "for the count"),
            "start": _f("str", "first column of the week", default="rolling",
                        values=_WEEK_START),
            "title": _f("str", "overrides the derived title"),
            "text": _f("str", "one sentence under the title"),
        },
    },
    "legend": {
        "describe": "Person-colour key.",
        "fields": {
            "people": _f("list", "names; defaults to the household"),
            "family": _f("str|bool", "label for shared events; false hides it",
                         default="Everyone"),
        },
    },
    "week_grid": {
        "describe": "Seven day columns, today wider; multi-day events as bars.",
        "fields": {
            "events": _f("list", _EVENTS, required=True),
            "forecast": _f("list", "weather forecast rows, per-day icon and high/low"),
            "start": _f("str", "first column", default="rolling", values=_WEEK_START),
            "max_per_day": _f("int", "then '+N more' (1-8)", default=5),
        },
    },
    "day_grid": {
        "describe": "Today on an hour scale, a column per person.",
        "fields": {
            "events": _f("list", _EVENTS, required=True),
            "people": _f("list", "column order; defaults to the household"),
            "from": _f("str", "'HH:MM'", default="07:30"),
            "to": _f("str", "'HH:MM'", default="20:30"),
            "background": _f("list", "{label, start, end, people?} context blocks, e.g. school hours"),
        },
    },
    # -- photos ---------------------------------------------------------
    "picture": {
        "describe": (
            "A photo with its caption. Portrait sits beside the living scene; "
            "landscape fills the screen. Bind photos to {photos.items} "
            "(filled from args.image_ids)."
        ),
        "fields": {
            "photos": _f("list", "photo records; index 0 is on screen"),
            "layout": _f("str", "auto picks by aspect ratio", default="auto",
                         values=("auto", "side", "fill", "fit", "diptych")),
            "pair": _f("bool", "let auto show two portraits side by side", default=False),
            "reason": _f("str", "why it is showing, one line"),
            "position": _f("dict|str", "{index, count} (1-based) or '3/12'"),
            "ui": _f("str", "minimal fades the caption", default="full", values=_UI),
            "minimal_after_s": _f("number", "seconds until minimal; 0 = never", default=6),
            "dwell_s": _f("number", "fill only: slow zoom over this many seconds", default=0),
            "temp": _f("str", "degrees, beside the clock"),
            "condition": _f("str", "weather word"),
            "icon": _f("str", "weather icon"),
        },
    },
    "photo_caption": {
        "describe": "A photo's caption alone: reason, title, people, date, source.",
        "fields": {
            "photo": _f("dict", "photo record supplying the fields below"),
            "title": _f("str", "<=34 chars"),
            "description": _f("str", "shown when there is no title"),
            "people": _f("list", "names or {name}"),
            "taken": _f("str", "ISO date"),
            "source": _f("str", "signal, whatsapp, camera, …"),
            "reason": _f("str", "why it is showing"),
            "position": _f("dict|str", "{index, count} or '3/12'"),
            "ui": _f("str", "minimal fades it", default="full", values=_UI),
            "minimal_after_s": _f("number", "seconds until minimal; 0 = never", default=6),
        },
    },
    "sky_chip": {
        "describe": "Small pill with time and weather, for full-bleed layouts.",
        "fields": {
            "temp": _f("str", "degrees"),
            "icon": _f("str", "weather icon"),
            "corner": _f("str", "where it sits", default="top-right",
                         values=("top-right", "top-left")),
        },
    },
    "capture": {
        "describe": "A camera still just taken ('here's what I see'), not a library photo.",
        "fields": {
            "image": _f("str|dict", "'photo:<path or id>' or a photo record", required=True),
            "say": _f("str", "the agent's sentence about it"),
            "people": _f("list", "{name, box?}; boxed people get a name tag"),
            "saved": _f("bool", "false shows 'Not saved'", default=False),
            "at": _f("str", "ISO time it was taken; always send it"),
            "temp": _f("str", "degrees"),
            "icon": _f("str", "weather icon"),
            "ui": _f("str", "minimal fades the text", default="full", values=_UI),
            "minimal_after_s": _f("number", "seconds until minimal; 0 = never", default=0),
        },
    },
    # -- inbox ----------------------------------------------------------
    "inbox": {
        "describe": (
            "Mail that needs a person, as the agent triaged it. The agent "
            "supplies every field (static source + update_data)."
        ),
        "fields": {
            "mode": _f("str", "triage = list, thread = one open, sent = just "
                       "sent with undo, cleared = nothing left", default="triage",
                       values=("triage", "thread", "sent", "cleared")),
            "focus": _f("int", "1-based thread in focus", default=1),
            "privacy": _f("str", "discreet shows counts, senders and whose inbox "
                          "ONLY: no subjects, summaries, drafts or deadlines. "
                          "Set it when someone other than the owners is in the room.",
                          default="open", values=("open", "discreet")),
            "privacy_note": _f("str", "discreet footer text"),
            "count": _f("int", "threads needing a reply; defaults to len(threads)"),
            "owners": _f("dict", "{name: count}; derived from threads[].for"),
            "threads": _f("list", "{sender, role?, for?, summary?, short?, due?, "
                          "urgent?, suggest?, quote?, draft?, say?}"),
            "handled": _f("dict", "{count, breakdown?} the agent dealt with itself"),
            "fyi": _f("list", "short FYI strings"),
            "quote": _f("str", "thread mode: one quoted line"),
            "draft": _f("dict|str", "{text, from?, sign?, effect?, effect_done?}"),
            "say": _f("list", "phrases the person can speak"),
            "undo_seconds": _f("int", "sent mode: undo countdown"),
            "arrived": _f("int", "cleared mode: mail that arrived today"),
            "last": _f("dict", "cleared mode: {at, text} last item filed"),
            "temp": _f("str", "degrees"),
            "condition": _f("str", "weather word"),
            "icon": _f("str", "weather icon"),
        },
    },
}


def validate_sky_block(block_type: str, params: dict[str, Any]) -> list[str]:
    """Errors for one sky block's params (required fields, closed sets)."""
    errors: list[str] = []
    fields = SKY_BLOCKS[block_type].get("fields", {})
    for name, field in fields.items():
        value = params.get(name)
        if field.get("required") and value in (None, "", []):
            errors.append(f"{block_type} block requires '{name}'")
            continue
        values = field.get("values")
        # "" is what an unset binding resolves to: the default applies.
        if values is None or value in (None, ""):
            continue
        if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
            continue
        if value not in values:
            errors.append(
                f"{block_type}.{name} value '{value}' is not one of: "
                f"{', '.join(str(v) for v in values)}"
            )
    return errors
