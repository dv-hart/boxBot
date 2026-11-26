"""State for the Living Sky page (HTML renderer).

Everything the page shows is one JSON document, built here from the
same inputs the PIL renderer drew from: the active spec with its
bindings resolved, the sources' data, the display args. Two things are
added that the old renderer did not have — the atmosphere behind every
display and the voice-presence dock on top of it.

The contract with the page is documented in ``displays/sky/README.md``.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from boxbot.displays.blocks import Block
from boxbot.displays.spec import DisplaySpec, resolve_value

logger = logging.getLogger(__name__)

STATE_VERSION = 1

# A dock left behind by a missed event must not sit on the screen
# forever. Longer than any spoken reply or tool call; every voice event
# refreshes it.
PRESENCE_TTL_SECONDS = 180.0

# Scene defaults the inbox block was designed against: the waterline
# sits just above the dock, whose height follows the amount of work.
_INBOX_SCENE: dict[str, dict[str, Any]] = {
    "triage": {"horizon": 300, "scale": 0.62},
    "thread": {"horizon": 276, "scale": 0.56},
    "sent": {"horizon": 336, "scale": 0.78},
    "cleared": {"horizon": 372, "scale": 1},
    "discreet": {"horizon": 376, "scale": 0.92},
}


# ---------------------------------------------------------------------------
# Voice presence
# ---------------------------------------------------------------------------


@dataclass
class Presence:
    """What the voice-state dock shows."""

    state: str  # listening | thinking | speaking
    who: str | None = None
    text: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"state": self.state}
        if self.who:
            d["who"] = self.who
        if self.text:
            d["text"] = self.text
        if self.detail:
            d["detail"] = self.detail
        return d


_KEEP: Any = object()


class PresenceTracker:
    """Follows the room's voice conversation from bus events.

    A plain state machine with no I/O: the display manager feeds it
    events and asks for :meth:`current`. Only the voice channel moves
    it — a WhatsApp turn is for someone who is not looking at the box.
    """

    def __init__(self, ttl: float = PRESENCE_TTL_SECONDS) -> None:
        self._ttl = ttl
        self._presence: Presence | None = None
        self._deadline = 0.0
        self._who: str | None = None

    def current(self) -> dict[str, Any] | None:
        if self._presence is None:
            return None
        if time.monotonic() > self._deadline:
            logger.warning(
                "Voice presence expired without a closing event (%s) — clearing",
                self._presence.state,
            )
            self.clear()
            return None
        return self._presence.to_dict()

    def clear(self) -> None:
        self._presence = None
        self._who = None

    def is_listening(self) -> bool:
        return self._presence is not None and self._presence.state == "listening"

    def _set(self, state: str, text: str = "", detail: str = "") -> None:
        self._presence = Presence(state=state, who=self._who, text=text, detail=detail)
        self._deadline = time.monotonic() + self._ttl

    # -- events --------------------------------------------------------

    def wake_word(self) -> None:
        self._who = None
        self._set("listening")

    def speaker(self, name: str | None) -> None:
        """The speaker of the utterance in hand was confirmed by voice
        (``SpeakerIdentified`` is published for nothing less)."""
        if not name:
            return
        self._who = name
        if self._presence is not None:
            self._presence.who = name

    def draft(self, text: str) -> None:
        """Words heard so far, before the utterance is final."""
        if self._presence is not None and self._presence.state == "speaking":
            return  # barge-in is decided by the conversation, not here
        self._set("listening", text=text.strip())

    def transcript(self, text: str, who: Any = _KEEP) -> None:
        """The utterance is final; the agent has it now.

        ``who`` is the confirmed speaker of *this* utterance, or None
        when it has none (a candidate, an unknown voice): the name on
        the dock is replaced either way. A name is never carried over
        from an earlier utterance — that would be a guess.
        """
        if who is not _KEEP:
            self._who = who or None
        self._set("thinking", text=text.strip())

    def tool(self, status_text: str) -> None:
        if self._presence is None:
            return
        self._set("thinking", text=self._presence.text, detail=status_text)

    def speaking(self, text: str) -> None:
        self._set("speaking", text=text.strip())

    def speaking_done(self) -> None:
        """Speech finished; the mic is open again until the session ends.

        Whoever speaks next is not known yet, so the dock names nobody.
        """
        if self._presence is not None:
            self._who = None
            self._set("listening")

    def ended(self) -> None:
        self.clear()


# ---------------------------------------------------------------------------
# Photos
# ---------------------------------------------------------------------------


def photo_items(refs: list[Any], paths: dict[str, str]) -> list[dict[str, Any]]:
    """Describe each photo the ``picture`` block may show.

    ``paths`` maps a ref (library id or allowed file path) to the
    absolute file it renders from — refs missing there are dropped, so
    the page is never handed a path nobody vetted. Library photos gain
    their stored metadata; a bare file (camera snapshot, workspace
    image) has only ``id`` and ``src``.
    """
    items: list[dict[str, Any]] = []
    meta = _photo_metadata([str(r) for r in refs if str(r) in paths])
    for ref in refs:
        ref = str(ref)
        src = paths.get(ref)
        if src is None:
            continue
        item: dict[str, Any] = {"id": ref, "src": src}
        item.update(meta.get(ref, {}))
        items.append(item)
    return items


def _photo_metadata(ids: list[str]) -> dict[str, dict[str, Any]]:
    if not ids:
        return {}
    try:
        from boxbot.core.config import get_config

        db_path = Path(get_config().photos.storage_path) / "photos.db"
    except Exception:
        return {}
    if not db_path.exists():
        return {}

    out: dict[str, dict[str, Any]] = {}
    marks = ",".join("?" for _ in ids)
    try:
        with contextlib.closing(sqlite3.connect(str(db_path))) as conn:
            conn.row_factory = sqlite3.Row
            for r in conn.execute(
                "SELECT id, source, description, width, height, created_at "
                f"FROM photos WHERE id IN ({marks}) AND deleted_at IS NULL",
                ids,
            ):
                entry: dict[str, Any] = {
                    "description": r["description"] or "",
                    "source": r["source"] or "",
                    "taken": r["created_at"],
                    "people": [],
                }
                if r["width"] and r["height"]:
                    entry["width"], entry["height"] = r["width"], r["height"]
                out[r["id"]] = entry
            for r in conn.execute(
                "SELECT photo_id, label, bbox_x, bbox_y, bbox_w, bbox_h "
                f"FROM photo_people WHERE photo_id IN ({marks})",
                ids,
            ):
                entry = out.get(r["photo_id"])
                if entry is None or not r["label"]:
                    continue
                person: dict[str, Any] = {"name": r["label"]}
                box = [r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"]]
                if all(v is not None for v in box):
                    width, height = entry.get("width"), entry.get("height")
                    # Stored either as fractions or as pixels; the page
                    # wants fractions of the photo.
                    if max(box) > 1.0 and width and height:
                        box = [box[0] / width, box[1] / height,
                               box[2] / width, box[3] / height]
                    if max(box) <= 1.0:
                        person["box"] = [round(float(v), 4) for v in box]
                entry["people"].append(person)
    except sqlite3.Error as e:
        logger.warning("photo metadata lookup failed: %s", e)
    return out


# ---------------------------------------------------------------------------
# The state document
# ---------------------------------------------------------------------------


def _first_block(block: dict[str, Any], block_type: str) -> dict[str, Any] | None:
    if block.get("type") == block_type:
        return block
    for child in block.get("children") or []:
        if isinstance(child, dict):
            found = _first_block(child, block_type)
            if found is not None:
                return found
    return None


def scene_options(spec: DisplaySpec, layout: dict[str, Any],
                  data: dict[str, Any]) -> dict[str, Any]:
    """Scene options for a display: the spec's ``scene``, bindings
    resolved, with unset values dropped so the page's defaults apply."""
    scene = resolve_value(copy.deepcopy(spec.scene), data) if spec.scene else {}
    scene = {k: v for k, v in scene.items() if v not in (None, "")}
    inbox = _first_block(layout, "inbox")
    if inbox is not None and "horizon" not in scene:
        mode = "discreet" if inbox.get("privacy") == "discreet" else inbox.get("mode")
        scene = {**_INBOX_SCENE.get(str(mode), _INBOX_SCENE["triage"]), **scene}
    return scene


def _layout(block: Block) -> dict[str, Any]:
    """The block tree as the page reads it.

    Like ``Block.to_dict``, except that a chart keeps its block type:
    there, the chart's own ``type`` param (line / bar / area) overwrites
    it. The page gets ``type: "chart"`` and ``chart_type`` instead.
    """
    params = dict(block.params)
    if block.block_type == "chart" and "type" in params:
        params["chart_type"] = params.pop("type")
    out: dict[str, Any] = {**params, "type": block.block_type}
    if block.children:
        out["children"] = [_layout(child) for child in block.children]
    return out


def build_state(
    spec: DisplaySpec,
    root: Block,
    data: dict[str, Any],
    args: dict[str, Any],
    *,
    atmosphere: dict[str, Any],
    people: list[str],
    status: str | None = None,
    presence: dict[str, Any] | None = None,
    frame: str | None = None,
    transition: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Assemble the page state for ``spec`` drawn against ``data``.

    ``root`` is the spec's block tree with bindings already resolved
    against ``data`` (which includes ``args`` under its own key).
    """
    layout = _layout(root)
    state: dict[str, Any] = {
        "v": STATE_VERSION,
        "display": spec.name,
        "transition": transition or spec.transition,
        "people": list(people),
        "atmosphere": atmosphere,
        "scene": scene_options(spec, layout, data),
        "layout": layout,
        "data": data,
        "args": dict(args),
        "status": status or None,
        "presence": presence,
        "frame": frame,
    }
    if now is not None:
        state["now"] = now.astimezone().isoformat(timespec="seconds")
    return state


def state_signature(state: dict[str, Any]) -> str:
    """Identity of what the page would draw for ``state``.

    Two states with the same signature look the same, so the second
    need not be pushed. The clock's seconds are left out: the page
    ticks its own clocks, and a display that declares the clock source
    would otherwise push once a second for nothing.
    """
    import json

    data = state.get("data") or {}
    clock = data.get("clock")
    if isinstance(clock, dict) and "second" in clock:
        data = {**data, "clock": {k: v for k, v in clock.items() if k != "second"}}
    slim = {k: v for k, v in state.items() if k != "now"}
    slim["data"] = data
    return json.dumps(slim, sort_keys=True, default=str)
