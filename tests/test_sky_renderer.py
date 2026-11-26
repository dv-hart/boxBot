"""HTML renderer (Living Sky): atmosphere, page state, browser surface,
and the display manager's html mode.

The browser is ``tests/fake_sky_browser.py`` — a DevTools-over-pipe
stand-in — so these run anywhere. Tests that need the real page in a
real Chromium live in ``test_sky_page.py`` and skip without one.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from PIL import Image

from boxbot.core.config import SkySurfaceConfig
from boxbot.displays import sky_state
from boxbot.displays.atmosphere import (
    build_atmosphere,
    local_sun_times,
    phase_for,
    weather_class,
)
from boxbot.displays.manager import DisplayManager
from boxbot.displays.sky_state import (
    PresenceTracker,
    build_state,
    photo_items,
    scene_options,
    state_signature,
)
from boxbot.displays.sky_surface import PAGE_DIR, SkySurface, SkySurfaceError
from boxbot.displays.spec import parse_spec, resolve_bindings, validate_spec

FAKE_BROWSER = str(Path(__file__).parent / "fake_sky_browser.py")
SEATTLE = (47.6062, -122.3321)
PACIFIC = ZoneInfo("America/Los_Angeles")


def seattle(hour: int, minute: int = 0, *, month: int = 11, day: int = 24) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=PACIFIC)


# ---------------------------------------------------------------------------
# Atmosphere
# ---------------------------------------------------------------------------


class TestAtmosphere:
    def test_seattle_november_sun(self) -> None:
        sunrise, sunset = local_sun_times(seattle(12), SEATTLE)
        # Almanac: 7:26 AM and 4:24 PM.
        assert (sunrise.hour, sunrise.minute) == (7, 26)
        assert sunset.hour == 16 and abs(sunset.minute - 24) <= 2

    def test_sun_times_use_the_local_day_in_the_evening(self) -> None:
        # 9 PM Pacific is already tomorrow in UTC; the sun must still be
        # today's, or sunset lands 24 hours out.
        sunrise, sunset = local_sun_times(seattle(21), SEATTLE)
        assert sunrise.date() == sunset.date() == seattle(21).date()

    @pytest.mark.parametrize("hour, minute, phase", [
        (4, 0, "night"), (7, 12, "dawn"), (12, 4, "noon"), (14, 41, "afternoon"),
        (16, 2, "golden"), (16, 48, "dusk"), (21, 36, "night"),
    ])
    def test_phases_through_a_seattle_november_day(self, hour, minute, phase) -> None:
        now = seattle(hour, minute)
        assert phase_for(now, *local_sun_times(now, SEATTLE)) == phase

    def test_midsummer_evening_is_still_golden(self) -> None:
        now = seattle(21, 0, month=6, day=21)
        assert phase_for(now, *local_sun_times(now, SEATTLE)) == "golden"

    @pytest.mark.parametrize("text, expected", [
        ("Sunny", "clear"), ("Partly Sunny", "partly"), ("Mostly Cloudy", "overcast"),
        ("Chance Rain Showers", "rain"), ("Areas Of Drizzle", "rain"),
        ("Slight Chance Rain", "overcast"), ("Rain And Snow", "snow"),
        ("Patchy Fog then Sunny", "fog"), ("", None), (None, None), ("Breezy", None),
    ])
    def test_weather_class(self, text, expected) -> None:
        assert weather_class(text) == expected

    def test_build_atmosphere(self) -> None:
        atmo = build_atmosphere(seattle(16, 2), {"condition": "Rain"}, location=SEATTLE)
        assert atmo["phase"] == "golden"
        assert atmo["weather"] == "rain" and atmo["weather_known"] is True
        assert atmo["time"] == "16:02"
        assert atmo["sun"]["next"] == "sunset" and atmo["sun"]["at"].startswith("4:2")

    def test_after_sunset_points_at_tomorrows_sunrise(self) -> None:
        atmo = build_atmosphere(seattle(21, 36), {}, location=SEATTLE)
        assert atmo["sun"]["next"] == "sunrise" and atmo["sun"]["at"].endswith("AM")

    def test_unknown_weather_is_flagged_not_invented(self) -> None:
        atmo = build_atmosphere(seattle(12), None, location=SEATTLE)
        assert atmo["weather"] == "partly" and atmo["weather_known"] is False

    def test_no_location_claims_no_sun_time(self, monkeypatch) -> None:
        monkeypatch.delenv("BOXBOT_WEATHER_LAT", raising=False)
        monkeypatch.delenv("BOXBOT_WEATHER_LON", raising=False)
        atmo = build_atmosphere(seattle(12), {})
        assert atmo["sun"] is None
        assert atmo["phase"] == "noon"  # nominal 7-to-7 day


# ---------------------------------------------------------------------------
# Voice presence
# ---------------------------------------------------------------------------


class TestPresence:
    def test_a_voice_turn(self) -> None:
        p = PresenceTracker()
        assert p.current() is None
        p.wake_word()
        assert p.current() == {"state": "listening"}
        p.speaker("Carina")
        p.draft("what time is swim")
        assert p.current() == {"state": "listening", "who": "Carina", "text": "what time is swim"}
        p.transcript("What time is swim practice?")
        assert p.current()["state"] == "thinking"
        p.tool("Checking the calendar…")
        assert p.current()["detail"] == "Checking the calendar…"
        assert p.current()["text"] == "What time is swim practice?"
        p.speaking("Five thirty.")
        assert p.current() == {"state": "speaking", "who": "Carina", "text": "Five thirty."}
        p.speaking_done()
        # Whoever speaks next is not known yet: no name carried over.
        assert p.current() == {"state": "listening"}
        p.ended()
        assert p.current() is None

    def test_a_draft_does_not_interrupt_speech(self) -> None:
        p = PresenceTracker()
        p.speaking("Five thirty.")
        p.draft("echo of the speaker")
        assert p.current()["state"] == "speaking"

    def test_tool_without_a_conversation_shows_nothing(self) -> None:
        p = PresenceTracker()
        p.tool("Searching…")
        assert p.current() is None

    def test_expires_without_a_closing_event(self, monkeypatch) -> None:
        clock = [100.0]
        monkeypatch.setattr(sky_state.time, "monotonic", lambda: clock[0])
        p = PresenceTracker(ttl=30)
        p.wake_word()
        clock[0] += 29
        assert p.current() is not None
        clock[0] += 2
        assert p.current() is None


# ---------------------------------------------------------------------------
# Page state
# ---------------------------------------------------------------------------


SPEC = {
    "name": "t",
    "theme": "living_sky",
    "scene": {"horizon": "{cfg.h}", "scale": "{cfg.missing}", "vista": "kerry"},
    "data_sources": [{"name": "cfg", "type": "static", "value": {"h": 300}}],
    "layout": {"type": "column", "children": [
        {"type": "text", "content": "{weather.temp}° in {args.place}"},
    ]},
}
ATMO = {"phase": "noon", "weather": "clear", "time": "12:00", "dim": False}


def make_state(data: dict[str, Any], **kw: Any) -> dict[str, Any]:
    spec = parse_spec(SPEC)
    assert validate_spec(spec) == []
    data = {"cfg": {"h": 300}, **data, "args": {"place": "Seattle"}}
    root = resolve_bindings(spec.root_block, data)
    return build_state(spec, root, data, {"place": "Seattle"},
                       atmosphere=ATMO, people=["Jacob", "Carina"], **kw)


class TestState:
    def test_shape(self) -> None:
        state = make_state({"weather": {"temp": "48"}}, status="Thinking…")
        assert state["v"] == 1 and state["display"] == "t"
        assert state["layout"]["children"][0]["content"] == "48° in Seattle"
        assert state["people"] == ["Jacob", "Carina"]
        assert state["status"] == "Thinking…" and state["presence"] is None
        json.dumps(state)  # must be plain JSON

    def test_scene_bindings_resolve_and_unset_values_drop(self) -> None:
        state = make_state({})
        assert state["scene"] == {"horizon": 300, "vista": "kerry"}

    def test_unknown_scene_option_is_a_validation_error(self) -> None:
        spec = parse_spec({**SPEC, "scene": {"zoom": 2}})
        assert any("scene option 'zoom'" in e for e in validate_spec(spec))

    def test_inbox_sets_the_scene_from_its_mode(self) -> None:
        spec = parse_spec({"name": "i", "layout": {"type": "column"}})
        assert scene_options(spec, {"type": "inbox", "mode": "thread"}, {}) == {
            "horizon": 276, "scale": 0.56,
        }
        nested = {"type": "column", "children": [
            {"type": "inbox", "mode": "triage", "privacy": "discreet"},
        ]}
        assert scene_options(spec, nested, {})["horizon"] == 376

    def test_signature_ignores_the_seconds_hand(self) -> None:
        a = make_state({"clock": {"minute": 4, "second": 1}})
        b = make_state({"clock": {"minute": 4, "second": 2}})
        c = make_state({"clock": {"minute": 5, "second": 2}})
        assert state_signature(a) == state_signature(b) != state_signature(c)

    def test_signature_sees_overlays(self) -> None:
        assert state_signature(make_state({})) != state_signature(
            make_state({}, presence={"state": "listening"}),
        )


class TestPhotoItems:
    @pytest.fixture
    def library(self, tmp_path, monkeypatch) -> Path:
        db = tmp_path / "photos.db"
        with sqlite3.connect(db) as conn:
            conn.executescript("""
                CREATE TABLE photos (id TEXT PRIMARY KEY, filename TEXT, source TEXT,
                    description TEXT, width INTEGER, height INTEGER, created_at TEXT,
                    deleted_at TEXT);
                CREATE TABLE photo_people (photo_id TEXT, label TEXT,
                    bbox_x REAL, bbox_y REAL, bbox_w REAL, bbox_h REAL);
                INSERT INTO photos VALUES ('p1', 'p1.jpg', 'signal', 'Kids at Alki',
                    1200, 1600, '2026-08-14T10:00:00', NULL);
                INSERT INTO photos VALUES ('gone', 'g.jpg', 'signal', 'x', 10, 10, 't', 'deleted');
                INSERT INTO photo_people VALUES ('p1', 'Mia', 0.25, 0.5, 0.2, 0.3);
                INSERT INTO photo_people VALUES ('p1', 'Erik', 600, 400, 120, 160);
                INSERT INTO photo_people VALUES ('p1', 'Jacob', NULL, NULL, NULL, NULL);
            """)

        class Cfg:
            class photos:
                storage_path = str(tmp_path)

        monkeypatch.setattr("boxbot.core.config.get_config", lambda: Cfg)
        return tmp_path

    def test_library_photo_carries_its_metadata(self, library) -> None:
        items = photo_items(["p1"], {"p1": "/photos/p1.jpg"})
        assert items[0]["src"] == "/photos/p1.jpg"
        assert items[0]["description"] == "Kids at Alki"
        assert (items[0]["width"], items[0]["height"]) == (1200, 1600)
        people = {p["name"]: p.get("box") for p in items[0]["people"]}
        assert people["Mia"] == [0.25, 0.5, 0.2, 0.3]
        assert people["Erik"] == [0.5, 0.25, 0.1, 0.1]  # pixels → fractions
        assert people["Jacob"] is None

    def test_unvetted_refs_never_reach_the_page(self, library) -> None:
        items = photo_items(["p1", "/etc/passwd", "gone"], {"p1": "/photos/p1.jpg"})
        assert [i["id"] for i in items] == ["p1"]

    def test_bare_file_has_only_id_and_src(self, library) -> None:
        assert photo_items(["snap.jpg"], {"snap.jpg": "/ws/snap.jpg"}) == [
            {"id": "snap.jpg", "src": "/ws/snap.jpg"},
        ]


# ---------------------------------------------------------------------------
# Browser surface
# ---------------------------------------------------------------------------


def make_config(tmp_path: Path, **overrides: Any) -> SkySurfaceConfig:
    values: dict[str, Any] = {
        "browser": FAKE_BROWSER,
        "headless": True,
        "state_dir": str(tmp_path / "display"),
        "launch_timeout": 5,
    }
    values.update(overrides)
    return SkySurfaceConfig(**values)


def commands(tmp_path: Path) -> list[dict[str, Any]]:
    log = tmp_path / "display" / "commands.log"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


def applied(tmp_path: Path) -> list[dict[str, Any]]:
    """Every state the page was handed, in order."""
    out = []
    for c in commands(tmp_path):
        expression = c["params"].get("expression", "") if c["method"] == "Runtime.evaluate" else ""
        if expression.startswith("BB.apply("):
            out.append(json.loads(expression[len("BB.apply("):-1]))
    return out


def last(tmp_path: Path) -> dict[str, Any]:
    """The newest state the page holds; ``{}`` before the first push."""
    states = applied(tmp_path)
    return states[-1] if states else {}


async def wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.02)


class TestSurface:
    async def test_page_ships_with_the_package(self) -> None:
        for name in ("index.html", "core.js", "core.css", "scene.js", "icons.js",
                     "fonts/fonts.css"):
            assert (PAGE_DIR / name).is_file(), name

    async def test_start_apply_screenshot_stop(self, tmp_path) -> None:
        surface = SkySurface(make_config(tmp_path))
        await surface.start()
        try:
            assert surface.is_running
            await surface.apply({"display": "home", "n": 1})
            png = await surface.screenshot()
            assert png.startswith(b"\x89PNG")
            assert applied(tmp_path) == [{"display": "home", "n": 1}]
        finally:
            await surface.stop()
        assert not surface.is_running

    async def test_launch_is_offline_and_pipe_only(self, tmp_path) -> None:
        surface = SkySurface(make_config(tmp_path))
        await surface.start()
        await surface.stop()
        argv = json.loads((tmp_path / "display" / "argv.json").read_text())
        assert "--remote-debugging-pipe" in argv
        assert not any(a.startswith("--remote-debugging-port") for a in argv)
        assert "--host-resolver-rules=MAP * ~NOTFOUND" in argv
        assert "--proxy-server=http://127.0.0.1:9" in argv
        assert argv[-1].startswith("file://") and "/sky/index.html" in argv[-1]

    async def test_payload_is_ascii_so_any_text_is_a_safe_literal(self, tmp_path) -> None:
        surface = SkySurface(make_config(tmp_path))
        await surface.start()
        try:
            text = "</script>  ü \U0001f327 '\"\\"
            await surface.apply({"t": text})
            expression = [
                c["params"]["expression"] for c in commands(tmp_path)
                if c["method"] == "Runtime.evaluate"
                and c["params"]["expression"].startswith("BB.apply(")
            ][-1]
            assert expression.isascii()
            assert applied(tmp_path)[-1] == {"t": text}
        finally:
            await surface.stop()

    async def test_submit_is_latest_wins(self, tmp_path) -> None:
        surface = SkySurface(make_config(tmp_path))
        await surface.start()
        try:
            for n in range(50):
                surface.submit({"n": n})
            await wait_for(lambda: bool(applied(tmp_path)) and applied(tmp_path)[-1] == {"n": 49})
            assert len(applied(tmp_path)) < 50
        finally:
            await surface.stop()

    async def test_oversized_state_drops_raw_data_then_refuses(self, tmp_path) -> None:
        surface = SkySurface(make_config(tmp_path))
        await surface.start()
        try:
            await surface.apply({"layout": "x", "data": {"blob": "y" * 900_000}})
            assert applied(tmp_path)[-1] == {"layout": "x", "data": {}}
            with pytest.raises(SkySurfaceError, match="bytes"):
                await surface.apply({"layout": "y" * 900_000})
        finally:
            await surface.stop()

    async def test_restarts_and_repushes_after_a_crash(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("boxbot.displays.sky_surface._RESTART_DELAYS", (0.05,))
        surface = SkySurface(make_config(tmp_path))
        await surface.start()
        try:
            await surface.apply({"n": 1})
            os.killpg(surface._proc.pid, signal.SIGKILL)
            await wait_for(lambda: len(applied(tmp_path)) >= 2)
            assert surface.is_running
            assert applied(tmp_path)[-1] == {"n": 1}
        finally:
            await surface.stop()

    async def test_missing_browser_is_an_error(self, tmp_path) -> None:
        surface = SkySurface(make_config(tmp_path, browser="no-such-browser"))
        with pytest.raises(SkySurfaceError, match="not installed"):
            await surface.start()
        assert not surface.is_running

    async def test_page_that_never_loads_fails_the_start(self, tmp_path) -> None:
        (tmp_path / "display").mkdir()
        (tmp_path / "display" / "not-ready").touch()
        surface = SkySurface(make_config(tmp_path, launch_timeout=1))
        with pytest.raises(SkySurfaceError, match="did not come up"):
            await surface.start()
        assert surface._proc is None


# ---------------------------------------------------------------------------
# Display manager, html mode
# ---------------------------------------------------------------------------


@pytest.fixture
async def manager(tmp_path):
    mgr = DisplayManager(
        renderer="html",
        sky_config=make_config(tmp_path, people=["Jacob", "Carina"]),
    )
    await mgr.start(user_dir=tmp_path / "none", agent_dir=tmp_path / "none")
    yield mgr
    await mgr.stop()


class TestManagerHtml:
    async def test_falls_back_to_pil_when_the_browser_will_not_start(self, tmp_path) -> None:
        mgr = DisplayManager(
            renderer="html", sky_config=make_config(tmp_path, browser="no-such-browser"),
        )
        await mgr.start(user_dir=tmp_path / "none", agent_dir=tmp_path / "none")
        try:
            assert mgr.renderer == "pil"
            assert await mgr.switch("clock")
            assert mgr.get_current_frame() is not None
        finally:
            await mgr.stop()

    async def test_switch_pushes_state_not_frames(self, manager, tmp_path) -> None:
        assert manager.renderer == "html"
        assert await manager.switch("notice", args={"title": "Hi", "lines": ["a", "b", "c"]})
        await wait_for(lambda: bool(applied(tmp_path)))
        state = last(tmp_path)
        assert state["display"] == "notice"
        assert state["people"] == ["Jacob", "Carina"]
        assert state["atmosphere"]["phase"] in (
            "dawn", "noon", "afternoon", "golden", "dusk", "night",
        )
        assert "Hi" in json.dumps(state["layout"])
        assert manager.get_current_frame() is None

    async def test_unchanged_state_is_not_pushed_again(self, manager, tmp_path) -> None:
        await manager.switch("notice", args={"title": "Hi", "lines": ["a", "b", "c"]})
        await wait_for(lambda: len(applied(tmp_path)) == 1)
        manager._render_and_update_frame()
        manager._render_and_update_frame()
        await asyncio.sleep(0.1)
        assert len(applied(tmp_path)) == 1

    async def test_status_pill_travels_in_the_state(self, manager, tmp_path) -> None:
        await manager.switch("clock")
        manager.set_status_text("Searching the web…")
        await wait_for(lambda: last(tmp_path).get("status") == "Searching the web…")
        manager.clear_status_text()
        await wait_for(lambda: last(tmp_path).get("status", 0) is None)

    async def test_voice_events_drive_the_dock(self, manager, tmp_path) -> None:
        from boxbot.core.events import (
            AgentSpeaking, ConversationEnded, TranscriptDraft, WakeWordHeard, get_event_bus,
        )

        bus = get_event_bus()
        await manager.switch("clock")
        await bus.publish(WakeWordHeard())
        await bus.publish(TranscriptDraft(text="what time is it"))
        await wait_for(lambda: (last(tmp_path).get("presence") or {}).get("text") == "what time is it")
        await bus.publish(AgentSpeaking(text="Ten past four."))
        await wait_for(lambda: (last(tmp_path).get("presence") or {}).get("state") == "speaking")
        # A WhatsApp thread closing is not the room's conversation.
        await bus.publish(ConversationEnded(channel="whatsapp"))
        await asyncio.sleep(0.05)
        assert manager._presence.current() is not None
        await bus.publish(ConversationEnded(channel="voice"))
        await wait_for(lambda: last(tmp_path).get("presence", 0) is None)

    async def test_capture_and_preview_return_images(self, manager) -> None:
        await manager.switch("clock")
        frame = await manager.capture_frame()
        assert isinstance(frame, Image.Image)
        preview = await manager.render_preview_image("weather_simple")
        assert isinstance(preview, Image.Image)
        assert manager._sky_preview is not None  # its own browser, not the screen's

    async def test_preview_state_carries_no_room_overlays(self, manager, tmp_path) -> None:
        await manager.switch("clock")
        manager.set_status_text("Thinking…")
        manager._presence.wake_word()
        await manager.render_preview_image("weather_simple")
        previews = [s for s in applied(tmp_path) if s["display"] == "weather_simple"]
        assert previews and previews[-1]["status"] is None and previews[-1]["presence"] is None

    async def test_pushed_frame_reaches_the_page_and_clears_on_switch(self, manager, tmp_path) -> None:
        await manager.switch("clock")
        manager.push_frame(Image.new("RGB", (64, 36), (10, 20, 30)))
        await wait_for(lambda: bool(last(tmp_path).get("frame")))
        assert last(tmp_path)["frame"].startswith("data:image/jpeg;base64,")
        await manager.switch("notice", args={"title": "x", "lines": ["a", "b", "c"]})
        await wait_for(lambda: last(tmp_path).get("display") == "notice")
        assert last(tmp_path)["frame"] is None


# ---------------------------------------------------------------------------
# Sky blocks and specs
# ---------------------------------------------------------------------------


class TestSkyBlocks:
    def test_sky_block_parses_with_params_and_children(self) -> None:
        spec = parse_spec({"name": "s", "theme": "living_sky", "layout": {
            "type": "zones", "children": [
                {"type": "zone", "place": "ground", "children": [
                    {"type": "next_event", "events": "{calendar.events}", "label": "Next"},
                ]},
            ],
        }})
        assert validate_spec(spec) == []
        event = spec.root_block.children[0].children[0]
        assert event.block_type == "next_event"
        assert event.params == {"events": "{calendar.events}", "label": "Next"}
        assert spec.root_block.to_dict()["children"][0]["place"] == "ground"

    def test_closed_sets_and_required_fields_are_checked(self) -> None:
        spec = parse_spec({"name": "s", "layout": {"type": "zones", "children": [
            {"type": "zone", "place": "basement"},
            {"type": "next_event"},
            {"type": "inbox", "privacy": "{inbox.privacy}", "mode": ""},
        ]}})
        errors = validate_spec(spec)
        assert any("zone.place value 'basement'" in e for e in errors)
        assert any("next_event block requires 'events'" in e for e in errors)
        assert not any("inbox" in e for e in errors)  # a binding and an unset value pass

    def test_unknown_block_is_still_an_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown block type"):
            parse_spec({"name": "s", "layout": {"type": "hologram"}})

    def test_schema_lists_sky_blocks_for_the_agent(self) -> None:
        from boxbot.tools._sandbox_actions import _build_display_schema

        schema = _build_display_schema()
        assert "living_sky" in schema["themes"]
        inbox = schema["blocks"]["inbox"]
        assert inbox["renderer"] == "html"
        assert inbox["fields"]["privacy"]["valid_values"] == ["open", "discreet"]
        assert schema["blocks"]["zones"]["accepts_children"] is True
        assert "renderer" not in schema["blocks"]["text"]


class TestPhotoParams:
    def test_photo_refs_in_any_param_resolve_or_vanish(self, monkeypatch) -> None:
        from boxbot.displays import manager as manager_mod
        from boxbot.displays.blocks import parse_block

        monkeypatch.setattr(
            manager_mod, "_resolve_photo_refs",
            lambda refs: {r: f"/vetted/{r}" for r in refs if r == "snap.jpg"},
        )
        block = parse_block({"type": "zones", "children": [
            {"type": "capture", "image": "photo:snap.jpg",
             "extra": {"thumbs": ["photo:/etc/passwd", "photo:snap.jpg"]}},
            {"type": "capture", "image": "photo:"},
        ]})
        manager_mod._resolve_photo_params(block)
        first, second = block.children
        assert first.params["image"] == "/vetted/snap.jpg"
        assert first.params["extra"]["thumbs"] == [None, "/vetted/snap.jpg"]
        assert second.params["image"] is None

    async def test_picture_state_carries_items_and_position(
        self, manager, tmp_path, monkeypatch,
    ) -> None:
        from boxbot.displays import manager as manager_mod

        if "picture" not in manager.list_available():
            pytest.skip("picture spec not present")
        monkeypatch.setattr(
            manager_mod, "_resolve_photo_refs",
            lambda refs: {r: f"/photos/{r}.jpg" for r in refs},
        )
        await manager.switch("picture", args={"image_ids": ["a", "b", "c"], "interval": 60})
        await wait_for(lambda: last(tmp_path).get("display") == "picture")
        state = last(tmp_path)
        assert state["layout"]["photos"][0] == {"id": "a", "src": "/photos/a.jpg"}
        assert state["layout"]["position"] == {"index": 1, "count": 3}
        # The next slideshow step: same display, next photo, and it fades.
        manager._active_args["image_ids"] = ["b", "c", "a"]
        manager._slideshow_index = 1
        manager._render_and_update_frame()
        await wait_for(lambda: last(tmp_path)["layout"]["position"]["index"] == 2)
        assert last(tmp_path)["layout"]["photos"][0]["id"] == "b"
        assert last(tmp_path)["transition"] == "always"


def test_chart_keeps_its_block_type_in_the_page_layout() -> None:
    spec = parse_spec({"name": "c", "layout": {"type": "column", "children": [
        {"type": "chart", "data": [1, 2, 3]},
    ]}})
    spec.root_block.children[0].params["type"] = "area"  # as the SDK builder stores it
    state = build_state(spec, spec.root_block, {}, {}, atmosphere=ATMO, people=[])
    chart = state["layout"]["children"][0]
    assert chart["type"] == "chart" and chart["chart_type"] == "area"
