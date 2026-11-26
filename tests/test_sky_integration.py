"""The HTML renderer (Living Sky) next to the rest of the display
system: the kiosk web view, the PIL-only shopping screens, pictures,
the agent's display actions, the voice dock, and losing the browser.

Same fake browser as ``test_sky_renderer.py``.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import signal
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from boxbot.displays import manager as manager_mod
from boxbot.displays.manager import DisplayManager
from boxbot.displays.shop_blocks import publish
from boxbot.displays.sky_state import PresenceTracker
from tests.test_sky_renderer import applied, last, make_config, wait_for

MERCHANT_TITLE = "<img src=x onerror=alert(1)> Kettle"


@pytest.fixture
async def manager(tmp_path):
    mgr = DisplayManager(
        renderer="html",
        sky_config=make_config(tmp_path, people=["Jacob", "Carina"]),
    )
    await mgr.start(user_dir=tmp_path / "none", agent_dir=tmp_path / "none")
    yield mgr
    await mgr.stop()


def products_payload(title: str = MERCHANT_TITLE) -> dict[str, Any]:
    return {
        "items": [{
            "number": 1, "title": title, "variant": "Hardcover · 2019",
            "price": "$24.00", "store": "Example Shop", "approved": True, "image": None,
        }],
        "hint": 'Say "add it", or ask for another',
    }


def frame_image(state: dict[str, Any]) -> Image.Image:
    prefix = "data:image/jpeg;base64,"
    assert state["frame"].startswith(prefix)
    image = Image.open(io.BytesIO(base64.b64decode(state["frame"][len(prefix):])))
    image.load()
    return image


class FakeWebView:
    """The parts of ``WebView`` the manager touches."""

    def __init__(self) -> None:
        self.is_open = False
        self.holds_mic_mute = False
        self.paused = 0
        self.speaking: list[bool] = []
        self.on_exit: Any = None

    async def open(self, target: Any, idle_minutes: Any = None) -> None:
        self.is_open = True

    async def close(self) -> None:
        self.is_open = False
        self.holds_mic_mute = False

    def state(self) -> dict[str, Any] | None:
        return {"url": "https://example.com/", "paused": False} if self.is_open else None

    def pause_now(self) -> None:
        self.paused += 1

    def set_speaking(self, value: bool) -> None:
        self.speaking.append(value)


# ---------------------------------------------------------------------------
# 3a — the kiosk web view
# ---------------------------------------------------------------------------


class TestWebView:
    async def test_open_then_self_close_brings_the_sky_page_back(self, manager, tmp_path) -> None:
        view = manager._web_view = FakeWebView()
        await manager.switch("clock")
        assert await manager.switch("web", args={"url": "https://example.com/watch"})
        # The page under the kiosk window shows the "Opening…" card.
        await wait_for(lambda: last(tmp_path).get("display") == "web")
        assert "example.com" in json.dumps(last(tmp_path)["layout"])
        assert view.is_open and manager.is_pinned()

        view.is_open = False  # the browser closed itself
        await manager._on_web_view_exit("media ended")
        await wait_for(lambda: last(tmp_path).get("display") == "clock")
        assert manager.get_active() == "clock" and not manager.is_pinned()
        assert manager.get_web_exit()["reason"] == "media ended"
        assert manager.renderer == "html"

    async def test_mic_mute_and_pause_hooks_do_not_depend_on_the_renderer(self, manager) -> None:
        from boxbot.core.events import (
            AgentSpeaking, AgentSpeakingDone, WakeWordHeard, get_event_bus,
        )

        view = manager._web_view = FakeWebView()
        await manager.switch("web", args={"url": "https://example.com/watch"})
        manager.note_web_mic_muted()
        assert manager.web_holds_mic_mute()
        bus = get_event_bus()
        await bus.publish(WakeWordHeard())
        await bus.publish(AgentSpeaking(text="hi"))
        await bus.publish(AgentSpeakingDone())
        assert view.paused == 1 and view.speaking == [True, False]
        manager.release_web_mic_mute()
        assert not manager.web_holds_mic_mute()

    async def test_screen_browser_restart_closes_the_web_view(
        self, manager, tmp_path, monkeypatch,
    ) -> None:
        # A relaunched screen window opens above the kiosk window; the
        # page must not play on behind it.
        monkeypatch.setattr("boxbot.displays.sky_surface._RESTART_DELAYS", (0.05,))
        view = manager._web_view = FakeWebView()
        await manager.switch("clock")
        await manager.switch("web", args={"url": "https://example.com/watch"})
        await wait_for(lambda: last(tmp_path).get("display") == "web")
        os.killpg(manager._sky._proc.pid, signal.SIGKILL)
        await wait_for(lambda: not view.is_open)
        await wait_for(lambda: manager.get_active() != "web")
        assert manager.get_web_exit()["reason"] == manager_mod.WEB_EXIT_SCREEN_RESTARTED
        assert manager.renderer == "html"

    async def test_sky_browser_launch_shares_nothing_with_the_kiosk(self, tmp_path, manager) -> None:
        argv = json.loads((tmp_path / "display" / "argv.json").read_text())
        joined = " ".join(argv)
        assert "sudo" not in argv and "boxbot-kiosk" not in joined
        assert f"--user-data-dir={tmp_path / 'display' / 'profile-screen'}" in argv
        assert "--mute-audio" in argv


# ---------------------------------------------------------------------------
# 3b — products / cart: drawn by PIL, shown as a frame
# ---------------------------------------------------------------------------


class TestShoppingScreens:
    async def test_products_reach_the_page_as_pixels_only(self, manager, tmp_path) -> None:
        token = publish(products_payload())
        assert await manager.switch("products", args={"view": token})
        await wait_for(lambda: last(tmp_path).get("display") == "products")
        state = last(tmp_path)
        image = frame_image(state)
        assert image.size == (1024, 600)
        assert len(set(image.resize((64, 36)).getdata())) > 4  # something was drawn
        # Nothing from the catalogue travels as text: the page gets the
        # token and an empty layout.
        rest = json.dumps({k: v for k, v in state.items() if k != "frame"})
        assert "Kettle" not in rest and "onerror" not in rest and "$24" not in rest
        assert state["layout"] == {"type": "column"}
        assert state["args"] == {"view": token}
        assert state["data"] == {"args": {"view": token}}
        assert manager.get_current_frame() is None  # PIL frame buffer stays unused

    async def test_frame_matches_the_pil_renderer(self, manager, tmp_path) -> None:
        token = publish(products_payload("Plain Kettle"))
        await manager.switch("products", args={"view": token})
        await wait_for(lambda: bool(last(tmp_path).get("frame")))
        shown = frame_image(last(tmp_path))
        expected = manager.render_preview("products", args={"view": token})
        diff = [
            abs(a - b)
            for pa, pb in zip(shown.resize((128, 75)).getdata(),
                              expected.convert("RGB").resize((128, 75)).getdata())
            for a, b in zip(pa, pb)
        ]
        assert sum(diff) / len(diff) < 3  # JPEG noise only

    async def test_redraw_on_new_view_and_no_redraw_for_voice_events(
        self, manager, tmp_path, monkeypatch,
    ) -> None:
        renders = []
        original = manager._renderer.render_block_tree

        def counting(*a: Any, **kw: Any) -> Any:
            renders.append(1)
            return original(*a, **kw)

        monkeypatch.setattr(manager._renderer, "render_block_tree", counting)
        first = publish(products_payload("First"))
        await manager.switch("products", args={"view": first})
        await wait_for(lambda: bool(last(tmp_path).get("frame")))
        assert len(renders) == 1
        frame_one = last(tmp_path)["frame"]

        manager.set_status_text("Adding to cart…")
        await wait_for(lambda: last(tmp_path).get("status") == "Adding to cart…")
        manager._presence.wake_word()
        manager._push_sky_state()
        await asyncio.sleep(0.05)
        assert len(renders) == 1
        assert last(tmp_path)["presence"] is None  # no dock over the cards

        # The image arrived: shopping republishes and switches again.
        second = publish(products_payload("First, now with a much longer title"))
        await manager.switch("products", args={"view": second})
        await wait_for(lambda: last(tmp_path)["args"] == {"view": second})
        assert len(renders) == 2 and last(tmp_path)["frame"] != frame_one

    async def test_cart_then_a_page_display_clears_the_frame(self, manager, tmp_path) -> None:
        token = publish({"lines": [], "empty": True, "total": "$0.00"})
        await manager.switch("cart", args={"view": token})
        await wait_for(lambda: last(tmp_path).get("display") == "cart")
        assert last(tmp_path)["frame"]
        await manager.switch("clock")
        await wait_for(lambda: last(tmp_path).get("display") == "clock")
        assert last(tmp_path)["frame"] is None

    async def test_preview_and_screenshot_of_a_pil_display(self, manager) -> None:
        token = publish(products_payload())
        await manager.switch("products", args={"view": token})
        preview = await manager.render_preview_image("products", args={"view": token})
        assert isinstance(preview, Image.Image) and preview.size == (1024, 600)
        assert manager._sky_preview is None  # PIL drew it; no second browser
        assert isinstance(await manager.capture_frame(), Image.Image)

    def test_which_displays_need_a_pil_frame(self) -> None:
        from boxbot.displays.builtins import get_builtin_specs
        from boxbot.displays.spec import parse_spec

        needs = {s.name: manager_mod._needs_pil_frame(s) for s in get_builtin_specs()}
        assert needs == {
            "clock": False, "weather_simple": False, "picture": False, "notice": False,
            "web": False, "products": True, "cart": True,
        }
        sky = parse_spec({"name": "s", "layout": {"type": "zones", "children": [
            {"type": "text", "content": "x"},
        ]}})
        assert not manager_mod._needs_pil_frame(sky)

    def test_frames_are_size_bounded(self, monkeypatch) -> None:
        noise = Image.frombytes("RGB", (2048, 1200), os.urandom(2048 * 1200 * 3))
        url = manager_mod._frame_data_url(noise, (1024, 600))
        # Scaled to the panel; worst-case content still fits the cap…
        if url is not None:
            raw = base64.b64decode(url.split(",", 1)[1])
            assert len(raw) <= manager_mod._MAX_FRAME_BYTES
            assert Image.open(io.BytesIO(raw)).size == (1024, 600)
        # …and what cannot fit is refused, not sent.
        monkeypatch.setattr(manager_mod, "_MAX_FRAME_BYTES", 1000)
        assert manager_mod._frame_data_url(noise, (1024, 600)) is None

    async def test_oversized_pushed_frame_is_dropped(self, manager, tmp_path, monkeypatch) -> None:
        await manager.switch("clock")
        await wait_for(lambda: last(tmp_path).get("display") == "clock")
        monkeypatch.setattr(manager_mod, "_MAX_FRAME_BYTES", 100)
        manager.push_frame(Image.new("RGB", (640, 360), (10, 20, 30)))
        await asyncio.sleep(0.1)
        assert last(tmp_path)["frame"] is None


# ---------------------------------------------------------------------------
# 3c — picture
# ---------------------------------------------------------------------------


class TestPicture:
    async def test_workspace_and_absolute_paths_and_unvetted_refs(
        self, manager, tmp_path, monkeypatch,
    ) -> None:
        from boxbot.core import paths
        from boxbot.tools import _sandbox_actions as actions

        workspace = tmp_path / "workspace"
        (workspace / "img").mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(workspace / "img" / "snap.jpg")
        outside = tmp_path / "secret.jpg"
        Image.new("RGB", (8, 8)).save(outside)
        monkeypatch.setattr(paths, "WORKSPACE_DIR", workspace)
        monkeypatch.setattr(
            actions, "_is_attach_allowed",
            lambda p: str(p).startswith(str(workspace.resolve())),
        )
        absolute = str((workspace / "img" / "snap.jpg").resolve())
        assert await manager.switch("picture", args={
            "image_ids": ["img/snap.jpg", absolute, str(outside), "no-such-id"],
            "interval": 60,
        })
        await wait_for(lambda: last(tmp_path).get("display") == "picture")
        photos = last(tmp_path)["layout"]["photos"]
        assert [p["src"] for p in photos] == [absolute, absolute]
        assert str(outside) not in json.dumps(last(tmp_path)["layout"])

    async def test_show_on_screen_action_switches_the_page(
        self, manager, tmp_path, monkeypatch,
    ) -> None:
        from boxbot.tools import _sandbox_actions as actions

        monkeypatch.setattr(actions, "get_display_manager", lambda: manager, raising=False)
        monkeypatch.setattr(manager_mod, "get_display_manager", lambda: manager)
        monkeypatch.setattr(
            manager_mod, "_resolve_photo_refs",
            lambda refs: {r: f"/photos/{r}.jpg" for r in refs},
        )
        ctx = SimpleNamespace(action_log=[], image_attachments=[])
        result = await actions.process_action(
            {"_sdk": "photos.show_on_screen", "photo_ids": ["p1"],
             "_expects_response": True}, ctx,
        )
        assert result.get("status") == "ok", result
        await wait_for(lambda: last(tmp_path).get("display") == "picture")
        assert last(tmp_path)["layout"]["photos"][0]["src"] == "/photos/p1.jpg"

    async def test_empty_slideshow_shows_the_notice(self, manager, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(manager, "_load_slideshow_ids", lambda limit=50: [])
        assert await manager.switch("picture")
        await wait_for(lambda: last(tmp_path).get("display") == "notice")
        assert "No photos yet" in json.dumps(last(tmp_path)["layout"])


# ---------------------------------------------------------------------------
# 3d — the agent's display actions
# ---------------------------------------------------------------------------


class TestAgentActions:
    @pytest.fixture
    def ctx(self, manager, monkeypatch, tmp_path):
        from boxbot.tools import _sandbox_actions as actions

        monkeypatch.setattr(manager_mod, "get_display_manager", lambda: manager)
        monkeypatch.setattr(actions, "_previews_dir", lambda: tmp_path)
        return SimpleNamespace(action_log=[], image_attachments=[])

    async def run(self, ctx: Any, sub: str, **payload: Any) -> dict[str, Any]:
        from boxbot.tools._sandbox_actions import process_action

        return await process_action(
            {"_sdk": f"display.{sub}", "_expects_response": True, **payload}, ctx,
        )

    async def test_get_active_reports_both_sides(self, manager, ctx) -> None:
        await manager.switch("clock")
        result = await self.run(ctx, "get_active")
        assert result["status"] == "ok" and result["name"] == "clock"
        assert result["renderer"] == "html"
        assert result["atmosphere"]["phase"]
        assert result["web"] is None and result["web_closed"] is None
        assert result["pinned"] is True
        # The echo in sdk_actions is trimmed: scalars stay, records go.
        entry = ctx.action_log[-1]
        assert entry["renderer"] == "html" and entry["name"] == "clock"
        assert "atmosphere" not in entry and "rotation" not in entry and "args" not in entry

    async def test_screenshot_returns_an_image(self, manager, ctx) -> None:
        await manager.switch("notice", args={"title": "Hi", "lines": ["a"]})
        result = await self.run(ctx, "screenshot")
        assert result["status"] == "ok" and result["attached"] is True
        assert len(ctx.image_attachments) == 1

    async def test_screenshot_with_the_web_view_open_is_refused(self, manager, ctx) -> None:
        manager._web_view = FakeWebView()
        await manager.switch("web", args={"url": "https://example.com/watch"})
        result = await self.run(ctx, "screenshot")
        assert result["status"] == "error" and "web view" in result["error"]

    async def test_schema_echo_is_trimmed(self, manager, ctx) -> None:
        result = await self.run(ctx, "schema")
        assert result["renderer"] == "html" and "zones" in result["blocks"]
        entry = ctx.action_log[-1]
        assert "blocks" not in entry and entry["renderer"] == "html"
        assert len(json.dumps(entry)) < 600

    async def test_rotation_and_pin_under_html(self, manager, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(manager_mod, "_persist_rotation_state", lambda state: None)
        await manager.set_rotation(["clock", "notice"], 60)
        await wait_for(lambda: last(tmp_path).get("display") == "clock")
        assert manager.is_rotating() and not manager.is_pinned()
        await manager.switch("notice", args={"title": "Pinned", "lines": ["x"]})
        assert manager.is_pinned() and not manager.is_rotating()
        await manager.unpin()
        assert manager.is_rotating()

    async def test_agent_declarative_display_draws_on_the_page(self, manager, tmp_path) -> None:
        from boxbot.displays.spec import parse_spec, validate_spec

        spec = parse_spec({
            "name": "chores", "theme": "boxbot",
            "data_sources": [{"name": "c", "type": "static", "value": {"who": "<b>Erik</b>"}}],
            "layout": {"type": "column", "children": [
                {"type": "text", "content": "Dishes: {c.who}"},
            ]},
        })
        assert validate_spec(spec) == []
        manager.register_spec(spec)
        assert await manager.switch("chores")
        await wait_for(lambda: last(tmp_path).get("display") == "chores")
        state = last(tmp_path)
        assert state["frame"] is None
        # Agent text travels as a JSON string in the layout, never as markup
        # the surface builds: the pushed expression is ASCII JSON only.
        assert state["layout"]["children"][0]["content"] == "Dishes: <b>Erik</b>"


# ---------------------------------------------------------------------------
# 3e — the voice dock
# ---------------------------------------------------------------------------


def identity(label: str, certainty: str, name: str | None) -> dict[str, Any]:
    return {label: {
        "label": label, "certainty": certainty, "person_name": name,
        "person_id": "p_1" if name else None,
    }}


class TestDock:
    @pytest.mark.parametrize("label, certainty, name, shown", [
        ("Jacob", "confirmed", "Jacob", "Jacob"),
        ("Jacob?", "uncertain", "Jacob", None),
        ("Speaker A", "unknown", None, None),
    ])
    def test_only_a_confirmed_speaker_is_named(self, label, certainty, name, shown) -> None:
        event = SimpleNamespace(
            transcript=f"[{label}]: what time is it",
            speaker_identities=identity(label, certainty, name),
        )
        assert manager_mod._confirmed_speaker(event) == shown

    def test_no_identity_block_names_nobody(self) -> None:
        event = SimpleNamespace(transcript="what time is it", speaker_identities={})
        assert manager_mod._confirmed_speaker(event) is None

    def test_the_last_line_decides(self) -> None:
        event = SimpleNamespace(
            transcript="[Jacob]: ask her\n[Speaker A]: what time is it",
            speaker_identities={
                **identity("Jacob", "confirmed", "Jacob"),
                **identity("Speaker A", "unknown", None),
            },
        )
        assert manager_mod._confirmed_speaker(event) is None

    def test_a_name_is_not_carried_to_the_next_utterance(self) -> None:
        p = PresenceTracker()
        p.wake_word()
        p.speaker("Jacob")
        p.transcript("What time is it?", who="Jacob")
        p.speaking("Ten past four.")
        assert p.current()["who"] == "Jacob"
        p.speaking_done()
        p.draft("and tomorrow")
        assert "who" not in p.current()
        p.transcript("And tomorrow?", who=None)
        assert "who" not in p.current()
        p.speaking("Rain.")
        assert "who" not in p.current()

    async def test_uncertain_speaker_reaches_the_page_unnamed(self, manager, tmp_path) -> None:
        from boxbot.core.events import TranscriptReady, WakeWordHeard, get_event_bus

        bus = get_event_bus()
        await manager.switch("clock")
        await bus.publish(WakeWordHeard())
        await bus.publish(TranscriptReady(
            transcript="[Jacob?]: what time is it", raw_text="what time is it",
            speaker_identities=identity("Jacob?", "uncertain", "Jacob"),
        ))
        await wait_for(lambda: (last(tmp_path).get("presence") or {}).get("state") == "thinking")
        presence = last(tmp_path)["presence"]
        assert "who" not in presence and "Jacob" not in json.dumps(presence)

        await bus.publish(TranscriptReady(
            transcript="[Carina]: and tomorrow", raw_text="and tomorrow",
            speaker_identities=identity("Carina", "confirmed", "Carina"),
        ))
        await wait_for(lambda: (last(tmp_path).get("presence") or {}).get("who") == "Carina")

    async def test_listening_dock_goes_when_the_mic_closes(self, manager, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(manager_mod, "_TICK_SECONDS", 0.02)
        # The running tick loop already sleeps on the old cadence once.
        mic = {"open": True}
        manager.set_mic_probe(lambda: mic["open"])
        await manager.switch("clock")
        manager._presence.wake_word()
        manager._push_sky_state()
        await wait_for(lambda: (last(tmp_path).get("presence") or {}).get("state") == "listening")
        manager._watch_mic()
        assert manager._presence.is_listening()

        mic["open"] = False  # dormant, or mute_mic
        manager._watch_mic()
        assert manager._presence.is_listening()  # one closed tick is not enough
        manager._watch_mic()
        assert manager._presence.current() is None
        await wait_for(lambda: last(tmp_path).get("presence", 0) is None)

    async def test_a_thinking_dock_ignores_the_mic(self, manager) -> None:
        manager.set_mic_probe(lambda: False)
        manager._presence.wake_word()
        manager._presence.transcript("hello", who=None)
        for _ in range(4):
            manager._watch_mic()
        assert manager._presence.current()["state"] == "thinking"


# ---------------------------------------------------------------------------
# 3f — losing the browser
# ---------------------------------------------------------------------------


class TestFallback:
    async def test_failed_start_registers_no_sky_displays(self, tmp_path) -> None:
        mgr = DisplayManager(
            renderer="html", sky_config=make_config(tmp_path, browser="no-such-browser"),
        )
        await mgr.start(user_dir=tmp_path / "none", agent_dir=tmp_path / "none")
        try:
            assert mgr.renderer == "pil"
            assert "home" not in mgr.list_available()
            # ``picture`` is the PIL built-in, not the page's version.
            assert mgr.get_spec("picture").root_block.block_type == "image"
        finally:
            await mgr.stop()

    async def test_html_mode_registers_the_sky_displays(self, manager) -> None:
        assert "home" in manager.list_available()
        assert manager.get_spec("picture").root_block.block_type == "picture"

    async def test_browser_that_stays_dead_falls_back_to_pil(
        self, manager, tmp_path, monkeypatch,
    ) -> None:
        monkeypatch.setattr("boxbot.displays.sky_surface._RESTART_DELAYS", (0.02,))
        started = []

        async def start_screen() -> None:
            started.append(manager.get_current_frame() is not None)

        manager.set_renderer_fallback(start_screen)
        manager._web_view = view = FakeWebView()
        manager._rotation_displays = ["home", "clock"]
        assert await manager.switch("home")
        await wait_for(lambda: last(tmp_path).get("display") == "home")

        # The page stops loading and the browser dies: every relaunch fails.
        (tmp_path / "display" / "not-ready").touch()
        manager._sky._config.launch_timeout = 0.3
        os.killpg(manager._sky._proc.pid, signal.SIGKILL)
        await wait_for(lambda: bool(started), timeout=20)

        assert manager.renderer == "pil" and manager._sky is None
        assert started == [True]  # a frame was ready before the screen was asked for
        assert "home" not in manager.list_available()
        assert manager.get_active() == "clock"
        assert manager.get_spec("picture").root_block.block_type == "image"
        assert isinstance(manager.get_current_frame(), Image.Image)
        assert not manager._presence_subscribed and not view.is_open
        # And it keeps working as the PIL renderer.
        assert await manager.switch("notice", args={"title": "Still here", "lines": ["x"]})
        assert manager.get_active() == "notice"
        frame = await manager.capture_frame()
        assert isinstance(frame, Image.Image) and frame.size == (1024, 600)

    async def test_crash_loop_falls_back_and_keeps_a_pinned_display(
        self, manager, tmp_path, monkeypatch,
    ) -> None:
        from boxbot.displays import sky_surface

        monkeypatch.setattr(sky_surface, "_RESTART_DELAYS", (0.02,))
        monkeypatch.setattr(sky_surface, "_MAX_EXITS", 2)
        await manager.switch("notice", args={"title": "Dinner", "lines": ["at six"]})
        await wait_for(lambda: last(tmp_path).get("display") == "notice")
        surface = manager._sky
        for _ in range(2):
            await wait_for(lambda: surface.is_running)
            os.killpg(surface._proc.pid, signal.SIGKILL)
            await wait_for(lambda: not surface.is_running)
        await wait_for(lambda: manager.renderer == "pil" and manager.get_active() == "notice")
        assert manager.is_pinned()
        assert manager.get_active_args()["title"] == "Dinner"
        await wait_for(lambda: manager.get_current_frame() is not None)

    async def test_one_crash_is_survived_on_html(self, manager, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("boxbot.displays.sky_surface._RESTART_DELAYS", (0.02,))
        await manager.switch("clock")
        await wait_for(lambda: len(applied(tmp_path)) >= 1)
        os.killpg(manager._sky._proc.pid, signal.SIGKILL)
        await wait_for(lambda: len(applied(tmp_path)) >= 2)
        assert manager.renderer == "html" and manager._sky.is_running


# ---------------------------------------------------------------------------
# Security review: image paths, crash/hang, environment, preview race
# ---------------------------------------------------------------------------


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """A workspace that is attach-allowed and a file beside it that is not."""
    from boxbot.core import paths
    from boxbot.tools import _sandbox_actions as actions

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    allowed = workspace / "ok.png"
    Image.new("RGB", (8, 8), (200, 10, 10)).save(allowed)
    secret = tmp_path / "perception" / "crop.png"
    secret.parent.mkdir()
    Image.new("RGB", (8, 8), (10, 200, 10)).save(secret)
    monkeypatch.setattr(paths, "WORKSPACE_DIR", workspace)
    monkeypatch.setattr(
        actions, "_is_attach_allowed",
        lambda p: str(p.resolve()).startswith(str(workspace.resolve())),
    )
    return SimpleNamespace(allowed=str(allowed.resolve()), secret=str(secret.resolve()))


class TestImagePaths:
    async def test_unvetted_paths_never_reach_the_page(self, manager, tmp_path, roots) -> None:
        from boxbot.displays.spec import parse_spec

        spec = parse_spec({"name": "leak", "layout": {"type": "zones", "children": [
            {"type": "image", "source": roots.secret},
            {"type": "image", "source": "file://" + roots.secret},
            {"type": "image", "source": roots.allowed},
            {"type": "image", "source": "file://" + roots.allowed},
            {"type": "capture", "image": roots.secret},
            {"type": "capture", "image": {"src": roots.secret, "thumb": roots.allowed}},
            {"type": "picture", "photos": [
                roots.secret, roots.allowed,
                {"id": "x", "src": roots.secret, "thumb": "file://" + roots.secret},
                {"id": "y", "src": roots.allowed},
            ]},
            {"type": "text", "content": roots.secret},  # text is text
        ]}})
        manager.register_spec(spec)
        assert await manager.switch("leak")
        await wait_for(lambda: last(tmp_path).get("display") == "leak")
        state = last(tmp_path)
        kids = state["layout"]["children"]
        assert [k.get("source") for k in kids[:4]] == [None, None, roots.allowed, roots.allowed]
        assert kids[4]["image"] is None
        assert kids[5]["image"] == {"src": None, "thumb": roots.allowed}
        assert kids[6]["photos"] == [
            roots.allowed, {"id": "x", "src": None, "thumb": None},
            {"id": "y", "src": roots.allowed},
        ]
        assert state["images"] == [roots.allowed]
        without_text = json.dumps([kids[:7], state["images"]])
        assert roots.secret not in without_text

    async def test_photo_refs_are_listed_and_bad_ones_dropped(
        self, manager, tmp_path, roots, monkeypatch,
    ) -> None:
        await manager.switch("picture", args={
            "image_ids": ["ok.png", roots.secret], "interval": 60,
        })
        await wait_for(lambda: last(tmp_path).get("display") == "picture")
        state = last(tmp_path)
        assert state["images"] == [roots.allowed]
        assert [p["src"] for p in state["layout"]["photos"]] == [roots.allowed]

    def test_pil_renderer_opens_only_vetted_files(self, roots) -> None:
        from boxbot.displays import renderer

        assert renderer._try_load_image(roots.secret) is None
        assert renderer._try_load_image("relative/ok.png") is None
        assert renderer._try_load_image(roots.allowed) is not None
        # A path the manager resolved from a ``photo:`` ref is trusted.
        renderer.trust_image_paths([roots.secret])
        try:
            assert renderer._try_load_image(roots.secret) is not None
        finally:
            renderer._TRUSTED_IMAGE_PATHS.clear()

    def test_pil_preview_of_an_unvetted_path_draws_no_pixels_of_it(self, roots, tmp_path) -> None:
        from boxbot.displays.spec import parse_spec

        mgr = DisplayManager()
        spec = parse_spec({"name": "leak", "layout": {
            "type": "image", "source": roots.secret, "fit": "fill",
        }})
        image = mgr.render_preview(spec=spec)
        colours = set(image.convert("RGB").resize((16, 16)).getdata())
        assert (10, 200, 10) not in colours
        ok = parse_spec({"name": "fine", "layout": {
            "type": "image", "source": roots.allowed, "fit": "fill",
        }})
        assert (200, 10, 10) in set(mgr.render_preview(spec=ok).convert("RGB").resize((16, 16)).getdata())


class TestBrowserHealth:
    async def test_environment_carries_no_secrets(self, manager, tmp_path, monkeypatch) -> None:
        env = json.loads((tmp_path / "display" / "env.json").read_text())
        assert set(env) <= {"PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TZ",
                            "XDG_RUNTIME_DIR", "WAYLAND_DISPLAY"}
        assert env["HOME"] == str(tmp_path / "display" / "profile-screen")

    async def test_secrets_in_the_service_environment_stay_there(self, tmp_path, monkeypatch) -> None:
        from boxbot.displays.sky_surface import SkySurface

        for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "ELEVENLABS_API_KEY",
                     "SIGNAL_PIN", "WHATSAPP_ACCESS_TOKEN", "BOXBOT_ANYTHING"):
            monkeypatch.setenv(name, "s3cret-value")
        surface = SkySurface(make_config(tmp_path))
        await surface.start()
        try:
            env = json.loads((tmp_path / "display" / "env.json").read_text())
            argv = json.loads((tmp_path / "display" / "argv.json").read_text())
        finally:
            await surface.stop()
        assert "s3cret-value" not in json.dumps(env)
        assert not [k for k in env if "KEY" in k or "TOKEN" in k or "PIN" in k or "SECRET" in k]
        assert "--deny-permission-prompts" in argv
        assert "--force-webrtc-ip-handling-policy=disable_non_proxied_udp" in argv
        assert not [a for a in argv if "no-sandbox" in a or "disable-web-security" in a
                    or a.startswith("--remote-debugging-port")]

    async def test_a_crashed_page_is_relaunched_and_gets_the_state_again(
        self, tmp_path, monkeypatch,
    ) -> None:
        from boxbot.displays import sky_surface

        monkeypatch.setattr(sky_surface, "_RESTART_DELAYS", (0.02,))
        monkeypatch.setattr(sky_surface, "_PUSH_TIMEOUT", 0.5)
        surface = sky_surface.SkySurface(make_config(tmp_path))
        await surface.start()
        try:
            first = surface._proc.pid
            (tmp_path / "display" / "crash-page").touch()
            surface.submit({"n": 1})  # answered with Inspector.targetCrashed
            await wait_for(lambda: surface._proc is not None and surface._proc.pid != first
                           and surface.is_running)
            await wait_for(lambda: surface.pushed >= 1)
            assert applied(tmp_path)[-1] == {"n": 1}
        finally:
            await surface.stop()

    async def test_a_hung_page_is_killed_after_repeated_timeouts(self, tmp_path, monkeypatch) -> None:
        from boxbot.displays import sky_surface

        monkeypatch.setattr(sky_surface, "_RESTART_DELAYS", (0.02,))
        monkeypatch.setattr(sky_surface, "_PUSH_TIMEOUT", 0.1)
        monkeypatch.setattr(sky_surface, "_MAX_PUSH_FAILURES", 2)
        original_sleep = asyncio.sleep
        monkeypatch.setattr(sky_surface.asyncio, "sleep",
                            lambda d, *a: original_sleep(min(d, 0.05), *a))
        surface = sky_surface.SkySurface(make_config(tmp_path))
        await surface.start()
        try:
            first = surface._proc.pid
            hang = tmp_path / "display" / "hang"
            hang.touch()
            surface.submit({"n": 2})
            await wait_for(lambda: surface._proc is not None and surface._proc.pid != first)
            hang.unlink()  # the relaunched page answers
            await wait_for(lambda: surface.pushed >= 1, timeout=10)
            assert surface.is_running
        finally:
            await surface.stop()

    async def test_a_page_that_stays_hung_ends_in_the_pil_fallback(
        self, manager, tmp_path, monkeypatch,
    ) -> None:
        from boxbot.displays import sky_surface

        monkeypatch.setattr(sky_surface, "_RESTART_DELAYS", (0.02,))
        monkeypatch.setattr(sky_surface, "_PUSH_TIMEOUT", 0.1)
        monkeypatch.setattr(sky_surface, "_MAX_PUSH_FAILURES", 1)
        monkeypatch.setattr(sky_surface, "_MAX_EXITS", 3)
        original_sleep = asyncio.sleep
        monkeypatch.setattr(sky_surface.asyncio, "sleep",
                            lambda d, *a: original_sleep(min(d, 0.05), *a))
        await manager.switch("clock")
        await wait_for(lambda: last(tmp_path).get("display") == "clock")
        (tmp_path / "display" / "hang").touch()
        await manager.switch("notice", args={"title": "Hi", "lines": ["x"]})
        await wait_for(lambda: manager.renderer == "pil", timeout=20)
        await wait_for(lambda: manager.get_current_frame() is not None)
        assert manager.get_active() == "notice"

    async def test_a_state_the_page_rejects_is_not_retried_or_fatal(self, tmp_path, monkeypatch) -> None:
        from boxbot.displays import sky_surface

        surface = sky_surface.SkySurface(make_config(tmp_path))
        await surface.start()
        try:
            pid = surface._proc.pid
            surface.submit({"layout": "y" * 900_000})  # too large for one message
            await asyncio.sleep(0.3)
            assert surface.is_running and surface._proc.pid == pid
            surface.submit({"n": 3})
            await wait_for(lambda: surface.pushed >= 1)
        finally:
            await surface.stop()


class TestPreviewRace:
    async def test_a_draft_is_never_registered_while_it_renders(
        self, manager, monkeypatch, tmp_path,
    ) -> None:
        from boxbot.tools import _sandbox_actions as actions

        monkeypatch.setattr(manager_mod, "get_display_manager", lambda: manager)
        monkeypatch.setattr(actions, "_previews_dir", lambda: tmp_path)
        await manager.switch("clock")
        seen: list[Any] = []
        original = manager.render_preview_image

        async def slow(*args: Any, **kwargs: Any) -> Any:
            # What a tick or a switch would find mid-render.
            seen.append((manager.get_spec("draft"), manager.get_spec("notice")))
            return await original(*args, **kwargs)

        monkeypatch.setattr(manager, "render_preview_image", slow)
        notice = manager.get_spec("notice")
        ctx = SimpleNamespace(action_log=[], image_attachments=[])
        for name in ("draft", "notice"):  # a new name, and one that shadows a saved display
            result = await actions.process_action({
                "_sdk": "display.preview", "_expects_response": True,
                "spec": {"name": name, "layout": {"type": "column", "children": [
                    {"type": "text", "content": "unsaved"},
                ]}},
            }, ctx)
            assert result["status"] == "ok", result
        assert seen == [(None, notice), (None, notice)]
        assert manager.get_spec("draft") is None and manager.get_spec("notice") is notice
        assert len(ctx.image_attachments) == 2
        previews = [s for s in applied(tmp_path) if s["display"] in ("draft", "notice")]
        assert all("unsaved" in json.dumps(s["layout"]) for s in previews) and len(previews) == 2

    async def test_pil_draft_preview_is_not_registered_either(self, monkeypatch, tmp_path) -> None:
        from boxbot.tools import _sandbox_actions as actions

        mgr = DisplayManager()
        await mgr.load_displays(user_dir=tmp_path / "none", agent_dir=tmp_path / "none")
        monkeypatch.setattr(manager_mod, "get_display_manager", lambda: mgr)
        monkeypatch.setattr(actions, "_previews_dir", lambda: tmp_path)
        ctx = SimpleNamespace(action_log=[], image_attachments=[])
        result = await actions.process_action({
            "_sdk": "display.preview", "_expects_response": True,
            "spec": {"name": "draft", "layout": {"type": "text", "content": "unsaved"}},
        }, ctx)
        assert result["status"] == "ok", result
        assert mgr.get_spec("draft") is None
