"""Tests for the web display: URL policy, the kiosk browser lifecycle
(against a fake browser speaking DevTools-over-pipe), and the display
manager's handling of the reserved ``web`` display."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from boxbot.core.config import WebViewConfig
from boxbot.displays.manager import DisplayManager
from boxbot.displays.web_view import (
    EMBED_REFERRER,
    WebTarget,
    WebView,
    WebViewError,
    normalize_url,
)

FAKE_BROWSER = str(Path(__file__).parent / "fake_kiosk_browser.py")


# ---------------------------------------------------------------------------
# URL policy
# ---------------------------------------------------------------------------


class TestNormalizeUrl:
    @pytest.mark.parametrize("url", [
        "https://example.com/",
        "https://en.wikipedia.org/wiki/Otter?x=1#top",
        "https://sub.domain.co.uk:443/path",
        "  https://example.com/padded  ",
    ])
    def test_public_https_pages_pass(self, url):
        target = normalize_url(url)
        assert target.kind == "page"
        assert target.url == url.strip()
        assert target.referrer is None

    @pytest.mark.parametrize("url", [
        "",
        None,
        42,
        "example.com",
        "http://example.com/",
        "file:///etc/passwd",
        "chrome://settings",
        "javascript:alert(1)",
        "data:text/html,<h1>x</h1>",
        "https://",
        "https://localhost/",
        "https://router/",
        "https://homeassistant.local:8123/",
        "https://box.lan/",
        "https://127.0.0.1/",
        "https://192.168.0.10/",
        "https://[::1]/",
        "https://2130706433/",
        "https://0x7f.0.0.1/",
        "https://127.1/",
        "https://example.com:8443/",
        "https://user:pw@example.com/",
        "https://user@example.com/",
        "https://exa mple.com/",
        "https://example.com/\npath",
        "https://example.com/" + "a" * 3000,
        "https://bad_host.example.com/",
    ])
    def test_everything_else_is_rejected(self, url):
        with pytest.raises(WebViewError):
            normalize_url(url)

    @pytest.mark.parametrize("url", [
        "https://www.youtube.com/watch?v=jNQXAC9IVRw",
        "https://youtube.com/watch?v=jNQXAC9IVRw&feature=share",
        "https://m.youtube.com/watch?v=jNQXAC9IVRw",
        "https://youtu.be/jNQXAC9IVRw",
        "https://www.youtube.com/shorts/jNQXAC9IVRw",
        "https://www.youtube.com/embed/jNQXAC9IVRw",
        "https://www.youtube.com/live/jNQXAC9IVRw?si=abc",
    ])
    def test_youtube_links_become_the_embed_player(self, url):
        target = normalize_url(url)
        assert target.kind == "youtube"
        assert target.url == (
            "https://www.youtube-nocookie.com/embed/jNQXAC9IVRw?autoplay=1&rel=0"
        )
        assert target.referrer == EMBED_REFERRER

    @pytest.mark.parametrize("t, seconds", [
        ("90", 90), ("90s", 90), ("1m30s", 90), ("1h2m3s", 3723),
    ])
    def test_youtube_start_time_is_kept(self, t, seconds):
        target = normalize_url(f"https://youtu.be/jNQXAC9IVRw?t={t}")
        assert target.url.endswith(f"&start={seconds}")

    def test_youtube_playlist_is_kept(self):
        target = normalize_url(
            "https://www.youtube.com/watch?v=jNQXAC9IVRw&list=PL12345abc"
        )
        assert target.url.endswith("&list=PL12345abc")

    @pytest.mark.parametrize("url", [
        "https://www.youtube.com/",
        "https://www.youtube.com/results?search_query=otters",
        "https://www.youtube.com/watch?v=bad id!",
        "https://www.youtube.com/@somechannel",
    ])
    def test_other_youtube_pages_stay_plain_pages(self, url):
        try:
            target = normalize_url(url)
        except WebViewError:
            return  # malformed id with a space — rejected outright
        assert target.kind == "page"


# ---------------------------------------------------------------------------
# Browser lifecycle
# ---------------------------------------------------------------------------


def make_config(tmp_path: Path, **overrides: Any) -> WebViewConfig:
    values: dict[str, Any] = {
        "browser": FAKE_BROWSER,
        "privilege_drop": "none",
        "state_dir": str(tmp_path / "kiosk"),
        "launch_timeout": 5,
        "load_timeout": 5,
    }
    values.update(overrides)
    return WebViewConfig(**values)


def commands(tmp_path: Path) -> list[dict[str, Any]]:
    log = tmp_path / "kiosk" / "commands.log"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


def evaluated(tmp_path: Path) -> list[str]:
    return [
        c["params"]["expression"] for c in commands(tmp_path)
        if c["method"] == "Runtime.evaluate"
    ]


async def wait_for(predicate, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.02)


class TestWebView:
    async def test_open_navigates_and_reports_state(self, tmp_path):
        view = WebView(make_config(tmp_path), listening_probe=lambda: False)
        try:
            state = await view.open(normalize_url("https://youtu.be/jNQXAC9IVRw"))
            assert view.is_open
            assert state["kind"] == "youtube"
            assert state["title"] == "Fake Page"
            assert state["paused"] is False

            navigate = next(
                c for c in commands(tmp_path) if c["method"] == "Page.navigate"
            )
            assert navigate["params"]["referrer"] == EMBED_REFERRER
            assert "youtube-nocookie.com/embed/jNQXAC9IVRw" in navigate["params"]["url"]
            assert {"method": "Browser.setDownloadBehavior",
                    "params": {"behavior": "deny"}} in commands(tmp_path)
        finally:
            await view.close()
        assert not view.is_open
        assert view.state() is None

    async def test_second_open_reuses_the_running_browser(self, tmp_path):
        view = WebView(make_config(tmp_path), listening_probe=lambda: False)
        try:
            await view.open(normalize_url("https://example.com/a"))
            pid = view._proc.pid
            await view.open(normalize_url("https://example.com/b"))
            assert view._proc.pid == pid
            assert view.state()["url"] == "https://example.com/b"
        finally:
            await view.close()

    async def test_profile_is_wiped_on_every_launch(self, tmp_path):
        view = WebView(make_config(tmp_path), listening_probe=lambda: False)
        leftover = tmp_path / "kiosk" / "profile" / "Cookies"
        try:
            await view.open(normalize_url("https://example.com/"))
            leftover.write_text("session")
            await view.close()
            await view.open(normalize_url("https://example.com/"))
            assert not leftover.exists()
        finally:
            await view.close()

    async def test_media_follows_the_room(self, tmp_path):
        listening = False
        view = WebView(make_config(tmp_path), listening_probe=lambda: listening)
        try:
            await view.open(normalize_url("https://example.com/"))
            assert view.state()["paused"] is False

            # Wake word: pause at once, before the session reports ACTIVE.
            await view.pause_now(grace=0.3)
            assert view.state()["paused"] is True
            assert any(".pause()" in e for e in evaluated(tmp_path))

            # The session is now listening; the pause outlives the grace.
            listening = True
            await asyncio.sleep(0.7)
            assert view.state()["paused"] is True

            # Mic closes (muted / dormant / ended): playback resumes.
            listening = False
            await wait_for(lambda: view.state()["paused"] is False)
            assert any(".play()" in e for e in evaluated(tmp_path))
        finally:
            await view.close()

    async def test_media_pauses_while_bb_speaks(self, tmp_path):
        view = WebView(make_config(tmp_path), listening_probe=lambda: False)
        try:
            await view.open(normalize_url("https://example.com/"))
            await view.set_speaking(True)
            assert view.state()["paused"] is True
            await asyncio.sleep(0.6)
            assert view.state()["paused"] is True
            await view.set_speaking(False)
            await wait_for(lambda: view.state()["paused"] is False)
        finally:
            await view.close()

    async def _open_with_fast_idle(self, tmp_path, monkeypatch, **flags):
        """Open a view whose idle clock runs in fractions of a second."""
        from boxbot.displays import web_view as module

        monkeypatch.setattr(module, "_ACTIVITY_INTERVAL", 0.1)
        reasons: list[str] = []

        async def on_exit(reason: str) -> None:
            reasons.append(reason)

        listening = {"value": False}
        view = WebView(
            make_config(tmp_path), on_exit=on_exit,
            listening_probe=lambda: listening["value"],
        )
        (tmp_path / "kiosk").mkdir(exist_ok=True)
        for name, on in flags.items():
            if on:
                (tmp_path / "kiosk" / name).touch()
        await view.open(normalize_url("https://example.com/"))
        view._idle_seconds = 0.5
        return view, reasons, listening

    async def test_idle_page_closes_itself(self, tmp_path, monkeypatch):
        view, reasons, _ = await self._open_with_fast_idle(tmp_path, monkeypatch)
        await wait_for(lambda: bool(reasons))
        assert "idle" in reasons[0]
        assert not view.is_open

    async def test_playing_media_keeps_the_view_open(self, tmp_path, monkeypatch):
        view, reasons, _ = await self._open_with_fast_idle(
            tmp_path, monkeypatch, playing=True,
        )
        try:
            await asyncio.sleep(1.2)
            assert view.is_open and reasons == []
            assert view.state()["playing"] is True

            # The video ends: the idle clock starts, and the view goes away.
            (tmp_path / "kiosk" / "playing").unlink()
            await wait_for(lambda: bool(reasons))
            assert not view.is_open
        finally:
            await view.close()

    async def test_touching_the_page_keeps_the_view_open(self, tmp_path, monkeypatch):
        view, reasons, _ = await self._open_with_fast_idle(
            tmp_path, monkeypatch, touched=True,
        )
        try:
            await asyncio.sleep(1.2)
            assert view.is_open and reasons == []
        finally:
            await view.close()

    async def test_a_conversation_does_not_run_the_idle_clock(
        self, tmp_path, monkeypatch,
    ):
        view, reasons, listening = await self._open_with_fast_idle(
            tmp_path, monkeypatch,
        )
        try:
            listening["value"] = True  # room is talking: playback held
            await asyncio.sleep(1.2)
            assert view.is_open and reasons == []
            assert view.state()["paused"] is True
        finally:
            await view.close()

    async def test_explicit_pause_holds_then_goes_idle(self, tmp_path, monkeypatch):
        view, reasons, _ = await self._open_with_fast_idle(
            tmp_path, monkeypatch, playing=True,
        )
        try:
            state = await view.media("pause")
            assert state["paused"] is True
            # Room is quiet, yet it stays paused — unlike the automatic pause.
            await asyncio.sleep(0.6)
            assert view.state()["paused"] is True
            # A paused video is not "playing": the page reports it stopped.
            (tmp_path / "kiosk" / "playing").unlink()
            await wait_for(lambda: bool(reasons))
            assert "idle" in reasons[0]
        finally:
            await view.close()

    async def test_explicit_play_resumes_and_starts_media(self, tmp_path):
        view = WebView(make_config(tmp_path), listening_probe=lambda: False)
        try:
            await view.open(normalize_url("https://example.com/"))
            await view.media("pause")
            state = await view.media("play")
            assert state["paused"] is False
            # Starts media whether or not this view was what paused it.
            assert any(
                "m.paused&&!m.ended" in e and ".play()" in e
                for e in evaluated(tmp_path)
            )
        finally:
            await view.close()

    async def test_media_rejects_unknown_actions_and_closed_views(self, tmp_path):
        view = WebView(make_config(tmp_path), listening_probe=lambda: False)
        with pytest.raises(WebViewError, match="not open"):
            await view.media("play")
        try:
            await view.open(normalize_url("https://example.com/"))
            with pytest.raises(WebViewError, match="'play' or 'pause'"):
                await view.media("seek")
        finally:
            await view.close()

    async def test_idle_minutes_override_is_capped(self, tmp_path):
        view = WebView(
            make_config(tmp_path, idle_minutes=5, max_minutes=60),
            listening_probe=lambda: False,
        )
        try:
            state = await view.open(normalize_url("https://example.com/"))
            assert state["closes_if_idle_seconds"] == 300
            state = await view.open(
                normalize_url("https://example.com/"), idle_minutes=30,
            )
            assert state["closes_if_idle_seconds"] == 1800
            state = await view.open(
                normalize_url("https://example.com/"), idle_minutes=600,
            )
            assert state["closes_if_idle_seconds"] == 3600
        finally:
            await view.close()

    async def test_no_audio_path_means_a_muted_browser(self, tmp_path):
        """Never the session's default sink: on the box that is the AEC
        reference output, not the speaker."""
        view = WebView(make_config(tmp_path))
        argv, _, _ = view._build_launch()
        assert "--mute-audio" in argv
        assert not any(a.startswith("PULSE_") for a in argv)
        assert argv[-1] == "about:blank"

        argv, _, _ = view._build_launch("boxbot_kiosk")
        assert "--mute-audio" not in argv
        assert "PULSE_SINK=boxbot_kiosk" in argv
        assert any(a.startswith("PULSE_SERVER=unix:") for a in argv)

    async def test_page_load_failure_closes_the_view(self, tmp_path):
        view = WebView(make_config(tmp_path), listening_probe=lambda: False)
        with pytest.raises(WebViewError, match="ERR_NAME_NOT_RESOLVED"):
            await view.open(normalize_url("https://fail.example.com/"))
        assert not view.is_open

    async def test_browser_crash_notifies_owner(self, tmp_path):
        reasons: list[str] = []

        async def on_exit(reason: str) -> None:
            reasons.append(reason)

        view = WebView(
            make_config(tmp_path), on_exit=on_exit, listening_probe=lambda: False,
        )
        try:
            # The fake exits right after acknowledging this navigation.
            try:
                await view.open(normalize_url("https://crash.example.com/"))
            except WebViewError:
                return  # died before open() finished: reported as an error
            await wait_for(lambda: bool(reasons))
            assert "exited" in reasons[0]
            assert not view.is_open
        finally:
            await view.close()

    async def test_time_limit_closes_and_notifies(self, tmp_path, monkeypatch):
        reasons: list[str] = []

        async def on_exit(reason: str) -> None:
            reasons.append(reason)

        view = WebView(
            make_config(tmp_path), on_exit=on_exit, listening_probe=lambda: False,
        )

        async def short_limit() -> None:
            await asyncio.sleep(0.2)
            view._limit_task = None
            await view.close()
            await view._notify_exit("time limit reached")

        monkeypatch.setattr(view, "_enforce_limit", short_limit)
        await view.open(normalize_url("https://example.com/"))
        await wait_for(lambda: bool(reasons))
        assert reasons == ["time limit reached"]
        assert not view.is_open

    async def test_missing_browser_is_a_clear_error(self, tmp_path):
        view = WebView(make_config(tmp_path, browser="no-such-browser-xyz"))
        with pytest.raises(WebViewError, match="not installed"):
            await view.open(normalize_url("https://example.com/"))

    async def test_missing_kiosk_user_refuses_to_launch(self, tmp_path):
        """No fallback to the service user: fail closed."""
        view = WebView(make_config(
            tmp_path, privilege_drop="auto", user="no-such-user-xyz",
        ))
        with pytest.raises(WebViewError, match="does not exist"):
            await view.open(normalize_url("https://example.com/"))
        assert not view.is_open

    async def test_disabled_by_config(self, tmp_path):
        view = WebView(make_config(tmp_path, enabled=False))
        with pytest.raises(WebViewError, match="disabled"):
            await view.open(normalize_url("https://example.com/"))

    async def test_launch_drops_to_the_kiosk_user(self, tmp_path, monkeypatch):
        import os
        import pwd

        me = pwd.getpwuid(os.getuid()).pw_name
        view = WebView(make_config(tmp_path, privilege_drop="sudo", user=me))
        monkeypatch.setattr(WebView, "_grant_session_access", lambda *a: None)
        argv, _, _ = view._build_launch()
        assert argv[:5] == ["sudo", "-n", "-u", me, "--"]
        # A clean environment: nothing of the service's leaks through.
        assert argv[5:7] == ["env", "-i"]
        assert "--remote-debugging-pipe" in argv
        assert not any(a.startswith("--remote-debugging-port") for a in argv)


# ---------------------------------------------------------------------------
# Display manager
# ---------------------------------------------------------------------------


class FakeWebView:
    def __init__(self) -> None:
        self.opened: list[WebTarget] = []
        self.closed = 0
        self.fail: str | None = None
        self.is_open = False
        self.idle_minutes: float | None = None
        self.media_actions: list[str] = []

    async def open(
        self, target: WebTarget, *, idle_minutes: float | None = None,
    ) -> dict[str, Any]:
        if self.fail:
            raise WebViewError(self.fail)
        self.opened.append(target)
        self.idle_minutes = idle_minutes
        self.is_open = True
        return {}

    async def media(self, action: str) -> dict[str, Any]:
        self.media_actions.append(action)
        return self.state() or {}

    async def close(self) -> None:
        if self.is_open:
            self.closed += 1
        self.is_open = False

    def state(self) -> dict[str, Any] | None:
        if not self.is_open:
            return None
        return {"url": self.opened[-1].url, "kind": "page", "title": "",
                "paused": False, "open_seconds": 0}


@pytest.fixture
async def manager():
    mgr = DisplayManager()
    await mgr.load_displays()
    mgr._web_view = FakeWebView()
    yield mgr
    await mgr.stop()


class TestManagerWebDisplay:
    async def test_web_is_a_registered_display(self, manager):
        assert "web" in manager.list_available()
        assert "product" in manager.list_available()

    async def test_switch_opens_the_view_pinned(self, manager):
        ok = await manager.switch(
            "web", args={"url": "https://youtu.be/jNQXAC9IVRw"}, pin=False,
        )
        assert ok
        assert manager.get_active() == "web"
        assert manager.is_pinned()
        assert manager._web_view.opened[0].kind == "youtube"
        assert manager.get_active_args()["host"] == "www.youtube-nocookie.com"
        assert manager.get_web_state()["url"].startswith("https://www.youtube-nocookie")

    async def test_rejected_url_leaves_the_screen_alone(self, manager):
        await manager.switch("clock")
        ok = await manager.switch("web", args={"url": "http://192.168.0.1/"})
        assert not ok
        assert "https" in manager.last_switch_error
        assert manager.get_active() == "clock"
        assert manager._web_view.opened == []

    async def test_missing_url_is_an_error(self, manager):
        assert not await manager.switch("web")
        assert "url" in manager.last_switch_error

    async def test_switching_away_closes_the_view(self, manager):
        await manager.switch("web", args={"url": "https://example.com/"})
        await manager.switch("clock")
        assert manager._web_view.closed == 1
        assert manager.get_web_state() is None

    async def test_same_url_again_still_reaches_the_browser(self, manager):
        """Not swallowed by the same-display no-op: the browser may have
        been navigated away by touch, or have died."""
        await manager.switch("web", args={"url": "https://example.com/"})
        await manager.switch("web", args={"url": "https://example.com/"})
        assert len(manager._web_view.opened) == 2

    async def test_launch_failure_restores_a_normal_display(self, manager):
        await manager.switch("clock")
        manager._web_view.fail = "browser 'chromium' is not installed"
        ok = await manager.switch("web", args={"url": "https://example.com/"})
        assert not ok
        assert manager.last_switch_error == "browser 'chromium' is not installed"
        assert manager.get_active() != "web"
        assert not manager.is_pinned()

    async def test_unpin_closes_the_view(self, manager):
        await manager.switch("web", args={"url": "https://example.com/"})
        await manager.unpin()
        assert manager._web_view.closed == 1
        assert manager.get_active() != "web"
        assert not manager.is_pinned()

    async def test_browser_exit_hands_the_screen_back(self, manager):
        await manager.switch("web", args={"url": "https://example.com/"})
        manager._web_view.is_open = False
        await manager._on_web_view_exit("browser exited (code 1)")
        assert manager.get_active() != "web"
        assert not manager.is_pinned()

    async def test_idle_minutes_arg_reaches_the_view(self, manager):
        await manager.switch(
            "web", args={"url": "https://example.com/", "idle_minutes": 30},
        )
        assert manager._web_view.idle_minutes == 30
        assert manager.get_active_args()["idle_minutes"] == 30

    @pytest.mark.parametrize("bad", [0, -5, "10", True])
    async def test_bad_idle_minutes_is_rejected(self, manager, bad):
        ok = await manager.switch(
            "web", args={"url": "https://example.com/", "idle_minutes": bad},
        )
        assert not ok
        assert "idle_minutes" in manager.last_switch_error
        assert manager._web_view.opened == []

    async def test_web_media_reaches_the_view(self, manager):
        await manager.switch("web", args={"url": "https://example.com/"})
        await manager.web_media("pause")
        assert manager._web_view.media_actions == ["pause"]

    async def test_web_media_without_a_view_is_an_error(self):
        mgr = DisplayManager()
        with pytest.raises(WebViewError, match="not open"):
            await mgr.web_media("play")

    async def test_stop_closes_the_view(self):
        mgr = DisplayManager()
        await mgr.load_displays()
        mgr._web_view = FakeWebView()
        await mgr.switch("web", args={"url": "https://example.com/"})
        await mgr.stop()
        assert mgr._web_view.closed == 1


# ---------------------------------------------------------------------------
# Sound: null sink capture → speaker
# ---------------------------------------------------------------------------


class FakeSpeaker:
    def __init__(self) -> None:
        self.fed: list[bytes] = []

    def feed(self, audio_data: bytes, sample_rate: int = 48000, channels: int = 1) -> bool:
        assert (sample_rate, channels) == (48000, 1)
        self.fed.append(audio_data)
        return True


class FakeCapture:
    """Stands in for the pw-record process: a stdout that yields chunks."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.stdout = asyncio.StreamReader()
        for chunk in chunks:
            self.stdout.feed_data(chunk)
        self.stdout.feed_eof()
        self.returncode = None


class TestWebAudioBridge:
    LOUD = (b"\x10\x27" * 960)   # 20 ms of a constant 10000
    QUIET = bytes(1920)           # 20 ms of digital silence

    async def test_sound_reaches_the_speaker_and_silence_does_not(self):
        from boxbot.displays.web_audio import WebAudioBridge

        speaker = FakeSpeaker()
        bridge = WebAudioBridge(speaker, muted=lambda: False)
        await bridge._pump(FakeCapture([self.LOUD, self.QUIET, self.LOUD]))
        # Feeding silence would keep the speaker "busy" forever and TTS
        # would never see its queue drain.
        assert speaker.fed == [self.LOUD, self.LOUD]

    async def test_nothing_is_fed_while_muted(self):
        from boxbot.displays.web_audio import WebAudioBridge

        speaker = FakeSpeaker()
        bridge = WebAudioBridge(speaker, muted=lambda: True)
        await bridge._pump(FakeCapture([self.LOUD, self.LOUD]))
        assert speaker.fed == []

    async def test_unavailable_tools_mean_no_sound_not_a_crash(self, monkeypatch):
        from boxbot.displays import web_audio

        monkeypatch.setattr(web_audio.shutil, "which", lambda name: None)
        bridge = web_audio.WebAudioBridge(FakeSpeaker(), muted=lambda: False)
        assert await bridge.start() is False
        assert not bridge.running
        await bridge.stop()


class TestSpeakerFeed:
    """Speaker.feed: the live-source entry the web audio bridge uses."""

    @pytest.fixture
    def speaker(self):
        np = pytest.importorskip("numpy")  # noqa: F841
        from boxbot.hardware.speaker import Speaker

        spk = Speaker(sample_rate=24000, gain_db=6.0, aec_reference_device=None)
        spk._started = True
        return spk

    @staticmethod
    def frames(queue) -> int:
        return sum(block.shape[0] for block in queue)

    def test_first_chunk_gets_a_cushion_later_chunks_do_not(self, speaker):
        chunk = (b"\x10\x27" * 960)
        assert speaker.feed(chunk)
        first = self.frames(speaker._queue)
        assert first == 960 + 150 * 48           # audio + 150 ms cushion
        assert speaker.feed(chunk)
        assert self.frames(speaker._queue) == first + 960

    def test_audio_is_not_run_through_the_speech_limiter(self, speaker):
        import numpy as np

        speaker.feed(np.full(960, 10000, dtype=np.int16).tobytes())
        audio = np.concatenate([b.ravel() for b in speaker._queue])[150 * 48:]
        assert int(audio.max()) == 10000          # +6 dB stage would give ~18000

    def test_backlog_is_bounded(self, speaker):
        chunk = (b"\x10\x27" * 960)
        accepted = sum(speaker.feed(chunk) for _ in range(100))   # 2 s offered
        assert 0 < accepted < 40
        assert self.frames(speaker._queue) <= (500 + 150 + 20) * 48 + 960

    def test_reference_delay_is_added_once_per_run(self, speaker):
        import numpy as np

        speaker._aec_stream = object()           # pretend the AEC path is up
        speaker._aec_delay_samples = 320         # 20 ms at 16 kHz
        chunk = np.full(960, 10000, dtype=np.int16).tobytes()
        speaker.feed(chunk)
        first = self.frames(speaker._aec_queue)
        assert first == 320 + 150 * 16 + 320     # delay + cushion + 20 ms audio
        speaker.feed(chunk)
        assert self.frames(speaker._aec_queue) == first + 320

    def test_not_started_or_empty_is_dropped(self, speaker):
        assert speaker.feed(b"") is False
        speaker._started = False
        assert speaker.feed(b"\x10\x27" * 960) is False
