"""Tests for the web display: URL policy, the kiosk browser lifecycle
(against a fake browser speaking DevTools-over-pipe), and the display
manager's handling of the reserved ``web`` display."""

from __future__ import annotations

import asyncio
import json
import os
import stat
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

    async def test_nothing_the_page_controls_reaches_the_agent(self, tmp_path):
        """A page title is attacker-written text; the state the agent
        reads must be built only from what BB itself supplied."""
        view = WebView(make_config(tmp_path), listening_probe=lambda: False)
        try:
            state = await view.open(normalize_url("https://example.com/"))
            assert set(state) == {
                "url", "kind", "playing", "paused", "audio",
                "open_seconds", "closes_if_idle_seconds",
            }
            assert not any("title" in e for e in evaluated(tmp_path))
        finally:
            await view.close()

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

    async def test_whole_home_is_wiped_on_every_launch(self, tmp_path):
        view = WebView(make_config(tmp_path), listening_probe=lambda: False)
        home = tmp_path / "kiosk" / "home"
        cookies = home / "profile" / "Cookies"
        planted_ca = home / ".pki" / "nssdb" / "cert9.db"
        try:
            await view.open(normalize_url("https://example.com/"))
            cookies.write_text("session")
            planted_ca.parent.mkdir(parents=True)
            planted_ca.write_text("rogue CA")
            await view.close()
            await view.open(normalize_url("https://example.com/"))
            assert not cookies.exists()
            # Outside the profile: Chromium's certificate store.
            assert not planted_ca.exists()
        finally:
            await view.close()

    async def test_media_follows_the_room(self, tmp_path):
        listening = False
        view = WebView(make_config(tmp_path), listening_probe=lambda: listening)
        try:
            await view.open(normalize_url("https://example.com/"))
            assert view.state()["paused"] is False

            # Wake word: pause at once, before the session reports ACTIVE.
            view.pause_now(grace=0.3)
            assert view.state()["paused"] is True
            await wait_for(lambda: any(".pause()" in e for e in evaluated(tmp_path)))

            # The session is now listening; the pause outlives the grace.
            listening = True
            await asyncio.sleep(0.7)
            assert view.state()["paused"] is True

            # Mic closes (muted / dormant / ended): playback resumes.
            listening = False
            await wait_for(lambda: view.state()["paused"] is False)
            await wait_for(lambda: any(".play()" in e for e in evaluated(tmp_path)))
        finally:
            await view.close()

    async def test_media_pauses_while_bb_speaks(self, tmp_path):
        view = WebView(make_config(tmp_path), listening_probe=lambda: False)
        try:
            await view.open(normalize_url("https://example.com/"))
            view.set_speaking(True)
            assert view.state()["paused"] is True
            await asyncio.sleep(0.6)
            assert view.state()["paused"] is True
            view.set_speaking(False)
            await wait_for(lambda: view.state()["paused"] is False)
        finally:
            await view.close()

    async def test_speech_in_progress_survives_opening_a_page(self, tmp_path):
        """Another conversation opens a page while BB is mid-sentence in
        the room: the page must not start over the speech."""
        view = WebView(make_config(tmp_path), listening_probe=lambda: False)
        try:
            view.set_speaking(True)
            await view.open(normalize_url("https://example.com/"))
            await wait_for(lambda: view.state()["paused"] is True)
            view.set_speaking(False)
            await wait_for(lambda: view.state()["paused"] is False)
        finally:
            await view.close()

    async def test_a_lost_speaking_done_does_not_pause_forever(
        self, tmp_path, monkeypatch,
    ):
        from boxbot.displays import web_view as module

        monkeypatch.setattr(module, "_SPEAKING_MAX_SECONDS", 0.4)
        view = WebView(make_config(tmp_path), listening_probe=lambda: False)
        try:
            await view.open(normalize_url("https://example.com/"))
            view.set_speaking(True)           # ...and no "done" ever arrives
            assert view.state()["paused"] is True
            await wait_for(lambda: view.state()["paused"] is False)
        finally:
            await view.close()

    async def test_pausing_never_waits_on_the_browser(self, tmp_path):
        """pause_now / set_speaking run inside the voice pipeline's
        event publish; a hung renderer must not delay speech."""
        import time as _time

        view = WebView(make_config(tmp_path), listening_probe=lambda: False)
        try:
            await view.open(normalize_url("https://example.com/"))

            async def hung(expression: str) -> None:
                await asyncio.sleep(30)

            view._evaluate = hung  # type: ignore[method-assign]
            started = _time.monotonic()
            view.pause_now()
            view.set_speaking(True)
            assert _time.monotonic() - started < 0.1
            assert view.state()["paused"] is True
        finally:
            for task in list(view._background):
                task.cancel()
            await view.close()

    async def test_cancelled_open_leaves_no_browser_behind(self, tmp_path):
        """The wake word cancels the agent's tool call mid page-load. A
        half-open browser has no idle clock and no time limit."""
        view = WebView(make_config(tmp_path), listening_probe=lambda: False)
        task = asyncio.ensure_future(
            view.open(normalize_url("https://slow.example.com/"))
        )
        await wait_for(lambda: any(
            c["method"] == "Page.navigate" for c in commands(tmp_path)
        ))
        proc = view._proc
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not view.is_open
        assert view.state() is None
        assert proc.returncode is not None
        assert view._fds == [] and view._cdp is None

    async def test_second_cancel_cannot_cut_the_teardown_short(self, tmp_path):
        """Two wake words in a row: the lock must stay held until the
        teardown is really finished, or it would go on to close the
        browser a later open() launches."""
        view = WebView(make_config(tmp_path), listening_probe=lambda: False)
        real_teardown = view._teardown
        finished: list[bool] = []

        async def slow_teardown() -> None:
            await asyncio.sleep(0.4)
            await real_teardown()
            finished.append(True)

        view._teardown = slow_teardown  # type: ignore[method-assign]
        task = asyncio.ensure_future(
            view.open(normalize_url("https://slow.example.com/"))
        )
        await wait_for(lambda: any(
            c["method"] == "Page.navigate" for c in commands(tmp_path)
        ))
        task.cancel()
        await asyncio.sleep(0.1)
        task.cancel()                       # lands mid-teardown
        await asyncio.sleep(0.05)
        assert view._lock.locked() and not finished
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished == [True]
        assert not view._lock.locked() and not view.is_open

    async def test_stale_self_close_spares_a_newer_page(self, tmp_path):
        """Idle/limit decided to close, then someone opened a new page
        before the close got the lock: the new page stays."""
        reasons: list[str] = []

        async def on_exit(reason: str) -> None:
            reasons.append(reason)

        view = WebView(
            make_config(tmp_path), on_exit=on_exit, listening_probe=lambda: False,
        )
        try:
            await view.open(normalize_url("https://example.com/a"))
            stale = view._epoch
            await view.open(normalize_url("https://example.com/b"))
            assert await view._close_self(stale, "nothing playing — idle timeout") is False
            assert view.is_open and reasons == []
            assert await view._close_self(view._epoch, "time limit reached") is True
            assert not view.is_open and reasons == ["time limit reached"]
        finally:
            await view.close()

    async def test_open_after_a_crash_does_not_leak_the_old_handles(self, tmp_path):
        view = WebView(make_config(tmp_path), listening_probe=lambda: False)
        try:
            await view.open(normalize_url("https://example.com/"))
            old_fds = list(view._fds)
            # The browser dies and open() wins the race for the lock
            # against the exit watcher.
            async with view._lock:
                view._proc.kill()
                await view._proc.wait()
            await view.open(normalize_url("https://example.com/again"))
            assert view.is_open
            for fd in old_fds:
                if fd not in view._fds:
                    with pytest.raises(OSError):
                        os.fstat(fd)
            await asyncio.sleep(0.3)   # the old watcher must not close it
            assert view.is_open
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
            # The fake loads the page, then dies 0.3 s later.
            await view.open(normalize_url("https://crash.example.com/"))
            assert view.is_open
            await wait_for(lambda: bool(reasons))
            assert reasons == ["browser exited (code 7)"]
            assert not view.is_open
            assert view._fds == []
        finally:
            await view.close()

    async def test_time_limit_closes_and_notifies(self, tmp_path, monkeypatch):
        reasons: list[str] = []

        async def on_exit(reason: str) -> None:
            reasons.append(reason)

        # 0.005 min = 0.3 s; "playing" keeps the idle timeout out of it.
        (tmp_path / "kiosk").mkdir()
        (tmp_path / "kiosk" / "playing").touch()
        view = WebView(
            make_config(tmp_path, max_minutes=0.005), on_exit=on_exit,
            listening_probe=lambda: False,
        )
        await view.open(normalize_url("https://example.com/"))
        await wait_for(lambda: bool(reasons))
        assert reasons == ["time limit reached"]
        assert not view.is_open

    async def test_startup_failure_says_why(self, tmp_path):
        """"exit code 1" alone cost a deploy cycle: the browser's own
        last words belong in the error."""
        browser = tmp_path / "broken-browser"
        browser.write_text("#!/bin/sh\necho 'Missing X server or $DISPLAY' >&2\nexit 1\n")
        browser.chmod(0o755)
        view = WebView(make_config(tmp_path, browser=str(browser), launch_timeout=3))
        with pytest.raises(WebViewError, match="exit code 1.*Missing X server"):
            await view.open(normalize_url("https://example.com/"))
        assert not view.is_open

    async def test_browser_is_told_to_use_wayland(self, tmp_path):
        """It starts from an empty environment, so it cannot detect the
        session type itself."""
        argv, _, _ = WebView(make_config(tmp_path))._build_launch()
        assert "--ozone-platform=wayland" in argv
        assert argv[:2] == ["env", "-i"]

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

    async def test_no_network_fence_refuses_to_launch(self, tmp_path, monkeypatch):
        """Without the firewall rule a page could reach the LAN: the
        browser must not start at all."""
        import os
        import pwd
        import subprocess

        me = pwd.getpwuid(os.getuid()).pw_name
        calls: list[tuple[str, ...]] = []

        def as_kiosk(self, *argv: str):
            calls.append(argv)
            # pkill: nothing to kill. Probe: loopback was reachable.
            return subprocess.CompletedProcess(argv, 1 if argv[0] == "pkill" else 3, "", "")

        monkeypatch.setattr(WebView, "_as_kiosk", as_kiosk)
        granted: list[str] = []
        monkeypatch.setattr(
            WebView, "_grant_session_access",
            staticmethod(lambda *a: granted.append("x")),
        )
        view = WebView(make_config(tmp_path, privilege_drop="sudo", user=me))
        with pytest.raises(WebViewError, match="network fence"):
            await view.open(normalize_url("https://example.com/"))
        assert not view.is_open
        assert [c[0] for c in calls] == ["pkill", "python3"]
        assert granted == []          # no session access was handed out

    async def test_control_pipe_must_be_a_real_fifo(self, tmp_path):
        """A compromised browser swapping the FIFO for a link to one of
        our files must not get the service to write into it."""
        victim = tmp_path / "victim"
        victim.write_text("precious")
        link = tmp_path / "to_browser"
        link.symlink_to(victim)
        with pytest.raises(OSError):
            WebView._open_fifo(str(link))       # O_NOFOLLOW
        with pytest.raises(WebViewError, match="not a FIFO"):
            WebView._open_fifo(str(victim))
        assert victim.read_text() == "precious"

    async def test_full_control_pipe_is_an_error_not_a_hang(self, tmp_path):
        """A stalled browser stops reading. write() must fail, never
        block the event loop."""
        from boxbot.displays.cdp_pipe import CDPError, CDPPipe

        to_browser, from_browser = tmp_path / "to", tmp_path / "from"
        os.mkfifo(to_browser)
        os.mkfifo(from_browser)
        write_fd = WebView._open_fifo(str(to_browser))    # nobody drains it
        read_fd = WebView._open_fifo(str(from_browser))
        pipe = CDPPipe(write_fd, read_fd)
        pipe.start()
        try:
            errors: list[str] = []
            for _ in range(400):                          # > 64 KiB of commands
                try:
                    await pipe.call(
                        "Runtime.evaluate", {"expression": "x" * 500}, timeout=0.001,
                    )
                except CDPError as e:
                    errors.append(str(e))
            assert any("pipe is full" in e for e in errors)
        finally:
            pipe.close()
            os.close(write_fd)
            os.close(read_fd)

    async def test_disabled_by_config(self, tmp_path):
        view = WebView(make_config(tmp_path, enabled=False))
        with pytest.raises(WebViewError, match="disabled"):
            await view.open(normalize_url("https://example.com/"))

    async def test_launch_drops_to_the_kiosk_user(self, tmp_path, monkeypatch):
        import os
        import pwd

        me = pwd.getpwuid(os.getuid()).pw_name
        view = WebView(make_config(tmp_path, privilege_drop="sudo", user=me))
        # On the box setup-sandbox.sh creates these; the service does not.
        (tmp_path / "kiosk" / "ctl").mkdir(parents=True)
        argv, ctl_in, ctl_out = view._build_launch()
        assert stat.S_IMODE(os.stat(ctl_in).st_mode) == 0o660
        assert stat.S_IMODE(os.stat(ctl_out).st_mode) == 0o660
        assert "--block-new-web-contents" in argv
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
        self.holds_mic_mute = False
        self.pauses = 0
        self.speaking: list[bool] = []
        self.block_open: asyncio.Event | None = None

    def pause_now(self, grace: float = 2.0) -> None:
        self.pauses += 1

    def set_speaking(self, speaking: bool) -> None:
        self.speaking.append(speaking)

    async def open(
        self, target: WebTarget, *, idle_minutes: float | None = None,
    ) -> dict[str, Any]:
        if self.fail:
            raise WebViewError(self.fail)
        if self.block_open is not None:
            await self.block_open.wait()
        self.opened.append(target)
        self.idle_minutes = idle_minutes
        self.is_open = True
        self.holds_mic_mute = False
        return {}

    async def media(self, action: str) -> dict[str, Any]:
        self.media_actions.append(action)
        return self.state() or {}

    async def close(self) -> None:
        if self.is_open:
            self.closed += 1
        self.is_open = False
        self.holds_mic_mute = False

    def state(self) -> dict[str, Any] | None:
        if not self.is_open:
            return None
        return {"url": self.opened[-1].url, "kind": "page",
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

    async def test_cancelled_switch_does_not_strand_the_opening_card(self, manager):
        await manager.switch("clock")
        manager._web_view.block_open = asyncio.Event()
        task = asyncio.ensure_future(
            manager.switch("web", args={"url": "https://example.com/"})
        )
        for _ in range(50):
            await asyncio.sleep(0.01)
            if manager.get_active() == "web":
                break
        assert manager.get_active() == "web"       # card is up, browser loading
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert manager.get_active() != "web"
        assert not manager.is_pinned()

    async def test_falls_back_to_any_display_when_the_usual_ones_are_gone(self, manager):
        await manager.switch("web", args={"url": "https://example.com/"})
        for name in ("clock", "picture"):
            manager._specs.pop(name)
        manager._rotation_displays = []
        manager._web_view.is_open = False
        await manager._on_web_view_exit("browser exited (code 1)")
        assert manager.get_active() not in (None, "web")

    async def test_voice_events_reach_the_view_through_the_bus(self, manager):
        from boxbot.core.events import (
            AgentSpeaking, AgentSpeakingDone, WakeWordHeard, get_event_bus,
        )

        manager._subscribe_events()
        try:
            bus = get_event_bus()
            await bus.publish(WakeWordHeard(confidence=0.9))
            await bus.publish(AgentSpeaking(conversation_id="c", text="hi"))
            await bus.publish(AgentSpeakingDone(conversation_id="c"))
        finally:
            manager._unsubscribe_events()
        assert manager._web_view.pauses == 1
        assert manager._web_view.speaking == [True, False]

    async def test_mic_mute_hold_lasts_only_as_long_as_the_view(self, manager):
        manager.note_web_mic_muted()               # nothing open: ignored
        assert not manager.web_holds_mic_mute()
        await manager.switch("web", args={"url": "https://example.com/"})
        assert not manager.web_holds_mic_mute()    # open alone is not a hold
        manager.note_web_mic_muted()
        assert manager.web_holds_mic_mute()
        await manager.switch("clock")
        assert not manager.web_holds_mic_mute()

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

    def test_video_sound_is_not_bb_talking(self, speaker):
        """is_playing drives barge-in ("user spoke during TTS"); a wake
        word over a video must not be mistaken for one."""
        speaker._playback_lock = asyncio.Lock()
        speaker.feed(b"\x10\x27" * 960)
        assert len(speaker._queue) > 0
        assert speaker.is_playing is False

    def test_reference_stays_aligned_across_an_underrun(self, speaker):
        """After the audible queue runs dry the reference queue still
        holds a tail. Padding by the full delay again would push the
        reference a further delay behind the speaker every time."""
        import numpy as np

        speaker._aec_stream = object()
        speaker._aec_delay_samples = 320
        chunk = np.full(960, 10000, dtype=np.int16).tobytes()
        speaker.feed(chunk)
        speaker._queue.clear()                     # audible path underran
        tail = 400
        while self.frames(speaker._aec_queue) > tail:   # reference: tail left
            speaker._aec_queue.popleft()
        tail = self.frames(speaker._aec_queue)
        speaker.feed(chunk)
        # Audible: cushion then audio. Reference: must come out
        # cushion + delay after "now", i.e. padded by that minus the tail.
        assert self.frames(speaker._aec_queue) == (150 * 16 + 320) + 320
        assert tail > 0

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


# ---------------------------------------------------------------------------
# Mic handling around a playing view
# ---------------------------------------------------------------------------


class TestMicMuteForPlayback:
    """Opening a video on a voice conversation closes the mic, and BB's
    own speech must not reopen it while that view is up — but a mute
    placed for any other reason keeps the normal mute_mic contract."""

    class Session:
        def __init__(self, can_mute: bool = True) -> None:
            self.can_mute = can_mute
            self.muted = False

        def mute_mic(self) -> bool:
            self.muted = self.can_mute
            return self.can_mute

    class Conv:
        def __init__(self, channel: str) -> None:
            self.channel = channel

    @pytest.fixture
    async def wired(self, manager, monkeypatch):
        from boxbot.displays import manager as manager_module

        monkeypatch.setattr(manager_module, "_display_manager_instance", manager)
        await manager.switch("web", args={"url": "https://example.com/"})
        return manager

    def _patch(self, monkeypatch, session, conv) -> None:
        import boxbot.communication.voice as voice
        import boxbot.tools._tool_context as tool_context

        monkeypatch.setattr(voice, "get_voice_session", lambda: session)
        monkeypatch.setattr(tool_context, "get_current_conversation", lambda: conv)

    async def test_voice_conversation_gets_muted_and_held(self, wired, monkeypatch):
        from boxbot.tools.builtins.switch_display import _mute_mic_for_playback

        session = self.Session()
        self._patch(monkeypatch, session, self.Conv("voice"))
        assert _mute_mic_for_playback() is True
        assert session.muted
        assert wired.web_holds_mic_mute()

    async def test_text_conversation_touches_nothing(self, wired, monkeypatch):
        from boxbot.tools.builtins.switch_display import _mute_mic_for_playback

        session = self.Session()
        self._patch(monkeypatch, session, self.Conv("whatsapp"))
        assert _mute_mic_for_playback() is False
        assert not session.muted
        assert not wired.web_holds_mic_mute()

    async def test_no_hold_when_the_mic_could_not_be_muted(self, wired, monkeypatch):
        from boxbot.tools.builtins.switch_display import _mute_mic_for_playback

        self._patch(monkeypatch, self.Session(can_mute=False), self.Conv("voice"))
        assert _mute_mic_for_playback() is False
        assert not wired.web_holds_mic_mute()

    async def test_tts_end_keeps_the_mute_only_while_the_view_holds_it(self, wired):
        from types import SimpleNamespace

        from boxbot.communication.voice import VoiceSession

        muted = SimpleNamespace(is_mic_muted=True)
        # A page is open, but this mute was the agent's own (mute_mic):
        # speech reopens the mic as documented.
        assert VoiceSession._web_view_holds_mute(muted) is False
        wired.note_web_mic_muted()
        assert VoiceSession._web_view_holds_mute(muted) is True
        assert VoiceSession._web_view_holds_mute(
            SimpleNamespace(is_mic_muted=False)
        ) is False
        await wired.switch("clock")
        assert VoiceSession._web_view_holds_mute(muted) is False
