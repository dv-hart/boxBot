"""Web view — a kiosk browser window over the display surface.

Backs the reserved ``web`` display: ``switch_display("web", args={"url":
...})`` puts a YouTube video, an image, or a web page on the screen. The
block renderer cannot do any of those (it draws PIL frames), so a
Chromium window is launched fullscreen on top of the pygame surface and
closed again when the display switches away.

Security model:

- **Separate OS user.** The browser runs as ``display.web.user``
  (default ``boxbot-kiosk``), not as the boxBot service user. Web
  content is the least-trusted thing the box renders; a browser escape
  must not land next to ``.env``. Launch refuses rather than fall back
  to the service user (``privilege_drop: none`` is the explicit dev
  override).
- **No listener.** Control is the DevTools protocol over a FIFO pair
  (``--remote-debugging-pipe``), never a debugging TCP port.
- **Throwaway profile.** The profile directory is wiped on every launch:
  no cookies, logins, or history persist.
- **URL policy.** ``https`` only, public hostnames only — see
  :func:`normalize_url`. Downloads are denied.

Lifetime: the view closes when the display switches away, on
``unpin``, when nothing has played or been touched for
``idle_minutes``, after ``max_minutes`` regardless, or if the browser
dies. In every self-initiated case the owner is told through
``on_exit`` so a normal display comes back.

Sound goes browser → PipeWire null sink → speaker HAL (see
:mod:`boxbot.displays.web_audio`), so it is echo-cancelled like speech.
BB still does not talk or listen over a video: **media plays only while
the room is quiet**, meaning the mic's utterance pipeline is closed and
BB is not speaking. The wake word pauses playback at once; a short poll
then follows the voice session (muted, dormant, or ended → resume), so
no state transition has to remember to emit an event.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import ipaddress
import json
import logging
import os
import pwd
import re
import shutil
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable
from urllib.parse import parse_qs, urlencode, urlsplit

from boxbot.displays.cdp_pipe import CDPError, CDPPipe
from boxbot.displays.web_audio import SINK_NAME, WebAudioBridge

if TYPE_CHECKING:
    from boxbot.core.config import WebViewConfig

logger = logging.getLogger(__name__)

MAX_URL_LENGTH = 2048

# YouTube's embed player only plays when the request carries a referrer
# from a site other than YouTube itself (verified against Chrome 142:
# none → stuck paused, youtube.com → "Error code: 152"). The view loads
# the player as a top-level page, which has no referrer, so one is set.
EMBED_REFERRER = "https://boxbot.invalid/"

_YOUTUBE_HOSTS = frozenset({
    "youtube.com", "www.youtube.com", "m.youtube.com",
    "music.youtube.com", "youtube-nocookie.com", "www.youtube-nocookie.com",
})
_YOUTUBE_ID = re.compile(r"^[A-Za-z0-9_-]{6,20}$")
_YOUTUBE_PATH_ID = re.compile(r"^/(?:embed|shorts|live|v)/([^/?#]+)")
_START_TIME = re.compile(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s?)?$")

# Names that never resolve on the public internet. The web view is for
# public pages; the LAN (router, Home Assistant, the box itself) is not
# something a URL handed to the agent should be able to reach.
_PRIVATE_SUFFIXES = (
    ".local", ".localhost", ".internal", ".lan", ".home", ".home.arpa",
    ".intranet", ".corp", ".test", ".invalid", ".example",
)

_PAUSE_JS = (
    "(()=>{let n=0;document.querySelectorAll('video,audio').forEach(m=>{"
    "if(!m.paused){m.dataset.bbHeld='1';m.pause();n++}});return n})()"
)
_RESUME_JS = (
    "(()=>{let n=0;document.querySelectorAll('[data-bb-held]').forEach(m=>{"
    "delete m.dataset.bbHeld;const p=m.play();if(p&&p.catch)p.catch(()=>{});"
    "n++});return n})()"
)

# Starts whatever media the page has, whether or not this view paused it.
_PLAY_JS = (
    "(()=>{let n=0;document.querySelectorAll('video,audio').forEach(m=>{"
    "delete m.dataset.bbHeld;if(m.paused&&!m.ended){const p=m.play();"
    "if(p&&p.catch)p.catch(()=>{});n++}});return n})()"
)
# Is anything playing, and how long since someone touched the page?
# Installs its input listeners on first run in each document.
_ACTIVITY_JS = (
    "(()=>{if(!window.__bbSeen){window.__bbSeen=Date.now();"
    "for(const t of['pointerdown','touchstart','keydown','wheel','scroll'])"
    "addEventListener(t,()=>{window.__bbSeen=Date.now()},"
    "{capture:true,passive:true})}"
    "const playing=[...document.querySelectorAll('video,audio')]"
    ".some(m=>!m.paused&&!m.ended&&m.readyState>2);"
    "return JSON.stringify({playing,quiet_ms:Date.now()-window.__bbSeen})})()"
)

# Runs as the kiosk user. Wipes the profile, makes a private runtime dir,
# then execs the browser with the DevTools pipe on fds 3 and 4. A shell
# redirect is the only portable way to hand a child those exact fd
# numbers across sudo, which closes every inherited fd above stderr.
_LAUNCH_SCRIPT = (
    'rm -rf "$1" && mkdir -p "$1" "$2" && chmod 700 "$1" "$2" && '
    'in="$3" out="$4" && shift 4 && exec "$@" 3<"$in" 4>"$out"'
)


# How often the view re-checks whether the room is quiet.
_SYNC_INTERVAL = 0.25
# How often it asks the page whether anything is playing.
_ACTIVITY_INTERVAL = 2.0


def room_is_listening() -> bool:
    """True while the voice session is capturing utterances.

    ACTIVE with an unmuted mic is the only state where room audio turns
    into transcripts. DORMANT, IDLE, and agent-muted all mean "wake word
    only", which playback does not disturb.
    """
    try:
        from boxbot.communication.voice import VoiceSessionState, get_voice_session
    except Exception:
        return False
    session = get_voice_session()
    if session is None:
        return False
    return session.state is VoiceSessionState.ACTIVE and not session.is_mic_muted


class WebViewError(Exception):
    """The web view could not show what was asked. Message is agent-facing."""


@dataclass(frozen=True)
class WebTarget:
    """A URL that passed policy, in the form the browser should load."""

    url: str
    kind: str  # "youtube" | "page"
    referrer: str | None = None


def normalize_url(raw: Any) -> WebTarget:
    """Validate ``raw`` against the web-view URL policy.

    Accepts ``https`` URLs on public hostnames. Rejects everything else:
    other schemes (``http``, ``file``, ``chrome``, ``javascript``,
    ``data``), IP-literal hosts, single-label and private-suffix names,
    embedded credentials, and non-default ports.

    YouTube watch / share / shorts links are rewritten to the
    privacy-enhanced embed player (``youtube-nocookie.com``), which
    fills the window with the video and nothing else.

    Raises:
        WebViewError: with a message naming what was wrong.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise WebViewError("args.url is required: an https:// URL")
    url = raw.strip()
    if len(url) > MAX_URL_LENGTH:
        raise WebViewError(f"url longer than {MAX_URL_LENGTH} characters")
    if any(ord(c) < 0x21 or ord(c) == 0x7F for c in url):
        raise WebViewError("url contains whitespace or control characters")

    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as e:
        raise WebViewError(f"url is not valid: {e}") from e

    if parts.scheme.lower() != "https":
        raise WebViewError("only https:// URLs can be shown")
    if parts.username is not None or parts.password is not None:
        raise WebViewError("url must not contain credentials")
    if port not in (None, 443):
        raise WebViewError("url must use the default https port")

    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        raise WebViewError("url has no hostname")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise WebViewError("url must use a public hostname, not an IP address")
    # Reject anything that is not a plain multi-label DNS name. This also
    # covers the numeric host forms browsers accept as IPv4 ("0x7f.1",
    # "2130706433"): a name whose last label is numeric is never a TLD.
    labels = host.split(".")
    if len(labels) < 2 or not all(
        re.fullmatch(r"[a-z0-9]([a-z0-9-]*[a-z0-9])?", label) for label in labels
    ):
        raise WebViewError("url must use a public hostname")
    if labels[-1].isdigit() or labels[-1].startswith("0x"):
        raise WebViewError("url must use a public hostname, not an IP address")
    if host == "localhost" or host.endswith(_PRIVATE_SUFFIXES):
        raise WebViewError("url must use a public hostname")

    youtube = _youtube_embed(host, parts.path, parts.query)
    if youtube is not None:
        return WebTarget(url=youtube, kind="youtube", referrer=EMBED_REFERRER)
    return WebTarget(url=url, kind="page")


def _youtube_embed(host: str, path: str, query: str) -> str | None:
    """Return the embed-player URL for a YouTube video link, else ``None``."""
    params = parse_qs(query)
    video_id: str | None = None
    if host == "youtu.be":
        video_id = path.strip("/").split("/")[0] or None
    elif host in _YOUTUBE_HOSTS:
        if path == "/watch":
            video_id = (params.get("v") or [None])[0]
        else:
            match = _YOUTUBE_PATH_ID.match(path)
            if match:
                video_id = match.group(1)
    if not video_id or not _YOUTUBE_ID.fullmatch(video_id):
        return None

    embed: dict[str, str] = {"autoplay": "1", "rel": "0"}
    start = _start_seconds((params.get("t") or params.get("start") or [""])[0])
    if start:
        embed["start"] = str(start)
    playlist = (params.get("list") or [""])[0]
    if playlist and re.fullmatch(r"[A-Za-z0-9_-]{2,64}", playlist):
        embed["list"] = playlist
    return f"https://www.youtube-nocookie.com/embed/{video_id}?{urlencode(embed)}"


def _start_seconds(value: str) -> int:
    """Parse a YouTube ``t=`` value (``90``, ``90s``, ``1m30s``) to seconds."""
    match = _START_TIME.fullmatch(value.strip().lower()) if value else None
    if not match or not any(match.groups()):
        return 0
    hours, minutes, seconds = (int(g) if g else 0 for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds


class WebView:
    """Owns the kiosk browser process and its DevTools connection.

    One instance per :class:`~boxbot.displays.manager.DisplayManager`.
    ``on_exit`` is awaited when the browser goes away for any reason
    other than :meth:`close` (crash, time limit) so the manager can put
    a normal display back.
    """

    def __init__(
        self,
        config: "WebViewConfig",
        *,
        size: tuple[int, int] = (1024, 600),
        on_exit: Callable[[str], Awaitable[None]] | None = None,
        listening_probe: Callable[[], bool] = room_is_listening,
        speaker: Any | None = None,
    ) -> None:
        self._config = config
        self._size = size
        self._on_exit = on_exit
        self._listening_probe = listening_probe
        # No speaker (dev, headless, tests): the view opens without sound.
        self._audio: WebAudioBridge | None = (
            WebAudioBridge(speaker, muted=lambda: self._paused)
            if speaker is not None else None
        )

        self._lock = asyncio.Lock()
        self._proc: asyncio.subprocess.Process | None = None
        self._cdp: CDPPipe | None = None
        self._fds: list[int] = []
        self._session_id: str | None = None
        self._watch_task: asyncio.Task[None] | None = None
        self._limit_task: asyncio.Task[None] | None = None
        self._sync_task: asyncio.Task[None] | None = None
        self._closing = False

        self._target: WebTarget | None = None
        self._title: str = ""
        self._opened_at: float = 0.0
        self._speaking = False
        self._pause_until = 0.0
        self._paused = False
        # Set by media("pause"): stays paused until media("play").
        self._user_paused = False
        self._playing = False
        self._idle_seconds = 0.0
        self._last_active = 0.0

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    def state(self) -> dict[str, Any] | None:
        """What the view is showing, or ``None`` when closed."""
        if not self.is_open or self._target is None:
            return None
        return {
            "url": self._target.url,
            "kind": self._target.kind,
            "title": self._title,
            "playing": self._playing and not self._paused,
            "paused": self._paused,
            "audio": self._audio is not None and self._audio.running,
            "open_seconds": int(time.monotonic() - self._opened_at),
            "closes_if_idle_seconds": int(self._idle_seconds),
        }

    # ------------------------------------------------------------------
    # Open / close
    # ------------------------------------------------------------------

    async def open(
        self, target: WebTarget, *, idle_minutes: float | None = None,
    ) -> dict[str, Any]:
        """Show ``target``, launching the browser if it is not running.

        ``idle_minutes`` overrides ``display.web.idle_minutes`` for this
        page (capped at ``max_minutes``) — a recipe someone is cooking
        from should outlast the default.

        Raises:
            WebViewError: the browser could not start or the page could
                not load. The view is closed when this is raised.
        """
        if not self._config.enabled:
            raise WebViewError("web view is disabled (display.web.enabled)")

        async with self._lock:
            try:
                if not self.is_open:
                    await self._launch()
                await self._navigate(target)
            except (WebViewError, CDPError, OSError) as e:
                await self._teardown()
                if isinstance(e, WebViewError):
                    raise
                raise WebViewError(f"web view failed: {e}") from e

            self._target = target
            self._opened_at = time.monotonic()
            # A fresh page starts playing; the sync loop pauses it again
            # within a tick if the room is not quiet.
            self._speaking = False
            self._pause_until = 0.0
            self._paused = False
            self._user_paused = False
            self._playing = False
            minutes = self._config.idle_minutes if idle_minutes is None else idle_minutes
            self._idle_seconds = 60.0 * max(1.0, min(minutes, self._config.max_minutes))
            self._last_active = time.monotonic()
            self._restart_limit_timer()
            if self._sync_task is None:
                self._sync_task = asyncio.create_task(
                    self._sync_loop(), name="web-view-sync",
                )
            state = self.state()
            return state if state is not None else {}

    async def close(self) -> None:
        """Close the browser. Safe to call when nothing is open."""
        async with self._lock:
            await self._teardown()

    async def screenshot(self) -> bytes:
        """PNG of what the browser window is showing right now."""
        async with self._lock:
            if not self.is_open or self._cdp is None:
                raise WebViewError("web view is not open")
            try:
                result = await self._cdp.call(
                    "Page.captureScreenshot", {"format": "png"},
                    session_id=self._session_id, timeout=15.0,
                )
            except CDPError as e:
                raise WebViewError(f"web view screenshot failed: {e}") from e
            return base64.b64decode(result.get("data", ""))

    # ------------------------------------------------------------------
    # Pause while the room is not quiet
    # ------------------------------------------------------------------

    async def pause_now(self, grace: float = 2.0) -> None:
        """Pause immediately (wake word heard).

        ``grace`` keeps the pause in force while the voice session
        catches up to ACTIVE; after that the sync loop decides.
        """
        self._pause_until = time.monotonic() + grace
        await self._set_paused(True)

    async def set_speaking(self, speaking: bool) -> None:
        """Track TTS playback: media stays paused while BB talks."""
        self._speaking = speaking
        if speaking:
            await self._set_paused(True)

    async def media(self, action: str) -> dict[str, Any]:
        """Explicit play / pause, as asked for by a person.

        ``pause`` holds until ``play`` — unlike the automatic pause,
        which lifts as soon as the room is quiet. ``play`` also starts
        media the page loaded without autoplaying.
        """
        if action not in ("play", "pause"):
            raise WebViewError("action must be 'play' or 'pause'")
        if not self.is_open:
            raise WebViewError("web view is not open")
        if action == "pause":
            self._user_paused = True
            await self._set_paused(True)
        else:
            self._user_paused = False
            self._pause_until = 0.0
            self._last_active = time.monotonic()
            if not self._wants_pause():
                self._paused = False
                await self._evaluate(_PLAY_JS)
        state = self.state()
        return state if state is not None else {}

    def _wants_pause(self) -> bool:
        if self._user_paused or self._speaking:
            return True
        if time.monotonic() < self._pause_until:
            return True
        try:
            return bool(self._listening_probe())
        except Exception:
            logger.debug("web view listening probe failed", exc_info=True)
            return False

    async def _set_paused(self, paused: bool) -> None:
        if not self.is_open or paused == self._paused:
            return
        self._paused = paused
        await self._evaluate(_PAUSE_JS if paused else _RESUME_JS)

    async def _sync_loop(self) -> None:
        """Keep playback in step with the room, and close when idle."""
        next_activity_check = 0.0
        while self.is_open:
            await asyncio.sleep(_SYNC_INTERVAL)
            # This loop is the only thing that pauses a video for the
            # room and the only thing that closes an idle page; one bad
            # tick must not end it.
            try:
                await self._set_paused(self._wants_pause())
                now = time.monotonic()
                if now < next_activity_check:
                    continue
                next_activity_check = now + _ACTIVITY_INTERVAL
                await self._check_activity()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("web view sync tick failed")
                continue
            if time.monotonic() - self._last_active > self._idle_seconds:
                break
        else:
            return

        logger.info("Web view idle for %.0fs; closing", self._idle_seconds)
        self._sync_task = None
        await self.close()
        await self._notify_exit("nothing playing — idle timeout")

    async def _check_activity(self) -> None:
        """Refresh ``_last_active``: something is playing, someone just
        touched the page, or BB itself is holding playback for the room
        (a conversation must not run the idle clock down)."""
        raw = await self._evaluate(_ACTIVITY_JS)
        try:
            activity = json.loads(raw) if isinstance(raw, str) else {}
        except ValueError:
            activity = {}
        if not isinstance(activity, dict):
            activity = {}
        self._playing = bool(activity.get("playing"))
        quiet_ms = activity.get("quiet_ms")
        touched = isinstance(quiet_ms, (int, float)) and (
            quiet_ms < _ACTIVITY_INTERVAL * 1500
        )
        held_for_room = self._paused and not self._user_paused
        if self._playing or touched or held_for_room:
            self._last_active = time.monotonic()

    async def _evaluate(self, expression: str) -> Any:
        """Run ``expression`` in the page. Best-effort: ``None`` on failure."""
        cdp = self._cdp
        if cdp is None:
            return None
        try:
            result = await cdp.call(
                "Runtime.evaluate",
                {"expression": expression, "returnByValue": True},
                session_id=self._session_id, timeout=5.0,
            )
        except CDPError as e:
            logger.debug("web view evaluate failed: %s", e)
            return None
        value = result.get("result")
        return value.get("value") if isinstance(value, dict) else None

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    async def _launch(self) -> None:
        # Before the browser starts, so its first sound has somewhere to go.
        audio_sink = (
            SINK_NAME if self._audio is not None and await self._audio.start()
            else None
        )
        argv, ctl_in, ctl_out = self._build_launch(audio_sink)

        # O_RDWR so neither side blocks in open() waiting for its peer:
        # a FIFO opened read-only or write-only waits for the other end.
        write_fd = os.open(ctl_in, os.O_RDWR | os.O_CLOEXEC)
        read_fd = os.open(ctl_out, os.O_RDWR | os.O_CLOEXEC)
        self._fds = [write_fd, read_fd]

        logger.info("Launching web view: %s", " ".join(argv[:6]) + " …")
        self._closing = False
        self._proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        self._cdp = CDPPipe(write_fd, read_fd)
        self._cdp.start()
        self._watch_task = asyncio.create_task(
            self._watch_exit(self._proc), name="web-view-exit",
        )

        timeout = float(self._config.launch_timeout)
        try:
            await self._cdp.call("Target.setDiscoverTargets", {"discover": True},
                                 timeout=timeout)
            targets = await self._cdp.call("Target.getTargets", timeout=timeout)
            page_id = next(
                (t.get("targetId") for t in targets.get("targetInfos", [])
                 if isinstance(t, dict) and t.get("type") == "page"),
                None,
            )
            if page_id is None:
                created = await self._cdp.call(
                    "Target.createTarget", {"url": "about:blank"}, timeout=timeout,
                )
                page_id = created.get("targetId")
            attached = await self._cdp.call(
                "Target.attachToTarget", {"targetId": page_id, "flatten": True},
                timeout=timeout,
            )
            self._session_id = attached.get("sessionId")
            if not self._session_id:
                raise CDPError("browser did not return a page session")
            await self._cdp.call("Page.enable", session_id=self._session_id)
            # Nothing the page offers may land on disk.
            await self._cdp.call("Browser.setDownloadBehavior", {"behavior": "deny"})
        except CDPError as e:
            if self._proc.returncode is not None:
                raise WebViewError(
                    "browser exited during startup "
                    f"(exit code {self._proc.returncode})"
                ) from e
            raise WebViewError(f"browser did not come up: {e}") from e

    def _build_launch(
        self, audio_sink: str | None = None,
    ) -> tuple[list[str], str, str]:
        """Assemble the launch argv and create the control FIFOs.

        ``audio_sink`` is the PipeWire sink the browser must play into
        (the web-audio bridge's null sink). Without one the browser gets
        no audio server at all, rather than whatever sink happens to be
        the session default.
        """
        cfg = self._config
        browser = shutil.which(cfg.browser) or (
            cfg.browser if os.path.isabs(cfg.browser) and os.access(cfg.browser, os.X_OK)
            else None
        )
        if browser is None:
            raise WebViewError(f"browser '{cfg.browser}' is not installed")

        drop = cfg.privilege_drop
        if drop == "auto":
            drop = "sudo"
        if drop == "sudo":
            try:
                pwd.getpwnam(cfg.user)
            except KeyError as e:
                raise WebViewError(
                    f"web view user '{cfg.user}' does not exist — run "
                    "scripts/setup-sandbox.sh on the box"
                ) from e

        state_dir = Path(cfg.state_dir)
        ctl_dir = state_dir / "ctl"
        ctl_in = ctl_dir / "to_browser"
        ctl_out = ctl_dir / "from_browser"
        try:
            ctl_dir.mkdir(parents=True, exist_ok=True)
            for fifo in (ctl_in, ctl_out):
                with contextlib.suppress(FileNotFoundError):
                    if not stat.S_ISFIFO(fifo.lstat().st_mode):
                        raise WebViewError(f"{fifo} exists and is not a FIFO")
                    fifo.unlink()
                # The directory's default ACL (setup-sandbox.sh) extends
                # these to the kiosk user; nobody else can open them.
                os.mkfifo(fifo, 0o660)
        except OSError as e:
            raise WebViewError(
                f"cannot prepare web view control dir {ctl_dir}: {e} — run "
                "scripts/setup-sandbox.sh on the box"
            ) from e

        runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
        wayland = os.environ.get("WAYLAND_DISPLAY") or "wayland-0"
        wayland_socket = wayland if os.path.isabs(wayland) else f"{runtime_dir}/{wayland}"
        pulse_socket = f"{runtime_dir}/pulse/native"
        if drop == "sudo":
            self._grant_session_access(cfg.user, runtime_dir, wayland_socket)

        width, height = self._size
        browser_argv = [
            browser,
            "--kiosk",
            "--remote-debugging-pipe",
            f"--user-data-dir={state_dir / 'profile'}",
            f"--window-size={width},{height}",
            "--window-position=0,0",
            "--ozone-platform-hint=auto",
            "--no-first-run",
            "--no-default-browser-check",
            "--noerrdialogs",
            "--disable-infobars",
            "--disable-extensions",
            "--disable-sync",
            "--disable-background-networking",
            "--disable-component-update",
            "--disable-session-crashed-bubble",
            "--disable-features=Translate,MediaRouter,PasswordManagerOnboarding",
            "--password-store=basic",
            "--autoplay-policy=no-user-gesture-required",
            "--overscroll-history-navigation=0",
            *cfg.extra_args,
            "about:blank",
        ]
        env = {
            "HOME": str(state_dir),
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "XDG_RUNTIME_DIR": str(state_dir / "run"),
            "WAYLAND_DISPLAY": wayland_socket,
            "LANG": os.environ.get("LANG", "C.UTF-8"),
        }
        if audio_sink is not None:
            env["PULSE_SERVER"] = f"unix:{pulse_socket}"
            env["PULSE_SINK"] = audio_sink
        else:
            browser_argv.insert(-1, "--mute-audio")
        argv = [
            "env", "-i", *(f"{k}={v}" for k, v in env.items()),
            "sh", "-c", _LAUNCH_SCRIPT, "boxbot-web-view",
            str(state_dir / "profile"), str(state_dir / "run"),
            str(ctl_in), str(ctl_out),
            *browser_argv,
        ]
        if drop == "sudo":
            argv = ["sudo", "-n", "-u", cfg.user, "--", *argv]
        return argv, str(ctl_in), str(ctl_out)

    @staticmethod
    def _grant_session_access(user: str, runtime_dir: str, wayland_socket: str) -> None:
        """Let the kiosk user reach this session's compositor and audio.

        The runtime dir is a tmpfs recreated at every login, so the ACLs
        cannot be installed once by the setup script. The service user
        owns these paths, so no privilege is needed to grant them.
        """
        import subprocess

        grants = [
            (runtime_dir, "x", True),
            (wayland_socket, "rw", True),
            (f"{runtime_dir}/pulse", "x", False),
            (f"{runtime_dir}/pulse/native", "rw", False),
        ]
        for path, perms, required in grants:
            if not os.path.exists(path):
                if required:
                    raise WebViewError(f"{path} not found — is the desktop session up?")
                continue
            result = subprocess.run(
                ["setfacl", "-m", f"u:{user}:{perms}", path],
                capture_output=True, text=True, timeout=5, check=False,
            )
            if result.returncode != 0:
                detail = result.stderr.strip() or f"exit {result.returncode}"
                if required:
                    raise WebViewError(f"setfacl on {path} failed: {detail}")
                logger.warning("web view: setfacl on %s failed: %s (no audio?)",
                               path, detail)

    # ------------------------------------------------------------------
    # Navigate
    # ------------------------------------------------------------------

    async def _navigate(self, target: WebTarget) -> None:
        assert self._cdp is not None
        params: dict[str, Any] = {"url": target.url}
        if target.referrer:
            # See EMBED_REFERRER.
            params["referrer"] = target.referrer

        loaded = asyncio.ensure_future(self._cdp.wait_event(
            "Page.loadEventFired", session_id=self._session_id,
            timeout=float(self._config.load_timeout),
        ))
        try:
            result = await self._cdp.call(
                "Page.navigate", params, session_id=self._session_id,
                timeout=float(self._config.load_timeout),
            )
            error_text = result.get("errorText")
            if error_text:
                raise WebViewError(f"page failed to load: {error_text}")
            # A slow page is still a shown page: carry on after the wait
            # whether or not the load event arrived.
            await loaded
        finally:
            if not loaded.done():
                loaded.cancel()
                with contextlib.suppress(asyncio.CancelledError, CDPError):
                    await loaded

        title = await self._evaluate("document.title")
        self._title = title[:200] if isinstance(title, str) else ""

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def _restart_limit_timer(self) -> None:
        if self._limit_task is not None:
            self._limit_task.cancel()
        self._limit_task = asyncio.create_task(
            self._enforce_limit(), name="web-view-limit",
        )

    async def _enforce_limit(self) -> None:
        """Close the view after ``max_minutes`` — a forgotten video must
        not play all night."""
        await asyncio.sleep(max(1, self._config.max_minutes) * 60)
        logger.info("Web view reached its %d-minute limit; closing",
                    self._config.max_minutes)
        self._limit_task = None
        await self.close()
        await self._notify_exit("time limit reached")

    async def _watch_exit(self, proc: asyncio.subprocess.Process) -> None:
        """Clean up and tell the manager when the browser dies on its own."""
        code = await proc.wait()
        if self._closing or proc is not self._proc:
            return
        logger.warning("Web view browser exited unexpectedly (code %s)", code)
        self._watch_task = None
        async with self._lock:
            if proc is self._proc:
                await self._teardown()
        await self._notify_exit(f"browser exited (code {code})")

    async def _notify_exit(self, reason: str) -> None:
        if self._on_exit is None:
            return
        try:
            await self._on_exit(reason)
        except Exception:
            logger.exception("web view on_exit callback failed")

    async def _teardown(self) -> None:
        """Stop the browser and release every resource. Caller holds the lock."""
        self._closing = True
        current = asyncio.current_task()
        for attr in ("_limit_task", "_watch_task", "_sync_task"):
            task = getattr(self, attr)
            setattr(self, attr, None)
            if task is not None and task is not current:
                task.cancel()

        proc, cdp = self._proc, self._cdp
        if proc is not None and proc.returncode is None:
            if cdp is not None:
                with contextlib.suppress(CDPError):
                    await cdp.call("Browser.close", timeout=2.0)
            if not await self._wait_exit(proc, 3.0):
                # Signal the whole session: under sudo the browser is a
                # grandchild, and sudo relays the signal to it.
                self._signal(proc, 15)
                if not await self._wait_exit(proc, 3.0):
                    self._signal(proc, 9)
                    await self._wait_exit(proc, 2.0)
                    await self._kill_strays()

        if self._audio is not None:
            await self._audio.stop()
        if cdp is not None:
            cdp.close()
        for fd in self._fds:
            with contextlib.suppress(OSError):
                os.close(fd)
        self._fds = []
        self._proc = None
        self._cdp = None
        self._session_id = None
        self._target = None
        self._title = ""
        self._paused = False
        self._user_paused = False
        self._playing = False
        self._speaking = False
        self._pause_until = 0.0

    @staticmethod
    async def _wait_exit(proc: asyncio.subprocess.Process, timeout: float) -> bool:
        try:
            await asyncio.wait_for(proc.wait(), timeout)
        except asyncio.TimeoutError:
            return False
        return True

    @staticmethod
    def _signal(proc: asyncio.subprocess.Process, signum: int) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signum)

    async def _kill_strays(self) -> None:
        """Last resort after SIGKILL: sudo itself died, so a browser running
        as the kiosk user may be orphaned where this process cannot signal
        it. Nothing else ever runs as that user."""
        cfg = self._config
        if cfg.privilege_drop not in ("auto", "sudo"):
            return
        try:
            killer = await asyncio.create_subprocess_exec(
                "sudo", "-n", "-u", cfg.user, "--", "pkill", "-KILL", "-U", cfg.user,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(killer.wait(), 5.0)
        except (OSError, asyncio.TimeoutError) as e:
            logger.warning("web view stray cleanup failed: %s", e)
