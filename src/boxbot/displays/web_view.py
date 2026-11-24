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
- **Throwaway home.** The kiosk user's whole home (profile, certificate
  store, caches) is wiped on every launch: nothing a page did persists.
- **URL policy.** ``https`` only, public hostnames only — see
  :func:`normalize_url`. That vets the link BB was handed; it cannot
  vet where the page goes next.
- **Network fence.** What the page can reach afterwards is bounded by
  the OS: an nftables rule keyed on the kiosk uid rejects loopback and
  every private range (``scripts/setup-sandbox.sh``). Launch probes for
  the rule and refuses without it. Downloads and popups are denied.
- **Service-owned control dir.** The FIFOs live in a directory only the
  service user can modify, and are opened ``O_NOFOLLOW``: a compromised
  browser cannot swap one for a link to a file of ours.
- **Nothing from the page reaches the agent.** Not pixels, not the
  title: :meth:`WebView.state` reports only what BB itself asked for.

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
import subprocess
import time
from collections import deque
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

# Runs as the kiosk user. Empties its home (profile, NSS certificate
# store, caches — everything a page could have left behind), then execs
# the browser with the DevTools pipe on fds 3 and 4. A shell redirect is
# the only portable way to hand a child those exact fd numbers across
# sudo, which closes every inherited fd above stderr.
_LAUNCH_SCRIPT = (
    'home="$1" in="$2" out="$3" && shift 3 && '
    # A page that got as far as this uid could chmod its droppings to
    # make the wipe fail, and with it every later launch.
    'chmod -R u+rwX "$home" && '
    'find "$home" -mindepth 1 -delete && '
    'mkdir -p "$home/profile" "$home/run" && '
    'chmod 700 "$home/profile" "$home/run" && '
    'exec "$@" 3<"$in" 4>"$out"'
)

# Runs as the kiosk user before every launch: is the network fence up?
# The fence drops the packet in the kernel's output hook, which makes a
# UDP send fail at once with EPERM. Without the fence the same send
# succeeds (UDP to a closed port reports nothing). Measured on the box;
# TCP is no use here — a dropped SYN just times out, slowly, and looks
# like any unreachable host.
_FENCE_PROBE = (
    "import socket,sys\n"
    "s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)\n"
    "try:\n s.connect(('127.0.0.1',9));s.send(b'x')\n"
    "except PermissionError:\n sys.exit(0)\n"
    "except OSError:\n sys.exit(4)\n"
    "sys.exit(3)\n"
)

# AgentSpeakingDone is published from a finally block, but a view stuck
# paused because one event went missing would be silent forever. Far
# longer than any spoken reply, so real speech never outlasts it.
_SPEAKING_MAX_SECONDS = 600.0


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
    ``on_exit`` is awaited when the view closes *itself* (browser died,
    idle, time limit) so the manager can put a normal display back. It
    is not called for :meth:`close`.
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
        self._background: set[asyncio.Future[Any]] = set()
        self._stderr_tail: deque[str] = deque(maxlen=6)
        self._closing = False
        # Bumped by every open(). A self-close (idle, time limit) decided
        # under an older epoch is stale: someone has asked for a new page
        # since, and it must not be torn down.
        self._epoch = 0

        self._target: WebTarget | None = None
        self._opened_at: float = 0.0
        # Tracks speaker events, not page state: survives open/close.
        self._speaking_since: float | None = None
        self._pause_until = 0.0
        self._paused = False
        # Set by media("pause"): stays paused until media("play").
        self._user_paused = False
        self._playing = False
        self._idle_seconds = 0.0
        self._last_active = 0.0
        # True when the mic was muted *because* this view started
        # playing (see switch_display); cleared with the view.
        self.holds_mic_mute = False

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    def state(self) -> dict[str, Any] | None:
        """What the view is showing, or ``None`` when closed.

        Agent-facing, so it carries nothing the page controls — no title,
        no page text. ``url`` is the one BB asked for, after policy.
        """
        if not self.is_open or self._target is None:
            return None
        return {
            "url": self._target.url,
            "kind": self._target.kind,
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
                not load. The view is closed when this is raised — and
                equally when the call is cancelled part-way (a wake word
                during page load cancels the agent's tool call).
        """
        if not self._config.enabled:
            raise WebViewError("web view is disabled (display.web.enabled)")

        async with self._lock:
            self._epoch += 1
            try:
                if self._proc is not None and not self.is_open:
                    # The browser died and its watcher has not got the
                    # lock yet: release the dead handles before reuse.
                    await self._teardown()
                if not self.is_open:
                    await self._launch()
                await self._navigate(target)
            except BaseException as e:
                # Including CancelledError: a half-open browser has no
                # idle clock and no time limit, so it must not survive.
                await self._teardown_uninterruptible()
                if isinstance(e, (CDPError, OSError)):
                    raise WebViewError(f"web view failed: {e}") from e
                raise

            self._target = target
            self._opened_at = time.monotonic()
            # A fresh page starts playing; the sync loop pauses it again
            # within a tick if the room is not quiet.
            self._pause_until = 0.0
            self._paused = False
            self._user_paused = False
            self._playing = False
            self.holds_mic_mute = False
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

    async def _teardown_uninterruptible(self) -> None:
        """Run ``_teardown`` to completion even if this task is cancelled
        again meanwhile. Caller holds the lock.

        A bare ``shield`` is not enough: a second cancel (another wake
        word) would end the wait, the lock would be released, and the
        still-running teardown would then close whatever the *next*
        ``open()`` had just launched.
        """
        teardown = asyncio.ensure_future(self._teardown())
        while not teardown.done():
            try:
                await asyncio.shield(teardown)
            except asyncio.CancelledError:
                continue
            except Exception:
                logger.exception("web view teardown failed")

    async def _close_self(self, epoch: int, reason: str) -> bool:
        """Close because the view decided to, unless a newer ``open()``
        has superseded the decision. Returns whether it closed."""
        async with self._lock:
            if epoch != self._epoch or self._proc is None:
                return False
            await self._teardown()
        await self._notify_exit(reason)
        return True

    async def screenshot(self) -> bytes:
        """PNG of the browser window. For developers and tests only —
        never route this to the agent (see the module docstring)."""
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

    def pause_now(self, grace: float = 2.0) -> None:
        """Pause immediately (wake word heard).

        ``grace`` keeps the pause in force while the voice session
        catches up to ACTIVE; after that the sync loop decides.

        Not a coroutine on purpose: this runs inside event-bus handlers
        that the voice pipeline awaits, and a slow renderer must never
        hold up the wake word or the start of speech.
        """
        self._pause_until = time.monotonic() + grace
        self._set_paused(True)

    def set_speaking(self, speaking: bool) -> None:
        """Track TTS playback: media stays paused while BB talks."""
        self._speaking_since = time.monotonic() if speaking else None
        if speaking:
            self._set_paused(True)

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
            self._set_paused(True)
        else:
            self._user_paused = False
            self._pause_until = 0.0
            self._last_active = time.monotonic()
            if not self._wants_pause():
                self._paused = False
                await self._evaluate(_PLAY_JS)
        state = self.state()
        return state if state is not None else {}

    def _is_speaking(self) -> bool:
        since = self._speaking_since
        return since is not None and time.monotonic() - since < _SPEAKING_MAX_SECONDS

    def _wants_pause(self) -> bool:
        if self._user_paused or self._is_speaking():
            return True
        if time.monotonic() < self._pause_until:
            return True
        try:
            return bool(self._listening_probe())
        except Exception:
            logger.debug("web view listening probe failed", exc_info=True)
            return False

    def _set_paused(self, paused: bool) -> None:
        """Flip the pause state. The audio gate follows ``_paused`` at
        once; the page is told in the background."""
        if not self.is_open or paused == self._paused:
            return
        self._paused = paused
        self._spawn(self._evaluate(_PAUSE_JS if paused else _RESUME_JS))

    async def _sync_loop(self) -> None:
        """Keep playback in step with the room, and close when idle."""
        next_activity_check = 0.0
        while self.is_open:
            await asyncio.sleep(_SYNC_INTERVAL)
            # This loop is the only thing that pauses a video for the
            # room and the only thing that closes an idle page; one bad
            # tick must not end it.
            try:
                self._set_paused(self._wants_pause())
                now = time.monotonic()
                if now < next_activity_check:
                    continue
                next_activity_check = now + _ACTIVITY_INTERVAL
                epoch = self._epoch
                await self._check_activity()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("web view sync tick failed")
                continue
            if time.monotonic() - self._last_active <= self._idle_seconds:
                continue
            logger.info("Web view idle for %.0fs; closing", self._idle_seconds)
            if await self._close_self(epoch, "nothing playing — idle timeout"):
                return

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
        """Run ``expression`` in the page. Best-effort: ``None`` on failure.

        The result comes from the page's own JavaScript world, so a
        hostile page can lie in it. It is only ever used to decide when
        to close the view — never shown to the agent.
        """
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

    def _privilege_drop(self) -> str:
        drop = self._config.privilege_drop
        return "sudo" if drop == "auto" else drop

    async def _launch(self) -> None:
        if self._privilege_drop() == "sudo":
            # setfacl / sudo round-trips: keep them off the event loop.
            await asyncio.to_thread(self._prepare_session)

        # Before the browser starts, so its first sound has somewhere to go.
        audio_sink = None
        if self._audio is not None:
            if not stat.S_ISSOCK(self._stat_mode(self._pulse_socket())):
                logger.warning(
                    "web view audio off: %s missing — run "
                    "scripts/setup-sandbox.sh on the box", self._pulse_socket(),
                )
            elif await self._audio.start():
                audio_sink = SINK_NAME
        argv, ctl_in, ctl_out = self._build_launch(audio_sink)

        self._fds = [self._open_fifo(ctl_in), self._open_fifo(ctl_out)]
        write_fd, read_fd = self._fds

        logger.info("Launching web view: %s", " ".join(argv[:6]) + " …")
        self._closing = False
        self._proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self._stderr_tail.clear()
        self._spawn(self._drain_stderr(self._proc))
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
                await asyncio.sleep(0.05)   # let the last stderr lines land
                detail = " | ".join(self._stderr_tail) or "no output"
                raise WebViewError(
                    "browser exited during startup "
                    f"(exit code {self._proc.returncode}): {detail}"
                ) from e
            raise WebViewError(f"browser did not come up: {e}") from e

    def _spawn(self, coro: Awaitable[Any]) -> None:
        """Run ``coro`` in the background, holding a reference to it."""
        task = asyncio.ensure_future(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _drain_stderr(self, proc: asyncio.subprocess.Process) -> None:
        """Keep the browser's last few stderr lines.

        Read continuously so the pipe can never fill and stall the
        browser. The lines are the browser's own startup diagnostics
        ("Missing X server", sandbox errors); they go to the log and
        into a launch-failure message, nowhere else.
        """
        assert proc.stderr is not None
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    return
                text = line.decode(errors="replace").strip()
                if text:
                    self._stderr_tail.append(text[:300])
        except (asyncio.CancelledError, ValueError, OSError):
            return

    @staticmethod
    def _open_fifo(path: str) -> int:
        """Open one end of the control pipe.

        ``O_RDWR`` so neither side blocks in ``open()`` waiting for its
        peer. ``O_NONBLOCK`` so a stalled browser can never block the
        event loop in ``write()``. ``O_NOFOLLOW`` plus the ``fstat``
        check so the path is the FIFO this process just created and not
        something swapped in under it.
        """
        fd = os.open(
            path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
        if not stat.S_ISFIFO(os.fstat(fd).st_mode):
            os.close(fd)
            raise WebViewError(f"{path} is not a FIFO")
        return fd

    def _session_paths(self) -> tuple[str, str]:
        """``(runtime_dir, wayland_socket)`` of this session."""
        runtime_dir = (
            os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
        ).rstrip("/")
        wayland = os.environ.get("WAYLAND_DISPLAY") or "wayland-0"
        wayland_socket = wayland if os.path.isabs(wayland) else f"{runtime_dir}/{wayland}"
        return runtime_dir, wayland_socket

    def _pulse_socket(self) -> str:
        """The PulseAudio socket set aside for the browser.

        Not the session's own ``$XDG_RUNTIME_DIR/pulse/native``: every
        libpulse client re-secures that directory to 0700 when it
        connects, silently revoking any ACL granted on it (seen on the
        box: the browser played, and nothing reached the sink).
        ``setup-sandbox.sh`` has pipewire-pulse listen on this second,
        restricted-access socket in a directory the kiosk user can
        enter by group.
        """
        return str(Path(self._config.state_dir) / "pulse" / "native")

    @staticmethod
    def _stat_mode(path: str) -> int:
        try:
            return os.stat(path).st_mode
        except OSError:
            return 0

    def _prepare_session(self) -> None:
        """Everything that must hold before the browser may start as the
        kiosk user. Blocking; runs in a worker thread. Raises
        :class:`WebViewError` — launch is refused, never degraded."""
        user = self._config.user
        try:
            pwd.getpwnam(user)
        except KeyError as e:
            raise WebViewError(
                f"web view user '{user}' does not exist — run "
                "scripts/setup-sandbox.sh on the box"
            ) from e

        try:
            # A browser left over from an unclean stop. Nothing else ever
            # runs as this user. pkill exits 1 when there is nothing to kill.
            self._as_kiosk("pkill", "-KILL", "-U", user)

            probe = self._as_kiosk("python3", "-c", _FENCE_PROBE)
            if probe.returncode != 0:
                raise WebViewError(
                    "web view network fence is not active (the kiosk user "
                    "can reach local addresses) — run scripts/setup-sandbox.sh "
                    "on the box"
                )

            runtime_dir, wayland_socket = self._session_paths()
            self._grant_session_access(user, runtime_dir, wayland_socket)
        except (OSError, subprocess.SubprocessError) as e:
            raise WebViewError(f"web view session setup failed: {e}") from e

    def _as_kiosk(self, *argv: str) -> "subprocess.CompletedProcess[str]":
        return subprocess.run(
            ["sudo", "-n", "-u", self._config.user, "--", *argv],
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=10, check=False,
        )

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

        drop = self._privilege_drop()
        state_dir = Path(cfg.state_dir)
        ctl_dir = state_dir / "ctl"
        home = state_dir / "home"
        ctl_in = ctl_dir / "to_browser"
        ctl_out = ctl_dir / "from_browser"
        try:
            if drop == "none":
                # Dev / tests: one user owns everything. On the box these
                # exist already, with ownership only root can arrange.
                ctl_dir.mkdir(parents=True, exist_ok=True)
                home.mkdir(parents=True, exist_ok=True)
            for fifo in (ctl_in, ctl_out):
                with contextlib.suppress(FileNotFoundError):
                    fifo.unlink()
                os.mkfifo(fifo, 0o660)
                # mkfifo honours the umask; the kiosk user reaches these
                # through the directory's setgid group and needs both bits.
                os.chmod(fifo, 0o660)
        except OSError as e:
            raise WebViewError(
                f"cannot prepare web view control dir {ctl_dir}: {e} — run "
                "scripts/setup-sandbox.sh on the box"
            ) from e

        _, wayland_socket = self._session_paths()
        width, height = self._size
        browser_argv = [
            browser,
            "--kiosk",
            "--remote-debugging-pipe",
            f"--user-data-dir={home / 'profile'}",
            f"--window-size={width},{height}",
            "--window-position=0,0",
            # Explicit: the browser starts from an empty environment
            # (env -i), so there is no XDG_SESSION_TYPE for "auto" to
            # read and Chromium would fall back to X11 and exit.
            "--ozone-platform=wayland",
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
            # No second window: the view drives, pauses and times out
            # exactly one page.
            "--block-new-web-contents",
            *cfg.extra_args,
            "about:blank",
        ]
        env = {
            "HOME": str(home),
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "XDG_RUNTIME_DIR": str(home / "run"),
            "WAYLAND_DISPLAY": wayland_socket,
            "LANG": os.environ.get("LANG", "C.UTF-8"),
        }
        if audio_sink is not None:
            env["PULSE_SERVER"] = f"unix:{self._pulse_socket()}"
            env["PULSE_SINK"] = audio_sink
        else:
            browser_argv.insert(-1, "--mute-audio")
        argv = [
            "env", "-i", *(f"{k}={v}" for k, v in env.items()),
            "sh", "-c", _LAUNCH_SCRIPT, "boxbot-web-view",
            str(home), str(ctl_in), str(ctl_out),
            *browser_argv,
        ]
        if drop == "sudo":
            argv = ["sudo", "-n", "-u", cfg.user, "--", *argv]
        return argv, str(ctl_in), str(ctl_out)

    @staticmethod
    def _grant_session_access(user: str, runtime_dir: str, wayland_socket: str) -> None:
        """Let the kiosk user reach this session's compositor.

        The runtime dir is a tmpfs recreated at every login, so the ACLs
        cannot be installed once by the setup script. The service user
        owns these paths, so no privilege is needed to grant them.

        Traversing the runtime dir would also expose every world-writable
        socket in it. The native PipeWire sockets are the dangerous ones
        (full graph access: record any node, rewire anything), so they
        are closed to other users here. Sound does not go through this
        directory at all — see :meth:`_pulse_socket`.
        """
        for name in ("pipewire-0", "pipewire-0-manager"):
            path = f"{runtime_dir}/{name}"
            if os.path.exists(path):
                os.chmod(path, 0o600)

        for path, perms in ((runtime_dir, "x"), (wayland_socket, "rw")):
            if not os.path.exists(path):
                raise WebViewError(f"{path} not found — is the desktop session up?")
            result = subprocess.run(
                ["setfacl", "-m", f"u:{user}:{perms}", path],
                capture_output=True, text=True, timeout=5, check=False,
            )
            if result.returncode != 0:
                detail = result.stderr.strip() or f"exit {result.returncode}"
                raise WebViewError(f"setfacl on {path} failed: {detail}")

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

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def _restart_limit_timer(self) -> None:
        if self._limit_task is not None:
            self._limit_task.cancel()
        self._limit_task = asyncio.create_task(
            self._enforce_limit(self._epoch), name="web-view-limit",
        )

    async def _enforce_limit(self, epoch: int) -> None:
        """Close the view after ``max_minutes`` — a forgotten video must
        not play all night."""
        await asyncio.sleep(self._config.max_minutes * 60)
        logger.info("Web view reached its %g-minute limit; closing",
                    self._config.max_minutes)
        await self._close_self(epoch, "time limit reached")

    async def _watch_exit(self, proc: asyncio.subprocess.Process) -> None:
        """Clean up and tell the manager when the browser dies on its own."""
        code = await proc.wait()
        if self._closing or proc is not self._proc:
            return
        logger.warning("Web view browser exited unexpectedly (code %s)", code)
        # A dead browser never answers and (the pipe being O_RDWR) never
        # produces EOF: fail the calls in flight now, so whoever holds
        # the lock stops waiting on them.
        if self._cdp is not None:
            self._cdp.close()
        async with self._lock:
            if proc is not self._proc:
                return  # an open() already cleaned up and moved on
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
        if proc is not None:
            if proc.returncode is None:
                if cdp is not None:
                    with contextlib.suppress(CDPError):
                        await cdp.call("Browser.close", timeout=2.0)
                if not await self._wait_exit(proc, 3.0):
                    # Signal the whole session: under sudo the browser is
                    # a grandchild, and sudo relays the signal to it.
                    self._signal(proc, 15)
                    if not await self._wait_exit(proc, 3.0):
                        self._signal(proc, 9)
                        await self._wait_exit(proc, 2.0)
            # Whatever the browser left running as the kiosk user —
            # helpers that outlived it, or anything a page managed to
            # start — goes with it.
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
        self._paused = False
        self._user_paused = False
        self._playing = False
        self._pause_until = 0.0
        self.holds_mic_mute = False

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
        """Kill every process of the kiosk user. Nothing else ever runs
        as that user, and this process cannot signal them directly."""
        if self._privilege_drop() != "sudo":
            return
        user = self._config.user
        try:
            killer = await asyncio.create_subprocess_exec(
                "sudo", "-n", "-u", user, "--", "pkill", "-KILL", "-U", user,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(killer.wait(), 5.0)
        except (OSError, asyncio.TimeoutError) as e:
            logger.warning("web view stray cleanup failed: %s", e)
