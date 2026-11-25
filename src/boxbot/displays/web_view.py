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
- **Nothing from the page reaches the agent** except a small fixed set
  of typed values. Not pixels, not the title, not a string of any
  kind: :meth:`WebView.state` reports what BB itself asked for plus
  playback facts read from the page as numbers and booleans, each
  type-checked and clamped (:func:`parse_probe`). A hostile page can
  lie in them; it cannot say anything with them.

Lifetime: the view closes when the display switches away, on
``unpin``, when nothing has played or been touched for
``idle_minutes``, ``close_after_end_seconds`` after its media ended,
after ``max_minutes`` regardless, or if the browser dies or its page
crashes. In every self-initiated case the owner is told through
``on_exit`` (one of the fixed ``EXIT_*`` strings) so a normal display
comes back.

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
import math
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

# Starts whatever media the page has, whether or not this view paused
# it. Media that has ended goes back to the beginning first.
_PLAY_JS = (
    "(()=>{let n=0;document.querySelectorAll('video,audio').forEach(m=>{"
    "delete m.dataset.bbHeld;if(m.paused||m.ended){"
    "if(m.ended){try{m.currentTime=0}catch(e){}}const p=m.play();"
    "if(p&&p.catch)p.catch(()=>{});n++}});return n})()"
)
# Is anything playing, how long since someone touched the page, and how
# is the primary media element doing? Installs its input listeners on
# first run in each document.
#
# "Primary" is chosen the same way every time: a <video> beats an
# <audio>, then the longest finite duration, then document order.
# Numbers and booleans only — see parse_probe, which trusts none of it.
_ACTIVITY_JS = (
    "(()=>{if(!window.__bbSeen){window.__bbSeen=Date.now();"
    "for(const t of['pointerdown','touchstart','keydown','wheel','scroll'])"
    "addEventListener(t,()=>{window.__bbSeen=Date.now()},"
    "{capture:true,passive:true})}"
    "const ms=[...document.querySelectorAll('video,audio')];"
    "const playing=ms.some(m=>!m.paused&&!m.ended&&m.readyState>2);"
    "const score=m=>(m.tagName==='VIDEO'?1e12:0)"
    "+(isFinite(m.duration)?m.duration:0);"
    "let p=null;for(const m of ms){if(!p||score(m)>score(p))p=m}"
    "let media=null;if(p){let q=null;"
    "try{q=p.getVideoPlaybackQuality?p.getVideoPlaybackQuality():null}"
    "catch(e){}"
    "media={t:p.currentTime,d:p.duration,paused:p.paused,ended:p.ended,"
    "rs:p.readyState,ns:p.networkState,err:p.error?p.error.code:0,"
    "frames:q?q.totalVideoFrames:null,dropped:q?q.droppedVideoFrames:null}}"
    "return JSON.stringify({playing,quiet_ms:Date.now()-window.__bbSeen,"
    "media})})()"
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
# Consecutive probes without progress (or without an answer) before
# playback counts as stalled.
_STALL_PROBES = 3
# Position must move by more than this between probes to count as progress.
_PROGRESS_EPSILON = 0.05
# Media sitting paused this close to its end has ended, whatever the
# element's own ``ended`` flag says (players differ on when they set it).
_END_SLACK = 0.5

# Why the view closed itself: passed to ``on_exit`` and readable by the
# agent, so fixed strings only. A dead browser adds its exit code.
EXIT_IDLE = "nothing playing — idle timeout"
EXIT_LIMIT = "time limit reached"
EXIT_ENDED = "media ended"
EXIT_CRASHED = "page crashed"
EXIT_RELOAD_FAILED = "reload failed"

# Why playback is being held (``state()["paused_by"]``).
PAUSED_LISTENING = "listening"
PAUSED_SPEAKING = "speaking"
PAUSED_USER = "user"

_MEDIA_ACTIONS = ("play", "pause", "restart")

# Bounds for what the probe may report. Anything outside is clamped;
# anything that is not a finite number / a real boolean is dropped.
_MAX_PROBE_CHARS = 2048
_MAX_MEDIA_SECONDS = 7 * 86400.0
_MAX_FRAMES = 2 ** 53
_MAX_ELEMENTS = 9999

# Browser stderr goes to the main log under its own name, so it can be
# filtered in or out: ``boxbot.displays.web_view.browser``.
browser_logger = logging.getLogger(__name__ + ".browser")
_STDERR_LINE_CHARS = 300
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]")


def _clean_log_line(raw: bytes) -> str:
    """One line of browser output, made safe to write to the log:
    escape sequences and control characters out, length capped."""
    text = _ANSI_ESCAPE.sub("", raw.decode(errors="replace"))
    text = _CONTROL_CHARS.sub(" ", text).strip()
    return text[:_STDERR_LINE_CHARS]


class _LogBudget:
    """At most ``lines`` log lines per ``per_seconds``; the rest are counted.

    Everything rate-limited here is driven by the page or the browser,
    which must not be able to fill the disk through the log.
    """

    def __init__(self, lines: int, per_seconds: float) -> None:
        self._lines = lines
        self._window = per_seconds
        self._window_start = 0.0
        self._used = 0
        self._suppressed = 0

    def take(self) -> int | None:
        """``None`` if this line must be dropped; otherwise how many
        lines were dropped since the last one that got through."""
        now = time.monotonic()
        if now - self._window_start >= self._window:
            self._window_start = now
            self._used = 0
        if self._used >= self._lines:
            self._suppressed += 1
            return None
        self._used += 1
        suppressed, self._suppressed = self._suppressed, 0
        return suppressed


def _number(value: Any, low: float, high: float) -> float | None:
    """``value`` as a finite float clamped to ``[low, high]``, else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    # Compared before converting: an int too large for a float clamps
    # here instead of raising in float().
    if value <= low:
        return float(low)
    if value >= high:
        return float(high)
    return float(value)


def _count(value: Any, high: int) -> int | None:
    """A non-negative whole number clamped to ``high``, else ``None``."""
    number = _number(value, 0, high)
    return None if number is None else int(number)


def _enum(value: Any, high: int) -> int | None:
    """An integer code in ``0..high``; anything else is ``None``."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 <= value <= high else None


def _flag(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


@dataclass(frozen=True)
class MediaProbe:
    """The primary media element, as far as the page can be believed.

    Every field is ``None`` when the page did not supply a value of the
    right type. Seconds for ``position`` / ``duration``; the HTML
    ``readyState`` (0-4), ``networkState`` (0-3) and ``MediaError.code``
    (0 = none, 1-4) enums as integers.
    """

    position: float | None = None
    duration: float | None = None
    paused: bool | None = None
    ended: bool | None = None
    ready_state: int | None = None
    network_state: int | None = None
    error_code: int | None = None
    frames: int | None = None
    dropped: int | None = None

    @property
    def at_end(self) -> bool:
        """Finished: flagged ended, or sitting paused at its last moment."""
        if self.ended is True:
            return True
        return (
            self.paused is True
            and self.position is not None
            and self.duration is not None
            and self.position >= self.duration - _END_SLACK
        )


@dataclass(frozen=True)
class Probe:
    """One answer from ``_ACTIVITY_JS`` after validation."""

    answered: bool = False
    playing: bool = False
    quiet_ms: float | None = None
    media: MediaProbe | None = None


def parse_probe(raw: Any) -> Probe:
    """Validate what the page returned for ``_ACTIVITY_JS``.

    The value comes out of the page's own JavaScript world: a hostile
    page can replace ``JSON.stringify`` or the media prototypes and
    return anything. Only finite numbers and real booleans survive, in
    fixed fields, clamped to sane ranges. No string from the page is
    kept in any form.
    """
    if not isinstance(raw, str) or len(raw) > _MAX_PROBE_CHARS:
        return Probe()
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError):
        return Probe()
    if not isinstance(data, dict):
        return Probe()

    media: MediaProbe | None = None
    m = data.get("media")
    if isinstance(m, dict):
        duration = _number(m.get("d"), 0.0, _MAX_MEDIA_SECONDS)
        if duration == 0.0:
            duration = None
        position = _number(m.get("t"), 0.0, _MAX_MEDIA_SECONDS)
        if position is not None and duration is not None:
            position = min(position, duration)
        media = MediaProbe(
            position=position,
            duration=duration,
            paused=_flag(m.get("paused")),
            ended=_flag(m.get("ended")),
            ready_state=_enum(m.get("rs"), 4),
            network_state=_enum(m.get("ns"), 3),
            error_code=_enum(m.get("err"), 4),
            frames=_count(m.get("frames"), _MAX_FRAMES),
            dropped=_count(m.get("dropped"), _MAX_FRAMES),
        )
    return Probe(
        answered=True,
        playing=data.get("playing") is True,
        quiet_ms=_number(data.get("quiet_ms"), 0.0, 1e12),
        media=media,
    )


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
        # Bumped by every page load (open() and reload). A self-close or
        # a queued reload decided under an older epoch is stale: someone
        # has asked for a new page since, and it must not be acted on.
        self._epoch = 0
        # True while a page load holds the lock. A crash reported then
        # cannot be pinned on the old page or the new one; see
        # _on_target_crashed.
        self._loading = False
        self._crashed_in_load = False
        # Bumped when ended media is told to play again: a probe sent
        # before that describes playback that no longer exists.
        self._playback_gen = 0
        # time.monotonic() at which the hard limit closes the view. Set
        # by open() only; a reload keeps it.
        self._limit_at = 0.0

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
        self._page_target_id: str | None = None
        # Why playback is held right now; None when it is not.
        self._paused_by: str | None = None
        # media("play") arrived while the room was not quiet: start the
        # page's media (not just what this view paused) on resume.
        self._play_on_resume = False
        # False when the page was opened with its own idle_minutes: the
        # caller said how long it may sit there, so ended media does
        # not close it early.
        self._close_on_end = True
        # Playback as of the last probe. See _track_playback.
        self._media: MediaProbe | None = None
        self._ended = False
        self._ended_since: float | None = None
        self._stalled = False
        self._stall_probes = 0
        self._probe_failures = 0
        self._last_position: float | None = None
        self._phase: str | None = None
        self._pause_log = _LogBudget(30, 60.0)
        self._playback_log = _LogBudget(12, 60.0)
        self._evaluate_log = _LogBudget(1, 30.0)
        self._stderr_log = _LogBudget(30, 60.0)
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

        Agent-facing, so it carries no text the page controls — no
        title, no page text. ``url`` is the one BB asked for, after
        policy. The playback fields come from the page as validated
        numbers and booleans (:func:`parse_probe`), up to one probe
        interval old:

        - ``playing`` — media is running and audible to the room.
        - ``paused`` — BB is holding playback; ``paused_by`` says why:
          ``"listening"`` (mic open or wake word just heard),
          ``"speaking"`` (BB talking), ``"user"`` (``media("pause")``).
          The first two lift by themselves; ``None`` when not held.
          This is BB's hold, never the page's own pause button.
        - ``ended`` — the primary media played to its end.
        - ``stalled`` — it should have been running and was not moving
          (or the page stopped answering).
        - ``position`` / ``duration`` — whole seconds, ``None`` if unknown.

        ``playing`` excludes the other three. ``ended`` and ``stalled``
        describe the media as last seen and stay readable while BB
        holds playback — the agent only ever looks during a
        conversation, which is exactly when ``paused`` is true. All
        four false: nothing to play, or media the page has not started.
        """
        if not self.is_open or self._target is None:
            return None
        media = self._media
        ended = self._ended
        stalled = self._stalled and not ended
        return {
            "url": self._target.url,
            "kind": self._target.kind,
            "playing": (
                self._playing and not self._paused and not ended and not stalled
            ),
            "paused": self._paused,
            "paused_by": self._paused_by if self._paused else None,
            "ended": ended,
            "stalled": stalled,
            "position": (
                round(media.position)
                if media is not None and media.position is not None else None
            ),
            "duration": (
                round(media.duration)
                if media is not None and media.duration is not None else None
            ),
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
            await self._load(target, may_launch=True)
            # Every open() starts a new hard limit — including one that
            # only navigates the running browser to a different (or the
            # same) URL: it is a deliberate switch_display('web', url)
            # by the agent, never something the page can cause. A reload
            # (media("restart")) goes through _reload and does not.
            self._opened_at = time.monotonic()
            self.holds_mic_mute = False
            self._close_on_end = idle_minutes is None
            minutes = self._config.idle_minutes if idle_minutes is None else idle_minutes
            self._idle_seconds = 60.0 * max(1.0, min(minutes, self._config.max_minutes))
            self._restart_limit_timer()
            if self._sync_task is None:
                self._sync_task = asyncio.create_task(
                    self._sync_loop(), name="web-view-sync",
                )
            state = self.state()
            return state if state is not None else {}

    async def _load(self, target: WebTarget, *, may_launch: bool) -> None:
        """Navigate to ``target`` under a new epoch. Caller holds the lock.

        ``may_launch=False`` is the reuse-only path: the caller has
        checked, with the lock held, that the browser is running, and
        nothing here will start one.

        On any failure — including cancellation — the view is torn down
        before the exception leaves.
        """
        self._epoch += 1
        self._loading = True
        self._crashed_in_load = False
        try:
            if may_launch:
                if self._proc is not None and not self.is_open:
                    # The browser died and its watcher has not got the
                    # lock yet: release the dead handles before reuse.
                    await self._teardown()
                if not self.is_open:
                    await self._launch()
            await self._navigate(target)
            if self._crashed_in_load:
                await self._confirm_page_alive()
        except BaseException as e:
            # Including CancelledError: a half-open browser has no
            # idle clock and no time limit, so it must not survive.
            await self._teardown_uninterruptible()
            if isinstance(e, (CDPError, OSError)):
                raise WebViewError(f"web view failed: {e}") from e
            raise
        finally:
            self._loading = False

        self._target = target
        # A fresh page starts playing; the sync loop pauses it again
        # within a tick if the room is not quiet.
        self._pause_until = 0.0
        self._reset_page_state()
        self._last_active = time.monotonic()

    async def _confirm_page_alive(self) -> None:
        """A crash was reported during this load. If it was the page
        being left, the navigation replaced it and the new page answers;
        if it was the new page, nothing does."""
        assert self._cdp is not None
        try:
            await self._cdp.call(
                "Runtime.evaluate", {"expression": "1", "returnByValue": True},
                session_id=self._session_id, timeout=5.0,
            )
        except CDPError as e:
            raise WebViewError("page crashed while loading") from e

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

    def _reset_page_state(self) -> None:
        """Forget everything known about the page that was showing."""
        self._paused = False
        self._paused_by = None
        self._user_paused = False
        self._play_on_resume = False
        self._playing = False
        self._media = None
        self._ended = False
        self._ended_since = None
        self._stalled = False
        self._stall_probes = 0
        self._probe_failures = 0
        self._last_position = None
        self._phase = None

    async def _close_self(self, epoch: int | None, reason: str) -> bool:
        """Close because the view decided to. Returns whether it closed.

        The decision was made before waiting for the lock, so it is
        checked again once the lock is held: a newer page load
        (``epoch``; ``None`` for the time limit, which outlives reloads)
        or a reason that no longer holds leaves the view alone.
        """
        async with self._lock:
            if self._proc is None or (epoch is not None and epoch != self._epoch):
                return False
            if not self._close_still_due(reason):
                return False
            await self._teardown()
        await self._notify_exit(reason)
        return True

    def _close_still_due(self, reason: str) -> bool:
        """Does ``reason`` for a self-close hold right now?"""
        if reason == EXIT_ENDED:
            return self._end_grace_over()     # replayed / held meanwhile
        if reason == EXIT_IDLE:
            return self._idle_expired()
        if reason == EXIT_LIMIT:
            return self._limit_reached()
        return True                           # crash: the epoch says it all

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
        self._set_paused(True, PAUSED_LISTENING)

    def set_speaking(self, speaking: bool) -> None:
        """Track TTS playback: media stays paused while BB talks."""
        self._speaking_since = time.monotonic() if speaking else None
        if speaking:
            self._set_paused(True, PAUSED_SPEAKING)

    async def media(self, action: str) -> dict[str, Any]:
        """Explicit play / pause / restart, as asked for by a person.

        ``pause`` holds until ``play`` — unlike the automatic pause,
        which lifts as soon as the room is quiet. ``play`` also starts
        media the page loaded without autoplaying, and plays media that
        has ended again from the beginning. ``restart`` loads the page
        BB opened afresh: the way out of a stalled or broken player.

        Returns :meth:`state` plus ``result``: ``"paused"``,
        ``"playing"``, ``"restarted"``, or ``"deferred"`` — the room is
        not quiet (see ``paused_by``), so play starts by itself once it
        is. Nothing needs calling again.
        """
        if action not in _MEDIA_ACTIONS:
            raise WebViewError("action must be 'play', 'pause' or 'restart'")
        if not self.is_open:
            raise WebViewError("web view is not open")
        if action == "pause":
            self._user_paused = True
            self._play_on_resume = False
            self._set_paused(True, PAUSED_USER)
            result = "paused"
        elif action == "restart":
            # The epoch as of this request, read before any await.
            await self._reload(self._epoch)
            result = "restarted"
        else:
            self._user_paused = False
            self._pause_until = 0.0
            self._last_active = time.monotonic()
            reason = self._pause_reason()
            if reason is None:
                was_paused, held_by = self._paused, self._paused_by
                was_ended = self._ended
                self._paused = False
                self._paused_by = None
                self._play_on_resume = False
                self._playback_resumed(replay=True)
                # A pause sent a moment ago is still waiting to start in
                # the background. Let it reach the pipe first, or the
                # page would get "play" then "pause".
                await asyncio.sleep(0)
                touched = _count(await self._evaluate(_PLAY_JS), _MAX_ELEMENTS)
                if was_paused:
                    self._log_pause_flip(False, held_by, touched)
                result = "restarted" if was_ended else "playing"
            else:
                # Not now — and when the room goes quiet, start the
                # page's media whether or not this view paused it.
                self._play_on_resume = True
                self._set_paused(True, reason)
                result = "deferred"
        state = self.state()
        if state is None:
            return {}
        state["result"] = result
        return state

    async def _reload(self, epoch: int) -> None:
        """Load the page BB opened again, from the start.

        Reuse only: this never launches a browser. ``epoch`` is the one
        the request was made under. If, once the lock is held, the view
        has closed (ended, idle, limit, crash — the owner has put
        another display back) or loaded something else, the request is
        void and nothing is touched.

        New epoch (a close decided about the old load cannot land on
        this one) and a fresh idle clock; the idle allowance, the mic
        mute, ``open_seconds`` and the hard time limit carry over — a
        page that reports ``stalled`` must not earn itself more time.
        If the reload fails the view is closed and the owner told, as
        for any other self-close.
        """
        load_failed = False
        try:
            async with self._lock:
                target = self._target
                if target is None or not self.is_open or epoch != self._epoch:
                    raise WebViewError("web view is not open")
                load_failed = True        # until _load returns: it tears down
                await self._load(target, may_launch=False)
                load_failed = False
        finally:
            if load_failed:
                # In the background, so a cancelled tool call still
                # hands the screen back.
                self._spawn(self._notify_exit(EXIT_RELOAD_FAILED))

    def _is_speaking(self) -> bool:
        since = self._speaking_since
        return since is not None and time.monotonic() - since < _SPEAKING_MAX_SECONDS

    def _pause_reason(self) -> str | None:
        """Why playback must be held right now, or ``None`` if the room
        is quiet and nobody asked for a pause."""
        if self._user_paused:
            return PAUSED_USER
        if self._is_speaking():
            return PAUSED_SPEAKING
        if time.monotonic() < self._pause_until:
            return PAUSED_LISTENING
        try:
            listening = bool(self._listening_probe())
        except Exception:
            logger.debug("web view listening probe failed", exc_info=True)
            return None
        return PAUSED_LISTENING if listening else None

    def _wants_pause(self) -> bool:
        return self._pause_reason() is not None

    def _set_paused(self, paused: bool, reason: str | None = None) -> None:
        """Flip the pause state. The audio gate follows ``_paused`` at
        once; the page is told in the background."""
        if not self.is_open:
            return
        if paused == self._paused:
            if paused and reason is not None:
                self._paused_by = reason    # e.g. listening → speaking
            return
        held_by = self._paused_by
        self._paused = paused
        self._paused_by = reason if paused else None
        if paused:
            script = _PAUSE_JS
        else:
            replay = self._play_on_resume
            script = _PLAY_JS if replay else _RESUME_JS
            self._play_on_resume = False
            self._playback_resumed(replay=replay)
        self._spawn(self._apply_pause(script, paused, reason if paused else held_by))

    def _playback_resumed(self, *, replay: bool) -> None:
        """The hold was just lifted. A stall seen before it is out of
        date, and the wait before closing ended media starts over.
        ``replay``: ended media was told to start again as well."""
        self._stalled = False
        self._stall_probes = 0
        self._probe_failures = 0
        self._ended_since = None
        if replay:
            self._ended = False
            self._playback_gen += 1

    async def _apply_pause(self, script: str, paused: bool, reason: str | None) -> None:
        touched = _count(await self._evaluate(script), _MAX_ELEMENTS)
        self._log_pause_flip(paused, reason, touched)

    def _log_pause_flip(self, paused: bool, reason: str | None, touched: int | None) -> None:
        """One line per pause / resume: why, and how many media elements
        the page-side script acted on (``None``: the page did not answer)."""
        suppressed = self._pause_log.take()
        if suppressed is None:
            return
        logger.info(
            "Web view %s (%s=%s, elements=%s)%s",
            "paused" if paused else "resumed",
            "reason" if paused else "was", reason, touched,
            f" [+{suppressed} flips not logged]" if suppressed else "",
        )

    async def _sync_loop(self) -> None:
        """Keep playback in step with the room; close when the media has
        ended or the page is idle."""
        next_activity_check = 0.0
        while self.is_open:
            await asyncio.sleep(_SYNC_INTERVAL)
            # This loop is the only thing that pauses a video for the
            # room and the only thing that closes an idle page; one bad
            # tick must not end it.
            try:
                reason = self._pause_reason()
                self._set_paused(reason is not None, reason)
                now = time.monotonic()
                if now < next_activity_check:
                    continue
                next_activity_check = now + _ACTIVITY_INTERVAL
                epoch = self._epoch
                # A pause / resume flipped just above is sent from a
                # background task: let it reach the pipe ahead of the
                # probe, so the probe sees its effect.
                await asyncio.sleep(0)
                await self._check_activity(epoch)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("web view sync tick failed")
                continue
            if self._end_grace_over():
                logger.info(
                    "Web view media ended %.0fs ago; closing",
                    self._config.close_after_end_seconds,
                )
                if await self._close_self(epoch, EXIT_ENDED):
                    return
                continue
            if not self._idle_expired():
                continue
            logger.info("Web view idle for %.0fs; closing", self._idle_seconds)
            if await self._close_self(epoch, EXIT_IDLE):
                return

    def _end_grace_over(self) -> bool:
        """The media ended, stayed ended for the configured grace, and
        nothing is holding the view for the room."""
        grace = float(self._config.close_after_end_seconds)
        return (
            grace > 0
            and self._close_on_end
            and self._ended
            and not self._paused
            and self._ended_since is not None
            and time.monotonic() - self._ended_since >= grace
        )

    def _idle_expired(self) -> bool:
        """Nothing played or was touched for the idle allowance, and BB
        is not holding the view for the room (a wake word since the
        last probe counts)."""
        if self._paused and not self._user_paused:
            return False
        return time.monotonic() - self._last_active > self._idle_seconds

    def _limit_reached(self) -> bool:
        return time.monotonic() >= self._limit_at

    async def _check_activity(self, epoch: int | None = None) -> None:
        """Ask the page how it is doing.

        Refreshes ``_last_active`` — something is playing, someone just
        touched the page, or BB itself is holding playback for the room
        (a conversation must not run the idle clock down) — and the
        playback picture behind :meth:`state`.
        """
        playback_gen = self._playback_gen
        raw = await self._evaluate(_ACTIVITY_JS)
        if not self.is_open or (epoch is not None and epoch != self._epoch):
            return  # the answer is about a page that has been replaced
        if playback_gen != self._playback_gen:
            return  # ...or about media that has been replayed since
        probe = parse_probe(raw)
        self._playing = probe.playing
        touched = probe.quiet_ms is not None and (
            probe.quiet_ms < _ACTIVITY_INTERVAL * 1500
        )
        held_for_room = self._paused and not self._user_paused
        if self._playing or touched or held_for_room:
            self._last_active = time.monotonic()
        self._track_playback(probe)
        if self._paused and probe.playing:
            # Media that started after the pause went out (a page still
            # loading when the wake word came). Sound is gated on
            # ``_paused`` regardless; this stops the picture too.
            if self._pause_log.take() is not None:
                logger.info("Web view media started while held; pausing it")
            self._spawn(self._evaluate(_PAUSE_JS))

    def _track_playback(self, probe: Probe) -> None:
        """Work out ended / stalled from one probe and log the changes."""
        if not probe.answered:
            # No answer says nothing about the media; several in a row
            # say the page is stuck.
            self._probe_failures += 1
            if self._probe_failures >= _STALL_PROBES and not self._stalled:
                self._stalled = True
                self._note_phase("stalled", self._media)
            return
        self._probe_failures = 0

        media = self._media = probe.media
        now = time.monotonic()
        ended = media is not None and media.at_end and not probe.playing

        position = media.position if media is not None else None
        previous = self._last_position
        self._last_position = position
        if self._paused:
            # Held media does not move, so a stall cannot be measured;
            # one seen before the hold stays on record until it lifts.
            stalled = self._stalled and not ended
        else:
            should_advance = (
                media is not None
                and position is not None
                and media.paused is False
                and not ended
            )
            if not should_advance:
                self._stall_probes = 0
            elif previous is not None:
                # Backwards counts too: a loop or the next playlist item.
                if abs(position - previous) > _PROGRESS_EPSILON:
                    self._stall_probes = 0
                else:
                    self._stall_probes += 1
            stalled = self._stall_probes >= _STALL_PROBES

        if not ended:
            self._ended_since = None
        elif self._ended_since is None or self._paused:
            # The grace before closing does not run while BB holds the
            # view for a conversation ("play that again").
            self._ended_since = now
        self._ended = ended
        self._stalled = stalled

        if ended:
            self._note_phase("ended", media)
        elif stalled:
            self._note_phase("stalled", media)
        elif probe.playing and not self._paused:
            self._note_phase("playing", media)

    def _note_phase(self, phase: str, media: MediaProbe | None) -> None:
        """Log a change between playing / stalled / ended.

        Numbers only: every value was validated by :func:`parse_probe`.
        """
        previous, self._phase = self._phase, phase
        if previous == phase:
            return
        suppressed = self._playback_log.take()
        if suppressed is None:
            return
        m = media if media is not None else MediaProbe()
        logger.info(
            "Web view playback %s -> %s (pos=%s dur=%s ready=%s net=%s "
            "err=%s frames=%s dropped=%s)%s",
            previous or "start", phase,
            None if m.position is None else round(m.position, 1),
            None if m.duration is None else round(m.duration, 1),
            m.ready_state, m.network_state, m.error_code, m.frames, m.dropped,
            f" [+{suppressed} changes not logged]" if suppressed else "",
        )

    def _on_target_crashed(self, params: dict[str, Any], session_id: str | None) -> None:
        """DevTools says the page's renderer died. The browser process
        is still up, showing its own crash page: close the view."""
        if self._closing or not self.is_open:
            return
        if session_id is not None:
            if session_id != self._session_id:          # Inspector.targetCrashed
                return
        elif params.get("targetId") != self._page_target_id:   # Target.targetCrashed
            return
        code = _number(params.get("errorCode"), -2 ** 31, 2 ** 31)
        if self._loading:
            # A page load holds the lock and has already taken the next
            # epoch. The event may be the page being left dying late:
            # closing under the new epoch would kill the page being
            # loaded for it. The load checks the new page itself.
            self._crashed_in_load = True
            logger.warning(
                "Web view page crashed during a page load (renderer exit "
                "code %s); checking the new page", code,
            )
            return
        logger.warning(
            "Web view page crashed (renderer exit code %s); closing", code,
        )
        # Not loading, so this epoch is the page the event is about.
        self._spawn(self._close_self(self._epoch, EXIT_CRASHED))

    async def _evaluate(self, expression: str) -> Any:
        """Run ``expression`` in the page. Best-effort: ``None`` on failure.

        The result comes from the page's own JavaScript world, so a
        hostile page can lie in it. Callers must validate it before use
        (:func:`parse_probe`, :func:`_count`); it never reaches the
        agent as it came.
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
            self._note_evaluate_failure(cdp, str(e))
            return None
        if "exceptionDetails" in result:
            # The details are page-written text: not logged.
            self._note_evaluate_failure(cdp, "script raised in the page")
            return None
        value = result.get("result")
        return value.get("value") if isinstance(value, dict) else None

    def _note_evaluate_failure(self, cdp: CDPPipe, detail: str) -> None:
        """A page that stops answering is how a stall or a hung renderer
        first shows: worth a WARNING, but not one every quarter second."""
        if self._closing or cdp is not self._cdp or cdp.closed:
            logger.debug("web view evaluate failed during close: %s", detail)
            return
        suppressed = self._evaluate_log.take()
        if suppressed is None:
            return
        logger.warning(
            "web view evaluate failed: %s%s", detail,
            f" [+{suppressed} more not logged]" if suppressed else "",
        )

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
        for event in ("Inspector.targetCrashed", "Target.targetCrashed"):
            self._cdp.on_event(event, self._on_target_crashed)
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
            self._page_target_id = page_id if isinstance(page_id, str) else None
            await self._cdp.call("Page.enable", session_id=self._session_id)
            # For Inspector.targetCrashed. Optional: the browser-level
            # Target.targetCrashed covers a browser that refuses it.
            with contextlib.suppress(CDPError):
                await self._cdp.call("Inspector.enable", session_id=self._session_id)
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
        """Read the browser's stderr for as long as it runs.

        Read continuously so the pipe can never fill and stall the
        browser. Each line is cleaned (:func:`_clean_log_line`) and then
        goes two places:

        - the last few are kept for a launch-failure message ("Missing
          X server", sandbox errors);
        - all of them go to the main log at INFO under
          ``boxbot.displays.web_view.browser``, at most 30 a minute —
          GPU, codec and network errors are the only record of why a
          video stopped. The count of dropped lines is logged with the
          next one that gets through.
        """
        assert proc.stderr is not None
        while True:
            try:
                line = await proc.stderr.readline()
            except ValueError:
                # A line longer than the reader's buffer. The reader has
                # discarded it; stopping here would let the pipe fill.
                continue
            except (asyncio.CancelledError, OSError):
                return
            if not line:
                return
            text = _clean_log_line(line)
            if not text:
                continue
            self._stderr_tail.append(text)
            suppressed = self._stderr_log.take()
            if suppressed is None:
                continue
            if suppressed:
                browser_logger.info("(%d lines dropped by the rate limit)", suppressed)
            browser_logger.info("%s", text)

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
        """Start the hard limit over. open() only — never a reload."""
        if self._limit_task is not None:
            self._limit_task.cancel()
        self._limit_at = time.monotonic() + self._config.max_minutes * 60
        self._limit_task = asyncio.create_task(
            self._enforce_limit(), name="web-view-limit",
        )

    async def _enforce_limit(self) -> None:
        """Close the view after ``max_minutes`` — a forgotten video must
        not play all night. Not tied to a page epoch: reloads do not
        move the deadline, and an open() replaces this task."""
        while (remaining := self._limit_at - time.monotonic()) > 0:
            await asyncio.sleep(remaining)
        logger.info("Web view reached its %g-minute limit; closing",
                    self._config.max_minutes)
        await self._close_self(None, EXIT_LIMIT)

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
        self._page_target_id = None
        self._target = None
        self._reset_page_state()
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
