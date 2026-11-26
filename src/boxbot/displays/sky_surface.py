"""Sky surface — the Chromium window that *is* the screen (HTML renderer).

With ``display.renderer: html`` the display manager stops drawing PIL
frames. One browser window shows the bundled Living Sky page
(``displays/sky/index.html``) and the manager pushes state into it:
``BB.apply({...})`` over the DevTools pipe, once per change. The page
draws the scene, the blocks, the voice-state dock and the status pill,
and ticks its own clocks.

This is not the ``web`` display (:mod:`boxbot.displays.web_view`). That
browser shows the open internet and is fenced accordingly: its own OS
user, a kernel network fence, a home wiped per launch. This one shows
only files shipped in the package, so it runs as the service user, and
is closed off the other way round — it cannot reach anything:

- **No network.** Every hostname resolves to nothing and every request
  is routed to a dead proxy; the page's CSP allows ``file:`` only.
- **No listener.** Control is the DevTools pipe on inherited fds,
  never a TCP port.
- **No script from data.** State is JSON. The page escapes every value
  it prints and its CSP refuses inline script, so a calendar title
  cannot become code.
- **Fresh profile per launch**, under ``display.html.state_dir``.

A second, headless instance of the same class renders previews for the
agent (``bb.display.preview``) so authoring never touches the window on
the wall.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import fcntl
import json
import logging
import os
import shutil
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from boxbot.displays.cdp_pipe import CDPError, CDPPipe

if TYPE_CHECKING:
    from boxbot.core.config import SkySurfaceConfig

logger = logging.getLogger(__name__)

PAGE_DIR = Path(__file__).parent / "sky"

# One pushed state. Layout + data for a busy week is ~50 KB; a pushed
# camera frame (JPEG data URL) ~150 KB. The pipe is sized so a whole
# message fits in the kernel buffer and the non-blocking write never
# tears (see CDPPipe.call).
MAX_STATE_BYTES = 768 * 1024
_PIPE_BYTES = 1024 * 1024

# Hands the browser its DevTools pipe on fds 3 and 4, where
# ``--remote-debugging-pipe`` expects it, then becomes the browser.
# argv[1]/argv[2] are the inherited fd numbers (kept >= 10 so they can
# never be 3 or 4 themselves). Python rather than ``sh``: dash cannot
# redirect from a two-digit fd.
_EXEC_SCRIPT = (
    "import os,sys;os.dup2(int(sys.argv[1]),3);os.dup2(int(sys.argv[2]),4);"
    "os.execv(sys.argv[3],sys.argv[3:])"
)

_RESTART_DELAYS = (1.0, 2.0, 5.0, 15.0, 30.0)

# Resolves once fonts are in and every image has decoded (or failed):
# a screenshot before that catches fallback type and empty photos.
_SETTLED_JS = (
    "(async()=>{await document.fonts.ready;"
    "await Promise.all([...document.images].map(i=>i.decode().catch(()=>{})));"
    "await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));"
    "return true})()"
)


class SkySurfaceError(Exception):
    """The surface could not start or did not answer."""


class SkySurface:
    """Owns one browser showing the Living Sky page.

    ``submit`` is the hot path: it only records the newest state and
    wakes the pump, so callers on the render path never wait on the
    browser. If the browser dies it is restarted with backoff and the
    last state is pushed again.
    """

    def __init__(
        self,
        config: "SkySurfaceConfig",
        *,
        size: tuple[int, int] = (1024, 600),
        headless: bool | None = None,
        name: str = "screen",
    ) -> None:
        self._config = config
        self._size = size
        self._headless = config.headless if headless is None else headless
        self._name = name

        self._proc: asyncio.subprocess.Process | None = None
        self._cdp: CDPPipe | None = None
        self._fds: list[int] = []
        self._session_id: str | None = None
        self._stderr_path: Path | None = None

        self._wanted = False
        self._ready = asyncio.Event()
        self._supervisor: asyncio.Task[None] | None = None
        self._pump: asyncio.Task[None] | None = None
        self._dirty = asyncio.Event()
        self._state: dict[str, Any] | None = None
        self._pushed = 0  # states delivered (tests, diagnostics)
        self._lock = asyncio.Lock()  # serialises DevTools calls that must not interleave

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._ready.is_set()

    @property
    def pushed(self) -> int:
        return self._pushed

    async def start(self) -> None:
        """Launch the browser and wait for the page.

        Raises :class:`SkySurfaceError` if the first launch fails; later
        crashes are handled by the supervisor.
        """
        if self._wanted:
            return
        self._wanted = True
        try:
            await self._launch()
        except BaseException:
            self._wanted = False
            await self._teardown()
            raise
        self._supervisor = asyncio.create_task(
            self._supervise(), name=f"sky-{self._name}-supervisor",
        )
        self._pump = asyncio.create_task(
            self._pump_loop(), name=f"sky-{self._name}-pump",
        )

    async def stop(self) -> None:
        self._wanted = False
        for task in (self._supervisor, self._pump):
            if task is not None:
                task.cancel()
        for task in (self._supervisor, self._pump):
            if task is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        self._supervisor = self._pump = None
        await self._teardown()

    # ------------------------------------------------------------------
    # State in
    # ------------------------------------------------------------------

    def submit(self, state: dict[str, Any]) -> None:
        """Show ``state``. Latest wins; never blocks."""
        self._state = state
        self._dirty.set()

    async def apply(self, state: dict[str, Any]) -> None:
        """Show ``state`` and wait until the page has taken it."""
        self._state = state
        self._dirty.clear()
        await self._push(state)

    async def screenshot(self, *, settle: bool = True) -> bytes:
        """PNG of what the page shows right now."""
        async with self._lock:
            cdp, session = self._require()
            width, height = self._size
            try:
                if settle:
                    await cdp.call(
                        "Runtime.evaluate",
                        {"expression": _SETTLED_JS, "awaitPromise": True,
                         "returnByValue": True},
                        session_id=session, timeout=15.0,
                    )
                result = await cdp.call(
                    "Page.captureScreenshot",
                    {"format": "png", "clip": {
                        "x": 0, "y": 0, "width": width, "height": height, "scale": 1,
                    }},
                    session_id=session, timeout=20.0,
                )
            except CDPError as e:
                raise SkySurfaceError(f"screenshot failed: {e}") from e
            return base64.b64decode(result.get("data", ""))

    async def render(self, state: dict[str, Any]) -> bytes:
        """Push ``state`` and return a PNG of the result (previews)."""
        await self.apply(state)
        return await self.screenshot()

    async def evaluate(self, expression: str) -> Any:
        """Run ``expression`` in the page and return its JSON value.

        For tests and diagnostics; nothing in the product path builds
        an expression from data.
        """
        async with self._lock:
            cdp, session = self._require()
            return await self._eval(cdp, session, expression)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require(self) -> tuple[CDPPipe, str]:
        if not self._ready.is_set() or self._cdp is None or self._session_id is None:
            raise SkySurfaceError("display surface is not running")
        return self._cdp, self._session_id

    @staticmethod
    async def _eval(cdp: CDPPipe, session: str, expression: str,
                    timeout: float = 10.0) -> Any:
        try:
            result = await cdp.call(
                "Runtime.evaluate",
                {"expression": expression, "returnByValue": True, "awaitPromise": True},
                session_id=session, timeout=timeout,
            )
        except CDPError as e:
            raise SkySurfaceError(str(e)) from e
        if "exceptionDetails" in result:
            detail = result["exceptionDetails"]
            text = (detail.get("exception") or {}).get("description") or detail.get("text")
            raise SkySurfaceError(f"page error: {text}")
        return (result.get("result") or {}).get("value")

    async def _push(self, state: dict[str, Any]) -> None:
        # ensure_ascii: the payload is spliced into a JS expression, and
        # pure ASCII JSON is a valid JS literal with no exceptions.
        payload = json.dumps(state, ensure_ascii=True, separators=(",", ":"), default=str)
        if len(payload) > MAX_STATE_BYTES:
            # Raw source data is a convenience for blocks; the layout
            # already carries every bound value.
            slim = {**state, "data": {}}
            payload = json.dumps(slim, ensure_ascii=True, separators=(",", ":"), default=str)
            if len(payload) > MAX_STATE_BYTES:
                raise SkySurfaceError(
                    f"display state is {len(payload)} bytes (limit {MAX_STATE_BYTES})"
                )
        async with self._lock:
            cdp, session = self._require()
            await self._eval(cdp, session, f"BB.apply({payload})")
        self._pushed += 1

    async def _pump_loop(self) -> None:
        while True:
            await self._dirty.wait()
            await self._ready.wait()
            self._dirty.clear()
            state = self._state
            if state is None:
                continue
            try:
                await self._push(state)
            except SkySurfaceError as e:
                if self._ready.is_set():
                    logger.warning("display state push failed: %s", e)
                    await asyncio.sleep(1.0)
                # Not ready: the supervisor re-pushes after the restart.

    async def _supervise(self) -> None:
        """Restart the browser whenever it exits while still wanted."""
        failures = 0
        while self._wanted:
            proc = self._proc
            if proc is not None:
                code = await proc.wait()
                if not self._wanted:
                    return
                logger.warning(
                    "Display browser (%s) exited (code %s): %s",
                    self._name, code, self._stderr_tail(),
                )
            await self._teardown()
            delay = _RESTART_DELAYS[min(failures, len(_RESTART_DELAYS) - 1)]
            await asyncio.sleep(delay)
            if not self._wanted:
                return
            try:
                await self._launch()
            except SkySurfaceError as e:
                failures += 1
                logger.error("Display browser (%s) restart failed: %s", self._name, e)
                await self._teardown()
                continue
            failures = 0
            if self._state is not None:
                self._dirty.set()

    def _argv(self, read_fd: int, write_fd: int, profile: Path) -> list[str]:
        cfg = self._config
        browser = shutil.which(cfg.browser) or (
            cfg.browser if os.path.isabs(cfg.browser) and os.access(cfg.browser, os.X_OK)
            else None
        )
        if browser is None:
            raise SkySurfaceError(f"browser '{cfg.browser}' is not installed")
        page = PAGE_DIR / "index.html"
        if not page.is_file():
            raise SkySurfaceError(f"display page missing: {page}")
        width, height = self._size
        flags = [
            "--remote-debugging-pipe",
            f"--user-data-dir={profile}",
            f"--window-size={width},{height}",
            "--window-position=0,0",
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
            "--mute-audio",
            "--overscroll-history-navigation=0",
            "--block-new-web-contents",
            # Offline twice over: no name resolves, and anything that
            # tries an address anyway is sent to a port nobody listens on.
            "--host-resolver-rules=MAP * ~NOTFOUND",
            "--proxy-server=http://127.0.0.1:9",
            "--proxy-bypass-list=<-loopback>",
        ]
        if self._headless:
            flags += ["--headless=new", "--hide-scrollbars"]
        else:
            flags.append("--kiosk")
            if "WAYLAND_DISPLAY" in self._session_env():
                # Explicit: a service started by systemd has no
                # XDG_SESSION_TYPE for "auto" to read, and Chromium
                # would fall back to X11 and exit.
                flags.append("--ozone-platform=wayland")
        return [
            sys.executable, "-c", _EXEC_SCRIPT, str(read_fd), str(write_fd),
            browser, *flags, *cfg.extra_args, page.as_uri() + ("?still" if self._headless else ""),
        ]

    @staticmethod
    def _session_env() -> dict[str, str]:
        """Where the compositor is, for a process whose own environment
        may not say (same defaults as the ``web`` display's browser)."""
        runtime_dir = (
            os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
        ).rstrip("/")
        wayland = os.environ.get("WAYLAND_DISPLAY") or "wayland-0"
        socket = wayland if os.path.isabs(wayland) else f"{runtime_dir}/{wayland}"
        if not os.path.exists(socket):
            return {}
        return {"XDG_RUNTIME_DIR": runtime_dir, "WAYLAND_DISPLAY": wayland}

    @staticmethod
    def _high_fd(fd: int) -> int:
        """Move ``fd`` to a number >= 10 (see ``_EXEC_SCRIPT``)."""
        high = fcntl.fcntl(fd, fcntl.F_DUPFD, 10)
        os.close(fd)
        return high

    async def _launch(self) -> None:
        cfg = self._config
        state_dir = Path(cfg.state_dir)
        profile = state_dir / f"profile-{self._name}"
        try:
            # Nothing in the profile is worth keeping, and a stale one
            # brings "restore pages?" bubbles after an unclean stop.
            await asyncio.to_thread(shutil.rmtree, profile, True)
            profile.mkdir(parents=True, exist_ok=True)
            os.chmod(profile, 0o700)
        except OSError as e:
            raise SkySurfaceError(f"cannot prepare {profile}: {e}") from e

        # to_browser: we write, the browser reads on fd 3.
        # from_browser: the browser writes on fd 4, we read.
        to_r, to_w = os.pipe()
        from_r, from_w = os.pipe()
        child_r, child_w = self._high_fd(to_r), self._high_fd(from_w)
        self._fds = [to_w, from_r]
        try:
            with contextlib.suppress(OSError):
                fcntl.fcntl(to_w, fcntl.F_SETPIPE_SZ, _PIPE_BYTES)
            os.set_blocking(to_w, False)
            argv = self._argv(child_r, child_w, profile)
            logger.info(
                "Launching display browser (%s%s)", self._name,
                ", headless" if self._headless else "",
            )
            # The browser's own diagnostics ("Missing X server", sandbox
            # errors) go to a file, rewritten per launch: a pipe would
            # need draining for as long as any helper process lives.
            self._stderr_path = state_dir / f"browser-{self._name}.log"
            with open(self._stderr_path, "wb") as stderr:
                self._proc = await asyncio.create_subprocess_exec(
                    *argv,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=stderr,
                    pass_fds=(child_r, child_w),
                    start_new_session=True,
                    env={**os.environ, **({} if self._headless else self._session_env())},
                )
        except OSError as e:
            raise SkySurfaceError(f"could not start browser: {e}") from e
        finally:
            for fd in (child_r, child_w):
                with contextlib.suppress(OSError):
                    os.close(fd)

        self._cdp = CDPPipe(to_w, from_r)
        self._cdp.start()

        timeout = float(cfg.launch_timeout)
        try:
            targets = await self._cdp.call("Target.getTargets", timeout=timeout)
            page_id = next(
                (t.get("targetId") for t in targets.get("targetInfos", [])
                 if isinstance(t, dict) and t.get("type") == "page"),
                None,
            )
            if page_id is None:
                raise CDPError("browser opened no page")
            attached = await self._cdp.call(
                "Target.attachToTarget", {"targetId": page_id, "flatten": True},
                timeout=timeout,
            )
            session = attached.get("sessionId")
            if not session:
                raise CDPError("browser did not return a page session")
            await self._cdp.call("Page.enable", session_id=session)
            await self._cdp.call("Browser.setDownloadBehavior", {"behavior": "deny"})
            if self._headless:
                # Headless windows lose part of their height to phantom
                # chrome; pin the viewport to the panel's exact size.
                width, height = self._size
                await self._cdp.call(
                    "Emulation.setDeviceMetricsOverride",
                    {"width": width, "height": height, "deviceScaleFactor": 1,
                     "mobile": False},
                    session_id=session,
                )
            await self._wait_page(session, timeout)
        except (CDPError, SkySurfaceError) as e:
            await asyncio.sleep(0.05)  # let the last stderr lines land
            raise SkySurfaceError(
                f"display browser did not come up: {e} ({self._stderr_tail()})"
            ) from e
        self._session_id = session
        self._ready.set()

    async def _wait_page(self, session: str, timeout: float) -> None:
        assert self._cdp is not None
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            try:
                if await self._eval(
                    self._cdp, session,
                    "typeof BB==='object'&&BB.ready===true", timeout=5.0,
                ):
                    return
            except SkySurfaceError:
                pass  # page still loading
            if self._proc is not None and self._proc.returncode is not None:
                raise SkySurfaceError(f"browser exited (code {self._proc.returncode})")
            if asyncio.get_running_loop().time() > deadline:
                raise SkySurfaceError("page did not become ready")
            await asyncio.sleep(0.1)

    def _stderr_tail(self, lines: int = 6) -> str:
        """The last few lines the browser wrote, for an error message."""
        if self._stderr_path is None:
            return "no output"
        try:
            data = self._stderr_path.read_bytes()[-4096:]
        except OSError:
            return "no output"
        tail = [t.strip()[:300] for t in data.decode(errors="replace").splitlines() if t.strip()]
        return " | ".join(tail[-lines:]) or "no output"

    async def _teardown(self) -> None:
        self._ready.clear()
        self._session_id = None
        proc, cdp = self._proc, self._cdp
        self._proc = self._cdp = None
        if proc is not None and proc.returncode is None:
            if cdp is not None:
                with contextlib.suppress(CDPError):
                    await cdp.call("Browser.close", timeout=2.0)
            for signum, wait in ((None, 3.0), (signal.SIGTERM, 3.0), (signal.SIGKILL, 2.0)):
                if signum is not None:
                    with contextlib.suppress(ProcessLookupError, PermissionError):
                        os.killpg(proc.pid, signum)
                try:
                    await asyncio.wait_for(proc.wait(), wait)
                    break
                except asyncio.TimeoutError:
                    continue
        if cdp is not None:
            cdp.close()
        for fd in self._fds:
            with contextlib.suppress(OSError):
                os.close(fd)
        self._fds = []
