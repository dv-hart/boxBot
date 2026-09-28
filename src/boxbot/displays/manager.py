"""Display manager for boxBot.

Orchestrates the display lifecycle: loading display specs, switching between
displays, managing data source refresh cycles, coordinating rendering, and
handling idle rotation. This is the main entry point for the display system.

The display manager:
- Maintains a registry of available displays (built-in + user-created + agent-created)
- Handles display switching via the `switch_display` tool or `DisplaySwitch` events
- Manages data source refresh cycles as async background tasks
- Coordinates with the renderer to produce frames
- Handles display-specific args (e.g. `picture` display with `image_ids`)
- Manages idle display rotation on a configurable timer
- Provides a thread-safe frame buffer for the screen HAL to consume

Usage:
    from boxbot.displays.manager import DisplayManager

    mgr = DisplayManager()
    await mgr.start()
    await mgr.switch("clock")
    frame = mgr.get_current_frame()  # thread-safe, called from screen HAL
    await mgr.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging

# Module-level accessor so sandbox action handlers (bb.photos.show_on_screen,
# bb.display.*) can reach the running DisplayManager without DI plumbing.
# ``boxbot.core.main`` calls ``set_display_manager`` once at startup.
_display_manager_instance: "DisplayManager | None" = None


def get_display_manager() -> "DisplayManager | None":
    return _display_manager_instance


def set_display_manager(mgr: "DisplayManager | None") -> None:
    global _display_manager_instance
    _display_manager_instance = mgr


# ---------------------------------------------------------------------------
# Rotation state persistence
# ---------------------------------------------------------------------------


def _rotation_state_path() -> "Path":
    """Resolve ``data/displays/rotation.json``.

    Anchored to ``boxbot.core.paths.DISPLAYS_DIR`` so the resolved
    path is independent of cwd and honors ``BOXBOT_DATA_DIR`` overrides
    (tests, alternate deployments).
    """
    from boxbot.core.paths import DISPLAYS_DIR
    return DISPLAYS_DIR / "rotation.json"


def _load_rotation_state() -> dict | None:
    """Read persisted rotation state, or None if absent / malformed.

    Returns a dict ``{"displays": list[str], "interval": int}`` on
    success. Missing file, bad JSON, or shape mismatch all return
    ``None`` so the caller falls back to config — never raises so a
    corrupt file can't take down the display manager.
    """
    import json as _json
    path = _rotation_state_path()
    if not path.is_file():
        return None
    try:
        data = _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, _json.JSONDecodeError) as exc:
        logger.warning("Could not read rotation state at %s: %s", path, exc)
        return None
    displays = data.get("displays")
    interval = data.get("interval")
    if not isinstance(displays, list) or not all(isinstance(d, str) for d in displays):
        return None
    if not isinstance(interval, int) or interval < 1:
        return None
    return {"displays": displays, "interval": interval}


def _persist_rotation_state(state: dict | None) -> None:
    """Write rotation state to disk (or delete when ``state`` is None).

    Atomically replaces the file via tmp+rename so a partial write
    can't corrupt the persisted record. Best-effort: I/O failures are
    logged but don't fail the SDK call.
    """
    import json as _json
    path = _rotation_state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if state is None:
            if path.exists():
                path.unlink()
            return
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(_json.dumps(state, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        logger.warning("Could not persist rotation state to %s: %s", path, exc)
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image

from boxbot.displays.blocks import Block
from boxbot.displays.data_sources import (
    DataSourceManager,
    StaticSource,
    create_source,
    placeholder_for_source,
    integration_for_source,
)
from boxbot.displays.renderer import DisplayRenderer
from boxbot.displays.spec import (
    DisplaySpec,
    parse_spec,
    resolve_bindings,
    validate_spec,
)
from boxbot.core.paths import DISPLAYS_DIR, PROJECT_ROOT
from boxbot.displays.themes import Theme, get_theme

logger = logging.getLogger(__name__)

# Repaint pump cadence — see _live_tick_loop. The tick is the floor on
# how often anything re-renders; _SLOW_LIVE_SECONDS is the cadence for a
# live block that shows no seconds (a minute clock).
_TICK_SECONDS = 1.0
_SLOW_LIVE_SECONDS = 60.0

# Status pill watchdog: a crashed turn must not strand "Searching the
# web…" on screen forever. Every AgentToolCalled refreshes the deadline;
# 90s outlives any sane single tool call (sandbox scripts included).
_STATUS_TTL_SECONDS = 90.0

# Default paths
_BUILTINS_DIR = Path(__file__).parent / "builtins"
_USER_DISPLAYS_DIR = PROJECT_ROOT / "displays"
_AGENT_DISPLAYS_DIR = DISPLAYS_DIR


def _draw_status_pill(base: Image.Image, text: str) -> Image.Image:
    """Composite the transient agent-status pill onto a rendered frame.

    Bottom-center rounded pill — slate surface, orange accent dot,
    theme-white text — sized relative to frame width so it reads the
    same at 1024x600 and 1280x800. Returns a new RGB
    image; ``base`` is untouched.
    """
    from PIL import ImageDraw

    from boxbot.displays.renderer import _get_font

    w, h = base.size
    font_size = max(16, round(w * 0.020))
    pad_x = round(font_size * 1.2)
    pad_y = round(font_size * 0.65)
    dot_r = max(3, round(font_size * 0.22))
    dot_gap = round(font_size * 0.6)

    font = _get_font("Inter", font_size, 500)
    frame = base.convert("RGBA")
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    pill_w = text_w + dot_r * 2 + dot_gap + pad_x * 2
    pill_h = text_h + pad_y * 2
    x0 = (w - pill_w) // 2
    y1 = h - round(h * 0.04)
    y0 = y1 - pill_h

    # Slate surface (#1d252d) at ~88%, hairline highlight, signal-orange
    # dot (#f25e0f), near-white text (#f0f2f3) — THEME_BOXBOT palette.
    draw.rounded_rectangle(
        (x0, y0, x0 + pill_w, y1),
        radius=pill_h // 2,
        fill=(29, 37, 45, 224),
        outline=(255, 255, 255, 28),
        width=1,
    )
    cy = (y0 + y1) // 2
    dot_x = x0 + pad_x + dot_r
    draw.ellipse(
        (dot_x - dot_r, cy - dot_r, dot_x + dot_r, cy + dot_r),
        fill=(242, 94, 15, 255),
    )
    draw.text(
        (dot_x + dot_r + dot_gap, cy - text_h // 2 - bbox[1]),
        text,
        font=font,
        fill=(240, 242, 243, 245),
    )

    frame.alpha_composite(overlay)
    return frame.convert("RGB")


class DisplayManager:
    """Manages display lifecycle, data sources, rendering, and frame output.

    The display manager is the central coordinator for boxBot's visual output.
    It handles:

    - **Discovery**: Loads display specs from built-in, user-contributed, and
      agent-created directories at startup.
    - **Switching**: Responds to `switch_display` tool calls and `DisplaySwitch`
      events on the event bus. Tears down old data sources and sets up new ones.
    - **Data refresh**: Runs async background tasks that fetch data on each
      source's configured interval. When data changes, triggers a re-render.
    - **Rendering**: Uses `DisplayRenderer` to produce PIL Images from resolved
      display specs. Handles binding resolution and theme application.
    - **Frame buffer**: Maintains a thread-safe current frame that the screen
      HAL reads from its own thread (pygame display loop).
    - **Rotation**: Cycles through configured idle displays on a timer when
      the agent is not actively controlling the display.
    - **Night mode**: Adjusts brightness based on time-of-day configuration.

    The display manager subscribes to the event bus at start() and unsubscribes
    at stop(). All display switches funnel through the same code path regardless
    of whether they come from the tool, event bus, or rotation timer.
    """

    def __init__(
        self,
        width: int = 1024,
        height: int = 600,
    ) -> None:
        """Initialize the display manager.

        Args:
            width: Display width in pixels. Defaults to 1024.
            height: Display height in pixels. Defaults to 600.
        """
        # Display specs registry
        self._specs: dict[str, DisplaySpec] = {}

        # Active display state
        self._active_display: str | None = None
        self._active_args: dict[str, Any] = {}
        self._active_theme: Theme | None = None
        self._active_spec: DisplaySpec | None = None

        # Pin flag: when True, the active display was chosen explicitly
        # by the agent (or another caller) and idle rotation is paused.
        # Cleared by ``unpin()`` or by ``start_rotation()``.
        self._pinned: bool = False
        self._last_switch_ts: float = 0.0

        # Slideshow tick (picture display with len(image_ids) > 1).
        # Lives next to the active display state because it's a
        # per-active-display behavior, not a global rotation.
        self._slideshow_task: asyncio.Task[None] | None = None

        # Data source management. Fresh data marks the screen dirty and
        # _live_tick_loop repaints it — see _on_source_update.
        self._data_manager = DataSourceManager(on_update=self._on_source_update)
        self._source_dirty = False

        # Rendering
        self._renderer = DisplayRenderer(width=width, height=height)
        self._width = width
        self._height = height

        # Thread-safe frame buffer for screen HAL consumption
        self._frame_lock = threading.Lock()
        self._current_frame: Image.Image | None = None
        self._frame_generation: int = 0  # Incremented on each new frame

        # Transient agent-status pill ("Searching the web…") composited
        # over whatever is on screen while the agent works a turn.
        # _base_frame is the un-overlaid render so the pill can change
        # or clear without re-rendering the display. Driven by
        # AgentToolCalled events; cleared on AgentSpeaking /
        # AgentTurnEnded / ConversationEnded and by the TTL watchdog.
        self._base_frame: Image.Image | None = None
        self._status_text: str | None = None
        self._status_deadline: float = 0.0

        # Rotation state
        self._rotation_task: asyncio.Task[None] | None = None
        self._rotation_displays: list[str] = []
        self._rotation_interval: int = 30
        self._rotation_index: int = 0
        self._rotation_active: bool = False

        # Background refresh state
        self._refresh_task: asyncio.Task[None] | None = None
        self._live_tick_task: asyncio.Task[None] | None = None
        self._running: bool = False

        # Data change callback tracking
        self._last_data_hash: str = ""

        # Event bus subscription tracking
        self._event_subscribed: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(
        self,
        builtins_dir: str | Path | None = None,
        user_dir: str | Path | None = None,
        agent_dir: str | Path | None = None,
    ) -> None:
        """Start the display manager.

        Loads display specs, subscribes to the event bus, and starts
        background tasks. Call this once at application startup.

        Args:
            builtins_dir: Path to built-in display specs.
            user_dir: Path to user-contributed displays.
            agent_dir: Path to agent-created displays.
        """
        self._running = True

        # Load display specs from all directories
        await self.load_displays(builtins_dir, user_dir, agent_dir)

        # Subscribe to DisplaySwitch events on the event bus
        self._subscribe_events()

        # Start background live-tick task for clock/countdown blocks
        self._live_tick_task = asyncio.create_task(
            self._live_tick_loop(),
            name="display-live-tick",
        )

        logger.info(
            "Display manager started (%d display(s) loaded)", len(self._specs)
        )

    async def stop(self) -> None:
        """Stop the display manager and clean up all resources.

        Stops rotation, data source fetching, background tasks, and
        unsubscribes from the event bus.
        """
        self._running = False

        # Stop rotation
        self.stop_rotation()

        # Stop slideshow tick if running
        self._stop_slideshow()

        # Stop live tick task
        if self._live_tick_task and not self._live_tick_task.done():
            self._live_tick_task.cancel()
            self._live_tick_task = None

        # Stop data sources
        await self._data_manager.stop_all()
        self._data_manager.clear()

        # Unsubscribe from event bus
        self._unsubscribe_events()

        logger.info("Display manager stopped")

    # ------------------------------------------------------------------
    # Event bus integration
    # ------------------------------------------------------------------

    def _subscribe_events(self) -> None:
        """Subscribe to display-related events on the event bus."""
        if self._event_subscribed:
            return

        try:
            from boxbot.core.events import (
                AgentSpeaking,
                AgentToolCalled,
                AgentTurnEnded,
                ConversationEnded,
                DisplaySwitch,
                get_event_bus,
            )

            bus = get_event_bus()
            bus.subscribe(DisplaySwitch, self._on_display_switch)
            bus.subscribe(AgentToolCalled, self._on_agent_tool_called)
            bus.subscribe(AgentSpeaking, self._on_status_clear)
            bus.subscribe(AgentTurnEnded, self._on_status_clear)
            bus.subscribe(ConversationEnded, self._on_status_clear)
            self._event_subscribed = True
            logger.debug("Subscribed to display events")
        except ImportError:
            logger.debug("Event bus not available, skipping subscription")
        except Exception:
            logger.exception("Failed to subscribe to event bus")

    def _unsubscribe_events(self) -> None:
        """Unsubscribe from the event bus."""
        if not self._event_subscribed:
            return

        try:
            from boxbot.core.events import (
                AgentSpeaking,
                AgentToolCalled,
                AgentTurnEnded,
                ConversationEnded,
                DisplaySwitch,
                get_event_bus,
            )

            bus = get_event_bus()
            bus.unsubscribe(DisplaySwitch, self._on_display_switch)
            bus.unsubscribe(AgentToolCalled, self._on_agent_tool_called)
            bus.unsubscribe(AgentSpeaking, self._on_status_clear)
            bus.unsubscribe(AgentTurnEnded, self._on_status_clear)
            bus.unsubscribe(ConversationEnded, self._on_status_clear)
            self._event_subscribed = False
            logger.debug("Unsubscribed from display events")
        except Exception:
            logger.debug("Failed to unsubscribe from event bus")

    async def _on_display_switch(self, event: Any) -> None:
        """Handle a DisplaySwitch event from the event bus.

        Args:
            event: A DisplaySwitch event with display_name and args.
        """
        display_name = getattr(event, "display_name", "")
        args = getattr(event, "args", {})

        if display_name:
            logger.info(
                "DisplaySwitch event received: '%s' (args=%s)",
                display_name, args,
            )
            await self.switch(display_name, args)

    async def _on_agent_tool_called(self, event: Any) -> None:
        """Show/refresh the status pill for a room-facing tool call.

        Voice-channel only: the screen is in the room with the person
        who is waiting. WhatsApp/Signal turns run for someone who isn't
        looking at the box.
        """
        if getattr(event, "channel", "") != "voice":
            return
        text = getattr(event, "status_text", "")
        if text:
            self.set_status_text(text)

    async def _on_status_clear(self, event: Any) -> None:
        """Drop the status pill — the turn resolved (speech started,
        turn ended, or the conversation closed)."""
        self.clear_status_text()

    # ------------------------------------------------------------------
    # Discovery & loading
    # ------------------------------------------------------------------

    async def load_displays(
        self,
        builtins_dir: str | Path | None = None,
        user_dir: str | Path | None = None,
        agent_dir: str | Path | None = None,
    ) -> None:
        """Discover and load display specs from all sources.

        Scans three directories for display specs:
        1. Built-in displays (shipped with boxBot)
        2. User-contributed displays (dropped into displays/)
        3. Agent-created displays (built via SDK, stored in data/displays/)

        Args:
            builtins_dir: Path to built-in display specs.
            user_dir: Path to user-contributed displays.
            agent_dir: Path to agent-created displays.
        """
        builtins_path = Path(builtins_dir) if builtins_dir else _BUILTINS_DIR
        user_path = Path(user_dir) if user_dir else _USER_DISPLAYS_DIR
        agent_path = Path(agent_dir) if agent_dir else _AGENT_DISPLAYS_DIR

        # Register programmatic built-in displays first
        from boxbot.displays.builtins import get_builtin_specs

        for spec in get_builtin_specs():
            self.register_spec(spec)

        # Load from all directories (file-based specs override builtins)
        self._load_specs_from_dir(builtins_path)
        self._load_specs_from_dir(user_path)
        self._load_specs_from_dir(agent_path)

        logger.info(
            "Loaded %d display(s): %s",
            len(self._specs),
            list(self._specs.keys()),
        )
        self._warn_unknown_integrations()

    def _warn_unknown_integrations(self) -> None:
        """Report integration sources that have no registered integration.

        Without this the source is still created and rediscovers the same
        "unknown integration 'calendar'" on every refresh for the life of
        the process. Once at startup is enough.
        """
        try:
            from boxbot.integrations.loader import discover_integrations

            known = {meta.name for meta in discover_integrations()}
        except Exception:
            logger.debug("Integration registry unavailable; skipping source check")
            return

        for spec in self._specs.values():
            for src in spec.data_sources:
                target = integration_for_source(
                    src.name, src.source_type, src.integration,
                )
                if target is not None and target not in known:
                    logger.warning(
                        "Display '%s' binds source '%s' to unknown "
                        "integration '%s' — it will never have data. "
                        "Registered: %s",
                        spec.name, src.name, target,
                        ", ".join(sorted(known)) or "(none)",
                    )

    def _load_specs_from_dir(self, directory: Path) -> None:
        """Load display specs from a directory.

        Scans for:
        - display.json files inside subdirectories
        - .json files directly in the directory

        Args:
            directory: Path to scan for display specs.
        """
        if not directory.is_dir():
            return

        # Subdirectories with display.json
        for subdir in sorted(directory.iterdir()):
            if subdir.is_dir():
                spec_file = subdir / "display.json"
                if spec_file.exists():
                    self._load_spec_file(spec_file)

        # Top-level JSON files
        for json_file in sorted(directory.glob("*.json")):
            self._load_spec_file(json_file)

    def _load_spec_file(self, path: Path) -> None:
        """Load and validate a single display spec file.

        Args:
            path: Path to the JSON spec file.
        """
        try:
            with open(path) as f:
                data = json.load(f)
            spec = parse_spec(data)
            errors = validate_spec(spec)
            if errors:
                logger.warning(
                    "Display spec '%s' has validation errors: %s", path, errors
                )
            self._specs[spec.name] = spec
            logger.debug("Loaded display spec '%s' from %s", spec.name, path)
        except Exception:
            logger.exception("Failed to load display spec from %s", path)

    def register_spec(self, spec: DisplaySpec) -> None:
        """Register a display spec programmatically.

        Used for built-in displays defined in code rather than JSON files,
        and for runtime registration of agent-created displays.

        Args:
            spec: The display spec to register.
        """
        self._specs[spec.name] = spec
        logger.debug("Registered display spec '%s'", spec.name)

    def unregister_spec(self, name: str) -> bool:
        """Remove a display spec from the registry.

        If the display is currently active, switches away first.

        Args:
            name: Display name to remove.

        Returns:
            True if the spec was found and removed.
        """
        if name not in self._specs:
            return False

        del self._specs[name]
        logger.debug("Unregistered display spec '%s'", name)
        return True

    # ------------------------------------------------------------------
    # Display switching
    # ------------------------------------------------------------------

    async def switch(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        pin: bool = True,
    ) -> bool:
        """Switch to a named display.

        This is the primary method for changing what's on screen. It:
        1. Validates the display name
        2. If pin=True, pauses idle rotation (default for agent calls)
        3. Tears down data sources for the old display
        4. Sets up data sources for the new display
        5. Performs an initial render
        6. Updates the frame buffer

        Args:
            name: Display name to activate.
            args: Display-specific arguments (e.g. {"image_ids": [...]} for
                  the picture display).
            pin: When True (default), mark this as an explicit pin so the
                  rotation loop will not clobber it. Internal callers (the
                  rotation loop) pass ``pin=False``.

        Returns:
            True if the switch succeeded.
        """
        spec = self._specs.get(name)
        if spec is None:
            logger.warning("Display '%s' not found. Available: %s", name, self.list_available())
            return False

        args = dict(args) if args else {}

        # Picture display without explicit image_ids => slideshow mode.
        # Populate the id list from the slideshow-enabled photo set. With
        # nothing to show, fall back to a friendly notice rather than the
        # image block's raw-source placeholder (which rendered a tiny
        # "photo:" on a black screen).
        if name == "picture":
            ids = args.get("image_ids")
            if not isinstance(ids, list) or not ids:
                slideshow_ids = self._load_slideshow_ids()
                if slideshow_ids:
                    args["image_ids"] = slideshow_ids
                elif "notice" in self._specs:
                    logger.info(
                        "picture slideshow requested but no photos in "
                        "rotation; showing empty-state notice",
                    )
                    name = "notice"
                    spec = self._specs["notice"]
                    args = {
                        "title": "No photos yet",
                        "lines": ["Send me a photo and it'll show up here."],
                    }

        # Already on screen with the same spec and args: only the pin
        # state can change. Tearing every data source down and rebuilding
        # it to render an identical frame is pure churn — the rotation
        # loop did exactly that 1,850 times in one boxbot log.
        if (name == self._active_display
                and spec is self._active_spec
                and args == self._active_args):
            if pin and not self._pinned:
                self._pinned = True
                self.stop_rotation()
            return True

        # Explicit pin: stop rotation and mark pinned. Internal calls
        # from the rotation loop pass pin=False to avoid stopping
        # themselves.
        if pin:
            self._pinned = True
            self.stop_rotation()

        # Stop old data sources
        await self._data_manager.stop_all()
        self._data_manager.clear()

        # Resolve and cache the theme
        try:
            self._active_theme = get_theme(spec.theme)
        except KeyError:
            logger.warning(
                "Theme '%s' not found for display '%s', using 'boxbot'",
                spec.theme, name,
            )
            self._active_theme = get_theme("boxbot")

        # Set up new data sources
        await self._setup_data_sources(spec)

        # Update active state. The spec object is kept so a re-save of
        # the same display still rebuilds rather than hitting the no-op.
        self._active_display = name
        self._active_spec = spec
        self._active_args = args

        # Render the initial frame
        self._render_and_update_frame()

        self._last_switch_ts = time.monotonic()

        # If the new display is a multi-photo picture slideshow,
        # start the per-photo tick. Always tear down any prior
        # slideshow first so switching away (or to a single photo)
        # cancels cleanly.
        self._stop_slideshow()
        self._maybe_start_slideshow()

        # Rotation-driven switches are routine; only explicit ones are
        # worth a line in the log.
        logger.log(
            logging.INFO if pin else logging.DEBUG,
            "Switched to display '%s' (args=%s, pinned=%s)",
            name, self._active_args, self._pinned,
        )
        return True

    # ------------------------------------------------------------------
    # Picture-display slideshow
    # ------------------------------------------------------------------

    def _load_slideshow_ids(self, limit: int = 50) -> list[str]:
        """Photo ids in the slideshow rotation, newest first.

        Mirrors the direct-sqlite approach of :func:`_resolve_photo_sources`
        so the manager doesn't depend on a live ``PhotoStore`` connection.
        Best-effort: returns ``[]`` if the photo DB is absent or unreadable
        (a fresh install, or photos disabled).
        """
        try:
            from boxbot.core.config import get_config
            config = get_config()
        except Exception:
            return []

        db_path = Path(config.photos.storage_path) / "photos.db"
        if not db_path.exists():
            return []

        import sqlite3

        try:
            with sqlite3.connect(str(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT id FROM photos "
                    "WHERE in_slideshow = 1 AND deleted_at IS NULL "
                    "ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [r["id"] for r in rows]
        except sqlite3.Error as e:
            logger.warning("slideshow id load failed: %s", e)
            return []

    def _maybe_start_slideshow(self) -> None:
        """Start a per-photo tick when the picture display has >1 id.

        Reads ``interval`` (seconds, default 8) from ``args``. Rotates
        ``args.image_ids`` left on each tick so the display spec
        (``photo:{args.image_ids[0]}``) stays unchanged — only the
        underlying list shifts, and a re-render shows the next photo.
        """
        if self._active_display != "picture":
            return
        ids = self._active_args.get("image_ids")
        if not isinstance(ids, list) or len(ids) < 2:
            return

        raw_interval = self._active_args.get("interval", 8)
        try:
            interval_s = float(raw_interval)
        except (TypeError, ValueError):
            interval_s = 8.0
        interval_s = max(1.0, interval_s)

        self._slideshow_task = asyncio.create_task(
            self._slideshow_loop(interval_s),
            name="picture-slideshow",
        )
        logger.info(
            "Started picture slideshow: %d photos every %.1fs",
            len(ids), interval_s,
        )

    def _stop_slideshow(self) -> None:
        """Cancel any running slideshow tick."""
        if self._slideshow_task and not self._slideshow_task.done():
            self._slideshow_task.cancel()
        self._slideshow_task = None

    async def _slideshow_loop(self, interval: float) -> None:
        """Cycle ``args.image_ids`` on the picture display.

        On each tick: rotate the id list left by one, then re-render
        so the bound source picks up the new ``image_ids[0]``.
        Exits silently if the display is switched away from
        ``picture`` or the id list shrinks below 2.
        """
        try:
            while self._running:
                await asyncio.sleep(interval)
                if self._active_display != "picture":
                    return
                ids = self._active_args.get("image_ids")
                if not isinstance(ids, list) or len(ids) < 2:
                    return
                # Rotate left: [a, b, c] -> [b, c, a]
                self._active_args["image_ids"] = ids[1:] + ids[:1]
                self._render_and_update_frame()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Slideshow loop error")

    async def unpin(self) -> bool:
        """Release the pin and resume idle rotation.

        After ``switch(..., pin=True)`` (the default), the display stays
        put until the agent calls ``unpin()`` or replaces it with
        another ``switch``. ``unpin`` clears the pin flag and restarts
        the rotation loop with the *current* rotation list — whatever
        ``set_rotation`` was last called with, or the config defaults
        if it never was. If the rotation list is empty, the current
        display simply stays on screen with the pin flag cleared.

        Returns:
            True on success.
        """
        self._pinned = False

        # Use the in-memory rotation list, not the config defaults —
        # otherwise unpin silently undoes any prior ``set_rotation``.
        if self._rotation_displays:
            self.start_rotation(
                list(self._rotation_displays),
                self._rotation_interval,
            )
            logger.info("Unpinned; rotation resumed")
        else:
            logger.info("Unpinned; no rotation configured, holding current display")
        return True

    async def set_rotation(
        self,
        displays: list[str] | None = None,
        interval: int | None = None,
    ) -> bool:
        """Configure and start idle rotation.

        Clears the pin flag (the caller is explicitly opting into
        rotation now). ``displays=None`` and ``interval=None`` reuse the
        config defaults. If ``displays`` is an empty list, rotation is
        stopped and the pin remains cleared.

        The chosen ``(displays, interval)`` is persisted to
        ``data/displays/rotation.json`` so it survives a restart;
        startup prefers this file over the config defaults.

        Returns:
            True on success.
        """
        self._pinned = False
        if displays is not None and not displays:
            self.stop_rotation()
            _persist_rotation_state(None)
            logger.info("Rotation cleared")
            return True
        self.start_rotation(displays, interval)
        # Persist after start_rotation so we save the resolved values
        # (None inputs already filled from config, list filtered to
        # registered displays). Saving raw inputs would re-introduce
        # the bug if a stale display name was supplied.
        _persist_rotation_state(
            {
                "displays": list(self._rotation_displays),
                "interval": self._rotation_interval,
            }
        )
        return True

    async def _setup_data_sources(self, spec: DisplaySpec) -> None:
        """Register and start data sources for a display spec.

        Creates DataSource instances from the spec's data source
        declarations and starts their async fetch loops.

        Args:
            spec: The display spec whose data sources to set up.
        """
        for src_spec in spec.data_sources:
            config: dict[str, Any] = {
                "integration": src_spec.integration,
                "inputs": src_spec.inputs,
                "url": src_spec.url,
                "params": src_spec.params,
                "secret": src_spec.secret,
                "refresh": src_spec.refresh,
                "fields": src_spec.fields,
                "value": src_spec.value,
                "query": src_spec.query,
                "limit": src_spec.limit,
            }
            try:
                source = create_source(
                    name=src_spec.name,
                    source_type=src_spec.source_type,
                    config=config,
                )
                self._data_manager.register(source)
            except ValueError as e:
                logger.warning(
                    "Could not create data source '%s': %s", src_spec.name, e
                )

        await self._data_manager.start_all()

    def _on_source_update(self, name: str) -> None:
        """Mark the screen dirty after a fetch changed the data behind it.

        The data manager only holds the active display's sources, so a
        changed fetch is by definition on screen. Without this a
        data-bound display renders once and freezes: the live tick loop
        only ticks clock and countdown blocks, and a single-entry
        rotation has no re-switch timer to hide the staleness.

        A flag, not a render: ``ClockSource`` refreshes once a second and
        its payload changes every tick, so rendering here would put a
        full render + frame push per second behind any display that
        declares it. ``_live_tick_loop`` owns the repaint budget.
        """
        logger.debug(
            "Source '%s' changed; queuing a repaint of '%s'",
            name, self._active_display,
        )
        self._source_dirty = True

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    def start_rotation(
        self,
        displays: list[str] | None = None,
        interval: int | None = None,
    ) -> None:
        """Start idle rotation through a list of displays.

        If displays or interval are not provided, reads from config.
        Only includes displays that are actually registered.

        Args:
            displays: List of display names to rotate through.
            interval: Seconds between display switches.
        """
        # Use config defaults if not specified
        if displays is None or interval is None:
            cfg_displays, cfg_interval = self._get_rotation_config()
            if displays is None:
                displays = cfg_displays
            if interval is None:
                interval = cfg_interval

        # Filter to displays we actually have
        valid = [d for d in displays if d in self._specs]
        if not valid:
            logger.warning(
                "No valid displays for rotation. Requested: %s, Available: %s",
                displays, list(self._specs.keys()),
            )
            return

        self.stop_rotation()
        self._rotation_displays = valid
        self._rotation_interval = interval
        self._rotation_index = 0
        self._rotation_active = True
        # Caller is opting into rotation — clear any prior pin.
        self._pinned = False

        self._rotation_task = asyncio.create_task(
            self._rotation_loop(),
            name="display-rotation",
        )
        logger.info(
            "Started display rotation: %s (every %ds)", valid, interval
        )

    def stop_rotation(self) -> None:
        """Stop the idle rotation timer."""
        self._rotation_active = False
        if self._rotation_task and not self._rotation_task.done():
            self._rotation_task.cancel()
            self._rotation_task = None
            logger.debug("Stopped display rotation")

    async def _rotation_loop(self) -> None:
        """Periodically switch to the next display in the rotation list.

        Runs until cancelled or until the active display gets pinned by
        an external caller. Each iteration switches with ``pin=False``
        so the rotation does not stop itself.
        """
        try:
            while self._running and self._rotation_active:
                if self._pinned:
                    # Defensive: rotation should already be stopped, but
                    # if it somehow wasn't, bail rather than clobber the
                    # pinned display.
                    return
                if self._rotation_displays:
                    name = self._rotation_displays[self._rotation_index]
                    await self.switch(name, pin=False)
                    if len(self._rotation_displays) == 1:
                        # Nothing to rotate between. Live blocks stay
                        # current through _live_tick_loop and data-bound
                        # ones through _on_source_update, so a timer
                        # here would only re-switch clock→clock.
                        self._rotation_active = False
                        logger.debug(
                            "Rotation list holds only '%s'; no timer needed",
                            name,
                        )
                        return
                    self._rotation_index = (
                        (self._rotation_index + 1) % len(self._rotation_displays)
                    )
                await asyncio.sleep(self._rotation_interval)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Display rotation loop error")

    def _get_rotation_config(self) -> tuple[list[str], int]:
        """Get rotation settings — persisted state first, config fallback.

        Persisted state at ``data/displays/rotation.json`` overrides
        config defaults so agent-set rotation lists survive a restart.
        Same "config seeds, runtime DB/state authoritative" pattern as
        the scheduler.

        Returns:
            Tuple of (display_names, interval_seconds).
        """
        persisted = _load_rotation_state()
        if persisted is not None:
            return persisted["displays"], persisted["interval"]
        try:
            from boxbot.core.config import get_config
            config = get_config()
            return (
                config.display.idle_displays,
                config.display.rotation_interval,
            )
        except (RuntimeError, ImportError):
            # Config not loaded or not available
            return (["clock", "weather"], 30)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_active(self) -> str | None:
        """Get the name of the currently active display."""
        return self._active_display

    def get_active_args(self) -> dict[str, Any]:
        """Get the args dict for the currently active display."""
        return dict(self._active_args)

    def get_active_theme(self) -> Theme | None:
        """Get the resolved theme for the currently active display."""
        return self._active_theme

    def list_available(self) -> list[str]:
        """List names of all registered displays, sorted alphabetically."""
        return sorted(self._specs.keys())

    def get_spec(self, name: str) -> DisplaySpec | None:
        """Get a display spec by name.

        Args:
            name: Display name.

        Returns:
            The DisplaySpec, or None if not found.
        """
        return self._specs.get(name)

    def is_rotating(self) -> bool:
        """True if idle rotation is currently active."""
        return self._rotation_active and self._rotation_task is not None

    def is_pinned(self) -> bool:
        """True if the current display was set by an explicit pin."""
        return self._pinned

    def get_rotation_state(self) -> dict[str, Any]:
        """Snapshot of the rotation subsystem for the agent.

        Returns:
            ``{"active": bool, "displays": [...], "interval": int,
              "next_in_sec": float | None}``
            where ``next_in_sec`` is roughly how many seconds remain
            until the next rotation tick (None if rotation isn't
            active).
        """
        active = self.is_rotating()
        next_in: float | None = None
        if active and self._last_switch_ts > 0:
            elapsed = time.monotonic() - self._last_switch_ts
            next_in = max(0.0, self._rotation_interval - elapsed)
        return {
            "active": active,
            "displays": list(self._rotation_displays),
            "interval": self._rotation_interval,
            "next_in_sec": next_in,
        }

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_data(self) -> dict[str, Any]:
        """Get all current data from the active display's data sources.

        Returns:
            Dict mapping source names to their cached data dicts.
        """
        return self._data_manager.get_all_data()

    def update_static_data(
        self,
        display_name: str,
        source_name: str,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        """Update a static data source on any registered display.

        Works whether or not the display is currently active. The spec's
        matching ``data_sources[i].value`` is always mutated in memory so
        the next activation initializes the source with the fresh value;
        if the display is also currently active, the live source
        instance is updated and a re-render is triggered immediately.

        The caller is responsible for persisting the updated spec to disk
        for agent-saved displays so the change survives a process
        restart (the sandbox action handler does this).

        Args:
            display_name: The display that owns the source.
            source_name: The static source name to update.
            value: New data value.

        Returns:
            ``{"ok": True, "live": bool}`` on success, where ``live``
            indicates whether the display was active and the in-memory
            source was updated (vs. only the spec). On failure, returns
            ``{"ok": False, "error": "..."}``.
        """
        spec = self._specs.get(display_name)
        if spec is None:
            return {
                "ok": False,
                "error": f"display '{display_name}' is not registered",
            }

        # Find the matching static source declaration on the spec. The
        # type check happens against the declared type, not a runtime
        # instance, so this works for inactive displays too.
        target = None
        for src in spec.data_sources:
            if src.name == source_name:
                target = src
                break
        if target is None:
            return {
                "ok": False,
                "error": (
                    f"display '{display_name}' has no source named "
                    f"'{source_name}'"
                ),
            }
        if target.source_type != "static":
            return {
                "ok": False,
                "error": (
                    f"source '{source_name}' is type '{target.source_type}', "
                    "not 'static'"
                ),
            }

        # Always update the spec so a later switch sees the new value.
        target.value = value

        # Live update + re-render only when this display is on screen.
        if display_name == self._active_display:
            source = self._data_manager.get_source(source_name)
            if isinstance(source, StaticSource):
                source.update(value)
                self._render_and_update_frame()
                logger.debug(
                    "Updated static data for active '%s.%s'",
                    display_name, source_name,
                )
                return {"ok": True, "live": True}
            # The display claims to be active but its source isn't
            # registered with the data manager — shouldn't happen, but
            # don't lie about live update.
            logger.warning(
                "Active display '%s' missing static source '%s' on data manager",
                display_name, source_name,
            )
            return {"ok": True, "live": False}

        logger.debug(
            "Updated static data on inactive spec '%s.%s'",
            display_name, source_name,
        )
        return {"ok": True, "live": False}

    # ------------------------------------------------------------------
    # Frame buffer (thread-safe)
    # ------------------------------------------------------------------

    def get_current_frame(self) -> Image.Image | None:
        """Get the current rendered frame for the screen HAL.

        This method is thread-safe -- the screen HAL calls it from
        the pygame display thread while the display manager runs in
        the async event loop thread.

        Returns:
            A PIL Image (RGB, 1024x600), or None if no display is active.
        """
        with self._frame_lock:
            return self._current_frame

    def get_frame_generation(self) -> int:
        """Get the frame generation counter.

        The screen HAL can use this to detect when a new frame is
        available without copying the full image. Compare the returned
        value against the last seen generation.

        Returns:
            Monotonically increasing integer, incremented on each new frame.
        """
        with self._frame_lock:
            return self._frame_generation

    @property
    def active_display(self) -> str | None:
        """Name of the display currently on screen, or None."""
        return self._active_display

    @property
    def size(self) -> tuple[int, int]:
        """The render canvas size as ``(width, height)``."""
        return (self._width, self._height)

    def push_frame(self, frame: Image.Image) -> None:
        """Push an externally rendered frame straight to the screen.

        An external frame pump (e.g. a camera live-view source) uses
        this to put decoded video on screen without going through the
        block renderer. The pusher owns the screen only while its placeholder
        display stays active and static — it must check
        :attr:`active_display` each frame and stop when the display
        changes, because any block render (switch, tick repaint) will
        overwrite pushed frames without notice.

        Thread-safe; same contract as the internal render path.
        """
        self._update_frame(frame)

    def _update_frame(self, frame: Image.Image) -> None:
        """Update the frame buffer with a new rendered frame.

        Thread-safe. Called after each render cycle. Keeps the
        un-overlaid render in ``_base_frame`` and re-applies the status
        pill when one is active, so the pill survives live-tick
        repaints and mid-turn display switches.

        Args:
            frame: The new frame image (RGB, 1024x600).
        """
        with self._frame_lock:
            self._base_frame = frame
            self._current_frame = self._composite_status(frame)
            self._frame_generation += 1

    def set_status_text(self, text: str) -> None:
        """Show (or update) the transient agent-status pill.

        Composites over the cached base frame and bumps the generation
        counter — the screen HAL picks it up within one poll (~33 ms).
        No display re-render happens. Refreshes the TTL watchdog even
        when the text is unchanged.
        """
        with self._frame_lock:
            self._status_deadline = time.monotonic() + _STATUS_TTL_SECONDS
            if text == self._status_text:
                return
            self._status_text = text
            if self._base_frame is not None:
                self._current_frame = self._composite_status(self._base_frame)
                self._frame_generation += 1

    def clear_status_text(self) -> None:
        """Remove the status pill and restore the un-overlaid frame."""
        with self._frame_lock:
            if self._status_text is None:
                return
            self._status_text = None
            if self._base_frame is not None:
                self._current_frame = self._base_frame
                self._frame_generation += 1

    def _composite_status(self, base: Image.Image) -> Image.Image:
        """Apply the status pill to ``base`` if one is active.

        Caller must hold ``_frame_lock``. Never raises — a drawing
        failure falls back to the clean frame.
        """
        if not self._status_text:
            return base
        try:
            return _draw_status_pill(base, self._status_text)
        except Exception:
            logger.exception("Status pill compositing failed")
            return base

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_and_update_frame(self) -> None:
        """Render the active display and update the frame buffer.

        This is the main rendering entry point. It:
        1. Gets the current spec and theme
        2. Gathers data from all active sources
        3. Resolves data bindings in the block tree
        4. Renders the resolved tree to a PIL Image
        5. Updates the thread-safe frame buffer
        """
        if not self._active_display:
            return

        spec = self._specs.get(self._active_display)
        if spec is None or spec.root_block is None:
            return

        theme = self._active_theme
        if theme is None:
            return

        # Gather current data from all sources. Display args are exposed
        # under the "args" key so blocks can bind to {args.image_ids[0]}
        # etc. — a way for switch(..., args=...) callers to parameterize
        # a display (used by the `picture` display for image_ids).
        data = self._data_manager.get_all_data()
        if self._active_args:
            data = {**data, "args": dict(self._active_args)}

        try:
            # Resolve bindings and render
            resolved = resolve_bindings(spec.root_block, data)
            _resolve_photo_sources(resolved)
            frame = self._renderer.render_block_tree(resolved, theme, data)
            self._update_frame(frame)
        except Exception:
            logger.exception(
                "Render failed for display '%s'", self._active_display
            )

    def build_preview_data(
        self,
        spec: DisplaySpec,
        data: dict[str, Any] | None = None,
        args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assemble the data dict the renderer (and warnings checker) sees.

        Resolution order, per source declared on the spec:

        1. ``data`` override, if the caller passed the key at all —
           ``{}``/``0``/``False`` is a value the caller chose, not an
           absence. Live and static payloads use truthiness instead:
           there, empty means "nothing fetched yet", and a placeholder
           previews better than a blank block.
        2. The source's own declared payload — for ``static`` sources,
           this is the ``value=`` the agent set when authoring. Critical
           for previewing a fresh spec before save/switch, since static
           sources are inert until they're registered with a manager.
        3. Live cached data from the running data manager, but only when
           this spec *is* the active display. Source names are a global
           namespace: another spec's ``climate`` source is not the
           on-screen display's ``climate`` source, and letting live data
           win there silently renders values the spec never declared.
        4. Built-in placeholder data (``weather``, ``calendar``, …).

        Self-preview of the active display still sees pushed values —
        ``update_static_data`` writes them back onto the spec, so step 2
        and the live cache agree.

        ``args`` are exposed to bindings as ``{args.<field>}``.
        """
        override = dict(data) if data else {}
        live = (
            self._data_manager.get_all_data()
            if spec.name == self._active_display
            else {}
        )

        preview_data: dict[str, Any] = {}
        for src_spec in spec.data_sources:
            name = src_spec.name
            if name in override:
                preview_data[name] = override[name]
            elif src_spec.source_type == "static" and src_spec.value is not None:
                v = src_spec.value
                preview_data[name] = v if isinstance(v, dict) else {"value": v}
            elif live.get(name):
                preview_data[name] = live[name]
            else:
                preview_data[name] = placeholder_for_source(
                    name, src_spec.source_type,
                )

        # Undeclared override keys still pass through — callers preview
        # data for sources the spec hasn't declared yet.
        for key, value in override.items():
            preview_data.setdefault(key, value)

        preview_data["args"] = dict(args) if args else override.get("args") or {}
        return preview_data

    def render_preview(
        self,
        name: str | None = None,
        data: dict[str, Any] | None = None,
        width: int | None = None,
        height: int | None = None,
        args: dict[str, Any] | None = None,
    ) -> Image.Image | None:
        """Render a display to a PIL Image for preview.

        Used by the SDK preview workflow. Fills missing data sources
        with placeholder data — and, for ``static`` sources, the
        declared ``value`` — so the agent sees a realistic layout
        even before the spec has been saved and switched to.

        Args:
            name: Display name (defaults to active display).
            data: Override data dict. If None, uses live data +
                placeholders + declared static values.
            width: Preview width override.
            height: Preview height override.
            args: Optional args dict, exposed as ``{args.<field>}``.

        Returns:
            A PIL Image of the rendered display, or None if the display
            is not found.
        """
        display_name = name or self._active_display
        if not display_name:
            logger.warning("No display to render preview for")
            return None

        spec = self._specs.get(display_name)
        if spec is None:
            logger.warning("Display '%s' not found for preview", display_name)
            return None

        pw = width or self._width
        ph = height or self._height
        if pw != self._width or ph != self._height:
            renderer = DisplayRenderer(width=pw, height=ph)
        else:
            renderer = self._renderer

        preview_data = self.build_preview_data(spec, data=data, args=args)
        return renderer.render_preview(spec, preview_data)

    # ------------------------------------------------------------------
    # Background tasks
    # ------------------------------------------------------------------

    async def _live_tick_loop(self) -> None:
        """The repaint pump: at most one render per tick, whoever asked.

        Two things ask for a repaint, and both come through here so they
        cannot stack:
        - live blocks (clock, countdown), which re-render on their own
          cadence — 1 fps with seconds showing, 1/60 fps without;
        - a data source whose fetch changed what is on screen, which sets
          ``_source_dirty`` (see :meth:`_on_source_update`).

        The tick is the floor on both: a source that refreshes faster than
        ``_TICK_SECONDS`` (``ClockSource`` is 1 Hz, and its payload changes
        every tick) coalesces into one render, and the flag is only cleared
        by a render, so the last change always lands.
        """
        last_live_render = 0.0
        try:
            while self._running:
                await asyncio.sleep(_TICK_SECONDS)

                # Status-pill watchdog: lazy expiry so a crashed turn
                # (no clearing event ever fires) can't strand the pill.
                if (
                    self._status_text is not None
                    and time.monotonic() > self._status_deadline
                ):
                    logger.warning(
                        "Status pill expired without a clearing event "
                        "(%r) — clearing", self._status_text,
                    )
                    self.clear_status_text()

                if not self._active_display:
                    continue

                spec = self._specs.get(self._active_display)
                has_live = has_seconds = False
                if spec and spec.root_block:
                    has_live, has_seconds = self._detect_live_blocks(spec.root_block)

                now = time.monotonic()
                live_due = has_live and (
                    has_seconds or now - last_live_render >= _SLOW_LIVE_SECONDS
                )
                if not (live_due or self._source_dirty):
                    continue

                if live_due:
                    last_live_render = now
                self._source_dirty = False
                self._render_and_update_frame()

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Live tick loop error")

    @staticmethod
    def _detect_live_blocks(block: Block) -> tuple[bool, bool]:
        """Check if a block tree contains live blocks (clock, countdown).

        Args:
            block: The root block to check.

        Returns:
            Tuple of (has_live_blocks, has_seconds). has_seconds is True
            if any clock block has show_seconds=True.
        """
        has_live = False
        has_seconds = False

        if block.block_type in ("clock", "countdown"):
            has_live = True
            if block.block_type == "clock" and block.params.get("show_seconds", False):
                has_seconds = True

        for child in block.children:
            child_live, child_seconds = DisplayManager._detect_live_blocks(child)
            has_live = has_live or child_live
            has_seconds = has_seconds or child_seconds

        return has_live, has_seconds

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def get_brightness(self) -> float:
        """Get the current display brightness setting.

        Checks night mode configuration and returns the appropriate
        brightness level.

        Returns:
            Brightness value between 0.0 and 1.0.
        """
        try:
            from boxbot.core.config import get_config
            config = get_config()

            if config.display.night_mode.enabled:
                if self._is_night_mode(
                    config.display.night_mode.start,
                    config.display.night_mode.end,
                ):
                    return config.display.night_mode.brightness

            return config.display.brightness
        except (RuntimeError, ImportError):
            return 0.8

    @staticmethod
    def _is_night_mode(start_str: str, end_str: str) -> bool:
        """Check if the current time falls within the night mode window.

        Args:
            start_str: Night mode start time (HH:MM format).
            end_str: Night mode end time (HH:MM format).

        Returns:
            True if current time is within the night mode window.
        """
        from datetime import datetime

        try:
            now = datetime.now()
            current_minutes = now.hour * 60 + now.minute

            start_parts = start_str.split(":")
            start_minutes = int(start_parts[0]) * 60 + int(start_parts[1])

            end_parts = end_str.split(":")
            end_minutes = int(end_parts[0]) * 60 + int(end_parts[1])

            if start_minutes <= end_minutes:
                # Same day: e.g. 06:00 to 18:00
                return start_minutes <= current_minutes <= end_minutes
            else:
                # Overnight: e.g. 22:00 to 07:00
                return current_minutes >= start_minutes or current_minutes <= end_minutes

        except (ValueError, IndexError):
            return False

    def get_night_theme(self) -> str:
        """Get the recommended theme for night mode.

        Returns:
            Theme name suitable for nighttime display.
        """
        return "midnight"


# ---------------------------------------------------------------------------
# Photo source pre-resolution
# ---------------------------------------------------------------------------


def _resolve_photo_sources(block: "Block") -> None:
    """Walk a resolved block tree and rewrite ``photo:<ref>`` image sources.

    After :func:`boxbot.displays.spec.resolve_bindings` runs, ImageBlock
    sources like ``photo:{args.image_ids[0]}`` are now ``photo:<ref>``.
    A ``<ref>`` is either a photo-library id or an absolute file path
    (the `picture` display renders both — camera snapshots and other
    workspace images arrive as paths). We resolve each ref to an absolute
    file path here and rewrite the block in place; the renderer only
    knows how to open a plain file path. Misses fall through to the
    renderer's placeholder.
    """
    from boxbot.displays.blocks import Block as _Block  # avoid circular

    # Walk once to collect all photo: refs.
    refs: list[str] = []

    def _collect(b: _Block) -> None:
        if b.block_type == "image":
            src = str(b.params.get("source", ""))
            if src.startswith("photo:"):
                refs.append(src.split(":", 1)[1])
        for child in b.children:
            _collect(child)

    _collect(block)
    if not refs:
        return

    paths = _resolve_photo_refs(refs)

    def _rewrite(b: _Block) -> None:
        if b.block_type == "image":
            src = str(b.params.get("source", ""))
            if src.startswith("photo:"):
                ref = src.split(":", 1)[1]
                if ref in paths:
                    b.params["source"] = paths[ref]
        for child in b.children:
            _rewrite(child)

    _rewrite(block)


def _resolve_photo_refs(refs: list[str]) -> dict[str, str]:
    """Map each ``photo:`` ref to an absolute file path it renders from.

    A ref is either a photo-library id (looked up in the photos DB) or a
    direct file path under an allowed root (workspace / photos / crops /
    previews / sandbox tmp — same roots that gate image attachment). Refs
    that resolve to neither are left out; the caller renders a placeholder.
    """
    paths: dict[str, str] = {}

    # Direct file paths (camera snapshots, workspace images). Validated
    # against the shared attach roots so a display can never be pointed at
    # an arbitrary file on disk.
    remaining: list[str] = []
    for ref in refs:
        direct = _resolve_direct_image_path(ref)
        if direct is not None:
            paths[ref] = direct
        else:
            remaining.append(ref)
    if not remaining:
        return paths

    # Photo-library ids. Best-effort: the DB may not exist yet on a fresh
    # install; swallow errors rather than break the whole render.
    try:
        from boxbot.core.config import get_config

        config = get_config()
    except Exception:
        return paths

    storage_path = Path(config.photos.storage_path)
    db_path = storage_path / "photos.db"
    if not db_path.exists():
        return paths

    import sqlite3

    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            placeholders = ",".join("?" for _ in remaining)
            rows = conn.execute(
                f"SELECT id, filename FROM photos "
                f"WHERE id IN ({placeholders}) AND deleted_at IS NULL",
                remaining,
            ).fetchall()
            for r in rows:
                abs_path = (storage_path / r["filename"]).resolve()
                if abs_path.exists():
                    paths[r["id"]] = str(abs_path)
    except sqlite3.Error as e:
        logger.warning("photo source resolution failed: %s", e)

    return paths


def _resolve_direct_image_path(ref: str) -> str | None:
    """Return ``ref`` as an absolute path iff it is an allowed image file.

    Reuses the sandbox attach-root allowlist so display sources are held
    to the same boundary as tool-result attachments.
    """
    if not ref or not Path(ref).is_absolute():
        return None
    try:
        from boxbot.tools._sandbox_actions import _is_attach_allowed

        abs_path = Path(ref).resolve()
    except Exception:
        return None
    if not abs_path.is_file() or not _is_attach_allowed(abs_path):
        return None
    return str(abs_path)
