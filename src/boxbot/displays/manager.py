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
import threading
from pathlib import Path
from typing import Any

from PIL import Image

from boxbot.displays.blocks import Block
from boxbot.displays.data_sources import (
    DataSourceManager,
    StaticSource,
    create_source,
    get_placeholder_data,
)
from boxbot.displays.renderer import DisplayRenderer
from boxbot.displays.spec import (
    DisplaySpec,
    parse_spec,
    resolve_bindings,
    validate_spec,
)
from boxbot.displays.themes import Theme, get_theme

logger = logging.getLogger(__name__)

# Default paths
_BUILTINS_DIR = Path(__file__).parent / "builtins"
_USER_DISPLAYS_DIR = Path("displays")
_AGENT_DISPLAYS_DIR = Path("data/displays")


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

        # Data source management
        self._data_manager = DataSourceManager()

        # Rendering
        self._renderer = DisplayRenderer(width=width, height=height)
        self._width = width
        self._height = height

        # Thread-safe frame buffer for screen HAL consumption
        self._frame_lock = threading.Lock()
        self._current_frame: Image.Image | None = None
        self._frame_generation: int = 0  # Incremented on each new frame

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
            from boxbot.core.events import DisplaySwitch, get_event_bus

            bus = get_event_bus()
            bus.subscribe(DisplaySwitch, self._on_display_switch)
            self._event_subscribed = True
            logger.debug("Subscribed to DisplaySwitch events")
        except ImportError:
            logger.debug("Event bus not available, skipping subscription")
        except Exception:
            logger.exception("Failed to subscribe to event bus")

    def _unsubscribe_events(self) -> None:
        """Unsubscribe from the event bus."""
        if not self._event_subscribed:
            return

        try:
            from boxbot.core.events import DisplaySwitch, get_event_bus

            bus = get_event_bus()
            bus.unsubscribe(DisplaySwitch, self._on_display_switch)
            self._event_subscribed = False
            logger.debug("Unsubscribed from DisplaySwitch events")
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

        # Load from all directories
        self._load_specs_from_dir(builtins_path)
        self._load_specs_from_dir(user_path)
        self._load_specs_from_dir(agent_path)

        logger.info(
            "Loaded %d display(s): %s",
            len(self._specs),
            list(self._specs.keys()),
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
    ) -> bool:
        """Switch to a named display.

        This is the primary method for changing what's on screen. It:
        1. Validates the display name
        2. Stops rotation if running
        3. Tears down data sources for the old display
        4. Sets up data sources for the new display
        5. Performs an initial render
        6. Updates the frame buffer

        Args:
            name: Display name to activate.
            args: Display-specific arguments (e.g. {"image_ids": [...]} for
                  the picture display).

        Returns:
            True if the switch succeeded.
        """
        spec = self._specs.get(name)
        if spec is None:
            logger.warning("Display '%s' not found. Available: %s", name, self.list_available())
            return False

        # Stop rotation (agent explicitly chose a display)
        if not self._rotation_active:
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

        # Update active state
        self._active_display = name
        self._active_args = args or {}

        # Render the initial frame
        self._render_and_update_frame()

        logger.info("Switched to display '%s' (args=%s)", name, self._active_args)
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
                "url": src_spec.url,
                "params": src_spec.params,
                "secret": src_spec.secret,
                "refresh": src_spec.refresh,
                "fields": src_spec.fields,
                "value": src_spec.value,
                "query": src_spec.query,
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

        Runs until cancelled. Each iteration switches to the next display
        and waits for the configured interval.
        """
        try:
            while self._running and self._rotation_active:
                if self._rotation_displays:
                    name = self._rotation_displays[self._rotation_index]
                    self._rotation_active = True  # Keep flag set during switch
                    await self.switch(name)
                    self._rotation_active = True  # Re-set after switch (switch may clear it)
                    self._rotation_index = (
                        (self._rotation_index + 1) % len(self._rotation_displays)
                    )
                await asyncio.sleep(self._rotation_interval)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Display rotation loop error")

    def _get_rotation_config(self) -> tuple[list[str], int]:
        """Get rotation settings from config, with safe fallbacks.

        Returns:
            Tuple of (display_names, interval_seconds).
        """
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
    ) -> bool:
        """Update a static data source without rebuilding the display.

        The agent can change what flows through the data pipe without
        touching the layout. No preview or approval needed.

        Args:
            display_name: The display that owns the source.
            source_name: The static source name to update.
            value: New data value.

        Returns:
            True if the update succeeded.
        """
        if display_name != self._active_display:
            logger.warning(
                "Cannot update data for inactive display '%s'", display_name
            )
            return False

        source = self._data_manager.get_source(source_name)
        if not isinstance(source, StaticSource):
            logger.warning(
                "Source '%s' is not a static source (type: %s)",
                source_name, type(source).__name__,
            )
            return False

        source.update(value)

        # Re-render with the updated data
        self._render_and_update_frame()

        logger.debug(
            "Updated static data for '%s.%s'", display_name, source_name
        )
        return True

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

    def _update_frame(self, frame: Image.Image) -> None:
        """Update the frame buffer with a new rendered frame.

        Thread-safe. Called after each render cycle.

        Args:
            frame: The new frame image (RGB, 1024x600).
        """
        with self._frame_lock:
            self._current_frame = frame
            self._frame_generation += 1

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

        # Gather current data from all sources
        data = self._data_manager.get_all_data()

        try:
            # Resolve bindings and render
            resolved = resolve_bindings(spec.root_block, data)
            frame = self._renderer.render_block_tree(resolved, theme, data)
            self._update_frame(frame)
        except Exception:
            logger.exception(
                "Render failed for display '%s'", self._active_display
            )

    def render_preview(
        self,
        name: str | None = None,
        data: dict[str, Any] | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> Image.Image | None:
        """Render a display to a PIL Image for preview.

        Used by the SDK preview workflow. Fills missing data sources
        with placeholder data so the agent sees a realistic layout.

        Args:
            name: Display name (defaults to active display).
            data: Override data dict. If None, uses live data + placeholders.
            width: Preview width override.
            height: Preview height override.

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

        # Create a preview renderer if dimensions differ
        pw = width or self._width
        ph = height or self._height
        if pw != self._width or ph != self._height:
            renderer = DisplayRenderer(width=pw, height=ph)
        else:
            renderer = self._renderer

        # Build data dict: live data + placeholders for missing sources
        preview_data = dict(data) if data else dict(self._data_manager.get_all_data())
        for src_spec in spec.data_sources:
            if src_spec.name not in preview_data or not preview_data[src_spec.name]:
                preview_data[src_spec.name] = get_placeholder_data(src_spec.name)

        return renderer.render_preview(spec, preview_data)

    # ------------------------------------------------------------------
    # Background tasks
    # ------------------------------------------------------------------

    async def _live_tick_loop(self) -> None:
        """Background task that re-renders live blocks (clock, countdown).

        Runs every second to keep live blocks updated. Only re-renders
        if the active display contains live blocks, otherwise sleeps
        longer to conserve resources.

        Frame rate management:
        - 1 fps when clock with seconds is showing
        - 1/60 fps when clock without seconds is showing
        - 0 fps when no live blocks (sleep until data change)
        """
        try:
            while self._running:
                if self._active_display:
                    spec = self._specs.get(self._active_display)
                    if spec and spec.root_block:
                        has_live, has_seconds = self._detect_live_blocks(spec.root_block)
                        if has_live:
                            self._render_and_update_frame()
                            if has_seconds:
                                await asyncio.sleep(1.0)
                            else:
                                await asyncio.sleep(60.0)
                            continue

                # No live blocks or no active display -- sleep longer
                await asyncio.sleep(5.0)

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
