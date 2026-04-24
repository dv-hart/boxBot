"""Screen HAL — renders display frames to the HDMI output via pygame.

Polls the DisplayManager's frame buffer from a dedicated thread and
blits PIL Images to a fullscreen pygame surface. The display manager
runs in the async event loop; this module bridges to the pygame display
thread which requires its own event pump.

Hardware: 7" HDMI LCD (H), 1024×600, IPS, capacitive touch (USB HID).
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any

from boxbot.hardware.base import HardwareModule, HardwareUnavailableError

if TYPE_CHECKING:
    from boxbot.displays.manager import DisplayManager

logger = logging.getLogger(__name__)

# Target display resolution
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 600
TARGET_FPS = 30


class Screen(HardwareModule):
    """Fullscreen pygame display that renders frames from the DisplayManager.

    The screen runs a dedicated thread that:
    1. Initializes pygame in fullscreen mode
    2. Polls ``DisplayManager.get_current_frame()`` for new PIL Images
    3. Converts and blits them to the pygame surface
    4. Pumps the pygame event loop (required by SDL)
    """

    name = "screen"

    def __init__(
        self,
        display_manager: DisplayManager | None = None,
        brightness: float = 1.0,
    ) -> None:
        super().__init__()
        self._display_manager = display_manager
        self._brightness = max(0.0, min(1.0, brightness))
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_generation = -1

    def set_display_manager(self, dm: DisplayManager) -> None:
        """Wire up the display manager after construction."""
        self._display_manager = dm

    async def start(self) -> None:
        """Start the pygame display thread."""
        # Ensure Wayland environment is set for headless SSH sessions
        if "WAYLAND_DISPLAY" not in os.environ:
            os.environ["WAYLAND_DISPLAY"] = "wayland-0"
        if "XDG_RUNTIME_DIR" not in os.environ:
            os.environ["XDG_RUNTIME_DIR"] = "/run/user/1000"

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._display_loop,
            name="screen-hal",
            daemon=True,
        )
        self._thread.start()

        # Wait briefly for pygame init to complete
        await asyncio.sleep(0.5)

        if not self._started:
            raise HardwareUnavailableError("pygame display failed to initialize")

        await self._emit_health(
            __import__("boxbot.hardware.base", fromlist=["HealthStatus"]).HealthStatus.OK
        )

    async def stop(self) -> None:
        """Stop the display thread and close pygame."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._started = False

    @property
    def is_available(self) -> bool:
        return self._started

    def set_brightness(self, level: float) -> None:
        """Set backlight brightness (0.0-1.0). No-op if unsupported."""
        self._brightness = max(0.0, min(1.0, level))
        # TODO: implement backlight control via sysfs if hardware supports it

    def _display_loop(self) -> None:
        """Main display thread — runs pygame event loop and blits frames."""
        try:
            import pygame

            pygame.init()

            # Fullscreen on the HDMI display
            screen = pygame.display.set_mode(
                (SCREEN_WIDTH, SCREEN_HEIGHT),
                pygame.FULLSCREEN | pygame.NOFRAME,
            )
            pygame.display.set_caption("boxBot")
            pygame.mouse.set_visible(False)

            # Black screen initially
            screen.fill((0, 0, 0))
            pygame.display.flip()

            self._started = True
            logger.info(
                "Screen started: %dx%d fullscreen (pygame %s)",
                SCREEN_WIDTH, SCREEN_HEIGHT, pygame.ver,
            )

            clock = pygame.time.Clock()

            while not self._stop_event.is_set():
                # Pump SDL events (required even if we don't handle them)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._stop_event.set()
                        break

                # Check for new frame from display manager
                if self._display_manager is not None:
                    gen = self._display_manager.get_frame_generation()
                    if gen != self._last_generation:
                        frame = self._display_manager.get_current_frame()
                        if frame is not None:
                            self._blit_frame(screen, frame)
                            pygame.display.flip()
                            self._last_generation = gen

                clock.tick(TARGET_FPS)

        except Exception:
            logger.exception("Screen display loop failed")
        finally:
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass
            self._started = False
            logger.info("Screen stopped")

    @staticmethod
    def _blit_frame(screen: Any, frame: Any) -> None:
        """Convert a PIL Image to a pygame surface and blit it."""
        import pygame

        # PIL Image → raw bytes → pygame surface
        raw = frame.tobytes()
        surface = pygame.image.frombuffer(raw, frame.size, "RGB")

        # Scale if the frame doesn't match screen resolution
        screen_w, screen_h = screen.get_size()
        if surface.get_size() != (screen_w, screen_h):
            surface = pygame.transform.smoothscale(surface, (screen_w, screen_h))

        screen.blit(surface, (0, 0))
