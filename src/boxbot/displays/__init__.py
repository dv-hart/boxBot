"""boxBot display system.

Manages the 7" screen with swappable layouts — dashboards, clocks, photo
slideshows, weather boards — using a declarative block system with data
binding and themes.

Key exports:
    DisplayManager   — Display lifecycle, switching, rotation, frame buffer
    DisplayRenderer  — Renders display specs to PIL Images
    Theme, get_theme — Theme system
    render_to_image  — Convenience function for one-shot rendering
"""

from boxbot.displays.manager import DisplayManager
from boxbot.displays.renderer import DisplayRenderer, render_to_image
from boxbot.displays.themes import Theme, get_theme, list_themes

__all__ = [
    "DisplayManager",
    "DisplayRenderer",
    "Theme",
    "get_theme",
    "list_themes",
    "render_to_image",
]
