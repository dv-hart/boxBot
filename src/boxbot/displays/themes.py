"""Theme system for boxBot displays.

Defines color palettes, typography, spacing, and visual styling. Every block
reads from the active theme using semantic tokens (accent, muted, success) —
never raw hex values.

Four built-in themes: boxbot (default), midnight, daylight, classic.
Community themes load from YAML files in themes/ at the project root.

Usage:
    from boxbot.displays.themes import get_theme, list_themes

    theme = get_theme("boxbot")
    bg_rgb = theme.color_rgb("background")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FontStyle:
    """Typography settings for a single text size level."""

    size: int
    weight: int
    tracking: float = 0.0


@dataclass(frozen=True)
class ThemeFonts:
    """Complete font configuration for a theme."""

    family: str
    title: FontStyle
    heading: FontStyle
    subtitle: FontStyle
    body: FontStyle
    caption: FontStyle
    small: FontStyle

    def get_style(self, name: str) -> FontStyle:
        """Get a font style by name, falling back to body."""
        return getattr(self, name, self.body)


@dataclass(frozen=True)
class ThemeColors:
    """Color palette for a theme. All values are hex strings."""

    background: str
    surface: str
    surface_alt: str
    text: str
    muted: str
    dim: str
    accent: str
    accent_soft: str
    secondary: str
    success: str
    warning: str
    error: str

    def get(self, name: str) -> str:
        """Get a color by semantic name, falling back to text."""
        return getattr(self, name, self.text)


@dataclass(frozen=True)
class ThemeSpacing:
    """Spacing scale in pixels."""

    xs: int = 4
    sm: int = 8
    md: int = 16
    lg: int = 24
    xl: int = 32

    def get(self, name: str) -> int:
        """Get spacing by name, falling back to md."""
        return getattr(self, name, self.md)


@dataclass(frozen=True)
class Theme:
    """Complete theme definition."""

    name: str
    description: str
    colors: ThemeColors
    fonts: ThemeFonts
    spacing: ThemeSpacing
    radius: int = 14
    shadow: bool = True
    icon_style: str = "outline"
    transition: str = "crossfade"

    def color_rgb(self, name: str) -> tuple[int, int, int]:
        """Get a color as an RGB tuple.

        Args:
            name: Semantic color name (e.g. 'accent', 'background').

        Returns:
            (R, G, B) tuple with values 0-255.
        """
        hex_str = self.colors.get(name)
        return hex_to_rgb(hex_str)

    def color_rgba(self, name: str) -> tuple[int, int, int, int]:
        """Get a color as an RGBA tuple.

        Args:
            name: Semantic color name.

        Returns:
            (R, G, B, A) tuple with values 0-255.
        """
        hex_str = self.colors.get(name)
        return hex_to_rgba(hex_str)

    def font_style(self, size_name: str) -> FontStyle:
        """Get font style for a given size name."""
        return self.fonts.get_style(size_name)


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a hex color string to an RGB tuple.

    Handles #RGB, #RRGGBB, and #RRGGBBAA formats.
    """
    h = hex_color.lstrip("#")
    if len(h) == 3:
        r, g, b = int(h[0] * 2, 16), int(h[1] * 2, 16), int(h[2] * 2, 16)
    elif len(h) in (6, 8):
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    else:
        logger.warning("Invalid hex color '%s', defaulting to black", hex_color)
        r, g, b = 0, 0, 0
    return (r, g, b)


def hex_to_rgba(hex_color: str) -> tuple[int, int, int, int]:
    """Convert a hex color string to an RGBA tuple.

    #RRGGBBAA format uses the last two hex digits as alpha.
    All other formats default to alpha=255 (fully opaque).
    """
    h = hex_color.lstrip("#")
    r, g, b = hex_to_rgb(hex_color)
    if len(h) == 8:
        a = int(h[6:8], 16)
    else:
        a = 255
    return (r, g, b, a)


def resolve_color(token: str, theme: Theme) -> str:
    """Resolve a color token to a hex string.

    If the token is already a hex color (#...), returns it directly.
    Otherwise, looks it up in the theme's color palette.
    """
    if token.startswith("#"):
        return token
    return theme.colors.get(token)


# ---------------------------------------------------------------------------
# Built-in themes (hardcoded from spec)
# ---------------------------------------------------------------------------

_INTER_FONTS = ThemeFonts(
    family="Inter",
    title=FontStyle(size=42, weight=700, tracking=-0.02),
    heading=FontStyle(size=28, weight=600, tracking=-0.01),
    subtitle=FontStyle(size=22, weight=500),
    body=FontStyle(size=18, weight=400),
    caption=FontStyle(size=15, weight=400),
    small=FontStyle(size=13, weight=400),
)

_DEFAULT_SPACING = ThemeSpacing(xs=4, sm=8, md=16, lg=24, xl=32)


THEME_BOXBOT = Theme(
    name="boxbot",
    description="Warm minimal — designed for the wooden enclosure",
    colors=ThemeColors(
        background="#191714",
        surface="#252018",
        surface_alt="#302a20",
        text="#ede8e0",
        muted="#8a8078",
        dim="#5a5550",
        accent="#d4845a",
        accent_soft="#d4845a22",
        secondary="#c4a46c",
        success="#7a9e6c",
        warning="#d4a043",
        error="#c45c5c",
    ),
    fonts=_INTER_FONTS,
    spacing=_DEFAULT_SPACING,
    radius=14,
    shadow=True,
    icon_style="outline",
    transition="crossfade",
)

THEME_MIDNIGHT = Theme(
    name="midnight",
    description="Near-black for nighttime — only essentials visible",
    colors=ThemeColors(
        background="#0c0b0a",
        surface="#141210",
        surface_alt="#1a1816",
        text="#6a6560",
        muted="#3a3835",
        dim="#2a2825",
        accent="#8a6040",
        accent_soft="#8a604015",
        secondary="#7a6840",
        success="#4a6a44",
        warning="#8a7030",
        error="#7a3a3a",
    ),
    fonts=_INTER_FONTS,
    spacing=_DEFAULT_SPACING,
    radius=14,
    shadow=False,
    icon_style="outline",
    transition="crossfade",
)

THEME_DAYLIGHT = Theme(
    name="daylight",
    description="Warm cream for bright rooms — high contrast readability",
    colors=ThemeColors(
        background="#f5f0e6",
        surface="#ffffff",
        surface_alt="#ece7dd",
        text="#2a2218",
        muted="#7a7068",
        dim="#a09890",
        accent="#c46a3c",
        accent_soft="#c46a3c18",
        secondary="#8a7040",
        success="#4a7a40",
        warning="#b08020",
        error="#b04040",
    ),
    fonts=_INTER_FONTS,
    spacing=_DEFAULT_SPACING,
    radius=14,
    shadow=True,
    icon_style="outline",
    transition="crossfade",
)

THEME_CLASSIC = Theme(
    name="classic",
    description="Vintage radio — amber on dark, warm and nostalgic",
    colors=ThemeColors(
        background="#141008",
        surface="#1e1810",
        surface_alt="#282018",
        text="#d4b88c",
        muted="#8a7860",
        dim="#5a5040",
        accent="#e0943c",
        accent_soft="#e0943c20",
        secondary="#c49030",
        success="#6a9050",
        warning="#c48830",
        error="#a84040",
    ),
    fonts=_INTER_FONTS,
    spacing=_DEFAULT_SPACING,
    radius=10,
    shadow=True,
    icon_style="outline",
    transition="crossfade",
)

# Registry of built-in themes
_BUILTIN_THEMES: dict[str, Theme] = {
    "boxbot": THEME_BOXBOT,
    "midnight": THEME_MIDNIGHT,
    "daylight": THEME_DAYLIGHT,
    "classic": THEME_CLASSIC,
}

# Community themes loaded at runtime
_community_themes: dict[str, Theme] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_theme(name: str) -> Theme:
    """Get a theme by name.

    Checks built-in themes first, then community themes.

    Args:
        name: Theme name (e.g. 'boxbot', 'midnight').

    Returns:
        The Theme instance.

    Raises:
        KeyError: If the theme name is not found.
    """
    if name in _BUILTIN_THEMES:
        return _BUILTIN_THEMES[name]
    if name in _community_themes:
        return _community_themes[name]
    raise KeyError(
        f"Theme '{name}' not found. Available: {', '.join(list_themes())}"
    )


def list_themes() -> list[str]:
    """List all available theme names (built-in + community)."""
    return list(_BUILTIN_THEMES.keys()) + list(_community_themes.keys())


def load_community_themes(themes_dir: str | Path) -> dict[str, Theme]:
    """Load community themes from a directory of YAML files.

    Scans for .yaml/.yml files directly in the directory, and for
    theme.yaml inside subdirectories (which may also contain bundled fonts).

    Args:
        themes_dir: Path to the themes directory.

    Returns:
        Dict mapping theme names to Theme instances.
    """
    themes_path = Path(themes_dir)
    if not themes_path.is_dir():
        logger.debug("Themes directory %s does not exist", themes_path)
        return {}

    loaded: dict[str, Theme] = {}

    # Top-level YAML files
    for yaml_file in themes_path.glob("*.yaml"):
        theme = _load_theme_yaml(yaml_file)
        if theme:
            loaded[theme.name] = theme

    for yaml_file in themes_path.glob("*.yml"):
        theme = _load_theme_yaml(yaml_file)
        if theme:
            loaded[theme.name] = theme

    # Subdirectories with theme.yaml
    for subdir in themes_path.iterdir():
        if subdir.is_dir():
            for fname in ("theme.yaml", "theme.yml"):
                yaml_file = subdir / fname
                if yaml_file.exists():
                    theme = _load_theme_yaml(yaml_file)
                    if theme:
                        loaded[theme.name] = theme
                    break

    _community_themes.update(loaded)
    if loaded:
        logger.info("Loaded %d community theme(s): %s", len(loaded), list(loaded.keys()))
    return loaded


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_theme_yaml(path: Path) -> Theme | None:
    """Parse a theme YAML file into a Theme instance."""
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            logger.warning("Theme file %s is not a valid YAML mapping", path)
            return None
        return _theme_from_dict(data)
    except Exception:
        logger.exception("Failed to load theme from %s", path)
        return None


def _theme_from_dict(data: dict[str, Any]) -> Theme:
    """Build a Theme from a parsed YAML/JSON dict."""
    colors_data = data.get("colors", {})
    colors = ThemeColors(
        background=colors_data.get("background", "#000000"),
        surface=colors_data.get("surface", "#111111"),
        surface_alt=colors_data.get("surface_alt", "#222222"),
        text=colors_data.get("text", "#ffffff"),
        muted=colors_data.get("muted", "#888888"),
        dim=colors_data.get("dim", "#555555"),
        accent=colors_data.get("accent", "#d4845a"),
        accent_soft=colors_data.get("accent_soft", "#d4845a22"),
        secondary=colors_data.get("secondary", "#c4a46c"),
        success=colors_data.get("success", "#7a9e6c"),
        warning=colors_data.get("warning", "#d4a043"),
        error=colors_data.get("error", "#c45c5c"),
    )

    fonts_data = data.get("fonts", {})
    family = fonts_data.get("family", "Inter")

    def _parse_font_style(d: dict | None, default: FontStyle) -> FontStyle:
        if not d:
            return default
        return FontStyle(
            size=d.get("size", default.size),
            weight=d.get("weight", default.weight),
            tracking=d.get("tracking", default.tracking),
        )

    fonts = ThemeFonts(
        family=family,
        title=_parse_font_style(fonts_data.get("title"), _INTER_FONTS.title),
        heading=_parse_font_style(fonts_data.get("heading"), _INTER_FONTS.heading),
        subtitle=_parse_font_style(fonts_data.get("subtitle"), _INTER_FONTS.subtitle),
        body=_parse_font_style(fonts_data.get("body"), _INTER_FONTS.body),
        caption=_parse_font_style(fonts_data.get("caption"), _INTER_FONTS.caption),
        small=_parse_font_style(fonts_data.get("small"), _INTER_FONTS.small),
    )

    spacing_data = data.get("spacing", {})
    spacing = ThemeSpacing(
        xs=spacing_data.get("xs", 4),
        sm=spacing_data.get("sm", 8),
        md=spacing_data.get("md", 16),
        lg=spacing_data.get("lg", 24),
        xl=spacing_data.get("xl", 32),
    )

    return Theme(
        name=data.get("name", "unnamed"),
        description=data.get("description", ""),
        colors=colors,
        fonts=fonts,
        spacing=spacing,
        radius=data.get("radius", 14),
        shadow=data.get("shadow", True),
        icon_style=data.get("icon_style", "outline"),
        transition=data.get("transition", "crossfade"),
    )
