"""Rendering engine for boxBot displays.

Renders a display spec to a PIL Image for preview/testing. The live pygame
renderer extends this with animation and dirty rectangling; this module
provides the static rendering foundation.

The layout engine resolves a block tree into positioned rectangles, then
draws each block using PIL/Pillow. Text wraps and truncates automatically.
Color tokens resolve to hex values via the active theme.

Usage:
    from boxbot.displays.renderer import DisplayRenderer

    renderer = DisplayRenderer()
    image = renderer.render(spec, theme)
    image.save("preview.png")

    # Or use the convenience function:
    from boxbot.displays.renderer import render_to_image
    image = render_to_image(root_block, theme, data)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from boxbot.displays.blocks import Block
from boxbot.displays.spec import DisplaySpec, resolve_bindings
from boxbot.displays.themes import Theme, get_theme, hex_to_rgb, hex_to_rgba, resolve_color

logger = logging.getLogger(__name__)

# Default screen dimensions (7" IPS display)
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 600

# Icon size mapping (Lucide icon sizes in pixels)
ICON_SIZES: dict[str, int] = {"sm": 16, "md": 24, "lg": 32, "xl": 48}

# Clock size mapping (font sizes for the time display)
CLOCK_SIZES: dict[str, int] = {"md": 36, "lg": 56, "xl": 80}

# Font weight to PIL weight name mapping
WEIGHT_MAP: dict[int, str] = {
    300: "Light",
    400: "Regular",
    500: "Medium",
    600: "SemiBold",
    700: "Bold",
    800: "ExtraBold",
}


# ---------------------------------------------------------------------------
# Layout rect
# ---------------------------------------------------------------------------


@dataclass
class Rect:
    """A positioned rectangle for layout calculations.

    Represents a region of the screen where a block should be rendered.
    The renderer passes rects through the block tree, subdividing
    available space as it descends into children.
    """

    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        """Right edge x-coordinate."""
        return self.x + self.w

    @property
    def bottom(self) -> int:
        """Bottom edge y-coordinate."""
        return self.y + self.h


# ---------------------------------------------------------------------------
# Font cache
# ---------------------------------------------------------------------------


# Bundled assets shipped with the package under assets/.
_ASSETS_DIR = __import__("pathlib").Path(__file__).parent / "assets"
_FONTS_DIR = _ASSETS_DIR / "fonts"
_LUCIDE_DIR = _ASSETS_DIR / "lucide"


def available_lucide_icons() -> list[str]:
    """Return every bundled Lucide icon name (sans the .svg suffix).

    Surfaces what's actually available — the bundled subset is much
    smaller than the full Lucide catalog. Used by ``display.schema``
    so the agent can introspect which icon names will resolve.
    """
    if not _LUCIDE_DIR.is_dir():
        return []
    return sorted(p.stem for p in _LUCIDE_DIR.glob("*.svg"))


def lucide_icon_exists(name: str) -> bool:
    """Whether ``name`` corresponds to a bundled Lucide SVG."""
    if not name:
        return False
    return (_LUCIDE_DIR / f"{name}.svg").exists()

# Named weight → numeric weight for when a TextBlock specifies weight by name.
_WEIGHT_NAMES: dict[str, int] = {
    "thin": 100, "extralight": 200, "light": 300,
    "regular": 400, "normal": 400,
    "medium": 500, "semibold": 600, "bold": 700,
    "extrabold": 800, "black": 900,
}


def _resolve_weight(override: str | int | None, default: int) -> int:
    """Resolve a text block weight override (name or number) to numeric."""
    if override is None:
        return default
    if isinstance(override, int):
        return override
    if isinstance(override, str):
        return _WEIGHT_NAMES.get(override.lower(), default)
    return default

_font_cache: dict[tuple[str, int, int], ImageFont.FreeTypeFont] = {}


def _get_font(family: str, size: int, weight: int = 400) -> ImageFont.FreeTypeFont:
    """Get a PIL font, caching for reuse.

    Lookup order:
      1. Bundled asset: assets/fonts/<family>-<weight>.ttf
      2. Font name via fontconfig (e.g. "Inter-Bold")
      3. Bare family name
      4. System fallback paths
      5. PIL default
    """
    cache_key = (family, size, weight)
    if cache_key in _font_cache:
        return _font_cache[cache_key]

    weight_name = WEIGHT_MAP.get(weight, "Regular")
    font: ImageFont.FreeTypeFont | None = None

    # 1. Bundled asset
    bundled = _FONTS_DIR / f"{family}-{weight_name}.ttf"
    if bundled.is_file():
        try:
            font = ImageFont.truetype(str(bundled), size)
        except (OSError, IOError):
            font = None

    # 2-3. fontconfig / installed
    if font is None:
        for candidate in (f"{family}-{weight_name}", family):
            try:
                font = ImageFont.truetype(candidate, size)
                break
            except (OSError, IOError):
                continue

    # 4. System fallbacks
    if font is None:
        for path in (
            f"/usr/share/fonts/truetype/inter/{family}-{weight_name}.ttf",
            f"/usr/share/fonts/truetype/{family.lower()}/{family}-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ):
            try:
                font = ImageFont.truetype(path, size)
                break
            except (OSError, IOError):
                continue

    # 5. Last resort
    if font is None:
        try:
            font = ImageFont.load_default(size=size)
        except TypeError:
            font = ImageFont.load_default()

    _font_cache[cache_key] = font
    return font


# ---------------------------------------------------------------------------
# Render context (carries state through the tree)
# ---------------------------------------------------------------------------


@dataclass
class RenderContext:
    """Carries rendering state through the block tree.

    Passed to every block renderer so it can access the drawing surface,
    theme colors/fonts, and live data for any remaining unresolved bindings.
    """

    draw: ImageDraw.ImageDraw
    image: Image.Image
    theme: Theme
    data: dict[str, Any]


# ---------------------------------------------------------------------------
# DisplayRenderer class
# ---------------------------------------------------------------------------


class DisplayRenderer:
    """Renders display specs to PIL Images.

    The renderer takes a resolved display spec (with data bindings already
    resolved) and draws it to a 1024x600 PIL Image. It implements rendering
    for all block types: layout containers, content blocks, composite widgets,
    and meta blocks.

    The renderer is stateless between calls -- all state is carried in the
    RenderContext. This means the same renderer instance can be used for
    concurrent preview and live rendering.

    Usage:
        renderer = DisplayRenderer()

        # Render a full spec (resolves bindings, gets theme)
        image = renderer.render(spec, theme, data)

        # Render a raw block tree (bindings already resolved)
        image = renderer.render_block_tree(root_block, theme, data)

        # Render for preview (uses placeholder data for missing sources)
        image = renderer.render_preview(spec, data)
    """

    def __init__(
        self,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
    ) -> None:
        """Initialize the renderer.

        Args:
            width: Output image width in pixels. Defaults to 1024.
            height: Output image height in pixels. Defaults to 600.
        """
        self._width = width
        self._height = height

    @property
    def width(self) -> int:
        """Output image width."""
        return self._width

    @property
    def height(self) -> int:
        """Output image height."""
        return self._height

    def render(
        self,
        spec: DisplaySpec,
        theme: Theme | None = None,
        data: dict[str, Any] | None = None,
    ) -> Image.Image:
        """Render a complete display spec to a PIL Image.

        Resolves the theme from the spec if not provided, resolves data
        bindings, and renders the block tree.

        Args:
            spec: The display specification to render.
            theme: Theme override. If None, uses the spec's theme name.
            data: Data dict for binding resolution. Keys are source names.

        Returns:
            A PIL Image (RGB) of the rendered display.
        """
        if theme is None:
            try:
                theme = get_theme(spec.theme)
            except KeyError:
                logger.warning(
                    "Theme '%s' not found, falling back to 'boxbot'", spec.theme
                )
                theme = get_theme("boxbot")

        data = data or {}

        if spec.root_block is None:
            # Empty spec -- return a blank image with background color
            return Image.new(
                "RGB", (self._width, self._height),
                hex_to_rgb(theme.colors.background),
            )

        # Resolve data bindings in the block tree
        resolved = resolve_bindings(spec.root_block, data)

        return self.render_block_tree(resolved, theme, data)

    def render_block_tree(
        self,
        root_block: Block,
        theme: Theme,
        data: dict[str, Any] | None = None,
    ) -> Image.Image:
        """Render a block tree to a PIL Image.

        The block tree should already have data bindings resolved.
        This is the lower-level rendering entry point.

        Args:
            root_block: Root block of the display layout.
            theme: The active theme for colors, fonts, spacing.
            data: Optional data dict (for widget blocks that read data directly).

        Returns:
            A PIL Image (RGB) of the rendered display.
        """
        img = Image.new(
            "RGBA", (self._width, self._height),
            hex_to_rgba(theme.colors.background),
        )
        draw = ImageDraw.Draw(img)
        ctx = RenderContext(draw=draw, image=img, theme=theme, data=data or {})

        rect = Rect(0, 0, self._width, self._height)
        _render_block(ctx, root_block, rect)

        return img.convert("RGB")

    def render_preview(
        self,
        spec: DisplaySpec,
        data: dict[str, Any] | None = None,
    ) -> Image.Image:
        """Render a display spec for SDK preview.

        Fills missing data sources with placeholder data so the agent
        sees a realistic layout even before sources are live.

        Args:
            spec: The display specification.
            data: Available data (will be supplemented with placeholders).

        Returns:
            A PIL Image (RGB) suitable for saving as PNG.
        """
        from boxbot.displays.data_sources import get_placeholder_data

        preview_data = dict(data) if data else {}

        # Fill in placeholders for any sources not in the data
        for src_spec in spec.data_sources:
            if src_spec.name not in preview_data or not preview_data[src_spec.name]:
                preview_data[src_spec.name] = get_placeholder_data(src_spec.name)

        return self.render(spec, data=preview_data)


# ---------------------------------------------------------------------------
# Module-level convenience function (backwards compatible)
# ---------------------------------------------------------------------------

# Shared renderer instance for the convenience function
_default_renderer: DisplayRenderer | None = None


def render_to_image(
    root_block: Block,
    theme: Theme,
    data: dict[str, Any] | None = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> Image.Image:
    """Render a block tree to a PIL Image (convenience function).

    This is a module-level wrapper around DisplayRenderer.render_block_tree()
    for simple use cases and backwards compatibility.

    Args:
        root_block: The root block of the display spec (already binding-resolved).
        theme: The active theme.
        data: Dict of source data (for any remaining unresolved bindings).
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        A PIL Image of the rendered display.
    """
    global _default_renderer
    if _default_renderer is None or _default_renderer.width != width or _default_renderer.height != height:
        _default_renderer = DisplayRenderer(width=width, height=height)

    return _default_renderer.render_block_tree(root_block, theme, data)


# ---------------------------------------------------------------------------
# Block rendering dispatch
# ---------------------------------------------------------------------------


def _render_block(ctx: RenderContext, block: Block, rect: Rect) -> None:
    """Dispatch rendering for a single block.

    Looks up the block type in the renderer registry and calls the
    appropriate render function. Unknown block types fall through
    to rendering their children vertically.

    Args:
        ctx: The current render context.
        block: The block to render.
        rect: The available rectangle for this block.
    """
    renderer = _BLOCK_RENDERERS.get(block.block_type)
    if renderer:
        renderer(ctx, block, rect)
    else:
        logger.debug("No renderer for block type '%s', rendering children", block.block_type)
        _render_children_vertical(ctx, block.children, rect, gap=0)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _parse_padding(padding: int | list[int] | None) -> tuple[int, int, int, int]:
    """Parse padding into (top, right, bottom, left).

    Accepts:
        - None or 0: no padding
        - int: uniform padding on all sides
        - [v, h]: vertical/horizontal
        - [t, r, b, l]: explicit per-side

    Args:
        padding: Padding specification.

    Returns:
        Tuple of (top, right, bottom, left) pixel values.
    """
    if padding is None or padding == 0:
        return (0, 0, 0, 0)
    if isinstance(padding, int):
        return (padding, padding, padding, padding)
    if isinstance(padding, list):
        if len(padding) == 2:
            return (padding[0], padding[1], padding[0], padding[1])
        if len(padding) == 4:
            return (padding[0], padding[1], padding[2], padding[3])
    return (0, 0, 0, 0)


def _inner_rect(rect: Rect, padding: tuple[int, int, int, int]) -> Rect:
    """Compute the inner rect after subtracting padding.

    Args:
        rect: The outer rectangle.
        padding: (top, right, bottom, left) padding values.

    Returns:
        A new Rect inset by the padding amounts.
    """
    t, r, b, l = padding
    return Rect(
        x=rect.x + l,
        y=rect.y + t,
        w=max(0, rect.w - l - r),
        h=max(0, rect.h - t - b),
    )


def _measure_text(text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    """Measure text dimensions using the font's bounding box.

    Args:
        text: The text string to measure.
        font: The PIL font to measure with.

    Returns:
        (width, height) tuple in pixels.
    """
    bbox = font.getbbox(text)
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


def _wrap_text(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    max_lines: int | None = None,
) -> list[str]:
    """Wrap text to fit within max_width, with optional line limit.

    When max_lines is set and the text exceeds that limit, the last
    line is truncated with an ellipsis.

    Args:
        text: The text to wrap.
        font: The PIL font for measurement.
        max_width: Maximum line width in pixels.
        max_lines: Optional maximum number of lines.

    Returns:
        List of wrapped text lines.
    """
    if max_width <= 0:
        return [text]

    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current_line = words[0]

    for word in words[1:]:
        test_line = f"{current_line} {word}"
        w, _ = _measure_text(test_line, font)
        if w <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
            if max_lines and len(lines) >= max_lines:
                break

    lines.append(current_line)

    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        # Truncate last line with ellipsis
        last = lines[-1]
        while last and _measure_text(last + "...", font)[0] > max_width:
            last = last[:-1]
        lines[-1] = last + "..." if last else "..."

    return lines


def _resolve_block_color(
    color_token: str | None,
    ctx: RenderContext,
    default: str = "text",
) -> tuple[int, int, int]:
    """Resolve a color token to an RGB tuple.

    Handles semantic tokens (accent, muted, etc.), hex colors (#RRGGBB),
    and the special "default" token.

    Args:
        color_token: The color token or hex string.
        ctx: Render context (for theme access).
        default: Fallback theme color name if token is None or "default".

    Returns:
        (R, G, B) tuple.
    """
    if not color_token or color_token == "default":
        return ctx.theme.color_rgb(default)
    return hex_to_rgb(resolve_color(color_token, ctx.theme))


# ---------------------------------------------------------------------------
# Layout container renderers (7)
# ---------------------------------------------------------------------------


def _render_row(ctx: RenderContext, block: Block, rect: Rect) -> None:
    """Render a row -- horizontal flow with children side by side.

    Children are distributed evenly across the available width,
    with configurable gap, alignment, and padding.
    """
    padding = _parse_padding(block.params.get("padding"))
    inner = _inner_rect(rect, padding)
    gap = block.params.get("gap", 0)
    align = block.params.get("align", "start")

    children = block.children
    if not children:
        return

    # Count non-spacer children for width calculation
    non_spacer = [c for c in children if c.block_type != "spacer"]
    n = len(non_spacer)
    if n == 0:
        return

    # Calculate total fixed spacer width
    fixed_spacer_width = sum(
        c.params.get("size", 0) for c in children if c.block_type == "spacer" and c.params.get("size")
    )
    flex_spacer_count = sum(
        1 for c in children if c.block_type == "spacer" and not c.params.get("size")
    )

    total_gap = gap * max(0, len(children) - 1)
    available_for_children = max(0, inner.w - total_gap - fixed_spacer_width)

    if align == "spread" and n > 1:
        # Spread: equal-size children distributed with equal gaps
        child_width = max(1, available_for_children // n)
        spread_gap = max(0, (inner.w - child_width * n) // (n - 1)) if n > 1 else 0
        x = inner.x
        for child in children:
            if child.block_type == "spacer":
                size = child.params.get("size")
                x += size if size else 0
                continue
            child_rect = Rect(x, inner.y, child_width, inner.h)
            _render_block(ctx, child, child_rect)
            x += child_width + spread_gap
    else:
        # start / center / end: pack children at their natural widths. Flex
        # spacers (spacer with no size) absorb any remaining space so you
        # can left/right-pack by sprinkling flex spacers.
        natural_widths = [_estimate_block_width(ctx, c, inner.w) for c in non_spacer]
        total_natural = sum(natural_widths) + total_gap + fixed_spacer_width

        flex_space = 0
        if flex_spacer_count > 0:
            flex_space = max(0, inner.w - total_natural) // flex_spacer_count
            leading = 0
        else:
            leading = max(0, inner.w - total_natural)
            if align == "center":
                leading //= 2
            elif align != "end":
                leading = 0

        x = inner.x + leading
        nw_iter = iter(natural_widths)
        for child in children:
            if child.block_type == "spacer":
                size = child.params.get("size")
                if size:
                    x += size
                else:
                    x += flex_space
                continue
            cw = next(nw_iter)
            _render_block(ctx, child, Rect(x, inner.y, cw, inner.h))
            x += cw + gap


def _render_column(ctx: RenderContext, block: Block, rect: Rect) -> None:
    """Render a column -- vertical flow with children stacked.

    Children flow top to bottom with configurable gap, alignment,
    and padding.
    """
    padding = _parse_padding(block.params.get("padding"))
    inner = _inner_rect(rect, padding)
    gap = block.params.get("gap", 0)

    _render_children_vertical(ctx, block.children, inner, gap)


def _render_children_vertical(
    ctx: RenderContext,
    children: list[Block],
    rect: Rect,
    gap: int,
) -> None:
    """Render a list of children in vertical flow with gap spacing.

    Each child gets the full width of the container. Heights are
    estimated based on block type and content.

    Args:
        ctx: Render context.
        children: List of child blocks.
        rect: Available rectangle.
        gap: Vertical gap between children in pixels.
    """
    y = rect.y
    for child in children:
        if child.block_type == "spacer":
            size = child.params.get("size")
            y += size if size else gap
            continue

        # Estimate child height for layout
        child_h = _estimate_block_height(ctx, child, rect.w)
        child_rect = Rect(rect.x, y, rect.w, child_h)
        _render_block(ctx, child, child_rect)
        y += child_h + gap


def _render_columns(ctx: RenderContext, block: Block, rect: Rect) -> None:
    """Render multi-column layout with weight ratios.

    Ratios are relative integers: [2, 1] means 2/3 + 1/3 width split.
    The layout engine converts ratios to pixel widths based on available
    space minus gaps and padding.
    """
    padding = _parse_padding(block.params.get("padding"))
    inner = _inner_rect(rect, padding)
    gap = block.params.get("gap", 0)
    ratios = block.params.get("ratios", [1, 1])

    total_ratio = sum(ratios)
    if total_ratio == 0:
        return

    total_gap = gap * max(0, len(ratios) - 1)
    available = max(0, inner.w - total_gap)

    x = inner.x
    for i, child in enumerate(block.children):
        if i >= len(ratios):
            break
        col_w = int(available * ratios[i] / total_ratio)
        child_rect = Rect(x, inner.y, col_w, inner.h)
        _render_block(ctx, child, child_rect)
        x += col_w + gap


def _render_card(ctx: RenderContext, block: Block, rect: Rect) -> None:
    """Render a card with surface background and rounded corners.

    Cards provide visual grouping with a colored background, optional
    shadow, and rounded corners from the theme.
    """
    color_token = block.params.get("color")
    if color_token:
        bg = hex_to_rgb(resolve_color(color_token, ctx.theme))
    else:
        bg = ctx.theme.color_rgb("surface")

    radius = block.params.get("radius", ctx.theme.radius)
    padding = _parse_padding(block.params.get("padding", 16))

    # Draw shadow if theme supports it
    if ctx.theme.shadow:
        shadow_offset = 2
        shadow_color = (0, 0, 0, 40)
        # Draw shadow as a slightly offset darker rectangle
        ctx.draw.rounded_rectangle(
            [rect.x + shadow_offset, rect.y + shadow_offset,
             rect.x + rect.w + shadow_offset, rect.y + rect.h + shadow_offset],
            radius=radius,
            fill=(0, 0, 0),
        )

    # Draw card background
    ctx.draw.rounded_rectangle(
        [rect.x, rect.y, rect.x + rect.w, rect.y + rect.h],
        radius=radius,
        fill=bg,
    )

    inner = _inner_rect(rect, padding)
    _render_children_vertical(ctx, block.children, inner, gap=8)


def _render_spacer(ctx: RenderContext, block: Block, rect: Rect) -> None:
    """Spacer is invisible -- just occupies space in the layout."""
    pass


def _render_divider(ctx: RenderContext, block: Block, rect: Rect) -> None:
    """Render a horizontal or vertical separator line.

    Draws a thin line centered within the block's rect.
    """
    color = _resolve_block_color(block.params.get("color"), ctx, default="dim")
    thickness = block.params.get("thickness", 1)
    orientation = block.params.get("orientation", "horizontal")

    if orientation == "vertical":
        mid_x = rect.x + rect.w // 2
        ctx.draw.line(
            [(mid_x, rect.y), (mid_x, rect.y + rect.h)],
            fill=color, width=thickness,
        )
    else:
        mid_y = rect.y + rect.h // 2
        ctx.draw.line(
            [(rect.x, mid_y), (rect.x + rect.w, mid_y)],
            fill=color, width=thickness,
        )


def _render_repeat(ctx: RenderContext, block: Block, rect: Rect) -> None:
    """Render expanded repeat children.

    By the time the renderer sees a repeat block, the binding resolution
    step has already expanded the template into concrete children. We
    just render them in a vertical flow.
    """
    gap = 8
    _render_children_vertical(ctx, block.children, rect, gap)


# ---------------------------------------------------------------------------
# Content block renderers (13)
# ---------------------------------------------------------------------------


def _render_text(ctx: RenderContext, block: Block, rect: Rect) -> None:
    """Render a text block with size, color, alignment, wrapping, and truncation.

    Supports all theme text sizes (title through small), semantic color
    tokens, horizontal alignment, and line-limited truncation with ellipsis.
    """
    content_raw = block.params.get("content", "")
    content = "" if content_raw is None else str(content_raw)
    size_name = block.params.get("size", "body")
    color_token = block.params.get("color", "default")
    text_align = block.params.get("align", "left")
    max_lines = block.params.get("max_lines")
    weight_override = block.params.get("weight")

    font_style = ctx.theme.font_style(size_name)
    weight = _resolve_weight(weight_override, font_style.weight)
    font = _get_font(ctx.theme.fonts.family, font_style.size, weight)
    color = _resolve_block_color(color_token, ctx)

    lines = _wrap_text(content, font, rect.w, max_lines)

    y = rect.y
    for line in lines:
        lw, lh = _measure_text(line, font)

        if text_align == "center":
            x = rect.x + (rect.w - lw) // 2
        elif text_align == "right":
            x = rect.x + rect.w - lw
        else:
            x = rect.x

        ctx.draw.text((x, y), line, fill=color, font=font)
        y += lh + 4  # Line spacing


def _render_metric(ctx: RenderContext, block: Block, rect: Rect) -> None:
    """Render a metric block: big value with optional label, icon, and change.

    The "big number with context" pattern for temperatures, prices,
    battery levels, etc. The change indicator auto-colors based on
    positive/negative when change_color is "auto".
    """
    value = str(block.params.get("value", ""))
    label = block.params.get("label")
    change = block.params.get("change")

    # Value (large heading font)
    font_style = ctx.theme.font_style("heading")
    font = _get_font(ctx.theme.fonts.family, font_style.size, font_style.weight)
    color = ctx.theme.color_rgb("text")

    y = rect.y
    ctx.draw.text((rect.x, y), value, fill=color, font=font)
    _, vh = _measure_text(value, font)
    y += vh + 4

    # Change indicator (auto-colored: green for positive, red for negative)
    if change:
        change_str = str(change)
        change_color_token = block.params.get("change_color", "auto")
        if change_color_token == "auto":
            if change_str.startswith("+") or (
                change_str[0].isdigit() and not change_str.startswith("-")
            ):
                change_color = ctx.theme.color_rgb("success")
            elif change_str.startswith("-"):
                change_color = ctx.theme.color_rgb("error")
            else:
                change_color = ctx.theme.color_rgb("muted")
        else:
            change_color = _resolve_block_color(change_color_token, ctx)

        cfont = _get_font(ctx.theme.fonts.family, ctx.theme.fonts.caption.size)
        ctx.draw.text((rect.x, y), change_str, fill=change_color, font=cfont)
        _, ch = _measure_text(change_str, cfont)
        y += ch + 4

    # Label (muted caption below)
    if label:
        lfont = _get_font(ctx.theme.fonts.family, ctx.theme.fonts.caption.size)
        lcolor = ctx.theme.color_rgb("muted")
        ctx.draw.text((rect.x, y), str(label), fill=lcolor, font=lfont)


def _render_badge(ctx: RenderContext, block: Block, rect: Rect) -> None:
    """Render a badge -- small colored label (e.g. "Active", "3 new", "Live").

    Draws a pill-shaped background tinted with the badge color, with
    the text rendered on top.
    """
    text = str(block.params.get("text", ""))
    color_token = block.params.get("color", "accent")

    font = _get_font(ctx.theme.fonts.family, ctx.theme.fonts.small.size)
    tw, th = _measure_text(text, font)

    pad = 8
    badge_w = tw + pad * 2
    badge_h = th + pad

    bg_color = _resolve_block_color(color_token, ctx, default="accent")
    # Create a tinted background (dimmed version of the badge color)
    bg_tint = tuple(max(0, c // 3) for c in bg_color)

    ctx.draw.rounded_rectangle(
        [rect.x, rect.y, rect.x + badge_w, rect.y + badge_h],
        radius=badge_h // 2,
        fill=bg_tint,
    )
    ctx.draw.text(
        (rect.x + pad, rect.y + pad // 2),
        text, fill=bg_color, font=font,
    )


def _render_list(ctx: RenderContext, block: Block, rect: Rect) -> None:
    """Render a bulleted, numbered, or check-style list.

    Supports bullet, number, check, and none styles. When max_items
    is set and exceeded, shows "+N more" at the bottom.
    """
    items = block.params.get("items", [])
    style = block.params.get("style", "bullet")
    max_items = block.params.get("max_items")

    if isinstance(items, str):
        items = [items]

    display_items = items[:max_items] if max_items else items

    font = _get_font(ctx.theme.fonts.family, ctx.theme.fonts.body.size)
    color = ctx.theme.color_rgb("text")
    muted = ctx.theme.color_rgb("muted")

    y = rect.y
    for i, item in enumerate(display_items):
        prefix = ""
        if style == "bullet":
            prefix = "  "
        elif style == "number":
            prefix = f"{i + 1}. "
        elif style == "check":
            prefix = "  "

        text = f"{prefix}{item}"
        ctx.draw.text((rect.x, y), text, fill=color, font=font)

        # Draw bullet/check marker
        if style == "bullet":
            cy = y + ctx.theme.fonts.body.size // 2
            ctx.draw.ellipse(
                [rect.x + 2, cy - 3, rect.x + 8, cy + 3],
                fill=muted,
            )
        elif style == "check":
            cy = y + ctx.theme.fonts.body.size // 2
            ctx.draw.rectangle(
                [rect.x + 1, cy - 5, rect.x + 11, cy + 5],
                outline=muted, width=1,
            )

        _, lh = _measure_text(text, font)
        y += lh + 6

    # Show "+N more" overflow indicator
    if max_items and len(items) > max_items:
        remaining = len(items) - max_items
        sfont = _get_font(ctx.theme.fonts.family, ctx.theme.fonts.caption.size)
        ctx.draw.text(
            (rect.x, y), f"+{remaining} more", fill=muted, font=sfont,
        )


def _render_table(ctx: RenderContext, block: Block, rect: Rect) -> None:
    """Render a data table with automatic column sizing.

    Columns are evenly distributed. Headers render in a muted caption
    style with a divider line below. Optional striped rows alternate
    the surface_alt background.
    """
    headers = block.params.get("headers", [])
    rows = block.params.get("rows", [])
    striped = block.params.get("striped", False)
    max_rows = block.params.get("max_rows")

    if isinstance(rows, str):
        rows = []

    display_rows = rows[:max_rows] if max_rows else rows

    if not headers:
        return

    hfont = _get_font(ctx.theme.fonts.family, ctx.theme.fonts.caption.size, 600)
    bfont = _get_font(ctx.theme.fonts.family, ctx.theme.fonts.body.size)
    header_color = ctx.theme.color_rgb("muted")
    text_color = ctx.theme.color_rgb("text")
    stripe_color = ctx.theme.color_rgb("surface_alt")

    col_count = len(headers)
    col_width = rect.w // max(col_count, 1)
    row_height = ctx.theme.fonts.body.size + 16

    # Headers
    y = rect.y
    for i, h in enumerate(headers):
        x = rect.x + i * col_width
        ctx.draw.text((x, y), str(h), fill=header_color, font=hfont)
    y += row_height

    # Divider under headers
    ctx.draw.line(
        [(rect.x, y), (rect.x + rect.w, y)],
        fill=ctx.theme.color_rgb("dim"), width=1,
    )
    y += 4

    # Data rows
    for row_idx, row in enumerate(display_rows):
        if striped and row_idx % 2 == 1:
            ctx.draw.rectangle(
                [rect.x, y, rect.x + rect.w, y + row_height],
                fill=stripe_color,
            )
        for i, cell in enumerate(row):
            if i < col_count:
                x = rect.x + i * col_width
                ctx.draw.text(
                    (x, y + 4), str(cell), fill=text_color, font=bfont,
                )
        y += row_height


def _render_key_value(ctx: RenderContext, block: Block, rect: Rect) -> None:
    """Render key-value pairs in a two-column layout.

    Keys render in muted medium-weight font on the left (1/3 width),
    values in normal text on the right (2/3 width).
    """
    data = block.params.get("data", {})
    if isinstance(data, str):
        data = {}

    kfont = _get_font(ctx.theme.fonts.family, ctx.theme.fonts.body.size, 500)
    vfont = _get_font(ctx.theme.fonts.family, ctx.theme.fonts.body.size)
    key_color = ctx.theme.color_rgb("muted")
    val_color = ctx.theme.color_rgb("text")

    y = rect.y
    key_width = rect.w // 3

    for key, value in data.items():
        ctx.draw.text((rect.x, y), str(key), fill=key_color, font=kfont)
        ctx.draw.text(
            (rect.x + key_width, y), str(value), fill=val_color, font=vfont,
        )
        _, lh = _measure_text(str(key), kfont)
        y += lh + 8


_icon_cache: dict[tuple[str, int, tuple[int, int, int]], Image.Image | None] = {}


def _load_lucide_icon(name: str, px: int, color: tuple[int, int, int]) -> Image.Image | None:
    """Rasterize a bundled Lucide SVG to a PIL image at `px` with `color`.

    Returns None if the icon isn't bundled or cairosvg isn't installed.
    Results are cached by (name, px, color) for reuse.
    """
    key = (name, px, color)
    if key in _icon_cache:
        return _icon_cache[key]

    path = _LUCIDE_DIR / f"{name}.svg"
    if not path.is_file():
        _icon_cache[key] = None
        return None

    try:
        import cairosvg
    except ImportError:
        _icon_cache[key] = None
        return None

    try:
        svg_text = path.read_text()
        # Lucide strokes use currentColor. Wrap in a container that sets
        # color so strokes and fills pick up the requested tint.
        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        svg_text = svg_text.replace("<svg", f'<svg color="{hex_color}"', 1)
        png_bytes = cairosvg.svg2png(
            bytestring=svg_text.encode("utf-8"),
            output_width=px,
            output_height=px,
        )
        import io
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    except Exception:  # noqa: BLE001
        logger.exception("Failed to rasterize Lucide icon '%s'", name)
        img = None

    _icon_cache[key] = img
    return img


def _render_icon(ctx: RenderContext, block: Block, rect: Rect) -> None:
    """Render a Lucide icon.

    Loads the named SVG from the bundled asset pack, rasterizes via cairosvg
    at the requested pixel size, and blits it at the block rect. Falls back
    to a circled-letter placeholder when the SVG or cairosvg is unavailable.
    """
    name_raw = block.params.get("name", "")
    name = "" if name_raw is None else str(name_raw)
    size_name = block.params.get("size", "md")
    color_token = block.params.get("color")

    px = ICON_SIZES.get(size_name, 24)
    color = _resolve_block_color(color_token, ctx, default="text")

    icon_img = _load_lucide_icon(name, px, color) if name else None
    if icon_img is not None:
        ctx.image.paste(icon_img, (rect.x, rect.y), icon_img)
        return

    # Fallback: outlined circle with the icon name's initial
    cx = rect.x + px // 2
    cy = rect.y + px // 2
    r = px // 2 - 2
    ctx.draw.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        outline=color, width=1,
    )
    if name:
        sfont = _get_font(ctx.theme.fonts.family, max(8, px // 3))
        initial = name[0].upper()
        tw, th = _measure_text(initial, sfont)
        ctx.draw.text(
            (cx - tw // 2, cy - th // 2), initial, fill=color, font=sfont,
        )


def _render_emoji(ctx: RenderContext, block: Block, rect: Rect) -> None:
    """Render an emoji placeholder.

    Until Twemoji image assets are loaded, renders the emoji name as
    text. Size options: md, lg, xl.
    """
    name_raw = block.params.get("name", "")
    name = "" if name_raw is None else str(name_raw)
    size_name = block.params.get("size", "md")
    px = ICON_SIZES.get(size_name, 24)

    font = _get_font(ctx.theme.fonts.family, px)
    color = ctx.theme.color_rgb("text")
    # Render the name (or first two chars) as a placeholder
    display_text = name if len(name) <= 4 else name[:2]
    ctx.draw.text((rect.x, rect.y), display_text, fill=color, font=font)


def _render_image(ctx: RenderContext, block: Block, rect: Rect) -> None:
    """Render an image block.

    Supports three source types: photo:id, url:https://..., asset:filename.
    For now, renders a placeholder rectangle with the source label. Future
    versions will load actual image data from the photo store or cache.
    """
    source = str(block.params.get("source", ""))
    fit = block.params.get("fit", "cover")
    radius = block.params.get("radius", ctx.theme.radius)

    # Try to load actual image from photo store
    actual_image = _try_load_image(source)

    if actual_image is not None:
        # Resize image to fit the rect according to fit mode
        resized = _fit_image(actual_image, rect.w, rect.h, fit)
        # Paste image into the main image at the rect position
        # Center the resized image within the rect
        paste_x = rect.x + (rect.w - resized.width) // 2
        paste_y = rect.y + (rect.h - resized.height) // 2
        ctx.image.paste(resized, (paste_x, paste_y))
    else:
        # Placeholder: surface_alt rectangle with source label
        bg = ctx.theme.color_rgb("surface_alt")
        ctx.draw.rounded_rectangle(
            [rect.x, rect.y, rect.x + rect.w, rect.y + rect.h],
            radius=radius, fill=bg,
        )
        font = _get_font(ctx.theme.fonts.family, ctx.theme.fonts.caption.size)
        label = source[:30] if source else "[image]"
        tw, th = _measure_text(label, font)
        ctx.draw.text(
            (rect.x + (rect.w - tw) // 2, rect.y + (rect.h - th) // 2),
            label, fill=ctx.theme.color_rgb("muted"), font=font,
        )


def _try_load_image(source: str) -> Image.Image | None:
    """Try to load an image from the given source string.

    Args:
        source: Image source (photo:id, url:..., asset:filename, or file path).

    Returns:
        PIL Image if loaded successfully, None otherwise.
    """
    if not source:
        return None

    # Direct file path (for preview/testing)
    from pathlib import Path
    if Path(source).is_file():
        try:
            return Image.open(source)
        except Exception:
            return None

    # photo:id -- would need photo store access (not available in renderer)
    # url:... -- would need HTTP fetch (not available in static render)
    # asset:filename -- would need asset directory lookup
    # All of these will be resolved by the display manager before rendering
    return None


def _fit_image(
    img: Image.Image,
    target_w: int,
    target_h: int,
    fit: str = "cover",
) -> Image.Image:
    """Resize an image to fit a target rectangle.

    Args:
        img: Source image.
        target_w: Target width.
        target_h: Target height.
        fit: Fit mode -- "cover" (fill, may crop), "contain" (fit inside),
             or "fill" (stretch to exact size).

    Returns:
        Resized PIL Image.
    """
    if fit == "fill":
        return img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    src_w, src_h = img.size
    if src_w == 0 or src_h == 0:
        return img

    src_ratio = src_w / src_h
    target_ratio = target_w / target_h if target_h > 0 else 1.0

    if fit == "cover":
        # Scale to cover the target, then crop
        if src_ratio > target_ratio:
            new_h = target_h
            new_w = int(new_h * src_ratio)
        else:
            new_w = target_w
            new_h = int(new_w / src_ratio)
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        # Center crop
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        return resized.crop((left, top, left + target_w, top + target_h))
    else:
        # contain: scale to fit inside
        if src_ratio > target_ratio:
            new_w = target_w
            new_h = int(new_w / src_ratio)
        else:
            new_h = target_h
            new_w = int(new_h * src_ratio)
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _render_chart(ctx: RenderContext, block: Block, rect: Rect) -> None:
    """Render a simple chart (line, bar, or area).

    Charts are rendered using PIL drawing primitives -- no external
    charting library. Supports single or multi-series data, grid lines,
    and area fills.
    """
    chart_type = block.params.get("type", "line")
    data = block.params.get("data", [])
    series = block.params.get("series", [])
    height = block.params.get("height", 200)
    color_token = block.params.get("color", "accent")
    show_grid = block.params.get("show_grid", True)
    fill_opacity = block.params.get("fill_opacity", 0.15)
    show_dots = block.params.get("show_dots", False)
    show_legend = block.params.get("show_legend", False)

    chart_rect = Rect(rect.x, rect.y, rect.w, min(height, rect.h))

    # Collect all data series as (values, color) tuples
    all_series: list[tuple[list[float], tuple[int, int, int], str | None]] = []
    if series:
        for s in series:
            s_data = s.get("data", [])
            if isinstance(s_data, list):
                s_color = _resolve_block_color(s.get("color"), ctx, default="accent")
                s_label = s.get("label")
                all_series.append((s_data, s_color, s_label))
    elif isinstance(data, list) and data:
        color = _resolve_block_color(color_token, ctx, default="accent")
        all_series.append((data, color, None))

    if not all_series:
        # Empty chart placeholder
        ctx.draw.rounded_rectangle(
            [chart_rect.x, chart_rect.y,
             chart_rect.x + chart_rect.w, chart_rect.y + chart_rect.h],
            radius=4, fill=ctx.theme.color_rgb("surface"),
        )
        return

    # Reserve space for legend at bottom if enabled
    legend_h = 0
    if show_legend and any(label for _, _, label in all_series):
        legend_h = ctx.theme.fonts.caption.size + 12
        chart_rect = Rect(
            chart_rect.x, chart_rect.y,
            chart_rect.w, max(20, chart_rect.h - legend_h),
        )

    # Grid lines
    if show_grid:
        grid_color = ctx.theme.color_rgb("dim")
        for i in range(5):
            gy = chart_rect.y + int(chart_rect.h * i / 4)
            ctx.draw.line(
                [(chart_rect.x, gy), (chart_rect.x + chart_rect.w, gy)],
                fill=grid_color, width=1,
            )

    # Find global min/max across all series
    all_vals = [
        v for vals, _, _ in all_series
        for v in vals
        if isinstance(v, (int, float))
    ]
    if not all_vals:
        return

    v_min = min(all_vals)
    v_max = max(all_vals)
    v_range = v_max - v_min or 1

    # Render each series
    for values, color, label in all_series:
        numeric = [v for v in values if isinstance(v, (int, float))]
        if not numeric:
            continue

        n = len(numeric)
        points: list[tuple[int, int]] = []
        for i, v in enumerate(numeric):
            x = chart_rect.x + int(chart_rect.w * i / max(n - 1, 1))
            y = chart_rect.y + chart_rect.h - int(
                (v - v_min) / v_range * chart_rect.h
            )
            points.append((x, y))

        if chart_type == "bar":
            bar_w = max(2, chart_rect.w // n - 2)
            for i, (x, y) in enumerate(points):
                bx = chart_rect.x + int(chart_rect.w * i / n)
                ctx.draw.rectangle(
                    [bx, y, bx + bar_w, chart_rect.y + chart_rect.h],
                    fill=color,
                )
        elif chart_type in ("line", "area"):
            if len(points) >= 2:
                if chart_type == "area":
                    # Fill under the line: interpolate between background and
                    # the series color by fill_opacity. Works correctly on
                    # both dark and light themes (simple division darkened
                    # the accent on every bg, which looked muddy on light
                    # themes and almost-black on dark themes).
                    bg = ctx.theme.color_rgb("surface")
                    muted_fill = tuple(
                        int(bg[i] + (color[i] - bg[i]) * fill_opacity)
                        for i in range(3)
                    )
                    fill_points = list(points) + [
                        (points[-1][0], chart_rect.y + chart_rect.h),
                        (points[0][0], chart_rect.y + chart_rect.h),
                    ]
                    ctx.draw.polygon(fill_points, fill=muted_fill)

                # Draw the line
                ctx.draw.line(points, fill=color, width=2)

                # Draw data point dots if enabled
                if show_dots:
                    for px, py in points:
                        dot_r = 3
                        ctx.draw.ellipse(
                            [px - dot_r, py - dot_r, px + dot_r, py + dot_r],
                            fill=color,
                        )

    # Legend
    if show_legend and legend_h > 0:
        legend_y = chart_rect.y + chart_rect.h + 4
        legend_x = chart_rect.x
        lfont = _get_font(ctx.theme.fonts.family, ctx.theme.fonts.caption.size)
        for _, color, label in all_series:
            if label:
                # Color swatch
                sw = 10
                sy = legend_y + (ctx.theme.fonts.caption.size - sw) // 2
                ctx.draw.rectangle(
                    [legend_x, sy, legend_x + sw, sy + sw],
                    fill=color,
                )
                legend_x += sw + 6
                ctx.draw.text(
                    (legend_x, legend_y), label,
                    fill=ctx.theme.color_rgb("muted"), font=lfont,
                )
                tw, _ = _measure_text(label, lfont)
                legend_x += tw + 16


def _render_progress(ctx: RenderContext, block: Block, rect: Rect) -> None:
    """Render a progress/capacity bar.

    Value is 0.0 to 1.0. When color is "auto", the bar shifts from
    green (success) through yellow (warning) to red (error) as it
    fills up.
    """
    value = block.params.get("value", 0.0)
    label = block.params.get("label")
    color_token = block.params.get("color", "auto")

    # Coerce to float, treating None / missing / unparseable as 0. A
    # progress bar bound to a data source that hasn't loaded yet (or
    # to a typo'd field) used to crash the whole render here.
    if value is None:
        value = 0.0
    elif isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            value = 0.0
    else:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0

    value = max(0.0, min(1.0, value))

    # Determine bar color
    if color_token == "auto":
        if value < 0.6:
            bar_color = ctx.theme.color_rgb("success")
        elif value < 0.85:
            bar_color = ctx.theme.color_rgb("warning")
        else:
            bar_color = ctx.theme.color_rgb("error")
    else:
        bar_color = _resolve_block_color(color_token, ctx, default="accent")

    bar_h = 12
    bar_y = rect.y
    radius = bar_h // 2

    # Label above the bar
    if label:
        font = _get_font(ctx.theme.fonts.family, ctx.theme.fonts.caption.size)
        ctx.draw.text(
            (rect.x, rect.y), str(label),
            fill=ctx.theme.color_rgb("text"), font=font,
        )
        bar_y += ctx.theme.fonts.caption.size + 6

    # Background track
    ctx.draw.rounded_rectangle(
        [rect.x, bar_y, rect.x + rect.w, bar_y + bar_h],
        radius=radius, fill=ctx.theme.color_rgb("surface_alt"),
    )

    # Filled portion
    fill_w = int(rect.w * value)
    if fill_w > 0:
        ctx.draw.rounded_rectangle(
            [rect.x, bar_y, rect.x + fill_w, bar_y + bar_h],
            radius=radius, fill=bar_color,
        )


def _render_clock(ctx: RenderContext, block: Block, rect: Rect) -> None:
    """Render a clock display.

    Self-updating live block. Reads the current time at render and
    displays it in 12h or 24h format with optional date and seconds.
    The display manager triggers re-renders to keep it ticking.
    """
    fmt = block.params.get("format", "12h")
    show_date = block.params.get("show_date", True)
    show_seconds = block.params.get("show_seconds", False)
    size_name = block.params.get("size", "lg")

    now = datetime.now()
    hour = now.hour
    if fmt == "12h":
        hour = hour % 12 or 12
        ampm = " AM" if now.hour < 12 else " PM"
    else:
        ampm = ""

    time_str = f"{hour}:{now.minute:02d}"
    if show_seconds:
        time_str += f":{now.second:02d}"
    time_str += ampm

    font_size = CLOCK_SIZES.get(size_name, 56)
    font = _get_font(ctx.theme.fonts.family, font_size, 700)
    color = ctx.theme.color_rgb("text")

    tw, th = _measure_text(time_str, font)
    # Center horizontally, slightly above vertical center
    x = rect.x + (rect.w - tw) // 2
    y = rect.y + (rect.h - th) // 3

    ctx.draw.text((x, y), time_str, fill=color, font=font)

    if show_date:
        date_str = now.strftime("%A, %B %d")
        dfont = _get_font(ctx.theme.fonts.family, ctx.theme.fonts.subtitle.size)
        dcolor = ctx.theme.color_rgb("muted")
        dw, dh = _measure_text(date_str, dfont)
        ctx.draw.text(
            (rect.x + (rect.w - dw) // 2, y + th + 12),
            date_str, fill=dcolor, font=dfont,
        )


def _render_countdown(ctx: RenderContext, block: Block, rect: Rect) -> None:
    """Render a countdown to a target datetime.

    Parses the target ISO datetime, computes the remaining time, and
    displays it in H:MM:SS format. Shows "0:00:00" when the target
    has passed.
    """
    target_str = str(block.params.get("target", ""))
    label = block.params.get("label")

    # Parse target and compute remaining time
    remaining = "0:00:00"
    try:
        target = datetime.fromisoformat(target_str)
        delta = target - datetime.now()
        if delta.total_seconds() > 0:
            hours = int(delta.total_seconds() // 3600)
            minutes = int((delta.total_seconds() % 3600) // 60)
            seconds = int(delta.total_seconds() % 60)
            remaining = f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            remaining = "0:00:00"
    except (ValueError, TypeError):
        remaining = target_str or "--:--:--"

    y = rect.y
    if label:
        font = _get_font(ctx.theme.fonts.family, ctx.theme.fonts.subtitle.size)
        ctx.draw.text(
            (rect.x, y), str(label),
            fill=ctx.theme.color_rgb("muted"), font=font,
        )
        y += ctx.theme.fonts.subtitle.size + 8

    font = _get_font(ctx.theme.fonts.family, ctx.theme.fonts.title.size, 700)
    ctx.draw.text(
        (rect.x, y), remaining,
        fill=ctx.theme.color_rgb("accent"), font=font,
    )


# ---------------------------------------------------------------------------
# Composite widget renderers (2)
# ---------------------------------------------------------------------------


def _render_weather_widget(ctx: RenderContext, block: Block, rect: Rect) -> None:
    """Render the pre-built weather widget.

    A complete weather card: icon, temperature, condition text, and
    optionally a forecast row. Reads from the weather data source.
    """
    source_name = block.params.get("data_source", "weather")
    data = ctx.data.get(source_name, {})

    # Draw card background
    bg = ctx.theme.color_rgb("surface")
    ctx.draw.rounded_rectangle(
        [rect.x, rect.y, rect.x + rect.w, rect.y + rect.h],
        radius=ctx.theme.radius, fill=bg,
    )

    padding = _parse_padding(16)
    inner = _inner_rect(rect, padding)

    y = inner.y

    # Icon + temperature row
    temp = data.get("temp", "?")
    condition = data.get("condition", "")
    icon_name = data.get("icon", "cloud")

    # Icon placeholder
    icon_px = 48
    _render_icon(
        ctx,
        Block(params={"name": icon_name, "size": "xl"}),
        Rect(inner.x, y, icon_px, icon_px),
    )

    # Temperature next to icon
    tfont = _get_font(ctx.theme.fonts.family, ctx.theme.fonts.title.size, 700)
    ctx.draw.text(
        (inner.x + icon_px + 12, y),
        f"{temp}\u00b0",
        fill=ctx.theme.color_rgb("text"), font=tfont,
    )
    y += icon_px + 8

    # Condition text
    cfont = _get_font(ctx.theme.fonts.family, ctx.theme.fonts.body.size)
    ctx.draw.text(
        (inner.x, y), condition,
        fill=ctx.theme.color_rgb("muted"), font=cfont,
    )
    y += ctx.theme.fonts.body.size + 12

    # Forecast row (if data available)
    forecast = data.get("forecast", [])
    if forecast:
        dfont = _get_font(ctx.theme.fonts.family, ctx.theme.fonts.caption.size)
        sfont = _get_font(ctx.theme.fonts.family, ctx.theme.fonts.small.size)
        num_days = min(5, len(forecast))
        day_w = inner.w // max(num_days, 1)

        for i, day_data in enumerate(forecast[:num_days]):
            dx = inner.x + i * day_w
            day_name = day_data.get("day", "")
            high = day_data.get("high", "")
            low = day_data.get("low", "")

            ctx.draw.text(
                (dx, y), day_name,
                fill=ctx.theme.color_rgb("muted"), font=dfont,
            )
            ctx.draw.text(
                (dx, y + ctx.theme.fonts.caption.size + 4),
                f"{high}/{low}",
                fill=ctx.theme.color_rgb("text"), font=sfont,
            )


def _render_calendar_widget(ctx: RenderContext, block: Block, rect: Rect) -> None:
    """Render the pre-built calendar widget.

    Shows today's agenda with time slots and event details. Reads from
    the calendar data source.
    """
    source_name = block.params.get("data_source", "calendar")
    data = ctx.data.get(source_name, {})
    events = data.get("events", [])

    # Draw card background
    bg = ctx.theme.color_rgb("surface")
    ctx.draw.rounded_rectangle(
        [rect.x, rect.y, rect.x + rect.w, rect.y + rect.h],
        radius=ctx.theme.radius, fill=bg,
    )

    padding = _parse_padding(16)
    inner = _inner_rect(rect, padding)

    # Title
    hfont = _get_font(ctx.theme.fonts.family, ctx.theme.fonts.heading.size, 600)
    ctx.draw.text(
        (inner.x, inner.y), "Today",
        fill=ctx.theme.color_rgb("text"), font=hfont,
    )

    y = inner.y + ctx.theme.fonts.heading.size + 12
    efont = _get_font(ctx.theme.fonts.family, ctx.theme.fonts.body.size)
    tfont = _get_font(ctx.theme.fonts.family, ctx.theme.fonts.caption.size)

    for event in events[:6]:
        time_str = event.get("time", "")
        title = event.get("title", "")

        ctx.draw.text(
            (inner.x, y), time_str,
            fill=ctx.theme.color_rgb("muted"), font=tfont,
        )
        ctx.draw.text(
            (inner.x + 80, y), title,
            fill=ctx.theme.color_rgb("text"), font=efont,
        )
        y += ctx.theme.fonts.body.size + 12


# ---------------------------------------------------------------------------
# Meta block renderers (2)
# ---------------------------------------------------------------------------


def _render_rotate(ctx: RenderContext, block: Block, rect: Rect) -> None:
    """Render a rotate block.

    For static rendering (preview), renders the first child only.
    The live display manager handles the rotation timer and re-renders
    with updated {current} bindings.
    """
    if block.children:
        _render_block(ctx, block.children[0], rect)


def _render_page_dots(ctx: RenderContext, block: Block, rect: Rect) -> None:
    """Render page dots indicator.

    Draws a row of small circles at the bottom of the block rect.
    The first dot is highlighted (active) and the rest are dimmed.
    In live rendering, the display manager updates which dot is active.
    """
    color_token = block.params.get("color", "accent")
    color = _resolve_block_color(color_token, ctx, default="accent")
    dim = ctx.theme.color_rgb("dim")

    dot_r = 4
    dot_spacing = 16
    num_dots = 5  # Placeholder count for preview
    total_w = num_dots * dot_spacing

    x = rect.x + (rect.w - total_w) // 2
    y = rect.y + rect.h - dot_r * 2 - 8

    for i in range(num_dots):
        cx = x + i * dot_spacing + dot_r
        if i == 0:  # Active dot
            ctx.draw.ellipse(
                [cx - dot_r, y - dot_r, cx + dot_r, y + dot_r],
                fill=color,
            )
        else:
            ctx.draw.ellipse(
                [cx - dot_r, y - dot_r, cx + dot_r, y + dot_r],
                fill=dim,
            )


# ---------------------------------------------------------------------------
# Height estimation
# ---------------------------------------------------------------------------


def _estimate_block_height(
    ctx: RenderContext,
    block: Block,
    available_width: int,
) -> int:
    """Estimate the height a block needs for layout purposes.

    This is used during vertical flow layout to determine how much
    vertical space to allocate for each child block. The estimates
    are approximate -- they don't account for all layout interactions
    but are sufficient for reasonable rendering.

    Args:
        ctx: Render context (for font metrics).
        block: The block to estimate.
        available_width: The width available for this block.

    Returns:
        Estimated height in pixels.
    """
    bt = block.block_type

    if bt == "text":
        size_name = block.params.get("size", "body")
        content_raw = block.params.get("content", "")
        content = "" if content_raw is None else str(content_raw)
        font_style = ctx.theme.font_style(size_name)
        font = _get_font(ctx.theme.fonts.family, font_style.size)
        max_lines = block.params.get("max_lines")
        lines = _wrap_text(content, font, available_width, max_lines)
        return len(lines) * (font_style.size + 4) + 4

    if bt == "metric":
        h = ctx.theme.fonts.heading.size + 8
        if block.params.get("change"):
            h += ctx.theme.fonts.caption.size + 4
        if block.params.get("label"):
            h += ctx.theme.fonts.caption.size + 4
        return h

    if bt == "badge":
        return ctx.theme.fonts.small.size + 16

    if bt == "list":
        items = block.params.get("items", [])
        if isinstance(items, str):
            items = [items]
        count = len(items)
        max_items = block.params.get("max_items")
        if max_items and count > max_items:
            count = max_items + 1  # +1 for the "+N more" line
        return count * (ctx.theme.fonts.body.size + 6) + 4

    if bt == "table":
        rows = block.params.get("rows", [])
        if isinstance(rows, str):
            rows = []
        max_rows = block.params.get("max_rows")
        display_rows = rows[:max_rows] if max_rows else rows
        row_h = ctx.theme.fonts.body.size + 16
        return row_h * (len(display_rows) + 1) + 8  # +1 for header

    if bt == "key_value":
        data = block.params.get("data", {})
        if isinstance(data, str):
            return 40
        return len(data) * (ctx.theme.fonts.body.size + 8)

    if bt == "chart":
        return block.params.get("height", 200)

    if bt == "progress":
        h = 12  # Bar height
        if block.params.get("label"):
            h += ctx.theme.fonts.caption.size + 6
        return h

    if bt == "clock":
        size_name = block.params.get("size", "lg")
        base = CLOCK_SIZES.get(size_name, 56)
        if block.params.get("show_date", True):
            base += ctx.theme.fonts.subtitle.size + 16
        return base + 20

    if bt == "countdown":
        h = ctx.theme.fonts.title.size + 8
        if block.params.get("label"):
            h += ctx.theme.fonts.subtitle.size + 8
        return h

    if bt in ("icon", "emoji"):
        size_name = block.params.get("size", "md")
        return ICON_SIZES.get(size_name, 24)

    if bt == "image":
        return min(200, available_width * 9 // 16)  # 16:9 aspect ratio

    if bt == "spacer":
        return block.params.get("size", 16)

    if bt == "divider":
        return block.params.get("thickness", 1) + 8

    if bt in ("card", "weather_widget", "calendar_widget"):
        # Estimate from children plus padding
        padding = _parse_padding(block.params.get("padding", 16))
        inner_w = available_width - padding[1] - padding[3]
        children_h = sum(
            _estimate_block_height(ctx, c, inner_w) + 8
            for c in block.children
        )
        # For weather/calendar widgets without children, provide a minimum
        if not block.children and bt in ("weather_widget", "calendar_widget"):
            children_h = 120
        return children_h + padding[0] + padding[2] + 16

    if bt in ("row", "columns"):
        # Height is the max of children heights
        if block.children:
            return max(
                _estimate_block_height(
                    ctx, c,
                    available_width // max(len(block.children), 1),
                )
                for c in block.children
            )
        return 0

    if bt in ("column", "stack"):
        # Height is the sum of children heights plus gaps
        padding = _parse_padding(block.params.get("padding"))
        gap = block.params.get("gap", 0)
        inner_w = available_width - padding[1] - padding[3]
        total = padding[0] + padding[2]
        for i, c in enumerate(block.children):
            total += _estimate_block_height(ctx, c, inner_w)
            if i < len(block.children) - 1:
                total += gap
        return total

    if bt == "repeat":
        # Estimate based on max or expanded children count
        if block.children and len(block.children) > 1:
            # Already expanded by binding resolution
            return sum(
                _estimate_block_height(ctx, c, available_width) + 8
                for c in block.children
            )
        n = block.params.get("max", 3)
        if block.children:
            item_h = _estimate_block_height(ctx, block.children[0], available_width)
            return n * (item_h + 8)
        return n * 40

    if bt == "page_dots":
        return 20

    if bt == "rotate":
        # Height of the first child (what we render in preview)
        if block.children:
            return _estimate_block_height(ctx, block.children[0], available_width)
        return 40

    # Default fallback
    return 40


def _estimate_block_width(
    ctx: RenderContext,
    block: Block,
    available_width: int,
) -> int:
    """Estimate the natural width a block needs.

    Used by row layout with align=center/end to cluster children at their
    content width rather than stretching to equal slots. Conservative —
    when a block is complex, returns a fraction of the available width.
    """
    bt = block.block_type

    if bt == "text":
        content_raw = block.params.get("content", "")
        content = "" if content_raw is None else str(content_raw)
        size_name = block.params.get("size", "body")
        font_style = ctx.theme.font_style(size_name)
        weight = _resolve_weight(block.params.get("weight"), font_style.weight)
        font = _get_font(ctx.theme.fonts.family, font_style.size, weight)
        # Single-line measurement; wrapping within a centred cluster is rare.
        w, _ = _measure_text(content, font)
        return w + 2

    if bt == "badge":
        text = str(block.params.get("text", ""))
        font = _get_font(ctx.theme.fonts.family, ctx.theme.fonts.small.size, 600)
        w, _ = _measure_text(text, font)
        return w + 24  # padding on each side

    if bt in ("icon", "emoji"):
        size_name = block.params.get("size", "md")
        return ICON_SIZES.get(size_name, 24)

    if bt == "clock":
        size_name = block.params.get("size", "lg")
        base = CLOCK_SIZES.get(size_name, 56)
        # Approximate width of "12:34 PM" at that size
        return int(base * 4.2)

    if bt == "metric":
        value = str(block.params.get("value", ""))
        label = str(block.params.get("label", ""))
        vfont = _get_font(
            ctx.theme.fonts.family, ctx.theme.fonts.heading.size, 700,
        )
        lfont = _get_font(
            ctx.theme.fonts.family, ctx.theme.fonts.caption.size,
        )
        vw, _ = _measure_text(value, vfont)
        lw, _ = _measure_text(label, lfont)
        return max(vw, lw) + 8

    if bt == "spacer":
        return block.params.get("size") or 0

    if bt == "divider":
        # Mostly used horizontally — tiny when inline
        return block.params.get("thickness", 1) + 8

    # Complex blocks: give them a sensible share of available width
    return min(available_width, max(80, available_width // 3))


# ---------------------------------------------------------------------------
# Block renderer registry
# ---------------------------------------------------------------------------

_BLOCK_RENDERERS: dict[str, Any] = {
    # Layout containers (7)
    "row": _render_row,
    "column": _render_column,
    "stack": _render_column,  # alias for column
    "columns": _render_columns,
    "card": _render_card,
    "spacer": _render_spacer,
    "divider": _render_divider,
    "repeat": _render_repeat,
    # Content blocks (13)
    "text": _render_text,
    "metric": _render_metric,
    "badge": _render_badge,
    "list": _render_list,
    "table": _render_table,
    "key_value": _render_key_value,
    "icon": _render_icon,
    "emoji": _render_emoji,
    "image": _render_image,
    "chart": _render_chart,
    "progress": _render_progress,
    "clock": _render_clock,
    "countdown": _render_countdown,
    # Composite widgets (2)
    "weather_widget": _render_weather_widget,
    "calendar_widget": _render_calendar_widget,
    # Meta blocks (2)
    "rotate": _render_rotate,
    "page_dots": _render_page_dots,
}
