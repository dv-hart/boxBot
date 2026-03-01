"""Block definitions for the boxBot display system.

Displays are composed from a fixed set of blocks. Layout containers arrange
children. Content blocks display data. The agent never specifies pixel
coordinates — containers handle all positioning.

Block categories:
- Layout containers (7): row, column, columns, card, spacer, divider, repeat
- Content blocks (13): text, metric, badge, list, table, key_value, icon,
  emoji, image, chart, progress, clock, countdown
- Composite widgets (2): weather_widget, calendar_widget
- Meta blocks (2): rotate, page_dots

Usage:
    from boxbot.displays.blocks import TextBlock, RowBlock, parse_block

    block = parse_block({"type": "text", "content": "Hello", "size": "title"})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


@dataclass
class Block:
    """Base class for all display blocks.

    Every block has a type identifier, optional children, and a params
    dict for block-specific configuration. The measure/render cycle is
    handled by the renderer — blocks are data structures, not renderers.
    """

    block_type: str = ""
    children: list[Block] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this block (and children) to a JSON-compatible dict."""
        d: dict[str, Any] = {"type": self.block_type}
        if self.params:
            d.update(self.params)
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


# ---------------------------------------------------------------------------
# Layout Containers (7)
# ---------------------------------------------------------------------------


@dataclass
class RowBlock(Block):
    """Horizontal flow — children side by side."""

    block_type: str = field(default="row", init=False)
    gap: int = 0
    align: str = "start"  # start, center, end, spread
    padding: int | list[int] = 0

    def __post_init__(self) -> None:
        self.params = {
            k: v for k, v in {
                "gap": self.gap,
                "align": self.align,
                "padding": self.padding,
            }.items() if v != (0 if k != "align" else "start")
        }


@dataclass
class ColumnBlock(Block):
    """Vertical flow — children stacked."""

    block_type: str = field(default="column", init=False)
    gap: int = 0
    align: str = "start"
    padding: int | list[int] = 0

    def __post_init__(self) -> None:
        self.params = {
            k: v for k, v in {
                "gap": self.gap,
                "align": self.align,
                "padding": self.padding,
            }.items() if v != (0 if k != "align" else "start")
        }


@dataclass
class ColumnsBlock(Block):
    """Multi-column layout with weight ratios.

    ratios is a list of ints: [2, 1] means 2/3 + 1/3 width split.
    """

    block_type: str = field(default="columns", init=False)
    ratios: list[int] = field(default_factory=lambda: [1, 1])
    gap: int = 0
    padding: int | list[int] = 0

    def __post_init__(self) -> None:
        self.params = {
            "ratios": self.ratios,
        }
        if self.gap:
            self.params["gap"] = self.gap
        if self.padding:
            self.params["padding"] = self.padding


@dataclass
class CardBlock(Block):
    """Surface with background, rounded corners, and optional shadow."""

    block_type: str = field(default="card", init=False)
    color: str | None = None  # defaults to theme surface
    radius: int | None = None  # defaults to theme radius
    padding: int | list[int] = 16

    def __post_init__(self) -> None:
        self.params = {"padding": self.padding}
        if self.color:
            self.params["color"] = self.color
        if self.radius is not None:
            self.params["radius"] = self.radius


@dataclass
class SpacerBlock(Block):
    """Fixed or flexible space between elements."""

    block_type: str = field(default="spacer", init=False)
    size: int | None = None  # None = flexible

    def __post_init__(self) -> None:
        self.params = {}
        if self.size is not None:
            self.params["size"] = self.size


@dataclass
class DividerBlock(Block):
    """Separator line (horizontal or vertical)."""

    block_type: str = field(default="divider", init=False)
    color: str | None = None
    thickness: int = 1
    orientation: str = "horizontal"  # horizontal, vertical

    def __post_init__(self) -> None:
        self.params = {}
        if self.color:
            self.params["color"] = self.color
        if self.thickness != 1:
            self.params["thickness"] = self.thickness
        if self.orientation != "horizontal":
            self.params["orientation"] = self.orientation


@dataclass
class RepeatBlock(Block):
    """Iterate over a data array, stamping a template per item.

    The single child is the template block. Fields on the current item
    use {.field} syntax.
    """

    block_type: str = field(default="repeat", init=False)
    source: str = ""
    max: int | None = None
    highlight_active: bool = False

    def __post_init__(self) -> None:
        self.params = {"source": self.source}
        if self.max is not None:
            self.params["max"] = self.max
        if self.highlight_active:
            self.params["highlight_active"] = True


# ---------------------------------------------------------------------------
# Content Blocks (13)
# ---------------------------------------------------------------------------


@dataclass
class TextBlock(Block):
    """Text display with size, color, weight, alignment, and truncation."""

    block_type: str = field(default="text", init=False)
    content: str = ""
    size: str = "body"
    color: str = "default"
    weight: str | None = None
    align: str = "left"
    max_lines: int | None = None
    animation: str = "none"
    min_width: int | None = None

    def __post_init__(self) -> None:
        self.params = {"content": self.content}
        if self.size != "body":
            self.params["size"] = self.size
        if self.color != "default":
            self.params["color"] = self.color
        if self.weight:
            self.params["weight"] = self.weight
        if self.align != "left":
            self.params["align"] = self.align
        if self.max_lines is not None:
            self.params["max_lines"] = self.max_lines
        if self.animation != "none":
            self.params["animation"] = self.animation
        if self.min_width is not None:
            self.params["min_width"] = self.min_width


@dataclass
class MetricBlock(Block):
    """Big number with context — temperature, price, battery, etc."""

    block_type: str = field(default="metric", init=False)
    value: str = ""
    label: str | None = None
    icon: str | None = None
    change: str | None = None
    change_color: str = "auto"
    animation: str = "none"

    def __post_init__(self) -> None:
        self.params = {"value": self.value}
        if self.label:
            self.params["label"] = self.label
        if self.icon:
            self.params["icon"] = self.icon
        if self.change:
            self.params["change"] = self.change
        if self.change_color != "auto":
            self.params["change_color"] = self.change_color
        if self.animation != "none":
            self.params["animation"] = self.animation


@dataclass
class BadgeBlock(Block):
    """Small colored label — "Active", "3 new", "Live"."""

    block_type: str = field(default="badge", init=False)
    text: str = ""
    color: str = "accent"

    def __post_init__(self) -> None:
        self.params = {"text": self.text}
        if self.color != "accent":
            self.params["color"] = self.color


@dataclass
class ListBlock(Block):
    """Bulleted/numbered list."""

    block_type: str = field(default="list", init=False)
    items: list[str] | str = field(default_factory=list)  # str = data binding
    style: str = "bullet"  # bullet, number, check, none
    icon: str | None = None
    max_items: int | None = None

    def __post_init__(self) -> None:
        self.params = {"items": self.items}
        if self.style != "bullet":
            self.params["style"] = self.style
        if self.icon:
            self.params["icon"] = self.icon
        if self.max_items is not None:
            self.params["max_items"] = self.max_items


@dataclass
class TableBlock(Block):
    """Data table with automatic column sizing."""

    block_type: str = field(default="table", init=False)
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] | str = field(default_factory=list)
    striped: bool = False
    max_rows: int | None = None

    def __post_init__(self) -> None:
        self.params = {"headers": self.headers, "rows": self.rows}
        if self.striped:
            self.params["striped"] = True
        if self.max_rows is not None:
            self.params["max_rows"] = self.max_rows


@dataclass
class KeyValueBlock(Block):
    """Two-column label/value pairs."""

    block_type: str = field(default="key_value", init=False)
    data: dict[str, str] | str = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.params = {"data": self.data}


@dataclass
class IconBlock(Block):
    """Lucide icon."""

    block_type: str = field(default="icon", init=False)
    name: str = ""
    size: str = "md"  # sm, md, lg, xl
    color: str | None = None

    def __post_init__(self) -> None:
        self.params = {"name": self.name}
        if self.size != "md":
            self.params["size"] = self.size
        if self.color:
            self.params["color"] = self.color


@dataclass
class EmojiBlock(Block):
    """Twemoji rendered as image for visual consistency."""

    block_type: str = field(default="emoji", init=False)
    name: str = ""
    size: str = "md"  # md, lg, xl

    def __post_init__(self) -> None:
        self.params = {"name": self.name}
        if self.size != "md":
            self.params["size"] = self.size


@dataclass
class ImageBlock(Block):
    """Image display: photo:id, url:https://..., or asset:filename."""

    block_type: str = field(default="image", init=False)
    source: str = ""
    fit: str = "cover"  # cover, contain, fill
    radius: int | None = None

    def __post_init__(self) -> None:
        self.params = {"source": self.source}
        if self.fit != "cover":
            self.params["fit"] = self.fit
        if self.radius is not None:
            self.params["radius"] = self.radius


@dataclass
class ChartBlock(Block):
    """Simple chart: line, bar, or area. Rendered directly without a library."""

    block_type: str = field(default="chart", init=False)
    data: str | list[float] | None = None
    series: list[dict[str, Any]] | None = None
    chart_type: str = "line"
    color: str = "accent"
    height: int = 200
    x_labels: str | list[str] | None = None
    show_grid: bool = True
    show_legend: bool = False
    fill_opacity: float = 0.15
    show_dots: bool = False

    def __post_init__(self) -> None:
        self.params = {"type": self.chart_type, "height": self.height}
        if self.data is not None:
            self.params["data"] = self.data
        if self.series is not None:
            self.params["series"] = self.series
        if self.color != "accent":
            self.params["color"] = self.color
        if self.x_labels is not None:
            self.params["x_labels"] = self.x_labels
        if not self.show_grid:
            self.params["show_grid"] = False
        if self.show_legend:
            self.params["show_legend"] = True
        if self.fill_opacity != 0.15:
            self.params["fill_opacity"] = self.fill_opacity
        if self.show_dots:
            self.params["show_dots"] = True


@dataclass
class ProgressBlock(Block):
    """Progress/capacity bar. Value is 0.0 to 1.0 or a data binding."""

    block_type: str = field(default="progress", init=False)
    value: float | str = 0.0
    label: str | None = None
    color: str = "auto"

    def __post_init__(self) -> None:
        self.params = {"value": self.value}
        if self.label:
            self.params["label"] = self.label
        if self.color != "auto":
            self.params["color"] = self.color


@dataclass
class ClockBlock(Block):
    """Self-updating clock. Live block — renderer ticks directly."""

    block_type: str = field(default="clock", init=False)
    format: str = "12h"  # 12h, 24h
    show_date: bool = True
    show_seconds: bool = False
    size: str = "lg"  # md, lg, xl

    def __post_init__(self) -> None:
        self.params = {}
        if self.format != "12h":
            self.params["format"] = self.format
        if not self.show_date:
            self.params["show_date"] = False
        if self.show_seconds:
            self.params["show_seconds"] = True
        if self.size != "lg":
            self.params["size"] = self.size


@dataclass
class CountdownBlock(Block):
    """Live countdown to a target datetime."""

    block_type: str = field(default="countdown", init=False)
    target: str = ""
    label: str | None = None

    def __post_init__(self) -> None:
        self.params = {"target": self.target}
        if self.label:
            self.params["label"] = self.label


# ---------------------------------------------------------------------------
# Composite Widgets (2)
# ---------------------------------------------------------------------------


@dataclass
class WeatherWidget(Block):
    """Pre-built weather card: icon, temp, condition, forecast row."""

    block_type: str = field(default="weather_widget", init=False)
    data_source: str = "weather"

    def __post_init__(self) -> None:
        self.params = {"data_source": self.data_source}


@dataclass
class CalendarWidget(Block):
    """Pre-built calendar widget: today's agenda with time slots."""

    block_type: str = field(default="calendar_widget", init=False)
    data_source: str = "calendar"

    def __post_init__(self) -> None:
        self.params = {"data_source": self.data_source}


# ---------------------------------------------------------------------------
# Meta Blocks (2)
# ---------------------------------------------------------------------------


@dataclass
class RotateBlock(Block):
    """Cycles through items in a data array on a timer.

    Makes {current} available as a binding prefix for the active item.
    """

    block_type: str = field(default="rotate", init=False)
    source: str = ""
    key: str = ""
    interval: int = 20

    def __post_init__(self) -> None:
        self.params = {
            "source": self.source,
            "key": self.key,
            "interval": self.interval,
        }


@dataclass
class PageDotsBlock(Block):
    """Visual indicator for the active item in a rotate sequence."""

    block_type: str = field(default="page_dots", init=False)
    color: str = "accent"

    def __post_init__(self) -> None:
        self.params = {}
        if self.color != "accent":
            self.params["color"] = self.color


# ---------------------------------------------------------------------------
# Block registry and parser
# ---------------------------------------------------------------------------

# Maps block type strings to their dataclass types
BLOCK_REGISTRY: dict[str, type[Block]] = {
    "row": RowBlock,
    "column": ColumnBlock,
    "stack": ColumnBlock,  # alias
    "columns": ColumnsBlock,
    "card": CardBlock,
    "spacer": SpacerBlock,
    "divider": DividerBlock,
    "repeat": RepeatBlock,
    "text": TextBlock,
    "metric": MetricBlock,
    "badge": BadgeBlock,
    "list": ListBlock,
    "table": TableBlock,
    "key_value": KeyValueBlock,
    "icon": IconBlock,
    "emoji": EmojiBlock,
    "image": ImageBlock,
    "chart": ChartBlock,
    "progress": ProgressBlock,
    "clock": ClockBlock,
    "countdown": CountdownBlock,
    "weather_widget": WeatherWidget,
    "calendar_widget": CalendarWidget,
    "rotate": RotateBlock,
    "page_dots": PageDotsBlock,
}


def parse_block(data: dict[str, Any]) -> Block:
    """Parse a JSON block dict into a typed Block instance.

    Args:
        data: Dict with at least a 'type' key, plus block-specific params.

    Returns:
        The appropriate Block subclass instance.

    Raises:
        ValueError: If the block type is unknown.
    """
    block_type = data.get("type", "")
    cls = BLOCK_REGISTRY.get(block_type)
    if cls is None:
        raise ValueError(
            f"Unknown block type '{block_type}'. "
            f"Available: {', '.join(sorted(BLOCK_REGISTRY.keys()))}"
        )

    # Extract children before passing remaining params
    children_data = data.get("children", [])
    children = [parse_block(c) for c in children_data]

    # Build params dict excluding 'type' and 'children'
    params = {k: v for k, v in data.items() if k not in ("type", "children")}

    # Map constructor args from the params
    block = _construct_block(cls, params)
    block.children = children
    return block


def _construct_block(cls: type[Block], params: dict[str, Any]) -> Block:
    """Construct a Block subclass from a params dict.

    Maps JSON param names to dataclass field names, handling the
    chart type→chart_type rename and other mismatches.
    """
    import dataclasses

    # Get the field names for this block class
    field_names = {f.name for f in dataclasses.fields(cls)}

    # Special mappings: JSON key → Python field name
    key_map = {
        "type": "chart_type",  # chart block uses 'type' in JSON for chart type
    }

    kwargs: dict[str, Any] = {}
    for k, v in params.items():
        mapped = key_map.get(k, k)
        if mapped in field_names and mapped not in ("block_type", "children", "params"):
            kwargs[mapped] = v

    return cls(**kwargs)
