"""Display builder — declarative block-based display construction.

The agent composes layout containers and content blocks into a tree.
The rendering engine handles positioning, text wrapping, overflow, and
theming. The agent describes **what** to show, not **how** to render it.

Usage:
    from boxbot_sdk import display

    d = display.create("weather_board")
    d.set_theme("boxbot")
    d.data("weather")
    header = d.row(padding=24, align="center")
    header.icon("{weather.icon}", size="xl")
    header.text("{weather.temp}°F", size="title")
    d.preview()
    d.save()
"""

from __future__ import annotations

from typing import Any

from . import _transport, _validators as v


def create(name: str) -> DisplayBuilder:
    """Create a new display builder.

    Args:
        name: Unique display name (alphanumeric, underscores, hyphens).

    Returns:
        A new DisplayBuilder instance.
    """
    v.validate_name(name, "display name")
    return DisplayBuilder(name)


def update_data(display_name: str, source_name: str, *,
                value: Any) -> None:
    """Update a static data source on an existing display.

    Args:
        display_name: Name of the display to update.
        source_name: Name of the data source to update.
        value: New value for the static data source.
    """
    v.require_str(display_name, "display_name")
    v.require_str(source_name, "source_name")
    _transport.emit_action("display.update_data", {
        "display": display_name,
        "source": source_name,
        "value": value,
    })


class _BlockNode:
    """Base class for all block nodes in the display tree."""

    def __init__(self, block_type: str, **props: Any) -> None:
        self.block_type = block_type
        self.props: dict[str, Any] = {
            k: val for k, val in props.items() if val is not None
        }
        self.children: list[_BlockNode] = []

    def to_dict(self) -> dict[str, Any]:
        """Serialize this node and its children to a dict."""
        result: dict[str, Any] = {"type": self.block_type}
        if self.props:
            result.update(self.props)
        if self.children:
            result["children"] = [child.to_dict() for child in self.children]
        return result


class ContainerBuilder:
    """Builder for layout containers that can hold child blocks.

    All layout methods return either a new ContainerBuilder (for nested
    containers) or self (for content blocks), enabling chained construction.
    """

    def __init__(self, node: _BlockNode, root: DisplayBuilder) -> None:
        self._node = node
        self._root = root

    # --- Layout containers ---

    def row(self, *, gap: int | None = None, align: str | None = None,
            padding: int | list[int] | None = None) -> ContainerBuilder:
        """Add a horizontal row container.

        Args:
            gap: Space between children in pixels.
            align: Child alignment — start, center, end, spread.
            padding: Padding around content.

        Returns:
            ContainerBuilder for the new row.
        """
        props: dict[str, Any] = {}
        if gap is not None:
            v.require_int(gap, "gap", min_val=0)
            props["gap"] = gap
        if align is not None:
            v.validate_one_of(align, "align", v.VALID_CONTAINER_ALIGNS)
            props["align"] = align
        if padding is not None:
            props["padding"] = v.validate_padding(padding)
        node = _BlockNode("row", **props)
        self._node.children.append(node)
        return ContainerBuilder(node, self._root)

    def column(self, *, gap: int | None = None, align: str | None = None,
               padding: int | list[int] | None = None) -> ContainerBuilder:
        """Add a vertical column container.

        Args:
            gap: Space between children in pixels.
            align: Child alignment — start, center, end, spread.
            padding: Padding around content.

        Returns:
            ContainerBuilder for the new column.
        """
        props: dict[str, Any] = {}
        if gap is not None:
            v.require_int(gap, "gap", min_val=0)
            props["gap"] = gap
        if align is not None:
            v.validate_one_of(align, "align", v.VALID_CONTAINER_ALIGNS)
            props["align"] = align
        if padding is not None:
            props["padding"] = v.validate_padding(padding)
        node = _BlockNode("column", **props)
        self._node.children.append(node)
        return ContainerBuilder(node, self._root)

    def stack(self, *, gap: int | None = None, align: str | None = None,
              padding: int | list[int] | None = None) -> ContainerBuilder:
        """Add a vertical stack container (alias for column).

        Args:
            gap: Space between children in pixels.
            align: Child alignment — start, center, end, spread.
            padding: Padding around content.

        Returns:
            ContainerBuilder for the new stack.
        """
        return self.column(gap=gap, align=align, padding=padding)

    def columns(self, ratios: list[int], *, gap: int | None = None,
                padding: int | list[int] | None = None) -> _ColumnsBuilder:
        """Add a multi-column layout with ratio-based widths.

        Args:
            ratios: List of integer width ratios, e.g. [2, 1] for 2/3 + 1/3.
            gap: Space between columns in pixels.
            padding: Padding around content.

        Returns:
            _ColumnsBuilder with one ContainerBuilder per column.
        """
        validated_ratios = v.validate_ratios(ratios)
        props: dict[str, Any] = {"ratios": validated_ratios}
        if gap is not None:
            v.require_int(gap, "gap", min_val=0)
            props["gap"] = gap
        if padding is not None:
            props["padding"] = v.validate_padding(padding)
        node = _BlockNode("columns", **props)
        self._node.children.append(node)
        # Create a child column node for each ratio
        builders = []
        for _ in validated_ratios:
            col_node = _BlockNode("column")
            node.children.append(col_node)
            builders.append(ContainerBuilder(col_node, self._root))
        return _ColumnsBuilder(builders)

    def card(self, *, color: str | None = None, radius: int | None = None,
             padding: int | list[int] | None = None) -> ContainerBuilder:
        """Add a card container with background and rounded corners.

        Args:
            color: Background color token.
            radius: Corner radius in pixels.
            padding: Padding around content.

        Returns:
            ContainerBuilder for the new card.
        """
        props: dict[str, Any] = {}
        if color is not None:
            props["color"] = v.require_str(color, "color")
        if radius is not None:
            v.require_int(radius, "radius", min_val=0)
            props["radius"] = radius
        if padding is not None:
            props["padding"] = v.validate_padding(padding)
        node = _BlockNode("card", **props)
        self._node.children.append(node)
        return ContainerBuilder(node, self._root)

    def spacer(self, size: int | None = None) -> None:
        """Add a spacer between elements.

        Args:
            size: Fixed size in pixels. None for flexible space.
        """
        props: dict[str, Any] = {}
        if size is not None:
            v.require_int(size, "size", min_val=0)
            props["size"] = size
        node = _BlockNode("spacer", **props)
        self._node.children.append(node)

    def divider(self, *, color: str | None = None, thickness: int | None = None,
                orientation: str | None = None) -> None:
        """Add a separator line.

        Args:
            color: Line color token.
            thickness: Line thickness in pixels.
            orientation: "horizontal" or "vertical".
        """
        props: dict[str, Any] = {}
        if color is not None:
            props["color"] = v.require_str(color, "color")
        if thickness is not None:
            v.require_int(thickness, "thickness", min_val=1)
            props["thickness"] = thickness
        if orientation is not None:
            v.validate_one_of(orientation, "orientation",
                              v.VALID_DIVIDER_ORIENTATIONS)
            props["orientation"] = orientation
        node = _BlockNode("divider", **props)
        self._node.children.append(node)

    def repeat(self, source: str, *, max: int | None = None,
               highlight_active: bool | None = None) -> ContainerBuilder:
        """Add a repeat block that iterates over a data array.

        Use {.field} syntax inside the repeat for the current item.

        Args:
            source: Data binding for the array, e.g. "{weather.forecast}".
            max: Maximum number of items to render.
            highlight_active: Highlight the active item (for use with rotate).

        Returns:
            ContainerBuilder for the repeat template.
        """
        v.require_str(source, "source")
        props: dict[str, Any] = {"source": source}
        if max is not None:
            v.require_int(max, "max", min_val=1)
            props["max"] = max
        if highlight_active is not None:
            v.require_bool(highlight_active, "highlight_active")
            props["highlight_active"] = highlight_active
        node = _BlockNode("repeat", **props)
        self._node.children.append(node)
        return ContainerBuilder(node, self._root)

    # --- Content blocks ---

    def text(self, content: str, *, size: str | None = None,
             color: str | None = None, weight: str | None = None,
             align: str | None = None, max_lines: int | None = None,
             animation: str | None = None,
             min_width: int | None = None) -> None:
        """Add a text block.

        Args:
            content: Text content (can include data bindings).
            size: title, heading, subtitle, body, caption, small.
            color: default, muted, dim, accent, success, warning, error.
            weight: normal, medium, semibold, bold.
            align: left, center, right.
            max_lines: Maximum lines before truncation with ellipsis.
            animation: none, fade, typewriter.
            min_width: Minimum width in pixels.
        """
        v.require_str(content, "content", allow_empty=True)
        props: dict[str, Any] = {"content": content}
        if size is not None:
            v.validate_one_of(size, "size", v.VALID_TEXT_SIZES)
            props["size"] = size
        if color is not None:
            v.validate_one_of(color, "color", v.VALID_TEXT_COLORS)
            props["color"] = color
        if weight is not None:
            v.validate_one_of(weight, "weight", v.VALID_TEXT_WEIGHTS)
            props["weight"] = weight
        if align is not None:
            v.validate_one_of(align, "align", v.VALID_TEXT_ALIGNS)
            props["align"] = align
        if max_lines is not None:
            v.require_int(max_lines, "max_lines", min_val=1)
            props["max_lines"] = max_lines
        if animation is not None:
            v.validate_one_of(animation, "animation", v.VALID_TEXT_ANIMATIONS)
            props["animation"] = animation
        if min_width is not None:
            v.require_int(min_width, "min_width", min_val=0)
            props["min_width"] = min_width
        node = _BlockNode("text", **props)
        self._node.children.append(node)

    def metric(self, value: str, *, label: str | None = None,
               icon: str | None = None, change: str | None = None,
               change_color: str | None = None,
               animation: str | None = None,
               color: str | None = None) -> None:
        """Add a metric block (big number with context).

        Args:
            value: The primary value (can be data-bound).
            label: Label text below/beside the value.
            icon: Lucide icon name.
            change: Change indicator, e.g. "+2.3%".
            change_color: "auto" or a color token.
            animation: none, fade, count_up.
            color: Color token for the metric value.
        """
        v.require_str(value, "value", allow_empty=True)
        props: dict[str, Any] = {"value": value}
        if label is not None:
            props["label"] = v.require_str(label, "label", allow_empty=True)
        if icon is not None:
            props["icon"] = v.require_str(icon, "icon")
        if change is not None:
            props["change"] = v.require_str(change, "change", allow_empty=True)
        if change_color is not None:
            props["change_color"] = v.require_str(change_color, "change_color")
        if animation is not None:
            v.validate_one_of(animation, "animation", v.VALID_METRIC_ANIMATIONS)
            props["animation"] = animation
        if color is not None:
            props["color"] = v.require_str(color, "color")
        node = _BlockNode("metric", **props)
        self._node.children.append(node)

    def badge(self, text: str, *, color: str | None = None) -> None:
        """Add a badge (small colored label).

        Args:
            text: Badge text.
            color: Color token.
        """
        v.require_str(text, "text", allow_empty=True)
        props: dict[str, Any] = {"text": text}
        if color is not None:
            props["color"] = v.require_str(color, "color")
        node = _BlockNode("badge", **props)
        self._node.children.append(node)

    def list(self, items: str | list[str], *, style: str | None = None,
             icon: str | None = None, max_items: int | None = None) -> None:
        """Add a list block.

        Args:
            items: List of strings or a data binding string.
            style: bullet, number, check, none.
            icon: Lucide icon name (replaces bullet).
            max_items: Maximum items to show.
        """
        props: dict[str, Any] = {}
        if isinstance(items, str):
            props["items"] = items
        else:
            v.require_list(items, "items")
            props["items"] = items
        if style is not None:
            v.validate_one_of(style, "style", v.VALID_LIST_STYLES)
            props["style"] = style
        if icon is not None:
            props["icon"] = v.require_str(icon, "icon")
        if max_items is not None:
            v.require_int(max_items, "max_items", min_val=1)
            props["max_items"] = max_items
        node = _BlockNode("list", **props)
        self._node.children.append(node)

    def table(self, headers: list[str] | str, rows: list[list[str]] | str, *,
              striped: bool | None = None,
              max_rows: int | None = None) -> None:
        """Add a data table.

        Args:
            headers: Column headers (list or data binding).
            rows: Table rows (list of lists or data binding).
            striped: Alternating row backgrounds.
            max_rows: Maximum rows to display.
        """
        props: dict[str, Any] = {}
        if isinstance(headers, str):
            props["headers"] = headers
        else:
            v.require_list(headers, "headers")
            props["headers"] = headers
        if isinstance(rows, str):
            props["rows"] = rows
        else:
            v.require_list(rows, "rows")
            props["rows"] = rows
        if striped is not None:
            v.require_bool(striped, "striped")
            props["striped"] = striped
        if max_rows is not None:
            v.require_int(max_rows, "max_rows", min_val=1)
            props["max_rows"] = max_rows
        node = _BlockNode("table", **props)
        self._node.children.append(node)

    def key_value(self, data: dict[str, str] | str) -> None:
        """Add a key-value pair display.

        Args:
            data: Dict of label→value pairs or a data binding string.
        """
        props: dict[str, Any] = {}
        if isinstance(data, str):
            props["data"] = data
        else:
            v.require_dict(data, "data")
            props["data"] = data
        node = _BlockNode("key_value", **props)
        self._node.children.append(node)

    def icon(self, name: str, *, size: str | None = None,
             color: str | None = None) -> None:
        """Add an icon from the Lucide library.

        Args:
            name: Lucide icon name (can be data-bound).
            size: sm, md, lg, xl.
            color: Color token.
        """
        v.require_str(name, "name", allow_empty=True)
        props: dict[str, Any] = {"name": name}
        if size is not None:
            v.validate_one_of(size, "size", v.VALID_ICON_SIZES)
            props["size"] = size
        if color is not None:
            props["color"] = v.require_str(color, "color")
        node = _BlockNode("icon", **props)
        self._node.children.append(node)

    def emoji(self, name: str, *, size: str | None = None) -> None:
        """Add an emoji from the Twemoji set.

        Args:
            name: Emoji name.
            size: md, lg, xl.
        """
        v.require_str(name, "name")
        props: dict[str, Any] = {"name": name}
        if size is not None:
            v.validate_one_of(size, "size", v.VALID_EMOJI_SIZES)
            props["size"] = size
        node = _BlockNode("emoji", **props)
        self._node.children.append(node)

    def image(self, source: str, *, fit: str | None = None,
              radius: int | None = None) -> None:
        """Add an image block.

        Args:
            source: Image source — "photo:id", "url:https://...", "asset:filename".
            fit: cover, contain, fill.
            radius: Corner radius in pixels.
        """
        v.require_str(source, "source")
        props: dict[str, Any] = {"source": source}
        if fit is not None:
            v.validate_one_of(fit, "fit", v.VALID_IMAGE_FITS)
            props["fit"] = fit
        if radius is not None:
            v.require_int(radius, "radius", min_val=0)
            props["radius"] = radius
        node = _BlockNode("image", **props)
        self._node.children.append(node)

    def chart(self, *, data: str | None = None,
              series: list[dict[str, Any]] | None = None,
              type: str | None = None, color: str | None = None,
              height: int | None = None, x_labels: str | list | None = None,
              show_grid: bool | None = None, show_legend: bool | None = None,
              fill_opacity: float | None = None,
              show_dots: bool | None = None,
              padding: int | list[int] | None = None) -> None:
        """Add a chart block (line, bar, or area).

        Args:
            data: Data binding for single series.
            series: List of series dicts for multi-series.
            type: line, bar, area.
            color: Color token (single series).
            height: Chart height in pixels.
            x_labels: X-axis labels (data binding or list).
            show_grid: Show grid lines.
            show_legend: Show series legend.
            fill_opacity: Fill opacity for area charts (0.0-1.0).
            show_dots: Show data points on line charts.
            padding: Padding around the chart.
        """
        if data is None and series is None:
            raise ValueError("chart requires either 'data' or 'series'")
        if data is not None and series is not None:
            raise ValueError("chart cannot have both 'data' and 'series'")

        props: dict[str, Any] = {}
        if data is not None:
            props["data"] = v.require_str(data, "data", allow_empty=True)
        if series is not None:
            v.require_list(series, "series")
            props["series"] = series
        if type is not None:
            v.validate_one_of(type, "type", v.VALID_CHART_TYPES)
            props["type"] = type
        if color is not None:
            props["color"] = v.require_str(color, "color")
        if height is not None:
            v.require_int(height, "height", min_val=50)
            props["height"] = height
        if x_labels is not None:
            props["x_labels"] = x_labels
        if show_grid is not None:
            v.require_bool(show_grid, "show_grid")
            props["show_grid"] = show_grid
        if show_legend is not None:
            v.require_bool(show_legend, "show_legend")
            props["show_legend"] = show_legend
        if fill_opacity is not None:
            v.require_float(fill_opacity, "fill_opacity", min_val=0.0, max_val=1.0)
            props["fill_opacity"] = fill_opacity
        if show_dots is not None:
            v.require_bool(show_dots, "show_dots")
            props["show_dots"] = show_dots
        if padding is not None:
            props["padding"] = v.validate_padding(padding)
        node = _BlockNode("chart", **props)
        self._node.children.append(node)

    def progress(self, value: str | float, *, label: str | None = None,
                 color: str | None = None) -> None:
        """Add a progress/capacity bar.

        Args:
            value: 0.0 to 1.0 or a data binding string.
            label: Label text.
            color: Color token or "auto" for level-based coloring.
        """
        props: dict[str, Any] = {}
        if isinstance(value, str):
            props["value"] = value
        else:
            v.require_float(value, "value", min_val=0.0, max_val=1.0)
            props["value"] = value
        if label is not None:
            props["label"] = v.require_str(label, "label", allow_empty=True)
        if color is not None:
            props["color"] = v.require_str(color, "color")
        node = _BlockNode("progress", **props)
        self._node.children.append(node)

    def clock(self, *, format: str | None = None, show_date: bool | None = None,
              show_seconds: bool | None = None,
              size: str | None = None) -> None:
        """Add a live clock block.

        Args:
            format: 12h or 24h.
            show_date: Show date below time.
            show_seconds: Show seconds.
            size: md, lg, xl.
        """
        props: dict[str, Any] = {}
        if format is not None:
            v.validate_one_of(format, "format", v.VALID_CLOCK_FORMATS)
            props["format"] = format
        if show_date is not None:
            v.require_bool(show_date, "show_date")
            props["show_date"] = show_date
        if show_seconds is not None:
            v.require_bool(show_seconds, "show_seconds")
            props["show_seconds"] = show_seconds
        if size is not None:
            v.validate_one_of(size, "size", v.VALID_CLOCK_SIZES)
            props["size"] = size
        node = _BlockNode("clock", **props)
        self._node.children.append(node)

    def countdown(self, target: str, *, label: str | None = None) -> None:
        """Add a live countdown block.

        Args:
            target: Target datetime string or data binding.
            label: Label text.
        """
        v.require_str(target, "target")
        props: dict[str, Any] = {"target": target}
        if label is not None:
            props["label"] = v.require_str(label, "label", allow_empty=True)
        node = _BlockNode("countdown", **props)
        self._node.children.append(node)

    # --- Composite widgets ---

    def weather_widget(self, data_source: str = "weather") -> None:
        """Add a pre-built weather widget.

        Args:
            data_source: Name of the weather data source.
        """
        v.require_str(data_source, "data_source")
        node = _BlockNode("weather_widget", data_source=data_source)
        self._node.children.append(node)

    def calendar_widget(self, data_source: str = "calendar") -> None:
        """Add a pre-built calendar widget.

        Args:
            data_source: Name of the calendar data source.
        """
        v.require_str(data_source, "data_source")
        node = _BlockNode("calendar_widget", data_source=data_source)
        self._node.children.append(node)


class _ColumnsBuilder:
    """Provides access to individual columns created by columns()."""

    def __init__(self, builders: list[ContainerBuilder]) -> None:
        self._builders = builders

    def __len__(self) -> int:
        return len(self._builders)

    def __getitem__(self, index: int) -> ContainerBuilder:
        return self._builders[index]

    def __iter__(self):
        return iter(self._builders)

    def column(self, index: int = 0) -> ContainerBuilder:
        """Get a specific column by index.

        Args:
            index: Zero-based column index.
        """
        if index < 0 or index >= len(self._builders):
            raise IndexError(
                f"Column index {index} out of range "
                f"(0-{len(self._builders) - 1})"
            )
        return self._builders[index]


class DisplayBuilder(ContainerBuilder):
    """Top-level display builder.

    Extends ContainerBuilder with display-level configuration:
    theme, data sources, transitions, rotate, page_dots, preview, save.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._theme: str | None = None
        self._data_sources: list[dict[str, Any]] = []
        self._transition: str | None = None
        self._rotate_config: dict[str, Any] | None = None
        # Root node is an implicit top-level column
        root_node = _BlockNode("root")
        super().__init__(root_node, self)

    @property
    def name(self) -> str:
        """Display name."""
        return self._name

    def set_theme(self, theme: str) -> None:
        """Set the display theme.

        Args:
            theme: Theme name — boxbot, midnight, daylight, classic,
                   or a community theme name.
        """
        v.require_str(theme, "theme")
        self._theme = theme

    def data(self, name: str, *, type: str | None = None, **kwargs: Any) -> None:
        """Declare a data source for this display.

        Built-in sources (weather, calendar, tasks, people, agent_status,
        clock) need only a name. Custom sources require a type and config.

        Args:
            name: Data source name.
            type: Source type for custom sources (http_json, http_text,
                  static, memory_query).
            **kwargs: Additional config (url, params, secret, refresh,
                      fields, value, query).
        """
        config = v.validate_data_source_config(name, type, **kwargs)
        self._data_sources.append(config)

    def set_transition(self, transition: str) -> None:
        """Set the display switch transition.

        Args:
            transition: crossfade, slide_left, slide_right, none.
        """
        v.validate_one_of(transition, "transition", v.VALID_TRANSITIONS)
        self._transition = transition

    def rotate(self, source: str, key: str, interval: int) -> None:
        """Set up rotation through items in a data array.

        Makes {current} available as a binding prefix for the active item.

        Args:
            source: Data source name.
            key: Key within the source data to iterate over.
            interval: Seconds between rotations.
        """
        v.require_str(source, "source")
        v.require_str(key, "key")
        v.require_int(interval, "interval", min_val=1)
        self._rotate_config = {
            "source": source,
            "key": key,
            "interval": interval,
        }

    def page_dots(self, *, color: str | None = None) -> None:
        """Add page dots indicator for rotate sequences.

        Args:
            color: Dot color token.
        """
        props: dict[str, Any] = {}
        if color is not None:
            props["color"] = v.require_str(color, "color")
        node = _BlockNode("page_dots", **props)
        self._node.children.append(node)

    def _build_spec(self) -> dict[str, Any]:
        """Build the complete display spec as a dict."""
        spec: dict[str, Any] = {"name": self._name}
        if self._theme:
            spec["theme"] = self._theme
        if self._data_sources:
            spec["data_sources"] = self._data_sources
        if self._transition:
            spec["transition"] = self._transition
        if self._rotate_config:
            spec["rotate"] = self._rotate_config
        # The root node's children form the display layout
        spec["layout"] = [child.to_dict() for child in self._node.children]
        return spec

    def preview(self) -> None:
        """Render the display to a 1024x600 PNG for agent review.

        The execute_script tool renders the spec and returns the image
        path in the tool response.
        """
        spec = self._build_spec()
        _transport.emit_action("display.preview", {
            "name": self._name,
            "spec": spec,
        })

    def save(self) -> None:
        """Save the display spec for activation.

        The execute_script tool writes the spec and queues it for
        user approval before activation.
        """
        spec = self._build_spec()
        _transport.emit_action("display.save", {
            "name": self._name,
            "spec": spec,
        })
