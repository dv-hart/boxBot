"""Tests for the display system — themes, blocks, spec, data binding, renderer."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from boxbot.displays.blocks import (
    BLOCK_REGISTRY,
    BadgeBlock,
    Block,
    CardBlock,
    ChartBlock,
    ClockBlock,
    ColumnBlock,
    ColumnsBlock,
    CountdownBlock,
    DividerBlock,
    EmojiBlock,
    IconBlock,
    ImageBlock,
    KeyValueBlock,
    ListBlock,
    MetricBlock,
    PageDotsBlock,
    ProgressBlock,
    RepeatBlock,
    RotateBlock,
    RowBlock,
    SpacerBlock,
    TableBlock,
    TextBlock,
    WeatherWidget,
    parse_block,
)
from boxbot.displays.spec import (
    DataSourceSpec,
    DisplaySpec,
    parse_spec,
    resolve_bindings,
    validate_spec,
)
from boxbot.displays.themes import (
    get_theme,
    hex_to_rgb,
    hex_to_rgba,
    list_themes,
)


# ---------------------------------------------------------------------------
# Theme tests
# ---------------------------------------------------------------------------


class TestThemes:
    """Test the theme system — built-in themes, color conversion."""

    def test_four_builtin_themes(self):
        themes = list_themes()
        names = set(themes)
        assert names == {"boxbot", "midnight", "daylight", "classic"}

    def test_get_theme_by_name(self):
        theme = get_theme("boxbot")
        assert theme is not None
        assert theme.name == "boxbot"

    def test_get_unknown_theme_raises_key_error(self):
        with pytest.raises(KeyError, match="nonexistent"):
            get_theme("nonexistent")

    def test_theme_has_colors(self):
        theme = get_theme("boxbot")
        assert theme.colors is not None
        assert theme.colors.background is not None
        assert theme.colors.text is not None

    def test_theme_has_fonts(self):
        theme = get_theme("boxbot")
        assert theme.fonts is not None

    def test_theme_has_spacing(self):
        theme = get_theme("boxbot")
        assert theme.spacing is not None

    def test_hex_to_rgb_valid(self):
        r, g, b = hex_to_rgb("#FF0000")
        assert (r, g, b) == (255, 0, 0)

    def test_hex_to_rgb_lowercase(self):
        r, g, b = hex_to_rgb("#00ff00")
        assert (r, g, b) == (0, 255, 0)

    def test_hex_to_rgba_full_opaque(self):
        r, g, b, a = hex_to_rgba("#0000FFFF")
        assert (r, g, b) == (0, 0, 255)
        assert a == 255

    def test_hex_to_rgba_default_opaque(self):
        r, g, b, a = hex_to_rgba("#0000FF")
        assert (r, g, b) == (0, 0, 255)
        assert a == 255  # 6-digit hex defaults to fully opaque

    def test_hex_to_rgba_with_alpha_in_hex(self):
        _, _, _, a = hex_to_rgba("#00000080")
        assert a == 128  # 0x80 = 128


# ---------------------------------------------------------------------------
# Block tests
# ---------------------------------------------------------------------------


class TestBlocks:
    """Test block dataclass construction and serialization."""

    def test_text_block_to_dict(self):
        block = TextBlock(content="Hello", size="title")
        d = block.to_dict()
        assert d["type"] == "text"
        assert d["content"] == "Hello"
        assert d.get("size") == "title"

    def test_row_block_with_children(self):
        row = RowBlock(gap=16)
        row.children = [TextBlock(content="A"), TextBlock(content="B")]
        d = row.to_dict()
        assert d["type"] == "row"
        assert len(d["children"]) == 2

    def test_column_block_defaults(self):
        col = ColumnBlock()
        assert col.block_type == "column"
        assert col.gap == 0

    def test_columns_block_ratios(self):
        cols = ColumnsBlock(ratios=[2, 1])
        d = cols.to_dict()
        assert d["ratios"] == [2, 1]

    def test_card_block_params(self):
        card = CardBlock(color="surface", radius=12, padding=20)
        d = card.to_dict()
        assert d["type"] == "card"
        assert d["color"] == "surface"
        assert d["radius"] == 12

    def test_metric_block_fields(self):
        metric = MetricBlock(value="72", label="Temperature", icon="thermometer")
        d = metric.to_dict()
        assert d["value"] == "72"
        assert d["label"] == "Temperature"

    def test_chart_block_type_mapping(self):
        chart = ChartBlock(data=[1.0, 2.0, 3.0], chart_type="bar")
        d = chart.to_dict()
        assert d["type"] == "bar"
        assert d["data"] == [1.0, 2.0, 3.0]

    def test_clock_block_defaults(self):
        clock = ClockBlock()
        assert clock.format == "12h"
        assert clock.show_date is True

    def test_spacer_block_optional_size(self):
        spacer = SpacerBlock(size=24)
        d = spacer.to_dict()
        assert d["size"] == 24

        flexible = SpacerBlock()
        assert "size" not in flexible.to_dict() or flexible.to_dict().get("size") is None

    def test_divider_block(self):
        div = DividerBlock(thickness=2, color="red")
        d = div.to_dict()
        assert d["type"] == "divider"
        assert d["thickness"] == 2

    def test_repeat_block_source(self):
        repeat = RepeatBlock(source="{weather.forecast}")
        d = repeat.to_dict()
        assert d["source"] == "{weather.forecast}"

    def test_badge_block(self):
        badge = BadgeBlock(text="Active", color="success")
        d = badge.to_dict()
        assert d["text"] == "Active"
        assert d["color"] == "success"

    def test_progress_block(self):
        prog = ProgressBlock(value=0.75, label="CPU")
        d = prog.to_dict()
        assert d["value"] == 0.75

    def test_image_block(self):
        img = ImageBlock(source="photo:abc123", fit="contain")
        d = img.to_dict()
        assert d["source"] == "photo:abc123"
        assert d["fit"] == "contain"

    def test_weather_widget(self):
        w = WeatherWidget(data_source="weather")
        d = w.to_dict()
        assert d["type"] == "weather_widget"


class TestBlockRegistry:
    """Test the block registry and parse_block function."""

    def test_registry_has_all_block_types(self):
        expected_types = {
            "row", "column", "stack", "columns", "card", "spacer",
            "divider", "repeat", "text", "metric", "badge", "list",
            "table", "key_value", "icon", "emoji", "image", "chart",
            "progress", "clock", "countdown", "weather_widget",
            "calendar_widget", "rotate", "page_dots",
        }
        assert expected_types.issubset(set(BLOCK_REGISTRY.keys()))

    def test_parse_block_text(self):
        block = parse_block({"type": "text", "content": "Hello"})
        assert isinstance(block, TextBlock)
        assert block.content == "Hello"

    def test_parse_block_with_children(self):
        data = {
            "type": "row",
            "gap": 8,
            "children": [
                {"type": "text", "content": "A"},
                {"type": "text", "content": "B"},
            ],
        }
        block = parse_block(data)
        assert isinstance(block, RowBlock)
        assert len(block.children) == 2

    def test_parse_block_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown block type"):
            parse_block({"type": "nonexistent_block"})

    def test_stack_is_alias_for_column(self):
        assert BLOCK_REGISTRY["stack"] is ColumnBlock


# ---------------------------------------------------------------------------
# Spec parsing and validation
# ---------------------------------------------------------------------------


class TestSpecParsing:
    """Test display spec parsing from JSON dicts."""

    def test_parse_minimal_spec(self):
        data = {"name": "test"}
        spec = parse_spec(data)
        assert spec.name == "test"
        assert spec.theme == "boxbot"
        assert spec.transition == "crossfade"

    def test_parse_spec_with_data_sources(self):
        data = {
            "name": "weather_dash",
            "data_sources": [
                {"name": "weather", "type": "builtin"},
                {"name": "stock", "type": "http_json", "url": "https://api.stock.com"},
            ],
        }
        spec = parse_spec(data)
        assert len(spec.data_sources) == 2
        assert spec.data_sources[0].name == "weather"
        assert spec.data_sources[1].url == "https://api.stock.com"

    def test_parse_spec_with_layout(self):
        data = {
            "name": "layout_test",
            "layout": {
                "type": "column",
                "children": [{"type": "text", "content": "Top"}],
            },
        }
        spec = parse_spec(data)
        assert spec.root_block is not None
        assert spec.root_block.block_type == "column"


class TestSpecValidation:
    """Test display spec validation."""

    def test_valid_spec_returns_no_errors(self):
        spec = DisplaySpec(
            name="valid",
            theme="boxbot",
            transition="crossfade",
            root_block=TextBlock(content="Hello"),
        )
        errors = validate_spec(spec)
        assert errors == []

    def test_missing_name_produces_error(self):
        spec = DisplaySpec(name="")
        errors = validate_spec(spec)
        assert any("name" in e.lower() for e in errors)

    def test_invalid_transition_produces_error(self):
        spec = DisplaySpec(name="test", transition="bounce")
        errors = validate_spec(spec)
        assert any("transition" in e.lower() for e in errors)

    def test_duplicate_data_source_names(self):
        spec = DisplaySpec(
            name="dup_test",
            data_sources=[
                DataSourceSpec(name="weather"),
                DataSourceSpec(name="weather"),
            ],
        )
        errors = validate_spec(spec)
        assert any("duplicate" in e.lower() for e in errors)

    def test_http_json_source_without_url(self):
        spec = DisplaySpec(
            name="bad_source",
            data_sources=[
                DataSourceSpec(name="stock", source_type="http_json"),
            ],
        )
        errors = validate_spec(spec)
        assert any("url" in e.lower() for e in errors)

    def test_text_block_without_content_produces_error(self):
        spec = DisplaySpec(
            name="empty_text",
            root_block=TextBlock(content=""),
        )
        errors = validate_spec(spec)
        assert any("content" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# Data binding resolution
# ---------------------------------------------------------------------------


class TestDataBindingResolution:
    """Test {source.field} data binding resolution."""

    def test_resolve_simple_binding(self):
        block = TextBlock(content="{weather.temp}")
        data = {"weather": {"temp": "72"}}
        resolved = resolve_bindings(block, data)
        assert resolved.params["content"] == "72"

    def test_resolve_mixed_text_and_binding(self):
        block = TextBlock(content="{weather.temp} degrees")
        data = {"weather": {"temp": "72"}}
        resolved = resolve_bindings(block, data)
        assert resolved.params["content"] == "72 degrees"

    def test_resolve_nested_binding(self):
        block = TextBlock(content="{weather.forecast[0].high}")
        data = {"weather": {"forecast": [{"high": "85"}]}}
        resolved = resolve_bindings(block, data)
        assert resolved.params["content"] == "85"

    def test_resolve_repeat_item_binding(self):
        block = TextBlock(content="{.name}")
        repeat_item = {"name": "Monday"}
        resolved = resolve_bindings(block, {}, repeat_item=repeat_item)
        assert resolved.params["content"] == "Monday"

    def test_resolve_current_item_binding(self):
        block = TextBlock(content="{current.title}")
        current_item = {"title": "Photo 1"}
        resolved = resolve_bindings(block, {}, current_item=current_item)
        assert resolved.params["content"] == "Photo 1"

    def test_resolve_missing_source_returns_none_as_empty_string(self):
        block = TextBlock(content="{missing.field} text")
        resolved = resolve_bindings(block, {})
        assert "text" in resolved.params["content"]

    def test_resolve_full_binding_returns_raw_value(self):
        """A binding that is the entire string returns the raw value type."""
        block = TextBlock(content="{data.items}")
        data = {"data": {"items": [1, 2, 3]}}
        resolved = resolve_bindings(block, data)
        assert resolved.params["content"] == [1, 2, 3]

    def test_resolve_does_not_mutate_original(self):
        block = TextBlock(content="{weather.temp}")
        original_content = block.params["content"]
        data = {"weather": {"temp": "72"}}
        resolve_bindings(block, data)
        assert block.params["content"] == original_content

    def test_resolve_children_recursively(self):
        row = RowBlock()
        row.children = [TextBlock(content="{data.label}")]
        data = {"data": {"label": "Resolved"}}
        resolved = resolve_bindings(row, data)
        assert resolved.children[0].params["content"] == "Resolved"
