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


class TestNumericTextSize:
    """The ramp stops at title/42px, so hero readouts ask for pixels.

    Before this, a numeric size crashed the whole render inside
    ``getattr(self, 96, ...)`` — a hero temperature was impossible.
    """

    def test_ramp_name_still_resolves(self):
        fonts = get_theme("boxbot").fonts
        assert fonts.get_style("title") is fonts.title

    def test_unknown_name_falls_back_to_body(self):
        fonts = get_theme("boxbot").fonts
        assert fonts.get_style("gigantic") is fonts.body

    def test_number_becomes_that_pixel_size(self):
        fonts = get_theme("boxbot").fonts
        assert fonts.get_style(96).size == 96

    def test_number_keeps_body_weight_and_tracking(self):
        from boxbot.displays.themes import FontStyle, ThemeFonts

        base = get_theme("boxbot").fonts
        fonts = ThemeFonts(
            family=base.family,
            title=base.title,
            heading=base.heading,
            subtitle=base.subtitle,
            body=FontStyle(size=18, weight=400, tracking=0.04),
            caption=base.caption,
            small=base.small,
        )
        style = fonts.get_style(96)
        assert style.weight == 400
        assert style.tracking == 0.04

    def test_out_of_range_numbers_clamp(self):
        fonts = get_theme("boxbot").fonts
        assert fonts.get_style(10_000).size == 240
        assert fonts.get_style(-4).size == 8

    def test_spec_accepts_a_pixel_size(self):
        spec = DisplaySpec(
            name="hero",
            theme="boxbot",
            root_block=TextBlock(content="71°", size=140),
        )
        assert validate_spec(spec) == []

    def test_spec_rejects_a_nonpositive_pixel_size(self):
        spec = DisplaySpec(
            name="hero",
            theme="boxbot",
            root_block=TextBlock(content="71°", size=0),
        )
        assert validate_spec(spec) == ["text.size in pixels must be positive"]

    def test_bad_size_degrades_instead_of_crashing_the_render(self):
        from boxbot.displays.renderer import render_to_image

        theme = get_theme("boxbot")
        for size in (140, 10_000, "gigantic", None):
            block = TextBlock(content="71°")
            block.params["size"] = size
            img = render_to_image(block, theme, {}, width=400, height=300)
            assert img.size == (400, 300)


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

    def test_parse_spec_memory_query_source(self):
        data = {
            "name": "reminders_board",
            "data_sources": [
                {"name": "recent", "type": "memory_query",
                 "query": "kitchen renovation", "refresh": 600, "limit": 3},
            ],
        }
        spec = parse_spec(data)
        src = spec.data_sources[0]
        assert src.source_type == "memory_query"
        assert src.query == "kitchen renovation"
        assert src.refresh == 600
        assert src.limit == 3

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

    def test_memory_query_source_without_query(self):
        spec = DisplaySpec(
            name="bad_source",
            data_sources=[
                DataSourceSpec(name="recent", source_type="memory_query"),
            ],
        )
        errors = validate_spec(spec)
        assert any("query" in e.lower() for e in errors)

    def test_memory_query_source_bad_limit(self):
        spec = DisplaySpec(
            name="bad_limit",
            data_sources=[
                DataSourceSpec(name="recent", source_type="memory_query",
                               query="reminders", limit=0),
            ],
        )
        errors = validate_spec(spec)
        assert any("limit" in e.lower() for e in errors)

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


# ---------------------------------------------------------------------------
# Live-display introspection: get_active + screenshot dispatcher actions
# ---------------------------------------------------------------------------


class TestActiveDisplayIntrospection:
    """The agent must be able to ask 'what's on screen right now?' and
    'show me the pixels'. Both go through the display action dispatcher.
    """

    @pytest.mark.asyncio
    async def test_get_active_returns_none_when_idle(self, monkeypatch):
        from boxbot.displays.manager import (
            DisplayManager,
            set_display_manager,
        )
        from boxbot.tools._sandbox_actions import (
            ActionContext,
            _handle_display_action,
        )

        mgr = DisplayManager()
        set_display_manager(mgr)
        try:
            ctx = ActionContext()
            resp = await _handle_display_action(
                "display.get_active", {}, ctx,
            )
            assert resp["status"] == "ok"
            assert resp["name"] is None
            assert resp["args"] == {}
            assert resp["theme"] is None
        finally:
            set_display_manager(None)

    @pytest.mark.asyncio
    async def test_get_active_after_switch(self, tmp_path, monkeypatch):
        from boxbot.displays.manager import (
            DisplayManager,
            set_display_manager,
        )
        from boxbot.displays.spec import DisplaySpec
        from boxbot.displays.blocks import TextBlock
        from boxbot.tools._sandbox_actions import (
            ActionContext,
            _handle_display_action,
        )

        mgr = DisplayManager()
        spec = DisplaySpec(
            name="hello",
            theme="boxbot",
            data_sources=[],
            root_block=TextBlock(content="hi"),
        )
        mgr.register_spec(spec)
        await mgr.switch("hello", args={"who": "jacob"})
        set_display_manager(mgr)
        try:
            ctx = ActionContext()
            resp = await _handle_display_action(
                "display.get_active", {}, ctx,
            )
            assert resp["status"] == "ok"
            assert resp["name"] == "hello"
            assert resp["args"] == {"who": "jacob"}
            assert resp["theme"] == "boxbot"
        finally:
            await mgr._data_manager.stop_all()
            set_display_manager(None)

    @pytest.mark.asyncio
    async def test_screenshot_attaches_live_frame(self, tmp_path, monkeypatch):
        # Anchor PREVIEWS_DIR inside tmp_path so the test doesn't write
        # into the project's data/ tree.
        monkeypatch.setenv("BOXBOT_DATA_DIR", str(tmp_path))
        # Reload paths so PREVIEWS_DIR picks up the env override.
        import importlib
        import boxbot.core.paths as paths
        importlib.reload(paths)
        import boxbot.tools._sandbox_actions as sa
        importlib.reload(sa)

        from boxbot.displays.manager import (
            DisplayManager,
            set_display_manager,
        )
        from boxbot.displays.spec import DisplaySpec
        from boxbot.displays.blocks import TextBlock

        mgr = DisplayManager()
        spec = DisplaySpec(
            name="hello",
            theme="boxbot",
            data_sources=[],
            root_block=TextBlock(content="hi"),
        )
        mgr.register_spec(spec)
        await mgr.switch("hello")
        set_display_manager(mgr)
        try:
            ctx = sa.ActionContext()
            resp = await sa._handle_display_action(
                "display.screenshot", {}, ctx,
            )
            assert resp["status"] == "ok", resp
            assert resp["name"] == "hello"
            assert resp["attached"] is True
            assert len(ctx.image_attachments) == 1
            assert ctx.image_attachments[0].exists()
            # Path lives under the test-scoped previews dir.
            assert paths.PREVIEWS_DIR in ctx.image_attachments[0].parents
        finally:
            await mgr._data_manager.stop_all()
            set_display_manager(None)

    @pytest.mark.asyncio
    async def test_screenshot_errors_when_idle(self):
        from boxbot.displays.manager import (
            DisplayManager,
            set_display_manager,
        )
        from boxbot.tools._sandbox_actions import (
            ActionContext,
            _handle_display_action,
        )

        mgr = DisplayManager()
        set_display_manager(mgr)
        try:
            ctx = ActionContext()
            resp = await _handle_display_action(
                "display.screenshot", {}, ctx,
            )
            assert resp["status"] == "error"
            assert "no display" in resp["error"]
        finally:
            set_display_manager(None)


class TestUpdateData:
    """Refresh triggers pre-stage data for displays that aren't on screen
    yet, so update_data has to work whether the display is active or not.
    Active updates re-render live; inactive updates mutate the spec
    (and persist to disk for agent-saved displays).
    """

    @staticmethod
    def _static_spec(name: str = "weekly_glance") -> DisplaySpec:
        return DisplaySpec(
            name=name,
            theme="boxbot",
            data_sources=[
                DataSourceSpec(
                    name="agenda",
                    source_type="static",
                    value={"entries": []},
                ),
            ],
            root_block=TextBlock(content="{agenda.entries[0]}"),
        )

    @pytest.mark.asyncio
    async def test_update_inactive_mutates_spec(self):
        from boxbot.displays.manager import DisplayManager

        mgr = DisplayManager()
        mgr.register_spec(self._static_spec())

        result = mgr.update_static_data(
            "weekly_glance", "agenda", {"entries": ["Tue: oil change"]},
        )

        assert result == {"ok": True, "live": False}
        spec = mgr.get_spec("weekly_glance")
        assert spec is not None
        assert spec.data_sources[0].value == {"entries": ["Tue: oil change"]}

    @pytest.mark.asyncio
    async def test_update_active_updates_live_source(self):
        from boxbot.displays.manager import DisplayManager
        from boxbot.displays.data_sources import StaticSource

        mgr = DisplayManager()
        mgr.register_spec(self._static_spec())
        await mgr.switch("weekly_glance")
        try:
            result = mgr.update_static_data(
                "weekly_glance", "agenda", {"entries": ["fresh"]},
            )
            assert result == {"ok": True, "live": True}
            source = mgr._data_manager.get_source("agenda")
            assert isinstance(source, StaticSource)
            assert await source.fetch() == {"entries": ["fresh"]}
            assert mgr.get_spec("weekly_glance").data_sources[0].value == {
                "entries": ["fresh"],
            }
        finally:
            await mgr._data_manager.stop_all()

    @pytest.mark.asyncio
    async def test_update_unknown_display_returns_error(self):
        from boxbot.displays.manager import DisplayManager

        mgr = DisplayManager()
        result = mgr.update_static_data("nope", "x", {})
        assert result["ok"] is False
        assert "not registered" in result["error"]

    @pytest.mark.asyncio
    async def test_update_unknown_source_returns_error(self):
        from boxbot.displays.manager import DisplayManager

        mgr = DisplayManager()
        mgr.register_spec(self._static_spec())
        result = mgr.update_static_data("weekly_glance", "ghost", {})
        assert result["ok"] is False
        assert "no source named" in result["error"]

    @pytest.mark.asyncio
    async def test_update_non_static_source_returns_error(self):
        from boxbot.displays.manager import DisplayManager

        spec = DisplaySpec(
            name="brief",
            theme="boxbot",
            data_sources=[
                DataSourceSpec(name="weather", source_type="builtin"),
            ],
            root_block=TextBlock(content="hi"),
        )
        mgr = DisplayManager()
        mgr.register_spec(spec)
        result = mgr.update_static_data("brief", "weather", {})
        assert result["ok"] is False
        assert "not 'static'" in result["error"]

    @pytest.mark.asyncio
    async def test_dispatcher_persists_inactive_update_for_agent_display(
        self, tmp_path, monkeypatch,
    ):
        # Point the agent displays dir at a tmp path so we don't write
        # into the real data/ tree. Reload sandbox_actions so the helper
        # _agent_displays_dir picks up the override.
        monkeypatch.setenv("BOXBOT_DATA_DIR", str(tmp_path))
        import importlib
        import boxbot.core.paths as paths
        importlib.reload(paths)
        import boxbot.tools._sandbox_actions as sa
        importlib.reload(sa)

        from boxbot.displays.manager import (
            DisplayManager,
            set_display_manager,
        )

        mgr = DisplayManager()
        spec = self._static_spec()
        mgr.register_spec(spec)
        # Simulate an agent-saved display: a JSON file already exists at
        # the canonical path so the dispatcher knows to persist updates.
        import json as _json
        agent_path = paths.DISPLAYS_DIR / "weekly_glance.json"
        agent_path.parent.mkdir(parents=True, exist_ok=True)
        agent_path.write_text(_json.dumps({"name": "weekly_glance"}))

        set_display_manager(mgr)
        try:
            ctx = sa.ActionContext()
            resp = await sa._handle_display_action(
                "display.update_data",
                {
                    "display": "weekly_glance",
                    "source": "agenda",
                    "value": {"entries": ["Mon: dentist"]},
                },
                ctx,
            )
            assert resp["status"] == "ok", resp
            assert resp["live"] is False
            assert resp["persisted"] is True

            on_disk = _json.loads(agent_path.read_text())
            # The dispatcher round-trips the spec through _spec_to_dict
            # before writing, so the persisted shape matches what
            # display.load would see.
            assert on_disk["name"] == "weekly_glance"
            agenda = next(
                s for s in on_disk["data_sources"] if s["name"] == "agenda"
            )
            assert agenda["value"] == {"entries": ["Mon: dentist"]}
        finally:
            set_display_manager(None)

    @pytest.mark.asyncio
    async def test_dispatcher_skips_persist_for_non_agent_display(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv("BOXBOT_DATA_DIR", str(tmp_path))
        import importlib
        import boxbot.core.paths as paths
        importlib.reload(paths)
        import boxbot.tools._sandbox_actions as sa
        importlib.reload(sa)

        from boxbot.displays.manager import (
            DisplayManager,
            set_display_manager,
        )

        mgr = DisplayManager()
        mgr.register_spec(self._static_spec("builtin_glance"))

        set_display_manager(mgr)
        try:
            ctx = sa.ActionContext()
            resp = await sa._handle_display_action(
                "display.update_data",
                {
                    "display": "builtin_glance",
                    "source": "agenda",
                    "value": {"entries": ["x"]},
                },
                ctx,
            )
            assert resp["status"] == "ok", resp
            assert resp["persisted"] is False
            # No file was created in the agent dir.
            assert not (paths.DISPLAYS_DIR / "builtin_glance.json").exists()
        finally:
            set_display_manager(None)


class TestPreviewDataPrecedence:
    """Source names are a global namespace, so live data from the display
    that happens to be on screen must not leak into the preview of a
    *different* spec that reuses the name. That shadowing silently
    rendered live values for a spec's own declared statics and cost an
    on-device agent a whole turn budget chasing a phantom binding bug.
    """

    @staticmethod
    def _spec(name: str, value: dict[str, Any]) -> DisplaySpec:
        return DisplaySpec(
            name=name,
            theme="boxbot",
            data_sources=[
                DataSourceSpec(name="climate", source_type="static", value=value),
            ],
            root_block=TextBlock(content="{climate.temp}"),
        )

    @pytest.mark.asyncio
    async def test_other_spec_declared_values_win_over_live_source(self):
        from boxbot.displays.manager import DisplayManager

        mgr = DisplayManager()
        mgr.register_spec(self._spec("onscreen", {"temp": 71, "mode": "Heat"}))
        await mgr.switch("onscreen")
        try:
            draft = self._spec("draft", {"temp": 64, "humidity": 43})
            preview = mgr.build_preview_data(draft)
            assert preview["climate"] == {"temp": 64, "humidity": 43}
        finally:
            await mgr._data_manager.stop_all()

    @pytest.mark.asyncio
    async def test_self_preview_sees_pushed_update_data(self):
        from boxbot.displays.manager import DisplayManager

        mgr = DisplayManager()
        mgr.register_spec(self._spec("onscreen", {"temp": 71}))
        await mgr.switch("onscreen")
        try:
            mgr.update_static_data("onscreen", "climate", {"temp": 68})
            preview = mgr.build_preview_data(mgr.get_spec("onscreen"))
            assert preview["climate"]["temp"] == 68
        finally:
            await mgr._data_manager.stop_all()

    @pytest.mark.asyncio
    async def test_caller_override_wins(self):
        from boxbot.displays.manager import DisplayManager

        mgr = DisplayManager()
        spec = self._spec("draft", {"temp": 64})
        mgr.register_spec(spec)
        preview = mgr.build_preview_data(spec, data={"climate": {"temp": 99}})
        assert preview["climate"] == {"temp": 99}

    @pytest.mark.asyncio
    async def test_falsy_caller_override_still_wins(self):
        """An empty override is a choice, not an absence."""
        from boxbot.displays.manager import DisplayManager

        mgr = DisplayManager()
        spec = self._spec("draft", {"temp": 64})
        mgr.register_spec(spec)
        preview = mgr.build_preview_data(spec, data={"climate": {}})
        assert preview["climate"] == {}

    @pytest.mark.asyncio
    async def test_undeclared_source_falls_back_to_placeholder(self):
        from boxbot.displays.manager import DisplayManager

        spec = DisplaySpec(
            name="brief",
            theme="boxbot",
            data_sources=[DataSourceSpec(name="weather", source_type="builtin")],
            root_block=TextBlock(content="{weather.temp}"),
        )
        mgr = DisplayManager()
        preview = mgr.build_preview_data(spec, args={"who": "jacob"})
        assert preview["weather"]
        assert preview["args"] == {"who": "jacob"}

    def test_unresolved_binding_warning_lists_available_fields(self):
        from boxbot.tools._sandbox_actions import _collect_unresolved_bindings

        spec_dict = {
            "name": "draft",
            "data_sources": [
                {
                    "name": "climate",
                    "type": "static",
                    "value": {"temp": 71, "mode": "Heat"},
                },
            ],
            "layout": {"type": "text", "content": "{climate.humidity}"},
        }
        warnings = _collect_unresolved_bindings(
            spec_dict, render_data={"climate": {"temp": 71, "mode": "Heat"}},
        )
        assert len(warnings) == 1
        assert "Available fields on 'climate': mode, temp." in warnings[0]


class TestPictureSlideshow:
    """Picture display slideshow mode.

    Regression: switching to ``picture`` with no ``image_ids`` rendered a
    tiny "photo:" placeholder on a black screen because nothing populated
    the slideshow list. Slideshow mode must pull the ``in_slideshow`` set,
    and an empty set must fall back to a friendly notice rather than a
    broken image block.
    """

    @staticmethod
    def _register_builtins(mgr) -> None:
        from boxbot.displays.builtins import get_builtin_specs

        for spec in get_builtin_specs():
            mgr.register_spec(spec)

    @staticmethod
    def _fake_config(storage_path):
        import types

        return types.SimpleNamespace(
            photos=types.SimpleNamespace(storage_path=str(storage_path)),
        )

    @pytest.mark.asyncio
    async def test_slideshow_populates_ids_from_set(self, tmp_path, monkeypatch):
        from boxbot.displays.manager import DisplayManager
        from boxbot.photos.store import PhotoStore

        storage = tmp_path / "photos"
        storage.mkdir()
        store = PhotoStore(db_path=storage / "photos.db")
        await store.initialize()
        try:
            await store.create_photo(
                filename="a.jpg", source="camera", photo_id="aaa")
            await store.create_photo(
                filename="b.jpg", source="camera", photo_id="bbb")
            # Excluded from the slideshow rotation.
            await store.create_photo(
                filename="c.jpg", source="camera",
                in_slideshow=False, photo_id="ccc")
        finally:
            await store.close()

        monkeypatch.setattr(
            "boxbot.core.config.get_config",
            lambda: self._fake_config(storage),
        )

        mgr = DisplayManager()
        self._register_builtins(mgr)
        try:
            ok = await mgr.switch("picture", args={})
            assert ok
            assert mgr._active_display == "picture"
            ids = mgr._active_args.get("image_ids")
            assert set(ids) == {"aaa", "bbb"}
            assert "ccc" not in ids
        finally:
            mgr._stop_slideshow()
            await mgr._data_manager.stop_all()

    @pytest.mark.asyncio
    async def test_empty_slideshow_falls_back_to_notice(
        self, tmp_path, monkeypatch,
    ):
        from boxbot.displays.manager import DisplayManager
        from boxbot.photos.store import PhotoStore

        storage = tmp_path / "photos"
        storage.mkdir()
        store = PhotoStore(db_path=storage / "photos.db")
        await store.initialize()
        await store.close()  # empty DB — nothing in the slideshow set

        monkeypatch.setattr(
            "boxbot.core.config.get_config",
            lambda: self._fake_config(storage),
        )

        mgr = DisplayManager()
        self._register_builtins(mgr)
        try:
            ok = await mgr.switch("picture", args={})
            assert ok
            # Falls back to the notice display, not a broken picture.
            assert mgr._active_display == "notice"
            assert mgr._active_args.get("title") == "No photos yet"
        finally:
            await mgr._data_manager.stop_all()

    @pytest.mark.asyncio
    async def test_explicit_ids_bypass_slideshow(self, tmp_path, monkeypatch):
        from boxbot.displays.manager import DisplayManager

        # No photos DB at all -> _load_slideshow_ids returns [], but explicit
        # ids must still be honored without consulting the slideshow set.
        storage = tmp_path / "photos"
        storage.mkdir()
        monkeypatch.setattr(
            "boxbot.core.config.get_config",
            lambda: self._fake_config(storage),
        )

        mgr = DisplayManager()
        self._register_builtins(mgr)
        try:
            ok = await mgr.switch("picture", args={"image_ids": ["xyz"]})
            assert ok
            assert mgr._active_display == "picture"
            assert mgr._active_args["image_ids"] == ["xyz"]
        finally:
            mgr._stop_slideshow()
            await mgr._data_manager.stop_all()


# ---------------------------------------------------------------------------
# Status pill (transient agent-activity overlay)
# ---------------------------------------------------------------------------


class TestStatusPill:
    """The tool-derived status pill composited over the active frame."""

    def _mgr_with_frame(self, size=(1024, 600)):
        from PIL import Image

        from boxbot.displays.manager import DisplayManager

        mgr = DisplayManager(width=size[0], height=size[1])
        base = Image.new("RGB", size, (20, 26, 33))
        mgr._update_frame(base)
        return mgr, base

    def test_set_status_composites_and_bumps_generation(self):
        mgr, base = self._mgr_with_frame()
        g0 = mgr.get_frame_generation()

        mgr.set_status_text("Searching the web…")

        assert mgr.get_frame_generation() == g0 + 1
        assert mgr._current_frame.tobytes() != base.tobytes()
        # Base frame is preserved un-overlaid.
        assert mgr._base_frame.tobytes() == base.tobytes()

    def test_same_text_is_noop_but_refreshes_deadline(self):
        mgr, _ = self._mgr_with_frame()
        mgr.set_status_text("Recalling…")
        g1 = mgr.get_frame_generation()
        d1 = mgr._status_deadline

        mgr.set_status_text("Recalling…")

        assert mgr.get_frame_generation() == g1
        assert mgr._status_deadline >= d1

    def test_new_base_frame_reapplies_pill(self):
        from PIL import Image

        mgr, _ = self._mgr_with_frame()
        mgr.set_status_text("Checking the camera…")

        base2 = Image.new("RGB", (1024, 600), (40, 40, 40))
        mgr._update_frame(base2)

        assert mgr._base_frame is base2
        assert mgr._current_frame.tobytes() != base2.tobytes()

    def test_clear_restores_base_frame(self):
        mgr, _ = self._mgr_with_frame()
        mgr.set_status_text("Fetching data…")
        g1 = mgr.get_frame_generation()

        mgr.clear_status_text()

        assert mgr.get_frame_generation() == g1 + 1
        assert mgr._current_frame is mgr._base_frame
        assert mgr._status_text is None

    def test_clear_without_status_is_noop(self):
        mgr, _ = self._mgr_with_frame()
        g0 = mgr.get_frame_generation()
        mgr.clear_status_text()
        assert mgr.get_frame_generation() == g0

    def test_pill_renders_at_non_default_resolution(self):
        mgr, base = self._mgr_with_frame(size=(1280, 800))
        mgr.set_status_text("Searching the web…")
        assert mgr._current_frame.size == (1280, 800)
        assert mgr._current_frame.tobytes() != base.tobytes()

    @pytest.mark.asyncio
    async def test_tool_called_event_voice_only(self):
        from boxbot.core.events import AgentToolCalled

        mgr, base = self._mgr_with_frame()

        await mgr._on_agent_tool_called(
            AgentToolCalled(
                conversation_id="whatsapp_1", channel="whatsapp",
                tool_name="web_search", status_text="Searching the web…",
            )
        )
        assert mgr._status_text is None

        await mgr._on_agent_tool_called(
            AgentToolCalled(
                conversation_id="voice_room", channel="voice",
                tool_name="web_search", status_text="Searching the web…",
            )
        )
        assert mgr._status_text == "Searching the web…"

    @pytest.mark.asyncio
    async def test_status_clear_handler(self):
        from boxbot.core.events import AgentTurnEnded

        mgr, _ = self._mgr_with_frame()
        mgr.set_status_text("Working on it…")

        await mgr._on_status_clear(
            AgentTurnEnded(conversation_id="voice_room", channel="voice")
        )
        assert mgr._status_text is None
