"""Tests for SDK modules — transport, validators, display, skill, memory, photos, tasks, packages, secrets."""

from __future__ import annotations

import io
import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from boxbot.sdk import _validators as v
from boxbot.sdk._transport import _MARKER, collect_response, emit_action


# ---------------------------------------------------------------------------
# Transport layer
# ---------------------------------------------------------------------------


class TestTransport:
    """Test the SDK <-> main process transport mechanism."""

    def test_marker_constant(self):
        assert _MARKER == "__BOXBOT_SDK_ACTION__:"

    def test_emit_action_writes_to_stdout(self):
        fake_stdout = io.StringIO()
        with patch("sys.stdout", fake_stdout):
            emit_action("memory.save", {"content": "test data"})

        output = fake_stdout.getvalue()
        assert _MARKER in output
        # Parse the JSON after the marker
        json_str = output.split(_MARKER, 1)[1].strip()
        data = json.loads(json_str)
        assert data["_sdk"] == "memory.save"
        assert data["content"] == "test data"

    def test_emit_action_includes_sdk_type(self):
        fake_stdout = io.StringIO()
        with patch("sys.stdout", fake_stdout):
            emit_action("display.save", {"name": "weather"})

        output = fake_stdout.getvalue()
        data = json.loads(output.split(_MARKER, 1)[1].strip())
        assert data["_sdk"] == "display.save"
        assert data["name"] == "weather"

    def test_collect_response_reads_json_from_stdin(self):
        response_data = {"approved": True, "reason": "User approved"}
        fake_stdin = io.StringIO(json.dumps(response_data) + "\n")
        with patch("sys.stdin", fake_stdin):
            # We need to also mock select.select to indicate stdin is ready
            with patch("select.select", return_value=([fake_stdin], [], [])):
                result = collect_response(timeout=5)
        assert result["approved"] is True
        assert result["reason"] == "User approved"

    def test_collect_response_timeout_raises(self):
        with patch("select.select", return_value=([], [], [])):
            with pytest.raises(TimeoutError, match="No response"):
                collect_response(timeout=1)

    def test_collect_response_malformed_json_raises(self):
        fake_stdin = io.StringIO("not valid json\n")
        with patch("sys.stdin", fake_stdin):
            with patch("select.select", return_value=([fake_stdin], [], [])):
                with pytest.raises(RuntimeError, match="Malformed response"):
                    collect_response(timeout=5)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


class TestValidators:
    """Test input validation functions and constants."""

    def test_require_raises_on_none(self):
        with pytest.raises(ValueError, match="required"):
            v.require(None, "test_field")

    def test_require_passes_on_value(self):
        v.require("something", "test_field")  # should not raise

    def test_require_str_validates_type(self):
        with pytest.raises(ValueError, match="must be a string"):
            v.require_str(123, "field")

    def test_require_str_rejects_empty(self):
        with pytest.raises(ValueError, match="must not be empty"):
            v.require_str("  ", "field")

    def test_require_str_allows_empty_when_flagged(self):
        result = v.require_str("", "field", allow_empty=True)
        assert result == ""

    def test_require_int_validates_type(self):
        with pytest.raises(ValueError, match="must be an integer"):
            v.require_int("5", "field")

    def test_require_int_validates_range(self):
        with pytest.raises(ValueError, match="must be >="):
            v.require_int(-1, "field", min_val=0)

    def test_require_int_rejects_bool(self):
        with pytest.raises(ValueError, match="must be an integer"):
            v.require_int(True, "field")

    def test_require_float_validates_range(self):
        with pytest.raises(ValueError, match="must be >="):
            v.require_float(-0.5, "field", min_val=0.0)

    def test_require_bool_validates_type(self):
        with pytest.raises(ValueError, match="must be a boolean"):
            v.require_bool(1, "field")

    def test_require_list_validates_type(self):
        with pytest.raises(ValueError, match="must be a list"):
            v.require_list("not a list", "field")

    def test_require_list_accepts_tuple(self):
        result = v.require_list((1, 2, 3), "field")
        assert result == [1, 2, 3]

    def test_require_dict_validates_type(self):
        with pytest.raises(ValueError, match="must be a dict"):
            v.require_dict([1, 2], "field")

    def test_validate_one_of_accepts_valid(self):
        result = v.validate_one_of("boxbot", "theme", v.VALID_THEMES)
        assert result == "boxbot"

    def test_validate_one_of_rejects_invalid(self):
        with pytest.raises(ValueError, match="must be one of"):
            v.validate_one_of("neon", "theme", v.VALID_THEMES)

    def test_validate_padding_int(self):
        assert v.validate_padding(10) == 10

    def test_validate_padding_list_of_two(self):
        assert v.validate_padding([10, 20]) == [10, 20]

    def test_validate_padding_list_of_four(self):
        assert v.validate_padding([10, 20, 30, 40]) == [10, 20, 30, 40]

    def test_validate_padding_rejects_list_of_three(self):
        with pytest.raises(ValueError, match="2 or 4"):
            v.validate_padding([10, 20, 30])

    def test_validate_padding_rejects_negative(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            v.validate_padding(-1)

    def test_validate_ratios_requires_at_least_two(self):
        with pytest.raises(ValueError, match="at least 2"):
            v.validate_ratios([1])

    def test_validate_ratios_rejects_zero(self):
        with pytest.raises(ValueError, match="positive integer"):
            v.validate_ratios([1, 0])

    def test_validate_name_accepts_valid(self):
        assert v.validate_name("my-display_1", "display name") == "my-display_1"

    def test_validate_name_rejects_special_chars(self):
        with pytest.raises(ValueError, match="alphanumeric"):
            v.validate_name("my display!", "display name")

    def test_validate_data_source_builtin(self):
        config = v.validate_data_source_config("weather")
        assert config["name"] == "weather"

    def test_validate_data_source_unknown_builtin_raises(self):
        with pytest.raises(ValueError, match="Unknown built-in"):
            v.validate_data_source_config("unknown_source")

    def test_validate_data_source_http_json(self):
        config = v.validate_data_source_config(
            "stock_data", source_type="http_json", url="https://api.example.com"
        )
        assert config["type"] == "http_json"
        assert config["url"] == "https://api.example.com"

    def test_validate_data_source_http_json_requires_url(self):
        with pytest.raises(ValueError, match="requires 'url'"):
            v.validate_data_source_config("bad_source", source_type="http_json")

    def test_valid_themes_contains_all_four(self):
        assert v.VALID_THEMES == {"boxbot", "midnight", "daylight", "classic"}


# ---------------------------------------------------------------------------
# Display builder
# ---------------------------------------------------------------------------


class TestDisplayBuilder:
    """Test the declarative display builder API."""

    def test_create_returns_display_builder(self):
        from boxbot.sdk.display import create
        d = create("test_display")
        assert d.name == "test_display"

    def test_create_rejects_invalid_name(self):
        from boxbot.sdk.display import create
        with pytest.raises(ValueError):
            create("bad name!")

    def test_set_theme(self):
        from boxbot.sdk.display import create
        d = create("themed")
        d.set_theme("midnight")
        spec = d._build_spec()
        assert spec["theme"] == "midnight"

    def test_add_data_source(self):
        from boxbot.sdk.display import create
        d = create("data_test")
        d.data("weather")
        spec = d._build_spec()
        assert len(spec["data_sources"]) == 1
        assert spec["data_sources"][0]["name"] == "weather"

    def test_add_text_block(self):
        from boxbot.sdk.display import create
        d = create("text_test")
        d.text("Hello World", size="title", color="accent")
        spec = d._build_spec()
        # Single child unwraps — layout IS the text block.
        assert spec["layout"]["type"] == "text"
        assert spec["layout"]["content"] == "Hello World"

    def test_nested_row_with_children(self):
        from boxbot.sdk.display import create
        d = create("nested_test")
        row = d.row(gap=16)
        row.text("Left")
        row.text("Right")
        spec = d._build_spec()
        # Single top-level row → layout unwraps to the row itself.
        assert spec["layout"]["type"] == "row"
        assert len(spec["layout"]["children"]) == 2

    def test_columns_builder(self):
        from boxbot.sdk.display import create
        d = create("cols_test")
        cols = d.columns([2, 1], gap=8)
        assert len(cols) == 2
        cols[0].text("Wide column")
        cols[1].text("Narrow column")
        spec = d._build_spec()
        columns_block = spec["layout"]
        assert columns_block["type"] == "columns"
        assert columns_block["ratios"] == [2, 1]

    def test_metric_block(self):
        from boxbot.sdk.display import create
        d = create("metric_test")
        d.metric("{weather.temp}", label="Temperature", icon="thermometer")
        spec = d._build_spec()
        block = spec["layout"]
        assert block["type"] == "metric"
        assert block["value"] == "{weather.temp}"

    def test_multiple_top_level_blocks_wrap_in_column(self):
        from boxbot.sdk.display import create
        d = create("multi_test")
        d.text("first")
        d.text("second")
        spec = d._build_spec()
        # Two siblings → wrapped in implicit column.
        assert spec["layout"]["type"] == "column"
        assert len(spec["layout"]["children"]) == 2

    def test_chart_requires_data_or_series(self):
        from boxbot.sdk.display import create
        d = create("chart_test")
        with pytest.raises(ValueError, match="requires either"):
            d.chart()

    def test_save_round_trips_through_load(self):
        from boxbot.sdk import display as display_mod
        from boxbot.sdk.display import create

        d = create("rt_test")
        d.set_theme("midnight")
        d.data("weather")
        d.text("{weather.temp}", size="title")
        spec = d._build_spec()

        # Load uses the same layout shape that save would produce.
        rebuilt = display_mod.DisplayBuilder._from_spec(spec)
        rebuilt_spec = rebuilt._build_spec()
        assert rebuilt_spec["theme"] == "midnight"
        assert rebuilt_spec["layout"] == spec["layout"]

    def test_set_transition(self):
        from boxbot.sdk.display import create
        d = create("transition_test")
        d.set_transition("slide_left")
        spec = d._build_spec()
        assert spec["transition"] == "slide_left"

    def test_rotate_config(self):
        from boxbot.sdk.display import create
        d = create("rotate_test")
        d.rotate("photos", "items", interval=15)
        spec = d._build_spec()
        assert spec["rotate"]["source"] == "photos"
        assert spec["rotate"]["interval"] == 15


# ---------------------------------------------------------------------------
# Skill builder
# ---------------------------------------------------------------------------


class TestSkillBuilder:
    """Test the declarative skill builder API.

    Skills are structured prompt data: a SKILL.md (frontmatter + body)
    plus optional bundled scripts and Level 3 sub-docs. Parameters,
    env vars, and scheduling belong to integrations, not skills.
    """

    def test_create_returns_skill_builder(self):
        from boxbot.sdk.skill import create
        s = create("test-skill")
        assert s.name == "test-skill"

    def test_create_rejects_uppercase(self):
        from boxbot.sdk.skill import create
        with pytest.raises(ValueError, match="lowercase"):
            create("TestSkill")

    def test_create_rejects_reserved_names(self):
        from boxbot.sdk.skill import create
        with pytest.raises(ValueError, match="reserved"):
            create("anthropic")
        with pytest.raises(ValueError, match="reserved"):
            create("claude")

    def test_create_rejects_overlong_name(self):
        from boxbot.sdk.skill import create
        with pytest.raises(ValueError, match="≤64"):
            create("a" * 65)

    def test_save_without_description_raises(self):
        from boxbot.sdk.skill import create
        s = create("incomplete")
        s.body = "# Hi"
        with pytest.raises(ValueError, match="description is required"):
            s.save()

    def test_save_without_body_raises(self):
        from boxbot.sdk.skill import create
        s = create("no-body")
        s.description = "A skill without a body"
        with pytest.raises(ValueError, match="body is required"):
            s.save()

    def test_description_rejects_overlong(self):
        from boxbot.sdk.skill import create
        s = create("longdesc")
        with pytest.raises(ValueError, match="≤1024"):
            s.description = "x" * 1025

    def test_description_rejects_xml(self):
        from boxbot.sdk.skill import create
        s = create("xmldesc")
        with pytest.raises(ValueError, match="XML"):
            s.description = "Use this <evil>tag</evil>"

    def test_save_emits_minimal_payload(self):
        from boxbot.sdk.skill import create
        s = create("weather")
        s.description = "Get weather forecasts. Use when asked about weather."
        s.body = "# Weather\n\nUse bb.weather.forecast()."

        fake_stdout = io.StringIO()
        with patch("sys.stdout", fake_stdout):
            s.save()
        output = fake_stdout.getvalue()
        data = json.loads(output.split(_MARKER, 1)[1].strip())
        assert data["_sdk"] == "skill.save"
        assert data["name"] == "weather"
        assert data["description"].startswith("Get weather")
        assert data["body"].startswith("# Weather")
        assert "scripts" not in data
        assert "resources" not in data

    def test_add_script_emits_in_payload(self):
        from boxbot.sdk.skill import create
        s = create("with-script")
        s.description = "Skill bundling a helper script."
        s.body = "# With script\n\nSee scripts/helper.py."
        s.add_script("helper.py", "def go():\n    return 42\n")

        fake_stdout = io.StringIO()
        with patch("sys.stdout", fake_stdout):
            s.save()
        data = json.loads(fake_stdout.getvalue().split(_MARKER, 1)[1].strip())
        assert len(data["scripts"]) == 1
        assert data["scripts"][0]["filename"] == "helper.py"
        assert "def go" in data["scripts"][0]["content"]

    def test_add_script_rejects_path_traversal(self):
        from boxbot.sdk.skill import create
        s = create("badpath")
        s.description = "x" * 30
        s.body = "# x"
        with pytest.raises(ValueError, match="bare basename"):
            s.add_script("../escape.py", "x")

    def test_add_script_requires_py_extension(self):
        from boxbot.sdk.skill import create
        s = create("ext")
        s.description = "x" * 30
        s.body = "# x"
        with pytest.raises(ValueError, match=r"\.py"):
            s.add_script("helper.txt", "x")

    def test_add_resource_emits_in_payload(self):
        from boxbot.sdk.skill import create
        s = create("with-ref")
        s.description = "Skill with a Level 3 sub-doc."
        s.body = "# Skill\n\nSee REFERENCE.md."
        s.add_resource("REFERENCE.md", "# Reference\n\n…")

        fake_stdout = io.StringIO()
        with patch("sys.stdout", fake_stdout):
            s.save()
        data = json.loads(fake_stdout.getvalue().split(_MARKER, 1)[1].strip())
        assert data["resources"][0]["filename"] == "REFERENCE.md"

    def test_add_resource_rejects_skill_md(self):
        from boxbot.sdk.skill import create
        s = create("res-skill-md")
        s.description = "x" * 30
        s.body = "# x"
        with pytest.raises(ValueError, match="reserved"):
            s.add_resource("SKILL.md", "x")


# ---------------------------------------------------------------------------
# SDK memory module
# ---------------------------------------------------------------------------


class TestSDKMemory:
    """Test the SDK memory module functions."""

    def test_save_emits_action(self):
        from boxbot.sdk import memory
        fake_stdout = io.StringIO()
        with patch("sys.stdout", fake_stdout):
            memory.save(
                content="Jacob likes coffee",
                memory_type="person",
                people=["Jacob"],
                importance=0.8,
            )
        output = fake_stdout.getvalue()
        data = json.loads(output.split(_MARKER, 1)[1].strip())
        assert data["_sdk"] == "memory.save"
        assert data["content"] == "Jacob likes coffee"
        assert data["importance"] == 0.8

    def test_save_validates_memory_type(self):
        from boxbot.sdk import memory
        with pytest.raises(ValueError, match="must be one of"):
            memory.save(content="test", memory_type="invalid")

    def test_delete_emits_action(self):
        from boxbot.sdk import memory
        fake_stdout = io.StringIO()
        with patch("sys.stdout", fake_stdout):
            memory.delete("mem-123")
        output = fake_stdout.getvalue()
        data = json.loads(output.split(_MARKER, 1)[1].strip())
        assert data["_sdk"] == "memory.delete"
        assert data["id"] == "mem-123"

    def test_memory_record_properties(self):
        from boxbot.sdk.memory import MemoryRecord
        record = MemoryRecord({
            "id": "m1",
            "content": "Test",
            "type": "person",
            "people": ["Alice"],
            "tags": ["health"],
            "importance": 0.9,
            "created_at": "2025-01-01",
        })
        assert record.id == "m1"
        assert record.content == "Test"
        assert record.memory_type == "person"
        assert record.importance == 0.9
        assert "Alice" in record.people


# ---------------------------------------------------------------------------
# SDK secrets module
# ---------------------------------------------------------------------------


class TestSDKSecrets:
    """Test the SDK secrets module."""

    def test_store_emits_action(self):
        from boxbot.sdk import secrets
        fake_stdout = io.StringIO()
        with patch("sys.stdout", fake_stdout):
            secrets.store("my_key", "my_value")
        output = fake_stdout.getvalue()
        data = json.loads(output.split(_MARKER, 1)[1].strip())
        assert data["_sdk"] == "secrets.store"
        assert data["name"] == "my_key"

    def test_use_returns_env_var_name(self):
        from boxbot.sdk import secrets
        result = secrets.use("polygon_api_key")
        assert result == "BOXBOT_SECRET_POLYGON_API_KEY"

    def test_use_uppercases_name(self):
        from boxbot.sdk import secrets
        result = secrets.use("my_secret")
        assert result == "BOXBOT_SECRET_MY_SECRET"


# ---------------------------------------------------------------------------
# SDK packages module
# ---------------------------------------------------------------------------


class TestSDKPackages:
    """Test the SDK packages module."""

    def test_package_result_repr(self):
        from boxbot.sdk.packages import PackageResult
        approved = PackageResult(approved=True)
        assert "approved" in repr(approved)
        denied = PackageResult(approved=False, reason="Not needed")
        assert "denied" in repr(denied)
        assert "Not needed" in repr(denied)
