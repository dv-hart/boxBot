"""Tests for the bb.integrations SDK module — payload emission shape.

End-to-end (SDK → dispatcher → runner → output) is covered in
``test_integration_runner.py``. These tests verify the SDK side
emits the right action types and payload shapes.
"""

from __future__ import annotations

import io
import json
from unittest.mock import patch

import pytest

from boxbot.sdk import integrations as bb_integrations


# The SDK's action marker is the same constant the dispatcher reads.
_MARKER = "__BOXBOT_SDK_ACTION__:"


def _capture(callable_, *args, **kwargs):
    """Capture the action(s) the SDK emits + return value of the call."""
    fake_stdout = io.StringIO()
    fake_stdin = io.StringIO('{"status": "ok"}\n')  # canned response
    with patch("sys.stdout", fake_stdout), patch("sys.stdin", fake_stdin):
        result = callable_(*args, **kwargs)
    line = fake_stdout.getvalue().strip()
    assert _MARKER in line, f"no action marker in output: {line!r}"
    payload = json.loads(line.split(_MARKER, 1)[1])
    return payload, result


class TestList:
    def test_emits_list_action(self):
        payload, _ = _capture(bb_integrations.list)
        assert payload["_sdk"] == "integrations.list"


class TestGet:
    def test_emits_get_with_inputs(self):
        payload, _ = _capture(
            bb_integrations.get, "weather", lat=45.5, lon=-122.7, days=3
        )
        assert payload["_sdk"] == "integrations.get"
        assert payload["name"] == "weather"
        assert payload["inputs"] == {"lat": 45.5, "lon": -122.7, "days": 3}

    def test_get_requires_name(self):
        with pytest.raises(ValueError):
            _capture(bb_integrations.get, "")


class TestLogs:
    def test_emits_logs_action(self):
        payload, _ = _capture(bb_integrations.logs, "weather", limit=5)
        assert payload["_sdk"] == "integrations.logs"
        assert payload["name"] == "weather"
        assert payload["limit"] == 5

    def test_logs_rejects_invalid_limit(self):
        with pytest.raises(ValueError):
            _capture(bb_integrations.logs, "weather", limit=0)
        with pytest.raises(ValueError):
            _capture(bb_integrations.logs, "weather", limit=999)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class TestIntegrationBuilder:
    def test_create_returns_builder(self):
        i = bb_integrations.create("weather")
        assert i.name == "weather"

    def test_create_rejects_uppercase(self):
        with pytest.raises(ValueError, match="lowercase"):
            bb_integrations.create("Weather")

    def test_save_emits_create_action(self):
        i = bb_integrations.create("weather")
        i.description = "Get NOAA forecasts."
        i.add_input("lat", type="float", required=True)
        i.add_output("today", type="object", description="Today's forecast.")
        i.add_secret("NWS_USER_AGENT")
        i.script = (
            "from boxbot_sdk.integration import return_output\n"
            "return_output({'ok': True})\n"
        )
        i.timeout = 20

        payload, _ = _capture(i.save)
        assert payload["_sdk"] == "integrations.create"
        assert payload["name"] == "weather"
        assert payload["description"] == "Get NOAA forecasts."
        assert payload["inputs"]["lat"] == {"type": "float", "required": True}
        assert payload["outputs"]["today"]["description"] == "Today's forecast."
        assert payload["secrets"] == ["NWS_USER_AGENT"]
        assert payload["timeout"] == 20
        assert "return_output" in payload["script"]

    def test_save_without_description_raises(self):
        i = bb_integrations.create("nodescr")
        i.script = "from boxbot_sdk.integration import return_output\nreturn_output({})\n"
        with pytest.raises(ValueError, match="description"):
            i.save()

    def test_save_without_script_raises(self):
        i = bb_integrations.create("noscript")
        i.description = "x" * 30
        with pytest.raises(ValueError, match="script"):
            i.save()

    def test_add_secret_dedupes(self):
        i = bb_integrations.create("dedup")
        i.add_secret("FOO")
        i.add_secret("FOO")
        i.add_secret("BAR")
        i.description = "x"
        i.script = "from boxbot_sdk.integration import return_output\nreturn_output({})\n"
        payload, _ = _capture(i.save)
        assert payload["secrets"] == ["FOO", "BAR"]

    def test_timeout_validation(self):
        i = bb_integrations.create("badtimeout")
        with pytest.raises(ValueError):
            i.timeout = 0
        with pytest.raises(ValueError):
            i.timeout = 9999


# ---------------------------------------------------------------------------
# Update / delete
# ---------------------------------------------------------------------------


class TestUpdateDelete:
    def test_update_emits_action(self):
        payload, _ = _capture(
            bb_integrations.update, "weather", script="# new\n"
        )
        assert payload["_sdk"] == "integrations.update"
        assert payload["name"] == "weather"
        assert payload["script"] == "# new\n"

    def test_update_requires_payload(self):
        with pytest.raises(ValueError, match="manifest|script"):
            bb_integrations.update("weather")

    def test_delete_emits_action(self):
        payload, _ = _capture(bb_integrations.delete, "weather")
        assert payload["_sdk"] == "integrations.delete"
        assert payload["name"] == "weather"
