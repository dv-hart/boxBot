"""Tests for the integration persist writer (CRUD on disk)."""

from __future__ import annotations

from pathlib import Path

import pytest

from boxbot.integrations import loader as integ_loader
from boxbot.integrations.persist import (
    create_integration,
    delete_integration,
    update_integration,
)


_BASE_PAYLOAD = {
    "name": "weather",
    "description": "Get NOAA weather forecasts.",
    "inputs": {"lat": {"type": "float", "required": True}},
    "secrets": ["NWS_USER_AGENT"],
    "timeout": 15,
    "script": (
        "from boxbot_sdk.integration import inputs, return_output\n"
        "return_output({'today': {'high': 79}})\n"
    ),
}


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


class TestCreate:
    def test_creates_directory_and_files(self, tmp_path: Path):
        result = create_integration(_BASE_PAYLOAD, integrations_root=tmp_path)
        assert result["status"] == "ok"
        assert result["files"] == ["manifest.yaml", "script.py"]

        target = tmp_path / "weather"
        assert (target / "manifest.yaml").is_file()
        assert (target / "script.py").is_file()
        manifest_text = (target / "manifest.yaml").read_text()
        assert "name: weather" in manifest_text
        assert "NWS_USER_AGENT" in manifest_text

    def test_round_trip_through_loader(self, tmp_path: Path):
        create_integration(_BASE_PAYLOAD, integrations_root=tmp_path)
        metas = integ_loader.discover_integrations(root=tmp_path)
        assert len(metas) == 1
        assert metas[0].name == "weather"
        assert metas[0].timeout == 15
        assert "NWS_USER_AGENT" in metas[0].secrets

    def test_existing_returns_exists(self, tmp_path: Path):
        create_integration(_BASE_PAYLOAD, integrations_root=tmp_path)
        result = create_integration(_BASE_PAYLOAD, integrations_root=tmp_path)
        assert result["status"] == "exists"
        # Original untouched.
        assert (tmp_path / "weather" / "script.py").read_text().startswith(
            "from boxbot_sdk.integration"
        )

    def test_invalid_name_rejected(self, tmp_path: Path):
        result = create_integration(
            {**_BASE_PAYLOAD, "name": "Anthropic"},
            integrations_root=tmp_path,
        )
        assert result["status"] == "error"
        assert not list(tmp_path.iterdir())

    def test_missing_script_rejected(self, tmp_path: Path):
        payload = {k: v for k, v in _BASE_PAYLOAD.items() if k != "script"}
        result = create_integration(payload, integrations_root=tmp_path)
        assert result["status"] == "error"
        assert not list(tmp_path.iterdir())

    def test_invalid_secret_name_rejected(self, tmp_path: Path):
        result = create_integration(
            {**_BASE_PAYLOAD, "secrets": ["lower_case"]},
            integrations_root=tmp_path,
        )
        assert result["status"] == "error"
        assert not list(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_updates_script(self, tmp_path: Path):
        create_integration(_BASE_PAYLOAD, integrations_root=tmp_path)
        result = update_integration(
            {"name": "weather", "script": "# new script\n"},
            integrations_root=tmp_path,
        )
        assert result["status"] == "ok"
        assert result["files"] == ["script.py"]
        assert (tmp_path / "weather" / "script.py").read_text() == "# new script\n"
        # Manifest untouched.
        assert "NWS_USER_AGENT" in (tmp_path / "weather" / "manifest.yaml").read_text()

    def test_updates_manifest(self, tmp_path: Path):
        create_integration(_BASE_PAYLOAD, integrations_root=tmp_path)
        new_manifest = {
            "description": "Updated description.",
            "inputs": {"lat": {"type": "float"}, "lon": {"type": "float"}},
            "secrets": ["NWS_USER_AGENT", "EXTRA_KEY"],
            "timeout": 20,
        }
        result = update_integration(
            {"name": "weather", "manifest": new_manifest},
            integrations_root=tmp_path,
        )
        assert result["status"] == "ok"
        text = (tmp_path / "weather" / "manifest.yaml").read_text()
        assert "Updated description" in text
        assert "EXTRA_KEY" in text
        assert "timeout: 20" in text

    def test_missing_returns_missing(self, tmp_path: Path):
        result = update_integration(
            {"name": "ghost", "script": "# x"},
            integrations_root=tmp_path,
        )
        assert result["status"] == "missing"

    def test_no_payload_errors(self, tmp_path: Path):
        create_integration(_BASE_PAYLOAD, integrations_root=tmp_path)
        result = update_integration(
            {"name": "weather"},
            integrations_root=tmp_path,
        )
        assert result["status"] == "error"
        assert "manifest" in result["message"] or "script" in result["message"]


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_removes_directory(self, tmp_path: Path):
        create_integration(_BASE_PAYLOAD, integrations_root=tmp_path)
        result = delete_integration("weather", integrations_root=tmp_path)
        assert result["status"] == "ok"
        assert not (tmp_path / "weather").exists()

    def test_missing_returns_missing(self, tmp_path: Path):
        result = delete_integration("ghost", integrations_root=tmp_path)
        assert result["status"] == "missing"

    def test_empty_name_errors(self, tmp_path: Path):
        result = delete_integration("", integrations_root=tmp_path)
        assert result["status"] == "error"
