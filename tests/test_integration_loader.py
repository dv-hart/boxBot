"""Tests for the integration manifest schema and loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from boxbot.integrations import loader as integ_loader
from boxbot.integrations.manifest import (
    DEFAULT_TIMEOUT,
    ManifestError,
    load_manifest_file,
    render_manifest_yaml,
    validate_manifest,
)


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------


class TestValidateManifest:
    def test_minimal_valid_manifest(self):
        m = validate_manifest(
            {"name": "weather", "description": "Get weather forecasts."}
        )
        assert m["name"] == "weather"
        assert m["inputs"] == {}
        assert m["outputs"] == {}
        assert m["secrets"] == []
        assert m["timeout"] == DEFAULT_TIMEOUT

    def test_full_manifest(self):
        m = validate_manifest(
            {
                "name": "weather",
                "description": "Get NOAA weather forecasts.",
                "inputs": {
                    "lat": {"type": "float", "required": True},
                    "lon": {"type": "float", "required": True},
                    "days": {"type": "int", "default": 5},
                },
                "outputs": {"today": {"high": "int"}},
                "secrets": ["NWS_USER_AGENT"],
                "timeout": 15,
            }
        )
        assert m["timeout"] == 15
        assert m["secrets"] == ["NWS_USER_AGENT"]
        assert m["inputs"]["lat"]["required"] is True

    def test_secrets_accepts_dict_or_string(self):
        m = validate_manifest(
            {
                "name": "x",
                "description": "y",
                "secrets": ["FOO", {"name": "BAR"}],
            }
        )
        assert m["secrets"] == ["FOO", "BAR"]

    def test_secrets_dedupe_preserves_order(self):
        m = validate_manifest(
            {
                "name": "x",
                "description": "y",
                "secrets": ["FOO", "BAR", "FOO"],
            }
        )
        assert m["secrets"] == ["FOO", "BAR"]

    @pytest.mark.parametrize(
        "name,reason",
        [
            ("Anthropic", "lowercase"),
            ("anthropic", "reserved"),
            ("claude", "reserved"),
            ("a" * 65, "≤64"),
            ("has space", "[a-z0-9_-]"),
            ("../escape", "[a-z0-9_-]"),
            ("", "non-empty"),
        ],
    )
    def test_rejects_bad_name(self, name: str, reason: str):
        with pytest.raises(ManifestError, match=reason):
            validate_manifest({"name": name, "description": "x" * 30})

    def test_rejects_overlong_description(self):
        with pytest.raises(ManifestError, match="≤1024"):
            validate_manifest({"name": "x", "description": "y" * 1025})

    def test_rejects_xml_in_description(self):
        with pytest.raises(ManifestError, match="XML"):
            validate_manifest({"name": "x", "description": "Use <evil>"})

    def test_rejects_overlong_timeout(self):
        with pytest.raises(ManifestError, match="≤300"):
            validate_manifest(
                {"name": "x", "description": "y", "timeout": 999}
            )

    def test_rejects_negative_timeout(self):
        with pytest.raises(ManifestError, match="≥1"):
            validate_manifest(
                {"name": "x", "description": "y", "timeout": 0}
            )

    def test_rejects_unknown_input_type(self):
        with pytest.raises(ManifestError, match="must be one of"):
            validate_manifest(
                {
                    "name": "x",
                    "description": "y",
                    "inputs": {"foo": {"type": "magic"}},
                }
            )

    def test_rejects_lowercase_secret_name(self):
        with pytest.raises(ManifestError, match="SCREAMING_SNAKE_CASE"):
            validate_manifest(
                {"name": "x", "description": "y", "secrets": ["lower_case"]}
            )


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------


class TestManifestYaml:
    def test_render_then_parse_round_trips(self, tmp_path: Path):
        original = validate_manifest(
            {
                "name": "weather",
                "description": "Get NOAA forecasts.",
                "inputs": {"lat": {"type": "float", "required": True}},
                "secrets": ["NWS_USER_AGENT"],
                "timeout": 15,
            }
        )
        yaml_text = render_manifest_yaml(original)
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(yaml_text)
        reparsed = load_manifest_file(manifest_path)
        assert reparsed == original

    def test_render_preserves_key_order(self):
        text = render_manifest_yaml(
            validate_manifest(
                {
                    "name": "x",
                    "description": "y",
                    "secrets": ["A"],
                    "timeout": 5,
                }
            )
        )
        # name appears before description appears before secrets appears before timeout
        positions = [text.find(k) for k in ("name:", "description:", "secrets:", "timeout:")]
        assert positions == sorted(positions)
        assert all(p >= 0 for p in positions)


# ---------------------------------------------------------------------------
# load_manifest_file forgiveness
# ---------------------------------------------------------------------------


class TestLoadManifestFile:
    def test_missing_file_returns_none(self, tmp_path: Path):
        assert load_manifest_file(tmp_path / "nope.yaml") is None

    def test_malformed_yaml_returns_none(self, tmp_path: Path, caplog):
        path = tmp_path / "manifest.yaml"
        path.write_text("name: weather\n  bad indent: oops\n   - mixed\n")
        assert load_manifest_file(path) is None
        assert any("Malformed YAML" in r.message or "Invalid" in r.message
                   for r in caplog.records)

    def test_invalid_manifest_returns_none(self, tmp_path: Path):
        path = tmp_path / "manifest.yaml"
        path.write_text("name: Anthropic\ndescription: nope\n")
        assert load_manifest_file(path) is None

    def test_non_dict_yaml_returns_none(self, tmp_path: Path):
        path = tmp_path / "manifest.yaml"
        path.write_text("- just\n- a list\n")
        assert load_manifest_file(path) is None


# ---------------------------------------------------------------------------
# Loader / discovery
# ---------------------------------------------------------------------------


def _make_integration(
    root: Path,
    name: str,
    *,
    description: str = "Test integration.",
    script: str = "from boxbot_sdk.integration import return_output\nreturn_output({'ok': True})\n",
    extra_manifest: str = "",
) -> Path:
    """Create a minimal integration directory under ``root``."""
    d = root / name
    d.mkdir(parents=True)
    (d / "manifest.yaml").write_text(
        f"name: {name}\ndescription: {description}\n{extra_manifest}"
    )
    (d / "script.py").write_text(script)
    return d


class TestDiscoverIntegrations:
    def test_empty_root(self, tmp_path: Path):
        assert integ_loader.discover_integrations(root=tmp_path) == []

    def test_nonexistent_root(self, tmp_path: Path):
        assert integ_loader.discover_integrations(root=tmp_path / "ghost") == []

    def test_finds_one_integration(self, tmp_path: Path):
        _make_integration(tmp_path, "weather")
        metas = integ_loader.discover_integrations(root=tmp_path)
        assert len(metas) == 1
        assert metas[0].name == "weather"
        assert metas[0].timeout == DEFAULT_TIMEOUT
        assert metas[0].script_path.is_file()

    def test_alphabetical_order(self, tmp_path: Path):
        _make_integration(tmp_path, "zebra")
        _make_integration(tmp_path, "alpha")
        _make_integration(tmp_path, "mike")
        names = [m.name for m in integ_loader.discover_integrations(root=tmp_path)]
        assert names == ["alpha", "mike", "zebra"]

    def test_skips_dir_without_manifest(self, tmp_path: Path):
        d = tmp_path / "no-manifest"
        d.mkdir()
        (d / "script.py").write_text("# no manifest")
        assert integ_loader.discover_integrations(root=tmp_path) == []

    def test_skips_dir_without_script(self, tmp_path: Path):
        d = tmp_path / "no-script"
        d.mkdir()
        (d / "manifest.yaml").write_text(
            "name: no-script\ndescription: Has no script.\n"
        )
        assert integ_loader.discover_integrations(root=tmp_path) == []

    def test_skips_malformed_manifest(self, tmp_path: Path, caplog):
        d = tmp_path / "bad"
        d.mkdir()
        (d / "manifest.yaml").write_text("name: Anthropic\ndescription: nope\n")
        (d / "script.py").write_text("# x")
        assert integ_loader.discover_integrations(root=tmp_path) == []

    def test_skips_when_dirname_mismatches_manifest_name(self, tmp_path: Path):
        d = tmp_path / "dirname"
        d.mkdir()
        (d / "manifest.yaml").write_text(
            "name: not-dirname\ndescription: Mismatch.\n"
        )
        (d / "script.py").write_text("# x")
        assert integ_loader.discover_integrations(root=tmp_path) == []

    def test_get_integration_returns_meta(self, tmp_path: Path):
        _make_integration(tmp_path, "weather")
        meta = integ_loader.get_integration("weather", root=tmp_path)
        assert meta is not None
        assert meta.name == "weather"

    def test_get_integration_returns_none_for_missing(self, tmp_path: Path):
        assert integ_loader.get_integration("ghost", root=tmp_path) is None
