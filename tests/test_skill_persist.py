"""Integration tests for the bb.skill.save → on-disk → discover round-trip.

These exercise the writer in :mod:`boxbot.skills.persist` directly,
plus the dispatcher routing in ``_sandbox_actions``. Each test points
the writer at a temp directory so they don't touch the real
``skills/`` tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from boxbot.skills import loader as skills_loader
from boxbot.skills.persist import write_skill
from boxbot.tools._sandbox_actions import ActionContext, process_action


# ---------------------------------------------------------------------------
# Direct writer — payload → files
# ---------------------------------------------------------------------------


class TestWriteSkill:
    def test_minimal_skill_writes_skill_md(self, tmp_path: Path):
        result = write_skill(
            {
                "name": "weather",
                "description": "Get NOAA forecasts. Use when asked about weather.",
                "body": "# Weather\n\nUse `bb.weather.forecast()`.\n",
            },
            skills_root=tmp_path,
        )
        assert result["status"] == "ok"
        assert result["name"] == "weather"
        assert result["files"] == ["SKILL.md"]

        skill_md = tmp_path / "weather" / "SKILL.md"
        text = skill_md.read_text()
        assert text.startswith("---\n")
        assert "name: weather" in text
        assert "description: " in text
        assert "# Weather" in text

    def test_skill_round_trips_through_loader(self, tmp_path: Path):
        write_skill(
            {
                "name": "weather",
                "description": "Get NOAA forecasts. Use when asked about weather.",
                "body": "# Weather\n\nUse the integration.\n",
            },
            skills_root=tmp_path,
        )
        skills = skills_loader.discover_skills(root=tmp_path)
        assert len(skills) == 1
        assert skills[0].name == "weather"
        assert "NOAA" in skills[0].description

    def test_scripts_get_init_py_sibling(self, tmp_path: Path):
        result = write_skill(
            {
                "name": "with-script",
                "description": "Bundles a helper.",
                "body": "# Skill\n\nSee scripts/helper.py.\n",
                "scripts": [{"filename": "helper.py", "content": "def go():\n    return 42\n"}],
            },
            skills_root=tmp_path,
        )
        assert result["status"] == "ok"
        skill_dir = tmp_path / "with-script"
        assert (skill_dir / "scripts" / "__init__.py").is_file()
        assert (skill_dir / "scripts" / "helper.py").is_file()
        assert "def go" in (skill_dir / "scripts" / "helper.py").read_text()
        # Auto-stamped __init__.py mentions its purpose.
        init_text = (skill_dir / "scripts" / "__init__.py").read_text()
        assert "Auto-generated" in init_text

    def test_resources_land_at_skill_root(self, tmp_path: Path):
        write_skill(
            {
                "name": "with-ref",
                "description": "Skill with a Level 3 sub-doc.",
                "body": "# Skill\n\nSee REFERENCE.md.\n",
                "resources": [{"filename": "REFERENCE.md", "content": "# Reference\n"}],
            },
            skills_root=tmp_path,
        )
        skill_dir = tmp_path / "with-ref"
        assert (skill_dir / "REFERENCE.md").read_text() == "# Reference\n"
        # Resources do not create scripts/.
        assert not (skill_dir / "scripts").exists()

    def test_existing_skill_returns_exists_status(self, tmp_path: Path):
        (tmp_path / "weather").mkdir()
        (tmp_path / "weather" / "SKILL.md").write_text("---\nname: weather\n---\n")

        result = write_skill(
            {
                "name": "weather",
                "description": "Trying to clobber.",
                "body": "# Replacement",
            },
            skills_root=tmp_path,
        )
        assert result["status"] == "exists"
        assert result["name"] == "weather"
        # Original file untouched.
        assert (tmp_path / "weather" / "SKILL.md").read_text() == "---\nname: weather\n---\n"


# ---------------------------------------------------------------------------
# Validation — defense-in-depth on the trusted side
# ---------------------------------------------------------------------------


class TestWriterValidation:
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
    def test_rejects_bad_name(self, tmp_path: Path, name: str, reason: str):
        result = write_skill(
            {"name": name, "description": "x" * 30, "body": "# x"},
            skills_root=tmp_path,
        )
        assert result["status"] == "error"
        assert reason in result["message"]
        assert not list(tmp_path.iterdir())  # nothing written

    def test_rejects_overlong_description(self, tmp_path: Path):
        result = write_skill(
            {"name": "long", "description": "x" * 1025, "body": "# x"},
            skills_root=tmp_path,
        )
        assert result["status"] == "error"
        assert "1024" in result["message"]
        assert not list(tmp_path.iterdir())

    def test_rejects_xml_in_description(self, tmp_path: Path):
        result = write_skill(
            {"name": "xml", "description": "Use this <tag>", "body": "# x"},
            skills_root=tmp_path,
        )
        assert result["status"] == "error"
        assert "XML" in result["message"]

    def test_rejects_empty_body(self, tmp_path: Path):
        result = write_skill(
            {"name": "empty-body", "description": "x" * 30, "body": ""},
            skills_root=tmp_path,
        )
        assert result["status"] == "error"
        assert "body" in result["message"]

    def test_rejects_skill_md_resource(self, tmp_path: Path):
        result = write_skill(
            {
                "name": "skill-md-res",
                "description": "x" * 30,
                "body": "# x",
                "resources": [{"filename": "SKILL.md", "content": "x"}],
            },
            skills_root=tmp_path,
        )
        assert result["status"] == "error"
        assert "reserved" in result["message"]
        # Directory must not have been created.
        assert not (tmp_path / "skill-md-res").exists()

    def test_rejects_path_traversal_in_script_filename(self, tmp_path: Path):
        result = write_skill(
            {
                "name": "trav",
                "description": "x" * 30,
                "body": "# x",
                "scripts": [{"filename": "../escape.py", "content": "x"}],
            },
            skills_root=tmp_path,
        )
        assert result["status"] == "error"
        assert "basename" in result["message"]
        assert not (tmp_path / "trav").exists()


# ---------------------------------------------------------------------------
# Dispatcher routing — sandbox action → write
# ---------------------------------------------------------------------------


class TestDispatcherRouting:
    @pytest.mark.asyncio
    async def test_skill_save_routes_to_handler(
        self, tmp_path: Path, monkeypatch
    ):
        # Point the writer at our temp tree.
        from boxbot.skills import persist as persist_mod

        monkeypatch.setattr(persist_mod, "_DEFAULT_SKILLS_ROOT", tmp_path)

        ctx = ActionContext()
        result = await process_action(
            {
                "_sdk": "skill.save",
                "name": "from-dispatch",
                "description": "Saved through the dispatcher.",
                "body": "# Hello\n",
            },
            ctx,
        )
        assert result["status"] == "ok"
        assert (tmp_path / "from-dispatch" / "SKILL.md").is_file()
        assert ctx.action_log[-1]["action"] == "skill.save"

    @pytest.mark.asyncio
    async def test_unknown_skill_subaction_returns_error(self):
        ctx = ActionContext()
        result = await process_action(
            {"_sdk": "skill.delete", "name": "x"}, ctx
        )
        assert result["status"] == "error"
        assert "skill.delete" in result["message"]
