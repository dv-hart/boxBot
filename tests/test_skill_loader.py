"""Tests for the filesystem-based skill loader.

Covers discovery, index rendering, full-body loading, sub-file loading,
and the path-escape guards on ``subpath``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from boxbot.skills.loader import (
    SkillMeta,
    discover_skills,
    get_skill_index,
    load_skill,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_skill(
    root: Path,
    name: str,
    description: str = "A test skill.",
    when_to_use: str = "When tests run.",
    body: str = "# Body\n\nSome content.\n",
    extra_files: dict[str, str] | None = None,
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    frontmatter = (
        f"---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"when_to_use: {when_to_use}\n"
        f"---\n"
    )
    skill_md.write_text(frontmatter + body, encoding="utf-8")
    for rel, content in (extra_files or {}).items():
        fpath = skill_dir / rel
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(content, encoding="utf-8")
    return skill_dir


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_empty_root_returns_empty_list(self, tmp_path: Path):
        assert discover_skills(tmp_path) == []

    def test_missing_root_returns_empty_list(self, tmp_path: Path):
        assert discover_skills(tmp_path / "does-not-exist") == []

    def test_discovers_single_skill(self, tmp_path: Path):
        _write_skill(tmp_path, "alpha")
        skills = discover_skills(tmp_path)
        assert len(skills) == 1
        meta = skills[0]
        assert isinstance(meta, SkillMeta)
        assert meta.name == "alpha"
        assert meta.description == "A test skill."
        assert meta.when_to_use == "When tests run."
        assert meta.root_path.is_dir()
        assert meta.root_path.name == "alpha"

    def test_discovers_multiple_sorted_by_name(self, tmp_path: Path):
        _write_skill(tmp_path, "charlie")
        _write_skill(tmp_path, "alpha")
        _write_skill(tmp_path, "bravo")
        names = [s.name for s in discover_skills(tmp_path)]
        assert names == ["alpha", "bravo", "charlie"]

    def test_skips_directory_without_skill_md(self, tmp_path: Path):
        (tmp_path / "not-a-skill").mkdir()
        _write_skill(tmp_path, "real")
        names = [s.name for s in discover_skills(tmp_path)]
        assert names == ["real"]

    def test_skips_entry_with_missing_name_frontmatter(self, tmp_path: Path):
        bad = tmp_path / "broken"
        bad.mkdir()
        (bad / "SKILL.md").write_text(
            "---\ndescription: has no name\n---\nbody\n",
            encoding="utf-8",
        )
        _write_skill(tmp_path, "good")
        names = [s.name for s in discover_skills(tmp_path)]
        assert names == ["good"]

    def test_skips_plain_files_at_root(self, tmp_path: Path):
        (tmp_path / "stray.txt").write_text("ignore me", encoding="utf-8")
        _write_skill(tmp_path, "real")
        names = [s.name for s in discover_skills(tmp_path)]
        assert names == ["real"]

    def test_does_not_follow_symlinks(self, tmp_path: Path):
        # Build a real skill outside the root, then symlink it in.
        outside_root = tmp_path / "outside"
        outside_root.mkdir()
        _write_skill(outside_root, "sneaky")
        inside_root = tmp_path / "inside"
        inside_root.mkdir()
        try:
            (inside_root / "sneaky").symlink_to(outside_root / "sneaky")
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported on this platform")
        assert discover_skills(inside_root) == []


# ---------------------------------------------------------------------------
# Index rendering
# ---------------------------------------------------------------------------


class TestIndex:
    def test_empty_when_no_skills(self, tmp_path: Path):
        assert get_skill_index(tmp_path) == ""

    def test_includes_header_and_entries(self, tmp_path: Path):
        _write_skill(
            tmp_path,
            "hal-audio",
            when_to_use=(
                "Scripts that need to play sounds, capture raw audio, or "
                "control the LED ring programmatically."
            ),
        )
        _write_skill(
            tmp_path,
            "hal-sandbox-ref",
            when_to_use="Using the boxbot_sdk from sandboxed scripts.",
        )
        out = get_skill_index(tmp_path)
        assert "## Available skills" in out
        assert "- hal-audio: Scripts that need to play sounds" in out
        assert "- hal-sandbox-ref: Using the boxbot_sdk from sandboxed scripts." in out
        assert "load_skill" in out

    def test_falls_back_to_description_if_no_when_to_use(self, tmp_path: Path):
        skill_dir = tmp_path / "plain"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: plain\ndescription: Just a description.\n---\nbody\n",
            encoding="utf-8",
        )
        out = get_skill_index(tmp_path)
        assert "- plain: Just a description." in out


# ---------------------------------------------------------------------------
# load_skill — body + sub-files
# ---------------------------------------------------------------------------


class TestLoadSkill:
    def test_returns_body_without_frontmatter(self, tmp_path: Path):
        _write_skill(
            tmp_path,
            "alpha",
            body="# Alpha\n\nHello body.\n",
        )
        body = load_skill("alpha", root=tmp_path)
        assert body.startswith("# Alpha")
        assert "description:" not in body  # frontmatter stripped
        assert "name: alpha" not in body

    def test_returns_subfile_content(self, tmp_path: Path):
        _write_skill(
            tmp_path,
            "alpha",
            extra_files={"examples/play.py": "print('hi')\n"},
        )
        contents = load_skill("alpha", subpath="examples/play.py", root=tmp_path)
        assert contents == "print('hi')\n"

    def test_unknown_skill_raises(self, tmp_path: Path):
        _write_skill(tmp_path, "alpha")
        with pytest.raises(ValueError, match="Unknown skill"):
            load_skill("nope", root=tmp_path)

    def test_missing_subfile_raises(self, tmp_path: Path):
        _write_skill(tmp_path, "alpha")
        with pytest.raises(ValueError, match="not found"):
            load_skill("alpha", subpath="does/not/exist.py", root=tmp_path)


# ---------------------------------------------------------------------------
# Path-escape safety
# ---------------------------------------------------------------------------


class TestSubpathSafety:
    def test_rejects_absolute_path(self, tmp_path: Path):
        _write_skill(tmp_path, "alpha")
        with pytest.raises(ValueError, match="relative"):
            load_skill("alpha", subpath="/etc/passwd", root=tmp_path)

    def test_rejects_parent_traversal(self, tmp_path: Path):
        _write_skill(tmp_path, "alpha")
        _write_skill(tmp_path, "beta")
        # This would resolve to tmp_path/beta/SKILL.md — still within the
        # skills root, but NOT within the alpha skill root. Must reject.
        with pytest.raises(ValueError, match="'\\.\\.'|escapes"):
            load_skill("alpha", subpath="../beta/SKILL.md", root=tmp_path)

    def test_rejects_hidden_traversal(self, tmp_path: Path):
        _write_skill(tmp_path, "alpha")
        with pytest.raises(ValueError, match="'\\.\\.'|escapes"):
            load_skill(
                "alpha",
                subpath="subdir/../../etc/passwd",
                root=tmp_path,
            )

    def test_rejects_symlink_subfile(self, tmp_path: Path):
        skill_dir = _write_skill(tmp_path, "alpha")
        target = tmp_path / "secret.txt"
        target.write_text("TOP SECRET", encoding="utf-8")
        try:
            (skill_dir / "link.txt").symlink_to(target)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported on this platform")
        with pytest.raises(ValueError):
            load_skill("alpha", subpath="link.txt", root=tmp_path)

    def test_rejects_invalid_skill_name_with_separator(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Invalid skill name"):
            load_skill("../outside", root=tmp_path)

    def test_rejects_empty_subpath(self, tmp_path: Path):
        _write_skill(tmp_path, "alpha")
        with pytest.raises(ValueError):
            load_skill("alpha", subpath="", root=tmp_path)


# ---------------------------------------------------------------------------
# Seed skill sanity check — uses the real default root
# ---------------------------------------------------------------------------


class TestSeedSkill:
    def test_seed_skill_is_discoverable(self):
        skills = discover_skills()
        names = {s.name for s in skills}
        assert "hal-sandbox-ref" in names, (
            f"Expected the seed skill in the default skills root; found {names}"
        )

    def test_seed_skill_body_loads(self):
        body = load_skill("hal-sandbox-ref")
        assert "boxbot_sdk" in body
        assert "---" not in body.splitlines()[0]  # frontmatter stripped
