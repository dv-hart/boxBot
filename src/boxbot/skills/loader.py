"""Filesystem-based skill loader.

Lightweight discovery layer for the top-level ``skills/`` directory
(sibling to ``src/``). Each skill is a folder with a ``SKILL.md`` file
containing YAML frontmatter + markdown body. The loader is used for
progressive disclosure: a short index is injected into the system
prompt, and the agent calls the ``load_skill`` tool to read the full
body of a skill only when the task calls for it.

This is **not** Anthropic's ``/v1/skills`` API. That API is bound to
Anthropic's code-execution container and is useless for boxBot's local
hardware tools. We build our own.

Public API:
    SkillMeta          — frontmatter-only record of a skill
    discover_skills()  — scan the repo-level ``skills/`` dir
    get_skill_index()  — markdown block for system-prompt injection
    load_skill()       — read full body or a sub-file with path-escape guard
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml  # type: ignore[import-untyped]
except Exception:  # pragma: no cover — PyYAML is a project dependency
    yaml = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# Repo-root ``skills/`` directory. loader.py lives at
# ``<repo>/src/boxbot/skills/loader.py``, so four parents up is the repo root.
_DEFAULT_SKILLS_ROOT: Path = (
    Path(__file__).resolve().parent.parent.parent.parent / "skills"
)

_SKILL_FILE = "SKILL.md"


@dataclass(frozen=True)
class SkillMeta:
    """Frontmatter-only record of a skill.

    ``root_path`` is the absolute path to the skill's directory, not to
    the ``SKILL.md`` file. Sub-files are resolved relative to this.
    """

    name: str
    description: str
    when_to_use: str
    root_path: Path


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a ``SKILL.md`` into (frontmatter_dict, body).

    Expects the file to start with ``---\n`` followed by YAML, then a
    closing ``---\n``. If the document has no frontmatter, returns an
    empty dict and the full text as the body.
    """
    if not text.startswith("---"):
        return {}, text

    # Find the closing delimiter on its own line.
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        return {}, text

    end_idx: int | None = None
    for i in range(1, len(lines)):
        if lines[i].rstrip("\r\n") == "---":
            end_idx = i
            break
    if end_idx is None:
        return {}, text

    fm_text = "".join(lines[1:end_idx])
    body = "".join(lines[end_idx + 1 :])

    meta: dict[str, str] = {}
    if yaml is not None:
        try:
            parsed = yaml.safe_load(fm_text) or {}
            if isinstance(parsed, dict):
                # Coerce all values to strings for our minimal schema.
                meta = {str(k): ("" if v is None else str(v)) for k, v in parsed.items()}
        except Exception as exc:
            logger.warning("Failed to parse YAML frontmatter: %s", exc)
            meta = _manual_frontmatter_parse(fm_text)
    else:
        meta = _manual_frontmatter_parse(fm_text)

    return meta, body


def _manual_frontmatter_parse(fm_text: str) -> dict[str, str]:
    """Minimal ``key: value`` YAML fallback for when PyYAML is unavailable.

    Only handles flat scalar keys; anything more complex is dropped.
    """
    out: dict[str, str] = {}
    for raw in fm_text.splitlines():
        line = raw.rstrip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        # Strip matching quotes if present.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _resolve_root(root: Path | None) -> Path:
    return (root if root is not None else _DEFAULT_SKILLS_ROOT).resolve()


def discover_skills(root: Path | None = None) -> list[SkillMeta]:
    """Scan the skills root directory and return frontmatter records.

    - Never raises. Logs warnings for malformed skills and skips them.
    - Returns an empty list if ``root`` does not exist.
    - Does not follow symlinks (security: skills are read-only, local-only).
    - Skills are sorted by name for stable ordering.
    """
    resolved = _resolve_root(root)
    if not resolved.exists() or not resolved.is_dir():
        return []

    skills: list[SkillMeta] = []
    try:
        entries = sorted(resolved.iterdir(), key=lambda p: p.name)
    except OSError as exc:
        logger.warning("Cannot list skills root %s: %s", resolved, exc)
        return []

    for entry in entries:
        # Security: do not follow symlinks — local skills only.
        if entry.is_symlink():
            logger.warning("Skipping symlinked skill entry: %s", entry.name)
            continue
        if not entry.is_dir():
            continue
        skill_md = entry / _SKILL_FILE
        if not skill_md.is_file():
            continue
        try:
            text = skill_md.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Cannot read %s: %s", skill_md, exc)
            continue

        meta, _body = _split_frontmatter(text)
        name = meta.get("name", "").strip()
        description = meta.get("description", "").strip()
        when_to_use = meta.get("when_to_use", "").strip()

        if not name:
            logger.warning(
                "Skill at %s has no 'name' in frontmatter — skipping", entry
            )
            continue
        # Prefer directory name as authoritative if they disagree.
        if name != entry.name:
            logger.debug(
                "Skill frontmatter name %r differs from directory %r; using directory name",
                name,
                entry.name,
            )
            name = entry.name

        skills.append(
            SkillMeta(
                name=name,
                description=description,
                when_to_use=when_to_use,
                root_path=entry.resolve(),
            )
        )

    return skills


# ---------------------------------------------------------------------------
# Index rendering
# ---------------------------------------------------------------------------


_INDEX_HEADER = """## Available skills

You have on-demand access to skills — topic-specific guidance loaded via the
load_skill tool. Read the full body only when the task calls for it.

"""

_INDEX_FOOTER = """
To read a full skill, call load_skill(name="<skill-name>").
To read a sub-file of a skill, call load_skill(name="<skill-name>", subpath="examples/<file>.py").
"""


def get_skill_index(root: Path | None = None) -> str:
    """Return a markdown-formatted skill index for system-prompt injection.

    - Returns ``""`` if no skills exist or discovery fails.
    - Never raises.
    """
    try:
        skills = discover_skills(root)
    except Exception as exc:  # defensive — discover_skills already swallows
        logger.warning("Skill discovery failed: %s", exc)
        return ""

    if not skills:
        return ""

    lines = [_INDEX_HEADER.rstrip("\n")]
    lines.append("")
    for meta in skills:
        # Prefer when_to_use; fall back to description.
        blurb = meta.when_to_use or meta.description or "(no description)"
        # Collapse embedded newlines so each skill is one list item.
        blurb = " ".join(blurb.split())
        lines.append(f"- {meta.name}: {blurb}")
    lines.append(_INDEX_FOOTER.rstrip("\n"))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def _find_skill(name: str, root: Path | None) -> SkillMeta:
    """Locate a skill by name. Raises ValueError if not found."""
    if not name or not isinstance(name, str):
        raise ValueError("Skill name must be a non-empty string.")
    # Reject any path-ish characters in the skill name itself.
    if "/" in name or "\\" in name or name in ("", ".", ".."):
        raise ValueError(f"Invalid skill name: {name!r}")
    for meta in discover_skills(root):
        if meta.name == name:
            return meta
    raise ValueError(f"Unknown skill: {name!r}")


def _safe_subpath(skill_root: Path, subpath: str) -> Path:
    """Resolve ``subpath`` under ``skill_root`` and refuse any escape.

    Rejects absolute paths and any resolved location outside the skill
    root. Raises ``ValueError`` on violation.
    """
    if not subpath or not isinstance(subpath, str):
        raise ValueError("subpath must be a non-empty string when provided.")
    candidate = Path(subpath)
    if candidate.is_absolute():
        raise ValueError(f"subpath must be relative: {subpath!r}")
    # Explicitly reject parent-traversal segments regardless of resolution.
    if any(part == ".." for part in candidate.parts):
        raise ValueError(f"subpath may not contain '..': {subpath!r}")

    skill_root_resolved = skill_root.resolve()
    full = (skill_root_resolved / candidate).resolve()
    if not full.is_relative_to(skill_root_resolved):
        raise ValueError(
            f"subpath escapes skill directory: {subpath!r}"
        )
    # Refuse symlinks for defense-in-depth.
    if full.is_symlink():
        raise ValueError(f"subpath is a symlink: {subpath!r}")
    return full


def load_skill(
    name: str,
    subpath: str | None = None,
    root: Path | None = None,
) -> str:
    """Read a skill's full body, or a sub-file within the skill directory.

    Args:
        name: Skill name as listed by ``discover_skills`` / the index.
        subpath: Optional path to a file inside the skill directory. Must
            be relative and must not escape the skill root. If ``None``,
            returns the ``SKILL.md`` body with YAML frontmatter stripped.
        root: Override for the skills root directory (tests).

    Returns:
        UTF-8 text content.

    Raises:
        ValueError: unknown skill, invalid subpath, or path escape attempt.
    """
    meta = _find_skill(name, root)

    if subpath is None:
        skill_md = meta.root_path / _SKILL_FILE
        try:
            text = skill_md.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"Cannot read skill body for {name!r}: {exc}") from exc
        _, body = _split_frontmatter(text)
        return body.lstrip("\n")

    target = _safe_subpath(meta.root_path, subpath)
    if not target.exists():
        raise ValueError(f"Sub-file not found: {subpath!r}")
    if not target.is_file():
        raise ValueError(f"Sub-file is not a regular file: {subpath!r}")
    try:
        return target.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read sub-file {subpath!r}: {exc}") from exc


__all__ = [
    "SkillMeta",
    "discover_skills",
    "get_skill_index",
    "load_skill",
]
