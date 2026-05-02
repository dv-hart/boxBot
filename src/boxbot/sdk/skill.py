"""Skill builder — create new agent skills declaratively.

A skill is **structured prompt data**: a SKILL.md file (YAML frontmatter
+ markdown body) you, the agent, will read on demand later, optionally
bundled with helper scripts under ``scripts/`` and Level 3 sub-docs.

Skills are not callable functions. They have no parameters, no env
vars, and no schedule. If your use case needs any of those — credential
storage, scheduled fetches, multi-consumer data — you want an
**integration** (``src/boxbot/integrations/``), not a skill. See
``skills/skill_authoring/SKILL.md`` for the full authoring guide.

Usage::

    import boxbot_sdk as bb

    s = bb.skill.create("weather")
    s.description = (
        "Get NOAA weather forecasts for the configured location. "
        "Use when the user asks about weather, temperature, or conditions."
    )
    s.body = '''
    # Weather

    Use ``bb.weather.forecast(days=N)`` for an N-day forecast.
    For hourly precipitation, see HOURLY.md.
    '''
    s.add_resource("HOURLY.md", "# Hourly forecast\\n…")
    s.add_script("nws_raw.py", "import requests\\n…")
    s.save()

The main process writes ``skills/<name>/SKILL.md``, optional resources
at the skill root, and bundled scripts under ``scripts/`` (with an
auto-generated ``__init__.py`` so ``from skills.<name>.scripts import
<file>`` resolves inside ``execute_script``). Files are owned
``boxbot:boxbot`` mode 0644 — the sandbox can read but not modify them
after save.

The loader picks the new skill up on the next discovery scan.
"""

from __future__ import annotations

from typing import Any

from . import _transport, _validators as v


def create(name: str) -> SkillBuilder:
    """Create a new skill builder.

    Args:
        name: Skill name. ≤64 chars, lowercase, ``[a-z0-9_-]+``,
            not ``anthropic`` or ``claude``.

    Returns:
        A new SkillBuilder instance.
    """
    v.validate_skill_name(name)
    return SkillBuilder(name)


class SkillBuilder:
    """Builder for defining skills declaratively.

    Skills are documentation, not callable functions. Set ``description``
    (frontmatter) and ``body`` (markdown), optionally bundle scripts and
    Level 3 sub-docs, then call ``save()``.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._description: str | None = None
        self._body: str | None = None
        self._scripts: list[dict[str, str]] = []
        self._resources: list[dict[str, str]] = []

    @property
    def name(self) -> str:
        """Skill name."""
        return self._name

    @property
    def description(self) -> str | None:
        """Skill description (frontmatter)."""
        return self._description

    @description.setter
    def description(self, value: str) -> None:
        v.validate_skill_description(value)
        self._description = value

    @property
    def body(self) -> str | None:
        """SKILL.md body (markdown)."""
        return self._body

    @body.setter
    def body(self, value: str) -> None:
        v.require_str(value, "body")
        self._body = value

    def add_script(self, filename: str, content: str) -> None:
        """Bundle a Python helper script under ``scripts/<filename>``.

        Bundled scripts are importable from ``execute_script`` via
        ``from skills.<name>.scripts import <module>``. The writer also
        stamps a ``scripts/__init__.py`` so the import resolves; you do
        not need to add it yourself.

        Subprocess execution is not available (seccomp blocks
        ``execve``/``fork``); always import.

        Args:
            filename: Script filename, must end in ``.py`` and be a
                bare basename (no slashes, no traversal).
            content: Python source code.
        """
        v.require_str(filename, "filename")
        v.require_str(content, "content")
        if not filename.endswith(".py"):
            raise ValueError(f"script filename must end in '.py', got '{filename}'")
        if "/" in filename or "\\" in filename or filename.startswith("."):
            raise ValueError(f"script filename must be a bare basename, got '{filename}'")
        self._scripts.append({"filename": filename, "content": content})

    def add_resource(self, filename: str, content: str) -> None:
        """Bundle a Level 3 sub-doc at the skill root (e.g. ``REFERENCE.md``).

        Use for guidance that's too long for the main SKILL.md body.
        Reference it from the body by filename so future-you knows it
        exists without paying for it upfront.

        Args:
            filename: Filename, bare basename, conventionally ``.md``.
            content: File content.
        """
        v.require_str(filename, "filename")
        v.require_str(content, "content")
        if "/" in filename or "\\" in filename or filename.startswith("."):
            raise ValueError(f"resource filename must be a bare basename, got '{filename}'")
        if filename == "SKILL.md":
            raise ValueError("'SKILL.md' is reserved — set s.body instead")
        self._resources.append({"filename": filename, "content": content})

    def save(self) -> None:
        """Save the skill. The loader picks it up on next discovery scan.

        Refuses if a skill with the same name already exists — call
        ``skill.delete`` first (when implemented), or pick a different
        name. Never silently overwrites a community skill.
        """
        if self._description is None:
            raise ValueError("Skill description is required — set s.description")
        if self._body is None:
            raise ValueError("Skill body is required — set s.body")

        payload: dict[str, Any] = {
            "name": self._name,
            "description": self._description,
            "body": self._body,
        }
        if self._scripts:
            payload["scripts"] = self._scripts
        if self._resources:
            payload["resources"] = self._resources

        _transport.emit_action("skill.save", payload)
