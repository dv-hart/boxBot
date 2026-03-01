"""Skill builder — create new agent skills declaratively.

Skills are modular capabilities the agent can invoke. The skill's execution
logic is a Python script that runs in the sandbox.

Usage:
    from boxbot_sdk import skill

    s = skill.create("check_gmail")
    s.description = "Check for unread emails and return summaries"
    s.add_parameter("max_results", type="integer", default=10)
    s.set_script('''
    import imaplib
    # ...
    ''')
    s.add_env_var("GMAIL_USER", secret=True)
    s.save()
"""

from __future__ import annotations

from typing import Any

from . import _transport, _validators as v


def create(name: str) -> SkillBuilder:
    """Create a new skill builder.

    Args:
        name: Unique skill name (alphanumeric, underscores, hyphens).

    Returns:
        A new SkillBuilder instance.
    """
    v.validate_name(name, "skill name")
    return SkillBuilder(name)


class SkillBuilder:
    """Builder for defining skills declaratively."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._description: str | None = None
        self._parameters: list[dict[str, Any]] = []
        self._script: str | None = None
        self._env_vars: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        """Skill name."""
        return self._name

    @property
    def description(self) -> str | None:
        """Skill description."""
        return self._description

    @description.setter
    def description(self, value: str) -> None:
        v.require_str(value, "description")
        self._description = value

    def add_parameter(self, name: str, *, type: str = "string",
                      default: Any = None,
                      required: bool = False,
                      description: str | None = None) -> None:
        """Add a parameter to the skill.

        Args:
            name: Parameter name.
            type: Parameter type — string, integer, float, boolean.
            default: Default value.
            required: Whether the parameter is required.
            description: Parameter description.
        """
        v.require_str(name, "parameter name")
        v.validate_one_of(type, "type", v.VALID_SKILL_PARAM_TYPES)

        param: dict[str, Any] = {"name": name, "type": type}
        if default is not None:
            param["default"] = default
        if required:
            param["required"] = True
        if description is not None:
            param["description"] = v.require_str(description, "description")
        self._parameters.append(param)

    def set_script(self, script: str) -> None:
        """Set the Python script that runs when the skill is invoked.

        The script runs in the sandbox with access to the SDK and any
        declared environment variables.

        Args:
            script: Python source code.
        """
        v.require_str(script, "script")
        self._script = script

    def add_env_var(self, name: str, *, secret: bool = False) -> None:
        """Declare an environment variable the skill needs at runtime.

        Args:
            name: Environment variable name.
            secret: If True, the value is stored in boxbot_sdk.secrets.
        """
        v.require_str(name, "env var name")
        env_var: dict[str, Any] = {"name": name}
        if secret:
            env_var["secret"] = True
        self._env_vars.append(env_var)

    def save(self) -> None:
        """Save the skill. Auto-activates since skill logic is sandboxed."""
        if self._description is None:
            raise ValueError("Skill description is required — set s.description")
        if self._script is None:
            raise ValueError("Skill script is required — call s.set_script()")

        payload: dict[str, Any] = {
            "name": self._name,
            "description": self._description,
            "script": self._script,
        }
        if self._parameters:
            payload["parameters"] = self._parameters
        if self._env_vars:
            payload["env_vars"] = self._env_vars

        _transport.emit_action("skill.save", payload)
