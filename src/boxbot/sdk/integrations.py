"""bb.integrations — call, list, create, update, delete, and inspect integrations.

An integration is a sandbox-runnable data pipe — manifest plus
script — that other code can call to fetch data from external
services or compute outputs from declared inputs. Unlike skills,
integrations are stateful (credentials, OAuth tokens, last-fetched
caches if they want to maintain them) and are designed to be called
by many consumers (the agent, displays, scheduled briefings).

Pipe model: integrations don't run on their own. **Consumers decide
when to call them.** There is no schedule in the manifest; if you
want recurring fetches, register a trigger that calls the integration.

The full lifecycle::

    import boxbot_sdk as bb

    # Discover what's available
    bb.integrations.list()
    # → [{"name": "weather", "description": "...", "inputs": {...}, "outputs": {...}}, ...]

    # Call one
    forecast = bb.integrations.get("weather", lat=45.5, lon=-122.7, days=5)

    # Author a new one
    i = bb.integrations.create("solar")
    i.description = "Solar production forecast for the household array."
    i.add_input("date", type="string", required=True)
    i.add_output("kwh", type="float")
    i.add_secret("SOLAR_API_KEY")
    i.script = "from boxbot_sdk.integration import inputs, return_output\\n…"
    i.timeout = 20
    i.save()

    # Debug a misbehaving one
    bb.integrations.logs("weather", limit=5)
    # → [{"started_at": …, "status": "error", "error": "401 Unauthorized"}, …]

    # Update or remove
    bb.integrations.update("solar", script="…new script…")
    bb.integrations.delete("solar")

For the full authoring guide, load the ``integrations_authoring`` skill.
"""

from __future__ import annotations

from typing import Any

from . import _transport, _validators as v


# ---------------------------------------------------------------------------
# Read-side
# ---------------------------------------------------------------------------


def list() -> Any:  # noqa: A001 — shadows the builtin intentionally; this is an SDK module
    """List all registered integrations.

    Returns the result the dispatcher sends back: typically a list of
    ``{name, description, inputs, outputs, timeout, secrets}`` dicts.
    """
    return _transport.request("integrations.list", {})


def get(name: str, **inputs: Any) -> Any:
    """Run integration ``name`` with the given inputs and return its output.

    Consumers control freshness: every call spawns a sandbox subprocess
    and runs the integration's script. If you want caching, cache on
    your end. Returns the dispatcher response — typically
    ``{"status": "ok", "output": <integration's return value>}`` or
    ``{"status": "error" | "timeout", "error": "..."}``.
    """
    v.require_str(name, "name")
    return _transport.request("integrations.get", {"name": name, "inputs": inputs})


def logs(name: str, *, limit: int = 20) -> Any:
    """Return the most recent runs of integration ``name``, newest first.

    Each row carries status, timing, inputs, outputs (or error). Useful
    for self-debugging — five consecutive auth errors usually mean a
    secret needs refreshing.
    """
    v.require_str(name, "name")
    v.require_int(limit, "limit", min_val=1, max_val=200)
    return _transport.request(
        "integrations.logs", {"name": name, "limit": limit}
    )


# ---------------------------------------------------------------------------
# Write-side: create / update / delete
# ---------------------------------------------------------------------------


def create(name: str) -> IntegrationBuilder:
    """Start building a new integration.

    Args:
        name: Integration name. ≤64 chars, lowercase, ``[a-z0-9_-]+``,
            not ``anthropic`` / ``claude``.

    Returns:
        A new :class:`IntegrationBuilder`. Set description / inputs /
        outputs / secrets / script / timeout, then call ``save()``.
    """
    # Light sanity check up here so obvious bad names error before any
    # builder mutation. Strict re-validation happens main-side.
    v.require_str(name, "name")
    if name != name.lower():
        raise ValueError(f"integration name must be lowercase, got '{name}'")
    return IntegrationBuilder(name)


def update(name: str, *, manifest: dict[str, Any] | None = None,
           script: str | None = None) -> Any:
    """Replace the manifest and/or script of an existing integration.

    Errors with ``status: "missing"`` if no integration is registered
    under that name — call :func:`create` instead.

    Either ``manifest`` (a dict matching the manifest schema, minus
    ``name`` which is implicit) or ``script`` (a string) must be provided.
    """
    v.require_str(name, "name")
    if manifest is None and script is None:
        raise ValueError("update requires manifest=… and/or script=…")
    payload: dict[str, Any] = {"name": name}
    if manifest is not None:
        payload["manifest"] = v.require_dict(manifest, "manifest")
    if script is not None:
        payload["script"] = v.require_str(script, "script")
    return _transport.request("integrations.update", payload)


def delete(name: str) -> Any:
    """Remove an integration. Errors with ``status: "missing"`` if absent."""
    v.require_str(name, "name")
    return _transport.request("integrations.delete", {"name": name})


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class IntegrationBuilder:
    """Builder for declaring an integration's manifest + script.

    Construct via :func:`create`. ``description`` and ``script`` are
    required; everything else is optional. ``save()`` emits the
    structured action that the main process turns into files on disk.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._description: str | None = None
        self._inputs: dict[str, dict[str, Any]] = {}
        self._outputs: dict[str, Any] = {}
        self._secrets: list[str] = []
        self._script: str | None = None
        self._timeout: int | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str | None:
        return self._description

    @description.setter
    def description(self, value: str) -> None:
        v.require_str(value, "description")
        self._description = value

    @property
    def script(self) -> str | None:
        return self._script

    @script.setter
    def script(self, value: str) -> None:
        v.require_str(value, "script")
        self._script = value

    @property
    def timeout(self) -> int | None:
        return self._timeout

    @timeout.setter
    def timeout(self, value: int) -> None:
        v.require_int(value, "timeout", min_val=1, max_val=300)
        self._timeout = value

    def add_input(
        self,
        name: str,
        *,
        type: str = "string",  # noqa: A002 — matches manifest field
        required: bool = False,
        default: Any = None,
        description: str | None = None,
    ) -> None:
        """Declare an input the script accepts.

        v1: input shape is descriptive only — the runner applies
        ``default`` if the caller omits the input, and rejects calls
        missing a ``required`` field. No type coercion.
        """
        v.require_str(name, "input name")
        spec: dict[str, Any] = {"type": v.require_str(type, "type")}
        if required:
            spec["required"] = True
        if default is not None:
            spec["default"] = default
        if description is not None:
            spec["description"] = v.require_str(description, "description")
        self._inputs[name] = spec

    def add_output(
        self,
        name: str,
        *,
        type: str = "string",  # noqa: A002
        description: str | None = None,
    ) -> None:
        """Declare an output field the script returns. v1: descriptive only."""
        v.require_str(name, "output name")
        spec: dict[str, Any] = {"type": v.require_str(type, "type")}
        if description is not None:
            spec["description"] = v.require_str(description, "description")
        self._outputs[name] = spec

    def add_secret(self, name: str) -> None:
        """Declare a secret the script needs.

        The main process resolves secret values from ``bb.secrets`` at
        runtime. If a declared secret is missing, the script can detect
        that and surface a helpful error.
        """
        v.require_str(name, "secret name")
        if name not in self._secrets:
            self._secrets.append(name)

    def save(self) -> Any:
        """Persist the integration. Errors if a same-name one exists.

        Builds the create payload from the current builder state and
        emits it as an ``integrations.create`` action. The main process
        validates again, writes ``manifest.yaml`` and ``script.py``,
        and returns the file list.
        """
        if self._description is None:
            raise ValueError("integration description is required — set i.description")
        if self._script is None:
            raise ValueError("integration script is required — set i.script")

        payload: dict[str, Any] = {
            "name": self._name,
            "description": self._description,
            "script": self._script,
        }
        if self._inputs:
            payload["inputs"] = self._inputs
        if self._outputs:
            payload["outputs"] = self._outputs
        if self._secrets:
            payload["secrets"] = list(self._secrets)
        if self._timeout is not None:
            payload["timeout"] = self._timeout

        return _transport.request("integrations.create", payload)
