"""Secret store — write-only credential storage.

Secrets are stored securely and cannot be viewed after storage. The agent
can reference them by name (e.g., in data source configs or skill env vars).

Usage:
    from boxbot_sdk import secrets

    secrets.store("polygon_api_key", "pk_abc123...")
    env_name = secrets.use("polygon_api_key")
    # env_name = "BOXBOT_SECRET_POLYGON_API_KEY"
"""

from __future__ import annotations

from . import _transport, _validators as v


def store(name: str, value: str) -> None:
    """Store a secret value.

    The value is stored securely and cannot be retrieved by agent scripts
    after storage. It can be referenced by name in data source `secret`
    fields and skill `env_var` declarations.

    Args:
        name: Secret name (used as reference key).
        value: Secret value to store.
    """
    v.require_str(name, "name")
    v.require_str(value, "value")
    _transport.emit_action("secrets.store", {
        "name": name,
        "value": value,
    })


def use(name: str) -> str:
    """Get the environment variable name for a stored secret.

    Returns the env var name that will contain the secret value when
    scripts or skills run. The agent cannot see the actual value.

    Args:
        name: Secret name.

    Returns:
        Environment variable name, e.g. "BOXBOT_SECRET_POLYGON_API_KEY".
    """
    v.require_str(name, "name")
    return f"BOXBOT_SECRET_{name.upper()}"
