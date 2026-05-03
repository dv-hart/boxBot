"""Secret store — write-only credential storage for the agent.

The agent stores third-party credentials (API keys, OAuth refresh tokens,
…) via ``bb.secrets.store(name, value)``. Values land in a JSON file on
disk that the sandbox user cannot read; integrations and sandbox scripts
receive only the specific values they declared they need, injected as
``BOXBOT_SECRET_<NAME>`` env vars at launch.

The agent never reads values back. ``bb.secrets.list()`` returns names
(and timestamps) only; ``bb.secrets.use(name)`` returns the env-var name
iff the secret exists.
"""

from __future__ import annotations

from boxbot.secrets.store import (
    SecretStore,
    SecretStoreError,
    get_secret_store,
)

__all__ = ["SecretStore", "SecretStoreError", "get_secret_store"]
