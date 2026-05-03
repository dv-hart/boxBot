"""Integration script helpers — used **inside** an integration's script.py.

When an integration's ``script.py`` runs in the sandbox, the runner
sets ``BOXBOT_INTEGRATION_OUTPUT_PATH`` in the environment and
JSON-encodes the inputs as ``BOXBOT_INTEGRATION_INPUTS_PATH``. This
module is the small surface the script uses to read inputs and
return its output.

This is **not** part of the agent-facing ``bb.integrations`` API
(which lives in :mod:`boxbot_sdk.integrations`). The agent never
calls these helpers directly — they're for the script being
executed by the integration runner.

Usage inside an integration's script.py::

    from boxbot_sdk.integration import inputs, return_output

    args = inputs()
    forecast = fetch_weather(args["lat"], args["lon"])
    return_output({"today": forecast})
"""

from __future__ import annotations

import json
import os
from typing import Any

_OUTPUT_PATH_ENV = "BOXBOT_INTEGRATION_OUTPUT_PATH"
_INPUTS_PATH_ENV = "BOXBOT_INTEGRATION_INPUTS_PATH"


class IntegrationContextError(RuntimeError):
    """Raised when an integration helper is called outside an integration run."""


def inputs() -> dict[str, Any]:
    """Return the inputs dict the runner provided.

    Reads JSON from ``BOXBOT_INTEGRATION_INPUTS_PATH``. Returns an
    empty dict if the integration was invoked without inputs. Raises
    :class:`IntegrationContextError` if the env var isn't set —
    indicating the script wasn't launched as an integration.
    """
    path = os.environ.get(_INPUTS_PATH_ENV)
    if not path:
        raise IntegrationContextError(
            f"{_INPUTS_PATH_ENV} not set — this helper only works inside "
            "an integration run launched by boxbot.integrations.runner.run."
        )
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    if not text.strip():
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise IntegrationContextError(
            f"integration inputs file at {path} did not parse to a dict"
        )
    return parsed


def return_output(value: Any) -> None:
    """Set this integration call's output. Last call wins.

    The runner reads this file after the script exits. Calling it
    once with a JSON-serializable value is the contract.
    """
    path = os.environ.get(_OUTPUT_PATH_ENV)
    if not path:
        raise IntegrationContextError(
            f"{_OUTPUT_PATH_ENV} not set — this helper only works inside "
            "an integration run launched by boxbot.integrations.runner.run."
        )
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(value, fh)
