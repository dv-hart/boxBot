"""load_skill tool — read a skill's body or sub-file on demand.

Progressive-disclosure wrapper around ``boxbot.skills.loader.load_skill``.
The short skills index is injected into the system prompt; this tool
exists so the agent can pull the full body (or a specific sub-file)
only when the task calls for it.

Registration
------------
This file defines ``LoadSkillTool`` following the ``Tool`` base-class
convention used by the other builtins. It is **intentionally not
registered** in ``src/boxbot/tools/registry.py`` yet — wiring is left
for a human review step so the caller can decide when to make it live
(the spec pairs it with removing ``speak`` from ``_load_tools`` to keep
the always-loaded tool count at nine).

To register:
    1. Add ``from boxbot.tools.builtins.load_skill import LoadSkillTool``
       to ``src/boxbot/tools/registry.py``.
    2. Add ``LoadSkillTool()`` to the list returned by ``_load_tools``.

Error handling
--------------
The agent runtime (``_process_tool_calls`` in ``core/agent.py``) wraps any
exception raised by ``execute()`` into a tool-result string. To match the
existing pattern used by ``search_memory`` etc., this tool returns a
JSON-encoded error envelope on ``ValueError`` rather than raising, so the
agent sees a clean machine-parseable message. If the runtime grows real
``is_error`` support, swap the return path for that.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from boxbot.tools.base import Tool

logger = logging.getLogger(__name__)


class LoadSkillTool(Tool):
    """Load a specific skill's full body or a sub-file for on-demand guidance."""

    name = "load_skill"
    description = (
        "Load a specific skill's full body or a sub-file for on-demand "
        "guidance. Call when a skill's index description matches the "
        "current task."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Skill name from the skills index.",
            },
            "subpath": {
                "type": "string",
                "description": (
                    "Optional relative path to a sub-file within the "
                    "skill directory."
                ),
            },
        },
        "required": ["name"],
        "additionalProperties": False,
    }

    async def execute(self, **kwargs: Any) -> str:
        name: str = kwargs["name"]
        subpath: str | None = kwargs.get("subpath")

        logger.info("load_skill: name=%s subpath=%s", name, subpath)

        try:
            from boxbot.skills.loader import load_skill as _load_skill

            return _load_skill(name=name, subpath=subpath)
        except ValueError as exc:
            logger.info("load_skill rejected request: %s", exc)
            return json.dumps({"error": str(exc)})
        except Exception as exc:
            logger.exception("load_skill unexpected failure")
            return json.dumps({"error": f"Skill load failed: {exc}"})
