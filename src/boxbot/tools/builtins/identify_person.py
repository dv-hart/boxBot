"""identify_person tool — name or identify a person from the perception pipeline.

Bridges the agent's semantic understanding of identity to the perception
backend's embedding storage. The agent provides a name and a perception
reference label; the backend handles all embedding bookkeeping.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from boxbot.tools.base import Tool

logger = logging.getLogger(__name__)


class IdentifyPersonTool(Tool):
    """Name or identify a person detected by the perception pipeline."""

    name = "identify_person"
    description = (
        "Name or identify a person detected by the perception pipeline. "
        "Provide the person's name and the perception reference label "
        "(e.g., 'Person B'). If the name matches an existing person, "
        "the session's voice and visual embeddings are linked to that "
        "record. If the name is new, a new person record is created. "
        "You never handle embeddings directly — just provide semantic labels."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The person's name.",
            },
            "ref": {
                "type": "string",
                "description": (
                    "The perception reference label for the person "
                    "(e.g., 'Person A', 'Person B')."
                ),
            },
        },
        "required": ["name", "ref"],
        "additionalProperties": False,
    }

    async def execute(self, **kwargs: Any) -> str:
        name: str = kwargs["name"]
        ref: str = kwargs["ref"]

        logger.info("identify_person: name=%s, ref=%s", name, ref)

        # Stub: In production, this:
        # 1. Looks up existing person records by name
        # 2. If found: links ref's session embeddings (voice + visual) to
        #    that record
        # 3. If not found: creates a new person record and stores ref's
        #    embeddings
        # The agent never sees or handles embeddings.

        return json.dumps({
            "status": "identified",
            "name": name,
            "ref": ref,
            "message": (
                f"Person '{ref}' has been identified as '{name}'. "
                "Embeddings will be linked to their person record."
            ),
        })
