"""identify_person tool — record that a session speaker is a specific person.

Bridges the agent's structured-output identity decisions to the perception
backend's session-claims + embedding buffer. The agent provides a name and
the session speaker ref (the raw pyannote label). The backend figures out
whether this is a create, confirm, correct, rename, or no-op and reports
back so the agent can phrase its reply naturally.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from boxbot.tools.base import Tool

logger = logging.getLogger(__name__)


_OUTCOME_MESSAGES = {
    "create": "Created a new person record for '{name}'. Embeddings from this session will commit to them at session end.",
    "confirm": "Confirmed '{ref}' as '{name}'. Session embeddings will reinforce the existing record.",
    "no_op": "Already had '{ref}' as '{name}'; nothing changed.",
    "correct": (
        "Corrected: '{ref}' was previously believed to be someone else "
        "and is now recorded as '{name}'. Embeddings from this session "
        "commit to '{name}'."
    ),
    "rename": (
        "Treated '{name}' as a new person for this session (prior claim "
        "differed and '{name}' did not exist in the store)."
    ),
}


class IdentifyPersonTool(Tool):
    """Record that a session speaker is a specific person."""

    name = "identify_person"
    description = (
        "Record that a session speaker is a specific person. Use when a "
        "speaker has told you their name (first meeting) OR when they've "
        "corrected a mis-identification (\"I'm actually Sarah, not "
        "Carina\"). The system handles both — it decides whether to "
        "create, confirm, correct, rename, or no-op and tells you which "
        "outcome happened so you can respond naturally. Buffered voice "
        "and visual embeddings from this session commit to the named "
        "person when the voice session ends. This is NOT for lookup; it "
        "is only for ESTABLISHING or CORRECTING identity."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The person's name as you should address them.",
            },
            "ref": {
                "type": "string",
                "description": (
                    "The session speaker reference — the stable pyannote "
                    "label for this speaker. You'll find this in the "
                    "speaker_identities block of the conversation context; "
                    "each entry is keyed by the display name you see in "
                    "the transcript, and carries the raw ref internally. "
                    "If unsure, use the display label shown in brackets "
                    "in the transcript (e.g. 'Speaker A')."
                ),
            },
        },
        "required": ["name", "ref"],
        "additionalProperties": False,
    }

    async def execute(self, **kwargs: Any) -> str:
        name: str = kwargs["name"]
        ref: str = kwargs["ref"]

        logger.info("identify_person: name=%s ref=%s", name, ref)

        try:
            from boxbot.perception.pipeline import get_pipeline
        except Exception:
            return json.dumps({
                "status": "error",
                "message": "Perception module unavailable.",
            })

        try:
            pipeline = get_pipeline()
        except RuntimeError:
            logger.debug(
                "Perception pipeline not running; acknowledging without commit"
            )
            return json.dumps({
                "status": "acknowledged",
                "name": name,
                "ref": ref,
                "message": (
                    f"Noted: '{ref}' is '{name}'. Perception pipeline is "
                    f"not active — identity will not be persisted."
                ),
            })

        enrollment = pipeline.enrollment
        if enrollment is None:
            return json.dumps({
                "status": "error",
                "message": "Enrollment manager not available.",
            })

        try:
            result = await enrollment.identify(name, ref)
        except ValueError as exc:
            return json.dumps({
                "status": "error",
                "name": name,
                "ref": ref,
                "message": str(exc),
            })

        outcome = result.get("outcome", "unknown")
        message = _OUTCOME_MESSAGES.get(
            outcome,
            "Recorded '{ref}' as '{name}' (outcome: {outcome}).",
        ).format(name=name, ref=ref, outcome=outcome)

        return json.dumps({
            "status": result.get("status", "ok"),
            "outcome": outcome,
            "name": result.get("name", name),
            "ref": ref,
            "person_id": result.get("person_id"),
            "embeddings_buffered": result.get("embeddings_buffered", 0),
            "prior_claim_name": result.get("prior_claim_name"),
            "prior_claim_source": result.get("prior_claim_source"),
            "message": message,
        })
