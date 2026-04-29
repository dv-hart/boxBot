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

    async def execute(self, **kwargs: Any) -> Any:
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

        body = {
            "status": result.get("status", "ok"),
            "outcome": outcome,
            "name": result.get("name", name),
            "ref": ref,
            "person_id": result.get("person_id"),
            "embeddings_buffered": result.get("embeddings_buffered", 0),
            "prior_claim_name": result.get("prior_claim_name"),
            "prior_claim_source": result.get("prior_claim_source"),
            "message": message,
        }

        # On first-meeting / correction outcomes, attach the speaker's
        # most recent crop so the agent can look at their face and write
        # appearance notes into person memory. The crop is already on
        # disk under data/perception/crops/ (allowlisted for attach).
        if outcome in ("create", "correct", "rename"):
            crop_path = _find_latest_crop_for_ref(pipeline, ref)
            if crop_path is not None:
                from boxbot.tools._sandbox_actions import build_image_block

                block = build_image_block(crop_path)
                if block is not None:
                    body["crop_attached"] = True
                    body["crop_path"] = str(crop_path)
                    body["appearance_prompt"] = (
                        "The speaker's face is attached. Write a short "
                        "appearance description to bb.workspace (e.g. "
                        f"notes/people/{name.lower()}.md) and a memory "
                        "pointer to it so you recognise them later."
                    )
                    return [{"type": "text", "text": json.dumps(body)}, block]

        return json.dumps(body)


def _find_latest_crop_for_ref(pipeline: Any, ref: str) -> "Path | None":
    """Locate the most recent crop image for a speaker ref, if any.

    Lives alongside the tool so it can pick up CropManager state either
    from the pipeline (if it exposes one) or from the on-disk default
    layout. Kept tolerant — this is a UX enhancement, not a required
    path, so exceptions degrade gracefully.
    """
    from pathlib import Path

    try:
        mgr = getattr(pipeline, "_crop_manager", None) or getattr(
            pipeline, "crop_manager", None
        )
        if mgr is None:
            from boxbot.perception.crops import CropManager

            mgr = CropManager()
        p = mgr.latest_for_ref(ref)
        if p is None:
            return None
        return Path(p)
    except Exception:
        logger.debug("crop lookup for ref=%s failed", ref, exc_info=True)
        return None
