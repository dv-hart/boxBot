"""identify_person tool — the agent's single gateway for person identity.

Four actions, one tool (decision D3: extend the existing tool rather
than grow a second identity surface):

- ``identify`` (default) — record that a session speaker is a specific
  person. Bridges the agent's structured-output identity decisions to
  the perception backend's session-claims + embedding buffer.
- ``rename`` — rename an existing person record ("actually, call me
  Jake"). Pure metadata; embeddings untouched.
- ``merge`` — merge a duplicate person record into the surviving one
  (e.g. the Eric/Erik pair). Destructive; the agent must confirm with
  the humans involved first.
- ``list_flags`` — read the latest nightly identity-reconcile audit
  report (duplicate-person candidates etc.) so the agent can act on it.

Rename and merge re-point everything keyed on the person: photo person
tags, active person-condition triggers, the structured ``person``/
``people`` name fields on memories (what retrieval keys on), in-session
enrollment claims, and live speaker mappings (via the ``PersonRenamed``
event). Only free-text prose inside memory bodies is left as-is — the
nightly dream cycle reconciles that wording over time.
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
        "differed and '{name}' did not exist in the store). If the person "
        "meant they want their EXISTING record renamed (same human, new "
        "name), call identify_person with action=\"rename\" instead."
    ),
}


class IdentifyPersonTool(Tool):
    """Establish, correct, rename, or merge person identities."""

    name = "identify_person"
    description = (
        "Gateway for person identity. Actions:\n"
        "- identify (default): record a session speaker as a person — "
        "first meetings (\"I'm Erik\") or corrections (\"I'm actually "
        "Sarah\"). Needs `name` + `ref`; buffered embeddings commit at "
        "voice-session end.\n"
        "- rename: change an existing person's name (\"call me Jake\"). "
        "Needs `name` (current) + `new_name`. Metadata only; errors if "
        "`new_name` already belongs to someone else.\n"
        "- merge: combine two records for the SAME human. Needs `name` "
        "(the record to KEEP) + `duplicate_name` (merged away). "
        "DESTRUCTIVE and not undoable — confirm with the humans first "
        "(\"Are Eric and Erik the same person?\").\n"
        "- list_flags: read the nightly duplicate-audit findings (use on "
        "an [id-reconcile] to-do).\n"
        "Not for looking up who is present — that's injected via the "
        "[Present: ...] header."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["identify", "rename", "merge", "list_flags"],
                "description": (
                    "What to do. Defaults to \"identify\" (session "
                    "speaker → person). See the tool description for "
                    "when to use each action."
                ),
            },
            "name": {
                "type": "string",
                "description": (
                    "identify: the person's name as you should address "
                    "them. rename: the person's CURRENT name. merge: the "
                    "name of the record to KEEP."
                ),
            },
            "ref": {
                "type": "string",
                "description": (
                    "identify only: the speaker's stable pyannote label, "
                    "from the speaker_identities block in the conversation "
                    "context. If unsure, use the bracketed display label "
                    "in the transcript (e.g. 'Speaker A')."
                ),
            },
            "new_name": {
                "type": "string",
                "description": "rename only: the new name for the person.",
            },
            "duplicate_name": {
                "type": "string",
                "description": (
                    "merge only: the duplicate record to merge away. Its "
                    "embeddings move into `name`'s record. Confirm with "
                    "the humans involved before merging."
                ),
            },
        },
        "required": [],
        "additionalProperties": False,
    }

    async def execute(self, **kwargs: Any) -> Any:
        action: str = kwargs.get("action") or "identify"

        if action == "list_flags":
            return await self._list_flags()
        if action == "rename":
            return await self._rename(
                kwargs.get("name"), kwargs.get("new_name")
            )
        if action == "merge":
            return await self._merge(
                kwargs.get("name"), kwargs.get("duplicate_name")
            )
        return await self._identify(kwargs.get("name"), kwargs.get("ref"))

    # ------------------------------------------------------------------
    # action="identify" (default — original behavior)
    # ------------------------------------------------------------------

    async def _identify(self, name: str | None, ref: str | None) -> Any:
        if not name or not ref:
            return _error("action=\"identify\" requires both name and ref.")

        logger.info("identify_person: name=%s ref=%s", name, ref)

        pipeline, err = _get_running_pipeline()
        if pipeline is None:
            if err is not None:
                return err
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
            return _error("Enrollment manager not available.")

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

    # ------------------------------------------------------------------
    # action="rename"
    # ------------------------------------------------------------------

    async def _rename(self, name: str | None, new_name: str | None) -> str:
        if not name or not new_name:
            return _error(
                "action=\"rename\" requires name (current) and new_name."
            )
        if name == new_name:
            return _error("name and new_name are identical; nothing to do.")

        pipeline, err = _get_running_pipeline()
        if pipeline is None:
            return err or _error(
                "Perception pipeline is not active; cannot rename."
            )
        store = pipeline.cloud_store
        if store is None:
            return _error("Person store not available.")

        person = await store.get_person_by_name(name)
        if person is None:
            return _error(
                f"No person named {name!r} on record. Use search via "
                f"list_flags or check the spelling."
            )

        try:
            result = await store.rename_person(person["id"], new_name)
        except ValueError as exc:
            return _error(str(exc))

        repointed = await _repoint_references(
            pipeline,
            old_person_id=person["id"],
            new_person_id=person["id"],
            old_name=result["old_name"],
            new_name=new_name,
        )

        logger.info(
            "identify_person rename: %r -> %r (%s)",
            result["old_name"], new_name, person["id"],
        )
        return json.dumps({
            "status": "ok",
            "action": "rename",
            "old_name": result["old_name"],
            "new_name": new_name,
            "person_id": person["id"],
            **repointed,
            "message": (
                f"Renamed '{result['old_name']}' to '{new_name}'. Photos "
                f"and triggers were re-pointed automatically. Memory "
                f"records still say '{result['old_name']}' — save a "
                f"memory noting the rename and update any important "
                f"person memories."
            ),
        })

    # ------------------------------------------------------------------
    # action="merge"
    # ------------------------------------------------------------------

    async def _merge(
        self, name: str | None, duplicate_name: str | None
    ) -> str:
        if not name or not duplicate_name:
            return _error(
                "action=\"merge\" requires name (record to KEEP) and "
                "duplicate_name (record to merge away). Confirm with the "
                "humans involved before merging."
            )

        pipeline, err = _get_running_pipeline()
        if pipeline is None:
            return err or _error(
                "Perception pipeline is not active; cannot merge."
            )
        store = pipeline.cloud_store
        if store is None:
            return _error("Person store not available.")

        winner = await store.get_person_by_name(name)
        loser = await store.get_person_by_name(duplicate_name)
        if winner is None:
            return _error(f"No person named {name!r} on record.")
        if loser is None:
            return _error(f"No person named {duplicate_name!r} on record.")
        if winner["id"] == loser["id"]:
            return _error(
                f"{name!r} and {duplicate_name!r} already resolve to the "
                f"same person record; nothing to merge."
            )

        try:
            result = await store.merge_persons(loser["id"], winner["id"])
        except ValueError as exc:
            return _error(str(exc))

        repointed = await _repoint_references(
            pipeline,
            old_person_id=loser["id"],
            new_person_id=winner["id"],
            old_name=result["loser_name"],
            new_name=result["winner_name"],
        )

        logger.info(
            "identify_person merge: %r (%s) -> %r (%s)",
            result["loser_name"], loser["id"],
            result["winner_name"], winner["id"],
        )
        return json.dumps({
            "status": "ok",
            "action": "merge",
            "kept": result["winner_name"],
            "merged_away": result["loser_name"],
            "person_id": winner["id"],
            "visual_embeddings_moved": result["visual_moved"],
            "voice_embeddings_moved": result["voice_moved"],
            **repointed,
            "message": (
                f"Merged '{result['loser_name']}' into "
                f"'{result['winner_name']}'. Embeddings, photo tags, "
                f"triggers, and memory tags now point at "
                f"'{result['winner_name']}'. Older free-text notes may "
                f"still say '{result['loser_name']}'; the nightly "
                f"consolidation reconciles that wording over time."
            ),
        })

    # ------------------------------------------------------------------
    # action="list_flags"
    # ------------------------------------------------------------------

    async def _list_flags(self) -> str:
        from boxbot.perception.reconcile import load_latest_report

        report = load_latest_report()
        if report is None:
            return json.dumps({
                "status": "ok",
                "action": "list_flags",
                "message": (
                    "No identity-reconcile report on file yet — the "
                    "nightly audit hasn't run since this feature shipped."
                ),
            })

        judge = report.get("judge") or {}
        return json.dumps({
            "status": "ok",
            "action": "list_flags",
            "ran_at": report.get("ran_at"),
            "audit_only": report.get("audit_only"),
            "persons": report.get("persons"),
            "duplicate_persons": report.get("duplicate_persons", []),
            "outlier_count": len(report.get("outliers", [])),
            "mislabel_count": len(report.get("mislabels", [])),
            "judge_duplicate_verdicts": judge.get("duplicate_persons", []),
            "message": (
                "duplicate_persons lists records that may be the same "
                "human. Verify with the household before acting; if "
                "confirmed, call identify_person(action=\"merge\", "
                "name=<keep>, duplicate_name=<merge away>)."
            ),
        })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message})


def _get_running_pipeline() -> tuple[Any | None, str | None]:
    """Return (pipeline, None) or (None, error_json | None).

    ``(None, None)`` means the perception module imported fine but the
    pipeline isn't running (identify degrades to an acknowledgement;
    rename/merge return their own error).
    """
    try:
        from boxbot.perception.pipeline import get_pipeline
    except Exception:
        return None, _error("Perception module unavailable.")
    try:
        return get_pipeline(), None
    except RuntimeError:
        logger.debug("Perception pipeline not running")
        return None, None


async def _repoint_references(
    pipeline: Any,
    *,
    old_person_id: str,
    new_person_id: str,
    old_name: str,
    new_name: str,
) -> dict[str, int]:
    """Re-point everything keyed on a renamed/merged person.

    Covers: photo person tags (by person_id), active person-condition
    triggers (by name), the structured ``person``/``people`` name fields on
    memories (by name — what retrieval keys on), in-session enrollment
    claims (by person_id), and — via the ``PersonRenamed`` event — the
    agent's and voice session's live speaker/presence maps. Free-text
    memory prose is left alone; the nightly dream cycle reconciles stale
    wording over time.

    Each leg is best-effort: a photos or scheduler hiccup must not
    abort an already-committed store mutation.
    """
    counts = {"photos_repointed": 0, "triggers_repointed": 0,
              "memories_repointed": 0, "session_claims_repointed": 0}

    # Photo person tags (person_id-keyed, label refreshed to new name).
    try:
        from boxbot.photos.search import get_store

        photo_store = await get_store()
        counts["photos_repointed"] = await photo_store.repoint_person(
            old_person_id, new_person_id, new_label=new_name,
        )
    except Exception:
        logger.exception("identify_person: photo repoint failed")

    # Person-condition triggers (name-keyed).
    try:
        from boxbot.core.scheduler import repoint_person_triggers

        counts["triggers_repointed"] = await repoint_person_triggers(
            old_name, new_name,
        )
    except Exception:
        logger.exception("identify_person: trigger repoint failed")

    # Structured memory name fields (name-keyed, like triggers). Retrieval
    # boosts on person/people, so a merged-away name would orphan its
    # memories from the survivor's recall.
    try:
        from boxbot.tools.builtins.search_memory import _get_memory_store

        mem_store = await _get_memory_store()
        counts["memories_repointed"] = await mem_store.repoint_person_name(
            old_name, new_name,
        )
    except Exception:
        logger.exception("identify_person: memory repoint failed")

    # In-session enrollment claims (so commit_session at voice-session
    # end doesn't write embeddings to a stale/merged-away person).
    try:
        enrollment = pipeline.enrollment
        if enrollment is not None:
            counts["session_claims_repointed"] = enrollment.repoint_person(
                old_person_id, new_person_id, new_name,
            )
    except Exception:
        logger.exception("identify_person: enrollment repoint failed")

    # Live session maps (agent + voice adapter) refresh via the event.
    try:
        from boxbot.core.events import PersonRenamed, get_event_bus

        await get_event_bus().publish(
            PersonRenamed(
                old_name=old_name,
                new_name=new_name,
                person_id=new_person_id,
                merged_from_id=(
                    old_person_id if old_person_id != new_person_id else ""
                ),
            )
        )
    except Exception:
        logger.exception("identify_person: PersonRenamed publish failed")

    return counts


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
