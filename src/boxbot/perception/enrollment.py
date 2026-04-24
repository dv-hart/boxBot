"""Person enrollment, session-scoped identity state, and embedding commits.

The EnrollmentManager is the bridge between the perception pipeline (which
produces raw embeddings and speaker labels) and the persistent person store
(``CloudStore``). It holds session-scoped state during a live interaction —
both the raw embeddings accumulated for each label, and a "claim" describing
what boxBot currently believes about that label.

Key concepts:

- **Session ref**: a temporary label for an unidentified speaker/person in
  the current session (e.g. ``"Speaker A"``, ``"Person B"``). Stable within
  a session; never persisted.

- **Embeddings buffer**: voice and visual embeddings captured under a ref
  during the session. Nothing hits the persistent CloudStore until
  ``identify()`` or ``commit_session()`` decides who they belong to.

- **Session claim**: what the system currently believes about a ref. Seeded
  from voice/visual ReID matches at first contact; overwritten by the
  agent's ``identify_person`` tool calls. Drives the end-of-session commit.

- **Identify outcome**: each call to ``identify()`` returns one of five
  outcomes (CREATE/CONFIRM/CORRECT/RENAME/NO_OP). The agent uses this to
  choose natural-language feedback ("nice to meet you" vs "sorry about
  that" vs silent reinforcement).

Usage (inside the pipeline):
    manager = EnrollmentManager(cloud_store)
    manager.on_reid_match("Speaker A", "voice",
                          person_id="p_123", person_name="Sarah",
                          tier="medium", score=0.74)
    manager.buffer_voice_embedding("Speaker A", embedding)
    manager.buffer_visual_embedding("Speaker A", embedding, crop_path=...)

    # Agent calls identify_person:
    outcome = await manager.identify(name="Sarah", ref="Speaker A")

    # On voice session end:
    await manager.commit_session()
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

import numpy as np

from boxbot.perception.clouds import CloudStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class IdentifyOutcome(str, enum.Enum):
    """What happened when ``identify()`` was called.

    - CREATE: no prior claim; name didn't exist; new person record created.
    - CONFIRM: no prior claim (or same name); name exists; embeddings will
      be appended to the existing record.
    - NO_OP: prior claim already matches the given name. Nothing to change.
    - CORRECT: prior claim named a different existing person; re-route
      embeddings to the corrected target. Log a correction event.
    - RENAME: prior claim named someone; new name doesn't exist in the
      store. Treated as creating a new person and pointing the session
      at them — the prior-claim person is NOT renamed in the store.
    """

    CREATE = "create"
    CONFIRM = "confirm"
    NO_OP = "no_op"
    CORRECT = "correct"
    RENAME = "rename"


Modality = Literal["voice", "visual"]
Tier = Literal["high", "medium", "low", "unknown"]
ClaimSource = Literal[
    "voice_reid_match",
    "visual_reid_match",
    "agent_identify",
    "unknown",
]


@dataclass
class SessionClaim:
    """What the system believes about a session ref.

    Both voice and visual ReID can independently seed a claim. The agent's
    ``identify_person`` tool always wins — it sets the canonical claim and
    marks the source as ``"agent_identify"``.
    """

    person_id: str | None
    name: str | None
    source: ClaimSource
    match_tier: Tier | None = None
    match_score: float | None = None
    established_at: float = field(default_factory=time.monotonic)


@dataclass
class SessionPerson:
    """Buffered state for one session ref.

    Attributes:
        ref: Temporary label (e.g. "Speaker A", "Person B").
        visual_embeddings: Visual embeddings captured under this ref.
        voice_embeddings: Voice embeddings captured under this ref.
        first_seen: When first detected/heard this session.
        last_seen: Most recent detection/utterance for this ref.
    """

    ref: str
    visual_embeddings: list[np.ndarray] = field(default_factory=list)
    voice_embeddings: list[np.ndarray] = field(default_factory=list)
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class EnrollmentManager:
    """Session-scoped embedding buffering + person enrollment.

    Args:
        cloud_store: The CloudStore for persistent embedding storage.
    """

    def __init__(self, cloud_store: CloudStore) -> None:
        self._store = cloud_store
        self._session: dict[str, SessionPerson] = {}
        self._claims: dict[str, SessionClaim] = {}

    # --- state inspection -----------------------------------------------

    def get_session_refs(self) -> list[str]:
        """Return all active session refs in this session."""
        return list(self._session.keys())

    def get_session_person(self, ref: str) -> SessionPerson | None:
        """Return buffered state for a ref, or None if unknown."""
        return self._session.get(ref)

    def get_claim(self, ref: str) -> SessionClaim | None:
        """Return the current session claim for a ref, or None."""
        return self._claims.get(ref)

    def get_all_claims(self) -> dict[str, SessionClaim]:
        """Return a snapshot of all current claims."""
        return dict(self._claims)

    # --- buffering ------------------------------------------------------

    def _touch(self, ref: str) -> SessionPerson:
        """Create-or-return the SessionPerson for ref, updating last_seen."""
        if ref not in self._session:
            self._session[ref] = SessionPerson(ref=ref)
        else:
            self._session[ref].last_seen = datetime.now(timezone.utc)
        return self._session[ref]

    def buffer_visual_embedding(
        self,
        ref: str,
        embedding: np.ndarray,
    ) -> None:
        """Buffer a visual embedding under a session ref."""
        self._touch(ref).visual_embeddings.append(embedding)

    def buffer_voice_embedding(
        self,
        ref: str,
        embedding: np.ndarray,
    ) -> None:
        """Buffer a voice embedding under a session ref."""
        self._touch(ref).voice_embeddings.append(embedding)

    # Legacy alias — older perception code calls buffer_embedding for visual.
    def buffer_embedding(self, ref: str, embedding: np.ndarray) -> None:
        """Legacy alias for ``buffer_visual_embedding``."""
        self.buffer_visual_embedding(ref, embedding)

    # --- claim seeding (from perception ReID) ---------------------------

    def on_reid_match(
        self,
        ref: str,
        modality: Modality,
        *,
        person_id: str | None,
        person_name: str | None,
        tier: Tier,
        score: float,
    ) -> None:
        """Seed or reinforce a session claim from a ReID match.

        Called by the perception pipeline when voice or visual ReID
        produces a match (or a confident "unknown") for a session ref.

        Behaviour:
        - If there is no claim yet, create one from this match.
        - If a claim from ``"agent_identify"`` exists, do NOT overwrite —
          the agent's explicit decision is the canonical source.
        - If an older weaker ReID claim exists and this match is stronger
          (higher tier or higher score on the same tier), upgrade it.
        """
        existing = self._claims.get(ref)
        source: ClaimSource = (
            "voice_reid_match" if modality == "voice" else "visual_reid_match"
        )

        if existing is not None and existing.source == "agent_identify":
            # Agent has already spoken — don't overwrite.
            return

        if existing is None:
            self._claims[ref] = SessionClaim(
                person_id=person_id,
                name=person_name,
                source=source,
                match_tier=tier,
                match_score=score,
            )
            return

        # Compare strengths. Higher tier wins; same tier, higher score wins.
        if _tier_rank(tier) > _tier_rank(existing.match_tier) or (
            _tier_rank(tier) == _tier_rank(existing.match_tier)
            and (existing.match_score is None or score > existing.match_score)
        ):
            self._claims[ref] = SessionClaim(
                person_id=person_id,
                name=person_name,
                source=source,
                match_tier=tier,
                match_score=score,
            )

    # --- agent-facing identification -----------------------------------

    async def identify(
        self,
        name: str,
        ref: str,
    ) -> dict:
        """Record that ``ref`` belongs to ``name`` and describe the outcome.

        This does NOT flush buffered embeddings to the store immediately —
        the commit happens at ``commit_session()`` (session end). Instead,
        it updates the session claim so the commit routes correctly.

        Returns a dict the ``identify_person`` tool surfaces to the agent:
        ``{status, outcome, name, ref, person_id, embeddings_buffered,
           prior_claim}``.

        Raises:
            ValueError: if the ref has no buffered state and no prior claim.
        """
        session_person = self._session.get(ref)
        prior_claim = self._claims.get(ref)

        if session_person is None and prior_claim is None:
            raise ValueError(
                f"Session ref '{ref}' is unknown. "
                f"Active refs: {list(self._session.keys())}"
            )

        # Look up target person by name
        existing = await self._store.get_person_by_name(name)

        # Decide outcome based on the matrix in the class docstring
        if existing is not None:
            target_id = existing["id"]
            target_name = existing["name"]
            if prior_claim is None:
                outcome = IdentifyOutcome.CONFIRM
            elif prior_claim.person_id == target_id:
                outcome = IdentifyOutcome.NO_OP
            else:
                outcome = IdentifyOutcome.CORRECT
        else:
            # Name doesn't exist in the store
            target_id = await self._store.create_person(name)
            target_name = name
            if prior_claim is None or prior_claim.person_id is None:
                outcome = IdentifyOutcome.CREATE
            else:
                outcome = IdentifyOutcome.RENAME

        # Overwrite the claim with the agent's decision
        self._claims[ref] = SessionClaim(
            person_id=target_id,
            name=target_name,
            source="agent_identify",
            match_tier="high",
            match_score=1.0,
        )

        buffered = 0
        if session_person is not None:
            buffered = (
                len(session_person.visual_embeddings)
                + len(session_person.voice_embeddings)
            )

        prior_name = prior_claim.name if prior_claim else None
        prior_source = prior_claim.source if prior_claim else None

        logger.info(
            "identify: ref=%s → %s (outcome=%s, buffered=%d, "
            "prior=%s/%s)",
            ref, target_name, outcome.value, buffered,
            prior_name, prior_source,
        )

        if outcome is IdentifyOutcome.CORRECT:
            logger.warning(
                "identify correction: ref=%s was '%s' (%s), now '%s' — "
                "session embeddings will commit to %s",
                ref, prior_name, prior_source, target_name, target_name,
            )

        return {
            "status": "ok",
            "outcome": outcome.value,
            "name": target_name,
            "ref": ref,
            "person_id": target_id,
            "embeddings_buffered": buffered,
            "prior_claim_name": prior_name,
            "prior_claim_source": prior_source,
        }

    # --- commit at session end -----------------------------------------

    async def commit_session(self) -> dict[str, dict]:
        """Flush all buffered embeddings based on session claims.

        For each session ref:
        - If a claim with a ``person_id`` exists, commit buffered voice and
          visual embeddings to that person.
        - If no claim or claim has no person_id (stranger who never got
          introduced), drop the embeddings. Privacy default: no orphan
          records.

        After committing, clear the session state.

        Returns:
            A summary ``{ref: {person_id, name, voice_added, visual_added,
                               dropped}}`` for logging/telemetry.
        """
        summary: dict[str, dict] = {}

        for ref, person in list(self._session.items()):
            claim = self._claims.get(ref)
            if claim is None or claim.person_id is None:
                # Stranger, no identity ever resolved — drop.
                logger.info(
                    "commit_session: dropping %d embeddings for unidentified "
                    "ref=%s (no claim or no person_id)",
                    len(person.voice_embeddings) + len(person.visual_embeddings),
                    ref,
                )
                summary[ref] = {
                    "person_id": None,
                    "name": None,
                    "voice_added": 0,
                    "visual_added": 0,
                    "dropped": (
                        len(person.voice_embeddings)
                        + len(person.visual_embeddings)
                    ),
                }
                continue

            person_id = claim.person_id
            voice_added = 0
            visual_added = 0

            for emb in person.voice_embeddings:
                try:
                    await self._store.add_voice_embedding(person_id, emb)
                    voice_added += 1
                except Exception:
                    logger.exception(
                        "Failed to commit voice embedding for ref=%s → %s",
                        ref, claim.name,
                    )

            for emb in person.visual_embeddings:
                try:
                    # Visual embeddings committed via identify() or reid-match
                    # are treated as voice-confirmed (agent-blessed).
                    await self._store.add_visual_embedding(
                        person_id,
                        emb,
                        voice_confirmed=(claim.source == "agent_identify"),
                    )
                    visual_added += 1
                except Exception:
                    logger.exception(
                        "Failed to commit visual embedding for ref=%s → %s",
                        ref, claim.name,
                    )

            # Recompute centroids now that new embeddings are in.
            try:
                if voice_added > 0:
                    await self._store.recompute_voice_centroid(person_id)
                if visual_added > 0:
                    await self._store.recompute_centroid(person_id)
            except Exception:
                logger.exception(
                    "Centroid recompute failed for person_id=%s", person_id
                )

            logger.info(
                "commit_session: ref=%s → %s (+%d voice, +%d visual, "
                "claim source=%s tier=%s)",
                ref, claim.name, voice_added, visual_added,
                claim.source, claim.match_tier,
            )
            summary[ref] = {
                "person_id": person_id,
                "name": claim.name,
                "voice_added": voice_added,
                "visual_added": visual_added,
                "dropped": 0,
            }

        # Session fully flushed — clear state.
        self._session.clear()
        self._claims.clear()

        return summary

    def clear_session(self) -> None:
        """Clear all buffered session and claim state without committing."""
        self._session.clear()
        self._claims.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_TIER_ORDER: dict[Tier | None, int] = {
    None: -1,
    "unknown": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
}


def _tier_rank(tier: Tier | None) -> int:
    return _TIER_ORDER.get(tier, -1)
