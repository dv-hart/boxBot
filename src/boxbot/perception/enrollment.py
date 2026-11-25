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

- **Voice anchor**: an ``identify_person`` claim binds to a *voice*, not
  to the ref's whole bucket. The anchor is the utterance that carried the
  claim (most recent under the ref) plus the buffered utterances that
  agree with it. Later utterances inherit the name only if they are close
  to the anchor; only anchor members (or utterances that independently
  matched the person's existing cloud with a margin) are ever committed.
  "Store only when certain" — incident 2026-11-24, where one anonymous
  bucket mixed an adult and a child and all of it was written to the
  child's voiceprint.

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
      (To actually rename an existing record — "call me Jake" — use
      ``identify_person(action="rename")``, which hits
      ``CloudStore.rename_person`` instead of this path.)
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
    "voice_doa",          # voice-confirmed + DOA-associated detection (bootstrap)
    "voice_visual_agree",  # voice match and visual ReID agree on the person
    "agent_identify",
    "unknown",
]

# Claim sources that admit embeddings to a cloud at commit (vs. held-back
# passive reid-only claims). See docs/plans/person-id-overhaul.md.
# Visual admits on the bootstrap path too (voice teaches vision); voice only
# admits on agreement or explicit identify (decision 2026-06-07).
_VISUAL_ADMIT_SOURCES = frozenset(
    {"agent_identify", "voice_visual_agree", "voice_doa"}
)
_VOICE_ADMIT_SOURCES = frozenset({"agent_identify", "voice_visual_agree"})

# Map claim source → embedding provenance label (clouds.PROVENANCE_*).
_SOURCE_TO_PROVENANCE = {
    "agent_identify": "agent_identify",
    "voice_visual_agree": "voice_visual_agree",
    "voice_doa": "voice_doa",
}


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
    # Crop image path per visual embedding (parallel to visual_embeddings;
    # None when no crop was saved). Lets commit persist crop_path alongside.
    visual_crops: list[str | None] = field(default_factory=list)
    voice_embeddings: list[np.ndarray] = field(default_factory=list)
    # Per-utterance metadata, parallel to voice_embeddings.
    voice_utterances: list[VoiceUtterance] = field(default_factory=list)
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class VoiceGate:
    """Thresholds for session labelling and the commit gate.

    Defaults mirror ``PerceptionConfig`` (``voice_*``); the manager reads
    the live config when no explicit gate is supplied.
    """

    confirmed: float = 0.55
    maybe: float = 0.44
    anchor: float = 0.55
    anchor_commit: float = 0.60
    cluster: float = 0.44
    margin: float = 0.10
    min_seconds: float = 2.0
    max_refs: int = 6
    topk: int = 3


@dataclass
class VoiceUtterance:
    """One buffered utterance embedding plus what we knew when it arrived.

    ``best_*`` / ``runner_up_score`` are the utterance's own scores against
    the enrolled clouds (None when the caller didn't supply them — legacy
    ``buffer_voice_embedding`` use). ``anchor_member`` marks it as carrying
    the ref's agent claim; ``anchor_score`` is how close it was to the
    anchor (1.0 for the seed itself).
    """

    embedding: np.ndarray
    unit: np.ndarray
    duration: float | None = None
    best_person_id: str | None = None
    best_score: float | None = None
    runner_up_score: float | None = None
    anchor_member: bool = False
    anchor_score: float | None = None


@dataclass
class VoiceAnchor:
    """The voice an ``identify_person`` claim is bound to.

    ``vectors`` are unit embeddings of the anchoring utterances — at most
    top-k of them, fixed when the claim is made. The anchor deliberately
    does not grow as later utterances match it: scoring against a fixed
    set of ≤ k vectors is a plain mean (no best-of-many upward bias, which
    let half-overlapped speech clear the threshold) and cannot drift.
    ``corrected_from`` is the person the ref was believed to be before the
    agent's claim (a correction) — the only enrolled person whose
    confirmed match the anchor may outrank.
    """

    ref: str
    person_id: str
    name: str
    vectors: list[np.ndarray] = field(default_factory=list)
    corrected_from: str | None = None


Certainty = Literal["confirmed", "uncertain", "unknown"]


@dataclass
class VoiceAttribution:
    """Who one utterance is attributed to, and how sure we are.

    Attributes:
        label: What the transcript shows (``Jacob``, ``Jacob?``,
            ``Speaker A``). Also the identity-block key.
        ref: Enrollment buffer the embedding went to. Usually equals
            ``label``; an anchor-inherited utterance shows the claimed
            name but buffers under the claimed ref.
        person_id / person_name: Best belief; a *candidate* only when
            ``certainty == "uncertain"``; None when unknown.
        certainty: ``confirmed`` (own cloud match ≥ confirmed, or
            anchor-consistent with an agent claim), ``uncertain`` (maybe
            band), ``unknown``.
        source: ``voice_reid_match`` | ``agent_identify`` |
            ``session_anchor`` (maybe-band match to an agent anchor) |
            ``unknown``.
        voice_tier / voice_score: Tier + the score that decided it.
    """

    label: str
    ref: str
    person_id: str | None
    person_name: str | None
    certainty: Certainty
    source: str
    voice_tier: Tier
    voice_score: float


_ANON_OVERFLOW_REF = "Speaker ?"
# Cluster/anchor scoring looks at the most recent N members only — bounds
# per-utterance cost on a long session.
_MAX_SCORED_MEMBERS = 24


def _unit(vec: np.ndarray) -> np.ndarray:
    """Flattened float32 L2-unit copy (zero vectors returned as-is)."""
    v = np.asarray(vec, dtype=np.float32).flatten()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _topk_mean(vectors: list[np.ndarray], q: np.ndarray, k: int) -> float:
    """Mean of the top-k cosines of ``q`` to ``vectors`` (-1 if none)."""
    usable = [v for v in vectors[-_MAX_SCORED_MEMBERS:] if v.shape == q.shape]
    if not usable:
        return -1.0
    sims = np.stack(usable) @ q
    k = max(1, min(k, sims.shape[0]))
    return float(np.sort(sims)[-k:].mean())


def _anon_label(index: int) -> str:
    """0 → 'Speaker A', 1 → 'Speaker B', …"""
    if index < 26:
        return f"Speaker {chr(ord('A') + index)}"
    return (
        f"Speaker {chr(ord('A') + (index // 26) - 1)}"
        f"{chr(ord('A') + (index % 26))}"
    )


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class EnrollmentManager:
    """Session-scoped embedding buffering + person enrollment.

    Args:
        cloud_store: The CloudStore for persistent embedding storage.
        voice_gate: Labelling / commit thresholds. Omit to follow the
            live ``PerceptionConfig`` (falls back to defaults when no
            config is loaded).
    """

    def __init__(
        self,
        cloud_store: CloudStore,
        voice_gate: VoiceGate | None = None,
    ) -> None:
        self._store = cloud_store
        self._voice_gate = voice_gate
        self._session: dict[str, SessionPerson] = {}
        self._claims: dict[str, SessionClaim] = {}
        # Agent-claim anchors, keyed by the claimed ref.
        self._anchors: dict[str, VoiceAnchor] = {}
        # Anonymous session voices in first-heard order (Speaker A, B, …).
        self._anon_refs: list[str] = []

    def _gate(self) -> VoiceGate:
        """Current thresholds: explicit gate, else config, else defaults."""
        if self._voice_gate is not None:
            return self._voice_gate
        gate = VoiceGate()
        try:
            from boxbot.core.config import get_config

            p = get_config().perception
        except Exception:
            return gate
        for attr, key in (
            ("confirmed", "voice_confirmed_threshold"),
            ("maybe", "voice_maybe_threshold"),
            ("anchor", "voice_anchor_threshold"),
            ("anchor_commit", "voice_anchor_commit_threshold"),
            ("cluster", "voice_session_cluster_threshold"),
            ("margin", "voice_commit_margin"),
            ("min_seconds", "voice_commit_min_seconds"),
            ("max_refs", "voice_session_max_refs"),
            ("topk", "voice_cloud_topk"),
        ):
            value = getattr(p, key, None)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                setattr(gate, attr, type(getattr(gate, attr))(value))
        return gate

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

    def get_anchor(self, ref: str) -> VoiceAnchor | None:
        """Return the agent-claim voice anchor bound to a ref, or None."""
        return self._anchors.get(ref)

    def resolve_ref(self, ref: str) -> str | None:
        """Map what the agent typed to a live session ref.

        The agent passes the bracketed transcript label. That is usually
        the ref itself, but an anchor-inherited utterance is displayed
        under the claimed *name* while buffered under the claimed ref,
        and an uncertain label carries a trailing ``?``. Returns None if
        nothing matches.
        """
        known = set(self._session) | set(self._claims)
        if ref in known:
            return ref
        wanted = ref.strip()
        folded = wanted.casefold()
        for candidate in known:
            if candidate.casefold() == folded:
                return candidate
        # Claimed name → the ref its anchor lives on (latest claim wins).
        bare = folded.rstrip("?").strip()
        for anchor in reversed(list(self._anchors.values())):
            if anchor.name.casefold() in (folded, bare):
                return anchor.ref
        # "Jacob" ↔ "Jacob?"
        for candidate in known:
            if candidate.casefold().rstrip("?").strip() == bare:
                return candidate
        return None

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
        crop_path: str | None = None,
    ) -> None:
        """Buffer a visual embedding (and optional saved crop) under a ref."""
        person = self._touch(ref)
        person.visual_embeddings.append(embedding)
        person.visual_crops.append(crop_path)

    def buffer_voice_embedding(
        self,
        ref: str,
        embedding: np.ndarray,
        *,
        duration: float | None = None,
        best_person_id: str | None = None,
        best_score: float | None = None,
        runner_up_score: float | None = None,
        anchor_member: bool = False,
        anchor_score: float | None = None,
    ) -> VoiceUtterance:
        """Buffer a voice embedding (and its match metadata) under a ref.

        The metadata feeds the commit gate. Without it the embedding can
        still anchor an ``identify`` claim, but can never be committed on
        the strength of its own cloud match.
        """
        utterance = VoiceUtterance(
            embedding=embedding,
            unit=_unit(embedding),
            duration=duration,
            best_person_id=best_person_id,
            best_score=best_score,
            runner_up_score=runner_up_score,
            anchor_member=anchor_member,
            anchor_score=anchor_score,
        )
        person = self._touch(ref)
        person.voice_embeddings.append(embedding)
        person.voice_utterances.append(utterance)
        return utterance

    # --- per-utterance voice attribution ---------------------------------

    def attribute_voice(
        self,
        embedding: np.ndarray,
        ranked: list[tuple[str, str, float]] | None = None,
        *,
        duration: float | None = None,
    ) -> VoiceAttribution:
        """Label one utterance, buffer it, and seed its ref's claim.

        ``ranked`` is ``VoiceReID.rank_clouds`` output — every enrolled
        person's score, best first. Decision order:

        1. **Own match beats claim.** Best enrolled score ≥ confirmed →
           that person, whatever claims exist. (Sole exception: an anchor
           the agent created by *correcting away from* that same person,
           when the utterance is at least as close to the anchor.)
        2. **Anchor-consistent.** ≥ anchor threshold to an agent-claim
           anchor and no other enrolled person scores as high → the
           claimed name; the utterance is marked committable.
        3. **Maybe band.** Best of (enrolled, anchor) ≥ maybe → shown as
           ``Name?``; a candidate, never committed.
        4. **Unknown.** Nearest anonymous session voice ≥ cluster
           threshold, else a new ``Speaker X`` (capped).

        Cost: a handful of dot products per enrolled person / anchor /
        session voice.
        """
        gate = self._gate()
        ranked = ranked or []
        q = _unit(embedding)

        best_id, best_name, best_score = ranked[0] if ranked else (None, None, 0.0)
        runner_up = ranked[1][2] if len(ranked) > 1 else 0.0
        meta = {
            "duration": duration,
            "best_person_id": best_id,
            "best_score": float(best_score) if ranked else None,
            "runner_up_score": float(runner_up) if ranked else None,
        }

        # Best agent-claim anchor for this voice.
        anchor: VoiceAnchor | None = None
        anchor_score = -1.0
        for cand in self._anchors.values():
            score = _topk_mean(cand.vectors, q, gate.topk)
            if score > anchor_score:
                anchor, anchor_score = cand, score

        # Strongest enrolled person who is NOT the anchor's person.
        rival_id: str | None = None
        rival_score = -1.0
        if anchor is not None:
            for pid, _name, score in ranked:
                if pid != anchor.person_id:
                    rival_id, rival_score = pid, score
                    break
        anchor_wins = (
            anchor is not None
            and anchor_score >= gate.anchor
            and (
                anchor_score > rival_score
                or (
                    rival_id is not None
                    and rival_id == anchor.corrected_from
                    and anchor_score >= rival_score
                )
            )
        )

        # 1. Own confirmed match.
        if best_id is not None and best_score >= gate.confirmed:
            corrected_away = (
                anchor_wins
                and anchor is not None
                and anchor.corrected_from == best_id
            )
            own_is_anchor_person = (
                anchor_wins and anchor is not None
                and anchor.person_id == best_id
            )
            if not corrected_away and not own_is_anchor_person:
                self.buffer_voice_embedding(best_name, embedding, **meta)
                self.on_reid_match(
                    best_name, "voice", person_id=best_id,
                    person_name=best_name, tier="high", score=best_score,
                )
                return VoiceAttribution(
                    label=best_name, ref=best_name, person_id=best_id,
                    person_name=best_name, certainty="confirmed",
                    source="voice_reid_match", voice_tier="high",
                    voice_score=float(best_score),
                )

        # 2. Anchor-consistent with an agent claim.
        if anchor_wins and anchor is not None:
            self.buffer_voice_embedding(
                anchor.ref, embedding, anchor_member=True,
                anchor_score=float(anchor_score), **meta
            )
            return VoiceAttribution(
                label=anchor.name, ref=anchor.ref,
                person_id=anchor.person_id, person_name=anchor.name,
                certainty="confirmed", source="agent_identify",
                voice_tier="high", voice_score=float(anchor_score),
            )

        # 3. Maybe band — a candidate name, shown as uncertain.
        enrolled_maybe = best_id is not None and best_score >= gate.maybe
        anchor_maybe = anchor is not None and anchor_score >= gate.maybe
        if enrolled_maybe or anchor_maybe:
            use_anchor = anchor_maybe and (
                not enrolled_maybe or anchor_score > best_score
            )
            if use_anchor and anchor is not None:
                cand_id, cand_name = anchor.person_id, anchor.name
                cand_score, source = anchor_score, "session_anchor"
            else:
                cand_id, cand_name = best_id, best_name
                cand_score, source = best_score, "voice_reid_match"
            label = f"{cand_name}?"
            self.buffer_voice_embedding(label, embedding, **meta)
            self.on_reid_match(
                label, "voice", person_id=cand_id, person_name=cand_name,
                tier="medium", score=float(cand_score),
            )
            return VoiceAttribution(
                label=label, ref=label, person_id=cand_id,
                person_name=cand_name, certainty="uncertain",
                source=source, voice_tier="medium",
                voice_score=float(cand_score),
            )

        # 4. Unknown — session-local grouping of anonymous voices.
        ref = self._assign_anon_ref(q, gate)
        self.buffer_voice_embedding(ref, embedding, **meta)
        self.on_reid_match(
            ref, "voice", person_id=None, person_name=None,
            tier="unknown", score=float(best_score) if ranked else 0.0,
        )
        return VoiceAttribution(
            label=ref, ref=ref, person_id=None, person_name=None,
            certainty="unknown", source="unknown", voice_tier="unknown",
            voice_score=float(best_score) if ranked else 0.0,
        )

    def _assign_anon_ref(self, q: np.ndarray, gate: VoiceGate) -> str:
        """Nearest anonymous session voice ≥ cluster threshold, else new.

        Past ``max_refs`` distinct voices, further strangers share the
        overflow ref rather than being forced into someone else's.
        """
        best_ref: str | None = None
        best_score = -1.0
        for ref in self._anon_refs:
            person = self._session.get(ref)
            if person is None:
                continue
            score = _topk_mean(
                [u.unit for u in person.voice_utterances], q, gate.topk
            )
            if score > best_score:
                best_ref, best_score = ref, score
        if best_ref is not None and best_score >= gate.cluster:
            return best_ref
        if len(self._anon_refs) >= max(1, gate.max_refs):
            return _ANON_OVERFLOW_REF
        ref = _anon_label(len(self._anon_refs))
        self._anon_refs.append(ref)
        return ref

    def _bind_anchor(
        self,
        ref: str,
        person_id: str,
        name: str,
        corrected_from: str | None,
    ) -> dict:
        """(Re)build the voice anchor for an agent claim on ``ref``.

        Seed = the most recent utterance under the ref — the one that
        carried the self-identification the agent just acted on. Other
        buffered utterances join only if they agree with the seed
        (cosine ≥ anchor threshold). Utterances clearly unlike the seed
        (< cluster threshold) are counted as another voice.
        """
        gate = self._gate()
        person = self._session.get(ref)
        utterances = person.voice_utterances if person is not None else []
        for u in utterances:
            u.anchor_member = False
            u.anchor_score = None
        if not utterances:
            self._anchors.pop(ref, None)
            return {"anchored": 0, "other_voice": 0, "unanchored": 0}

        seed = utterances[-1]
        seed.anchor_member = True
        seed.anchor_score = 1.0
        vectors = [seed.unit]
        other_voice = 0
        unanchored = 0
        for u in utterances[:-1]:
            sim = (
                float(u.unit @ seed.unit)
                if u.unit.shape == seed.unit.shape else -1.0
            )
            if sim >= gate.anchor:
                u.anchor_member = True
                u.anchor_score = sim
                vectors.append(u.unit)
            elif sim < gate.cluster:
                other_voice += 1
            else:
                unanchored += 1
        # Keep the seed plus the most recent agreeing utterances, ≤ top-k.
        k = max(1, gate.topk)
        vectors = vectors[:1] + vectors[1:][-(k - 1):] if k > 1 else vectors[:1]
        # Re-insert so the newest claim is the last anchor (resolve_ref).
        self._anchors.pop(ref, None)
        self._anchors[ref] = VoiceAnchor(
            ref=ref, person_id=person_id, name=name,
            vectors=vectors, corrected_from=corrected_from,
        )
        return {
            "anchored": sum(1 for u in utterances if u.anchor_member),
            "other_voice": other_voice,
            "unanchored": unanchored,
        }

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

    def mark_admission(
        self,
        ref: str,
        *,
        person_id: str,
        person_name: str | None,
        source: ClaimSource,
        score: float,
    ) -> None:
        """Record a fusion-driven *admitting* claim for a ref.

        Called when fusion concludes an identity strong enough to grow a cloud:
        ``voice_doa`` (voice-confirmed + DOA-associated detection → teach
        vision) or ``voice_visual_agree`` (voice and visual agree). Never
        overrides an ``agent_identify`` claim (the agent is canonical), and
        never downgrades a stronger admitting source already set this session.
        """
        existing = self._claims.get(ref)
        if existing is not None and existing.source == "agent_identify":
            return
        if (
            existing is not None
            and _admit_rank(source) < _admit_rank(existing.source)
        ):
            return
        self._claims[ref] = SessionClaim(
            person_id=person_id,
            name=person_name,
            source=source,
            match_tier="high",
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

        The claim binds to a *voice*: see :meth:`_bind_anchor`. Only the
        anchoring utterances (and later ones that match them) take the
        name or reach the voiceprint — not the ref's whole bucket.

        Returns a dict the ``identify_person`` tool surfaces to the agent:
        ``{status, outcome, name, ref, person_id, embeddings_buffered,
           prior_claim_name, prior_claim_source, voice_utterances_anchored,
           voice_utterances_other_voice, voice_utterances_unanchored,
           mixed_voices}`` (``ref`` is the resolved session ref).

        Raises:
            ValueError: if the ref has no buffered state and no prior claim.
        """
        resolved = self.resolve_ref(ref)
        if resolved is None:
            raise ValueError(
                f"Session ref '{ref}' is unknown. "
                f"Active refs: {list(self._session.keys())}"
            )
        ref = resolved
        session_person = self._session.get(ref)
        prior_claim = self._claims.get(ref)

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

        # Bind the claim to the voice that made it, not the bucket.
        corrected_from = (
            prior_claim.person_id
            if prior_claim is not None
            and prior_claim.person_id
            and prior_claim.person_id != target_id
            else None
        )
        binding = self._bind_anchor(ref, target_id, target_name, corrected_from)

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
            "prior=%s/%s, voice anchored=%d other-voice=%d unanchored=%d)",
            ref, target_name, outcome.value, buffered,
            prior_name, prior_source,
            binding["anchored"], binding["other_voice"], binding["unanchored"],
        )

        if outcome is IdentifyOutcome.CORRECT:
            logger.warning(
                "identify correction: ref=%s was '%s' (%s), now '%s' — "
                "anchored session embeddings will commit to %s",
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
            "voice_utterances_anchored": binding["anchored"],
            "voice_utterances_other_voice": binding["other_voice"],
            "voice_utterances_unanchored": binding["unanchored"],
            "mixed_voices": binding["other_voice"] > 0,
        }

    # --- commit at session end -----------------------------------------

    async def commit_session(
        self, session_id: str | None = None
    ) -> dict[str, dict]:
        """Flush buffered embeddings that pass the commit gate, then clear.

        Voice — "store only when certain". An utterance embedding is
        written to a person's voiceprint only if:

        (a) it carries an ``agent_identify`` claim made this session —
            anchor-consistent at the stricter *commit* threshold (→ the
            claimed person, provenance ``agent_identify``), or
        (b) its ref carries an admitting claim (``agent_identify`` /
            ``voice_visual_agree``) for person P *and* the utterance
            itself matched P's existing cloud ≥ confirmed with a clear
            margin over the next-best person (→ P, provenance
            ``voice_reid`` / the claim source).

        Everything else is dropped and counted per ref: ``dropped_low``
        (not certain enough), ``dropped_conflict`` (looked like a
        different enrolled person), ``dropped_short`` (under the minimum
        duration). Passive voice-ReID claims still never grow a cloud.

        Visual — unchanged: all of a ref's embeddings commit when its
        claim source admits visual (agent_identify / voice_visual_agree /
        voice_doa); strangers are dropped (no orphan records).

        Args:
            session_id: Voice session id, stamped on each committed voice
                row so the session can be undone
                (``CloudStore.delete_voice_embeddings_by_session``).

        Returns:
            ``{ref: {person_id, name, voice_added, visual_added, dropped,
            voice_dropped_low, voice_dropped_conflict,
            voice_dropped_short}}``.
        """
        gate = self._gate()
        summary: dict[str, dict] = {}

        for ref, person in list(self._session.items()):
            claim = self._claims.get(ref)
            anchor = self._anchors.get(ref)
            has_person = claim is not None and claim.person_id is not None
            admit_visual = has_person and claim.source in _VISUAL_ADMIT_SOURCES
            admit_voice = has_person and claim.source in _VOICE_ADMIT_SOURCES
            person_id = claim.person_id if has_person else None

            voice_added = 0
            dropped_low = 0
            dropped_conflict = 0
            dropped_short = 0

            for u in person.voice_utterances:
                verdict, provenance, confidence = self._voice_commit_verdict(
                    u, claim if admit_voice else None, anchor, gate
                )
                if verdict == "low":
                    dropped_low += 1
                    continue
                if verdict == "conflict":
                    dropped_conflict += 1
                    continue
                if verdict == "short":
                    dropped_short += 1
                    continue
                try:
                    added = await self._store.add_voice_embedding(
                        person_id, u.embedding,
                        provenance=provenance,
                        confidence=confidence,
                        source_ref=ref,
                        session_id=session_id,
                    )
                    if added:
                        voice_added += 1
                except Exception:
                    logger.exception(
                        "Failed to commit voice embedding for ref=%s → %s",
                        ref, claim.name if claim else None,
                    )

            visual_added = 0
            visual_dropped = 0
            if admit_visual:
                provenance = _SOURCE_TO_PROVENANCE.get(claim.source, "legacy")
                crops = person.visual_crops
                for idx, emb in enumerate(person.visual_embeddings):
                    crop_path = crops[idx] if idx < len(crops) else None
                    try:
                        added = await self._store.add_visual_embedding(
                            person_id,
                            emb,
                            voice_confirmed=True,
                            crop_path=crop_path,
                            provenance=provenance,
                            confidence=claim.match_score,
                            source_ref=ref,
                        )
                        if added:
                            visual_added += 1
                    except Exception:
                        logger.exception(
                            "Failed to commit visual embedding for ref=%s → %s",
                            ref, claim.name,
                        )
            else:
                visual_dropped = len(person.visual_embeddings)

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
                "commit_session: ref=%s → %s | voice committed=%d "
                "dropped-low=%d dropped-conflict=%d dropped-short=%d | "
                "visual committed=%d dropped=%d | claim source=%s "
                "session=%s",
                ref, claim.name if has_person else None,
                voice_added, dropped_low, dropped_conflict, dropped_short,
                visual_added, visual_dropped,
                claim.source if claim else None, session_id,
            )
            summary[ref] = {
                "person_id": person_id,
                "name": claim.name if has_person else None,
                "voice_added": voice_added,
                "visual_added": visual_added,
                "dropped": (
                    dropped_low + dropped_conflict + dropped_short
                    + visual_dropped
                ),
                "voice_dropped_low": dropped_low,
                "voice_dropped_conflict": dropped_conflict,
                "voice_dropped_short": dropped_short,
            }

        # Session fully flushed — clear state.
        self.clear_session()

        return summary

    @staticmethod
    def _voice_commit_verdict(
        u: VoiceUtterance,
        claim: SessionClaim | None,
        anchor: VoiceAnchor | None,
        gate: VoiceGate,
    ) -> tuple[str, str, float | None]:
        """Decide one utterance's fate at commit.

        ``claim`` is the ref's claim only if it admits voice. Returns
        ``(verdict, provenance, confidence)`` with verdict one of
        ``commit`` / ``low`` / ``conflict`` / ``short``.
        """
        if claim is None or claim.person_id is None:
            return "low", "", None

        target = claim.person_id
        looks_like_other = (
            u.best_person_id is not None
            and u.best_person_id != target
            and u.best_score is not None
        )
        is_anchor_member = (
            claim.source == "agent_identify"
            and anchor is not None
            and anchor.person_id == target
            and u.anchor_member
        )
        self_matched = (
            u.best_person_id == target
            and u.best_score is not None
            and u.best_score >= gate.confirmed
            and (u.best_score - (u.runner_up_score or 0.0)) >= gate.margin
        )

        if is_anchor_member:
            # Labelled on the anchor threshold; stored only on the
            # stricter one — a half-overlapped utterance can scrape past
            # the first.
            if u.anchor_score is None or u.anchor_score < gate.anchor_commit:
                return "low", "", None
            # A confirmed match to someone the agent did not correct away
            # from means the anchor swallowed another enrolled voice.
            if (
                looks_like_other
                and u.best_score >= gate.confirmed
                and u.best_person_id != anchor.corrected_from
            ):
                return "conflict", "", None
            provenance, confidence = "agent_identify", claim.match_score
        elif self_matched:
            provenance = (
                "voice_reid" if claim.source == "agent_identify"
                else _SOURCE_TO_PROVENANCE.get(claim.source, "legacy")
            )
            confidence = u.best_score
        elif looks_like_other and u.best_score >= gate.maybe:
            return "conflict", "", None
        else:
            return "low", "", None

        if (
            gate.min_seconds > 0
            and u.duration is not None
            and u.duration < gate.min_seconds
        ):
            return "short", "", None
        return "commit", provenance, confidence

    def repoint_person(
        self,
        old_person_id: str,
        new_person_id: str,
        new_name: str | None = None,
    ) -> int:
        """Re-point in-session claims after a rename or merge.

        After ``identify_person(action="rename")`` the person keeps
        their id but changes name (call with ``old == new`` id); after
        ``action="merge"`` the loser's id is dead and claims must point
        at the winner. Without this, ``commit_session`` at voice-session
        end would write buffered embeddings to a stale/merged-away
        person.

        Returns the number of claims updated.
        """
        updated = 0
        for ref, claim in list(self._claims.items()):
            if claim.person_id != old_person_id:
                continue
            self._claims[ref] = SessionClaim(
                person_id=new_person_id,
                name=new_name if new_name is not None else claim.name,
                source=claim.source,
                match_tier=claim.match_tier,
                match_score=claim.match_score,
                established_at=claim.established_at,
            )
            updated += 1
        for anchor in self._anchors.values():
            if anchor.person_id == old_person_id:
                anchor.person_id = new_person_id
                if new_name is not None:
                    anchor.name = new_name
            if anchor.corrected_from == old_person_id:
                anchor.corrected_from = new_person_id
        if updated:
            logger.info(
                "repoint_person: %d session claim(s) moved %s -> %s (%s)",
                updated, old_person_id, new_person_id, new_name,
            )
        return updated

    def clear_session(self) -> None:
        """Clear all buffered session and claim state without committing."""
        self._session.clear()
        self._claims.clear()
        self._anchors.clear()
        self._anon_refs.clear()


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


# Strength order for admitting claim sources (higher wins, never downgraded).
_ADMIT_ORDER: dict[str, int] = {
    "voice_reid_match": 0,
    "visual_reid_match": 0,
    "voice_doa": 1,
    "voice_visual_agree": 2,
    "agent_identify": 3,
}


def _admit_rank(source: str | None) -> int:
    return _ADMIT_ORDER.get(source or "", -1)
