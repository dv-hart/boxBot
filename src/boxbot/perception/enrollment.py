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
  to the ref's whole bucket. The anchor's seed is the utterance the agent
  acted on: the one it names by id, else the most recent utterance under
  the label that had been *delivered to the agent* (``mark_shown``) —
  never one still queued behind the agent's turn. Buffered utterances
  that agree with the seed join it. Later utterances inherit the name
  only if they are close to the anchor; only corroborated anchor members
  (or utterances that independently matched the person's existing cloud
  with a margin) are ever committed. "Store only when certain" —
  incident 2026-11-24, where one anonymous bucket mixed an adult and a
  child and all of it was written to the child's voiceprint.

- **Utterance id**: every buffered utterance gets a short id (``u7``),
  unique for the life of the manager. It rides the identity block the
  agent sees, so ``identify_person(utterance="u7")`` binds to exactly
  that line.

Usage (inside the pipeline):
    manager = EnrollmentManager(cloud_store)
    manager.on_reid_match("Speaker A", "voice",
                          person_id="p_123", person_name="Sarah",
                          tier="medium", score=0.74)
    manager.buffer_voice_embedding("Speaker A", embedding)
    manager.buffer_visual_embedding("Speaker A", embedding, crop_path=...)

    # The line reached the agent's thread; then it calls identify_person:
    manager.mark_shown([utterance_id])
    outcome = await manager.identify(name="Sarah", utterance=utterance_id)

    # On voice session end:
    await manager.commit_session()
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Literal

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
    anchor (1.0 for the seed itself); ``anchor_seed`` marks the seed.

    ``uid`` is the short id the agent sees beside the line's label.
    ``shown`` turns True once the utterance's transcript line has
    entered the agent's thread (``EnrollmentManager.mark_shown``) — an
    utterance still queued behind the agent's turn is not something a
    label-only ``identify_person`` call can have been about.
    """

    embedding: np.ndarray
    unit: np.ndarray
    duration: float | None = None
    best_person_id: str | None = None
    best_score: float | None = None
    runner_up_score: float | None = None
    anchor_member: bool = False
    anchor_score: float | None = None
    anchor_seed: bool = False
    uid: str = ""
    ref: str = ""
    shown: bool = False


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

    ``label_only``: the ref held another voice when the claim was made
    and the agent did not say which utterance it meant. The anchor then
    labels (seed vector only) and nothing under the ref is committed.
    ``rivals``: unit embeddings that sat under the ref but did NOT agree
    with the seed — known not-this-voice. An utterance closer to a rival
    than to the anchor never takes the anchor's name.
    """

    ref: str
    person_id: str
    name: str
    vectors: list[np.ndarray] = field(default_factory=list)
    corrected_from: str | None = None
    seed_uid: str | None = None
    label_only: bool = False
    rivals: list[np.ndarray] = field(default_factory=list)


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
        utterance_id: Short id of the buffered utterance (``u7``).
    """

    label: str
    ref: str
    person_id: str | None
    person_name: str | None
    certainty: Certainty
    source: str
    voice_tier: Tier
    voice_score: float
    utterance_id: str = ""


@dataclass
class _SessionState:
    """Everything scoped to one voice session.

    Held as one object so ``commit_session`` can detach the ending
    session in a single synchronous swap (a session starting while the
    commit's DB writes are in flight gets fresh state).
    """

    session: dict[str, SessionPerson] = field(default_factory=dict)
    claims: dict[str, SessionClaim] = field(default_factory=dict)
    # Agent-claim anchors, keyed by the claimed ref.
    anchors: dict[str, VoiceAnchor] = field(default_factory=dict)
    # Anonymous session voices in first-heard order (Speaker A, B, …).
    anon_refs: list[str] = field(default_factory=list)
    # Case-folded transcript label → ref, for every label actually put
    # on a line this session.
    labels: dict[str, str] = field(default_factory=dict)
    # Utterance id → utterance.
    utterances: dict[str, VoiceUtterance] = field(default_factory=dict)


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


def _fold_uid(utterance_id: str | None) -> str:
    """Normalise an utterance id as typed by the agent."""
    return str(utterance_id or "").strip().strip("[]").casefold()


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
        self._state = _SessionState()
        # Sessions detached by ``commit_session`` and still being written.
        self._detached: list[_SessionState] = []
        # Utterance-id counter. Never reset: an id from an ended session
        # must not name an utterance of the next one.
        self._utterance_seq = 0

    @property
    def _session(self) -> dict[str, SessionPerson]:
        return self._state.session

    @property
    def _claims(self) -> dict[str, SessionClaim]:
        return self._state.claims

    @property
    def _anchors(self) -> dict[str, VoiceAnchor]:
        return self._state.anchors

    @property
    def _anon_refs(self) -> list[str]:
        return self._state.anon_refs

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

    def live_labels(self) -> list[str]:
        """Labels the agent may pass as ``ref`` — session refs plus every
        label put on a transcript line this session."""
        state = self._state
        out = list(dict.fromkeys([*state.session, *state.claims]))
        folded = {label.casefold() for label in out}
        for anchor in state.anchors.values():
            if (
                anchor.name.casefold() in state.labels
                and anchor.name.casefold() not in folded
            ):
                out.append(anchor.name)
                folded.add(anchor.name.casefold())
        return out

    def resolve_ref(self, ref: str) -> str | None:
        """Map what the agent typed to a live session ref.

        Exact (case-folded) matches only:

        1. a session ref (``Speaker A``, ``Jacob?``, ``Jacob``);
        2. a label that was actually put on a transcript line this
           session — an anchor-inherited utterance is displayed under
           the claimed *name* while buffered under the claimed ref.

        Nothing else. In particular ``Name?`` never resolves to the
        ``Name`` anchor and ``Name`` never to ``Name?``: those are
        different voices by construction, and a claim that lands on the
        wrong one re-labels a person (review 2026-11-25). Returns None
        when nothing matches.
        """
        state = self._state
        known = set(state.session) | set(state.claims)
        if ref in known:
            return ref
        folded = (ref or "").strip().casefold()
        if not folded:
            return None
        for candidate in known:
            if candidate.casefold() == folded:
                return candidate
        target = state.labels.get(folded)
        if target is not None and target in known:
            return target
        return None

    def get_utterance(self, utterance_id: str) -> VoiceUtterance | None:
        """Return this session's utterance with that id, or None."""
        return self._state.utterances.get(_fold_uid(utterance_id))

    def mark_shown(self, utterance_ids: Iterable[str] | None = None) -> int:
        """Record that utterances have been delivered to the agent.

        Call with the ids of the transcript lines that have entered the
        agent's thread (``Conversation.shown_utterance_ids``). ``None``
        marks everything buffered so far — for callers with no queue
        between hearing and showing (tests, offline replay). Sticky and
        idempotent; unknown ids are ignored. Returns how many flipped.
        """
        state = self._state
        if utterance_ids is None:
            targets = list(state.utterances.values())
        else:
            targets = [
                u for u in (
                    state.utterances.get(_fold_uid(uid))
                    for uid in utterance_ids
                ) if u is not None
            ]
        flipped = 0
        for u in targets:
            if not u.shown:
                u.shown = True
                flipped += 1
        return flipped

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
        shown: bool = False,
    ) -> VoiceUtterance:
        """Buffer a voice embedding (and its match metadata) under a ref.

        The metadata feeds the commit gate. Without it the embedding can
        still anchor an ``identify`` claim, but can never be committed on
        the strength of its own cloud match. The utterance gets its id
        here; ``shown`` stays False until :meth:`mark_shown`.
        """
        self._utterance_seq += 1
        utterance = VoiceUtterance(
            uid=f"u{self._utterance_seq}",
            ref=ref,
            shown=shown,
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
        self._state.utterances[utterance.uid] = utterance
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
            # Closer to a voice known NOT to be the anchor's (it sat in
            # the same bucket and disagreed with the seed) than to the
            # anchor: this is that other voice. No name, not even "Name?".
            if cand.rivals and any(
                r.shape == q.shape and float(r @ q) >= score
                for r in cand.rivals[-_MAX_SCORED_MEMBERS:]
            ):
                continue
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
                u = self.buffer_voice_embedding(best_name, embedding, **meta)
                self.on_reid_match(
                    best_name, "voice", person_id=best_id,
                    person_name=best_name, tier="high", score=best_score,
                )
                return self._attributed(VoiceAttribution(
                    label=best_name, ref=best_name, person_id=best_id,
                    person_name=best_name, certainty="confirmed",
                    source="voice_reid_match", voice_tier="high",
                    voice_score=float(best_score), utterance_id=u.uid,
                ))

        # 2. Anchor-consistent with an agent claim.
        if anchor_wins and anchor is not None:
            u = self.buffer_voice_embedding(
                anchor.ref, embedding, anchor_member=True,
                anchor_score=float(anchor_score), **meta
            )
            return self._attributed(VoiceAttribution(
                label=anchor.name, ref=anchor.ref,
                person_id=anchor.person_id, person_name=anchor.name,
                certainty="confirmed", source="agent_identify",
                voice_tier="high", voice_score=float(anchor_score),
                utterance_id=u.uid,
            ))

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
            u = self.buffer_voice_embedding(label, embedding, **meta)
            self.on_reid_match(
                label, "voice", person_id=cand_id, person_name=cand_name,
                tier="medium", score=float(cand_score),
            )
            return self._attributed(VoiceAttribution(
                label=label, ref=label, person_id=cand_id,
                person_name=cand_name, certainty="uncertain",
                source=source, voice_tier="medium",
                voice_score=float(cand_score), utterance_id=u.uid,
            ))

        # 4. Unknown — session-local grouping of anonymous voices.
        ref = self._assign_anon_ref(q, gate)
        u = self.buffer_voice_embedding(ref, embedding, **meta)
        self.on_reid_match(
            ref, "voice", person_id=None, person_name=None,
            tier="unknown", score=float(best_score) if ranked else 0.0,
        )
        return self._attributed(VoiceAttribution(
            label=ref, ref=ref, person_id=None, person_name=None,
            certainty="unknown", source="unknown", voice_tier="unknown",
            voice_score=float(best_score) if ranked else 0.0,
            utterance_id=u.uid,
        ))

    def _attributed(self, attribution: VoiceAttribution) -> VoiceAttribution:
        """Record the label a line is about to be shown under."""
        self._state.labels[attribution.label.casefold()] = attribution.ref
        return attribution

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

    def _latest_shown(self, ref: str) -> VoiceUtterance | None:
        """Most recent utterance under ``ref`` the agent has been shown."""
        person = self._session.get(ref)
        if person is None:
            return None
        for u in reversed(person.voice_utterances):
            if u.shown:
                return u
        return None

    def _extract_utterance(self, u: VoiceUtterance) -> str:
        """Move one utterance out of its bucket into a ref of its own.

        Used when a claim is about a single line in a bucket that is not
        one voice: a ``Name?`` pool, or a ref already anchored on a
        different voice. Returns the new ref.
        """
        old = self._session.get(u.ref)
        if old is not None:
            for idx, member in enumerate(old.voice_utterances):
                if member is u:
                    del old.voice_utterances[idx]
                    if idx < len(old.voice_embeddings):
                        del old.voice_embeddings[idx]
                    break
        new_ref = f"voice:{u.uid}"
        person = self._touch(new_ref)
        person.voice_embeddings.append(u.embedding)
        person.voice_utterances.append(u)
        u.ref = new_ref
        return new_ref

    def _bind_anchor(
        self,
        ref: str,
        seed: VoiceUtterance | None,
        person_id: str,
        name: str,
        corrected_from: str | None,
        *,
        explicit: bool,
    ) -> dict:
        """(Re)build the voice anchor for an agent claim on ``ref``.

        ``seed`` is the utterance the claim is about (see ``identify``).
        Other utterances under the ref are compared with it:

        - cosine ≥ anchor threshold → agrees;
        - anything lower → another voice (``other_voice`` below the
          cluster threshold, ``unanchored`` in the band between). Either
          makes the bucket *mixed*.

        Mixed and the agent did not name the utterance (``explicit``
        False): the anchor is the seed alone and is ``label_only`` —
        nothing under the ref is committed. Otherwise agreeing
        utterances join the anchor.
        """
        gate = self._gate()
        person = self._session.get(ref)
        utterances = person.voice_utterances if person is not None else []
        for u in utterances:
            u.anchor_member = False
            u.anchor_score = None
            u.anchor_seed = False
        if seed is None or not any(u is seed for u in utterances):
            self._anchors.pop(ref, None)
            return {
                "anchored": 0, "other_voice": 0, "unanchored": 0,
                "label_only": False, "seed": None,
            }

        seed.anchor_member = True
        seed.anchor_score = 1.0
        seed.anchor_seed = True
        agreeing: list[tuple[VoiceUtterance, float]] = []
        rivals: list[np.ndarray] = []
        other_voice = 0
        unanchored = 0
        for u in utterances:
            if u is seed:
                continue
            sim = (
                float(u.unit @ seed.unit)
                if u.unit.shape == seed.unit.shape else -1.0
            )
            if sim >= gate.anchor:
                agreeing.append((u, sim))
                continue
            rivals.append(u.unit)
            if sim < gate.cluster:
                other_voice += 1
            else:
                unanchored += 1

        label_only = bool(rivals) and not explicit
        vectors = [seed.unit]
        if not label_only:
            for u, sim in agreeing:
                u.anchor_member = True
                u.anchor_score = sim
            # Seed plus the most recent agreeing utterances, ≤ top-k.
            extra = max(0, gate.topk - 1)
            if extra:
                vectors += [u.unit for u, _ in agreeing][-extra:]
        # Re-insert so the newest claim is the last anchor.
        self._anchors.pop(ref, None)
        self._anchors[ref] = VoiceAnchor(
            ref=ref, person_id=person_id, name=name,
            vectors=vectors, corrected_from=corrected_from,
            seed_uid=seed.uid, label_only=label_only, rivals=rivals,
        )
        return {
            "anchored": sum(1 for u in utterances if u.anchor_member),
            "other_voice": other_voice,
            "unanchored": unanchored,
            "label_only": label_only,
            "seed": seed.uid,
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
        ref: str | None = None,
        *,
        utterance: str | None = None,
    ) -> dict:
        """Record that a session voice belongs to ``name``.

        This does NOT flush buffered embeddings to the store — the commit
        happens at ``commit_session()`` (session end). It updates the
        session claim so the commit routes correctly, and binds the claim
        to a *voice* (see :meth:`_bind_anchor`).

        Which utterance the claim is about:

        - ``utterance`` (id from the identity block) — exactly that one.
          It must have been delivered to the agent.
        - ``ref`` only — the most recent utterance under that label that
          had been delivered to the agent (:meth:`mark_shown`). Lines
          queued behind the agent's turn are never the seed.

        What the claim lands on:

        - an anonymous / passively-matched ref → that ref;
        - a ``Name?`` pool (different voices by design) → the seed
          utterance alone, moved to a ref of its own;
        - a ref already anchored to someone else → refused without
          ``utterance`` (a correction must say which line). With it: the
          anchor is replaced if the seed is that same voice ("I'm
          actually Sarah"); otherwise the seed is split off and the
          established anchor stays.

        Returns a dict the ``identify_person`` tool surfaces to the agent:
        ``{status, outcome, name, ref, person_id, embeddings_buffered,
           prior_claim_name, prior_claim_source, utterance, binding,
           voice_bound, label_only, voice_utterances_anchored,
           voice_utterances_other_voice, voice_utterances_unanchored,
           mixed_voices}`` (``ref`` is the session ref the claim is on).

        Raises:
            ValueError: unknown ref / utterance, an utterance not yet
                delivered, or an implicit re-claim of an anchored label.
        """
        state = self._state
        gate = self._gate()
        explicit = bool(utterance and str(utterance).strip())

        def _resolve() -> tuple[str, VoiceUtterance | None]:
            if explicit:
                seed = state.utterances.get(_fold_uid(utterance))
                if seed is None:
                    raise ValueError(
                        f"Utterance '{utterance}' is unknown this session. "
                        f"Use the id shown beside the speaker's label, or "
                        f"ref=<label>. Live labels: {self.live_labels()}"
                    )
                if not seed.shown:
                    raise ValueError(
                        f"Utterance '{utterance}' has not been delivered "
                        f"to you yet; nothing bound."
                    )
                return seed.ref, seed
            if not ref:
                raise ValueError("identify needs `utterance` or `ref`.")
            resolved = self.resolve_ref(ref)
            if resolved is None:
                raise ValueError(
                    f"Session ref '{ref}' is unknown — pass a label "
                    f"exactly as shown on a transcript line this session "
                    f"('Name?' and 'Name' are different labels). "
                    f"Live labels: {self.live_labels()}"
                )
            return resolved, self._latest_shown(resolved)

        # Fail on a bad ref before touching the store.
        _resolve()
        existing = await self._store.get_person_by_name(name)
        if state is not self._state:
            raise ValueError("Voice session ended; nothing bound.")
        resolved, seed = _resolve()
        is_pool = resolved.endswith("?")
        anchor = state.anchors.get(resolved)
        prior_claim = state.claims.get(resolved)
        existing_id = existing["id"] if existing is not None else None

        extract = False
        if anchor is not None:
            same_person = existing_id is not None and (
                anchor.person_id == existing_id
            )
            seed_is_anchor_voice = seed is not None and (
                seed.anchor_member
                or (
                    bool(anchor.vectors)
                    and seed.unit.shape == anchor.vectors[0].shape
                    and float(seed.unit @ anchor.vectors[0]) >= gate.anchor
                )
            )
            if same_person and (not explicit or seed_is_anchor_voice):
                # Already bound to this person: leave the anchor alone.
                return self._identify_result(
                    IdentifyOutcome.NO_OP, anchor.name, resolved,
                    anchor.person_id, prior_claim, explicit=explicit,
                    binding={
                        "anchored": sum(
                            1 for u in state.session[resolved].voice_utterances
                            if u.anchor_member
                        ) if resolved in state.session else 0,
                        "other_voice": 0, "unanchored": 0,
                        "label_only": anchor.label_only,
                        "seed": anchor.seed_uid,
                    },
                )
            if not explicit:
                raise ValueError(
                    f"'{ref}' is already bound to {anchor.name} this "
                    f"session; not moved. If a line under that label was "
                    f"really {name}, pass utterance=<id beside that line's "
                    f"label>. Same human, new name: action=\"rename\"."
                )
            extract = not seed_is_anchor_voice
        elif is_pool:
            if seed is None:
                raise ValueError(
                    f"No line under '{ref}' has been delivered to you yet; "
                    f"nothing bound. '{ref}' pools unverified voices — "
                    f"pass utterance=<id beside the line's label>."
                )
            extract = True

        # Decide outcome based on the matrix in IdentifyOutcome.
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
            target_id = await self._store.create_person(name)
            target_name = name
            if state is not self._state:
                raise ValueError("Voice session ended; nothing bound.")
            if prior_claim is None or prior_claim.person_id is None:
                outcome = IdentifyOutcome.CREATE
            else:
                outcome = IdentifyOutcome.RENAME

        # Everything below is synchronous: the claim and its anchor land
        # together or not at all.
        corrected_from = (
            prior_claim.person_id
            if prior_claim is not None
            and prior_claim.person_id
            and prior_claim.person_id != target_id
            else None
        )
        bind_ref = resolved
        if extract and seed is not None:
            if anchor is not None:
                # Split off from someone else's anchored voice: this line
                # was never believed to be them by voice.
                corrected_from = None
            bind_ref = self._extract_utterance(seed)

        # Overwrite the claim with the agent's decision.
        state.claims[bind_ref] = SessionClaim(
            person_id=target_id,
            name=target_name,
            source="agent_identify",
            match_tier="high",
            match_score=1.0,
        )
        binding = self._bind_anchor(
            bind_ref, seed, target_id, target_name, corrected_from,
            explicit=explicit,
        )

        result = self._identify_result(
            outcome, target_name, bind_ref, target_id, prior_claim,
            explicit=explicit, binding=binding,
        )
        logger.info(
            "identify: ref=%s → %s (outcome=%s, buffered=%d, prior=%s/%s, "
            "seed=%s by %s, voice anchored=%d other-voice=%d unanchored=%d"
            "%s)",
            bind_ref, target_name, outcome.value,
            result["embeddings_buffered"],
            result["prior_claim_name"], result["prior_claim_source"],
            binding["seed"], result["binding"],
            binding["anchored"], binding["other_voice"],
            binding["unanchored"],
            ", LABEL ONLY — mixed bucket, nothing will be stored"
            if binding["label_only"] else "",
        )
        if outcome is IdentifyOutcome.CORRECT:
            logger.warning(
                "identify correction: ref=%s was '%s' (%s), now '%s'",
                bind_ref, result["prior_claim_name"],
                result["prior_claim_source"], target_name,
            )
        return result

    def _identify_result(
        self,
        outcome: IdentifyOutcome,
        name: str,
        ref: str,
        person_id: str,
        prior_claim: SessionClaim | None,
        *,
        explicit: bool,
        binding: dict,
    ) -> dict:
        session_person = self._session.get(ref)
        buffered = 0
        if session_person is not None:
            buffered = (
                len(session_person.visual_embeddings)
                + len(session_person.voice_embeddings)
            )
        return {
            "status": "ok",
            "outcome": outcome.value,
            "name": name,
            "ref": ref,
            "person_id": person_id,
            "embeddings_buffered": buffered,
            "prior_claim_name": prior_claim.name if prior_claim else None,
            "prior_claim_source": prior_claim.source if prior_claim else None,
            "utterance": binding["seed"],
            "binding": "utterance" if explicit else "label",
            "voice_bound": binding["seed"] is not None,
            "label_only": bool(binding["label_only"]),
            "voice_utterances_anchored": binding["anchored"],
            "voice_utterances_other_voice": binding["other_voice"],
            "voice_utterances_unanchored": binding["unanchored"],
            "mixed_voices": (
                binding["other_voice"] + binding["unanchored"]
            ) > 0,
        }

    # --- commit at session end -----------------------------------------

    async def commit_session(
        self, session_id: str | None = None
    ) -> dict[str, dict]:
        """Detach the session, then flush what passes the commit gate.

        The session's state is swapped out synchronously before the first
        DB write: a session that starts while this one is being written
        gets fresh buffers, and nothing it hears is committed under this
        session's claims.

        Voice — "store only when certain". An utterance embedding is
        written to a person's voiceprint only if:

        (a) it carries an ``agent_identify`` claim made this session —
            anchor-consistent at the stricter *commit* threshold (→ the
            claimed person, provenance ``agent_identify``) — AND the
            anchor is corroborated: at least one utterance other than
            the seed agrees with it at that threshold and is long
            enough. One line on its own never writes a voiceprint. A
            ``label_only`` anchor (mixed bucket, no utterance named)
            commits no voice at all; or
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
        state = self._state
        self._state = _SessionState()
        self._detached.append(state)
        try:
            return await self._commit_state(state, gate, session_id)
        finally:
            self._detached.remove(state)

    async def _commit_state(
        self,
        state: _SessionState,
        gate: VoiceGate,
        session_id: str | None,
    ) -> dict[str, dict]:
        """Write one detached session. Touches ``state`` only."""
        summary: dict[str, dict] = {}

        for ref, person in list(state.session.items()):
            claim = state.claims.get(ref)
            anchor = state.anchors.get(ref)
            label_only = anchor is not None and anchor.label_only
            anchor_usable = (
                anchor is not None
                and not label_only
                and self._anchor_corroborated(person, claim, anchor, gate)
            )
            has_person = claim is not None and claim.person_id is not None
            admit_visual = has_person and claim.source in _VISUAL_ADMIT_SOURCES
            admit_voice = (
                has_person
                and claim.source in _VOICE_ADMIT_SOURCES
                and not label_only
            )
            person_id = claim.person_id if has_person else None

            voice_added = 0
            dropped_low = 0
            dropped_conflict = 0
            dropped_short = 0

            for u in person.voice_utterances:
                verdict, provenance, confidence = self._voice_commit_verdict(
                    u, claim if admit_voice else None,
                    anchor if anchor_usable else None, gate,
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
                "session=%s%s",
                ref, claim.name if has_person else None,
                voice_added, dropped_low, dropped_conflict, dropped_short,
                visual_added, visual_dropped,
                claim.source if claim else None, session_id,
                " | voice withheld: label-only claim on a mixed bucket"
                if label_only
                else " | anchor uncorroborated: seed alone is not stored"
                if anchor is not None and not anchor_usable
                else "",
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
                "voice_label_only": label_only,
            }

        return summary

    @classmethod
    def _anchor_corroborated(
        cls,
        person: SessionPerson,
        claim: SessionClaim | None,
        anchor: VoiceAnchor,
        gate: VoiceGate,
    ) -> bool:
        """True when an utterance other than the seed backs the anchor.

        "Backs" = would itself be committed as an anchor member: agrees
        at the commit threshold, long enough, no conflicting match. A
        seed is one line the agent (or the latest-shown rule) picked; if
        that pick was wrong, it must not be able to write a voiceprint
        on its own.
        """
        if claim is None or claim.source != "agent_identify":
            return False
        for u in person.voice_utterances:
            if u.anchor_seed or not u.anchor_member:
                continue
            verdict, provenance, _ = cls._voice_commit_verdict(
                u, claim, anchor, gate,
            )
            if verdict == "commit" and provenance == "agent_identify":
                return True
        return False

    @staticmethod
    def _voice_commit_verdict(
        u: VoiceUtterance,
        claim: SessionClaim | None,
        anchor: VoiceAnchor | None,
        gate: VoiceGate,
    ) -> tuple[str, str, float | None]:
        """Decide one utterance's fate at commit.

        ``claim`` is the ref's claim only if it admits voice; ``anchor``
        is the ref's anchor only if it may commit (not label-only, and
        corroborated — see ``_anchor_corroborated``). Returns
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
        # A session being committed right now must not write to the
        # merged-away / renamed id either.
        for state in self._detached:
            for ref, claim in list(state.claims.items()):
                if claim.person_id == old_person_id:
                    state.claims[ref] = SessionClaim(
                        person_id=new_person_id,
                        name=new_name if new_name is not None else claim.name,
                        source=claim.source,
                        match_tier=claim.match_tier,
                        match_score=claim.match_score,
                        established_at=claim.established_at,
                    )
            for anchor in state.anchors.values():
                if anchor.person_id == old_person_id:
                    anchor.person_id = new_person_id
        if updated:
            logger.info(
                "repoint_person: %d session claim(s) moved %s -> %s (%s)",
                updated, old_person_id, new_person_id, new_name,
            )
        return updated

    def clear_session(self) -> None:
        """Drop the current session's buffered state without committing.

        A session detached by an in-flight ``commit_session`` is not
        affected.
        """
        self._state = _SessionState()


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
