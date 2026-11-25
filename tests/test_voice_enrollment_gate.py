"""Speaker labelling + voice-enrollment gate ("store only when certain").

Reconstructs the 2026-11-24 incident synthetically: an enrolled adult
(weak cloud, so his utterances straddle the confirmed threshold) and an
unenrolled child talk to the box in the same session, interleaved and
sometimes over each other; the child says his name mid-session and the
agent calls ``identify_person(name="Erik", ref="Speaker A")``.

Before the fix every utterance that missed the adult's threshold was
shown as the child, and the whole anonymous bucket (17 embeddings, both
voices) was written to the child's voiceprint.

Synthetic voices: an utterance is ``q·base + sqrt(1-q²)·noise`` with
fresh noise per utterance, so two utterances of one speaker have cosine
≈ q₁·q₂. q = 0.8 gives ≈ 0.64 — the measured same-speaker clip-to-clip
figure (docs/voice-id-redesign.md). The two bases share a 0.2 component,
the measured cross-speaker cosine.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import numpy as np
import pytest
import pytest_asyncio

from boxbot.core.events import TranscriptReady, VoiceSessionEnded, get_event_bus
from boxbot.perception.clouds import CloudStore
from boxbot.perception.enrollment import EnrollmentManager, VoiceGate
from boxbot.perception.voice_reid import VoiceReID

DIM = 192
LONG = 3.0   # seconds — passes the duration gate
SHORT = 0.8  # seconds — fails it


def _unit(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    return v / np.linalg.norm(v)


class Voices:
    """Deterministic two-speaker utterance generator."""

    def __init__(self, seed: int = 20261124) -> None:
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.adult = _unit(self._rng.standard_normal(DIM))
        other = self._rng.standard_normal(DIM)
        other -= (other @ self.adult) * self.adult
        other = _unit(other)
        # cos(adult, child) = 0.2
        self.child = _unit(0.2 * self.adult + np.sqrt(1 - 0.04) * other)

    def utt(self, base: np.ndarray, quality: float) -> np.ndarray:
        noise = self._rng.standard_normal(DIM)
        noise -= (noise @ base) * base
        noise = _unit(noise)
        return _unit(quality * base + np.sqrt(1 - quality**2) * noise)

    def mixed(self) -> np.ndarray:
        """Both talking in one VAD utterance."""
        return _unit(self.utt(self.adult, 0.8) + self.utt(self.child, 0.8))


@pytest_asyncio.fixture
async def store(tmp_path):
    s = CloudStore(db_path=tmp_path / "gate.db")
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
def manager(store):
    # Explicit gate = the shipped defaults, independent of any config.
    return EnrollmentManager(store, voice_gate=VoiceGate())


async def _hear(manager, store, emb, duration=LONG):
    """What voice.py does per utterance: rank against clouds, attribute."""
    ranked = VoiceReID().rank_clouds(emb, await store.get_voice_clouds())
    return manager.attribute_voice(emb, ranked, duration=duration)


async def _rows(store, person_id):
    db = store._ensure_db()
    async with db.execute(
        "SELECT embedding, provenance, session_id, source_ref "
        "FROM voice_embeddings WHERE person_id = ?",
        (person_id,),
    ) as cur:
        return [dict(r) async for r in cur]


async def _enroll_weak_adult(store, voices):
    """Jacob with a deliberately weak cloud (3 noisy clips)."""
    pid = await store.create_person("Jacob")
    for _ in range(3):
        await store.add_voice_embedding(
            pid, voices.utt(voices.adult, 0.7), provenance="seed",
        )
    return pid


# ---------------------------------------------------------------------------
# The incident
# ---------------------------------------------------------------------------


async def _replay_incident(manager, store, seed):
    """Run the incident session; return everything the asserts need."""
    v = Voices(seed)
    jacob = await _enroll_weak_adult(store, v)
    out = {
        "jacob": jacob,
        "adult": [],        # (embedding, attribution)
        "child": [],        # (embedding, attribution, duration)
        "mixed": [],        # (embedding, attribution)
    }

    async def adult(quality):
        e = v.utt(v.adult, quality)
        a = await _hear(manager, store, e)
        out["adult"].append((e, a))
        return a

    async def child(quality=0.8, duration=LONG):
        e = v.utt(v.child, quality)
        a = await _hear(manager, store, e, duration)
        out["child"].append((e, a, duration))
        return a

    # Before anyone is named: adult strong / maybe-band / lost, child x2.
    out["pre"] = [
        await adult(0.95), await child(), await adult(0.72),
        await adult(0.5), await child(),
    ]
    # "I'm Erik" → identify_person(name="Erik", ref="Speaker A")
    out["intro"] = await child()
    out["identify"] = await manager.identify("Erik", "Speaker A")
    out["erik"] = out["identify"]["person_id"]

    # Interleaved after the claim.
    out["post_child"] = []
    for quality in (0.95, 0.72, 0.5, 0.72, 0.5, 0.95, 0.6, 0.72):
        await adult(quality)
        out["post_child"].append(await child())
    # Child, too short to trust.
    out["short"] = await child(duration=SHORT)
    # Both talking / finishing each other's sentences.
    for _ in range(4):
        e = v.mixed()
        out["mixed"].append((e, await _hear(manager, store, e)))

    out["summary"] = await manager.commit_session(session_id="voice_incident")
    out["erik_rows"] = await _rows(store, out["erik"])
    out["jacob_rows"] = await _rows(store, jacob)
    return out


class TestIncidentReplay:
    @pytest.mark.asyncio
    async def test_what_the_agent_sees(self, manager, store):
        r = await _replay_incident(manager, store, seed=20261124)

        a_strong, c1, a_maybe, a_lost, c2 = r["pre"]
        assert (a_strong.label, a_strong.certainty) == ("Jacob", "confirmed")
        assert (c1.label, c1.certainty) == ("Speaker A", "unknown")
        assert (a_maybe.label, a_maybe.certainty) == ("Jacob?", "uncertain")
        # The adult missing his own cloud is a distinct anonymous voice —
        # not the child's ref.
        assert a_lost.certainty == "unknown"
        assert a_lost.label.startswith("Speaker ")
        assert a_lost.label != "Speaker A"
        assert c2.label == "Speaker A"
        assert r["intro"].label == "Speaker A"

        assert r["identify"]["outcome"] == "create"
        assert r["identify"]["voice_utterances_anchored"] == 3
        assert r["identify"]["mixed_voices"] is False

        # After the claim the child's lines carry his name…
        assert [a.label for a in r["post_child"]] == ["Erik"] * 8
        assert {a.source for a in r["post_child"]} == {"agent_identify"}
        assert r["short"].label == "Erik"
        # …and the adult's never do.
        adult_labels = [a.label for _, a in r["adult"]]
        assert not any("Erik" in label for label in adult_labels)
        assert "Speaker A" not in adult_labels
        assert {"Jacob", "Jacob?"} <= set(adult_labels)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("seed", [20261124, *range(1, 12)])
    async def test_only_the_childs_certain_utterances_are_stored(
        self, manager, store, seed,
    ):
        r = await _replay_incident(manager, store, seed)
        stored = {row["embedding"] for row in r["erik_rows"]}

        # Adult utterances are never labelled as the child…
        for _, a in r["adult"]:
            assert a.person_name != "Erik" and "Erik" not in a.label
        # …and nothing of his, and nothing overlapped, reaches the child.
        for e, _ in r["adult"] + r["mixed"]:
            assert e.tobytes() not in stored

        # What is stored is exactly the child's long utterances that were
        # certain: anchor-consistent at the commit threshold.
        child_long = {
            e.tobytes() for e, _, duration in r["child"] if duration >= 2.0
        }
        assert stored <= child_long
        assert len(stored) >= 8          # of 11 — most of them
        assert {row["session_id"] for row in r["erik_rows"]} == {
            "voice_incident"
        }
        assert {row["provenance"] for row in r["erik_rows"]} == {
            "agent_identify"
        }
        # The adult's cloud is untouched: a passive match never grows it.
        assert len(r["jacob_rows"]) == 3

        # Accounting: every utterance is either committed or counted.
        summary = r["summary"]
        total = len(r["adult"]) + len(r["child"]) + len(r["mixed"])
        committed = sum(i["voice_added"] for i in summary.values())
        dropped = sum(
            i["voice_dropped_low"] + i["voice_dropped_conflict"]
            + i["voice_dropped_short"]
            for i in summary.values()
        )
        assert committed == len(stored)
        assert committed + dropped == total
        assert summary["Speaker A"]["voice_added"] == committed
        assert summary["Speaker A"]["voice_dropped_short"] == 1
        assert manager.get_session_refs() == []
        assert manager.get_anchor("Speaker A") is None

    @pytest.mark.asyncio
    async def test_old_style_mixed_bucket_binds_only_the_claiming_voice(
        self, manager, store,
    ):
        """The exact pre-fix shape: both voices already under one ref."""
        v = Voices()
        jacob = await _enroll_weak_adult(store, v)

        adult = [v.utt(v.adult, 0.8) for _ in range(5)]
        child = [v.utt(v.child, 0.9) for _ in range(4)]
        order = [adult[0], child[0], adult[1], adult[2], child[1],
                 adult[3], child[2], adult[4], child[3]]  # child speaks last
        for e in order:
            manager.buffer_voice_embedding("Speaker A", e, duration=LONG)

        result = await manager.identify("Erik", "Speaker A")
        assert result["voice_utterances_anchored"] == 4
        assert result["voice_utterances_other_voice"] == 5
        assert result["mixed_voices"] is True

        summary = await manager.commit_session(session_id="s1")
        erik = result["person_id"]
        stored = {r["embedding"] for r in await _rows(store, erik)}
        assert stored == {e.tobytes() for e in child}
        assert summary["Speaker A"]["voice_added"] == 4
        assert summary["Speaker A"]["voice_dropped_low"] == 5
        assert len(await _rows(store, jacob)) == 3


# ---------------------------------------------------------------------------
# Labelling rules
# ---------------------------------------------------------------------------


class TestAttribution:
    @pytest.mark.asyncio
    async def test_own_match_beats_claim(self, manager, store):
        """A confirmed cloud match wins even right after a claim on the
        bucket the previous utterance sat in."""
        v = Voices()
        await _enroll_weak_adult(store, v)
        await _hear(manager, store, v.utt(v.child, 0.8))
        await manager.identify("Erik", "Speaker A")

        a = await _hear(manager, store, v.utt(v.adult, 0.95))
        assert (a.label, a.person_name, a.source) == (
            "Jacob", "Jacob", "voice_reid_match",
        )

    @pytest.mark.asyncio
    async def test_maybe_band_is_uncertain_not_claimed_name(
        self, manager, store,
    ):
        v = Voices()
        jacob = await _enroll_weak_adult(store, v)
        await _hear(manager, store, v.utt(v.child, 0.8))
        await manager.identify("Erik", "Speaker A")

        a = await _hear(manager, store, v.utt(v.adult, 0.72))
        assert a.label == "Jacob?"
        assert a.certainty == "uncertain"
        assert a.person_id == jacob  # candidate only
        assert a.voice_tier == "medium"

    @pytest.mark.asyncio
    async def test_distinct_anonymous_voices_get_distinct_refs(
        self, manager, store,
    ):
        v = Voices()
        labels = []
        for base in (v.adult, v.child, v.adult, v.child, v.adult):
            labels.append((await _hear(manager, store, v.utt(base, 0.8))).label)
        assert labels == [
            "Speaker A", "Speaker B", "Speaker A", "Speaker B", "Speaker A",
        ]

    @pytest.mark.asyncio
    async def test_anonymous_refs_are_capped(self, store):
        manager = EnrollmentManager(store, voice_gate=VoiceGate(max_refs=2))
        rng = np.random.default_rng(3)
        labels = [
            manager.attribute_voice(_unit(rng.standard_normal(DIM)), []).label
            for _ in range(4)
        ]
        assert labels == ["Speaker A", "Speaker B", "Speaker ?", "Speaker ?"]

    @pytest.mark.asyncio
    async def test_weak_anchor_match_is_uncertain_and_not_stored(
        self, manager, store,
    ):
        v = Voices()
        await _hear(manager, store, v.utt(v.child, 0.8))
        result = await manager.identify("Erik", "Speaker A")

        # cos to the anchor ≈ 0.8 · 0.6 = 0.48 → maybe band.
        a = await _hear(manager, store, v.utt(v.child, 0.6))
        assert (a.label, a.certainty, a.source) == (
            "Erik?", "uncertain", "session_anchor",
        )
        summary = await manager.commit_session(session_id="s")
        assert summary["Erik?"]["voice_added"] == 0
        assert len(await _rows(store, result["person_id"])) == 1

    @pytest.mark.asyncio
    async def test_correction_binds_voice_and_outranks_the_wrong_match(
        self, manager, store,
    ):
        """"I'm actually Sarah": her voice matched Jacob's cloud. After the
        correction her voice is Sarah; Jacob's own voice is still Jacob."""
        v = Voices()
        jacob = await store.create_person("Jacob")
        # Jacob's cloud is polluted with clips of Sarah's voice.
        for _ in range(3):
            await store.add_voice_embedding(jacob, v.utt(v.child, 0.9))
            await store.add_voice_embedding(jacob, v.utt(v.adult, 0.9))
        sarah_clip = v.utt(v.child, 0.8)

        first = await _hear(manager, store, sarah_clip)
        assert first.label == "Jacob"  # the mislabel being corrected

        result = await manager.identify("Sarah", "Jacob")
        assert result["outcome"] == "rename"  # Sarah is new to the store
        assert result["prior_claim_name"] == "Jacob"

        again = await _hear(manager, store, sarah_clip)
        assert (again.label, again.source) == ("Sarah", "agent_identify")
        real_jacob = await _hear(manager, store, v.utt(v.adult, 0.9))
        assert real_jacob.label == "Jacob"

        summary = await manager.commit_session(session_id="s")
        assert len(await _rows(store, result["person_id"])) == 2
        # Real Jacob's line sat under a ref now claimed for Sarah: dropped.
        assert summary["Jacob"]["voice_dropped_conflict"] == 1

    @pytest.mark.asyncio
    async def test_identify_resolves_displayed_labels(self, manager, store):
        v = Voices()
        await _hear(manager, store, v.utt(v.child, 0.8))
        await manager.identify("Erik", "Speaker A")
        a = await _hear(manager, store, v.utt(v.child, 0.8))
        assert a.label == "Erik" and a.ref == "Speaker A"

        # The agent passes what it sees in the transcript.
        assert manager.resolve_ref("Erik") == "Speaker A"
        assert manager.resolve_ref("speaker a") == "Speaker A"
        again = await manager.identify("Erik", "Erik")
        assert again["ref"] == "Speaker A"
        assert again["outcome"] == "no_op"
        with pytest.raises(ValueError, match="Session ref"):
            await manager.identify("Erik", "Speaker Z")


# ---------------------------------------------------------------------------
# Commit gate (b): self-matched utterances under an admitting claim
# ---------------------------------------------------------------------------


class TestCommitGate:
    @pytest.mark.asyncio
    async def test_self_match_needs_threshold_and_margin(self, manager, store):
        jacob = await store.create_person("Jacob")
        carina = await store.create_person("Carina")
        rng = np.random.default_rng(9)

        def emb():
            return _unit(rng.standard_normal(DIM))

        clear, narrow, low, other, seed = emb(), emb(), emb(), emb(), emb()
        kw = {"duration": LONG}
        manager.buffer_voice_embedding(
            "Jacob", clear, best_person_id=jacob, best_score=0.70,
            runner_up_score=0.25, **kw)
        manager.buffer_voice_embedding(   # margin 0.04 < 0.10
            "Jacob", narrow, best_person_id=jacob, best_score=0.58,
            runner_up_score=0.54, **kw)
        manager.buffer_voice_embedding(   # below confirmed
            "Jacob", low, best_person_id=jacob, best_score=0.50,
            runner_up_score=0.20, **kw)
        manager.buffer_voice_embedding(   # looks like Carina
            "Jacob", other, best_person_id=carina, best_score=0.60,
            runner_up_score=0.30, **kw)
        manager.buffer_voice_embedding(   # the claim's own utterance
            "Jacob", seed, best_person_id=jacob, best_score=0.62,
            runner_up_score=0.20, **kw)
        await manager.identify("Jacob", "Jacob")

        summary = await manager.commit_session(session_id="s")
        rows = await _rows(store, jacob)
        by_emb = {r["embedding"]: r["provenance"] for r in rows}
        assert by_emb == {
            seed.tobytes(): "agent_identify",   # (a) anchor
            clear.tobytes(): "voice_reid",      # (b) own match + margin
        }
        info = summary["Jacob"]
        assert info["voice_added"] == 2
        assert info["voice_dropped_low"] == 2
        assert info["voice_dropped_conflict"] == 1
        assert await _rows(store, carina) == []

    @pytest.mark.asyncio
    async def test_passive_match_never_commits(self, manager, store):
        v = Voices()
        jacob = await _enroll_weak_adult(store, v)
        for _ in range(3):
            a = await _hear(manager, store, v.utt(v.adult, 0.95))
            assert a.label == "Jacob"
        summary = await manager.commit_session(session_id="s")
        assert summary["Jacob"]["voice_added"] == 0
        assert len(await _rows(store, jacob)) == 3


# ---------------------------------------------------------------------------
# Store: session traceability, undo, migration, eviction
# ---------------------------------------------------------------------------


class TestStoreSession:
    @pytest.mark.asyncio
    async def test_delete_by_session_removes_exactly_that_session(self, store):
        rng = np.random.default_rng(1)
        erik = await store.create_person("Erik")
        jacob = await store.create_person("Jacob")
        keep = [_unit(rng.standard_normal(DIM)) for _ in range(2)]
        bad = [_unit(rng.standard_normal(DIM)) for _ in range(5)]
        await store.add_voice_embedding(erik, keep[0], session_id="good")
        await store.add_voice_embedding(erik, keep[1])  # no session
        for e in bad:
            await store.add_voice_embedding(erik, e, session_id="bad")
        await store.add_voice_embedding(jacob, bad[0], session_id="bad")
        await store.recompute_voice_centroid(erik)

        assert await store.delete_voice_embeddings_by_session(erik, "bad") == 5

        left = {r["embedding"] for r in await _rows(store, erik)}
        assert left == {e.tobytes() for e in keep}
        # Other people's rows from that session are not touched.
        assert len(await _rows(store, jacob)) == 1
        # Centroid rebuilt from what is left.
        centroid = (await store.get_voice_centroids())[erik][1]
        expected = _unit(np.mean(keep, axis=0))
        assert np.allclose(centroid, expected, atol=1e-5)
        # Idempotent.
        assert await store.delete_voice_embeddings_by_session(erik, "bad") == 0

    @pytest.mark.asyncio
    async def test_delete_last_rows_clears_centroid(self, store):
        erik = await store.create_person("Erik")
        e = _unit(np.random.default_rng(2).standard_normal(DIM))
        await store.add_voice_embedding(erik, e, session_id="only")
        await store.recompute_voice_centroid(erik)
        assert erik in await store.get_voice_centroids()

        assert await store.delete_voice_embeddings_by_session(erik, "only") == 1
        assert erik not in await store.get_voice_centroids()
        with pytest.raises(ValueError):
            await store.delete_voice_embeddings_by_session(erik, "")

    @pytest.mark.asyncio
    async def test_migration_adds_session_id_to_old_db(self, tmp_path):
        path = tmp_path / "old.db"
        async with aiosqlite.connect(str(path)) as db:
            await db.executescript(
                """
                CREATE TABLE persons (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE,
                    first_seen TEXT NOT NULL, last_seen TEXT NOT NULL,
                    is_user INTEGER NOT NULL DEFAULT 0, merged_into TEXT);
                CREATE TABLE voice_embeddings (
                    id TEXT PRIMARY KEY, person_id TEXT NOT NULL,
                    embedding BLOB NOT NULL, timestamp TEXT NOT NULL,
                    provenance TEXT NOT NULL DEFAULT 'legacy',
                    confidence REAL, source_ref TEXT);
                INSERT INTO persons VALUES ('p1','Jacob','t','t',0,NULL);
                """
            )
            await db.execute(
                "INSERT INTO voice_embeddings "
                "(id, person_id, embedding, timestamp) VALUES (?,?,?,?)",
                ("e1", "p1", np.ones(DIM, dtype=np.float32).tobytes(), "t"),
            )
            await db.commit()

        s = CloudStore(db_path=path)
        await s.initialize()
        try:
            rows = await _rows(s, "p1")
            assert len(rows) == 1 and rows[0]["session_id"] is None
            await s.add_voice_embedding(
                "p1", np.ones(DIM, dtype=np.float32), session_id="s9",
            )
            assert await s.delete_voice_embeddings_by_session("p1", "s9") == 1
            assert len(await _rows(s, "p1")) == 1
        finally:
            await s.close()

    @pytest.mark.asyncio
    async def test_cap_evicts_non_anchors_before_old_anchors(self, store):
        """Old agent-confirmed rows used to go first (age term)."""
        v = Voices()
        pid = await store.create_person("Jacob")
        anchors = [v.utt(v.adult, 0.8) for _ in range(4)]
        for e in anchors:  # oldest rows
            await store.add_voice_embedding(
                pid, e, provenance="agent_identify",
            )
        for _ in range(6):
            await store.add_voice_embedding(
                pid, v.utt(v.adult, 0.8), provenance="voice_reid",
            )
        await store._enforce_voice_cap(pid, max_count=8)

        rows = await _rows(store, pid)
        assert len(rows) == 7  # low-water = 90% of cap
        kept = {r["embedding"] for r in rows}
        assert {e.tobytes() for e in anchors} <= kept

    @pytest.mark.asyncio
    async def test_cap_still_holds_when_everything_is_an_anchor(self, store):
        v = Voices()
        pid = await store.create_person("Jacob")
        for _ in range(10):
            await store.add_voice_embedding(
                pid, v.utt(v.adult, 0.8), provenance="agent_identify",
            )
        await store._enforce_voice_cap(pid, max_count=8)
        assert len(await _rows(store, pid)) == 7


# ---------------------------------------------------------------------------
# End to end: voice adapter → identify_person tool → session end → undo
# ---------------------------------------------------------------------------


class TestVoiceAdapterEndToEnd:
    @pytest_asyncio.fixture
    async def identity(self, store):
        from boxbot.perception.identity import IdentityService

        svc = IdentityService(cloud_store=store)
        await svc.start()
        yield svc
        await svc.stop()

    def _session(self, embeddings):
        from boxbot.communication.stt import STTResult
        from boxbot.communication.voice import VoiceSession, VoiceSessionState
        from boxbot.core.config import VoiceConfig

        mic = MagicMock()
        mic.set_led_pattern = AsyncMock()
        speaker = MagicMock()
        speaker.is_playing = False
        config = VoiceConfig()
        config.diarization.enabled = False  # production setting
        session = VoiceSession(mic, speaker, config)
        session._state = VoiceSessionState.ACTIVE
        session._conversation_id = "voice_e2e"
        session._stt = MagicMock()
        session._stt.transcribe = AsyncMock(
            return_value=STTResult(text="hello there", language="en", words=[])
        )
        session._diarizer = MagicMock()
        session._diarizer.embed_utterance = AsyncMock(side_effect=embeddings)
        session._diarizer_loaded = True
        return session

    @staticmethod
    def _utterance(seconds=LONG):
        from boxbot.communication.audio_capture import Utterance

        return Utterance(
            audio=b"\x00" * int(seconds * 16000 * 2),
            duration=seconds, sample_rate=16000,
            timestamp_start=1.0, timestamp_end=1.0 + seconds,
        )

    @pytest.mark.asyncio
    async def test_transcript_block_tool_commit_and_undo(
        self, identity, store, mock_config, monkeypatch,
    ):
        import sys

        from boxbot.tools.builtins.identify_person import IdentifyPersonTool

        monkeypatch.setitem(sys.modules, "boxbot.perception.pipeline", None)

        v = Voices()
        jacob = await _enroll_weak_adult(store, v)
        script = [
            ("adult", v.utt(v.adult, 0.95)),
            ("child", v.utt(v.child, 0.9)),      # "I'm Erik" → identify
            ("adult", v.utt(v.adult, 0.72)),
            ("child", v.utt(v.child, 0.9)),
            ("adult", v.utt(v.adult, 0.5)),
            ("adult", v.utt(v.adult, 0.95)),
        ]
        session = self._session([e for _, e in script])

        events: list[TranscriptReady] = []
        bus = get_event_bus()
        bus.subscribe(TranscriptReady, lambda e: events.append(e))

        tool = IdentifyPersonTool()
        tool_result = None
        for i, _ in enumerate(script):
            await session._on_utterance(self._utterance())
            if i == 1:
                tool_result = json.loads(
                    await tool.execute(name="Erik", ref="Speaker A")
                )

        prefixes = [e.transcript.split("]:")[0] + "]" for e in events]
        assert prefixes == [
            "[Jacob]", "[Speaker A]", "[Jacob?]", "[Erik]",
            prefixes[4], "[Jacob]",
        ]
        # The adult's weakest line is an anonymous voice — not the child,
        # not the child's ref, and not stuck on the previous speaker.
        assert prefixes[4].startswith("[Speaker ")
        assert prefixes[4] != "[Speaker A]"

        # Header data and transcript prefix agree: the block is keyed by
        # the very label the line carries, and says how sure we are.
        for event, prefix in zip(events, prefixes):
            assert list(event.speaker_identities) == [prefix[1:-1]]
            (info,) = event.speaker_identities.values()
            assert info["label"] == prefix[1:-1]
        certainties = [
            next(iter(e.speaker_identities.values()))["certainty"]
            for e in events
        ]
        assert certainties == [
            "confirmed", "unknown", "uncertain", "confirmed", "unknown",
            "confirmed",
        ]
        maybe = events[2].speaker_identities["Jacob?"]
        assert maybe["person_name"] == "Jacob" and maybe["person_id"] == jacob

        # Tool told the agent what it bound.
        assert tool_result["voice_utterances_anchored"] == 1
        assert tool_result["mixed_voices"] is False
        assert "most recent 'Speaker A' utterance" in tool_result["message"]

        # Session end → commit, stamped with the voice session id.
        await bus.publish(VoiceSessionEnded(conversation_id="voice_e2e"))
        erik = (await store.get_person_by_name("Erik"))["id"]
        rows = await _rows(store, erik)
        assert {r["embedding"] for r in rows} == {
            e.tobytes() for who, e in script if who == "child"
        }
        assert {r["session_id"] for r in rows} == {"voice_e2e"}
        assert len(await _rows(store, jacob)) == 3

        # Undo the session.
        assert await store.delete_voice_embeddings_by_session(
            erik, "voice_e2e"
        ) == 2
        assert await _rows(store, erik) == []

    @pytest.mark.asyncio
    async def test_tool_warns_on_mixed_bucket(self, identity, store):
        from boxbot.tools.builtins.identify_person import IdentifyPersonTool

        v = Voices()
        enrollment = identity.enrollment
        for base in (v.adult, v.adult, v.child):
            enrollment.buffer_voice_embedding("Speaker A", v.utt(base, 0.8))

        result = json.loads(
            await IdentifyPersonTool().execute(name="Erik", ref="Speaker A")
        )
        assert result["mixed_voices"] is True
        assert result["voice_utterances_anchored"] == 1
        assert result["voice_utterances_other_voice"] == 2
        assert "DIFFERENT voice" in result["message"]
        assert "ask again" in result["message"]
