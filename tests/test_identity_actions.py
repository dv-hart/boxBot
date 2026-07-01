"""Tests for ws5 identity actions — rename, merge, flag surfacing.

Covers:
- CloudStore.rename_person (incl. name-collision guard)
- CloudStore.merge_persons (embedding move, cap enforcement,
  merged_into tombstone, name resolution through the merge chain)
- EnrollmentManager.repoint_person (session-claim re-pointing)
- PhotoStore.repoint_person (photo tag re-pointing)
- scheduler.repoint_person_triggers (person-condition re-pointing)
- identify_person tool actions (rename / merge / list_flags)
- reconcile report persistence + duplicate-person to-do nudge
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio


def _vec(dim: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest_asyncio.fixture
async def cloud_store(tmp_path):
    from boxbot.perception.clouds import CloudStore

    store = CloudStore(db_path=tmp_path / "perception.db")
    await store.initialize()
    yield store
    await store.close()


# ---------------------------------------------------------------------------
# CloudStore.rename_person
# ---------------------------------------------------------------------------


class TestRenamePerson:
    @pytest.mark.asyncio
    async def test_rename_changes_name_only(self, cloud_store):
        pid = await cloud_store.create_person("Jacob")
        emb_id = await cloud_store.add_visual_embedding(pid, _vec(seed=1))
        assert emb_id

        result = await cloud_store.rename_person(pid, "Jake")
        assert result == {
            "old_name": "Jacob", "new_name": "Jake", "person_id": pid,
        }
        person = await cloud_store.get_person(pid)
        assert person["name"] == "Jake"
        # Embeddings untouched.
        assert await cloud_store.count_visual_embeddings(pid) == 1
        # Old name no longer resolves; new one does.
        assert await cloud_store.get_person_by_name("Jacob") is None
        assert (await cloud_store.get_person_by_name("Jake"))["id"] == pid

    @pytest.mark.asyncio
    async def test_rename_collision_suggests_merge(self, cloud_store):
        pid = await cloud_store.create_person("Erik")
        await cloud_store.create_person("Eric")
        with pytest.raises(ValueError, match="merge"):
            await cloud_store.rename_person(pid, "Eric")

    @pytest.mark.asyncio
    async def test_rename_missing_person_raises(self, cloud_store):
        with pytest.raises(ValueError, match="No person"):
            await cloud_store.rename_person("nope", "Jake")

    @pytest.mark.asyncio
    async def test_rename_logs_correction(self, cloud_store):
        pid = await cloud_store.create_person("Jacob")
        await cloud_store.rename_person(pid, "Jake")
        db = cloud_store._ensure_db()
        async with db.execute(
            "SELECT source FROM id_corrections"
        ) as cur:
            rows = [r["source"] async for r in cur]
        assert "rename_person" in rows


# ---------------------------------------------------------------------------
# CloudStore.merge_persons
# ---------------------------------------------------------------------------


class TestMergePersons:
    @pytest.mark.asyncio
    async def test_merge_moves_embeddings_and_tombstones(self, cloud_store):
        winner = await cloud_store.create_person("Eric")
        loser = await cloud_store.create_person("Erik")
        await cloud_store.add_visual_embedding(winner, _vec(seed=1))
        await cloud_store.add_visual_embedding(loser, _vec(seed=2))
        await cloud_store.add_visual_embedding(loser, _vec(seed=3))
        await cloud_store.add_voice_embedding(loser, _vec(192, seed=4))

        result = await cloud_store.merge_persons(loser, winner)
        assert result["visual_moved"] == 2
        assert result["voice_moved"] == 1
        assert result["winner_name"] == "Eric"
        assert result["loser_name"] == "Erik"

        # Winner owns everything; loser owns nothing.
        assert await cloud_store.count_visual_embeddings(winner) == 3
        assert await cloud_store.count_visual_embeddings(loser) == 0
        assert len(await cloud_store.get_voice_embeddings(winner)) == 1

        # Loser row soft-kept with merged_into.
        loser_row = await cloud_store.get_person(loser)
        assert loser_row["merged_into"] == winner

    @pytest.mark.asyncio
    async def test_merged_name_resolves_to_winner(self, cloud_store):
        winner = await cloud_store.create_person("Eric")
        loser = await cloud_store.create_person("Erik")
        await cloud_store.merge_persons(loser, winner)

        resolved = await cloud_store.get_person_by_name("Erik")
        assert resolved["id"] == winner
        assert resolved["name"] == "Eric"

    @pytest.mark.asyncio
    async def test_merged_person_hidden_from_list(self, cloud_store):
        winner = await cloud_store.create_person("Eric")
        loser = await cloud_store.create_person("Erik")
        await cloud_store.merge_persons(loser, winner)

        names = {p["name"] for p in await cloud_store.list_persons()}
        assert names == {"Eric"}
        names_all = {
            p["name"]
            for p in await cloud_store.list_persons(include_merged=True)
        }
        assert names_all == {"Eric", "Erik"}

    @pytest.mark.asyncio
    async def test_merge_enforces_visual_cap(self, cloud_store):
        from boxbot.perception import clouds as clouds_mod

        winner = await cloud_store.create_person("Eric")
        loser = await cloud_store.create_person("Erik")
        with patch.object(clouds_mod, "MAX_VISUAL_EMBEDDINGS", 10):
            for i in range(8):
                await cloud_store.add_visual_embedding(
                    winner, _vec(seed=100 + i)
                )
            for i in range(8):
                await cloud_store.add_visual_embedding(
                    loser, _vec(seed=200 + i)
                )
            # patch.object on the module constant doesn't change the
            # default arg already bound on _enforce_visual_cap, so call
            # merge with the cap monkeypatched at the method level.
            orig = cloud_store._enforce_visual_cap

            async def capped(person_id, max_count=10):
                return await orig(person_id, max_count=10)

            cloud_store._enforce_visual_cap = capped
            await cloud_store.merge_persons(loser, winner)

        count = await cloud_store.count_visual_embeddings(winner)
        assert count <= 10  # janitor brought 16 back under the cap

    @pytest.mark.asyncio
    async def test_merge_guards(self, cloud_store):
        a = await cloud_store.create_person("Eric")
        b = await cloud_store.create_person("Erik")
        c = await cloud_store.create_person("Sarah")
        with pytest.raises(ValueError, match="themselves"):
            await cloud_store.merge_persons(a, a)
        await cloud_store.merge_persons(b, a)
        # Already-merged loser can't merge again.
        with pytest.raises(ValueError, match="already merged"):
            await cloud_store.merge_persons(b, c)
        # Can't merge into a tombstone.
        with pytest.raises(ValueError, match="tombstone"):
            await cloud_store.merge_persons(c, b)

    @pytest.mark.asyncio
    async def test_merge_recomputes_winner_centroid(self, cloud_store):
        winner = await cloud_store.create_person("Eric")
        loser = await cloud_store.create_person("Erik")
        await cloud_store.add_visual_embedding(loser, _vec(seed=5))
        await cloud_store.merge_persons(loser, winner)
        centroids = await cloud_store.get_centroids()
        assert winner in centroids
        assert loser not in centroids


# ---------------------------------------------------------------------------
# EnrollmentManager.repoint_person
# ---------------------------------------------------------------------------


class TestEnrollmentRepoint:
    @pytest.mark.asyncio
    async def test_claims_follow_merge(self, cloud_store):
        from boxbot.perception.enrollment import EnrollmentManager

        manager = EnrollmentManager(cloud_store)
        manager.buffer_visual_embedding("Person A", _vec(seed=1))
        await cloud_store.create_person("Erik")
        result = await manager.identify("Erik", "Person A")
        old_id = result["person_id"]

        updated = manager.repoint_person(old_id, "winner-id", "Eric")
        assert updated == 1
        claim = manager.get_claim("Person A")
        assert claim.person_id == "winner-id"
        assert claim.name == "Eric"
        assert claim.source == "agent_identify"  # provenance preserved

    @pytest.mark.asyncio
    async def test_unrelated_claims_untouched(self, cloud_store):
        from boxbot.perception.enrollment import EnrollmentManager

        manager = EnrollmentManager(cloud_store)
        manager.on_reid_match(
            "Person B", "visual", person_id="p-other",
            person_name="Sarah", tier="high", score=0.9,
        )
        assert manager.repoint_person("p-missing", "p-new", "X") == 0
        assert manager.get_claim("Person B").person_id == "p-other"


# ---------------------------------------------------------------------------
# PhotoStore.repoint_person
# ---------------------------------------------------------------------------


class TestPhotoRepoint:
    @pytest_asyncio.fixture
    async def photo_store(self, tmp_path):
        from boxbot.photos.store import PhotoStore

        store = PhotoStore(db_path=tmp_path / "photos.db")
        await store.initialize()
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_repoint_updates_id_and_label(self, photo_store):
        photo_id = await photo_store.create_photo(
            filename="x.jpg", source="whatsapp",
            people=[
                {"label": "Erik", "person_id": "loser-id"},
                {"label": "Sarah", "person_id": "sarah-id"},
            ],
        )

        n = await photo_store.repoint_person(
            "loser-id", "winner-id", new_label="Eric"
        )
        assert n == 1
        photo = await photo_store.get_photo(photo_id)
        people = {p["label"]: p["person_id"] for p in photo.people}
        assert people == {"Eric": "winner-id", "Sarah": "sarah-id"}


# ---------------------------------------------------------------------------
# scheduler.repoint_person_triggers
# ---------------------------------------------------------------------------


class TestTriggerRepoint:
    @pytest.fixture
    def patch_db(self, tmp_path):
        with patch(
            "boxbot.core.scheduler.DB_PATH",
            tmp_path / "scheduler" / "scheduler.db",
        ):
            yield

    @pytest.mark.asyncio
    async def test_active_person_triggers_repointed(self, patch_db):
        from boxbot.core import scheduler

        await scheduler.init_db()
        tid = await scheduler.create_trigger(
            "remind on sight", "say hi", person="Erik",
        )
        done = await scheduler.create_trigger(
            "old one", "say bye", person="Erik",
        )
        await scheduler.cancel_trigger(done)

        n = await scheduler.repoint_person_triggers("Erik", "Eric")
        assert n == 1
        trig = await scheduler.get_trigger(tid)
        assert trig["person"] == "Eric"
        # Cancelled trigger left alone.
        assert (await scheduler.get_trigger(done))["person"] == "Erik"


# ---------------------------------------------------------------------------
# identify_person tool actions
# ---------------------------------------------------------------------------


class _FakePipeline:
    def __init__(self, cloud_store, enrollment=None):
        self.cloud_store = cloud_store
        self.enrollment = enrollment


@pytest_asyncio.fixture
async def tool_env(cloud_store, monkeypatch):
    """identify_person tool wired to a real CloudStore + enrollment.

    The real pipeline module pulls cv2 at import time (not available on
    dev boxes), so a stub module supplying ``get_pipeline`` is injected
    into sys.modules instead.
    """
    import sys
    import types

    from boxbot.perception.enrollment import EnrollmentManager
    from boxbot.tools.builtins import identify_person as ip_mod

    enrollment = EnrollmentManager(cloud_store)
    pipeline = _FakePipeline(cloud_store, enrollment)

    fake_mod = types.ModuleType("boxbot.perception.pipeline")
    fake_mod.get_pipeline = lambda: pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "boxbot.perception.pipeline", fake_mod
    )

    tool = ip_mod.IdentifyPersonTool()
    return tool, cloud_store, enrollment


def _no_side_effects(monkeypatch):
    """Silence the cross-store re-point legs not under test."""

    async def _zero(*a, **k):
        return 0

    import boxbot.core.scheduler as sched

    monkeypatch.setattr(sched, "repoint_person_triggers", _zero)

    class _FakePhotoStore:
        async def repoint_person(self, *a, **k):
            return 0

    async def _get_store():
        return _FakePhotoStore()

    import boxbot.photos.search as photo_search

    monkeypatch.setattr(photo_search, "get_store", _get_store)

    class _FakeMemoryStore:
        async def repoint_person_name(self, *a, **k):
            return 0

    async def _get_memory_store():
        return _FakeMemoryStore()

    import boxbot.tools.builtins.search_memory as search_memory

    monkeypatch.setattr(
        search_memory, "_get_memory_store", _get_memory_store
    )


class TestIdentifyPersonToolActions:
    @pytest.mark.asyncio
    async def test_default_action_is_identify(self, tool_env):
        tool, store, enrollment = tool_env
        enrollment.buffer_visual_embedding("Person A", _vec(seed=1))
        result = json.loads(await tool.execute(name="Jacob", ref="Person A"))
        assert result["outcome"] == "create"
        assert (await store.get_person_by_name("Jacob")) is not None

    @pytest.mark.asyncio
    async def test_rename_happy_path(self, tool_env, monkeypatch):
        tool, store, enrollment = tool_env
        _no_side_effects(monkeypatch)
        pid = await store.create_person("Jacob")
        # In-session claim must follow the rename.
        enrollment.buffer_visual_embedding("Person A", _vec(seed=1))
        await enrollment.identify("Jacob", "Person A")

        result = json.loads(
            await tool.execute(action="rename", name="Jacob", new_name="Jake")
        )
        assert result["status"] == "ok"
        assert result["old_name"] == "Jacob"
        assert result["new_name"] == "Jake"
        assert (await store.get_person(pid))["name"] == "Jake"
        assert result["session_claims_repointed"] == 1
        assert enrollment.get_claim("Person A").name == "Jake"

    @pytest.mark.asyncio
    async def test_rename_collision_returns_error(self, tool_env, monkeypatch):
        tool, store, _ = tool_env
        _no_side_effects(monkeypatch)
        await store.create_person("Erik")
        await store.create_person("Eric")
        result = json.loads(
            await tool.execute(action="rename", name="Erik", new_name="Eric")
        )
        assert result["status"] == "error"
        assert "merge" in result["message"]

    @pytest.mark.asyncio
    async def test_rename_requires_params(self, tool_env):
        tool, _, _ = tool_env
        result = json.loads(await tool.execute(action="rename", name="X"))
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_merge_happy_path(self, tool_env, monkeypatch):
        tool, store, enrollment = tool_env
        _no_side_effects(monkeypatch)
        winner = await store.create_person("Eric")
        loser = await store.create_person("Erik")
        await store.add_visual_embedding(loser, _vec(seed=2))
        # Session claim pointing at the loser must follow the merge.
        enrollment.on_reid_match(
            "Person A", "visual", person_id=loser,
            person_name="Erik", tier="high", score=0.9,
        )

        result = json.loads(
            await tool.execute(
                action="merge", name="Eric", duplicate_name="Erik",
            )
        )
        assert result["status"] == "ok"
        assert result["kept"] == "Eric"
        assert result["merged_away"] == "Erik"
        assert result["visual_embeddings_moved"] == 1
        assert result["session_claims_repointed"] == 1
        assert enrollment.get_claim("Person A").person_id == winner
        assert (await store.get_person(loser))["merged_into"] == winner
        # Destructive-action guidance present.
        assert "memory" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_merge_same_person_errors(self, tool_env, monkeypatch):
        tool, store, _ = tool_env
        _no_side_effects(monkeypatch)
        winner = await store.create_person("Eric")
        loser = await store.create_person("Erik")
        await store.merge_persons(loser, winner)
        # Both names now resolve to the winner — second merge is a no-op
        # error, not a self-merge crash.
        result = json.loads(
            await tool.execute(
                action="merge", name="Eric", duplicate_name="Erik",
            )
        )
        assert result["status"] == "error"
        assert "same person" in result["message"]

    @pytest.mark.asyncio
    async def test_merge_publishes_person_renamed(self, tool_env, monkeypatch):
        from boxbot.core.events import PersonRenamed, get_event_bus

        tool, store, _ = tool_env
        _no_side_effects(monkeypatch)
        await store.create_person("Eric")
        await store.create_person("Erik")

        seen: list[PersonRenamed] = []

        async def handler(event: PersonRenamed) -> None:
            seen.append(event)

        bus = get_event_bus()
        bus.subscribe(PersonRenamed, handler)
        try:
            await tool.execute(
                action="merge", name="Eric", duplicate_name="Erik",
            )
        finally:
            bus.unsubscribe(PersonRenamed, handler)

        assert len(seen) == 1
        assert seen[0].old_name == "Erik"
        assert seen[0].new_name == "Eric"
        assert seen[0].merged_from_id  # set on merge

    @pytest.mark.asyncio
    async def test_list_flags_no_report(self, tool_env, tmp_path, monkeypatch):
        from boxbot.perception import reconcile

        tool, _, _ = tool_env
        monkeypatch.setattr(
            reconcile, "REPORT_PATH", tmp_path / "missing.json"
        )
        result = json.loads(await tool.execute(action="list_flags"))
        assert result["status"] == "ok"
        assert "No identity-reconcile report" in result["message"]

    @pytest.mark.asyncio
    async def test_list_flags_surfaces_duplicates(
        self, tool_env, tmp_path, monkeypatch
    ):
        from boxbot.perception import reconcile

        report_path = tmp_path / "id-reconcile-latest.json"
        report_path.write_text(json.dumps({
            "ran_at": "2026-06-09T03:00:00+00:00",
            "audit_only": True,
            "persons": 4,
            "outliers": [{"embedding_id": "e1"}],
            "mislabels": [],
            "duplicate_persons": [{
                "a": "Eric", "a_id": "id-a", "b": "Erik", "b_id": "id-b",
                "name_distance": 1, "centroid_sim": 0.91,
                "reason": "similar names, similar faces",
            }],
            "judge": {"calls": 1, "duplicate_persons": [], "clusters": []},
        }))
        monkeypatch.setattr(reconcile, "REPORT_PATH", report_path)

        tool, _, _ = tool_env
        result = json.loads(await tool.execute(action="list_flags"))
        assert result["status"] == "ok"
        assert len(result["duplicate_persons"]) == 1
        assert result["duplicate_persons"][0]["a"] == "Eric"
        assert result["outlier_count"] == 1
        assert "merge" in result["message"]


# ---------------------------------------------------------------------------
# Reconcile report persistence + to-do nudge
# ---------------------------------------------------------------------------


class TestReconcileFlagSurface:
    @pytest.mark.asyncio
    async def test_run_persists_report(self, cloud_store, tmp_path):
        from boxbot.perception.reconcile import run_id_reconcile

        await cloud_store.create_person("Eric")
        await cloud_store.create_person("Erik")
        path = tmp_path / "latest.json"
        report = await run_id_reconcile(
            cloud_store=cloud_store, audit_only=True, report_path=path,
        )
        assert len(report["duplicate_persons"]) == 1
        on_disk = json.loads(path.read_text())
        assert on_disk["duplicate_persons"] == report["duplicate_persons"]
        assert on_disk["ran_at"]

        from boxbot.perception.reconcile import load_latest_report

        assert load_latest_report(path)["persons"] == 2

    @pytest.mark.asyncio
    async def test_nudge_creates_todo_once(self, tmp_path):
        from boxbot.perception import reconcile
        from boxbot.core import scheduler

        report = {
            "duplicate_persons": [{
                "a": "Eric", "b": "Erik",
                "name_distance": 1, "centroid_sim": 0.9,
                "reason": "similar names",
            }],
        }
        with patch(
            "boxbot.core.scheduler.DB_PATH",
            tmp_path / "scheduler" / "scheduler.db",
        ):
            await scheduler.init_db()
            assert await reconcile.nudge_duplicate_todos(report) == 1
            # Second night, same flag → no duplicate to-do.
            assert await reconcile.nudge_duplicate_todos(report) == 0
            todos = await scheduler.list_todos(status="pending")
            assert len(todos) == 1
            assert "[id-reconcile]" in todos[0]["description"]
            assert "Eric" in todos[0]["description"]
            # Completed to-dos also suppress re-nudging.
            await scheduler.complete_todo(todos[0]["id"])
            assert await reconcile.nudge_duplicate_todos(report) == 0

    @pytest.mark.asyncio
    async def test_nudge_empty_report_noop(self):
        from boxbot.perception.reconcile import nudge_duplicate_todos

        assert await nudge_duplicate_todos({"duplicate_persons": []}) == 0
