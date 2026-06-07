"""Tests for the sandbox ``photos.*`` action handlers.

These exercise the bridge between the SDK (``bb.photos.*``) and the
``PhotoStore`` — i.e. that management actions actually mutate state
rather than returning a silent stub. The handler reaches the store via
the ``boxbot.photos.search`` singleton, so each test injects a
temp-backed store into that singleton.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

import boxbot.photos.search as search_mod
from boxbot.tools._sandbox_actions import ActionContext, process_action


@pytest_asyncio.fixture
async def wired_store(photo_store, mock_config, monkeypatch):
    """Inject a temp PhotoStore as the shared search singleton.

    Depends on ``mock_config`` so store methods that read
    ``config.photos`` (e.g. ``get_storage_info``) work under test.
    """
    monkeypatch.setattr(search_mod, "_store_singleton", photo_store)
    yield photo_store


async def _act(action_type: str, **payload):
    ctx = ActionContext()
    return await process_action({"_sdk": action_type, **payload}, ctx)


@pytest.mark.asyncio
class TestPhotoMutationActions:
    async def test_delete_then_restore(self, wired_store):
        pid = await wired_store.create_photo(filename="a.jpg", source="signal")

        res = await _act("photos.delete", id=pid)
        assert res["status"] == "ok" and res["deleted"] is True
        rec = await wired_store.get_photo(pid)
        assert rec.deleted_at is not None

        res = await _act("photos.restore", id=pid)
        assert res["status"] == "ok" and res["restored"] is True
        rec = await wired_store.get_photo(pid)
        assert rec.deleted_at is None

    async def test_delete_missing_photo_errors(self, wired_store):
        res = await _act("photos.delete", id="photo_nope")
        assert res["status"] == "error"

    async def test_delete_requires_id(self, wired_store):
        res = await _act("photos.delete")
        assert res["status"] == "error"

    async def test_set_tags_replaces(self, wired_store):
        pid = await wired_store.create_photo(
            filename="b.jpg", source="signal", tags=["old"]
        )
        res = await _act("photos.set_tags", id=pid, tags=["beach", "family"])
        assert res["status"] == "ok"
        rec = await wired_store.get_photo(pid)
        assert set(rec.tags) == {"beach", "family"}

    async def test_set_tags_requires_list(self, wired_store):
        pid = await wired_store.create_photo(filename="c.jpg", source="signal")
        res = await _act("photos.set_tags", id=pid, tags="notalist")
        assert res["status"] == "error"

    async def test_update_recomputes_embedding(self, wired_store):
        pid = await wired_store.create_photo(filename="d.jpg", source="signal")
        res = await _act("photos.update", id=pid, description="a red bicycle")
        assert res["status"] == "ok" and res["updated"] is True
        rec = await wired_store.get_photo(pid)
        assert rec.description == "a red bicycle"

    async def test_slideshow_toggle(self, wired_store):
        pid = await wired_store.create_photo(
            filename="e.jpg", source="signal", in_slideshow=False
        )
        res = await _act("photos.add_to_slideshow", id=pid)
        assert res["status"] == "ok" and res["in_slideshow"] is True
        assert (await wired_store.get_photo(pid)).in_slideshow is True

        res = await _act("photos.remove_from_slideshow", id=pid)
        assert res["status"] == "ok" and res["in_slideshow"] is False
        assert (await wired_store.get_photo(pid)).in_slideshow is False

    async def test_set_person_by_index(self, wired_store):
        pid = await wired_store.create_photo(
            filename="f.jpg",
            source="signal",
            people=[{"label": "unknown person", "bbox_x": 0.1}],
        )
        res = await _act("photos.set_person", id=pid, person_index=0, name="Jacob")
        assert res["status"] == "ok"
        rec = await wired_store.get_photo(pid)
        assert rec.people[0]["label"] == "Jacob"

    async def test_set_person_out_of_range(self, wired_store):
        pid = await wired_store.create_photo(filename="g.jpg", source="signal")
        res = await _act("photos.set_person", id=pid, person_index=5, name="Jacob")
        assert res["status"] == "error"

    async def test_merge_rename_delete_tag(self, wired_store):
        p1 = await wired_store.create_photo(
            filename="h.jpg", source="signal", tags=["beech"]
        )
        # rename typo
        res = await _act("photos.rename_tag", old="beech", new="beach")
        assert res["status"] == "ok" and res["renamed"] is True
        assert "beach" in (await wired_store.get_photo(p1)).tags

        # merge a synonym into beach
        await wired_store.create_photo(
            filename="i.jpg", source="signal", tags=["seaside"]
        )
        res = await _act("photos.merge_tags", source="seaside", into="beach")
        assert res["status"] == "ok" and res["merged_count"] == 1

        # delete the tag entirely
        res = await _act("photos.delete_tag", tag="beach")
        assert res["status"] == "ok" and res["removed_count"] >= 1

    async def test_list_deleted(self, wired_store):
        pid = await wired_store.create_photo(filename="j.jpg", source="signal")
        await wired_store.soft_delete_photo(pid)
        res = await _act("photos.list_deleted")
        assert res["status"] == "ok"
        assert any(r["id"] == pid for r in res["results"])

    async def test_storage_info_shape(self, wired_store):
        await wired_store.create_photo(filename="k.jpg", source="signal")
        res = await _act("photos.storage_info")
        assert res["status"] == "ok"
        for key in ("used_bytes", "quota_bytes", "used_percent",
                    "used_gb", "quota_gb", "photo_count"):
            assert key in res
        assert res["photo_count"] == 1

    async def test_unknown_action_is_stub(self, wired_store):
        res = await _act("photos.frobnicate")
        assert res["status"] == "stub"
