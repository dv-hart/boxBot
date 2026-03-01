"""Tests for the photo system — store, search, intake pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from boxbot.photos.store import PhotoRecord, PhotoStore, StorageInfo, TagRecord


# ---------------------------------------------------------------------------
# Photo Store CRUD
# ---------------------------------------------------------------------------


class TestPhotoStoreCRUD:
    """Test photo creation, retrieval, update, and listing."""

    @pytest.mark.asyncio
    async def test_create_photo_returns_prefixed_id(self, photo_store):
        pid = await photo_store.create_photo(
            filename="test.jpg",
            source="camera",
            description="A sunset",
        )
        assert pid.startswith("photo_")

    @pytest.mark.asyncio
    async def test_create_photo_with_explicit_id(self, photo_store):
        pid = await photo_store.create_photo(
            filename="explicit.jpg",
            source="upload",
            photo_id="photo_explicit_01",
        )
        assert pid == "photo_explicit_01"

    @pytest.mark.asyncio
    async def test_get_photo_returns_record(self, photo_store):
        pid = await photo_store.create_photo(
            filename="get_test.jpg",
            source="whatsapp",
            sender="Jacob",
            description="Family photo",
            width=1920,
            height=1080,
            file_size=500000,
        )
        photo = await photo_store.get_photo(pid)
        assert photo is not None
        assert isinstance(photo, PhotoRecord)
        assert photo.filename == "get_test.jpg"
        assert photo.source == "whatsapp"
        assert photo.sender == "Jacob"
        assert photo.description == "Family photo"
        assert photo.width == 1920
        assert photo.file_size == 500000

    @pytest.mark.asyncio
    async def test_get_nonexistent_photo_returns_none(self, photo_store):
        result = await photo_store.get_photo("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_create_photo_with_tags(self, photo_store):
        pid = await photo_store.create_photo(
            filename="tagged.jpg",
            source="camera",
            tags=["sunset", "nature", "orange"],
        )
        photo = await photo_store.get_photo(pid)
        assert set(photo.tags) == {"sunset", "nature", "orange"}

    @pytest.mark.asyncio
    async def test_create_photo_with_people(self, photo_store):
        pid = await photo_store.create_photo(
            filename="people.jpg",
            source="camera",
            people=[
                {"label": "Jacob", "person_id": "p1", "bbox_x": 0.1},
                {"label": "Alice", "person_id": "p2", "bbox_x": 0.5},
            ],
        )
        photo = await photo_store.get_photo(pid)
        assert len(photo.people) == 2
        labels = {p["label"] for p in photo.people}
        assert labels == {"Jacob", "Alice"}

    @pytest.mark.asyncio
    async def test_update_photo_description(self, photo_store):
        pid = await photo_store.create_photo(
            filename="update.jpg",
            source="camera",
            description="Original",
        )
        updated = await photo_store.update_photo(pid, description="Updated desc")
        assert updated is True
        photo = await photo_store.get_photo(pid)
        assert photo.description == "Updated desc"

    @pytest.mark.asyncio
    async def test_list_photos_excludes_deleted(self, photo_store):
        p1 = await photo_store.create_photo(
            filename="keep.jpg", source="camera"
        )
        p2 = await photo_store.create_photo(
            filename="delete.jpg", source="camera"
        )
        await photo_store.soft_delete_photo(p2)

        photos = await photo_store.list_photos()
        ids = {p.id for p in photos}
        assert p1 in ids
        assert p2 not in ids

    @pytest.mark.asyncio
    async def test_list_photos_includes_deleted_when_flagged(self, photo_store):
        p1 = await photo_store.create_photo(
            filename="a.jpg", source="camera"
        )
        p2 = await photo_store.create_photo(
            filename="b.jpg", source="camera"
        )
        await photo_store.soft_delete_photo(p2)

        photos = await photo_store.list_photos(include_deleted=True)
        ids = {p.id for p in photos}
        assert p1 in ids
        assert p2 in ids


# ---------------------------------------------------------------------------
# Soft delete / restore
# ---------------------------------------------------------------------------


class TestPhotoSoftDelete:
    """Test soft delete and restore operations."""

    @pytest.mark.asyncio
    async def test_soft_delete_sets_deleted_at(self, photo_store):
        pid = await photo_store.create_photo(
            filename="delete_me.jpg", source="camera"
        )
        result = await photo_store.soft_delete_photo(pid)
        assert result is True
        photo = await photo_store.get_photo(pid)
        assert photo.deleted_at is not None
        assert photo.in_slideshow is False

    @pytest.mark.asyncio
    async def test_restore_clears_deleted_at(self, photo_store):
        pid = await photo_store.create_photo(
            filename="restore_me.jpg", source="camera"
        )
        await photo_store.soft_delete_photo(pid)
        result = await photo_store.restore_photo(pid)
        assert result is True
        photo = await photo_store.get_photo(pid)
        assert photo.deleted_at is None

    @pytest.mark.asyncio
    async def test_list_deleted_returns_only_deleted(self, photo_store):
        p1 = await photo_store.create_photo(filename="a.jpg", source="camera")
        p2 = await photo_store.create_photo(filename="b.jpg", source="camera")
        await photo_store.soft_delete_photo(p2)

        deleted = await photo_store.list_deleted()
        ids = {p.id for p in deleted}
        assert p2 in ids
        assert p1 not in ids


# ---------------------------------------------------------------------------
# Tag management
# ---------------------------------------------------------------------------


class TestTagManagement:
    """Test tag library operations — update, merge, rename, delete."""

    @pytest.mark.asyncio
    async def test_update_tags_replaces_all(self, photo_store):
        pid = await photo_store.create_photo(
            filename="tag_test.jpg", source="camera", tags=["old"]
        )
        await photo_store.update_tags(pid, ["new1", "new2"])
        photo = await photo_store.get_photo(pid)
        assert set(photo.tags) == {"new1", "new2"}
        assert "old" not in photo.tags

    @pytest.mark.asyncio
    async def test_merge_tags(self, photo_store):
        p1 = await photo_store.create_photo(
            filename="m1.jpg", source="camera", tags=["sunset"]
        )
        p2 = await photo_store.create_photo(
            filename="m2.jpg", source="camera", tags=["sundown"]
        )
        count = await photo_store.merge_tags("sundown", "sunset")
        assert count == 1
        # p2 should now have "sunset"
        photo2 = await photo_store.get_photo(p2)
        assert "sunset" in photo2.tags
        assert "sundown" not in photo2.tags

    @pytest.mark.asyncio
    async def test_rename_tag(self, photo_store):
        await photo_store.create_photo(
            filename="r.jpg", source="camera", tags=["kitty"]
        )
        result = await photo_store.rename_tag("kitty", "cat")
        assert result is True
        tags = await photo_store.list_tags()
        names = {t.name for t in tags}
        assert "cat" in names
        assert "kitty" not in names

    @pytest.mark.asyncio
    async def test_delete_tag(self, photo_store):
        await photo_store.create_photo(
            filename="d.jpg", source="camera", tags=["delete_me"]
        )
        count = await photo_store.delete_tag("delete_me")
        assert count >= 1
        tags = await photo_store.list_tags()
        names = {t.name for t in tags}
        assert "delete_me" not in names

    @pytest.mark.asyncio
    async def test_list_tags_with_counts(self, photo_store):
        await photo_store.create_photo(
            filename="c1.jpg", source="camera", tags=["nature"]
        )
        await photo_store.create_photo(
            filename="c2.jpg", source="camera", tags=["nature", "sunset"]
        )
        tags = await photo_store.list_tags()
        nature_tag = next((t for t in tags if t.name == "nature"), None)
        assert nature_tag is not None
        assert nature_tag.count == 2

    @pytest.mark.asyncio
    async def test_merge_nonexistent_tag_returns_zero(self, photo_store):
        count = await photo_store.merge_tags("nonexistent", "target")
        assert count == 0


# ---------------------------------------------------------------------------
# Slideshow
# ---------------------------------------------------------------------------


class TestSlideshow:
    """Test slideshow membership management."""

    @pytest.mark.asyncio
    async def test_default_in_slideshow_is_true(self, photo_store):
        pid = await photo_store.create_photo(
            filename="slide.jpg", source="camera"
        )
        photo = await photo_store.get_photo(pid)
        assert photo.in_slideshow is True

    @pytest.mark.asyncio
    async def test_remove_from_slideshow(self, photo_store):
        pid = await photo_store.create_photo(
            filename="remove.jpg", source="camera"
        )
        await photo_store.remove_from_slideshow(pid)
        photo = await photo_store.get_photo(pid)
        assert photo.in_slideshow is False

    @pytest.mark.asyncio
    async def test_add_back_to_slideshow(self, photo_store):
        pid = await photo_store.create_photo(
            filename="add_back.jpg", source="camera"
        )
        await photo_store.remove_from_slideshow(pid)
        await photo_store.add_to_slideshow(pid)
        photo = await photo_store.get_photo(pid)
        assert photo.in_slideshow is True

    @pytest.mark.asyncio
    async def test_get_slideshow_photos(self, photo_store):
        p1 = await photo_store.create_photo(
            filename="show.jpg", source="camera"
        )
        p2 = await photo_store.create_photo(
            filename="noshow.jpg", source="camera"
        )
        await photo_store.remove_from_slideshow(p2)

        slideshow = await photo_store.get_slideshow_photos()
        ids = {p.id for p in slideshow}
        assert p1 in ids
        assert p2 not in ids


# ---------------------------------------------------------------------------
# Photo search (hybrid)
# ---------------------------------------------------------------------------


class TestPhotoSearch:
    """Test the photo hybrid search system."""

    @pytest.mark.asyncio
    async def test_search_by_description(self, photo_store):
        from boxbot.memory.embeddings import embed

        desc = "beautiful sunset over the ocean"
        embedding = embed(desc)
        await photo_store.create_photo(
            filename="sunset.jpg",
            source="camera",
            description=desc,
            embedding=embedding,
        )
        await photo_store.create_photo(
            filename="cat.jpg",
            source="camera",
            description="a fluffy cat sleeping",
            embedding=embed("a fluffy cat sleeping"),
        )

        # Import the search function
        from boxbot.photos.search import hybrid_search

        results = await hybrid_search(photo_store, query="sunset ocean")
        assert len(results) > 0
        # Sunset photo should rank higher
        assert results[0].photo.description == desc
