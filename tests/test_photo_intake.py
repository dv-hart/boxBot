"""Tests for photo intake hardening — sniffing, size/queue/quota caps."""

from __future__ import annotations

import io

import pytest
import pytest_asyncio
from PIL import Image

from boxbot.photos.imageutil import sniff_image_file, sniff_image_mime
from boxbot.photos.intake import IntakePipeline


def _jpeg_bytes(w: int = 16, h: int = 16) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 50, 50)).save(buf, format="JPEG")
    return buf.getvalue()


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 200, 10)).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Content sniffing
# ---------------------------------------------------------------------------


class TestSniff:
    def test_sniff_jpeg(self):
        assert sniff_image_mime(_jpeg_bytes()) == "image/jpeg"

    def test_sniff_png(self):
        assert sniff_image_mime(_png_bytes()) == "image/png"

    def test_sniff_gif(self):
        assert sniff_image_mime(b"GIF89a" + b"\x00" * 16) == "image/gif"

    def test_sniff_webp(self):
        assert sniff_image_mime(b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 4) == "image/webp"

    def test_sniff_rejects_text(self):
        assert sniff_image_mime(b"#!/bin/sh\necho hi\n") is None

    def test_sniff_rejects_short(self):
        assert sniff_image_mime(b"\xff\xd8") is None

    def test_sniff_file(self, tmp_path):
        p = tmp_path / "x.bin"  # extension lies; content is a real jpeg
        p.write_bytes(_jpeg_bytes())
        assert sniff_image_file(p) == "image/jpeg"
        bad = tmp_path / "y.jpg"  # extension lies the other way
        bad.write_text("not an image")
        assert sniff_image_file(bad) is None


# ---------------------------------------------------------------------------
# Intake pipeline
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def pipeline(photo_store, mock_config):
    return IntakePipeline(photo_store)


@pytest.mark.asyncio
class TestIntakePipeline:
    async def test_happy_path_stores_resized_photo(
        self, pipeline, photo_store, tmp_path, mock_config
    ):
        src = tmp_path / "in.jpg"
        src.write_bytes(_jpeg_bytes(4000, 3000))  # larger than max_image_resolution
        result = await pipeline._run_pipeline(
            src, "signal", "Jacob", caption="a red square"
        )
        rec = await photo_store.get_photo(result.photo_id)
        assert rec is not None
        assert rec.source == "signal"
        assert rec.sender == "Jacob"
        # caption seeds the description (tagger is still a stub)
        assert rec.description == "a red square"
        # downscaled within the configured max resolution
        max_w, max_h = mock_config.photos.max_image_resolution
        assert rec.width <= max_w and rec.height <= max_h

    async def test_oversize_source_rejected(
        self, pipeline, tmp_path, mock_config
    ):
        mock_config.photos.max_ingest_bytes = 10
        src = tmp_path / "big.jpg"
        src.write_bytes(_jpeg_bytes())
        with pytest.raises(ValueError, match="ingest limit"):
            await pipeline.enqueue(src, source="signal")

    async def test_queue_saturation_rejected(
        self, pipeline, tmp_path, mock_config
    ):
        mock_config.photos.max_intake_queue_depth = 0
        src = tmp_path / "ok.jpg"
        src.write_bytes(_jpeg_bytes())
        with pytest.raises(ValueError, match="saturated"):
            await pipeline.enqueue(src, source="signal")

    async def test_quota_block_rejected(
        self, pipeline, tmp_path, mock_config
    ):
        mock_config.photos.quota_block_percent = 0.0  # always over
        src = tmp_path / "ok.jpg"
        src.write_bytes(_jpeg_bytes())
        with pytest.raises(ValueError, match="quota"):
            await pipeline.enqueue(src, source="signal")

    async def test_non_image_fails_pipeline(
        self, pipeline, tmp_path, mock_config
    ):
        src = tmp_path / "fake.jpg"
        src.write_text("definitely not an image")
        with pytest.raises(Exception):
            await pipeline._run_pipeline(src, "signal", None)


class _FakeBlock:
    def __init__(self, name, payload):
        self.type = "tool_use"
        self.name = name
        self.input = payload


class _FakeResp:
    def __init__(self, name, payload):
        self.content = [_FakeBlock(name, payload)]


class _FakeMessages:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResp("describe_photo", self._payload)


class _FakeClient:
    def __init__(self, payload):
        self.messages = _FakeMessages(payload)


@pytest.mark.asyncio
class TestTagging:
    async def test_generate_tags_parses_and_dedups(self, photo_store, mock_config):
        client = _FakeClient(
            {"description": "  A dog on a beach.  ",
             "tags": ["Dog", "beach", "BEACH", "  "]}
        )
        pipe = IntakePipeline(photo_store, anthropic_client=client)
        img = Image.new("RGB", (32, 24), (1, 2, 3))
        desc, tags, new_tags = await pipe._generate_tags(img, [])
        assert desc == "A dog on a beach."
        assert tags == ["dog", "beach"]  # lowercased + deduped + blanks dropped
        assert set(new_tags) == {"dog", "beach"}  # no prior library
        # the model actually got an image block
        sent = client.messages.calls[0]["messages"][0]["content"]
        assert any(b["type"] == "image" for b in sent)

    async def test_new_tags_excludes_existing_library(self, photo_store, mock_config):
        await photo_store.create_photo(
            filename="seed.jpg", source="signal", tags=["beach"]
        )
        client = _FakeClient({"description": "x", "tags": ["beach", "dog"]})
        pipe = IntakePipeline(photo_store, anthropic_client=client)
        img = Image.new("RGB", (10, 10), (0, 0, 0))
        _, tags, new_tags = await pipe._generate_tags(img, [])
        assert set(tags) == {"beach", "dog"}
        assert new_tags == ["dog"]  # beach already in library

    async def test_run_pipeline_uses_tagger(
        self, photo_store, tmp_path, mock_config
    ):
        client = _FakeClient(
            {"description": "a sunset over water", "tags": ["sunset", "ocean"]}
        )
        pipe = IntakePipeline(photo_store, anthropic_client=client)
        src = tmp_path / "p.jpg"
        src.write_bytes(_jpeg_bytes())
        result = await pipe._run_pipeline(src, "signal", None)
        rec = await photo_store.get_photo(result.photo_id)
        assert rec.description == "a sunset over water"
        assert set(rec.tags) == {"sunset", "ocean"}

    async def test_tagger_failure_falls_back_to_caption(
        self, photo_store, tmp_path, mock_config
    ):
        class _BoomMessages:
            async def create(self, **kwargs):
                raise RuntimeError("api down")

        class _BoomClient:
            messages = _BoomMessages()

        pipe = IntakePipeline(photo_store, anthropic_client=_BoomClient())
        src = tmp_path / "q.jpg"
        src.write_bytes(_jpeg_bytes())
        result = await pipe._run_pipeline(
            src, "signal", None, caption="grandma's birthday"
        )
        rec = await photo_store.get_photo(result.photo_id)
        assert rec.description == "grandma's birthday"


@pytest.mark.asyncio
class TestPeopleDetectorSeam:
    async def test_injected_detector_annotations_stored(
        self, photo_store, tmp_path, mock_config
    ):
        async def detector(image):
            return [{"label": "Jacob", "bbox_x": 0.1, "bbox_y": 0.2}]

        pipe = IntakePipeline(photo_store, people_detector=detector)
        src = tmp_path / "d.jpg"
        src.write_bytes(_jpeg_bytes())
        result = await pipe._run_pipeline(src, "signal", None)
        rec = await photo_store.get_photo(result.photo_id)
        assert rec.people and rec.people[0]["label"] == "Jacob"

    async def test_detector_failure_is_swallowed(
        self, photo_store, tmp_path, mock_config
    ):
        async def detector(image):
            raise RuntimeError("hailo busy")

        pipe = IntakePipeline(photo_store, people_detector=detector)
        src = tmp_path / "e.jpg"
        src.write_bytes(_jpeg_bytes())
        result = await pipe._run_pipeline(src, "signal", None)  # must not raise
        rec = await photo_store.get_photo(result.photo_id)
        assert rec.people == []

    async def test_no_detector_means_no_people(
        self, photo_store, tmp_path, mock_config
    ):
        pipe = IntakePipeline(photo_store)
        src = tmp_path / "f.jpg"
        src.write_bytes(_jpeg_bytes())
        result = await pipe._run_pipeline(src, "signal", None)
        rec = await photo_store.get_photo(result.photo_id)
        assert rec.people == []


class TestAttachContentSniff:
    """build_image_block must key on content, not the filename extension."""

    def test_real_image_under_root_attaches(self, tmp_path, monkeypatch):
        import boxbot.tools._sandbox_actions as sa

        monkeypatch.setattr(sa, "_attach_roots", lambda: (tmp_path.resolve(),))
        p = tmp_path / "photo.bin"  # extension is not .jpg
        p.write_bytes(_jpeg_bytes())
        block = sa.build_image_block(p)
        assert block is not None
        assert block["source"]["media_type"] == "image/jpeg"

    def test_mislabeled_non_image_refused(self, tmp_path, monkeypatch):
        import boxbot.tools._sandbox_actions as sa

        monkeypatch.setattr(sa, "_attach_roots", lambda: (tmp_path.resolve(),))
        p = tmp_path / "evil.jpg"  # claims jpg, is a script
        p.write_text("#!/bin/sh\nrm -rf /\n")
        assert sa.build_image_block(p) is None
