"""Tests for the memory system — store, search, embeddings, retrieval, extraction, maintenance."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from boxbot.memory import embeddings
from boxbot.memory.embeddings import (
    EMBEDDING_DIM,
    active_model,
    cosine_similarity,
    embed,
    embed_batch,
)
from boxbot.memory.search import (
    SearchCandidate,
    _escape_fts_query,
    _merge_candidates,
    hybrid_search,
    search_memories,
)
from boxbot.memory.store import (
    DEFAULT_SYSTEM_MEMORY,
    MEMORY_TYPES,
    SYSTEM_MEMORY_MAX_BYTES,
    MemoryStore,
    _apply_section_update,
    _contains_secret,
)


# ---------------------------------------------------------------------------
# Embedding tests
# ---------------------------------------------------------------------------


class _StubEncoder:
    """Stand-in for SentenceTransformer: deterministic unit vectors."""

    def encode(self, text, normalize_embeddings=False):
        if isinstance(text, list):
            return [self.encode(t) for t in text]
        rng = np.random.RandomState(len(text))
        vec = rng.randn(EMBEDDING_DIM).astype(np.float32)
        return vec / np.linalg.norm(vec)


class TestEmbeddings:
    """Test the embedding generation functions.

    Both modes are forced explicitly so the suite behaves the same with
    or without sentence-transformers installed.
    """

    @pytest.fixture
    def with_model(self, monkeypatch):
        """Force the loaded-model path with a stub encoder."""
        monkeypatch.setattr(embeddings, "_model", _StubEncoder())
        monkeypatch.setattr(embeddings, "_unavailable", False)

    @pytest.fixture
    def without_model(self, monkeypatch):
        """Force the degraded path (no sentence-transformers, no API)."""
        monkeypatch.setattr(embeddings, "_model", None)
        monkeypatch.setattr(embeddings, "_unavailable", True)
        monkeypatch.setattr(embeddings, "_api_client", None)
        monkeypatch.setattr(embeddings, "_api_unavailable", True)

    @pytest.fixture
    def with_api(self, monkeypatch):
        """Force the API-fallback path with a stub OpenAI client."""

        class _StubAPI:
            class embeddings:  # noqa: N801 - mirrors openai client shape
                @staticmethod
                def create(model, input, dimensions):
                    assert dimensions == EMBEDDING_DIM

                    class _Item:
                        def __init__(self, text):
                            rng = np.random.RandomState(len(text))
                            vec = rng.randn(dimensions).astype(np.float32)
                            self.embedding = (vec / np.linalg.norm(vec)).tolist()

                    class _Response:
                        data = [_Item(t) for t in input]

                    return _Response()

        monkeypatch.setattr(embeddings, "_model", None)
        monkeypatch.setattr(embeddings, "_unavailable", True)
        monkeypatch.setattr(embeddings, "_api_client", _StubAPI())
        monkeypatch.setattr(embeddings, "_api_model", "text-embedding-3-small")
        monkeypatch.setattr(embeddings, "_api_unavailable", False)

    @pytest.fixture
    def with_failing_api(self, monkeypatch, with_api):
        """API configured but every call errors."""

        class _Broken:
            class embeddings:  # noqa: N801
                @staticmethod
                def create(model, input, dimensions):
                    raise RuntimeError("deployment not found")

        monkeypatch.setattr(embeddings, "_api_client", _Broken())
        monkeypatch.setattr(embeddings, "_api_call_failed", False)

    def test_embed_returns_correct_dimension(self, with_model):
        vec = embed("hello world")
        assert vec.shape == (EMBEDDING_DIM,)
        assert vec.dtype == np.float32

    def test_embed_same_text_produces_same_vector(self, with_model):
        a = embed("test text")
        b = embed("test text")
        np.testing.assert_array_equal(a, b)

    def test_embed_batch_returns_list_of_correct_size(self, with_model):
        results = embed_batch(["hello", "world", "testing"])
        assert len(results) == 3
        for vec in results:
            assert vec.shape == (EMBEDDING_DIM,)

    def test_embed_batch_empty_input(self):
        assert embed_batch([]) == []

    def test_embed_returns_none_without_model(self, without_model):
        """No fabricated vectors: noise would outrank real keyword hits."""
        assert embed("hello world") is None

    def test_embed_batch_returns_nones_without_model(self, without_model):
        assert embed_batch(["a", "b"]) == [None, None]

    def test_active_model_reports_availability(self, without_model):
        assert active_model() is None

    def test_active_model_names_the_model(self, with_model):
        assert active_model() == embeddings.MODEL_NAME

    def test_api_embed_returns_correct_shape(self, with_api):
        vec = embed("hello world")
        assert vec.shape == (EMBEDDING_DIM,)
        assert vec.dtype == np.float32

    def test_api_embed_batch_preserves_order_and_size(self, with_api):
        results = embed_batch(["hello", "world", "hello"])
        assert len(results) == 3
        np.testing.assert_array_equal(results[0], results[2])
        assert all(r.shape == (EMBEDDING_DIM,) for r in results)

    def test_api_active_model_names_api_model(self, with_api):
        assert active_model() == "text-embedding-3-small"

    def test_local_model_wins_over_api(self, with_api, monkeypatch):
        """Local sentence-transformers is preferred when importable."""
        monkeypatch.setattr(embeddings, "_model", _StubEncoder())
        monkeypatch.setattr(embeddings, "_unavailable", False)
        assert active_model() == embeddings.MODEL_NAME

    def test_api_failure_returns_none_not_noise(self, with_failing_api):
        """A failed API call stores NULL, never a fabricated vector."""
        assert embed("hello") is None
        assert embed_batch(["a", "b"]) == [None, None]

    def test_cosine_similarity_identical_vectors(self):
        vec = np.linspace(0.0, 1.0, EMBEDDING_DIM, dtype=np.float32)
        sim = cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.01

    def test_cosine_similarity_orthogonal_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        sim = cosine_similarity(a, b)
        assert abs(sim) < 0.01

    def test_cosine_similarity_zero_vector_returns_zero(self):
        a = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        b = np.ones(EMBEDDING_DIM, dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0


# ---------------------------------------------------------------------------
# Memory store tests
# ---------------------------------------------------------------------------


class TestMemoryStoreCRUD:
    """Test MemoryStore create, get, update, and delete operations."""

    @pytest.mark.asyncio
    async def test_create_memory_returns_uuid(self, memory_store):
        mid = await memory_store.create_memory(
            type="person",
            content="Jacob likes coffee",
            summary="Jacob coffee preference",
            person="Jacob",
        )
        assert isinstance(mid, str)
        assert len(mid) > 0

    @pytest.mark.asyncio
    async def test_get_memory_returns_record(self, memory_store):
        mid = await memory_store.create_memory(
            type="household",
            content="The wifi password is fish",
            summary="Wifi password",
        )
        memory = await memory_store.get_memory(mid)
        assert memory is not None
        assert memory.type == "household"
        assert memory.content == "The wifi password is fish"
        assert memory.status == "active"

    @pytest.mark.asyncio
    async def test_get_memory_updates_last_relevant_at(self, memory_store):
        mid = await memory_store.create_memory(
            type="methodology",
            content="Test relevance",
            summary="Test",
        )
        m1 = await memory_store.get_memory_no_touch(mid)
        m2 = await memory_store.get_memory(mid)  # updates last_relevant_at
        m3 = await memory_store.get_memory_no_touch(mid)
        assert m3.last_relevant_at >= m1.last_relevant_at

    @pytest.mark.asyncio
    async def test_get_memory_no_touch_does_not_update(self, memory_store):
        mid = await memory_store.create_memory(
            type="person",
            content="Static check",
            summary="Static",
            person="Alice",
        )
        m1 = await memory_store.get_memory_no_touch(mid)
        m2 = await memory_store.get_memory_no_touch(mid)
        assert m1.last_relevant_at == m2.last_relevant_at

    @pytest.mark.asyncio
    async def test_get_nonexistent_memory_returns_none(self, memory_store):
        result = await memory_store.get_memory("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_memory_type_raises(self, memory_store):
        with pytest.raises(ValueError, match="Invalid memory type"):
            await memory_store.create_memory(
                type="invalid_type",
                content="bad",
                summary="bad",
            )

    @pytest.mark.asyncio
    async def test_archive_and_unarchive_memory(self, memory_store):
        mid = await memory_store.create_memory(
            type="methodology",
            content="Test method",
            summary="Method",
        )
        await memory_store.archive_memory(mid)
        m = await memory_store.get_memory_no_touch(mid)
        assert m.status == "archived"

        await memory_store.unarchive_memory(mid)
        m = await memory_store.get_memory_no_touch(mid)
        assert m.status == "active"

    @pytest.mark.asyncio
    async def test_invalidate_memory(self, memory_store):
        mid = await memory_store.create_memory(
            type="person",
            content="Old fact",
            summary="Old",
            person="Jacob",
        )
        affected = await memory_store.invalidate_memory(
            mid, invalidated_by="conv-1", superseded_by=None
        )
        assert affected == 1
        m = await memory_store.get_memory_no_touch(mid)
        assert m.status == "invalidated"
        assert m.invalidated_by == "conv-1"

    @pytest.mark.asyncio
    async def test_invalidate_unknown_id_affects_zero_rows(self, memory_store):
        """A full-id miss must report 0, not look like a success.

        Regression for the misattribution bug: a 0-row UPDATE is not a
        SQLite error, so callers must check the count.
        """
        affected = await memory_store.invalidate_memory(
            "no-such-id", invalidated_by="conv-1"
        )
        assert affected == 0

    @pytest.mark.asyncio
    async def test_repoint_person_name_updates_person_and_people(
        self, memory_store
    ):
        """A merge/rename must move the structured name fields so the
        survivor's recall reaches the merged-away memories."""
        mid = await memory_store.create_memory(
            type="person",
            content="Eric likes Mario",
            summary="Eric Mario",
            person="Eric",
            people=["Eric"],
        )
        n = await memory_store.repoint_person_name("Eric", "Erik")
        assert n >= 1
        m = await memory_store.get_memory_no_touch(mid)
        assert m.person == "Erik"
        assert m.people == ["Erik"]
        # Free-text prose is deliberately left untouched.
        assert m.content == "Eric likes Mario"

    @pytest.mark.asyncio
    async def test_repoint_person_name_preserves_other_people(
        self, memory_store
    ):
        mid = await memory_store.create_memory(
            type="person",
            content="Eric and Jacob at the park",
            summary="Eric Jacob park",
            person="Erik",
            people=["Eric", "Jacob"],
        )
        await memory_store.repoint_person_name("Eric", "Erik")
        m = await memory_store.get_memory_no_touch(mid)
        assert m.people == ["Erik", "Jacob"]

    @pytest.mark.asyncio
    async def test_repoint_person_name_no_substring_clobber(
        self, memory_store
    ):
        """'Eric' must not rewrite 'Erica' — quoted-token match only."""
        mid = await memory_store.create_memory(
            type="person",
            content="Erica fact",
            summary="Erica",
            person="Erica",
            people=["Erica"],
        )
        await memory_store.repoint_person_name("Eric", "Erik")
        m = await memory_store.get_memory_no_touch(mid)
        assert m.person == "Erica"
        assert m.people == ["Erica"]

    @pytest.mark.asyncio
    async def test_repoint_person_name_skips_invalidated(self, memory_store):
        mid = await memory_store.create_memory(
            type="person", content="dead", summary="dead", person="Eric",
            people=["Eric"],
        )
        await memory_store.invalidate_memory(mid, invalidated_by="c")
        await memory_store.repoint_person_name("Eric", "Erik")
        m = await memory_store.get_memory_no_touch(mid)
        assert m.person == "Eric"  # untouched: dead rows stay as-is

    @pytest.mark.asyncio
    async def test_repoint_person_name_noop_when_same(self, memory_store):
        assert await memory_store.repoint_person_name("Eric", "Eric") == 0
        assert await memory_store.repoint_person_name("", "Erik") == 0

    @pytest.mark.asyncio
    async def test_resolve_memory_id_exact_and_prefix(self, memory_store):
        mid = await memory_store.create_memory(
            type="person",
            content="Resolve me",
            summary="Resolve",
            person="Jacob",
        )
        # Exact id.
        assert await memory_store.resolve_memory_id(mid) == [mid]
        # 8-char prefix (the form the agent actually sees).
        assert await memory_store.resolve_memory_id(mid[:8]) == [mid]

    @pytest.mark.asyncio
    async def test_resolve_memory_id_no_match(self, memory_store):
        assert await memory_store.resolve_memory_id("ffffffff") == []

    @pytest.mark.asyncio
    async def test_resolve_memory_id_skips_invalidated_on_prefix(
        self, memory_store
    ):
        """Prefix matching ignores already-invalidated rows so a correction
        doesn't resolve to a dead duplicate."""
        mid = await memory_store.create_memory(
            type="person", content="Dead", summary="Dead", person="Jacob"
        )
        await memory_store.invalidate_memory(mid, invalidated_by="c")
        # Exact id still resolves (idempotent); prefix does not.
        assert await memory_store.resolve_memory_id(mid) == [mid]
        assert await memory_store.resolve_memory_id(mid[:8]) == []

    @pytest.mark.asyncio
    async def test_delete_memory_removes_permanently(self, memory_store):
        mid = await memory_store.create_memory(
            type="methodology",
            content="Delete me",
            summary="Delete",
        )
        await memory_store.delete_memory(mid)
        result = await memory_store.get_memory_no_touch(mid)
        assert result is None

    @pytest.mark.asyncio
    async def test_list_memories_with_filters(self, memory_store):
        await memory_store.create_memory(
            type="person", content="P1", summary="S1", person="Jacob"
        )
        await memory_store.create_memory(
            type="household", content="H1", summary="S2"
        )

        person_mems = await memory_store.list_memories(type="person")
        assert len(person_mems) == 1
        assert person_mems[0].type == "person"

    @pytest.mark.asyncio
    async def test_count_memories(self, memory_store):
        await memory_store.create_memory(
            type="person", content="C1", summary="S1", person="A"
        )
        await memory_store.create_memory(
            type="person", content="C2", summary="S2", person="B"
        )
        count = await memory_store.count_memories(status="active")
        assert count == 2

    @pytest.mark.asyncio
    async def test_update_memory_content(self, memory_store):
        mid = await memory_store.create_memory(
            type="person",
            content="Original content",
            summary="Original",
            person="Alice",
        )
        await memory_store.update_memory_content(
            mid,
            content="Updated content",
            summary="Updated",
            tags=["new-tag"],
        )
        m = await memory_store.get_memory_no_touch(mid)
        assert m.content == "Updated content"
        assert m.summary == "Updated"
        assert "new-tag" in m.tags


# ---------------------------------------------------------------------------
# Conversation CRUD
# ---------------------------------------------------------------------------


class TestConversationCRUD:
    """Test conversation log operations."""

    @pytest.mark.asyncio
    async def test_create_and_get_conversation(self, memory_store):
        cid = await memory_store.create_conversation(
            channel="voice",
            participants=["Jacob", "BB"],
            summary="Talked about weather",
            topics=["weather"],
        )
        conv = await memory_store.get_conversation(cid)
        assert conv is not None
        assert conv.channel == "voice"
        assert "Jacob" in conv.participants

    @pytest.mark.asyncio
    async def test_list_conversations_ordered_by_date(self, memory_store):
        await memory_store.create_conversation(
            channel="voice", participants=["A"], summary="First"
        )
        await memory_store.create_conversation(
            channel="whatsapp", participants=["B"], summary="Second"
        )
        convs = await memory_store.list_conversations(limit=10)
        assert len(convs) == 2
        # Newest first
        assert convs[0].summary == "Second"

    @pytest.mark.asyncio
    async def test_delete_conversation_detaches_source_memories(
        self, memory_store
    ):
        """Deleting a conversation must not fail when a memory still points
        at it, and must leave that memory intact with a null source.

        Regression: memories outlive their source conversation (180d vs
        60d retention), so nightly maintenance deletes conversations that
        still have child memories. Without detaching first, the enforced
        ``source_conversation`` FK raised ``FOREIGN KEY constraint failed``
        and aborted the whole maintenance pass.
        """
        cid = await memory_store.create_conversation(
            channel="voice", participants=["Jacob", "BB"], summary="Chat"
        )
        mid = await memory_store.create_memory(
            type="person",
            content="Jacob likes hiking",
            summary="Jacob likes hiking",
            person="Jacob",
            source_conversation=cid,
        )

        # Must not raise despite the surviving child memory.
        await memory_store.delete_conversation(cid)

        assert await memory_store.get_conversation(cid) is None
        survivor = await memory_store.get_memory_no_touch(mid)
        assert survivor is not None
        assert survivor.content == "Jacob likes hiking"
        assert survivor.source_conversation is None


# ---------------------------------------------------------------------------
# System memory
# ---------------------------------------------------------------------------


class TestSystemMemory:
    """Test system memory read/write/versioning."""

    @pytest.mark.asyncio
    async def test_read_default_system_memory(self, memory_store, tmp_memory_db):
        sys_mem_path = tmp_memory_db.parent / "system.md"
        with patch("boxbot.memory.store.SYSTEM_MEMORY_PATH", sys_mem_path):
            content = await memory_store.read_system_memory()
        assert "Household" in content

    @pytest.mark.asyncio
    async def test_add_entry_to_system_memory(self, memory_store, tmp_memory_db):
        sys_mem_path = tmp_memory_db.parent / "system.md"
        with patch("boxbot.memory.store.SYSTEM_MEMORY_PATH", sys_mem_path):
            await memory_store.update_system_memory(
                section="Household",
                action="add_entry",
                content="Jacob is allergic to peanuts",
                updated_by="test",
            )
            content = await memory_store.read_system_memory()
        assert "peanuts" in content

    @pytest.mark.asyncio
    async def test_invalid_section_raises(self, memory_store, tmp_memory_db):
        sys_mem_path = tmp_memory_db.parent / "system.md"
        with patch("boxbot.memory.store.SYSTEM_MEMORY_PATH", sys_mem_path):
            with pytest.raises(ValueError, match="Invalid section"):
                await memory_store.update_system_memory(
                    section="BadSection",
                    action="set",
                    content="...",
                    updated_by="test",
                )

    @pytest.mark.asyncio
    async def test_secret_content_rejected(self, memory_store, tmp_memory_db):
        sys_mem_path = tmp_memory_db.parent / "system.md"
        with patch("boxbot.memory.store.SYSTEM_MEMORY_PATH", sys_mem_path):
            with pytest.raises(ValueError, match="secrets"):
                await memory_store.update_system_memory(
                    section="Household",
                    action="add_entry",
                    content="api_key: sk-abc123defghijklmnopqrst",
                    updated_by="test",
                )


class TestContainsSecret:
    """Test the secret detection patterns."""

    def test_detects_api_key_pattern(self):
        assert _contains_secret("api_key: abc123") is True

    def test_detects_sk_prefix(self):
        assert _contains_secret("use sk-abcdefghijklmnopqrstuvwx for auth") is True

    def test_detects_bearer_token(self):
        assert _contains_secret("Bearer eyJhbGciOiJIUzI1NiIsInR5c") is True

    def test_detects_aws_access_key(self):
        assert _contains_secret("key is AKIAIOSFODNN7EXAMPLE") is True

    def test_detects_github_token(self):
        assert _contains_secret("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl") is True

    def test_detects_private_key(self):
        assert _contains_secret("-----BEGIN RSA PRIVATE KEY-----") is True

    def test_detects_ec_private_key(self):
        assert _contains_secret("-----BEGIN EC PRIVATE KEY-----") is True

    def test_normal_text_is_clean(self):
        assert _contains_secret("Jacob likes coffee in the morning") is False


class TestApplySectionUpdate:
    """Test the system memory section update logic."""

    def test_set_replaces_section(self):
        current = "## Household\n- old entry\n\n## Standing Instructions\n- rule 1\n"
        result = _apply_section_update(current, "Household", "set", "- new content")
        assert "new content" in result
        assert "old entry" not in result

    def test_add_entry_appends(self):
        current = "## Household\n- entry 1\n\n## Standing Instructions\n- rule 1\n"
        result = _apply_section_update(
            current, "Household", "add_entry", "new item"
        )
        assert "entry 1" in result
        assert "- new item" in result

    def test_remove_entry_removes_matching(self):
        current = "## Household\n- remove me\n- keep me\n"
        result = _apply_section_update(
            current, "Household", "remove_entry", "remove me"
        )
        assert "remove me" not in result
        assert "keep me" in result


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------


class TestHybridSearch:
    """Test the hybrid vector + BM25 search pipeline."""

    @pytest.mark.asyncio
    async def test_search_finds_relevant_memory(self, memory_store):
        await memory_store.create_memory(
            type="person",
            content="Jacob is allergic to peanuts and tree nuts",
            summary="Jacob has nut allergies",
            person="Jacob",
            tags=["health", "allergy"],
        )
        await memory_store.create_memory(
            type="household",
            content="The fridge brand is Samsung",
            summary="Samsung fridge",
        )

        candidates = await hybrid_search(
            memory_store, "allergies", include_conversations=False
        )
        assert len(candidates) > 0
        # The allergy memory should score higher
        top = candidates[0]
        assert "allergic" in top.content.lower() or "allerg" in top.summary.lower()

    @pytest.mark.asyncio
    async def test_search_filters_by_person(self, memory_store):
        await memory_store.create_memory(
            type="person",
            content="Jacob likes chess",
            summary="Jacob chess",
            person="Jacob",
        )
        await memory_store.create_memory(
            type="person",
            content="Alice likes painting",
            summary="Alice painting",
            person="Alice",
        )

        candidates = await hybrid_search(
            memory_store,
            "hobbies",
            person="Jacob",
            include_conversations=False,
        )
        # All results should relate to Jacob
        for c in candidates:
            assert c.person == "Jacob" or "Jacob" in str(c.metadata.get("people", []))


class TestDegradedEmbeddings:
    """No embedding model: NULL vectors on write, keyword-only ranking."""

    @pytest.fixture(autouse=True)
    def no_model(self, monkeypatch):
        monkeypatch.setattr(embeddings, "_model", None)
        monkeypatch.setattr(embeddings, "_unavailable", True)

    @pytest.mark.asyncio
    async def test_create_memory_stores_null_embedding(self, memory_store):
        mem_id = await memory_store.create_memory(
            type="household", content="The fridge is a Samsung", summary="fridge"
        )
        cursor = await memory_store.db.execute(
            "SELECT embedding FROM memories WHERE id = ?", (mem_id,)
        )
        row = await cursor.fetchone()
        assert row["embedding"] is None

    @pytest.mark.asyncio
    async def test_search_ranks_on_keywords_alone(self, memory_store):
        await memory_store.create_memory(
            type="person",
            content="Jacob is allergic to peanuts and tree nuts",
            summary="Jacob has nut allergies",
            person="Jacob",
        )
        await memory_store.create_memory(
            type="household",
            content="The fridge brand is Samsung",
            summary="Samsung fridge",
        )

        candidates = await hybrid_search(
            memory_store, "allergies", include_conversations=False
        )
        assert candidates
        top = candidates[0]
        assert "allerg" in top.summary.lower()
        # BM25 carries the whole score — no vector contribution at all.
        assert all(c.vector_score == 0.0 for c in candidates)
        assert top.combined_score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_search_matches_a_whole_utterance(self, memory_store):
        """FTS5 ANDs bare terms — an utterance must still find keywords."""
        mem_id = await memory_store.create_memory(
            type="person",
            content="Jacob loves chicken pesto pizza.",
            summary="Jacob's pizza preference",
            person="Jacob",
        )
        candidates = await hybrid_search(
            memory_store,
            "Jacob what should I eat tonight?",
            person="Jacob",
            include_conversations=False,
        )
        assert [c.id for c in candidates] == [mem_id]

    @pytest.mark.asyncio
    async def test_exact_phrase_wins_over_loose_matches(self, memory_store):
        """The relaxed OR pass only runs when the strict pass finds nothing."""
        exact = await memory_store.create_memory(
            type="household",
            content="The fridge brand is Samsung",
            summary="Samsung fridge",
        )
        await memory_store.create_memory(
            type="household",
            content="The washing machine brand is Bosch",
            summary="Bosch washer",
        )
        candidates = await hybrid_search(
            memory_store, "Samsung fridge", include_conversations=False
        )
        assert [c.id for c in candidates] == [exact]

    @pytest.mark.asyncio
    async def test_search_tolerates_stored_vectors(self, memory_store):
        """Rows embedded by an earlier install must not break search."""
        mem_id = await memory_store.create_memory(
            type="household", content="The fridge brand is Samsung", summary="fridge"
        )
        await memory_store.db.execute(
            "UPDATE memories SET embedding = ? WHERE id = ?",
            (np.ones(EMBEDDING_DIM, dtype=np.float32).tobytes(), mem_id),
        )
        candidates = await hybrid_search(
            memory_store, "fridge", include_conversations=False
        )
        assert [c.id for c in candidates] == [mem_id]


class TestEmbeddingModelMarker:
    """The store records which embedder produced its vectors."""

    @pytest.fixture(autouse=True)
    def stub_model(self, monkeypatch):
        monkeypatch.setattr(embeddings, "_model", _StubEncoder())
        monkeypatch.setattr(embeddings, "_unavailable", False)

    @pytest.mark.asyncio
    async def test_marker_recorded_on_first_init(self, memory_store):
        cursor = await memory_store.db.execute(
            "SELECT value FROM store_meta WHERE key = 'embedding_model'"
        )
        row = await cursor.fetchone()
        assert row["value"] == embeddings.MODEL_NAME

    @pytest.mark.asyncio
    async def test_model_swap_warns(self, memory_store, monkeypatch, caplog):
        monkeypatch.setattr(embeddings, "MODEL_NAME", "some-other-model")
        with caplog.at_level("WARNING"):
            await memory_store._check_embedding_model()
        assert "Embedding model changed" in caplog.text

    @pytest.mark.asyncio
    async def test_unmarked_store_on_default_model_is_claimed_once(
        self, memory_store, caplog
    ):
        """Rows embedded before the marker existed, still on the default
        local model: claim them once at INFO, no standing warning."""
        mem_id = await memory_store.create_memory(
            type="household", content="The fridge is a Samsung", summary="fridge"
        )
        await memory_store.db.execute(
            "UPDATE memories SET embedding = ? WHERE id = ?",
            (np.ones(EMBEDDING_DIM, dtype=np.float32).tobytes(), mem_id),
        )
        await memory_store.db.execute("DELETE FROM store_meta")

        with caplog.at_level("INFO"):
            await memory_store._check_embedding_model()
        assert "unknown provenance" not in caplog.text
        assert "historical default" in caplog.text

        cursor = await memory_store.db.execute(
            "SELECT value FROM store_meta WHERE key = 'embedding_model'"
        )
        row = await cursor.fetchone()
        assert row is not None and row["value"] == embeddings.MODEL_NAME

        caplog.clear()
        with caplog.at_level("INFO"):
            await memory_store._check_embedding_model()
        assert caplog.text == ""

    @pytest.mark.asyncio
    async def test_unmarked_store_on_other_model_warns_every_boot(
        self, memory_store, caplog, monkeypatch
    ):
        """Unmarked vectors with a NON-default active backend cannot be
        assumed to match: warn, leave the marker absent, warn again."""
        mem_id = await memory_store.create_memory(
            type="household", content="The fridge is a Samsung", summary="fridge"
        )
        await memory_store.db.execute(
            "UPDATE memories SET embedding = ? WHERE id = ?",
            (np.ones(EMBEDDING_DIM, dtype=np.float32).tobytes(), mem_id),
        )
        await memory_store.db.execute("DELETE FROM store_meta")
        monkeypatch.setattr(
            "boxbot.memory.store.active_model", lambda: "text-embedding-3-small"
        )

        with caplog.at_level("WARNING"):
            await memory_store._check_embedding_model()
        assert "unknown provenance" in caplog.text

        cursor = await memory_store.db.execute(
            "SELECT value FROM store_meta WHERE key = 'embedding_model'"
        )
        assert await cursor.fetchone() is None

        caplog.clear()
        with caplog.at_level("WARNING"):
            await memory_store._check_embedding_model()
        assert "unknown provenance" in caplog.text

    @pytest.mark.asyncio
    async def test_unmarked_empty_store_is_quiet(self, memory_store, caplog):
        await memory_store.db.execute("DELETE FROM store_meta")
        with caplog.at_level("WARNING"):
            await memory_store._check_embedding_model()
        assert "unknown provenance" not in caplog.text

    @pytest.mark.asyncio
    async def test_degraded_mode_leaves_marker_alone(
        self, memory_store, monkeypatch
    ):
        monkeypatch.setattr(embeddings, "_model", None)
        monkeypatch.setattr(embeddings, "_unavailable", True)
        await memory_store._check_embedding_model()
        cursor = await memory_store.db.execute(
            "SELECT value FROM store_meta WHERE key = 'embedding_model'"
        )
        row = await cursor.fetchone()
        assert row["value"] == embeddings.MODEL_NAME


class TestMergeCandidates:
    """Test the score merging/normalization logic."""

    def test_merge_combines_vector_and_bm25(self):
        vec_cands = [
            SearchCandidate(
                id="a", source="memory", type="person", person=None,
                content="", summary="A", vector_score=1.0
            ),
        ]
        bm25_cands = [
            SearchCandidate(
                id="a", source="memory", type="person", person=None,
                content="", summary="A", bm25_score=0.5
            ),
        ]
        merged = _merge_candidates(vec_cands, bm25_cands, limit=10)
        assert len(merged) == 1
        assert merged[0].combined_score > 0

    def test_merge_deduplicates_by_id(self):
        vec_cands = [
            SearchCandidate(
                id="x", source="memory", type="person", person=None,
                content="", summary="X", vector_score=0.8
            ),
        ]
        bm25_cands = [
            SearchCandidate(
                id="x", source="memory", type="person", person=None,
                content="", summary="X", bm25_score=0.6
            ),
        ]
        merged = _merge_candidates(vec_cands, bm25_cands, limit=10)
        assert len(merged) == 1

    def test_merge_respects_limit(self):
        candidates = [
            SearchCandidate(
                id=f"m{i}", source="memory", type="person", person=None,
                content="", summary=f"S{i}", vector_score=float(i) / 10
            )
            for i in range(20)
        ]
        merged = _merge_candidates(candidates, [], limit=5)
        assert len(merged) == 5


class TestEscapeFtsQuery:
    """Test FTS5 query escaping."""

    def test_simple_words_quoted(self):
        result = _escape_fts_query("hello world")
        assert '"hello"' in result
        assert '"world"' in result

    def test_special_chars_removed(self):
        result = _escape_fts_query("test@email.com OR 1=1")
        assert "@" not in result


# ---------------------------------------------------------------------------
# Search entry point
# ---------------------------------------------------------------------------


class TestSearchMemoriesEntryPoint:
    """Test the main search_memories() function."""

    @pytest.mark.asyncio
    async def test_get_mode_returns_memory(self, memory_store):
        mid = await memory_store.create_memory(
            type="person",
            content="Get mode test",
            summary="Get test",
            person="Alice",
        )
        result = await search_memories(
            memory_store, mode="get", memory_id=mid
        )
        assert result["id"] == mid
        assert result["content"] == "Get mode test"

    @pytest.mark.asyncio
    async def test_get_mode_nonexistent_returns_error(self, memory_store):
        result = await search_memories(
            memory_store, mode="get", memory_id="no-such-id"
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_lookup_mode_returns_facts_and_conversations(self, memory_store):
        await memory_store.create_memory(
            type="person",
            content="Jacob tests lookup mode",
            summary="Lookup test",
            person="Jacob",
        )
        result = await search_memories(
            memory_store, mode="lookup", query="lookup test"
        )
        assert "facts" in result
        assert "conversations" in result

    @pytest.mark.asyncio
    async def test_summary_mode_returns_answer(self, memory_store):
        await memory_store.create_memory(
            type="household",
            content="The house has 3 bedrooms",
            summary="House size",
        )
        result = await search_memories(
            memory_store, mode="summary", query="house"
        )
        assert "answer" in result
        assert "sources" in result

    @pytest.mark.asyncio
    async def test_invalid_mode_raises(self, memory_store):
        with pytest.raises(ValueError, match="Invalid mode"):
            await search_memories(
                memory_store, mode="bad_mode", query="test"
            )

    @pytest.mark.asyncio
    async def test_get_mode_without_id_raises(self, memory_store):
        with pytest.raises(ValueError, match="memory_id is required"):
            await search_memories(memory_store, mode="get")


class TestMemoryDeleteHandler:
    """The memory.delete sandbox action: prefix resolution, honest errors,
    and an informative response. Regression suite for the misattribution
    bug where a prefix delete silently no-op'd but returned ok."""

    @pytest.mark.asyncio
    async def test_delete_resolves_prefix_and_returns_record(
        self, memory_store, monkeypatch
    ):
        from boxbot.tools import _sandbox_actions as sa

        monkeypatch.setattr(sa, "_memory_store", memory_store)
        mid = await memory_store.create_memory(
            type="person",
            content="Zara's preschool graduation: Mon Jun 8 2026, 8:45 AM.",
            summary="Zara's preschool graduation",
            person="Zara",
        )
        ctx = sa.ActionContext()
        # The agent only ever sees the 8-char prefix.
        resp = await sa.process_action(
            {"_sdk": "memory.delete", "id": mid[:8]}, ctx
        )
        assert resp["status"] == "ok"
        assert resp["invalidated"]["id"] == mid
        assert resp["invalidated"]["person"] == "Zara"
        assert resp["invalidated"]["status"] == "invalidated"
        m = await memory_store.get_memory_no_touch(mid)
        assert m.status == "invalidated"

    @pytest.mark.asyncio
    async def test_delete_no_match_errors(self, memory_store, monkeypatch):
        from boxbot.tools import _sandbox_actions as sa

        monkeypatch.setattr(sa, "_memory_store", memory_store)
        ctx = sa.ActionContext()
        resp = await sa.process_action(
            {"_sdk": "memory.delete", "id": "deadbeef"}, ctx
        )
        assert resp["status"] == "error"
        assert "no active memory" in resp["message"]

    @pytest.mark.asyncio
    async def test_delete_ambiguous_prefix_errors(
        self, memory_store, monkeypatch
    ):
        from boxbot.tools import _sandbox_actions as sa

        monkeypatch.setattr(sa, "_memory_store", memory_store)

        async def _two(_prefix):
            return ["id-aaaa-1", "id-aaaa-2"]

        monkeypatch.setattr(memory_store, "resolve_memory_id", _two)
        ctx = sa.ActionContext()
        resp = await sa.process_action(
            {"_sdk": "memory.delete", "id": "id-aaaa"}, ctx
        )
        assert resp["status"] == "error"
        assert "ambiguous" in resp["message"]
        assert resp["matches"] == ["id-aaaa-1", "id-aaaa-2"]
