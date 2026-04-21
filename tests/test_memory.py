"""Tests for the memory system — store, search, embeddings, retrieval, extraction, maintenance."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from boxbot.memory.embeddings import EMBEDDING_DIM, cosine_similarity, embed, embed_batch
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


class TestEmbeddings:
    """Test the embedding generation functions."""

    def test_embed_returns_correct_dimension(self):
        vec = embed("hello world")
        assert vec.shape == (EMBEDDING_DIM,)
        assert vec.dtype == np.float32

    def test_embed_same_text_produces_same_vector(self):
        """Deterministic fallback: same text should produce same embedding."""
        a = embed("test text")
        b = embed("test text")
        np.testing.assert_array_equal(a, b)

    def test_embed_different_text_produces_different_vector(self):
        a = embed("cats are great")
        b = embed("quantum physics theory")
        # These should differ substantially
        assert not np.allclose(a, b)

    def test_embed_batch_returns_list_of_correct_size(self):
        results = embed_batch(["hello", "world", "test"])
        assert len(results) == 3
        for vec in results:
            assert vec.shape == (EMBEDDING_DIM,)

    def test_embed_batch_empty_input(self):
        assert embed_batch([]) == []

    def test_cosine_similarity_identical_vectors(self):
        vec = embed("identical")
        sim = cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.01

    def test_cosine_similarity_orthogonal_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        sim = cosine_similarity(a, b)
        assert abs(sim) < 0.01

    def test_cosine_similarity_zero_vector_returns_zero(self):
        a = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        b = embed("something")
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
            type="operational",
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
        await memory_store.invalidate_memory(
            mid, invalidated_by="conv-1", superseded_by=None
        )
        m = await memory_store.get_memory_no_touch(mid)
        assert m.status == "invalidated"
        assert m.invalidated_by == "conv-1"

    @pytest.mark.asyncio
    async def test_delete_memory_removes_permanently(self, memory_store):
        mid = await memory_store.create_memory(
            type="operational",
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
        assert _contains_secret("AWS key is AKIAIOSFODNN7EXAMPLE") is True

    def test_detects_github_token(self):
        assert _contains_secret("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij") is True

    def test_detects_pem_private_key(self):
        assert _contains_secret("-----BEGIN RSA PRIVATE KEY-----") is True
        assert _contains_secret("-----BEGIN EC PRIVATE KEY-----") is True

    def test_detects_slack_token(self):
        assert _contains_secret("xoxb-1234567890-abcdefghij") is True

    def test_normal_text_is_clean(self):
        assert _contains_secret("Jacob likes coffee in the morning") is False

    def test_technical_text_without_secrets_is_clean(self):
        assert _contains_secret("The API documentation says to use version 3") is False
        assert _contains_secret("GitHub is a code hosting platform") is False


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
