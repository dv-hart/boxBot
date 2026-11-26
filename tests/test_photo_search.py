"""Photo search: soft person match, FTS over tags/people, relevance floor,
slim rows, one-call show, and tool/SDK parity. Fakes only.

Dev has no embedding model (``embed`` returns None), so vector behaviour
is tested with injected unit vectors.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest
import pytest_asyncio

import boxbot.displays.manager as display_manager_mod
import boxbot.photos.search as search_mod
from boxbot.memory.embeddings import EMBEDDING_DIM
from boxbot.photos.search import (
    DEFAULT_LIMIT,
    fts_match_expr,
    hybrid_search,
    query_terms,
    search_photos,
    split_query_names,
)
from boxbot.sdk import _transport, photos as sdk_photos
from boxbot.tools._sandbox_actions import ActionContext, process_action
from boxbot.tools.builtins.search_photos import SearchPhotosTool


@pytest_asyncio.fixture
async def wired_store(photo_store, mock_config, monkeypatch):
    monkeypatch.setattr(search_mod, "_store_singleton", photo_store)
    yield photo_store


def _unit(axis: int) -> np.ndarray:
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vec[axis] = 1.0
    return vec


def _mix(a: int, b: int, weight: float) -> np.ndarray:
    """Unit vector with cosine *weight* against axis *a*."""
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vec[a] = weight
    vec[b] = float(np.sqrt(1.0 - weight * weight))
    return vec


async def _family_library(store):
    """Four photos; none has a photo_people row (today's production state)."""
    await store.create_photo(
        filename="1.jpg", source="signal", photo_id="photo_tag",
        description="A little girl with curly hair on a swing",
        tags=["zara", "park"],
    )
    await store.create_photo(
        filename="2.jpg", source="signal", photo_id="photo_desc",
        description="Zara blowing out birthday candles", tags=["birthday"],
    )
    await store.create_photo(
        filename="3.jpg", source="signal", photo_id="photo_cat",
        description="a fluffy cat sleeping on the sofa", tags=["cat"],
    )
    await store.create_photo(
        filename="4.jpg", source="camera", photo_id="photo_bazaar",
        description="stalls at the bazaar", tags=["market"],
    )


class TestQuerySanitising:
    def test_terms_drop_filler_and_punctuation(self):
        assert query_terms("Show us photos of Zara!") == ["zara"]
        assert query_terms('zara\'s "birthday" (2026) AND NOT') == [
            "zara", "s", "birthday", "2026", "not"
        ]
        # Nothing but filler = no terms (the search then lists newest first).
        assert query_terms("photos") == []
        assert query_terms("show me photos") == []
        assert query_terms("  ") == [] and query_terms(None) == []

    def test_names_are_pulled_out_whole(self):
        known = {n.casefold(): n for n in
                 ("Zara", "Ann", "Ann-Marie", "Mary", "Mary Jane", "Arcy")}
        assert split_query_names("zara's birthday", known) == (
            ["Zara"], "       birthday")
        # Hyphenated and multi-word names are one unit, never fragments.
        assert split_query_names("Ann-Marie", known)[0] == ["Ann-Marie"]
        assert split_query_names("ann", known)[0] == ["Ann"]
        assert split_query_names("mary  jane at the park", known)[0] == [
            "Mary Jane"]
        assert split_query_names("Joann and D'Arcy", known)[0] == []
        assert split_query_names("Erik and Zara", {
            "zara": "Zara", "erik": "Erik"})[0] == ["Erik", "Zara"]
        assert split_query_names("the bazaar", known) == ([], "the bazaar")

    def test_match_expr_is_quoted_or(self):
        assert fts_match_expr(["zara", "beach"]) == '"zara" OR "beach"'


@pytest.mark.asyncio
class TestPersonSearch:
    async def test_name_found_in_tags_and_description(self, photo_store):
        """'photos of Zara' works with an empty photo_people table."""
        await _family_library(photo_store)
        results = await hybrid_search(photo_store, query="photos of Zara")
        assert {r.photo.id for r in results} == {"photo_tag", "photo_desc"}

    async def test_people_filter_is_soft_across_fields(self, photo_store):
        await _family_library(photo_store)
        results = await hybrid_search(photo_store, people=["zara"])
        assert {r.photo.id for r in results} == {"photo_tag", "photo_desc"}
        # Whole words only: "bazaar" is not Zara.
        assert "photo_bazaar" not in {r.photo.id for r in results}

    async def test_labels_outrank_text_mentions(self, photo_store):
        await _family_library(photo_store)
        await photo_store.create_photo(
            filename="5.jpg", source="signal", photo_id="photo_confirmed",
            description="two kids in a paddling pool")
        await photo_store.update_people(
            "photo_confirmed", [{"label": "Zara"}], agent_reported=True)
        await photo_store.create_photo(
            filename="6.jpg", source="signal", photo_id="photo_likely",
            description="a child eating ice cream")
        await photo_store.apply_inferred_people(
            "photo_likely", [{"label": "Zara", "confidence": "likely"}])
        await photo_store.create_photo(
            filename="7.jpg", source="signal", photo_id="photo_unsure",
            description="someone far away on a hill")
        await photo_store.apply_inferred_people(
            "photo_unsure", [{"label": "Zara", "confidence": "unsure"}])

        results = await hybrid_search(photo_store, query="Zara")
        ids = [r.photo.id for r in results]
        assert ids[0] == "photo_confirmed"
        assert ids[1] == "photo_likely"
        assert set(ids[2:]) == {"photo_tag", "photo_desc"}
        # An "unsure" suggestion is not a person: no match.
        assert "photo_unsure" not in ids and "photo_cat" not in ids

    async def test_name_plus_topic(self, photo_store):
        """The name restricts; the topic only ranks within that person."""
        await _family_library(photo_store)
        await photo_store.set_person_note("Zara", "girl, curly hair")
        await photo_store.set_person_note("Erik", "boy")
        await photo_store.create_photo(
            filename="8.jpg", source="signal", photo_id="photo_erik_bday",
            description="A boy with a birthday cake", tags=["birthday"],
            people=[{"label": "Erik", "confidence": "captioned"}])
        results = await hybrid_search(photo_store, query="Zara birthday")
        assert [r.photo.id for r in results] == ["photo_desc", "photo_tag"]
        assert results[0].score > results[1].score
        # Same answer however the name is spelled or decorated.
        for query in ("zara's birthday", "ZARA birthday!", "birthday, Zara"):
            assert [r.photo.id for r in await hybrid_search(
                photo_store, query=query)] == ["photo_desc", "photo_tag"]

    async def test_people_and_query_excludes_other_peoples_photos(self, photo_store):
        await _family_library(photo_store)
        await photo_store.create_photo(
            filename="8.jpg", source="signal", photo_id="photo_erik_bday",
            description="Erik's birthday cake", tags=["birthday"])
        await photo_store.create_photo(
            filename="9.jpg", source="signal", photo_id="photo_erik_label",
            description="birthday candles and balloons", tags=["birthday"],
            people=[{"label": "Erik", "confidence": "confirmed"}])
        await photo_store.create_photo(
            filename="10.jpg", source="signal", photo_id="photo_zara_unsure",
            description="a birthday party, kids at a table",
            tags=["birthday"])
        await photo_store.apply_inferred_people(
            "photo_zara_unsure", [{"label": "Zara", "confidence": "unsure"}])
        results = await hybrid_search(
            photo_store, query="birthday", people=["Zara"])
        # Zara's birthday first, then other Zara photos. Erik's birthday
        # photos — a better keyword match — and the unsure guess are out.
        assert [r.photo.id for r in results] == ["photo_desc", "photo_tag"]
        # Putting a known name in the query is the same search.
        await photo_store.set_person_note("Zara", "girl, curly hair")
        in_query = await hybrid_search(photo_store, query="Zara birthday")
        assert [r.photo.id for r in in_query] == ["photo_desc", "photo_tag"]

    async def test_name_in_query_never_returns_another_persons_photo(
        self, wired_store, monkeypatch
    ):
        """The reviewer's repro: 'Zara at the beach' must not show Erik."""
        store = wired_store
        for name in ("Zara", "Erik", "Ann"):
            await store.set_person_note(name, "")
        await store.create_photo(
            filename="a.jpg", source="s", photo_id="photo_zara_beach",
            description="A girl at the beach", tags=["beach"],
            people=[{"label": "Zara", "confidence": "captioned"}])
        await store.create_photo(
            filename="b.jpg", source="s", photo_id="photo_erik_beach",
            description="A boy at the beach", tags=["beach"],
            people=[{"label": "Erik", "confidence": "captioned"}])
        await store.create_photo(
            filename="c.jpg", source="s", photo_id="photo_annmarie",
            description="Ann-Marie with a cake", tags=["cake", "ann-marie"],
            people=[{"label": "Ann-Marie", "confidence": "confirmed"}])
        await store.create_photo(
            filename="d.jpg", source="s", photo_id="photo_swing",
            description="kid on swing", tags=["park"],
            people=[{"label": "Zara", "confidence": "unsure"}])
        await store.create_photo(
            filename="e.jpg", source="s", photo_id="photo_ann",
            description="a woman reading", tags=["sofa"],
            people=[{"label": "Ann", "confidence": "likely"}])

        async def ids(**kwargs):
            return [r.photo.id for r in await hybrid_search(store, **kwargs)]

        assert await ids(query="Zara at the beach") == ["photo_zara_beach"]
        assert await ids(query="beach") == [
            "photo_erik_beach", "photo_zara_beach"]  # no name: both, newest first
        assert await ids(query="beach", people=["Zara"]) == ["photo_zara_beach"]
        # Whole names, both ways round.
        assert await ids(query="Ann") == ["photo_ann"]
        assert await ids(people=["ann"]) == ["photo_ann"]
        assert await ids(query="Ann-Marie") == ["photo_annmarie"]
        assert await ids(query="ann-marie cake") == ["photo_annmarie"]
        assert await ids(people=["Ann-Marie"]) == ["photo_annmarie"]
        # A name with photos but no topic match still returns only theirs.
        assert await ids(query="Erik cake") == ["photo_erik_beach"]

        # ...and with show=True only the right child reaches the screen.
        mgr = _FakeDisplayManager()
        monkeypatch.setattr(display_manager_mod, "get_display_manager", lambda: mgr)
        out = await search_photos(query="Zara at the beach", show=True)
        assert mgr.calls == [("picture", {"image_ids": ["photo_zara_beach"]})]
        assert out["shown"] == 1

    async def test_several_names_need_all_of_them(self, wired_store):
        store = wired_store
        await store.set_person_note("Zara", "")
        await store.set_person_note("Erik", "")
        await store.create_photo(
            filename="a.jpg", source="s", photo_id="photo_zara",
            description="on a swing",
            people=[{"label": "Zara", "confidence": "confirmed"}])
        await store.create_photo(
            filename="b.jpg", source="s", photo_id="photo_erik",
            description="on a bike",
            people=[{"label": "Erik", "confidence": "confirmed"}])
        out = await search_photos(query="Zara and Erik")
        assert out["results"] == []
        assert "No photo has all of Zara, Erik together" in out["hint"]
        assert "Zara alone: 1; Erik alone: 1" in out["hint"]
        out = await search_photos(people=["Zara", "Erik"])
        assert out["results"] == [] and "Zara alone: 1" in out["hint"]

        await store.create_photo(
            filename="c.jpg", source="s", photo_id="photo_both",
            description="two kids",
            people=[{"label": "Zara", "confidence": "likely"},
                    {"label": "Erik", "confidence": "confirmed"}])
        for kwargs in ({"query": "Zara and Erik"}, {"people": ["Zara", "Erik"]},
                       {"query": "Erik", "people": ["zara"]}):
            out = await search_photos(**kwargs)
            assert [r["id"] for r in out["results"]] == ["photo_both"]
            assert out["results"][0]["score"] == 0.9

        out = await search_photos(query="Zara", date_from="2099-01-01")
        assert "without date range: 2" in out["hint"]

    async def test_unsure_guess_is_no_text_match_either(self, photo_store):
        """An unsure suggestion must not match through tags/description."""
        await photo_store.set_person_note("Zara", "")
        await photo_store.create_photo(
            filename="a.jpg", source="s", photo_id="photo_guess",
            description="Zara, maybe, far away on a hill", tags=["zara"])
        await photo_store.apply_inferred_people(
            "photo_guess", [{"label": "Zara", "confidence": "unsure"}])
        assert await hybrid_search(photo_store, query="Zara") == []
        assert await hybrid_search(photo_store, people=["Zara"]) == []
        assert await hybrid_search(photo_store, query="Zara on a hill") == []


@pytest.mark.asyncio
class TestRelevanceFloor:
    async def test_unrelated_query_returns_nothing(self, photo_store):
        await _family_library(photo_store)
        assert await hybrid_search(photo_store, query="dinosaur spaceship") == []

    async def test_unrelated_query_with_embeddings(self, photo_store, monkeypatch):
        """With vectors on, a weakly-similar photo no longer rides along."""
        await photo_store.create_photo(
            filename="a.jpg", source="s", photo_id="photo_weak",
            description="a fluffy cat", embedding=_mix(0, 1, 0.15))
        await photo_store.create_photo(
            filename="b.jpg", source="s", photo_id="photo_close",
            description="waves at dusk", embedding=_mix(0, 2, 0.7))
        await photo_store.create_photo(
            filename="c.jpg", source="s", photo_id="photo_opposite",
            description="a spreadsheet", embedding=-_unit(0))
        monkeypatch.setattr(search_mod, "embed", lambda text: _unit(0))

        results = await hybrid_search(photo_store, query="sunset ocean")
        assert [r.photo.id for r in results] == ["photo_close"]
        assert results[0].score == pytest.approx(0.42, abs=0.01)

    async def test_floor_comes_from_config(self, photo_store, mock_config, monkeypatch):
        await photo_store.create_photo(
            filename="a.jpg", source="s", photo_id="photo_weak",
            description="a fluffy cat", embedding=_mix(0, 1, 0.25))
        monkeypatch.setattr(search_mod, "embed", lambda text: _unit(0))
        assert await hybrid_search(photo_store, query="kitten") == []
        monkeypatch.setattr(mock_config.photos, "search_min_score", 0.1)
        assert len(await hybrid_search(photo_store, query="kitten")) == 1

    async def test_keyword_hit_always_clears_the_floor(self, photo_store, monkeypatch):
        """A literal term match survives even with an unhelpful vector."""
        await _family_library(photo_store)
        monkeypatch.setattr(search_mod, "embed", lambda text: _unit(5))
        results = await hybrid_search(photo_store, query="sofa")
        assert [r.photo.id for r in results] == ["photo_cat"]

    async def test_no_query_lists_newest_first(self, photo_store):
        await _family_library(photo_store)
        results = await hybrid_search(photo_store)
        assert [r.photo.id for r in results] == [
            "photo_bazaar", "photo_cat", "photo_desc", "photo_tag"
        ]
        results = await hybrid_search(photo_store, source="camera")
        assert [r.photo.id for r in results] == ["photo_bazaar"]

    async def test_filler_only_query_lists_newest_first(self, wired_store):
        await _family_library(wired_store)
        newest_first = ["photo_bazaar", "photo_cat", "photo_desc", "photo_tag"]
        for query in ("show me photos", "photos", "any pictures?", "-", '"'):
            results = await hybrid_search(wired_store, query=query)
            assert [r.photo.id for r in results] == newest_first, query
        out = await search_photos(query="show me some photos", limit=2)
        assert [r["id"] for r in out["results"]] == newest_first[:2]
        assert "hint" not in out
        # Filler around a real word is still a real query.
        results = await hybrid_search(wired_store, query="show me cat photos")
        assert [r.photo.id for r in results] == ["photo_cat"]

    async def test_deleted_photos_stay_hidden(self, photo_store):
        await _family_library(photo_store)
        await photo_store.soft_delete_photo("photo_desc")
        results = await hybrid_search(photo_store, query="Zara")
        assert [r.photo.id for r in results] == ["photo_tag"]


@pytest.mark.asyncio
class TestFts:
    async def test_tags_and_people_are_indexed_and_kept_in_sync(self, photo_store):
        await _family_library(photo_store)
        results = await hybrid_search(photo_store, query="market")  # tag only
        assert [r.photo.id for r in results] == ["photo_bazaar"]

        await photo_store.update_tags("photo_cat", ["pets"])
        assert [r.photo.id for r in await hybrid_search(
            photo_store, query="pets")] == ["photo_cat"]
        assert await hybrid_search(photo_store, query="cat") != []  # description
        await photo_store.rename_tag("pets", "animals")
        assert [r.photo.id for r in await hybrid_search(
            photo_store, query="animals")] == ["photo_cat"]
        assert await hybrid_search(photo_store, query="pets") == []
        await photo_store.merge_tags("animals", "creatures")
        assert [r.photo.id for r in await hybrid_search(
            photo_store, query="creatures")] == ["photo_cat"]
        await photo_store.delete_tag("creatures")
        assert await hybrid_search(photo_store, query="creatures") == []

        await photo_store.update_photo("photo_cat", description="a tabby asleep")
        assert await hybrid_search(photo_store, query="fluffy") == []
        assert [r.photo.id for r in await hybrid_search(
            photo_store, query="tabby")] == ["photo_cat"]

    async def test_rebuild_matches_incremental(self, photo_store):
        await _family_library(photo_store)
        await photo_store.rebuild_fts_index()
        await photo_store.rebuild_fts_index()
        results = await hybrid_search(photo_store, query="zara")
        assert {r.photo.id for r in results} == {"photo_tag", "photo_desc"}
        db = photo_store._ensure_db()
        async with db.execute("SELECT COUNT(*) AS n FROM photos_fts") as cur:
            assert (await cur.fetchone())["n"] == 4

    async def test_punctuation_is_harmless(self, photo_store, caplog):
        await _family_library(photo_store)
        with caplog.at_level(logging.WARNING, logger="boxbot.photos.search"):
            for query in ('zara\'s "birthday" (2026)!', "cat AND NOT *", "-", '"'):
                await hybrid_search(photo_store, query=query)
        assert caplog.records == []
        results = await hybrid_search(photo_store, query='"birthday" (candles)')
        assert results[0].photo.id == "photo_desc"

    async def test_index_fault_is_logged_not_swallowed(self, photo_store, caplog):
        await _family_library(photo_store)
        await photo_store._ensure_db().execute("DROP TABLE photos_fts")
        with caplog.at_level(logging.ERROR, logger="boxbot.photos.search"):
            assert await hybrid_search(photo_store, query="cat") == []
        assert any("FTS query failed" in r.message for r in caplog.records)


@pytest.mark.asyncio
class TestSearchEntryPoint:
    async def test_rows_are_slim_and_get_is_full(self, wired_store):
        await wired_store.create_photo(
            filename="a.jpg", source="signal", sender="Jacob",
            photo_id="photo_a", description="Zara " + "x" * 300, tags=["park"])
        await wired_store.apply_inferred_people("photo_a", [
            {"label": "Zara", "confidence": "likely", "basis": "curls"},
            {"label": "Erik", "confidence": "unsure", "basis": "blurry"},
        ])
        out = await search_photos(query="Zara")
        row = out["results"][0]
        assert set(row) == {
            "id", "description", "tags", "people", "created", "source",
            "sender", "unsure", "score",
        }
        assert row["people"] == [{"name": "Zara", "confidence": "likely"}]
        assert row["unsure"] == ["Erik"]
        assert len(row["description"]) == 160 and row["description"].endswith("…")
        assert len(row["created"]) == 10
        assert row["sender"] == "Jacob" and row["source"] == "signal"

        full = await search_photos(mode="get", photo_id="photo_a")
        assert len(full["description"]) == 305
        assert full["filename"] == "a.jpg" and "width" in full
        assert full["people"][0]["basis"] == "curls"
        assert full["unsure"] == [{"name": "Erik", "basis": "blurry"}]
        assert full["people_confirmed"] is False

    async def test_empty_result_carries_a_hint_not_broadened_rows(self, wired_store):
        await _family_library(wired_store)
        out = await search_photos(query="Zara", tags=["beach"])
        assert out["results"] == []
        assert "without tags: 2" in out["hint"]

        out = await search_photos(query="Zara", date_from="2099-01-01")
        assert out["results"] == [] and "without date range: 2" in out["hint"]

        out = await search_photos(query="dinosaur spaceship")
        assert out["results"] == []
        assert "do not show unrelated photos" in out["hint"]

        out = await search_photos(query="Zara")
        assert "hint" not in out

    async def test_query_count_does_not_grow_with_the_library(self, wired_store):
        """One search (hint included) is a fixed handful of SQL statements."""
        statements: list[str] = []

        async def run(n_photos: int, **kwargs) -> int:
            for i in range(n_photos):
                await wired_store.create_photo(
                    filename=f"{i}.jpg", source="signal",
                    description=f"Zara photo {i}", tags=["park", f"t{i}"],
                    people=[{"label": "Zara", "confidence": "likely"}])
            statements.clear()
            await search_photos(**kwargs)
            return len([s for s in statements if s.lstrip().startswith("SELECT")])

        await wired_store._ensure_db().set_trace_callback(statements.append)
        hit = {"query": "Zara park"}
        miss = {"query": "Zara park", "tags": ["nope"], "people": ["Zara"],
                "date_from": "2000-01-01", "source": "signal",
                "in_slideshow": True}  # empty: every hint relaxation runs
        small_hit, small_miss = await run(3, **hit), await run(0, **miss)
        big_hit, big_miss = await run(40, **hit), await run(0, **miss)
        assert small_hit == big_hit == small_miss == big_miss
        assert big_hit <= 6
        out = await search_photos(**miss)
        assert out["results"] == [] and "without tags: 43" in out["hint"]

    async def test_tool_and_sdk_share_backend_and_defaults(self, wired_store):
        for i in range(DEFAULT_LIMIT + 5):
            await wired_store.create_photo(
                filename=f"{i}.jpg", source="signal",
                description=f"Zara photo {i}")
        tool_out = json.loads(
            await SearchPhotosTool().execute(mode="search", query="Zara"))
        sdk_out = await process_action(
            {"_sdk": "photos.search", "query": "Zara"}, ActionContext())
        assert sdk_out.pop("status") == "ok"
        assert len(tool_out["results"]) == DEFAULT_LIMIT
        assert tool_out == sdk_out

        # query is optional on the tool.
        assert "query" not in SearchPhotosTool.parameters["required"]
        assert "Required" not in (
            SearchPhotosTool.parameters["properties"]["query"]["description"])
        listing = json.loads(await SearchPhotosTool().execute(mode="search"))
        assert len(listing["results"]) == DEFAULT_LIMIT

    async def test_search_action_validates_filters(self, wired_store):
        res = await process_action(
            {"_sdk": "photos.search", "people": "Zara"}, ActionContext())
        assert res["status"] == "error"
        res = await process_action(
            {"_sdk": "photos.search", "limit": "lots"}, ActionContext())
        assert res["status"] == "error"


class _FakeDisplayManager:
    def __init__(self, ok=True):
        self.ok = ok
        self.calls = []

    async def switch(self, name, args=None):
        self.calls.append((name, args))
        return self.ok


@pytest.mark.asyncio
class TestShow:
    async def test_show_true_displays_results_in_one_call(
        self, wired_store, monkeypatch
    ):
        await _family_library(wired_store)
        mgr = _FakeDisplayManager()
        monkeypatch.setattr(display_manager_mod, "_display_manager_instance", mgr,
                            raising=False)
        monkeypatch.setattr(display_manager_mod, "get_display_manager", lambda: mgr)
        out = json.loads(await SearchPhotosTool().execute(
            mode="search", query="Zara", show=True))
        assert out["shown"] == 2
        name, args = mgr.calls[0]
        assert name == "picture"
        assert args["image_ids"] == [r["id"] for r in out["results"]]

    async def test_nothing_found_shows_nothing(self, wired_store, monkeypatch):
        await _family_library(wired_store)
        mgr = _FakeDisplayManager()
        monkeypatch.setattr(display_manager_mod, "get_display_manager", lambda: mgr)
        out = await search_photos(query="dinosaur spaceship", show=True)
        assert out["shown"] == 0 and out["results"] == [] and "hint" in out
        assert mgr.calls == []

    async def test_show_without_display_reports_it(self, wired_store, monkeypatch):
        await _family_library(wired_store)
        monkeypatch.setattr(display_manager_mod, "get_display_manager", lambda: None)
        out = await search_photos(query="Zara", show=True)
        assert out["shown"] == 0
        assert out["show_error"] == "display manager not running"
        assert len(out["results"]) == 2

    async def test_default_does_not_touch_the_screen(self, wired_store, monkeypatch):
        await _family_library(wired_store)
        mgr = _FakeDisplayManager()
        monkeypatch.setattr(display_manager_mod, "get_display_manager", lambda: mgr)
        out = await search_photos(query="Zara")
        assert "shown" not in out and mgr.calls == []


# ---------------------------------------------------------------------------
# SDK surface (sandbox side)
# ---------------------------------------------------------------------------


@pytest.fixture
def recorder(monkeypatch):
    calls: list[tuple[str, dict]] = []
    replies: dict[str, dict] = {}

    def stub(action_type, payload, *, timeout=30):
        calls.append((action_type, payload))
        return {"status": "ok", **replies.get(action_type, {})}

    monkeypatch.setattr(_transport, "request", stub)
    stub.calls = calls
    stub.replies = replies
    return stub


class TestSdkPhotos:
    def test_search_leaves_defaults_to_the_backend(self, recorder):
        recorder.replies["photos.search"] = {
            "results": [{
                "id": "photo_a", "description": "d", "tags": ["park"],
                "people": [{"name": "Zara", "confidence": "likely"},
                           {"name": "Erik", "confidence": "confirmed"}],
                "created": "2026-11-01", "source": "signal", "sender": "Jacob",
                "score": 1.2,
            }],
        }
        results = sdk_photos.search("Zara")
        assert recorder.calls[-1] == ("photos.search", {"query": "Zara"})
        rec = results[0]
        assert rec.names == ["Zara", "Erik"]
        assert rec.people[0]["confidence"] == "likely"
        assert "Zara (likely)" in repr(rec) and "'Erik'" in repr(rec)
        assert rec.created_at == "2026-11-01"
        assert results.hint is None and results.shown is None

    def test_search_show_and_filters(self, recorder):
        recorder.replies["photos.search"] = {
            "results": [], "hint": "No match", "shown": 0}
        results = sdk_photos.search(
            "Zara", tags=["park"], date_from="2026-01-01", limit=5, show=True)
        assert recorder.calls[-1][1] == {
            "query": "Zara", "tags": ["park"], "date_from": "2026-01-01",
            "limit": 5, "show": True,
        }
        assert results == [] and results.hint == "No match" and results.shown == 0

    def test_search_rejects_bad_args_before_sending(self, recorder):
        with pytest.raises(ValueError):
            sdk_photos.search("Zara", people="Zara")
        with pytest.raises(ValueError):
            sdk_photos.search("Zara", limit=0)
        assert recorder.calls == []

    def test_people_and_roster_calls(self, recorder):
        recorder.replies["photos.set_people"] = {"people": ["Zara"]}
        assert sdk_photos.set_people("photo_a", names=["zara"]) == ["Zara"]
        assert recorder.calls[-1] == (
            "photos.set_people", {"id": "photo_a", "names": ["zara"]})
        sdk_photos.set_people("photo_a", names=[])
        assert recorder.calls[-1][1]["names"] == []

        sdk_photos.confirm_people("photo_a")
        assert recorder.calls[-1] == ("photos.confirm_people", {"id": "photo_a"})
        sdk_photos.unconfirmed(limit=5)
        assert recorder.calls[-1] == ("photos.unconfirmed", {"limit": 5})

        sdk_photos.set_roster("Zara", note="girl, curly hair")
        assert recorder.calls[-1] == (
            "photos.set_roster", {"name": "Zara", "note": "girl, curly hair"})
        sdk_photos.set_roster("Zara", note="")
        recorder.replies["photos.roster"] = {"roster": [{"name": "Zara"}]}
        assert sdk_photos.roster() == [{"name": "Zara"}]
        sdk_photos.remove_from_roster("Zara")
        assert recorder.calls[-1] == ("photos.remove_from_roster", {"name": "Zara"})
