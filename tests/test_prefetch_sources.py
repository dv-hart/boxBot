"""Tests for prefetch source lanes: SDK section selection + memory gather."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from boxbot.prefetch import sources as sources_mod
from boxbot.prefetch.bundle import PrefetchBundle
from boxbot.prefetch.request import PrefetchRequest
from boxbot.prefetch.sources import (
    _sdk_source,
    _split_module_doc,
    gather_memory_candidates,
)

_DOC = """# bb.display — declarative screens

Intro paragraph.

## Blocks

block docs

## Data sources

source docs

## Preview

preview docs
"""

_SHORT_DOC = """# bb.camera — stills

capture(), capture_cropped()
"""


def _req(**kw):
    defaults = dict(
        key="conv-1", key_kind="conversation", channel="voice",
        person="Jacob", text="show me the thermostat",
    )
    defaults.update(kw)
    return PrefetchRequest(**defaults)


class TestSplitModuleDoc:
    def test_preamble_and_sections(self):
        preamble, sections = _split_module_doc(_DOC)
        assert preamble.startswith("# bb.display")
        assert "Intro paragraph." in preamble
        assert [h for h, _ in sections] == ["Blocks", "Data sources", "Preview"]
        assert sections[0][1].startswith("## Blocks")
        assert "block docs" in sections[0][1]

    def test_sectionless_doc(self):
        preamble, sections = _split_module_doc(_SHORT_DOC)
        assert "capture()" in preamble
        assert sections == []


class TestSdkSectionSource:
    @pytest.fixture(autouse=True)
    def _corpus(self, monkeypatch):
        corpus = {
            "display": _split_module_doc(_DOC),
            "camera": _split_module_doc(_SHORT_DOC),
        }
        monkeypatch.setattr(
            sources_mod, "_sdk_section_corpus", lambda: corpus,
        )

    def test_candidates_list_sections_and_whole_short_docs(self):
        run = _sdk_source(_req())
        assert run is not None
        assert "- display.0: Blocks" in run.candidates
        assert "- display.2: Preview" in run.candidates
        # Sectionless doc offered whole, by module name.
        assert "- camera: bb.camera — stills" in run.candidates

    @pytest.mark.asyncio
    async def test_materialize_splices_selected_sections(self):
        run = _sdk_source(_req())
        bundle = PrefetchBundle()
        await run.materialize({"sections": ["display.0", "display.2"]}, bundle)

        text = bundle.sdk_modules["display"]
        assert text.startswith("# bb.display")  # preamble always rides
        assert "block docs" in text and "preview docs" in text
        assert "source docs" not in text
        assert "2 of 3 sections" in text  # pointer at the rest
        assert 'load_skill("bb", "modules/display.md")' in text
        assert bundle.sdk_sections["display"] == ["display.0", "display.2"]

    def test_menu_excludes_already_loaded_sections(self):
        run = _sdk_source(_req(already_loaded=["display.0"]))
        assert run is not None
        assert "display.0" not in run.candidates
        assert "- display.2: Preview" in run.candidates

    def test_menu_excludes_fully_loaded_modules(self):
        run = _sdk_source(_req(already_loaded=["bb/modules/display.md"]))
        assert run is not None
        assert "display" not in run.candidates
        assert "camera" in run.candidates

    def test_bare_names_do_not_blank_modules(self):
        """A SKILL named like a module (agent-created skills pick any
        name) must not remove the module from the menu — whole-module
        state matches only the unambiguous bb/modules path form."""
        run = _sdk_source(_req(already_loaded=["display"]))
        assert run is not None
        assert "- display.0: Blocks" in run.candidates

    def test_everything_loaded_skips_the_lane(self):
        run = _sdk_source(_req(already_loaded=[
            "bb/modules/display.md", "bb/modules/camera.md",
        ]))
        assert run is None, "no candidates left → no selector call at all"

    @pytest.mark.asyncio
    async def test_materialize_whole_module_key(self):
        run = _sdk_source(_req())
        bundle = PrefetchBundle()
        await run.materialize({"sections": ["camera"]}, bundle)

        assert "capture()" in bundle.sdk_modules["camera"]
        assert "partial doc" not in bundle.sdk_modules["camera"]
        assert bundle.sdk_sections["camera"] == ["camera"]

    @pytest.mark.asyncio
    async def test_invalid_keys_ignored_and_cap_applied(self):
        run = _sdk_source(_req())
        bundle = PrefetchBundle()
        await run.materialize(
            {"sections": [
                "display.99", "nope.0", "display.0", "display.1",
                "display.2", "camera", "camera",
            ]},
            bundle,
        )
        # Cap is 4 valid picks: display.0/1/2 + camera.
        assert set(bundle.sdk_modules) == {"display", "camera"}
        total = sum(len(v) for v in bundle.sdk_sections.values())
        assert total <= 4

    @pytest.mark.asyncio
    async def test_all_sections_selected_drops_the_pointer(self):
        run = _sdk_source(_req())
        bundle = PrefetchBundle()
        await run.materialize(
            {"sections": ["display.0", "display.1", "display.2"]}, bundle,
        )
        assert "partial doc" not in bundle.sdk_modules["display"]


class TestPredictedSdkSections:
    def test_section_keys_win(self):
        b = PrefetchBundle(
            sdk_modules={"display": "..."},
            sdk_sections={"display": ["display.0", "display.2"]},
        )
        assert b.predicted_sdk_sections() == ["display.0", "display.2"]

    def test_legacy_bundles_fall_back_to_module_paths(self):
        b = PrefetchBundle(sdk_modules={"zz_legacy": "..."})
        assert b.predicted_sdk_sections() == ["bb/modules/zz_legacy.md"]


class TestSplitModuleDocFences:
    def test_h2_inside_code_fence_is_not_a_section(self):
        doc = (
            "# bb.x\n\nintro\n\n## Real\n\nbody\n\n"
            "```markdown\n## Not a section\nfenced\n```\n\ntail\n"
        )
        preamble, sections = _split_module_doc(doc)
        assert [h for h, _ in sections] == ["Real"]
        # The fenced pseudo-heading stays inside the Real section, with
        # its fence intact.
        assert "## Not a section" in sections[0][1]
        assert sections[0][1].count("```") == 2


class TestDottedModuleNames:
    @pytest.mark.asyncio
    async def test_dotted_filename_selects_cleanly(self, monkeypatch):
        corpus = {"zz_legacy.cameras": _split_module_doc(_DOC)}
        monkeypatch.setattr(
            sources_mod, "_sdk_section_corpus", lambda: corpus,
        )
        run = _sdk_source(_req())
        assert "- zz_legacy.cameras.0: Blocks" in run.candidates
        bundle = PrefetchBundle()
        await run.materialize({"sections": ["zz_legacy.cameras.0"]}, bundle)
        assert "block docs" in bundle.sdk_modules["zz_legacy.cameras"]
        assert bundle.sdk_sections["zz_legacy.cameras"] == ["zz_legacy.cameras.0"]


class TestGatherMemoryCandidates:
    @pytest.mark.asyncio
    async def test_person_pass_merges_and_dedups(self, monkeypatch):
        def _c(cid):
            return SimpleNamespace(id=cid, summary=f"s-{cid}", type="person")

        calls: list[dict] = []

        async def fake_search(store, query, **kw):
            calls.append({"query": query, **kw})
            if query == "Jacob":
                return [_c("m2"), _c("m3")]
            return [_c("m1"), _c("m2")]

        monkeypatch.setattr(
            "boxbot.memory.search.hybrid_search", fake_search,
        )
        cands = await gather_memory_candidates(
            object(), text="show me the thermostat", person="Jacob",
        )
        assert [c.id for c in cands] == ["m1", "m2", "m3"]
        # The supplemental person pass must not spend a second embed on
        # the reply path.
        person_call = next(c for c in calls if c["query"] == "Jacob")
        assert person_call["allow_vector"] is False

    @pytest.mark.asyncio
    async def test_memory_lane_materialize_bumps_relevance(self, monkeypatch):
        def _c(cid):
            return SimpleNamespace(id=cid, summary=f"s-{cid}", type="person")

        async def fake_search(store, query, **kw):
            return [_c("m1")]

        monkeypatch.setattr(
            "boxbot.memory.search.hybrid_search", fake_search,
        )
        bumped: list[str] = []

        class _Store:
            async def update_memory_relevance(self, mid):
                bumped.append(mid)

        run = await sources_mod._memory_source(_req(), _Store())
        bundle = PrefetchBundle()
        await run.materialize({"memory_ids": ["m1"]}, bundle)
        assert bundle.memories == [("m1", "s-m1")]
        assert bumped == ["m1"]

    @pytest.mark.asyncio
    async def test_no_store_returns_empty(self):
        assert await gather_memory_candidates(
            None, text="x", person="Jacob",
        ) == []
