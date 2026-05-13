"""Tests for dropping the ``operational`` memory type (lifecycle step 7).

The taxonomy is now {person, household, methodology}. The dream phase
no longer skips memories typed ``operational``. The extraction tool
schema rejects ``operational``. The classifier for the migration
script correctly triages remaining rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def test_extraction_schema_no_longer_accepts_operational():
    """Tool schema enum must not include operational."""
    from boxbot.memory.extraction import EXTRACTION_TOOL

    memory_schema = (
        EXTRACTION_TOOL["input_schema"]["properties"]
        ["extracted_memories"]["items"]
    )
    type_enum = memory_schema["properties"]["type"]["enum"]
    assert "operational" not in type_enum
    assert set(type_enum) == {"person", "household", "methodology"}


def test_extraction_prompt_no_longer_documents_operational():
    """Documentation/operational paragraphs must be gone — otherwise
    the model will still try to emit it and fail validation."""
    from boxbot.memory.extraction import EXTRACTION_SYSTEM_PROMPT
    text = EXTRACTION_SYSTEM_PROMPT
    # No paragraph defining operational
    assert "**operational**:" not in text
    # Affirmative "no operational type" sentence is present
    assert "There is no \"operational\" type" in text


def test_store_memory_types_set_excludes_operational():
    from boxbot.memory.store import MEMORY_TYPES
    assert "operational" not in MEMORY_TYPES
    assert MEMORY_TYPES == {"person", "household", "methodology"}


def test_search_memory_tool_enum_excludes_operational():
    from boxbot.tools.builtins.search_memory import SearchMemoryTool
    schema = SearchMemoryTool.parameters["properties"]["types"]["items"]
    assert "operational" not in schema["enum"]
    assert set(schema["enum"]) == {"person", "household", "methodology"}


def test_dream_no_longer_exempts_operational_typed_rows():
    """Old rows in the DB may still carry ``type='operational'``;
    those must now be eligible for dedup."""
    from boxbot.memory.dream import find_near_duplicates
    import inspect

    src = inspect.getsource(find_near_duplicates)
    # The is_op skip block must be gone
    assert "is_op" not in src
    assert 'type == "operational"' not in src


class TestMigrationClassifier:
    """Verify the step-7 migration script triages real-corpus
    operational summaries the way we want."""

    @staticmethod
    def _classify(summary: str, content: str = "") -> str:
        from migrate_operational_to_methodology import _classify
        return _classify(summary, content)

    def test_workspace_pointer_promotes_to_methodology(self):
        assert self._classify(
            "Weekly weight log at workspace/data/weight.csv (date, lbs)."
        ) == "promote"
        assert self._classify(
            "Saved Q1 expense report to workspace/notes/jacob/q1_summary.md"
        ) == "promote"

    def test_durable_lesson_promotes_to_methodology(self):
        """Real corpus from Pi: the DST recheck note carries a lesson."""
        assert self._classify(
            "Cron triggers fixed to PDT offsets on 2026-04-30; "
            "DST re-check needed Nov 2026."
        ) == "promote"

    def test_routine_delivery_log_invalidates(self):
        assert self._classify(
            "Morning briefing delivered to Jacob on May 3 2026."
        ) == "invalidate"
        assert self._classify(
            "Evening review May 6 delivered to Jacob via text."
        ) == "invalidate"

    def test_ambiguous_keeps_as_is(self):
        """Some operational rows don't fit any of the patterns —
        leave them alone (operational has budget=0 in injection)."""
        assert self._classify(
            "Some random operational note with no pattern match"
        ) == "keep"

    def test_real_corpus_kept_lesson_rows_promote(self):
        """The three rows the step-2 backfill preserved via the
        lesson-signal veto: ideally these promote so they remain
        useful as methodology under step 7."""
        assert self._classify(
            "Cron triggers fixed to PDT offsets on 2026-04-30; "
            "DST re-check needed Nov 2026."
        ) == "promote"

    def test_real_corpus_calendar_working_promote_or_keep(self):
        """The 5/12 + 5/13 'calendar working' rows are stale-state
        cleanups, not durable lessons. Acceptable for either to
        promote (kept as methodology) or invalidate (cleaned up
        with the rest of the calendar churn). They should NOT be
        ambiguous-keep, since they have signal content."""
        v1 = self._classify(
            "Calendar integration confirmed working 5/12; reauth todos closed"
        )
        v2 = self._classify(
            "Google Calendar integration confirmed working 5/13; "
            "stale todos closed."
        )
        # Either bucket is acceptable; "keep" would mean classification
        # is too weak. Lock in that some decision was made.
        assert v1 in {"promote", "invalidate"}
        assert v2 in {"promote", "invalidate"}
