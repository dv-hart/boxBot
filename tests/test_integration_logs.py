"""Tests for the per-integration run log."""

from __future__ import annotations

import time
from pathlib import Path

from boxbot.integrations import logs as run_logs


def _record(
    root: Path,
    name: str,
    *,
    status: str = "ok",
    output: dict | None = None,
    error: str | None = None,
    inputs: dict | None = None,
    started: float | None = None,
) -> int:
    started = started if started is not None else time.time()
    return run_logs.record_run(
        name=name,
        started_at=started,
        finished_at=started + 0.05,
        status=status,
        inputs=inputs or {},
        output=output,
        error=error,
        root=root,
    )


class TestRunLog:
    def test_record_and_list(self, tmp_path: Path):
        rid = _record(tmp_path, "weather", output={"today": {"high": 79}})
        assert rid > 0
        runs = run_logs.list_runs("weather", root=tmp_path)
        assert len(runs) == 1
        assert runs[0]["status"] == "ok"
        assert runs[0]["output"] == {"today": {"high": 79}}
        assert runs[0]["duration_ms"] >= 0

    def test_records_error_with_message(self, tmp_path: Path):
        _record(tmp_path, "weather", status="error", error="API key expired")
        runs = run_logs.list_runs("weather", root=tmp_path)
        assert runs[0]["status"] == "error"
        assert runs[0]["error"] == "API key expired"
        assert runs[0]["output"] is None

    def test_list_returns_newest_first(self, tmp_path: Path):
        base = time.time()
        _record(tmp_path, "weather", started=base, output={"id": 1})
        _record(tmp_path, "weather", started=base + 1, output={"id": 2})
        _record(tmp_path, "weather", started=base + 2, output={"id": 3})
        runs = run_logs.list_runs("weather", root=tmp_path)
        assert [r["output"]["id"] for r in runs] == [3, 2, 1]

    def test_list_per_integration_isolation(self, tmp_path: Path):
        _record(tmp_path, "weather", output={"a": 1})
        _record(tmp_path, "calendar", output={"b": 2})
        weather = run_logs.list_runs("weather", root=tmp_path)
        calendar = run_logs.list_runs("calendar", root=tmp_path)
        assert len(weather) == len(calendar) == 1
        assert weather[0]["output"] == {"a": 1}
        assert calendar[0]["output"] == {"b": 2}

    def test_pruning_caps_at_max(self, tmp_path: Path, monkeypatch):
        # Force a tiny cap so the test is fast.
        monkeypatch.setattr(run_logs, "MAX_RUNS_PER_INTEGRATION", 5)
        base = time.time()
        for i in range(12):
            _record(tmp_path, "weather", started=base + i, output={"i": i})
        runs = run_logs.list_runs("weather", limit=20, root=tmp_path)
        assert len(runs) == 5
        # The five we keep are the most recent five (i = 7..11).
        kept = sorted(r["output"]["i"] for r in runs)
        assert kept == [7, 8, 9, 10, 11]

    def test_pruning_per_integration_independent(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(run_logs, "MAX_RUNS_PER_INTEGRATION", 3)
        base = time.time()
        for i in range(5):
            _record(tmp_path, "weather", started=base + i)
        for i in range(2):
            _record(tmp_path, "calendar", started=base + i)
        # Weather gets pruned to 3, calendar untouched at 2.
        assert len(run_logs.list_runs("weather", root=tmp_path)) == 3
        assert len(run_logs.list_runs("calendar", root=tmp_path)) == 2

    def test_truncates_giant_payloads(self, tmp_path: Path):
        big = {"data": "x" * (run_logs.MAX_LOGGED_BYTES * 2)}
        _record(tmp_path, "big", output=big)
        runs = run_logs.list_runs("big", root=tmp_path)
        assert len(runs) == 1
        # Truncated marker survived; the value is no longer parseable JSON
        # because we cut the string mid-stream — list_runs returns the raw text.
        assert "truncated" in str(runs[0]["output"])

    def test_list_limit_zero_returns_empty(self, tmp_path: Path):
        _record(tmp_path, "weather")
        assert run_logs.list_runs("weather", limit=0, root=tmp_path) == []

    def test_unknown_integration_returns_empty(self, tmp_path: Path):
        assert run_logs.list_runs("ghost", root=tmp_path) == []
