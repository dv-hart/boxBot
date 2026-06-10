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


class TestTruncationKeepsTail:
    """Head+tail truncation — Python tracebacks put the actual
    exception at the END, so the tail must survive."""

    def test_head_and_tail_both_kept(self):
        head = "Traceback (most recent call last):\n"
        middle = "x" * (run_logs.MAX_LOGGED_BYTES * 3)
        tail = "\nKeyError: 'access_token'"
        out = run_logs._truncate_for_log(head + middle + tail)
        assert out.startswith(head)
        assert out.endswith(tail)
        assert "truncated" in out
        # Bounded: head + tail + a small marker.
        assert len(out) <= (
            run_logs.HEAD_LOGGED_BYTES + run_logs.TAIL_LOGGED_BYTES + 200
        )

    def test_short_values_untouched(self):
        assert run_logs._truncate_for_log("hello") == "hello"
        exact = "y" * run_logs.MAX_LOGGED_BYTES
        assert run_logs._truncate_for_log(exact) == exact

    def test_error_tail_survives_in_recorded_run(self, tmp_path: Path):
        error = (
            "Traceback (most recent call last):\n"
            + ("  File \"script.py\", line 1, in <module>\n" * 4000)
            + "RuntimeError: token refresh failed"
        )
        _record(tmp_path, "calendar", status="error", error=error)
        runs = run_logs.list_runs("calendar", root=tmp_path)
        assert runs[0]["error"].endswith("RuntimeError: token refresh failed")
        assert runs[0]["error"].startswith("Traceback")


class TestInputRedaction:
    """Sensitive-looking input values must never land in the run log —
    the agent can read logs back via list_runs."""

    def test_sensitive_keys_redacted(self, tmp_path: Path):
        _record(
            tmp_path,
            "stocks",
            inputs={
                "api_key": "sk-live-12345",
                "Authorization": "Bearer abc",
                "refresh_token": "rt-999",
                "PASSWORD": "hunter2",
                "client_secret": "shhh",
                "credentials": "user:pass",
                "symbol": "AAPL",
            },
        )
        runs = run_logs.list_runs("stocks", root=tmp_path)
        logged = runs[0]["inputs"]
        for key in (
            "api_key", "Authorization", "refresh_token",
            "PASSWORD", "client_secret", "credentials",
        ):
            assert logged[key] == "***redacted***", key
        # Non-sensitive values pass through; key names stay visible.
        assert logged["symbol"] == "AAPL"
        raw = str(runs[0]["inputs"])
        assert "sk-live-12345" not in raw
        assert "hunter2" not in raw

    def test_redaction_recurses_into_nested_structures(self, tmp_path: Path):
        _record(
            tmp_path,
            "stocks",
            inputs={
                "config": {"auth": {"token": "deep"}, "region": "us"},
                "accounts": [
                    {"name": "a", "secret": "s1"},
                    {"name": "b", "api_token": "s2"},
                ],
            },
        )
        logged = run_logs.list_runs("stocks", root=tmp_path)[0]["inputs"]
        # "auth" matches the sensitive pattern, so the whole subtree goes.
        assert logged["config"]["auth"] == "***redacted***"
        assert logged["config"]["region"] == "us"
        assert logged["accounts"][0]["secret"] == "***redacted***"
        assert logged["accounts"][0]["name"] == "a"
        assert logged["accounts"][1]["api_token"] == "***redacted***"

    def test_outputs_not_redacted(self, tmp_path: Path):
        """Scope is inputs only — outputs keep their shape (e.g. a
        calendar event titled 'pick up key')."""
        _record(tmp_path, "calendar", output={"events": [{"title": "pick up key"}]})
        runs = run_logs.list_runs("calendar", root=tmp_path)
        assert runs[0]["output"] == {"events": [{"title": "pick up key"}]}
