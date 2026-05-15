"""Tests for display rotation state persistence.

The rotation list is in-memory on the DisplayManager, but
``set_rotation`` writes through to ``data/displays/rotation.json``
so it survives a restart. ``_get_rotation_config`` reads the
persisted file first, falling back to config defaults only when no
state has ever been written.

The companion bug fixed alongside this: ``unpin`` previously read
the config defaults, silently clobbering any agent-set rotation.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point boxbot.core.paths.DISPLAYS_DIR at a tmp dir for the test."""
    monkeypatch.setenv("BOXBOT_DATA_DIR", str(tmp_path))
    # paths.py snapshots BOXBOT_DATA_DIR at import time, so we have to
    # reload to pick up the override. Cache-bust both paths and the
    # manager module since the manager imports DISPLAYS_DIR by name.
    import importlib

    import boxbot.core.paths as paths_mod
    importlib.reload(paths_mod)
    import boxbot.displays.manager as mgr_mod
    importlib.reload(mgr_mod)
    yield tmp_path / "displays"


class TestRotationStatePersistence:
    def test_round_trip(self, isolated_data_dir: Path):
        from boxbot.displays.manager import (
            _load_rotation_state,
            _persist_rotation_state,
        )

        _persist_rotation_state({"displays": ["clock", "weather"], "interval": 45})
        loaded = _load_rotation_state()
        assert loaded == {"displays": ["clock", "weather"], "interval": 45}

    def test_missing_file_returns_none(self, isolated_data_dir: Path):
        from boxbot.displays.manager import _load_rotation_state

        assert _load_rotation_state() is None

    def test_malformed_json_returns_none(self, isolated_data_dir: Path):
        from boxbot.displays.manager import _load_rotation_state, _rotation_state_path

        path = _rotation_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not json {", encoding="utf-8")
        assert _load_rotation_state() is None

    def test_wrong_shape_returns_none(self, isolated_data_dir: Path):
        from boxbot.displays.manager import _load_rotation_state, _rotation_state_path

        path = _rotation_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # interval missing → invalid shape
        path.write_text(
            json.dumps({"displays": ["clock"]}), encoding="utf-8"
        )
        assert _load_rotation_state() is None

    def test_negative_interval_returns_none(self, isolated_data_dir: Path):
        from boxbot.displays.manager import _load_rotation_state, _rotation_state_path

        path = _rotation_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"displays": ["clock"], "interval": -1}), encoding="utf-8"
        )
        assert _load_rotation_state() is None

    def test_persist_none_deletes_file(self, isolated_data_dir: Path):
        from boxbot.displays.manager import (
            _persist_rotation_state,
            _rotation_state_path,
        )

        _persist_rotation_state({"displays": ["a"], "interval": 10})
        path = _rotation_state_path()
        assert path.is_file()
        _persist_rotation_state(None)
        assert not path.exists()

    def test_persist_atomic_write(self, isolated_data_dir: Path):
        from boxbot.displays.manager import (
            _persist_rotation_state,
            _rotation_state_path,
        )

        # First write seeds the file.
        _persist_rotation_state({"displays": ["clock"], "interval": 30})
        # Second write overwrites cleanly.
        _persist_rotation_state({"displays": ["weather"], "interval": 60})
        loaded = json.loads(_rotation_state_path().read_text())
        assert loaded == {"displays": ["weather"], "interval": 60}
        # No tmp file leaked.
        assert not _rotation_state_path().with_suffix(".json.tmp").exists()


class TestGetRotationConfigPreference:
    def test_persisted_state_overrides_config(
        self, isolated_data_dir: Path
    ):
        from boxbot.displays.manager import (
            DisplayManager,
            _persist_rotation_state,
        )

        _persist_rotation_state(
            {"displays": ["clock", "weekly_glance"], "interval": 45}
        )
        mgr = DisplayManager()
        displays, interval = mgr._get_rotation_config()
        assert displays == ["clock", "weekly_glance"]
        assert interval == 45

    def test_config_fallback_when_no_state(
        self, isolated_data_dir: Path
    ):
        from boxbot.displays.manager import DisplayManager

        # No persisted state; should fall through to ``get_config`` and
        # then the static fallback when config isn't loaded in this
        # test process.
        mgr = DisplayManager()
        displays, interval = mgr._get_rotation_config()
        # Either the test-config defaults or the hardcoded fallback —
        # both are valid here; we just want a sane shape.
        assert isinstance(displays, list)
        assert isinstance(interval, int)
        assert interval > 0


class TestSetRotationPersists:
    """``set_rotation`` writes through to the persisted state file.

    Uses a real DisplayManager but stubs ``start_rotation`` so we
    don't need an event loop or registered display specs.
    """

    @pytest.mark.asyncio
    async def test_set_rotation_persists_resolved_values(
        self, isolated_data_dir: Path
    ):
        from boxbot.displays.manager import (
            DisplayManager,
            _load_rotation_state,
        )

        mgr = DisplayManager()

        # Pretend these displays are registered so start_rotation's
        # filter doesn't drop them.
        mgr._specs = {"clock": object(), "weekly_glance": object()}  # type: ignore[assignment]

        # start_rotation does loop creation we don't need here.
        with patch.object(mgr, "_rotation_task", None):
            with patch("asyncio.create_task", return_value=None):
                await mgr.set_rotation(
                    displays=["clock", "weekly_glance"], interval=42,
                )

        loaded = _load_rotation_state()
        assert loaded == {
            "displays": ["clock", "weekly_glance"],
            "interval": 42,
        }

    @pytest.mark.asyncio
    async def test_set_rotation_empty_clears_persisted(
        self, isolated_data_dir: Path
    ):
        from boxbot.displays.manager import (
            DisplayManager,
            _load_rotation_state,
            _persist_rotation_state,
        )

        # Seed something to clear.
        _persist_rotation_state({"displays": ["clock"], "interval": 30})
        assert _load_rotation_state() is not None

        mgr = DisplayManager()
        with patch("asyncio.create_task", return_value=None):
            await mgr.set_rotation(displays=[])
        assert _load_rotation_state() is None


class TestUnpinPreservesRotation:
    """``unpin`` must restart rotation from the *current* in-memory
    list, not re-read config defaults. Otherwise it silently undoes
    any prior ``set_rotation``.
    """

    @pytest.mark.asyncio
    async def test_unpin_uses_current_rotation(
        self, isolated_data_dir: Path
    ):
        from boxbot.displays.manager import DisplayManager

        mgr = DisplayManager()
        mgr._specs = {"a": object(), "b": object()}  # type: ignore[assignment]
        # Manually seed the in-memory rotation, bypassing start_rotation
        # so we can assert what unpin uses.
        mgr._rotation_displays = ["a", "b"]
        mgr._rotation_interval = 99
        mgr._pinned = True

        captured: dict = {}

        def fake_start(displays=None, interval=None):
            captured["displays"] = displays
            captured["interval"] = interval

        with patch.object(mgr, "start_rotation", side_effect=fake_start):
            await mgr.unpin()

        assert captured["displays"] == ["a", "b"]
        assert captured["interval"] == 99
        assert mgr._pinned is False

    @pytest.mark.asyncio
    async def test_unpin_holds_display_when_no_rotation(
        self, isolated_data_dir: Path
    ):
        from boxbot.displays.manager import DisplayManager

        mgr = DisplayManager()
        mgr._rotation_displays = []  # nothing to rotate to
        mgr._pinned = True

        called = False

        def fake_start(displays=None, interval=None):
            nonlocal called
            called = True

        with patch.object(mgr, "start_rotation", side_effect=fake_start):
            await mgr.unpin()

        assert called is False
        assert mgr._pinned is False
