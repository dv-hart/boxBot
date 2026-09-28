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


class TestRotationChurn:
    """Re-switching to the display already on screen must do nothing.

    The rotation loop fired every interval regardless of the list
    length, so a one-entry list (``['clock']``) tore down and rebuilt
    every data source and pushed a fresh frame every 30s — 1,850
    redundant switches in one panel log.
    """

    @staticmethod
    def _spec(name: str):
        from boxbot.displays.blocks import TextBlock
        from boxbot.displays.spec import DataSourceSpec, DisplaySpec

        return DisplaySpec(
            name=name,
            theme="boxbot",
            data_sources=[
                DataSourceSpec(name="climate", source_type="static",
                               value={"temp": 71}),
            ],
            root_block=TextBlock(content="{climate.temp}"),
        )

    @pytest.mark.asyncio
    async def test_same_display_switch_keeps_sources(self, isolated_data_dir):
        from boxbot.displays.manager import DisplayManager

        mgr = DisplayManager()
        mgr.register_spec(self._spec("thermostat"))
        await mgr.switch("thermostat", pin=False)
        try:
            source = mgr._data_manager.get_source("climate")
            with patch.object(
                mgr._data_manager, "stop_all", side_effect=AssertionError,
            ):
                assert await mgr.switch("thermostat", pin=False) is True
            assert mgr._data_manager.get_source("climate") is source
        finally:
            await mgr._data_manager.stop_all()

    @pytest.mark.asyncio
    async def test_no_op_switch_still_pins(self, isolated_data_dir):
        from boxbot.displays.manager import DisplayManager

        mgr = DisplayManager()
        mgr.register_spec(self._spec("thermostat"))
        await mgr.switch("thermostat", pin=False)
        try:
            assert mgr.is_pinned() is False
            with patch.object(
                mgr._data_manager, "stop_all", side_effect=AssertionError,
            ):
                await mgr.switch("thermostat", pin=True)
            assert mgr.is_pinned() is True
        finally:
            await mgr._data_manager.stop_all()

    @pytest.mark.asyncio
    async def test_new_args_still_rebuild(self, isolated_data_dir):
        from boxbot.displays.manager import DisplayManager

        mgr = DisplayManager()
        mgr.register_spec(self._spec("thermostat"))
        await mgr.switch("thermostat", args={"unit": "F"})
        try:
            source = mgr._data_manager.get_source("climate")
            await mgr.switch("thermostat", args={"unit": "C"})
            assert mgr._data_manager.get_source("climate") is not source
            assert mgr.get_active_args() == {"unit": "C"}
        finally:
            await mgr._data_manager.stop_all()

    @pytest.mark.asyncio
    async def test_resaved_spec_rebuilds(self, isolated_data_dir):
        from boxbot.displays.manager import DisplayManager

        mgr = DisplayManager()
        mgr.register_spec(self._spec("thermostat"))
        await mgr.switch("thermostat")
        try:
            source = mgr._data_manager.get_source("climate")
            mgr.register_spec(self._spec("thermostat"))  # agent re-saved it
            await mgr.switch("thermostat")
            assert mgr._data_manager.get_source("climate") is not source
        finally:
            await mgr._data_manager.stop_all()

    @pytest.mark.asyncio
    async def test_single_entry_rotation_stops_after_one_switch(
        self, isolated_data_dir,
    ):
        import asyncio

        from boxbot.displays.manager import DisplayManager

        mgr = DisplayManager()
        mgr.register_spec(self._spec("clock"))
        mgr._running = True
        mgr._rotation_active = True
        mgr._rotation_displays = ["clock"]
        mgr._rotation_interval = 30

        switches = []

        async def fake_switch(name, args=None, pin=True):
            switches.append(name)
            return True

        with patch.object(mgr, "switch", side_effect=fake_switch):
            # Returns immediately: a sleeping loop would blow the timeout.
            await asyncio.wait_for(mgr._rotation_loop(), timeout=1.0)

        assert switches == ["clock"]
        assert mgr.is_rotating() is False

    @pytest.mark.asyncio
    async def test_source_change_repaints_the_active_display(
        self, isolated_data_dir,
    ):
        """With no re-switch timer, fresh data is the only thing that can
        repaint a data-bound display. The tick pump does the painting."""
        import asyncio

        from boxbot.displays.manager import DisplayManager

        mgr = DisplayManager()
        mgr.register_spec(self._spec("thermostat"))
        await mgr.switch("thermostat", pin=False)
        mgr._running = True
        tick = asyncio.create_task(mgr._live_tick_loop())
        try:
            source = mgr._data_manager.get_source("climate")
            generation = mgr._frame_generation

            source.update({"temp": 68})
            mgr._data_manager._notify("climate")  # what a changed fetch does
            await asyncio.sleep(1.4)

            assert mgr._frame_generation == generation + 1  # exactly once
            assert mgr._source_dirty is False
            assert mgr._data_manager.get_source("climate") is source

            # Nothing else asked for a repaint, so nothing else happens.
            await asyncio.sleep(1.2)
            assert mgr._frame_generation == generation + 1
        finally:
            mgr._running = False
            tick.cancel()
            await mgr._data_manager.stop_all()

    @pytest.mark.asyncio
    async def test_a_one_hz_source_cannot_drive_a_one_hz_render(
        self, isolated_data_dir,
    ):
        """morning_brief declares the 1 Hz ClockSource, whose payload
        changes every tick. Repainting per changed fetch would mean a full
        render and a ~3 MB frame push every second."""
        import asyncio

        from boxbot.displays.manager import DisplayManager

        mgr = DisplayManager()
        mgr.register_spec(self._spec("brief"))
        await mgr.switch("brief", pin=False)
        mgr._running = True
        tick = asyncio.create_task(mgr._live_tick_loop())
        try:
            generation = mgr._frame_generation
            # 20 changed fetches inside one tick, as a 1 Hz source in a
            # busy loop would look.
            for _ in range(20):
                mgr._data_manager._notify("climate")
                await asyncio.sleep(0.05)
            await asyncio.sleep(1.1)
            renders = mgr._frame_generation - generation
        finally:
            mgr._running = False
            tick.cancel()
            await mgr._data_manager.stop_all()
        assert 1 <= renders <= 3, renders


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
