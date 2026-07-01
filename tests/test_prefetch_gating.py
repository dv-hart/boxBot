"""Tests for prefetch gating helpers (enabled/channels/mode)."""

from __future__ import annotations

from types import SimpleNamespace

import boxbot.prefetch as prefetch


def _install_cfg(monkeypatch, *, enabled, mode="shadow",
                 channels=("whatsapp", "signal", "trigger")):
    cfg = SimpleNamespace(
        prefetch=SimpleNamespace(
            enabled=enabled, mode=mode, channels=list(channels),
            token_budget=1500,
        )
    )
    monkeypatch.setattr("boxbot.core.config.get_config", lambda: cfg)


class TestGating:
    def test_disabled_never_prefetches(self, monkeypatch):
        _install_cfg(monkeypatch, enabled=False)
        assert prefetch.should_prefetch("whatsapp") is False

    def test_enabled_respects_channel_list(self, monkeypatch):
        _install_cfg(monkeypatch, enabled=True, channels=("whatsapp",))
        assert prefetch.should_prefetch("whatsapp") is True
        assert prefetch.should_prefetch("signal") is False
        assert prefetch.should_prefetch("voice") is False

    def test_mode_and_is_active(self, monkeypatch):
        _install_cfg(monkeypatch, enabled=True, mode="shadow")
        assert prefetch.prefetch_mode() == "shadow"
        assert prefetch.is_active() is False

        _install_cfg(monkeypatch, enabled=True, mode="active")
        assert prefetch.prefetch_mode() == "active"
        assert prefetch.is_active() is True

    def test_missing_config_fails_safe(self, monkeypatch):
        def _boom():
            raise RuntimeError("config not loaded")

        monkeypatch.setattr("boxbot.core.config.get_config", _boom)
        assert prefetch.should_prefetch("whatsapp") is False
        assert prefetch.prefetch_mode() == "shadow"
        assert prefetch.is_active() is False
