"""Tests for the home_assistant integration.

Mix of two levels:

- ``test_manifest_loads_via_real_loader`` exercises the real loader so
  the manifest can't drift out of spec (bad name, unknown input type,
  bad secret name, etc.) without a test breaking.
- The rest of the tests import the integration's ``script.py`` as a
  module with a stubbed ``boxbot_sdk`` and a faked ``httpx`` so each
  action handler can be exercised in isolation. The integration runner
  has its own end-to-end tests; we don't re-test that path here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from boxbot.integrations.loader import discover_integrations


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "integrations" / "home_assistant" / "script.py"


# ---------------------------------------------------------------------------
# Manifest — real loader
# ---------------------------------------------------------------------------


def test_manifest_loads_via_real_loader():
    """The shipped integrations/ tree should include home_assistant
    and the manifest should pass validation."""
    metas = {m.name: m for m in discover_integrations()}
    assert "home_assistant" in metas, f"home_assistant not in {sorted(metas)}"
    meta = metas["home_assistant"]
    assert "action" in meta.inputs
    # Secrets declared, in canonical order.
    assert set(meta.secrets) == {"HOME_ASSISTANT_URL", "HOME_ASSISTANT_TOKEN"}
    assert 1 <= meta.timeout <= 300


# ---------------------------------------------------------------------------
# Script import — stub boxbot_sdk first, then load script as module
# ---------------------------------------------------------------------------


def _install_boxbot_sdk_stub(workspace_write: MagicMock) -> None:
    """Insert a fake boxbot_sdk package into sys.modules.

    The script's top-level import is
    ``from boxbot_sdk.integration import inputs, return_output`` —
    those need to resolve. ``camera_snapshot`` also does a late
    ``import boxbot_sdk as bb`` and calls ``bb.workspace.write``.
    """
    integration_mod = SimpleNamespace(
        inputs=lambda: {},
        return_output=lambda _value: None,
    )
    workspace_mod = SimpleNamespace(write=workspace_write)
    bb_mod = SimpleNamespace(
        integration=integration_mod,
        workspace=workspace_mod,
    )
    sys.modules["boxbot_sdk"] = bb_mod  # type: ignore[assignment]
    sys.modules["boxbot_sdk.integration"] = integration_mod  # type: ignore[assignment]
    sys.modules["boxbot_sdk.workspace"] = workspace_mod  # type: ignore[assignment]


def _load_script_module(workspace_write: MagicMock):
    """Import integrations/home_assistant/script.py as a module."""
    _install_boxbot_sdk_stub(workspace_write)
    spec = importlib.util.spec_from_file_location("ha_integration_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def workspace_write() -> MagicMock:
    return MagicMock(return_value={"path": "", "size": 0, "kind": "binary"})


@pytest.fixture
def script(workspace_write: MagicMock):
    return _load_script_module(workspace_write)


# ---------------------------------------------------------------------------
# Fake httpx — canned response router
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        json_body: Any = None,
        content: bytes | None = None,
        text: str = "",
    ):
        self.status_code = status_code
        self._json = json_body
        # Real httpx populates ``content`` from the response body. Mirror
        # that so the script's ``if resp.content`` guard sees a body
        # whenever the test set json_body.
        if content is not None:
            self.content = content
        elif json_body is not None:
            self.content = b"<json>"  # placeholder; only truthiness matters
        else:
            self.content = b""
        self.text = text

    def json(self) -> Any:
        return self._json

    def raise_for_status(self) -> None:
        if 400 <= self.status_code < 600:
            import httpx as real_httpx
            raise real_httpx.HTTPStatusError(
                f"{self.status_code}",
                request=real_httpx.Request("GET", "http://test"),
                response=real_httpx.Response(self.status_code, text=self.text),
            )


class _FakeHttpx:
    """Drop-in for the ``httpx`` module the script imports.

    Tests set ``responses`` to a dict keyed by ``(method, path)`` (with
    ``path`` the URL suffix after the base) or override ``handler``
    for ad-hoc routing.
    """

    def __init__(self):
        self.responses: dict[tuple[str, str], _FakeResponse] = {}
        self.calls: list[dict[str, Any]] = []
        # Mirror httpx's exception classes — the script catches these.
        import httpx as real_httpx
        self.HTTPStatusError = real_httpx.HTTPStatusError
        self.ConnectError = real_httpx.ConnectError
        self.Timeout = real_httpx.Timeout

    def _route(self, method: str, url: str, **kwargs) -> _FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        # Match by URL suffix to keep test setup decoupled from base URL.
        for (m, suffix), resp in self.responses.items():
            if m == method and url.endswith(suffix):
                return resp
        return _FakeResponse(404, text=f"no canned response for {method} {url}")

    def get(self, url: str, **kwargs) -> _FakeResponse:
        return self._route("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> _FakeResponse:
        return self._route("POST", url, **kwargs)


@pytest.fixture
def fake_httpx(script, monkeypatch) -> _FakeHttpx:
    fh = _FakeHttpx()
    monkeypatch.setattr(script, "httpx", fh)
    return fh


@pytest.fixture
def with_secrets(monkeypatch):
    monkeypatch.setenv("BOXBOT_SECRET_HOME_ASSISTANT_URL", "http://ha.test:8123")
    monkeypatch.setenv("BOXBOT_SECRET_HOME_ASSISTANT_TOKEN", "tok-abc")


# ---------------------------------------------------------------------------
# Connection / secrets
# ---------------------------------------------------------------------------


def test_missing_url_returns_error(script, monkeypatch):
    monkeypatch.delenv("BOXBOT_SECRET_HOME_ASSISTANT_URL", raising=False)
    monkeypatch.setenv("BOXBOT_SECRET_HOME_ASSISTANT_TOKEN", "tok")
    captured: list[Any] = []
    monkeypatch.setattr(script, "return_output", captured.append)
    assert script._load_connection() is None
    assert "HOME_ASSISTANT_URL" in captured[0]["error"]


def test_missing_token_returns_error(script, monkeypatch):
    monkeypatch.setenv("BOXBOT_SECRET_HOME_ASSISTANT_URL", "http://ha.test:8123")
    monkeypatch.delenv("BOXBOT_SECRET_HOME_ASSISTANT_TOKEN", raising=False)
    captured: list[Any] = []
    monkeypatch.setattr(script, "return_output", captured.append)
    assert script._load_connection() is None
    assert "HOME_ASSISTANT_TOKEN" in captured[0]["error"]


def test_trailing_slash_stripped(script, monkeypatch):
    monkeypatch.setenv("BOXBOT_SECRET_HOME_ASSISTANT_URL", "http://ha.test:8123/")
    monkeypatch.setenv("BOXBOT_SECRET_HOME_ASSISTANT_TOKEN", "tok")
    url, _ = script._load_connection()  # type: ignore[misc]
    assert url == "http://ha.test:8123"


# ---------------------------------------------------------------------------
# get_states / get_state
# ---------------------------------------------------------------------------


def test_get_states_returns_trimmed_entities(script, fake_httpx, with_secrets):
    fake_httpx.responses[("GET", "/api/states")] = _FakeResponse(
        200,
        json_body=[
            {
                "entity_id": "light.living_room",
                "state": "on",
                "attributes": {"friendly_name": "Living Room", "brightness": 200},
                "last_changed": "2026-05-17T14:22:01Z",
            },
            {
                "entity_id": "switch.basement",
                "state": "off",
                "attributes": {"friendly_name": "Basement Fan"},
                "last_changed": "2026-05-17T10:00:00Z",
            },
        ],
    )
    result = script._get_states("http://ha.test:8123", "tok", {})
    assert len(result["entities"]) == 2
    # Trimmed payload — no `attributes` key.
    assert set(result["entities"][0].keys()) == {
        "entity_id", "state", "friendly_name", "last_changed",
    }
    assert result["entities"][0]["friendly_name"] == "Living Room"


def test_get_states_domain_filter(script, fake_httpx, with_secrets):
    fake_httpx.responses[("GET", "/api/states")] = _FakeResponse(
        200,
        json_body=[
            {"entity_id": "light.a", "state": "on", "attributes": {}, "last_changed": ""},
            {"entity_id": "switch.b", "state": "off", "attributes": {}, "last_changed": ""},
            {"entity_id": "light.c", "state": "off", "attributes": {}, "last_changed": ""},
        ],
    )
    result = script._get_states("http://ha.test:8123", "tok", {"domain": "light"})
    ids = [e["entity_id"] for e in result["entities"]]
    assert ids == ["light.a", "light.c"]


def test_get_state_returns_attributes(script, fake_httpx, with_secrets):
    fake_httpx.responses[("GET", "/api/states/light.living_room")] = _FakeResponse(
        200,
        json_body={
            "state": "on",
            "attributes": {"brightness": 200, "rgb_color": [255, 0, 0]},
            "last_changed": "2026-05-17T14:22:01Z",
        },
    )
    result = script._get_state(
        "http://ha.test:8123", "tok", {"entity_id": "light.living_room"}
    )
    assert result["state"] == "on"
    assert result["attributes"]["brightness"] == 200
    assert result["last_changed"] == "2026-05-17T14:22:01Z"


def test_get_state_404(script, fake_httpx, with_secrets):
    fake_httpx.responses[("GET", "/api/states/light.ghost")] = _FakeResponse(404)
    result = script._get_state(
        "http://ha.test:8123", "tok", {"entity_id": "light.ghost"}
    )
    assert "not found" in result["error"]


def test_get_state_missing_entity_id(script):
    result = script._get_state("http://ha.test:8123", "tok", {})
    assert "entity_id" in result["error"]


# ---------------------------------------------------------------------------
# call_service — happy + mutation policy
# ---------------------------------------------------------------------------


def test_call_service_happy_path_with_data(script, fake_httpx, with_secrets):
    fake_httpx.responses[("POST", "/api/services/light/turn_on")] = _FakeResponse(
        200, json_body=[{"entity_id": "light.living_room"}],
    )
    result = script._call_service(
        "http://ha.test:8123",
        "tok",
        {
            "domain": "light",
            "service": "turn_on",
            "entity_id": "light.living_room",
            "service_data": {"rgb_color": [180, 30, 90], "brightness": 80},
        },
    )
    assert result["ok"] is True
    assert result["affected"] == ["light.living_room"]
    # Confirm the body merged entity_id + service_data.
    call = fake_httpx.calls[-1]
    body = call["json"]
    assert body == {
        "rgb_color": [180, 30, 90],
        "brightness": 80,
        "entity_id": "light.living_room",
    }


@pytest.mark.parametrize("domain", ["alarm_control_panel", "lock", "cover"])
def test_call_service_blocked_domains(script, fake_httpx, with_secrets, domain):
    result = script._call_service(
        "http://ha.test:8123",
        "tok",
        {"domain": domain, "service": "turn_on", "entity_id": f"{domain}.x"},
    )
    assert "blocked" in result["error"]
    assert domain in result["error"]
    # And: no HTTP call was made.
    assert fake_httpx.calls == []


def test_call_service_missing_domain(script):
    result = script._call_service("http://ha.test:8123", "tok", {"service": "turn_on"})
    assert "domain" in result["error"]


# ---------------------------------------------------------------------------
# camera_snapshot
# ---------------------------------------------------------------------------


def test_camera_snapshot_writes_workspace_path(
    script, fake_httpx, workspace_write, with_secrets
):
    fake_httpx.responses[("GET", "/api/camera_proxy/camera.front_door")] = _FakeResponse(
        200, content=b"\xff\xd8\xff\xe0fakejpegbytes",
    )
    result = script._camera_snapshot(
        "http://ha.test:8123", "tok", {"entity_id": "camera.front_door"}
    )
    # Path shape: tmp/ha/<safe_entity>_<stamp>.jpg
    assert result["image_path"].startswith("tmp/ha/camera_front_door_")
    assert result["image_path"].endswith(".jpg")
    assert result["entity_id"] == "camera.front_door"
    assert result["captured_at"].endswith("Z")
    # And: the JPEG bytes hit bb.workspace.write.
    workspace_write.assert_called_once()
    written_path, written_bytes = workspace_write.call_args.args
    assert written_path == result["image_path"]
    assert written_bytes == b"\xff\xd8\xff\xe0fakejpegbytes"


def test_camera_snapshot_rejects_non_camera_entity(script, with_secrets):
    result = script._camera_snapshot(
        "http://ha.test:8123", "tok", {"entity_id": "light.living_room"}
    )
    assert "camera.* entity" in result["error"]


def test_camera_snapshot_404(script, fake_httpx, with_secrets):
    fake_httpx.responses[("GET", "/api/camera_proxy/camera.ghost")] = _FakeResponse(404)
    result = script._camera_snapshot(
        "http://ha.test:8123", "tok", {"entity_id": "camera.ghost"}
    )
    assert "not found" in result["error"]


def test_camera_snapshot_empty_body(script, fake_httpx, with_secrets):
    fake_httpx.responses[("GET", "/api/camera_proxy/camera.muted")] = _FakeResponse(
        200, content=b"",
    )
    result = script._camera_snapshot(
        "http://ha.test:8123", "tok", {"entity_id": "camera.muted"}
    )
    assert "empty snapshot" in result["error"]


# ---------------------------------------------------------------------------
# list_services
# ---------------------------------------------------------------------------


def test_list_services_filters_by_domain(script, fake_httpx, with_secrets):
    fake_httpx.responses[("GET", "/api/services")] = _FakeResponse(
        200,
        json_body=[
            {"domain": "light", "services": {
                "turn_on": {"description": "Turn on light"},
                "turn_off": {"description": "Turn off light"},
            }},
            {"domain": "switch", "services": {
                "turn_on": {"description": "Turn on switch"},
            }},
        ],
    )
    result = script._list_services("http://ha.test:8123", "tok", {"domain": "light"})
    services = result["entities"]
    assert len(services) == 2
    assert {s["service"] for s in services} == {"turn_on", "turn_off"}
    assert all(s["domain"] == "light" for s in services)


# ---------------------------------------------------------------------------
# Dispatcher (main)
# ---------------------------------------------------------------------------


def test_main_unknown_action_returns_error(script, monkeypatch, with_secrets):
    captured: list[Any] = []
    monkeypatch.setattr(script, "get_inputs", lambda: {"action": "blow_up"})
    monkeypatch.setattr(script, "return_output", captured.append)
    script.main()
    assert "unknown action" in captured[0]["error"]


def test_main_dispatches_to_action(script, fake_httpx, monkeypatch, with_secrets):
    captured: list[Any] = []
    fake_httpx.responses[("GET", "/api/states")] = _FakeResponse(200, json_body=[])
    monkeypatch.setattr(script, "get_inputs", lambda: {"action": "get_states"})
    monkeypatch.setattr(script, "return_output", captured.append)
    script.main()
    assert captured[0] == {"entities": []}


def test_main_401_token_hint(script, fake_httpx, monkeypatch, with_secrets):
    """A 401 from HA should surface a token-refresh hint, not a raw stack trace."""
    captured: list[Any] = []
    fake_httpx.responses[("GET", "/api/states")] = _FakeResponse(401, text="Unauthorized")
    monkeypatch.setattr(script, "get_inputs", lambda: {"action": "get_states"})
    monkeypatch.setattr(script, "return_output", captured.append)
    script.main()
    assert "HOME_ASSISTANT_TOKEN" in captured[0]["error"]
