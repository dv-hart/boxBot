"""Tests for the package-install request flow (WS3 Part B).

Covers:
- :func:`boxbot.packages.service.validate_package_spec` — strict
  PyPI-name / exact-pin validation (no URLs, paths, specifiers).
- :class:`boxbot.packages.store.PackageStore` — durable queue
  roundtrip and compare-and-set status transitions.
- The ``packages.*`` sandbox action handler — request / status / list.
- Admin approval replies through :class:`MessageRouter` — admin vs
  non-admin, approve vs deny, unknown id; outbound channels and the
  pip installer are mocked throughout.
- The ``bb.packages`` SDK surface — pending-on-request semantics,
  raise-on-dispatch-error per the SDK-wide rule.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from boxbot.communication.channels import (
    Channel,
    register_outbound_channel,
)
from boxbot.communication.router import MessageRouter
from boxbot.core.events import SignalMessage, WhatsAppMessage
from boxbot.packages.service import (
    handle_admin_reply,
    parse_admin_reply,
    submit_request,
    validate_package_spec,
)
from boxbot.packages.store import PackageStore, set_package_store
from boxbot.tools._sandbox_actions import ActionContext, process_action


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def package_store(tmp_path):
    """A PackageStore on a temp DB, installed as the process singleton."""
    store = PackageStore(db_path=tmp_path / "packages.db")
    set_package_store(store)
    yield store
    set_package_store(None)


class FakeOutbound:
    """Minimal OutboundChannel stub that records sent texts."""

    name = "fake"

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    async def send_text(self, phone: str, message: str) -> bool:
        self.sent.append((phone, message))
        return True

    async def send_attachment(self, phone, file_path, caption=None) -> bool:
        return True

    async def download_media(self, media_id):
        return None


@pytest.fixture
def signal_outbound():
    out = FakeOutbound()
    register_outbound_channel(Channel.SIGNAL, out)
    yield out
    register_outbound_channel(Channel.SIGNAL, None)


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------


class TestValidatePackageSpec:
    @pytest.mark.parametrize("spec", [
        "requests",
        "google-api-python-client",
        "Pillow",
        "ruamel.yaml",
        "typing_extensions",
        "numpy==2.1.0",
        "requests==2.32.3",
        "foo==1!2.0.post1+local",
    ])
    def test_accepts_valid_specs(self, spec):
        assert validate_package_spec(spec) == spec

    def test_strips_whitespace(self):
        assert validate_package_spec("  requests  ") == "requests"

    @pytest.mark.parametrize("spec", [
        "",
        "   ",
        "https://evil.example/pkg.whl",
        "git+https://github.com/x/y",
        "./local-dir",
        "../escape",
        "/abs/path",
        "pkg[extra]",
        "pkg>=1.0",
        "pkg~=1.0",
        "pkg<2",
        "pkg==1.0; os_name=='posix'",
        "pkg name",
        "-e .",
        "--upgrade",
        "-requests",
        "pkg==",
        "pkg==*",
        "pkg==1.0 --index-url=http://evil",
        "pkg@file:///etc",
    ])
    def test_rejects_invalid_specs(self, spec):
        with pytest.raises(ValueError):
            validate_package_spec(spec)


# ---------------------------------------------------------------------------
# Reply parsing
# ---------------------------------------------------------------------------


class TestParseAdminReply:
    def test_approve(self):
        assert parse_admin_reply("approve pkg ab12cd34") == (
            "approve", "ab12cd34", ""
        )

    def test_deny_with_note(self):
        verb, rid, note = parse_admin_reply("deny pkg ab12cd34 too risky")
        assert (verb, rid, note) == ("deny", "ab12cd34", "too risky")

    def test_case_insensitive_and_id_lowercased(self):
        assert parse_admin_reply("  APPROVE  pkg AB12CD34 ") == (
            "approve", "ab12cd34", ""
        )

    @pytest.mark.parametrize("text", [
        "",
        "hello there",
        "approve",
        "approve pkg",
        "approve pkg xyz",          # not hex
        "approve package ab12cd34",
        "please approve pkg ab12cd34",  # must be the whole message
    ])
    def test_non_commands_return_none(self, text):
        assert parse_admin_reply(text) is None


# ---------------------------------------------------------------------------
# Store roundtrip
# ---------------------------------------------------------------------------


class TestPackageStore:
    @pytest.mark.asyncio
    async def test_create_and_get(self, package_store):
        req = await package_store.create_request(
            "requests", "http calls", requested_by="whatsapp:+1555"
        )
        assert req["status"] == "pending"
        assert len(req["id"]) == 8

        fetched = await package_store.get_request(req["id"])
        assert fetched == req

    @pytest.mark.asyncio
    async def test_get_unknown_returns_none(self, package_store):
        assert await package_store.get_request("deadbeef") is None

    @pytest.mark.asyncio
    async def test_find_pending(self, package_store):
        req = await package_store.create_request("numpy", "math")
        found = await package_store.find_pending("numpy")
        assert found is not None and found["id"] == req["id"]
        assert await package_store.find_pending("scipy") is None

    @pytest.mark.asyncio
    async def test_list_with_status_filter(self, package_store):
        a = await package_store.create_request("a", "r")
        await package_store.create_request("b", "r")
        await package_store.set_status(a["id"], "denied", expect="pending")

        pending = await package_store.list_requests(status="pending")
        assert [r["package"] for r in pending] == ["b"]
        all_rows = await package_store.list_requests()
        assert len(all_rows) == 2

    @pytest.mark.asyncio
    async def test_set_status_compare_and_set(self, package_store):
        req = await package_store.create_request("requests", "r")
        updated = await package_store.set_status(
            req["id"], "approved", resolved_by="+1555", expect="pending"
        )
        assert updated["status"] == "approved"
        assert updated["resolved_by"] == "+1555"

        # A second approve must fail the CAS — status is no longer pending.
        again = await package_store.set_status(
            req["id"], "approved", expect="pending"
        )
        assert again is None

    @pytest.mark.asyncio
    async def test_set_status_rejects_invalid(self, package_store):
        req = await package_store.create_request("requests", "r")
        with pytest.raises(ValueError):
            await package_store.set_status(req["id"], "banana")

    @pytest.mark.asyncio
    async def test_survives_reopen(self, tmp_path):
        """Durability: a second store on the same path sees the row."""
        path = tmp_path / "packages.db"
        first = PackageStore(db_path=path)
        req = await first.create_request("requests", "r")

        second = PackageStore(db_path=path)
        assert (await second.get_request(req["id"]))["package"] == "requests"


# ---------------------------------------------------------------------------
# Action handler
# ---------------------------------------------------------------------------


class TestPackagesActionHandler:
    @pytest.mark.asyncio
    async def test_request_queues_pending(self, package_store):
        result = await process_action(
            {
                "_sdk": "packages.request",
                "package": "requests",
                "reason": "http calls for the rss skill",
            },
            ActionContext(),
        )
        assert result["status"] == "ok"
        assert result["request"]["status"] == "pending"
        assert result["duplicate"] is False
        # No auth manager / no admins in tests → notified 0; the
        # request must still be durably queued.
        row = await package_store.get_request(result["request"]["id"])
        assert row is not None and row["status"] == "pending"

    @pytest.mark.asyncio
    async def test_request_rejects_bad_spec(self, package_store):
        result = await process_action(
            {
                "_sdk": "packages.request",
                "package": "pkg>=1.0",
                "reason": "nope",
            },
            ActionContext(),
        )
        assert result["status"] == "error"
        assert "specifier" in result["message"]

    @pytest.mark.asyncio
    async def test_request_requires_reason(self, package_store):
        result = await process_action(
            {"_sdk": "packages.request", "package": "requests"},
            ActionContext(),
        )
        assert result["status"] == "error"
        assert "reason" in result["message"]

    @pytest.mark.asyncio
    async def test_duplicate_pending_not_requeued(self, package_store):
        first = await process_action(
            {"_sdk": "packages.request", "package": "requests",
             "reason": "r"},
            ActionContext(),
        )
        second = await process_action(
            {"_sdk": "packages.request", "package": "requests",
             "reason": "r again"},
            ActionContext(),
        )
        assert second["duplicate"] is True
        assert second["request"]["id"] == first["request"]["id"]
        assert len(await package_store.list_requests()) == 1

    @pytest.mark.asyncio
    async def test_status_roundtrip(self, package_store):
        req = await package_store.create_request("numpy", "math")
        result = await process_action(
            {"_sdk": "packages.status", "id": req["id"]},
            ActionContext(),
        )
        assert result["status"] == "ok"
        assert result["request"]["package"] == "numpy"

    @pytest.mark.asyncio
    async def test_status_unknown_id_errors(self, package_store):
        result = await process_action(
            {"_sdk": "packages.status", "id": "deadbeef"},
            ActionContext(),
        )
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_list_filters(self, package_store):
        await package_store.create_request("a", "r")
        result = await process_action(
            {"_sdk": "packages.list", "request_status": "pending"},
            ActionContext(),
        )
        assert result["status"] == "ok"
        assert len(result["requests"]) == 1

        bad = await process_action(
            {"_sdk": "packages.list", "request_status": "banana"},
            ActionContext(),
        )
        assert bad["status"] == "error"

    @pytest.mark.asyncio
    async def test_unknown_action_errors(self, package_store):
        result = await process_action(
            {"_sdk": "packages.approve", "id": "x"},  # no such verb — ever
            ActionContext(),
        )
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Admin approval through the router
# ---------------------------------------------------------------------------


async def _register_admin_and_user(auth_manager):
    """Bootstrap an admin (+15551230001) and a plain user (+15551230002)."""
    code = await auth_manager.generate_bootstrap_code()
    await auth_manager.register_user(
        "+15551230001", "Admin", code, channel="signal"
    )
    code2 = await auth_manager.generate_registration_code("+15551230001")
    await auth_manager.register_user(
        "+15551230002", "Plain", code2, channel="signal"
    )


async def _drain_install_tasks():
    from boxbot.packages import service

    tasks = list(service._install_tasks)
    if tasks:
        await asyncio.gather(*tasks)


class TestRouterApprovalInterception:
    @pytest.mark.asyncio
    async def test_admin_approve_installs_and_replies(
        self, auth_manager, event_bus, package_store, signal_outbound
    ):
        await _register_admin_and_user(auth_manager)
        req = await package_store.create_request("requests", "http calls")

        events = []

        async def collect(event):
            events.append(event)

        event_bus.subscribe(SignalMessage, collect)

        router = MessageRouter(auth=auth_manager)
        with patch(
            "boxbot.packages.installer.install_package",
            new=AsyncMock(return_value=(True, "Successfully installed")),
        ) as mock_install:
            routed = await router.route_incoming(
                Channel.SIGNAL, "+15551230001", f"approve pkg {req['id']}"
            )
            await _drain_install_tasks()

        assert routed is True
        # Intercepted: the agent never saw the approval as a message.
        assert events == []
        mock_install.assert_awaited_once_with("requests")

        row = await package_store.get_request(req["id"])
        assert row["status"] == "installed"
        assert row["resolved_by"] == "+15551230001"

        # Admin got the immediate ack and the install confirmation.
        texts = [m for _, m in signal_outbound.sent]
        assert any("Approved requests" in t for t in texts)
        assert any("Installed requests" in t for t in texts)

    @pytest.mark.asyncio
    async def test_admin_deny_records_note(
        self, auth_manager, event_bus, package_store, signal_outbound
    ):
        await _register_admin_and_user(auth_manager)
        req = await package_store.create_request("requests", "r")

        router = MessageRouter(auth=auth_manager)
        routed = await router.route_incoming(
            Channel.SIGNAL, "+15551230001",
            f"deny pkg {req['id']} we do not need this",
        )
        assert routed is True

        row = await package_store.get_request(req["id"])
        assert row["status"] == "denied"
        assert row["note"] == "we do not need this"
        assert any("Denied" in m for _, m in signal_outbound.sent)

    @pytest.mark.asyncio
    async def test_pip_failure_marks_failed(
        self, auth_manager, event_bus, package_store, signal_outbound
    ):
        await _register_admin_and_user(auth_manager)
        req = await package_store.create_request("requests", "r")

        router = MessageRouter(auth=auth_manager)
        with patch(
            "boxbot.packages.installer.install_package",
            new=AsyncMock(return_value=(False, "ERROR: no matching distribution")),
        ):
            await router.route_incoming(
                Channel.SIGNAL, "+15551230001", f"approve pkg {req['id']}"
            )
            await _drain_install_tasks()

        row = await package_store.get_request(req["id"])
        assert row["status"] == "failed"
        assert "no matching distribution" in row["note"]
        assert any("failed" in m for _, m in signal_outbound.sent)

    @pytest.mark.asyncio
    async def test_non_admin_reply_routes_as_normal_message(
        self, auth_manager, event_bus, package_store, signal_outbound
    ):
        await _register_admin_and_user(auth_manager)
        req = await package_store.create_request("requests", "r")

        events = []

        async def collect(event):
            events.append(event)

        event_bus.subscribe(SignalMessage, collect)

        router = MessageRouter(auth=auth_manager)
        routed = await router.route_incoming(
            Channel.SIGNAL, "+15551230002", f"approve pkg {req['id']}"
        )
        assert routed is True
        # Delivered to the agent as ordinary text…
        assert len(events) == 1
        assert events[0].text == f"approve pkg {req['id']}"
        # …and the request is untouched. Non-admins cannot approve.
        row = await package_store.get_request(req["id"])
        assert row["status"] == "pending"
        assert signal_outbound.sent == []

    @pytest.mark.asyncio
    async def test_unknown_request_id_gets_polite_reply(
        self, auth_manager, event_bus, package_store, signal_outbound
    ):
        await _register_admin_and_user(auth_manager)
        router = MessageRouter(auth=auth_manager)
        routed = await router.route_incoming(
            Channel.SIGNAL, "+15551230001", "approve pkg deadbeef"
        )
        assert routed is True
        assert any("No package request" in m for _, m in signal_outbound.sent)

    @pytest.mark.asyncio
    async def test_double_approve_is_idempotent(
        self, auth_manager, event_bus, package_store, signal_outbound
    ):
        await _register_admin_and_user(auth_manager)
        req = await package_store.create_request("requests", "r")

        router = MessageRouter(auth=auth_manager)
        with patch(
            "boxbot.packages.installer.install_package",
            new=AsyncMock(return_value=(True, "ok")),
        ) as mock_install:
            await router.route_incoming(
                Channel.SIGNAL, "+15551230001", f"approve pkg {req['id']}"
            )
            await _drain_install_tasks()
            await router.route_incoming(
                Channel.SIGNAL, "+15551230001", f"approve pkg {req['id']}"
            )
            await _drain_install_tasks()

        # Second approve resolved nothing and spawned no second install.
        assert mock_install.await_count == 1
        assert any("already" in m for _, m in signal_outbound.sent)

    @pytest.mark.asyncio
    async def test_whatsapp_path_also_intercepts(
        self, auth_manager, event_bus, package_store
    ):
        """The interception is channel-agnostic — WhatsApp inbound too."""
        out = FakeOutbound()
        register_outbound_channel(Channel.WHATSAPP, out)
        try:
            code = await auth_manager.generate_bootstrap_code()
            await auth_manager.register_user(
                "+15551230001", "Admin", code, channel="whatsapp"
            )
            req = await package_store.create_request("requests", "r")

            events = []

            async def collect(event):
                events.append(event)

            event_bus.subscribe(WhatsAppMessage, collect)

            router = MessageRouter(auth=auth_manager)
            routed = await router.route_incoming(
                Channel.WHATSAPP, "+15551230001",
                f"deny pkg {req['id']}",
            )
            assert routed is True
            assert events == []
            row = await package_store.get_request(req["id"])
            assert row["status"] == "denied"
            assert any("Denied" in m for _, m in out.sent)
        finally:
            register_outbound_channel(Channel.WHATSAPP, None)


# ---------------------------------------------------------------------------
# submit_request notification fan-out
# ---------------------------------------------------------------------------


class TestSubmitRequestNotification:
    @pytest.mark.asyncio
    async def test_notifies_admins_on_their_channel(
        self, auth_manager, package_store, signal_outbound
    ):
        await _register_admin_and_user(auth_manager)
        from boxbot.communication.auth import set_auth_manager

        set_auth_manager(auth_manager)
        try:
            result = await submit_request(
                "requests", "http calls", requested_by="voice:room"
            )
        finally:
            set_auth_manager(None)

        # Only the admin is messaged — the plain user is not.
        assert result["admins_notified"] == 1
        assert result["admins"] == 1
        phone, text = signal_outbound.sent[0]
        assert phone == "+15551230001"
        assert "requests" in text
        assert result["request"]["id"] in text
        assert "approve pkg" in text
        assert "voice:room" in text


# ---------------------------------------------------------------------------
# Installer (pip + subprocess mocked / absent)
# ---------------------------------------------------------------------------


class TestInstaller:
    @pytest.mark.asyncio
    async def test_missing_pip_fails_cleanly(self, tmp_path):
        from boxbot.packages.installer import install_package

        ok, output = await install_package(
            "requests", venv=tmp_path / "no-venv"
        )
        assert ok is False
        assert "pip not found" in output

    def test_fix_permissions_applies_policy(self, tmp_path):
        from boxbot.packages.installer import fix_sandbox_permissions

        venv = tmp_path / "venv"
        site = venv / "lib" / "python3.13" / "site-packages" / "pkg"
        site.mkdir(parents=True)
        (site / "mod.py").write_text("x = 1\n")
        bin_dir = venv / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "pip").write_text("#!/bin/sh\n")
        (bin_dir / "python3.13").write_text("ELF\n")
        (bin_dir / "console-script").write_text("#!/bin/sh\n")

        # Group that won't exist — chown is skipped, chmod still applies.
        fix_sandbox_permissions(venv, group="no-such-group-xyz")

        assert (site / "mod.py").stat().st_mode & 0o777 == 0o640
        assert site.stat().st_mode & 0o777 == 0o750
        assert (bin_dir / "pip").stat().st_mode & 0o777 == 0o700
        assert (bin_dir / "python3.13").stat().st_mode & 0o777 == 0o750
        assert (bin_dir / "console-script").stat().st_mode & 0o777 == 0o640


# ---------------------------------------------------------------------------
# SDK surface (bb.packages)
# ---------------------------------------------------------------------------


class TestPackagesSDK:
    def test_request_returns_pending_record(self):
        from boxbot.sdk import packages as bb_packages

        canned = {
            "status": "ok",
            "request": {
                "id": "ab12cd34",
                "package": "requests",
                "status": "pending",
            },
            "duplicate": False,
            "admins_notified": 2,
            "admins": 2,
        }
        with patch(
            "boxbot.sdk._transport.request", return_value=canned
        ) as mock_req:
            record = bb_packages.request("requests", reason="http calls")

        action, payload = mock_req.call_args[0]
        assert action == "packages.request"
        assert payload == {"package": "requests", "reason": "http calls"}
        assert record["id"] == "ab12cd34"
        assert record["status"] == "pending"
        assert record["admins_notified"] == 2
        assert record["duplicate"] is False

    def test_request_raises_action_error_on_rejection(self):
        import boxbot.sdk as bb
        from boxbot.sdk import packages as bb_packages

        with patch(
            "boxbot.sdk._transport.request",
            return_value={"status": "error", "message": "invalid package name"},
        ):
            with pytest.raises(bb.ActionError, match="invalid package name"):
                bb_packages.request("bad name", reason="r")

    def test_denied_is_a_status_not_an_exception(self):
        """A human 'no' must read as data, never raise."""
        from boxbot.sdk import packages as bb_packages

        canned = {
            "status": "ok",
            "request": {"id": "ab12cd34", "status": "denied",
                        "note": "not needed"},
        }
        with patch("boxbot.sdk._transport.request", return_value=canned):
            record = bb_packages.status("ab12cd34")
        assert record["status"] == "denied"
        assert record["note"] == "not needed"

    def test_status_unknown_id_raises(self):
        import boxbot.sdk as bb
        from boxbot.sdk import packages as bb_packages

        with patch(
            "boxbot.sdk._transport.request",
            return_value={"status": "error",
                          "message": "no package request 'x' found"},
        ):
            with pytest.raises(bb.ActionError, match="no package request"):
                bb_packages.status("deadbeef")

    def test_list_passes_filter(self):
        from boxbot.sdk import packages as bb_packages

        with patch(
            "boxbot.sdk._transport.request",
            return_value={"status": "ok", "requests": [{"id": "a"}]},
        ) as mock_req:
            rows = bb_packages.list("pending")
        _, payload = mock_req.call_args[0]
        assert payload == {"request_status": "pending"}
        assert rows == [{"id": "a"}]
