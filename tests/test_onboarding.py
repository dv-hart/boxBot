"""Tests for the onboarding orchestration: bb.auth SDK actions,
UserRegistered event, and first-run setup todo seeding.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from boxbot.communication.auth import AuthManager
from boxbot.communication.router import Channel, MessageRouter
from boxbot.core.events import UserRegistered, WhatsAppMessage


# ---------------------------------------------------------------------------
# AuthManager.get_code_creator (new helper used by router)
# ---------------------------------------------------------------------------


class TestCodeCreatorLookup:
    @pytest.mark.asyncio
    async def test_bootstrap_code_creator(self, auth_manager: AuthManager):
        code = await auth_manager.generate_bootstrap_code()
        assert await auth_manager.get_code_creator(code) == "bootstrap"

    @pytest.mark.asyncio
    async def test_admin_invite_code_creator(self, auth_manager: AuthManager):
        boot = await auth_manager.generate_bootstrap_code()
        await auth_manager.register_user("+15550000001", "Admin", boot)
        invite = await auth_manager.generate_registration_code(
            created_by="+15550000001"
        )
        assert await auth_manager.get_code_creator(invite) == "+15550000001"

    @pytest.mark.asyncio
    async def test_unknown_code_returns_none(self, auth_manager: AuthManager):
        assert await auth_manager.get_code_creator("999999") is None


# ---------------------------------------------------------------------------
# UserRegistered event publication from the router
# ---------------------------------------------------------------------------


class TestUserRegisteredEvent:
    @pytest.mark.asyncio
    async def test_bootstrap_emits_admin_event_with_blank_inviter(
        self, auth_manager: AuthManager, event_bus
    ):
        code = await auth_manager.generate_bootstrap_code()
        events: list[UserRegistered] = []

        async def on_event(e):
            events.append(e)
        event_bus.subscribe(UserRegistered, on_event)

        router = MessageRouter(auth=auth_manager)
        ok = await router.route_incoming(
            Channel.WHATSAPP, "+15551110001", code, sender_name="A"
        )
        assert ok is True
        assert len(events) == 1
        assert events[0].phone == "+15551110001"
        assert events[0].role == "admin"
        assert events[0].invited_by_phone == ""

    @pytest.mark.asyncio
    async def test_invite_emits_user_event_with_admin_inviter(
        self, auth_manager: AuthManager, event_bus
    ):
        boot = await auth_manager.generate_bootstrap_code()
        await auth_manager.register_user("+15550000001", "Admin", boot)
        invite = await auth_manager.generate_registration_code(
            created_by="+15550000001"
        )
        events: list[UserRegistered] = []

        async def on_event(e):
            events.append(e)
        event_bus.subscribe(UserRegistered, on_event)

        router = MessageRouter(auth=auth_manager)
        ok = await router.route_incoming(
            Channel.WHATSAPP, "+15552220002", invite, sender_name="B"
        )
        assert ok is True
        assert len(events) == 1
        assert events[0].role == "user"
        assert events[0].invited_by_phone == "+15550000001"

    @pytest.mark.asyncio
    async def test_legacy_whatsapp_message_event_still_fires(
        self, auth_manager: AuthManager, event_bus
    ):
        """Existing consumers reading the [REGISTRATION] message must keep working."""
        code = await auth_manager.generate_bootstrap_code()
        wa_events: list[WhatsAppMessage] = []

        async def on_msg(e):
            wa_events.append(e)
        event_bus.subscribe(WhatsAppMessage, on_msg)

        router = MessageRouter(auth=auth_manager)
        await router.route_incoming(
            Channel.WHATSAPP, "+15551110002", code, sender_name="A"
        )
        assert len(wa_events) == 1
        assert "[REGISTRATION]" in wa_events[0].text


# ---------------------------------------------------------------------------
# Sandbox action handler: bb.auth.*
# ---------------------------------------------------------------------------


class TestAuthActions:
    @pytest.mark.asyncio
    async def test_list_users_empty(self, auth_manager: AuthManager):
        from boxbot.communication import auth as auth_module
        from boxbot.tools._sandbox_actions import _handle_auth_action

        auth_module.set_auth_manager(auth_manager)
        try:
            r = await _handle_auth_action("auth.list_users", {})
            assert r["status"] == "ok"
            assert r["users"] == []
        finally:
            auth_module.set_auth_manager(None)

    @pytest.mark.asyncio
    async def test_generate_bootstrap_code_via_action(
        self, auth_manager: AuthManager
    ):
        from boxbot.communication import auth as auth_module
        from boxbot.tools._sandbox_actions import _handle_auth_action

        auth_module.set_auth_manager(auth_manager)
        try:
            r = await _handle_auth_action("auth.generate_bootstrap_code", {})
            assert r["status"] == "ok"
            assert r["code"].isdigit()
            assert len(r["code"]) == 6
        finally:
            auth_module.set_auth_manager(None)

    @pytest.mark.asyncio
    async def test_generate_bootstrap_code_fails_when_admin_exists(
        self, auth_manager: AuthManager
    ):
        from boxbot.communication import auth as auth_module
        from boxbot.tools._sandbox_actions import _handle_auth_action

        boot = await auth_manager.generate_bootstrap_code()
        await auth_manager.register_user("+15550000001", "Admin", boot)
        auth_module.set_auth_manager(auth_manager)
        try:
            r = await _handle_auth_action("auth.generate_bootstrap_code", {})
            assert r["status"] == "error"
            assert "admin" in r["error"].lower()
        finally:
            auth_module.set_auth_manager(None)

    @pytest.mark.asyncio
    async def test_generate_registration_code_requires_admin_context(
        self, auth_manager: AuthManager
    ):
        """The sandbox cannot mint codes — only admins in a WhatsApp turn can."""
        from boxbot.communication import auth as auth_module
        from boxbot.tools._sandbox_actions import _handle_auth_action

        boot = await auth_manager.generate_bootstrap_code()
        await auth_manager.register_user("+15550000001", "Admin", boot)
        auth_module.set_auth_manager(auth_manager)
        try:
            # No conversation context → refuse
            r = await _handle_auth_action(
                "auth.generate_registration_code", {}
            )
            assert r["status"] == "error"
            assert "whatsapp" in r["error"].lower()
        finally:
            auth_module.set_auth_manager(None)

    @pytest.mark.asyncio
    async def test_generate_registration_code_admin_in_whatsapp_conv(
        self, auth_manager: AuthManager
    ):
        from boxbot.communication import auth as auth_module
        from boxbot.tools._sandbox_actions import _handle_auth_action
        from boxbot.tools._tool_context import current_conversation

        boot = await auth_manager.generate_bootstrap_code()
        await auth_manager.register_user("+15550000001", "Admin", boot)
        auth_module.set_auth_manager(auth_manager)

        # Fake conversation with WhatsApp channel from the admin's phone
        fake_conv = MagicMock()
        fake_conv.channel = "whatsapp"
        fake_conv.channel_key = "whatsapp:+15550000001"

        token = current_conversation.set(fake_conv)
        try:
            r = await _handle_auth_action(
                "auth.generate_registration_code", {}
            )
            assert r["status"] == "ok"
            assert r["code"].isdigit()
        finally:
            current_conversation.reset(token)
            auth_module.set_auth_manager(None)

    @pytest.mark.asyncio
    async def test_generate_registration_code_non_admin_refused(
        self, auth_manager: AuthManager
    ):
        from boxbot.communication import auth as auth_module
        from boxbot.tools._sandbox_actions import _handle_auth_action
        from boxbot.tools._tool_context import current_conversation

        boot = await auth_manager.generate_bootstrap_code()
        await auth_manager.register_user("+15550000001", "Admin", boot)
        invite = await auth_manager.generate_registration_code(
            created_by="+15550000001"
        )
        await auth_manager.register_user("+15550000002", "User", invite)

        auth_module.set_auth_manager(auth_manager)
        fake_conv = MagicMock()
        fake_conv.channel = "whatsapp"
        fake_conv.channel_key = "whatsapp:+15550000002"  # the non-admin

        token = current_conversation.set(fake_conv)
        try:
            r = await _handle_auth_action(
                "auth.generate_registration_code", {}
            )
            assert r["status"] == "error"
            assert "admin" in r["error"].lower()
        finally:
            current_conversation.reset(token)
            auth_module.set_auth_manager(None)

    @pytest.mark.asyncio
    async def test_notify_admins_calls_whatsapp_send(
        self, auth_manager: AuthManager
    ):
        from boxbot.communication import auth as auth_module
        from boxbot.communication import whatsapp as wa_module
        from boxbot.tools._sandbox_actions import _handle_auth_action

        boot = await auth_manager.generate_bootstrap_code()
        await auth_manager.register_user("+15550000001", "Admin", boot)

        fake_wa = MagicMock()
        fake_wa.send_text = AsyncMock(return_value=True)
        wa_module.set_whatsapp_client(fake_wa)
        auth_module.set_auth_manager(auth_manager)
        try:
            r = await _handle_auth_action(
                "auth.notify_admins", {"text": "hello admins"}
            )
            assert r["status"] == "ok"
            assert r["delivered"] == 1
            fake_wa.send_text.assert_awaited_once_with(
                "+15550000001", "hello admins"
            )
        finally:
            wa_module.set_whatsapp_client(None)
            auth_module.set_auth_manager(None)


# ---------------------------------------------------------------------------
# notice display registration
# ---------------------------------------------------------------------------


class TestNoticeDisplay:
    def test_notice_in_builtins(self):
        from boxbot.displays.builtins import get_builtin_specs

        names = {s.name for s in get_builtin_specs()}
        assert "notice" in names

    def test_notice_uses_args_substitution(self):
        from boxbot.displays.builtins import get_builtin_specs

        spec = next(s for s in get_builtin_specs() if s.name == "notice")
        # Walk the block tree looking for {args.title} / {args.lines[N]}
        seen = []

        def walk(node):
            if hasattr(node, "content"):
                seen.append(node.content)
            for child in getattr(node, "children", []) or []:
                walk(child)
        walk(spec.root_block)
        assert any("{args.title}" in s for s in seen)
        assert any("{args.lines[0]}" in s for s in seen)
