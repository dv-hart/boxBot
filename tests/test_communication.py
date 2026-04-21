"""Tests for the communication layer — auth, WhatsApp, message routing."""

from __future__ import annotations

import hashlib
import hmac
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from boxbot.communication.auth import AuthManager, RegisterResult, User
from boxbot.communication.router import Channel, MessageRouter
from boxbot.communication.whatsapp import (
    IncomingMessage,
    WhatsAppClient,
    WhatsAppWebhook,
)


# ---------------------------------------------------------------------------
# Auth Manager
# ---------------------------------------------------------------------------


class TestAuthManagerUserCRUD:
    """Test user management — registration, lookup, roles."""

    @pytest.mark.asyncio
    async def test_no_users_initially(self, auth_manager):
        users = await auth_manager.list_users()
        assert len(users) == 0

    @pytest.mark.asyncio
    async def test_bootstrap_code_generation(self, auth_manager):
        code = await auth_manager.generate_bootstrap_code()
        assert isinstance(code, str)
        assert len(code) == 6
        assert code.isdigit()

    @pytest.mark.asyncio
    async def test_register_user_with_bootstrap_code(self, auth_manager):
        code = await auth_manager.generate_bootstrap_code()
        result = await auth_manager.register_user(
            phone="+15551234567",
            name="Jacob",
            code=code,
        )
        assert result.success is True
        assert result.user is not None
        assert result.user.role == "admin"  # bootstrap = admin
        assert result.user.name == "Jacob"

    @pytest.mark.asyncio
    async def test_is_authorized_returns_true_for_registered(self, auth_manager):
        code = await auth_manager.generate_bootstrap_code()
        await auth_manager.register_user("+15551234567", "Jacob", code)
        assert await auth_manager.is_authorized("+15551234567") is True

    @pytest.mark.asyncio
    async def test_is_authorized_returns_false_for_unknown(self, auth_manager):
        assert await auth_manager.is_authorized("+19999999999") is False

    @pytest.mark.asyncio
    async def test_get_user_returns_user_object(self, auth_manager):
        code = await auth_manager.generate_bootstrap_code()
        await auth_manager.register_user("+15551234567", "Jacob", code)
        user = await auth_manager.get_user("+15551234567")
        assert isinstance(user, User)
        assert user.name == "Jacob"
        assert user.phone == "+15551234567"

    @pytest.mark.asyncio
    async def test_get_nonexistent_user_returns_none(self, auth_manager):
        user = await auth_manager.get_user("+10000000000")
        assert user is None

    @pytest.mark.asyncio
    async def test_remove_user(self, auth_manager):
        code = await auth_manager.generate_bootstrap_code()
        await auth_manager.register_user("+15551234567", "Jacob", code)
        removed = await auth_manager.remove_user("+15551234567")
        assert removed is True
        assert await auth_manager.is_authorized("+15551234567") is False

    @pytest.mark.asyncio
    async def test_update_role(self, auth_manager):
        code = await auth_manager.generate_bootstrap_code()
        await auth_manager.register_user("+15551234567", "Jacob", code)
        result = await auth_manager.update_role("+15551234567", "user")
        assert result is True
        user = await auth_manager.get_user("+15551234567")
        assert user.role == "user"

    @pytest.mark.asyncio
    async def test_update_role_invalid_role_returns_false(self, auth_manager):
        code = await auth_manager.generate_bootstrap_code()
        await auth_manager.register_user("+15551234567", "Jacob", code)
        result = await auth_manager.update_role("+15551234567", "superadmin")
        assert result is False


class TestAuthManagerRegistrationCodes:
    """Test registration code lifecycle."""

    @pytest.mark.asyncio
    async def test_code_is_single_use(self, auth_manager):
        code = await auth_manager.generate_bootstrap_code()
        result1 = await auth_manager.register_user("+15551111111", "Alice", code)
        assert result1.success is True
        result2 = await auth_manager.register_user("+15552222222", "Bob", code)
        assert result2.success is False
        assert "Invalid or expired" in result2.error

    @pytest.mark.asyncio
    async def test_duplicate_phone_rejected(self, auth_manager):
        code1 = await auth_manager.generate_bootstrap_code()
        await auth_manager.register_user("+15551234567", "Jacob", code1)
        # Try to register same phone with a new code (make admin first for code gen)
        code2 = await auth_manager.generate_registration_code("+15551234567")
        result = await auth_manager.register_user("+15551234567", "Jacob2", code2)
        assert result.success is False
        assert "already registered" in result.error

    @pytest.mark.asyncio
    async def test_admin_can_generate_registration_code(self, auth_manager):
        # Register admin first
        boot_code = await auth_manager.generate_bootstrap_code()
        await auth_manager.register_user("+15551234567", "Admin", boot_code)
        # Admin generates a code
        code = await auth_manager.generate_registration_code("+15551234567")
        assert isinstance(code, str)
        assert len(code) == 6

    @pytest.mark.asyncio
    async def test_non_admin_cannot_generate_code(self, auth_manager):
        # Register admin, then a regular user
        boot_code = await auth_manager.generate_bootstrap_code()
        await auth_manager.register_user("+15551234567", "Admin", boot_code)
        reg_code = await auth_manager.generate_registration_code("+15551234567")
        await auth_manager.register_user("+15559999999", "Regular", reg_code)
        # Regular user tries to generate
        with pytest.raises(PermissionError, match="Only admins"):
            await auth_manager.generate_registration_code("+15559999999")

    @pytest.mark.asyncio
    async def test_bootstrap_fails_when_admin_exists(self, auth_manager):
        code = await auth_manager.generate_bootstrap_code()
        await auth_manager.register_user("+15551234567", "Admin", code)
        with pytest.raises(RuntimeError, match="admin already exists"):
            await auth_manager.generate_bootstrap_code()

    @pytest.mark.asyncio
    async def test_regular_code_creates_user_role(self, auth_manager):
        # Bootstrap admin
        boot_code = await auth_manager.generate_bootstrap_code()
        await auth_manager.register_user("+15551234567", "Admin", boot_code)
        # Generate regular code
        reg_code = await auth_manager.generate_registration_code("+15551234567")
        result = await auth_manager.register_user("+15559999999", "User", reg_code)
        assert result.success is True
        assert result.user.role == "user"


class TestBruteForceProtection:
    """Test rate limiting and blocking for unknown numbers."""

    @pytest.mark.asyncio
    async def test_not_blocked_initially(self, auth_manager):
        assert await auth_manager.is_blocked("+19998887777") is False

    @pytest.mark.asyncio
    async def test_temp_block_after_max_attempts(self, auth_manager):
        phone = "+19998887777"
        # Auth manager is configured with max_attempts_per_window=3
        for _ in range(3):
            await auth_manager.record_failed_attempt(phone)
        assert await auth_manager.is_blocked(phone) is True

    @pytest.mark.asyncio
    async def test_unblock_clears_block(self, auth_manager):
        phone = "+19998887777"
        for _ in range(3):
            await auth_manager.record_failed_attempt(phone)
        assert await auth_manager.is_blocked(phone) is True
        result = await auth_manager.unblock(phone)
        assert result is True
        assert await auth_manager.is_blocked(phone) is False


# ---------------------------------------------------------------------------
# WhatsApp Webhook
# ---------------------------------------------------------------------------


class TestWhatsAppWebhook:
    """Test webhook verification and message parsing."""

    def test_verify_webhook_success(self):
        webhook = WhatsAppWebhook(verify_token="my-secret")
        result = webhook.verify_webhook("subscribe", "my-secret", "challenge_123")
        assert result == "challenge_123"

    def test_verify_webhook_wrong_token(self):
        webhook = WhatsAppWebhook(verify_token="my-secret")
        result = webhook.verify_webhook("subscribe", "wrong-token", "challenge_123")
        assert result is None

    def test_verify_webhook_wrong_mode(self):
        webhook = WhatsAppWebhook(verify_token="my-secret")
        result = webhook.verify_webhook("unsubscribe", "my-secret", "challenge_123")
        assert result is None

    def test_validate_signature_valid(self):
        secret = "test-app-secret"
        webhook = WhatsAppWebhook(verify_token="token", app_secret=secret)
        body = b"test payload"
        sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        assert webhook.validate_signature(body, f"sha256={sig}") is True

    def test_validate_signature_invalid(self):
        webhook = WhatsAppWebhook(verify_token="token", app_secret="secret")
        assert webhook.validate_signature(b"payload", "sha256=invalid") is False

    def test_validate_signature_no_secret_passes(self):
        webhook = WhatsAppWebhook(verify_token="token")
        assert webhook.validate_signature(b"payload", "sha256=anything") is True

    def test_parse_webhook_text_message(self):
        webhook = WhatsAppWebhook(verify_token="token")
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messaging_product": "whatsapp",
                                "contacts": [
                                    {
                                        "wa_id": "15551234567",
                                        "profile": {"name": "Jacob"},
                                    }
                                ],
                                "messages": [
                                    {
                                        "from": "15551234567",
                                        "type": "text",
                                        "text": {"body": "Hello BB"},
                                        "id": "msg_1",
                                        "timestamp": "1234567890",
                                    }
                                ],
                            }
                        }
                    ]
                }
            ]
        }
        messages = webhook.parse_webhook(payload)
        assert len(messages) == 1
        msg = messages[0]
        assert isinstance(msg, IncomingMessage)
        assert msg.sender_phone == "15551234567"
        assert msg.sender_name == "Jacob"
        assert msg.message_text == "Hello BB"

    def test_parse_webhook_image_message(self):
        webhook = WhatsAppWebhook(verify_token="token")
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messaging_product": "whatsapp",
                                "contacts": [
                                    {
                                        "wa_id": "15551234567",
                                        "profile": {"name": "Alice"},
                                    }
                                ],
                                "messages": [
                                    {
                                        "from": "15551234567",
                                        "type": "image",
                                        "image": {
                                            "id": "media_123",
                                            "caption": "Look at this",
                                        },
                                        "id": "msg_2",
                                        "timestamp": "1234567890",
                                    }
                                ],
                            }
                        }
                    ]
                }
            ]
        }
        messages = webhook.parse_webhook(payload)
        assert len(messages) == 1
        msg = messages[0]
        assert msg.media_type == "image"
        assert msg.media_url == "media_123"
        assert msg.message_text == "Look at this"

    def test_parse_webhook_no_messages(self):
        webhook = WhatsAppWebhook(verify_token="token")
        payload = {"entry": [{"changes": [{"value": {"messaging_product": "whatsapp"}}]}]}
        messages = webhook.parse_webhook(payload)
        assert messages == []

    def test_parse_webhook_empty_payload(self):
        webhook = WhatsAppWebhook(verify_token="token")
        messages = webhook.parse_webhook({})
        assert messages == []


# ---------------------------------------------------------------------------
# Message Router
# ---------------------------------------------------------------------------


class TestMessageRouter:
    """Test message routing through auth to the agent."""

    @pytest.mark.asyncio
    async def test_voice_always_routed(self, auth_manager, event_bus):
        router = MessageRouter(auth=auth_manager)
        result = await router.route_incoming(
            Channel.VOICE, "+15551234567", "Hello"
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_whatsapp_authorized_user_routed(self, auth_manager, event_bus):
        # Register a user
        code = await auth_manager.generate_bootstrap_code()
        await auth_manager.register_user("+15551234567", "Jacob", code)

        received = []

        async def on_msg(event):
            received.append(event)

        from boxbot.core.events import WhatsAppMessage
        event_bus.subscribe(WhatsAppMessage, on_msg)

        router = MessageRouter(auth=auth_manager)
        result = await router.route_incoming(
            Channel.WHATSAPP, "+15551234567", "Hello BB"
        )
        assert result is True
        assert len(received) == 1
        assert received[0].sender_name == "Jacob"

    @pytest.mark.asyncio
    async def test_whatsapp_unknown_number_silently_dropped(
        self, auth_manager, event_bus
    ):
        received = []

        async def on_msg(event):
            received.append(event)

        from boxbot.core.events import WhatsAppMessage
        event_bus.subscribe(WhatsAppMessage, on_msg)

        router = MessageRouter(auth=auth_manager)
        result = await router.route_incoming(
            Channel.WHATSAPP, "+19999999999", "Hello"
        )
        assert result is False
        assert len(received) == 0  # no event emitted

    @pytest.mark.asyncio
    async def test_whatsapp_valid_code_registers_user(
        self, auth_manager, event_bus
    ):
        code = await auth_manager.generate_bootstrap_code()

        received = []

        async def on_msg(event):
            received.append(event)

        from boxbot.core.events import WhatsAppMessage
        event_bus.subscribe(WhatsAppMessage, on_msg)

        router = MessageRouter(auth=auth_manager)
        result = await router.route_incoming(
            Channel.WHATSAPP,
            "+15559999999",
            code,
            sender_name="NewUser",
        )
        assert result is True
        assert await auth_manager.is_authorized("+15559999999") is True
        # Should have emitted a registration event
        assert len(received) == 1
        assert "[REGISTRATION]" in received[0].text

    @pytest.mark.asyncio
    async def test_resolve_recipient_case_insensitive(self, auth_manager):
        code = await auth_manager.generate_bootstrap_code()
        await auth_manager.register_user("+15551234567", "Jacob", code)

        router = MessageRouter(auth=auth_manager)
        phone = await router.resolve_recipient("jacob")
        assert phone == "+15551234567"

    @pytest.mark.asyncio
    async def test_resolve_unknown_recipient_returns_none(self, auth_manager):
        router = MessageRouter(auth=auth_manager)
        phone = await router.resolve_recipient("Nobody")
        assert phone is None

    @pytest.mark.asyncio
    async def test_route_outgoing_no_whatsapp_client(self, auth_manager):
        router = MessageRouter(auth=auth_manager, whatsapp=None)
        result = await router.route_outgoing("Jacob", "Hello")
        assert result is False

    @pytest.mark.asyncio
    async def test_blocked_number_is_dropped(self, auth_manager, event_bus):
        phone = "+19998887777"
        # Trigger a temp block
        for _ in range(3):
            await auth_manager.record_failed_attempt(phone)

        router = MessageRouter(auth=auth_manager)
        result = await router.route_incoming(
            Channel.WHATSAPP, phone, "Let me in"
        )
        assert result is False


# ---------------------------------------------------------------------------
# Auth Manager — Cleanup
# ---------------------------------------------------------------------------


class TestAuthManagerCleanup:
    """Test database maintenance / cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_purges_expired_codes(self, auth_manager):
        """Expired unused codes older than 24h are removed."""
        # Create a code that's already expired (expiry in the past)
        import aiosqlite
        from datetime import datetime, timedelta

        old_time = (datetime.now() - timedelta(hours=25)).isoformat()
        old_expiry = (datetime.now() - timedelta(hours=24, minutes=50)).isoformat()

        async with aiosqlite.connect(auth_manager._db_path) as db:
            await db.execute(
                "INSERT INTO registration_codes "
                "(code, created_by, created_at, expires_at) "
                "VALUES (?, ?, ?, ?)",
                ("999999", "bootstrap", old_time, old_expiry),
            )
            await db.commit()

        result = await auth_manager.cleanup()
        assert result["expired_codes"] >= 1

        # Verify the code was actually removed
        assert await auth_manager.validate_code("999999") is False

    @pytest.mark.asyncio
    async def test_cleanup_purges_old_failed_attempts(self, auth_manager):
        """Failed attempts older than 24h are removed."""
        import aiosqlite
        from datetime import datetime, timedelta

        old_time = (datetime.now() - timedelta(hours=25)).isoformat()

        async with aiosqlite.connect(auth_manager._db_path) as db:
            for _ in range(5):
                await db.execute(
                    "INSERT INTO failed_attempts (phone, attempted_at) "
                    "VALUES (?, ?)",
                    ("+19998887777", old_time),
                )
            await db.commit()

        result = await auth_manager.cleanup()
        assert result["failed_attempts"] >= 5

    @pytest.mark.asyncio
    async def test_cleanup_preserves_valid_codes(self, auth_manager):
        """Active, unexpired codes are not removed."""
        code = await auth_manager.generate_bootstrap_code()
        result = await auth_manager.cleanup()
        # The fresh code should still be valid
        assert await auth_manager.validate_code(code) is True
