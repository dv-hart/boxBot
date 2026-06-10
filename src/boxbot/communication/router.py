"""Message routing between channels, auth, and the agent.

Routes incoming messages through authentication and delivers them to
the agent via the event bus. Routes outgoing messages from the agent
to the correct channel based on user name resolution.

Security:
- Incoming messages from unknown numbers are silently dropped
- Only the registration code exception allows unknown number processing
- Package-approval replies (``approve pkg <id>`` / ``deny pkg <id>``)
  from registered ADMINS are intercepted here (mirroring the
  registration-code pattern) and handled by the package service instead
  of reaching the agent; the same text from a non-admin routes as a
  normal message and resolves nothing
- All routing decisions are logged without message content

Usage:
    from boxbot.communication.router import MessageRouter, Channel

    router = MessageRouter(auth=auth_manager, whatsapp=whatsapp_client)
    await router.route_incoming(
        Channel.WHATSAPP, "+15551234567", "Hello BB", media=None
    )
"""

from __future__ import annotations

import logging

from boxbot.communication.auth import AuthManager, User
from boxbot.communication.channels import Channel
from boxbot.communication.whatsapp import WhatsAppClient
from boxbot.core.events import (
    SignalMessage,
    UserRegistered,
    WhatsAppMessage,
    get_event_bus,
)

logger = logging.getLogger(__name__)

__all__ = ["Channel", "MessageRouter"]


class MessageRouter:
    """Routes messages between external channels and the agent.

    Handles the full message lifecycle:
    - Incoming: auth check → silent drop or event emission
    - Outgoing: name resolution → channel dispatch

    Args:
        auth: The AuthManager for user verification.
        whatsapp: The WhatsAppClient for sending outbound messages.
    """

    def __init__(
        self,
        auth: AuthManager,
        whatsapp: WhatsAppClient | None = None,
    ) -> None:
        self._auth = auth
        self._whatsapp = whatsapp
        self._event_bus = get_event_bus()

    async def route_incoming(
        self,
        channel: Channel,
        sender_phone: str,
        message: str | None,
        *,
        media_url: str | None = None,
        media_type: str | None = None,
        sender_name: str | None = None,
        message_id: str = "",
    ) -> bool:
        """Route an incoming message through auth and to the agent.

        For WhatsApp: checks if sender is authorized. If not, checks if
        the message contains a valid registration code. Otherwise, silent drop.

        For voice: messages are always routed (physical presence = auth).

        Args:
            channel: The channel the message arrived on.
            sender_phone: Sender's phone number.
            message: The message text.
            media_url: Optional media URL/ID.
            media_type: Optional media type.
            sender_name: Display name from the channel (e.g., WhatsApp profile name).
            message_id: Channel-issued message id (e.g. ``wamid.*`` for
                WhatsApp). Propagated onto the event so the agent's
                inbound staging can name files deterministically.

        Returns:
            True if the message was routed to the agent, False if dropped.
        """
        if channel == Channel.VOICE:
            # Voice = physical presence, always routed
            logger.debug("Voice message from %s routed to agent", sender_phone)
            return True

        if channel == Channel.WHATSAPP:
            return await self._route_whatsapp(
                sender_phone,
                message,
                media_url=media_url,
                media_type=media_type,
                sender_name=sender_name,
                message_id=message_id,
            )

        if channel == Channel.SIGNAL:
            return await self._route_signal(
                sender_phone,
                message,
                media_url=media_url,
                media_type=media_type,
                sender_name=sender_name,
                message_id=message_id,
            )

        logger.warning("Unknown channel: %s", channel)
        return False

    async def _route_whatsapp(
        self,
        sender_phone: str,
        message: str | None,
        *,
        media_url: str | None = None,
        media_type: str | None = None,
        sender_name: str | None = None,
        message_id: str = "",
    ) -> bool:
        """Route a WhatsApp message through auth to the agent.

        Returns True if routed, False if silently dropped.
        """
        # Check if sender is blocked (brute-force protection)
        if await self._auth.is_blocked(sender_phone):
            logger.debug("Blocked number, silent drop: %s", sender_phone)
            return False

        # Check if sender is a registered user
        user = await self._auth.get_user(sender_phone)
        if user is not None:
            # Authorized user — route to agent
            await self._auth.update_last_seen(sender_phone)

            if await self._maybe_handle_package_reply(
                user, message, Channel.WHATSAPP
            ):
                return True

            event = WhatsAppMessage(
                sender_name=user.name,
                sender_phone=sender_phone,
                text=message or "",
                media_url=media_url,
                media_type=media_type,
                message_id=message_id,
            )
            await self._event_bus.publish(event)
            logger.info(
                "WhatsApp message from %s (%s) routed to agent",
                user.name,
                sender_phone,
            )
            return True

        # Unknown number — check if message is a valid registration code
        if message and message.strip():
            code = message.strip()
            if await self._auth.validate_code(code):
                # Capture inviter before consuming the code (register_user
                # marks it used). "bootstrap" → first admin; otherwise
                # the inviting admin's phone for the UserRegistered event.
                inviter = await self._auth.get_code_creator(code)
                # Valid code — register with channel name or placeholder
                name = sender_name or "New User"
                result = await self._auth.register_user(
                    phone=sender_phone,
                    name=name,
                    code=code,
                )
                if result.success and result.user:
                    # Two events fire on a successful registration:
                    # 1) UserRegistered — the structured fact for any
                    #    consumer (agent welcome flow, audit, etc.).
                    # 2) A WhatsAppMessage tagged "[REGISTRATION] …" so
                    #    the agent's existing inbound-message path
                    #    surfaces this as a user-driven turn it can
                    #    react to. Both intentionally — older code only
                    #    reads the WhatsAppMessage, newer flows can
                    #    subscribe to UserRegistered for cleaner
                    #    semantics.
                    invited_by = (
                        "" if not inviter or inviter == "bootstrap" else inviter
                    )
                    await self._event_bus.publish(
                        UserRegistered(
                            phone=sender_phone,
                            name=result.user.name,
                            role=result.user.role,
                            invited_by_phone=invited_by,
                        )
                    )
                    event = WhatsAppMessage(
                        sender_name=result.user.name,
                        sender_phone=sender_phone,
                        text=f"[REGISTRATION] {code}",
                        media_url=None,
                        media_type=None,
                    )
                    await self._event_bus.publish(event)
                    logger.info(
                        "New user registered via code: %s (%s) as %s",
                        result.user.name,
                        sender_phone,
                        result.user.role,
                    )
                    return True

            # Invalid code or not a code — record failed attempt, silent drop
            await self._auth.record_failed_attempt(sender_phone)

        # SILENT DROP — no response, no error, no information leakage
        logger.debug("Unknown number, silent drop: %s", sender_phone)
        return False

    async def _route_signal(
        self,
        sender_phone: str,
        message: str | None,
        *,
        media_url: str | None = None,
        media_type: str | None = None,
        sender_name: str | None = None,
        message_id: str = "",
    ) -> bool:
        """Route a Signal message through auth to the agent.

        Mirrors ``_route_whatsapp``: silent drop for unknown senders,
        registration-code exception, ``UserRegistered`` event on success.
        Auth lookup ignores channel — a whatsapp-registered user texting
        from Signal during the migration is still authorised.

        Returns True if routed, False if silently dropped.
        """
        if await self._auth.is_blocked(sender_phone):
            logger.debug("Blocked number, silent drop: %s", sender_phone)
            return False

        user = await self._auth.get_user(sender_phone)
        if user is not None:
            await self._auth.update_last_seen(sender_phone)

            if await self._maybe_handle_package_reply(
                user, message, Channel.SIGNAL
            ):
                return True

            event = SignalMessage(
                sender_name=user.name,
                sender_phone=sender_phone,
                text=message or "",
                media_url=media_url,
                media_type=media_type,
                message_id=message_id,
            )
            await self._event_bus.publish(event)
            logger.info(
                "Signal message from %s (%s) routed to agent",
                user.name,
                sender_phone,
            )
            return True

        # Unknown number — check if message is a valid registration code.
        if message and message.strip():
            code = message.strip()
            if await self._auth.validate_code(code):
                inviter = await self._auth.get_code_creator(code)
                name = sender_name or "New User"
                result = await self._auth.register_user(
                    phone=sender_phone,
                    name=name,
                    code=code,
                    channel="signal",
                )
                if result.success and result.user:
                    invited_by = (
                        "" if not inviter or inviter == "bootstrap" else inviter
                    )
                    await self._event_bus.publish(
                        UserRegistered(
                            phone=sender_phone,
                            name=result.user.name,
                            role=result.user.role,
                            invited_by_phone=invited_by,
                        )
                    )
                    event = SignalMessage(
                        sender_name=result.user.name,
                        sender_phone=sender_phone,
                        text=f"[REGISTRATION] {code}",
                        media_url=None,
                        media_type=None,
                    )
                    await self._event_bus.publish(event)
                    logger.info(
                        "New Signal user registered: %s (%s) as %s",
                        result.user.name,
                        sender_phone,
                        result.user.role,
                    )
                    return True

            await self._auth.record_failed_attempt(sender_phone)

        # SILENT DROP — no response, no error.
        logger.debug("Unknown number on Signal, silent drop: %s", sender_phone)
        return False

    async def _maybe_handle_package_reply(
        self,
        user: "User",
        message: str | None,
        inbound_channel: Channel,
    ) -> bool:
        """Intercept an admin's package approve/deny reply.

        Returns True iff the message was a package-approval command from
        a registered **admin** and was fully handled here (the agent
        never sees it). The confirmation is sent back on the channel the
        reply arrived on — the admin is demonstrably reachable there.

        A matching command from a non-admin returns False: the message
        routes to the agent as normal text and resolves nothing. No
        privilege, no information leak — they're registered users anyway.
        """
        if not message:
            return False

        from boxbot.packages.service import handle_admin_reply, parse_admin_reply

        parsed = parse_admin_reply(message)
        if parsed is None:
            return False
        if user.role != "admin":
            logger.warning(
                "Non-admin %s sent a package-approval command; "
                "routing as a normal message",
                user.phone,
            )
            return False

        verb, request_id, note = parsed
        reply = await handle_admin_reply(
            verb, request_id, note, admin=user
        )
        logger.info(
            "Package %s for request %s handled from admin %s",
            verb, request_id, user.phone,
        )

        from boxbot.communication.channels import get_outbound_channel

        out = get_outbound_channel(inbound_channel)
        if out is None:
            # Fall back to the admin's registered outbound channel.
            try:
                out = get_outbound_channel(Channel(user.channel))
            except ValueError:
                out = None
        if out is not None:
            try:
                await out.send_text(user.phone, reply)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Could not confirm package %s to admin %s",
                    verb, user.phone,
                )
        else:
            logger.warning(
                "No outbound client to confirm package %s to admin %s",
                verb, user.phone,
            )
        return True

    async def route_outgoing(
        self,
        recipient_name: str,
        message: str,
        *,
        media_path: str | None = None,
        channel: Channel = Channel.WHATSAPP,
    ) -> bool:
        """Route an outgoing message from the agent to a user.

        Resolves the recipient name to a phone number and sends via
        the appropriate channel.

        Args:
            recipient_name: Display name of the recipient.
            message: The message text to send.
            media_path: Optional local file path for an image to send.
            channel: The channel to send on.

        Returns:
            True if the message was sent successfully, False otherwise.
        """
        phone = await self.resolve_recipient(recipient_name)
        if phone is None:
            logger.warning("Could not resolve recipient: %s", recipient_name)
            return False

        if channel == Channel.WHATSAPP:
            return await self._send_whatsapp(phone, message, media_path=media_path)

        logger.warning("Outgoing not supported on channel: %s", channel)
        return False

    async def _send_whatsapp(
        self,
        phone: str,
        message: str,
        *,
        media_path: str | None = None,
    ) -> bool:
        """Send a message via WhatsApp."""
        if self._whatsapp is None:
            logger.error("WhatsApp client not configured")
            return False

        if media_path:
            return await self._whatsapp.upload_and_send_image(
                phone, media_path, caption=message
            )
        return await self._whatsapp.send_text(phone, message)

    async def resolve_recipient(self, name: str) -> str | None:
        """Resolve a display name to a phone number.

        Performs case-insensitive matching against registered user names.

        Args:
            name: Display name to look up.

        Returns:
            Phone number in E.164 format, or None if not found.
        """
        users = await self._auth.list_users()
        name_lower = name.lower()
        for user in users:
            if user.name.lower() == name_lower:
                return user.phone
        return None

    async def send_to_admins(self, message: str) -> None:
        """Send a message to all admin users. Used for security notifications.

        Args:
            message: The message text.
        """
        if self._whatsapp is None:
            logger.warning("WhatsApp client not configured, cannot notify admins")
            return

        users = await self._auth.list_users()
        for user in users:
            if user.role == "admin":
                await self._whatsapp.send_text(user.phone, message)
