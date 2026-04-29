"""Auth — read user/admin state, mint registration codes, notify admins.

Two distinct flows the agent owns end-to-end:

1. First-admin bootstrap. ``generate_bootstrap_code()`` works only when no
   admin exists yet (the auth layer enforces this). The agent surfaces
   the returned code on the screen via ``switch_display`` so a human
   physically present at the box can text it from their phone.

2. Admin-initiated user registration. ``generate_registration_code()``
   uses the *current conversation's* sender as the inviting admin —
   the sandbox cannot pass an arbitrary ``created_by``. The agent
   should send the returned code back to the admin (via WhatsApp
   reply) for them to share with the new user out-of-band.

``notify_admins(text)`` reaches every registered admin via WhatsApp.
Use sparingly — security notifications and "new user joined" pings only.

The agent never sees raw secrets via this module. The four
``WHATSAPP_*`` env vars stay in the main process; this SDK is just an
RPC façade onto ``AuthManager``.
"""

from __future__ import annotations

from typing import Any

from . import _transport


def list_users() -> list[dict[str, Any]]:
    """Return all registered users.

    Each entry has: ``id``, ``name``, ``phone``, ``role`` ("admin" or
    "user"), ``created_at``, ``last_seen``. Empty list if no admin has
    been bootstrapped yet — the canonical "is BB set up?" signal.
    """
    resp = _transport.request("auth.list_users", {})
    if resp.get("status") != "ok":
        raise RuntimeError(resp.get("error") or "auth.list_users failed")
    return list(resp.get("users") or [])


def generate_bootstrap_code() -> str:
    """Mint the one-time first-admin code. Raises if an admin exists.

    Returns:
        6-digit numeric code. Expires in 10 minutes. Single-use.
        Show this on the box's HDMI screen (e.g. via
        ``switch_display("notice", args={...})``) — the security
        property is *physical presence*, so don't relay it over voice
        or any messaging channel.
    """
    resp = _transport.request("auth.generate_bootstrap_code", {})
    if resp.get("status") != "ok":
        raise RuntimeError(resp.get("error") or "auth.generate_bootstrap_code failed")
    return str(resp["code"])


def generate_registration_code() -> str:
    """Mint a code that registers the bearer as a standard user.

    Caller must be in a WhatsApp conversation with a registered admin
    as the sender; the main process resolves who you are from the
    conversation context. The sandbox cannot spoof this.

    Returns:
        6-digit numeric code. Expires in 10 minutes. Single-use.
        The agent should reply to the inviting admin with this code so
        they can forward it to the new user out-of-band.
    """
    resp = _transport.request("auth.generate_registration_code", {})
    if resp.get("status") != "ok":
        raise RuntimeError(
            resp.get("error") or "auth.generate_registration_code failed"
        )
    return str(resp["code"])


def notify_admins(text: str) -> None:
    """Send a WhatsApp message to every registered admin.

    Used for security alerts and "new user joined" notifications.
    No-op (with a warning logged on the main side) if WhatsApp isn't
    configured. Does not return a per-recipient delivery report.
    """
    resp = _transport.request("auth.notify_admins", {"text": str(text)})
    if resp.get("status") != "ok":
        raise RuntimeError(resp.get("error") or "auth.notify_admins failed")
