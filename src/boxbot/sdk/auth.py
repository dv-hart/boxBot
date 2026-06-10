"""Auth — read user/admin state, mint registration codes, notify admins.

Two distinct flows the agent owns end-to-end:

1. First-admin bootstrap. ``generate_bootstrap_code()`` works only when no
   admin exists yet (the auth layer enforces this). The agent surfaces
   the returned code on the screen via ``switch_display`` so a human
   physically present at the box can text it from their phone.

2. Admin-initiated user registration. ``generate_registration_code()``
   uses the *current conversation's* sender as the inviting admin —
   the sandbox cannot pass an arbitrary ``created_by``. The agent
   should send the returned code back to the admin (as a messaging
   reply) for them to share with the new user out-of-band.

``notify_admins(text)`` reaches every registered admin on their
registered channel (Signal or WhatsApp — each user row carries a
``channel`` field). Use sparingly — security notifications and
"new user joined" pings only.

The agent never sees raw secrets via this module. Channel credentials
stay in the main process; this SDK is just an RPC façade onto
``AuthManager``.

Error semantics: the writes (code minting, ``notify_admins``) raise
``bb.ActionError`` on rejection; ``list_users`` returns its documented
list shape.
"""

from __future__ import annotations

from typing import Any

from . import _transport


def list_users() -> list[dict[str, Any]]:
    """Return all registered users.

    Each entry has: ``id``, ``name``, ``phone``, ``role`` ("admin" or
    "user"), ``created_at``, ``last_seen``. Empty list if no admin has
    been bootstrapped yet — the canonical "is BB set up?" signal.

    Raises:
        ActionError: if the auth manager is unavailable.
    """
    resp = _transport.dispatch_or_raise("auth.list_users", {})
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
    resp = _transport.dispatch_or_raise("auth.generate_bootstrap_code", {})
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
    resp = _transport.dispatch_or_raise("auth.generate_registration_code", {})
    return str(resp["code"])


def notify_admins(text: str) -> None:
    """Send a message to every registered admin on their registered channel.

    Each admin's user row carries a ``channel`` field ("signal" or
    "whatsapp"); the main process resolves the right outbound client
    per admin. Used for security alerts and "new user joined"
    notifications. Admins whose channel has no client registered are
    skipped (logged main-side). Does not return a per-recipient
    delivery report.

    Raises:
        ActionError: if the text is empty or the auth manager is
            unavailable.
    """
    _transport.dispatch_or_raise("auth.notify_admins", {"text": str(text)})
