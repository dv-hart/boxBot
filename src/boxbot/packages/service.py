"""Package-request orchestration: validation, admin notify, approval.

Three responsibilities, all channel-agnostic:

- :func:`validate_package_spec` — strict PyPI-name (or ``name==version``)
  validation. URLs, local paths, extras, and every other specifier shape
  are rejected before anything touches the queue.
- :func:`submit_request` — durably queue a pending request and notify
  every registered admin on their **registered channel** (Signal or
  WhatsApp — resolved per-admin via the outbound-channel registry, the
  same pattern as the ``auth.notify_admins`` action handler).
- :func:`parse_admin_reply` / :func:`handle_admin_reply` — interpret an
  admin's ``approve pkg <id>`` / ``deny pkg <id>`` reply (the router
  intercepts these inbound, mirroring registration-code handling) and
  drive the install / denial. The install runs as a background task in
  the main process; its outcome lands on the request row and is
  messaged back to the approving admin.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any

from boxbot.packages.store import get_package_store

if TYPE_CHECKING:
    from boxbot.communication.auth import User

logger = logging.getLogger(__name__)

# PEP 503/508 project name: alnum, with dots/hyphens/underscores inside.
_NAME_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9._-]{0,78}[A-Za-z0-9])?$")
# PEP 440-ish version: starts with a digit (epoch or release), then
# alnum/dot/bang/plus. No spaces, slashes, or leading dashes — nothing
# pip could mistake for an option or a URL.
_VERSION_RE = re.compile(r"^[0-9][A-Za-z0-9.!+]{0,63}$")

_REPLY_RE = re.compile(
    r"^\s*(approve|deny)\s+pkg\s+([0-9a-fA-F]{6,12})\b[\s:,-]*(.*?)\s*$",
    re.IGNORECASE | re.DOTALL,
)

_MAX_REASON_LEN = 500


def validate_package_spec(spec: str) -> str:
    """Validate and normalise a requested package spec.

    Accepts a bare PyPI project name (``requests``) or an exact pin
    (``requests==2.32.3``). Everything else — URLs, local paths, VCS
    refs, extras (``pkg[extra]``), range specifiers (``>=``, ``~=``),
    whitespace, option-looking strings — raises ``ValueError``.

    Returns the stripped spec on success.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("package name is required")
    s = spec.strip()
    if "==" in s:
        name, _, version = s.partition("==")
        if not _VERSION_RE.match(version):
            raise ValueError(
                f"invalid version {version!r} — use an exact PEP 440 "
                "version, e.g. 'requests==2.32.3'"
            )
    else:
        name, version = s, None
        for marker in ("<", ">", "~", "=", "!", "@", ";", "["):
            if marker in name:
                raise ValueError(
                    f"unsupported specifier in {spec!r} — request a bare "
                    "name or an exact 'name==version' pin"
                )
    if not _NAME_RE.match(name):
        raise ValueError(
            f"invalid package name {name!r} — PyPI names are "
            "letters/digits with internal dots, hyphens, underscores"
        )
    return s


def parse_admin_reply(text: str) -> tuple[str, str, str] | None:
    """Parse an ``approve pkg <id>`` / ``deny pkg <id> [note]`` reply.

    Returns ``(verb, request_id, note)`` with the verb lowercased and
    the id lowercased, or None if the text isn't an approval command.
    """
    if not text:
        return None
    m = _REPLY_RE.match(text)
    if m is None:
        return None
    return m.group(1).lower(), m.group(2).lower(), m.group(3).strip()


async def submit_request(
    package: str,
    reason: str,
    *,
    requested_by: str | None = None,
) -> dict[str, Any]:
    """Validate, queue, and broadcast a new package request.

    Returns ``{"request": <row>, "duplicate": bool, "admins_notified":
    int, "admins": int}``. If a pending request for the same spec
    already exists it is returned as-is (``duplicate: True``) and no
    new notification goes out — re-asking must not spam the admins.

    Raises:
        ValueError: on an invalid package spec or empty reason.
    """
    spec = validate_package_spec(package)
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("a reason for the install is required")
    reason = reason.strip()[:_MAX_REASON_LEN]

    store = get_package_store()
    existing = await store.find_pending(spec)
    if existing is not None:
        return {
            "request": existing,
            "duplicate": True,
            "admins_notified": 0,
            "admins": 0,
        }

    request = await store.create_request(
        spec, reason, requested_by=requested_by
    )
    delivered, admins = await _notify_admins_of_request(request)
    logger.info(
        "Package request %s queued: %s (notified %d/%d admins)",
        request["id"], spec, delivered, admins,
    )
    return {
        "request": request,
        "duplicate": False,
        "admins_notified": delivered,
        "admins": admins,
    }


async def _notify_admins_of_request(request: dict[str, Any]) -> tuple[int, int]:
    """Message every registered admin on their registered channel.

    Returns ``(delivered, admin_count)``. Zero admins (pre-bootstrap)
    or zero registered outbound clients leaves the request pending —
    an admin can still resolve it later.
    """
    context = request.get("requested_by") or "agent (no conversation context)"
    text = (
        f"📦 boxBot package request {request['id']}\n"
        f"Package: {request['package']}\n"
        f"Reason: {request['reason']}\n"
        f"From: {context}\n"
        f'Reply "approve pkg {request["id"]}" or '
        f'"deny pkg {request["id"]}".'
    )
    return await _broadcast_to_admins(text)


async def _broadcast_to_admins(text: str) -> tuple[int, int]:
    from boxbot.communication.auth import get_auth_manager
    from boxbot.communication.channels import Channel, get_outbound_channel

    auth = get_auth_manager()
    if auth is None:
        logger.warning("Auth manager not initialised — cannot notify admins")
        return 0, 0

    admins = [u for u in await auth.list_users() if u.role == "admin"]
    sent = 0
    for admin in admins:
        try:
            channel = Channel(admin.channel)
        except ValueError:
            logger.warning(
                "Admin %s has unknown channel %r; skipping notify",
                admin.phone, admin.channel,
            )
            continue
        out = get_outbound_channel(channel)
        if out is None:
            continue
        try:
            if await out.send_text(admin.phone, text):
                sent += 1
        except Exception:  # noqa: BLE001
            logger.exception("notify admin %s failed", admin.phone)
    return sent, len(admins)


# ---------------------------------------------------------------------------
# Approval handling (called by MessageRouter for admin replies)
# ---------------------------------------------------------------------------

# Strong refs to in-flight install tasks so they aren't GC'd mid-install.
_install_tasks: set[asyncio.Task] = set()


async def handle_admin_reply(
    verb: str,
    request_id: str,
    note: str,
    *,
    admin: "User",
) -> str:
    """Apply an admin's approve/deny to a request; return the reply text.

    The caller (the router) has already verified ``admin.role ==
    "admin"`` — this function trusts that and records ``admin.phone``
    as the resolver. The returned string is sent back to the admin on
    the channel the reply arrived on.
    """
    store = get_package_store()
    request = await store.get_request(request_id)
    if request is None:
        return f"No package request '{request_id}' found."
    if request["status"] != "pending":
        return (
            f"Request {request_id} ({request['package']}) was already "
            f"{request['status']}."
        )

    if verb == "deny":
        updated = await store.set_status(
            request_id,
            "denied",
            resolved_by=admin.phone,
            note=note or None,
            expect="pending",
        )
        if updated is None:
            return f"Request {request_id} was already resolved."
        logger.info(
            "Package request %s (%s) denied by %s",
            request_id, request["package"], admin.phone,
        )
        return f"Denied package request {request_id} ({request['package']})."

    # approve
    updated = await store.set_status(
        request_id,
        "approved",
        resolved_by=admin.phone,
        expect="pending",
    )
    if updated is None:
        return f"Request {request_id} was already resolved."
    logger.info(
        "Package request %s (%s) approved by %s — installing",
        request_id, request["package"], admin.phone,
    )
    _spawn_install(updated, admin)
    return (
        f"Approved {request['package']} — installing into the sandbox "
        "now. You'll get a confirmation when it finishes."
    )


def _spawn_install(request: dict[str, Any], admin: "User") -> None:
    task = asyncio.get_running_loop().create_task(
        _install_and_record(request, admin),
        name=f"pkg-install-{request['id']}",
    )
    _install_tasks.add(task)
    task.add_done_callback(_install_tasks.discard)


async def _install_and_record(request: dict[str, Any], admin: "User") -> None:
    """Run pip, record the outcome, and message the approving admin."""
    from boxbot.packages.installer import install_package

    request_id = request["id"]
    package = request["package"]
    try:
        ok, output = await install_package(package)
    except Exception as e:  # noqa: BLE001 — never lose the outcome
        logger.exception("install task for %s crashed", package)
        ok, output = False, f"installer crashed: {e}"

    store = get_package_store()
    status = "installed" if ok else "failed"
    note = None if ok else output.strip()[-500:]
    await store.set_status(
        request_id, status, note=note, expect="approved"
    )

    if ok:
        text = (
            f"✅ Installed {package} into the sandbox "
            f"(request {request_id})."
        )
    else:
        text = (
            f"❌ Install of {package} failed (request {request_id}):\n"
            f"{note}"
        )
    await _send_to_admin(admin, text)


async def _send_to_admin(admin: "User", text: str) -> None:
    from boxbot.communication.channels import Channel, get_outbound_channel

    try:
        channel = Channel(admin.channel)
    except ValueError:
        logger.warning(
            "Cannot notify admin %s: unknown channel %r",
            admin.phone, admin.channel,
        )
        return
    out = get_outbound_channel(channel)
    if out is None:
        logger.warning(
            "Cannot notify admin %s: no outbound client for %s",
            admin.phone, channel.value,
        )
        return
    try:
        await out.send_text(admin.phone, text)
    except Exception:  # noqa: BLE001
        logger.exception("install-outcome notify to %s failed", admin.phone)
