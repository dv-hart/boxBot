"""Package installation — request packages with user approval.

Triggers an out-of-band approval flow (screen tap or admin WhatsApp).
Blocks until the user approves or denies.

Usage:
    from boxbot_sdk import packages

    result = packages.request("google-api-python-client",
                              reason="Needed for Gmail integration")
    if result.approved:
        import googleapiclient
    else:
        print(f"Denied: {result.reason}")
"""

from __future__ import annotations

from typing import Any

from . import _transport, _validators as v


class PackageResult:
    """Result of a package installation request."""

    def __init__(self, approved: bool, reason: str | None = None) -> None:
        self.approved = approved
        self.reason = reason

    def __repr__(self) -> str:
        status = "approved" if self.approved else "denied"
        if self.reason:
            return f"PackageResult({status}, reason={self.reason!r})"
        return f"PackageResult({status})"


def request(package: str, *, reason: str, timeout: int = 300) -> PackageResult:
    """Request installation of a Python package.

    Emits an SDK action and blocks until the user approves or denies
    via the out-of-band approval flow (screen tap or admin WhatsApp).

    Args:
        package: Package name (pip-installable).
        reason: Why this package is needed (shown to user during approval).
        timeout: Maximum seconds to wait for user response.

    Returns:
        PackageResult with approved status and optional reason.
    """
    v.require_str(package, "package")
    v.require_str(reason, "reason")
    v.require_int(timeout, "timeout", min_val=1)

    _transport.emit_action("packages.request", {
        "package": package,
        "reason": reason,
    })

    response = _transport.collect_response(timeout=timeout)
    return PackageResult(
        approved=response.get("approved", False),
        reason=response.get("reason"),
    )
