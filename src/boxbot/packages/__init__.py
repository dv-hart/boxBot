"""Package-install request flow — sandbox request, admin approval, install.

The sandbox cannot install packages (seccomp, owner-only pip, read-only
site-packages). The only path is:

1. ``bb.packages.request(...)`` emits a ``packages.request`` SDK action.
2. The action handler validates the package spec, durably queues a
   pending request (:mod:`boxbot.packages.store`), and notifies every
   registered admin on their registered channel
   (:mod:`boxbot.packages.service`).
3. An admin replies ``approve pkg <id>`` / ``deny pkg <id>`` on Signal
   or WhatsApp; the inbound :class:`~boxbot.communication.router.
   MessageRouter` intercepts the reply (same pattern as registration
   codes) and hands it to :func:`service.handle_admin_reply`.
4. On approval the **main process** runs the sandbox venv's pip
   (:mod:`boxbot.packages.installer`) and records the outcome on the
   request row. The agent observes the lifecycle via
   ``bb.packages.status(id)`` / ``bb.packages.list()``.

Request lifecycle: ``pending → approved → installed | failed`` or
``pending → denied``.
"""

from boxbot.packages.store import (
    PackageStore,
    get_package_store,
    set_package_store,
)

__all__ = [
    "PackageStore",
    "get_package_store",
    "set_package_store",
]
