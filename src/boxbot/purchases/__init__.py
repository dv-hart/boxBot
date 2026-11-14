"""Purchasing on the household's behalf through the Shop agent API.

The sandbox can only *propose* a purchase. The main process builds the
checkout, decides, and either texts the admins a review link or — when
automatic purchasing is on and every check passes — pays from the Shop
spending budget after a veto window.

| Module | Role |
|---|---|
| :mod:`shop_auth` | device-flow sign-in, token files, refresh |
| :mod:`shop_client` | Shop API calls; sanitised views of results |
| :mod:`rules` | deterministic limits (caps, store allow-list, requester) |
| :mod:`checker` | isolated policy-checker model call, fail closed |
| :mod:`store` | purchases + full decision log (``data/purchases/``) |
| :mod:`service` | orchestration, notifications, admin commands |

Trust boundaries:

- Shop tokens, checkout review links, order links, addresses, contact
  details, and payment credentials never reach the sandbox or any
  model context.
- The agent cannot set an address, a payment method, or a price; it
  names a store, variant ids, and quantities.
- Review links and veto notices are composed and sent by the service,
  to registered admins only. Every string in them that the service did
  not establish itself (item titles, the agent's reason, user display
  names) passes through ``shop_client.plain_text``; the ``message``
  tool refuses agent text that imitates a notice or carries a Shop
  sign-in/checkout link.
- A store outside ``purchases.allowed_shops`` gets a checkout — and
  with it the household's saved contact and address — only after an
  admin's own ``allow shop <domain>`` text (24 hours, review only).
- A veto window interrupted by a restart is never resumed: the
  purchase goes to review.
- Admin commands (``cancel buy <id>``, ``shop connect``,
  ``allow shop <domain>``) are honoured
  only from an admin's registered number, intercepted by the router
  before the agent sees them.

- What a person asked for is read from the conversation's provenance
  ledger (``Conversation.exchange_log``), never from agent-written
  text.

Lifecycle: ``review → (human pays in Shop)`` or ``review → scheduled →
verifying → paying → ordered | failed | unknown``; ``cancelled`` from
``review``/``scheduled``/``verifying``.
"""

from boxbot.purchases.store import (
    PurchaseStore,
    get_purchase_store,
    set_purchase_store,
)

__all__ = [
    "PurchaseStore",
    "get_purchase_store",
    "set_purchase_store",
]
