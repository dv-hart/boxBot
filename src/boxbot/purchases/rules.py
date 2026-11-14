"""Deterministic purchase rules — the gate no model can argue with.

Every rule is a plain comparison over facts the main process read
itself: the checkout as Shop returned it, the purchase ledger, the
configured limits, and who the request came from. Nothing here reads
agent-written text.

All rules are evaluated every time (no short-circuit) so the decision
log shows the complete picture, not just the first failure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from boxbot.purchases.shop_client import BUDGET_INSTRUMENT_TYPE


@dataclass(frozen=True, slots=True)
class RuleResult:
    rule: str
    ok: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Requester:
    """Who asked, as established by the transport — not by the agent."""

    channel: str | None        # "signal" | "whatsapp" | "voice" | "trigger" | None
    is_admin_text: bool        # a text from a registered admin's own number
    label: str                 # human-readable, for notifications


def evaluate(
    *,
    config: Any,
    shop: str,
    facts: dict[str, Any],
    requester: Requester,
    spent_today_cents: int,
    spent_month_cents: int,
    duplicate_of: str | None,
    latest_person_at: float | None,
    last_proposal_at: float | None,
) -> list[RuleResult]:
    """Evaluate every automatic-purchase rule.

    ``facts`` is :func:`shop_client.summarize_checkout` output.
    ``spent_*`` are ledger totals that already exclude this purchase.
    ``duplicate_of`` is the id of an earlier proposal for the same item
    from the same store inside the duplicate window, if any — whatever
    became of it (bought, sent for review, cancelled by an admin…).
    ``latest_person_at`` is when a person last spoke in the requesting
    conversation; ``last_proposal_at`` is when that conversation last
    had any purchase proposed (both epoch seconds).
    """
    from boxbot.core.config import MIN_AUTO_VETO_SECONDS

    total = facts.get("total_cents")
    known_total = isinstance(total, int) and total > 0
    results = [
        RuleResult(
            "mode_auto",
            config.mode == "auto",
            f"purchases.mode is {config.mode!r}",
        ),
        RuleResult(
            "shop_allowed",
            shop in config.allowed_shops,
            f"{shop} {'is' if shop in config.allowed_shops else 'is not'} "
            "in purchases.allowed_shops",
        ),
        RuleResult(
            "requester_admin_text",
            requester.is_admin_text or not config.auto_requires_admin_text,
            f"request came from {requester.label}",
        ),
        RuleResult(
            "veto_window",
            config.veto_window_seconds >= MIN_AUTO_VETO_SECONDS,
            f"veto window {config.veto_window_seconds}s vs minimum "
            f"{MIN_AUTO_VETO_SECONDS}s",
        ),
        RuleResult(
            # The same item is bought automatically at most once per
            # window. An earlier proposal that went to review may have
            # been paid by a person; one an admin cancelled was vetoed.
            # Either way a repeat needs a human.
            "no_recent_duplicate",
            duplicate_of is None,
            "no earlier proposal for this item" if duplicate_of is None
            else f"same item already proposed as purchase {duplicate_of}",
        ),
        RuleResult(
            # One spoken/typed request leads to at most one proposal
            # that can buy automatically: a person must have said
            # something since this conversation's last proposal.
            "new_request_since_last_proposal",
            latest_person_at is not None
            and (last_proposal_at is None or latest_person_at > last_proposal_at),
            "no person's message on record" if latest_person_at is None
            else "a person spoke after the last proposal"
            if last_proposal_at is None or latest_person_at > last_proposal_at
            else "no new message from a person since the last proposal",
        ),
        RuleResult(
            "checkout_ready",
            facts.get("status") == "ready_for_complete",
            f"checkout status is {facts.get('status')!r}",
        ),
        RuleResult(
            "no_blocking_messages",
            not any(
                m.get("type") in ("error", "warning")
                for m in facts.get("messages") or []
            ),
            "checkout carries "
            + (", ".join(
                str(m.get("code")) for m in facts.get("messages") or []
                if m.get("type") in ("error", "warning")
            ) or "no errors or warnings"),
        ),
        RuleResult(
            "currency_usd",
            facts.get("currency") == "USD",
            f"currency is {facts.get('currency')!r}",
        ),
        RuleResult(
            "total_known",
            known_total,
            f"total is {total!r}",
        ),
        RuleResult(
            "per_purchase_cap",
            known_total and total <= config.max_per_purchase_cents,
            f"total {total!r} vs cap {config.max_per_purchase_cents}",
        ),
        RuleResult(
            "daily_cap",
            known_total
            and spent_today_cents + total <= config.max_per_day_cents,
            f"today {spent_today_cents} + {total!r} vs cap "
            f"{config.max_per_day_cents}",
        ),
        RuleResult(
            "monthly_cap",
            known_total
            and spent_month_cents + total <= config.max_per_month_cents,
            f"month {spent_month_cents} + {total!r} vs cap "
            f"{config.max_per_month_cents}",
        ),
        RuleResult(
            "item_count",
            0 < (facts.get("item_count") or 0) <= config.max_items,
            f"{facts.get('item_count')!r} items vs max {config.max_items}",
        ),
        RuleResult(
            "budget_instrument",
            facts.get("selected_instrument_type") == BUDGET_INSTRUMENT_TYPE,
            "Shop pre-selected instrument type "
            f"{facts.get('selected_instrument_type')!r}",
        ),
        RuleResult(
            "single_destination",
            len(facts.get("destinations") or []) == 1,
            f"{len(facts.get('destinations') or [])} delivery destination(s)",
        ),
    ]
    return results


def all_pass(results: list[RuleResult]) -> bool:
    return all(r.ok for r in results)


def failed_names(results: list[RuleResult]) -> list[str]:
    return [r.rule for r in results if not r.ok]
