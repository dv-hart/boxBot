"""Persistence for the prefetch layer: the event log and the trigger cache.

Both live in ``memory.db`` (see ``memory/store.py`` for DDL). All writes
are best-effort at the call site — a telemetry/cache failure must never
break a message or a firing trigger.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from boxbot.prefetch.bundle import PrefetchBundle

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def record_prefetch_event(
    store: Any,
    *,
    key: str,
    key_kind: str,
    channel: str,
    mode: str,
    bundle: PrefetchBundle,
    latency_ms: int | None,
    cost_usd: float | None,
) -> None:
    """Append one row describing what this prefetch run predicted."""
    calls = bundle.predicted_integration_calls()
    pulled_at = calls[0]["pulled_at"] if calls else None
    await store.db.execute(
        """
        INSERT INTO prefetch_events (
            timestamp, key, key_kind, channel, mode,
            predicted_memory_ids, predicted_skills,
            predicted_workspace_paths, predicted_integration_calls,
            bundle_token_estimate, prefetch_latency_ms, prefetch_cost_usd,
            note, pulled_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _now().isoformat(),
            key,
            key_kind,
            channel,
            mode,
            json.dumps(bundle.predicted_memory_ids()),
            # sdk-module picks ride the skills column as "bb/modules/x.md"
            # so the offline join needs no schema migration.
            json.dumps(
                bundle.predicted_skills()
                + [f"bb/modules/{m}.md" for m in bundle.predicted_sdk_modules()]
            ),
            json.dumps(bundle.predicted_workspace_paths()),
            json.dumps(calls),
            bundle.token_estimate,
            latency_ms,
            cost_usd,
            "",  # note column retired with the likely_next_note field
            pulled_at,
        ),
    )
    await store.db.commit()


async def cache_put(
    store: Any,
    *,
    trigger_id: str,
    bundle: PrefetchBundle,
    expires_at: datetime,
) -> None:
    """Upsert a precomputed bundle for a scheduled trigger."""
    now = _now().isoformat()
    await store.db.execute(
        """
        INSERT INTO prefetch_cache (trigger_id, bundle_json, pulled_at, expires_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(trigger_id) DO UPDATE SET
            bundle_json=excluded.bundle_json,
            pulled_at=excluded.pulled_at,
            expires_at=excluded.expires_at,
            conversation_id=NULL
        """,
        (trigger_id, json.dumps(bundle.to_dict()), now, expires_at.isoformat()),
    )
    await store.db.commit()


async def cache_get(store: Any, trigger_id: str) -> PrefetchBundle | None:
    """Return the cached bundle if present and not expired, else None."""
    cur = await store.db.execute(
        "SELECT bundle_json, expires_at FROM prefetch_cache WHERE trigger_id = ?",
        (trigger_id,),
    )
    row = await cur.fetchone()
    if not row:
        return None
    try:
        expires_at = datetime.fromisoformat(row[1])
    except Exception:
        expires_at = None
    if expires_at is not None and _now() > expires_at:
        return None
    try:
        return PrefetchBundle.from_dict(json.loads(row[0]))
    except Exception:
        return None


async def cache_has_fresh(store: Any, trigger_id: str) -> bool:
    """True if a non-expired cache row already exists for this trigger."""
    return (await cache_get(store, trigger_id)) is not None


# Hot-task rows (``hot:*`` keys) ride the prefetch_cache table with a
# sentinel expiry: validity is content-fingerprinted by the caller
# (prefetch/hot.py), not time-based.
_KV_EXPIRY = "9999-12-31T00:00:00+00:00"


async def kv_put(store: Any, key: str, payload: dict[str, Any]) -> None:
    """Upsert an arbitrary JSON payload row (hot-task bundles/centroids)."""
    await store.db.execute(
        """
        INSERT INTO prefetch_cache (trigger_id, bundle_json, pulled_at, expires_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(trigger_id) DO UPDATE SET
            bundle_json=excluded.bundle_json,
            pulled_at=excluded.pulled_at,
            expires_at=excluded.expires_at
        """,
        (key, json.dumps(payload), _now().isoformat(), _KV_EXPIRY),
    )
    await store.db.commit()


async def kv_get(store: Any, key: str) -> dict[str, Any] | None:
    """Return the JSON payload stored under ``key``, or None."""
    cur = await store.db.execute(
        "SELECT bundle_json FROM prefetch_cache WHERE trigger_id = ?",
        (key,),
    )
    row = await cur.fetchone()
    if not row:
        return None
    try:
        payload = json.loads(row[0])
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


async def cache_stamp_conversation(
    store: Any, trigger_id: str, conversation_id: str
) -> None:
    """Back-fill the minted conversation_id so the offline join resolves."""
    await store.db.execute(
        "UPDATE prefetch_cache SET conversation_id = ? WHERE trigger_id = ?",
        (conversation_id, trigger_id),
    )
    await store.db.commit()
