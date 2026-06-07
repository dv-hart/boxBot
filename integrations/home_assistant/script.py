"""Home Assistant integration — sandbox-runnable.

Wraps the HA REST API behind an action dispatcher. URL and long-lived
access token are loaded from the secret store (HOME_ASSISTANT_URL,
HOME_ASSISTANT_TOKEN); the script makes one HTTP call per action and
returns a normalized dict.

Mutation policy (V1): call_service refuses domains that can compromise
physical security — alarm_control_panel, lock, cover. Read operations
on those domains (get_state) are fine; only state-changing service
calls are blocked. The full per-action confirmation gate (matching the
package-install approval flow) is a V2 deliverable.

Camera snapshots: the integration fetches the JPEG via
/api/camera_proxy/<entity_id>, writes it to a workspace tmp path via
bb.workspace.write, and returns the path. It does NOT interpret the
image — the caller (agent script, skill, or display) chooses whether
to view pixels with bb.workspace.view, run a small-model tag, or just
put it on screen.
"""

from __future__ import annotations

import datetime
import os
import re
import sys
from typing import Any

import httpx

from boxbot_sdk.integration import inputs as get_inputs, return_output


URL_SECRET = "HOME_ASSISTANT_URL"
TOKEN_SECRET = "HOME_ASSISTANT_TOKEN"

_TIMEOUT = httpx.Timeout(15.0)
# HA's /api/services/<domain>/<service> blocks until the service finishes. For
# state-tracked cloud devices (Alarm.com / Z-Wave lights, etc.) the first "cold"
# command waits on a full cloud confirmation round-trip that routinely exceeds
# 15s — so service calls get a longer read budget than plain state reads.
_CALL_TIMEOUT = httpx.Timeout(15.0, read=45.0)

# Domains whose state-changing service calls are blocked in V1.
# Read operations (get_state) on these are always allowed.
_BLOCKED_MUTATION_DOMAINS = frozenset({
    "alarm_control_panel",
    "lock",
    "cover",
})

# Workspace folder for camera snapshots. Each call writes one file.
_SNAPSHOT_DIR = "tmp/ha"


# ---------------------------------------------------------------------------
# Connection setup
# ---------------------------------------------------------------------------


def _load_connection() -> tuple[str, str] | None:
    """Read URL + token from env-injected secrets.

    Returns (url, token) or None after writing an error output.
    """
    url = os.environ.get(f"BOXBOT_SECRET_{URL_SECRET}", "").strip()
    token = os.environ.get(f"BOXBOT_SECRET_{TOKEN_SECRET}", "").strip()
    if not url:
        return_output({
            "error": (
                f"Home Assistant URL not stored. Run "
                f"bb.secrets.store('{URL_SECRET}', 'http://<host>:8123') "
                f"with your HA instance's address."
            )
        })
        return None
    if not token:
        return_output({
            "error": (
                f"Home Assistant token not stored. Generate a long-lived "
                f"access token in HA (profile → Long-Lived Access Tokens), "
                f"then bb.secrets.store('{TOKEN_SECRET}', '<token>')."
            )
        })
        return None
    # Strip trailing slash so we can always do f"{url}/api/..."
    return url.rstrip("/"), token


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------


def _get_states(url: str, token: str, args: dict[str, Any]) -> dict[str, Any]:
    domain_filter = (args.get("domain") or "").strip()
    resp = httpx.get(f"{url}/api/states", headers=_headers(token), timeout=_TIMEOUT)
    resp.raise_for_status()
    items = resp.json()
    if domain_filter:
        items = [e for e in items if e.get("entity_id", "").startswith(f"{domain_filter}.")]
    # Trim each entity to the fields a consumer typically wants — full
    # attribute payloads are large and rarely all needed up front.
    entities = [
        {
            "entity_id": e.get("entity_id", ""),
            "state": e.get("state", ""),
            "friendly_name": e.get("attributes", {}).get("friendly_name", ""),
            "last_changed": e.get("last_changed", ""),
        }
        for e in items
    ]
    return {"entities": entities}


def _get_state(url: str, token: str, args: dict[str, Any]) -> dict[str, Any]:
    entity_id = (args.get("entity_id") or "").strip()
    if not entity_id:
        return {"error": "get_state requires entity_id"}
    resp = httpx.get(
        f"{url}/api/states/{entity_id}",
        headers=_headers(token),
        timeout=_TIMEOUT,
    )
    if resp.status_code == 404:
        return {"error": f"entity not found: {entity_id}"}
    resp.raise_for_status()
    payload = resp.json()
    return {
        "state": payload.get("state", ""),
        "attributes": payload.get("attributes", {}),
        "last_changed": payload.get("last_changed", ""),
    }


def _read_state(url: str, token: str, entity_id: str) -> dict[str, Any] | None:
    """Best-effort re-read of one entity's state. Returns None on any failure —
    callers use this only to enrich a service-call result, never as the result."""
    try:
        resp = httpx.get(
            f"{url}/api/states/{entity_id}",
            headers=_headers(token),
            timeout=_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        return {
            "state": data.get("state"),
            "attributes": data.get("attributes", {}),
        }
    except Exception:  # noqa: BLE001 — enrichment only, never fatal
        return None


def _call_service(url: str, token: str, args: dict[str, Any]) -> dict[str, Any]:
    domain = (args.get("domain") or "").strip()
    service = (args.get("service") or "").strip()
    if not domain or not service:
        return {"error": "call_service requires domain and service"}
    if domain in _BLOCKED_MUTATION_DOMAINS:
        return {
            "error": (
                f"service calls in domain '{domain}' are blocked in V1 — "
                "alarm/lock/cover mutations need the confirmation gate "
                "(not yet implemented). State reads (get_state) on these "
                "entities still work."
            )
        }
    body: dict[str, Any] = {}
    extra = args.get("service_data")
    if isinstance(extra, dict):
        body.update(extra)
    entity_id = (args.get("entity_id") or "").strip()
    if entity_id:
        body["entity_id"] = entity_id
    try:
        resp = httpx.post(
            f"{url}/api/services/{domain}/{service}",
            headers=_headers(token),
            json=body,
            timeout=_CALL_TIMEOUT,
        )
    except httpx.TimeoutException:
        # HA didn't confirm within the window, but the command was almost
        # certainly dispatched (slow cloud/Z-Wave confirmation). Don't report a
        # hard failure — re-read the entity so the agent sees the real outcome.
        observed = _read_state(url, token, entity_id) if entity_id else None
        result = {
            "ok": True,
            "confirmed": False,
            "affected": [],
            "note": (
                "HA did not confirm the service within the timeout — common for "
                "Alarm.com/Z-Wave devices on a cold call. The command was sent; "
                "the resulting_state below was re-read afterward. Verify it "
                "matches intent rather than assuming the call failed."
            ),
        }
        if observed:
            result["resulting_state"] = observed
        return result
    if resp.status_code == 400:
        return {"error": f"HA rejected call: {resp.text[:200]}"}
    resp.raise_for_status()
    # HA returns a list of entities whose state changed as a result. This list is
    # empty when HA's cached state already matched the request (e.g. turning off
    # an already-off light) — which is NOT a failure. Re-read the entity so the
    # agent can verify the actual resulting state instead of inferring from it.
    payload = resp.json() if resp.content else []
    affected = [e.get("entity_id", "") for e in payload if isinstance(e, dict)]
    result = {"ok": True, "confirmed": True, "affected": affected}
    if entity_id:
        observed = _read_state(url, token, entity_id)
        if observed:
            result["resulting_state"] = observed
    return result


def _camera_snapshot(url: str, token: str, args: dict[str, Any]) -> dict[str, Any]:
    entity_id = (args.get("entity_id") or "").strip()
    if not entity_id:
        return {"error": "camera_snapshot requires entity_id"}
    if not entity_id.startswith("camera."):
        return {"error": f"camera_snapshot expects a camera.* entity, got '{entity_id}'"}
    resp = httpx.get(
        f"{url}/api/camera_proxy/{entity_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=_TIMEOUT,
    )
    if resp.status_code == 404:
        return {"error": f"camera not found or no snapshot available: {entity_id}"}
    resp.raise_for_status()
    image_bytes = resp.content
    if not image_bytes:
        return {"error": f"empty snapshot returned by HA for {entity_id}"}
    now = datetime.datetime.now(datetime.timezone.utc)
    captured_at = now.isoformat().replace("+00:00", "Z")
    # Filesystem-friendly compact timestamp: 20260517T143052Z
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    safe_entity = re.sub(r"[^a-zA-Z0-9_]", "_", entity_id)
    path = f"{_SNAPSHOT_DIR}/{safe_entity}_{stamp}.jpg"

    # Late import — only camera_snapshot needs the workspace transport.
    import boxbot_sdk as bb

    bb.workspace.write(path, image_bytes)
    return {
        "image_path": path,
        "entity_id": entity_id,
        "captured_at": captured_at,
    }


def _list_services(url: str, token: str, args: dict[str, Any]) -> dict[str, Any]:
    domain_filter = (args.get("domain") or "").strip()
    resp = httpx.get(f"{url}/api/services", headers=_headers(token), timeout=_TIMEOUT)
    resp.raise_for_status()
    items = resp.json()  # [{"domain": "light", "services": {"turn_on": {...}, ...}}, ...]
    entries: list[dict[str, Any]] = []
    for block in items:
        domain = block.get("domain", "")
        if domain_filter and domain != domain_filter:
            continue
        services = block.get("services", {}) or {}
        for service_name, meta in services.items():
            entries.append({
                "domain": domain,
                "service": service_name,
                "description": (meta or {}).get("description", ""),
            })
    return {"entities": entries}


_ACTIONS = {
    "get_states": _get_states,
    "get_state": _get_state,
    "call_service": _call_service,
    "camera_snapshot": _camera_snapshot,
    "list_services": _list_services,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = get_inputs()
    action = args.get("action")
    if action not in _ACTIONS:
        return_output({
            "error": (
                f"unknown action '{action}' — "
                f"expected one of {sorted(_ACTIONS.keys())}"
            )
        })
        return

    conn = _load_connection()
    if conn is None:
        return  # _load_connection already wrote the output
    url, token = conn

    try:
        result = _ACTIONS[action](url, token, args)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status == 401:
            return_output({
                "error": (
                    "HA returned 401 — token rejected. Generate a fresh "
                    f"long-lived access token in HA and bb.secrets.store('{TOKEN_SECRET}', ...)."
                )
            })
            return
        return_output({"error": f"HA API {status}: {exc.response.text[:200]}"})
        return
    except httpx.ConnectError as exc:
        return_output({"error": f"could not reach HA at {url}: {exc}"})
        return
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"home_assistant action {action} failed: {exc}\n")
        return_output({"error": str(exc)})
        return

    return_output(result)


if __name__ == "__main__":
    main()
