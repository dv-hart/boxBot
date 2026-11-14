"""Shop account connection: OAuth device flow + token files.

boxBot connects to a Shop account once with the device flow (a human
opens a link and approves a short code). Shop returns a one-hour access
token and a refresh token; :func:`refresh` swaps the refresh token for a
new access token with no human involved.

Storage, all under ``data/auth/shop/`` (dir 0700, files 0600):

| File | Holds |
|---|---|
| ``auth.header`` | ``Authorization: Bearer …`` line, ready to send |
| ``refresh.token`` | refresh token |
| ``token_meta.json`` | non-secret facts (obtained_at, expires_in, scope) |
| ``device.json`` | pending device request (deleted once resolved) |

Tokens never leave the main process: they are not returned to the
sandbox, not logged, and not placed in any model context.

Stdlib only and synchronous on purpose, so ``scripts/shop_sign_in.py``
can use it without the rest of boxBot; async callers wrap these in
``asyncio.to_thread``.

Flow and constants: https://shop.com/references/sign-in.md
"""

from __future__ import annotations

import http.client
import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from boxbot.core.paths import AUTH_DIR

DEVICE_ENDPOINT = "https://accounts.shop.app/oauth/device"
TOKEN_ENDPOINT = "https://accounts.shop.app/oauth/token"
CLIENT_ID = "5c733ab2-1903-400a-891e-7ba20c09e2a3"  # Shop's public client
RESOURCE = "https://mcp.shop.com"
SCOPE = "openid email personal_agent"
DEVICE_NAME = "boxBot"

SHOP_AUTH_DIR = AUTH_DIR / "shop"

# Refresh this long before the access token's stated expiry.
_EXPIRY_MARGIN_SECONDS = 120


def _dir() -> Path:
    return SHOP_AUTH_DIR


def _device_file() -> Path:
    return _dir() / "device.json"


def _header_file() -> Path:
    return _dir() / "auth.header"


def _meta_file() -> Path:
    return _dir() / "token_meta.json"


def _refresh_file() -> Path:
    return _dir() / "refresh.token"


def _post(url: str, fields: dict[str, str]) -> dict:
    """POST a form; return the JSON body for 2xx and OAuth-error 4xx."""
    req = urllib.request.Request(
        url,
        data=urllib.parse.urlencode(fields).encode(),
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read())
        except Exception:
            return {"error": f"http_{e.code}"}
    except (OSError, http.client.HTTPException, ValueError) as e:
        # URLError and timeouts are OSErrors; a dropped or half-read
        # connection surfaces as an HTTPException.
        return {"error": f"network: {type(e).__name__}"}
    return body if isinstance(body, dict) else {"error": "bad_response"}


def _write_private(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically, mode 0600.

    Temp file + rename, so a reader never sees a half-written token and
    a crash mid-write leaves the previous file intact.
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    # mkstemp: unique name (two writers never share a temp file) and
    # mode 0600 from creation.
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# A refresh (client thread) and a device-flow poll (sign-in thread) can
# both land tokens; the three files must change together.
_save_lock = threading.Lock()


def _save_tokens(body: dict) -> None:
    with _save_lock:
        _save_tokens_locked(body)


def _save_tokens_locked(body: dict) -> None:
    # Refresh token first: if Shop rotated it, losing the new one would
    # strand the connection, while a stale access token only costs one
    # more refresh. A response may omit it; then the old one stays.
    if body.get("refresh_token"):
        _write_private(_refresh_file(), body["refresh_token"])
    _write_private(
        _header_file(), f"Authorization: Bearer {body['access_token']}\n"
    )
    _write_private(_meta_file(), json.dumps({
        "obtained_at": time.time(),
        "expires_in": body.get("expires_in"),
        "token_type": body.get("token_type"),
        "scope": body.get("scope"),
        "has_refresh_token": _refresh_file().exists(),
    }))


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def is_connected() -> bool:
    """True when a refresh token or an access token is on disk."""
    return _refresh_file().exists() or _header_file().exists()


def token_is_fresh() -> bool:
    """True when the saved access token has not reached its expiry margin."""
    if not _header_file().exists():
        return False
    try:
        meta = json.loads(_meta_file().read_text())
        expires_at = float(meta["obtained_at"]) + float(meta["expires_in"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return time.time() < expires_at - _EXPIRY_MARGIN_SECONDS


def auth_header() -> tuple[str, str] | None:
    """``(name, value)`` of the saved Authorization header, or None."""
    try:
        line = _header_file().read_text().strip()
    except OSError:
        return None
    name, sep, value = line.partition(": ")
    if not sep or not value:
        return None
    return name, value


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------


def refresh() -> tuple[bool, str]:
    """Swap the refresh token for a new access token.

    Returns ``(ok, detail)``. ``detail`` is the OAuth error code on
    failure (``invalid_grant`` means the human must sign in again),
    never a token.
    """
    try:
        token = _refresh_file().read_text().strip()
    except OSError:
        return False, "no_refresh_token"
    if not token:
        return False, "no_refresh_token"
    body = _post(TOKEN_ENDPOINT, {
        "grant_type": "refresh_token",
        "client_id": CLIENT_ID,
        "refresh_token": token,
    })
    if not body.get("access_token"):
        return False, str(body.get("error", "unknown"))
    _save_tokens(body)
    return True, f"expires_in={body.get('expires_in')}"


# ---------------------------------------------------------------------------
# Device flow
# ---------------------------------------------------------------------------


def device_start() -> dict:
    """Begin a device sign-in.

    Returns ``{"ok": True, "link", "user_code", "expires_in",
    "interval"}`` — only the public, human-facing values — or
    ``{"ok": False, "error"}``. The private device code stays in
    ``device.json``.
    """
    body = _post(DEVICE_ENDPOINT, {
        "client_id": CLIENT_ID,
        "resource": RESOURCE,
        "scope": SCOPE,
        "device_name": DEVICE_NAME,
    })
    if "device_code" not in body:
        detail = f"{body.get('error', 'unknown')} " \
                 f"{body.get('error_description', '')}".strip()
        return {"ok": False, "error": detail}
    body["_requested_at"] = time.time()
    _write_private(_device_file(), json.dumps(body))
    return {
        "ok": True,
        "link": body.get("verification_uri_complete")
        or body.get("verification_uri"),
        "user_code": body.get("user_code"),
        "expires_in": int(body.get("expires_in", 0)),
        "interval": float(body.get("interval", 5)),
    }


def device_poll_once() -> str:
    """Ask the token endpoint once whether the human approved.

    Returns ``"approved"`` (tokens saved), ``"pending"``,
    ``"slow_down"``, ``"expired"``, ``"none"`` (no pending request), or
    the OAuth error code that ended the request.
    """
    try:
        device = json.loads(_device_file().read_text())
    except (OSError, ValueError):
        return "none"
    deadline = float(device.get("_requested_at", 0)) + float(
        device.get("expires_in", 600)
    )
    if time.time() >= deadline:
        _device_file().unlink(missing_ok=True)
        return "expired"
    body = _post(TOKEN_ENDPOINT, {
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        "client_id": CLIENT_ID,
        "device_code": device["device_code"],
    })
    if body.get("access_token"):
        _save_tokens(body)
        _device_file().unlink(missing_ok=True)
        return "approved"
    err = str(body.get("error", "unknown"))
    if err == "authorization_pending" or err.startswith("network"):
        return "pending"
    if err == "slow_down":
        return "slow_down"
    _device_file().unlink(missing_ok=True)
    return err


def device_wait() -> str:
    """Poll until the pending device request resolves. Blocking.

    Returns ``"approved"`` or the reason it stopped.
    """
    try:
        device = json.loads(_device_file().read_text())
    except (OSError, ValueError):
        return "none"
    interval = float(device.get("interval", 5))
    while True:
        time.sleep(interval)
        state = device_poll_once()
        if state == "pending":
            continue
        if state == "slow_down":
            interval += 5
            continue
        return state
