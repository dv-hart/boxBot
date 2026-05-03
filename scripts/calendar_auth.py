#!/usr/bin/env python3
"""One-time Google Calendar OAuth bootstrap.

Runs the InstalledAppFlow and saves the resulting token directly into
the boxBot secret store under ``GOOGLE_CALENDAR_TOKEN_JSON``. The
saved token auto-refreshes on subsequent use (the calendar integration
handles that) — this script only needs to be run once per device, or
again whenever Google revokes the refresh token.

Setup:
    1. In Google Cloud Console, create OAuth 2.0 credentials of type
       "Desktop app". Download the JSON.
    2. Save it as ``data/credentials/google_client_secrets.json`` (or
       set GOOGLE_CALENDAR_CLIENT_SECRETS to its path).
    3. Run this script in one of three modes (below).

Modes:
    --auto          Open a browser locally and run a local server on
                    --port to receive the callback. Best on a desktop.
    --no-browser    Same as --auto but doesn't open a browser. Useful
                    if you're tunneling localhost over SSH.
    --manual        Print the auth URL, prompt for the full redirect URL
                    you land on. Use when neither browser nor local
                    server is reachable. Interactive (TTY required).

For headless devices (no keyboard/mouse, accessed only over SSH) use
the two-phase form:

    --print-url     Phase 1: print the auth URL and persist OAuth state
                    to /tmp/boxbot_oauth_state.json. Run via SSH.
    --redirect-url <URL>
                    Phase 2: complete the flow with the URL the browser
                    landed on. Run via SSH after the user finishes the
                    consent screen on whatever device has a browser.

After authenticating, the calendar integration is ready: a fresh
conversation will see ``Secrets: N stored`` include
``GOOGLE_CALENDAR_TOKEN_JSON``, and
``bb.integrations.get("calendar", action="list_upcoming_events")``
will succeed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from boxbot.core.paths import CREDENTIALS_DIR  # noqa: E402
from boxbot.secrets import SecretStore  # noqa: E402


SCOPES = ["https://www.googleapis.com/auth/calendar"]
SECRET_NAME = "GOOGLE_CALENDAR_TOKEN_JSON"

_CLIENT_SECRETS_ENV = "GOOGLE_CALENDAR_CLIENT_SECRETS"
_STATE_PATH = Path("/tmp/boxbot_oauth_state.json")
_REDIRECT_URI = "http://localhost:8765"


def _client_secrets_path() -> Path:
    return Path(
        os.environ.get(_CLIENT_SECRETS_ENV)
        or CREDENTIALS_DIR / "google_client_secrets.json"
    )


def _check_secrets_file(secrets_path: Path) -> int:
    """Return 0 if present, non-zero with an explanation if not."""
    if secrets_path.exists():
        return 0
    print(f"ERROR: client secrets not found at {secrets_path}")
    print()
    print("Create one in Google Cloud Console:")
    print("  1. https://console.cloud.google.com/apis/credentials")
    print("  2. Create OAuth client ID → Desktop app")
    print("  3. Download JSON, save it to the path above")
    print(f"     (or set {_CLIENT_SECRETS_ENV}=/your/path)")
    return 1


def _phase_print_url(secrets_path: Path) -> int:
    """Print the auth URL and save state for phase 2.

    OAuth2 state continuity matters: phase 2 must construct a Flow
    with the same state value, otherwise google-auth-oauthlib's
    ``fetch_token`` raises MismatchingStateError. We write the state
    to ``_STATE_PATH`` (mode 0600) so the second invocation can pick
    it up. This file is not a credential — it's a CSRF-prevention
    nonce — but we still keep it owner-only.
    """
    os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
    from google_auth_oauthlib.flow import Flow

    flow = Flow.from_client_secrets_file(
        str(secrets_path), SCOPES, redirect_uri=_REDIRECT_URI,
    )
    auth_url, state = flow.authorization_url(
        access_type="offline", prompt="consent",
    )
    # PKCE: from_client_secrets_file defaults autogenerate_code_verifier
    # to True for installed-app flows in recent google-auth-oauthlib
    # versions, so the URL carries a code_challenge. Phase 2 must
    # present the matching code_verifier when exchanging the auth code,
    # otherwise Google returns invalid_grant. Persist whatever the
    # library generated.
    payload = {
        "state": state,
        "redirect_uri": _REDIRECT_URI,
        "code_verifier": getattr(flow, "code_verifier", None),
    }
    _STATE_PATH.write_text(json.dumps(payload))
    _STATE_PATH.chmod(0o600)

    print()
    print("=" * 70)
    print("Open this URL in any browser, sign in, grant calendar access:")
    print()
    print(auth_url)
    print()
    print("After granting, the browser redirects to a URL like:")
    print("  http://localhost:8765/?state=...&code=...&scope=...")
    print("(the page itself will fail to load — that's expected).")
    print()
    print("Capture that full redirect URL, then run phase 2:")
    print(f"  python3 scripts/calendar_auth.py --redirect-url '<URL>'")
    print("=" * 70)
    return 0


def _phase_redirect_url(secrets_path: Path, redirect_url: str) -> int:
    """Complete the flow using the redirect URL the browser landed on."""
    if not _STATE_PATH.exists():
        print(
            f"ERROR: no OAuth state at {_STATE_PATH}. Run "
            f"--print-url first (and within the same boot — /tmp is "
            f"cleared on reboot)."
        )
        return 1
    payload = json.loads(_STATE_PATH.read_text())

    os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
    from google_auth_oauthlib.flow import Flow

    flow = Flow.from_client_secrets_file(
        str(secrets_path), SCOPES,
        redirect_uri=payload.get("redirect_uri", _REDIRECT_URI),
        state=payload["state"],
    )
    # Restore the PKCE code_verifier that phase 1 generated; without
    # it, Google rejects the auth code exchange.
    if payload.get("code_verifier"):
        flow.code_verifier = payload["code_verifier"]
    try:
        flow.fetch_token(authorization_response=redirect_url)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: token exchange failed: {exc}")
        return 1
    creds = flow.credentials
    result = SecretStore().store(SECRET_NAME, creds.to_json())
    _STATE_PATH.unlink(missing_ok=True)
    print(f"Stored {SECRET_NAME} in the secret store ({result['previous']}).")
    print("Calendar integration is ready.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port", type=int, default=8765,
        help="Local port for OAuth redirect (default: 8765)",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Run a local server but don't open a browser.",
    )
    parser.add_argument(
        "--manual", action="store_true",
        help="Print URL, prompt for the redirect URL on stdin (TTY).",
    )
    parser.add_argument(
        "--print-url", action="store_true",
        help="Phase 1: print auth URL, save OAuth state, exit.",
    )
    parser.add_argument(
        "--redirect-url", default=None,
        help="Phase 2: complete the flow with this redirect URL.",
    )
    args = parser.parse_args()

    secrets_path = _client_secrets_path()
    rc = _check_secrets_file(secrets_path)
    if rc != 0:
        return rc

    # Two-phase headless flow takes precedence — they're explicit opt-ins.
    if args.print_url:
        return _phase_print_url(secrets_path)
    if args.redirect_url:
        return _phase_redirect_url(secrets_path, args.redirect_url)

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("ERROR: google-auth-oauthlib is not installed.")
        print("  pip install google-auth-oauthlib")
        return 1

    if args.manual:
        os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
        from google_auth_oauthlib.flow import Flow

        flow = Flow.from_client_secrets_file(
            str(secrets_path), SCOPES, redirect_uri=_REDIRECT_URI,
        )
        auth_url, _ = flow.authorization_url(
            access_type="offline", prompt="consent",
        )
        print()
        print("=" * 70)
        print("Open this URL in any browser, sign in, grant calendar access:")
        print()
        print(auth_url)
        print()
        print("Paste the full redirect URL here:")
        print("=" * 70)
        redirect_url = input("Redirect URL: ").strip()
        if not redirect_url:
            print("No URL provided.")
            return 1
        flow.fetch_token(authorization_response=redirect_url)
        creds = flow.credentials
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            str(secrets_path), SCOPES,
        )
        creds = flow.run_local_server(
            port=args.port,
            open_browser=not args.no_browser,
            prompt="consent",
        )

    token_json = creds.to_json()
    result = SecretStore().store(SECRET_NAME, token_json)
    print(f"Stored {SECRET_NAME} in the secret store ({result['previous']}).")
    print("Calendar integration is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
