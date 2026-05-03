#!/usr/bin/env python3
"""One-time Google Calendar OAuth bootstrap.

Runs the InstalledAppFlow with a local server callback. Saves the
resulting token directly into the boxBot secret store under
``GOOGLE_CALENDAR_TOKEN_JSON``. The saved token auto-refreshes on
subsequent use (the calendar integration handles that) — this script
only needs to be run once per device.

Setup:
    1. In Google Cloud Console, create OAuth 2.0 credentials of type
       "Desktop app". Download the JSON.
    2. Save it as ``data/credentials/google_client_secrets.json`` (or
       set GOOGLE_CALENDAR_CLIENT_SECRETS to its path).
    3. Run this script. A browser window will open; sign in and grant
       calendar access.

Usage:
    python3 scripts/calendar_auth.py [--port 8765] [--no-browser]
                                     [--manual]

After authenticating, the calendar integration is ready: a fresh
conversation will see ``Secrets: N stored`` include
``GOOGLE_CALENDAR_TOKEN_JSON`` (via ``bb.secrets.list()``), and
``bb.integrations.get("calendar", action="list_upcoming_events")``
will succeed.
"""

from __future__ import annotations

import argparse
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


def _client_secrets_path() -> Path:
    return Path(
        os.environ.get(_CLIENT_SECRETS_ENV)
        or CREDENTIALS_DIR / "google_client_secrets.json"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port", type=int, default=8765,
        help="Local port for OAuth redirect (default: 8765)",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Print the auth URL instead of opening a browser (for SSH)",
    )
    parser.add_argument(
        "--manual", action="store_true",
        help="Manual mode: print URL, then prompt for the redirect URL "
             "you land on (most reliable for WSL2/SSH).",
    )
    args = parser.parse_args()

    secrets_path = _client_secrets_path()

    if not secrets_path.exists():
        print(f"ERROR: client secrets not found at {secrets_path}")
        print()
        print("Create one in Google Cloud Console:")
        print("  1. https://console.cloud.google.com/apis/credentials")
        print("  2. Create OAuth client ID → Desktop app")
        print("  3. Download JSON, save it to the path above")
        print(f"     (or set {_CLIENT_SECRETS_ENV}=/your/path)")
        return 1

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print(
            "ERROR: google-auth-oauthlib is not installed. "
            "Install it with:"
        )
        print("  pip install google-auth-oauthlib")
        return 1

    if args.manual:
        os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
        from google_auth_oauthlib.flow import Flow

        flow = Flow.from_client_secrets_file(
            str(secrets_path), SCOPES,
            redirect_uri="http://localhost:8765",
        )
        auth_url, _ = flow.authorization_url(
            access_type="offline", prompt="consent"
        )
        print()
        print("=" * 70)
        print("Open this URL in any browser, sign in, grant calendar access:")
        print()
        print(auth_url)
        print()
        print("After granting, your browser will redirect to a URL like:")
        print("  http://localhost:8765/?state=...&code=...&scope=...")
        print("(it'll fail to load — that's fine)")
        print()
        print("Copy that ENTIRE redirect URL and paste it here:")
        print("=" * 70)
        redirect_url = input("Redirect URL: ").strip()
        if not redirect_url:
            print("No URL provided.")
            return 1

        flow.fetch_token(authorization_response=redirect_url)
        creds = flow.credentials
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            str(secrets_path), SCOPES
        )
        if args.no_browser:
            creds = flow.run_local_server(
                port=args.port, open_browser=False, prompt="consent"
            )
        else:
            creds = flow.run_local_server(port=args.port, prompt="consent")

    token_json = creds.to_json()
    result = SecretStore().store(SECRET_NAME, token_json)
    print(f"Stored {SECRET_NAME} in the secret store ({result['previous']}).")
    print("Calendar integration is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
