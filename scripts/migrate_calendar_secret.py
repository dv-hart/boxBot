#!/usr/bin/env python3
"""Migrate the existing Google Calendar token from disk into the secret store.

Use this once after deploying the calendar-as-integration migration. The
old token at ``data/credentials/google_calendar_token.json`` becomes a
secret named ``GOOGLE_CALENDAR_TOKEN_JSON``; the file is renamed to
``.migrated`` so a future deploy of this script is a no-op rather than
silently overwriting an updated secret.

Usage (run on the Pi or in dev — the project venv is required because
the import path pulls in boxbot.core which depends on anthropic):

    .venv/bin/python3 scripts/migrate_calendar_secret.py

Idempotent: if the secret already exists and the file is already
suffixed ``.migrated``, the script reports the state and exits 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from boxbot.core.paths import CREDENTIALS_DIR  # noqa: E402
from boxbot.secrets import SecretStore  # noqa: E402


SECRET_NAME = "GOOGLE_CALENDAR_TOKEN_JSON"
DEFAULT_TOKEN = CREDENTIALS_DIR / "google_calendar_token.json"


def _validate_token_shape(text: str) -> dict:
    """Confirm the file parses and looks like a google-auth token."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERROR: token file is not valid JSON: {exc}")
    if not isinstance(data, dict):
        raise SystemExit("ERROR: token JSON must be an object")
    required = {"refresh_token", "client_id", "client_secret"}
    missing = required - data.keys()
    if missing:
        raise SystemExit(
            f"ERROR: token JSON missing required fields: {sorted(missing)}. "
            "Re-run scripts/calendar_auth.py to mint a fresh token."
        )
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token-path",
        type=Path,
        default=DEFAULT_TOKEN,
        help=f"Path to the saved token JSON (default: {DEFAULT_TOKEN}).",
    )
    parser.add_argument(
        "--keep-file",
        action="store_true",
        help="Don't rename the source file to .migrated after import.",
    )
    args = parser.parse_args()

    token_path: Path = args.token_path
    store = SecretStore()

    already_stored = store.has(SECRET_NAME)

    if not token_path.exists():
        if already_stored:
            print(
                f"OK: {SECRET_NAME} is already in the secret store and the "
                f"source file at {token_path} is gone — nothing to do."
            )
            return 0
        migrated = token_path.with_suffix(token_path.suffix + ".migrated")
        if migrated.exists():
            print(
                f"OK: source file already renamed to {migrated} — nothing "
                f"to do. (If you want to re-import, rename it back first.)"
            )
            return 0
        print(
            f"ERROR: no token file at {token_path} and {SECRET_NAME} is not "
            f"stored. Run scripts/calendar_auth.py first."
        )
        return 1

    text = token_path.read_text(encoding="utf-8")
    _validate_token_shape(text)

    if already_stored:
        print(
            f"WARNING: {SECRET_NAME} is already stored. Overwriting with the "
            f"contents of {token_path}."
        )

    result = store.store(SECRET_NAME, text)
    print(f"Stored {SECRET_NAME} ({result['previous']}).")

    if not args.keep_file:
        new_path = token_path.with_suffix(token_path.suffix + ".migrated")
        token_path.rename(new_path)
        print(f"Renamed source to {new_path}")

    print("Migration complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
