#!/usr/bin/env python3
"""Connect boxBot to a Shop account from the command line.

    python3 scripts/shop_sign_in.py start    # prints the link + code
    python3 scripts/shop_sign_in.py wait     # polls until approved
    python3 scripts/shop_sign_in.py refresh  # new access token from the
                                             # saved refresh token
    python3 scripts/shop_sign_in.py status   # connected? token fresh?

Thin wrapper over ``boxbot.purchases.shop_auth`` (storage layout and
flow are documented there). Prints only the public user code, the
verification link, and status words — never a token.

On the box the same flow runs over the admin's messaging channel: an
admin texts ``shop connect`` and boxBot replies with the link and code.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from boxbot.purchases import shop_auth  # noqa: E402


def start() -> int:
    result = shop_auth.device_start()
    if not result["ok"]:
        print(f"device request failed: {result['error']}")
        return 1
    print(f"Open:  {result['link']}")
    print(f"Code:  {result['user_code']}")
    print(f"Valid: {result['expires_in'] // 60} minutes")
    print(f"Label: {shop_auth.DEVICE_NAME}")
    return 0


def wait() -> int:
    state = shop_auth.device_wait()
    if state == "approved":
        print("approved; token saved")
        return 0
    if state == "none":
        print("no pending sign-in; run `start` first")
    else:
        print(f"sign-in stopped: {state}")
    return 1


def refresh() -> int:
    ok, detail = shop_auth.refresh()
    print(f"refreshed; token saved ({detail})" if ok
          else f"refresh failed: {detail}")
    return 0 if ok else 1


def status() -> int:
    print(f"connected: {'yes' if shop_auth.is_connected() else 'no'}")
    print(f"access token fresh: "
          f"{'yes' if shop_auth.token_is_fresh() else 'no'}")
    return 0


if __name__ == "__main__":
    commands = {"start": start, "wait": wait, "refresh": refresh,
                "status": status}
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in commands:
        print(__doc__)
        sys.exit(2)
    sys.exit(commands[cmd]())
