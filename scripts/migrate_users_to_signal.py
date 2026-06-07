#!/usr/bin/env python3
"""Phase 4 of the Signal migration — flip registered users' channel.

Two-stage UX:

  1. ``status``: show every registered user and which channel they're
     currently set to. Run this first to see what needs flipping.
  2. ``notify``: send the user a heads-up via their CURRENT channel —
     "I'm moving to Signal — text me there to confirm." Useful when
     the operator wants the user to opt in instead of being moved
     silently.
  3. ``flip``: change the ``users.channel`` column. After this, the
     agent's output dispatcher sends to the new channel.

Designed to be run ON the Pi against the live ``data/auth/users.db``.

Auth lookups in the router are channel-agnostic, so a user still on
``channel='whatsapp'`` who sends from Signal is accepted — their
inbound thread is on Signal, BB's outbound still goes via WhatsApp
until ``flip`` is run for them. That's the intended transitional state.

Examples:

  # See current state
  python3 scripts/migrate_users_to_signal.py status

  # Notify Jacob via his current channel that we're moving
  python3 scripts/migrate_users_to_signal.py notify --phone +15035086292

  # Flip Jacob to Signal
  python3 scripts/migrate_users_to_signal.py flip --phone +15035086292

  # Flip everyone (use after you've confirmed they all have Signal)
  python3 scripts/migrate_users_to_signal.py flip --all
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Make src/ importable when running directly off the Pi.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


async def _list_users() -> list:
    from boxbot.communication.auth import AuthManager
    auth = AuthManager()
    await auth.init_db()
    return await auth.list_users()


async def cmd_status(_args) -> int:
    users = await _list_users()
    if not users:
        print("No registered users.")
        return 0
    print(f"{'phone':<16}  {'name':<20}  {'role':<8}  channel")
    print("-" * 60)
    for u in users:
        print(f"{u.phone:<16}  {u.name:<20}  {u.role:<8}  {u.channel}")
    on_whatsapp = sum(1 for u in users if u.channel == "whatsapp")
    on_signal = sum(1 for u in users if u.channel == "signal")
    print()
    print(f"Total: {len(users)}  (whatsapp: {on_whatsapp}, signal: {on_signal})")
    return 0


async def cmd_flip(args) -> int:
    from boxbot.communication.auth import AuthManager
    auth = AuthManager()
    await auth.init_db()
    users = await auth.list_users()

    if args.all:
        targets = [u for u in users if u.channel != "signal"]
    else:
        targets = [u for u in users if u.phone == args.phone]
        if not targets:
            print(f"No user with phone {args.phone!r}.")
            return 1
        if targets[0].channel == "signal":
            print(f"{targets[0].name} ({targets[0].phone}) is already on signal.")
            return 0

    if not targets:
        print("Nothing to flip — every user is already on signal.")
        return 0

    print("About to flip:")
    for u in targets:
        print(f"  {u.name:<20} {u.phone:<16} {u.channel} → signal")
    if not args.yes:
        confirm = input("Proceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return 1

    for u in targets:
        ok = await auth.update_channel(u.phone, "signal")
        if ok:
            print(f"  ✓ flipped {u.name} ({u.phone}) → signal")
        else:
            print(f"  ✗ failed to flip {u.name} ({u.phone})")
    return 0


async def cmd_notify(args) -> int:
    """Send a heads-up via the user's CURRENT outbound channel.

    Needs the daemons / clients to be reachable — i.e. run on the Pi
    while the agent is running, OR with env vars set so the clients
    can initialise. We piggy-back on the running boxbot's auth+channels
    by going through the OutboundChannel registry once it's populated.
    """
    from boxbot.communication.auth import AuthManager
    from boxbot.communication.channels import (
        Channel, get_outbound_channel, register_outbound_channel,
    )

    # Minimal client init for the standalone script: same code paths
    # boxbot.service uses, but skipping the rest of the agent.
    from boxbot.core.config import get_config
    config = get_config()

    if config.whatsapp.enabled and (
        config.api_keys.whatsapp_access_token
        and config.api_keys.whatsapp_phone_number_id
    ):
        from boxbot.communication.whatsapp import WhatsAppClient, set_whatsapp_client
        wa = WhatsAppClient(
            access_token=config.api_keys.whatsapp_access_token,
            phone_number_id=config.api_keys.whatsapp_phone_number_id,
        )
        set_whatsapp_client(wa)

    if config.signal.enabled and config.api_keys.signal_account:
        from boxbot.communication.signal_client import (
            SignalClient, set_signal_client,
        )
        sig = SignalClient(
            socket_path=config.signal.socket_path,
            account=config.api_keys.signal_account,
            attachments_dir=config.signal.attachments_dir,
        )
        try:
            await sig.connect()
            set_signal_client(sig)
        except ConnectionError as e:
            print(f"  Signal client could not connect: {e}", file=sys.stderr)

    auth = AuthManager()
    await auth.init_db()
    user = await auth.get_user(args.phone)
    if user is None:
        print(f"No registered user with phone {args.phone!r}.")
        return 1

    try:
        target_channel = Channel(user.channel)
    except ValueError:
        print(f"User has unknown channel {user.channel!r}.")
        return 1
    out = get_outbound_channel(target_channel)
    if out is None:
        print(
            f"No outbound client registered for {user.channel!r}. "
            f"Make sure that channel is enabled in config + reachable.",
            file=sys.stderr,
        )
        return 1

    message = args.message or (
        f"Hi {user.name} — BB is moving from WhatsApp to Signal as the "
        f"main messaging channel. You should already have a thread "
        f"with me on Signal at this same number (+15039858519). "
        f"Reply YES on Signal and I'll switch you over. "
        f"WhatsApp will keep working for now as a fallback."
    )
    print(f"Sending via {out.name} to {user.phone}:\n  {message[:120]}…")
    ok = await out.send_text(user.phone, message)
    print("  ✓ sent" if ok else "  ✗ failed")
    return 0 if ok else 1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="list every registered user + their channel")

    fp = sub.add_parser("flip", help="change a user's channel to signal")
    fp.add_argument("--phone", help="user phone in E.164")
    fp.add_argument(
        "--all", action="store_true",
        help="flip every user currently on whatsapp → signal",
    )
    fp.add_argument(
        "-y", "--yes", action="store_true",
        help="skip the confirmation prompt",
    )

    np_ = sub.add_parser(
        "notify",
        help="send a transition heads-up via the user's current channel",
    )
    np_.add_argument("--phone", required=True, help="user phone in E.164")
    np_.add_argument(
        "--message",
        help="override the default heads-up text",
    )

    return p


def main() -> int:
    args = _build_parser().parse_args()
    if args.cmd == "flip" and not args.phone and not args.all:
        print("flip requires --phone or --all", file=sys.stderr)
        return 2
    if args.cmd == "flip" and args.phone and args.all:
        print("flip: --phone and --all are mutually exclusive", file=sys.stderr)
        return 2

    handler = {
        "status": cmd_status,
        "flip": cmd_flip,
        "notify": cmd_notify,
    }[args.cmd]
    return asyncio.run(handler(args))


if __name__ == "__main__":
    sys.exit(main())
