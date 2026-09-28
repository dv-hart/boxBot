#!/usr/bin/env python3
"""Terminal chat client for boxBot's local Signal driver (communication/local_chat.py).

The panel/box runs local_chat as a TCP line server (spoofs the Signal channel);
this is a clean client for it — one line you type = one inbound message, the
agent's replies stream back as ``BB> …``. Nicer than raw ``nc`` for a
screen-share: separate reader thread so replies never garble your typing.

    # If the server is on another host (e.g. the box itself), forward it first:
    #   adb forward tcp:8765 tcp:8765
    python3 scripts/local_chat_client.py            # 127.0.0.1:8765
    python3 scripts/local_chat_client.py --host 1.2.3.4 --port 8765
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading


def _reader(sock: socket.socket) -> None:
    """Print everything the server sends until it closes."""
    while True:
        try:
            data = sock.recv(4096)
        except OSError:
            break
        if not data:
            print("\n[disconnected]")
            return
        sys.stdout.write(data.decode(errors="replace"))
        sys.stdout.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description="boxBot local-chat client")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    try:
        sock = socket.create_connection((args.host, args.port), timeout=10)
    except OSError as exc:
        print(f"cannot connect to {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 1
    print(f"connected to {args.host}:{args.port} — type a message, Ctrl-C to quit\n")

    threading.Thread(target=_reader, args=(sock,), daemon=True).start()
    try:
        for line in sys.stdin:
            sock.sendall(line.rstrip("\n").encode() + b"\n")
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
