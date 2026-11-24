#!/usr/bin/env python3
"""Stand-in for Chromium in web-view tests.

Speaks just enough DevTools-over-pipe (fd 3 in, fd 4 out, NUL-framed
JSON) for ``boxbot.displays.web_view.WebView``. Every command received
is appended to ``commands.log`` beside the profile directory so tests
can assert on what the view asked the browser to do.

A file named ``playing`` / ``touched`` beside the log makes the activity
probe report media playing / a recent touch.

Behaviour switches, read from the navigated URL's host:
- ``fail.example.com``  → Page.navigate returns an errorText
- ``crash.example.com`` → the page loads, then the process dies 0.3 s later
- ``slow.example.com``  → Page.navigate is never answered
"""

import json
import os
import sys
import threading
import time
from pathlib import Path


def main() -> int:
    profile = next(
        a.split("=", 1)[1] for a in sys.argv if a.startswith("--user-data-dir=")
    )
    # <state_dir>/home/profile → the log lives in <state_dir>, outside
    # the home that the launch wrapper wipes.
    log = Path(profile).parent.parent / "commands.log"
    reader = os.fdopen(3, "rb", buffering=0)
    buffer = b""

    def send(message: dict) -> None:
        os.write(4, json.dumps(message).encode() + b"\0")

    while True:
        chunk = reader.read(65536)
        if not chunk:
            return 0
        buffer += chunk
        while b"\0" in buffer:
            raw, buffer = buffer.split(b"\0", 1)
            message = json.loads(raw)
            method = message["method"]
            params = message.get("params", {})
            with log.open("a") as f:
                f.write(json.dumps({"method": method, "params": params}) + "\n")

            reply: dict = {"id": message["id"], "result": {}}
            if "sessionId" in message:
                reply["sessionId"] = message["sessionId"]

            if method == "Target.getTargets":
                reply["result"] = {"targetInfos": [{"targetId": "T1", "type": "page"}]}
            elif method == "Target.attachToTarget":
                reply["result"] = {"sessionId": "S1"}
            elif method == "Page.navigate":
                if "fail.example.com" in params["url"]:
                    reply["result"] = {"errorText": "net::ERR_NAME_NOT_RESOLVED"}
                if "slow.example.com" in params["url"]:
                    time.sleep(3600)
                send(reply)
                send({"method": "Page.loadEventFired", "sessionId": "S1",
                      "params": {"timestamp": 1}})
                if "crash.example.com" in params["url"]:
                    threading.Timer(0.3, os._exit, [7]).start()
                continue
            elif method == "Runtime.evaluate":
                expression = params["expression"]
                if "__bbSeen" in expression:
                    # Tests flip these by touching files beside the log.
                    value = json.dumps({
                        "playing": (log.parent / "playing").exists(),
                        "quiet_ms": 0 if (log.parent / "touched").exists() else 10**7,
                    })
                else:
                    value = 1
                reply["result"] = {"result": {"type": "string", "value": value}}
            elif method == "Browser.close":
                send(reply)
                return 0
            send(reply)


if __name__ == "__main__":
    sys.exit(main())
