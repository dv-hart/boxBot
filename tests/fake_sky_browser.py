#!/usr/bin/env python3
"""Stand-in for Chromium in sky-surface tests.

Speaks just enough DevTools-over-pipe (fd 3 in, fd 4 out, NUL-framed
JSON) for ``boxbot.displays.sky_surface.SkySurface``. Every command is
appended to ``commands.log`` in the state dir (the parent of the
profile directory) so tests can assert on what the surface sent.

Switches, as files in the state dir:
- ``not-ready``  → the page never reports ``BB.ready``
- ``hang``       → ``BB.apply`` is never answered (a hung renderer)
- ``crash-page`` → the next ``BB.apply`` gets ``Inspector.targetCrashed``
  instead of a reply (the file is removed: the relaunch is healthy)

``env.json`` holds the environment the browser was started with.
- ``die-after-N`` is not needed: tests kill the process themselves.
"""

import base64
import json
import os
import sys
from pathlib import Path

# 1x1 PNG.
PNG = base64.b64encode(bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360606060000000050001a5f645400000000049454e44ae426082"
)).decode()


def main() -> int:
    profile = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--user-data-dir="))
    state_dir = Path(profile).parent
    log = state_dir / "commands.log"
    (state_dir / "argv.json").write_text(json.dumps(sys.argv))
    (state_dir / "env.json").write_text(json.dumps(dict(os.environ)))
    reader = os.fdopen(3, "rb", buffering=0)
    buffer = b""
    while True:
        chunk = reader.read(65536)
        if not chunk:
            return 0
        buffer += chunk
        while b"\0" in buffer:
            raw, buffer = buffer.split(b"\0", 1)
            message = json.loads(raw)
            method, params = message["method"], message.get("params", {})
            with log.open("a") as f:
                f.write(json.dumps({"method": method, "params": params}) + "\n")
            if method == "Runtime.evaluate" and params.get("expression", "").startswith("BB.apply("):
                if (state_dir / "hang").exists():
                    continue
                if (state_dir / "crash-page").exists():
                    (state_dir / "crash-page").unlink()
                    event = {"method": "Inspector.targetCrashed", "params": {}, "sessionId": "S1"}
                    os.write(4, json.dumps(event).encode() + b"\0")
                    continue
            reply: dict = {"id": message["id"], "result": {}}
            if "sessionId" in message:
                reply["sessionId"] = message["sessionId"]
            if method == "Target.getTargets":
                reply["result"] = {"targetInfos": [{"targetId": "T1", "type": "page"}]}
            elif method == "Target.attachToTarget":
                reply["result"] = {"sessionId": "S1"}
            elif method == "Runtime.evaluate":
                expression = params.get("expression", "")
                if "BB.ready" in expression:
                    value = not (state_dir / "not-ready").exists()
                else:
                    value = True
                reply["result"] = {"result": {"type": "boolean", "value": value}}
            elif method == "Page.captureScreenshot":
                reply["result"] = {"data": PNG}
            os.write(4, json.dumps(reply).encode() + b"\0")
            if method == "Browser.close":
                return 0


if __name__ == "__main__":
    sys.exit(main())
