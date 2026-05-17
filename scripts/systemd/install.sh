#!/usr/bin/env bash
# Install (or re-install) the boxbot systemd unit on the Pi.
#
# Run ON THE PI, once, after the unit file or its memory caps change:
#   sudo bash scripts/systemd/install.sh
#
# After this:
#   sudo systemctl start boxbot       # start
#   sudo systemctl stop boxbot        # stop (graceful, up to TimeoutStopSec)
#   sudo systemctl status boxbot      # check
#   journalctl -u boxbot -f           # systemd-side log (stdout/stderr)
#
# Application logs continue to go to logs/boxbot.log via Python's
# RotatingFileHandler. scripts/restart-boxbot.sh's pgrep-based bounce
# also still works (it just signals the service's PID), but on a
# systemd-managed install you should prefer `systemctl restart boxbot`
# so the unit's restart counter and memory caps stay coherent.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root (use sudo)." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_SRC="$SCRIPT_DIR/boxbot.service"
UNIT_DEST="/etc/systemd/system/boxbot.service"

if [[ ! -f "$UNIT_SRC" ]]; then
    echo "Error: $UNIT_SRC not found" >&2
    exit 1
fi

echo "Installing $UNIT_SRC → $UNIT_DEST"
install -m 0644 "$UNIT_SRC" "$UNIT_DEST"

echo "Reloading systemd"
systemctl daemon-reload

if systemctl is-enabled boxbot >/dev/null 2>&1; then
    echo "boxbot.service already enabled"
else
    echo "Enabling boxbot.service for boot"
    systemctl enable boxbot.service
fi

cat <<'EOF'

Done. Next steps:

  # Stop any nohup'd boxbot started by restart-boxbot.sh:
  pkill -TERM -f '\.venv/bin/boxbot$' || true

  # Start it under systemd:
  sudo systemctl start boxbot

  # Watch it come up:
  sudo systemctl status boxbot
  tail -f logs/boxbot.log
EOF
