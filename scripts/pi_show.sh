#!/bin/bash
# Runs on the Pi. Swaps the fullscreen display to a new PNG.
# Called by render_and_show.sh over SSH. Uses setsid to detach cleanly
# so the process survives SSH disconnect.
#
# Usage (on Pi):
#   pi_show.sh <png-path>

set -euo pipefail

PNG="${1:-}"
if [[ -z "$PNG" ]] || [[ ! -f "$PNG" ]]; then
    echo "usage: $0 <png-path>" >&2
    exit 2
fi

pkill -f show_on_screen.py || true
sleep 1

# Start detached so SSH disconnect doesn't kill it.
setsid --fork \
    env DISPLAY=:0 \
    python3 -u "$HOME/software/boxBot/scripts/show_on_screen.py" "$PNG" --driver x11 \
    > /tmp/show_log.txt 2>&1 </dev/null

sleep 2
echo "pid: $(pgrep -f show_on_screen.py | head -1)"
tail -3 /tmp/show_log.txt
