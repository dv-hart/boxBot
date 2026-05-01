#!/usr/bin/env bash
# Restart the boxBot service. Designed to run ON THE PI — typically
# invoked by ``scripts/deploy.sh`` via SSH after a ``git pull``, but can
# also be run manually for a quick bounce.
#
# What it does:
#   1. SIGTERM the current boxbot process and wait up to 15s for clean exit.
#   2. SIGKILL if it's still alive after that.
#   3. Spawn a fresh boxbot via ``nohup setsid`` with a fresh dated log file.
#   4. Sleep briefly, then tail the new log so the operator can see startup.
#
# Idempotent: if no boxbot is running, just starts one.
#
# Exits non-zero if the new process didn't survive startup.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/software/boxBot}"
VENV_BOXBOT="$PROJECT_DIR/.venv/bin/boxbot"

if [[ ! -x "$VENV_BOXBOT" ]]; then
    echo "Error: $VENV_BOXBOT not found or not executable" >&2
    echo "Has the venv been set up? (cd $PROJECT_DIR && ./.venv/bin/pip install -e .)" >&2
    exit 1
fi

cd "$PROJECT_DIR"

# -------------------------------------------------------------------
# Stop existing boxbot (if any)
# -------------------------------------------------------------------

CUR_PID="$(pgrep -f '\.venv/bin/boxbot$' | head -1 || true)"
if [[ -n "$CUR_PID" ]]; then
    echo "Stopping current boxbot (pid $CUR_PID)..."
    kill -TERM "$CUR_PID" || true
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
        if ! kill -0 "$CUR_PID" 2>/dev/null; then
            echo "  exited cleanly"
            break
        fi
        sleep 1
    done
    if kill -0 "$CUR_PID" 2>/dev/null; then
        echo "  still alive after 15s, sending SIGKILL"
        kill -KILL "$CUR_PID" || true
        sleep 1
    fi
else
    echo "No running boxbot found"
fi

# -------------------------------------------------------------------
# Start fresh
# -------------------------------------------------------------------

mkdir -p logs
LOG="logs/boxbot-$(date +%Y%m%d-%H%M%S).log"
echo "Starting boxbot, logging to $LOG"

nohup setsid "$VENV_BOXBOT" > "$LOG" 2>&1 < /dev/null &
NEW_PID=$!
disown

# Give it a few seconds to either start cleanly or fail loudly
sleep 8

if ! kill -0 "$NEW_PID" 2>/dev/null; then
    echo "ERROR: boxbot died during startup. Last log lines:" >&2
    tail -30 "$LOG" >&2
    exit 1
fi

echo "boxbot started (pid $NEW_PID)"
echo ""
echo "=== last startup lines ==="
tail -25 "$LOG"
echo ""
echo "=== log path: $LOG ==="
