#!/usr/bin/env bash
# Restart the boxBot service. Designed to run ON THE PI — typically
# invoked by ``scripts/deploy.sh`` via SSH after a ``git pull``, but can
# also be run manually for a quick bounce.
#
# What it does:
#   - If boxbot is installed as a systemd unit (boxbot.service), restart
#     it via ``systemctl restart`` so the unit's PID tracking, memory
#     caps, and restart counter stay coherent. This is the normal case
#     on a provisioned Pi (scripts/systemd/install.sh).
#   - Otherwise, fall back to the legacy bounce: SIGTERM the running
#     process (SIGKILL after 15s), then spawn a fresh one via
#     ``nohup setsid`` with a dated log file.
#
# Idempotent: if no boxbot is running, just starts one.
#
# Exits non-zero if boxbot didn't survive startup.

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
# Preferred path: systemd-managed restart.
# -------------------------------------------------------------------
# When boxbot.service is installed, a pgrep SIGTERM + nohup respawn would
# orphan the process from systemd — the unit would show inactive while a
# rogue copy ran, Restart=on-failure won't relaunch after a clean
# SIGTERM, and the memory caps in the unit wouldn't apply. So defer to
# systemctl whenever the unit exists.
if command -v systemctl >/dev/null 2>&1 \
        && systemctl cat boxbot.service >/dev/null 2>&1; then
    echo "boxbot.service is installed — restarting via systemd"
    sudo systemctl restart boxbot.service
    # Give it a few seconds to either come up or fail loudly.
    sleep 5
    if ! systemctl is-active --quiet boxbot.service; then
        echo "ERROR: boxbot.service did not come up. Recent state + logs:" >&2
        sudo systemctl status boxbot.service --no-pager -l 2>&1 | tail -20 >&2
        journalctl -u boxbot.service -n 30 --no-pager >&2 2>/dev/null || true
        exit 1
    fi
    echo "boxbot.service active (pid $(systemctl show -p MainPID --value boxbot.service))"
    echo ""
    echo "=== recent app log ==="
    tail -25 logs/boxbot.log 2>/dev/null || echo "(no logs/boxbot.log yet)"
    exit 0
fi

# -------------------------------------------------------------------
# Fallback (no systemd unit): stop existing boxbot (if any)
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

# Cap glibc malloc arenas (default 8*ncores = 32 on the Pi 5). Without
# this, concurrent mixed-size allocation across the voice/perception/
# agent subsystems fragments the heap so freed memory is stranded in
# per-arena free lists and never returns to the OS — RSS climbs across
# conversations until OOM. 2 arenas bounds fragmentation; the in-process
# malloc_trim() (diagnostics.memory) hands back the rest. Mirrors the
# Environment= line in scripts/systemd/boxbot.service.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

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
