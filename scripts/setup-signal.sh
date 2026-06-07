#!/usr/bin/env bash
# setup-signal.sh — Install signal-cli daemon as a systemd service on the Pi.
#
# Idempotent: safe to re-run. Each step checks current state before acting.
#
# Prerequisites (NOT installed by this script — see memory/signal-cli-setup.md
# in the auto-memory store):
#   1. signal-cli at /usr/local/bin/signal-cli (download + extract from
#      https://github.com/AsamK/signal-cli/releases)
#   2. Java 25 (Debian Trixie: openjdk-25-jre-headless)
#   3. libsignal_jni.so for aarch64 at /usr/lib/aarch64-linux-gnu/jni/
#      (from https://github.com/exquo/signal-libs-build — must match
#      signal-cli's bundled libsignal-client version)
#   4. The account is already registered: `signal-cli -a +X… register/verify`
#
# This script:
#   1. Verifies the prerequisites
#   2. Resolves the account number (env, /etc/default file, or prompt)
#   3. Writes /etc/default/boxbot-signal
#   4. Installs scripts/systemd/signal-cli.service to systemd
#   5. Enables and starts the service
#   6. Verifies the unix socket appears and JSON-RPC responds
#
# Usage:
#   ./scripts/setup-signal.sh
#   SIGNAL_ACCOUNT=+15551234567 ./scripts/setup-signal.sh   # non-interactive
#
# Do NOT run as root — the script uses sudo where needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REAL_USER="$(whoami)"

echo ""
echo "====================================="
echo "  boxBot signal-cli Daemon Setup"
echo "====================================="
echo ""

if [[ $EUID -eq 0 ]]; then
    echo "Error: Do not run as root. Run as your normal user." >&2
    echo "The script will use sudo where needed." >&2
    exit 1
fi

# -------------------------------------------------------------------
# 1. Prerequisites
# -------------------------------------------------------------------

echo "--- Verifying prerequisites ---"

if ! command -v signal-cli >/dev/null 2>&1; then
    echo "Error: signal-cli not found on PATH." >&2
    echo "Install per memory/signal-cli-setup.md before running this." >&2
    exit 1
fi
SIGNAL_VERSION="$(signal-cli --version 2>&1 || true)"
echo "  signal-cli: $SIGNAL_VERSION"

if ! command -v java >/dev/null 2>&1; then
    echo "Error: java not found on PATH." >&2
    echo "Install: sudo apt install openjdk-25-jre-headless" >&2
    exit 1
fi
JAVA_VERSION="$(java -version 2>&1 | head -1)"
echo "  java: $JAVA_VERSION"

# -------------------------------------------------------------------
# 2. Resolve account number
# -------------------------------------------------------------------

SIGNAL_ACCOUNT="${SIGNAL_ACCOUNT:-}"
if [[ -z "$SIGNAL_ACCOUNT" && -f /etc/default/boxbot-signal ]]; then
    SIGNAL_ACCOUNT="$(
        grep -E '^SIGNAL_ACCOUNT=' /etc/default/boxbot-signal \
            | head -1 | cut -d= -f2-
    )"
    SIGNAL_ACCOUNT="${SIGNAL_ACCOUNT%\"}"; SIGNAL_ACCOUNT="${SIGNAL_ACCOUNT#\"}"
    SIGNAL_ACCOUNT="${SIGNAL_ACCOUNT%\'}"; SIGNAL_ACCOUNT="${SIGNAL_ACCOUNT#\'}"
fi
if [[ -z "$SIGNAL_ACCOUNT" ]]; then
    echo ""
    echo "Enter the registered Signal account number (E.164, e.g. +15039858519)"
    echo "Must already be registered via 'signal-cli -a <number> register/verify'."
    read -r -p "  SIGNAL_ACCOUNT: " SIGNAL_ACCOUNT
fi
if [[ ! "$SIGNAL_ACCOUNT" =~ ^\+[0-9]+$ ]]; then
    echo "Error: SIGNAL_ACCOUNT must be E.164 format (e.g. +15039858519)" >&2
    exit 1
fi

# Verify the account works (probes signal-cli's local state for that number)
if ! signal-cli -a "$SIGNAL_ACCOUNT" listDevices >/dev/null 2>&1; then
    echo "Error: 'signal-cli -a $SIGNAL_ACCOUNT listDevices' failed." >&2
    echo "Either the account isn't registered, libsignal native lib is" >&2
    echo "missing for this arch, or the account data dir isn't owned by" >&2
    echo "$REAL_USER (signal-cli stores state under ~/.local/share/signal-cli/)." >&2
    exit 1
fi
echo "  account: $SIGNAL_ACCOUNT (registered, listDevices OK)"

# -------------------------------------------------------------------
# 3. Write env file
# -------------------------------------------------------------------

echo ""
echo "--- Writing /etc/default/boxbot-signal ---"

# Account number is NOT a secret — it's the public phone number. 0644
# so systemd can read it without privileged file capabilities.
sudo install -m 0644 /dev/stdin /etc/default/boxbot-signal <<EOF
# Managed by scripts/setup-signal.sh — re-run to change account.
SIGNAL_ACCOUNT=$SIGNAL_ACCOUNT
EOF
echo "  wrote /etc/default/boxbot-signal"

# -------------------------------------------------------------------
# 4. Install systemd unit
# -------------------------------------------------------------------

echo ""
echo "--- Installing systemd unit ---"

UNIT_SRC="$SCRIPT_DIR/systemd/signal-cli.service"
UNIT_DST="/etc/systemd/system/signal-cli.service"

if [[ ! -f "$UNIT_SRC" ]]; then
    echo "Error: $UNIT_SRC not found." >&2
    exit 1
fi

sudo install -m 0644 "$UNIT_SRC" "$UNIT_DST"
echo "  installed $UNIT_DST"

sudo systemctl daemon-reload

# -------------------------------------------------------------------
# 5. Enable + start
# -------------------------------------------------------------------

echo ""
echo "--- Enabling + starting signal-cli.service ---"

sudo systemctl enable --now signal-cli.service

# -------------------------------------------------------------------
# 6. Verify socket + JSON-RPC
# -------------------------------------------------------------------

echo ""
echo "--- Verifying daemon ---"

SOCKET=/run/signal-cli/socket
for _ in 1 2 3 4 5 6 7 8 9 10; do
    if [[ -S "$SOCKET" ]]; then
        break
    fi
    sleep 1
done
if [[ ! -S "$SOCKET" ]]; then
    echo "Error: $SOCKET did not appear within 10s." >&2
    echo "Check: journalctl -u signal-cli.service --no-pager -n 50" >&2
    exit 1
fi
SOCKET_PERMS=$(stat -c '%U:%G %a' "$SOCKET")
echo "  socket: $SOCKET ($SOCKET_PERMS)"

# JSON-RPC reachability — send a version request via Python (stdlib only,
# no socat dependency).
RPC_OUT="$(python3 - "$SOCKET" <<'PY'
import json, socket, sys
sock_path = sys.argv[1]
try:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(5.0)
    s.connect(sock_path)
    req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "version"}) + "\n"
    s.sendall(req.encode())
    buf = b""
    while b"\n" not in buf:
        chunk = s.recv(4096)
        if not chunk:
            break
        buf += chunk
    line = buf.split(b"\n", 1)[0]
    resp = json.loads(line.decode())
    result = resp.get("result", {})
    print("OK", result.get("version", "(unknown version)"))
except Exception as e:
    print("ERR", e)
PY
)"
case "$RPC_OUT" in
    OK*) echo "  JSON-RPC: $RPC_OUT" ;;
    *)
        echo "  WARNING: JSON-RPC probe failed: $RPC_OUT"
        echo "  Daemon is running but the JSON-RPC interface didn't respond."
        echo "  Check: journalctl -u signal-cli.service --no-pager -n 50"
        ;;
esac

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------

echo ""
echo "====================================="
echo "  signal-cli Setup Complete"
echo "====================================="
echo ""
echo "  Account:  $SIGNAL_ACCOUNT"
echo "  Socket:   $SOCKET"
echo "  Status:   systemctl status signal-cli.service"
echo "  Logs:     journalctl -u signal-cli.service -f"
echo ""
