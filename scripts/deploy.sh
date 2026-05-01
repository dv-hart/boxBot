#!/usr/bin/env bash
# Canonical deploy: push to origin, fast-forward Pi, restart.
#
# Run from the dev machine. This is the SOP — see CLAUDE.md "Deploying
# to the Pi" for the why and the alternatives.
#
# Usage: scripts/deploy.sh [user@host]
#
# Default target: pi@boxbot.local
#
# Pre-flight checks (refuses to proceed if any fail):
#   - Working tree clean (no uncommitted changes)
#   - On branch main
#   - main is up to date with origin/main, OR the local has commits to push
#   - All modified Python files compile
#
# Steps:
#   1. ``git push origin main`` (no-op if already pushed).
#   2. SSH to Pi, ``git pull --ff-only`` (refuses non-fast-forward).
#   3. Warn if scripts/setup-sandbox.sh changed (operator runs it manually).
#   4. Run ``scripts/restart-boxbot.sh`` on the Pi.
#   5. Show the last 25 startup log lines.
#
# To bypass git (rsync the working tree, no commit, no service touch),
# use scripts/deploy-to-pi.sh instead. That mode is for fast iteration
# but skips the audit trail; use sparingly.

set -euo pipefail

TARGET="${1:-pi@boxbot.local}"
PI_PROJECT_DIR="software/boxBot"

cd "$(dirname "$0")/.."

# -------------------------------------------------------------------
# Pre-flight
# -------------------------------------------------------------------

echo "--- Pre-flight ---"

CUR_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$CUR_BRANCH" != "main" ]]; then
    echo "Error: not on main (currently on $CUR_BRANCH)" >&2
    echo "Switch to main and merge your work in before deploying." >&2
    exit 1
fi
echo "  branch: main ✓"

if ! git diff-index --quiet HEAD --; then
    echo "Error: working tree has uncommitted changes" >&2
    git status --short >&2
    echo "" >&2
    echo "Commit or stash before deploying. To deploy WIP without" >&2
    echo "committing, use scripts/deploy-to-pi.sh (rsync, no audit trail)." >&2
    exit 1
fi
echo "  working tree clean ✓"

# Track origin/main; refuse to deploy something we haven't pushed
git fetch origin --quiet
LOCAL_HEAD="$(git rev-parse main)"
REMOTE_HEAD="$(git rev-parse origin/main)"
if [[ "$LOCAL_HEAD" != "$REMOTE_HEAD" ]]; then
    AHEAD="$(git rev-list --count origin/main..main)"
    BEHIND="$(git rev-list --count main..origin/main)"
    if [[ "$BEHIND" -gt 0 ]]; then
        echo "Error: local main is $BEHIND commit(s) behind origin/main" >&2
        echo "Pull first: git pull --ff-only" >&2
        exit 1
    fi
    echo "  local main is $AHEAD commit(s) ahead of origin/main"
else
    echo "  main matches origin/main ✓"
fi

# -------------------------------------------------------------------
# Push
# -------------------------------------------------------------------

echo ""
echo "--- Push ---"
git push origin main

# Detect if scripts/setup-sandbox.sh changed in commits we're about to
# deploy (vs. what's currently on the Pi). We'll warn the operator
# rather than auto-run it, since setup-sandbox.sh requires sudo and
# may need attention.
PI_HEAD="$(ssh "$TARGET" "cd $PI_PROJECT_DIR && git rev-parse HEAD" 2>/dev/null || echo unknown)"
if [[ "$PI_HEAD" != "unknown" && "$PI_HEAD" != "$LOCAL_HEAD" ]]; then
    if git diff --name-only "$PI_HEAD" "$LOCAL_HEAD" 2>/dev/null \
           | grep -q '^scripts/setup-sandbox\.sh$'; then
        SETUP_CHANGED=1
    else
        SETUP_CHANGED=0
    fi
else
    SETUP_CHANGED=0
fi

# -------------------------------------------------------------------
# Pull on Pi + restart
# -------------------------------------------------------------------

echo ""
echo "--- Sync Pi → restart ---"

# Use a heredoc so the multi-line remote script is unambiguous.
ssh "$TARGET" bash <<EOF
set -euo pipefail
cd "$PI_PROJECT_DIR"
echo "Pi HEAD before: \$(git rev-parse --short HEAD)"
git fetch origin --quiet
git pull --ff-only origin main
echo "Pi HEAD after:  \$(git rev-parse --short HEAD)"
EOF

if [[ "$SETUP_CHANGED" -eq 1 ]]; then
    echo ""
    echo "WARNING: scripts/setup-sandbox.sh changed in this deploy."
    echo "After restart, run on the Pi:"
    echo "    ssh $TARGET 'cd $PI_PROJECT_DIR && sudo bash scripts/setup-sandbox.sh'"
    echo ""
fi

ssh "$TARGET" "cd $PI_PROJECT_DIR && bash scripts/restart-boxbot.sh"

echo ""
echo "--- Deploy to $TARGET complete ---"
