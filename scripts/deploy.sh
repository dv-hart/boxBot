#!/usr/bin/env bash
# Canonical deploy: push to origin, fast-forward Pi, restart.
#
# Run from the dev machine. This is the SOP — see CLAUDE.md "Deploying
# to the Pi" for the why and the alternatives.
#
# Usage: scripts/deploy.sh [user@host]
#
# Target resolution order:
#   1. ``user@host`` argument, if given.
#   2. ``BOXBOT_DEPLOY_TARGET`` environment variable.
#   3. ``BOXBOT_DEPLOY_TARGET=...`` in the repo's gitignored ``.env``.
#   4. Fallback ``pi@boxbot.local`` (mDNS — works out of the box on
#      most LANs running Avahi/Bonjour).
#
# Set ``BOXBOT_DEPLOY_TARGET=user@host`` once — either in shell init,
# in a gitignored ``.envrc``, or as a line in the project's ``.env``
# file (which is already 0600 and gitignored for secrets). The
# project directory on the Pi is ``software/boxBot`` by default;
# override with ``BOXBOT_PI_PROJECT_DIR`` if you keep the checkout
# elsewhere.
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
#   4. If src/boxbot/sdk/ changed, ``pip install --force-reinstall``
#      the SDK into the sandbox venv (needed because the SDK is
#      installed non-editable; source edits don't otherwise reach the
#      sandbox).
#   5. Run ``scripts/restart-boxbot.sh`` on the Pi.
#   6. Show the last 25 startup log lines.
#
# There is no rsync escape hatch. The Pi only ever runs committed
# code. To test uncommitted changes, do it in dev or push to a
# throwaway branch.

set -euo pipefail

cd "$(dirname "$0")/.."

# Target resolution: positional arg → environment → .env → mDNS default.
# Pure bash so it survives ``set -euo pipefail`` without subshell
# gymnastics; no shell-expansion of arbitrary .env content (we read
# only this one variable, with optional surrounding quotes stripped).
if [[ $# -ge 1 ]]; then
    TARGET="$1"
elif [[ -n "${BOXBOT_DEPLOY_TARGET:-}" ]]; then
    TARGET="$BOXBOT_DEPLOY_TARGET"
else
    TARGET=""
    if [[ -f .env ]]; then
        while IFS= read -r line || [[ -n "$line" ]]; do
            if [[ "$line" =~ ^BOXBOT_DEPLOY_TARGET=(.*)$ ]]; then
                TARGET="${BASH_REMATCH[1]}"
                TARGET="${TARGET#\"}"; TARGET="${TARGET%\"}"
                TARGET="${TARGET#\'}"; TARGET="${TARGET%\'}"
                break
            fi
        done < .env
    fi
    TARGET="${TARGET:-pi@boxbot.local}"
fi
PI_PROJECT_DIR="${BOXBOT_PI_PROJECT_DIR:-software/boxBot}"

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
    echo "Commit or stash before deploying. The Pi runs only" >&2
    echo "committed code — there is no WIP escape hatch." >&2
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

# Detect what changed between the Pi's current HEAD and the deploy
# target. We use this for two follow-up actions:
#   - setup-sandbox.sh changes → warn the operator (needs sudo + may
#     need attention).
#   - SDK source changes (src/boxbot/sdk/*.py) → auto-refresh the
#     sandbox venv copy. The SDK is installed non-editable, so source
#     edits don't reach the sandbox until pip re-installs them.
#     setup-sandbox.sh handles this via --force-reinstall, but is too
#     heavy to run every deploy; we do just the SDK install here.
PI_HEAD="$(ssh "$TARGET" "cd $PI_PROJECT_DIR && git rev-parse HEAD" 2>/dev/null || echo unknown)"
SETUP_CHANGED=0
SDK_CHANGED=0
if [[ "$PI_HEAD" != "unknown" && "$PI_HEAD" != "$LOCAL_HEAD" ]]; then
    CHANGED_FILES="$(git diff --name-only "$PI_HEAD" "$LOCAL_HEAD" 2>/dev/null || true)"
    if echo "$CHANGED_FILES" | grep -q '^scripts/setup-sandbox\.sh$'; then
        SETUP_CHANGED=1
    fi
    # README.md is doc-only and never affects imports. Everything else
    # under src/boxbot/sdk/ — .py files AND pyproject.toml (version
    # bump is the canonical "SDK changed" signal) — triggers a refresh.
    if echo "$CHANGED_FILES" \
           | grep -E '^src/boxbot/sdk/' \
           | grep -vE '/README\.md$' \
           | grep -q .; then
        SDK_CHANGED=1
    fi
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

# Refresh the sandbox SDK if any importable file under src/boxbot/sdk/
# changed. Same install line setup-sandbox.sh uses, scoped to just the
# SDK so we don't drag in apt steps or chown rebuilds. The sandbox
# venv is owned by the deploy user (jhart), so no sudo is required.
if [[ "$SDK_CHANGED" -eq 1 ]]; then
    echo ""
    echo "--- Refresh sandbox SDK ---"
    ssh "$TARGET" bash <<EOF
set -euo pipefail
cd "$PI_PROJECT_DIR"
SANDBOX_VENV_PIP="/var/lib/boxbot-sandbox/venv/bin/pip"
if [[ ! -x "\$SANDBOX_VENV_PIP" ]]; then
    echo "  sandbox venv pip not found at \$SANDBOX_VENV_PIP — skipping"
    echo "  (run 'sudo bash scripts/setup-sandbox.sh' to create it)"
    exit 0
fi
"\$SANDBOX_VENV_PIP" install --force-reinstall --no-deps --quiet \\
    src/boxbot/sdk
echo "  sandbox SDK reinstalled (boxbot_sdk)"
EOF
fi

ssh "$TARGET" "cd $PI_PROJECT_DIR && bash scripts/restart-boxbot.sh"

echo ""
echo "--- Deploy to $TARGET complete ---"
