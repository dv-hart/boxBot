#!/usr/bin/env bash
# FAST-ITERATION DEPLOY (rsync, no commit, no restart).
#
# This is NOT the canonical deploy path — see ``scripts/deploy.sh`` and
# the "Deploying to the Pi" section of CLAUDE.md for the SOP. Use that
# for any deploy you want to be auditable.
#
# Use THIS script when:
#   - Iterating on a tight loop and don't want to commit each change.
#   - Testing a quick fix you may not keep.
#
# Trade-offs:
#   - No git audit trail. The Pi's tracked files diverge from its
#     committed state (uncommitted local modifications).
#   - Does NOT restart boxbot — must run scripts/restart-boxbot.sh on
#     the Pi yourself if needed.
#   - Skips the pre-flight checks in scripts/deploy.sh (clean tree, on
#     main, push to origin).
#
# Usage: scripts/deploy-to-pi.sh [user@host]
#
# Default target: pi@boxbot.local
#
# CRITICAL: excludes .env — that file lives ONLY on the Pi and must never be
# overwritten from a dev machine. If you need to push model overrides or
# other env changes, SSH in and edit ~/software/boxBot/.env directly.
#
# Also excludes .venv (Pi has its own venv with ARM wheels), data/ (runtime
# state, credentials, memory DB), logs/ (runtime logs), and standard dev
# noise (__pycache__, .git, *.pyc, .pytest_cache).

set -euo pipefail

TARGET="${1:-pi@boxbot.local}"

rsync -azh \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '.pytest_cache' \
    --exclude 'data' \
    --exclude 'logs' \
    --exclude '.git' \
    --exclude '*.pyc' \
    --exclude '.env' \
    --exclude '.env.local' \
    --exclude '.env.*.local' \
    /home/jhart/software/boxBot/ \
    "${TARGET}":software/boxBot/

echo "Deploy to ${TARGET} complete. .env on the Pi was NOT touched."
