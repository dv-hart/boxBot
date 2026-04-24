#!/usr/bin/env bash
# Deploy the working tree to the Pi safely.
#
# Usage: scripts/deploy-to-pi.sh [user@host]
#
# Default target: jhart@192.168.0.10
#
# CRITICAL: excludes .env — that file lives ONLY on the Pi and must never be
# overwritten from a dev machine. If you need to push model overrides or
# other env changes, SSH in and edit ~/software/boxBot/.env directly.
#
# Also excludes .venv (Pi has its own venv with ARM wheels), data/ (runtime
# state, credentials, memory DB), logs/ (runtime logs), and standard dev
# noise (__pycache__, .git, *.pyc, .pytest_cache).

set -euo pipefail

TARGET="${1:-jhart@192.168.0.10}"

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
