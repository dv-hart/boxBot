#!/bin/bash
# Render a display to PNG, sync to the Pi, and show it fullscreen on the HDMI
# screen. Intended for tight iteration during display design work.
#
# Usage:
#   scripts/render_and_show.sh <spec-path-or-builtin> [theme]
#   scripts/render_and_show.sh displays/morning_brief/display.json boxbot
#   scripts/render_and_show.sh morning_brief        # shorthand for displays/<name>/display.json
#   scripts/render_and_show.sh --builtin clock

set -euo pipefail

PI_HOST="${PI_HOST:-jhart@192.168.0.10}"
TARGET=""
THEME=""
LIVE_ARG=""
for arg in "$@"; do
    case "$arg" in
        --live) LIVE_ARG="--live" ;;
        *)
            if [[ -z "$TARGET" ]]; then
                TARGET="$arg"
            elif [[ -z "$THEME" ]]; then
                THEME="$arg"
            fi
            ;;
    esac
done

if [[ -z "$TARGET" ]]; then
    echo "usage: $0 <spec-path-or-name> [theme] [--live]" >&2
    exit 2
fi

# Resolve shorthand: "morning_brief" → "displays/morning_brief/display.json"
if [[ "$TARGET" != "--builtin" ]] && [[ ! -f "$TARGET" ]] && [[ -f "displays/${TARGET}/display.json" ]]; then
    SPEC_ARG="displays/${TARGET}/display.json"
    LABEL="$TARGET"
elif [[ "$TARGET" == "--builtin" ]]; then
    SPEC_ARG="--builtin ${2:-clock}"
    LABEL="${2:-clock}"
    THEME=""  # can't theme-override builtins here
else
    SPEC_ARG="$TARGET"
    LABEL=$(basename "$TARGET" .json | sed 's|displays/||')
fi

THEME_SUFFIX=""
THEME_ARG=""
if [[ -n "$THEME" ]]; then
    THEME_ARG="-t $THEME"
    THEME_SUFFIX="_${THEME}"
fi

OUT="/tmp/preview_${LABEL}${THEME_SUFFIX}.png"

echo "=> rendering locally: $SPEC_ARG $THEME_ARG $LIVE_ARG"
PYTHONPATH=src python3 scripts/preview_display.py $SPEC_ARG $THEME_ARG $LIVE_ARG -o "$OUT"

echo "=> copying to Pi"
scp -q "$OUT" "$PI_HOST:/tmp/$(basename "$OUT")"

echo "=> swapping display on Pi"
ssh "$PI_HOST" "bash ~/software/boxBot/scripts/pi_show.sh /tmp/$(basename "$OUT")"
echo "=> done — check the box"
