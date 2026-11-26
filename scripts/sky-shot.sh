#!/bin/bash
# Render Living Sky fixtures to PNG with headless Chrome (dev only).
#   scripts/sky-shot.sh OUTDIR "name|fixture=home&phase=night&weather=rain" ...
# Each pair is "<output name>|<query string for sky/index.html>". Writes OUTDIR/<name>.png at 1024x600.
#
# At most three Chromes run at once, machine-wide (flock slots): an unthrottled batch of ~50 once ran the
# dev VM out of memory. Headless Chrome also loses ~87px of window height, so each shot is rendered
# 1024x720 and cropped.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PAGE="file://$ROOT/src/boxbot/displays/sky/index.html"
OUT="$1"; shift
mkdir -p "$OUT"
CHROME="$(command -v google-chrome || command -v chromium || command -v chromium-browser)"
SLOTS=${SKY_SHOT_SLOTS:-3}

shot() {  # $1 = png path, $2 = url
  while :; do
    for i in $(seq 1 "$SLOTS"); do
      exec 9>"/tmp/boxbot-chrome-shot.$i.lock"
      if flock -n 9; then
        timeout 90 "$CHROME" --headless=new --no-sandbox --hide-scrollbars --disable-gpu \
          --disable-dev-shm-usage --virtual-time-budget=4000 --window-size=1024,720 \
          --screenshot="$1" "$2" >/dev/null 2>&1
        return $?
      fi
      exec 9>&-
    done
    sleep 0.4
  done
}

NAMES=()
for pair in "$@"; do
  name="${pair%%|*}"; query="${pair#*|}"; NAMES+=("$name")
  ( shot "$OUT/$name.png" "$PAGE?$query" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do wait -n; done
done
wait
python3 - "$OUT" "${NAMES[@]}" << 'PY'
import sys
from PIL import Image
out, names = sys.argv[1], sys.argv[2:]
for n in names:
    p = f"{out}/{n}.png"
    try:
        Image.open(p).crop((0, 0, 1024, 600)).save(p)
    except OSError as e:
        print(f"missing: {p} ({e})", file=sys.stderr)
PY
