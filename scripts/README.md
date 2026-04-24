# scripts/

Setup and utility scripts for boxBot.

## Files

### `setup.sh`
Full system setup for a fresh Raspberry Pi. Idempotent — safe to re-run:
- System package installation (SDL2, audio libs, build tools, etc.)
- Data directory creation (`data/`, `data/displays/`, `logs/`, `models/`)
- Python virtual environment creation (`.venv/`)
- pip dependency installation (`pip install -e ".[dev]"`)
- Extra package installation: sounddevice, pyusb, sentence-transformers,
  openwakeword, elevenlabs, pyannote.audio (torchcodec removed — CUDA
  incompatible with Pi)
- System package symlinks into venv: libcamera, pykms, hailo_platform
  (Pi OS packages with no PyPI equivalent)
- Config file initialization from templates (won't overwrite existing)
- Calls `setup-sandbox.sh` to create the sandbox environment
- Offers to run `harden-os.sh` for OS-level network hardening

Run as your normal user (not root): `./scripts/setup.sh`

### `harden-os.sh`
OS-level network hardening. Standalone and idempotent — can be run
independently of `setup.sh` and is safe to re-run:
- Installs and configures UFW firewall (deny all inbound, allow SSH +
  webhook port)
- Hardens SSH: key-only auth, no root login, idle timeout
- Lockout protection: refuses to disable password auth if no SSH keys found
- Enables automatic security updates (Debian security patches only)
- Disables unnecessary services (bluetooth, avahi-daemon)
- See `docs/os-hardening.md` for full rationale and revert instructions

### `setup-sandbox.sh`
Sandbox environment setup. Creates the isolated execution environment
for agent-written scripts. Idempotent — safe to re-run:
- Creates `boxbot-sandbox` system user (no login, no home, no shell)
- Creates `boxbot` group, adds both users
- Creates sandbox venv at `data/sandbox/venv` (owned by main user)
- Installs sandbox packages from `config/sandbox-requirements.txt`
- Installs `boxbot_sdk` from `src/boxbot/sdk/` (non-editable, copied
  into site-packages for security)
- Pre-compiles bytecode (sandbox can't write to `__pycache__/`)
- Sets filesystem permissions (10 verification tests confirm isolation):
  - `data/sandbox/venv/` → owned by main user, read+execute for sandbox
  - `data/sandbox/venv/bin/python3` → mode `0750` (sandbox can execute)
  - `data/sandbox/venv/bin/pip` → mode `0700` (sandbox cannot execute)
  - `data/sandbox/output/`, `tmp/`, `scripts/` → owned by `boxbot-sandbox`
  - `skills/` → group-writable (both users in `boxbot` group)
  - `.env` → mode `0600` (sandbox cannot read secrets)
  - `src/`, `.git/` → owner-only (sandbox has no access)
- Installs seccomp profile (`config/seccomp-sandbox.json`) — blocks
  `execve`, `fork`, `vfork`, `clone` (except `CLONE_THREAD`), `clone3`
- Runs 10 automated verification tests to confirm sandbox isolation

Run as your normal user (not root): `./scripts/setup-sandbox.sh`

### `install-hailo.sh`
Hailo-specific setup:
- HailoRT library installation
- Model download (YOLOv8n, OSNet)
- Model compilation to HEF format for Hailo-8L
- Verification test

### `download-models.sh`
Download pre-trained model files:
- YOLOv8n (person detection) — ONNX and HEF
- OSNet (visual ReID) — ONNX and HEF
- ECAPA-TDNN (speaker embedding) — ONNX
- Silero VAD — ONNX
- openWakeWord — custom wake word model

### `enroll-user.sh`
Interactive user enrollment helper:
- Guides a new user through the enrollment process
- Captures visual and audio embeddings
- Adds to person database and optionally to WhatsApp whitelist

### `preview_display.py`
Render a display spec (JSON or YAML) to a PNG. Runs headless on any
machine — no HDMI or pygame needed. Uses the same `DisplayRenderer` as
the live system so previews are pixel-accurate. Fills missing data
sources with placeholder data.

```bash
# Render a built-in spec in the default theme
python3 scripts/preview_display.py --builtin clock -o /tmp/clock.png

# Render a user spec with a theme override
python3 scripts/preview_display.py displays/morning_brief/display.json \
    -t midnight -o data/previews/morning_brief_midnight.png

# List available built-ins and themes
python3 scripts/preview_display.py --list
```

### `block_gallery.py`
Render every block type (24 total) across every theme (4 total). Emits
per-block 1024x600 PNGs for close inspection and a composite contact
sheet per theme for at-a-glance review. Acts as a visual regression
baseline — rerun after changes to themes, blocks, or the renderer and
diff the output.

```bash
python3 scripts/block_gallery.py
# outputs → data/previews/gallery/
```

### `show_on_screen.py`
Display a PNG (or live-render a spec) fullscreen on the 7" HDMI screen.
Designed to run on the Pi so you can eyeball real renders during
iteration. Tries SDL drivers in order (kmsdrm, fbcon, x11, wayland) so
it works over SSH against a framebuffer console as well as under a
desktop session.

```bash
# From dev machine: render locally, copy, and show
scripts/preview_display.py displays/my_display/display.json -o /tmp/d.png
scp /tmp/d.png jhart@192.168.0.10:/tmp/d.png
ssh jhart@192.168.0.10 "python3 ~/boxBot/scripts/show_on_screen.py /tmp/d.png"

# Or render in-place on the Pi (matches live fonts/paths)
ssh jhart@192.168.0.10 "python3 ~/boxBot/scripts/show_on_screen.py \
    --spec ~/boxBot/displays/morning_brief/display.json -t boxbot"
```
Controls: `q`/`Esc` quit, `r` re-render, `1`-`4` swap themes
(boxbot/midnight/daylight/classic).
