#!/usr/bin/env bash
# setup.sh — Full system setup for a fresh Raspberry Pi
#
# Idempotent: safe to re-run. Each step checks current state before acting.
#
# What it does:
#   1. Installs system packages (SDL2, audio libs, build tools)
#   2. Creates data directories
#   3. Creates main Python virtual environment
#   4. Installs Python dependencies
#   5. Initializes config files from templates
#   6. Calls setup-sandbox.sh to create the sandbox environment
#   7. Offers to run harden-os.sh for OS hardening
#
# Usage:
#   ./scripts/setup.sh
#
# Do NOT run as root — the script uses sudo for steps that need it.

set -euo pipefail

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
DATA_DIR="$PROJECT_DIR/data"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=11

# -------------------------------------------------------------------
# Preflight
# -------------------------------------------------------------------

echo ""
echo "====================================="
echo "  boxBot Setup"
echo "====================================="
echo ""

if [[ $EUID -eq 0 ]]; then
    echo "Error: Do not run this script as root."
    echo "Run as your normal user: ./scripts/setup.sh"
    echo "The script will use sudo where needed."
    exit 1
fi

# Check Python version
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found. Install Python $MIN_PYTHON_MAJOR.$MIN_PYTHON_MINOR+."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if (( PYTHON_MAJOR < MIN_PYTHON_MAJOR || (PYTHON_MAJOR == MIN_PYTHON_MAJOR && PYTHON_MINOR < MIN_PYTHON_MINOR) )); then
    echo "Error: Python $MIN_PYTHON_MAJOR.$MIN_PYTHON_MINOR+ required (found $PYTHON_VERSION)."
    exit 1
fi

# Check architecture
ARCH=$(uname -m)
if [[ "$ARCH" != "aarch64" ]]; then
    echo "  Warning: Expected aarch64 (Raspberry Pi 5), found $ARCH."
    echo "  Hardware-specific packages (picamera2, RPi.GPIO) may fail to install."
    echo ""
    read -rp "  Continue anyway? [y/N]: " CONTINUE
    if [[ ! "$CONTINUE" =~ ^[Yy] ]]; then
        echo "  Aborted."
        exit 0
    fi
    echo ""
fi

echo "  Python:    $PYTHON_VERSION"
echo "  Platform:  $ARCH"
echo "  Project:   $PROJECT_DIR"
echo ""

CHANGES=()

# -------------------------------------------------------------------
# 1. System packages
# -------------------------------------------------------------------

echo "--- Installing system packages ---"

SYSTEM_PACKAGES=(
    # Build tools
    build-essential
    python3-dev
    python3-venv
    cmake
    git

    # SDL2 (pygame)
    libsdl2-dev
    libsdl2-mixer-dev
    libsdl2-image-dev
    libsdl2-ttf-dev

    # Audio (pyaudio, ALSA)
    portaudio19-dev
    libasound2-dev
    alsa-utils

    # Image processing (Pillow, OpenCV)
    libjpeg-dev
    libpng-dev
    zlib1g-dev
    libopenjp2-7-dev
    libtiff-dev

    # XML (lxml)
    libxml2-dev
    libxslt-dev

    # Math / ML (numpy, faiss)
    libopenblas-dev

    # HDF5 (h5py, model loading)
    libhdf5-dev

    # Audio / video processing
    ffmpeg

    # Sandbox security
    libseccomp-dev
    libseccomp2

    # Misc
    libffi-dev
    curl
)

PACKAGES_TO_INSTALL=()
for pkg in "${SYSTEM_PACKAGES[@]}"; do
    if ! dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
        PACKAGES_TO_INSTALL+=("$pkg")
    fi
done

if [[ ${#PACKAGES_TO_INSTALL[@]} -gt 0 ]]; then
    echo "Installing ${#PACKAGES_TO_INSTALL[@]} packages..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq "${PACKAGES_TO_INSTALL[@]}"
    CHANGES+=("Installed ${#PACKAGES_TO_INSTALL[@]} system packages")
else
    echo "All system packages already installed."
fi

# -------------------------------------------------------------------
# 2. Create data directories
# -------------------------------------------------------------------

echo ""
echo "--- Creating data directories ---"

DIRECTORIES=(
    "$DATA_DIR/sandbox/output"
    "$DATA_DIR/sandbox/tmp"
    "$DATA_DIR/sandbox/scripts"
    "$DATA_DIR/scheduler"
    "$DATA_DIR/memory"
    "$DATA_DIR/photos"
    "$DATA_DIR/perception/crops"
    "$DATA_DIR/perception/models"
    "$PROJECT_DIR/logs"
    "$PROJECT_DIR/models/wake_word"
)

DIRS_CREATED=0
for dir in "${DIRECTORIES[@]}"; do
    if [[ ! -d "$dir" ]]; then
        mkdir -p "$dir"
        ((DIRS_CREATED++))
    fi
done

if [[ $DIRS_CREATED -gt 0 ]]; then
    echo "Created $DIRS_CREATED directories."
    CHANGES+=("Created $DIRS_CREATED data directories")
else
    echo "All directories already exist."
fi

# -------------------------------------------------------------------
# 3. Create main virtual environment
# -------------------------------------------------------------------

echo ""
echo "--- Setting up Python virtual environment ---"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating venv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    CHANGES+=("Created main venv at .venv/")
else
    echo "Venv already exists at $VENV_DIR."
fi

# Upgrade pip
echo "Upgrading pip..."
"$VENV_DIR/bin/pip" install --upgrade pip --quiet

# -------------------------------------------------------------------
# 4. Install Python dependencies
# -------------------------------------------------------------------

echo ""
echo "--- Installing Python dependencies ---"

"$VENV_DIR/bin/pip" install -e "$PROJECT_DIR[dev]" --quiet
CHANGES+=("Installed Python dependencies")

echo "Dependencies installed."

# -------------------------------------------------------------------
# 5. Initialize config files
# -------------------------------------------------------------------

echo ""
echo "--- Initializing configuration ---"

CONFIG_COPIED=0

copy_template() {
    local src="$1"
    local dest="$2"
    local name
    name=$(basename "$dest")

    if [[ ! -f "$dest" ]]; then
        if [[ -f "$src" ]]; then
            cp "$src" "$dest"
            echo "  Created $name from template."
            ((CONFIG_COPIED++))
        else
            echo "  Warning: Template $src not found, skipping $name."
        fi
    else
        echo "  $name already exists (not overwritten)."
    fi
}

copy_template "$PROJECT_DIR/config/config.example.yaml" "$PROJECT_DIR/config/config.yaml"
copy_template "$PROJECT_DIR/config/whatsapp.example.yaml" "$PROJECT_DIR/config/whatsapp.yaml"
copy_template "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"

# Secure .env permissions (sandbox cannot read secrets)
if [[ -f "$PROJECT_DIR/.env" ]]; then
    chmod 600 "$PROJECT_DIR/.env"
fi

if [[ $CONFIG_COPIED -gt 0 ]]; then
    CHANGES+=("Initialized $CONFIG_COPIED config file(s) from templates")
fi

# Initialize empty system memory file
if [[ ! -f "$DATA_DIR/memory/system.md" ]]; then
    cat > "$DATA_DIR/memory/system.md" << 'EOF'
# System Memory

<!-- This file is automatically maintained by boxBot. -->
<!-- Manual edits are allowed but may be overwritten. -->
EOF
    echo "  Created empty system memory file."
    CHANGES+=("Created data/memory/system.md")
else
    echo "  System memory file already exists."
fi

# -------------------------------------------------------------------
# 6. Set up sandbox environment
# -------------------------------------------------------------------

echo ""
echo "--- Setting up sandbox environment ---"
echo ""

if [[ -f "$SCRIPT_DIR/setup-sandbox.sh" ]]; then
    bash "$SCRIPT_DIR/setup-sandbox.sh"
    CHANGES+=("Ran setup-sandbox.sh")
else
    echo "Warning: setup-sandbox.sh not found."
    echo "Sandbox must be set up manually."
fi

# -------------------------------------------------------------------
# 7. Offer OS hardening
# -------------------------------------------------------------------

echo ""
echo "--- OS Hardening ---"

if [[ -f "$SCRIPT_DIR/harden-os.sh" ]]; then
    echo "The hardening script configures the firewall, hardens SSH, and"
    echo "enables automatic security updates."
    echo ""
    read -rp "Run OS hardening now? [y/N]: " RUN_HARDEN
    if [[ "$RUN_HARDEN" =~ ^[Yy] ]]; then
        sudo bash "$SCRIPT_DIR/harden-os.sh"
        CHANGES+=("Ran harden-os.sh")
    else
        echo "Skipped. Run later with: sudo ./scripts/harden-os.sh"
    fi
else
    echo "harden-os.sh not found. Skipping."
fi

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------

echo ""
echo "====================================="
echo "  Setup Complete"
echo "====================================="
echo ""

if [[ ${#CHANGES[@]} -eq 0 ]]; then
    echo "  No changes needed — everything was already configured."
else
    for change in "${CHANGES[@]}"; do
        echo "  ✓ $change"
    done
fi

echo ""
echo "  Next steps:"
echo "    1. Edit .env with your API keys (ANTHROPIC_API_KEY, etc.)"
echo "    2. Edit config/config.yaml for your preferences"
echo "    3. Activate the venv:  source .venv/bin/activate"
echo "    4. Run boxBot:         python3 -m boxbot"
echo ""
echo "  Hardware setup (run when hardware is connected):"
echo "    - Audio (ALSA dual-output):  TBD — see docs/hal.md"
echo "    - Hailo AI HAT+:            ./scripts/install-hailo.sh"
echo "    - ML models:                ./scripts/download-models.sh"
echo ""
