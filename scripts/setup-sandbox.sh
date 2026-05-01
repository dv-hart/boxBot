#!/usr/bin/env bash
# setup-sandbox.sh — Create the isolated sandbox environment
#
# Idempotent: safe to re-run. Each step checks current state before acting.
#
# What it does:
#   1. Creates boxbot-sandbox system user (no login, no home, no shell)
#   2. Creates boxbot group and adds both users
#   3. Creates sandbox venv at $SANDBOX_DIR/venv
#   4. Installs sandbox packages + boxbot_sdk
#   5. Pre-compiles bytecode for performance
#   6. Sets filesystem permissions (OS-level sandbox enforcement)
#   7. Installs seccomp profile for subprocess blocking
#   8. Verifies sandbox isolation
#
# Sandbox location:
#   By default the sandbox lives at /var/lib/boxbot-sandbox/. This is
#   *outside* the project tree on purpose: an open-source project can be
#   cloned into any user's home directory, and the boxbot-sandbox system
#   user can't necessarily traverse a 0700 home directory just to reach
#   the venv. /var/lib is the standard Linux location for application
#   state and is reachable by all system users.
#
#   Override with the BOXBOT_SANDBOX_DIR environment variable, e.g.
#       BOXBOT_SANDBOX_DIR=/opt/boxbot-sandbox ./scripts/setup-sandbox.sh
#   Keep the override in sync with the ``sandbox.runtime_dir`` field in
#   config.yaml.
#
# Usage:
#   Called by setup.sh, or run standalone:
#   ./scripts/setup-sandbox.sh
#
# Do NOT run as root — the script uses sudo where needed.

set -euo pipefail

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SANDBOX_DIR="${BOXBOT_SANDBOX_DIR:-/var/lib/boxbot-sandbox}"
SANDBOX_VENV="$SANDBOX_DIR/venv"
SANDBOX_USER="boxbot-sandbox"
SANDBOX_GROUP="boxbot"
REAL_USER="$(whoami)"

echo ""
echo "====================================="
echo "  boxBot Sandbox Setup"
echo "====================================="
echo ""

# -------------------------------------------------------------------
# Preflight
# -------------------------------------------------------------------

if [[ $EUID -eq 0 ]]; then
    echo "Error: Do not run as root. Run as your normal user."
    echo "The script will use sudo where needed."
    exit 1
fi

echo "  User:    $REAL_USER"
echo "  Sandbox: $SANDBOX_VENV"
echo ""

CHANGES=()

# -------------------------------------------------------------------
# 1. Create sandbox user
# -------------------------------------------------------------------

echo "--- Creating sandbox user ---"

if id "$SANDBOX_USER" &>/dev/null; then
    echo "User $SANDBOX_USER already exists."
else
    sudo useradd \
        --system \
        --no-create-home \
        --shell /usr/sbin/nologin \
        "$SANDBOX_USER"
    echo "Created system user: $SANDBOX_USER"
    CHANGES+=("Created system user: $SANDBOX_USER")
fi

# -------------------------------------------------------------------
# 2. Create group and add users
# -------------------------------------------------------------------

echo ""
echo "--- Configuring group ---"

if getent group "$SANDBOX_GROUP" &>/dev/null; then
    echo "Group $SANDBOX_GROUP already exists."
else
    sudo groupadd "$SANDBOX_GROUP"
    echo "Created group: $SANDBOX_GROUP"
    CHANGES+=("Created group: $SANDBOX_GROUP")
fi

for user in "$REAL_USER" "$SANDBOX_USER"; do
    # Match the group name as a *whole token* — ``grep -w`` treats
    # hyphens as word boundaries, so a literal ``boxbot`` would falsely
    # match a primary group named ``boxbot-sandbox``. Compare the
    # space-separated group list line-by-line with ``-x`` instead.
    if id -nG "$user" 2>/dev/null | tr ' ' '\n' | grep -qx "$SANDBOX_GROUP"; then
        echo "$user already in $SANDBOX_GROUP group."
    else
        sudo usermod -aG "$SANDBOX_GROUP" "$user"
        echo "Added $user to $SANDBOX_GROUP group."
        CHANGES+=("Added $user to $SANDBOX_GROUP group")
    fi
done

# -------------------------------------------------------------------
# 3. Create sandbox venv
# -------------------------------------------------------------------

echo ""
echo "--- Creating sandbox virtual environment ---"

# /var/lib/boxbot-sandbox doesn't exist on a fresh system, and the
# operator can't create directories there without sudo. Create the
# parent owned by the operator so the rest of the script can manage it
# without escalation; the per-subdir owner/perm fixups happen below.
if [[ ! -d "$SANDBOX_DIR" ]]; then
    sudo mkdir -p "$SANDBOX_DIR"
    sudo chown "$REAL_USER:$SANDBOX_GROUP" "$SANDBOX_DIR"
    sudo chmod 750 "$SANDBOX_DIR"
    echo "Created sandbox root at $SANDBOX_DIR"
    CHANGES+=("Created sandbox root at $SANDBOX_DIR")
fi

mkdir -p "$SANDBOX_DIR/output" "$SANDBOX_DIR/tmp" "$SANDBOX_DIR/scripts"

if [[ ! -d "$SANDBOX_VENV" ]]; then
    python3 -m venv "$SANDBOX_VENV"
    echo "Created sandbox venv at $SANDBOX_VENV"
    CHANGES+=("Created sandbox venv at $SANDBOX_VENV")
else
    echo "Sandbox venv already exists at $SANDBOX_VENV"
fi

"$SANDBOX_VENV/bin/pip" install --upgrade pip --quiet

# -------------------------------------------------------------------
# 4. Install sandbox packages
# -------------------------------------------------------------------

echo ""
echo "--- Installing sandbox packages ---"

REQUIREMENTS_FILE="$PROJECT_DIR/config/sandbox-requirements.txt"

if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    echo "Error: $REQUIREMENTS_FILE not found."
    exit 1
fi

# Install third-party packages (filter out the SDK line — its relative
# path is unreliable and we install the SDK separately below)
grep -v "^-e" "$REQUIREMENTS_FILE" | grep -v "^#" | grep -v "^$" | \
    "$SANDBOX_VENV/bin/pip" install -r /dev/stdin --quiet

echo "Installed third-party sandbox packages."

# Install SDK as a regular (non-editable) package so the code is copied
# into site-packages. This means the sandbox user does NOT need read
# access to src/boxbot/ — the SDK lives entirely inside the venv.
SDK_DIR="$PROJECT_DIR/src/boxbot/sdk"

if [[ ! -f "$SDK_DIR/pyproject.toml" ]]; then
    echo "Error: SDK not found at $SDK_DIR/pyproject.toml"
    exit 1
fi

"$SANDBOX_VENV/bin/pip" install "$SDK_DIR" --quiet

echo "Installed boxbot_sdk into sandbox venv."
CHANGES+=("Installed sandbox packages + boxbot_sdk")

# Install the libseccomp Python binding system-wide so the sandbox
# bootstrap (scripts/sandbox_bootstrap.py) can apply the syscall filter
# at startup. ``python3-seccomp`` provides the ``import seccomp`` API.
# We install it via apt because libseccomp is a C library — pip
# installs would compile from source and pull build deps.
#
# If apt is unavailable (non-Debian) the operator can ``pip install
# pyseccomp`` into the sandbox venv as a fallback; the bootstrap tries
# both module names.
if command -v apt-get >/dev/null 2>&1; then
    if ! dpkg -s python3-seccomp >/dev/null 2>&1; then
        echo "--- Installing python3-seccomp ---"
        apt-get install -y python3-seccomp >/dev/null
        CHANGES+=("Installed python3-seccomp (libseccomp Python binding)")
    fi
    # Also expose the apt-installed binding to the sandbox venv. The
    # venv was created with ``--system-site-packages`` (see step 3) so
    # this should already be visible; we just verify.
    if ! sudo -u "$SANDBOX_USER" "$SANDBOX_VENV/bin/python3" \
           -c "import seccomp" >/dev/null 2>&1; then
        echo "  Note: seccomp module not visible to sandbox venv —"
        echo "  the venv was probably created without --system-site-packages."
        echo "  Falling back to pyseccomp via pip (PyPI binding)."
        "$SANDBOX_VENV/bin/pip" install pyseccomp --quiet
        CHANGES+=("Installed pyseccomp into sandbox venv (fallback)")
    fi
else
    echo "  apt-get not found — installing pyseccomp via pip"
    "$SANDBOX_VENV/bin/pip" install pyseccomp --quiet
    CHANGES+=("Installed pyseccomp into sandbox venv (no apt available)")
fi

# -------------------------------------------------------------------
# 5. Pre-compile bytecode
# -------------------------------------------------------------------

echo ""
echo "--- Pre-compiling Python bytecode ---"

# Compile all .py files to .pyc so the sandbox user doesn't need write
# access to __pycache__ directories at runtime.
"$SANDBOX_VENV/bin/python3" -m compileall "$SANDBOX_VENV/lib" -q 2>/dev/null || true

echo "Bytecode compiled."

# -------------------------------------------------------------------
# 6. Set filesystem permissions
# -------------------------------------------------------------------

echo ""
echo "--- Setting filesystem permissions ---"

# -- Sandbox venv: owned by main user, group-readable for sandbox --
# Strategy:
#   - Directories: 750 (owner rwx, group rx)
#   - Files: 640 (owner rw, group r)
#   - python3 binary: 750 (group can execute)
#   - pip binaries: 700 (owner-only — sandbox CANNOT install packages)

sudo chown -R "$REAL_USER:$SANDBOX_GROUP" "$SANDBOX_VENV"

# CRITICAL: ``find -type f`` and ``find -type d`` exclude symlinks, but
# the bare ``chmod`` calls below need to stay symlink-safe too —
# ``chmod`` follows symlinks by default, and the venv's python3 is a
# symlink to /usr/bin/python3.13. A naive ``chmod 750`` on the symlink
# would silently chmod the *system* Python, breaking it for every user
# on the box. We skip symlinks (their perms come from the target file,
# which already has the right system perms).
find "$SANDBOX_VENV" -type d -exec sudo chmod 750 {} +
find "$SANDBOX_VENV" -type f -exec sudo chmod 640 {} +

# Python interpreter binary inside the venv (real file, not a symlink)
# must be executable by the sandbox group. The bin/python3 *symlink*
# inherits its target's perms, so we leave it alone.
for pybin in "$SANDBOX_VENV"/bin/python3*; do
    if [[ -f "$pybin" && ! -L "$pybin" ]]; then
        sudo chmod 750 "$pybin"
    fi
done

# pip is a regular shell script with a shebang. Lock to owner-only so
# the sandbox user cannot install packages.
for pipbin in "$SANDBOX_VENV"/bin/pip*; do
    if [[ -f "$pipbin" && ! -L "$pipbin" ]]; then
        sudo chmod 700 "$pipbin"
    fi
done

# -- Sandbox working directories: owned by sandbox user --
sudo chown -R "$SANDBOX_USER:$SANDBOX_GROUP" "$SANDBOX_DIR/output"
sudo chown -R "$SANDBOX_USER:$SANDBOX_GROUP" "$SANDBOX_DIR/tmp"
sudo chown -R "$SANDBOX_USER:$SANDBOX_GROUP" "$SANDBOX_DIR/scripts"
sudo chmod -R 770 "$SANDBOX_DIR/output"
sudo chmod -R 770 "$SANDBOX_DIR/tmp"
sudo chmod -R 770 "$SANDBOX_DIR/scripts"

# -- Skills directory: group-writable (sandbox can create skills) --
if [[ -d "$PROJECT_DIR/skills" ]]; then
    sudo chown -R "$REAL_USER:$SANDBOX_GROUP" "$PROJECT_DIR/skills"
    find "$PROJECT_DIR/skills" -type d -exec sudo chmod 775 {} +
    find "$PROJECT_DIR/skills" -type f -exec sudo chmod 664 {} +
fi

# -- Displays directory: group-readable --
if [[ -d "$PROJECT_DIR/displays" ]]; then
    sudo chown -R "$REAL_USER:$SANDBOX_GROUP" "$PROJECT_DIR/displays"
    find "$PROJECT_DIR/displays" -type d -exec sudo chmod 750 {} +
    find "$PROJECT_DIR/displays" -type f -exec sudo chmod 640 {} +
fi

# -- .env: owner-only (sandbox CANNOT read secrets) --
if [[ -f "$PROJECT_DIR/.env" ]]; then
    sudo chown "$REAL_USER:$REAL_USER" "$PROJECT_DIR/.env"
    chmod 600 "$PROJECT_DIR/.env"
fi

# -- Data directories: group-readable where sandbox needs access --
for dir in "$PROJECT_DIR/data/memory" "$PROJECT_DIR/data/photos" \
           "$PROJECT_DIR/data/scheduler"; do
    if [[ -d "$dir" ]]; then
        sudo chown -R "$REAL_USER:$SANDBOX_GROUP" "$dir"
        find "$dir" -type d -exec sudo chmod 750 {} +
        find "$dir" -type f -exec sudo chmod 640 {} +
    fi
done

# -- Config directory: group-readable --
if [[ -d "$PROJECT_DIR/config" ]]; then
    sudo chown -R "$REAL_USER:$SANDBOX_GROUP" "$PROJECT_DIR/config"
    find "$PROJECT_DIR/config" -type d -exec sudo chmod 750 {} +
    find "$PROJECT_DIR/config" -type f -exec sudo chmod 640 {} +
fi

# -- Source code: owner-only (sandbox has NO access) --
if [[ -d "$PROJECT_DIR/src" ]]; then
    sudo chown -R "$REAL_USER:$REAL_USER" "$PROJECT_DIR/src"
    find "$PROJECT_DIR/src" -type d -exec sudo chmod 750 {} +
    find "$PROJECT_DIR/src" -type f -exec sudo chmod 640 {} +
fi

# -- .git: owner-only --
if [[ -d "$PROJECT_DIR/.git" ]]; then
    sudo chown -R "$REAL_USER:$REAL_USER" "$PROJECT_DIR/.git"
fi

echo "Filesystem permissions configured."
CHANGES+=("Set filesystem permissions (OS-level sandbox enforcement)")

# -------------------------------------------------------------------
# 7. Install seccomp profile
# -------------------------------------------------------------------

echo ""
echo "--- Installing seccomp profile ---"

SECCOMP_PROFILE="$PROJECT_DIR/config/seccomp-sandbox.json"

# This profile is loaded by the main process when launching sandbox
# scripts. It blocks process spawning while allowing Python threading.
#
# Format follows the OCI/Docker seccomp spec. The execute_script tool
# reads this file and applies it via libseccomp before exec.
#
# Rule ordering: argument-filtered rules are checked before catch-all
# rules for the same syscall, so the CLONE_THREAD allow takes priority
# over the clone block.

cat > "$SECCOMP_PROFILE" << 'SECCOMP_EOF'
{
  "_comment": "boxBot sandbox seccomp profile — blocks process spawning",
  "defaultAction": "SCMP_ACT_ALLOW",
  "syscalls": [
    {
      "names": [
        "execve",
        "execveat"
      ],
      "action": "SCMP_ACT_ERRNO",
      "errnoRet": 1,
      "_comment": "Block all exec — sandbox cannot spawn processes"
    },
    {
      "names": [
        "fork",
        "vfork"
      ],
      "action": "SCMP_ACT_ERRNO",
      "errnoRet": 1,
      "_comment": "Block fork/vfork — no child processes"
    },
    {
      "names": ["clone"],
      "action": "SCMP_ACT_ALLOW",
      "args": [
        {
          "index": 0,
          "value": 65536,
          "op": "SCMP_CMP_MASKED_EQ",
          "valueTwo": 65536
        }
      ],
      "_comment": "Allow clone with CLONE_THREAD (0x10000) for Python threading"
    },
    {
      "names": ["clone"],
      "action": "SCMP_ACT_ERRNO",
      "errnoRet": 1,
      "_comment": "Block clone without CLONE_THREAD"
    },
    {
      "names": ["clone3"],
      "action": "SCMP_ACT_ERRNO",
      "errnoRet": 1,
      "_comment": "Block clone3 — newer syscall, same restriction"
    }
  ]
}
SECCOMP_EOF

echo "Installed seccomp profile at config/seccomp-sandbox.json"
CHANGES+=("Installed seccomp profile")

# -------------------------------------------------------------------
# 8. Verify sandbox isolation
# -------------------------------------------------------------------

echo ""
echo "--- Verifying sandbox isolation ---"
echo ""

VERIFY_PASSED=0
VERIFY_FAILED=0

verify() {
    local description="$1"
    local expected="$2"   # "should_succeed" or "should_fail"
    local exit_code="$3"

    if [[ "$expected" == "should_succeed" && "$exit_code" -eq 0 ]] || \
       [[ "$expected" == "should_fail" && "$exit_code" -ne 0 ]]; then
        echo "  ✓ $description"
        # Plain assignment, not ((…++)). With ``set -e`` and the var
        # starting at 0, the post-increment expression evaluates to 0,
        # which bash treats as a failed command and aborts the script.
        VERIFY_PASSED=$((VERIFY_PASSED + 1))
    else
        echo "  ✗ $description"
        VERIFY_FAILED=$((VERIFY_FAILED + 1))
    fi
}

# Test 1: Sandbox user can execute Python
sudo -u "$SANDBOX_USER" "$SANDBOX_VENV/bin/python3" -c "print('ok')" &>/dev/null && rc=0 || rc=$?
verify "Sandbox can run Python interpreter" "should_succeed" "$rc"

# Test 2: Sandbox user can import boxbot_sdk
sudo -u "$SANDBOX_USER" "$SANDBOX_VENV/bin/python3" -c "import boxbot_sdk" &>/dev/null && rc=0 || rc=$?
verify "Sandbox can import boxbot_sdk" "should_succeed" "$rc"

# Test 3: Sandbox user can import third-party packages
sudo -u "$SANDBOX_USER" "$SANDBOX_VENV/bin/python3" -c "import requests; import numpy" &>/dev/null && rc=0 || rc=$?
verify "Sandbox can import requests, numpy" "should_succeed" "$rc"

# Test 4: Sandbox user can write to output directory
sudo -u "$SANDBOX_USER" touch "$SANDBOX_DIR/output/.verify_test" &>/dev/null && rc=0 || rc=$?
rm -f "$SANDBOX_DIR/output/.verify_test" 2>/dev/null
verify "Sandbox can write to $SANDBOX_DIR/output/" "should_succeed" "$rc"

# Test 5: Sandbox user CANNOT read project config directly. The
# project tree often lives under a 0700 home directory (typical Linux
# default), and the SDK accesses everything via the action protocol
# over stdin/stdout — there is no scenario where the sandbox needs
# direct filesystem access to config/. Treat readable config as a
# misconfiguration to flag.
if [[ -f "$PROJECT_DIR/config/config.example.yaml" ]]; then
    sudo -u "$SANDBOX_USER" test -r "$PROJECT_DIR/config/config.example.yaml" \
        &>/dev/null && rc=0 || rc=$?
    verify "Sandbox cannot read project config/" "should_fail" "$rc"
fi

# Test 6: Sandbox user CANNOT read .env
if [[ -f "$PROJECT_DIR/.env" ]]; then
    sudo -u "$SANDBOX_USER" cat "$PROJECT_DIR/.env" &>/dev/null && rc=0 || rc=$?
    verify "Sandbox cannot read .env" "should_fail" "$rc"
else
    echo "  - Skipped .env test (file not created yet)"
fi

# Test 7: Sandbox user CANNOT execute pip
sudo -u "$SANDBOX_USER" "$SANDBOX_VENV/bin/pip" --version &>/dev/null && rc=0 || rc=$?
verify "Sandbox cannot execute pip" "should_fail" "$rc"

# Test 8: Sandbox user CANNOT write to site-packages
SITE_PACKAGES=$("$SANDBOX_VENV/bin/python3" -c "import site; print(site.getsitepackages()[0])")
sudo -u "$SANDBOX_USER" touch "$SITE_PACKAGES/.verify_test" &>/dev/null && rc=0 || rc=$?
rm -f "$SITE_PACKAGES/.verify_test" 2>/dev/null
verify "Sandbox cannot write to site-packages" "should_fail" "$rc"

# Test 9: Sandbox user CANNOT read source code
if [[ -d "$PROJECT_DIR/src/boxbot" ]]; then
    sudo -u "$SANDBOX_USER" ls "$PROJECT_DIR/src/boxbot/" &>/dev/null && rc=0 || rc=$?
    verify "Sandbox cannot access src/boxbot/" "should_fail" "$rc"
else
    echo "  - Skipped src/ test (directory not populated yet)"
fi

# Test 10: Sandbox user CANNOT write to the venv
sudo -u "$SANDBOX_USER" touch "$SANDBOX_VENV/.verify_test" &>/dev/null && rc=0 || rc=$?
rm -f "$SANDBOX_VENV/.verify_test" 2>/dev/null
verify "Sandbox cannot write to venv" "should_fail" "$rc"

echo ""
if [[ $VERIFY_FAILED -eq 0 ]]; then
    echo "All $VERIFY_PASSED verification checks passed."
    CHANGES+=("Sandbox verification: $VERIFY_PASSED/$VERIFY_PASSED passed")
else
    echo "WARNING: $VERIFY_FAILED of $((VERIFY_PASSED + VERIFY_FAILED)) checks failed!"
    echo "Review the failures above and re-run this script after fixing."
    CHANGES+=("Sandbox verification: $VERIFY_FAILED FAILED")
fi

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------

echo ""
echo "====================================="
echo "  Sandbox Setup Complete"
echo "====================================="
echo ""

for change in "${CHANGES[@]}"; do
    echo "  ✓ $change"
done

echo ""
echo "  Sandbox user:   $SANDBOX_USER"
echo "  Sandbox venv:   $SANDBOX_VENV"
echo "  Seccomp:        config/seccomp-sandbox.json"
echo ""
echo "  Permission summary:"
echo "    CAN read:     config/, data/memory/, data/photos/, data/scheduler/"
echo "    CAN write:    $SANDBOX_DIR/output/, $SANDBOX_DIR/tmp/, skills/"
echo "    CAN execute:  $SANDBOX_VENV/bin/python3"
echo "    CANNOT read:  .env, src/boxbot/, .git/"
echo "    CANNOT write: site-packages, venv/"
echo "    CANNOT exec:  pip, any subprocess (seccomp)"
echo ""
