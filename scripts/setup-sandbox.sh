#!/usr/bin/env bash
# setup-sandbox.sh — Create the isolated sandbox environment
#
# Idempotent: safe to re-run. Each step checks current state before acting.
#
# What it does:
#   1. Creates boxbot-sandbox system user (no login, no home, no shell)
#   2. Creates boxbot group and adds both users
#   3. Creates sandbox venv at data/sandbox/venv
#   4. Installs sandbox packages + boxbot_sdk
#   5. Pre-compiles bytecode for performance
#   6. Sets filesystem permissions (OS-level sandbox enforcement)
#   7. Installs seccomp profile for subprocess blocking
#   8. Verifies sandbox isolation
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
SANDBOX_DIR="$PROJECT_DIR/data/sandbox"
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
    if id -nG "$user" 2>/dev/null | grep -qw "$SANDBOX_GROUP"; then
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

mkdir -p "$SANDBOX_DIR/output" "$SANDBOX_DIR/tmp" "$SANDBOX_DIR/scripts"

if [[ ! -d "$SANDBOX_VENV" ]]; then
    python3 -m venv "$SANDBOX_VENV"
    echo "Created sandbox venv."
    CHANGES+=("Created sandbox venv at data/sandbox/venv/")
else
    echo "Sandbox venv already exists."
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

find "$SANDBOX_VENV" -type d -exec sudo chmod 750 {} +
find "$SANDBOX_VENV" -type f -exec sudo chmod 640 {} +

# Python interpreter must be executable by sandbox
sudo chmod 750 "$SANDBOX_VENV/bin/python3"
# Also handle the versioned symlink (python3.11, python3.12, etc.)
for pybin in "$SANDBOX_VENV"/bin/python3.*; do
    [[ -e "$pybin" ]] && sudo chmod 750 "$pybin"
done

# pip must NOT be executable by sandbox user
for pipbin in "$SANDBOX_VENV"/bin/pip*; do
    [[ -e "$pipbin" ]] && sudo chmod 700 "$pipbin"
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
        ((VERIFY_PASSED++))
    else
        echo "  ✗ $description"
        ((VERIFY_FAILED++))
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
verify "Sandbox can write to data/sandbox/output/" "should_succeed" "$rc"

# Test 5: Sandbox user can read config
sudo -u "$SANDBOX_USER" test -r "$PROJECT_DIR/config/config.example.yaml" &>/dev/null && rc=0 || rc=$?
verify "Sandbox can read config/" "should_succeed" "$rc"

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
echo "    CAN write:    data/sandbox/output/, data/sandbox/tmp/, skills/"
echo "    CAN execute:  $SANDBOX_VENV/bin/python3"
echo "    CANNOT read:  .env, src/boxbot/, .git/"
echo "    CANNOT write: site-packages, venv/"
echo "    CANNOT exec:  pip, any subprocess (seccomp)"
echo ""
