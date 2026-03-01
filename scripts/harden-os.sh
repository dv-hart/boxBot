#!/usr/bin/env bash
# harden-os.sh — Raspberry Pi OS hardening for boxBot
#
# Idempotent: safe to re-run. Each step checks current state before acting.
#
# What it does:
#   1. Installs ufw and unattended-upgrades
#   2. Configures firewall (deny inbound, allow SSH + webhook port)
#   3. Hardens SSH (key-only auth, no root login, idle timeout)
#   4. Enables automatic security updates
#   5. Disables unnecessary services (bluetooth, avahi-daemon)
#
# Usage:
#   sudo ./scripts/harden-os.sh
#   sudo WEBHOOK_PORT=8443 ./scripts/harden-os.sh
#
# See docs/os-hardening.md for full rationale.

set -euo pipefail

# -------------------------------------------------------------------
# Preflight
# -------------------------------------------------------------------

if [[ $EUID -ne 0 ]]; then
    echo "Error: This script must be run as root (use sudo)."
    exit 1
fi

# Determine the real user who invoked sudo (for SSH key check)
REAL_USER="${SUDO_USER:-$(logname 2>/dev/null || echo "")}"
if [[ -z "$REAL_USER" ]]; then
    echo "Error: Cannot determine the non-root user."
    echo "Run this script with: sudo $0"
    exit 1
fi

REAL_HOME=$(eval echo "~$REAL_USER")

# Webhook port: from environment, or prompt
WEBHOOK_PORT="${WEBHOOK_PORT:-}"
if [[ -z "$WEBHOOK_PORT" ]]; then
    read -rp "WhatsApp webhook port [8443]: " WEBHOOK_PORT
    WEBHOOK_PORT="${WEBHOOK_PORT:-8443}"
fi

# Validate port number
if ! [[ "$WEBHOOK_PORT" =~ ^[0-9]+$ ]] || (( WEBHOOK_PORT < 1 || WEBHOOK_PORT > 65535 )); then
    echo "Error: Invalid port number: $WEBHOOK_PORT"
    exit 1
fi

echo ""
echo "====================================="
echo "  boxBot OS Hardening"
echo "====================================="
echo ""
echo "  User:         $REAL_USER"
echo "  Webhook port: $WEBHOOK_PORT"
echo ""

CHANGES=()

# -------------------------------------------------------------------
# 1. Install packages
# -------------------------------------------------------------------

echo "--- Checking required packages ---"

PACKAGES_TO_INSTALL=()
for pkg in ufw unattended-upgrades; do
    if ! dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
        PACKAGES_TO_INSTALL+=("$pkg")
    fi
done

if [[ ${#PACKAGES_TO_INSTALL[@]} -gt 0 ]]; then
    echo "Installing: ${PACKAGES_TO_INSTALL[*]}"
    apt-get update -qq
    apt-get install -y -qq "${PACKAGES_TO_INSTALL[@]}"
    CHANGES+=("Installed packages: ${PACKAGES_TO_INSTALL[*]}")
else
    echo "All required packages already installed."
fi

# -------------------------------------------------------------------
# 2. Configure UFW firewall
# -------------------------------------------------------------------

echo ""
echo "--- Configuring firewall (UFW) ---"

# Set default policies
ufw default deny incoming >/dev/null 2>&1
ufw default allow outgoing >/dev/null 2>&1

# Allow SSH with rate limiting
if ufw status | grep -q "22/tcp.*LIMIT"; then
    echo "SSH rate limiting already configured."
else
    # Remove any existing SSH rules first to avoid duplicates
    ufw delete allow ssh 2>/dev/null || true
    ufw delete allow 22/tcp 2>/dev/null || true
    ufw limit ssh
    CHANGES+=("UFW: Added rate-limited SSH rule (port 22)")
fi

# Allow webhook port
if ufw status | grep -q "${WEBHOOK_PORT}/tcp.*ALLOW"; then
    echo "Webhook port $WEBHOOK_PORT already allowed."
else
    ufw allow "${WEBHOOK_PORT}/tcp" comment "boxBot WhatsApp webhook"
    CHANGES+=("UFW: Allowed webhook port $WEBHOOK_PORT/tcp")
fi

# Enable UFW (idempotent — prints "already enabled" if active)
if ufw status | grep -q "Status: active"; then
    echo "UFW already active."
else
    echo "y" | ufw enable
    CHANGES+=("UFW: Enabled firewall")
fi

# -------------------------------------------------------------------
# 3. Harden SSH
# -------------------------------------------------------------------

echo ""
echo "--- Hardening SSH ---"

SSHD_CONF="/etc/ssh/sshd_config.d/10-boxbot.conf"

# Lockout protection: verify at least one SSH key exists
AUTHORIZED_KEYS="$REAL_HOME/.ssh/authorized_keys"
if [[ ! -f "$AUTHORIZED_KEYS" ]] || [[ ! -s "$AUTHORIZED_KEYS" ]]; then
    echo ""
    echo "============================================================"
    echo "  WARNING: No SSH authorized keys found!"
    echo "============================================================"
    echo ""
    echo "  File checked: $AUTHORIZED_KEYS"
    echo ""
    echo "  Disabling password auth without an SSH key would lock you"
    echo "  out of the Pi. Add your public key first:"
    echo ""
    echo "    ssh-copy-id $REAL_USER@<pi-address>"
    echo ""
    echo "  Then re-run this script."
    echo ""
    echo "  Skipping SSH hardening (other steps still applied)."
    echo ""
    CHANGES+=("SSH: SKIPPED — no authorized_keys found (lockout protection)")
else
    KEY_COUNT=$(grep -c "^ssh-" "$AUTHORIZED_KEYS" 2>/dev/null || echo "0")
    echo "Found $KEY_COUNT SSH public key(s) in $AUTHORIZED_KEYS"

    # Remove cloud-init SSH config if present — it sets PasswordAuthentication yes
    # and takes precedence over later-sorted config files (sshd uses first match)
    CLOUD_INIT_CONF="/etc/ssh/sshd_config.d/50-cloud-init.conf"
    if [[ -f "$CLOUD_INIT_CONF" ]]; then
        echo "Removing $CLOUD_INIT_CONF (conflicts with boxBot SSH hardening)"
        rm -f "$CLOUD_INIT_CONF"
        CHANGES+=("SSH: Removed 50-cloud-init.conf (PasswordAuthentication override)")
    fi

    # Remove old boxbot.conf if it exists (renamed to 10-boxbot.conf for precedence)
    OLD_CONF="/etc/ssh/sshd_config.d/boxbot.conf"
    if [[ -f "$OLD_CONF" ]]; then
        echo "Removing old $OLD_CONF (migrated to 10-boxbot.conf)"
        rm -f "$OLD_CONF"
    fi

    SSHD_CONTENT="# boxBot SSH hardening — managed by scripts/harden-os.sh
# Do not edit manually; re-run the script to update.
PasswordAuthentication no
PermitRootLogin no
PubkeyAuthentication yes
ClientAliveInterval 300
ClientAliveCountMax 2"

    if [[ -f "$SSHD_CONF" ]] && [[ "$(cat "$SSHD_CONF")" == "$SSHD_CONTENT" ]]; then
        echo "SSH hardening config already in place."
    else
        echo "$SSHD_CONTENT" > "$SSHD_CONF"
        chmod 644 "$SSHD_CONF"

        # Validate config before restarting
        if sshd -t 2>/dev/null; then
            systemctl restart sshd
            CHANGES+=("SSH: Configured key-only auth, no root login, idle timeout")
        else
            echo "Error: sshd config validation failed. Removing 10-boxbot.conf."
            rm -f "$SSHD_CONF"
            echo "SSH hardening was NOT applied. Check your sshd configuration."
            CHANGES+=("SSH: FAILED — config validation error, reverted")
        fi
    fi
fi

# -------------------------------------------------------------------
# 4. Enable automatic security updates
# -------------------------------------------------------------------

echo ""
echo "--- Configuring automatic security updates ---"

AUTO_UPGRADES_CONF="/etc/apt/apt.conf.d/20auto-upgrades"
AUTO_UPGRADES_CONTENT='APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";'

if [[ -f "$AUTO_UPGRADES_CONF" ]] && grep -q 'Unattended-Upgrade "1"' "$AUTO_UPGRADES_CONF"; then
    echo "Automatic security updates already enabled."
else
    echo "$AUTO_UPGRADES_CONTENT" > "$AUTO_UPGRADES_CONF"
    CHANGES+=("Auto-updates: Enabled unattended security upgrades")
fi

# Ensure only security updates are applied (not full dist-upgrade)
UNATTENDED_CONF="/etc/apt/apt.conf.d/50unattended-upgrades"
if [[ -f "$UNATTENDED_CONF" ]]; then
    # Check that the security origin is not commented out
    if grep -q '//.*"origin=Debian,codename=\${distro_codename}-security' "$UNATTENDED_CONF" 2>/dev/null; then
        # Uncomment the security origin line
        sed -i 's|//\(.*"origin=Debian,codename=${distro_codename}-security\)|\1|' "$UNATTENDED_CONF"
        CHANGES+=("Auto-updates: Uncommented Debian security origin")
    fi
    echo "Unattended-upgrades configuration checked."
else
    echo "Note: $UNATTENDED_CONF not found. unattended-upgrades will use defaults."
fi

# -------------------------------------------------------------------
# 5. Disable unnecessary services
# -------------------------------------------------------------------

echo ""
echo "--- Disabling unnecessary services ---"

for service in bluetooth avahi-daemon; do
    if systemctl is-enabled "$service" 2>/dev/null | grep -q "enabled"; then
        systemctl disable --now "$service" 2>/dev/null
        CHANGES+=("Disabled service: $service")
        echo "Disabled $service."
    elif systemctl list-unit-files "${service}.service" 2>/dev/null | grep -q "$service"; then
        echo "$service already disabled."
    else
        echo "$service not installed (skipped)."
    fi
done

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------

echo ""
echo "====================================="
echo "  Summary"
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
echo "  Run 'sudo ufw status verbose' to verify firewall rules."
echo "  See docs/os-hardening.md for details and how to revert."
echo ""
