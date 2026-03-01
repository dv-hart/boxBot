# OS Hardening

boxBot's application-level security — sandbox isolation, seccomp filters,
file permissions — is designed to contain the agent. But none of that
matters if the underlying Raspberry Pi is wide open on the network.

A default Raspberry Pi OS (Bookworm) install has SSH enabled with password
auth, no firewall, and several services (Bluetooth, mDNS) listening on
the network. This page explains what the hardening script does and why.

## What the Script Does

Run `scripts/harden-os.sh` as root (or with sudo). It's idempotent — safe
to re-run at any time.

### 1. Firewall (UFW)

**Default deny inbound, allow outbound.** Only two ports are opened:

| Port | Rule | Why |
|------|------|-----|
| 22 (SSH) | `ufw limit` (rate-limited) | Remote access for tinkering |
| WhatsApp webhook | `ufw allow` | Required by WhatsApp Business API |

Rate-limiting SSH means UFW automatically drops connections from IPs that
attempt more than 6 connections in 30 seconds. This is a lightweight
defense — the real protection is key-only auth (see below).

### 2. SSH Hardening

SSH stays enabled. This is a tinkerer project — you'll want to SSH in.
But password authentication is disabled in favor of SSH keys:

| Setting | Value | Why |
|---------|-------|-----|
| `PasswordAuthentication` | `no` | Keys only — no brute-force risk |
| `PermitRootLogin` | `no` | Use a regular user + sudo |
| `PubkeyAuthentication` | `yes` | Explicit (default, but stated for clarity) |
| `ClientAliveInterval` | `300` | Drop idle sessions after 10 minutes |
| `ClientAliveCountMax` | `2` | Two missed keepalives = disconnect |

These settings are applied as a drop-in file at
`/etc/ssh/sshd_config.d/boxbot.conf` so they don't touch the system
defaults.

**Lockout protection:** The script checks that at least one SSH public key
exists in `~/.ssh/authorized_keys` (for the user running the script)
before disabling password auth. If no keys are found, it aborts with
instructions on how to add one.

### 3. Automatic Security Updates

The script enables `unattended-upgrades` for **Debian security patches
only**. This means:

- Critical security fixes are applied automatically
- Normal package updates and distribution upgrades are **not** automatic
- You stay in control of feature updates

The configuration targets only the `${distro_id}:${distro_codename}-security`
origin, matching Raspberry Pi OS's Debian-based repos.

### 4. Disable Unnecessary Services

| Service | Why disabled |
|---------|-------------|
| `bluetooth` | boxBot has no Bluetooth peripherals; reduces attack surface |
| `avahi-daemon` | mDNS/DNS-SD service discovery; unnecessary for boxBot's communication model |

If you use Bluetooth peripherals or rely on mDNS (e.g., accessing
`boxbot.local`), you can skip these by editing the script or re-enabling
the services after running it.

## What We Don't Do (and Why)

### No ICMP blocking
Blocking ping is security theater. boxBot sits behind a home router's NAT
— it's not directly reachable from the internet. Blocking ICMP on the LAN
just makes debugging harder.

### No fail2ban
fail2ban watches auth logs for brute-force attempts and bans IPs. With
key-only SSH, password brute-force is irrelevant — there's no password to
guess. UFW's rate limiting handles connection floods. fail2ban would add
complexity with no meaningful security benefit.

### No VPN / Tailscale
Wrong threat model. boxBot's attack surface is the home LAN. If someone
is on your WiFi, they can already see all your other devices too. A VPN
would add latency to the WhatsApp webhook and complexity to the setup for
no real gain in this context.

### No port knocking
Obscurity, not security. And it makes SSH access annoying.

## Running the Script

```bash
# Review what it will do first
cat scripts/harden-os.sh

# Run it
sudo scripts/harden-os.sh

# Or with a specific webhook port
sudo WEBHOOK_PORT=8443 scripts/harden-os.sh
```

The script prints a summary of every change it makes. If something is
already configured, it skips that step and tells you.

## Verifying

After running, you can verify the configuration:

```bash
# Firewall status
sudo ufw status verbose

# SSH config
sudo sshd -T | grep -E 'passwordauthentication|permitrootlogin|pubkeyauthentication'

# Unattended upgrades
apt-config dump | grep -i unattended

# Disabled services
systemctl is-enabled bluetooth avahi-daemon
```

## Reverting

Everything the script does is reversible:

```bash
# Disable firewall
sudo ufw disable

# Re-enable password auth
sudo rm /etc/ssh/sshd_config.d/boxbot.conf
sudo systemctl restart sshd

# Re-enable services
sudo systemctl enable --now bluetooth avahi-daemon

# Remove unattended-upgrades
sudo apt remove unattended-upgrades
```
