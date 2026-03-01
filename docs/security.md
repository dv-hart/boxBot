# Security Model

## Principles

1. **Minimal attack surface** — no web UI, no unnecessary open ports, SSH
   key-only (password auth disabled)
2. **Explicit trust** — every communication channel requires authentication
3. **Physical presence** — high-privilege actions (enrollment, admin) require
   being physically at the box
4. **Local processing** — perception runs on-device; no images or audio
   leave the box except explicit API calls
5. **No information leakage** — unauthenticated requests receive no response

## Communication Channels

| Channel | Auth Method | Trust Basis |
|---------|------------|-------------|
| Voice | Physical presence | You're in the room |
| Buttons | Physical presence | You're at the box |
| WhatsApp | Phone number whitelist | Explicit enrollment |

## User Registration

Users are people with authenticated remote channels (WhatsApp by default).
Registration uses single-use, time-limited codes. See
[user-registration.md](user-registration.md) for the complete flow.

### First Admin (Bootstrap)
- Code displayed on the 7" screen during initial setup (physical presence)
- User texts code to BB's WhatsApp → registered as primary admin
- Bootstrap permanently disabled after first admin registers
- This is the ONLY registration that doesn't require an existing admin

### New User Registration
- Admin requests a code → BB sends code to admin via WhatsApp
- Admin shares code with new user out-of-band (in person, text, etc.)
- New user texts code to BB → registered as standard user
- BB **never** initiates contact with unknown numbers — admin gatekeeps

### Registration Codes
- 6 digits, cryptographically random, single-use, expires in 10 minutes
- One active code at a time, rate-limited (3 per hour per admin)
- Failed attempts: silent drop, rate limiting, temp/permanent blocking

### Permission Levels
- **admin** — register/remove users, approve packages, promote admins
- **standard** — send/receive messages, interact with skills

## WhatsApp Security

### Hard Block on Unknown Numbers
- Messages from unregistered numbers → **silent drop** (no response)
- No error message, no acknowledgment, no information leakage
- Only exception: message contains a valid, unexpired registration code
- Failed code attempts also produce no response

### Privileged Channel Separation
BB's WhatsApp is for **user ↔ BB communication only**. It is not a
proxy for other services:
- Email checking → skill (sandbox, IMAP/Gmail API, own credentials)
- Calendar sync → skill (sandbox, CalDAV, own credentials)
- RSS/news → skill (sandbox, own packages)
- Any inbox-style service → skill, never through the messaging path

BB may relay results via WhatsApp ("you have 3 unread emails"), but
the data fetching runs in the sandbox. This prevents privilege confusion
and keeps the authenticated channel clean.

### Webhook Security
- Incoming webhooks validated against WhatsApp's request signature
- Forged webhooks rejected (HTTP 403) before any processing
- Webhook port is the only open listener (required by WhatsApp API)

## Data Security

### Secrets Management
- API keys and tokens stored in `.env` (gitignored)
- WhatsApp credentials in `config/whatsapp.yaml` (gitignored)
- No secrets in code or committed config files

### Local Data
- Person embeddings stored locally in `data/` (gitignored)
- Photos stored locally in `data/photos/` (gitignored)
- Memory database stored locally (gitignored)
- Optional encrypted cloud backup (S3, disabled by default)

### Sandbox Isolation (OS-Level)

Sandbox security is enforced at the **operating system level**. Python-
level restrictions can always be bypassed; OS-level enforcement cannot
be circumvented from userspace code.

**Dedicated user:** Scripts run as `boxbot-sandbox`, a restricted system
user with minimal permissions. The main process runs as `boxbot`.

**seccomp filter:** The sandbox process has `execve`, `fork`, `vfork`,
and `clone` syscalls blocked at the kernel level. No subprocess spawning
of any kind — `subprocess.run()`, `os.system()`, `os.exec*()`, and
even `ctypes`-based syscalls are all killed by the kernel.

**Filesystem permissions:** Enforced by Unix file ownership, not Python
checks:
- `.env` → mode `0600` owned by `boxbot` → sandbox cannot read
- `src/boxbot/` → owned by `boxbot` → sandbox cannot write
- `data/sandbox/venv/lib/` → owned by `boxbot` → sandbox cannot write
  (read + execute only)
- `data/sandbox/venv/bin/pip` → mode `0700` owned by `boxbot` → sandbox
  cannot execute
- `data/sandbox/output/`, `tmp/` → owned by `boxbot-sandbox` → writable
- `skills/` → group-writable → sandbox can create skill directories

**Resource limits:** 30s timeout (main process kills), 256MB memory
(`ulimit`/cgroup), single core.

**Secrets:** Scripts receive only specific API keys needed for the current
task, passed as env vars by the main process. Never the full `.env`.

### Package Installation (Out-of-Band Approval)

The sandbox **cannot install packages**. This is enforced by four
independent OS-level controls:

1. seccomp blocks `execve` → can't run pip as subprocess
2. pip binary is mode `0700` owned by `boxbot` → sandbox can't execute it
3. site-packages is owned by `boxbot` → sandbox can't write to it
4. venv directory is read-only to `boxbot-sandbox`

Only the main process (as `boxbot`) can install packages, and it will
only do so after **out-of-band human approval** through a channel the
sandbox cannot access:
- **Display tap:** approval prompt on the touchscreen (physical input)
- **WhatsApp reply:** message to the admin user, response validated
  against the admin's phone number

The approval channel is completely separate from the sandbox's stdout.
The sandbox can only emit a REQUEST action — there is no SDK action that
means "already approved." The agent cannot spoof approval because the
approval comes from a different input pathway entirely.

All installs are logged to `data/sandbox/installed.txt` (append-only,
owned by `boxbot`).

### Agent Authoring via SDK

The agent creates skills and displays through the `boxbot_sdk` — a
constrained, immutable API pre-installed in the sandbox venv. The agent
**cannot write code that runs directly in the main process**:
- SDK is in read-only site-packages (owned by `boxbot`)
- Skills auto-activate (logic runs in sandbox, contained)
- Displays are declarative (agent uses building blocks, main process
  generates validated render code — no raw `render()` from agent input)
- Displays require user confirmation before activation
- Core code (`src/boxbot/`) is off-limits (owned by `boxbot`)

## OS Hardening

The application-level controls above protect against a compromised agent.
OS-level hardening protects the Pi itself from network-based attacks.

**`scripts/harden-os.sh`** is a standalone, idempotent script that:
- Enables a UFW firewall (deny all inbound; allow only SSH and the
  WhatsApp webhook port)
- Hardens SSH: key-only authentication, no root login, idle timeout
- Enables automatic security updates (Debian security patches only)
- Disables unnecessary services (Bluetooth, mDNS)

SSH remains enabled — this is a tinkerer project — but password auth is
disabled. The script includes lockout protection: it will not disable
password auth unless at least one SSH public key is present.

See [os-hardening.md](os-hardening.md) for full rationale, verification
steps, and how to revert each change.

## Threat Model

### Registration & Access

| Threat | Mitigation |
|--------|-----------|
| Unauthorized WhatsApp message | Hard block: silent drop, no response, no info leakage |
| Brute-force registration code | Rate limiting (5 attempts/10min), temp/perm blocking, no response on failure |
| Registration code interception | Single-use, 10-min expiry, admin shares out-of-band |
| Bootstrap hijacking | Code on physical screen (requires presence), bootstrap disabled after first admin |
| Admin impersonation | WhatsApp verifies phone ownership; admin phone number is the identity |
| Privilege escalation | Promotion requires existing admin confirmation via WhatsApp |
| Self-lockout | Primary admin cannot be demoted; physical button reset as fallback |

### Sandbox & Code Execution

| Threat | Mitigation |
|--------|-----------|
| API key exposure | `.env` mode `0600`, owned by `boxbot`, not readable by sandbox |
| Direct pip install | seccomp blocks execve + pip not executable + site-packages read-only |
| Spoof package approval | Approval is out-of-band (physical tap / admin WhatsApp); sandbox stdout can only emit requests |
| Malicious skill code | Sandbox: separate user, seccomp, filesystem restrictions |
| Malicious display code | Declarative SDK (no raw code); user approval for activation |
| Supply chain attack (pip) | Package install requires out-of-band human approval |
| Sandbox escape | OS-level: separate user, seccomp, filesystem permissions, resource limits |

### Agent & Communication

| Threat | Mitigation |
|--------|-----------|
| Prompt injection via WhatsApp | Only registered users can message; agent instruction hierarchy |
| Prompt injection via web content | Small-model content firewall: web results filtered by isolated small agent before reaching large model. Small agent has no boxBot tools/SDK access; output is plain text only. Defense in depth: small model filters, plain text boundary, large model judgment |
| Prompt injection → install package | Approval requires human on separate channel; agent can't self-approve |
| Privilege confusion (email via WhatsApp) | Inbox services are skills (sandbox), not WhatsApp features; privileged channel separation |
| Camera/mic data exfiltration | All perception on-device, no streaming |
| Network scanning/probing | UFW denies all inbound except SSH (key-only, rate-limited) and webhook (signature-validated) |

## Reporting Vulnerabilities

If you discover a security vulnerability, **do not open a public issue**.
Contact the maintainers directly.
