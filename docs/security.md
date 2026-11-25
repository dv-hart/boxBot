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
- `WHATSAPP_APP_SECRET` is **required** — boxBot will not start the
  WhatsApp listener without it. Set it in `.env`
- Incoming webhooks validated against WhatsApp's `X-Hub-Signature-256`
  header using HMAC-SHA256 with the app secret
- Forged webhooks rejected (HTTP 403) before any processing
- Webhook port is the only open listener (required by WhatsApp API)

## Data Security

### Secrets Management

Two distinct surfaces, by who needs the value:

**1. Main-process secrets — `.env`**
- API keys the main process uses directly: `ANTHROPIC_API_KEY`,
  `WHATSAPP_ACCESS_TOKEN`, `ELEVENLABS_API_KEY`, AWS keys, etc.
- File mode `0600`, owned by the main-process user
- WhatsApp credentials also in `config/whatsapp.yaml` (gitignored)
- The sandbox user has no read access (and home-dir `0700` blocks
  traversal regardless)

**2. Agent-managed secrets — `bb.secrets` / `data/credentials/secrets.json`**
- Credentials the agent stores at runtime (third-party API keys,
  OAuth refresh tokens, e.g. `GOOGLE_CALENDAR_TOKEN_JSON`)
- File mode `0600`, owned by the main-process user, atomic write,
  64-secret cap, ≤8 KB per value
- Sandbox scripts and integrations receive their values **only** as
  `BOXBOT_SECRET_<NAME>` env vars at launch, scoped to:
  - The names declared in an integration's manifest `secrets:` list, or
  - The names the agent passed in `execute_script`'s `secrets=[…]`
- The agent itself **cannot read values back**. SDK calls return name
  lists and existence checks; values cross only the runner→subprocess
  boundary
- Action log redaction: `secrets.store` records `{name, status}` only;
  the value is never written to the action log the agent observes

Neither store is committed; both are gitignored.

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

### Web View (Kiosk Browser)

The `web` display shows a YouTube video or web page by opening Chromium
over the screen. Web content is the least-trusted thing the box renders,
so it is fenced separately from both the main process and the sandbox:

- **Own OS user** (`boxbot-kiosk`, created by `setup-sandbox.sh`): no
  access to `.env`, the project tree, `data/`, or the sandbox. Launch
  is refused if the user is missing — there is no fallback to the
  service user. Every process of that user is killed when the view
  closes.
- **No listener:** boxBot drives the browser over a FIFO pair
  (`--remote-debugging-pipe`), never a debugging port. The FIFOs live
  in a directory owned by the service user; the kiosk user can open
  them but cannot add, remove or replace entries, and boxBot opens them
  `O_NOFOLLOW`.
- **URL policy:** `https` on public hostnames only — no IPs, LAN names,
  ports, or credentials. This vets the link BB was handed. It cannot
  vet what the page does next (redirects, scripts, a DNS name that
  resolves to a private address), which is what the next item is for.
- **Network fence (kernel):** an nftables rule keyed on the kiosk uid
  drops everything it sends to loopback, RFC 1918, link-local, CGNAT
  and multicast ranges, and all IPv6 (LAN devices have global v6
  addresses no prefix list covers); DNS is the only exception. A
  page cannot reach the router, Home Assistant, or the box itself no
  matter how it gets the address. Installed by `setup-sandbox.sh` as
  `boxbot-kiosk-firewall.service`; every launch probes for it and
  refuses to start the browser if loopback is reachable.
- **Throwaway home:** the kiosk user's whole home — profile,
  certificate store, caches — is emptied at every launch. Downloads
  and popup windows are denied.
- **Not an agent input:** nothing the page controls reaches the agent's
  context: not pixels, not the page title. Web text reaches the agent
  only through `web_search`'s content firewall.

Residual exposure, accepted: to draw on the screen the kiosk user is
granted the session's Wayland socket (ACL set at launch), and for
sound it gets a PulseAudio socket of its own, opened by pipewire-pulse
with `client.access = "restricted"` in a directory only it and the
service user can enter. A browser fully compromised to that uid could
therefore capture or draw on the screen and use the audio server
through that socket. It cannot reach the session's own PulseAudio
socket or the native PipeWire sockets (closed to other users at
launch), cannot load PulseAudio modules (disabled by
`setup-sandbox.sh`), and cannot reach anything on disk or on the LAN. A hostile page can also lie to the idle check and stay up
until the `max_minutes` ceiling. The touchscreen stays live while a
page is open, so a person at the box can follow links — physical
presence, same trust as the buttons.

### Shopping (ids, cart, product images)

The agent shops by short result id. `bb.purchases.find` returns slim
rows (`a1`, `a2`, …); the full catalogue row — variant id, store
domain, price, image URL — stays in a main-process cache keyed by the
conversation the tool call runs in. The key comes from the tool call's
own context, never from the sandbox payload, so an id from one
conversation resolves to nothing in another. Sessions expire after 30
idle minutes, are capped in number and size, and are dropped when the
conversation ends. Product cards and the cart screen are drawn from
that cache: display args carry only an opaque view token, so a title,
price, or seller on the screen is the catalogue's, not something the
model typed. Merchant text is sanitised (`plain_text`) and rendered as
text only.

- **Cart and quote move nothing.** A cart is in-memory state. A quote
  builds a store checkout to read shipping and tax — only at an
  *approved* store, because creating a checkout hands the store the
  household's saved contact and delivery details. It creates no
  purchase record and texts nobody. Hourly ceiling
  (`purchases.max_quotes_per_hour`).
- **One definition of "approved".** `purchases.allowed_shops` (config)
  ∪ stores an admin approved by their own `allow shop <domain>` text.
  Every path — quote, send, product-feed read — asks the same function.
  Automatic purchase stays narrower: config list only.
- **Send is propose.** `send` runs each store's lines through the
  existing proposal path (rules, isolated checker, review link or veto
  window). Money still moves only by an admin paying from the link on
  their phone, or inside the configured automatic limits.
- **Voice identity authorises nothing.** The requester recorded and
  shown to admins is the transport ("voice at the box, 4:12 pm"), never
  a name the perception stack attached to a voice.
- **Cancel is open to anyone, and local.** Cancelling stops boxBot's
  own purchase row; the store's checkout link stays payable. So when a
  purchase whose link was sent is cancelled by the agent, the main
  process texts the admins a fixed-format "ignore the earlier link"
  notice (rate-limited). A changed cart sent again cancels the earlier
  purchase and its one text is marked as a replacement.

**Product images — a deliberate, bounded exception.** The main process
otherwise never fetches a URL that came from a store. To put a photo
on a product card it does, in `boxbot.purchases.images`, under these
limits:

| Bound | Value |
|---|---|
| Switch | `purchases.fetch_product_images` (default true) |
| Scheme / port | `https`, default port, no userinfo |
| Host | exact match on `purchases.product_image_hosts` (default `["cdn.shopify.com"]`) |
| Redirects | not followed automatically; ≤2 hops, each re-checked against the same host list |
| Content type | `image/jpeg`, `image/png`, `image/webp` |
| Size | 3 MB, enforced while streaming (and on `Content-Length`) |
| Time | 8 s per request, 12 s per screen |
| Decode | pixel-count cap, then re-encoded to a plain JPEG (≤800 px) — metadata dropped; the raw download is never written to disk |
| Request | GET, no cookies, no credentials, fixed User-Agent |
| Source of the URL | the main-process result cache only — the sandbox cannot name a URL to fetch |
| Cache | `data/workspace/tmp/product-images/<sha256>.jpg`, pruned to 300 files |

The product blocks open image files from that cache directory only.
Cards render at once with a placeholder and redraw when the image
lands. Residual risk: an image decoder bug reachable from an
allow-listed CDN; widening the host list widens that surface.

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
