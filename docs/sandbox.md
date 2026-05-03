# Sandbox & Agent Authoring

## Overview

boxBot's agent can write and execute Python scripts, create new skills, and
create new displays. This power is what makes boxBot different from a static
assistant — it can build its own capabilities on the fly.

But code execution needs boundaries. This document defines the sandbox
environment, the trust model, and the authoring rules.

## The Two Venvs

boxBot runs two separate Python virtual environments:

```
┌─────────────────────────────┐  ┌─────────────────────────────┐
│      Main venv (.venv)      │  │   Sandbox venv              │
│                             │  │   (data/sandbox/venv)       │
│  boxBot application code    │  │                             │
│  Hardware drivers            │  │  boxbot_sdk (immutable)     │
│  picamera2, RPi.GPIO        │  │  Agent-written scripts      │
│  hailort, pygame            │  │  Skill script execution     │
│  Claude SDK, pydantic       │  │  General-purpose packages   │
│                             │  │  No hardware access         │
│  ⛔ Agent cannot modify     │  │  ✅ Agent operates here     │
└─────────────────────────────┘  └─────────────────────────────┘
```

**Why separate?**
- Sandbox scripts cannot import hardware drivers or boxBot internals
- The only bridge is the `boxbot_sdk` — an immutable, declarative API
  pre-installed in the sandbox venv
- Package installation in the sandbox never destabilizes boxBot
- The sandbox can be rebuilt without affecting the application
- Clear security boundary: agent code runs in a different environment
  than the code that controls hardware

## The boxbot_sdk

The SDK (`src/boxbot/sdk/`) is installed into the sandbox venv as a
standalone package. It has no dependency on boxBot internals — only
stdlib. Agent scripts `from boxbot_sdk import display, memory, ...`
to interact with the system.

Under the hood, SDK calls emit structured JSON to stdout. The
`execute_script` tool (running in the main process) parses these
actions and applies them safely — creating files, updating the DB,
queuing user approvals, etc. The agent never directly writes code
that runs in the main process.

This is the key constraint for **displays**: instead of writing raw
pygame rendering code, the agent uses `boxbot_sdk.display` to compose
displays from a block library (layout containers, content blocks,
composite widgets). The rendering engine draws the spec using validated
block implementations. The agent says **what** to show; the system
decides **how** to render it.

See `src/boxbot/sdk/README.md` for the full API reference.

## Sandbox Venv: Pre-Installed Packages

The sandbox venv is created by `scripts/setup.sh` and pre-loaded with
packages the agent needs for general-purpose scripting and skill
execution. These are defined in `config/sandbox-requirements.txt`.

### Vanilla Packages

**HTTP & APIs** (essential for skills that call external services):
- `requests` — HTTP client
- `httpx` — async HTTP client
- `beautifulsoup4` — HTML parsing
- `lxml` — XML/HTML parser (bs4 backend)

**Data & Computation:**
- `numpy` — numerical computing
- `python-dateutil` — date/time parsing and manipulation
- `pyyaml` — YAML parsing

**Text & Formatting:**
- `jinja2` — templating (useful for generating display code, reports)
- `markdown` — markdown to HTML conversion

**Images:**
- `Pillow` — image processing (photo tagging, display asset creation)

**Standard Library (no install needed):**
`json`, `csv`, `sqlite3`, `urllib`, `email`, `imaplib`, `html`, `re`,
`math`, `collections`, `itertools`, `datetime`, `pathlib`, `os`,
`subprocess` (restricted), `typing`, `dataclasses`, `asyncio`

### What's NOT Included (and Why)

| Package | Why Not |
|---------|---------|
| `pandas` | ~100MB+ installed, too heavy as default on Pi |
| `matplotlib` | ~40MB, niche — install on demand |
| `scikit-learn` | ~200MB+, way too heavy |
| `torch` / `tensorflow` | Inference runs on Hailo, not in sandbox |
| `picamera2` | Hardware access — main venv only |
| `RPi.GPIO` | Hardware access — main venv only |
| `pygame` | Display rendering — main venv only |

The agent can request additional packages (see Package Installation below).

## Script Execution Model

When the agent calls the `execute_script` tool:

```
1. Agent provides Python source code + description
2. Tool (main process) writes script to data/sandbox/scripts/{uuid}.py
3. Tool builds an allowlist environment (PATH, HOME, LANG, TZ, etc.)
   — no secrets from .env are inherited
4. Tool spawns subprocess with privilege drop:
     sudo -n -u boxbot-sandbox -- data/sandbox/venv/bin/python3 {script_path}
   (Set BOXBOT_SANDBOX_ENFORCE=0 to disable for development)
5. Captures stdout + stderr
5. Parses SDK actions from stdout, applies them in main process
6. Returns regular output + action results to agent
7. Cleans up script file
```

### OS-Level Enforcement

Sandbox security is enforced at the **operating system level**, not
Python level. Python-level restrictions (import hooks, removing modules)
can always be bypassed via `__import__`, `importlib`, `ctypes`, etc.
OS-level enforcement cannot be circumvented from userspace Python.

#### Sandbox User: `boxbot-sandbox`

Sandbox scripts run as a dedicated, unprivileged system user with
minimal permissions. The main boxBot process runs as `boxbot` (or the
installing user). The `execute_script` tool uses `sudo -u boxbot-sandbox`
to drop privileges before executing the script.

```
boxbot (main process user):
  - Owns src/boxbot/, config/, .env
  - Read-write to data/, skills/, displays/
  - Can run pip against the sandbox venv
  - Runs the agent, hardware drivers, display rendering

boxbot-sandbox (sandbox script user):
  - Cannot write to data/sandbox/venv/   (read + execute only)
  - Cannot read .env                     (no permissions)
  - Cannot write to src/boxbot/          (no permissions)
  - Can write to data/sandbox/output/    (owned by boxbot-sandbox)
  - Can write to data/sandbox/tmp/       (owned by boxbot-sandbox)
  - Can write to skills/                 (group-writable for skill creation)
```

#### No Subprocess Spawning: seccomp

Sandbox scripts are launched with a **seccomp filter** that blocks
syscalls outside the sandbox's needs. Currently blocked:

- `execve`, `execveat` — exec family. Blocks `subprocess.run`, `os.system`,
  `os.exec*`, `ctypes` calling exec, every spawn path.
- `fork`, `vfork` — process duplication.
- `clone` without `CLONE_THREAD` — full process clone (Python threading
  uses CLONE_THREAD and is allowed).
- `ptrace` — debugger attach.
- `kexec_load`, `init_module`, `delete_module`, `create_module`,
  `finit_module` — kernel manipulation.
- `mount`, `umount2`, `pivot_root`, `chroot` — filesystem namespace games.
- `swapon`, `swapoff`, `reboot`, `perf_event_open`, `bpf` — system controls.

The only process that runs is the Python interpreter itself, which was
spawned by the main process before seccomp was applied.

##### How it's wired

`scripts/sandbox_bootstrap.py` is the entry point of every sandboxed
Python execution. The bootstrap:

1. Reads `BOXBOT_SECCOMP_MODE` from env (`disabled` | `log` | `enforce`).
2. Imports the libseccomp Python binding (`seccomp` from
   `python3-seccomp`, with PyPI `pyseccomp` as fallback).
3. Installs the BPF filter on its own process. The filter is inherited
   by child processes — there are none, since `clone`/`fork`/`exec` are
   themselves blocked.
4. Hands off to the user script via `runpy.run_path`.

`setup-sandbox.sh` copies the bootstrap from the project tree to
`<runtime_dir>/sandbox_bootstrap.py` and chmods it `750
<operator>:boxbot`. The per-call `execute_script` path execs that copy,
not the in-tree one — the sandbox user typically can't traverse the
operator's home directory (often 0700) to reach the project tree. Re-run
`setup-sandbox.sh` whenever the bootstrap changes; the in-tree copy is
the fallback that's only used in tests/dev where the runtime dir
doesn't exist.

For the long-lived per-conversation runner, the same logic is inlined
at the top of the server script (`src/boxbot/tools/_sandbox_server.py`,
prepended at host side by `sandbox_runner.py`) — it can't import the
bootstrap module because the sandbox user may not have read access to
the project tree.

##### Modes

- `disabled` — no filter applied. Currently the on-disk default for
  `SandboxConfig.seccomp_mode` until the operator confirms enforcement.
- `log` — filter installed with `SCMP_ACT_LOG`. The kernel logs every
  forbidden syscall to the audit log / dmesg, but does **not** kill the
  process. This is the soak mode: run real workloads for days, scrape
  the kernel log, see which (if any) unexpected syscalls show up.
- `enforce` — filter installed with `SCMP_ACT_KILL_PROCESS`. The first
  forbidden syscall kills the process with SIGSYS. Production setting.

##### Recommended rollout

1. Run `setup-sandbox.sh` (installs `python3-seccomp` via apt).
2. Set `memory.sandbox.seccomp_mode: log` in `config/config.yaml` (the
   shipped default), restart boxBot.
3. Use BB normally for several days. Skills, voice, WhatsApp work
   exactly as before — `log` mode never blocks.
4. Inspect the audit log for SECCOMP entries:

       sudo journalctl -k -g 'seccomp\|SCMP' --since '2 days ago'
       # or, if auditd is running:
       sudo ausearch -m SECCOMP -i

   Each entry shows the syscall name and process. Anything unexpected
   in the list = a real workload needs that syscall, and it should
   either be added to the allow set or the script changed.
5. When the log goes quiet (only the deliberate blocks remain), flip
   `seccomp_mode: enforce` and restart.

##### Kill switch

If `enforce` mode breaks something unexpectedly, set
`BOXBOT_SECCOMP_DISABLE=1` in the environment and restart boxBot. The
bootstrap honours it before applying any filter, regardless of config.
No code change needed; recovery is just an env var.

#### Filesystem Permissions

```
READ + EXECUTE (sandbox user can read):
  data/sandbox/venv/lib/       → installed packages (read-only!)
  data/sandbox/venv/bin/python3 → the interpreter itself
  config/                      → configuration (read-only)
  data/photos/, data/photos/photos.db, data/memory/ → project data (read-only)
  data/scheduler/scheduler.db            → trigger and to-do data (read-only)
  skills/                      → existing skill code
  displays/                    → existing display code

WRITE (sandbox user can write):
  data/sandbox/output/         → script outputs
  data/sandbox/tmp/            → working space
  skills/                      → new skill directories (group-writable)

NO ACCESS (sandbox user has no permissions):
  .env                         → secrets (mode 0600, owned by boxbot)
  src/boxbot/                  → core code (owned by boxbot)
  data/sandbox/venv/lib/       → site-packages (read-only to sandbox)
  data/sandbox/venv/bin/pip    → pip binary (not executable by sandbox)
  .git/                        → repository internals
  /etc, /usr, ~                → system directories
```

#### pip Is Inaccessible from the Sandbox

The sandbox cannot install packages because:

1. **seccomp blocks execve** — `subprocess.run(["pip", ...])` is killed
   by the kernel
2. **pip binary is not executable** — `chmod 0700` owned by `boxbot`,
   so `boxbot-sandbox` cannot execute it
3. **site-packages is read-only** — owned by `boxbot`, the sandbox user
   cannot write files into it. Even if Python's `pip` module were
   somehow importable, it could not write the installed package
4. **No write to venv directory** — the entire `data/sandbox/venv/` tree
   is read-only to `boxbot-sandbox` (except executing the interpreter)

All four vectors must be defeated for a package to be installed. Each
one independently prevents it.

### Resource Limits

| Limit | Default | Enforcement |
|-------|---------|------------|
| Timeout | 30 seconds | Main process kills after deadline |
| Memory | 256 MB | `ulimit -v` / cgroup |
| CPU | Single core | OS scheduling |
| Disk write | writable_paths only | Filesystem permissions |
| Network | Allowed | (needed for API calls) |
| Processes | None | seccomp blocks fork/exec |

### Environment Variables

Sandbox scripts receive an **allowlist-only** environment. The parent
process's environment is NOT inherited — only these safe variables are
passed through:
- `PATH`, `HOME`, `LANG`, `LC_ALL`, `LC_CTYPE`, `TZ`
- `PYTHONPATH`, `VIRTUAL_ENV`, `PYTHONDONTWRITEBYTECODE`, `PYTHONUNBUFFERED`
- Specific API keys needed for the current skill (passed explicitly via
  the tool's `env_vars` parameter, not from .env)

All secrets (`ANTHROPIC_API_KEY`, `WHATSAPP_ACCESS_TOKEN`,
`ELEVENLABS_API_KEY`, AWS keys, etc.) are excluded by default because
the allowlist does not include them. This is safer than a blocklist,
which must be updated every time a new secret is added.

## Package Installation

**The sandbox cannot install packages.** This is enforced by the OS, not
by policy. The only entity that can install packages into the sandbox
venv is the **main boxBot process**, and it will only do so after
receiving **out-of-band human approval**.

### How It Actually Works

```
Agent script calls packages.request("google-api-python-client")
         │
         ▼
SDK emits JSON to stdout:
  {"_sdk": "packages.request", "package": "google-api-python-client",
   "reason": "Gmail API access"}
         │
         ▼
execute_script tool (main process) sees the request
         │
         ▼
Script finishes or is suspended — the agent CANNOT proceed
until the main process resolves the request
         │
         ▼
Main process initiates approval on TWO out-of-band channels:
  ┌──────────────────────────────────────────────────────┐
  │  1. DISPLAY: Show approval prompt on 7" screen       │
  │     "boxBot wants to install: google-api-python-..."  │
  │     [ YES ]  [ NO ]    ← physical tap on touchscreen │
  │                                                      │
  │  2. WHATSAPP: Message to primary admin user          │
  │     "boxBot wants to install google-api-python-..."   │
  │     Reply YES to approve.                            │
  │     ← only accepted from the admin's phone number    │
  └──────────────────────────────────────────────────────┘
         │
         ▼
Main process waits for approval from EITHER channel
(timeout after configurable period, default: denied)
         │
         ├─ Approved ──► Main process (as boxbot user) runs:
         │               data/sandbox/venv/bin/pip install <package>
         │               Logs to data/sandbox/installed.txt
         │               Returns success to agent
         │
         └─ Denied ────► Returns denial to agent
                         Agent must find alternative approach
```

### Why the Agent Cannot Spoof Approval

The approval flow is **out-of-band** — it uses input channels that the
sandbox has no access to:

| Spoofing Vector | Why It Fails |
|----------------|-------------|
| Forge "approved" JSON on stdout | The main process only accepts `packages.request` actions (a REQUEST, not an approval). There is no SDK action that means "already approved." The approval comes from a different input channel entirely |
| Call pip directly | seccomp blocks execve; pip not executable by sandbox user; site-packages read-only |
| Write to site-packages | Owned by `boxbot`, read-only to `boxbot-sandbox` |
| Inject "YES" into WhatsApp | WhatsApp approval validates the sender's phone number matches the admin. Sandbox has no access to WhatsApp API credentials |
| Simulate screen tap | Physical touchscreen input. Sandbox is a subprocess with no display access |
| Import pip module | Even if importable, pip cannot write to site-packages (filesystem permissions) |
| Download .whl and extract | Cannot write to site-packages |
| Trick main process parser | SDK actions are schema-validated. Unknown action types are rejected. Only `packages.request` is valid for packages |

The fundamental principle: **approval comes from a human through a
physical or authenticated channel that is completely separate from the
sandbox's stdout/stderr**. The sandbox's only output channel is stdout,
which can only emit requests, never approvals.

### Installed Package Audit

Every installation is logged to `data/sandbox/installed.txt`:
```
2026-02-20T14:30:00 google-api-python-client==2.x.x approved=Jacob channel=whatsapp
2026-02-20T15:00:00 feedparser==6.x.x approved=Jacob channel=display_tap
```

This log is append-only (owned by `boxbot`, not writable by sandbox)
and lets users see exactly what's been added and how it was approved.

## Agent Authoring: Skills vs Displays

The agent creates skills and displays through the `boxbot_sdk`, not by
writing raw files. The SDK provides constrained builder APIs that validate
inputs and generate correct output.

### Creating Skills (Lower Risk)

Skills are structured prompt data — markdown instructions BB reads on
demand, optionally bundled with helper scripts. They are *not* callable
functions with parameters. Data pipelines, scheduled fetchers, and
service connectors belong in **integrations** instead (see
[docs/integrations.md](integrations.md) for the paradigm and
`integrations/` at the repo root for the bundles); a skill points
at the relevant integration when one exists.

The agent writes a script that uses `boxbot_sdk.skill`:

```python
from boxbot_sdk import skill

s = skill.create("weather")
s.description = (
    "Get NOAA weather forecasts for the configured location. "
    "Use when the user asks about weather, temperature, rain, or conditions."
)
s.body = """
# Weather

Use `bb.weather.forecast(days=N)` to get an N-day forecast. For hourly
detail, see HOURLY.md. For raw NWS request structure, see scripts/nws_raw.py.
"""
s.add_resource("HOURLY.md", "# Hourly forecast\n...")  # Level 3 sub-doc
s.add_script("nws_raw.py", "import requests\n...")     # bundled helper
s.save()
```

The SDK emits a structured `skill.save` action. The main-process handler
generates the skill directory:

```
skills/
  weather/
    SKILL.md          # frontmatter (name, description) + body markdown
    HOURLY.md         # optional Level 3 sub-doc
    scripts/
      __init__.py     # auto-written so `from skills.weather.scripts import nws_raw` works
      nws_raw.py      # the helper the agent provided
```

`SKILL.md` is the canonical descriptor — frontmatter plus body, matching
Anthropic's Agent Skills format. `discover_skills()` scans for it on
startup. There is no `skill.yaml` and no generated `__init__.py` at the
skill root; skills are read by BB as documentation, not imported as
Python modules.

**Policy:** `auto_activate_skills: true` — agent-created skills activate
immediately because their logic is sandboxed and the loader picks them
up on the next discovery scan.

**Permissions:** `skills/` files are written by the main process owned
`boxbot:boxbot` mode `0644`. The sandbox can read but not modify them
after save, so a later sandbox script cannot tamper with what an
earlier one wrote.

### Creating Displays (JSON Specs)

Displays run in the main process, so the agent **cannot write arbitrary
display code**. The display SDK exposes a tiny CRUD interface over JSON
spec dicts — the agent reads, edits, and writes the dict like any other
data file. Layout containers handle positioning, content blocks handle
rendering, themes handle styling.

```python
from boxbot_sdk import display

spec = {
    "name": "stock_tracker",
    "theme": "boxbot",
    "data_sources": [{
        "name": "stocks", "type": "http_json",
        "url": "https://api.example.com/quotes",
        "secret": "stock_api_key",
        "refresh": 60,
        "fields": {
            "symbol": "ticker",
            "price": "lastPrice",
            "trend_icon": {"from": "direction",
                           "map": {"up": "trending-up",
                                   "down": "trending-down"}},
        },
    }],
    "rotate": {"source": "stocks", "key": "quotes", "interval": 20},
    "layout": {"type": "column", "padding": [16, 24], "children": [
        {"type": "row", "align": "center", "children": [
            {"type": "text", "content": "{current.symbol}",
             "size": "title", "weight": "bold"},
            {"type": "spacer"},
            {"type": "metric", "value": "${current.price}",
             "change": "{current.change_pct}%", "animation": "count_up"},
        ]},
        {"type": "chart", "data": "{current.history}",
         "type": "area", "color": "accent", "height": 300},
    ]},
}

display.preview(spec)    # render to PNG, attached to the tool result
display.save(spec)       # validate, write, register live
```

The agent never produces render code — the spec is purely declarative.
The dispatcher validates against the block schema; the rendering engine
draws each block from a fixed registry. The agent describes **what** to
show; the system decides **how** to render it.

The block library includes 7 layout containers (`row`, `column`,
`columns` with ratio-based widths, `card`, `spacer`, `divider`,
`repeat`), 13 content blocks (`text`, `metric`, `chart`, `icon`,
`list`, `table`, etc.), 2 composite widgets (`weather_widget`,
`calendar_widget`), and 2 meta blocks (`rotate`, `page_dots`).

Data sources are declared in the spec and fetched by the display
manager on a schedule — no scripts or cron needed. Built-in sources
(`clock`, `weather`, `calendar`, `tasks`, `people`, `agent_status`)
deliver display-ready data including resolved icon names and color
tokens. Custom sources (`http_json`, `http_text`, `memory_query`,
`static`) support `fields` with `map` transforms for declarative
value → icon/color mapping without conditional logic.

See [display-system.md](display-system.md) for the complete design.

**Policy:** displays auto-activate. The render spec is purely
declarative (a tree of validated blocks bound to data sources) — no
executable code reaches the main process, so there is nothing to gate.
``save()`` writes the spec and registers it live; the agent can call
``switch_display(name)`` immediately.

### Why This Is Safer Than Raw Code

| | Old Approach | SDK Approach |
|---|-------------|-------------|
| **Skills** | Agent writes `__init__.py` | Agent provides script + metadata via SDK; main process generates `__init__.py` from template |
| **Displays** | Agent writes Display subclass with raw `render()` | Agent composes from block library; rendering engine draws validated blocks. Agent previews result before saving |
| **Boundary** | Agent writes files that run in main process | Agent only writes sandbox scripts; SDK actions are validated and applied by main process |

The agent cannot inject code into the main process. It can only:
1. Write sandbox scripts (run in isolated venv)
2. Emit SDK actions (validated and applied by the main process)

### What the Agent CANNOT Do

- Write files in `src/boxbot/` — core application code is off-limits
- Write raw Python that runs in the main process
- Bypass the SDK to write display `render()` methods directly
- Modify `.env`, config/*.yaml, or .git/
- Create systemd units, cron jobs, or other OS-level changes
- Modify the SDK itself (installed in read-only site-packages)

## Putting It Together: The Gmail Example

User: "I want you to check my email" → sends credentials via WhatsApp.

1. **Agent writes a script** that uses the SDK:
   ```python
   from boxbot_sdk import skill, packages
   packages.request("google-api-python-client",
                     reason="Gmail API access")
   # ... rest of skill creation (see above) ...
   ```
2. **Package approval:** `execute_script` tool intercepts the SDK action,
   asks user "Install google-api-python-client?" → user confirms via voice
3. **Skill creation:** `execute_script` tool applies the `skill.save()`
   action — generates `skills/check_gmail/` from template
4. **Credential storage:** agent stores credentials encrypted in `data/`,
   skill's `env_var` declarations ensure they're injected at runtime
5. **Skill auto-activates** (logic is sandboxed)
6. **Ongoing:** agent can now check gmail on schedule or on request

The user was involved at exactly one point: approving the package install.
Everything else is autonomous and constrained by the SDK.

## Rebuilding the Sandbox

If the sandbox gets corrupted or needs a reset:

```bash
# Recreate sandbox venv from scratch
scripts/setup-sandbox.sh

# This preserves installed.txt so you can see what was added
# and optionally reinstall user-requested packages
```
