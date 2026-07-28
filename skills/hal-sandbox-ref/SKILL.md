---
name: hal-sandbox-ref
description: Sandbox execution model for boxbot_sdk scripts — what execute_script can and cannot reach, and where the per-module API docs live.
when_to_use: Writing a script via execute_script and you need the security boundary — what's importable, what's blocked, why a call was refused. For API detail, load the `bb` skill instead.
---

# Sandbox execution model

`boxbot_sdk` (importable as `bb`) is preinstalled and immutable. It is
the **only** way an `execute_script` script affects boxBot. `boxbot.*`
core imports are absent from the sandbox venv — you cannot reach
internals, only declare intent through the SDK.

The SDK never acts directly. It emits structured JSON actions on
stdout; the main process parses, validates, and applies them. You never
think about the transport.

**API docs live in the `bb` skill** — `load_skill(name="bb")` for the
module index, then `subpath="modules/<x>.md"` for depth. This page is
the boundary, not the reference.

## Cannot

- Import `boxbot.*` — not in the sandbox venv.
- Spawn subprocesses — seccomp blocks `execve` and `fork`. Import
  bundled scripts instead of shelling out.
- Read `.env` or on-disk secrets — mode 0600, different owner.
- Install packages — `site-packages` is read-only. Use
  `bb.packages.request()`; an admin approves out-of-band and there is no
  way to spoof approval from inside.
- Read a stored secret back — `bb.secrets` is write-only. Values reach
  scripts only as `BOXBOT_SECRET_<NAME>` env vars.
- Write raw display render code — display specs are declarative block
  trees, validated in the main process.
- Modify a saved skill or integration — files are `boxbot:boxbot` 0644
  after save. Delete and recreate.

## Can

Everything in `bb`: `workspace`, `camera`, `photos`, `audio`,
`display`, `memory`, `tasks`, `auth`, `skill`, `integrations`,
`packages`, `secrets` — plus stdlib and whatever is installed in the
sandbox venv (`httpx`, `requests`, …).

Images from `bb.workspace.view`, `bb.camera.capture`, `bb.photos.view`,
and `bb.display.preview` attach to the tool result, up to 8 per call.

## Error contract

Mutating calls raise `bb.ActionError` when the main process rejects
them — a failed write never looks like a success. Read calls return
response dicts; a failing *run* (an integration returning
`status: "error"`) is data to inspect, not an exception.

## Further reading

- `bb` skill — the SDK's own docs, module by module
- `docs/sandbox.md` — the full security model
- `src/boxbot/sdk/README.md` — SDK internals
- `docs/memory.md`, `docs/display-system.md`
