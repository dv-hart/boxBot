# Plan: Fix ReSpeaker LEDs (protocol is wrong + udev rule missing)

## Probe results (2026-04-23)

Ran empirical probes on-device after stopping boxBot. **Two root causes confirmed:**

### Root cause #1 — missing udev rule (FIXED)
Before the fix, every `ctrl_transfer` from `jhart` returned:
```
USBError: [Errno 13] Access denied (insufficient permissions)
```
Device enumeration worked (`usb.core.find` succeeded — explaining the misleading "ReSpeaker USB interface initialized" INFO log at `microphone.py:597`), but any actual I/O failed. boxBot swallowed these errors at DEBUG level at `microphone.py:483-484`, so they never surfaced.

**Fix applied**: installed `/etc/udev/rules.d/60-respeaker.rules`:
```
SUBSYSTEM=="usb", ATTRS{idVendor}=="2886", ATTRS{idProduct}=="0018", MODE="0666", GROUP="plugdev"
```
After `udevadm control --reload-rules && udevadm trigger` the device `/dev/bus/usb/001/002` is now `crw-rw-rw-`.

### Root cause #2 — LED + DOA protocols in our code are wrong (NOT YET FIXED)

After the udev fix, re-running the probe:

| Probe | Request | Result |
|---|---|---|
| **TEST 1: current code's per-LED write** | `ctrl_transfer(0x40, 0x00, 0x03, 0, [0xFF,0,0,255])` | **STALL** — `USBError [Errno 32] Pipe error`. Device actively rejects the format. |
| **TEST 2: Seeed pixel_ring `mono`** | `ctrl_transfer(0x40, 0x00, 1, 0x1C, [255,255,0,0])` | **SUCCESS** — 4 bytes written, device accepts. |
| **TEST 5: current code's DOA read** | `ctrl_transfer(0xC0, 0x00, 0x21, 0, 8)` | **FAIL** — `USBError [Errno 5] I/O Error`. Wrong command encoding. |

So the `microphone.py` code is addressing the wrong protocol entirely. It writes per-LED APA102-style frames to register `0x03+i`, but the Seeed ReSpeaker 4-Mic Array v2.0 firmware expects **command-based writes to `wIndex=0x1C`**. Our existing pattern-rendering code (pulse, chase) was running fine — the animation loop computes frame colors correctly — but every `_set_leds_raw` call ended with a STALL, silently caught.

This also explains why the DOA ring is visible even though `get_doa()` returns `None` when called: the DOA ring is the **XMOS firmware's own built-in reactive animation**, unrelated to our code. Our DOA read is broken, our LED writes are broken; the chip is just running its default behaviour.

## What the correct protocol looks like

Seeed's reference implementation lives in the `respeaker/pixel_ring` Python library. All LED commands use:

```
ctrl_transfer(
    bmRequestType = 0x40,              # vendor OUT
    bRequest      = 0x00,
    wValue        = <command_id>,      # which pattern / operation
    wIndex        = 0x1C,              # pixel_ring interface
    wLength       = len(data),
    data          = <payload bytes>,
    timeout       = 8000,
)
```

Command IDs (confirmed for this device family):
| cmd | meaning | payload |
|---|---|---|
| 0 | trace (firmware-driven, reactive) | `[0]` |
| 1 | mono (all 12 LEDs one colour) | `[R, G, B, 0]` (4 bytes) |
| 2 | listen | `[0]` or `[angle_lo, angle_hi]` for directional |
| 3 | speak | `[0]` |
| 4 | think | `[0]` |
| 5 | spin | `[0]` |
| 6 | show (custom) | 48 bytes = 12 LEDs × RGBA |
| 0x20 | set_brightness | `[brightness & 0xFF]` |

TEST 3/4 in the probe returned `USBTimeoutError` — I passed an empty payload; the firmware expects at least a single byte. Non-empty payload is required even for the pattern commands with no parameters.

Probe-validated: `cmd=1` + 4-byte mono returns success. That proves the protocol; the other commands are consistent per the reference library.

For DOA, the correct read uses the parameter ID scheme (not a raw register address):
```
cmd = 0x80 | 0x40 | 21  # read | int | DOAANGLE id=21
ctrl_transfer(0xC0, 0x00, cmd, 0, 8)
# parse response as int32 LE (angle 0-359) + int32 LE (max value)
```

## Plan: rewrite `microphone.py` LED + DOA internals

### Option A — vendor the protocol ourselves
Rewrite `_set_leds_raw` and `get_doa` in `microphone.py` to use the Seeed protocol. ~50 lines of replacement code. Keeps pyusb as the only dependency.

### Option B — depend on `pixel_ring` / `usb_4_mic_array` packages
Import Seeed's libraries and call their helpers. Fewer lines, but pulls in a third-party dep whose release cadence we don't control.

**Recommendation: Option A.** It's not a lot of code, and we've already vendored the device-specific constants.

### Specific changes

1. **Add a tiny protocol helper** (new section in `microphone.py` near the USB constants):

   ```python
   _PIXEL_RING_IFACE = 0x1C
   _CMD_TRACE = 0
   _CMD_MONO = 1
   _CMD_LISTEN = 2
   _CMD_SPEAK = 3
   _CMD_THINK = 4
   _CMD_SPIN = 5
   _CMD_SHOW = 6
   _CMD_SET_BRIGHTNESS = 0x20

   _PARAM_DOAANGLE_ID = 21
   ```

2. **Rewrite `_set_leds_raw(colors)`** — use `_CMD_SHOW` with 48-byte payload (12 LEDs × RGBA). Preserves existing animation loop behaviour (per-frame full-ring colour updates).

3. **Add `_set_pattern_cmd(cmd, data)`** — a thin helper used for `mono`, `trace`, predefined patterns. Our `_PATTERN_CONFIG` and rendering helpers don't need to change meaningfully — the pulse/chase renderers can keep computing per-LED colours and push them via the new `_set_leds_raw`.

4. **Rewrite `get_doa()`** — use the parameter-ID protocol above. Parse first 4 bytes as little-endian int32.

5. **Keep the udev rule installed**. Also add it to `scripts/setup.sh` / `scripts/harden-os.sh` (or a new `scripts/install-udev-rules.sh`) so fresh Pi installs don't repeat this debugging.

6. **Improve error logging**. `microphone.py:483-484` should log USB write failures at `WARNING` the first time (with request details), then throttle to avoid spam. The silent DEBUG catch is what let this bug hide for weeks.

### Wiring gaps already known (cover these while you're in there)
- `voice.py:388` sets LEDs to `"off"` on session end — change to `"idle"` for ambient warm-amber breathing when BB is awake but quiet. Keep `"off"` only for explicit shutdown.
- Add a DOA overlay when in `"listening"` state: once DOA read works, have the animation loop sample `get_doa()` at ~5 Hz and combine the directional highlight with the blue pulse base. Trivial once the protocol is fixed.

## Files touched
- `src/boxbot/hardware/microphone.py` — the rewrite
- `scripts/setup.sh` or new `scripts/install-udev-rules.sh` — udev rule installation
- `/etc/udev/rules.d/60-respeaker.rules` on Pi — already installed
- `src/boxbot/communication/voice.py:388` — `"off"` → `"idle"` on session end (small)

## Verification
After the rewrite, run the probe test again: commands 1–5 should all succeed and the physical LEDs should change visibly. Then start boxBot, trigger a wake word, and watch:
- IDLE → warm amber breathing
- Wake word heard → blue pulse (listening), with DOA highlight tracking the speaker
- Processing → amber-coral chase (thinking)
- TTS playing → green pulse (speaking)
- Conversation ends → back to idle amber

## Open questions
1. Does `_CMD_SHOW` actually take 48 bytes of RGBA on this device revision? Confirm by writing `[255,0,0,0]*12` and expecting all-red. (Not tested — would need boxBot down again.)
2. Does the XMOS firmware's built-in DOA animation suppress our `_CMD_SHOW` writes, or does ours override? Quick test by writing a solid colour and seeing if it holds.
3. Throttling: writing 30 FPS worth of 48-byte ctrl_transfers generates ~1.5 KB/s of USB control traffic — fine, but worth verifying it doesn't interfere with audio capture on the same device.

## Effort
3-4 hours including testing. Most of the time is in verifying each pattern visually on the device.
