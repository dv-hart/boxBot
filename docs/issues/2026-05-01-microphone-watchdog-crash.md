# Microphone watchdog crashes process when ReSpeaker USB drops

**Date:** 2026-05-01 21:23 (boxbot-000-01)
**Log:** `logs/boxbot-20260501-205728.log` (last lines)
**Severity:** Crash — whole `boxbot` process aborted, no auto-restart.

## Symptom

Process exited with a PortAudio C-level assertion abort. The Python
exception handler in the microphone watchdog never caught it because
PortAudio calls `abort()` directly:

```
ReSpeaker LED USB write failed (USBError: [Errno 5] Input/Output Error). …
Audio stream appears stalled (6.2s since last chunk). Restarting stream (restart #1).
Hardware microphone health: degraded (audio stream stalled for 6.2s)
Expression 'alsa_snd_pcm_drop( stream->capture.pcm )' failed in 'src/hostapi/alsa/pa_linux_alsa.c', line: 3046
ALSA lib confmisc.c:165:(snd_config_get_card) Cannot get card index for 2
Expression 'ret' failed in 'src/hostapi/alsa/pa_linux_alsa.c', line: 1736
Expression 'AlsaOpen( &alsaApi->baseHostApiRep, params, streamDir, &self->pcm )' failed in 'src/hostapi/alsa/pa_linux_alsa.c', line: 1904
python3: src/hostapi/alsa/pa_linux_alsa.c:2178: PaAlsaStream_Initialize: Assertion `self->capture.nfds || self->playback.nfds' failed.
```

## Root cause sequence

1. ReSpeaker USB endpoint glitched — vendor LED write returned
   `USBError [Errno 5] Input/Output Error` (visible in the log a few
   seconds before the stall).
2. Audio callbacks stopped firing for 6.2s. Watchdog at
   `src/boxbot/hardware/microphone.py:317` correctly detected the
   stall, closed the stream, and called `_open_stream()`.
3. While the watchdog was closing/reopening, the ALSA card for the
   ReSpeaker was momentarily gone (`Cannot get card index for 2`).
4. PortAudio's ALSA host API hit
   `assert(self->capture.nfds || self->playback.nfds)` in
   `PaAlsaStream_Initialize` — a hard `abort()` from C, bypassing
   the watchdog's `try/except Exception`.
5. Process died. Nothing restarted it.

When checked later (~30 min after crash), the ReSpeaker was back at
card index 2 and `lsusb` listed it normally — confirming a transient
USB hiccup, not a permanent failure.

## Why the existing watchdog didn't save us

`src/boxbot/hardware/microphone.py:357-373`:

```python
self._close_stream()
try:
    self._open_stream()
    …
except Exception:
    logger.exception("Failed to reopen audio stream after stall; …")
```

This only catches Python exceptions. PortAudio's `assert()` in
`pa_linux_alsa.c:2178` is a C-level abort — Python never sees it.

Two contributing problems:

1. The watchdog calls `sd.InputStream(device=self._device_index, …)`
   with a **stale** device index. If the USB device has flickered
   away and come back at a different ALSA card number, this hits a
   half-present device that triggers the assertion path.
2. There is no presence check before the reopen attempt. Even with a
   stable index, opening immediately after a USB transient is racy.

## Fix (implemented 2026-05-02)

Watchdog rewritten in `src/boxbot/hardware/microphone.py:_watchdog_loop`
into a two-phase recovery:

1. **Detect** — close the stalled stream, drop the stale USB handle
   via the new `_dispose_usb()` helper, mark health DEGRADED, and
   transition to a "waiting for device" state. Crucially, **no
   reopen attempt** here — the ALSA card may still be mid-flicker.
2. **Poll** — every 2s, force PortAudio to re-enumerate ALSA via the
   new `_refresh_portaudio()` helper (`sd._terminate(); sd._initialize()`,
   defensively `getattr`-guarded against future renames), then look
   up the ReSpeaker by name through the existing
   `_find_device_index()`. While it's absent, log at DEBUG and keep
   polling. Once it shows up, re-init USB and only then open a new
   `sd.InputStream`. If the open raises a Python exception, swallow
   and stay in wait mode.

The renamed counter `_stream_recovery_count` reflects what we now do
— recover after the device comes back — rather than the old
"restart" semantics which papered over the C-abort failure mode.

### Still open (not addressed here)

- **Process-level supervisor.** A C abort from any future PortAudio /
  ALSA bug would still kill the process with no auto-restart. The
  watchdog rewrite removes the only known trigger of the abort, but
  doesn't bound the failure mode itself. `scripts/restart-boxbot.sh`
  should be wrapped by a systemd unit with `Restart=on-failure` as
  defense in depth.

## Reproduction

Hard to reproduce on demand — needs a real USB transient on the
ReSpeaker. Approximations:

- `sudo udevadm trigger --action=remove …` then `--action=add` on
  the ReSpeaker bus device.
- Briefly unplug-and-replug the ReSpeaker USB while boxbot is
  running.

Confirm that the watchdog logs a stall and that the process either
recovers (after fix) or aborts with the same assertion (current
behavior).

## Related

- HAL microphone watchdog: `src/boxbot/hardware/microphone.py:317`
- Stream open: `src/boxbot/hardware/microphone.py:290`
- Restart script: `scripts/restart-boxbot.sh` (currently launched
  manually by deploy; not auto-triggered on crash).
