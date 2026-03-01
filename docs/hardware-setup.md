# Hardware Setup

## Bill of Materials

| Component | Part | Purpose |
|-----------|------|---------|
| Compute | Raspberry Pi 5 (8 GB) | Main processor |
| AI Accelerator | Raspberry Pi AI HAT+ (13 TOPS) | On-device ML inference |
| Storage | SanDisk MicroSD Extreme Pro 512 GB | OS + data storage |
| Display | 7" HDMI LCD (H), 1024x600 IPS | Visual output + touch input + speaker audio |
| Camera | Pi Camera Module 3 Wide NoIR (12MP, 120 deg) | Visual perception |
| Microphone | ReSpeaker XMOS XVF3000 4-Mic USB Array | Audio input + AEC + DOA + LED ring |
| Speaker | Waveshare 8ohm 5W | Audio output |
| Input | Adafruit KB2040 (RP2040) | Physical buttons/knobs (optional) |
| Power | Official Pi 27W PD Supply (5.1V/5A) | Power delivery |
| Mounting | M2.5 nylon standoff set | Internal mounting |
| Cooling | Heat sink thermal sticker tabs | Pi 5 thermal management |
| Wiring | 22AWG hookup wire, F-F jumper cables | Internal connections |

**Possible addition:** PAM8302 inline amplifier (~$3) if the display's
3.5mm headphone output is too quiet driving the 8Ω speaker. Test without
it first.

## Assembly Notes

> Detailed assembly guide with photos will be added as the enclosure
> design is finalized.

### Connections Overview

```
Pi 5 GPIO ◄──────► AI HAT+ (sits on top via GPIO header, PCIe)
Pi 5 CSI  ◄──────► Camera Module 3 (ribbon cable)
Pi 5 HDMI ◄──────► 7" LCD (micro-HDMI to HDMI — video + audio)
Pi 5 USB  ◄──────► 7" LCD (USB-A — touch input)
Pi 5 USB  ◄──────► ReSpeaker 4-Mic Array (USB-A — audio in + AEC ref out + LEDs)
Pi 5 USB  ◄──────► KB2040 (USB-C — optional physical controls)
Pi 5 J2   ◄──────► Momentary switch (power button — PMIC)
Display 3.5mm ──► [optional PAM8302 amp] ──► Waveshare Speaker
Pi 5 USB-C ◄─────► 27W PD Supply
```

### Audio Routing (Path C — Dual Output)

boxBot uses a dual-output audio architecture. TTS audio is routed to two
destinations simultaneously via an ALSA virtual device:

1. **HDMI → Display → 3.5mm → Speaker** — actual sound output
2. **USB → ReSpeaker** — silent AEC reference signal (XMOS subtracts
   this from mic input to enable barge-in detection)

This eliminates the need for a separate amplifier board in most cases.
The display decodes HDMI audio and outputs it through its 3.5mm jack.
The ReSpeaker receives the same audio via USB purely as an echo
cancellation reference — nothing is connected to its 3.5mm output.

See [hal.md](hal.md) for the full ALSA configuration and validation
items.

### Power Button

The Pi 5 has a 2-pin header (J2) for an external power button. Wire a
momentary switch to J2 — the PMIC firmware handles power on/off. No
KB2040 needed for this function. boxBot's `system.py` HAL module
intercepts the SIGTERM from PMIC to perform a graceful shutdown.

### Software Prerequisites

- Raspberry Pi OS Bookworm (64-bit)
- Hailo runtime and HailoRT (for AI HAT+)
- picamera2 (pre-installed on Pi OS)
- pygame (for display rendering)
- pyaudio (for ReSpeaker audio capture)
- pixel_ring (for ReSpeaker LED ring)
- See `scripts/setup.sh` for full dependency installation

### Display Setup

The 7" HDMI LCD (H) may need display rotation and resolution config in
`/boot/firmware/config.txt`. Details TBD based on enclosure orientation.

Backlight control method varies by display revision — validate during
hardware bring-up. See [hal.md](hal.md) for details.

### Audio Setup

The ReSpeaker handles echo cancellation and noise suppression on-board
via the XMOS DSP. It appears as a standard USB audio device for both
input (mic) and output (AEC reference).

ALSA configuration for the dual-output virtual device is documented in
[hal.md](hal.md). The exact ALSA card names are determined by running
`aplay -l` and `arecord -l` on the target hardware.

### Touch Screen

The display's USB connection provides capacitive touch input, which
appears as a Linux HID device. pygame captures touch events as mouse
events. No additional driver setup needed on Raspberry Pi OS Bookworm.
