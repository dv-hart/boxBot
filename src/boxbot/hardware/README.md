# hardware/

The hardware abstraction layer (HAL). All hardware access goes through
this module — no other module touches GPIO, I2C, camera, or audio
devices directly.

This separation allows:
- Core logic to be tested without hardware (mock the HAL)
- Hardware drivers to be swapped without changing business logic
- Clear ownership of device lifecycle (open/close, error recovery)

**Full design spec:** [docs/hal.md](../../../docs/hal.md) — audio routing
architecture, module interfaces, contention management, configuration
reference, error handling, and validation items.

## Base Protocol

All modules implement `HardwareModule`:
- `async start()` — initialize hardware
- `async stop()` — release hardware
- `async health_check()` → `HealthStatus` (OK, DEGRADED, ERROR, STOPPED)
- `is_available` / `is_started` properties

Modules emit health change events to the internal event bus. Upper layers
use these to degrade gracefully — no single hardware failure takes down
the whole system.

## Files

### `base.py`
`HardwareModule` abstract base class, `HealthStatus` enum, shared types
(`AudioChunk`, `ModelInfo`, `SystemHealth`).

### `camera.py`
Pi Camera Module 3 Wide NoIR interface via picamera2:
- **Dual-stream:** low-res (320×240) always-on for CPU motion detection,
  main stream (1280×720) on-demand for YOLO and ReID
- **Full-res photo capture:** 12MP still capture for the photo library
  via picamera2 mode switch (~100-200ms)
- NoIR capability for near-IR (with illuminator, not in v1 BOM)
- 120° horizontal FOV (Wide variant)

### `microphone.py`
ReSpeaker XMOS XVF3000 4-Mic USB Array:
- **Capture:** 6 channels at 16-bit 16kHz (4 raw mic + 2 processed).
  HAL extracts the configured output channel (default: channel 0,
  processed/beamformed mono) for downstream consumers
- **Audio fan-out:** registered consumer pattern — multiple async
  consumers receive independent copies of each audio chunk. Voice
  session state machine registers/deregisters consumers per state
  (IDLE: wake word only → ACTIVE: VAD + accumulator + diarization)
- **DOA:** direction of arrival from 4-mic array via USB vendor
  commands (pyusb). ±15° accuracy
- **LED ring:** 12× APA102 RGB LEDs — pattern-based API via USB vendor
  commands (`set_led_pattern("listening")`,
  `set_led_pattern("doa", {"angle": 90})`). Internal async animation
  loop at ~30 FPS for animated patterns
- XMOS DSP handles AEC, noise suppression, beamforming, AGC on-chip

### `speaker.py`
Waveshare 8ohm 5W speaker via dual-output ALSA routing:
- **Path C architecture:** TTS audio routes to BOTH HDMI (→ display
  3.5mm → speaker for sound) AND ReSpeaker USB (→ XMOS AEC reference,
  no physical output). ALSA virtual device handles duplication
- Streaming playback for low-latency TTS (`play_stream()`)
- Volume control with smooth fading for barge-in graduated yielding
- See [docs/hal.md](../../../docs/hal.md) for ALSA configuration

### `screen.py`
7" HDMI LCD (H) 1024×600 IPS with capacitive touch:
- **Display:** pygame surface management, flip-to-screen
- **Touch:** USB HID input captured as pygame events, published to
  event bus as `touch_down(x, y)` / `touch_up(x, y)` / `touch_move(x, y)`.
  Display manager and approval system register touch regions
- **Backlight:** `set_brightness()` interface with no-op fallback if
  hardware doesn't support software control (validate during bring-up)

### `hailo.py`
Raspberry Pi AI HAT+ (Hailo-8L, 13 TOPS):
- **Raw inference:** `infer(model_name, input_tensor)` → output tensors.
  Callers handle pre/post-processing (NMS, embedding normalization).
  HAL doesn't know about YOLO or ReID — it runs tensors through models
- **Priority contention:** `inference_session(priority="realtime"|"batch")`
  context manager. Realtime (perception) always preempts batch (photo
  intake). Multi-step sessions hold the lock to prevent interleaving
- **Model management:** load/unload HEF models, query input/output shapes
- Both models (YOLOv5s-personface + RepVGG-A0) fit in on-chip memory simultaneously

### `buttons.py`
Adafruit KB2040 (RP2040) input controller — **optional:**
- USB serial communication with CircuitPython firmware
- JSON event protocol: button press/release, encoder rotation
- Published to event bus. Module gracefully handles device absence
- Touch screen covers the critical input path for v1

### `system.py`
Cross-cutting hardware concerns:
- **Thermal monitoring:** SoC temperature (sysfs) + Hailo temperature
  (HailoRT). Configurable warning/throttle/critical thresholds.
  Emits `thermal_warning` and `thermal_critical` events
- **Graceful shutdown:** intercepts SIGTERM/SIGINT (from Pi 5 power
  button via PMIC). Runs registered shutdown handlers with timeout,
  then executes HAL shutdown sequence
- **Health aggregation:** `get_system_health()` returns temperatures,
  memory/disk usage, and per-module health status

## Hardware Notes

- All modules expose async interfaces for non-blocking operation
- Each module handles its own error recovery (USB disconnect, driver
  errors) and emits health events
- Configuration lives under `hardware:` in `config.yaml`
- See [docs/hal.md](../../../docs/hal.md) for the complete spec
