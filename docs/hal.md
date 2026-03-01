# Hardware Abstraction Layer (HAL)

## Overview

The HAL (`src/boxbot/hardware/`) is the only module that touches hardware
directly. No other part of boxBot imports `picamera2`, `pyaudio`,
`hailort`, or writes to GPIO. This strict boundary enables:

- **Testability** — mock the HAL to run the full stack without hardware
- **Swapability** — change a sensor or actuator without touching business
  logic
- **Clear lifecycle** — each device is opened, monitored, and closed in
  one place

The HAL exposes async interfaces. Long-running hardware operations (audio
streaming, frame capture, inference) never block the event loop.

## Hardware Inventory

| Module | Hardware | Interface | Role |
|--------|----------|-----------|------|
| `camera.py` | Pi Camera Module 3 Wide NoIR | CSI-2 | Visual perception, photo capture |
| `microphone.py` | ReSpeaker XMOS XVF3000 4-Mic Array | USB Audio + USB Vendor | Audio input, DOA, LED ring |
| `speaker.py` | Waveshare 8ohm 5W (via display 3.5mm) | ALSA (HDMI + USB) | Audio output, AEC reference |
| `screen.py` | 7" HDMI LCD (H) 1024x600 IPS | HDMI + USB HID | Visual output, touch input |
| `hailo.py` | Raspberry Pi AI HAT+ (Hailo-8L, 13 TOPS) | PCIe (via GPIO header) | ML inference |
| `buttons.py` | Adafruit KB2040 (RP2040) | USB Serial | Physical controls (optional) |
| `system.py` | Pi 5 SoC, PMIC, thermals | sysfs, signals | Thermal monitoring, graceful shutdown |

## Base Protocol

All HAL modules implement a common async lifecycle:

```python
class HardwareModule(ABC):
    """Base for all HAL modules."""

    name: str                     # "camera", "microphone", etc.

    @abstractmethod
    async def start(self) -> None:
        """Initialize hardware. Called during system startup.
        May raise HardwareUnavailableError if device not found."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Release hardware. Called during system shutdown.
        Must be safe to call even if start() was never called."""
        ...

    async def health_check(self) -> HealthStatus:
        """Return current health. Default: OK if started.
        Override for device-specific checks (temperature, USB connected)."""
        return HealthStatus.OK if self._started else HealthStatus.STOPPED

    @property
    def is_available(self) -> bool:
        """Whether the hardware is connected and responsive."""
        ...

    @property
    def is_started(self) -> bool:
        """Whether start() has completed successfully."""
        ...
```

```python
class HealthStatus(Enum):
    OK = "ok"                 # operating normally
    DEGRADED = "degraded"     # functional with reduced capability
    ERROR = "error"           # not functional, may recover
    STOPPED = "stopped"       # intentionally stopped
```

Every module emits events on the internal event bus when health changes:
`hardware_health(module="camera", status=HealthStatus.ERROR, detail="USB
disconnect detected")`. Upper layers use these to degrade gracefully.

## Startup and Shutdown

### Startup Order

Order matters — some modules depend on others being ready.

```
1. system.py     — register signal handlers, start thermal monitoring
2. screen.py     — initialize display (needed for bootstrap code, splash)
3. hailo.py      — load ML models (~2-3s, runs in background)
4. camera.py     — start low-res streaming for motion detection
5. microphone.py — open audio stream, start wake word listener
6. speaker.py    — ready for TTS output
7. buttons.py    — optional, last (non-critical)
```

The startup sequence is managed by `core/startup.py` (not by the HAL
modules themselves). Each module's `start()` is called in order. If a
non-critical module fails (buttons, LED ring), startup continues with
a warning. If a critical module fails (screen, microphone), the system
enters a degraded mode and logs the failure.

### Shutdown Order

Reverse of startup. `system.py` coordinates:

```
1. buttons.py    — close serial
2. speaker.py    — stop playback, close audio output
3. microphone.py — stop audio stream, turn off LEDs
4. camera.py     — stop streaming, release camera
5. hailo.py      — unload models, release device
6. screen.py     — clear display, release pygame
7. system.py     — final cleanup
```

`system.py` registers handlers for SIGTERM and SIGINT. On signal:

1. Emit `shutdown_requested` event
2. Upper layers flush state (pending memory extraction, DB writes)
3. Call `stop()` on each module in shutdown order
4. Exit cleanly

This ensures the system handles power button presses (Pi 5 J2 header →
PMIC → SIGTERM) and thermal shutdowns gracefully.

---

## Module Specifications

### `camera.py` — Pi Camera Module 3 Wide NoIR

#### Hardware

- **Sensor:** 12MP Sony IMX708 (4608 × 2592 max)
- **Lens:** 120° horizontal FOV (Wide variant)
- **Filter:** NoIR — no infrared filter, can see near-IR (~850nm)
- **Interface:** CSI-2 ribbon cable (reliable, not USB)
- **Library:** picamera2

#### Capabilities

The camera operates in two modes simultaneously using picamera2's
dual-stream configuration:

**Low-resolution stream (always running in DORMANT):**
CPU motion detection at 5-10 FPS. Frame differencing on 320×240
grayscale frames costs ~1ms per frame on a single A76 core. This
stream runs continuously and triggers the perception cascade.

**Main stream (on-demand for perception):**
1280×720 frames for YOLO person detection and ReID embedding
extraction. Captured only when motion detection triggers the
CHECKING state or when periodic presence heartbeats fire.

**Still capture (on-demand for photos):**
Full 12MP (4608×2592) capture for the photo library. picamera2's
`switch_mode_and_capture_array()` temporarily switches to the still
configuration, captures one frame, and returns to the preview
configuration. This brief mode switch (~100-200ms) is acceptable
because photo capture is infrequent and non-time-critical.

#### Interface

```python
class Camera(HardwareModule):
    name = "camera"

    async def start(self) -> None:
        """Initialize picamera2 with dual-stream preview configuration.
        lores: 320x240 (motion detection)
        main:  configured main_resolution (default 1280x720)
        """

    async def stop(self) -> None:
        """Stop all streams, release camera."""

    # ── Low-res stream (motion detection) ────────────────────────

    async def get_lores_frame(self) -> np.ndarray:
        """Get latest low-res frame. Non-blocking — returns most recent
        buffered frame. Shape: (240, 320) grayscale or (240, 320, 3) BGR.
        Used by CPU motion detection at 5-10 FPS."""

    # ── Main stream (perception) ─────────────────────────────────

    async def capture_frame(self) -> np.ndarray:
        """Capture a single frame from the main stream.
        Shape: (720, 1280, 3) BGR at default resolution.
        Used by YOLO person detection and ReID. On-demand, not
        continuously streaming at this resolution."""

    # ── Still capture (photo library) ─────────────────────────────

    async def capture_photo(self) -> np.ndarray:
        """Capture a full-resolution still image.
        Temporarily switches to still configuration (up to 12MP),
        captures one frame, then returns to preview configuration.
        Returns: (H, W, 3) BGR numpy array at photo_resolution.
        Used by photo intake pipeline. ~100-200ms mode switch."""

    # ── Properties ────────────────────────────────────────────────

    @property
    def main_resolution(self) -> tuple[int, int]:
        """Current main stream resolution (width, height)."""

    @property
    def photo_resolution(self) -> tuple[int, int]:
        """Still capture resolution (width, height)."""

    @property
    def is_streaming(self) -> bool:
        """Whether the low-res stream is active."""
```

#### Configuration

```yaml
hardware:
  camera:
    main_resolution: [1280, 720]     # main stream for perception
    photo_resolution: [4608, 2592]   # full-res still capture (12MP)
    lores_resolution: [320, 240]     # motion detection stream
    scan_fps: 5                      # low-res stream FPS
    rotation: 180                    # 0, 90, 180, 270 (enclosure dependent)
```

**Lores stream format note:** picamera2 delivers the lores YUV420 stream
with plane-padded dimensions (e.g., 360×384 for a 320×240 request). The
camera module crops to the configured resolution before returning frames
to consumers.

#### Notes

- **Why not capture at 12MP and downsize for YOLO?** Wasteful. YOLO runs
  at 640×640 input. Capturing at 1280×720 and letting the Hailo input
  pipeline resize is efficient. The 12MP capture path is only for photos.
- **NoIR capability:** Without an IR filter, the camera can see near-
  infrared light. An IR illuminator (not included in v1 BOM) would enable
  low-light person detection. The camera works in normal lighting without
  any IR source — the NoIR designation just means it won't block IR if
  present.
- **picamera2 dual-stream:** The `lores` and `main` streams run from a
  single sensor readout. The ISP (Image Signal Processor) on the Pi 5
  handles the downscaling — no CPU cost for the low-res stream.

---

### `microphone.py` — ReSpeaker XMOS XVF3000 4-Mic Array

#### Hardware

- **Mics:** 4× MEMS microphones in a circular array
- **DSP:** XMOS XVF3000 — AEC, noise suppression, beamforming, AGC
- **LEDs:** 12× APA102 RGB LEDs in a ring (addressable)
- **DOA:** Direction of arrival estimation from the 4-mic array
- **Interface:** USB Audio (audio stream) + USB Vendor (DOA, LEDs)
- **Library:** pyaudio (audio), usb.core / pixel_ring (DOA, LEDs)

The XMOS chip processes audio in real-time (~10-20ms latency). The audio
that reaches our software is already cleaned — AEC has subtracted BB's
own speaker output, noise suppression has filtered non-speech sounds, and
beamforming has focused on the dominant source. Our VAD and diarization
operate on this cleaned signal.

#### Audio Fan-Out

Multiple consumers need the same audio stream simultaneously. The
microphone module uses a **registered consumer** pattern:

```
microphone.py (audio stream)
    │
    ├── wake_word.py      (always registered)
    ├── vad.py            (registered during SESSION ACTIVE)
    ├── audio_capture.py  (registered during SESSION ACTIVE)
    └── diarization.py    (registered during SESSION ACTIVE)
```

The voice session state machine manages registration:
- **IDLE:** only wake_word is registered
- **SESSION ACTIVE:** VAD, audio capture, and diarization are added
- **SESSION SUSPENDED:** all removed except wake_word
- **SESSION ENDED:** all removed except wake_word

Each consumer receives an independent copy of every audio chunk.
Consumers are async callables — the microphone fires them concurrently
via `asyncio.gather()`. A slow consumer does not block others.

#### LED Ring

The 12 APA102 LEDs share the same USB device as the mic array. The
`pixel_ring` library (from Seeed) provides the low-level interface. The
microphone module exposes a high-level pattern API.

**Patterns:**

| Pattern | Effect | Triggered By |
|---------|--------|-------------|
| `off` | All LEDs off | Idle state, night mode |
| `idle` | Single dim LED at 12 o'clock | DORMANT state (optional) |
| `listening` | Gentle blue pulse (breathe) | Wake word detected, SESSION ACTIVE |
| `thinking` | Amber chase/spin | Agent processing (waiting for response) |
| `speaking` | Steady warm white | TTS playback active |
| `doa` | Highlight LED nearest to speaker direction | During conversation (ambient) |
| `error` | Red flash | Hardware error, agent error |

Pattern animations (pulsing, chasing) are driven by an internal async
loop in the microphone module. The loop runs at ~30 FPS while an
animated pattern is active, and sleeps when the pattern is static or off.
Consumers set the pattern — they don't drive individual frames.

#### Interface

```python
class Microphone(HardwareModule):
    name = "microphone"

    async def start(self) -> None:
        """Open audio stream, initialize DOA and LED ring.
        Begins producing audio chunks immediately."""

    async def stop(self) -> None:
        """Stop audio stream, turn off LEDs, release USB device."""

    # ── Audio fan-out ─────────────────────────────────────────────

    def add_consumer(
        self,
        callback: Callable[[AudioChunk], Awaitable[None]],
        name: str = ""
    ) -> None:
        """Register an audio consumer. Receives a copy of every chunk."""

    def remove_consumer(
        self,
        callback: Callable[[AudioChunk], Awaitable[None]]
    ) -> None:
        """Unregister an audio consumer."""

    # ── DOA ───────────────────────────────────────────────────────

    def get_doa(self) -> int | None:
        """Current direction of arrival in degrees (0-359).
        0 = forward (configurable via doa_forward_angle).
        None if no active speech source detected."""

    # ── LED ring ──────────────────────────────────────────────────

    def set_led_pattern(
        self,
        pattern: str,
        params: dict | None = None
    ) -> None:
        """Set the LED ring pattern.

        Args:
            pattern: "off", "idle", "listening", "thinking",
                     "speaking", "doa", "error"
            params:  Pattern-specific parameters:
                     doa: {"angle": int}
                     listening: {"color": (r, g, b)}
                     Brightness is applied globally from config.
        """

    # ── Properties ────────────────────────────────────────────────

    @property
    def sample_rate(self) -> int:
        """Audio sample rate (default 16000 Hz)."""

    @property
    def channels(self) -> int:
        """Output audio channels (1 = mono, extracted from 6-ch capture).
        The device captures 6 channels (4 raw mic + 2 processed).
        The HAL extracts the configured output channel for consumers."""

    @property
    def is_streaming(self) -> bool:
        """Whether the audio stream is active."""
```

```python
@dataclass
class AudioChunk:
    """A chunk of audio from the microphone."""
    data: bytes              # raw PCM audio (16-bit signed, little-endian)
    timestamp: float         # time.monotonic() at capture
    sample_rate: int         # e.g. 16000
    channels: int            # e.g. 1
    frames: int              # number of frames in this chunk
```

#### Configuration

```yaml
hardware:
  microphone:
    device_name: "ReSpeaker"          # ALSA device name substring match
    sample_rate: 16000
    capture_channels: 6               # raw device channels (4 mic + 2 processed)
    output_channel: 0                 # which channel to output (0 = processed/beamformed)
    chunk_size: 1024                  # frames per read (~64ms at 16kHz)
    doa_enabled: true
    led_ring:
      enabled: true
      brightness: 0.5                # 0.0-1.0, global LED brightness
```

The ReSpeaker captures 6 channels at 16-bit 16kHz: 4 raw microphone
channels plus 2 processed channels (beamformed/AEC'd). The HAL reads
all 6 channels and extracts the configured `output_channel` (channel 0
= processed mono) for downstream consumers. Raw channels are available
for DOA computation internally.

#### Notes

- **USB hot-plug:** The ReSpeaker is a USB device. If it disconnects,
  the microphone module emits `hardware_health(status=ERROR)` and
  attempts reconnection on a backoff schedule. During disconnection, the
  voice pipeline is inactive — WhatsApp remains available.
- **LED ring ownership:** LEDs and audio share the same USB device handle
  via different USB interfaces (audio class vs vendor class). Keeping
  them in one module avoids device handle conflicts.
- **DOA accuracy:** ±15° at 1-3 meters. Sufficient for mapping to camera
  bounding boxes (120° FOV) but not precise enough for exact localization.

---

### `speaker.py` — Audio Output (Dual-Path)

#### Hardware

- **Speaker:** Waveshare 8ohm 5W
- **Primary audio path:** Display's 3.5mm headphone jack → speaker
- **AEC reference path:** ReSpeaker USB audio output (silent)
- **Interface:** ALSA (virtual multi-device)

#### Audio Routing: Path C (Dual Output)

boxBot uses a dual-output audio architecture to maintain AEC (Acoustic
Echo Cancellation) while routing audio through the display's 3.5mm jack.

```
TTS audio → ALSA virtual device ("boxbot_speaker")
                │
                ├──► HDMI audio ──► Display board ──► 3.5mm jack ──► Speaker
                │    (actual sound output)
                │
                └──► ReSpeaker USB audio output (plays nothing — no speaker connected)
                     │
                     └──► XMOS AEC reference signal
                          (XMOS subtracts this from mic input in real-time)
```

**Why not just HDMI?** The ReSpeaker's XMOS chip needs a reference signal
to perform echo cancellation. Without it, the mics hear BB's own voice
through the speaker and interpret it as someone talking. AEC is what makes
barge-in detection possible.

**Why not just ReSpeaker 3.5mm out?** That works too (Path B), but
requires a separate amplifier board. The display's built-in audio DAC
and headphone amplifier are already there — less wiring.

**How ALSA handles it:** A virtual ALSA device (`boxbot_speaker`)
duplicates the audio stream to both the HDMI output and the ReSpeaker USB
output simultaneously. The speaker module writes to this single virtual
device — the duplication is transparent.

**AEC timing:** The XMOS adaptive filter handles acoustic propagation
delay (speaker → air → mic, ~5-30ms). The digital path difference between
HDMI and USB outputs should be <10ms, well within the filter's adaptation
window. **⚠ VALIDATE: confirm AEC convergence with dual-output during
hardware bring-up. Fallback is Path B (everything through ReSpeaker,
requires PAM8302 amp board).**

**Volume caveat:** The display's 3.5mm is a headphone-level output,
designed for 32Ω+ loads. Into an 8Ω speaker, it may be quieter than
desired. If volume is insufficient, add a PAM8302 inline amplifier (~$3,
three wires) between the display's 3.5mm and the speaker. Test without
the amp first.

#### ALSA Configuration

The dual-output routing is configured in `/etc/asound.conf` (or
`~/.asoundrc`). This is a one-time system-level setup, not managed by
boxBot at runtime.

```
# /etc/asound.conf — boxBot dual-output audio routing
#
# Exact card names determined by `aplay -l` on the target hardware.
# The names below are representative — adjust for your system.

# HDMI audio output (Pi 5)
pcm.hdmi_out {
    type hw
    card "vc4hdmi0"               # verify with: aplay -l
}

# ReSpeaker USB audio output (AEC reference only)
pcm.respeaker_out {
    type hw
    card "ArrayUAC10"             # verify with: aplay -l
}

# Virtual device: duplicates audio to both outputs
pcm.boxbot_speaker {
    type route
    slave {
        pcm {
            type multi
            slaves {
                a { pcm "hdmi_out"      channels 2 }
                b { pcm "respeaker_out" channels 2 }
            }
            bindings {
                0 { slave a channel 0 }
                1 { slave a channel 1 }
                2 { slave b channel 0 }
                3 { slave b channel 1 }
            }
        }
        channels 4
    }
    ttable {
        0.0 1.0       # left  → HDMI left
        1.1 1.0       # right → HDMI right
        0.2 1.0       # left  → ReSpeaker left  (AEC ref)
        1.3 1.0       # right → ReSpeaker right (AEC ref)
    }
}

pcm.!default {
    type plug
    slave.pcm "boxbot_speaker"
}
```

#### Interface

```python
class Speaker(HardwareModule):
    name = "speaker"

    async def start(self) -> None:
        """Open ALSA output device (boxbot_speaker virtual device)."""

    async def stop(self) -> None:
        """Stop any active playback, close ALSA device."""

    # ── Playback ──────────────────────────────────────────────────

    async def play(
        self,
        audio_data: bytes,
        sample_rate: int = 24000,
        channels: int = 1
    ) -> None:
        """Play a complete audio buffer. Blocks until playback finishes
        or stop_playback() is called."""

    async def play_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        sample_rate: int = 24000,
        channels: int = 1
    ) -> None:
        """Play audio from a streaming source (e.g., TTS chunks).
        Begins playback as soon as the first chunk arrives.
        Blocks until the stream ends or stop_playback() is called."""

    async def stop_playback(self) -> None:
        """Immediately stop current playback. Safe to call when
        nothing is playing."""

    # ── Volume ────────────────────────────────────────────────────

    def set_volume(self, level: float) -> None:
        """Set playback volume (0.0 to 1.0).
        Applied to the ALSA mixer, affects both HDMI and USB outputs."""

    def get_volume(self) -> float:
        """Current volume level (0.0 to 1.0)."""

    def fade_volume(
        self,
        target: float,
        duration_ms: int = 200
    ) -> None:
        """Smoothly transition volume to target over duration.
        Used by barge-in graduated yielding."""

    # ── State ─────────────────────────────────────────────────────

    @property
    def is_playing(self) -> bool:
        """Whether audio is currently playing."""
```

#### Configuration

```yaml
hardware:
  speaker:
    device_name: "boxbot_speaker"    # ALSA virtual device name
    sample_rate: 24000               # match TTS output format
    default_volume: 0.7              # 0.0-1.0
```

---

### `screen.py` — 7" HDMI LCD with Capacitive Touch

#### Hardware

- **Display:** 7" IPS, 1024×600, HDMI input
- **Touch:** Capacitive, via USB HID (appears as mouse/touch input)
- **Audio:** 3.5mm headphone jack (decodes HDMI audio) — used for speaker
- **Backlight:** Control method varies by model revision
- **Interface:** HDMI (video), USB (touch), possibly USB/GPIO (backlight)
- **Library:** pygame (rendering + touch event capture)

The display has two independent connections to the Pi:
1. **HDMI** — video output (one-way, Pi → display)
2. **USB** — touch input (one-way, display → Pi) + possibly backlight control

#### Touch Input

The capacitive touchscreen appears as a Linux input device. pygame
captures touch events as mouse events by default. The screen module
captures these events and publishes them to the event bus.

**Touch replaces most physical button needs:**
- Package install approval — tap YES/NO on screen
- Dismiss notifications — tap to dismiss
- Tap to wake — alternative to wake word
- Display interaction — scroll, select

**What touch can't replace (and KB2040 could):**
- Blind/ambient interaction (mute mic by feel, volume knob)
- Hardware mic kill switch (physical disconnect, stronger privacy)

For v1, touch + voice + wake word covers the core UX. The KB2040 is a
nice-to-have for later.

#### Touch Event Dispatch

screen.py captures raw touch events and publishes them to the event bus.
Routing depends on what's currently displayed:

```
screen.py (pygame event loop)
    │
    └──► event bus: touch_down(x, y) / touch_up(x, y) / touch_move(x, y)
            │
            ├── Display manager  (navigation, interaction)
            ├── Approval system  (YES/NO taps for package install)
            └── Bootstrap        (first-admin registration screen)
```

The display manager and approval system register touch regions ("button
at rect (x, y, w, h) → callback") with the event bus. screen.py
doesn't interpret touch events — it just publishes coordinates.

#### Backlight Control

The Waveshare 7" HDMI LCD (H) backlight control method varies by hardware
revision. The screen module defines the interface and provides a no-op
fallback if software control is unavailable.

**⚠ VALIDATE: determine backlight control method during hardware
bring-up. Possible methods: USB serial command, GPIO PWM, HDMI CEC, or
none (physical potentiometer only).**

If no software backlight control is available, night mode relies on the
`midnight` theme's dim colors. On an IPS panel, this still provides
significant effective dimming — near-black backgrounds emit very little
light.

#### Interface

```python
class Screen(HardwareModule):
    name = "screen"

    async def start(self) -> None:
        """Initialize pygame display and detect touch device.
        Sets up the rendering surface and touch event capture loop."""

    async def stop(self) -> None:
        """Clear display, release pygame resources."""

    # ── Display ───────────────────────────────────────────────────

    def get_surface(self) -> pygame.Surface:
        """Get the 1024x600 render target. The display manager draws
        onto this surface. Call flip() to push to screen."""

    def flip(self) -> None:
        """Push the current surface contents to the physical display.
        Called by the display manager after rendering."""

    # ── Backlight ─────────────────────────────────────────────────

    def set_brightness(self, level: float) -> None:
        """Set backlight brightness (0.0 to 1.0).
        No-op if hardware doesn't support software control."""

    def get_brightness(self) -> float:
        """Current brightness level (0.0 to 1.0).
        Returns the last set value, or 1.0 if unsupported."""

    # ── Touch ─────────────────────────────────────────────────────
    # Touch events are published to the event bus automatically
    # by the internal pygame event loop:
    #   touch_down(x: int, y: int)
    #   touch_up(x: int, y: int)
    #   touch_move(x: int, y: int)
    #
    # Consumers subscribe via the event bus, not through this module.

    # ── Properties ────────────────────────────────────────────────

    @property
    def width(self) -> int:
        """Display width in pixels (1024)."""

    @property
    def height(self) -> int:
        """Display height in pixels (600)."""

    @property
    def touch_available(self) -> bool:
        """Whether a touch input device was detected."""

    @property
    def backlight_supported(self) -> bool:
        """Whether software backlight control is available."""
```

#### Configuration

```yaml
hardware:
  screen:
    width: 1024
    height: 600
    rotation: 0                      # 0, 90, 180, 270 (enclosure dependent)
    touch_device: "auto"             # auto-detect USB HID touch input
    backlight_control: "auto"        # "auto", "none", "gpio", "usb"
    default_brightness: 0.8
```

---

### `hailo.py` — Raspberry Pi AI HAT+ (Hailo-8L)

#### Hardware

- **Accelerator:** Hailo-8L, 13 TOPS (INT8)
- **Interface:** PCIe via the GPIO header (AI HAT+ sits on top of the Pi)
- **Models:** Compiled HEF files (Hailo Executable Format)
- **Library:** HailoRT Python API

#### Raw Inference Interface

The Hailo module exposes raw model-level operations. Callers handle their
own pre-processing (resize, normalize, pad) and post-processing (NMS,
L2-normalize embeddings, decode outputs). The HAL doesn't know what YOLO
or ReID is — it runs tensors through models.

This keeps the HAL thin and flexible. The perception pipeline and photo
intake pipeline each implement their own model-specific logic on top of
the raw interface.

#### Priority-Based Contention

Two subsystems share the Hailo:
- **Perception pipeline** — real-time person detection and ReID during
  idle scanning and conversation
- **Photo intake pipeline** — batch person detection and ReID on incoming
  photos during idle processing

Live perception always preempts photo intake. The Hailo module implements
this via a priority-aware inference lock:

```
Perception requests inference (priority="realtime")
  → runs immediately, even if a batch inference is in progress

Photo intake requests inference (priority="batch")
  → waits if a realtime inference is running or queued
  → yields if a realtime request arrives mid-inference
```

The `inference_session` context manager holds priority for multi-step
operations (e.g., YOLO detection followed by multiple ReID crops). This
prevents interleaving between perception and photo intake within a
logical operation.

#### Interface

```python
class Hailo(HardwareModule):
    name = "hailo"

    async def start(self) -> None:
        """Initialize HailoRT, detect device.
        Optionally preload models (if preload_models config is true)."""

    async def stop(self) -> None:
        """Unload all models, release HailoRT device."""

    # ── Model management ──────────────────────────────────────────

    async def load_model(self, name: str, hef_path: str) -> None:
        """Load a compiled HEF model onto the device.
        Args:
            name:     Logical name for this model (e.g., "yolo_person")
            hef_path: Path to the .hef file
        """

    async def unload_model(self, name: str) -> None:
        """Unload a model from the device."""

    def is_model_loaded(self, name: str) -> bool:
        """Check if a named model is currently loaded."""

    def get_model_info(self, name: str) -> ModelInfo:
        """Get input/output tensor shapes and types for a loaded model.
        Returns:
            ModelInfo with input_shapes, output_shapes, input_dtypes, etc.
        """

    # ── Inference ─────────────────────────────────────────────────

    async def infer(
        self,
        model_name: str,
        input_data: np.ndarray,
        priority: str = "realtime"
    ) -> list[np.ndarray]:
        """Run a single inference on a loaded model.

        Args:
            model_name: Name of a loaded model
            input_data: Input tensor. Shape must match model input spec.
            priority:   "realtime" (preempts batch) or "batch" (yields
                        to realtime)

        Returns:
            List of output tensors as numpy arrays. The number and shape
            of outputs depend on the model. Caller handles post-processing.
        """

    @asynccontextmanager
    async def inference_session(
        self,
        priority: str = "realtime"
    ) -> AsyncIterator["InferenceSession"]:
        """Hold priority for a multi-step inference operation.

        Usage:
            async with hailo.inference_session(priority="realtime") as session:
                boxes = await session.infer("yolo_person", frame)
                for crop in extract_crops(frame, boxes):
                    embedding = await session.infer("reid_osnet", crop)

        The session holds the priority lock for its duration, preventing
        interleaving with other priority levels.
        """

    # ── Health ────────────────────────────────────────────────────

    @property
    def temperature(self) -> float | None:
        """Hailo die temperature in °C. None if unavailable."""

    @property
    def power_draw(self) -> float | None:
        """Estimated power draw in watts. None if unavailable."""
```

```python
@dataclass
class ModelInfo:
    """Metadata for a loaded Hailo model."""
    name: str
    input_shapes: list[tuple[int, ...]]
    output_shapes: list[tuple[int, ...]]
    input_dtypes: list[np.dtype]
    output_dtypes: list[np.dtype]
```

```python
class InferenceSession:
    """Returned by inference_session(). Holds priority lock."""
    async def infer(
        self,
        model_name: str,
        input_data: np.ndarray
    ) -> list[np.ndarray]:
        """Same as Hailo.infer() but within the held priority context."""
```

#### Configuration

```yaml
hardware:
  hailo:
    device: "auto"                   # auto-detect Hailo-8L
    models:
      yolo_person: "data/perception/models/yolov8n_person.hef"
      reid_osnet: "data/perception/models/osnet_ain_x1.hef"
    preload_models: true             # load models at startup
```

#### Notes

- **Model compilation:** HEF files are compiled from ONNX using the Hailo
  Dataflow Compiler. Pre-compiled models are stored in
  `data/perception/models/`. The compilation step is offline and not part
  of boxBot's runtime.
- **Batch preemption:** When a realtime request arrives during a batch
  inference, the batch inference completes its current frame (inference
  is atomic at the frame level) and then yields. The realtime request
  runs, and the batch resumes after. This adds at most one frame of
  latency (~12-15ms) to the realtime path.
- **Memory:** Both YOLOv8n and OSNet-AIN-x1.0 fit comfortably in the
  Hailo-8L's on-chip memory simultaneously. No model swapping needed.
- **Compute budget:** YOLOv8n at 1 FPS uses ~0.07% of the 13 TOPS
  capacity. Even at the perception pipeline's peak (YOLO + multiple
  ReID crops), utilization stays well under 5%.

---

### `buttons.py` — Physical Controls (Optional)

#### Hardware

- **Controller:** Adafruit KB2040 (RP2040 chip)
- **Interface:** USB Serial (CDC ACM)
- **Firmware:** CircuitPython running on the KB2040
- **Inputs:** Buttons, rotary encoder (configurable by firmware)

#### Status: Optional / Nice-to-Have

The capacitive touchscreen covers the critical input path for v1. The
KB2040 adds blind/ambient interaction — mute by feel, volume knob, wake
button. It can be added later without changing any other module.

The HAL defines the interface so the rest of the system is ready when
the hardware is connected.

#### Interface

```python
class Buttons(HardwareModule):
    name = "buttons"

    async def start(self) -> None:
        """Open serial port, detect KB2040.
        Sets is_available=False (not is_started=False) if device
        not found — this is an optional module."""

    async def stop(self) -> None:
        """Close serial port."""

    # Events published to event bus:
    #   button_pressed(button_id: str)     e.g. "mute", "wake", "dismiss"
    #   button_released(button_id: str)
    #   encoder_rotated(direction: int, steps: int)   +1 = clockwise

    @property
    def available(self) -> bool:
        """Whether the KB2040 is connected. False is normal —
        the module is optional."""
```

#### Configuration

```yaml
hardware:
  buttons:
    enabled: false                   # optional KB2040
    serial_port: "/dev/ttyACM0"
    baud_rate: 115200
```

#### Serial Protocol

The KB2040 firmware sends JSON-encoded events over USB serial:

```json
{"type": "button", "id": "mute", "state": "pressed"}
{"type": "button", "id": "mute", "state": "released"}
{"type": "encoder", "direction": 1, "steps": 3}
```

The firmware is responsible for debouncing. The HAL module parses these
messages and publishes events to the internal event bus. Button ID
mapping (physical button → logical function) is defined in the KB2040
firmware, not in boxBot config.

---

### `system.py` — Thermal Monitoring & Graceful Shutdown

#### Purpose

Cross-cutting hardware concerns that don't belong to any single device
module: SoC temperature, thermal management, signal handling, and system
health aggregation.

#### Thermal Monitoring

The Pi 5 in a wooden enclosure with an AI HAT+ drawing power generates
meaningful heat. The system module monitors thermals and emits events
when thresholds are crossed.

```
Temperature sources:
  SoC:   /sys/class/thermal/thermal_zone0/temp    (Pi 5 CPU/GPU)
  Hailo: HailoRT API (via hailo.py temperature property)
```

**Thermal response:**

| SoC Temp | Action |
|----------|--------|
| < 70°C | Normal operation |
| 70-80°C | `thermal_warning` event — informational |
| 80-85°C | Throttle Hailo usage (reduce scan FPS, pause photo intake) |
| > 85°C | `thermal_critical` event — notify user, further throttle |

The Pi 5's firmware performs its own thermal throttling of CPU frequency
at ~82°C. The system module's thresholds trigger application-level
responses (reduce workload) before the hardware throttle kicks in.

#### Graceful Shutdown

The Pi 5 power button (J2 header → PMIC) sends SIGTERM to all processes.
The system module intercepts this and orchestrates a clean shutdown:

```
SIGTERM / SIGINT received
    │
    ▼
system.py emits: shutdown_requested(reason="signal")
    │
    ▼
Registered handlers run (in registration order):
  1. Agent: flush pending memory extraction, close conversations
  2. Memory: flush pending writes, close database
  3. Scheduler: persist trigger state
  4. Photos: flush intake queue status
    │
    ▼
HAL shutdown sequence (reverse startup order)
    │
    ▼
Process exits cleanly
```

Handlers have a configurable timeout (default 10 seconds total). If
handlers don't complete in time, the system proceeds with HAL shutdown
to avoid hanging on a stuck process.

#### Interface

```python
class System(HardwareModule):
    name = "system"

    async def start(self) -> None:
        """Register signal handlers, start thermal monitoring loop."""

    async def stop(self) -> None:
        """Stop monitoring. (Usually not called directly — this module
        is the one orchestrating shutdown.)"""

    # ── Temperature ───────────────────────────────────────────────

    def get_soc_temperature(self) -> float:
        """SoC temperature in °C."""

    def get_hailo_temperature(self) -> float | None:
        """Hailo die temperature in °C. None if unavailable or Hailo
        not started."""

    # ── Shutdown ──────────────────────────────────────────────────

    def register_shutdown_handler(
        self,
        callback: Callable[[], Awaitable[None]],
        name: str = ""
    ) -> None:
        """Register a handler to run on shutdown. Handlers run in
        registration order with a shared timeout."""

    # ── Health ────────────────────────────────────────────────────

    def get_system_health(self) -> SystemHealth:
        """Aggregate health from all HAL modules + thermals."""
```

```python
@dataclass
class SystemHealth:
    soc_temperature: float           # °C
    hailo_temperature: float | None  # °C
    memory_used_percent: float       # 0-100
    disk_used_percent: float         # 0-100
    modules: dict[str, HealthStatus] # per-module health
```

#### Configuration

```yaml
hardware:
  system:
    thermal_warning_soc: 70          # °C — emit warning event
    thermal_throttle_soc: 80         # °C — reduce Hailo workload
    thermal_critical_soc: 85         # °C — notify user, max throttle
    health_check_interval: 30        # seconds between health polls
    shutdown_timeout: 10             # seconds for shutdown handlers
```

---

## Error Handling and Graceful Degradation

Each HAL module handles its own device errors and reports status via
the event bus. Upper layers adapt to hardware failures:

| Failure | Detection | Degradation |
|---------|-----------|-------------|
| Camera disconnect | picamera2 exception | Perception disabled. Voice + WhatsApp still work. Agent cannot see people but can hear them |
| Microphone disconnect | pyaudio stream error | Voice pipeline disabled. WhatsApp-only mode. Perception continues (visual only) |
| Speaker failure | ALSA write error | TTS silent. Agent communicates via WhatsApp text. Conversations continue (agent responds in text) |
| Hailo error | HailoRT exception | ML inference disabled. Perception falls back to CPU-only motion detection (no person ID). Photo intake paused |
| Screen failure | pygame init error | Headless mode. Voice + WhatsApp only. Package approval via WhatsApp only (no touch) |
| Touch failure | No HID device | Touch unavailable. Voice + WhatsApp for all interaction. Package approval via WhatsApp only |
| KB2040 disconnect | Serial timeout | No impact — module is optional |
| Thermal throttle | Temperature exceeds threshold | Reduced scan FPS, paused photo intake. Notification to user |

**Recovery:** Modules attempt reconnection for USB devices (microphone,
buttons) on a backoff schedule: 1s, 2s, 4s, 8s, max 30s. CSI (camera)
and PCIe (Hailo) devices don't typically disconnect but can error on
driver issues — these require a module restart.

**No single point of failure for core functionality.** The agent can
always be reached via WhatsApp, even if every hardware module fails.
Voice requires microphone + speaker. Vision requires camera + Hailo.
Display requires screen. But none of these are gated on each other.

---

## Configuration Reference

All hardware configuration under the `hardware:` key in `config.yaml`:

```yaml
hardware:
  camera:
    main_resolution: [1280, 720]     # perception stream
    photo_resolution: [4608, 2592]   # still capture (12MP)
    lores_resolution: [320, 240]     # motion detection stream
    scan_fps: 5                      # low-res FPS
    rotation: 180                    # 0, 90, 180, 270

  microphone:
    device_name: "ReSpeaker"         # ALSA device name match
    sample_rate: 16000
    capture_channels: 6              # raw device: 4 mic + 2 processed
    output_channel: 0                # channel index to extract (0 = processed)
    chunk_size: 1024                 # frames per read
    doa_enabled: true
    led_ring:
      enabled: true
      brightness: 0.5

  speaker:
    device_name: "boxbot_speaker"    # ALSA virtual device
    sample_rate: 24000
    default_volume: 0.7

  screen:
    width: 1024
    height: 600
    rotation: 0
    touch_device: "auto"
    backlight_control: "auto"
    default_brightness: 0.8

  hailo:
    device: "auto"
    models:
      yolo_person: "data/perception/models/yolov8n_person.hef"
      reid_osnet: "data/perception/models/osnet_ain_x1.hef"
    preload_models: true

  buttons:
    enabled: false
    serial_port: "/dev/ttyACM0"
    baud_rate: 115200

  system:
    thermal_warning_soc: 70
    thermal_throttle_soc: 80
    thermal_critical_soc: 85
    health_check_interval: 30
    shutdown_timeout: 10
```

The `hardware:` section covers device-level configuration. Feature-level
configuration (voice processing thresholds, display rotation intervals,
perception matching thresholds) lives in the respective feature sections
(`voice:`, `display:`, `perception:`, etc.) documented in their own
design docs.

---

## Testing

### Mock HAL

Every HAL module can be replaced with a mock for testing. Mocks implement
the same interface but return synthetic data:

```python
class MockCamera(Camera):
    async def start(self): self._started = True
    async def stop(self): self._started = False

    async def get_lores_frame(self) -> np.ndarray:
        return np.zeros((240, 320), dtype=np.uint8)

    async def capture_frame(self) -> np.ndarray:
        return np.zeros((720, 1280, 3), dtype=np.uint8)

    async def capture_photo(self) -> np.ndarray:
        return np.zeros((2592, 4608, 3), dtype=np.uint8)


class MockMicrophone(Microphone):
    """Produces silence. Consumers can inject test audio via
    inject_audio(chunk) for pipeline testing."""

    async def start(self): self._started = True
    async def stop(self): self._started = False

    async def inject_audio(self, chunk: AudioChunk) -> None:
        """Test helper: push audio to all registered consumers."""
        await asyncio.gather(*(c(chunk) for c in self._consumers))


class MockHailo(Hailo):
    """Returns zero tensors of the correct shape. Can be configured
    with canned responses for specific test scenarios."""

    async def infer(self, model_name, input_data, priority="realtime"):
        info = self.get_model_info(model_name)
        return [np.zeros(shape, dtype=dt)
                for shape, dt in zip(info.output_shapes, info.output_dtypes)]
```

### Hardware Integration Tests

Integration tests run on the actual Pi with real hardware. They verify
device detection, basic I/O, and lifecycle:

```python
@pytest.mark.hardware
async def test_camera_capture():
    cam = Camera(config)
    await cam.start()
    frame = await cam.capture_frame()
    assert frame.shape == (720, 1280, 3)
    await cam.stop()

@pytest.mark.hardware
async def test_microphone_stream():
    mic = Microphone(config)
    chunks = []
    mic.add_consumer(lambda c: chunks.append(c))
    await mic.start()
    await asyncio.sleep(0.5)
    await mic.stop()
    assert len(chunks) > 0
    assert chunks[0].sample_rate == 16000
```

Hardware tests are tagged `@pytest.mark.hardware` and excluded from
CI — they run only on-device during development and integration testing.

---

## Validation Items

Items marked for validation during hardware bring-up:

1. **⚠ AEC with dual-output audio** — confirm XMOS AEC converges when
   the reference comes from USB and the speaker plays from HDMI. Measure
   echo suppression quality. Fallback: Path B (ReSpeaker 3.5mm → amp →
   speaker, single ALSA output).

2. **⚠ Display backlight control** — determine the 7" HDMI LCD (H)
   backlight control method (USB serial, GPIO PWM, HDMI CEC, or none).
   Implement accordingly or fall back to theme-only dimming.

3. **⚠ Display 3.5mm speaker volume** — test headphone output driving
   an 8Ω speaker. If too quiet, add PAM8302 inline amplifier.

4. **⚠ ALSA device names** — run `aplay -l` and `arecord -l` on the
   target hardware to determine exact card names for the ALSA config.

5. **⚠ Thermal profile in enclosure** — measure SoC and Hailo
   temperatures under load in the wooden box. Adjust thermal thresholds
   and heat sink placement if needed.

6. **⚠ picamera2 mode switch latency** — measure the actual latency of
   `switch_mode_and_capture_array()` for full-res photo capture. Confirm
   it's acceptable (~100-200ms expected).

---

## File Layout

```
src/boxbot/hardware/
  __init__.py         # exports all module classes + base
  base.py             # HardwareModule ABC, HealthStatus, shared types
  camera.py           # Pi Camera Module 3 Wide NoIR
  microphone.py       # ReSpeaker XMOS XVF3000 + LED ring
  speaker.py          # Dual-output audio (HDMI + USB AEC reference)
  screen.py           # 7" HDMI LCD + capacitive touch
  hailo.py            # AI HAT+ (Hailo-8L) inference engine
  buttons.py          # KB2040 physical controls (optional)
  system.py           # Thermal monitoring, graceful shutdown, health
```
