"""Web view sound: browser → PipeWire null sink → speaker HAL.

The speaker HAL holds the audio output device exclusively, so the
browser cannot open it (verified on the box: a PipeWire stream aimed at
the HDMI sink never starts while boxBot runs). Rather than fight over
the device, the browser plays into a PipeWire *null sink* that goes
nowhere, and this bridge records that sink and hands the PCM to
:meth:`boxbot.hardware.speaker.Speaker.feed`.

Two things fall out of sending it through the HAL:

- one owner of the output device, as before — TTS is untouched;
- the audio joins the AEC reference, so the mic does not hear the
  video as room sound and the wake word still works over it.

The cost is latency: sound trails picture by roughly the speaker's
buffer depth (a few hundred ms).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
from typing import Any, Callable

logger = logging.getLogger(__name__)

SINK_NAME = "boxbot_kiosk"
_RATE = 48000
_CHUNK_BYTES = _RATE // 50 * 2  # 20 ms of mono s16
# Samples at or below this are digital silence from an idle sink.
_SILENCE_PEAK = 4

_SINK_PROPS = (
    "{ factory.name=support.null-audio-sink "
    f"node.name={SINK_NAME} "
    'node.description="boxBot web view" '
    "media.class=Audio/Sink audio.position=[FL FR] "
    # Outlives the pw-cli process that creates it.
    "object.linger=true }"
)


class WebAudioBridge:
    """Pumps the null sink's audio into the speaker while it runs.

    ``muted`` is polled per chunk: while it returns True (BB speaking,
    room listening) chunks are discarded, so nothing the browser still
    has in flight can land on top of speech.
    """

    def __init__(self, speaker: Any, *, muted: Callable[[], bool]) -> None:
        self._speaker = speaker
        self._muted = muted
        self._proc: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def start(self) -> bool:
        """Create the sink if needed and begin pumping.

        Returns False (with a logged reason) when sound is unavailable;
        the web view still opens, silently.
        """
        if self.running:
            return True
        if shutil.which("pw-cli") is None or shutil.which("pw-record") is None:
            logger.warning("web view audio off: pw-cli / pw-record not installed")
            return False
        if not await self._ensure_sink():
            return False
        try:
            self._proc = await asyncio.create_subprocess_exec(
                "pw-record",
                "-P", "{ stream.capture.sink=true }",
                "--target", SINK_NAME,
                "--rate", str(_RATE), "--channels", "1", "--format", "s16",
                "--latency", "20ms",
                "--raw", "-",
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError as e:
            logger.warning("web view audio off: pw-record failed to start: %s", e)
            return False
        self._task = asyncio.create_task(self._pump(self._proc), name="web-audio")
        return True

    async def stop(self) -> None:
        task, proc = self._task, self._proc
        self._task = None
        self._proc = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), 2.0)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()

    async def _ensure_sink(self) -> bool:
        listing = await self._run("pw-cli", "ls", "Node")
        if listing is None:
            logger.warning("web view audio off: PipeWire not reachable")
            return False
        if f'"{SINK_NAME}"' in listing:
            return True
        if await self._run("pw-cli", "create-node", "adapter", _SINK_PROPS) is None:
            logger.warning("web view audio off: could not create %s sink", SINK_NAME)
            return False
        logger.info("Created PipeWire sink %s for web view audio", SINK_NAME)
        return True

    @staticmethod
    async def _run(*argv: str) -> str | None:
        """Run a short PipeWire command. Output on success, else ``None``."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), 5.0)
        except (OSError, asyncio.TimeoutError):
            return None
        return out.decode(errors="replace") if proc.returncode == 0 else None

    async def _pump(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None
        try:
            while True:
                chunk = await proc.stdout.readexactly(_CHUNK_BYTES)
                if self._muted() or _is_silent(chunk):
                    continue
                self._speaker.feed(chunk, sample_rate=_RATE, channels=1)
        except asyncio.IncompleteReadError:
            logger.warning("web view audio capture ended (pw-record exited)")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("web view audio pump failed")


def _is_silent(chunk: bytes) -> bool:
    """True if every s16 sample in ``chunk`` is (near) zero."""
    samples = memoryview(chunk).cast("h")
    return max(samples) <= _SILENCE_PEAK and min(samples) >= -_SILENCE_PEAK
