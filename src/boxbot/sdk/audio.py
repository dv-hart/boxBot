"""bb.audio — play audio files through the speaker.

Stores live in the workspace; play picks them up from there. Supported
formats: ``.wav``, ``.flac``, ``.ogg``, ``.mp3``.

Usage:

    import boxbot_sdk as bb

    # Play a song stored in the workspace
    bb.audio.play("music/erik_birthday_song.mp3")

    # A short chime, quieter than usual
    bb.audio.play("audio/chime.wav", volume=0.4)

While the file is playing, the mic's STT consumer is detached so
household chatter and BB's own output don't enter the transcript
pipeline. Saying the wake word stops playback and reactivates STT —
this matches BB's behaviour during TTS exactly. ``play`` returns when
the audio drains naturally OR when it's interrupted; either outcome is
in the returned dict's ``status`` field.

The call blocks the sandbox script until playback completes, so a
typical "play my favorite song" interaction looks like:

    matches = bb.workspace.search("favorite song")
    if matches:
        bb.audio.play(matches[0]["path"])
        # script returns; agent's turn ends; conversation lands in
        # LISTENING with the post-response idle window armed
"""

from __future__ import annotations

from typing import Any

from . import _transport, _validators as v


# Long-running by design — leave generous slack above the player's
# hard duration cap so an interrupted song still gets a clean reply.
_TIMEOUT = 600

# Mirror of communication/audio_player.py — duplicated so the SDK
# never imports main-process modules.
SUPPORTED_FORMATS = (".wav", ".flac", ".ogg", ".mp3")


class AudioError(Exception):
    """Raised on audio playback failures.

    Covers: file not found, format unsupported, file too large,
    decoded audio too long, decoder failure, no speaker available,
    or any error returned by the main-process dispatcher.
    """


def play(path: str, *, volume: float | None = None) -> dict[str, Any]:
    """Play a workspace audio file through the speaker.

    Args:
        path: Workspace-relative path, e.g. ``audio/chime.wav`` or
            ``music/favorite.mp3``. No absolute paths, no ``..``.
        volume: Optional override (0.0 – 1.0). The speaker's prior
            volume is restored on completion so subsequent TTS isn't
            affected.

    Returns:
        Dict with:

        - ``status``: ``"ok"`` (drained naturally) or ``"interrupted"``
          (wake word stopped playback)
        - ``duration_ms``: decoded length of the audio
        - ``elapsed_ms``: actual time playback ran (less than
          ``duration_ms`` when interrupted)
        - ``format``: ``"wav"`` / ``"flac"`` / ``"ogg"`` / ``"mp3"``
        - ``sample_rate``: source sample rate (pre-resample)
        - ``channels``: source channel count

    Raises:
        AudioError: on any playback failure surfaced by the main
            process.
    """
    v.require_str(path, "path")
    payload: dict[str, Any] = {"path": path}
    if volume is not None:
        payload["volume"] = v.require_float(
            volume, "volume", min_val=0.0, max_val=1.0
        )

    response = _transport.request("audio.play", payload, timeout=_TIMEOUT)
    status = response.get("status")
    if status == "error":
        raise AudioError(response.get("error", "audio.play failed"))
    if status not in {"ok", "interrupted"}:
        raise AudioError(
            f"unexpected response from audio.play: {response!r}"
        )
    return response
