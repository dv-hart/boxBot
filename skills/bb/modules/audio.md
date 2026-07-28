# bb.audio — play files through the speaker

Plays `.wav` / `.flac` / `.ogg` / `.mp3` stored in the workspace.

**Use for:** a song, sound effect, recorded clip, or chime the user
stored; an alert tone from a trigger; replaying an inbound voice note
after it lands in the workspace.

**Not for:** text *you* should say — that is TTS, and it happens
automatically through structured voice output. Not for internet URLs —
download to the workspace first (`requests.get`), then play. The player
only reads workspace-resident files.

## Conversation interaction

Playback shares the TTS path:

- The mic's STT consumer detaches for the duration, so household
  chatter and your own output stay out of the transcript.
- `AgentSpeaking` / `AgentSpeakingDone` fire; room state flips to
  SPEAKING and back to LISTENING.
- The wake word ("BB") interrupts cleanly and re-activates STT —
  exactly like cutting BB off mid-sentence.

`play()` blocks the script until playback drains or is interrupted.
After it returns your script usually has nothing left to do; the turn
ends and the conversation lands in LISTENING.

## API

```python
import boxbot_sdk as bb

result = bb.audio.play("audio/chime.wav")
bb.audio.play("music/favorite_song.mp3", volume=0.5)   # volume restores on return
```

`play(path, *, volume=None) -> dict`:

| Key | Type | Meaning |
|-----|------|---------|
| `status` | `"ok"` \| `"interrupted"` | drained naturally vs. wake-word stopped it |
| `duration_ms` | int | full decoded length |
| `elapsed_ms` | int | how long playback ran |
| `format` | `"wav"`/`"flac"`/`"ogg"`/`"mp3"` | source format |
| `sample_rate` | int | source rate, pre-resample |
| `channels` | int | source channel count |

`AudioError` on path / format / quota / decoder failure. Catch it for a
graceful fallback; otherwise let it propagate so you see the failure.

## Patterns

```python
hits = bb.workspace.search("favorite song")
if not hits:
    print("no matches in workspace")
else:
    bb.audio.play(hits[0]["path"])
```

```python
bb.workspace.write("audio/voicenote_carina.ogg", voice_bytes)
result = bb.audio.play("audio/voicenote_carina.ogg")
if result["status"] == "interrupted":
    print("user wake-word'd over playback — they want to talk")
```

## Limits

- Formats: wav, flac, ogg, mp3. Decoded by `miniaudio` in the main
  process. No ffmpeg.
- File size cap 25 MB (default). Decoded duration cap 5 minutes
  (default, configurable), enforced before playback so oversized files
  fail fast.
- One playback at a time. If TTS is mid-sentence, `play()` waits for it
  to drain. To start immediately, finish the spoken response first.

## Layout suggestion

```
workspace/audio/chimes/timer_done.wav
workspace/audio/voicenotes/2026-05-06_carina.ogg
workspace/music/favorite_song.mp3
```
