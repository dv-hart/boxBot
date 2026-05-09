# bb.audio — play audio files through the speaker

Decode and play `.wav` / `.flac` / `.ogg` / `.mp3` files stored in the
workspace through the speaker.

## When to use it

- The user asks to hear a song, sound effect, recorded clip, or chime
  they've stored in the workspace.
- A skill or trigger wants to play an alert tone.
- Replaying an inbound voice message someone sent over WhatsApp (after
  it's been moved into the workspace).

## When NOT to use it

- For text the agent itself should *say*. That's TTS — it happens
  automatically through the structured `voice` output, you don't call
  it from a script.
- For arbitrary URLs from the internet. Download to the workspace
  first (e.g. via `requests.get`), then play. The player only reads
  workspace-resident files.

## How it interacts with the conversation

Playback runs through the same path as TTS:

- The mic's STT consumer detaches for the duration so household
  chatter and the speaker's own output don't enter the transcript.
- `AgentSpeaking` / `AgentSpeakingDone` events fire, so the
  conversation's room state flips to SPEAKING and back to LISTENING
  when playback ends.
- Saying the wake word ("BB") interrupts playback cleanly and
  re-activates STT — exactly like cutting BB off mid-sentence.

`bb.audio.play()` blocks the sandbox script until playback drains
naturally OR is interrupted. After it returns, your script normally
has nothing more to do — the agent's turn ends, the conversation
lands in LISTENING with the post-response idle window armed.

## API

```python
import boxbot_sdk as bb

# Play a file from the workspace
result = bb.audio.play("audio/chime.wav")

# Quieter playback (volume restores when the call returns)
bb.audio.play("music/favorite_song.mp3", volume=0.5)
```

`play(path, *, volume=None) -> dict` — returns:

| Key | Type | Meaning |
|-----|------|---------|
| `status` | `"ok"` or `"interrupted"` | drained naturally vs. wake-word stopped it |
| `duration_ms` | int | full decoded length |
| `elapsed_ms` | int | how long playback actually ran |
| `format` | `"wav"` / `"flac"` / `"ogg"` / `"mp3"` | source format |
| `sample_rate` | int | source sample rate (pre-resample) |
| `channels` | int | source channel count |

`AudioError` is raised on path/format/quota/decoder failures. Catch it
if you want a graceful fallback; otherwise let it propagate so the
agent sees the failure clearly.

## Patterns

### Find and play a song the user asked for

```python
import boxbot_sdk as bb

hits = bb.workspace.search("favorite song")
if not hits:
    print("no matches in workspace")
else:
    bb.audio.play(hits[0]["path"])
```

### Save an inbound audio attachment, then play it back

```python
# An inbound WhatsApp voice note has been staged into the workspace
bb.workspace.write("audio/voicenote_carina.ogg", voice_bytes)
result = bb.audio.play("audio/voicenote_carina.ogg")
if result["status"] == "interrupted":
    print("user wake-word'd over the playback — they want to talk")
```

### Quick alert chime from a trigger

```python
bb.audio.play("audio/timer_done.wav", volume=0.7)
```

## Limits

- **Format:** wav, flac, ogg, mp3. Decoding handled in the main
  process by `miniaudio` (no ffmpeg needed).
- **File size cap:** 25 MB by default — plenty for a 5-minute MP3.
- **Decoded duration cap:** 5 minutes by default. Adjustable in
  config; enforced before playback starts so a hostile or oversized
  file fails fast.
- **Concurrency:** the speaker serialises one playback at a time. If
  TTS is mid-sentence when you call `bb.audio.play()`, the audio call
  waits for it to drain. If you want the audio to start immediately,
  finish the spoken response first.

## Workspace layout suggestion

```
workspace/
  audio/
    chimes/
      timer_done.wav
      task_complete.wav
    voicenotes/
      2026-05-06_carina.ogg
  music/
    favorite_song.mp3
```

Matches the `notes/` and `data/` conventions elsewhere in the
workspace — pick a structure that makes search obvious.
