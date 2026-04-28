# Voice Pipeline

## Overview

boxBot's voice pipeline handles the full audio lifecycle: wake word
detection, speech capture, transcription, speaker attribution, agent
response, and text-to-speech playback. It is designed around a key
architectural principle: **BB is a conversational participant that
chooses when to speak, not a real-time voice endpoint that must always
respond.**

Unlike streaming voice AI (e.g. OpenAI Realtime), boxBot captures
diarized, attributed transcripts and presents them to the agent as
text. The agent sees `[Jacob]: "What's the weather?"` and decides
whether a response is warranted. This enables multi-speaker awareness,
contextual silence, and human-like conversational dynamics.

## Hardware Foundation

### ReSpeaker XMOS XVF3000 4-Mic Array

The microphone array is not just a microphone — it is a voice
processing frontend. The XMOS XVF3000 chip provides:

| Capability | What It Does | Why It Matters |
|------------|-------------|----------------|
| **Acoustic Echo Cancellation (AEC)** | Subtracts BB's own speaker output from mic input | Enables barge-in detection while BB is speaking |
| **Noise suppression** | Filters non-speech sounds (taps, fan, HVAC) | Reduces false VAD triggers from mechanical noise |
| **Beamforming** | Focuses on sound from a specific direction using 4 mics | Improves speech quality, especially at distance |
| **Automatic Gain Control** | Normalizes volume across near/far speakers | Consistent audio levels for STT |
| **Direction of Arrival (DOA)** | Estimates which direction sound comes from | Resolves voice-visual conflicts in perception |

These run in real-time on the XMOS chip with negligible latency
(~10-20ms). The audio that reaches our software pipeline is already
cleaned — most non-speech noise is removed before our VAD sees it.

### Speaker

Waveshare 8ohm 5W, driven by the Pi 5's audio output. Positioned
close to the mic array, which makes AEC critical — without it, the
mics would hear BB's own voice as "someone speaking."

## Pipeline Architecture

```
┌───────────────────────────────────────────────────────────┐
│  ReSpeaker XMOS XVF3000 (hardware, always running)        │
│  AEC · Noise Suppression · Beamforming · AGC              │
└──────────────────────────┬────────────────────────────────┘
                           │ cleaned audio stream
                           ▼
┌───────────────────────────────────────────────────────────┐
│  Wake Word Detection (CPU, always running)                 │
│  OpenWakeWord — listening for "BB" (bee-bee)               │
│  Low CPU cost (~2-3% single core)                         │
└──────────┬────────────────────────────────────────────────┘
           │ wake word detected
           ▼
┌───────────────────────────────────────────────────────────┐
│  Voice Activity Detection (CPU, session active)            │
│  Silero VAD — neural network speech detector               │
│  Distinguishes speech from residual noise                  │
└──────────┬────────────────────────────────────────────────┘
           │ speech regions identified
           ▼
┌───────────────────────────────────────────────────────────┐
│  Audio Accumulation                                        │
│  Buffer speech segments until silence threshold met        │
│  Silence persistence: configurable (default 800ms)         │
└──────────┬────────────────────────────────────────────────┘
           │ finalized utterance audio
           ▼
┌───────────────────────────────────────────────────────────┐
│  Parallel Processing                                       │
│  ┌─────────────────────┐  ┌─────────────────────────────┐ │
│  │ pyannote diarization │  │ ElevenLabs Scribe (STT)     │ │
│  │ Who spoke, when      │  │ Audio → transcript text     │ │
│  │ Speaker embeddings   │  │ Batch mode (v1)             │ │
│  └─────────┬───────────┘  └──────────────┬──────────────┘ │
│            │                              │                │
│            └──────────┬───────────────────┘                │
│                       ▼                                    │
│            Attributed Transcript                           │
│            [Jacob]: "What's the weather tomorrow?"         │
└───────────────────────┬───────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────┐
│  Agent (large model)                                       │
│  Full context: transcript + person identity + memories     │
│  Decides: respond / stay quiet / wait for more             │
└──────────┬────────────────────────────────────────────────┘
           │ response text (if responding)
           ▼
┌───────────────────────────────────────────────────────────┐
│  ElevenLabs TTS (streaming)                                │
│  Start playback before full response is generated          │
│  Barge-in monitoring active during playback                │
└───────────────────────────────────────────────────────────┘
```

## Wake Word Detection

### Engine: OpenWakeWord

Open source, community-trained, runs on CPU. Selected over Picovoice
Porcupine (proprietary) because boxBot is an open-source project.

**Wake word:** "BB" (bee-bee)

**Two-syllable wake word risks:** Short wake words have higher false
positive rates. "BB" could trigger on "baby," "maybe," "BT," etc.
Mitigations:
- Custom-trained OpenWakeWord model on "bee-bee" specifically
- Tuned confidence threshold (higher than default)
- False positives are low-stakes — BB starts listening, nobody
  addresses it, session times out. No harm done

### Configuration

```yaml
voice:
  wake_word:
    engine: "openwakeword"
    word: "bb"
    confidence_threshold: 0.7       # higher = fewer false positives
    model_path: "models/wake_word/"  # custom trained model
```

### Wake Word During Session

When a session is SUSPENDED (mic off, context retained), the wake
word listener remains active. Detecting the wake word reactivates the
session with existing context rather than starting fresh.

## Voice Activity Detection (VAD)

### Engine: Silero VAD

Neural network-based VAD, more accurate than energy-based approaches.
Runs on CPU with minimal overhead (~1-2% single core). Silero VAD
handles the cases that hardware noise suppression misses — quiet
speech, speech mixed with music, edge-case sounds that resemble
speech.

The XMOS hardware handles the bulk of noise filtering. Silero VAD
provides a second layer of confidence that what we're hearing is
actually human speech.

### Configuration

```yaml
voice:
  vad:
    engine: "silero"
    threshold: 0.5             # speech probability threshold
    min_speech_duration: 250   # ms — ignore speech shorter than this
    min_silence_duration: 100  # ms — ignore silence shorter than this
```

## Turn Detection: Silence Persistence

The core problem with aggressive turn detection (as in OpenAI
Realtime): any pause triggers the AI to respond. Natural
conversation has pauses — between clauses, while thinking, for
emphasis. These are not turn-yields.

### Human Conversational Timing Reference

| Pause Type | Duration | Meaning |
|-----------|----------|---------|
| Within-clause | 200-600ms | Thinking mid-sentence, choosing a word |
| Between-clause | 500-1000ms | Finishing a thought, starting another |
| Turn-yielding | 800-1500ms | Done speaking, expecting a response |
| Topic boundary | 1000-2000ms | Shifting to a new subject |

### Silence Persistence Model

```
Speech detected → start accumulating audio
  │
  Silence detected (VAD says no speech) → start persistence timer
  │
  ├── Speech resumes before threshold → reset timer, keep accumulating
  │
  └── Silence persists beyond threshold → finalize utterance
      │
      └── Send to diarization + STT → agent
```

The agent receives the full utterance, not fragments. No mid-sentence
cutoffs.

### Configuration

```yaml
voice:
  turn_detection:
    silence_threshold: 800       # ms — silence before finalizing utterance
    max_utterance_duration: 60   # seconds — hard cap to prevent runaway capture
    inter_utterance_gap: 300     # ms — minimum gap between separate utterances
```

### Why This Works Without an Intent Check

Earlier iterations considered a small model "intent check" between
STT and the agent to classify whether BB is being addressed. This was
removed because:

1. **The agent is better at it.** With full conversation context,
   person identity, and memories, the agent makes superior judgment
   calls. A small model classifying isolated utterances misses context
   like "Carina saying 'also check tomorrow' as a follow-up to BB."
2. **It adds latency.** ~200ms+ for a decision that's usually obvious
   to the agent anyway.
3. **The agent can simply not respond.** If an utterance doesn't
   warrant a response, the agent stays quiet. The utterance still
   enters the conversation log for context.

## Barge-In: Graduated Yielding

When BB is speaking and someone interrupts, the system needs to
distinguish genuine interruption from backchannels and noise.

### The Problem

Sounds that should NOT interrupt BB:
- Coughs, throat clearing
- "Mm-hmm," "yeah" (backchannels — confirming you're listening)
- Phone buzzing on a table
- Shifting in a chair, finger taps

Sounds that SHOULD interrupt BB:
- "Actually, wait—"
- "BB, stop"
- Any sustained, intentional speech

### The Difference

Backchannels are brief and stop. Interruptions persist. A "yeah" lasts
200-300ms. "Actually, the doctor called to cancel" continues for
seconds.

### Wake-Word-Gated Interruption

While BB is speaking, the STT/diarization consumer is **detached** from
the microphone — chatter, residual echo from BB's own voice, and any
other ambient sound cannot reach the transcript pipeline. The wake
word is the only path back to a hot mic. This is the v2 model after
the graduated VAD-based three-stage yielding was retired (see commit
history): it nullifies the household-chatter problem and removes the
fragility of AEC-aligned barge-in detection. The trade-off is that
you cannot cut BB off mid-sentence with raw speech — you say the wake
word.

### After a Wake-Word Interrupt

1. The wake-word handler stops TTS playback and publishes
   `AgentSpeakingDone(interrupted=True)`.
2. The agent translates that into `Conversation.interrupt()`, which
   cancels the in-flight generation, folds any partial spoken
   segments into the thread as an interrupted assistant turn, and
   drops any utterances that had been queued during SPEAKING (the
   user has explicitly taken back the floor — overheard chatter is
   irrelevant to what they are about to say).
3. The conversation transitions to LISTENING; the next utterance the
   user speaks runs through the normal `handle_input` path.

### After Natural TTS Completion

1. The voice layer publishes `AgentSpeakingDone(interrupted=False)`.
2. STT is re-attached to the mic so the user can continue without
   re-saying the wake word.
3. `_run_generation` either drains anything queued during SPEAKING
   (overheard utterances the agent should now consider) or settles
   in LISTENING.

### XMOS AEC: Why Barge-In Works at All

Without echo cancellation, the microphones hear BB's own voice through
the speaker. The XMOS XVF3000 models the acoustic path from speaker to
each microphone and subtracts BB's output from the input signal in real
time. What remains is only external speech.

This is why we selected a mic array with a dedicated DSP chip rather
than a simple USB microphone. Barge-in detection with a single mic and
no AEC would require the speaker to shout over BB — terrible UX.

### AEC Reference Delay Calibration

The XMOS chip can only cancel BB's voice if the reference signal it
receives (over USB) is **time-aligned** with the speaker output it
hears coming back through the mics. On the boxBot reference hardware,
the HDMI playback path has a much deeper buffer than the USB AEC
reference path, so the XMOS sees the reference tens of milliseconds
before it actually hears BB. The adaptive filter struggles or fails
under that misalignment, and BB's own speech bleeds back in as
phantom user turns.

To fix this, run the calibration script once after any audio-stack
change (new speaker, ALSA tuning, USB re-plug):

```bash
python3 scripts/calibrate_aec.py
```

The script:

1. Opens the speaker with the AEC reference path **disabled** so XMOS
   doesn't try to subtract anything.
2. Plays a 1.5 s linear chirp (200 Hz → 6 kHz) through HDMI.
3. Captures the chirp via the ReSpeaker mic array.
4. Cross-correlates capture against the known chirp to measure the
   HDMI playback latency.
5. Writes the measured offset to
   ``data/calibration/aec_delay_samples.json``.

On the next boxbot start, ``Speaker._open_aec_reference_stream``
reads the file and prepends that many samples of silence to every
chunk it pushes onto the AEC reference queue. Net effect: the XMOS
chip receives the reference signal at the same wall-clock instant
the mics capture the speaker's output, so its adaptive filter has
something it can actually subtract.

Use ``--dry-run`` to measure without persisting, ``--verbose`` for
debug output, ``--volume`` to scale the chirp amplitude.

### Wake-word-gated STT

Empirically AEC alignment alone is not enough. The XMOS adaptive
filter is cold-start at the top of every reply (no reference signal =
no learning), so the first ~1–2 s of TTS leaks past AEC and BB
transcribes its own voice. Separately, household chatter during BB's
reply is itself a problem: kids talking over BB get cleanly captured
and treated as user turns. Timing knobs cannot fix either.

Wake-word-gated STT solves both at once:

1. When ``AgentSpeaking`` fires, the audio_capture consumer
   (STT/diarization) is **detached** from the microphone. The
   wake-word consumer stays attached.
2. While BB is speaking, no mic audio reaches STT. Chatter, residual
   echo, sneezes, anything: not transcribed. To cut BB off
   mid-sentence the user says the wake word again, which stops TTS
   and re-attaches STT.
3. When TTS completes naturally, STT is re-attached automatically
   so the user can continue the conversation without re-saying the
   wake word.

Trade-off: you cannot cut BB off mid-sentence with raw speech — only
with the wake word. In practice this matches how households actually
interrupt ("BB, stop"). It also nullifies the chatter problem
completely.

Implementation: see ``VoiceSession.speak()`` and
``VoiceSession._on_wake_word`` in
``src/boxbot/communication/voice.py``. Conversation-side handling
of overheard utterances queued during SPEAKING and partial-fold on
interrupt lives in ``Conversation.handle_input`` and
``Conversation.interrupt`` in ``src/boxbot/core/conversation.py``.

## Speaker Diarization and Attribution

### Engine: pyannote.audio

pyannote provides speaker diarization ("who spoke when") and speaker
embedding extraction. It runs on CPU during active sessions.

pyannote is NOT an STT engine — it does not transcribe speech. It
answers "who is speaking?" while ElevenLabs Scribe answers "what did
they say?"

### How Attribution Works

1. Audio captured during the session is processed by pyannote
2. pyannote segments the audio by speaker and extracts embeddings
3. Speaker embeddings are matched against known person voice profiles
   (from the perception pipeline's voice cloud)
4. Each segment is labeled: `Speaker A = Jacob`, `Speaker B = Carina`,
   `Speaker C = Unknown`
5. The STT transcript is aligned with diarization segments
6. The agent receives attributed text:

```
[Jacob]: Hey, what should we do for dinner tonight?
[Carina]: I was thinking maybe Italian?
[Jacob]: That sounds good. BB, do we have pasta?
```

Unknown speakers appear as `[Person A]`, `[Person B]`, etc. The agent
can use the `identify_person` tool to name them if it learns their
identity through conversation.

### Configuration

```yaml
voice:
  diarization:
    engine: "pyannote"
    model: "pyannote/speaker-diarization-3.1"
    embedding_model: "pyannote/wespeaker-voxceleb-resnet34-LM"
    min_speakers: 1
    max_speakers: 6             # reasonable household limit
    match_threshold: 0.65       # cosine similarity for speaker matching
```

## Speech-to-Text (STT)

### Engine: ElevenLabs Scribe

Pluggable STT with ElevenLabs Scribe as the initial provider. The
STT interface is provider-agnostic so alternative engines (Whisper,
Deepgram, AssemblyAI) can be swapped in.

### Batch Mode (v1)

In v1, audio is sent to STT after the silence threshold is met —
the full utterance is transcribed as a batch. This is simpler and
sufficient for household conversation latency:

```
Speech ends → 800ms silence → send audio to Scribe → transcript
                                      ~1-2s
```

### Streaming Mode (Future Optimization)

Streaming STT would process audio while the person is still speaking,
reducing perceived latency by 1-2s. The transcript builds
incrementally and is finalized when silence is detected. This is a
latency optimization, not a functional change — the agent still waits
for the finalized transcript before responding.

### STT Provider Interface

```python
class STTProvider(Protocol):
    async def transcribe(self, audio: bytes, sample_rate: int,
                         language: str = "en") -> str:
        """Transcribe audio to text. Returns transcript string."""
        ...

    async def transcribe_stream(self, audio_stream: AsyncIterator[bytes],
                                 sample_rate: int,
                                 language: str = "en") -> AsyncIterator[str]:
        """Stream audio, yield partial transcripts. Optional."""
        ...
```

### Configuration

```yaml
voice:
  stt:
    provider: "elevenlabs"
    model: "scribe_v1"
    language: "en"
    # Provider-specific settings
    elevenlabs:
      api_key_secret: "elevenlabs_api_key"  # references boxbot_sdk.secrets
```

## Text-to-Speech (TTS)

### Engine: ElevenLabs TTS

High-quality neural TTS with streaming support. BB starts speaking
before the full agent response is generated — the TTS API receives
text chunks as the agent streams its response.

### Streaming Playback

```
Agent starts generating response
  │
  First text chunk available (~1-2s)
  │
  ├──► ElevenLabs TTS API (streaming)
  │         │
  │         ├──► Audio chunk 1 → speaker playback begins
  │         ├──► Audio chunk 2 → continues playing
  │         └──► ...
  │
  Agent finishes response
  │
  Final audio chunk → playback completes
```

This reduces perceived latency significantly. The user hears BB start
responding ~200ms after the first text chunk is available, rather than
waiting for the entire response to be generated and synthesized.

### Voice Selection

BB's voice is configured once and persists. ElevenLabs offers voice
cloning and a library of pre-built voices. The voice should match the
product personality — warm, clear, not robotic, not overly enthusiastic.

### Configuration

```yaml
voice:
  tts:
    provider: "elevenlabs"
    voice_id: "configured_voice_id"
    model: "eleven_turbo_v2_5"
    stability: 0.5
    similarity_boost: 0.75
    # Provider-specific settings
    elevenlabs:
      api_key_secret: "elevenlabs_api_key"
      optimize_streaming_latency: 3  # 0-4, higher = lower latency
```

## Conversation Session Model

### State Machine

```
IDLE
  │
  ├── Wake word detected ─────────────────┐
  │   (user initiates)                    │
  │                                       ▼
  ├── Trigger fires + person present ──► SESSION ACTIVE
  │   (BB initiates)                      │
  │                                       │
  │   BB-initiated: BB just starts        │
  │   talking. "Jacob, Carina asked       │
  │   me to remind you about dinner."     │
  │   No chime, no preamble.             │
  │                                       │
  │   If "Not now BB" → agent creates     │
  │   person trigger to retry later       │
  │                                       │
  └───────────────────────────────────────┘

SESSION ACTIVE (mic on, full pipeline running)
  │
  │  Continuous loop:
  │    1. VAD detects speech → accumulate
  │    2. Silence threshold → finalize
  │    3. Diarize + STT → attributed transcript
  │    4. Agent receives transcript
  │    5. Agent decides:
  │         Respond → TTS → speak (barge-in monitored)
  │         Stay quiet → log to context, keep listening
  │         Wait → keep listening for more
  │
  │  If BB is speaking:
  │    Barge-in monitoring active
  │    Graduated yielding (ignore → fade → stop)
  │
  └── session_active_timeout silence ──► SESSION SUSPENDED


SESSION SUSPENDED (mic off, context retained in memory)
  │
  │  Wake word listener remains active
  │  Display may show "BB is listening" indicator dimmed
  │
  ├── Wake word detected ──► SESSION ACTIVE (with existing context)
  │   Conversation resumes where it left off
  │
  └── session_suspend_timeout ──► SESSION ENDED


SESSION ENDED (post-conversation processing)
  │
  ├── Memory extraction (large model)
  │   Extract facts, person memories, invalidations
  │
  ├── Voice-confirmed visual embeddings stored
  │   (perception pipeline — voice gates vision)
  │
  ├── Centroid recomputation for identified speakers
  │
  ├── Conversation summary → conversation log
  │
  └── Return to IDLE
```

### BB-Initiated Conversations

When a trigger fires and the target person is present (perception
confirms), BB initiates conversation by speaking directly:

- "Jacob, your wife has an update for you."
- "Hey, just a reminder — the dentist appointment is at four."
- "The package you were expecting just arrived, I saw the delivery."

No chime, no notification sound. BB speaks as a person in the room
would. If the person responds, the session proceeds normally. If they
say "Not now BB" or similar, the agent creates a new person trigger
to retry later using the existing task system.

BB only initiates when perception confirms the person is present.
If the target person hasn't been seen, the trigger remains pending
(this is how person triggers already work in the scheduler).

### Multi-Speaker Conversations

Because pyannote provides speaker attribution, BB handles multi-speaker
conversations naturally:

```
[Jacob]: What should we do for dinner tonight?
[Carina]: I was thinking maybe Italian?
[Jacob]: That sounds good. BB, do we have pasta?
[BB]: You have spaghetti and penne in the pantry.
[Carina]: What about sauce?
[BB]: I don't have inventory for sauces. Want me to add
      "check pasta sauce" to your shopping list?
```

The agent sees a multi-party conversation and participates as a third
member. Its behavior is governed by personality guidance in system
memory:

- **When addressed directly** → respond
- **When relevant context exists** → may interject during a natural
  pause
- **When two people are talking to each other** → stay quiet unless
  time-sensitive or directly relevant information warrants interjection
- **When told "not now"** → back off, retry later if needed

These guidelines live in system memory (the always-loaded markdown
file) as natural language, not as code-level settings. Users can adjust
them through conversation: "BB, don't interrupt me when I'm working"
creates a person-level memory. "BB, always tell me about my reminders
right away" creates another. The agent reads these memories and adapts
its behavior per person.

### Configuration

```yaml
voice:
  session:
    active_timeout: 30          # seconds of silence before suspending
    suspend_timeout: 180        # seconds before ending suspended session
    max_session_duration: 600   # seconds — hard cap on session length
```

## Response Latency Budget

### Standard Response (User Speaks → BB Responds)

| Stage | Duration | Notes |
|-------|----------|-------|
| Silence persistence | ~800ms | Configurable, waits for turn yield |
| STT (batch) | ~1-2s | Full utterance to ElevenLabs Scribe |
| Diarization | ~500ms | Parallel with STT |
| Agent processing | ~1-2s | Large model, streaming response |
| TTS first audio | ~200ms | Streaming from ElevenLabs |
| **Total to first spoken word** | **~3-4s** | From end of speech |

This is honest latency. It's not instant, but it's natural for a
household assistant — like asking someone across the room a question.
They think for a moment, then answer.

### Barge-In Response (User Interrupts BB)

| Stage | Duration | Notes |
|-------|----------|-------|
| Speech detected (AEC) | ~20ms | XMOS real-time |
| VAD confirmation | ~30ms | Silero |
| Volume fade begins | 200ms | Configurable |
| BB fully stops | 400ms | Configurable |
| **Total to BB stopping** | **~400ms** | Natural yielding speed |

After BB stops, the interrupting speech is processed through the
standard pipeline (accumulate → STT → agent).

### Future Latency Optimizations

**Streaming STT** — process audio while the person is still speaking.
The transcript builds incrementally. Would save ~1-2s from the total
latency budget. Implementation complexity is moderate; deferred to
post-v1.

**Speculative processing** — begin agent processing on partial
transcripts, discard if the utterance changes. High complexity, high
reward. Deferred to post-v1.

## Audio Format and Quality

### Capture

| Parameter | Value |
|-----------|-------|
| Sample rate | 16kHz (sufficient for speech, standard for STT) |
| Bit depth | 16-bit PCM |
| Channels | 1 (mono, extracted from 6-channel capture) |
| Format | WAV/PCM for processing, compressed for API calls |

The ReSpeaker captures 6 channels (16-bit, 16kHz): 4 raw microphone
channels plus 2 processed channels (beamformed/AEC'd). The HAL
extracts channel 0 (processed mono) for downstream consumers.

### Playback

| Parameter | Value |
|-----------|-------|
| Sample rate | 24kHz or 44.1kHz (from TTS provider) |
| Format | PCM stream from ElevenLabs |
| Output | ALSA → Waveshare speaker |

## Privacy

- All speech processing is local except STT and TTS API calls
- Audio is never stored permanently — processed and discarded
- Only the text transcript enters the conversation log
- pyannote diarization and speaker embeddings run entirely on-device
- Wake word detection runs entirely on-device
- VAD runs entirely on-device
- The only data leaving the box: audio to ElevenLabs (STT/TTS),
  text to Claude API

## Integration with Other Systems

### Perception Pipeline

The voice pipeline and perception pipeline share data:
- **pyannote speaker embeddings** → matched against person voice clouds
  managed by the perception system
- **DOA from ReSpeaker** → used by perception to resolve voice-visual
  conflicts (someone speaking off-camera)
- **Person identity from perception** → injected into conversation
  context as speaker attribution
- **Post-session** → voice-confirmed visual embeddings stored via
  perception's enrollment rules

### Memory System

- **Conversation context** → memories injected at session start based
  on who is speaking and what they said
- **Post-session** → memory extraction agent processes the conversation
  transcript
- **Personality preferences** → stored as person memories, read by the
  agent to adjust conversational behavior

### Task/Trigger System

- **Person triggers** → fire when perception detects the target person,
  causing BB to initiate a conversation
- **"Not now"** → agent creates a new person trigger to retry later
- **Wake cycle triggers** → can initiate BB-initiated conversations
  when the person is present

### Display System

- **Agent state** → display shows "Listening," "Thinking," "Speaking"
  during active sessions
- **Session suspended** → display may show a subtle indicator that
  context is retained

## Configuration Reference

All voice pipeline parameters in one place:

```yaml
voice:
  # Wake word detection
  wake_word:
    engine: "openwakeword"
    word: "bb"
    confidence_threshold: 0.7
    model_path: "models/wake_word/"

  # Voice activity detection
  vad:
    engine: "silero"
    threshold: 0.5
    min_speech_duration: 250     # ms
    min_silence_duration: 100    # ms

  # Turn detection
  turn_detection:
    silence_threshold: 800       # ms — silence before finalizing
    max_utterance_duration: 60   # seconds — hard cap
    inter_utterance_gap: 300     # ms — minimum between utterances

  # Interruption is wake-word-only — STT detaches while BB speaks; the
  # wake word is the only path back to a hot mic mid-reply. STT
  # automatically re-attaches on natural TTS completion so the user
  # can continue without re-saying the wake word.

  # Speaker diarization
  diarization:
    engine: "pyannote"
    model: "pyannote/speaker-diarization-3.1"
    embedding_model: "pyannote/wespeaker-voxceleb-resnet34-LM"
    min_speakers: 1
    max_speakers: 6
    match_threshold: 0.65

  # Speech-to-text
  stt:
    provider: "elevenlabs"
    model: "scribe_v1"
    language: "en"
    elevenlabs:
      api_key_secret: "elevenlabs_api_key"

  # Text-to-speech
  tts:
    provider: "elevenlabs"
    voice_id: "configured_voice_id"
    model: "eleven_turbo_v2_5"
    stability: 0.5
    similarity_boost: 0.75
    elevenlabs:
      api_key_secret: "elevenlabs_api_key"
      optimize_streaming_latency: 3

  # Session management
  session:
    active_timeout: 30           # seconds before suspending
    suspend_timeout: 180         # seconds before ending
    max_session_duration: 600    # seconds hard cap

  # Audio capture
  audio:
    sample_rate: 16000
    bit_depth: 16
    channels: 1                  # mono after beamforming
```

## File Layout

```
src/boxbot/communication/
  voice.py          # Voice adapter, orchestrates pipeline
  wake_word.py      # OpenWakeWord integration
  vad.py            # Silero VAD wrapper
  stt.py            # STT provider interface + ElevenLabs Scribe
  tts.py            # TTS provider interface + ElevenLabs
  audio_capture.py  # ReSpeaker audio stream management
  diarization.py    # pyannote integration (shared with perception)
```

Note: `diarization.py` is shared between the voice pipeline and the
perception pipeline. During conversation, pyannote runs for both
speaker attribution (voice pipeline) and speaker identification
(perception pipeline). These share the same audio stream and
diarization output to avoid duplicate processing.
