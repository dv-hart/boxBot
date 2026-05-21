# Voice-ID diagnosis harness

Tools to find out **why same-speaker speaker-embedding similarity is so
low** (clean same-session same-speaker cosine was topping out ~0.58,
below the `speaker_threshold` of 0.75 — so nobody ever matches). We
characterise the real pipeline on realistic short utterances before
changing anything.

Two scripts:

- `capture_voice.py` — a hands-off **continuous recorder**. Saves all
  six ReSpeaker channels for every VAD-gated utterance, each with a
  wall-clock timestamp (production VAD + OpenWakeWord run on channel 0,
  exactly as live). No prompts, no typing. Output:
  `data/voice_diag/_inbox/utt_<epoch_ms>.{wav,json}`.
- `analyze_voice.py` — offline analyses over the (labelled) recordings.

Recordings live under `data/voice_diag/` (gitignored, Pi-local) — voice
audio is never committed.

## Why 6 channels

Production extracts `output_channel=0` and *assumes* it's the
beamformed/AEC channel (`microphone.py`), but that was never verified
against this unit's firmware. Saving all six lets analysis #1 prove
which channel is actually cleanest without re-recording.

## Hypotheses under test

| # | Analysis | Hypothesis |
|---|----------|-----------|
| 1 | per-channel SNR | We may not be embedding the beamformed channel |
| 2 | whole vs diarization fragments | pyannote shreds one clean utterance into weak sub-second embeddings (and costs 3.6–4.5 s latency); single-speaker should bypass diarization |
| 3 | cosine vs clip length | matching must work down to ~1.5 s real commands |
| 4 | quiet vs dishwasher, per channel | noise tanks embeddings even when a human hears fine; does the beamformed channel recover it? |
| 5 | within- vs cross-speaker → EER | `speaker_threshold` should come from the measured margin, not a guess |

Also on the suspect list (documented in `wipe_voice_enrollment.py`):
BB's own TTS leaking into the mic during cold-start AEC, enrolling its
own voice as the user.

## Workflow

The recorder is hands-off; the only manual step is jotting down
"time / who / condition" while you talk. Claude does the rest.

1. **Stop boxbot** (the capture device is exclusive; also avoids running
   heavy diagnostics against the live process):
   ```
   pkill -f '\.venv/bin/boxbot$'
   ```
2. **Start the recorder** (background):
   ```
   nohup python3 scripts/diag/capture_voice.py > logs/voice_capture.log 2>&1 &
   ```
3. **Talk.** Walk up and say the wake word + a command a few
   times under each condition. For each batch, note the clock time,
   speaker, and condition, e.g.:
   ```
   14:05  jacob  quiet_close       (3 commands)
   14:08  sarah  quiet_close
   14:12  jacob  dishwasher_close  (dishwasher running)
   ...
   ```
   Suggested matrix: `{quiet,dishwasher} × {close ~1 m, far ~3 m}` for
   both speakers; ≥3 utterances each. Talk however you normally do —
   `analyze_voice.py` is text-independent.
4. **Stop the recorder** (`kill <pid>` / Ctrl-C). Hand the notes to
   Claude. Each clip's timestamp is in its `.json`; Claude writes the
   matching `speaker`/`condition` into every sidecar, then runs:
   ```
   python3 scripts/diag/analyze_voice.py --in-dir data/voice_diag --channel 0 --noise-channels 0,1,5
   ```
   (unlabeled clips are skipped automatically). After analysis #1 names
   the cleanest channel, re-run with `--channel <best>`.
5. **Restart boxbot:** `./scripts/restart-boxbot.sh`.

## Reading the results → the fix

- #2 shows WHOLE ≫ FRAG and frequent false splits → **bypass diarization
  for single-speaker utterances, embed the whole utterance.**
- #3 rises sharply with length → **min-duration floor + merge contiguous
  same-speaker segments before embedding.**
- #1/#4 show another channel beats ch0 (esp. under noise) → **change
  `output_channel`.**
- #5 gives the real same/cross margin → **set `speaker_threshold` to the
  EER point** (and gate enrollment well above it).
