# Voice-ID diagnosis harness

Tools to find out **why same-speaker speaker-embedding similarity is so
low** (clean same-session same-speaker cosine was topping out ~0.58,
below the `speaker_threshold` of 0.75 — so nobody ever matches). We
characterise the real pipeline on realistic short utterances before
changing anything.

Two scripts:

- `capture_voice.py` — records the **full 6-channel** ReSpeaker stream
  for each wake-word-gated utterance (production VAD + OpenWakeWord on
  channel 0 drive the gating; all six channels are saved). Output:
  `data/voice_diag/<speaker>/<condition>/trialNN_<cmd>.wav` + `.json`.
- `analyze_voice.py` — offline analyses over those recordings.

Recordings live under `data/` (gitignored, Pi-local) — voice audio is
never committed.

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

## Protocol

Run on the Pi, **with boxbot stopped** (the capture device is
exclusive). Asymmetry to keep in mind: enrollment can be picky (prefer
long/clean/quiet), but matching must survive short + noisy.

1. Stop boxbot.
2. For each condition below, run one capture pass per speaker. Speak
   each prompted command naturally, starting with "hey jarvis".

   ```
   python3 scripts/diag/capture_voice.py --speaker jacob --condition quiet_close
   python3 scripts/diag/capture_voice.py --speaker jacob --condition quiet_far
   python3 scripts/diag/capture_voice.py --speaker jacob --condition dishwasher_close
   python3 scripts/diag/capture_voice.py --speaker jacob --condition dishwasher_far
   # repeat all four for --speaker sarah  (needed for analysis #5)
   ```

   - `*_close` ≈ 1 m, `*_far` ≈ 3 m.
   - Run `dishwasher_*` with the dishwasher actually running.
   - The default 6-command list repeats naturally; add more passes for
     more samples per condition (more is better for #3/#5).

3. Analyse:

   ```
   HF_TOKEN=$HF_TOKEN python3 scripts/diag/analyze_voice.py \
       --in-dir data/voice_diag --channel 0 --noise-channels 0,1,5
   ```

   `--channel` is the primary channel for #2/#3/#5; `--noise-channels`
   is the per-channel sweep in #4. After analysis #1 names the cleanest
   channel, re-run with `--channel <best>` to see the ceiling.
   `--skip-diarization` skips the heavy pyannote pipeline in #2.

## Reading the results → the fix

- #2 shows WHOLE ≫ FRAG and frequent false splits → **bypass diarization
  for single-speaker utterances, embed the whole utterance.**
- #3 rises sharply with length → **min-duration floor + merge contiguous
  same-speaker segments before embedding.**
- #1/#4 show another channel beats ch0 (esp. under noise) → **change
  `output_channel`.**
- #5 gives the real same/cross margin → **set `speaker_threshold` to the
  EER point** (and gate enrollment well above it).
