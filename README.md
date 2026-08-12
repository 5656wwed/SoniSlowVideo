# Gudmarv Easy dubbing

**Slow the VIDEO to fit natural voice** — not the other way around.

Fork of [5656wwed/SoniTranslate](https://github.com/5656wwed/SoniTranslate) with stretch-video mode ON by default.

## Colab

dots-tts-clean branch (Pocket TTS with voice cloning + Kokoro):
https://colab.research.google.com/github/5656wwed/SoniSlowVideo/blob/dots-tts-clean/SoniSlowVideo_Colab.ipynb

Main (stable) branch:
https://colab.research.google.com/github/5656wwed/SoniSlowVideo/blob/main/SoniSlowVideo_Colab.ipynb

## Pocket TTS with voice cloning (dots-tts-clean branch)

- **Pocket TTS** now accepts **cloned voices**: drop a `.wav`/`.mp3` sample into
  `./_POCKET_/` and it appears in the TTS list as `en-Pocket-<name> Pocket-TTS`.
  No transcript needed — Pocket clones straight from the sample.
- To **clone**: upload your sample **once** through the Colab UI (Step 1.5). It's
  saved to **Google Drive** (`MyDrive/SoniSlow_Pocket_Voices/`) and automatically
  reloaded on every future run.
- Keeps `torch==2.5.1` so whisperX / pyannote are unaffected.
- Also includes the 19 built-in Pocket voices plus **Kokoro** (natural, fast).

## Behavior

- Kokoro stays near natural speed
- Audio is not force-sped to fit short SRT slots
- Final video is slowed (`setpts`) so length matches the voice track
- SRT mode still skips translation/diarization

## Do not confuse with

- Main stable app: https://colab.research.google.com/github/5656wwed/SoniTranslate/blob/main/SoniTranslate_Colab.ipynb
