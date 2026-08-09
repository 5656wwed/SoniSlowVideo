# SoniSlowVideo

**Slow the VIDEO to fit natural voice** — not the other way around.

Fork of [5656wwed/SoniTranslate](https://github.com/5656wwed/SoniTranslate) with stretch-video mode ON by default.

## Colab (different link)

dots.tts branch (voice cloning + clean natural voice):
https://colab.research.google.com/github/5656wwed/SoniSlowVideo/blob/dots-tts/SoniSlowVideo_Colab.ipynb

Main (stable) branch:
https://colab.research.google.com/github/5656wwed/SoniSlowVideo/blob/main/SoniSlowVideo_Colab.ipynb

## dots.tts engine (dots-tts branch)

- Adds a **dots.tts** engine: fully continuous 2B autoregressive TTS (Apache-2.0,
  public HF checkpoints — free, no API key) with zero-shot **voice cloning**.
- To **clone**: upload your sample **once** through the Colab UI (Step 1.5). It's
  saved to **Google Drive** (`MyDrive/SoniSlow_Dots_Voices/`) and automatically
  reloaded on every future run — no re-uploading, no editing files.
  - The sample needs a `.wav`/`.mp3` plus the **exact text** it says (you type
    it once when uploading).
  - Your clone appears in the TTS list as `en-Dots-<name> Dots-TTS`.
- Requires `torch>=2.8.0` (the notebook installs it).

## Behavior

- Kokoro stays near natural speed

- Audio is not force-sped to fit short SRT slots

- Final video is slowed (`setpts`) so length matches the voice track

- SRT mode still skips translation/diarization

## Do not confuse with

- Main stable app: https://colab.research.google.com/github/5656wwed/SoniTranslate/blob/main/SoniTranslate_Colab.ipynb
