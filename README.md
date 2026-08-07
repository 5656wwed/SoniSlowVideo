# SoniSlowVideo

**Slow the VIDEO to fit natural voice** — not the other way around.

Fork of [5656wwed/SoniTranslate](https://github.com/5656wwed/SoniTranslate) with stretch-video mode ON by default.

## Colab (different link)

F5-TTS branch (clean modern English + voice cloning):
https://colab.research.google.com/github/5656wwed/SoniSlowVideo/blob/f5-tts/SoniSlowVideo_Colab.ipynb

Main (stable) branch:
https://colab.research.google.com/github/5656wwed/SoniSlowVideo/blob/main/SoniSlowVideo_Colab.ipynb

## F5-TTS engine (f5-tts branch)

- Adds a **F5-TTS** engine: clean modern English, zero-shot **voice cloning**.
- Pick `en-F5-Default F5-TTS` for a good default voice (no sample needed).
- To **clone**: upload your sample **once** through the Colab UI (Step 1.5). It's
  saved to **Google Drive** (`MyDrive/SoniSlow_F5_Voices/`) and automatically
  reloaded on every future run — no re-uploading, no editing files.
  - The sample needs a `.wav`/`.mp3` plus the **exact text** it says (you type
    it once when uploading).
  - Your clone appears in the TTS list as `en-F5-<name> F5-TTS`.

## Behavior

- Kokoro stays near natural speed

- Audio is not force-sped to fit short SRT slots

- Final video is slowed (`setpts`) so length matches the voice track

- SRT mode still skips translation/diarization

## Do not confuse with

- Main stable app: https://colab.research.google.com/github/5656wwed/SoniTranslate/blob/main/SoniTranslate_Colab.ipynb
