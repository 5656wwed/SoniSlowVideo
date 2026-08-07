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
- To **clone**: put your sample + its exact text into the repo's `_F5TTS_/` folder:
  - `_F5TTS_/myvoice.wav` (or .mp3) + `_F5TTS_/myvoice.txt` (the exact words it says)
  - Restart → a new `en-F5-myvoice F5-TTS` voice appears in the TTS list.

## Behavior

- Kokoro stays near natural speed

- Audio is not force-sped to fit short SRT slots

- Final video is slowed (`setpts`) so length matches the voice track

- SRT mode still skips translation/diarization

## Do not confuse with

- Main stable app: https://colab.research.google.com/github/5656wwed/SoniTranslate/blob/main/SoniTranslate_Colab.ipynb
