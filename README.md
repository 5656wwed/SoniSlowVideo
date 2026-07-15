# SoniSlowVideo

**Slow the VIDEO to fit natural voice** — not the other way around.

Fork of [5656wwed/SoniTranslate](https://github.com/5656wwed/SoniTranslate) with stretch-video mode ON by default.

## Colab (different link)

https://colab.research.google.com/github/5656wwed/SoniSlowVideo/blob/main/SoniSlowVideo_Colab.ipynb

## Behavior

- Kokoro stays near natural speed

- Audio is not force-sped to fit short SRT slots

- Final video is slowed (`setpts`) so length matches the voice track

- SRT mode still skips translation/diarization

## Do not confuse with

- Main stable app: https://colab.research.google.com/github/5656wwed/SoniTranslate/blob/main/SoniTranslate_Colab.ipynb
