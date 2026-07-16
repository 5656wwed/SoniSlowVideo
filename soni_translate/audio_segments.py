from pydub import AudioSegment
from tqdm import tqdm
from .utils import run_command
from .logging_setup import logger
import numpy as np


class Mixer:
    def __init__(self):
        self.parts = []

    def __len__(self):
        parts = self._sync()
        seg = parts[0][1]
        frame_count = max(offset + seg.frame_count() for offset, seg in parts)
        return int(1000.0 * frame_count / seg.frame_rate)

    def overlay(self, sound, position=0):
        self.parts.append((position, sound))
        return self

    def _sync(self):
        positions, segs = zip(*self.parts)

        frame_rate = segs[0].frame_rate
        array_type = segs[0].array_type # noqa

        offsets = [int(frame_rate * pos / 1000.0) for pos in positions]
        segs = AudioSegment.empty()._sync(*segs)
        return list(zip(offsets, segs))

    def append(self, sound):
        self.overlay(sound, position=len(self))

    def to_audio_segment(self):
        parts = self._sync()
        seg = parts[0][1]
        channels = seg.channels

        frame_count = max(offset + seg.frame_count() for offset, seg in parts)
        sample_count = int(frame_count * seg.channels)

        output = np.zeros(sample_count, dtype="int32")
        for offset, seg in parts:
            sample_offset = offset * channels
            samples = np.frombuffer(seg.get_array_of_samples(), dtype="int32")
            samples = np.int16(samples/np.max(np.abs(samples)) * 32767)
            start = sample_offset
            end = start + len(samples)
            output[start:end] += samples

        return seg._spawn(
            output, overrides={"sample_width": 4}).normalize(headroom=0.0)


def create_translated_audio(
    result_diarize,
    audio_files,
    final_file,
    concat=False,
    avoid_overlap=False,
    smart_pack=False,
):
    total_duration = result_diarize["segments"][-1]["end"]  # in seconds

    if concat:
        """
        file .\audio\1.ogg
        file .\audio\2.ogg
        file .\audio\3.ogg
        file .\audio\4.ogg
        ...
        """

        # Write the file paths to list.txt
        with open("list.txt", "w") as file:
            for i, audio_file in enumerate(audio_files):
                if i == len(audio_files) - 1:  # Check if it's the last item
                    file.write(f"file {audio_file}")
                else:
                    file.write(f"file {audio_file}\n")

        # command = f"ffmpeg -f concat -safe 0 -i list.txt {final_file}"
        command = (
            f"ffmpeg -f concat -safe 0 -i list.txt -c:a pcm_s16le {final_file}"
        )
        run_command(command)

    else:
        # First pass: decide placement times (smart_pack collapses dead air)
        BREATH = 0.05  # max silence between lines when smart packing
        placements = []  # (start_sec, audio_file, AudioSegment)
        last_end_time = 0.0
        previous_speaker = ""
        collapsed = 0

        for idx, (line, audio_file) in enumerate(
            tqdm(zip(result_diarize["segments"], audio_files))
        ):
            try:
                audio = AudioSegment.from_file(audio_file)
            except Exception as error:
                logger.debug(str(error))
                logger.error(f"Error audio file {audio_file}")
                continue

            srt_start = float(line["start"])
            dur = len(audio) / 1000.0
            start = srt_start

            if smart_pack:
                if idx == 0:
                    start = srt_start
                else:
                    natural_gap = srt_start - last_end_time
                    if natural_gap > BREATH:
                        # Dead air from SRT timing → collapse to tiny breath
                        start = last_end_time + BREATH
                        collapsed += 1
                        if collapsed <= 12 or collapsed % 25 == 0:
                            logger.info(
                                f"Smart pack {audio_file}: "
                                f"gap {natural_gap:.2f}s → {BREATH:.2f}s"
                            )
                    elif natural_gap < 0:
                        # Would overlap → push after previous + breath
                        start = last_end_time + BREATH
                    else:
                        start = srt_start
                last_end_time = start + dur
            elif avoid_overlap:
                speaker = line.get("speaker", "")
                if (last_end_time - 0.500) > start:
                    overlap_time = last_end_time - start
                    if previous_speaker and previous_speaker != speaker:
                        start = (last_end_time - 0.500)
                    else:
                        start = (last_end_time - 0.200)
                    if overlap_time > 2.5:
                        start = start - 0.3
                    logger.info(
                        f"Avoid overlap for {str(audio_file)} "
                        f"with {str(start)}"
                    )
                previous_speaker = speaker
                last_end_time = start + dur
            else:
                last_end_time = max(last_end_time, start + dur)

            placements.append((start, audio_file, audio))

        if smart_pack:
            logger.info(
                f"Smart pack: collapsed {collapsed} dead gaps "
                f"(breath={BREATH:.2f}s); audio ends {last_end_time:.1f}s"
            )
            # Canvas = packed length only (no trailing SRT dead air)
            total_duration = max(last_end_time + 0.05, 0.2)
        else:
            # Canvas long enough for SRT timeline + any overrun
            total_duration = max(float(total_duration), last_end_time + 0.05)
        base_audio = AudioSegment.silent(
            duration=int(total_duration * 1000), frame_rate=41000
        )
        combined_audio = Mixer()
        combined_audio.overlay(base_audio)

        logger.debug(
            f"Audio duration: {total_duration // 60} "
            f"minutes and {int(total_duration % 60)} seconds"
        )

        for start, audio_file, audio in placements:
            try:
                combined_audio = combined_audio.overlay(
                    audio, position=int(start * 1000)
                )
            except Exception as error:
                logger.debug(str(error))
                logger.error(f"Error audio file {audio_file}")

        # combined audio as a file
        combined_audio_data = combined_audio.to_audio_segment()
        combined_audio_data.export(
            final_file, format="wav"
        )  # best than ogg, change if the audio is anomalous
