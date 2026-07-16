from pydub import AudioSegment
from tqdm import tqdm
from .utils import run_command
from .logging_setup import logger
import numpy as np
import json
import os
import subprocess
import time


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
        array_type = segs[0].array_type  # noqa

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
            samples = np.int16(samples / np.max(np.abs(samples)) * 32767)
            start = sample_offset
            end = start + len(samples)
            output[start:end] += samples

        return seg._spawn(
            output, overrides={"sample_width": 4}
        ).normalize(headroom=0.0)


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
        with open("list.txt", "w") as file:
            for i, audio_file in enumerate(audio_files):
                if i == len(audio_files) - 1:
                    file.write(f"file {audio_file}")
                else:
                    file.write(f"file {audio_file}\n")

        command = (
            f"ffmpeg -f concat -safe 0 -i list.txt -c:a pcm_s16le {final_file}"
        )
        run_command(command)

    else:
        # First pass: decide placement times (smart_pack collapses dead air)
        BREATH = 0.05  # max silence between lines when smart packing
        placements = []  # (pack_start, audio_file, audio, srt_start, srt_end)
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
            srt_end = float(line.get("end", srt_start + 0.5))
            dur = len(audio) / 1000.0
            start = srt_start

            if smart_pack:
                if idx == 0:
                    start = 0.0  # continuous narration from t=0
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
                        start = last_end_time - 0.500
                    else:
                        start = last_end_time - 0.200
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

            placements.append((start, audio_file, audio, srt_start, srt_end))

        if smart_pack:
            logger.info(
                f"Smart pack: collapsed {collapsed} dead gaps "
                f"(breath={BREATH:.2f}s); audio ends {last_end_time:.1f}s"
            )
            # Canvas = packed length only (no trailing SRT dead air)
            total_duration = max(last_end_time + 0.05, 0.2)
            # Timeline for per-scene video sync
            tl = {
                "breath": BREATH,
                "audio_end": float(last_end_time),
                "segments": [
                    {
                        "pack_start": float(ps),
                        "pack_dur": float(len(au) / 1000.0),
                        "srt_start": float(ss),
                        "srt_end": float(se),
                    }
                    for (ps, _af, au, ss, se) in placements
                ],
            }
            with open("smart_pack_timeline.json", "w") as fh:
                json.dump(tl, fh)
            logger.info(
                f"Wrote smart_pack_timeline.json ({len(tl['segments'])} scenes)"
            )
        else:
            total_duration = max(float(total_duration), last_end_time + 0.05)
            if os.path.isfile("smart_pack_timeline.json"):
                try:
                    os.remove("smart_pack_timeline.json")
                except Exception:
                    pass

        base_audio = AudioSegment.silent(
            duration=int(total_duration * 1000), frame_rate=41000
        )
        combined_audio = Mixer()
        combined_audio.overlay(base_audio)

        logger.debug(
            f"Audio duration: {total_duration // 60} "
            f"minutes and {int(total_duration % 60)} seconds"
        )

        for start, audio_file, audio, _ss, _se in placements:
            try:
                combined_audio = combined_audio.overlay(
                    audio, position=int(start * 1000)
                )
            except Exception as error:
                logger.debug(str(error))
                logger.error(f"Error audio file {audio_file}")

        combined_audio_data = combined_audio.to_audio_segment()
        combined_audio_data.export(
            final_file, format="wav"
        )  # best than ogg, change if the audio is anomalous


def _ffprobe_duration(path):
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        text=True,
    ).strip()
    return float(out)


def build_scene_synced_video(
    video_in,
    audio_in,
    video_out,
    timeline_path="smart_pack_timeline.json",
):
    """Smart-pack audio + per-SRT-scene video retime so picture matches each line.

    Each original SRT span is sped/slowed to the packed audio span for that line.
    Keeps continuous narration flow AND scene sync.
    """
    if not os.path.isfile(timeline_path):
        raise FileNotFoundError(timeline_path)
    with open(timeline_path) as fh:
        tl = json.load(fh)
    segs = tl.get("segments") or []
    if not segs:
        raise ValueError("empty smart_pack timeline")

    v_dur = _ffprobe_duration(video_in)
    a_dur = _ffprobe_duration(audio_in)
    n = len(segs)
    work = "video_pieces_sync"
    os.makedirs(work, exist_ok=True)
    for name in os.listdir(work):
        try:
            os.remove(os.path.join(work, name))
        except Exception:
            pass

    piece_paths = []
    logger.info(
        f"Scene-sync video: {n} pieces, voice {a_dur:.1f}s / src video {v_dur:.1f}s"
    )
    t0 = time.time()

    for i, seg in enumerate(segs):
        src0 = float(seg["srt_start"])
        if i + 1 < n:
            src1 = float(segs[i + 1]["srt_start"])
            out_dur = float(segs[i + 1]["pack_start"]) - float(seg["pack_start"])
        else:
            src1 = min(v_dur, max(float(seg["srt_end"]), src0 + 0.15))
            out_dur = max(
                float(seg["pack_dur"]), a_dur - float(seg["pack_start"])
            )
        src_dur = max(0.05, src1 - src0)
        out_dur = max(0.05, out_dur)
        ratio = out_dur / src_dur
        # Soft clamp extreme warps
        ratio = max(0.45, min(2.2, ratio))

        piece = os.path.join(work, f"p{i:05d}.mp4")
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{src0:.3f}",
            "-i",
            video_in,
            "-t",
            f"{src_dur:.3f}",
            "-an",
            "-filter:v",
            f"setpts={ratio:.6f}*PTS",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-threads",
            "0",
            piece,
        ]
        rc = subprocess.run(cmd, capture_output=True, text=True)
        if rc.returncode != 0 or not os.path.isfile(piece):
            logger.warning(f"piece {i} failed, retry trim-only")
            cmd2 = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{src0:.3f}",
                "-i",
                video_in,
                "-t",
                f"{min(src_dur, out_dur):.3f}",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "23",
                piece,
            ]
            subprocess.run(cmd2, capture_output=True, text=True)

        if os.path.isfile(piece) and os.path.getsize(piece) > 500:
            piece_paths.append(piece)
        if i < 5 or i % 50 == 0 or i == n - 1:
            logger.info(
                f"Scene-sync piece {i + 1}/{n}: "
                f"srt {src0:.1f}-{src1:.1f}s → {out_dur:.2f}s (x{ratio:.2f})"
            )

    if not piece_paths:
        raise RuntimeError("no video pieces built")

    list_file = os.path.join(work, "concat.txt")
    with open(list_file, "w") as fh:
        for p in piece_paths:
            fh.write("file '" + os.path.abspath(p) + "'\n")

    silent_video = os.path.join(work, "video_only.mp4")
    cmd_cat = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_file,
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "23",
        "-an",
        silent_video,
    ]
    logger.info("Scene-sync: concatenating pieces...")
    rc = subprocess.run(cmd_cat, capture_output=True, text=True)
    if rc.returncode != 0 or not os.path.isfile(silent_video):
        err = (rc.stderr or "")[-400:]
        raise RuntimeError(f"concat failed: {err}")

    cmd_mux = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        silent_video,
        "-i",
        audio_in,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        video_out,
    ]
    logger.info("Scene-sync: muxing packed voice...")
    rc = subprocess.run(cmd_mux, capture_output=True, text=True)
    if rc.returncode != 0 or not os.path.isfile(video_out):
        cmd_mux = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            silent_video,
            "-i",
            audio_in,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            video_out,
        ]
        rc = subprocess.run(cmd_mux, capture_output=True, text=True)
        if rc.returncode != 0:
            err = (rc.stderr or "")[-400:]
            raise RuntimeError(f"mux failed: {err}")

    logger.info(
        f"Scene-sync done in {(time.time() - t0) / 60:.1f} min → {video_out}"
    )
    return video_out
