from gtts import gTTS
import edge_tts, asyncio, json, glob # noqa
from tqdm import tqdm
import librosa, os, re, torch, gc, subprocess, time # noqa
from .language_configuration import (
    fix_code_language,
    BARK_VOICES_LIST,
    VITS_VOICES_LIST,
)
from .utils import (
    download_manager,
    create_directories,
    copy_files,
    rename_file,
    remove_directory_contents,
    remove_files,
    run_command,
    write_chunked,
)
import numpy as np
from typing import Any, Dict
from pathlib import Path
import soundfile as sf
import platform
import logging
import traceback
import sys
from .logging_setup import logger


class TTS_OperationError(Exception):
    def __init__(self, message="The operation did not complete successfully."):
        self.message = message
        super().__init__(self.message)


def verify_saved_file_and_size(filename):
    if not os.path.exists(filename):
        raise TTS_OperationError(f"File '{filename}' was not saved.")
    if os.path.getsize(filename) == 0:
        raise TTS_OperationError(
            f"File '{filename}' has a zero size. "
            "Related to incorrect TTS for the target language"
        )


def error_handling_in_tts(error, segment, TRANSLATE_AUDIO_TO, filename):
    traceback.print_exc()
    logger.error(f"Error: {str(error)}")
    try:
        from tempfile import TemporaryFile

        tts = gTTS(segment["text"], lang=fix_code_language(TRANSLATE_AUDIO_TO))
        # tts.save(filename)
        f = TemporaryFile()
        tts.write_to_fp(f)

        # Reset the file pointer to the beginning of the file
        f.seek(0)

        # Read audio data from the TemporaryFile using soundfile
        audio_data, samplerate = sf.read(f)
        f.close()  # Close the TemporaryFile
        write_chunked(
            filename, audio_data, samplerate, format="ogg", subtype="vorbis"
        )

        logger.warning(
            'TTS auxiliary will be utilized '
            f'rather than TTS: {segment["tts_name"]}'
        )
        verify_saved_file_and_size(filename)
    except Exception as error:
        logger.critical(f"Error: {str(error)}")
        sample_rate_aux = 22050
        duration = float(segment["end"]) - float(segment["start"])
        data = np.zeros(int(sample_rate_aux * duration)).astype(np.float32)
        write_chunked(
            filename, data, sample_rate_aux, format="ogg", subtype="vorbis"
        )
        logger.error("Audio will be replaced -> [silent audio].")
        verify_saved_file_and_size(filename)


def pad_array(array, sr):

    if isinstance(array, list):
        array = np.array(array)

    if not array.shape[0]:
        raise ValueError("The generated audio does not contain any data")

    valid_indices = np.where(np.abs(array) > 0.001)[0]

    if len(valid_indices) == 0:
        logger.debug(f"No valid indices: {array}")
        return array

    try:
        pad_indice = int(0.1 * sr)
        start_pad = max(0, valid_indices[0] - pad_indice)
        end_pad = min(len(array), valid_indices[-1] + 1 + pad_indice)
        padded_array = array[start_pad:end_pad]
        return padded_array
    except Exception as error:
        logger.error(str(error))
        return array


# =====================================
# EDGE TTS
# =====================================


def edge_tts_voices_list():
    try:
        completed_process = subprocess.run(
            ["edge-tts", "--list-voices"], capture_output=True, text=True
        )
        lines = completed_process.stdout.strip().split("\n")
    except Exception as error:
        logger.debug(str(error))
        lines = []

    formatted_voices = []

    for line in lines:
        line = line.strip()

        if not line or line.startswith("Name") or line.startswith("---"):
            continue

        parts = line.split()

        if len(parts) >= 2:
            name = parts[0]
            gender = parts[1]
            formatted_voices.append(f"{name}-{gender}")

    if not formatted_voices:
        logger.warning(
            "The list of Edge TTS voices could not be obtained, "
            "switching to an alternative method"
        )
        tts_voice_list = asyncio.new_event_loop().run_until_complete(
            edge_tts.list_voices()
        )
        formatted_voices = sorted(
            [f"{v['ShortName']}-{v['Gender']}" for v in tts_voice_list]
        )

    # Only show US English voices (male + female)
    formatted_voices = sorted(
        v for v in formatted_voices if v.startswith("en-US-")
    )

    if not formatted_voices:
        logger.error("Can't get EDGE TTS - list voices")

    return formatted_voices


def segments_egde_tts(filtered_edge_segments, TRANSLATE_AUDIO_TO, is_gui):
    for segment in tqdm(filtered_edge_segments["segments"]):
        speaker = segment["speaker"] # noqa
        text = segment["text"]
        start = segment["start"]
        tts_name = segment["tts_name"]

        # make the tts audio
        filename = f"audio/{start}.ogg"
        temp_file = filename[:-3] + "mp3"

        logger.info(f"{text} >> {filename}")
        try:
            if is_gui:
                asyncio.run(
                    edge_tts.Communicate(
                        text, "-".join(tts_name.split("-")[:-1])
                    ).save(temp_file)
                )
            else:
                # nest_asyncio.apply() if not is_gui else None
                command = f'edge-tts -t "{text}" -v "{tts_name.replace("-Male", "").replace("-Female", "")}" --write-media "{temp_file}"'
                run_command(command)
            verify_saved_file_and_size(temp_file)

            data, sample_rate = sf.read(temp_file)
            data = pad_array(data, sample_rate)
            # os.remove(temp_file)

            # Save file
            write_chunked(
                file=filename,
                samplerate=sample_rate,
                data=data,
                format="ogg",
                subtype="vorbis",
            )
            verify_saved_file_and_size(filename)

        except Exception as error:
            error_handling_in_tts(error, segment, TRANSLATE_AUDIO_TO, filename)


# =====================================
# BARK TTS
# =====================================


def segments_bark_tts(
    filtered_bark_segments, TRANSLATE_AUDIO_TO, model_id_bark="suno/bark-small"
):
    from transformers import AutoProcessor, BarkModel
    from optimum.bettertransformer import BetterTransformer

    device = os.environ.get("SONITR_DEVICE")
    torch_dtype_env = torch.float16 if device == "cuda" else torch.float32

    # load model bark
    model = BarkModel.from_pretrained(
        model_id_bark, torch_dtype=torch_dtype_env
    ).to(device)
    model = model.to(device)
    processor = AutoProcessor.from_pretrained(
        model_id_bark, return_tensors="pt"
    )  # , padding=True
    if device == "cuda":
        # convert to bettertransformer
        model = BetterTransformer.transform(model, keep_original_model=False)
        # enable CPU offload
        # model.enable_cpu_offload()
    sampling_rate = model.generation_config.sample_rate

    # filtered_segments = filtered_bark_segments['segments']
    # Sorting the segments by 'tts_name'
    # sorted_segments = sorted(filtered_segments, key=lambda x: x['tts_name'])
    # logger.debug(sorted_segments)

    for segment in tqdm(filtered_bark_segments["segments"]):
        speaker = segment["speaker"] # noqa
        text = segment["text"]
        start = segment["start"]
        tts_name = segment["tts_name"]

        inputs = processor(text, voice_preset=BARK_VOICES_LIST[tts_name]).to(
            device
        )

        # make the tts audio
        filename = f"audio/{start}.ogg"
        logger.info(f"{text} >> {filename}")
        try:
            # Infer
            with torch.inference_mode():
                speech_output = model.generate(
                    **inputs,
                    do_sample=True,
                    fine_temperature=0.4,
                    coarse_temperature=0.8,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            # Save file
            data_tts = pad_array(
                speech_output.cpu().numpy().squeeze().astype(np.float32),
                sampling_rate,
            )
            write_chunked(
                file=filename,
                samplerate=sampling_rate,
                data=data_tts,
                format="ogg",
                subtype="vorbis",
            )
            verify_saved_file_and_size(filename)
        except Exception as error:
            error_handling_in_tts(error, segment, TRANSLATE_AUDIO_TO, filename)
        gc.collect()
        torch.cuda.empty_cache()
    try:
        del processor
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as error:
        logger.error(str(error))
        gc.collect()
        torch.cuda.empty_cache()


# =====================================
# VITS TTS
# =====================================


def uromanize(input_string):
    """Convert non-Roman strings to Roman using the `uroman` perl package."""
    # script_path = os.path.join(uroman_path, "bin", "uroman.pl")

    if not os.path.exists("./uroman"):
        logger.info(
            "Clonning repository uroman https://github.com/isi-nlp/uroman.git"
            " for romanize the text"
        )
        process = subprocess.Popen(
            ["git", "clone", "https://github.com/isi-nlp/uroman.git"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate()
    script_path = os.path.join("./uroman", "uroman", "uroman.pl")

    command = ["perl", script_path]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Execute the perl command
    stdout, stderr = process.communicate(input=input_string.encode())

    if process.returncode != 0:
        raise ValueError(f"Error {process.returncode}: {stderr.decode()}")

    # Return the output as a string and skip the new-line character at the end
    return stdout.decode()[:-1]


def segments_vits_tts(filtered_vits_segments, TRANSLATE_AUDIO_TO):
    from transformers import VitsModel, AutoTokenizer

    filtered_segments = filtered_vits_segments["segments"]
    # Sorting the segments by 'tts_name'
    sorted_segments = sorted(filtered_segments, key=lambda x: x["tts_name"])
    logger.debug(sorted_segments)

    model_name_key = None
    for segment in tqdm(sorted_segments):
        speaker = segment["speaker"] # noqa
        text = segment["text"]
        start = segment["start"]
        tts_name = segment["tts_name"]

        if tts_name != model_name_key:
            model_name_key = tts_name
            model = VitsModel.from_pretrained(VITS_VOICES_LIST[tts_name])
            tokenizer = AutoTokenizer.from_pretrained(
                VITS_VOICES_LIST[tts_name]
            )
            sampling_rate = model.config.sampling_rate

        if tokenizer.is_uroman:
            romanize_text = uromanize(text)
            logger.debug(f"Romanize text: {romanize_text}")
            inputs = tokenizer(romanize_text, return_tensors="pt")
        else:
            inputs = tokenizer(text, return_tensors="pt")

        # make the tts audio
        filename = f"audio/{start}.ogg"
        logger.info(f"{text} >> {filename}")
        try:
            # Infer
            with torch.no_grad():
                speech_output = model(**inputs).waveform

            data_tts = pad_array(
                speech_output.cpu().numpy().squeeze().astype(np.float32),
                sampling_rate,
            )
            # Save file
            write_chunked(
                file=filename,
                samplerate=sampling_rate,
                data=data_tts,
                format="ogg",
                subtype="vorbis",
            )
            verify_saved_file_and_size(filename)
        except Exception as error:
            error_handling_in_tts(error, segment, TRANSLATE_AUDIO_TO, filename)
        gc.collect()
        torch.cuda.empty_cache()
    try:
        del tokenizer
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as error:
        logger.error(str(error))
        gc.collect()
        torch.cuda.empty_cache()


# =====================================
# Coqui XTTS
# =====================================


def coqui_xtts_voices_list():
    main_folder = "_XTTS_"
    pattern_coqui = re.compile(r".+\.(wav|mp3|ogg|m4a)$")
    pattern_automatic_speaker = re.compile(r"AUTOMATIC_SPEAKER_\d+\.wav$")

    # List only files in the directory matching the pattern but not matching
    # AUTOMATIC_SPEAKER_00.wav, AUTOMATIC_SPEAKER_01.wav, etc.
    wav_voices = [
        "_XTTS_/" + f
        for f in os.listdir(main_folder)
        if os.path.isfile(os.path.join(main_folder, f))
        and pattern_coqui.match(f)
        and not pattern_automatic_speaker.match(f)
    ]

    return ["_XTTS_/AUTOMATIC.wav"] + wav_voices


def seconds_to_hhmmss_ms(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return "%02d:%02d:%02d.%03d" % (hours, minutes, int(seconds), milliseconds)


def audio_trimming(audio_path, destination, start, end):
    if isinstance(start, (int, float)):
        start = seconds_to_hhmmss_ms(start)
    if isinstance(end, (int, float)):
        end = seconds_to_hhmmss_ms(end)

    if destination:
        file_directory = destination
    else:
        file_directory = os.path.dirname(audio_path)

    file_name = os.path.splitext(os.path.basename(audio_path))[0]
    file_ = f"{file_name}_trim.wav"
    # file_ = f'{os.path.splitext(audio_path)[0]}_trim.wav'
    output_path = os.path.join(file_directory, file_)

    # -t (duration from -ss) | -to (time stop) | -af silenceremove=1:0:-50dB (remove silence)
    command = f'ffmpeg -y -loglevel error -i "{audio_path}" -ss {start} -to {end} -acodec pcm_s16le -f wav "{output_path}"'
    run_command(command)

    return output_path


def convert_to_xtts_good_sample(audio_path: str = "", destination: str = ""):
    if destination:
        file_directory = destination
    else:
        file_directory = os.path.dirname(audio_path)

    file_name = os.path.splitext(os.path.basename(audio_path))[0]
    file_ = f"{file_name}_good_sample.wav"
    # file_ = f'{os.path.splitext(audio_path)[0]}_good_sample.wav'
    mono_path = os.path.join(file_directory, file_)  # get root

    command = f'ffmpeg -y -loglevel error -i "{audio_path}" -ac 1 -ar 22050 -sample_fmt s16 -f wav "{mono_path}"'
    run_command(command)

    return mono_path


def sanitize_file_name(file_name):
    import unicodedata

    # Normalize the string to NFKD form to separate combined characters into
    # base characters and diacritics
    normalized_name = unicodedata.normalize("NFKD", file_name)
    # Replace any non-ASCII characters or special symbols with an underscore
    sanitized_name = re.sub(r"[^\w\s.-]", "_", normalized_name)
    return sanitized_name


def create_wav_file_vc(
    sample_name="",  # name final file
    audio_wav="",  # path
    start=None,  # trim start
    end=None,  # trim end
    output_final_path="_XTTS_",
    get_vocals_dereverb=True,
):
    sample_name = sample_name if sample_name else "default_name"
    sample_name = sanitize_file_name(sample_name)
    audio_wav = audio_wav if isinstance(audio_wav, str) else audio_wav.name

    BASE_DIR = (
        "."  # os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    output_dir = os.path.join(BASE_DIR, "clean_song_output")  # remove content
    # remove_directory_contents(output_dir)

    if start or end:
        # Cut file
        audio_segment = audio_trimming(audio_wav, output_dir, start, end)
    else:
        # Complete file
        audio_segment = audio_wav

    from .mdx_net import process_uvr_task

    try:
        _, _, _, _, audio_segment = process_uvr_task(
            orig_song_path=audio_segment,
            main_vocals=True,
            dereverb=get_vocals_dereverb,
        )
    except Exception as error:
        logger.error(str(error))

    sample = convert_to_xtts_good_sample(audio_segment)

    sample_name = f"{sample_name}.wav"
    sample_rename = rename_file(sample, sample_name)

    copy_files(sample_rename, output_final_path)

    final_sample = os.path.join(output_final_path, sample_name)
    if os.path.exists(final_sample):
        logger.info(final_sample)
        return final_sample
    else:
        raise Exception(f"Error wav: {final_sample}")


def create_new_files_for_vc(
    speakers_coqui,
    segments_base,
    dereverb_automatic=True
):
    # before function delete automatic delete_previous_automatic
    output_dir = os.path.join(".", "clean_song_output")  # remove content
    remove_directory_contents(output_dir)

    for speaker in speakers_coqui:
        filtered_speaker = [
            segment
            for segment in segments_base
            if segment["speaker"] == speaker
        ]
        if len(filtered_speaker) > 4:
            filtered_speaker = filtered_speaker[1:]
        if filtered_speaker[0]["tts_name"] == "_XTTS_/AUTOMATIC.wav":
            name_automatic_wav = f"AUTOMATIC_{speaker}"
            if os.path.exists(f"_XTTS_/{name_automatic_wav}.wav"):
                logger.info(f"WAV automatic {speaker} exists")
                # path_wav = path_automatic_wav
                pass
            else:
                # create wav
                wav_ok = False
                for seg in filtered_speaker:
                    duration = float(seg["end"]) - float(seg["start"])
                    if duration > 7.0 and duration < 12.0:
                        logger.info(
                            f'Processing segment: {seg["start"]}, {seg["end"]}, {seg["speaker"]}, {duration}, {seg["text"]}'
                        )
                        create_wav_file_vc(
                            sample_name=name_automatic_wav,
                            audio_wav="audio.wav",
                            start=(float(seg["start"]) + 1.0),
                            end=(float(seg["end"]) - 1.0),
                            get_vocals_dereverb=dereverb_automatic,
                        )
                        wav_ok = True
                        break

                if not wav_ok:
                    logger.info("Taking the first segment")
                    seg = filtered_speaker[0]
                    logger.info(
                        f'Processing segment: {seg["start"]}, {seg["end"]}, {seg["speaker"]}, {seg["text"]}'
                    )
                    max_duration = float(seg["end"]) - float(seg["start"])
                    max_duration = max(2.0, min(max_duration, 9.0))

                    create_wav_file_vc(
                        sample_name=name_automatic_wav,
                        audio_wav="audio.wav",
                        start=(float(seg["start"])),
                        end=(float(seg["start"]) + max_duration),
                        get_vocals_dereverb=dereverb_automatic,
                    )


def segments_coqui_tts(
    filtered_coqui_segments,
    TRANSLATE_AUDIO_TO,
    model_id_coqui="tts_models/multilingual/multi-dataset/xtts_v2",
    speakers_coqui=None,
    delete_previous_automatic=True,
    dereverb_automatic=True,
    emotion=None,
):
    """XTTS
    Install:
    pip install -q TTS==0.21.1
    pip install -q numpy==1.23.5

    Notes:
    - tts_name is the wav|mp3|ogg|m4a file for VC
    """
    from TTS.api import TTS

    TRANSLATE_AUDIO_TO = fix_code_language(TRANSLATE_AUDIO_TO, syntax="coqui")
    supported_lang_coqui = [
        "zh-cn",
        "en",
        "fr",
        "de",
        "it",
        "pt",
        "pl",
        "tr",
        "ru",
        "nl",
        "cs",
        "ar",
        "es",
        "hu",
        "ko",
        "ja",
    ]
    if TRANSLATE_AUDIO_TO not in supported_lang_coqui:
        raise TTS_OperationError(
            f"'{TRANSLATE_AUDIO_TO}' is not a supported language for Coqui XTTS"
        )
    # Emotion and speed can only be used with Coqui Studio models. discontinued
    # emotions = ["Neutral", "Happy", "Sad", "Angry", "Dull"]

    if delete_previous_automatic:
        for spk in speakers_coqui:
            remove_files(f"_XTTS_/AUTOMATIC_{spk}.wav")

    directory_audios_vc = "_XTTS_"
    create_directories(directory_audios_vc)
    create_new_files_for_vc(
        speakers_coqui,
        filtered_coqui_segments["segments"],
        dereverb_automatic,
    )

    # Init TTS
    device = os.environ.get("SONITR_DEVICE")
    model = TTS(model_id_coqui).to(device)
    sampling_rate = 24000

    # filtered_segments = filtered_coqui_segments['segments']
    # Sorting the segments by 'tts_name'
    # sorted_segments = sorted(filtered_segments, key=lambda x: x['tts_name'])
    # logger.debug(sorted_segments)

    for segment in tqdm(filtered_coqui_segments["segments"]):
        speaker = segment["speaker"]
        text = segment["text"]
        start = segment["start"]
        tts_name = segment["tts_name"]
        if tts_name == "_XTTS_/AUTOMATIC.wav":
            tts_name = f"_XTTS_/AUTOMATIC_{speaker}.wav"

        # make the tts audio
        filename = f"audio/{start}.ogg"
        logger.info(f"{text} >> {filename}")
        try:
            # Infer
            wav = model.tts(
                text=text, speaker_wav=tts_name, language=TRANSLATE_AUDIO_TO
            )
            data_tts = pad_array(
                wav,
                sampling_rate,
            )
            # Save file
            write_chunked(
                file=filename,
                samplerate=sampling_rate,
                data=data_tts,
                format="ogg",
                subtype="vorbis",
            )
            verify_saved_file_and_size(filename)
        except Exception as error:
            error_handling_in_tts(error, segment, TRANSLATE_AUDIO_TO, filename)
        gc.collect()
        torch.cuda.empty_cache()
    try:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as error:
        logger.error(str(error))
        gc.collect()
        torch.cuda.empty_cache()


# =====================================
# PIPER TTS
# =====================================


def piper_tts_voices_list():
    file_path = download_manager(
        url="https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json",
        path="./PIPER_MODELS",
    )

    with open(file_path, "r", encoding="utf8") as file:
        data = json.load(file)
    piper_id_models = [key + " VITS-onnx" for key in data.keys()]

    return piper_id_models


def replace_text_in_json(file_path, key_to_replace, new_text, condition=None):
    # Read the JSON file
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Modify the specified key's value with the new text
    if key_to_replace in data:
        if condition:
            value_condition = condition
        else:
            value_condition = data[key_to_replace]

        if data[key_to_replace] == value_condition:
            data[key_to_replace] = new_text

    # Write the modified content back to the JSON file
    with open(file_path, "w") as file:
        json.dump(
            data, file, indent=2
        )  # Write the modified data back to the file with indentation for readability


def load_piper_model(
    model: str,
    data_dir: list,
    download_dir: str = "",
    update_voices: bool = False,
):
    from piper import PiperVoice
    from piper.download import ensure_voice_exists, find_voice, get_voices

    try:
        import onnxruntime as rt

        if rt.get_device() == "GPU" and os.environ.get("SONITR_DEVICE") == "cuda":
            logger.debug("onnxruntime device > GPU")
            cuda = True
        else:
            logger.info(
                "onnxruntime device > CPU"
            )  # try pip install onnxruntime-gpu
            cuda = False
    except Exception as error:
        raise TTS_OperationError(f"onnxruntime error: {str(error)}")

    # Disable CUDA in Windows
    if platform.system() == "Windows":
        logger.info("Employing CPU exclusivity with Piper TTS")
        cuda = False

    if not download_dir:
        # Download to first data directory by default
        download_dir = data_dir[0]
    else:
        data_dir = [os.path.join(data_dir[0], download_dir)]

    # Download voice if file doesn't exist
    model_path = Path(model)
    if not model_path.exists():
        # Load voice info
        voices_info = get_voices(download_dir, update_voices=update_voices)

        # Resolve aliases for backwards compatibility with old voice names
        aliases_info: Dict[str, Any] = {}
        for voice_info in voices_info.values():
            for voice_alias in voice_info.get("aliases", []):
                aliases_info[voice_alias] = {"_is_alias": True, **voice_info}

        voices_info.update(aliases_info)
        ensure_voice_exists(model, data_dir, download_dir, voices_info)
        model, config = find_voice(model, data_dir)

        replace_text_in_json(
            config, "phoneme_type", "espeak", "PhonemeType.ESPEAK"
        )

    # Load voice
    voice = PiperVoice.load(model, config_path=config, use_cuda=cuda)

    return voice


def synthesize_text_to_audio_np_array(voice, text, synthesize_args):
    audio_stream = voice.synthesize_stream_raw(text, **synthesize_args)

    # Collect the audio bytes into a single NumPy array
    audio_data = b""
    for audio_bytes in audio_stream:
        audio_data += audio_bytes

    # Ensure correct data type and convert audio bytes to NumPy array
    audio_np = np.frombuffer(audio_data, dtype=np.int16)
    return audio_np


def segments_vits_onnx_tts(filtered_onnx_vits_segments, TRANSLATE_AUDIO_TO):
    """
    Install:
    pip install -q piper-tts==1.2.0 onnxruntime-gpu # for cuda118
    """

    data_dir = [
        str(Path.cwd())
    ]  # "Data directory to check for downloaded models (default: current directory)"
    download_dir = "PIPER_MODELS"
    # model_name = "en_US-lessac-medium" tts_name in a dict like VITS
    update_voices = True  # "Download latest voices.json during startup",

    synthesize_args = {
        "speaker_id": None,
        "length_scale": 1.0,
        "noise_scale": 0.667,
        "noise_w": 0.8,
        "sentence_silence": 0.0,
    }

    filtered_segments = filtered_onnx_vits_segments["segments"]
    # Sorting the segments by 'tts_name'
    sorted_segments = sorted(filtered_segments, key=lambda x: x["tts_name"])
    logger.debug(sorted_segments)

    model_name_key = None
    for segment in tqdm(sorted_segments):
        speaker = segment["speaker"] # noqa
        text = segment["text"]
        start = segment["start"]
        tts_name = segment["tts_name"].replace(" VITS-onnx", "")

        if tts_name != model_name_key:
            model_name_key = tts_name
            model = load_piper_model(
                tts_name, data_dir, download_dir, update_voices
            )
            sampling_rate = model.config.sample_rate

        # make the tts audio
        filename = f"audio/{start}.ogg"
        logger.info(f"{text} >> {filename}")
        try:
            # Infer
            speech_output = synthesize_text_to_audio_np_array(
                model, text, synthesize_args
            )
            data_tts = pad_array(
                speech_output,  # .cpu().numpy().squeeze().astype(np.float32),
                sampling_rate,
            )
            # Save file
            write_chunked(
                file=filename,
                samplerate=sampling_rate,
                data=data_tts,
                format="ogg",
                subtype="vorbis",
            )
            verify_saved_file_and_size(filename)
        except Exception as error:
            error_handling_in_tts(error, segment, TRANSLATE_AUDIO_TO, filename)
        gc.collect()
        torch.cuda.empty_cache()
    try:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as error:
        logger.error(str(error))
        gc.collect()
        torch.cuda.empty_cache()


# =====================================
# CLOSEAI TTS
# =====================================


def segments_openai_tts(
    filtered_openai_tts_segments, TRANSLATE_AUDIO_TO
):
    from openai import OpenAI

    client = OpenAI()
    sampling_rate = 24000

    # filtered_segments = filtered_openai_tts_segments['segments']
    # Sorting the segments by 'tts_name'
    # sorted_segments = sorted(filtered_segments, key=lambda x: x['tts_name'])

    for segment in tqdm(filtered_openai_tts_segments["segments"]):
        speaker = segment["speaker"] # noqa
        text = segment["text"].strip()
        start = segment["start"]
        tts_name = segment["tts_name"]

        # make the tts audio
        filename = f"audio/{start}.ogg"
        logger.info(f"{text} >> {filename}")

        try:
            # Request
            response = client.audio.speech.create(
                model="tts-1-hd" if "HD" in tts_name else "tts-1",
                voice=tts_name.split()[0][1:],
                response_format="wav",
                input=text
            )

            audio_bytes = b''
            for data in response.iter_bytes(chunk_size=4096):
                audio_bytes += data

            speech_output = np.frombuffer(audio_bytes, dtype=np.int16)

            # Save file
            data_tts = pad_array(
                speech_output[240:],
                sampling_rate,
            )

            write_chunked(
                file=filename,
                samplerate=sampling_rate,
                data=data_tts,
                format="ogg",
                subtype="vorbis",
            )
            verify_saved_file_and_size(filename)

        except Exception as error:
            error_handling_in_tts(error, segment, TRANSLATE_AUDIO_TO, filename)


# =====================================
# KOKORO TTS
# =====================================


def segments_kokoro_tts(filtered_kokoro_segments, TRANSLATE_AUDIO_TO):
    """Generate TTS using Kokoro — lightweight, high-quality, GPU-accelerated."""
    from kokoro import KPipeline
    from .language_configuration import KOKORO_VOICES_LIST
    import re as _re

    filtered_segments = filtered_kokoro_segments["segments"]
    starts_sorted = sorted(float(s["start"]) for s in filtered_segments)

    def _slot_for(start, end):
        start = float(start)
        end = float(end)
        nxt = [s for s in starts_sorted if s > start + 1e-6]
        gap = 0.06  # tiny breath only — keep SRT scene timing
        if nxt:
            return max(0.35, nxt[0] - start - gap)
        return max(0.35, end - start)

    sorted_segments = sorted(filtered_segments, key=lambda x: x["tts_name"])

    pipeline = None
    current_key = None

    for segment in tqdm(sorted_segments):
        text = segment["text"]
        start = segment["start"]
        end = segment.get("end", float(start) + 3.0)
        tts_name = segment["tts_name"]

        lang_voice = KOKORO_VOICES_LIST.get(tts_name, "a:af_heart")
        lang_code, voice = lang_voice.split(":")

        tts_key = lang_code
        if tts_key != current_key:
            current_key = tts_key
            if pipeline is not None:
                del pipeline
                gc.collect()
                torch.cuda.empty_cache()
            logger.info(f"Loading Kokoro pipeline for lang={lang_code}")
            pipeline = KPipeline(lang_code=lang_code)

        filename = f"audio/{start}.ogg"
        text = _re.sub(r"\s+", " ", text).strip()

        # Locked pace: default 1.1x only. Never auto-rush higher.
        slot = _slot_for(start, end)
        base = float(os.environ.get("SONI_VOICE_SPEED", "1.1") or "1.1")
        base = max(0.95, min(1.15, base))  # hard cap range
        speed = base
        speed = round(float(speed), 2)

        logger.info(
            f"Kokoro [{voice}] speed={speed} (locked) slot={slot:.2f}s: "
            f"{text[:55]}... → {filename}"
        )

        try:
            generator = pipeline(text, voice=voice, speed=speed)
            audio_chunks = []
            for _, _, audio in generator:
                audio_chunks.append(audio.numpy().astype(np.float32))
            if audio_chunks:
                combined = np.concatenate(audio_chunks)
                # Gentle trim on final audio only (preserves inter-sentence breaths)
                combined_trimmed, _ = librosa.effects.trim(
                    combined, top_db=25, ref=np.max
                )
                if len(combined_trimmed) > 0:
                    combined = combined_trimmed
                write_chunked(
                    file=filename,
                    samplerate=24000,
                    data=combined,
                    format="ogg",
                    subtype="vorbis",
                )
            verify_saved_file_and_size(filename)
        except Exception as error:
            logger.error(f"Kokoro failed for segment: {error}")
            error_handling_in_tts(error, segment, TRANSLATE_AUDIO_TO, filename)

    if pipeline is not None:
        del pipeline
    gc.collect()
    torch.cuda.empty_cache()


# =====================================
# POCKET TTS
# =====================================


# =====================================
# POCKET TTS
# =====================================


_pocket_voice_cache = {}


def _pocket_normalize_voice(voice):
    """If `voice` is a file path (a clone sample), normalize it to a clean
    24kHz mono PCM WAV first. Pocket reads .wav with the built-in `wave`
    module (PCM only) — Telegram voice notes are often OGG/Opus misnamed as
    .wav and would crash audio_read. Preset names pass through untouched."""
    is_path = (
        os.sep in str(voice)
        or str(voice).lower().endswith((".wav", ".mp3", ".ogg", ".m4a", ".flac"))
    )
    if not is_path or not os.path.exists(str(voice)):
        return voice
    if voice in _pocket_voice_cache:
        return _pocket_voice_cache[voice]

    norm = os.path.join("_POCKET_", "_norm_" + os.path.splitext(os.path.basename(str(voice)))[0] + ".wav")
    try:
        run_command(
            f'ffmpeg -y -loglevel error -i "{voice}" -ar 24000 -ac 1 -c:a pcm_s16le "{norm}"'
        )
        logger.info(f"Pocket TTS [normalize] {voice} -> {norm}")
    except Exception as error:
        logger.warning(f"Pocket TTS normalize failed, using original: {error}")
        return voice
    _pocket_voice_cache[voice] = norm
    return norm


def segments_pocket_tts(filtered_pocket_segments, TRANSLATE_AUDIO_TO):
    """Pocket TTS — load model ONCE via `serve`, reuse for every segment.

    Re-launching the CLI per segment causes the model to degrade and fill
    audio with garbage (rumble/mush). The `serve` server holds the model in
    memory (GPU if available) and reuses it — this matches how the Telegram
    Pocket bot runs and is far more stable.
    """
    from .language_configuration import POCKET_TTS_VOICES_LIST
    import requests as _requests
    import shutil

    pocket_bin = shutil.which("pocket-tts")
    if not pocket_bin:
        venv = os.path.join(os.path.dirname(sys.executable), "pocket-tts")
        pocket_bin = venv if os.path.exists(venv) else "pocket-tts"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logger.info(f"Pocket TTS using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Pocket TTS using CPU (no CUDA detected)")

    logger.info(f"Pocket TTS binary: {pocket_bin}")

    # Start the serve server ONCE (loads model into memory, reused per segment)
    port = 17980
    server_proc = None
    try:
        server_proc = subprocess.Popen(
            [pocket_bin, "serve", "--host", "127.0.0.1", "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        base = f"http://127.0.0.1:{port}"
        # Wait for /health to be ready (model load can take a while on first run)
        ready = False
        for _ in range(240):  # up to ~4 min
            if server_proc.poll() is not None:
                logger.error("Pocket serve exited early")
                break
            try:
                r = _requests.get(base + "/health", timeout=2)
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(1)
        if not ready:
            raise TTS_OperationError(
                "Pocket TTS server did not become ready. See _pocket_err.log."
            )
        logger.info(f"Pocket TTS server ready on {base}")
    except Exception as error:
        logger.error(f"Pocket TTS server start failed: {error}")
        if server_proc is not None:
            server_proc.kill()
        error_handling_in_tts(error, {}, TRANSLATE_AUDIO_TO, "audio/0.00.ogg")
        return

    try:
        for segment in tqdm(sorted(
            filtered_pocket_segments["segments"],
            key=lambda x: x["tts_name"]
        )):
            text = segment["text"]
            start = segment["start"]
            tts_name = segment["tts_name"]
            voice = POCKET_TTS_VOICES_LIST.get(tts_name, "alba")
            voice = _pocket_normalize_voice(voice)
            filename = f"audio/{start}.ogg"

            logger.info(f"Pocket TTS [{voice}]: {text[:60]}... → {filename}")

            try:
                max_attempts = 3
                last_err = None
                ok = False
                for attempt in range(1, max_attempts + 1):
                    try:
                        # Send text + clone sample to the running server
                        with open(voice, "rb") as vf:
                            resp = _requests.post(
                                base + "/tts",
                                data={"text": text},
                                files={"voice_wav": ("voice.wav", vf, "audio/wav")},
                                timeout=240,
                            )
                        if resp.status_code != 200:
                            last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                            if attempt < max_attempts:
                                logger.warning(
                                    f"Pocket TTS attempt {attempt} failed ({last_err}); retrying..."
                                )
                                time.sleep(3)
                                continue
                            raise TTS_OperationError(last_err)

                        # Save returned WAV bytes, then convert to ogg
                        tmp_wav = filename[:-4] + ".pkt.wav"
                        with open(tmp_wav, "wb") as f:
                            f.write(resp.content)
                        os.system(
                            f'ffmpeg -y -loglevel panic -i "{tmp_wav}" '
                            f'-c:a libvorbis -q:a 4 "{filename}"'
                        )
                        if os.path.exists(tmp_wav):
                            os.remove(tmp_wav)
                        verify_saved_file_and_size(filename)
                        ok = True
                        break
                    except Exception as error:
                        last_err = str(error)
                        if attempt < max_attempts:
                            logger.warning(
                                f"Pocket TTS attempt {attempt} failed; retrying..."
                            )
                            time.sleep(3)
                            continue
                        raise
                if not ok:
                    raise TTS_OperationError(last_err)
            except Exception as error:
                logger.error(f"Pocket TTS failed: {error}")
                error_handling_in_tts(error, segment, TRANSLATE_AUDIO_TO, filename)
    finally:
        if server_proc is not None:
            server_proc.kill()

    gc.collect()
    torch.cuda.empty_cache()



# =====================================
# Select task TTS
# =====================================


FISH_API_KEY = os.environ.get("FISH_API_KEY", "")
FISH_MODEL = os.environ.get("FISH_MODEL", "s2.1-pro-free")


def fish_audio_voices_list():
    """List Fish Audio voices (id, title).

    Sources (merged):
      1. FISH_VOICES env var  -> JSON  {"VoiceName": "voice_id", ...}
      2. GET api.fish.audio/model (clones + public) — model must be in HEADER.
    """
    out = []
    envj = os.environ.get("FISH_VOICES", "")
    if envj:
        try:
            import json as _json
            for name, vid in _json.loads(envj).items():
                if name and vid:
                    out.append((str(vid), str(name)))
        except Exception as e:
            logger.warning(f"FISH_VOICES parse error: {e}")
    if FISH_API_KEY:
        import requests as _rq
        for self_only in (True, False):
            try:
                resp = _rq.get(
                    f"https://api.fish.audio/model"
                    f"?page_size=100&page_number=1&self_only={'true' if self_only else 'false'}",
                    headers={"Authorization": f"Bearer {FISH_API_KEY}"},
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    items = data.get("items", []) if isinstance(data, dict) else []
                    for it in items:
                        vid = it.get("_id") or it.get("id")
                        title = (it.get("title") or it.get("name") or "").strip()
                        if not title:
                            title = vid or "Unknown"
                        if self_only:
                            title = f"{title} (Clone)"
                        if vid:
                            out.append((str(vid), title))
                else:
                    logger.warning(
                        f"Fish voices list skipped (HTTP {resp.status_code})"
                    )
            except Exception as e:
                logger.warning(f"Fish voices list error: {e}")
    # dedupe by id
    seen, res = set(), []
    for vid, title in out:
        if vid not in seen:
            seen.add(vid)
            res.append((vid, title))
    return res


def _fish_title_to_id():
    return {title: vid for vid, title in fish_audio_voices_list()}


def segments_fish_tts(filtered_fish_segments, TRANSLATE_AUDIO_TO):
    """Fish Audio — cloud TTS via api.fish.audio (voice cloning via reference_id)."""
    import requests as _rq
    if not FISH_API_KEY:
        logger.error("Fish Audio: no FISH_API_KEY set.")
        return
    title_to_id = _fish_title_to_id()
    for segment in tqdm(sorted(
        filtered_fish_segments["segments"], key=lambda x: x["tts_name"]
    )):
        text = segment["text"]
        start = segment["start"]
        tts_name = segment["tts_name"]
        title = tts_name.replace(" FishAudio", "").strip()
        voice_id = title_to_id.get(title, "") or tts_name
        filename = f"audio/{start}.ogg"
        logger.info(f"Fish Audio [{title}]: {text[:60]}... → {filename}")
        try:
            payload = {"text": text, "format": "mp3", "normalize": True}
            if voice_id and len(voice_id) > 12:
                payload["reference_id"] = voice_id
            resp = _rq.post(
                "https://api.fish.audio/v1/tts",
                headers={
                    "Authorization": f"Bearer {FISH_API_KEY}",
                    "Content-Type": "application/json",
                    "model": FISH_MODEL,  # Fish Audio reads model from the HEADER
                },
                json=payload,
                timeout=180,
            )
            if resp.status_code != 200:
                raise TTS_OperationError(
                    f"Fish HTTP {resp.status_code}: {resp.text[:300]}"
                )
            tmp = filename[:-4] + ".fish.mp3"
            with open(tmp, "wb") as f:
                f.write(resp.content)
            os.system(
                f'ffmpeg -y -loglevel panic -i "{tmp}" '
                f'-c:a libvorbis -q:a 4 "{filename}"'
            )
            if os.path.exists(tmp):
                os.remove(tmp)
            verify_saved_file_and_size(filename)
        except Exception as error:
            logger.error(f"Fish TTS failed: {error}")
            error_handling_in_tts(error, segment, TRANSLATE_AUDIO_TO, filename)


def preview_voice_audio(voice, phrase="Welcome! This is your selected voice. Let me know if you like it."):
    """Synthesize a short sample of the given voice for a preview. Returns a path or None."""
    out = "/tmp/voice_preview.mp3"
    if os.path.exists(out):
        os.remove(out)
    if not voice:
        return None
    try:
        if voice.endswith(" FishAudio"):
            import requests as _rq
            if not FISH_API_KEY:
                return None
            vid = _fish_title_to_id().get(voice.replace(" FishAudio", "").strip(), "")
            if not vid:
                return None
            resp = _rq.post(
                "https://api.fish.audio/v1/tts",
                headers={"Authorization": f"Bearer {FISH_API_KEY}",
                         "Content-Type": "application/json", "model": FISH_MODEL},
                json={"text": phrase, "reference_id": vid, "format": "mp3", "normalize": True},
                timeout=180,
            )
            if resp.status_code == 200:
                with open(out, "wb") as f:
                    f.write(resp.content)
        elif voice.endswith(" Pocket-TTS"):
            from .language_configuration import POCKET_TTS_VOICES_LIST
            import shutil as _sh
            v = POCKET_TTS_VOICES_LIST.get(voice, "alba")
            pocket_bin = _sh.which("pocket-tts") or "pocket-tts"
            subprocess.run(
                [pocket_bin, "generate", "--text", phrase, "--voice", str(v),
                 "--output-path", out, "--device", "cpu", "--quiet"],
                capture_output=True, timeout=300,
            )
        elif voice.endswith(" Kokoro"):
            from kokoro import KPipeline
            kid = voice.replace("en-Kokoro-", "").replace(" Kokoro", "").strip()
            kp = KPipeline(lang_code="a")
            for _, _, audio in kp(phrase, voice=kid, speed=1.0):
                audio.export(out, format="mp3")
                break
        else:
            import edge_tts
            import asyncio as _asyncio
            _asyncio.run(edge_tts.Communicate(phrase, voice).save(out))
        return out if os.path.exists(out) else None
    except Exception as e:
        logger.error(f"Voice preview failed ({voice}): {e}")
        return None


def find_spkr(pattern, speaker_to_voice, segments):
    return [
        speaker
        for speaker, voice in speaker_to_voice.items()
        if pattern.match(voice) and any(
            segment["speaker"] == speaker for segment in segments
        )
    ]


def filter_by_speaker(speakers, segments):
    return {
        "segments": [
            segment
            for segment in segments
            if segment["speaker"] in speakers
        ]
    }


def audio_segmentation_to_voice(
    result_diarize,
    TRANSLATE_AUDIO_TO,
    is_gui,
    tts_voice00,
    tts_voice01="",
    tts_voice02="",
    tts_voice03="",
    tts_voice04="",
    tts_voice05="",
    tts_voice06="",
    tts_voice07="",
    tts_voice08="",
    tts_voice09="",
    tts_voice10="",
    tts_voice11="",
    dereverb_automatic=True,
    model_id_bark="suno/bark-small",
    model_id_coqui="tts_models/multilingual/multi-dataset/xtts_v2",
    delete_previous_automatic=True,
):

    remove_directory_contents("audio")
    create_directories(["audio"])

    # Mapping speakers to voice variables
    speaker_to_voice = {
        "SPEAKER_00": tts_voice00,
        "SPEAKER_01": tts_voice01,
        "SPEAKER_02": tts_voice02,
        "SPEAKER_03": tts_voice03,
        "SPEAKER_04": tts_voice04,
        "SPEAKER_05": tts_voice05,
        "SPEAKER_06": tts_voice06,
        "SPEAKER_07": tts_voice07,
        "SPEAKER_08": tts_voice08,
        "SPEAKER_09": tts_voice09,
        "SPEAKER_10": tts_voice10,
        "SPEAKER_11": tts_voice11,
    }

    # Assign 'SPEAKER_00' to segments without a 'speaker' key
    for segment in result_diarize["segments"]:
        if "speaker" not in segment:
            segment["speaker"] = "SPEAKER_00"
            logger.warning(
                "NO SPEAKER DETECT IN SEGMENT: First TTS will be used in the"
                f" segment time {segment['start'], segment['text']}"
            )
        # Assign the TTS name
        segment["tts_name"] = speaker_to_voice[segment["speaker"]]

    # Merge consecutive segments that belong to the same sentence
    # for natural TTS flow (no mid-sentence break).
    # Only splits on actual sentence-ending punctuation (. ! ? …)
    # so every TTS segment is a complete thought.
    merged = []
    buf = None
    for seg in result_diarize["segments"]:
        if buf is None:
            buf = dict(seg)
            continue
        prev_text = buf["text"].strip()
        if not re.search(r'[.!?…]\s*$', prev_text):
            # Previous segment is a sentence fragment — merge
            buf["text"] = prev_text + " " + seg["text"].strip()
            buf["end"] = seg.get("end", seg["start"] + 0.5)
        else:
            merged.append(buf)
            buf = dict(seg)
    if buf is not None:
        merged.append(buf)
    n_merged = len(result_diarize["segments"]) - len(merged)
    if n_merged > 0:
        logger.info(f"Merged {n_merged} sentence fragments for natural TTS flow")
    result_diarize["segments"] = merged

    # Find TTS method
    pattern_edge = re.compile(r".*-(Male|Female)$")
    pattern_bark = re.compile(r".* BARK$")
    pattern_vits = re.compile(r".* VITS$")
    pattern_coqui = re.compile(r".+\.(wav|mp3|ogg|m4a)$")
    pattern_vits_onnx = re.compile(r".* VITS-onnx$")
    pattern_openai_tts = re.compile(r".* OpenAI-TTS$")
    pattern_kokoro = re.compile(r".* Kokoro$")
    pattern_pocket_tts = re.compile(r".* Pocket-TTS$")
    pattern_fish = re.compile(r".* FishAudio$")
    

    all_segments = result_diarize["segments"]

    speakers_edge = find_spkr(pattern_edge, speaker_to_voice, all_segments)
    speakers_bark = find_spkr(pattern_bark, speaker_to_voice, all_segments)
    speakers_vits = find_spkr(pattern_vits, speaker_to_voice, all_segments)
    speakers_coqui = find_spkr(pattern_coqui, speaker_to_voice, all_segments)
    speakers_vits_onnx = find_spkr(
        pattern_vits_onnx, speaker_to_voice, all_segments
    )
    speakers_openai_tts = find_spkr(
        pattern_openai_tts, speaker_to_voice, all_segments
    )
    speakers_kokoro = find_spkr(pattern_kokoro, speaker_to_voice, all_segments)
    speakers_pocket_tts = find_spkr(pattern_pocket_tts, speaker_to_voice, all_segments)
    speakers_fish = find_spkr(pattern_fish, speaker_to_voice, all_segments)
    

    # Filter method in segments
    filtered_edge = filter_by_speaker(speakers_edge, all_segments)
    filtered_bark = filter_by_speaker(speakers_bark, all_segments)
    filtered_vits = filter_by_speaker(speakers_vits, all_segments)
    filtered_coqui = filter_by_speaker(speakers_coqui, all_segments)
    filtered_vits_onnx = filter_by_speaker(speakers_vits_onnx, all_segments)
    filtered_openai_tts = filter_by_speaker(speakers_openai_tts, all_segments)
    filtered_kokoro = filter_by_speaker(speakers_kokoro, all_segments)
    filtered_pocket_tts = filter_by_speaker(speakers_pocket_tts, all_segments)
    filtered_fish = filter_by_speaker(speakers_fish, all_segments)
    

    # Infer
    if filtered_edge["segments"]:
        logger.info(f"EDGE TTS: {speakers_edge}")
        segments_egde_tts(filtered_edge, TRANSLATE_AUDIO_TO, is_gui)  # mp3
    if filtered_bark["segments"]:
        logger.info(f"BARK TTS: {speakers_bark}")
        segments_bark_tts(
            filtered_bark, TRANSLATE_AUDIO_TO, model_id_bark
        )  # wav
    if filtered_vits["segments"]:
        logger.info(f"VITS TTS: {speakers_vits}")
        segments_vits_tts(filtered_vits, TRANSLATE_AUDIO_TO)  # wav
    if filtered_coqui["segments"]:
        logger.info(f"Coqui TTS: {speakers_coqui}")
        segments_coqui_tts(
            filtered_coqui,
            TRANSLATE_AUDIO_TO,
            model_id_coqui,
            speakers_coqui,
            delete_previous_automatic,
            dereverb_automatic,
        )  # wav
    if filtered_vits_onnx["segments"]:
        logger.info(f"PIPER TTS: {speakers_vits_onnx}")
        segments_vits_onnx_tts(filtered_vits_onnx, TRANSLATE_AUDIO_TO)  # wav
    if filtered_openai_tts["segments"]:
        logger.info(f"OpenAI TTS: {speakers_openai_tts}")
        segments_openai_tts(filtered_openai_tts, TRANSLATE_AUDIO_TO)  # wav
    if filtered_kokoro["segments"]:
        logger.info(f"Kokoro TTS: {speakers_kokoro}")
        segments_kokoro_tts(filtered_kokoro, TRANSLATE_AUDIO_TO)  # wav
    if filtered_pocket_tts["segments"]:
        logger.info(f"Pocket TTS: {speakers_pocket_tts}")
        segments_pocket_tts(filtered_pocket_tts, TRANSLATE_AUDIO_TO)
    if filtered_fish["segments"]:
        logger.info(f"Fish Audio: {speakers_fish}")
        segments_fish_tts(filtered_fish, TRANSLATE_AUDIO_TO)


    [result.pop("tts_name", None) for result in result_diarize["segments"]]
    return [
        speakers_edge,
        speakers_bark,
        speakers_vits,
        speakers_coqui,
        speakers_vits_onnx,
        speakers_openai_tts,
    ]


def _atempo_filter_chain(rate):
    """Build ffmpeg atempo chain. Single atempo only allows 0.5–2.0."""
    rate = float(rate)
    if rate <= 0:
        rate = 1.0
    filters = []
    r = rate
    while r > 2.0 + 1e-9:
        filters.append("atempo=2.0")
        r /= 2.0
    while r < 0.5 - 1e-9:
        filters.append("atempo=0.5")
        r /= 0.5
    filters.append(f"atempo={round(r, 3)}")
    return ",".join(filters)


def _smart_sentence_gap(start, end, next_start, duration_tts=None):
    """Micro clearance only for force-fit path (not stretch/smart-pack).

    Stretch-video mode does NOT use this for dead air — packing removes gaps.
    """
    start = float(start)
    end = float(end)
    next_start = float(next_start)
    # Tiny anti-glue only; real gaps are handled by smart_pack stitch
    return 0.03


def accelerate_segments(
    result_diarize,
    max_accelerate_audio,
    valid_speakers,
    acceleration_rate_regulation=False,
    folder_output="audio2",
    stretch_video=False,
):
    if stretch_video:
        logger.info(
            "Smart-pack mode: keep voice as-is "
            "(per-scene video match handles timing)"
        )
    else:
        logger.info("Force-fit each line into its SRT slot")

    (
        speakers_edge,
        speakers_bark,
        speakers_vits,
        speakers_coqui,
        speakers_vits_onnx,
        speakers_openai_tts,
    ) = valid_speakers

    create_directories(f"{folder_output}/audio/")
    remove_directory_contents(f"{folder_output}/audio/")

    audio_files = []
    speakers_list = []

    max_count_segments_idx = len(result_diarize["segments"]) - 1

    for i, segment in tqdm(enumerate(result_diarize["segments"])):
        text = segment["text"]  # noqa
        start = segment["start"]
        end = segment["end"]
        speaker = segment["speaker"]

        filename = f"audio/{start}.ogg"
        duration_tts = librosa.get_duration(filename=filename)
        out_file = f"{folder_output}/{filename}"

        if stretch_video:
            # Smart pack + piecewise video: do not squash to SRT slots
            acc_percentage = 1.0
            duration_true = max(0.2, float(end) - float(start))
        else:
            duration_true = max(0.2, float(end) - float(start))
            sentence_gap = 0.05
            if i < max_count_segments_idx:
                next_start = float(result_diarize["segments"][i + 1]["start"])
                duration_true = max(
                    0.2, next_start - float(start) - sentence_gap
                )

            if duration_true <= 0:
                acc_percentage = 1.0
            else:
                acc_percentage = duration_tts / duration_true

            if (
                acceleration_rate_regulation
                and acc_percentage >= 1.3
                and i < max_count_segments_idx
            ):
                try:
                    next_start = float(
                        result_diarize["segments"][i + 1]["start"]
                    )
                    duration_true = max(
                        0.2, next_start - float(start) - 0.03
                    )
                    acc_percentage = duration_tts / duration_true
                except Exception as error:
                    logger.error(str(error))

            if acc_percentage <= 1.03:
                acc_percentage = 1.0
            else:
                # Cap so we never chipmunk the line
                if acc_percentage > 1.35:
                    logger.info(
                        f"Cap fit {filename}: need x{acc_percentage:.2f} "
                        f"→ max 1.35 for slot {duration_true:.2f}s"
                    )
                    acc_percentage = 1.35
                elif acc_percentage > float(max_accelerate_audio):
                    logger.info(
                        f"Heavy fit {filename}: need x{acc_percentage:.2f} "
                        f"for slot {duration_true:.2f}s"
                    )

        if speaker in speakers_edge:
            info_enc = sf.info(filename).format
        else:
            info_enc = "OGG"

        if acc_percentage == 1.0 and info_enc == "OGG":
            copy_files(filename, f"{folder_output}{os.sep}audio")
        else:
            # Speed up long audio (up to 1.35). Do NOT slow short audio here —
            # short audio is padded with silence below (keeps voice natural).
            acc_percentage = max(1.03, min(float(acc_percentage), 1.35))
            atempo = _atempo_filter_chain(acc_percentage)
            os.system(
                f"ffmpeg -y -loglevel panic -i {filename} "
                f"-filter:a {atempo} {out_file}"
            )

        if not stretch_video:
            try:
                duration_create = librosa.get_duration(filename=out_file)
                if (
                    duration_create > duration_true * 1.05
                    and duration_true > 0.2
                ):
                    force_rate = min(
                        max(duration_create / duration_true, 1.03), 1.35
                    )
                    if force_rate > 1.05:
                        tmp_force = out_file + ".fit.ogg"
                        atempo = _atempo_filter_chain(force_rate)
                        rc = os.system(
                            f"ffmpeg -y -loglevel panic -i {out_file} "
                            f"-filter:a {atempo} {tmp_force}"
                        )
                        if rc == 0 and os.path.isfile(tmp_force):
                            os.replace(tmp_force, out_file)
                            logger.info(
                                f"Force-fit {filename}: "
                                f"{duration_create:.2f}s → "
                                f"slot {duration_true:.2f}s "
                                f"(x{force_rate:.2f})"
                            )
                        elif os.path.isfile(tmp_force):
                            os.remove(tmp_force)
                # Pad pass: if audio is SHORTER than the slot, append silence to
                # fill the visual (voice stays natural speed — no slow-down).
                # Cap padding so we don't add huge dead air on a very short line.
                if duration_create < duration_true * 0.95 and duration_true > 0.2:
                    pad_sec = duration_true - duration_create
                    max_pad = 1.5  # don't add more than 1.5s of silence
                    if pad_sec > 0.15:
                        pad_sec = min(pad_sec, max_pad)
                        tmp_pad = out_file + ".pad.ogg"
                        rc = os.system(
                            f"ffmpeg -y -loglevel panic -i {out_file} "
                            f"-af apad=pad_dur={pad_sec:.3f} "
                            f"-c:a libvorbis -q:a 4 {tmp_pad}"
                        )
                        if rc == 0 and os.path.isfile(tmp_pad):
                            os.replace(tmp_pad, out_file)
                            logger.info(
                                f"Pad {filename}: "
                                f"{duration_create:.2f}s → "
                                f"slot {duration_true:.2f}s "
                                f"(+{pad_sec:.2f}s silence)"
                            )
                        elif os.path.isfile(tmp_pad):
                            os.remove(tmp_pad)
            except Exception as error:
                logger.error(f"Force-fit failed for {filename}: {error}")

        audio_files.append(out_file)
        speaker = "TTS Speaker {:02d}".format(int(speaker[-2:]) + 1)
        speakers_list.append(speaker)

    return audio_files, speakers_list


# =====================================
# Tone color converter
# =====================================


def se_process_audio_segments(
    source_seg, tone_color_converter, device, remove_previous_processed=True
):
    # list wav seg
    source_audio_segs = glob.glob(f"{source_seg}/*.wav")
    if not source_audio_segs:
        raise ValueError(
            f"No audio segments found in {str(source_audio_segs)}"
        )

    source_se_path = os.path.join(source_seg, "se.pth")

    # if exist not create wav
    if os.path.isfile(source_se_path):
        se = torch.load(source_se_path).to(device)
        logger.debug(f"Previous created {source_se_path}")
    else:
        se = tone_color_converter.extract_se(source_audio_segs, source_se_path)

    return se


def create_wav_vc(
    valid_speakers,
    segments_base,
    audio_name,
    max_segments=10,
    target_dir="processed",
    get_vocals_dereverb=False,
):
    # valid_speakers = list({item['speaker'] for item in segments_base})

    # Before function delete automatic delete_previous_automatic
    output_dir = os.path.join(".", target_dir)  # remove content
    # remove_directory_contents(output_dir)

    path_source_segments = []
    path_target_segments = []
    for speaker in valid_speakers:
        filtered_speaker = [
            segment
            for segment in segments_base
            if segment["speaker"] == speaker
        ]
        if len(filtered_speaker) > 4:
            filtered_speaker = filtered_speaker[1:]

        dir_name_speaker = speaker + audio_name
        dir_name_speaker_tts = "tts" + speaker + audio_name
        dir_path_speaker = os.path.join(output_dir, dir_name_speaker)
        dir_path_speaker_tts = os.path.join(output_dir, dir_name_speaker_tts)
        create_directories([dir_path_speaker, dir_path_speaker_tts])

        path_target_segments.append(dir_path_speaker)
        path_source_segments.append(dir_path_speaker_tts)

        # create wav
        max_segments_count = 0
        for seg in filtered_speaker:
            duration = float(seg["end"]) - float(seg["start"])
            if duration > 3.0 and duration < 18.0:
                logger.info(
                    f'Processing segment: {seg["start"]}, {seg["end"]}, {seg["speaker"]}, {duration}, {seg["text"]}'
                )
                name_new_wav = str(seg["start"])

                check_segment_audio_target_file = os.path.join(
                    dir_path_speaker, f"{name_new_wav}.wav"
                )

                if os.path.exists(check_segment_audio_target_file):
                    logger.debug(
                        "Segment vc source exists: "
                        f"{check_segment_audio_target_file}"
                    )
                    pass
                else:
                    create_wav_file_vc(
                        sample_name=name_new_wav,
                        audio_wav="audio.wav",
                        start=(float(seg["start"]) + 1.0),
                        end=(float(seg["end"]) - 1.0),
                        output_final_path=dir_path_speaker,
                        get_vocals_dereverb=get_vocals_dereverb,
                    )

                    file_name_tts = f"audio2/audio/{str(seg['start'])}.ogg"
                    # copy_files(file_name_tts, os.path.join(output_dir, dir_name_speaker_tts)
                    convert_to_xtts_good_sample(
                        file_name_tts, dir_path_speaker_tts
                    )

                max_segments_count += 1
                if max_segments_count == max_segments:
                    break

        if max_segments_count == 0:
            logger.info("Taking the first segment")
            seg = filtered_speaker[0]
            logger.info(
                f'Processing segment: {seg["start"]}, {seg["end"]}, {seg["speaker"]}, {seg["text"]}'
            )
            max_duration = float(seg["end"]) - float(seg["start"])
            max_duration = max(1.0, min(max_duration, 18.0))

            name_new_wav = str(seg["start"])
            create_wav_file_vc(
                sample_name=name_new_wav,
                audio_wav="audio.wav",
                start=(float(seg["start"])),
                end=(float(seg["start"]) + max_duration),
                output_final_path=dir_path_speaker,
                get_vocals_dereverb=get_vocals_dereverb,
            )

            file_name_tts = f"audio2/audio/{str(seg['start'])}.ogg"
            # copy_files(file_name_tts, os.path.join(output_dir, dir_name_speaker_tts)
            convert_to_xtts_good_sample(file_name_tts, dir_path_speaker_tts)

    logger.debug(f"Base: {str(path_source_segments)}")
    logger.debug(f"Target: {str(path_target_segments)}")

    return path_source_segments, path_target_segments


def toneconverter_openvoice(
    result_diarize,
    preprocessor_max_segments,
    remove_previous_process=True,
    get_vocals_dereverb=False,
    model="openvoice",
):
    audio_path = "audio.wav"
    # se_path = "se.pth"
    target_dir = "processed"
    create_directories(target_dir)

    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter

    audio_name = f"{os.path.basename(audio_path).rsplit('.', 1)[0]}_{se_extractor.hash_numpy_array(audio_path)}"
    # se_path = os.path.join(target_dir, audio_name, 'se.pth')

    # create wav seg original and target

    valid_speakers = list(
        {item["speaker"] for item in result_diarize["segments"]}
    )

    logger.info("Openvoice preprocessor...")

    if remove_previous_process:
        remove_directory_contents(target_dir)

    path_source_segments, path_target_segments = create_wav_vc(
        valid_speakers,
        result_diarize["segments"],
        audio_name,
        max_segments=preprocessor_max_segments,
        get_vocals_dereverb=get_vocals_dereverb,
    )

    logger.info("Openvoice loading model...")
    model_path_openvoice = "./OPENVOICE_MODELS"
    url_model_openvoice = "https://huggingface.co/myshell-ai/OpenVoice/resolve/main/checkpoints/converter"

    if "v2" in model:
        model_path = os.path.join(model_path_openvoice, "v2")
        url_model_openvoice = url_model_openvoice.replace(
            "OpenVoice", "OpenVoiceV2"
        ).replace("checkpoints/", "")
    else:
        model_path = os.path.join(model_path_openvoice, "v1")
    create_directories(model_path)

    config_url = f"{url_model_openvoice}/config.json"
    checkpoint_url = f"{url_model_openvoice}/checkpoint.pth"

    config_path = download_manager(url=config_url, path=model_path)
    checkpoint_path = download_manager(
        url=checkpoint_url, path=model_path
    )

    device = os.environ.get("SONITR_DEVICE")
    tone_color_converter = ToneColorConverter(config_path, device=device)
    tone_color_converter.load_ckpt(checkpoint_path)

    logger.info("Openvoice tone color converter:")
    global_progress_bar = tqdm(total=len(result_diarize["segments"]), desc="Progress")

    for source_seg, target_seg, speaker in zip(
        path_source_segments, path_target_segments, valid_speakers
    ):
        # source_se_path = os.path.join(source_seg, 'se.pth')
        source_se = se_process_audio_segments(source_seg, tone_color_converter, device)
        # target_se_path = os.path.join(target_seg, 'se.pth')
        target_se = se_process_audio_segments(target_seg, tone_color_converter, device)

        # Iterate throw segments
        encode_message = "@MyShell"
        filtered_speaker = [
            segment
            for segment in result_diarize["segments"]
            if segment["speaker"] == speaker
        ]
        for seg in filtered_speaker:
            src_path = (
                save_path
            ) = f"audio2/audio/{str(seg['start'])}.ogg"  # overwrite
            logger.debug(f"{src_path}")

            tone_color_converter.convert(
                audio_src_path=src_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=save_path,
                message=encode_message,
            )

            global_progress_bar.update(1)

    global_progress_bar.close()

    try:
        del tone_color_converter
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as error:
        logger.error(str(error))
        gc.collect()
        torch.cuda.empty_cache()


def toneconverter_freevc(
    result_diarize,
    remove_previous_process=True,
    get_vocals_dereverb=False,
):
    audio_path = "audio.wav"
    target_dir = "processed"
    create_directories(target_dir)

    from openvoice import se_extractor

    audio_name = f"{os.path.basename(audio_path).rsplit('.', 1)[0]}_{se_extractor.hash_numpy_array(audio_path)}"

    # create wav seg; original is target and dubbing is source
    valid_speakers = list(
        {item["speaker"] for item in result_diarize["segments"]}
    )

    logger.info("FreeVC preprocessor...")

    if remove_previous_process:
        remove_directory_contents(target_dir)

    path_source_segments, path_target_segments = create_wav_vc(
        valid_speakers,
        result_diarize["segments"],
        audio_name,
        max_segments=1,
        get_vocals_dereverb=get_vocals_dereverb,
    )

    logger.info("FreeVC loading model...")
    device_id = os.environ.get("SONITR_DEVICE")
    device = None if device_id == "cpu" else device_id
    try:
        from TTS.api import TTS
        tts = TTS(
            model_name="voice_conversion_models/multilingual/vctk/freevc24",
            progress_bar=False
        ).to(device)
    except Exception as error:
        logger.error(str(error))
        logger.error("Error loading the FreeVC model.")
        return

    logger.info("FreeVC process:")
    global_progress_bar = tqdm(total=len(result_diarize["segments"]), desc="Progress")

    for source_seg, target_seg, speaker in zip(
        path_source_segments, path_target_segments, valid_speakers
    ):

        filtered_speaker = [
            segment
            for segment in result_diarize["segments"]
            if segment["speaker"] == speaker
        ]

        files_and_directories = os.listdir(target_seg)
        wav_files = [file for file in files_and_directories if file.endswith(".wav")]
        original_wav_audio_segment = os.path.join(target_seg, wav_files[0])

        for seg in filtered_speaker:

            src_path = (
                  save_path
              ) = f"audio2/audio/{str(seg['start'])}.ogg"  # overwrite
            logger.debug(f"{src_path} - {original_wav_audio_segment}")

            wav = tts.voice_conversion(
                source_wav=src_path,
                target_wav=original_wav_audio_segment,
            )

            write_chunked(
                file=save_path,
                samplerate=tts.voice_converter.vc_config.audio.output_sample_rate,
                data=wav,
                format="ogg",
                subtype="vorbis",
            )

            global_progress_bar.update(1)

    global_progress_bar.close()

    try:
        del tts
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as error:
        logger.error(str(error))
        gc.collect()
        torch.cuda.empty_cache()


def toneconverter(
    result_diarize,
    preprocessor_max_segments,
    remove_previous_process=True,
    get_vocals_dereverb=False,
    method_vc="freevc"
):

    if method_vc == "freevc":
        if preprocessor_max_segments > 1:
            logger.info("FreeVC only uses one segment.")
        return toneconverter_freevc(
                    result_diarize,
                    remove_previous_process=remove_previous_process,
                    get_vocals_dereverb=get_vocals_dereverb,
                )
    elif "openvoice" in method_vc:
        return toneconverter_openvoice(
                    result_diarize,
                    preprocessor_max_segments,
                    remove_previous_process=remove_previous_process,
                    get_vocals_dereverb=get_vocals_dereverb,
                    model=method_vc,
                )


if __name__ == "__main__":
    from segments import result_diarize

    audio_segmentation_to_voice(
        result_diarize,
        TRANSLATE_AUDIO_TO="en",
        max_accelerate_audio=2.1,
        is_gui=True,
        tts_voice00="en-facebook-mms VITS",
        tts_voice01="en-CA-ClaraNeural-Female",
        tts_voice02="en-GB-ThomasNeural-Male",
        tts_voice03="en-GB-SoniaNeural-Female",
        tts_voice04="en-NZ-MitchellNeural-Male",
        tts_voice05="en-GB-MaisieNeural-Female",
    )
