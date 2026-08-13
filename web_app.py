#!/usr/bin/env python3
"""
Gudmarv Easy dubbing — custom web app (no Gradio).
Frontend = the mock design (static HTML/CSS/JS). Backend = the real
SoniTranslate engine exposed as a FastAPI service.

Run:  python web_app.py  (cwd must be the repo dir)
"""
import os, sys, json, time, threading, traceback, shutil, uuid, re, tempfile, subprocess
from pathlib import Path

# --- engine must run from the repo dir (fixed output filenames in cwd) ---
REPO = str(Path(__file__).resolve().parent)
os.chdir(REPO)
sys.path.insert(0, REPO)

OUTPUTS = Path(REPO) / "outputs"
UPLOADS = Path(REPO) / "web_uploads"
UPLOADS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

os.environ.setdefault("SAVE_DIR", str(OUTPUTS))

# Log file used for progress + live tail. Defaults to the VPS path (works there);
# Colab overrides it via SONI_LOG_FILE (the notebook redirects stdout to /content/server.log).
LOG_FILE = os.environ.get("SONI_LOG_FILE", "/home/ubuntu/sonislow/web_app.log")

# --- FastAPI + engine imports (gradio is imported by app_rvc but unused here) ---
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Gudmarv Easy dubbing API")

# Instantiate the engine ONCE (CPU mode). Heavy: loads TTS voice lists + VCI.
_engine = None
_engine_lock = threading.Lock()

def get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            from app_rvc import SoniTranslate  # imports gradio internally, but no GUI
            # cpu_mode is on by default; set SONI_CPU_MODE=0 to let GPU be used
            # (e.g. on Colab for faster Kokoro).
            _cmode = os.environ.get("SONI_CPU_MODE", "1") != "0"
            _engine = SoniTranslate(cpu_mode=_cmode)
    return _engine

# --- job store (in-memory; single serial worker) ---
JOBS = {}           # id -> {status, message, output, files:[]}
_job_queue = []
_job_worker = None
_next_job = 0

LUT_PRESETS = {
    "none": None,
    "Vintage-Faded": "Vintage-Faded.cube",
    "Cinematic-Teal": "Cinematic-Teal.cube",
    "Warm-Gold": "Warm-Gold.cube",
    "Mono": "Mono.cube",
    "Cool-Icy": "Cool-Icy.cube",
    "Vivid-Punch": "Vivid-Punch.cube",
    "Warm-Glow": "Warm-Glow.cube",
    "Bleach-Bypass": "Bleach-Bypass.cube",
    "Moody-Monochrome": "Moody-Monochrome.cube",
    "Cinematic-Teal-Orange": "Cinematic-Teal-Orange.cube",
    "Amber-Faded-Classic": "TH1_AmberFadedClassicRetro.cube",
    "Bright-Daylight": "TH1_BrightDaylightEditorial.cube",
    "Eerie-Olive-Horror": "TH1_EerieOliveHorrorNarrative.cube",
}

class _NamedPath:
    """Wrap a filesystem path so the engine's `bgm_file.name` works (it expects a gradio-style file)."""
    def __init__(self, path):
        self.name = str(path)

class _Progress:
    """Gradio-style progress callable that records percent into the job dict."""
    def __init__(self, job_id):
        self.job_id = job_id
    def __call__(self, percent, desc=None):
        try:
            JOBS[self.job_id]["progress"] = max(0.0, min(1.0, float(percent)))
            if desc:
                JOBS[self.job_id]["message"] = str(desc)[:200]
        except Exception:
            pass

def _run_job(job_id):
    st = JOBS[job_id]
    payload = st.get("_payload", {})
    try:
        st["status"] = "running"
        st["message"] = "Loading engine..."
        eng = get_engine()

        media = payload["media_path"]
        srt = payload.get("srt_path")

        st["message"] = "Preparing params..."
        # SRT mode: no transcription, no whisperx, no diarization.
        origin_lang = payload.get("origin_language") or "English (en)"
        if srt:
            # SRT mode requires an explicit origin language (not "Automatic detection")
            pass

        args = dict(
            media_file=media,
            link_media="",
            directory_input="",
            YOUR_HF_TOKEN=os.environ.get("HF_TOKEN", ""),
            preview=False,
            transcriber_model="large-v3",
            batch_size=4,
            compute_type="float32",
            origin_language=origin_lang,
            target_language=payload.get("target_language") or origin_lang,
            min_speakers=1,
            max_speakers=1,
            tts_voice00=payload.get("voice", "en-US-EmmaMultilingualNeural-Female"),
            video_output_name=payload.get("output_name", ""),
            mix_method_audio="Adjusting volumes and mixing audio",
            # Mute the original track so only the clone voice is heard; raise dub
            volume_original_audio=float(payload.get("volume_original_audio", 0.0)),
            volume_translated_audio=float(payload.get("volume_translated_audio", 1.80)),
            output_format_subtitle="srt",
            translate_process="google_translator_batch",
            subtitle_file=srt,
            output_type="video (mp4)" if payload.get("voice") else "raw media",
            soft_subtitles_to_video=False,
            # Captions are burned inside the `burn_subtitles_to_video` block;
            # enable it when captions are requested (SRT subtitle file is present).
            burn_subtitles_to_video=bool(payload.get("caption_enable", False)) and bool(srt),
            # --- captions (CapCut style) ---
            caption_enable=bool(payload.get("caption_enable", False)),
            caption_size=int(payload.get("caption_size", 24)),
            caption_color=payload.get("caption_color", "#FFFFFF"),
            caption_hl=payload.get("caption_hl", "#FFD400"),
            caption_box=bool(payload.get("caption_box", True)),
            caption_pos=payload.get("caption_pos", "lower"),
            caption_karaoke=bool(payload.get("caption_karaoke", True)),
            caption_font=payload.get("caption_font", "Arial Black"),
            caption_glow=bool(payload.get("caption_glow", False)),
            caption_glow_color=payload.get("caption_glow_color", "#FFD400"),
            caption_glow_strength=int(payload.get("caption_glow_strength", 6)),
            caption_style=payload.get("caption_style", "highlight"),
            # --- pre-dub crop / color / lut / cut-mirror ---
            edit_crop_enable=bool(payload.get("edit_crop_enable", False)),
            edit_zoom=int(payload.get("edit_zoom", 100) or 100),
            edit_crop_x=int(payload.get("edit_crop_x", 0)),
            edit_crop_y=int(payload.get("edit_crop_y", 0)),
            edit_crop_w=int(payload.get("edit_crop_w", 0)),
            edit_crop_h=int(payload.get("edit_crop_h", 0)),
            edit_bright=float(payload.get("edit_bright", 0.0)),
            edit_contrast=float(payload.get("edit_contrast", 1.0)),
            edit_sat=float(payload.get("edit_sat", 1.0)),
            edit_gamma=float(payload.get("edit_gamma", 1.0)),
            edit_hue=float(payload.get("edit_hue", 0.0)),
            edit_warmth=float(payload.get("edit_warmth", 0.0)),
            color_grade=payload.get("color_grade", "none") or "none",
            edit_lut=None,
            lut_preset=payload.get("lut_custom") or LUT_PRESETS.get(payload.get("lut", "none")),
            output_speed=float(payload.get("output_speed", 1.0) or 1.0),
            cut_mirror_enable=bool(payload.get("cut_mirror_enable", False)),
            cut_mirror_sec=int(payload.get("cut_mirror_sec", 5)),
            # --- bgm ---
            bgm_file=_NamedPath(payload["bgm_path"]) if payload.get("bgm_path") else None,
            bgm_volume=int(payload.get("bgm_volume", 15)),
            bgm_preset=None,
            enable_cache=False,
            is_gui=True,
            progress=_Progress(job_id),
        )

        # Per-engine narration speed sliders (0.5-1.5) -> set each engine's env var
        # before the run so _engine_speed() in text_to_speech.py picks up the UI value.
        for _env, _key in (("SONI_EDGE_SPEED", "edge_speed"),
                           ("SONI_KOKORO_SPEED", "kokoro_speed"),
                           ("SONI_FISH_SPEED", "fish_speed"),
                           ("SONI_POCKET_SPEED", "pocket_speed")):
            if _key in payload and payload.get(_key) not in (None, ""):
                try:
                    _v = max(0.5, min(1.5, float(payload.get(_key))))
                    os.environ[_env] = str(_v)
                    print(f"[SPEED] {_key}={_v} -> {_env}")
                except Exception:
                    pass

        st["message"] = "Engine running..."
        result = eng.multilingual_media_conversion(**args)
        if isinstance(result, str):
            result = [result]
        files = []
        for r in result or []:
            p = Path(r)
            if p.exists():
                files.append(p.name)
        st["status"] = "done"
        st["message"] = "Done"
        st["output"] = files
        st["files"] = files
        _copy_to_drive(files)
    except Exception as e:
        st["status"] = "error"
        st["message"] = str(e)[:500]
        st["traceback"] = traceback.format_exc()[-2000:]


def _copy_to_drive(files):
    """Copy finished outputs to Google Drive (if mounted) so the user can
    download them fast instead of through the slow Cloudflare tunnel.
    Drive mount (Colab) lives at /content/drive; on the VPS this path won't
    exist, so it silently no-ops."""
    try:
        drive_dir = Path("/content/drive/MyDrive/SoniSlowVideo_outputs")
        if not drive_dir.exists():
            return
        drive_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            src = OUTPUTS / f
            if not src.is_file():
                continue
            dst = drive_dir / f
            try:
                shutil.copy2(src, dst)
                print(f"[DRIVE] copied {f} -> {dst}")
            except Exception as e:
                print(f"[DRIVE] copy failed {f}: {e}")
    except Exception:
        pass

def _worker_loop():
    global _job_worker
    while True:
        if not _job_queue:
            return
        job_id = _job_queue.pop(0)
        _run_job(job_id)

def _kick_worker():
    global _job_worker
    if _job_worker is None or not _job_worker.is_alive():
        _job_worker = threading.Thread(target=_worker_loop, daemon=True)
        _job_worker.start()

@app.post("/api/upload")
async def upload_file(
    kind: str = Form(...),
    file: UploadFile = File(...),
):
    """Stage a single uploaded file immediately (per selection). kind in {video,srt,bgm,lut}.
    Returns an id the frontend passes to /api/render later. LUTs go into assets/lut/."""
    import uuid as _uuid
    uid = _uuid.uuid4().hex[:8]
    safe = os.path.basename(file.filename or "file")
    if kind == "lut":
        lut_dir = Path(REPO) / "assets" / "lut"
        lut_dir.mkdir(parents=True, exist_ok=True)
        # keep original .cube filename so the engine's lut_preset path resolves
        dest = lut_dir / safe
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        print(f"[UPLOAD] lut -> {dest.name} ({dest.stat().st_size} bytes)")
        return {"id": str(uid), "kind": "lut", "path": str(dest), "size": dest.stat().st_size, "name": safe}
    dest = UPLOADS / f"{uid}_{kind}_{safe}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    size = dest.stat().st_size
    print(f"[UPLOAD] {kind} -> {dest.name} ({size} bytes)")
    return {"id": str(uid), "kind": kind, "path": str(dest), "size": size, "name": safe}

@app.get("/api/speeds")
def get_speeds():
    """Current + default per-engine narration speeds so the UI sliders init correctly."""
    specs = [("SONI_EDGE_SPEED", "edge_speed", 1.0, "Edge-TTS"),
             ("SONI_KOKORO_SPEED", "kokoro_speed", 1.1, "Kokoro"),
             ("SONI_FISH_SPEED", "fish_speed", 0.8, "Fish Audio"),
             ("SONI_POCKET_SPEED", "pocket_speed", 1.0, "Pocket-TTS")]
    out = {}
    for env, key, dflt, label in specs:
        try:
            cur = max(0.5, min(1.5, float(os.environ.get(env, dflt))))
        except Exception:
            cur = dflt
        out[key] = {"current": round(cur, 2), "default": dflt, "label": label, "min": 0.5, "max": 1.5}
    return out

@app.post("/api/render")
async def render(
    video: UploadFile = File(None),
    srt: UploadFile = File(None),
    bgm: UploadFile = File(None),
    json_data: str = Form("{}"),
):
    try:
        payload = json.loads(json_data)
    except Exception:
        payload = {}
    # Log what the browser actually sent (for debugging cut/mirror/crop)
    print(f"[REQ] payload keys: {sorted(payload.keys())}")
    print(f"[REQ] cut_mirror_enable={payload.get('cut_mirror_enable')} "
          f"cut_mirror_sec={payload.get('cut_mirror_sec')} "
          f"edit_crop_enable={payload.get('edit_crop_enable')} "
          f"crop=({payload.get('edit_crop_x')},{payload.get('edit_crop_y')},"
          f"{payload.get('edit_crop_w')},{payload.get('edit_crop_h')}) "
          f"voice={payload.get('voice','')[:30]!r} output_name={payload.get('output_name')!r}")
    # Files may arrive either staged (payload has media_path/srt_path/bgm_path from
    # /api/upload) OR as direct multipart uploads in this same request.
    if payload.get("media_path"):
        vpath = payload["media_path"]
    elif video is not None and video.filename:
        uid = uuid.uuid4().hex[:8]
        vpath = UPLOADS / f"{uid}_{os.path.basename(video.filename)}"
        with open(vpath, "wb") as f:
            shutil.copyfileobj(video.file, f)
    else:
        raise HTTPException(400, "no video provided")

    srt_path = payload.get("srt_path")
    if not srt_path and srt is not None and srt.filename:
        uid = uuid.uuid4().hex[:8]
        srt_path = UPLOADS / f"{uid}_{os.path.basename(srt.filename)}"
        with open(srt_path, "wb") as f:
            shutil.copyfileobj(srt.file, f)

    bgm_path = payload.get("bgm_path")
    if not bgm_path and bgm is not None and bgm.filename:
        uid = uuid.uuid4().hex[:8]
        bgm_path = UPLOADS / f"{uid}_{os.path.basename(bgm.filename)}"
        with open(bgm_path, "wb") as f:
            shutil.copyfileobj(bgm.file, f)

    payload["media_path"] = str(vpath)
    payload["srt_path"] = str(srt_path) if srt_path else None
    payload["bgm_path"] = str(bgm_path) if bgm_path else None

    global _next_job
    job_id = f"job_{int(time.time())}_{_next_job}"
    _next_job += 1
    # record the log byte offset at submission so progress parsing only reads THIS job's lines
    try:
        with open(LOG_FILE, "rb") as f:
            _log_start = f.seek(0, os.SEEK_END)
    except Exception:
        _log_start = 0
    JOBS[job_id] = {"status": "queued", "message": "Queued", "output": [], "files": [], "_payload": payload, "_created": time.time(), "progress": 0.0, "_log_start": _log_start}
    _job_queue.append(job_id)
    _kick_worker()
    return {"job_id": job_id}

def _estimate_progress(job_id):
    """Blend the engine's coarse progress with per-line TTS progress from the log
    so the bar moves smoothly during the (long) dubbing stage."""
    global _tts_total
    j = JOBS.get(job_id) or {}
    base = float(j.get("progress", 0.0) or 0.0)
    if j.get("status") == "done":
        return 1.0
    if j.get("status") == "error":
        return base
    # running / queued
    try:
        with open(LOG_FILE, "rb") as f:
            f.seek(int(j.get("_log_start", 0)))
            lines = f.read().decode("utf-8", "replace").splitlines()
    except Exception:
        lines = []
    # TTS progress: lines like " 45%|... 13/28 [00:xx<..."  -> fraction of the dub
    tts_frac = None
    total = None
    done = 0
    for ln in lines:
        m = re.search(r"(\d+)/(\d+)\s+\[[0-9:]+<", ln)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
    if total:
        tts_frac = done / total
    # Engine reports 0.0 until "Text to speech" (0.80). Map:
    #   before TTS  -> engine progress as-is
    #   during TTS  -> 0.15..0.85 blended by tts_frac
    #   after TTS   -> engine progress (0.80+) as-is
    if base >= 0.80 and tts_frac is not None:
        # TTS stage: interpolate 0.15 -> 0.85 by tts_frac
        return 0.15 + 0.70 * tts_frac
    if base >= 0.15 and base < 0.80:
        # between "Processing video" and TTS end, nudge by tts_frac too
        return max(base, 0.15 + 0.70 * tts_frac if tts_frac is not None else base)
    return base

@app.get("/api/status/{job_id}")
def status(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "unknown job")
    return {
        "status": j["status"],
        "message": j["message"],
        "files": j.get("files", []),
        "output": j.get("output", []),
        "progress": round(_estimate_progress(job_id), 3),
        "traceback": j.get("traceback", ""),
    }

@app.get("/api/jobs")
def jobs(limit: int = 10):
    """Recent jobs (newest first) + live tail of the server log."""
    items = []
    for jid in list(JOBS.keys())[-limit:]:
        j = JOBS[jid]
        items.append({
            "job_id": jid,
            "status": j["status"],
            "message": j["message"],
            "files": j.get("files", []),
            "created": j.get("_created", 0),
        })
    items.reverse()
    return {"jobs": items}

_LOG_TAIL = []

def _tail_log(lines=120):
    """Return last `lines` lines of the app log file, cleaned of progress spinners."""
    try:
        with open(LOG_FILE, "r", errors="replace") as f:
            raw = f.readlines()
    except Exception:
        return []
    out = []
    for ln in raw[-lines:]:
        # collapse \r progress bars, keep clean lines
        if "%|" in ln or "MB/s" in ln or "it/s]" in ln:
            continue
        # drop uvicorn HTTP access-log noise ("GET /api/status ... HTTP/1.1" 200 OK)
        # so the log only shows real render/engine output the user can understand.
        if ' HTTP/1.1"' in ln or ('INFO: ' in ln and '- "' in ln and '" 200' in ln):
            continue
        out.append(ln.rstrip())
    return out[-lines:]

@app.get("/api/logs")
def logs(limit: int = 120):
    """Live tail of the render log for monitoring."""
    return {"log": _tail_log(limit)}

@app.get("/api/download/{filename}")
def download(filename: str):
    # serve from outputs dir (safe: only allow files under OUTPUTS)
    p = OUTPUTS / filename
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(str(p), filename=filename)


@app.get("/api/luts")
def list_luts():
    """List every saved .cube LUT in assets/lut/ (the persistent library).
    Anything uploaded via /api/upload kind=lut lands here and is reusable."""
    lut_dir = Path(REPO) / "assets" / "lut"
    if not lut_dir.is_dir():
        return {"luts": []}
    return {"luts": sorted(f.name for f in lut_dir.glob("*.cube"))}


@app.post("/api/lut_preview")
def lut_preview(data: dict):
    """Render a live preview frame with the chosen LUT + color adjustments.

    Body: {"video": "<path>", "lut": "<preset-name|'custom'>",
           "lut_custom": "<uploaded .cube filename>", "bright": f,
           "contrast": f, "sat": f, "hue": f}
    Returns a PNG of one video frame graded with the same chain the render uses,
    so the user can SEE what a (custom) LUT does before committing a render.
    """
    import io
    from fastapi.responses import Response as _Resp
    vpath = data.get("video")
    if not vpath or not os.path.isfile(vpath):
        raise HTTPException(400, "no video")
    # resolve the .cube file
    lut_file = data.get("lut_custom") or LUT_PRESETS.get(data.get("lut"))
    lut_path = None
    if lut_file:
        cand = Path(REPO) / "assets" / "lut" / lut_file
        if cand.is_file():
            lut_path = cand
    # pick a representative frame (~15% in)
    try:
        dur = float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", vpath],
            text=True).strip())
    except Exception:
        dur = 3.0
    ts = max(0.1, dur * 0.15)
    tmp = Path(tempfile.gettempdir()) / f"lutpv_{uuid.uuid4().hex[:8]}.png"
    vf = []
    m = 1.0 + float(data.get("bright", 0.0) or 0.0)
    if abs(m - 1.0) > 1e-3:
        vf.append(f"lutyuv=y='clip(val*{m:.4f},0,255)'")
    c = float(data.get("contrast", 1.0) or 1.0)
    s = float(data.get("sat", 1.0) or 1.0)
    if abs(c - 1.0) > 1e-3 or abs(s - 1.0) > 1e-3:
        vf.append(f"eq=contrast={c:.3f}:saturation={s:.3f}")
    h = float(data.get("hue", 0.0) or 0.0)
    if h:
        vf.append(f"hue=h={h:.1f}")
    w = float(data.get("warmth", 0.0) or 0.0)
    if w:
        vf.append(f"colortemperature=temperature={int(6500 - w * 2000)}")
    if lut_path:
        vf.append(f"lut3d=file='{lut_path}'")
    if not vf:
        vf = ["null"]
    cmd = ["ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", vpath,
           "-vf", ",".join(vf), "-frames:v", "1", "-q:v", "3", str(tmp)]
    rc = subprocess.run(cmd, capture_output=True, text=True)
    if rc.returncode != 0 or not tmp.exists():
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise HTTPException(500, f"preview failed: {rc.stderr[-200:]}")
    data_bytes = tmp.read_bytes()
    tmp.unlink(missing_ok=True)
    return _Resp(content=data_bytes, media_type="image/png")


FAVORITES_FILE = Path(REPO) / "favorites.txt"

def _load_favorites():
    """Return the list of voice names the user wants to keep (exact matches).
    Empty file / missing file = show ALL voices."""
    try:
        if not FAVORITES_FILE.exists():
            return []
        names = []
        for ln in FAVORITES_FILE.read_text(encoding="utf-8").splitlines():
            s = ln.strip()
            if s and not s.startswith("#"):
                names.append(s)
        return names
    except Exception:
        return []

def _filter_voices(all_voices):
    """Keep the user's 6 Fish favorites PLUS all Edge/Kokoro/Pocket default voices.
    Everything else (other Fish clones, etc.) is hidden."""
    favs = _load_favorites()
    fav_set = set(favs)
    def is_edge_kokoro_pocket(v):
        low = v.lower()
        return ("fishaudio" not in low) and (
            "edge" in low or "kokoro" in low or "pocket" in low
            or not any(e in low for e in ["fishaudio", "(clone)"])
        )
    out = []
    seen = set()
    for v in all_voices:
        low = v.lower()
        # Always include exact/partial Fish favorites
        if any(f == v or f.lower() in low for f in favs):
            pass
        elif "fishaudio" in low:
            continue  # other fish clones hidden
        elif not is_edge_kokoro_pocket(v):
            continue
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out

_voices_cache = {"at": 0, "list": []}

def _get_voice_list_cached():
    """Cache the full voice list for 15 min (fetching all Fish pages is slow)."""
    import time as _t
    if _voices_cache["list"] and (_t.time() - _voices_cache["at"] < 900):
        return _voices_cache["list"]
    eng = get_engine()
    lst = eng.get_tts_voice_list()
    _voices_cache["at"] = _t.time()
    _voices_cache["list"] = lst
    return lst

@app.get("/api/voices")
def voices():
    try:
        return {"voices": _filter_voices(_get_voice_list_cached())}
    except Exception as e:
        return {"voices": [], "error": str(e)[:300]}

@app.post("/api/favorites")
async def set_favorites(names: dict):
    """Set the favorites list (body: {"names": [...]}). Used to hide unwanted voices."""
    names = names.get("names", [])
    with open(FAVORITES_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(names) + ("\n" if names else ""))
    return {"ok": True, "count": len(names), "file": str(FAVORITES_FILE)}

@app.get("/api/favorites")
def get_favorites():
    return {"names": _load_favorites()}

@app.get("/api/health")
def health():
    return {"ok": True}

# ---------------------------------------------------------------------------
# Pocket TTS voice cloning (saves a .safetensors clone for reuse)
# ---------------------------------------------------------------------------
POCKET_VOICES_DIR = Path(os.environ.get("POCKET_CLONE_DIR", "/home/ubuntu/pocket_tts_voices"))
POCKET_CLI = os.environ.get("POCKET_TTS_CLI", "/home/ubuntu/.local/bin/pocket-tts")

@app.get("/api/pocket/clones")
def pocket_clones():
    POCKET_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    clones = sorted(f.stem for f in POCKET_VOICES_DIR.glob("*.safetensors"))
    return {"clones": clones}

@app.post("/api/pocket/clone")
async def pocket_clone(name: str = Form(...), file: UploadFile = File(...)):
    """Clone a Pocket voice from an uploaded sample (WAV only)."""
    import uuid as _uuid, re as _re
    if not (file.filename or "").lower().endswith(".wav"):
        raise HTTPException(400, "Only .wav voice samples are supported for cloning. Convert your audio to WAV and try again.")
    voice_name = _re.sub(r"[^A-Za-z0-9 _\-]+", "", (name or "").strip()).strip() or "clone"
    POCKET_VOICES_DIR.mkdir(parents=True, exist_ok=True)

    tmp = Path(tempfile.gettempdir()) / f"clone_src_{_uuid.uuid4().hex[:8]}"
    with tmp.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    wav16k = Path(tempfile.gettempdir()) / f"clone_wav_{_uuid.uuid4().hex[:8]}.wav"
    conv = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp),
                           "-ac", "1", "-ar", "16000", str(wav16k)],
                          capture_output=True, text=True, timeout=120)
    out_path = POCKET_VOICES_DIR / f"{voice_name}.safetensors"
    exp = subprocess.run([POCKET_CLI, "export-voice", str(wav16k), str(out_path), "--quiet"],
                         capture_output=True, text=True, timeout=300)
    # Also drop a wav conditioning sample into the engine's _POCKET_/ folder so
    # this clone shows up in the Gudmarv Easy dubbing Pocket voice dropdown + is usable.
    pocket_scan = Path(REPO) / "_POCKET_"
    pocket_scan.mkdir(parents=True, exist_ok=True)
    wav_copy = pocket_scan / f"{voice_name}.wav"
    try:
        shutil.copyfile(wav16k, wav_copy)
    except Exception:
        pass
    try:
        tmp.unlink(missing_ok=True); wav16k.unlink(missing_ok=True)
    except Exception:
        pass
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise HTTPException(500, f"Clone failed: {(exp.stderr or '')[-300:]}")
    return {"name": voice_name, "filename": out_path.name,
            "clones": sorted(f.stem for f in POCKET_VOICES_DIR.glob("*.safetensors"))}

# Serve the frontend as the web root. Works both when the frontend is checked
# into the repo (Colab clone: <repo>/web_frontend) and when it lives as a
# sibling next to the repo (VPS: <repo>/../web_frontend).
_inrepo = Path(__file__).resolve().parent / "web_frontend"
_sibling = Path(__file__).resolve().parent.parent / "web_frontend"
WEB_DIR = _inrepo if (_inrepo / "index.html").exists() else _sibling
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    idx = WEB_DIR / "index.html"
    if idx.exists():
        return HTMLResponse(idx.read_text())
    return HTMLResponse("<h1>Gudmarv Easy dubbing</h1><p>frontend not found</p>")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    print(f"Gudmarv Easy dubbing web app -> http://127.0.0.1:{port}")
    # Warm the heavy engine + voice list once at boot (background) so the first
    # page load and first render don't block ~50s. The voice list is cached 15
    # min, and the engine is built once and reused, so this is a one-time cost.
    def _warm_startup():
        try:
            _get_voice_list_cached()
            print("[WARM] engine + voice list preloaded.")
        except Exception as e:
            print(f"[WARM] preload failed (will lazy-load): {str(e)[:200]}")
    threading.Thread(target=_warm_startup, daemon=True, name="voice-warmup").start()
    uvicorn.run(app, host="0.0.0.0", port=port)
