"""End-to-end inference: one class video (file or URL) -> hybrid quality assessment.

Everything derived from the media (downloaded video, WAV) lives in a
`tempfile.TemporaryDirectory(prefix="class_assess_")` that is removed in `finally:`.
Media never leaves the machine; only the de-identified transcript + numeric features
reach Gemini (when a key is configured), and only after the pre-flight privacy gate.

Stages and progress fractions:
  1 ingest + fast SHA-256 file hash           0.00 -> 0.08
  2 PyAV 16 kHz mono WAV                      0.08 -> 0.15
  3 visual compliance + audio prosody         0.15 -> 0.40
  4 faster-whisper transcription              0.40 -> 0.82
  5 de-id, roles, features, Gemini, scorer    0.82 -> 1.00
"""
from __future__ import annotations

import gc
import json
import os
import re
import tempfile
import time
import urllib.request
from pathlib import Path

from src.audio_extract import extract_wav_16k_mono
from src.audio_features import extract_audio_features
from src.deidentify import deidentify_transcript
from src.speaker_roles import attribute_speaker_roles
from src.transcribe import audio_fingerprint, release_models, transcribe_audio
from src.transcript_features import extract_transcript_features
from src.utils.cuda_setup import register_cuda_dlls
from src.visual_features import FaceDetector, aggregate_visual, default_models_dir, load_logo_templates, measure_video
from src import llm_judge as lj
from src import scorer as sc

CALIBRATION_ASSET = Path(__file__).resolve().parent / "assets" / "calibration.json"
MAX_DOWNLOAD_BYTES = 4 * 2**30
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}


# ===================================================================== ingest
def _is_url(x) -> bool:
    return isinstance(x, str) and re.match(r"(?i)^https?://", x.strip()) is not None


def _download(url: str, dest_dir: Path) -> Path:
    url = url.strip()
    if re.search(r"(?i)(drive|docs)\.google\.com", url):
        if re.search(r"/folders/|folderview|/drive/u/\d+/folders", url):
            raise ValueError("Google Drive folder links are not supported — share a link to a single video file.")
        import gdown
        out = gdown.download(url, str(dest_dir / "input_video"), quiet=True, fuzzy=True)
        p = Path(out) if out else None
        if not p or not p.is_file() or p.stat().st_size == 0:
            raise ValueError("Could not download that Google Drive file. Make sure it is a single video shared as "
                             "'Anyone with the link can view'.")
        return p
    req = urllib.request.Request(url, headers={"User-Agent": "class-assessor/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            ctype = (r.headers.get("Content-Type") or "").lower()
            if "text/html" in ctype:
                raise ValueError("That link returned a web page, not a video file. Use a direct video link "
                                 "(e.g. ending in .mp4) or a Google Drive file link.")
            dest = dest_dir / "input_video.mp4"
            total = 0
            with dest.open("wb") as f:
                while chunk := r.read(2**20):
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise ValueError("Video is larger than 4 GB.")
                    f.write(chunk)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Could not download the video link ({type(e).__name__}).") from None
    if total == 0:
        raise ValueError("The video link returned an empty file.")
    return dest


def _probe_video(path: Path) -> None:
    import av
    try:
        with av.open(str(path)) as c:
            if not c.streams.audio:
                raise ValueError("The file has no audio track.")
    except ValueError:
        raise
    except Exception:
        raise ValueError("The file could not be opened as a video.") from None


# ================================================================ helpers
def pick_device() -> tuple[str, str]:
    """('cuda', 'int8_float16') when a CUDA device is available, else ('cpu', 'int8').
    ASSESSOR_DEVICE=cpu forces the CPU path."""
    if os.environ.get("ASSESSOR_DEVICE", "").lower() == "cpu":
        return "cpu", "int8"
    register_cuda_dlls()
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "int8_float16"
    except Exception:
        pass
    return "cpu", "int8"


def load_calibration() -> dict:
    try:
        return json.loads(CALIBRATION_ASSET.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"label_source": "deterministic", "deterministic_cutoffs": list(sc.DEFAULT_CUTOFFS)}


_SNIPPET_KEYS = {"individual_question_examples", "homework_snippet"}


# ================================================================ pipeline
def assess_video_pipeline(video_path_or_url: str | Path, language: str = "en", progress_cb=None, *,
                          use_llm: bool = True, whisper_model: str | None = None, models_dir: str | Path | None = None,
                          llm_cache_dir: str | Path | None = None, sample_every_sec: float = 10.0,
                          transcribe_fn=None, judge_fn=None, detector=None) -> dict:
    """Assess one class recording. Returns {"assessment", "telemetry", "llm", ...}.

    `transcribe_fn`, `judge_fn` and `detector` are injection points for tests.
    Raises ValueError with a user-facing message for bad inputs.
    """
    def report(frac: float, desc: str) -> None:
        if progress_cb:
            progress_cb(round(min(1.0, max(0.0, frac)), 3), desc)

    # WHISPER_MODEL (e.g. "base" in Streamlit secrets / env) trades accuracy for memory.
    whisper_model = whisper_model or os.environ.get("WHISPER_MODEL", "small")
    timings: dict[str, float] = {}
    tmp = tempfile.TemporaryDirectory(prefix="class_assess_", ignore_cleanup_errors=True)
    tmp_path = Path(tmp.name)
    try:
        # 1 — ingest
        t0 = time.perf_counter()
        report(0.0, "Ingesting video")
        if _is_url(str(video_path_or_url)):
            src = _download(str(video_path_or_url), tmp_path)
        else:
            src = Path(video_path_or_url)
            if not src.is_file():
                raise ValueError("Video file not found.")
        _probe_video(src)
        file_hash = audio_fingerprint(src)
        timings["ingest"] = time.perf_counter() - t0
        report(0.08, "Extracting audio")

        # 2 — audio
        t0 = time.perf_counter()
        wav = tmp_path / "audio.wav"
        ext = extract_wav_16k_mono(src, wav)
        timings["audio_extract"] = time.perf_counter() - t0
        gc.collect()
        report(0.15, "Checking camera, branded virtual background and audio")

        # 3 — visual + audio features
        t0 = time.perf_counter()
        mdir = Path(models_dir) if models_dir else default_models_dir()
        measured = measure_video(src, detector or FaceDetector(mdir), sample_every_sec)
        visual = aggregate_visual(measured, logo_template=load_logo_templates(mdir))
        audio = extract_audio_features(wav)
        timings["visual_audio"] = time.perf_counter() - t0
        del measured
        gc.collect()
        report(0.40, "Transcribing speech")

        # 4 — transcription
        t0 = time.perf_counter()
        device, compute = pick_device() if transcribe_fn is None else ("mock", "mock")
        tr = (transcribe_fn or transcribe_audio)(
            wav, model_name=whisper_model, language=language, device=device, compute_type=compute, beam_size=1,
            progress_callback=lambda f: report(0.40 + 0.42 * f, "Transcribing speech"))
        timings["transcribe"] = time.perf_counter() - t0
        release_models()  # drop the Whisper model before de-id / roles / scoring (small containers)
        gc.collect()
        report(0.82, "De-identifying and scoring")

        # 5 — de-id, roles, features, judge, scorer
        t0 = time.perf_counter()
        deid = deidentify_transcript(tr["segments"])
        roles = attribute_speaker_roles(deid["masked_segments"], wav)
        segs = roles.pop("segments")
        duration = float(tr.get("duration_sec") or ext["duration_sec"])
        feats = extract_transcript_features(segs, deid["stats"], roles, duration)
        logprobs = [s["avg_logprob"] for s in segs if s.get("avg_logprob") is not None]
        avg_lp = round(sum(logprobs) / len(logprobs), 3) if logprobs else None
        inp = {"duration_sec": duration, "segments": segs, "visual": visual, "audio": audio,
               "transcript": feats, "asr": {"avg_logprob_mean": avg_lp}}
        cam = sc.refine_camera_off(visual.get("camera_off_times_sec", []), visual.get("sample_every_sec", sample_every_sec),
                                   segs, visual.get("sampled_frames", 0))
        inp["camera_refined"] = cam

        llm_res, llm_note = None, None
        if use_llm:
            key, status = lj.api_key_status() if judge_fn is None else ("injected", "ok")
            if key is None:
                llm_note = status
            else:
                sid = f"upload_{file_hash[:16]}"
                prompt, block, _info = lj.build_assessment_prompt(
                    sid, segs, duration, lj.build_prompt_features(visual, audio, feats, cam, avg_lp,
                                                                  sc.G["code_mix_logprob"]))
                try:
                    llm_res = (judge_fn or lj.call_gemini_judge)(sid, prompt, block, set(), cache_dir=llm_cache_dir)
                except lj.PrivacyViolation:
                    llm_res, llm_note = None, "Gemini skipped: the privacy pre-flight check blocked the prompt."
                if llm_res is None and llm_note is None:
                    llm_note = "Gemini unavailable (quota or model errors); deterministic assessment shown."
        else:
            llm_note = "Gemini judge disabled; deterministic assessment shown."
        calib = load_calibration()
        # The label always comes from the deterministic scorer; Gemini adds summary + evidence.
        cutoffs = tuple(calib["deterministic_cutoffs"])
        assessment = sc.assess_class_recording(inp, llm_res, cutoffs)
        timings["score"] = time.perf_counter() - t0
        report(1.0, "Done")

        return {
            "assessment": assessment,
            "llm": {"status": assessment["llm_status"], "note": llm_note,
                    "model": (llm_res or {}).get("model"), "usage": (llm_res or {}).get("usage")},
            "telemetry": {
                "file_hash": file_hash[:16], "duration_min": round(duration / 60, 1),
                "visual": {k: v for k, v in visual.items() if k != "camera_off_times_sec"},
                "audio": audio,
                "transcript_features": {k: v for k, v in feats.items() if k not in _SNIPPET_KEYS},
                "camera_refined": cam,
                "deid": deid["stats"],
                "speaker": roles,
                "asr": {"model": whisper_model, "language": language, "device": device, "compute_type": compute,
                        "segments": len(segs), "words": tr.get("word_count"), "rtf": tr.get("rtf"),
                        "avg_logprob_mean": avg_lp, "non_latin_pct": tr.get("non_latin_pct")},
                "timings_sec": {k: round(v, 1) for k, v in timings.items()},
            },
            "temp_dir": str(tmp_path),
        }
    finally:
        tmp.cleanup()
        gc.collect()
