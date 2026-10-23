"""Production transcription: faster-whisper with a locked config + disk cache.

Locked config: model "small", language "en", beam_size 1, Silero VAD
(min_silence 500 ms). Small/base auto-detect mislabels these classes as
Malayalam, so language is always passed explicitly.

PRIVACY: returns transcript text to the caller but never prints it.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import time
import wave
from pathlib import Path

from src.utils.cuda_setup import register_cuda_dlls
from src.wer import non_latin_pct, normalize_words

VAD_PARAMETERS = dict(min_silence_duration_ms=500)
CACHE_VERSION = 1  # bump when the output schema / decoding settings change
_HASH_CHUNK = 4 * 2**20

_models: dict[tuple, object] = {}


def audio_fingerprint(wav_path: str | Path) -> str:
    """SHA-256 of first 4 MB + last 4 MB + file size (fast, stable for large WAVs)."""
    p = Path(wav_path)
    size = p.stat().st_size
    h = hashlib.sha256()
    with p.open("rb") as f:
        h.update(f.read(_HASH_CHUNK))
        if size > _HASH_CHUNK:
            f.seek(max(_HASH_CHUNK, size - _HASH_CHUNK))
            h.update(f.read(_HASH_CHUNK))
    h.update(str(size).encode())
    return h.hexdigest()


def cache_key(wav_path: str | Path, model_name: str, language: str | None, beam_size: int) -> str:
    h = hashlib.sha256()
    h.update(audio_fingerprint(wav_path).encode())
    h.update(json.dumps([model_name, language, beam_size, VAD_PARAMETERS, CACHE_VERSION]).encode())
    return h.hexdigest()


def _load_model(model_name: str, device: str, compute_type: str, cpu_threads: int):
    """Load (or reuse) a WhisperModel. Kept in-process so a batch loads it once."""
    key = (model_name, device, compute_type, cpu_threads)
    if key not in _models:
        register_cuda_dlls()
        from faster_whisper import WhisperModel
        kw = {"cpu_threads": cpu_threads} if cpu_threads else {}
        _models[key] = WhisperModel(model_name, device=device, compute_type=compute_type, **kw)
    return _models[key]


def release_models() -> None:
    """Drop all loaded models and free host / GPU memory."""
    _models.clear()
    gc.collect()


def _wav_duration(wav_path: Path) -> float:
    try:
        with wave.open(str(wav_path), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except (wave.Error, EOFError):
        return 0.0


def transcribe_audio(wav_path: str | Path, model_name: str = "small", language: str = "en",
                     device: str = "auto", compute_type: str = "default", beam_size: int = 1,
                     cache_dir: str | Path | None = None, progress_callback=None,
                     cpu_threads: int = 0) -> dict:
    """Transcribe a 16 kHz WAV into timestamped segments (cached by audio hash + config)."""
    register_cuda_dlls()
    wav_path = Path(wav_path)
    cache_file = None
    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"{cache_key(wav_path, model_name, language, beam_size)}.json"
        if cache_file.is_file():
            result = json.loads(cache_file.read_text(encoding="utf-8"))
            result["cache_hit"] = True
            if progress_callback:
                progress_callback(1.0)
            return result

    if device == "cpu" and not cpu_threads:
        # Small CPU containers (e.g. Streamlit Community Cloud: 2 vCPU, ~2.7 GB RAM): don't oversubscribe.
        cpu_threads = int(os.environ.get("WHISPER_CPU_THREADS", "2"))
    model = _load_model(model_name, device, compute_type, cpu_threads)
    t0 = time.perf_counter()
    seg_iter, info = model.transcribe(str(wav_path), language=language, beam_size=beam_size,
                                      vad_filter=True, vad_parameters=VAD_PARAMETERS)
    total = float(getattr(info, "duration", 0.0) or _wav_duration(wav_path))
    segments = []
    for i, s in enumerate(seg_iter):  # decoding happens while iterating
        segments.append({"id": i, "start": round(s.start, 2), "end": round(s.end, 2), "text": s.text,
                         "avg_logprob": round(s.avg_logprob, 4),
                         "no_speech_prob": round(s.no_speech_prob, 4)})
        if progress_callback and total > 0:
            progress_callback(min(1.0, s.end / total))
    wall = time.perf_counter() - t0
    if progress_callback:
        progress_callback(1.0)

    text = " ".join(s["text"] for s in segments)
    result = {
        "model_name": model_name, "language": language, "beam_size": beam_size,
        "duration_sec": round(total, 2), "wall_time_sec": round(wall, 2),
        "rtf": round(wall / total, 4) if total else None,
        "segment_count": len(segments), "word_count": len(normalize_words(text)),
        "non_latin_pct": non_latin_pct(text), "segments": segments, "cache_hit": False,
    }
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(".json.part")
        tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, cache_file)
    return result
