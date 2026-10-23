"""Memory-safe audio extraction with PyAV (no system ffmpeg binary needed).

Audio frames are decoded, resampled to 16 kHz mono s16 and streamed straight
into a WAV file, so peak RAM stays at a few frames regardless of class length.
"""
from __future__ import annotations

import os
import time
import wave
from pathlib import Path

import av

SAMPLE_RATE = 16000


def _wav_info(path: Path) -> tuple[float, int]:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / wf.getframerate(), wf.getframerate()


def extract_wav_16k_mono(video_path: str | Path, out_wav_path: str | Path, force: bool = False) -> dict:
    """Decode the first audio stream of ``video_path`` into a 16 kHz mono int16 WAV.

    Reuses an existing non-empty ``out_wav_path`` unless ``force`` is True.
    Returns duration_sec, sample_rate, extract_time_sec, wav_size_mb (+ reused).
    """
    out = Path(out_wav_path)
    t0 = time.perf_counter()
    if out.is_file() and out.stat().st_size > 44 and not force:
        dur, sr = _wav_info(out)
        return {"duration_sec": dur, "sample_rate": sr, "extract_time_sec": 0.0,
                "wav_size_mb": out.stat().st_size / 2**20, "reused": True}

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".wav.part")
    n_samples = 0
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    with av.open(str(video_path)) as container, wave.open(str(tmp), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        stream = container.streams.audio[0]
        stream.thread_type = "AUTO"

        def _write(frames):
            nonlocal n_samples
            for f in frames:
                pcm = f.to_ndarray()  # (1, n) int16 for packed mono s16
                wf.writeframes(pcm.tobytes())
                n_samples += pcm.shape[-1]

        for frame in container.decode(stream):
            frame.pts = None  # let the resampler re-time; avoids pts gaps errors
            _write(resampler.resample(frame))
        _write(resampler.resample(None))  # flush buffered samples
    os.replace(tmp, out)
    return {"duration_sec": n_samples / SAMPLE_RATE, "sample_rate": SAMPLE_RATE,
            "extract_time_sec": time.perf_counter() - t0,
            "wav_size_mb": out.stat().st_size / 2**20, "reused": False}


def slice_wav(src_wav: str | Path, dst_wav: str | Path, start_sec: float, end_sec: float) -> dict:
    """Copy [start_sec, end_sec) of a WAV into a new WAV in fixed-size chunks."""
    src, dst = Path(src_wav), Path(dst_wav)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(src), "rb") as r, wave.open(str(dst), "wb") as w:
        sr = r.getframerate()
        w.setparams(r.getparams())
        start = min(int(start_sec * sr), r.getnframes())
        end = min(int(end_sec * sr), r.getnframes())
        r.setpos(start)
        remaining = max(0, end - start)
        while remaining:
            n = min(remaining, sr * 30)
            w.writeframes(r.readframes(n))
            remaining -= n
    dur, _ = _wav_info(dst)
    return {"duration_sec": dur, "start_sec": start / sr, "end_sec": end / sr}
