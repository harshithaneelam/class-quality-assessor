"""Audio prosody & silence features from a 16 kHz mono WAV (streamed, ~1 MB RAM).

Speech/silence is decided per 1-second window: RMS dB above
max(p10_db + 8, -48 dBFS). Pitch is estimated with YIN on a uniform sample of
up to 120 active windows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

WINDOW_SEC = 1.0
SILENCE_MIN_GAP = 5
PITCH_SAMPLE_WINDOWS = 120


def _ts(sec: float) -> str:
    sec = max(0, int(sec))
    return f"[{sec // 60:02d}:{sec % 60:02d}]"


def _window_db(wav_path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    sr = sf.info(str(wav_path)).samplerate
    dbs = []
    for block in sf.blocks(str(wav_path), blocksize=int(sr * WINDOW_SEC), dtype="float32", always_2d=False):
        if block.ndim > 1:
            block = block.mean(axis=1)
        rms = float(np.sqrt(np.mean(block ** 2))) if block.size else 0.0
        dbs.append(20 * np.log10(rms + 1e-10))
    return np.array(dbs), sr


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs where mask is True."""
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _pitch_sample(wav_path: Path, sr: int, windows: np.ndarray) -> np.ndarray:
    """Median voiced F0 per sampled 1-s window."""
    import librosa
    import soundfile as sf
    meds = []
    with sf.SoundFile(str(wav_path)) as f:
        for w in windows:
            f.seek(int(w * sr * WINDOW_SEC))
            y = f.read(int(sr * WINDOW_SEC), dtype="float32", always_2d=False)
            if y.size < sr // 2:
                continue
            f0 = librosa.yin(y, fmin=70, fmax=500, sr=sr, frame_length=1024)
            e = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0][: len(f0)]
            voiced = f0[: len(e)][e > np.percentile(e, 40)]
            if voiced.size:
                meds.append(float(np.median(voiced)))
    return np.array(meds)


def extract_audio_features(wav_path: str | Path) -> dict:
    wav_path = Path(wav_path)
    db, sr = _window_db(wav_path)
    n = len(db)
    if n == 0:
        return {"duration_sec": 0.0}
    p10 = float(np.percentile(db, 10))
    thr = max(p10 + 8.0, -48.0)
    active = db > thr
    speech_idx = np.flatnonzero(active)

    gaps = []
    if speech_idx.size:
        first, last = int(speech_idx[0]), int(speech_idx[-1])
        gaps = [(a, b) for a, b in _runs(~active[first:last + 1])]
        gaps = [(a + first, b + first) for a, b in gaps]
    long5 = [(a, b) for a, b in gaps if b - a >= SILENCE_MIN_GAP]
    long10 = [(a, b) for a, b in long5 if b - a >= 10]
    max_gap = max(gaps, key=lambda g: g[1] - g[0]) if gaps else None

    act_db = db[active]
    rms_mean = float(np.mean(act_db)) if act_db.size else None
    rms_std = float(np.std(act_db)) if act_db.size else None
    dyn = float(np.percentile(act_db, 95) - np.percentile(act_db, 15)) if act_db.size else None

    f0 = np.array([])
    if speech_idx.size:
        pick = speech_idx[np.linspace(0, speech_idx.size - 1, min(PITCH_SAMPLE_WINDOWS, speech_idx.size)).astype(int)]
        f0 = _pitch_sample(wav_path, sr, pick)
    f0_med = float(np.median(f0)) if f0.size else None
    f0_iqr = float(np.percentile(f0, 75) - np.percentile(f0, 25)) if f0.size else None
    f0_cv = float(np.std(f0) / np.mean(f0)) if f0.size and np.mean(f0) > 0 else None

    # Liveliness: each component scaled so ~1.0 is an animated delivery.
    parts = [min(1.0, rms_std / 8.0) if rms_std is not None else None,
             min(1.0, dyn / 20.0) if dyn is not None else None,
             min(1.0, f0_cv / 0.30) if f0_cv is not None else None]
    parts = [p for p in parts if p is not None]

    return {
        "duration_sec": round(n * WINDOW_SEC, 1),
        "silence_threshold_db": round(thr, 1),
        "speech_ratio_pct": round(100 * float(active.mean()), 1),
        "silence_gaps_gt_5s_count": len(long5),
        "silence_gaps_gt_10s_count": len(long10),
        "dead_air_total_sec": float(sum(b - a for a, b in long5) * WINDOW_SEC),
        "max_silence_gap_sec": float((max_gap[1] - max_gap[0]) * WINDOW_SEC) if max_gap else 0.0,
        "max_silence_gap_ts": f"{_ts(max_gap[0])}-{_ts(max_gap[1])}" if max_gap else None,
        "longest_silences_ts": [f"{_ts(a)}-{_ts(b)}" for a, b in sorted(long10, key=lambda g: g[0] - g[1])[:3]],
        "rms_mean_db": round(rms_mean, 1) if rms_mean is not None else None,
        "rms_std_db": round(rms_std, 2) if rms_std is not None else None,
        "dynamic_range_db": round(dyn, 1) if dyn is not None else None,
        "pitch_windows_used": int(f0.size),
        "pitch_f0_hz_median": round(f0_med, 1) if f0_med else None,
        "pitch_f0_hz_iqr": round(f0_iqr, 1) if f0_iqr is not None else None,
        "pitch_variation_cv": round(f0_cv, 3) if f0_cv is not None else None,
        "energy_liveliness_score": round(float(np.mean(parts)), 3) if parts else None,
    }
