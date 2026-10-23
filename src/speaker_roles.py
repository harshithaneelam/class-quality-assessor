"""Segment-level TEACHER / STUDENT attribution without speaker labels.

Score = 0.5 teacher prior + lexical / dialog-turn cues + an acoustic vote.
The acoustic vote comes from a per-recording 2-means clustering of segments on
(log F0, RMS dB); the cluster holding more speech time is the teacher (the
teacher dominates class talk). Positive score -> TEACHER.

Validated against VTT speaker prefixes on the 12 labelled classes (step 04):
~67% segment accuracy, ~60% balanced accuracy. A fixed "long-segment" F0
baseline was tried first and was worse than always guessing TEACHER, because
Whisper small rarely emits segments > 25 words.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_STUDENT_PH = re.compile(r"\[STUDENT_\d+\]")
_TEACHER_CUES = re.compile(
    r"(?i)\b(can you|let's see|look at the board|homework|puzzle|next question|everyone|unmute)\b")
_PROMPT = re.compile(r"(?i)(\?|\btell me\b)")
_STUDENT_REPLY = re.compile(
    r"(?i)(\bknight to\b|\bqueen\b|\bbishop\b|\b[a-h][1-8]\b|\bcapture[sd]?\b|\bcheck\b|\byes,? sir\b|"
    r"\byes,? ma'?am\b|\bgood evening\b)")
_MIN_ACOUSTIC_SEC = 0.8
_MAX_ACOUSTIC_SEC = 6.0


def _n_words(text: str) -> int:
    return len(re.findall(r"[\w']+", _STUDENT_PH.sub("x", text)))


def segment_acoustics(wav_path: str | Path, segments: list[dict]) -> list[tuple[float | None, float | None]]:
    """(median F0 Hz, RMS dB) per segment; None for segments shorter than 0.8 s.

    Reads only the needed slice (centre <= 6 s) of each segment from disk.
    """
    import librosa
    import soundfile as sf

    out = []
    with sf.SoundFile(str(wav_path)) as f:
        sr, n_frames = f.samplerate, f.frames
        for s in segments:
            dur = float(s["end"]) - float(s["start"])
            if dur < _MIN_ACOUSTIC_SEC:
                out.append((None, None))
                continue
            take = min(dur, _MAX_ACOUSTIC_SEC)
            start = int((float(s["start"]) + (dur - take) / 2) * sr)
            if start >= n_frames - sr // 2:  # Whisper timestamps can run past EOF
                out.append((None, None))
                continue
            f.seek(start)
            y = f.read(min(int(take * sr), n_frames - start), dtype="float32", always_2d=False)
            if y.size < sr * 0.5:
                out.append((None, None))
                continue
            rms = float(np.sqrt(np.mean(y ** 2)) + 1e-9)
            rms_db = 20 * np.log10(rms)
            try:
                f0 = librosa.yin(y, fmin=70, fmax=500, sr=sr, frame_length=1024)
                # yin returns a value for every frame; keep frames with energy (voiced-ish)
                frame_rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0][: len(f0)]
                voiced = f0[: len(frame_rms)][frame_rms > np.percentile(frame_rms, 40)]
                med = float(np.median(voiced)) if voiced.size else None
            except Exception:
                med = None
            out.append((med, rms_db))
    return out


def acoustic_votes(segs: list[dict], acoustics) -> tuple[dict[int, float], float | None, float | None]:
    """Per-recording 2-means on standardised (log F0, RMS dB).

    Returns ({segment index: vote in [-2, 2], + = teacher-like}, teacher F0 Hz, teacher RMS dB).
    """
    ix = [i for i, (f0, rms) in enumerate(acoustics) if f0 and rms is not None]
    if len(ix) < 4:
        return {}, None, None
    X = np.array([[np.log(acoustics[i][0]), acoustics[i][1]] for i in ix])
    Z = (X - X.mean(0)) / (X.std(0) + 1e-6)
    c = Z[np.argsort(Z[:, 0])[[len(Z) // 4, 3 * len(Z) // 4]]].copy()  # init: low / high pitch
    for _ in range(30):
        lab = np.argmin(((Z[:, None, :] - c[None]) ** 2).sum(-1), 1)
        c = np.array([Z[lab == k].mean(0) if (lab == k).any() else c[k] for k in range(2)])
    # Orient by lexical mass, not talk time: the teacher cluster holds more words and
    # instructional cues (talk-time orientation inverted poor_weak_1 to 16% teacher).
    mass = [sum(_n_words(segs[ix[j]]["text"]) + 5 * len(_TEACHER_CUES.findall(segs[ix[j]]["text"]))
                + 5 * len(_STUDENT_PH.findall(segs[ix[j]]["text"]))
                for j in range(len(ix)) if lab[j] == k) for k in range(2)]
    tk = int(np.argmax(mass))
    d_t = ((Z - c[tk]) ** 2).sum(-1)
    d_s = ((Z - c[1 - tk]) ** 2).sum(-1)
    votes = {i: float(np.clip((d_s[j] - d_t[j]) / 2, -2, 2)) for j, i in enumerate(ix)}
    members = [j for j in range(len(ix)) if lab[j] == tk]
    t_f0 = float(np.exp(np.median(X[members, 0])))
    t_rms = float(np.median(X[members, 1]))
    return votes, t_f0, t_rms


def attribute_speaker_roles(masked_segments: list[dict], wav_path: str | Path | None = None) -> dict:
    segs = [dict(s) for s in masked_segments]
    n = len(segs)
    acoustics = segment_acoustics(wav_path, segs) if (wav_path and n) else [(None, None)] * n
    votes, t_f0, t_rms = acoustic_votes(segs, acoustics)
    scores = _score(segs, votes)
    # Guard against cluster inversion: the teacher carries the majority of class talk,
    # so if the teacher word share comes out < 45 %, take the flipped orientation when
    # that raises it.
    if votes and _teacher_word_share(segs, scores) < MIN_TEACHER_WORD_SHARE:
        flipped = _score(segs, {i: -v for i, v in votes.items()})
        if _teacher_word_share(segs, flipped) > _teacher_word_share(segs, scores):
            scores = flipped

    for s, sc, (f0, rms) in zip(segs, scores, acoustics):
        s["speaker_role"] = "TEACHER" if sc >= 0 else "STUDENT"
        s["role_score"] = round(sc, 2)
        s["f0_hz"] = round(f0, 1) if f0 else None
        s["rms_db"] = round(rms, 1) if rms is not None else None
    return _summary(segs, t_f0, t_rms)


MIN_TEACHER_WORD_SHARE = 0.45


def _teacher_word_share(segs, scores) -> float:
    total = sum(_n_words(s["text"]) for s in segs)
    t = sum(_n_words(s["text"]) for s, sc in zip(segs, scores) if sc >= 0)
    return t / total if total else 1.0


def _score(segs, votes) -> list[float]:
    scores = []
    for i, s in enumerate(segs):
        text, nw = s["text"], _n_words(s["text"])
        sc = 0.5  # teacher prior
        if _STUDENT_PH.search(text):
            sc += 1.5
        sc += len(_TEACHER_CUES.findall(text))
        if nw > 12:
            sc += 1
        if 1 <= nw <= 12:
            prev = segs[i - 1] if i else None
            if prev and s["start"] - prev["end"] <= 4 and (
                    _PROMPT.search(prev["text"]) or _STUDENT_PH.search(prev["text"])):
                sc -= 0.75
            if _STUDENT_REPLY.search(text):
                sc -= 0.75
            if nw <= 3:
                sc -= 0.25
        sc += votes.get(i, 0.0)
        scores.append(sc)
    return scores


def _summary(segs, t_f0, t_rms) -> dict:
    n = len(segs)
    dur_t = sum(s["end"] - s["start"] for s in segs if s["speaker_role"] == "TEACHER")
    dur_s = sum(s["end"] - s["start"] for s in segs if s["speaker_role"] == "STUDENT")
    total = dur_t + dur_s
    switches = sum(1 for a, b in zip(segs, segs[1:]) if a["speaker_role"] != b["speaker_role"])
    span_min = ((segs[-1]["end"] - segs[0]["start"]) / 60) if n else 0.0
    return {
        "segments": segs,
        "teacher_talk_time_pct": round(100 * dur_t / total, 1) if total else 0.0,
        "student_talk_time_pct": round(100 * dur_s / total, 1) if total else 0.0,
        "student_turn_count": sum(1 for s in segs if s["speaker_role"] == "STUDENT"),
        "interaction_alternations_per_10min": round(10 * switches / span_min, 2) if span_min else 0.0,
        "teacher_f0_baseline_hz": round(t_f0, 1) if t_f0 else None,
        "teacher_rms_baseline_db": round(t_rms, 1) if t_rms is not None else None,
    }
