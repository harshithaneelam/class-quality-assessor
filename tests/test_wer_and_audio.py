"""Synthetic tests for PyAV audio extraction and WER (no real dataset content)."""
import wave
from fractions import Fraction

import av
import numpy as np
import pytest

from src.audio_extract import extract_wav_16k_mono, slice_wav
from src.wer import normalize_words, parse_vtt_cues, parse_vtt_speaker_cues, reference_words, wer


@pytest.fixture
def synthetic_video(tmp_path):
    """2.0 s, 44.1 kHz stereo 440 Hz tone in an MP4/AAC container."""
    path = tmp_path / "tone.mp4"
    sr, secs = 44100, 2.0
    t = np.arange(int(sr * secs)) / sr
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    with av.open(str(path), "w") as c:
        s = c.add_stream("aac", rate=sr)
        s.layout = "stereo"
        for i in range(0, len(tone), 1024):
            chunk = tone[i:i + 1024]
            frame = av.AudioFrame.from_ndarray(np.stack([chunk, chunk]), format="fltp", layout="stereo")
            frame.sample_rate = sr
            frame.pts = i
            frame.time_base = Fraction(1, sr)
            for p in s.encode(frame):
                c.mux(p)
        for p in s.encode(None):
            c.mux(p)
    return path


def test_extract_wav_16k_mono(synthetic_video, tmp_path):
    out = tmp_path / "audio" / "x.wav"
    info = extract_wav_16k_mono(synthetic_video, out)
    assert info["sample_rate"] == 16000 and not info["reused"]
    assert abs(info["duration_sec"] - 2.0) < 0.1
    with wave.open(str(out), "rb") as wf:
        assert (wf.getnchannels(), wf.getsampwidth(), wf.getframerate()) == (1, 2, 16000)
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    assert np.abs(pcm[4000:-4000]).max() > 5000  # tone survived, not silence

    again = extract_wav_16k_mono(synthetic_video, out)
    assert again["reused"] and again["extract_time_sec"] == 0.0
    assert not extract_wav_16k_mono(synthetic_video, out, force=True)["reused"]


def test_slice_wav(synthetic_video, tmp_path):
    full = tmp_path / "full.wav"
    extract_wav_16k_mono(synthetic_video, full)
    info = slice_wav(full, tmp_path / "cut.wav", 0.5, 1.5)
    assert abs(info["duration_sec"] - 1.0) < 1e-3


def test_wer_basic():
    ref = "the cat sat on the mat".split()
    assert wer(ref, ref) == 0.0
    assert wer(ref, "the cat sat on mat".split()) == pytest.approx(1 / 6)       # 1 deletion
    assert wer(ref, "the dog sat on the mat now".split()) == pytest.approx(2 / 6)  # 1 sub + 1 ins
    assert wer([], []) == 0.0


def test_normalize_words():
    assert normalize_words("Hello, World!  It's [inaudible] FINE.") == ["hello", "world", "its", "fine"]


def test_vtt_prefix_stripping(tmp_path):
    p = tmp_path / "s.vtt"
    p.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nZorblat Q: Good morning, class!\n\n"
                 "00:01:00.000 --> 00:01:02.500\n<v Mxyzzy>Hi miss.</v>\n\n"
                 "00:05:01.000 --> 00:05:02.000\nno prefix here\n", encoding="utf-8")
    cues = parse_vtt_cues(p)
    assert [c[0] for c in cues] == [1.0, 60.0, 301.0]
    assert reference_words(cues) == ["good", "morning", "class", "hi", "miss", "no", "prefix", "here"]
    assert reference_words(cues, 0, 300) == ["good", "morning", "class", "hi", "miss"]
    assert "zorblat" not in reference_words(cues) and "mxyzzy" not in reference_words(cues)


def test_vtt_speaker_cues(tmp_path):
    p = tmp_path / "s.vtt"
    p.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nZorblat Quenterly: Good morning, class!\n\n"
                 "00:00:03.000 --> 00:00:04.000\n<v Mxyzzy>Hi miss.</v>\n\n"
                 "00:00:05.000 --> 00:00:06.000\nno prefix here\n", encoding="utf-8")
    cues = parse_vtt_speaker_cues(p)
    assert [c[2] for c in cues] == ["Zorblat Quenterly", "Mxyzzy", None]
    assert normalize_words(cues[0][3]) == ["good", "morning", "class"]
    assert normalize_words(cues[1][3]) == ["hi", "miss"]
