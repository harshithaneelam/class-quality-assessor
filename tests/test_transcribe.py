"""transcribe_audio with a mocked WhisperModel and a synthetic WAV."""
import wave
from types import SimpleNamespace

import numpy as np
import pytest

import src.transcribe as tr


@pytest.fixture
def wav(tmp_path):
    p = tmp_path / "a.wav"
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes((np.random.default_rng(0).normal(0, 500, 16000 * 10)).astype(np.int16).tobytes())
    return p


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, path, **kw):
        self.calls.append(kw)
        segs = [SimpleNamespace(start=0.5, end=4.0, text=" Good morning, class.", avg_logprob=-0.21,
                                no_speech_prob=0.01),
                SimpleNamespace(start=5.0, end=10.0, text=" Open your books!", avg_logprob=-0.35,
                                no_speech_prob=0.02)]
        return iter(segs), SimpleNamespace(duration=10.0, language=kw.get("language"))


@pytest.fixture
def fake(monkeypatch):
    m = FakeModel()
    monkeypatch.setattr(tr, "_load_model", lambda *a, **k: m)
    return m


def test_transcribe_shape_and_progress(wav, fake):
    progress = []
    r = tr.transcribe_audio(wav, progress_callback=progress.append)
    assert fake.calls[0]["language"] == "en" and fake.calls[0]["beam_size"] == 1
    assert fake.calls[0]["vad_filter"] is True
    assert fake.calls[0]["vad_parameters"] == {"min_silence_duration_ms": 500}
    assert r["segment_count"] == 2 and r["word_count"] == 6
    assert r["duration_sec"] == 10.0 and r["non_latin_pct"] == 0.0 and r["rtf"] is not None
    assert set(r["segments"][0]) == {"id", "start", "end", "text", "avg_logprob", "no_speech_prob"}
    assert progress == [0.4, 1.0, 1.0]


def test_cache_hit_skips_model(wav, fake, tmp_path):
    cache = tmp_path / "cache"
    first = tr.transcribe_audio(wav, cache_dir=cache)
    second = tr.transcribe_audio(wav, cache_dir=cache)
    assert len(fake.calls) == 1
    assert not first["cache_hit"] and second["cache_hit"]
    assert second["segments"] == first["segments"]
    tr.transcribe_audio(wav, language="ml", cache_dir=cache)  # different config -> new run
    assert len(fake.calls) == 2


def test_cache_key_depends_on_config_and_audio(wav, tmp_path):
    k = tr.cache_key(wav, "small", "en", 1)
    assert k == tr.cache_key(wav, "small", "en", 1)
    assert k != tr.cache_key(wav, "small", "ml", 1)
    assert k != tr.cache_key(wav, "base", "en", 1)
    assert k != tr.cache_key(wav, "small", "en", 3)
    other = tmp_path / "b.wav"
    other.write_bytes(wav.read_bytes()[:-2] + b"\x01\x02")
    assert tr.audio_fingerprint(other) != tr.audio_fingerprint(wav)
