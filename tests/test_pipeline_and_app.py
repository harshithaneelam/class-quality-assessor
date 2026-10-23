"""Offline end-to-end pipeline + Gradio app tests (synthetic video, mocked Whisper / Gemini)."""
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from src import llm_judge as lj
from src import pipeline as pl
from src import visual_features as vf


class NoFaces:
    kind = "fake"

    def detect(self, bgr):
        return []


@pytest.fixture(scope="module")
def synthetic_mp4(tmp_path_factory):
    """45 s, 320x180 @ 2 fps dark frames + 16 kHz mono tone with pauses."""
    path = tmp_path_factory.mktemp("vid") / "class.mp4"
    fps, secs, sr = 2, 45, 16000
    with av.open(str(path), "w") as c:
        vs = c.add_stream("mpeg4", rate=fps)
        vs.width, vs.height, vs.pix_fmt = 320, 180, "yuv420p"
        aus = c.add_stream("aac", rate=sr)
        aus.layout = "mono"
        for i in range(fps * secs):
            img = np.full((180, 320, 3), 25, np.uint8)
            img[10:30, 20:100] = 235
            for p in vs.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)
        t = np.arange(sr * secs) / sr
        tone = (0.3 * np.sin(2 * np.pi * 180 * t) * (np.sin(2 * np.pi * 0.1 * t) > -0.3)).astype(np.float32)
        for i in range(0, len(tone), 1024):
            fr = av.AudioFrame.from_ndarray(tone[None, i:i + 1024], format="fltp", layout="mono")
            fr.sample_rate, fr.pts, fr.time_base = sr, i, Fraction(1, sr)
            for p in aus.encode(fr):
                c.mux(p)
        for s in (vs, aus):
            for p in s.encode(None):
                c.mux(p)
    return path


def fake_transcribe(wav, **kw):
    kw["progress_callback"](0.5)
    segs = [
        {"id": 0, "start": 1.0, "end": 5.0, "text": "Good evening everyone, welcome! How are you?", "avg_logprob": -0.2, "no_speech_prob": 0.01},
        {"id": 1, "start": 6.0, "end": 9.0, "text": "Ananya, can you tell me the best move?", "avg_logprob": -0.2, "no_speech_prob": 0.01},
        {"id": 2, "start": 9.5, "end": 10.5, "text": "Knight to f3.", "avg_logprob": -0.3, "no_speech_prob": 0.02},
        {"id": 3, "start": 12.0, "end": 16.0, "text": "Well done! For homework solve these puzzles.", "avg_logprob": -0.2, "no_speech_prob": 0.01},
        {"id": 4, "start": 40.0, "end": 44.0, "text": "Thank you, bye, see you next class.", "avg_logprob": -0.2, "no_speech_prob": 0.01},
    ]
    return {"segments": segs, "duration_sec": 45.0, "word_count": 30, "rtf": 0.01, "non_latin_pct": 0.0}


def fake_judge(short_id, prompt, block, forbidden, cache_dir=None):
    assert "Ananya" not in prompt and "[STUDENT_1]" in prompt  # de-identified before the judge
    lj.preflight_privacy_check(prompt, forbidden)
    payload = {"pedagogical_scores": {k: 80 for k in ("intro_and_onboarding_0_100", "individual_attention_0_100",
                                                      "communication_energy_and_rapport_0_100",
                                                      "lesson_structure_and_homework_0_100",
                                                      "visual_and_brand_compliance_0_100")},
               "llm_raw_label": "Good", "poor_sub_category": "none", "executive_summary": "Friendly short class.",
               "strengths": [{"timestamp": "00:01", "dimension": "intro_onboarding", "observation": "Warm welcome."}],
               "improvements": [{"timestamp": "[00:40]", "dimension": "structure_homework", "observation": "Longer wrap-up."}]}
    return {"assessment": lj.normalize_assessment(lj.LLMClassAssessment.model_validate(payload)),
            "model": "mock-gemini", "usage": {}, "llm_status": "ok", "cache_hit": False}


# ------------------------------------------------------------ assets
def test_bundled_logo_templates_load_without_work_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CHALLENGE_DATA_DIR", str(tmp_path / "missing"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert not vf.default_models_dir().exists()
    t = vf.load_logo_templates()
    assert t is not None and t.shape[1:] == (18, 32) and t.shape[0] == 3 * len(vf.LOGO_SCALES)
    assert vf.LOGO_TEMPLATES_ASSET.stat().st_size < 50_000
    with np.load(vf.LOGO_TEMPLATES_ASSET) as z:
        assert set(z.files) == {"templates", "grid_w", "grid_h"} and z["templates"].dtype == np.float32
        assert 0.0 <= z["templates"].min() and z["templates"].max() <= 1.0


def test_calibration_asset_present():
    c = pl.load_calibration()
    assert c["label_source"] == "deterministic" and len(c["deterministic_cutoffs"]) == 3


# ------------------------------------------------------------ pipeline
def test_pipeline_end_to_end_offline_and_temp_cleanup(synthetic_mp4, tmp_path):
    progress = []
    r = pl.assess_video_pipeline(synthetic_mp4, "en", progress_cb=lambda f, d: progress.append((f, d)),
                                 transcribe_fn=fake_transcribe, judge_fn=fake_judge, detector=NoFaces(),
                                 models_dir=tmp_path)
    a = r["assessment"]
    assert a["label"] in ("Poor", "Average", "Good", "Excellent") and a["llm_status"] == "ok"
    assert r["llm"]["model"] == "mock-gemini"
    fr = [f for f, _ in progress]
    assert fr[0] == 0.0 and fr[-1] == 1.0 and fr == sorted(fr)
    assert {0.08, 0.15, 0.40, 0.82}.issubset(set(fr))
    assert 0.61 in fr  # transcription progress mapped into 0.40-0.82 (fake reports 50 %)
    assert not Path(r["temp_dir"]).exists()  # ephemeral workspace removed
    assert r["telemetry"]["visual"]["sampled_frames"] >= 1 and r["telemetry"]["deid"]["distinct_student_placeholders"] == 1
    assert "individual_question_examples" not in r["telemetry"]["transcript_features"]
    assert all(o["timestamp"] for o in a["strengths"] + a["improvements"])


def test_pipeline_deterministic_when_llm_disabled(synthetic_mp4, tmp_path):
    r = pl.assess_video_pipeline(synthetic_mp4, transcribe_fn=fake_transcribe, detector=NoFaces(),
                                 models_dir=tmp_path, use_llm=False)
    assert r["assessment"]["llm_status"] == "fallback_deterministic" and r["llm"]["note"]


def test_pipeline_cleans_temp_on_error(tmp_path, monkeypatch):
    made = []
    real = pl.tempfile.TemporaryDirectory

    def spy(*a, **k):
        d = real(*a, **k)
        made.append(d.name)
        return d
    monkeypatch.setattr(pl.tempfile, "TemporaryDirectory", spy)
    bad = tmp_path / "not_a_video.mp4"
    bad.write_bytes(b"nope")
    with pytest.raises(ValueError):
        pl.assess_video_pipeline(bad, transcribe_fn=fake_transcribe, detector=NoFaces())
    assert made and not Path(made[0]).exists()


def test_url_validation_messages():
    with pytest.raises(ValueError, match="folder"):
        pl.assess_video_pipeline("https://drive.google.com/drive/folders/abc123")
    with pytest.raises(ValueError, match="not found"):
        pl.assess_video_pipeline("C:/definitely/missing/video.mp4")


# ------------------------------------------------------------ app
def sample_result(label="Poor"):
    return {
        "assessment": {
            "label": label, "poor_sub_category": "compliance_quality_issue", "confidence_pct": 91.0, "composite": 41.0,
            "executive_summary": "Summary with <script>alert(1)</script> injection attempt.",
            "llm_status": "fallback_deterministic", "language_warning": "Low ASR confidence — likely code-mixed.",
            "sub_scores": {"visual_compliance": 10, "intro_onboarding": 60, "individual_attention": 70,
                           "energy_rapport": 50, "structure_homework": 40},
            "deterministic_scores": {}, "llm_scores": None, "triggered_gates": ["C1_no_branded_bg"],
            "compliance_flags": {"no_8x8": True, "face_not_visible": False, "dark_face": False, "bg_clarity": False,
                                 "camera_off_while_teaching": False},
            "strengths": [{"timestamp": "[01:00]", "dimension": "intro_onboarding", "observation": "Warm start.",
                           "source": "deterministic"}],
            "improvements": [{"timestamp": "[05:00]", "dimension": "visual_compliance",
                              "observation": "Keep the branded virtual background on <b>.", "source": "llm"}],
        },
        "llm": {"status": "fallback_deterministic", "note": "GEMINI_API_KEY is missing", "model": None},
        "telemetry": {
            "visual": {"flag_no_8x8_background": True, "bg_8x8_when_cam_active_pct": 20.0,
                       "tutor_face_visible_when_8x8_pct": 90.0, "tutor_face_luminance_mean": 100.0},
            "audio": {"dead_air_total_sec": 120.0, "max_silence_gap_sec": 30.0},
            "transcript_features": {"has_warm_intro": True, "intro_score": 0.75, "intro_evidence_ts": "[00:20]",
                                    "distinct_students_called": 6, "teacher_question_count": 80},
            "camera_refined": {"max_camera_off_during_speech_streak_sec": 0.0},
        },
    }


def test_app_builds_and_renders():
    import app
    demo = app.build_app()
    assert demo is not None
    r = sample_result()
    hero = app.render_hero(r)
    assert "POOR" in hero and "Compliance / Quality Issue" in hero and "91%" in hero
    assert "<script>" not in hero and "&lt;script&gt;" in hero
    assert "Deterministic Rubric Fallback" in hero and "code-mixed" in hero
    scores = app.render_scores(r["assessment"])
    assert scores.count("ca-row") == 5 and "Vocal Energy &amp; Rapport" in scores
    check = app.render_checklist(r)
    assert "Branded virtual background kept" in check and ">FAIL<" in check and ">PASS<" in check
    ev = app.render_evidence(r["assessment"])
    assert "[05:00]" in ev and "&lt;b&gt;" in ev and "AI judge" in ev and "Measured" in ev
    assert app.telemetry_payload(r)["label"] == "Poor"


def test_app_handler_requires_input():
    import gradio as gr
    import app
    with pytest.raises(gr.Error):
        app.run_assessment(None, "", "en", progress=lambda *a, **k: None)


# ------------------------------------------------------------ bundled assets (relative paths)
def test_bundled_assets_resolve_from_repo_root():
    root_assets = Path(vf.__file__).resolve().parent.parent / "assets"
    assert vf.find_asset("logo_templates.npz") == root_assets / "logo_templates.npz"
    onnx = vf.find_asset(vf.YUNET_FILE)
    assert onnx == root_assets / vf.YUNET_FILE and 150_000 < onnx.stat().st_size < 400_000


def test_face_detector_uses_bundled_onnx_without_download(monkeypatch):
    import urllib.request

    def no_network(*a, **k):
        raise AssertionError("must not download when the model is bundled")
    monkeypatch.setattr(urllib.request, "urlretrieve", no_network)
    d = vf.FaceDetector()
    assert d.kind == "yunet"
    assert d.detect(np.zeros((180, 320, 3), np.uint8)) == []


# ------------------------------------------------------------ streamlit app helpers
def test_streamlit_helpers(tmp_path, monkeypatch):
    import io
    import streamlit_app as sa
    assert sa.TITLE == "Online Class Quality Assessor"
    assert sa.is_supported_link("https://drive.google.com/file/d/x/view") and not sa.is_supported_link("notaurl")
    assert sa.LANGUAGES == {"English": "en", "Malayalam": "ml"}

    class Upload(io.BytesIO):
        name = "lesson.MP4"
    dest = sa.save_upload(Upload(b"x" * (3 * sa.CHUNK + 5)), tmp_path)
    assert dest.name == "upload.mp4" and dest.stat().st_size == 3 * sa.CHUNK + 5

    class Bad(io.BytesIO):
        name = "notes.txt"
    with pytest.raises(ValueError):
        sa.save_upload(Bad(b"x"), tmp_path)

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    assert sa.configure_gemini_key() is True
    sa.render_results(sample_result())  # renders in Streamlit "bare mode" without errors


def test_telemetry_keys_have_no_brand():
    import json
    from src.ui_render import telemetry_payload
    r = sample_result()
    brand = f"{8}x{8}"  # built at runtime so the brand string never appears in the repo
    r["telemetry"]["visual"][f"bg_{brand}_frame_pct"] = 40.0
    assert brand not in json.dumps(telemetry_payload(r))
