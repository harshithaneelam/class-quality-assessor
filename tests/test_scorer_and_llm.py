"""Scorer + LLM judge tests — fully offline (mock Gemini client, synthetic features)."""
import json
from types import SimpleNamespace

import pytest

from src import llm_judge as lj
from src import scorer as sc


# ------------------------------------------------------------------ fixtures
def llm_payload(label="Good", score=70):
    return {
        "pedagogical_scores": {k: score for k in (
            "intro_and_onboarding_0_100", "individual_attention_0_100", "communication_energy_and_rapport_0_100",
            "lesson_structure_and_homework_0_100", "visual_and_brand_compliance_0_100")},
        "llm_raw_label": label, "poor_sub_category": "none",
        "executive_summary": "Solid class with good coverage.",
        "strengths": [{"timestamp": "[01:00]", "dimension": "intro", "observation": "Warm welcome."},
                      {"timestamp": "[10:00]", "dimension": "attention", "observation": "Called on [STUDENT_1]."}],
        "improvements": [{"timestamp": "[40:00]", "dimension": "homework", "observation": "Assign homework."},
                         {"timestamp": "[42:00]", "dimension": "energy", "observation": "More praise."}],
    }


class FakeAPIError(Exception):
    def __init__(self, code):
        super().__init__(f"{code} error")
        self.code = code


class FakeModels:
    def __init__(self, script):
        self.script = list(script)  # per call: Exception instance or payload dict
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        parsed = lj.LLMClassAssessment.model_validate(item)
        return SimpleNamespace(parsed=parsed, text=json.dumps(item), model_version=model + "-001",
                               usage_metadata=SimpleNamespace(prompt_token_count=5000, candidates_token_count=600,
                                                              thoughts_token_count=300, total_token_count=5900))


def fake_client(script):
    return SimpleNamespace(models=FakeModels(script))


SEGS = [{"start": 30.0 * i, "end": 30.0 * i + 25, "text": f"[TEACHER] explains move {i}. [STUDENT_1] can you try?",
         "speaker_role": "TEACHER" if i % 3 else "STUDENT"} for i in range(100)]


def base_inputs(**over):
    inp = {
        "duration_sec": 3000.0,
        "segments": SEGS,
        "visual": {"flag_no_8x8_background": False, "flag_face_not_visible": False, "flag_bg_clarity_issue": False,
                   "tutor_face_luminance_mean": 110.0, "camera_off_times_sec": [], "sample_every_sec": 10.0,
                   "sampled_frames": 300, "bg_8x8_when_cam_active_pct": 80.0,
                   "visual_evidence_timestamps": {"bg_8x8_missing_samples": ["[05:00]"],
                                                  "face_missing_on_8x8_samples": ["[06:00]"],
                                                  "low_face_lighting_samples": ["[07:00]"]}},
        "audio": {"dead_air_total_sec": 120.0, "max_silence_gap_sec": 20.0, "max_silence_gap_ts": "[10:00]-[10:20]",
                  "speech_ratio_pct": 85.0, "energy_liveliness_score": 0.8},
        "transcript": {"has_warm_intro": True, "intro_score": 1.0, "intro_evidence_ts": "[00:20]",
                       "has_self_intro_or_new_joiner_time": True, "self_intro_evidence_ts": "[01:00]",
                       "jumped_straight_to_revision": False, "distinct_students_called": 8,
                       "individual_questions_count": 15, "teacher_question_count": 100,
                       "individual_question_examples": ["[12:00] [TEACHER] asked [STUDENT_2]: ..."],
                       "praise_per_10min": 4.0, "has_homework": True, "homework_evidence_ts": "[47:00]",
                       "has_proper_conclusion": True, "conclusion_evidence_ts": "[49:00]", "first_speech_ts": "[00:20]"},
        "asr": {"avg_logprob_mean": -0.25},
    }
    for path, val in over.items():
        sect, key = path.split("__")
        inp[sect][key] = val
    return inp


# ------------------------------------------------------------------ schema
def test_schema_validation_and_normalisation():
    p = llm_payload()
    p["pedagogical_scores"]["individual_attention_0_100"] = 140
    p["strengths"] = p["strengths"] * 3  # 6 items
    d = lj.normalize_assessment(lj.LLMClassAssessment.model_validate(p))
    assert d["pedagogical_scores"]["individual_attention_0_100"] == 100 and len(d["strengths"]) == 4
    bad = llm_payload()
    bad["llm_raw_label"] = "Superb"
    with pytest.raises(Exception):
        lj.LLMClassAssessment.model_validate(bad)


# ------------------------------------------------------------------ privacy gate
def test_preflight_blocks_names_paths_and_binary():
    lj.preflight_privacy_check("[00:10] (T) hello [STUDENT_1]", {"zorblat"})
    with pytest.raises(lj.PrivacyViolation) as e:
        lj.preflight_privacy_check("[00:10] (T) hello Zorblat", {"zorblat"})
    assert "zorblat" not in str(e.value).lower()
    with pytest.raises(lj.PrivacyViolation):
        lj.preflight_privacy_check(r"see C:\challenge-data\x.mp4", set())
    with pytest.raises(lj.PrivacyViolation):
        lj.preflight_privacy_check("A" * 200, set())


def test_privacy_violation_raised_before_any_network_call(tmp_path):
    client = fake_client([llm_payload()])
    with pytest.raises(lj.PrivacyViolation):
        lj.call_gemini_judge("x", "teacher Zorblat said", "blk", {"zorblat"}, cache_dir=tmp_path, client=client)
    assert client.models.calls == []


# ------------------------------------------------------------------ retry / fallback / cache
def test_429_backoff_then_model_fallback_then_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_PRIMARY_MODEL", "m-primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "m-lite")
    sleeps = []
    script = [FakeAPIError(429)] * 4 + [FakeAPIError(503), llm_payload("Excellent", 90)]
    client = fake_client(script)
    r = lj.call_gemini_judge("id1", "prompt", "block", set(), cache_dir=tmp_path, client=client, sleep=sleeps.append)
    assert client.models.calls == ["m-primary"] * 4 + ["m-lite"] * 2
    assert [s for s in sleeps if s >= 2] == [2, 5, 10, 2]
    assert r["assessment"]["llm_raw_label"] == "Excellent" and r["model"] == "m-lite-001"
    assert r["usage"]["prompt_token_count"] == 5000 and not r["cache_hit"]
    again = lj.call_gemini_judge("id1", "prompt", "block", set(), cache_dir=tmp_path, client=fake_client([]))
    assert again["cache_hit"] and again["assessment"]["llm_raw_label"] == "Excellent"


def test_all_models_fail_returns_none(tmp_path):
    client = fake_client([FakeAPIError(400)] * 4)
    assert lj.call_gemini_judge("id2", "p", "b", set(), cache_dir=tmp_path, client=client, sleep=lambda s: None) is None
    assert len(client.models.calls) == 4


def test_missing_api_key_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(lj, "api_key_status", lambda: (None, "missing"))
    assert lj.call_gemini_judge("id3", "p", "b", set(), cache_dir=tmp_path) is None


def test_cache_key_depends_on_transcript_and_version():
    k = lj.cache_key("a", "block")
    assert k == lj.cache_key("a", "block") and k != lj.cache_key("a", "block2")
    assert k != lj.cache_key("a", "block", prompt_version="judge-v99")


# ------------------------------------------------------------------ prompt
def test_prompt_compact_merge_and_cap():
    segs = [{"start": 0.0, "end": 1.0, "text": "hello", "speaker_role": "TEACHER"},
            {"start": 1.5, "end": 3.0, "text": "everyone", "speaker_role": "TEACHER"},
            {"start": 3.2, "end": 4.0, "text": "hi", "speaker_role": "STUDENT"}]
    block, info = lj.compact_transcript(segs, 10.0)
    assert block.splitlines() == ["[00:00] (T) hello everyone", "[00:03] (S) hi"]
    long = [{"start": 10.0 * i, "end": 10.0 * i + 9, "text": "word " * 60, "speaker_role": "TEACHER"} for i in range(400)]
    block, info = lj.compact_transcript(long, 4000.0, max_words=5500)
    assert info["sampled"] and info["words_sent"] <= 5500 and "[...]" in block
    prompt, _, _ = lj.build_assessment_prompt("x", segs, 10.0, {"flag": True})
    lj.preflight_privacy_check(prompt, {"zorblat"})


# ------------------------------------------------------------------ camera-off refinement
def test_camera_off_during_speech_vs_silent_break():
    segs = [{"start": 0.0, "end": 100.0}, {"start": 600.0, "end": 700.0}]
    silent = sc.refine_camera_off([300.0, 310.0, 320.0, 330.0], 10.0, segs, 100)
    assert silent["camera_off_during_speech_pct"] == 0.0 and silent["camera_off_silent_samples"] == 4
    teaching = sc.refine_camera_off([600.0, 610.0, 620.0, 630.0, 640.0, 650.0, 660.0], 10.0, segs, 100)
    assert teaching["camera_off_during_speech_pct"] == 7.0
    assert teaching["max_camera_off_during_speech_streak_sec"] == 70.0


# ------------------------------------------------------------------ gates & labels
def test_excellent_offline_deterministic():
    r = sc.assess_class_recording(base_inputs())
    assert r["label"] == "Excellent" and r["llm_status"] == "fallback_deterministic"
    assert 50 <= r["confidence_pct"] <= 98 and len(r["strengths"]) >= 2


def test_compliance_gate_poor():
    r = sc.assess_class_recording(base_inputs(visual__flag_no_8x8_background=True))
    assert r["label"] == "Poor" and r["poor_sub_category"] == "compliance_quality_issue"
    assert "C1_no_branded_bg" in r["triggered_gates"]
    assert any(o["timestamp"] == "[05:00]" for o in r["improvements"])


def test_dark_face_is_lighting_not_hidden_face_and_caps_average():
    r = sc.assess_class_recording(base_inputs(visual__flag_face_not_visible=True, visual__tutor_face_luminance_mean=25.0))
    assert r["label"] == "Average" and "A1_dark_face" in r["triggered_gates"]
    assert "C2_face_hidden" not in r["triggered_gates"]


def test_pedagogical_gates():
    r = sc.assess_class_recording(base_inputs(transcript__jumped_straight_to_revision=True, transcript__praise_per_10min=0.0,
                                              transcript__early_revision_ts="[00:30]"))
    assert r["label"] == "Poor" and r["poor_sub_category"] == "pedagogical_weakness"
    r2 = sc.assess_class_recording(base_inputs(transcript__has_self_intro_or_new_joiner_time=False,
                                               transcript__distinct_students_called=2,
                                               transcript__teacher_question_count=40))
    assert r2["label"] == "Poor" and "P2_no_onboarding_low_coverage" in r2["triggered_gates"]


def test_missing_homework_without_high_praise_caps_good():
    r = sc.assess_class_recording(base_inputs(transcript__has_homework=False))
    assert r["label"] == "Good" and "E0_excellent_prereq_unmet" in r["triggered_gates"]


def test_label_is_deterministic_llm_supplies_text():
    llm = {"assessment": lj.normalize_assessment(lj.LLMClassAssessment.model_validate(llm_payload("Poor", 10))),
           "llm_status": "ok"}
    det = sc.assess_class_recording(base_inputs())
    hyb = sc.assess_class_recording(base_inputs(), llm)
    assert hyb["label"] == det["label"] == "Excellent" and hyb["composite"] == det["composite"]
    assert hyb["sub_scores"] == det["deterministic_scores"] and hyb["label_source"] == "deterministic"
    assert hyb["executive_summary"] == "Solid class with good coverage." and hyb["strengths"][0]["source"] == "llm"
    assert hyb["llm_raw_label"] == "Poor" and hyb["confidence_pct"] < det["confidence_pct"] + 5  # disagreement noted


def test_fused_path_still_available_for_comparison():
    llm = {"assessment": lj.normalize_assessment(lj.LLMClassAssessment.model_validate(llm_payload("Good", 50))),
           "llm_status": "ok"}
    det = sc.assess_class_recording(base_inputs())
    fused = sc.assess_class_recording(base_inputs(), llm, label_source="fused")
    assert fused["composite"] < det["composite"] and fused["label_source"] == "fused"
    assert fused["sub_scores"]["visual_compliance"] == det["deterministic_scores"]["visual_compliance"]


def test_code_mix_language_warning():
    r = sc.assess_class_recording(base_inputs(asr__avg_logprob_mean=-0.6))
    assert r["language_warning"] and r["confidence_pct"] < sc.assess_class_recording(base_inputs())["confidence_pct"]


def test_fit_cutoffs_learns_bands():
    ok = {"poor": False, "cap_average": False, "excellent_ok": True}
    samples = [{"composite": c, "gates": ok, "truth": t} for c, t in
               [(20, "Poor"), (30, "Poor"), (45, "Average"), (50, "Average"), (62, "Good"), (66, "Good"),
                (80, "Excellent"), (85, "Excellent")]]
    c = sc.fit_cutoffs(samples)
    assert all(sc.label_from(s["composite"], ok, c) == s["truth"] for s in samples)


# ------------------------------------------------------------------ v2
def test_v2_visual_verdicts_hide_raw_numbers_when_passing():
    v = base_inputs()["visual"]
    out = lj.visual_verdicts(v)
    assert out["branded_virtual_background"].startswith("PASS") and out["camera_continuity"].startswith("PASS")
    assert "missing_samples" not in json.dumps(out)
    v_fail = {**v, "flag_no_8x8_background": True, "bg_8x8_when_cam_active_pct": 20.0}
    f = lj.visual_verdicts(v_fail)["branded_virtual_background"]
    assert f["verdict"].startswith("FAIL") and f["missing_samples"] == ["[05:00]"]


def test_llm_bullets_contradicting_pass_flags_are_dropped():
    p = llm_payload("Good", 60)
    p["improvements"] = [{"timestamp": "[10:00]", "dimension": "visual", "observation": "The camera was off for a while."},
                         {"timestamp": "[11:00]", "dimension": "visual", "observation": "Missing branded virtual background."},
                         {"timestamp": "[40:00]", "dimension": "homework", "observation": "Assign homework."}]
    llm = {"assessment": lj.normalize_assessment(lj.LLMClassAssessment.model_validate(p)), "llm_status": "ok"}
    r = sc.assess_class_recording(base_inputs(), llm)
    texts = " ".join(o["observation"] for o in r["improvements"])
    assert "camera was off" not in texts and "Missing branded-background" not in texts
    assert r["llm_observations_dropped_as_contradicting"] == 2


def test_asr_name_drop_does_not_fire_p2():
    common = dict(transcript__has_self_intro_or_new_joiner_time=False, transcript__distinct_students_called=1)
    dropped = sc.assess_class_recording(base_inputs(**common, transcript__teacher_question_count=75))
    assert dropped["asr_name_drop_assumed"] and "P2_no_onboarding_low_coverage" not in dropped["triggered_gates"]
    few_q = sc.assess_class_recording(base_inputs(**common, transcript__teacher_question_count=44))
    assert few_q["label"] == "Poor" and "P2_no_onboarding_low_coverage" in few_q["triggered_gates"]
    cam_off = base_inputs(**common, transcript__teacher_question_count=75)
    cam_off["visual"]["flag_intermittent_camera_off"] = True
    assert "P2_no_onboarding_low_coverage" in sc.assess_class_recording(cam_off)["triggered_gates"]


def test_llm_timestamps_normalised_and_untimed_dropped():
    assert sc.normalize_ts("12:34") == "[12:34]" and sc.normalize_ts("[05:07]-[06:00]") == "[05:07]"
    assert sc.normalize_ts("PASS") is None and sc.normalize_ts(None) is None
    p = llm_payload()
    p["strengths"] = [{"timestamp": "3:05", "dimension": "intro", "observation": "Warm welcome."},
                      {"timestamp": "N/A", "dimension": "attention", "observation": "General remark."}]
    llm = {"assessment": lj.normalize_assessment(lj.LLMClassAssessment.model_validate(p)), "llm_status": "ok"}
    r = sc.assess_class_recording(base_inputs(), llm)
    llm_obs = [o for o in r["strengths"] if o["source"] == "llm"]
    assert [o["timestamp"] for o in llm_obs] == ["[03:05]"]


def test_brand_name_scrubbed_from_llm_text_and_filtered():
    brand = f"{8}x{8}"  # built at runtime so the brand string never appears in the repo
    assert brand not in lj.scrub_brand(f"Kept the {brand} background on.")
    kept, dropped = lj.filter_contradicting_observations(
        [{"timestamp": "[01:00]", "dimension": "visual", "observation": f"Did not keep the {brand} background."},
         {"timestamp": "[02:00]", "dimension": "rapport", "observation": f"Great energy, a true {brand} coach."}], {})
    assert dropped == 1 and brand not in kept[0]["observation"]
