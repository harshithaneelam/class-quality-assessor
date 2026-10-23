"""Hybrid class-quality scorer: deterministic 5-dimension rubric + gates + optional LLM.

Pipeline (`assess_class_recording`):
  1. refine camera-off with speech timing (camera off while the tutor talks counts;
     camera off during a silent puzzle/break window does not);
  2. deterministic 0-100 scores for 5 dimensions from visual / audio / transcript features;
  3. composite of the deterministic scores -> label via cutoffs (fit on labelled data,
     see `fit_cutoffs`). The Gemini judge supplies the executive summary and timestamped
     evidence (and its agreement feeds confidence) but does not move the label: on the
     10 labelled classes deterministic LOO was 8/10 vs 7/10 for the 0.45/0.55 fusion.
     `label_source="fused"` keeps the fused path for comparison;
  4. (fused path only) 0.45 det + 0.55 LLM on the four pedagogical dimensions;
  5. interpretable gates override the band: Poor (compliance / pedagogical), caps at
     Average (dark face, background clarity) and Good (Excellent prerequisites);
  6. confidence and timestamped strengths / improvements (LLM + deterministic).

The gate rules were written after looking at all 10 labelled classes, so the
leave-one-out numbers in step 06 (which only refit the cutoffs) are optimistic.
"""
from __future__ import annotations

import re
from itertools import combinations

import numpy as np

LABELS = ("Poor", "Average", "Good", "Excellent")
DIMENSIONS = ("visual_compliance", "intro_onboarding", "individual_attention", "energy_rapport", "structure_homework")
LLM_DIM_KEYS = {
    "visual_compliance": "visual_and_brand_compliance_0_100",
    "intro_onboarding": "intro_and_onboarding_0_100",
    "individual_attention": "individual_attention_0_100",
    "energy_rapport": "communication_energy_and_rapport_0_100",
    "structure_homework": "lesson_structure_and_homework_0_100",
}
W_DET, W_LLM = 0.45, 0.55
DET_ONLY_DIMS = ("visual_compliance",)
DEFAULT_CUTOFFS = (40.0, 55.0, 70.0)  # Poor | Average | Good | Excellent on the composite

G = dict(  # gate thresholds
    dark_face_lum=55.0, speech_cam_off_streak=60.0, near_zero_praise=0.1, low_coverage_students=3,
    exc_students=5, exc_questions=60, exc_dead_air=600.0, exc_praise=5.0,
    code_mix_logprob=-0.50,
    asr_drop_max_students=2, asr_drop_min_questions=60, asr_drop_effective_students=4,
)


def _ts(sec: float) -> str:
    sec = max(0, int(sec))
    return f"[{sec // 60:02d}:{sec % 60:02d}]"


def _lead_ts(text: str | None) -> str | None:
    """Leading "[MM:SS]" of an evidence string (the text itself is never used)."""
    m = re.match(r"\[\d+:\d{2}\]", text or "")
    return m.group(0) if m else None


# ===================================================== 1. camera-off refinement
def refine_camera_off(camera_off_times: list[float], sample_every_sec: float, segments: list[dict],
                      sampled_frames: int) -> dict:
    """Camera-off samples that overlap speech (a Whisper segment within +/- half a step)."""
    half = sample_every_sec / 2
    spans = sorted((s["start"], s["end"]) for s in segments)
    starts = np.array([a for a, _ in spans]) if spans else np.array([])
    ends = np.array([b for _, b in spans]) if spans else np.array([])

    def speaking(t):
        if not spans:
            return False
        return bool(np.any((starts < t + half) & (ends > t - half)))

    speech_off = [t for t in sorted(camera_off_times) if speaking(t)]
    best = run = 0
    prev = None
    for t in speech_off:
        run = run + 1 if (prev is not None and abs(t - prev - sample_every_sec) < 1e-6) else 1
        best = max(best, run)
        prev = t
    silent_off = len(camera_off_times) - len(speech_off)
    return {
        "camera_off_during_speech_pct": round(100.0 * len(speech_off) / sampled_frames, 1) if sampled_frames else 0.0,
        "max_camera_off_during_speech_streak_sec": float(best * sample_every_sec),
        "camera_off_silent_samples": silent_off,
        "camera_off_during_speech_ts": [_ts(t) for t in speech_off[:3]],
    }


# ================================================ 2. deterministic dimensions
def _clip(x: float) -> float:
    return float(min(100.0, max(0.0, x)))


def gate_inputs(inp: dict) -> dict:
    """Flatten the features the gates and scores use (with safe defaults)."""
    v, a, t = inp.get("visual", {}), inp.get("audio", {}), inp.get("transcript", {})
    cam = inp.get("camera_refined") or refine_camera_off(
        v.get("camera_off_times_sec", []), v.get("sample_every_sec", 10.0), inp.get("segments", []),
        v.get("sampled_frames", 0))
    dur = float(inp.get("duration_sec") or 1.0)
    lum = float(v.get("tutor_face_luminance_mean") or 0.0)
    dark = 0 < lum < G["dark_face_lum"]
    students = int(t.get("distinct_students_called") or 0)
    questions = int(t.get("teacher_question_count") or 0)
    cam_flag = bool(v.get("flag_intermittent_camera_off"))
    # Whisper often drops Indian student names, so few distinct [STUDENT_k] can appear even
    # when the coach questions students one by one. Many questions + camera on -> treat as
    # ASR name drop (avg_1: 1 student detected, 75 questions; reviewers noted 1-to-1
    # questioning) and use a moderate effective coverage instead of firing P2.
    asr_drop = (students <= G["asr_drop_max_students"] and questions >= G["asr_drop_min_questions"]
                and not cam_flag)
    name_masks = sum(s.get("text", "").count("[NAME]") for s in inp.get("segments", []))
    return {
        "no_8x8": bool(v.get("flag_no_8x8_background")),
        # a dark face is often not *detected*; that is a lighting issue, not a hidden face
        "face_not_visible": bool(v.get("flag_face_not_visible")) and not dark,
        "dark_face": dark, "face_lum": lum,
        "bg_clarity": bool(v.get("flag_bg_clarity_issue")),
        "cam_off_speech_pct": cam["camera_off_during_speech_pct"],
        "cam_off_speech_streak": cam["max_camera_off_during_speech_streak_sec"],
        "warm_intro": bool(t.get("has_warm_intro")), "intro_score": float(t.get("intro_score") or 0.0),
        "self_intro": bool(t.get("has_self_intro_or_new_joiner_time")),
        "jumped": bool(t.get("jumped_straight_to_revision")),
        "students": students,
        "students_eff": max(students, G["asr_drop_effective_students"]) if asr_drop else students,
        "asr_name_drop": asr_drop, "name_masks": name_masks, "cam_off_flag": cam_flag,
        "indiv_q": int(t.get("individual_questions_count") or 0),
        "questions": questions,
        "praise10": float(t.get("praise_per_10min") or 0.0),
        "homework": bool(t.get("has_homework")), "conclusion": bool(t.get("has_proper_conclusion")),
        "dead_air": float(a.get("dead_air_total_sec") or 0.0), "dead_air_frac": float(a.get("dead_air_total_sec") or 0.0) / dur,
        "max_gap": float(a.get("max_silence_gap_sec") or 0.0),
        "speech_ratio": float(a.get("speech_ratio_pct") or 0.0),
        "liveliness": float(a.get("energy_liveliness_score") or 0.0),
        "camera": cam,
    }


def deterministic_scores(g: dict) -> dict:
    vis = 100.0
    vis -= 45 if g["no_8x8"] else 0
    vis -= 45 if g["face_not_visible"] else 0
    vis -= min(30.0, 1.5 * g["cam_off_speech_pct"])
    vis -= 15 if g["cam_off_speech_streak"] >= 120 else 0
    vis -= 25 if g["dark_face"] else 0
    vis -= 15 if g["bg_clarity"] else 0
    intro = 45 * g["intro_score"] + 30 * g["self_intro"] + 25 * (not g["jumped"])
    indiv = 45 * min(g["students_eff"], 8) / 8 + 25 * min(g["indiv_q"], 20) / 20 + 30 * min(g["questions"], 100) / 100
    energy = (35 * min(g["praise10"], 6.0) / 6 + 20 * g["liveliness"]
              + 30 * (1 - min(g["dead_air_frac"] / 0.4, 1.0)) + 15 * g["speech_ratio"] / 100)
    struct = 45 * g["homework"] + 35 * g["conclusion"] + 20 * (1 - min(g["max_gap"] / 300, 1.0))
    return {"visual_compliance": _clip(vis), "intro_onboarding": _clip(intro), "individual_attention": _clip(indiv),
            "energy_rapport": _clip(energy), "structure_homework": _clip(struct)}


# ======================================================================= gates
def evaluate_gates(g: dict) -> dict:
    compliance = g["no_8x8"] or g["face_not_visible"]
    ped_a = g["jumped"] and (g["cam_off_speech_streak"] >= G["speech_cam_off_streak"] or g["praise10"] < G["near_zero_praise"])
    ped_b = (not g["self_intro"]) and (g["students_eff"] <= G["low_coverage_students"] or (not g["warm_intro"] and not g["conclusion"]))
    pedagogical = ped_a or ped_b
    cap_avg = g["dark_face"] or g["bg_clarity"]
    exc_ok = (g["students"] >= G["exc_students"] and g["questions"] >= G["exc_questions"]
              and g["dead_air"] < G["exc_dead_air"] and (g["homework"] or g["praise10"] >= G["exc_praise"])
              and g["warm_intro"] and not g["jumped"])
    codes = []
    if g["no_8x8"]:
        codes.append("C1_no_branded_bg")
    if g["face_not_visible"]:
        codes.append("C2_face_hidden")
    if ped_a:
        codes.append("P1_jump_no_rapport")
    if ped_b:
        codes.append("P2_no_onboarding_low_coverage")
    if g["dark_face"]:
        codes.append("A1_dark_face")
    if g["bg_clarity"]:
        codes.append("A2_bg_clarity")
    if not exc_ok:
        codes.append("E0_excellent_prereq_unmet")
    sub = ("both" if compliance and pedagogical else "compliance_quality_issue" if compliance
           else "pedagogical_weakness" if pedagogical else "none")
    return {"poor": compliance or pedagogical, "poor_sub_category": sub, "cap_average": cap_avg,
            "excellent_ok": exc_ok, "codes": codes}


def label_from(composite: float, gates: dict, cutoffs=DEFAULT_CUTOFFS) -> str:
    if gates["poor"]:
        return "Poor"
    idx = int(np.searchsorted(np.asarray(cutoffs), composite, side="right"))
    if gates["cap_average"]:
        idx = min(idx, 1)
    if not gates["excellent_ok"]:
        idx = min(idx, 2)
    return LABELS[idx]


def fit_cutoffs(samples: list[dict]) -> tuple[float, float, float]:
    """Choose (c1, c2, c3) maximising training accuracy of the final (gated) label.

    samples: [{"composite": float, "gates": dict, "truth": str}]. Candidate cut points
    are midpoints between sorted training composites; ties prefer the widest margin.
    """
    comps = sorted({round(s["composite"], 3) for s in samples})
    cands = [0.0] + [(a + b) / 2 for a, b in zip(comps, comps[1:])] + [100.0]
    best, best_key = DEFAULT_CUTOFFS, (-1, -1.0)
    for c in combinations(cands, 3):
        acc = sum(label_from(s["composite"], s["gates"], c) == s["truth"] for s in samples)
        margin = min(abs(s["composite"] - x) for s in samples for x in c)
        key = (acc, margin)
        if key > best_key:
            best, best_key = c, key
    return tuple(round(x, 2) for x in best)


# =============================================================== observations
def deterministic_observations(inp: dict, g: dict) -> tuple[list[dict], list[dict]]:
    t, v, a = inp.get("transcript", {}), inp.get("visual", {}), inp.get("audio", {})
    ev = v.get("visual_evidence_timestamps", {})
    dur = float(inp.get("duration_sec") or 0.0)
    start = t.get("first_speech_ts") or "[00:00]"
    S, I = [], []

    def add(lst, ts, dim, text):
        if ts:
            lst.append({"timestamp": ts, "dimension": dim, "observation": text, "source": "deterministic"})

    if g["warm_intro"]:
        add(S, t.get("intro_evidence_ts"), "intro_onboarding", "Opened with a warm greeting and rapport building.")
    else:
        add(I, start, "intro_onboarding", "Open with a warmer greeting and a short rapport-building chat before teaching.")
    if g["self_intro"]:
        add(S, t.get("self_intro_evidence_ts"), "intro_onboarding", "Gave time for introductions / new joiners.")
    else:
        add(I, start, "intro_onboarding", "Give new joiners time to introduce themselves at the start of class.")
    if g["jumped"]:
        add(I, t.get("early_revision_ts"), "intro_onboarding",
            "Went straight into revision of the previous class; build rapport first.")
    exs = t.get("individual_question_examples") or []
    if g["students"] >= 5 and exs:
        add(S, _lead_ts(exs[0]), "individual_attention",
            f"Addressed {g['students']} different students individually ({g['indiv_q']} direct questions).")
    elif g["students"] <= 3:
        add(I, (_lead_ts(exs[0]) if exs else None) or start, "individual_attention",
            "Call on each student individually by name, especially during puzzle solving.")
    if g["homework"]:
        add(S, t.get("homework_evidence_ts"), "structure_homework", "Homework / practice was assigned.")
    elif dur:
        add(I, _ts(max(0.0, dur - 300)), "structure_homework", "Assign homework before closing the class.")
    if g["conclusion"]:
        add(S, t.get("conclusion_evidence_ts"), "structure_homework", "Closed the class with a proper wrap-up.")
    elif dur:
        add(I, _ts(max(0.0, dur - 120)), "structure_homework", "End with a clear wrap-up and goodbye.")
    if g["praise10"] >= 3:
        add(S, start, "energy_rapport", f"Frequent praise and encouragement ({g['praise10']:.1f} per 10 min).")
    elif g["praise10"] < 1:
        add(I, start, "energy_rapport", "Praise and encourage students more often; keep the tone energetic and pleasant.")
    if g["max_gap"] >= 60 and a.get("max_silence_gap_ts"):
        add(I, a["max_silence_gap_ts"].split("-")[0], "energy_rapport",
            f"Long silence of {g['max_gap']:.0f} s; keep students engaged during pauses.")
    if g["no_8x8"]:
        add(I, (ev.get("bg_8x8_missing_samples") or [None])[0], "visual_compliance", "Keep the branded virtual background on.")
    if g["face_not_visible"]:
        add(I, (ev.get("face_missing_on_8x8_samples") or [None])[0], "visual_compliance",
            "Keep your face clearly visible on camera.")
    if g["cam_off_speech_streak"] >= 60:
        add(I, (g["camera"]["camera_off_during_speech_ts"] or [None])[0], "visual_compliance",
            "Camera was off while teaching; keep it on throughout the class.")
    if g["dark_face"]:
        add(I, (ev.get("low_face_lighting_samples") or [None])[0], "visual_compliance",
            "Improve lighting on your face (face too dark on camera).")
    if not (g["no_8x8"] or g["face_not_visible"] or g["dark_face"]) and v.get("bg_8x8_when_cam_active_pct", 0) >= 50:
        add(S, start, "visual_compliance", "Kept the branded virtual background with the face visible.")
    return S, I


def normalize_ts(ts: str | None) -> str | None:
    """'12:34', '[12:34]', '[12:34]-[13:00]' -> '[12:34]'; anything else -> None."""
    m = re.search(r"(\d{1,3}):(\d{2})", ts or "")
    return f"[{int(m.group(1)):02d}:{m.group(2)}]" if m else None


def _merge_obs(llm_obs: list[dict], det_obs: list[dict], cap: int = 5) -> list[dict]:
    """LLM observations first (only those citing a real timestamp), then deterministic
    ones for dimensions the LLM did not cover."""
    out = [{**o, "timestamp": normalize_ts(o.get("timestamp")), "source": "llm"} for o in llm_obs]
    out = [o for o in out if o["timestamp"]]
    have = {o["dimension"] for o in out}
    for o in det_obs:
        if len(out) >= cap:
            break
        if o["dimension"] not in have or not out:
            out.append(o)
            have.add(o["dimension"])
    return out[:cap]


# ===================================================================== fusion
def assess_class_recording(inp: dict, llm_result: dict | None = None, cutoffs=DEFAULT_CUTOFFS,
                           label_source: str = "deterministic") -> dict:
    """inp: {"visual", "audio", "transcript", "segments", "duration_sec", "asr"} -> assessment."""
    g = gate_inputs(inp)
    det = deterministic_scores(g)
    gates = evaluate_gates(g)
    llm_a = (llm_result or {}).get("assessment")
    llm_scores = {d: float(llm_a["pedagogical_scores"][k]) for d, k in LLM_DIM_KEYS.items()} if llm_a else None
    # The LLM never sees video — only our measured visual percentages — and on the 10
    # labelled classes it scored visual compliance 0-40 for every class (it reads normal
    # detector misses as non-compliance). Visual stays deterministic; the 4 pedagogical
    # dimensions are fused.
    fused = {d: round(W_DET * det[d] + W_LLM * llm_scores[d], 1) if (llm_scores and d not in DET_ONLY_DIMS)
             else round(det[d], 1) for d in DIMENSIONS}
    det_comp = float(np.mean(list(det.values())))
    fused_comp = float(np.mean(list(fused.values())))
    use_fused = label_source == "fused" and llm_scores is not None
    shown = fused if use_fused else {k: round(v, 1) for k, v in det.items()}
    composite = round(fused_comp if use_fused else det_comp, 1)
    label = label_from(composite, gates, cutoffs)

    logprob = (inp.get("asr") or {}).get("avg_logprob_mean")
    lang_warn = logprob is not None and logprob < G["code_mix_logprob"]
    conf = 72.0
    if llm_scores:
        ped = [d for d in DIMENSIONS if d not in DET_ONLY_DIMS]  # LLM visual scores are not trusted
        diff = abs(float(np.mean([det[d] for d in ped])) - float(np.mean([llm_scores[d] for d in ped])))
        conf += 12 if diff < 10 else 4 if diff < 20 else -10
        conf += 8 if llm_a["llm_raw_label"] == label else -8
    else:
        conf -= 5
    margin = min(abs(composite - c) for c in cutoffs)
    conf += min(10.0, margin / 2)
    if gates["poor"] or gates["cap_average"]:
        conf += 5
    if lang_warn:
        conf -= 20
    conf = float(min(98.0, max(50.0, conf)))

    det_s, det_i = deterministic_observations(inp, g)
    llm_s, llm_i, dropped = [], [], 0
    if llm_a:
        from src.llm_judge import filter_contradicting_observations
        vis = inp.get("visual", {})
        llm_s, d1 = filter_contradicting_observations(llm_a["strengths"], vis)
        llm_i, d2 = filter_contradicting_observations(llm_a["improvements"], vis)
        dropped = d1 + d2
    strengths = _merge_obs(llm_s, det_s)
    improvements = _merge_obs(llm_i, det_i)
    from src.llm_judge import scrub_brand
    summary = scrub_brand(llm_a["executive_summary"]) if llm_a else (
        f"Deterministic assessment: {label}. Gates: {', '.join(c for c in gates['codes'] if not c.startswith('E0')) or 'none'}.")
    return {
        "label": label,
        "poor_sub_category": gates["poor_sub_category"] if label == "Poor" else "none",
        "confidence_pct": round(conf, 1),
        "composite": composite, "deterministic_composite": round(det_comp, 1), "fused_composite": round(fused_comp, 1),
        "label_source": "fused" if use_fused else "deterministic",
        "sub_scores": shown, "deterministic_scores": {k: round(v, 1) for k, v in det.items()}, "fused_scores": fused,
        "llm_scores": llm_scores, "llm_raw_label": llm_a["llm_raw_label"] if llm_a else None,
        "triggered_gates": gates["codes"],
        "camera_refined": g["camera"],
        "compliance_flags": {k: g[k] for k in ("no_8x8", "face_not_visible", "dark_face", "bg_clarity")} | {
            "camera_off_while_teaching": g["cam_off_speech_streak"] >= 60},
        "language_warning": ("Low ASR confidence — likely code-mixed / non-English speech; transcript-based "
                             "signals are less reliable.") if lang_warn else None,
        "executive_summary": summary,
        "strengths": strengths, "improvements": improvements,
        "llm_status": (llm_result or {}).get("llm_status", "fallback_deterministic") if llm_a else "fallback_deterministic",
        "llm_observations_dropped_as_contradicting": dropped,
        "asr_name_drop_assumed": g["asr_name_drop"],
        "cutoffs": list(cutoffs),
    }
