"""Framework-free HTML renderers shared by the Gradio (app.py) and Streamlit apps.

All model-written text is HTML-escaped. Telemetry keys are renamed so the client's
brand never appears in the UI.
"""
from __future__ import annotations

import html
import re

TITLE = "Online Class Quality Assessor"

CSS = """
.ca-card{border:1px solid var(--border-color-primary, rgba(128,128,128,.35));border-radius:14px;padding:18px 20px;margin:4px 0}
.ca-hero{display:flex;flex-wrap:wrap;gap:16px;align-items:center}
.ca-label{font-size:30px;font-weight:800;letter-spacing:.06em}
.ca-pill{display:inline-block;padding:3px 10px;border-radius:999px;font-size:13px;font-weight:600}
.ca-muted{color:var(--body-text-color-subdued, rgba(128,128,128,.95));font-size:13px}
.ca-banner{border-radius:10px;padding:8px 12px;font-size:13px;margin-top:10px}
.ca-bar{height:10px;border-radius:999px;background:var(--border-color-primary, rgba(128,128,128,.35));overflow:hidden}
.ca-bar>div{height:100%;border-radius:999px}
.ca-row{display:grid;grid-template-columns:minmax(150px,230px) 1fr 42px;gap:12px;align-items:center;margin:9px 0}
.ca-table{width:100%;border-collapse:collapse;font-size:14px}
.ca-table td,.ca-table th{padding:8px 6px;border-bottom:1px solid var(--border-color-primary, rgba(128,128,128,.35));text-align:left;vertical-align:top}
.ca-ev{border:1px solid var(--border-color-primary, rgba(128,128,128,.35));border-radius:10px;padding:10px 12px;margin:8px 0}
.ca-ts{font-family:ui-monospace,Menlo,Consolas,monospace;font-weight:700;margin-right:8px}
.ca-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}
"""

LABEL_COLORS = {  # accent, translucent fill (readable on light and dark themes)
    "Excellent": ("#16a34a", "rgba(22,163,74,0.14)"),
    "Good": ("#2563eb", "rgba(37,99,235,0.14)"),
    "Average": ("#d97706", "rgba(217,119,6,0.16)"),
    "Poor": ("#dc2626", "rgba(220,38,38,0.14)"),
}
DIMENSIONS = [
    ("visual_compliance", "Visual & Brand Compliance"),
    ("intro_onboarding", "Intro & Onboarding"),
    ("individual_attention", "1-to-1 Student Engagement"),
    ("energy_rapport", "Vocal Energy & Rapport"),
    ("structure_homework", "Lesson Structure & Homework"),
]
DIM_NAMES = dict(DIMENSIONS)
POOR_SUB = {"compliance_quality_issue": "Compliance / Quality Issue",
            "pedagogical_weakness": "Pedagogical / Rapport Weakness",
            "both": "Compliance + Pedagogical Issues"}
BADGE = {"PASS": ("#16a34a", "rgba(22,163,74,0.14)"), "WARN": ("#d97706", "rgba(217,119,6,0.16)"),
         "FAIL": ("#dc2626", "rgba(220,38,38,0.14)")}

def _e(x) -> str:
    return html.escape(str(x if x is not None else ""))


def _pill(text: str, fg: str, bg: str) -> str:
    return f'<span class="ca-pill" style="color:{fg};background:{bg};border:1px solid {fg}">{_e(text)}</span>'


# ================================================================ renderers
def render_hero(result: dict) -> str:
    a, llm = result["assessment"], result.get("llm", {})
    fg, bg = LABEL_COLORS.get(a["label"], ("#64748b", "rgba(100,116,139,.14)"))
    sub = _pill(POOR_SUB[a["poor_sub_category"]], fg, bg) if a["label"] == "Poor" and a["poor_sub_category"] in POOR_SUB else ""
    if a["llm_status"] in ("ok", "cached"):
        mode = (f"Deterministic Rubric + Gemini Evidence · rating from measured visual / audio / transcript "
                f"features; summary and observations by {_e(llm.get('model') or 'Gemini')}")
        mode_bg = "rgba(37,99,235,0.10)"
    else:
        mode = "Deterministic Rubric Fallback · rating and evidence from measured features only" + (
            f" — {_e(llm.get('note'))}" if llm.get("note") else "")
        mode_bg = "rgba(217,119,6,0.12)"
    warn = (f'<div class="ca-banner" style="background:rgba(217,119,6,0.14)">⚠ {_e(a["language_warning"])}</div>'
            if a.get("language_warning") else "")
    return (f'<div class="ca-card" style="border-left:6px solid {fg}">'
            f'<div class="ca-hero"><span class="ca-label" style="color:{fg}">{_e(a["label"].upper())}</span>{sub}'
            f'<span class="ca-muted">Confidence <b>{a["confidence_pct"]:.0f}%</b> · composite {a["composite"]:.0f}/100</span></div>'
            f'<p style="margin:10px 0 0">{_e(a["executive_summary"])}</p>'
            f'<div class="ca-banner" style="background:{mode_bg}">{mode}</div>{warn}</div>')


def render_scores(assessment: dict) -> str:
    rows = []
    for key, name in DIMENSIONS:
        v = float(assessment["sub_scores"].get(key, 0))
        color = "#16a34a" if v >= 75 else "#2563eb" if v >= 60 else "#d97706" if v >= 40 else "#dc2626"
        rows.append(f'<div class="ca-row"><span>{_e(name)}</span><div class="ca-bar"><div style="width:{v:.0f}%;'
                    f'background:{color}"></div></div><b>{v:.0f}</b></div>')
    return f'<div class="ca-card"><h3 style="margin:0 0 6px">Sub-scores (0–100)</h3>{"".join(rows)}</div>'


def checklist_rows(result: dict) -> list[tuple[str, str, str]]:
    """(check, PASS/WARN/FAIL, measurement)."""
    t = result["telemetry"]
    v, au, tf, cam = t["visual"], t["audio"], t["transcript_features"], t["camera_refined"]
    a = result["assessment"]
    rows = [
        ("Branded virtual background kept", "FAIL" if v.get("flag_no_8x8_background") else "PASS",
         f"{v.get('bg_8x8_when_cam_active_pct', 0):.0f}% of active-camera frames"),
        ("Tutor face visible", "FAIL" if a["compliance_flags"]["face_not_visible"] else
         ("WARN" if v.get("flag_face_not_visible") else "PASS"),
         f"{v.get('tutor_face_visible_when_8x8_pct', 0):.0f}% of branded-background frames show the face"),
        ("Camera on while teaching", "FAIL" if a["compliance_flags"]["camera_off_while_teaching"] else
         ("WARN" if v.get("flag_intermittent_camera_off") else "PASS"),
         f"longest camera-off while speaking: {cam.get('max_camera_off_during_speech_streak_sec', 0):.0f}s"),
        ("Face lighting", "FAIL" if a["compliance_flags"]["dark_face"] else "PASS",
         f"face luminance {v.get('tutor_face_luminance_mean', 0):.0f}/255"),
        ("Background clarity", "WARN" if v.get("flag_bg_clarity_issue") else "PASS",
         f"colour bleed {v.get('bg_color_bleed_mean_pct', 0):.1f}%"),
        ("Warm intro & rapport", "PASS" if tf.get("has_warm_intro") else "WARN",
         f"intro score {tf.get('intro_score', 0):.2f}" + (f" at {tf['intro_evidence_ts']}" if tf.get("intro_evidence_ts") else "")),
        ("Time for new joiners / self-intro", "PASS" if tf.get("has_self_intro_or_new_joiner_time") else "WARN",
         tf.get("self_intro_evidence_ts") or "not detected"),
        ("No jump straight into revision", "FAIL" if tf.get("jumped_straight_to_revision") else "PASS",
         f"revision cue at {tf['early_revision_ts']}" if tf.get("jumped_straight_to_revision") else "rapport first"),
        ("1-to-1 questioning", "PASS" if (tf.get("distinct_students_called", 0) >= 5 or tf.get("teacher_question_count", 0) >= 60)
         else "WARN", f"{tf.get('distinct_students_called', 0)} students named · {tf.get('teacher_question_count', 0)} questions"),
        ("Homework assigned", "PASS" if tf.get("has_homework") else "WARN", tf.get("homework_evidence_ts") or "not detected"),
        ("Proper conclusion", "PASS" if tf.get("has_proper_conclusion") else "WARN", tf.get("conclusion_evidence_ts") or "not detected"),
        ("Dead air", "PASS" if au.get("dead_air_total_sec", 0) < 600 else "WARN",
         f"{au.get('dead_air_total_sec', 0):.0f}s in pauses ≥5s · longest {au.get('max_silence_gap_sec', 0):.0f}s"),
    ]
    return rows


def render_checklist(result: dict) -> str:
    body = "".join(f"<tr><td>{_e(c)}</td><td>{_pill(s, *BADGE[s])}</td><td class='ca-muted'>{_e(m)}</td></tr>"
                   for c, s, m in checklist_rows(result))
    return (f'<div class="ca-card"><h3 style="margin:0 0 6px">Compliance & pedagogical checklist</h3>'
            f'<table class="ca-table"><tr><th>Check</th><th>Status</th><th>Measurement</th></tr>{body}</table></div>')


def _evidence_cards(items: list[dict], accent: str) -> str:
    if not items:
        return "<p class='ca-muted'>None recorded.</p>"
    out = []
    for o in items:
        dim = DIM_NAMES.get(o.get("dimension"), str(o.get("dimension", "")).replace("_", " ").title())
        src = "AI judge" if o.get("source") == "llm" else "Measured"
        out.append(f'<div class="ca-ev" style="border-left:4px solid {accent}"><span class="ca-ts">{_e(o.get("timestamp") or "—")}</span>'
                   f'<span class="ca-pill" style="background:var(--border-color-primary, rgba(128,128,128,.35))">{_e(dim)}</span> '
                   f'<span class="ca-muted">· {src}</span><div style="margin-top:6px">{_e(o.get("observation"))}</div></div>')
    return "".join(out)


def render_evidence(assessment: dict) -> str:
    return (f'<div class="ca-grid"><div class="ca-card"><h3 style="margin:0">Strengths</h3>'
            f'{_evidence_cards(assessment["strengths"], "#16a34a")}</div>'
            f'<div class="ca-card"><h3 style="margin:0">Areas for improvement</h3>'
            f'{_evidence_cards(assessment["improvements"], "#d97706")}</div></div>')


def _neutral_keys(obj):
    """Rename internal keys that carry the brand (e.g. bg_8x8_*) to neutral names."""
    if isinstance(obj, dict):
        return {re.sub(r"8x8", "branded_bg", k): _neutral_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_neutral_keys(v) for v in obj]
    return obj


def telemetry_payload(result: dict) -> dict:
    a = result["assessment"]
    return _neutral_keys({"label": a["label"], "confidence_pct": a["confidence_pct"],
                          "triggered_gates": a["triggered_gates"], "label_source": a.get("label_source"),
                          "deterministic_scores": a["deterministic_scores"], "llm_scores": a["llm_scores"],
                          "llm": result.get("llm"), **result["telemetry"]})
