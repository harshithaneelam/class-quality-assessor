"""Gemini structured class judge: de-identified prompt, privacy gate, cache, fallback.

PRIVACY
  * Only de-identified text is sent: masked transcript lines ([TEACHER],
    [STUDENT_k], [NAME]) plus numeric / boolean features. No media bytes.
  * `preflight_privacy_check` refuses to send a prompt that contains any known
    speaker-name token, a dataset path, or binary / base64-looking payloads.
  * Responses (which contain the LLM's evidence text) are cached to disk under
    the work dir and are never printed by this module.

QUOTA
  * Cache key = SHA-256(PROMPT_VERSION, short_id, SHA-256(masked transcript block)),
    so threshold tuning / LOO re-runs never spend quota; only a changed transcript
    or prompt version does.
  * 429 / 503 -> backoff 2 s, 5 s, 10 s, then the next model in the chain.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

PROMPT_VERSION = "judge-v2"
# Masked transcript lines cost ~2.4 tokens/word ([MM:SS] (T) prefixes, placeholders):
# v1's 5,500-word cap produced prompts up to 13k tokens. Measured ~3.0 chars/token incl.
# the system prompt; 3,000 words keeps the worst class (89 min) near ~8k input tokens.
MAX_PROMPT_WORDS = 3000
BACKOFF_SEC = (2, 5, 10)
PACING_SEC = 1.5
_last_call = 0.0
LAST_ERRORS: list[str] = []  # "<model>:<ExceptionClass>:<code>" from the most recent uncached call


# ======================================================================= schema
class PedagogicalScores(BaseModel):
    intro_and_onboarding_0_100: int
    individual_attention_0_100: int
    communication_energy_and_rapport_0_100: int
    lesson_structure_and_homework_0_100: int
    visual_and_brand_compliance_0_100: int


class Observation(BaseModel):
    timestamp: str
    dimension: str
    observation: str


class LLMClassAssessment(BaseModel):
    pedagogical_scores: PedagogicalScores
    llm_raw_label: Literal["Poor", "Average", "Good", "Excellent"]
    poor_sub_category: Literal["none", "compliance_quality_issue", "pedagogical_weakness", "both"]
    executive_summary: str
    strengths: list[Observation]
    improvements: list[Observation]


def normalize_assessment(a: LLMClassAssessment | dict) -> dict:
    """Clamp scores to 0..100 and keep 2..4 observations each (schema can't enforce it)."""
    d = a.model_dump() if isinstance(a, BaseModel) else dict(a)
    d["pedagogical_scores"] = {k: int(min(100, max(0, v))) for k, v in d["pedagogical_scores"].items()}
    d["strengths"] = d.get("strengths", [])[:4]
    d["improvements"] = d.get("improvements", [])[:4]
    return LLMClassAssessment.model_validate(d).model_dump()


# ================================================================ prompt building
SYSTEM_INSTRUCTION = """You are a senior quality reviewer for an online chess academy for children.
You assess ONE recorded online class from a de-identified transcript plus measured
visual/audio/transcript features. Names are masked: [TEACHER] is the coach, [STUDENT_k]
are children, [NAME] is an unidentified person. Never guess or invent names.

Transcript lines look like "[MM:SS] (T) text" (T = teacher, S = student; roles are
estimated automatically and can be wrong). Whisper transcription has ~20-35% word
errors, so judge patterns, not single words.

REVIEWER RUBRIC
- Poor has two failure modes:
  (a) Known quality / compliance issues: missing branded virtual background, tutor's face not
      visible even when the branded virtual background is kept, or tutor camera turned off repeatedly
      while teaching.
  (b) Weak teaching / rapport: jumping straight into revision of the previous class without
      rapport building, no time for new joiners' self-introduction, weak intro/conclusion,
      high dead air, flat or unpleasant interaction.
- Average: keeps the branded virtual background and asks individual 1-to-1 questions, but has noticeable
  secondary flaws: poor lighting on the tutor's face (face_luminance < 60), virtual
  background clarity/segmentation issues, low vocal energy plus missing homework, or an
  abrupt opening.
- Good: strong introduction, fixed branded virtual background, good 1-to-1 student coverage, but misses
  ONE minor item (no homework assigned OR interaction should be warmer / more pleasant).
- Excellent: fixed branded virtual background, clear face and lighting, warm rapport building, considers
  every student individually throughout, energetic and pleasant delivery, proper conclusion
  (and homework given or the topic enriched).

READING THE FEATURES
- Visual checks come as calibrated verdicts. A "PASS" verdict is reliable: do NOT contradict
  it (never claim the camera was off, the branded virtual background was missing, the face was hidden or
  the lighting was poor when that check is PASS). Only checks marked FAIL carry raw
  percentages and timestamps.
- Whisper often fails to transcribe Indian student names, so distinct_students_called can be
  low even when the coach addresses students one by one. If teacher_question_count >= 60,
  the coach IS asking frequent 1-to-1 questions across the class.

OUTPUT RULES
- Score each dimension 0-100.
- Give 2-4 strengths and 2-4 improvements. Each must cite a concrete [MM:SS] timestamp taken
  from the transcript or the feature timestamps, and refer to [TEACHER]/[STUDENT_k]
  interactions or to the measured visual/audio metrics.
- Write improvements constructively, as a reviewer would to the coach (e.g. give time for
  new joiners to introduce themselves, don't jump straight into revision, assign homework,
  improve face lighting, keep the branded virtual background and face visible, be more energetic and
  pleasant).
- Never reproduce long transcript passages; paraphrase in at most one short sentence each.
"""

_KEEP_RX = re.compile(r"\?|\[STUDENT_\d+\]|(?i:\bhomework\b|\bhome work\b|\bnext class\b|\bpuzzle\b|"
                      r"\blet'?s (?:start|move|begin)\b|\bnow we\b|\bthank you\b|\bbye\b)")


def _ts(sec: float) -> str:
    sec = max(0, int(sec))
    return f"[{sec // 60:02d}:{sec % 60:02d}]"


def merge_turns(segments: list[dict], max_gap: float = 1.5, max_turn_words: int = 120) -> list[dict]:
    """Merge consecutive same-role segments separated by < max_gap seconds
    (capped at max_turn_words so long monologues stay sampleable)."""
    out: list[dict] = []
    for s in segments:
        role = "S" if s.get("speaker_role") == "STUDENT" else "T"
        text = " ".join(s.get("text", "").split())
        if not text:
            continue
        if (out and out[-1]["role"] == role and s["start"] - out[-1]["end"] < max_gap
                and len(out[-1]["text"].split()) + len(text.split()) <= max_turn_words):
            out[-1]["text"] += " " + text
            out[-1]["end"] = s["end"]
        else:
            out.append({"start": s["start"], "end": s["end"], "role": role, "text": text})
    return out


def compact_transcript(segments: list[dict], duration_sec: float, max_words: int | None = None) -> tuple[str, dict]:
    """`[MM:SS] (T/S) text` lines; if over budget keep opening 12 min, closing 10 min,
    key moments (questions, students, homework, transitions) and a uniform middle sample."""
    max_words = MAX_PROMPT_WORDS if max_words is None else max_words
    turns = merge_turns(segments)
    words = [len(t["text"].split()) for t in turns]
    info = {"turns": len(turns), "words_total": sum(words), "sampled": False}
    keep = list(range(len(turns)))
    if sum(words) > max_words:
        info["sampled"] = True
        t0 = turns[0]["start"] if turns else 0.0
        core = {i for i, t in enumerate(turns) if t["start"] < t0 + 720 or t["start"] >= duration_sec - 600}
        key = {i for i, t in enumerate(turns) if i not in core and _KEEP_RX.search(t["text"])}
        chosen = set(core)
        budget = max_words - sum(words[i] for i in chosen)
        for i in sorted(key, key=lambda i: words[i]):  # short key moments first
            if words[i] <= budget:
                chosen.add(i)
                budget -= words[i]
        rest = [i for i in range(len(turns)) if i not in chosen]
        if rest and budget > 0:
            step = max(1, len(rest) // max(1, budget // max(1, int(sum(words[i] for i in rest) / len(rest)))))
            for i in rest[::step]:
                if words[i] <= budget:
                    chosen.add(i)
                    budget -= words[i]
        if sum(words[i] for i in chosen) > max_words:  # opening+closing alone too long: keep chronologically
            total, trimmed = 0, set()
            for i in sorted(chosen):
                if total + words[i] <= max_words:
                    trimmed.add(i)
                    total += words[i]
            chosen = trimmed
        keep = sorted(chosen)
    lines, prev = [], None
    for i in keep:
        if prev is not None and i != prev + 1:
            lines.append("[...]")
        t = turns[i]
        lines.append(f"{_ts(t['start'])} ({t['role']}) {t['text']}")
        prev = i
    info["words_sent"] = sum(words[i] for i in keep)
    return "\n".join(lines), info


def build_assessment_prompt(short_id: str, masked_segments: list[dict], duration_sec: float,
                            features: dict) -> tuple[str, str, dict]:
    """Return (prompt_text, transcript_block, info). `features` must be numbers/booleans/
    timestamps only (no transcript snippets)."""
    block, info = compact_transcript(masked_segments, duration_sec)
    prompt = (f"CLASS ID: {short_id}\nDURATION: {duration_sec / 60:.1f} min\n\n"
              f"MEASURED FEATURES (JSON):\n{json.dumps(features, separators=(',', ':'))}\n\n"
              f"DE-IDENTIFIED TRANSCRIPT ({info['words_sent']} of {info['words_total']} words"
              f"{', sampled: opening 12 min + closing 10 min + key moments + uniform middle' if info['sampled'] else ''}):\n"
              f"{block}\n\nAssess this class using the rubric. Return JSON only.")
    return prompt, block, info


# ============================================================ v2 feature block
def visual_verdicts(v: dict) -> dict:
    """PASS verdicts for passing checks; raw numbers + timestamps only for FAIL checks."""
    ev = v.get("visual_evidence_timestamps") or {}
    out = {}
    if v.get("flag_no_8x8_background"):
        out["branded_virtual_background"] = {"verdict": "FAIL (branded virtual background missing on many active camera frames)",
                                         "bg_8x8_when_cam_active_pct": v.get("bg_8x8_when_cam_active_pct"),
                                         "missing_samples": ev.get("bg_8x8_missing_samples")}
    else:
        out["branded_virtual_background"] = "PASS (verified present across active camera frames)"
    if v.get("flag_face_not_visible"):
        out["tutor_face_visibility"] = {"verdict": "FAIL (tutor face not visible on camera)",
                                        "face_visible_when_8x8_pct": v.get("tutor_face_visible_when_8x8_pct"),
                                        "overall_face_visible_pct": v.get("overall_face_visible_pct"),
                                        "face_missing_samples": ev.get("face_missing_on_8x8_samples")}
    else:
        out["tutor_face_visibility"] = "PASS (tutor face clearly visible)"
    if v.get("flag_intermittent_camera_off"):
        out["camera_continuity"] = {"verdict": "FAIL (camera off / missing repeatedly)",
                                    "camera_off_or_missing_pct": v.get("camera_off_or_missing_pct"),
                                    "max_camera_off_streak_sec": v.get("max_camera_off_streak_sec"),
                                    "camera_off_windows": ev.get("camera_off_windows")}
    else:
        out["camera_continuity"] = ("PASS (camera remained on during active teaching; any missing PiP frames were "
                                    "normal Zoom hidden-thumbnail panels during screen share)")
    if v.get("flag_poor_lighting"):
        out["face_lighting"] = {"verdict": "FAIL (face too dark)",
                                "tutor_face_luminance_mean": v.get("tutor_face_luminance_mean"),
                                "low_lighting_samples": ev.get("low_face_lighting_samples")}
    else:
        out["face_lighting"] = "PASS (well lit)"
    if v.get("flag_bg_clarity_issue"):
        out["background_clarity"] = {"verdict": "FAIL (virtual background segmentation / bleed issues)",
                                     "bg_color_bleed_mean_pct": v.get("bg_color_bleed_mean_pct"),
                                     "glitch_samples": ev.get("bg_glitch_samples")}
    else:
        out["background_clarity"] = "PASS (no segmentation issue measured)"
    return out


_AUDIO_KEYS = ["speech_ratio_pct", "silence_gaps_gt_10s_count", "dead_air_total_sec", "max_silence_gap_sec",
               "max_silence_gap_ts", "rms_std_db", "pitch_variation_cv", "energy_liveliness_score"]
_TRANSCRIPT_KEYS = ["first_speech_ts", "has_warm_intro", "intro_score", "intro_evidence_ts",
                    "has_self_intro_or_new_joiner_time", "self_intro_evidence_ts", "jumped_straight_to_revision",
                    "early_revision_ts", "distinct_students_called", "individual_questions_count",
                    "teacher_question_count", "student_equity_gini", "has_homework", "homework_evidence_ts",
                    "has_proper_conclusion", "conclusion_evidence_ts", "praise_count", "praise_per_10min",
                    "teacher_talk_time_pct", "student_turn_count", "interaction_alternations_per_10min"]


def build_prompt_features(visual: dict, audio: dict, transcript: dict, camera_refined: dict | None,
                          avg_logprob_mean: float | None, code_mix_threshold: float = -0.50) -> dict:
    """Numbers / booleans / timestamps / verdicts only — never transcript snippets."""
    out = {"visual_compliance": visual_verdicts(visual),
           "audio": {k: audio.get(k) for k in _AUDIO_KEYS},
           "transcript_signals": {k: transcript.get(k) for k in _TRANSCRIPT_KEYS}}
    if visual.get("flag_intermittent_camera_off") and camera_refined:
        out["camera_off_vs_speech"] = {k: camera_refined.get(k) for k in (
            "camera_off_during_speech_pct", "max_camera_off_during_speech_streak_sec",
            "camera_off_silent_samples", "camera_off_during_speech_ts")}
    if avg_logprob_mean is not None:
        out["asr"] = {"whisper_avg_logprob_mean": avg_logprob_mean,
                      "likely_code_mixed": avg_logprob_mean < code_mix_threshold}
    return out


_CONTRA = {
    "flag_intermittent_camera_off": re.compile(
        r"(?i)camera\s+(?:was\s+|is\s+|being\s+)?(?:off|turned\s+off|switched\s+off|not\s+on|disabled)|"
        r"turn(?:ed|s|ing)?\s+(?:the\s+|their\s+|his\s+|her\s+)?camera\s+off|camera[- ]off|no\s+camera|"
        r"without\s+(?:a\s+|the\s+)?camera|(?:pip|thumbnail|video\s+tile)\s+(?:was\s+)?(?:missing|hidden|absent)"),
    "flag_no_8x8_background": re.compile(
        r"(?i)(?:branded\s+(?:virtual\s+)?|virtual\s+)background\s+(?:was\s+|is\s+)?(?:missing|absent|removed|not\s+(?:used|kept|on|present|visible|maintained))|"
        r"(?:no|without|missing|lack(?:ed|s|ing)?|not\s+using|not\s+keep(?:ing)?|didn'?t\s+keep|did\s+not\s+keep)\s+"
        r"(?:the\s+|an?\s+)?(?:fixed\s+|official\s+)?(?:branded\s+(?:virtual\s+)?|virtual\s+)background"),
    "flag_face_not_visible": re.compile(
        r"(?i)face\s+(?:was\s+|is\s+)?(?:not|n't)\s+(?:clearly\s+)?(?:visible|seen|shown)|"
        r"face\s+(?:was\s+|is\s+)?(?:hidden|missing|obscured|absent)|(?:no|without)\s+(?:visible\s+)?face|"
        r"(?:could\s*n'?o?t|cannot|can't)\s+see\s+(?:the\s+)?(?:tutor'?s?\s+|coach'?s?\s+|teacher'?s?\s+)?face"),
    "flag_poor_lighting": re.compile(
        r"(?i)(?:poor|dim|dark|low|bad|insufficient)\s+(?:face\s+|room\s+)?lighting|"
        r"face\s+(?:is\s+|was\s+|appears\s+|looked\s+)?(?:too\s+)?(?:dark|dim|underexposed)"),
}


_BRAND_RX = re.compile(r"(?i)\b8\s*[x\u00d7]\s*8\b(?:\s+(?:virtual\s+)?background)?|\beight\s+times\s+eight\b")


def scrub_brand(text: str | None) -> str | None:
    """Replace the client's brand name in model-written text with a neutral phrase."""
    if not text:
        return text
    return _BRAND_RX.sub("branded virtual background", text)


def filter_contradicting_observations(observations: list[dict], visual: dict) -> tuple[list[dict], int]:
    """Drop LLM bullets that contradict a PASS visual verdict (flag False); scrub brand names."""
    kept, dropped = [], 0
    for o in observations:
        o = {**o, "observation": scrub_brand(o.get("observation")), "dimension": scrub_brand(o.get("dimension"))}
        text = f"{o.get('dimension', '')} {o.get('observation', '')}"
        if any(not visual.get(flag) and rx.search(text) for flag, rx in _CONTRA.items()):
            dropped += 1
            continue
        kept.append(o)
    return kept, dropped


# ============================================================ privacy pre-flight
class PrivacyViolation(Exception):
    pass


_B64_RX = re.compile(r"[A-Za-z0-9+/]{120,}={0,2}")


def preflight_privacy_check(prompt: str, forbidden_name_tokens: set[str] | None = None) -> None:
    """Raise PrivacyViolation if the outgoing prompt could leak names or media.

    Never includes the offending token in the error message."""
    if not isinstance(prompt, str):
        raise PrivacyViolation("prompt is not text")
    if "\x00" in prompt or _B64_RX.search(prompt) or "data:" in prompt.lower() and "base64" in prompt.lower():
        raise PrivacyViolation("binary / base64-like payload in prompt")
    if re.search(r"challenge-data|[A-Za-z]:\\|\.mp4|\.vtt|\.wav", prompt, re.I):
        raise PrivacyViolation("dataset path or media filename in prompt")
    hits = 0
    for tok in forbidden_name_tokens or ():
        if len(tok) >= 3 and re.search(rf"(?<![\w]){re.escape(tok)}(?![\w])", prompt, re.I):
            hits += 1
    if hits:
        raise PrivacyViolation(f"{hits} known speaker-name token(s) present in prompt")


# ===================================================================== caching
def default_cache_dir() -> Path:
    if os.environ.get("SPACE_ID"):  # Hugging Face Spaces
        return Path("/tmp/llm_cache")
    base = os.environ.get("CHALLENGE_DATA_DIR") or r"C:\challenge-data"
    return Path(base) / "_work" / "llm_cache"


def cache_key(short_id: str, transcript_block: str, prompt_version: str = PROMPT_VERSION) -> str:
    tx_hash = hashlib.sha256(transcript_block.encode("utf-8")).hexdigest()
    return hashlib.sha256(f"{prompt_version}|{short_id}|{tx_hash}".encode("utf-8")).hexdigest()


# ===================================================================== calling
def model_chain() -> list[str]:
    # gemini-2.0-flash is retired (404) and gemini-2.5-flash-lite returned 404 for this key
    # in Oct 2026; Google's rolling aliases keep the tail of the chain alive.
    chain = [os.getenv("GEMINI_PRIMARY_MODEL", "gemini-2.5-flash"),
             os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite"),
             "gemini-flash-lite-latest", "gemini-flash-latest"]
    return list(dict.fromkeys(m for m in chain if m))


def api_key_status() -> tuple[str | None, str]:
    """(key or None, human-readable status). Loads .env without printing it."""
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass
    key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key or key == "your_gemini_api_key_here":
        return None, ("GEMINI_API_KEY is missing or empty — add it to .env (see .env.example) "
                      "to enable the Gemini judge; using the deterministic fallback.")
    return key, "ok"


def _is_retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    msg = f"{type(exc).__name__} {exc}".upper()
    return code in (429, 503) or "RESOURCE_EXHAUSTED" in msg or "UNAVAILABLE" in msg or "429" in msg or "503" in msg


def _default_client(api_key: str):
    from google import genai
    return genai.Client(api_key=api_key)


def call_gemini_judge(short_id: str, prompt: str, transcript_block: str,
                      forbidden_name_tokens: set[str] | None = None, cache_dir: str | Path | None = None,
                      client=None, sleep=time.sleep, use_cache: bool = True) -> dict | None:
    """Return {"assessment", "model", "usage", "cache_hit", "llm_status"} or None (fallback).

    Raises PrivacyViolation before any network call if the prompt fails the gate.
    """
    global _last_call
    preflight_privacy_check(prompt, forbidden_name_tokens)
    cdir = Path(cache_dir) if cache_dir else default_cache_dir()
    cfile = cdir / f"{cache_key(short_id, transcript_block)}.json"
    if use_cache and cfile.is_file():
        cached = json.loads(cfile.read_text(encoding="utf-8"))
        return {**cached, "cache_hit": True, "llm_status": "cached"}

    if client is None:
        key, _status = api_key_status()
        if key is None:
            return None
        client = _default_client(key)

    from google.genai import types
    config = types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION,
                                         response_mime_type="application/json",
                                         response_schema=LLMClassAssessment, temperature=0.1)
    errors = LAST_ERRORS
    errors.clear()
    for model in model_chain():
        for attempt in range(len(BACKOFF_SEC) + 1):
            wait = PACING_SEC - (time.monotonic() - _last_call)
            if wait > 0:
                sleep(wait)
            try:
                _last_call = time.monotonic()
                resp = client.models.generate_content(model=model, contents=prompt, config=config)
                parsed = resp.parsed if getattr(resp, "parsed", None) is not None else \
                    LLMClassAssessment.model_validate_json(resp.text)
                um = getattr(resp, "usage_metadata", None)
                usage = {k: getattr(um, k, None) for k in
                         ("prompt_token_count", "candidates_token_count", "thoughts_token_count", "total_token_count")} if um else {}
                out = {"assessment": normalize_assessment(parsed), "model": getattr(resp, "model_version", None) or model,
                       "model_requested": model, "usage": usage, "prompt_version": PROMPT_VERSION,
                       "errors_before_success": list(errors)}
                cdir.mkdir(parents=True, exist_ok=True)
                tmp = cfile.with_suffix(".json.part")
                tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
                tmp.replace(cfile)
                return {**out, "cache_hit": False, "llm_status": "ok"}
            except Exception as exc:  # noqa: BLE001 - classify and fall through
                errors.append(f"{model}:{type(exc).__name__}:{getattr(exc, 'code', '')}")
                if _is_retryable(exc) and attempt < len(BACKOFF_SEC):
                    sleep(BACKOFF_SEC[attempt])
                    continue
                break  # non-retryable or retries exhausted -> next model
    return None
