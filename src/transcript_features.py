"""Pedagogical rubric signals from masked, role-attributed segments.

Signals follow the organiser's feedback themes: warm intro / rapport, time for
new joiners, jumping straight into revision, 1-to-1 questioning and coverage,
homework, a proper conclusion, and praise. Every boolean comes with an
[MM:SS] timestamp; excerpts are taken only from already-masked text.
"""
from __future__ import annotations

import re

_STUDENT_PH = re.compile(r"\[STUDENT_\d+\]")


def _rx(phrases):
    return re.compile(r"(?i)\b(?:" + "|".join(phrases) + r")\b")


_GREET_P = [r"hello", r"hi", r"good (?:morning|afternoon|evening)", r"welcome"]
_RAPPORT_P = [r"how are you", r"how was your day", r"nice to see you", r"how was school",
              r"how is everyone", r"how was your week", r"what did you do"]
GREETING, RAPPORT, WARM = _rx(_GREET_P), _rx(_RAPPORT_P), _rx(_GREET_P + _RAPPORT_P)
SELF_INTRO = _rx([r"introduce", r"my name is", r"tell me about yourself", r"new joiners?",
                  r"first class", r"which grade", r"what grade", r"which class are you", r"where are you from"])
REVISION = _rx([r"last class", r"previous class", r"revision", r"revise", r"we learned", r"we learnt",
                r"open the board", r"puzzles?", r"move", r"white to play", r"black to play"])
PROMPT = re.compile(r"(?i)(\?|\b(?:can you|could you|tell me|your turn|what is|what's|how many|which|why|"
                    r"what will|what would|unmute)\b)")
HOMEWORK = _rx([r"homework", r"home work", r"practice at home", r"assignment", r"worksheet",
                r"solve these", r"before next class", r"send (?:it )?in the group"])
CONCLUSION = _rx([r"thank you", r"bye", r"see you next class", r"see you (?:on|next)", r"great job today",
                  r"any doubts", r"any questions", r"wrap up", r"that's all for today"])
PRAISE = _rx([r"good job", r"well done", r"excellent", r"awesome", r"super", r"very good", r"great try",
              r"nice move", r"wonderful", r"fantastic", r"brilliant", r"great job"])


def ts(sec: float) -> str:
    sec = max(0, int(sec))
    return f"[{sec // 60:02d}:{sec % 60:02d}]"


def _excerpt(text: str, n: int = 100) -> str:
    t = " ".join(text.split())
    return t if len(t) <= n else t[: n - 1] + "…"


def gini(values: list[int]) -> float:
    """0 = perfectly equal attention across students, -> 1 = one student gets everything."""
    v = sorted(x for x in values if x >= 0)
    n, total = len(v), sum(v)
    if n == 0 or total == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(v))
    return round((2 * cum) / (n * total) - (n + 1) / n, 3)


def _first(segs, rx, t0=0.0, t1=float("inf")):
    for s in segs:
        if t0 <= s["start"] < t1 and rx.search(s["text"]):
            return s
    return None


def extract_transcript_features(masked_segments: list[dict], deid_stats: dict, speaker_stats: dict,
                                duration_sec: float) -> dict:
    segs = sorted(masked_segments, key=lambda s: s["start"])
    dur = float(duration_sec) or (segs[-1]["end"] if segs else 0.0)
    # Recordings often start minutes before anyone speaks: anchor the opening at first speech.
    t0 = segs[0]["start"] if segs else 0.0
    open_end = t0 + min(480.0, 0.18 * dur)
    close_start = dur - min(420.0, 0.15 * dur)
    opening = [s for s in segs if s["start"] < open_end]

    # 1. warm intro & rapport
    greet_hits = {m.group(0).lower() for s in opening for m in GREETING.finditer(s["text"])}
    rapport_hits = {m.group(0).lower() for s in opening for m in RAPPORT.finditer(s["text"])}
    intro_score = round(min(1.0, 0.25 * min(len(greet_hits), 2) + 0.25 * min(len(rapport_hits), 2)), 2)
    first_warm = _first(segs, WARM, 0, open_end)
    first_rapport = _first(segs, RAPPORT, 0, open_end)

    # 2. self-intro / new joiner time
    si = _first(segs, SELF_INTRO, 0, open_end)
    early7 = [s for s in segs if s["start"] < t0 + 420]
    early_students = {p for s in early7 for p in _STUDENT_PH.findall(s["text"])}
    early_student_turns = sum(1 for s in early7 if s.get("speaker_role") == "STUDENT")
    roll_call = len(early_students) >= 3 and early_student_turns >= 5
    if si is None and roll_call:
        si = next(s for s in early7 if _STUDENT_PH.search(s["text"]))

    # 3. jumped straight into revision
    rev = _first(segs, REVISION, 0, t0 + 180)
    warm_ts = first_rapport["start"] if first_rapport else (first_warm["start"] if first_warm else None)
    jumped = bool(rev) and (warm_ts is None or rev["start"] < warm_ts or intro_score < 0.35)

    # 4. individual questioning & coverage
    teacher = [s for s in segs if s.get("speaker_role", "TEACHER") == "TEACHER"]
    questions = [s for s in teacher if PROMPT.search(s["text"])]
    called: dict[str, int] = {}
    individual = []
    for s in questions:
        phs = list(dict.fromkeys(_STUDENT_PH.findall(s["text"])))
        if phs:
            individual.append((s, phs))
            for p in phs:
                called[p] = called.get(p, 0) + 1
    population = set(deid_stats.get("per_student_mention_counts", {})) | set(called)
    examples = [f"{ts(s['start'])} [TEACHER] asked {phs[0]}: {_excerpt(s['text'])}" for s, phs in individual[:3]]

    # 5. homework (prefer the last quarter of class)
    hw = _first(segs, HOMEWORK, 0.75 * dur) or _first(segs, HOMEWORK)
    # 6. conclusion
    concl = _first(segs, CONCLUSION, close_start)
    # 7. praise
    praise = sum(len(PRAISE.findall(s["text"])) for s in segs)

    return {
        "duration_min": round(dur / 60, 1),
        "first_speech_ts": ts(t0),
        "has_warm_intro": intro_score >= 0.35,
        "intro_score": intro_score,
        "intro_evidence_ts": ts(first_warm["start"]) if first_warm else None,
        "has_self_intro_or_new_joiner_time": si is not None,
        "self_intro_evidence_ts": ts(si["start"]) if si else None,
        "jumped_straight_to_revision": jumped,
        "early_revision_ts": ts(rev["start"]) if rev else None,
        "distinct_students_called": len(called),
        "individual_questions_count": len(individual),
        "teacher_question_count": len(questions),
        "student_equity_gini": gini([called.get(p, 0) for p in population]),
        "individual_question_examples": examples,
        "has_homework": hw is not None,
        "homework_evidence_ts": ts(hw["start"]) if hw else None,
        "homework_snippet": _excerpt(hw["text"], 120) if hw else None,
        "has_proper_conclusion": concl is not None,
        "conclusion_evidence_ts": ts(concl["start"]) if concl else None,
        "praise_count": praise,
        "praise_per_10min": round(10 * praise / (dur / 60), 2) if dur else 0.0,
        "teacher_talk_time_pct": speaker_stats.get("teacher_talk_time_pct"),
        "student_talk_time_pct": speaker_stats.get("student_talk_time_pct"),
        "student_turn_count": speaker_stats.get("student_turn_count"),
        "interaction_alternations_per_10min": speaker_stats.get("interaction_alternations_per_10min"),
        "distinct_student_placeholders": deid_stats.get("distinct_student_placeholders"),
    }
