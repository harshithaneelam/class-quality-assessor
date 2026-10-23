"""De-identification, speaker-role and rubric-feature tests on synthetic transcripts."""
import re
import wave

import numpy as np
import pytest

from src.deidentify import deidentify_transcript
from src.speaker_roles import attribute_speaker_roles
from src.transcript_features import extract_transcript_features, gini, ts


def seg(start, text, dur=4.0):
    return {"start": float(start), "end": float(start) + dur, "text": text}


def masked_text(res):
    return " ".join(s["text"] for s in res["masked_segments"])


# ------------------------------------------------------------- deidentify
def test_teacher_intro_with_unseen_name():
    r = deidentify_transcript([seg(5, "Good evening everyone, I am your coach Zorbina and today we will learn forks."),
                               seg(20, "Zorbina ma'am, can I share my screen?")])
    t = masked_text(r)
    assert "Zorbina" not in t and t.count("[TEACHER]") == 2
    assert r["stats"]["teacher_name_masked"] and r["stats"]["distinct_student_placeholders"] == 0


def test_indian_gazetteer_and_order_of_appearance():
    r = deidentify_transcript([seg(0, "Okay so we start now."),
                               seg(10, "Ananya, can you tell me the best move?"),
                               seg(20, "Well done Vihaan, that was a fork."),
                               seg(30, "ananya is right, the knight attacks two pieces.")])
    t = masked_text(r)
    assert "nanya" not in t.lower() and "vihaan" not in t.lower()
    segs = r["masked_segments"]
    assert segs[1]["text"].startswith("[STUDENT_1]") and "[STUDENT_2]" in segs[2]["text"]
    assert segs[3]["text"].startswith("[STUDENT_1]")  # lowercase global pass
    assert r["stats"]["per_student_mention_counts"] == {"[STUDENT_1]": 2, "[STUDENT_2]": 1}


def test_phonetic_variants_cluster_to_one_student():
    r = deidentify_transcript([seg(0, "Hi Kritagya, welcome to class."),
                               seg(30, "Very good Krithagya, now try the next one."),
                               seg(60, "Tell me Kritagyaa, where does the bishop go?")])
    t = masked_text(r)
    assert "ritag" not in t.lower() and "rithag" not in t.lower()
    assert r["stats"]["distinct_student_placeholders"] == 1
    assert t.count("[STUDENT_1]") == 3


def test_full_name_self_intro_and_surname():
    r = deidentify_transcript([seg(400, "My name is Ishaan Varghese and I am in grade four."),
                               seg(410, "Okay Ishaan, welcome. Varghese is a nice surname.")])
    t = masked_text(r)
    assert "Ishaan" not in t and "Varghese" not in t
    assert r["stats"]["distinct_student_placeholders"] == 1


def test_chess_terms_places_and_months_not_masked():
    text = ("This is Lichess. White to play: Knight to f3, then Queen takes, check and checkmate. "
            "In Kerala and Dubai we play the Sicilian on Sunday in March.")
    r = deidentify_transcript([seg(0, text)])
    assert masked_text(r) == text
    assert r["stats"]["total_mask_replacements"] == 0


def test_extra_known_names():
    r = deidentify_transcript([seg(0, "and then plorbix moved the rook")], extra_known_names=["Plorbix Q"])
    assert "plorbix" not in masked_text(r).lower()


# ---------------------------------------------------------- speaker roles
def test_lexical_roles_and_metrics():
    segs = [
        seg(0, "[STUDENT_1], can you tell me where the knight should go to attack the queen and the rook "
               "at the same time, everyone else please watch the board carefully?", 10),
        seg(10.5, "Knight to e5.", 1.5),
        seg(12.5, "Very good, that is a fork, now let's see the next puzzle on the board for everyone here.", 8),
        seg(21, "Yes sir.", 1),
    ]
    r = attribute_speaker_roles(segs)
    assert [s["speaker_role"] for s in r["segments"]] == ["TEACHER", "STUDENT", "TEACHER", "STUDENT"]
    assert r["student_turn_count"] == 2
    assert r["teacher_talk_time_pct"] == pytest.approx(100 * 18 / 20.5, abs=0.1)
    assert r["interaction_alternations_per_10min"] > 0


def test_acoustic_high_pitch_short_segment_leans_student(tmp_path):
    sr = 16000

    def tone(f, secs, amp=0.3):
        t = np.arange(int(sr * secs)) / sr
        return amp * np.sin(2 * np.pi * f * t)

    # 6 long low-pitch "teacher" segments, then one short high-pitch neutral-text segment.
    audio = np.concatenate([tone(140, 9) for _ in range(6)] + [tone(330, 3)])
    wav = tmp_path / "a.wav"
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    segs = [seg(9 * i, "we keep going with this long explanation about the opening ideas here", 9) for i in range(6)]
    segs.append(seg(54, "I think it is that one", 3))
    r = attribute_speaker_roles(segs, wav)
    assert r["teacher_f0_baseline_hz"] == pytest.approx(140, rel=0.1)
    assert r["segments"][-1]["speaker_role"] == "STUDENT"
    assert all(s["speaker_role"] == "TEACHER" for s in r["segments"][:6])


# -------------------------------------------------------- rubric features
def _features(segs, dur=3000):
    for s in segs:
        s.setdefault("speaker_role", "TEACHER")
    deid = {"per_student_mention_counts": {"[STUDENT_1]": 3, "[STUDENT_2]": 1, "[STUDENT_3]": 1}}
    return extract_transcript_features(segs, deid, {"teacher_talk_time_pct": 80.0}, dur)


def test_good_class_signals():
    f = _features([
        seg(10, "Good evening everyone, welcome! How are you all? How was your day?"),
        seg(60, "We have a new joiner today, [STUDENT_3] please introduce yourself."),
        seg(200, "Now let's revise what we learned in the last class."),
        seg(900, "[STUDENT_1], can you find the fork?"),
        seg(1000, "[STUDENT_2], what is the best move here?"),
        seg(1100, "Well done! Excellent, very good."),
        seg(2700, "For homework, solve these five puzzles before next class."),
        seg(2950, "Any doubts? Great job today, thank you, bye!"),
    ])
    assert f["has_warm_intro"] and f["intro_score"] >= 0.75 and f["intro_evidence_ts"] == "[00:10]"
    assert f["has_self_intro_or_new_joiner_time"] and f["self_intro_evidence_ts"] == "[01:00]"
    assert not f["jumped_straight_to_revision"]
    assert f["distinct_students_called"] == 2 and f["individual_questions_count"] == 2
    assert f["has_homework"] and f["homework_evidence_ts"] == "[45:00]"
    assert f["has_proper_conclusion"] and f["conclusion_evidence_ts"] == "[49:10]"
    assert f["praise_count"] == 4  # well done, excellent, very good, great job
    assert f["individual_question_examples"][0].startswith("[15:00] [TEACHER] asked [STUDENT_1]:")


def test_jumped_straight_no_homework_no_conclusion():
    f = _features([
        seg(15, "Okay open the board, last class we did pins, white to play."),
        seg(1500, "[STUDENT_1] what is the answer?"),
        seg(2980, "Okay."),
    ])
    assert f["jumped_straight_to_revision"] and f["early_revision_ts"] == "[00:15]"
    assert not f["has_warm_intro"] and not f["has_homework"] and not f["has_proper_conclusion"]


def test_opening_anchored_at_first_speech():
    # Silent first 4 minutes, then straight into revision; greeting only at 27 min.
    f = _features([
        seg(251, "Okay, last class we did the fork, open the board."),
        seg(1620, "Hello everyone."),
    ])
    assert f["first_speech_ts"] == "[04:11]"
    assert f["jumped_straight_to_revision"] and f["early_revision_ts"] == "[04:11]"
    assert not f["has_warm_intro"]


def test_helpers():
    assert ts(3725) == "[62:05]"
    assert gini([1, 1, 1]) == 0.0
    assert gini([0, 0, 9]) == pytest.approx(0.667, abs=1e-3)


def test_rare_lowercase_and_sentence_start_names_masked_as_low_confidence():
    r = deidentify_transcript([seg(0, "Okay so then zorbukov moved the rook."),
                               seg(10, "Plavtrinka. Good, now the bishop.")])
    t = masked_text(r)
    assert "zorbukov" not in t.lower() and "plavtrinka" not in t.lower()
    assert t.count("[NAME]") == 2 and r["stats"]["low_confidence_name_masks"] == 2
    assert r["stats"]["distinct_student_placeholders"] == 0  # low-confidence masks don't add students


def test_chess_jargon_not_masked_by_rare_rule():
    text = "That fianchetto bishop and zugzwang ideas on Chessbrainz and Lichess."
    assert masked_text(deidentify_transcript([seg(0, text)])) == text


def test_cluster_inversion_guard(monkeypatch):
    import src.speaker_roles as srm
    long = "now everyone look at the board and tell me which piece is attacking the queen here"
    segs = [seg(10 * i, long, 9) for i in range(6)] + [seg(60, "the knight", 2)]
    # Acoustic vote pointing the wrong way: teacher-heavy segments look "student".
    monkeypatch.setattr(srm, "acoustic_votes", lambda s, a: ({i: -5.0 for i in range(6)} | {6: 5.0}, 200.0, -20.0))
    r = attribute_speaker_roles(segs)
    assert all(s["speaker_role"] == "TEACHER" for s in r["segments"][:6])
    assert r["teacher_talk_time_pct"] > 45
