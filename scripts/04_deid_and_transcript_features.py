"""Step 04 — de-identification, speaker roles and rubric features for all transcripts.

Validates (numbers only):
  * blind de-identification recall against VTT speaker-prefix name tokens,
  * zero post-scrub leaks in everything written to disk,
  * TEACHER/STUDENT attribution against VTT speaker prefixes.

PRIVACY: prints ONLY numbers / booleans / labels — no transcript snippets, not
even masked ones. VTT speaker names stay in process memory. Masked transcripts
and features are written to <data>/_work/ (outside the repo).

Usage:  .venv\\Scripts\\python scripts\\04_deid_and_transcript_features.py [--wait-min 90]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.deidentify import deidentify_transcript  # noqa: E402
from src.speaker_roles import attribute_speaker_roles  # noqa: E402
from src.transcript_features import extract_transcript_features  # noqa: E402
from src.wer import normalize_words, parse_vtt_speaker_cues  # noqa: E402


def _safe_excepthook(exc_type, exc, tb):
    print(f"[FATAL] {exc_type.__name__} (details suppressed for privacy)", file=sys.stderr)


sys.excepthook = _safe_excepthook

GENERIC_SPEAKER_TOKENS = {
    "coach", "tutor", "teacher", "admin", "user", "ipad", "iphone", "zoom", "guest", "host", "mobile",
    "android", "samsung", "galaxy", "phone", "laptop", "speaker", "unknown", "sir", "maam", "mam",
    "miss", "mrs", "chess", "class", "student", "the", "and", "redmi", "oppo", "vivo", "realme",
    "lenovo", "tab", "desktop", "pc", "macbook", "huawei", "nokia", "oneplus", "pro", "max", "mini",
    # connection / device labels that appear in speaker prefixes (e.g. "<device> audio")
    "audio", "video", "call", "dial", "only", "meeting", "room", "camera", "screen", "share", "cohost",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def wait_for_transcripts(tx_dir: Path, ids: list[str], wait_min: float) -> list[str]:
    deadline = time.time() + wait_min * 60
    while True:
        missing = [i for i in ids if not (tx_dir / f"{i}.json").is_file()]
        if not missing or time.time() > deadline:
            return missing
        log(f"- waiting for {len(missing)} transcript(s) from step 03 ...")
        time.sleep(30)


def speaker_tokens(cues) -> tuple[set[str], str | None, set[str]]:
    """All name tokens, primary (most words) speaker, and that speaker's tokens. In memory only."""
    words: dict[str, int] = {}
    for _, _, spk, text in cues:
        if spk:
            words[spk] = words.get(spk, 0) + len(normalize_words(text))
    primary = max(words, key=words.get) if words else None

    def toks(name):
        return {t.lower() for t in re.findall(r"[A-Za-z]+", name)
                if len(t) >= 3 and t.lower() not in GENERIC_SPEAKER_TOKENS}
    all_toks = set().union(*(toks(s) for s in words)) if words else set()
    return all_toks, primary, toks(primary) if primary else set()


def token_rx(tokens: set[str]):
    if not tokens:
        return None
    alts = "|".join(re.escape(t) for t in sorted(tokens, key=len, reverse=True))
    return re.compile(rf"(?<![\w\[])(?:{alts})(?![\w\]])", re.I)


def vtt_role_at(cues, start, end, primary):
    """Speaker role of the VTT cue with the largest overlap with [start, end)."""
    best, best_ov = None, 0.0
    for cs, ce, spk, _ in cues:
        if ce <= start or cs >= end or not spk:
            continue
        ov = min(ce, end) - max(cs, start)
        if ov > best_ov:
            best, best_ov = spk, ov
    if best is None:
        return None
    return "TEACHER" if best == primary else "STUDENT"


def table(rows, cols) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join("—" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-min", type=float, default=90.0)
    args = ap.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    work = Path(os.environ.get("CHALLENGE_DATA_DIR") or r"C:\challenge-data") / "_work"
    tx_dir, out_dir = work / "transcripts_small_en", work / "masked_transcripts"
    index = json.loads((work / "video_index.json").read_text(encoding="utf-8"))
    ids = list(index)

    missing = wait_for_transcripts(tx_dir, ids, args.wait_min)
    if missing:
        raise SystemExit(f"{len(missing)} transcript(s) still missing after {args.wait_min} min — rerun later")

    log(f"## Step 04 — de-identification + speaker roles + rubric features ({len(ids)} recordings)")
    val_rows, feat_rows, all_out = [], [], {}
    for n, sid in enumerate(ids, 1):
        e = index[sid]
        label = e["label"] + (f" ({e['sub_origin']})" if e.get("sub_origin") else "")
        tx = json.loads((tx_dir / f"{sid}.json").read_text(encoding="utf-8"))
        segs = [{k: s[k] for k in ("id", "start", "end", "text", "avg_logprob", "no_speech_prob") if k in s}
                for s in tx["segments"]]

        cues = parse_vtt_speaker_cues(e["vtt_path"]) if e.get("vtt_path") else []
        name_toks, primary, _ = speaker_tokens(cues)
        n_speakers = len({c[2] for c in cues if c[2]})

        # Blind pass (no ground-truth names)
        blind = deidentify_transcript(segs, extra_known_names=None)
        rx = token_rx(name_toks)
        raw_text = " ".join(s["text"] for s in segs)
        blind_text = " ".join(s["text"] for s in blind["masked_segments"])
        occurred = {t for t in name_toks if re.search(rf"(?<![\w]){re.escape(t)}(?![\w])", raw_text, re.I)}
        caught = {t for t in occurred if not re.search(rf"(?<![\w]){re.escape(t)}(?![\w])", blind_text, re.I)}
        recall = round(100 * len(caught) / len(occurred), 1) if occurred else None

        # Secondary scrub of any remaining ground-truth tokens (privacy guarantee on disk)
        masked = blind["masked_segments"]
        scrubbed_hits = 0
        if rx is not None:
            for s in masked:
                s["text"], k = rx.subn("[NAME]", s["text"])
                scrubbed_hits += k

        roles = attribute_speaker_roles(masked, work / "audio" / f"{sid}.wav")
        rsegs = roles.pop("segments")
        feats = extract_transcript_features(rsegs, blind["stats"], roles, tx.get("duration_sec") or 0.0)

        # Validation vs VTT speaker prefixes
        tw = sum(len(normalize_words(t)) for _, _, spk, t in cues if spk and spk == primary)
        aw = sum(len(normalize_words(t)) for _, _, spk, t in cues if spk)
        vtt_teacher_pct = round(100 * tw / aw, 1) if aw else None
        ew_t = sum(len(normalize_words(s["text"])) for s in rsegs if s["speaker_role"] == "TEACHER")
        ew_a = sum(len(normalize_words(s["text"])) for s in rsegs)
        est_teacher_word_pct = round(100 * ew_t / ew_a, 1) if ew_a else None
        pairs = [(s["speaker_role"], vtt_role_at(cues, s["start"], s["end"], primary)) for s in rsegs]
        pairs = [(p, t) for p, t in pairs if t is not None]
        role_acc = round(100 * sum(p == t for p, t in pairs) / len(pairs), 1) if pairs else None
        confusion = {f"pred_{p.lower()}_vtt_{t.lower()}": sum(1 for a, b in pairs if (a, b) == (p, t))
                     for p in ("TEACHER", "STUDENT") for t in ("TEACHER", "STUDENT")}
        stud_true = [(p, t) for p, t in pairs if t == "STUDENT"]
        stud_recall = round(100 * sum(p == "STUDENT" for p, _ in stud_true) / len(stud_true), 1) if stud_true else None

        masked_doc = {"short_id": sid, "duration_sec": tx.get("duration_sec"), "model_name": tx.get("model_name"),
                      "deid_stats": blind["stats"], "speaker_stats": roles, "segments": rsegs}
        feat_doc = {"short_id": sid, "label": e["label"], "sub_origin": e.get("sub_origin"),
                    "features": feats, "deid_stats": blind["stats"], "speaker_stats": roles}
        # Leak check over everything that goes to disk
        leaks = len(rx.findall(json.dumps(masked_doc, ensure_ascii=False) + json.dumps(feat_doc, ensure_ascii=False))) if rx else 0
        if leaks:
            raise SystemExit(f"{sid}: post-scrub leak check failed ({leaks}) — nothing written")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{sid}.json").write_text(json.dumps(masked_doc, ensure_ascii=False), encoding="utf-8")

        validation = {"vtt_speakers": n_speakers, "vtt_name_tokens": len(name_toks),
                      "name_tokens_in_whisper": len(occurred), "blind_name_recall_pct": recall,
                      "post_scrub_replacements": scrubbed_hits, "post_scrub_leaks": leaks,
                      "vtt_teacher_word_pct": vtt_teacher_pct, "est_teacher_word_pct": est_teacher_word_pct,
                      "role_accuracy_pct": role_acc, "student_role_recall_pct": stud_recall,
                      "role_segments_compared": len(pairs), "role_confusion": confusion,
                      "segments_with_vtt_speaker_pct": round(100 * len(pairs) / len(rsegs), 1) if rsegs else None}
        feat_doc["validation"] = validation
        all_out[sid] = feat_doc

        val_rows.append({"short_id": sid, "label": label, "vtt_speakers": n_speakers,
                         "whisper_stud_placeholders": blind["stats"]["distinct_student_placeholders"],
                         "total_masks": blind["stats"]["total_mask_replacements"],
                         "blind_name_recall%": f"{recall} ({len(caught)}/{len(occurred)})" if occurred else "— (0 in audio)",
                         "post_scrub_leaks": leaks, "vtt_teacher_talk%": vtt_teacher_pct,
                         "est_teacher_talk%": est_teacher_word_pct, "est_teacher_time%": roles["teacher_talk_time_pct"],
                         "role_acc%": role_acc, "stud_recall%": stud_recall, "stud_turns": roles["student_turn_count"],
                         "conf TT/TS/ST/SS": "/".join(str(confusion[k]) for k in (
                             "pred_teacher_vtt_teacher", "pred_teacher_vtt_student",
                             "pred_student_vtt_teacher", "pred_student_vtt_student"))})
        feat_rows.append({"short_id": sid, "label": label, "warm_intro": feats["has_warm_intro"],
                          "intro_score": feats["intro_score"],
                          "self_intro": feats["has_self_intro_or_new_joiner_time"],
                          "jump_rev": feats["jumped_straight_to_revision"],
                          "studs_called": feats["distinct_students_called"],
                          "indiv_q": feats["individual_questions_count"],
                          "questions": feats["teacher_question_count"], "gini": feats["student_equity_gini"],
                          "praise/10m": feats["praise_per_10min"], "homework": feats["has_homework"],
                          "hw_ts": feats["homework_evidence_ts"], "conclusion": feats["has_proper_conclusion"]})
        log(f"[{n}/{len(ids)}] {sid}: done (masks {blind['stats']['total_mask_replacements']}, "
            f"leaks {leaks}, role_acc {role_acc}%)")

    (work / "04_transcript_features.json").write_text(json.dumps(all_out, indent=1, ensure_ascii=False), encoding="utf-8")
    log("\n## Table 1 — De-ID & speaker attribution validation\n")
    log(table(val_rows, ["short_id", "label", "vtt_speakers", "whisper_stud_placeholders", "total_masks",
                         "blind_name_recall%", "post_scrub_leaks", "vtt_teacher_talk%", "est_teacher_talk%",
                         "est_teacher_time%", "role_acc%", "stud_recall%", "stud_turns", "conf TT/TS/ST/SS"]))
    log("\n## Table 2 — Pedagogical rubric signals\n")
    log(table(feat_rows, ["short_id", "label", "warm_intro", "intro_score", "self_intro", "jump_rev",
                          "studs_called", "indiv_q", "questions", "gini", "praise/10m", "homework", "hw_ts",
                          "conclusion"]))
    log(f"\nSaved -> {work / '04_transcript_features.json'} and {out_dir}")


if __name__ == "__main__":
    main()
