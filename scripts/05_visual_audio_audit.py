"""Step 05 — visual compliance + audio prosody audit for all recordings.

Per-frame numeric metrics are cached in <data>/_work/05_frame_metrics/<id>.json,
so `--aggregate-only` re-applies thresholds without decoding video again.

PRIVACY: frames are analysed in memory and never saved, OCR'd or uploaded;
only numbers, booleans and [MM:SS] timestamps are printed.

Usage:  .venv\\Scripts\\python scripts\\05_visual_audio_audit.py [--aggregate-only] [--only ID ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.audio_features import extract_audio_features  # noqa: E402
from src.utils.ids import resolve_short_id  # noqa: E402
from src.visual_features import (FaceDetector, aggregate_visual, build_logo_template,  # noqa: E402
                                 classify_frames, expand_logo_templates, is_8x8_dark_rule,
                                 measure_video)


def _safe_excepthook(exc_type, exc, tb):
    print(f"[FATAL] {exc_type.__name__} (details suppressed for privacy)", file=sys.stderr)


sys.excepthook = _safe_excepthook


# Classes whose branded-background tile the dark-share rule already detects confidently; the coarse
# logo mask averaged over those frames is the reference for the logo-structure match.
LOGO_REFERENCE_IDS = ("avg_2", "good_2", "exc_2")
LOGO_REF_MIN_SCORE = 0.7


def make_logo_template(cache: Path, models_dir: Path, index: dict):
    """One template per reference class (logo scale/crop differs per tutor camera)."""
    tpls, n_rois = [], 0
    for short in LOGO_REFERENCE_IDS:
        try:
            sid = resolve_short_id(index, short)
        except KeyError:
            continue
        f = cache / f"{sid}.json"
        if not f.is_file():
            continue
        frames = json.loads(f.read_text(encoding="utf-8"))["frames"]
        rois = []
        for fr, mode in zip(frames, classify_frames(frames)):
            r = fr["full"] if mode == "speaker_fullscreen" else fr["pip"]
            if mode in ("speaker_fullscreen", "screenshare_pip") and is_8x8_dark_rule(r) and r["bg8_score"] >= LOGO_REF_MIN_SCORE:
                rois.append(r)
        tpl = build_logo_template(rois)
        if tpl is not None:
            tpls.append(tpl)
            n_rois += len(rois)
    if not tpls:
        return None, 0
    stack = np.stack(tpls).astype(np.float32)
    models_dir.mkdir(parents=True, exist_ok=True)
    np.save(models_dir / "8x8_logo_templates.npy", stack)
    return expand_logo_templates(stack), n_rois


def log(msg: str) -> None:
    print(msg, flush=True)


def table(rows, cols) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join("—" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def yn(b) -> str:
    return "**Y**" if b else "n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--every", type=float, default=10.0)
    args = ap.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    work = Path(os.environ.get("CHALLENGE_DATA_DIR") or r"C:\challenge-data") / "_work"
    cache = work / "05_frame_metrics"
    cache.mkdir(parents=True, exist_ok=True)
    index = json.loads((work / "video_index.json").read_text(encoding="utf-8"))
    ids = [i for i in index if not args.only or i in args.only]

    detector = None if args.aggregate_only else FaceDetector(work / "models")
    log(f"## Step 05 — visual + audio audit ({len(ids)} recordings, 1 frame / {args.every:.0f}s"
        f"{', detector ' + detector.kind if detector else ', aggregate-only'})")
    measured_all, timings = {}, {}
    for n, sid in enumerate(ids, 1):
        e = index[sid]
        cfile = cache / f"{sid}.json"
        t0 = time.perf_counter()
        if args.aggregate_only and cfile.is_file():
            measured = json.loads(cfile.read_text(encoding="utf-8"))
        else:
            try:
                measured = measure_video(e["video_path"], detector, args.every)
            except Exception as ex:
                log(f"[{n}/{len(ids)}] {sid}: visual FAILED ({type(ex).__name__})")
                continue
            cfile.write_text(json.dumps(measured), encoding="utf-8")
        measured_all[sid] = measured
        timings[sid] = time.perf_counter() - t0
        log(f"[{n}/{len(ids)}] {sid}: measured {len(measured.get('frames', []))} frames ({timings[sid]:.0f}s)")

    template, n_ref = make_logo_template(cache, work / "models", index)
    log(f"- logo template from {n_ref} reference branded-background ROIs ({', '.join(LOGO_REFERENCE_IDS)})")
    results = {}
    for sid, measured in measured_all.items():
        e = index[sid]
        visual = aggregate_visual(measured, logo_template=template)
        t1 = time.perf_counter()
        try:
            audio = extract_audio_features(work / "audio" / f"{sid}.wav")
        except Exception as ex:
            audio = {"error": type(ex).__name__}
        t_aud = time.perf_counter() - t1
        results[sid] = {"label": e["label"], "sub_origin": e.get("sub_origin"), "visual": visual, "audio": audio,
                        "detector": measured.get("detector")}
        log(f"- {sid}: aggregated | audio {t_aud:.1f}s")

    out_path = work / "05_visual_audio_features.json"
    prev = json.loads(out_path.read_text(encoding="utf-8")) if (args.only and out_path.is_file()) else {}
    prev.update(results)
    out_path.write_text(json.dumps(prev, indent=1), encoding="utf-8")

    vrows, arows = [], []
    for sid, r in results.items():
        v, a = r["visual"], r["audio"]
        lab = r["label"] + (f" ({r['sub_origin']})" if r.get("sub_origin") else "")
        vrows.append({"short_id": sid, "label": lab, "frames": v.get("sampled_frames"),
                      "full%": v.get("fullscreen_cam_pct"), "pip%": v.get("screenshare_pip_pct"),
                      "cam_off%": v.get("camera_off_or_missing_pct"), "off_streak_s": v.get("max_camera_off_streak_sec"),
                      "hidden%": v.get("pip_panel_hidden_pct"),
                      "branded-background%": v.get("bg_8x8_frame_pct"), "8x8_dark%": v.get("bg8_by_dark_rule_pct"),
                      "8x8_logo%": v.get("bg8_by_logo_match_pct"), "face_on_8x8%": v.get("tutor_face_visible_when_8x8_pct"),
                      "face_any%": v.get("overall_face_visible_pct"), "face_lum": v.get("tutor_face_luminance_mean"),
                      "bleed%": v.get("bg_color_bleed_mean_pct"), "glitch%": v.get("bg_glitch_frames_pct"),
                      "sharp": v.get("roi_sharpness_mean"), "flicker": v.get("bg_flicker_steady_face"),
                      "F_no8x8": yn(v.get("flag_no_8x8_background")), "F_noface": yn(v.get("flag_face_not_visible")),
                      "F_camoff": yn(v.get("flag_intermittent_camera_off")), "F_dark": yn(v.get("flag_poor_lighting")),
                      "F_bgclarity": yn(v.get("flag_bg_clarity_issue"))})
        arows.append({"short_id": sid, "label": lab, "speech%": a.get("speech_ratio_pct"),
                      "gaps>5s": a.get("silence_gaps_gt_5s_count"), "gaps>10s": a.get("silence_gaps_gt_10s_count"),
                      "dead_air_s": a.get("dead_air_total_sec"), "max_gap_s": a.get("max_silence_gap_sec"),
                      "max_gap_ts": a.get("max_silence_gap_ts"), "rms_db": a.get("rms_mean_db"),
                      "rms_std": a.get("rms_std_db"), "dyn_db": a.get("dynamic_range_db"),
                      "f0_med": a.get("pitch_f0_hz_median"), "f0_iqr": a.get("pitch_f0_hz_iqr"),
                      "f0_cv": a.get("pitch_variation_cv"), "liveliness": a.get("energy_liveliness_score")})
    log("\n## Table 1 — Visual metrics & flags\n")
    log(table(vrows, list(vrows[0]) if vrows else []))
    log("\n## Table 2 — Audio prosody & silence\n")
    log(table(arows, list(arows[0]) if arows else []))
    log(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
