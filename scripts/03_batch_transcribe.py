"""Step 03 — batch-transcribe all recordings with the locked config (small / en / beam 1).

Also: distil-small.en comparison on good_1 (GPU full + CPU 4-thread 5-min
slice) and a demo_1 language="ml" pass.

PRIVACY: prints only numeric stats. Transcripts go to
<data>/_work/transcripts_small_en/ (outside the repo) and are never printed.

Usage:  .venv\\Scripts\\python scripts\\03_batch_transcribe.py [--only ID ...] [--skip-extras]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.audio_extract import extract_wav_16k_mono, slice_wav  # noqa: E402
from src.transcribe import release_models, transcribe_audio  # noqa: E402
from src.utils.ids import resolve_short_id  # noqa: E402
from src.wer import normalize_words, parse_vtt_cues, reference_words, wer  # noqa: E402


def _safe_excepthook(exc_type, exc, tb):
    print(f"[FATAL] {exc_type.__name__} (details suppressed for privacy)", file=sys.stderr)


sys.excepthook = _safe_excepthook

GPU = dict(device="cuda", compute_type="int8_float16")
LOCKED = dict(model_name="small", language="en", beam_size=1)
DISTIL = "Systran/faster-distil-whisper-small.en"
BENCH_ID = "good_1"
CLASS_MIN = 50.0


def log(msg: str) -> None:
    print(msg, flush=True)


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def vtt_compare(vtt_path, result, start=0.0, end=None) -> dict:
    if not vtt_path:
        return {"vtt_words": None, "word_ratio": None, "wer_pct": None}
    ref = reference_words(parse_vtt_cues(vtt_path), start, end)
    hyp = normalize_words(" ".join(s["text"] for s in result["segments"]))
    return {"vtt_words": len(ref),
            "word_ratio": round(len(hyp) / len(ref), 3) if ref else None,
            "wer_pct": round(100 * wer(ref, hyp), 1) if ref else None}


def table(rows, cols) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join("—" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="restrict to these short_ids")
    ap.add_argument("--skip-extras", action="store_true", help="skip distil + demo_1 ml passes")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    work = Path(os.environ.get("CHALLENGE_DATA_DIR") or r"C:\challenge-data") / "_work"
    audio_dir, out_dir, cache_dir = work / "audio", work / "transcripts_small_en", work / "transcribe_cache"
    index = json.loads((work / "video_index.json").read_text(encoding="utf-8"))
    ids = [i for i in index if not args.only or i in args.only]
    global BENCH_ID
    try:
        BENCH_ID = resolve_short_id(index, BENCH_ID)
    except KeyError:
        pass

    t_start = time.perf_counter()
    log(f"## Step 03 — batch transcription ({len(ids)} recordings, "
        f"{LOCKED['model_name']}/{LOCKED['language']}/beam {LOCKED['beam_size']}, {GPU['device']}/{GPU['compute_type']})")
    rows = []
    for n, sid in enumerate(ids, 1):
        e = index[sid]
        row = {"short_id": sid, "label": e["label"] + (f" ({e['sub_origin']})" if e.get("sub_origin") else "")}
        try:
            ext = extract_wav_16k_mono(e["video_path"], audio_dir / f"{sid}.wav")
        except Exception as ex:
            row["error"] = f"extract:{type(ex).__name__}"
            log(f"[{n}/{len(ids)}] {sid}: audio extraction FAILED ({type(ex).__name__})")
            rows.append(row)
            continue
        try:
            res = transcribe_audio(audio_dir / f"{sid}.wav", cache_dir=cache_dir, **LOCKED, **GPU)
        except Exception as ex:
            row["error"] = f"transcribe:{type(ex).__name__}"
            log(f"[{n}/{len(ids)}] {sid}: transcription FAILED ({type(ex).__name__})")
            rows.append(row)
            continue
        save_json(out_dir / f"{sid}.json", {"short_id": sid, **{k: v for k, v in res.items() if k != "cache_hit"}})
        row.update(dur_min=round(ext["duration_sec"] / 60, 1), wav_s=round(ext["extract_time_sec"], 1),
                   whisper_s=res["wall_time_sec"], rtf=res["rtf"], seg_count=res["segment_count"],
                   whisper_words=res["word_count"], non_latin_pct=res["non_latin_pct"],
                   cache_hit=res["cache_hit"], **vtt_compare(e.get("vtt_path"), res))
        rows.append(row)
        log(f"[{n}/{len(ids)}] {sid}: {row['dur_min']} min | wav {row['wav_s']}s | whisper {row['whisper_s']}s "
            f"(RTF {row['rtf']}){' [cache]' if res['cache_hit'] else ''} | segs {row['seg_count']} | "
            f"words {row['whisper_words']}/{row['vtt_words']} | WER {row['wer_pct']}%")
        del res
        gc.collect()

    extras = {}
    if not args.skip_extras:
        # demo_1 with language="ml" (same small model instance, different decode language)
        if "demo_1" in index:
            try:
                ml = transcribe_audio(audio_dir / "demo_1.wav", model_name="small", language="ml",
                                      beam_size=1, cache_dir=cache_dir, **GPU)
                save_json(out_dir / "demo_1_ml.json",
                          {"short_id": "demo_1", **{k: v for k, v in ml.items() if k != "cache_hit"}})
                en_row = next((r for r in rows if r["short_id"] == "demo_1"), {})
                extras["demo_1"] = {
                    "en": {k: en_row.get(k) for k in ("seg_count", "whisper_words", "non_latin_pct", "wer_pct", "whisper_s")},
                    "ml": {"seg_count": ml["segment_count"], "whisper_words": ml["word_count"],
                           "non_latin_pct": ml["non_latin_pct"], "whisper_s": ml["wall_time_sec"],
                           **{k: v for k, v in vtt_compare(index["demo_1"].get("vtt_path"), ml).items() if k == "wer_pct"}},
                }
                log(f"- demo_1 ml pass: segs {ml['segment_count']} | words {ml['word_count']} | "
                    f"non-Latin {ml['non_latin_pct']}% | {ml['wall_time_sec']}s")
                del ml
            except Exception as ex:
                extras["demo_1"] = {"error": type(ex).__name__}
                log(f"- demo_1 ml pass FAILED ({type(ex).__name__})")
        release_models()

        # distil-small.en vs small/en on the benchmark recording
        if BENCH_ID in index:
            e = index[BENCH_ID]
            wav = audio_dir / f"{BENCH_ID}.wav"
            d = {}
            try:
                g = transcribe_audio(wav, model_name=DISTIL, language="en", beam_size=1, **GPU)
                d["gpu_full"] = {"wall_time_sec": g["wall_time_sec"], "rtf": g["rtf"],
                                 "word_count": g["word_count"], **vtt_compare(e.get("vtt_path"), g)}
                log(f"- distil-small.en GPU full: {g['wall_time_sec']}s (RTF {g['rtf']}) | words {g['word_count']} | "
                    f"WER {d['gpu_full']['wer_pct']}%")
                del g
            except Exception as ex:
                d["gpu_full"] = {"error": type(ex).__name__}
                log(f"- distil-small.en GPU FAILED ({type(ex).__name__})")
            release_models()
            try:
                wav5 = audio_dir / f"{BENCH_ID}_0-300s.wav"
                if not wav5.is_file():
                    slice_wav(wav, wav5, 0, 300)
                c = transcribe_audio(wav5, model_name=DISTIL, language="en", beam_size=1,
                                     device="cpu", compute_type="int8", cpu_threads=4)
                d["cpu4_5min"] = {"wall_time_sec": c["wall_time_sec"], "rtf": c["rtf"],
                                  "est_min_per_50min_class": round(c["rtf"] * CLASS_MIN, 2),
                                  "word_count": c["word_count"], **vtt_compare(e.get("vtt_path"), c, 0, 300)}
                log(f"- distil-small.en CPU 4t 5-min: {c['wall_time_sec']}s (RTF {c['rtf']}) | "
                    f"est {d['cpu4_5min']['est_min_per_50min_class']} min / 50-min class")
                del c
            except Exception as ex:
                d["cpu4_5min"] = {"error": type(ex).__name__}
                log(f"- distil-small.en CPU FAILED ({type(ex).__name__})")
            release_models()
            small_row = next((r for r in rows if r["short_id"] == BENCH_ID), {})
            d["small_en_gpu_full"] = {k: small_row.get(k) for k in ("whisper_s", "rtf", "whisper_words", "wer_pct")}
            extras["distil_vs_small"] = d

    release_models()
    total_min = (time.perf_counter() - t_start) / 60
    save_json(work / "03_batch_transcribe_metrics.json",
              {"config": {**LOCKED, **GPU}, "total_wall_min": round(total_min, 1), "recordings": rows, "extras": extras})

    log("\n## Recordings\n")
    log(table(rows, ["short_id", "label", "dur_min", "wav_s", "whisper_s", "rtf", "seg_count",
                     "whisper_words", "vtt_words", "word_ratio", "wer_pct"]))
    if "distil_vs_small" in extras:
        d = extras["distil_vs_small"]
        s, g, c = d["small_en_gpu_full"], d.get("gpu_full", {}), d.get("cpu4_5min", {})
        log(f"\n## distil-small.en vs small (en) on {BENCH_ID}\n")
        log(table([
            {"model": "small (en)", "gpu_full_s": s.get("whisper_s"), "gpu_rtf": s.get("rtf"),
             "words": s.get("whisper_words"), "wer_pct": s.get("wer_pct")},
            {"model": "distil-small.en", "gpu_full_s": g.get("wall_time_sec"), "gpu_rtf": g.get("rtf"),
             "words": g.get("word_count"), "wer_pct": g.get("wer_pct"),
             "cpu4_5min_s": c.get("wall_time_sec"), "cpu4_rtf": c.get("rtf"),
             "cpu4_est_min_per_50min": c.get("est_min_per_50min_class"), "cpu4_5min_wer_pct": c.get("wer_pct")},
        ], ["model", "gpu_full_s", "gpu_rtf", "words", "wer_pct", "cpu4_5min_s", "cpu4_rtf",
            "cpu4_est_min_per_50min", "cpu4_5min_wer_pct"]))
    if "demo_1" in extras and "en" in extras["demo_1"]:
        log("\n## demo_1 — en vs ml\n")
        log(table([{"language": k, **v} for k, v in extras["demo_1"].items()],
                  ["language", "seg_count", "whisper_words", "non_latin_pct", "wer_pct", "whisper_s"]))
    log(f"\nTotal wall time: {total_min:.1f} min | metrics -> {work / '03_batch_transcribe_metrics.json'}")


if __name__ == "__main__":
    main()
