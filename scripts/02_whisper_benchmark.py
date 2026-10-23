"""Step 02 — faster-whisper speed / accuracy benchmark (GPU vs 4-thread CPU).

PRIVACY: prints only numbers, language codes, timings and WER. Whisper output is
cached under <data>/_work/transcripts/ (outside the repo) and never printed.

Usage:  .venv\\Scripts\\python scripts\\02_whisper_benchmark.py [--skip-cpu] [--skip-demo]
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

from src.utils.cuda_setup import register_cuda_dlls  # noqa: E402

register_cuda_dlls()

from faster_whisper import WhisperModel  # noqa: E402

from src.audio_extract import extract_wav_16k_mono, slice_wav  # noqa: E402
from src.utils.ids import resolve_short_id  # noqa: E402
from src.wer import non_latin_pct, normalize_words, parse_vtt_cues, reference_words, wer  # noqa: E402


def _safe_excepthook(exc_type, exc, tb):
    print(f"[FATAL] {exc_type.__name__} (details suppressed for privacy)", file=sys.stderr)


sys.excepthook = _safe_excepthook

PRIMARY_ID = "good_1"
DEMO_IDS = ("demo_1", "demo_2")
TURBO_ALIASES = ("large-v3-turbo", "turbo", "deepdml/faster-whisper-large-v3-turbo-ct2")
VAD = dict(vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
CLASS_MIN = 50.0


def log(msg: str) -> None:
    print(msg, flush=True)


def rss_gb() -> float | None:
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 2**30, 2)
    except Exception:
        return None


def load_model(aliases, device, compute_type, cpu_threads):
    last_err = None
    for alias in aliases:
        try:
            t0 = time.perf_counter()
            kw = {"cpu_threads": cpu_threads} if cpu_threads else {}
            m = WhisperModel(alias, device=device, compute_type=compute_type, **kw)
            return m, alias, time.perf_counter() - t0
        except Exception as e:  # model-hub / alias errors carry no dataset content
            last_err = e
            log(f"    model alias '{alias}' failed: {type(e).__name__}: {str(e)[:120]}")
    raise RuntimeError(f"all aliases failed ({type(last_err).__name__})")


def run_config(name, aliases, device, compute_type, wav, audio_sec, ref_words, cache_dir,
               cache_key, beam_size=1, cpu_threads=0, language=None) -> dict:
    log(f"\n>> {name}: {aliases[0]} | {device}/{compute_type} | beam={beam_size} | "
        f"{audio_sec / 60:.1f} min audio")
    row = {"config_name": name, "device": device, "compute_type": compute_type,
           "beam_size": beam_size, "cpu_threads": cpu_threads or None,
           "audio_min_tested": round(audio_sec / 60, 2), "vtt_ref_words": len(ref_words)}
    try:
        model, alias, load_s = load_model(aliases, device, compute_type, cpu_threads)
    except Exception as e:
        row["error"] = type(e).__name__
        log(f"    FAILED to load ({type(e).__name__})")
        return row
    row.update(model=alias, load_sec=round(load_s, 1))
    try:
        t0 = time.perf_counter()
        segs_iter, info = model.transcribe(str(wav), beam_size=beam_size, language=language, **VAD)
        segs = [(s.start, s.end, s.text) for s in segs_iter]  # generator: decoding happens here
        wall = time.perf_counter() - t0
    except Exception as e:
        row["error"] = type(e).__name__
        log(f"    FAILED during transcribe ({type(e).__name__})")
        segs, info, wall = None, None, None
    finally:
        row["rss_gb_after"] = rss_gb()
        del model
        gc.collect()
    if segs is None:
        return row

    hyp_text = " ".join(t for _, _, t in segs)
    hyp = normalize_words(hyp_text)
    rtf = wall / audio_sec
    row["non_latin_pct"] = non_latin_pct(hyp_text)
    row.update(
        wall_time_sec=round(wall, 1), rtf=round(rtf, 4),
        est_time_per_50min_class_min=round((load_s + rtf * CLASS_MIN * 60) / 60, 2),
        segments=len(segs), whisper_words=len(hyp),
        word_count_ratio=round(len(hyp) / len(ref_words), 3) if ref_words else None,
        wer_pct=round(100 * wer(ref_words, hyp), 1) if ref_words else None,
        detected_lang=info.language, lang_prob=round(info.language_probability, 3),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{cache_key}.json").write_text(json.dumps(
        {"config": row, "segments": [{"start": round(a, 2), "end": round(b, 2), "text": t}
                                     for a, b, t in segs]}, ensure_ascii=False), encoding="utf-8")
    log(f"    load {load_s:.1f}s | transcribe {wall:.1f}s | RTF {rtf:.3f} | "
        f"WER {row['wer_pct']}% | words {len(hyp)}/{len(ref_words)} | "
        f"lang {info.language} ({info.language_probability:.2f}) | non-Latin {row['non_latin_pct']}% | "
        f"RSS {row['rss_gb_after']} GB")
    return row


def densest_window(cues, length: float = 180.0, min_start: float = 120.0) -> float:
    """Start time of the `length`-second window (>= min_start) with the most VTT words."""
    best_start, best_n = min_start, -1
    starts = sorted({max(min_start, s) for s, _, _ in cues if s >= min_start} | {min_start})
    for st in starts:
        n = len(reference_words(cues, st, st + length))
        if n > best_n:
            best_start, best_n = st, n
    return best_start


def pick_demo_model(gpu_rows: list[dict]) -> dict | None:
    """Fastest GPU run whose WER is within 2 pp of the best GPU WER."""
    ok = [r for r in gpu_rows if r.get("wer_pct") is not None]
    if not ok:
        return None
    best = min(r["wer_pct"] for r in ok)
    return min((r for r in ok if r["wer_pct"] <= best + 2.0), key=lambda r: r["rtf"])


def fmt_table(rows: list[dict]) -> str:
    cols = ["config_name", "device", "compute_type", "audio_min_tested", "wall_time_sec", "rtf",
            "est_time_per_50min_class_min", "whisper_words", "vtt_ref_words", "word_count_ratio",
            "wer_pct", "non_latin_pct", "detected_lang"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        vals = []
        for c in cols:
            if c == "detected_lang":
                v = f"{r['detected_lang']} ({r['lang_prob']:.2f})" if r.get("detected_lang") else "—"
            elif c == "wall_time_sec" and r.get("error"):
                v = f"ERROR: {r['error']}"
            else:
                v = r.get(c)
                v = "—" if v is None else v
            vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cpu", action="store_true")
    ap.add_argument("--skip-demo", action="store_true")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    data_dir = Path(os.environ.get("CHALLENGE_DATA_DIR") or r"C:\challenge-data")
    work = data_dir / "_work"
    audio_dir, tx_dir = work / "audio", work / "transcripts"
    index = json.loads((work / "video_index.json").read_text(encoding="utf-8"))

    # ---- primary recording -------------------------------------------------
    global PRIMARY_ID
    PRIMARY_ID = resolve_short_id(index, PRIMARY_ID)
    entry = index[PRIMARY_ID]
    if not entry.get("vtt_path"):
        raise SystemExit(f"{PRIMARY_ID} has no paired VTT")
    wav = audio_dir / f"{PRIMARY_ID}.wav"
    log(f"## Whisper benchmark — primary recording {PRIMARY_ID}")
    try:
        ext = extract_wav_16k_mono(entry["video_path"], wav)
    except Exception as e:
        raise SystemExit(f"audio extraction failed for {PRIMARY_ID} ({type(e).__name__})")
    log(f"- WAV: {ext['duration_sec'] / 60:.2f} min, {ext['wav_size_mb']:.1f} MB, "
        f"extract {ext['extract_time_sec']:.1f}s{' (cached)' if ext['reused'] else ''}, RSS {rss_gb()} GB")
    audio_sec = ext["duration_sec"]

    cues = parse_vtt_cues(entry["vtt_path"])
    ref_full = reference_words(cues)
    ref_5 = reference_words(cues, 0, 300)
    log(f"- VTT reference: {len(cues)} cues, {len(ref_full)} words (full), {len(ref_5)} words (first 5 min), "
        f"non-Latin {non_latin_pct(' '.join(t for _, _, t in cues))}%")

    rows: list[dict] = []
    # Auto-detect (spec) plus language="en" control runs: small/base mis-detect
    # this class as Malayalam, which inflates WER.
    gpu_cfgs = [("gpu_large-v3-turbo_b1", TURBO_ALIASES, None), ("gpu_small_b1", ("small",), None),
                ("gpu_base_b1", ("base",), None), ("gpu_small_b1_en", ("small",), "en"),
                ("gpu_base_b1_en", ("base",), "en")]
    for name, aliases, lang in gpu_cfgs:
        rows.append(run_config(name, aliases, "cuda", "int8_float16", wav, audio_sec, ref_full,
                               tx_dir, f"{PRIMARY_ID}.{name}", language=lang))

    if not args.skip_cpu:
        wav5 = audio_dir / f"{PRIMARY_ID}_0-300s.wav"
        sl = slice_wav(wav, wav5, 0, 300)
        for size in ("small", "base"):
            for lang in (None, "en"):
                name = f"cpu4_{size}_b1_5min" + ("_en" if lang else "")
                rows.append(run_config(name, (size,), "cpu", "int8", wav5, sl["duration_sec"], ref_5,
                                       tx_dir, f"{PRIMARY_ID}.{name}", cpu_threads=4, language=lang))

    # ---- multilingual demo check -------------------------------------------
    chosen = pick_demo_model([r for r in rows if r["device"] == "cuda"])
    if not args.skip_demo and chosen:
        log(f"\n## Language check — model chosen: {chosen['model']} "
            f"(fastest GPU run within 2 pp of best WER)")
        for did in DEMO_IDS:
            d = index.get(did) if did in index else index.get(resolve_short_id(index, did))
            if not d:
                log(f"- {did}: not in index")
                continue
            try:
                extract_wav_16k_mono(d["video_path"], audio_dir / f"{did}.wav")
            except Exception as e:
                log(f"- {did}: audio extraction failed ({type(e).__name__})")
                continue
            # Densest 180 s of VTT speech at/after t=120 s (a fixed 120-300 s window was
            # silent for demo_1).
            dcues = parse_vtt_cues(d["vtt_path"]) if d.get("vtt_path") else []
            st = densest_window(dcues) if dcues else 120.0
            clip = audio_dir / f"{did}_langclip.wav"
            sl = slice_wav(audio_dir / f"{did}.wav", clip, st, st + 180)
            ref = reference_words(dcues, st, st + 180)
            rows.append(run_config(f"{did}_langcheck_{int(st)}-{int(st) + 180}s", (chosen["model"],), "cuda",
                                   "int8_float16", clip, sl["duration_sec"], ref, tx_dir,
                                   f"{did}.langcheck", language=None))

    out = {"primary_id": PRIMARY_ID, "audio_min": round(audio_sec / 60, 2),
           "vtt_ref_words_full": len(ref_full), "vtt_ref_words_5min": len(ref_5),
           "demo_model": chosen["model"] if chosen else None, "runs": rows}
    (work / "02_whisper_benchmark.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    log("\n## Summary\n")
    log(fmt_table(rows))
    log(f"\nSaved metrics -> {work / '02_whisper_benchmark.json'}")


if __name__ == "__main__":
    main()
