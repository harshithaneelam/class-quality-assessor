"""Step 01 — environment audit + privacy-safe dataset inspection.

PRIVACY: this script reads C:\\challenge-data locally but prints ONLY aggregate
metrics, short IDs and sanitised filename labels (batch / level / date). It
never prints transcript text, VTT cue lines, speaker names or raw filenames.
The short_id -> real path map is written to <data>/_work/video_index.json,
outside the repo.

Usage:  .venv\\Scripts\\python scripts\\01_env_and_data_check.py [--no-whisper-smoke]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Silence OpenCV / FFmpeg native logging: their warnings can echo file paths.
os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.cuda_setup import register_cuda_dlls  # noqa: E402


def _safe_excepthook(exc_type, exc, tb):
    # Tracebacks can contain dataset paths (i.e. tutor names) — print type only.
    print(f"[FATAL] {exc_type.__name__} (details suppressed for privacy)", file=sys.stderr)


sys.excepthook = _safe_excepthook

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}
SUB_EXTS = {".vtt"}
SKIP_DIRS = {"_work", "frames"}

# ---------------------------------------------------------------- folder map
# (normalised folder key, target label, sub-origin, short-id prefix)
FOLDER_RULES = [
    (re.compile(r"quality[\s_-]*issue", re.I), "quality_issue", "Poor", "quality_issue", "poor_qi"),
    (re.compile(r"\bweak", re.I), "weak", "Poor", "weak", "poor_weak"),
    (re.compile(r"\bpoor", re.I), "poor", "Poor", "poor", "poor"),
    (re.compile(r"excellent", re.I), "excellent", "Excellent", "", "exc"),
    (re.compile(r"\bgood", re.I), "good", "Good", "", "good"),
    (re.compile(r"average", re.I), "average", "Average", "", "avg"),
    (re.compile(r"demo", re.I), "demo", "Unlabelled_Demo", "", "demo"),
]


def classify_folder(rel_parts: tuple[str, ...]):
    """Match the deepest folder component first; returns (key, label, origin, prefix)."""
    for part in reversed(rel_parts):
        for rx, key, label, origin, prefix in FOLDER_RULES:
            if rx.search(part):
                return key, label, origin, prefix
    return "unmapped", "Unmapped", "", "unk"


# ------------------------------------------------------- filename sanitiser
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
MONTH_RX = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
SAFE_WORDS = {
    "part", "pt", "class", "session", "recording", "rec", "batch", "level", "beginner",
    "beginners", "intermediate", "advanced", "demo", "zoom", "meeting", "video", "final",
    "copy", "am", "pm", "gmt", "ist", "utc", "the", "and", "of", "on", "at", "to", "by",
    "with", "lesson", "day", "week", "new", "old", "full", "hd", "mp4", "vtt", "transcript",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun", "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday", "st", "nd", "rd", "th", "online", "live",
    "kids", "children", "english", "maths", "math", "science", "coding",
} | set(MONTHS) | {
    "january", "february", "march", "april", "june", "july", "august", "september",
    "sept", "october", "november", "december",
}


def _norm_date(d: int, m: int, y: int) -> str | None:
    if y < 100:
        y += 2000
    if m > 12 and d <= 12:
        d, m = m, d
    if not (1 <= d <= 31 and 1 <= m <= 12 and 2000 <= y <= 2100):
        return None
    return f"{d:02d}/{m:02d}/{y:04d}"


def extract_date(stem: str) -> tuple[str | None, str]:
    """Return (dd/mm/yyyy or None, stem with the date removed)."""
    pats = [
        (rf"(\d{{4}})[-_.](\d{{1,2}})[-_.](\d{{1,2}})", lambda g: _norm_date(int(g[2]), int(g[1]), int(g[0]))),
        (rf"(\d{{1,2}})[-_./](\d{{1,2}})[-_./](\d{{2,4}})", lambda g: _norm_date(int(g[0]), int(g[1]), int(g[2]))),
        (rf"(\d{{1,2}})(?:st|nd|rd|th)?[\s_-]*{MONTH_RX}[\s_,-]*(\d{{4}})",
         lambda g: _norm_date(int(g[0]), MONTHS[g[1][:3].lower()], int(g[2]))),
        (rf"{MONTH_RX}[\s_-]*(\d{{1,2}})(?:st|nd|rd|th)?[\s_,-]*(\d{{4}})",
         lambda g: _norm_date(int(g[1]), MONTHS[g[0][:3].lower()], int(g[2]))),
        (r"(20\d{2})(\d{2})(\d{2})", lambda g: _norm_date(int(g[2]), int(g[1]), int(g[0]))),
    ]
    for pat, conv in pats:
        m = re.search(pat, stem, re.I)
        if m:
            d = conv(m.groups())
            if d:
                return d, stem[:m.start()] + " " + stem[m.end():]
    return None, stem


def sanitize_name(stem: str) -> dict:
    """Extract only safe tokens from a filename stem. Never returns raw text."""
    batch_m = re.search(r"\bB[-_ ]?(\d{2,5})\b", stem, re.I) or re.search(r"B(\d{3,5})", stem)
    batch = f"B{batch_m.group(1)}" if batch_m else None
    lvl_m = re.search(r"\b(beginners?|intermediate|advanced)\b", stem, re.I)
    level = lvl_m.group(1).rstrip("s").capitalize() if lvl_m else None
    date, rest = extract_date(stem)
    # "Part 2" / "Part-2", but not "Part - 28-08-2026" (a date, not a part number).
    part_m = re.search(r"\bpart[\s_-]?(\d{1,2})\b(?![-_./]\d)", stem, re.I)
    part = int(part_m.group(1)) if part_m else None
    if batch_m:
        rest = rest.replace(batch_m.group(0), " ")
    residual = [t for t in re.findall(r"[A-Za-z]+", rest)
                if t.lower() not in SAFE_WORDS and not re.fullmatch(r"b\d*", t, re.I)]
    return {"batch": batch, "level": level, "date": date, "part": part,
            "has_other_tokens": bool(residual)}


def safe_label(folder_key: str, s: dict) -> str:
    bits = [folder_key, s["batch"] or "B?"]
    if s["level"]:
        bits.append(s["level"])
    if s["part"] is not None:
        bits.append(f"Part {s['part']}")
    out = " ".join(bits)
    if s["date"]:
        out += f" ({s['date']})"
    if s["has_other_tokens"]:
        out += " [REDACTED_NAME]"
    return out


# ---------------------------------------------------------------- VTT stats
TIMING_RX = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3}\s*-->\s*\d{1,2}:\d{2}")
VOICE_RX = re.compile(r"<v(?:\.[^ >]*)?\s+([^>]+)>", re.I)
PREFIX_RX = re.compile(r"^\s*([A-Za-z][\w .'\-()]{0,40}?)\s*:\s+\S")
TAG_RX = re.compile(r"<[^>]+>")
END_RX = re.compile(r"-->\s*(?:(\d{1,2}):)?(\d{2}):(\d{2}[.,]\d{3})")


def vtt_stats(path: Path) -> dict:
    """Aggregate counts only. Speaker names are hashed into a set, never returned."""
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return {"ok": False}
    cues, cur, end_s = [], None, 0.0
    for line in text.splitlines():
        if TIMING_RX.search(line):
            cur = []
            cues.append(cur)
            em = END_RX.search(line)
            if em:
                h, mi, s = em.groups()
                end_s = max(end_s, int(h or 0) * 3600 + int(mi) * 60 + float(s.replace(",", ".")))
        elif not line.strip():
            cur = None
        elif cur is not None:
            cur.append(line)
    words, with_prefix, speakers = 0, 0, set()
    for lines in cues:
        body = " ".join(lines)
        spk = None
        vm = VOICE_RX.search(body)
        if vm:
            spk = vm.group(1)
        else:
            pm = PREFIX_RX.match(TAG_RX.sub("", lines[0])) if lines else None
            if pm and len(pm.group(1).split()) <= 4:
                spk = pm.group(1)
        clean = TAG_RX.sub(" ", body)
        if spk:
            with_prefix += 1
            speakers.add(hash(spk.strip().lower()))
            if not vm:
                clean = clean.split(":", 1)[1] if ":" in clean else clean
        words += len(clean.split())
    n = len(cues)
    return {"ok": True, "cue_count": n, "word_count": words, "end_s": round(end_s, 1),
            "speaker_prefix_count": len(speakers),
            "pct_cues_with_prefix": round(100 * with_prefix / n, 1) if n else 0.0}


# ------------------------------------------------------------- environment
def find_ff_tool(name: str) -> tuple[str | None, bool]:
    """Return (path, on_path)."""
    p = shutil.which(name)
    if p:
        return p, True
    local = os.environ.get("LOCALAPPDATA", "")
    candidates = [
        Path(local) / "Microsoft" / "WinGet" / "Links" / f"{name}.exe",
        Path("C:/ffmpeg/bin") / f"{name}.exe",
        Path("C:/Program Files/ffmpeg/bin") / f"{name}.exe",
        Path("C:/ProgramData/chocolatey/bin") / f"{name}.exe",
        Path.home() / "scoop" / "shims" / f"{name}.exe",
    ]
    candidates += [Path(g) for g in glob.glob(
        str(Path(local) / "Microsoft" / "WinGet" / "Packages" / "*ffmpeg*" / "**" / f"{name}.exe"),
        recursive=True)]
    for c in candidates:
        if c.is_file():
            return str(c), False
    return None, False


def ram_gb() -> tuple[float, float] | None:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total / 2**30, vm.available / 2**30
    except Exception:
        return None


def env_audit(whisper_smoke: bool) -> dict:
    print("## Environment audit\n")
    in_venv = sys.prefix != sys.base_prefix
    print(f"- Python: {sys.version.split()[0]} ({'venv: ' + Path(sys.prefix).name if in_venv else 'system interpreter'})")
    print(f"- OS: {platform.system()} {platform.release()} (build {platform.version()}), {platform.machine()}")
    print(f"- CPU logical cores: {os.cpu_count()}  ({platform.processor() or 'n/a'})")
    r = ram_gb()
    print(f"- RAM: {r[0]:.1f} GB total, {r[1]:.1f} GB available" if r else "- RAM: unknown (psutil missing)")

    ff = {}
    for tool in ("ffmpeg", "ffprobe"):
        p, on_path = find_ff_tool(tool)
        ff[tool] = p
        if p and on_path:
            print(f"- {tool}: OK on PATH")
        elif p:
            print(f"- {tool}: FOUND but NOT on PATH -> {p}")
        else:
            print(f"- {tool}: MISSING (install: `winget install --id Gyan.FFmpeg -e`, then reopen the shell)")

    try:
        nv = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                             "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        print(f"- nvidia-smi: {nv.stdout.strip() or 'no output'}")
    except Exception as e:
        print(f"- nvidia-smi: unavailable ({type(e).__name__})")

    dll_dirs = register_cuda_dlls()
    print(f"- cuda_setup: registered {len(dll_dirs)} DLL dir(s): "
          + ", ".join(Path(d).parent.name for d in dll_dirs))

    cuda_ok = False
    try:
        import ctranslate2
        n = ctranslate2.get_cuda_device_count()
        cuda_ok = n > 0
        types = sorted(ctranslate2.get_supported_compute_types("cuda")) if cuda_ok else []
        print(f"- ctranslate2 {ctranslate2.__version__}: CUDA devices = {n}; compute types = {types}")
    except Exception as e:
        print(f"- ctranslate2: import/CUDA check failed ({type(e).__name__}: {e})")

    if whisper_smoke and cuda_ok:
        # Synthetic silence through Whisper 'tiny' on GPU: exercises cuBLAS + cuDNN for real.
        try:
            import numpy as np
            from faster_whisper import WhisperModel
            m = WhisperModel("tiny", device="cuda", compute_type="float16")
            segs, info = m.transcribe(np.zeros(16000 * 2, dtype=np.float32), beam_size=1)
            list(segs)
            print("- faster-whisper GPU smoke test (tiny, float16, synthetic silence): OK")
        except Exception as e:
            print(f"- faster-whisper GPU smoke test: FAILED ({type(e).__name__}: {str(e)[:160]})")

    for mod in ("cv2", "librosa", "soundfile", "numpy", "scipy", "spacy", "rapidfuzz",
                "google.genai", "pydantic", "gradio", "faster_whisper"):
        try:
            m = __import__(mod, fromlist=["_"])
            print(f"- {mod}: {getattr(m, '__version__', 'ok')}")
        except Exception as e:
            print(f"- {mod}: NOT importable ({type(e).__name__})")
    print()
    return {"ffprobe": ff.get("ffprobe")}


# ------------------------------------------------------------ video helpers
def ffprobe_duration(ffprobe: str | None, path: Path) -> float | None:
    if not ffprobe:
        return None
    try:
        out = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration",
                              "-of", "default=nw=1:nk=1", str(path)],
                             capture_output=True, text=True, timeout=60)
        return float(out.stdout.strip())
    except Exception:
        return None


def video_metrics(path: Path, ffprobe: str | None, frames_dir: Path, short_id: str, cascade) -> dict:
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"open_ok": False}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = ffprobe_duration(ffprobe, path)
    if not dur and fps > 0 and n_frames > 0:
        dur = n_frames / fps
    dur = dur or 0.0

    targets = {"t60s": min(60.0, dur * 0.5), "10pct": 0.10 * dur, "50pct": 0.50 * dur, "85pct": 0.85 * dur}
    brightness, faces, saved = [], [], 0
    for tag, t in targets.items():
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok and fps > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, frame = cap.read()
        if not ok or frame is None:
            continue
        cv2.imwrite(str(frames_dir / f"{short_id}_{tag}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness.append(float(np.mean(gray)))
        min_side = max(24, int(min(gray.shape) * 0.04))
        det = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side))
        faces.append(len(det))
    cap.release()
    return {"open_ok": True, "duration_s": dur, "fps": fps, "width": w, "height": h,
            "frames_saved": saved,
            "brightness_mean": round(sum(brightness) / len(brightness), 1) if brightness else None,
            "faces_per_frame": faces}


# ------------------------------------------------------------------- main
def _pair_score(vid: Path, vid_dur: float, vtt: Path, vtt_end: float) -> tuple[float, list[str]]:
    """Score a (video, vtt) candidate from privacy-safe signals only."""
    score, why = 0.0, []
    a, b = vid.stem.lower(), vtt.stem.lower()
    if a == b or b.startswith(a) or a.startswith(b):
        score += 100
        why.append("stem")
    sa, sb = sanitize_name(vid.stem), sanitize_name(vtt.stem)
    for field, plus, minus in (("batch", 40, -40), ("date", 30, -20), ("part", 10, -10)):
        if sa[field] is not None and sb[field] is not None:
            if sa[field] == sb[field]:
                score += plus
                why.append(field)
            else:
                score += minus
    if vid_dur > 0 and vtt_end > 0:
        diff = abs(vid_dur - vtt_end)
        if vtt_end > vid_dur + 120:  # captions can't run past the recording
            score -= 30
        elif diff < 120:
            score += 40
            why.append("dur")
        elif diff < 300:
            score += 25
            why.append("dur~")
        elif diff < 600:
            score += 10
    try:
        from rapidfuzz import fuzz
        r = fuzz.token_set_ratio(a, b)
        score += r * 0.2
        if r >= 70:
            why.append("fuzzy")
    except ImportError:
        pass
    return score, why


def pair_folder(vids: list[Path], vid_durs: dict[Path, float], vtts: list[Path],
                vtt_ends: dict[Path, float], min_score: float = 40.0) -> dict[Path, tuple[Path, str]]:
    """Greedy best-score 1:1 assignment of VTTs to videos within one folder."""
    cands = []
    for v in vids:
        for t in vtts:
            s, why = _pair_score(v, vid_durs.get(v, 0.0), t, vtt_ends.get(t, 0.0))
            cands.append((s, v, t, why))
    out, used_v, used_t = {}, set(), set()
    for s, v, t, why in sorted(cands, key=lambda c: -c[0]):
        if s < min_score or v in used_v or t in used_t:
            continue
        out[v] = (t, "+".join(why) or "weak")
        used_v.add(v)
        used_t.add(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--no-whisper-smoke", action="store_true")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    data_dir = Path(args.data_dir or os.environ.get("CHALLENGE_DATA_DIR") or r"C:\challenge-data")

    env = env_audit(whisper_smoke=not args.no_whisper_smoke)

    print("## Dataset discovery\n")
    if not data_dir.is_dir():
        print(f"- Data dir {data_dir} NOT FOUND")
        return
    work = data_dir / "_work"
    frames_dir = work / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    by_folder: dict[Path, dict[str, list[Path]]] = defaultdict(lambda: {"video": [], "vtt": []})
    other_exts = Counter()
    for root, dirs, files in os.walk(data_dir):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        for f in sorted(files):
            p = Path(root) / f
            ext = p.suffix.lower()
            if ext in VIDEO_EXTS:
                by_folder[p.parent]["video"].append(p)
            elif ext in SUB_EXTS:
                by_folder[p.parent]["vtt"].append(p)
            else:
                other_exts[ext or "<none>"] += 1

    import cv2
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    counters: Counter = Counter()
    rows, index, pair_methods, orphan_vtts = [], {}, Counter(), []
    folder_keys_seen = Counter()
    # Deterministic order: by (folder rule, relative path).
    for folder in sorted(by_folder, key=lambda p: str(p.relative_to(data_dir)).lower()):
        rel_parts = folder.relative_to(data_dir).parts
        key, label, origin, prefix = classify_folder(rel_parts)
        folder_keys_seen[key] += 1
        vids, vtts = by_folder[folder]["video"], by_folder[folder]["vtt"]
        label_txt = label + (f" ({origin})" if origin else "")

        # Video metrics first: durations feed the VTT pairing.
        per_vid = {}
        for vid in vids:
            s = sanitize_name(vid.stem)
            counters[prefix] += 1
            short_id = f"{prefix}_{counters[prefix]}" + (f"_{s['batch']}" if s["batch"] else "")
            try:
                vm = video_metrics(vid, env["ffprobe"], frames_dir, short_id, cascade)
            except Exception as e:
                vm = {"open_ok": False, "error": type(e).__name__}
            per_vid[vid] = (short_id, s, vm)
        vtt_info = {t: vtt_stats(t) for t in vtts}
        pairs = pair_folder(vids, {v: per_vid[v][2].get("duration_s") or 0.0 for v in vids},
                            vtts, {t: vtt_info[t].get("end_s", 0.0) for t in vtts})
        used = {t for t, _ in pairs.values()}

        for vid in vids:
            short_id, s, vm = per_vid[vid]
            vtt, method = pairs.get(vid, (None, "none"))
            pair_methods[method] += 1
            vs = vtt_info[vtt] if vtt else {"ok": False}
            size_mb = vid.stat().st_size / 2**20
            dur_diff = (round((vs["end_s"] - (vm.get("duration_s") or 0)) / 60, 1)
                        if vtt and vs.get("ok") else None)

            row = {"short_id": short_id, "folder": key, "label": label_txt,
                   "source": safe_label(key, s), "size_mb": round(size_mb, 1),
                   "has_vtt": bool(vtt), "pairing": method,
                   "vtt_source": safe_label(key, sanitize_name(vtt.stem)) if vtt else None,
                   "vtt_minus_video_min": dur_diff, **vm,
                   **{f"vtt_{k}": v for k, v in vs.items()}}
            rows.append(row)
            index[short_id] = {"video_path": str(vid), "vtt_path": str(vtt) if vtt else None,
                               "label": label, "sub_origin": origin or None, "folder_key": key,
                               "batch": s["batch"], "level": s["level"], "date": s["date"],
                               "part": s["part"]}
        for t in vtts:
            if t not in used:
                orphan_vtts.append(f"{safe_label(key, sanitize_name(t.stem))} "
                                   f"[{vtt_info[t].get('end_s', 0) / 60:.1f} min]")

    (work / "video_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (work / "01_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    labels = Counter(r["label"] for r in rows)
    total_h = sum(r.get("duration_s") or 0 for r in rows) / 3600
    print(f"- Data dir: {data_dir}")
    print(f"- Folders with media: {len(by_folder)}  (folder keys: {dict(folder_keys_seen)})")
    print(f"- Videos: {len(rows)}  | total duration: {total_h:.2f} h")
    print(f"- Per label: {dict(labels)}")
    print(f"- VTT pairing signals: {dict(pair_methods)}  | unpaired VTT files: {len(orphan_vtts)}")
    for o in orphan_vtts:
        print(f"    - unpaired VTT: {o}")
    print(f"- Other file types (ignored): {dict(other_exts) or 'none'}")
    print(f"- video_index.json + frames written under {work} (outside repo)\n")

    hdr = ["short_id", "folder", "mapped_label", "source (sanitised)", "dur_min", "res", "fps",
           "size_mb", "has_vtt", "pair_signals", "vtt-vid min", "vtt_cues", "vtt_words", "spk_prefix (n / %cues)",
           "bright", "faces/frame", "frames"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        ok = r.get("open_ok")
        spk = (f"{'yes' if r['vtt_speaker_prefix_count'] else 'no'} "
               f"({r['vtt_speaker_prefix_count']} / {r['vtt_pct_cues_with_prefix']}%)"
               if r.get("vtt_ok") else "—")
        lines.append("| " + " | ".join(str(x) for x in [
            r["short_id"], r["folder"], r["label"], r["source"],
            f"{r['duration_s'] / 60:.1f}" if ok else "ERR",
            f"{r['width']}x{r['height']}" if ok else "—",
            f"{r['fps']:.1f}" if ok else "—",
            r["size_mb"], r["has_vtt"], r["pairing"],
            r["vtt_minus_video_min"] if r["vtt_minus_video_min"] is not None else "—",
            r.get("vtt_cue_count", "—"), r.get("vtt_word_count", "—"), spk,
            r.get("brightness_mean", "—"),
            ",".join(map(str, r.get("faces_per_frame", []))) or "—",
            f"{r.get('frames_saved', 0)}/4",
        ]) + " |")
    table = "\n".join(lines)
    print("## Recordings (privacy-safe)\n")
    print(table)
    (work / "01_summary.md").write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
