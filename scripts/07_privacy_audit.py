"""Step 07 — pre-push privacy & confidentiality audit (in-memory, prints counts only).

Reference sets are built in memory from the local dataset and never written or printed:
  * speaker-name tokens from the caption files' speaker prefixes,
  * person-name tokens left in the dataset file names (after removing dates / levels / codes),
  * the recordings' batch codes,
  * the Gemini API key value (from the environment / .env).
It scans every file that would be committed (tracked + untracked, not ignored) and the full
`git log -p --all` history (diffs and commit messages); `--rev <branch>` limits the
history scan to one branch. Output: counts, commit hashes, file
paths and categories — never the matched text.

Hits on a word from the built-in public first-name / surname gazetteer (src/deidentify.py)
are reported separately as `gazetteer_overlap`: that list was not derived from the data.
Hits on tokens that are ordinary vocabulary — in the de-identifier's whitelist (languages,
places, chess and common words) or a frequent English word (Zipf >= 4.5) matched only in
lowercase — are reported as `common_word_overlap` (e.g. a language name in a file name).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.deidentify import GAZETTEER, WHITELIST  # noqa: E402
from src.wer import parse_vtt_speaker_cues  # noqa: E402

sys.excepthook = lambda t, e, tb: print(f"[FATAL] {t.__name__} (details suppressed)", file=sys.stderr)


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.excepthook = lambda t, e, tb: print(f"[FATAL] {t.__name__} (details suppressed)", file=sys.stderr)
    return mod


MEDIA_EXT = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".wav", ".mp3", ".m4a", ".flac", ".vtt", ".srt"}


def reference_sets() -> tuple[set[str], set[str], set[str]]:
    step01 = _load("s01", "01_env_and_data_check.py")
    step04 = _load("s04", "04_deid_and_transcript_features.py")
    work = Path(os.environ.get("CHALLENGE_DATA_DIR") or r"C:\challenge-data") / "_work"
    index = json.loads((work / "video_index.json").read_text(encoding="utf-8"))
    names, codes = set(), set()
    for e in index.values():
        if e.get("batch"):
            codes.add(e["batch"].upper())
        if e.get("vtt_path"):
            toks, _, _ = step04.speaker_tokens(parse_vtt_speaker_cues(e["vtt_path"]))
            names |= toks
        for p in (e.get("video_path"), e.get("vtt_path")):
            if not p:
                continue
            stem = Path(p).stem
            for m in re.finditer(r"\bB[-_ ]?(\d{2,5})\b", stem, re.I):
                codes.add(f"B{m.group(1)}")
            _, rest = step01.extract_date(stem)
            for t in re.findall(r"[A-Za-z]+", rest):
                lt = t.lower()
                if (len(lt) >= 3 and lt not in step01.SAFE_WORDS and not re.fullmatch(r"b\d*", lt)
                        and lt not in step04.GENERIC_SPEAKER_TOKENS):
                    names.add(lt)
    key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    try:
        from dotenv import dotenv_values
        key = key or (dotenv_values(REPO / ".env").get("GEMINI_API_KEY") or "").strip()
    except ImportError:
        pass
    secrets = {key} if key and key != "your_gemini_api_key_here" else set()
    return names, codes, secrets


def scanner(names: set[str], codes: set[str], secrets: set[str]):
    name_rx = re.compile(r"(?<![A-Za-z])(" + "|".join(sorted(map(re.escape, names), key=len, reverse=True)) + r")(?![A-Za-z])",
                         re.I) if names else None
    code_rx = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(sorted(map(re.escape, codes))) + r")(?![0-9])") if codes else None
    key_rx = re.compile(r"AIza[0-9A-Za-z_\-]{30,}")

    try:
        from wordfreq import zipf_frequency
    except ImportError:
        zipf_frequency = None

    def category(match: str) -> str:
        t = match.lower()
        if t in GAZETTEER:
            return "gazetteer_overlap"
        if t in WHITELIST or (zipf_frequency and zipf_frequency(t, "en") >= 4.5 and match.islower()):
            return "common_word_overlap"
        return "name"

    def scan(text: str) -> Counter:
        c = Counter()
        if name_rx:
            for m in name_rx.finditer(text):
                c[category(m.group(1))] += 1
        if code_rx:
            c["batch_code"] += len(code_rx.findall(text))
        c["secret"] += sum(text.count(s) for s in secrets) + len(key_rx.findall(text))
        return c
    return scan


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                          errors="replace").stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", default="--all", help="history to scan (default: all refs); e.g. main")
    args = ap.parse_args()
    names, codes, secrets = reference_sets()
    scan = scanner(names, codes, secrets)

    # ---- files that would be committed
    files = [f for f in git("ls-files", "--cached", "--others", "--exclude-standard").splitlines() if f]
    tree = Counter()
    per_file: dict[str, Counter] = {}
    media = [f for f in files if Path(f).suffix.lower() in MEDIA_EXT or f.lower() == ".env" or f.endswith("secrets.toml")]
    for f in files:
        p = REPO / f
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:  # binary asset: only secrets apply
            text = p.read_bytes().decode("latin-1")
            c = Counter(secret=sum(text.count(s) for s in secrets))
        else:
            c = scan(text)
        if any(c.values()):
            per_file[f] = c
        tree += c
    public = ("README.md", "app.py", "streamlit_app.py")
    public_codes = sum(per_file.get(f, Counter())["batch_code"] for f in public)
    public_names = sum(per_file.get(f, Counter())["name"] for f in public)

    # ---- full history (messages + diffs)
    has_commits = bool(git("rev-parse", "--verify", "--quiet", "HEAD").strip()) or args.rev == "--all"
    log = git("log", "-p", args.rev, "--format=@@COMMIT %H%n%B@@END_MSG") if has_commits else ""
    commits = re.split(r"(?m)^@@COMMIT ", log)[1:]
    hist = Counter()
    hist_where: list[tuple[str, str, str, int]] = []
    for blob in commits:
        sha, rest = blob.split("\n", 1)
        msg, _, diff = rest.partition("@@END_MSG")
        for cat, n in scan(msg).items():
            if n:
                hist[cat] += n
                hist_where.append((sha[:7], "commit message", cat, n))
        for chunk in re.split(r"(?m)^diff --git ", diff)[1:]:
            path = chunk.split("\n", 1)[0].split(" b/")[-1]
            added = "\n".join(l for l in chunk.splitlines() if l.startswith("+") and not l.startswith("+++"))
            for cat, n in scan(added).items():
                if n:
                    hist[cat] += n
                    hist_where.append((sha[:7], f"diff:{path}", cat, n))

    passed = (tree["name"] == 0 and tree["secret"] == 0 and tree["batch_code"] == 0 and not media
              and hist["name"] == 0 and hist["secret"] == 0 and hist["batch_code"] == 0)
    print(f"PRIVACY_AUDIT_PASSED: {passed} (tracked_files={len(files)}, commits_checked={len(commits)}, "
          f"name_leaks={tree['name'] + hist['name']}, secret_leaks={tree['secret'] + hist['secret']}, "
          f"readme_batch_code_leaks={per_file.get('README.md', Counter())['batch_code']})")
    print(f"  current tree: batch_codes={tree['batch_code']} names={tree['name']} secrets={tree['secret']} "
          f"media_or_secret_files={len(media)} public_files(README/app/streamlit): codes={public_codes} names={public_names} "
          f"gazetteer_overlap={tree['gazetteer_overlap']} common_word_overlap={tree['common_word_overlap']}")
    for f, c in sorted(per_file.items()):
        if c["name"] or c["batch_code"] or c["secret"]:
            print(f"    file {f}: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items()) if v))
    print(f"  history: batch_codes={hist['batch_code']} names={hist['name']} secrets={hist['secret']} "
          f"gazetteer_overlap={hist['gazetteer_overlap']} common_word_overlap={hist['common_word_overlap']} (reference sets: {len(names)} name tokens, {len(codes)} codes)")
    for sha, where, cat, n in hist_where:
        if cat in ("name", "batch_code", "secret"):
            print(f"    commit {sha} {where}: {cat}={n}")


if __name__ == "__main__":
    main()
