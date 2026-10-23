"""VTT reference text + word-level WER helpers (no printing of text, ever)."""
from __future__ import annotations

import re
from pathlib import Path

from rapidfuzz.distance import Levenshtein

_TIMING = re.compile(
    r"(?:(\d{1,2}):)?(\d{2}):(\d{2}[.,]\d{3})\s*-->\s*(?:(\d{1,2}):)?(\d{2}):(\d{2}[.,]\d{3})")
_VOICE = re.compile(r"<v(?:\.[^ >]*)?\s+[^>]+>", re.I)
_PREFIX = re.compile(r"^[^:]{1,40}:\s*")
_TAG = re.compile(r"<[^>]+>")
_BRACKET = re.compile(r"\[[^\]]*\]|\([^)]*\)")
_PUNCT = re.compile(r"[^\w\s]|_")


def _secs(h, m, s) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + float(s.replace(",", "."))


def parse_vtt_cues(path: str | Path) -> list[tuple[float, float, str]]:
    """Return (start, end, spoken_text) per cue, with speaker prefixes stripped."""
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    cues, cur = [], None
    for line in text.splitlines():
        m = _TIMING.search(line)
        if m:
            cur = [_secs(*m.groups()[:3]), _secs(*m.groups()[3:]), []]
            cues.append(cur)
        elif not line.strip():
            cur = None
        elif cur is not None:
            cur[2].append(line.strip())
    out = []
    for start, end, lines in cues:
        body = " ".join(lines)
        if _VOICE.search(body):
            body = _VOICE.sub(" ", body)
        else:
            body = _PREFIX.sub("", body, count=1)
        out.append((start, end, _TAG.sub(" ", body)))
    return out


_VOICE_NAME = re.compile(r"<v(?:\.[^ >]*)?\s+([^>]+)>", re.I)
_PREFIX_NAME = re.compile(r"^\s*([A-Za-z][^:]{0,39}):\s*")


def parse_vtt_speaker_cues(path: str | Path) -> list[tuple[float, float, str | None, str]]:
    """(start, end, speaker or None, spoken_text) per cue.

    Speaker labels contain real names: keep the result in memory only — never
    print, log or persist it.
    """
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    cues, cur = [], None
    for line in text.splitlines():
        m = _TIMING.search(line)
        if m:
            cur = [_secs(*m.groups()[:3]), _secs(*m.groups()[3:]), []]
            cues.append(cur)
        elif not line.strip():
            cur = None
        elif cur is not None:
            cur[2].append(line.strip())
    out = []
    for start, end, lines in cues:
        body = " ".join(lines)
        spk = None
        vm = _VOICE_NAME.search(body)
        if vm:
            spk, body = vm.group(1).strip(), _VOICE.sub(" ", body)
        else:
            pm = _PREFIX_NAME.match(body)
            if pm and len(pm.group(1).split()) <= 4:
                spk, body = pm.group(1).strip(), body[pm.end():]
        out.append((start, end, spk, _TAG.sub(" ", body)))
    return out


def normalize_words(text: str) -> list[str]:
    """Lowercase, drop bracketed annotations and punctuation, split on whitespace."""
    text = _BRACKET.sub(" ", text.lower())
    text = _PUNCT.sub(" ", text.replace("'", "").replace("’", ""))
    return text.split()


def reference_words(cues, start_sec: float = 0.0, end_sec: float | None = None) -> list[str]:
    """Reference words from cues whose start falls inside [start_sec, end_sec)."""
    words = []
    for s, _e, t in cues:
        if s >= start_sec and (end_sec is None or s < end_sec):
            words.extend(normalize_words(t))
    return words


def non_latin_pct(text: str) -> float:
    """% of letters outside Latin script (e.g. Malayalam output) — numeric only."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return round(100 * sum(ord(c) > 0x024F for c in letters) / len(letters), 1)


def wer(ref_words: list[str], hyp_words: list[str]) -> float:
    """Word error rate (S + D + I) / N_ref via Levenshtein over word tokens."""
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return Levenshtein.distance(ref_words, hyp_words) / len(ref_words)
