"""Synthetic-name tests for the filename sanitiser (no real dataset names)."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "check01", Path(__file__).resolve().parents[1] / "scripts" / "01_env_and_data_check.py")
check01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check01)


def label(stem, key="excellent"):
    return check01.safe_label(key, check01.sanitize_name(stem))


def test_name_is_redacted():
    out = label("B901 intermediate Part 2 - 29-06-2026 Zorblat Quenterly")
    assert out == "excellent B901 Intermediate Part 2 (29/06/2026) [REDACTED_NAME]"
    assert "Zorblat" not in out and "Quenterly" not in out


def test_no_name_no_flag():
    assert label("B902 (29_06_2026)") == "excellent B902 (29/06/2026)"


def test_part_dash_date_is_not_part_number():
    assert label("B903 intermediate Part - 28-08-2026", "average") == "average B903 Intermediate (28/08/2026)"
    assert label("B903 Part-2 28-08-2026", "average") == "average B903 Part 2 (28/08/2026)"


def test_month_name_date():
    assert label("Class B904 12th July 2026 Mxyzzy", "good") == "good B904 (12/07/2026) [REDACTED_NAME]"


def test_iso_date_and_missing_batch():
    assert label("2026-06-29 recording", "demo") == "demo B? (29/06/2026)"


def test_vtt_stats_never_returns_names(tmp_path):
    p = tmp_path / "x.vtt"
    p.write_text("WEBVTT\n\n1\n00:00:01.000 --> 00:00:02.000\nZorblat Q: hello there kids\n\n"
                 "2\n00:00:03.000 --> 00:00:04.000\n<v Mxyzzy>hi miss</v>\n\n"
                 "3\n00:00:05.000 --> 00:00:06.000\nno prefix here\n", encoding="utf-8")
    s = check01.vtt_stats(p)
    assert s["cue_count"] == 3 and s["speaker_prefix_count"] == 2
    assert s["word_count"] == 3 + 2 + 3
    assert "Zorblat" not in repr(s) and "Mxyzzy" not in repr(s)
