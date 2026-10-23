"""Step 06 — Gemini judge + hybrid scorer: calibration and honest evaluation.

Modes
  --test-one ID   single Gemini call on one recording; prints model + token counts only
  (default)       all 12 recordings (cached LLM calls), LLM-only / deterministic LOO /
                  hybrid LOO evaluation on the 10 labelled classes, demos as held-out
  --no-llm        deterministic only (no API use)

PRIVACY: prompts are built only from step-04 masked transcripts (full de-identification
incl. the rare-word [NAME] rule — verified per file) plus numeric features; every
prompt passes a pre-flight check against the VTT speaker-name tokens (kept in memory
only). The console shows labels, scores, gate codes and timestamp counts only; LLM
evidence text goes to <data>/_work/06_calibration_report.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import llm_judge as lj  # noqa: E402
from src import scorer as sc  # noqa: E402
from src.wer import parse_vtt_speaker_cues  # noqa: E402


def _safe_excepthook(exc_type, exc, tb):
    print(f"[FATAL] {exc_type.__name__} (details suppressed for privacy)", file=sys.stderr)


sys.excepthook = _safe_excepthook

_spec = importlib.util.spec_from_file_location("step04", REPO_ROOT / "scripts" / "04_deid_and_transcript_features.py")
step04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step04)
sys.excepthook = _safe_excepthook  # step04 import re-installs its own hook; keep ours


def log(msg: str) -> None:
    print(msg, flush=True)


def table(rows, cols) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join("—" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def prompt_features(inp: dict, cam: dict) -> dict:
    """v2 feature block (verdicts for passing visual checks; numbers only)."""
    return lj.build_prompt_features(inp["visual"], inp["audio"], inp["transcript"], cam,
                                    inp["asr"]["avg_logprob_mean"], sc.G["code_mix_logprob"])


def load_inputs(work: Path, index: dict) -> dict:
    f04 = json.loads((work / "04_transcript_features.json").read_text(encoding="utf-8"))
    f05 = json.loads((work / "05_visual_audio_features.json").read_text(encoding="utf-8"))
    out = {}
    for sid in index:
        m = json.loads((work / "masked_transcripts" / f"{sid}.json").read_text(encoding="utf-8"))
        if "low_confidence_name_masks" not in m.get("deid_stats", {}):
            raise SystemExit(f"{sid}: masked transcript predates the rare-word [NAME] rule — rerun step 04")
        segs = m["segments"]
        out[sid] = {"duration_sec": m["duration_sec"], "segments": segs,
                    "visual": f05[sid]["visual"], "audio": f05[sid]["audio"],
                    "transcript": f04[sid]["features"],
                    "asr": {"avg_logprob_mean": round(float(np.mean([s["avg_logprob"] for s in segs])), 3)}}
    return out


def vtt_name_tokens(entry: dict) -> set[str]:
    if not entry.get("vtt_path"):
        return set()
    toks, _, _ = step04.speaker_tokens(parse_vtt_speaker_cues(entry["vtt_path"]))
    return toks  # in memory only


def run_llm(sid, inp, entry, llm_cache, use_llm=True):
    cam = sc.refine_camera_off(inp["visual"].get("camera_off_times_sec", []), inp["visual"].get("sample_every_sec", 10.0),
                               inp["segments"], inp["visual"].get("sampled_frames", 0))
    inp["camera_refined"] = cam
    prompt, block, info = lj.build_assessment_prompt(sid, inp["segments"], inp["duration_sec"], prompt_features(inp, cam))
    lj.preflight_privacy_check(prompt, vtt_name_tokens(entry))  # raises before any send
    res = lj.call_gemini_judge(sid, prompt, block, vtt_name_tokens(entry), cache_dir=llm_cache) if use_llm else None
    return res, info, len(prompt)


def tier_dist(a, b):
    return abs(sc.LABELS.index(a) - sc.LABELS.index(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-one")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    work = Path(os.environ.get("CHALLENGE_DATA_DIR") or r"C:\challenge-data") / "_work"
    llm_cache = work / "llm_cache"
    index = json.loads((work / "video_index.json").read_text(encoding="utf-8"))
    inputs = load_inputs(work, index)

    if not args.no_llm:
        key, status = lj.api_key_status()
        if key is None:
            log(f"!! {status}")
            args.no_llm = True

    if args.test_one:
        sid = args.test_one
        res, info, nchar = run_llm(sid, inputs[sid], index[sid], llm_cache, use_llm=not args.no_llm)
        log(f"## Test call — {sid}")
        log(f"- prompt: {nchar} chars, transcript {info['words_sent']}/{info['words_total']} words "
            f"({'sampled' if info['sampled'] else 'full'}), pre-flight privacy check: PASSED")
        if res is None:
            log("- LLM: no result (fallback_deterministic)")
            return
        u = res.get("usage") or {}
        log(f"- model: {res['model']} (requested {res.get('model_requested')}) | cache_hit: {res['cache_hit']}")
        log(f"- tokens: prompt {u.get('prompt_token_count')} | output {u.get('candidates_token_count')} | "
            f"thinking {u.get('thoughts_token_count')} | total {u.get('total_token_count')}")
        a = res["assessment"]
        log(f"- llm label: {a['llm_raw_label']} ({a['poor_sub_category']}) | scores {a['pedagogical_scores']} | "
            f"observations: {len(a['strengths'])} strengths, {len(a['improvements'])} improvements")
        if res.get("errors_before_success"):
            log(f"- errors before success: {res['errors_before_success']}")
        return

    # ---- all recordings
    llm = {}
    log(f"## Step 06 — {'deterministic only' if args.no_llm else 'Gemini judge + hybrid scorer'} ({len(index)} recordings)")
    for sid in index:
        res, info, _ = run_llm(sid, inputs[sid], index[sid], llm_cache, use_llm=not args.no_llm)
        llm[sid] = res
        u = (res or {}).get("usage") or {}
        if res is None and not args.no_llm:
            log(f"  ! {sid}: Gemini unavailable, errors: {lj.LAST_ERRORS[-4:]}")
        log(f"- {sid}: llm {'—' if res is None else res['llm_status']} "
            f"{'' if res is None else '(' + str(res['model']) + ')'} | prompt tokens {u.get('prompt_token_count', '—')} "
            f"| words sent {info['words_sent']}")

    labelled = [s for s in index if index[s]["label"] in sc.LABELS]
    demos = [s for s in index if s not in labelled]
    truth = {s: index[s]["label"] for s in labelled}

    # Production: label from the deterministic scorer, Gemini for summary + evidence text.
    # Reference only: the 0.45 det / 0.55 LLM fused label.
    gates = {s: sc.evaluate_gates(sc.gate_inputs(inputs[s])) for s in index}

    def loo(source):
        preds, folds = {}, {}
        for s in labelled:
            train = [{"composite": sc.assess_class_recording(inputs[o], llm[o], label_source=source)["composite"],
                      "gates": gates[o], "truth": truth[o]} for o in labelled if o != s]
            cut = sc.fit_cutoffs(train)
            folds[s] = cut
            preds[s] = sc.assess_class_recording(inputs[s], llm[s], cut, label_source=source)
        return preds, folds

    def fit_all(source):
        return sc.fit_cutoffs([{"composite": sc.assess_class_recording(inputs[s], llm[s], label_source=source)["composite"],
                                "gates": gates[s], "truth": truth[s]} for s in labelled])

    final_loo, final_folds = loo("deterministic")
    fused_loo, fused_folds = loo("fused") if any(llm.values()) else (final_loo, final_folds)

    def acc(pred_labels):
        exact = sum(pred_labels[s] == truth[s] for s in labelled)
        within = sum(tier_dist(pred_labels[s], truth[s]) <= 1 for s in labelled)
        return f"{exact}/{len(labelled)} ({100 * exact / len(labelled):.0f}%)", f"{within}/{len(labelled)} ({100 * within / len(labelled):.0f}%)"

    rows_modes = []
    if any(llm.values()):
        llm_lab = {s: (llm[s] or {}).get("assessment", {}).get("llm_raw_label") or "—" for s in labelled}
        ex = sum(llm_lab[s] == truth[s] for s in labelled)
        wi = sum(llm_lab[s] in sc.LABELS and tier_dist(llm_lab[s], truth[s]) <= 1 for s in labelled)
        rows_modes.append({"mode": "1 LLM-only zero-shot", "exact": f"{ex}/{len(labelled)} ({100 * ex / len(labelled):.0f}%)",
                           "within_1_tier": f"{wi}/{len(labelled)} ({100 * wi / len(labelled):.0f}%)"})
    e, w = acc({s: final_loo[s]["label"] for s in labelled})
    rows_modes.append({"mode": "2 Deterministic LOO = production label (Gemini adds text only)", "exact": e, "within_1_tier": w})
    if any(llm.values()):
        e, w = acc({s: fused_loo[s]["label"] for s in labelled})
        rows_modes.append({"mode": "3 Fused 0.45/0.55 LOO (reference, not used)", "exact": e, "within_1_tier": w})
    log("\n## Evaluation on the 10 labelled classes\n")
    log(table(rows_modes, ["mode", "exact", "within_1_tier"]))
    log("\nNote: gate rules were designed after inspecting all 10 labels; LOO refits only the score cutoffs, "
        "so these numbers are optimistic.")

    det_cut, fused_cut = fit_all("deterministic"), fit_all("fused")
    final = {s: (final_loo[s] if s in labelled else sc.assess_class_recording(inputs[s], llm[s], det_cut)) for s in index}
    rows = []
    for s in labelled + demos:
        h = final[s]
        fz = fused_loo[s] if s in labelled else sc.assess_class_recording(inputs[s], llm[s], fused_cut, label_source="fused")
        obs = h["strengths"] + h["improvements"]
        secs = [int(m.group(1)) * 60 + int(m.group(2)) for o in obs
                for m in [re.match(r"\[(\d+):(\d{2})\]", o["timestamp"] or "")] if m]
        rows.append({"short_id": s, "true_label": truth.get(s, "(demo)"), "final_label": h["label"],
                     "llm_label": h["llm_raw_label"], "fused_label": fz["label"], "conf%": h["confidence_pct"],
                     "vis": h["sub_scores"]["visual_compliance"], "intro": h["sub_scores"]["intro_onboarding"],
                     "indiv": h["sub_scores"]["individual_attention"], "energy": h["sub_scores"]["energy_rapport"],
                     "hw_struct": h["sub_scores"]["structure_homework"],
                     "gates": ",".join(c for c in h["triggered_gates"] if not c.startswith("E0")) or "—",
                     "obs": len(obs), "llm_obs": sum(o.get("source") == "llm" for o in obs),
                     "earliest_ts": sc._ts(min(secs)) if secs else None,
                     "latest_ts": sc._ts(max(secs)) if secs else None,
                     "lang_warn": bool(h["language_warning"])})
    log("\n## Per-recording breakdown (LOO for labelled, all-10 cutoffs for demos)\n")
    log(table(rows, list(rows[0])))
    log(f"\nFinal cutoffs (fit on all 10): production/deterministic {det_cut} | fused reference {fused_cut}")
    calib = {"label_source": "deterministic", "deterministic_cutoffs": list(det_cut),
             "fused_cutoffs_reference": list(fused_cut),
             "prompt_version": lj.PROMPT_VERSION, "fit_on": f"{len(labelled)} labelled classes",
             "note": "Composite cutoffs Poor|Average|Good|Excellent on the deterministic composite; gates in "
                     "src/scorer.py override. Gemini supplies summary + evidence text only."}
    (REPO_ROOT / "src" / "assets" / "calibration.json").write_text(json.dumps(calib, indent=1), encoding="utf-8")

    report = {"label_source": "deterministic", "final_cutoffs": det_cut, "fused_cutoffs_reference": fused_cut,
              "modes": rows_modes, "rows": rows, "det_loo_folds": final_folds, "fused_loo_folds": fused_folds,
              "assessments": final,
              "llm": {s: ({k: v for k, v in r.items()} if r else None) for s, r in llm.items()}}
    (work / "06_calibration_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str),
                                                     encoding="utf-8")
    log(f"Saved -> {work / '06_calibration_report.json'}")


if __name__ == "__main__":
    main()
