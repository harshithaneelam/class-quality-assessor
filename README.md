# Online Class Quality Assessor

**Live Application:** <TBD — insert Streamlit Community Cloud URL>

Rates a recorded online children's class (chess lessons on a video-call platform) as
**Excellent / Good / Average / Poor** and backs the rating with **timestamped `[MM:SS]` evidence**:
strengths, areas for improvement, a compliance & pedagogy checklist and five 0–100 sub-scores.

All media processing runs locally. Only de-identified text (`[TEACHER]`, `[STUDENT_1..N]`) and
calibrated numeric verdicts are ever sent to an LLM.

---

## 1. Architecture & multimodal signal pipeline

```
video file / Google Drive link (gdown) / direct URL
  └─ ephemeral TemporaryDirectory (deleted in `finally`)
      1. ingest + fast SHA-256 file hash
      2. PyAV → 16 kHz mono WAV (streamed, no ffmpeg binary needed)
      3. visual compliance (1 frame / 10 s, streamed)        audio prosody & silence (1 s windows)
         · Zoom-aware layout: full-screen camera,             · speech ratio, dead air, longest gaps
           screen share + top-right PiP, hidden panel,          · RMS dynamics, YIN pitch variation
           camera off                                          · liveliness score
         · branded virtual background: dark-field rule
           OR 3-template logo matcher (scale / shift / mirror)
         · YuNet face detection → face-box luminance
         · background bleed / sharpness
      4. faster-whisper `small` (explicit language, beam 1, Silero VAD)
      5. 3-layer local de-identification → speaker roles → transcript rubric features
         → deterministic rubric + gates  ──►  final label + 5 sub-scores + checklist
         → Gemini v2 judge (de-identified)  ──►  executive summary + timestamped observations
```

| Module | Role |
|---|---|
| `src/audio_extract.py` | PyAV streaming decode → 16 kHz mono int16 WAV |
| `src/visual_features.py` | layout classifier, branded-background detector, YuNet faces, lighting, bleed |
| `src/audio_features.py` | dead air, silence gaps, dynamics, pitch, liveliness |
| `src/transcribe.py` | faster-whisper with SHA-256 cache; 2 CPU threads on small containers |
| `src/deidentify.py` | 3-layer name masker (context patterns, gazetteer + spaCy + rare-word rule, fuzzy clustering) |
| `src/speaker_roles.py` | TEACHER / STUDENT attribution (lexical cues + per-class 2-cluster acoustics) |
| `src/transcript_features.py` | intro, onboarding, revision-jump, 1-to-1 questions, homework, conclusion, praise |
| `src/scorer.py` | deterministic 5-dimension rubric, gates, confidence, observation merging |
| `src/llm_judge.py` | prompt v2, pre-flight privacy gate, cache, backoff, model fallback |
| `src/pipeline.py` | end-to-end inference with progress callbacks |
| `streamlit_app.py` / `app.py` | Streamlit (Cloud) and Gradio (local) front-ends sharing `src/ui_render.py` |
| `assets/` | `logo_templates.npz` (3 × 18 × 32 numeric grids, 2.3 KB) and the YuNet face model (OpenCV Zoo, MIT) |

## 2. Privacy & free-tier LLM design

- **Local-only media.** Audio extraction, transcription, computer vision and prosody run on the
  host. Video and audio are never uploaded anywhere. Each run works inside a temporary folder that
  is deleted in a `finally` block; the apps also delete their copy of an uploaded file.
- **3-layer name masker.** (1) context patterns (self-introductions, vocatives, "coach X",
  "X ma'am"); (2) Indian + international name gazetteer, spaCy `PERSON`, capitalised proper nouns and a
  rare-word rule (Zipf < 1.5 → low-confidence `[NAME]`); (3) fuzzy clustering of Whisper's spelling
  variants into stable `[TEACHER]` / `[STUDENT_k]` placeholders. On the calibration set the **blind**
  masker caught **13 / 14 (92.9 %)** of the caption speaker-name tokens that Whisper actually transcribed,
  with **0 leaks** after the secondary scrub (the one miss is a name that is also a common English word).
- **Pre-flight leak check.** Every prompt is checked before any network call: no known speaker-name
  token, no dataset path or media filename, no binary / base64 payload. A failing prompt is never sent.
- **What the LLM sees.** Because Google's free tier may use prompts for review, only the
  de-identified transcript (capped at 3,000 words ≈ 5–8k tokens) and **calibrated `PASS` / `FAIL`
  verdicts** for the visual checks are sent — never raw media, file names or real names.
- **Quota-aware calling.** Responses are cached on disk by SHA-256 of (prompt version, recording id,
  transcript hash), so re-runs and threshold tuning cost no quota. 429 / 503 → backoff 2 s, 5 s, 10 s →
  next model: `gemini-2.5-flash` → `gemini-flash-lite-latest` → `gemini-flash-latest` → offline
  deterministic fallback (`GEMINI_PRIMARY_MODEL` / `GEMINI_FALLBACK_MODEL` can override the head of the
  chain). Without a key the app still produces a full, timestamped assessment.

## 3. Calibration & evaluation

Ten labelled recordings (`exc_1`, `exc_2`, `good_1`, `good_2`, `avg_1`, `avg_2`, `poor_weak_1`,
`poor_weak_2`, `poor_qi_1`, `poor_qi_2`) plus two unlabelled demos (`demo_1`, `demo_2`).
"Within one tier" counts e.g. Good ↔ Excellent as a near miss.

| Mode | Prompt v1 exact | v1 within one tier | **Prompt v2 exact** | **v2 within one tier** |
|---|---|---|---|---|
| LLM-only zero-shot | 4/10 (40 %) | 6/10 (60 %) | 4/10 (40 %) | **9/10 (90 %)** |
| Deterministic leave-one-out | 7/10 (70 %) | 10/10 (100 %) | **8/10 (80 %)** | **10/10 (100 %)** |
| Hybrid (0.45 det + 0.55 LLM) leave-one-out | 8/10 (80 %) | 10/10 (100 %) | 7/10 (70 %) | 10/10 (100 %) |

- **Prompt v1** sent raw detector percentages; the LLM read normal detector misses as non-compliance and
  rated **every** class Poor (its 4/10 were the four Poor classes).
- **Prompt v2** sends calibrated `PASS` / `FAIL` visual verdicts and explains that Whisper often drops
  student names; within-one-tier accuracy rose from 6/10 to 9/10.
- **Deterministic v2** gained `avg_1` through the ASR name-drop refinement (≤ 2 named students but
  ≥ 60 teacher questions with the camera on is treated as missed names, not missing 1-to-1 attention).
- **Hybrid v2** dropped to 7/10: `gemini-2.5-flash` rated `exc_2` Good and the 0.55 LLM weight pulled the
  fused composite below the Excellent cutoff. During the v2 run `gemini-2.5-flash` also hit its free-tier
  daily quota after 9 calls, so three Poor classes were judged by the fallback (`gemini-3.5-flash-lite`),
  i.e. the LLM rows mix two judges.
- **Production choice:** the final rating and the five sub-scores come from the **deterministic scorer**
  (8/10 exact, 10/10 within one tier); Gemini contributes the executive summary and timestamped
  observations (bullets contradicting a `PASS` visual verdict are filtered out), and its agreement feeds
  the confidence score.
- **Caveat:** the gate rules were written after looking at all ten labels; leave-one-out only refits the
  score cutoffs, so these figures are optimistic. Ten recordings is a very small calibration set.

Production (deterministic, leave-one-out) per recording:

| Recording | True | Predicted | Main reason |
|---|---|---|---|
| exc_1 | Excellent | Excellent | — |
| exc_2 | Excellent | Excellent | — |
| good_1 | Good | **Average** | no homework or wrap-up detected (miss) |
| good_2 | Good | Good | Excellent prerequisites unmet |
| avg_1 | Average | Average | dark face lighting cap |
| avg_2 | Average | **Good** | background segmentation glitch not detected (miss) |
| poor_qi_1 | Poor | Poor | branded background missing + face hidden |
| poor_qi_2 | Poor | Poor | face not visible on camera |
| poor_weak_1 | Poor | Poor | jumped into revision without rapport |
| poor_weak_2 | Poor | Poor | no onboarding + low student coverage |

## 4. What worked, what failed & limitations

**Worked**
- **Face-box luminance** separates poor lighting cleanly: `avg_1` 25 vs 85–148 for every other class.
- **Hidden panel vs camera off:** a pure-black PiP panel beside live screen share is Zoom's hidden
  thumbnail panel, not a camera-off (the real camera-off canvas is flat grey). `good_1` camera-off fell
  from 65 % to 3 %.
- **Dead air** tracks quality: `exc_1` 75 s vs 1,721–1,956 s for `poor_weak_1` / `poor_qi_2`.
- **Logo-structure matching** (three per-tutor templates with scale / shift / mirror search) recovers the
  branded background when the logo fills most of the tile, without firing on the class with no background.

**Rejected**
- **`distil-small.en`:** 2× faster on GPU but dropped 73 % of the words (733 vs 2,740) — WER 79.7 %.
- **Automatic language detection:** Whisper `small` labelled Indian-English classes as Malayalam with
  0.98 probability, so the language is always set explicitly.
- **Fixed-baseline speaker roles:** worse than always guessing TEACHER; replaced by per-class clustering.

**Limitations**
- `avg_2`'s background segmentation glitch is not detected (a temporal flicker measure scored 0.74 vs
  0.50 / 0.48 for the two Excellent classes — not a clean separation).
- `demo_1` is code-mixed: low ASR confidence triggers a language warning and lowers confidence.
- Speaker-role attribution is ~67 % accurate against caption speaker labels.
- Whisper `small` disagrees with the platform captions by 17–46 % WER, so transcript signals are
  pattern-level, not word-exact.
- Calibrated on 10 recordings; thresholds will need re-tuning on more data.

## 5. Local usage & deployment

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
copy .env.example .env            # then set GEMINI_API_KEY (optional)
streamlit run streamlit_app.py    # Streamlit UI
python app.py                     # or the Gradio UI
pytest tests/                     # offline test suite (mocked Whisper / Gemini)
```

A CUDA GPU is used automatically when available (`int8_float16`); otherwise CPU `int8` with 2 threads.

**Streamlit Community Cloud**
1. New app → repository `harshithaneelam/class-quality-assessor`, branch `main`, main file
   `streamlit_app.py`.
2. Advanced settings → Python **3.11** (also pinned by `.python-version`).
3. Secrets → add `GEMINI_API_KEY = "..."` (optional; without it the deterministic rubric writes the evidence).
4. Optional memory switches (Secrets or environment): `WHISPER_MODEL = "base"` uses a smaller speech
   model; `WHISPER_CHUNK_SEC` sets the CPU transcription window (default 600 s; `0` = whole file).
   With the defaults a 44-min class peaks at ~1.3 GB resident (measured locally on CPU).
5. Deploy. The first run downloads the Whisper `small` model (~460 MB); expect ~10–15 min per
   45–60 min class on the free 2-core CPU, and prefer Google Drive links for large recordings
   (`.streamlit/config.toml` caps uploads at 500 MB).
