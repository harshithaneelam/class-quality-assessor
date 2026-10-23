# Online Class Quality Assessor

## Project overview
24-hour build challenge. Given a recorded online children's class (chess lessons on
a video-call platform with a branded virtual background), classify its quality as
**Poor / Average / Good / Excellent** and justify the rating with **timestamped
evidence** (e.g. `12:34 — tutor checks understanding with [STUDENT_2]`).

**Confidentiality:** the GitHub repo is public. Never commit the client's company
name, reviewer feedback text, batch codes, or any person name; refer to recordings
only by short IDs (`exc_1`, `good_2`, `poor_qi_1`, `demo_1`, ...).

## Privacy rules (NON-NEGOTIABLE)
The dataset contains recordings of children and tutors.

1. The raw dataset lives at `C:\challenge-data\` — **outside** this repo. It is
   never copied, committed, or symlinked into the repo.
2. Claude / any AI assistant must **never** read raw recordings, `.vtt` files,
   or unmasked transcripts into its context, and never send them to any
   external AI API. `.claude/settings.json` denies Read/Edit on
   `C:\challenge-data\**`; do not work around it with Bash (`cat`, `head`,
   `type`, `Get-Content`, `grep`, `ls`/`dir` of the dataset, etc.).
3. Dataset access happens only through local Python scripts in `scripts/`
   that print **aggregate, privacy-safe metrics** only.
4. Scripts must never print raw transcript text, VTT cue lines, or speaker
   names. Any filename from the dataset printed to stdout/stderr must be
   sanitised to folder label + batch code + level + date, with any person name
   replaced by `[REDACTED_NAME]`. Prefer the stable short IDs
   (`exc_1`, `poor_qi_1`, `demo_1`, ...).
5. All derived artefacts — cached transcripts, extracted frames, audio, the
   short-ID → path map (`video_index.json`) — live in
   `C:\challenge-data\_work\`, outside the repo.
6. Only **de-identified** text (`[TEACHER]`, `[STUDENT_1]`, ...) may be sent
   to Gemini.

## Architecture
```
local ingestion (C:\challenge-data, short IDs)
  -> local signal extraction
       faster-whisper (GPU, RTX 3050) transcript + timestamps
       OpenCV  (webcam tile, faces, brightness, screen activity)
       librosa (talk-time, silence, turn-taking, energy)
  -> local 3-layer de-identification
       1. VTT speaker prefixes -> role tokens
       2. spaCy NER on PERSON entities
       3. rapidfuzz match against known-name roster
       => [TEACHER], [STUDENT_1], [STUDENT_2], ...
  -> Gemini free tier (gemini-2.5-flash, fallback flash-lite)
       structured rubric output (pydantic schema)
       SHA-256 cache of masked input -> response
       deterministic heuristic fallback if the API is unavailable
  -> Gradio UI (rating + rubric breakdown + timestamped evidence)
```

## Layout
- `src/` — library code (`src/utils/cuda_setup.py` registers CUDA DLL dirs on
  Windows; import it before `ctranslate2` / `faster_whisper`).
- `scripts/` — numbered pipeline steps (`01_env_and_data_check.py`, ...).
- `tests/` — tests (synthetic data only, never real transcripts).

## Environment
- Windows 11, Python venv at `.venv`, `pip install -r requirements.txt`.
- Config via `.env` (see `.env.example`); `.env` is gitignored.
