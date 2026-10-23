"""Online Class Quality Assessor — Streamlit app (Streamlit Community Cloud entry point).

Run locally:  streamlit run streamlit_app.py
Secrets:      GEMINI_API_KEY in .streamlit/secrets.toml (Cloud: App settings → Secrets)
              or in the environment / .env. Without a key the app uses the
              deterministic rubric for both the rating and the evidence.
"""
from __future__ import annotations

import gc
import os
import re
import tempfile
from pathlib import Path

import streamlit as st

from src.ui_render import (CSS, TITLE, render_checklist, render_evidence, render_hero, render_scores,
                           telemetry_payload)

RUNTIME_NOTE = (
    "**Runtime:** a 45–60 minute class takes about **10–15 minutes** on Streamlit Community Cloud's free "
    "2-core CPU (about **2 minutes** on a local GPU). For large recordings, paste a **Google Drive sharing link** "
    "instead of uploading, to avoid browser upload limits.")
PRIVACY_NOTE = (
    "**Privacy:** all media processing — audio extraction, speech transcription, face / background computer "
    "vision and prosody analysis — runs locally in an ephemeral temporary folder that is deleted in a "
    "`finally` block. Person names are masked locally (`[TEACHER]`, `[STUDENT_1..N]`) before any text is sent "
    "to Gemini; video and audio are never uploaded anywhere.")
LANGUAGES = {"English": "en", "Malayalam": "ml"}
UPLOAD_TYPES = ["mp4", "mkv", "mov", "webm", "avi"]
CHUNK = 8 * 2**20


def configure_gemini_key() -> bool:
    """Copy GEMINI_API_KEY from Streamlit secrets (if any) into the environment."""
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:  # no secrets file locally
        key = None
    key = key or os.environ.get("GEMINI_API_KEY")
    if key:
        os.environ["GEMINI_API_KEY"] = key
    return bool(key)


def save_upload(uploaded, dest_dir: Path) -> Path:
    """Stream an UploadedFile to disk in chunks, then drop the in-memory buffer."""
    suffix = Path(uploaded.name).suffix.lower()
    if suffix.lstrip(".") not in UPLOAD_TYPES:
        raise ValueError("Unsupported file type.")
    dest = dest_dir / f"upload{suffix}"
    uploaded.seek(0)
    with dest.open("wb") as f:
        while chunk := uploaded.read(CHUNK):
            f.write(chunk)
    try:
        uploaded.close()
    except Exception:
        pass
    gc.collect()
    return dest


def is_supported_link(url: str) -> bool:
    return bool(re.match(r"(?i)^https?://", (url or "").strip()))


def render_results(result: dict) -> None:
    a = result["assessment"]
    st.html(f"<style>{CSS}</style>" + render_hero(result))
    st.html(f"<style>{CSS}</style>" + render_scores(a))
    st.html(f"<style>{CSS}</style>" + render_checklist(result))
    st.subheader("Timestamped observations")
    st.html(f"<style>{CSS}</style>" + render_evidence(a))
    with st.expander("Multimodal Telemetry & De-Identification Summary"):
        st.json(telemetry_payload(result), expanded=False)


def run(source, language: str, progress_bar, status) -> dict:
    from src.pipeline import assess_video_pipeline  # heavy imports only when needed

    def cb(frac: float, desc: str) -> None:
        progress_bar.progress(min(1.0, max(0.0, frac)), text=f"{desc} · {frac * 100:.0f}%")
        status.write(desc)

    return assess_video_pipeline(source, language, progress_cb=cb)


def main() -> None:
    st.set_page_config(page_title=TITLE, page_icon="🎓", layout="wide")
    st.title(TITLE)
    st.caption("Rates a recorded online children's class as Excellent / Good / Average / Poor with timestamped evidence.")
    st.info(RUNTIME_NOTE + "\n\n" + PRIVACY_NOTE)
    has_key = configure_gemini_key()
    if not has_key:
        st.warning("No GEMINI_API_KEY configured — the deterministic rubric will produce the rating and the evidence.")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader("Class video", type=UPLOAD_TYPES)
        url = st.text_input("…or a Google Drive / direct video link", placeholder="https://drive.google.com/file/d/…/view")
    with col2:
        lang_name = st.selectbox("Class language", list(LANGUAGES), index=0)
        st.caption("Language is set explicitly: automatic detection on the `small` Whisper model mislabelled "
                   "Indian-English classes as Malayalam.")
        go = st.button("Assess Class Quality", type="primary", use_container_width=True)

    if not go:
        return
    if not uploaded and not is_supported_link(url):
        st.error("Upload a class video or paste a Google Drive / direct video link.")
        return

    progress_bar = st.progress(0.0, text="Starting")
    status = st.empty()
    tmp = tempfile.TemporaryDirectory(prefix="class_upload_", ignore_cleanup_errors=True)
    try:
        source = url.strip() if is_supported_link(url) else save_upload(uploaded, Path(tmp.name))
        result = run(source, LANGUAGES[lang_name], progress_bar, status)
    except ValueError as e:
        st.error(str(e))
        return
    except Exception as e:  # never surface paths or media details
        st.error(f"Assessment failed ({type(e).__name__}). Please try another file or link.")
        return
    finally:
        tmp.cleanup()
        gc.collect()
    status.empty()
    progress_bar.progress(1.0, text="Done")
    render_results(result)


if __name__ == "__main__":
    main()
