"""Online Class Quality Assessor — Gradio UI (Gradio 6.x).

Run locally:  .venv\\Scripts\\python app.py
"""
from __future__ import annotations

import os

import gradio as gr

from src.pipeline import assess_video_pipeline
from src.ui_render import (CSS, TITLE, render_checklist, render_evidence, render_hero,  # noqa: F401
                            render_scores, telemetry_payload, checklist_rows)

RUNTIME_NOTE = ("Runtime: about 1.5–2.5 min on a GPU, or 8–14 min on a free CPU Space, "
                "for a 45–60 minute class.")
PRIVACY_NOTE = ("Privacy: video and audio are processed locally and temporary files are deleted as soon "
                "as the assessment finishes. Names are masked as [TEACHER], [STUDENT_1..N] or [NAME] "
                "before any text is sent to the Gemini judge; no video or audio is ever uploaded.")

# ================================================================ handler
def run_assessment(video_file, url, language, progress=gr.Progress()):
    source = (url or "").strip() or video_file
    if not source:
        raise gr.Error("Upload a class video or paste a video / Google Drive link.")
    try:
        result = assess_video_pipeline(source, language or "en",
                                       progress_cb=lambda f, d: progress(f, desc=d))
    except ValueError as e:
        raise gr.Error(str(e))
    finally:
        # Gradio keeps its own copy of uploads; delete it as soon as we're done.
        if video_file and not (url or "").strip():
            try:
                os.remove(video_file)
            except OSError:
                pass
    a = result["assessment"]
    return render_hero(result), render_scores(a), render_checklist(result), render_evidence(a), telemetry_payload(result)


def build_app() -> gr.Blocks:
    with gr.Blocks(title=TITLE, delete_cache=(600, 3600)) as demo:
        gr.Markdown(f"# {TITLE}\nRates a recorded online children's class as **Excellent / Good / Average / Poor** "
                    f"with timestamped evidence.\n\n{RUNTIME_NOTE}\n\n{PRIVACY_NOTE}")
        with gr.Row():
            with gr.Column(scale=2):
                video = gr.File(label="Class video", file_types=[".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"],
                                type="filepath")
                url = gr.Textbox(label="…or a video / Google Drive file link",
                                 placeholder="https://drive.google.com/file/d/…/view")
            with gr.Column(scale=1):
                language = gr.Dropdown(label="Class language", choices=[("English (Default)", "en"), ("Malayalam", "ml")],
                                       value="en")
                go = gr.Button("Assess Class Quality", variant="primary")
        hero = gr.HTML()
        scores = gr.HTML()
        checklist = gr.HTML()
        evidence = gr.HTML()
        with gr.Accordion("Detailed multimodal telemetry & de-identification stats", open=False):
            telemetry = gr.JSON()
        go.click(run_assessment, inputs=[video, url, language], outputs=[hero, scores, checklist, evidence, telemetry])
    return demo


if __name__ == "__main__":
    build_app().queue(default_concurrency_limit=1).launch(css=CSS, theme=gr.themes.Soft())
