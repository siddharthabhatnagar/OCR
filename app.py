"""
HF Spaces entry point (Gradio SDK + ZeroGPU).

This is the file HF Spaces runs when the Space starts. The Gradio SDK
expects a `demo` variable that is a `gradio.Blocks` instance.

Layout
------
- `demo` is a gradio.Blocks with a file-upload UI for browser testing
- All FastAPI routes from `service.main` (/ocr, /health, /docs) are
  mounted onto `demo.app` so they work alongside the Gradio UI
- `if __name__ == "__main__": demo.launch(prevent_reloop=True)` is the
  canonical Gradio pattern. `prevent_reloop=True` prevents the infinite
  loop that occurs when the SDK re-executes app.py as a subprocess.

URLs (after deploy)
-------------------
  /          → Gradio UI (file upload form)
  /ocr       → FastAPI POST endpoint (multipart upload)
  /ocr/raw   → FastAPI POST endpoint (raw image bytes)
  /health    → health check (no auth)
  /docs      → Swagger UI
"""
from __future__ import annotations

import os
import sys

# Ensure the project root is on sys.path so `service.*` imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr

# Import the FastAPI app (has /ocr, /health, /docs routes)
from service.main import app as fastapi_app
from service.config import get_settings


# --------------------------------------------------------------------- #
# Gradio handler — calls the OCR pipeline directly
# --------------------------------------------------------------------- #
def ocr_file(file):
    if file is None:
        return "Please upload a PDF or image file."

    settings = get_settings()
    file_path = file.name if hasattr(file, "name") else str(file)

    with open(file_path, "rb") as f:
        payload = f.read()

    name = file_path.lower()
    try:
        if name.endswith(".pdf"):
            from service.ocr import ocr_pdf
            result = ocr_pdf(payload, settings)
        else:
            from service.ocr import ocr_image
            result = ocr_image(payload, settings)
    except Exception as e:
        return f"**OCR failed:**\n\n```\n{e}\n```"

    output = (
        f"### OCR Result\n\n"
        f"**Pages:** {result.total_pages} | "
        f"**Time:** {result.elapsed_ms / 1000:.1f}s | "
        f"**Model:** `{result.model}`\n\n"
    )
    if result.warnings:
        output += "**Warnings:** " + "; ".join(result.warnings) + "\n\n"
    output += "---\n\n"
    output += result.markdown if result.markdown else "_No text extracted._"
    return output


# --------------------------------------------------------------------- #
# Build the Gradio Blocks UI
# --------------------------------------------------------------------- #
demo = gr.Blocks(
    title="Unlimited-OCR API",
    theme=gr.themes.Soft(),
    css="footer {visibility: hidden}",
)

with demo:
    gr.Markdown(
        "# 📄 Unlimited-OCR API\n"
        "Upload a PDF or image to extract text, tables, equations, and figures.\n\n"
        "Powered by `baidu/Unlimited-OCR` (3B MoE, MIT-licensed) on HF Spaces ZeroGPU.\n\n"
        "**First request after idle takes 2-5 min** (cold start: download + load model)."
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload PDF or image",
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp"],
                file_count="single",
            )
            submit_btn = gr.Button("Run OCR", variant="primary")
            gr.Markdown(
                "---\n"
                "**REST API endpoints (for RAG integration):**\n"
                "- `POST /ocr` — multipart upload\n"
                "- `POST /ocr/raw` — raw image bytes\n"
                "- `GET /health` — health check\n"
                "- `GET /docs` — Swagger UI\n\n"
                "**Python client:**\n"
                "```python\n"
                "from client.ocr_client import UnlimitedOcrClient\n"
                'client = UnlimitedOcrClient(base_url="https://YOUR-SPACE.hf.space")\n'
                'result = client.ocr_pdf("paper.pdf")\n'
                "```"
            )
        with gr.Column(scale=2):
            output_md = gr.Markdown(label="OCR Result")

    submit_btn.click(fn=ocr_file, inputs=file_input, outputs=output_md)


# --------------------------------------------------------------------- #
# Merge FastAPI routes into the Gradio Blocks' underlying FastAPI app
# --------------------------------------------------------------------- #
# `demo.app` is the FastAPI app that Gradio uses internally. We add our
# custom routes (/ocr, /health, /ocr/raw) to it so they work alongside
# the Gradio UI when the SDK launches demo.

_existing_paths = {getattr(r, "path", None) for r in demo.app.routes}

for route in fastapi_app.routes:
    path = getattr(route, "path", None)
    if path is None:
        continue
    if path in _existing_paths:
        continue
    # Skip Gradio's internal routes
    if path.startswith("/gradio") or path in (
        "/", "/config", "/theme.css", "/favicon.ico",
        "/api", "/api/", "/api/predict", "/queue", "/queue/data",
        "/upload", "/file", "/stream",
    ):
        continue
    demo.app.routes.append(route)

# Copy our CORS middleware into Gradio's app
for middleware in fastapi_app.user_middleware:
    demo.app.user_middleware.append(middleware)

# Rebuild middleware stack to include the new middleware
demo.app.middleware_stack = None
demo.app.build_middleware_stack()


# --------------------------------------------------------------------- #
# Launch (only when run as `python app.py`, NOT when imported by SDK)
# --------------------------------------------------------------------- #
# `prevent_reloop=True` is critical on HF Spaces — the SDK may re-execute
# app.py as a subprocess, and without this flag Gradio would try to launch
# again, causing infinite recursion.
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        prevent_reloop=True,
        show_error=True,
    )
