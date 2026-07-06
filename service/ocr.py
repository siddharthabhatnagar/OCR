"""
High-level OCR pipeline:
  PDF bytes -> render pages -> chunk -> model.infer_multi() -> split <PAGE>
  -> per-page elements -> OcrResponse
"""
from __future__ import annotations

import os
import tempfile
import time
from io import BytesIO

from PIL import Image
from loguru import logger

from .config import Settings
from .model import DEFAULT_PROMPT, get_model_manager, split_pages
from .pdf import render_pages
from .postprocess import extract_elements
from .schemas import OcrResponse, PageMeta, PageResult


def ocr_pdf(pdf_bytes: bytes, settings: Settings) -> OcrResponse:
    """Top-level entrypoint used by the FastAPI route and Gradio UI."""
    t0 = time.perf_counter()
    warnings: list[str] = []
    mm = get_model_manager()

    pages = render_pages(pdf_bytes, settings)
    total_pages = len(pages)
    logger.info("Rendered {} pages @ {} DPI", total_pages, settings.pdf_dpi)

    if total_pages > settings.pdf_max_pages:
        warnings.append(
            f"Truncated from {total_pages} to {settings.pdf_max_pages} pages."
        )
        pages = pages[: settings.pdf_max_pages]
        total_pages = len(pages)

    # Materialize page images to disk (model.infer_multi wants paths)
    with tempfile.TemporaryDirectory(prefix="unlimited_ocr_") as tmp_dir:
        image_paths: list[str] = []
        for p in pages:
            out_path = os.path.join(tmp_dir, f"page_{p.page_index:04d}.png")
            p.image.save(out_path, format="PNG")
            image_paths.append(out_path)

        # Chunk pages and OCR each chunk via model.infer_multi()
        chunk_size = settings.chunk_size
        per_page_mds: list[str] = [""] * total_pages

        for chunk_start in range(0, total_pages, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_pages)
            chunk_paths = image_paths[chunk_start:chunk_end]
            n_in_chunk = len(chunk_paths)
            logger.info(
                "OCR chunk pages {}..{} ({} images)",
                chunk_start, chunk_end - 1, n_in_chunk,
            )
            raw = mm.infer_multi(chunk_paths, prompt=DEFAULT_PROMPT)
            chunk_pages = split_pages(raw, n_in_chunk)
            for i, md in enumerate(chunk_pages):
                per_page_mds[chunk_start + i] = md

    # Build typed page results
    page_results: list[PageResult] = []
    for p, md in zip(pages, per_page_mds):
        meta = PageMeta(
            page_index=p.page_index,
            width_px=p.width_px,
            height_px=p.height_px,
            dpi=p.dpi,
        )
        elements = extract_elements(md, p.page_index, meta)
        page_results.append(
            PageResult(page=meta, markdown=md, elements=elements)
        )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    full_md = "\n\n---\n\n".join(pr.markdown for pr in page_results)

    return OcrResponse(
        markdown=full_md,
        pages=page_results,
        total_pages=total_pages,
        elapsed_ms=elapsed_ms,
        model=settings.model_id,
        warnings=warnings,
    )


def ocr_image(image_bytes: bytes, settings: Settings) -> OcrResponse:
    """OCR a single image (PNG/JPEG). Wraps it as a 1-page document."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")

    with tempfile.NamedTemporaryFile(prefix="unlimited_ocr_", suffix=".png", delete=False) as tmp:
        img.save(tmp.name, format="PNG")
        tmp_path = tmp.name

    try:
        mm = get_model_manager()
        raw = mm.infer_multi([tmp_path], prompt=DEFAULT_PROMPT)
        pages_md = split_pages(raw, 1)
        md = pages_md[0] if pages_md else ""
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    meta = PageMeta(page_index=0, width_px=img.width, height_px=img.height, dpi=72)
    elements = extract_elements(md, 0, meta)
    return OcrResponse(
        markdown=md,
        pages=[PageResult(page=meta, markdown=md, elements=elements)],
        total_pages=1,
        elapsed_ms=0,
        model=settings.model_id,
    )
