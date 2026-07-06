"""
Core OCR pipeline using PyMuPDF4LLM.

PyMuPDF4LLM is the official PyMuPDF RAG helper:
- Pure CPU, no torch/transformers (30 MB total vs Docling's 5 GB)
- <1 sec cold start
- <1 sec per page inference
- Outputs clean markdown with tables, headings, lists
- Made by the PyMuPDF team specifically for RAG ingestion

API reference: https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/
"""
from __future__ import annotations

import io
import os
import re
import tempfile
import time
from typing import Any

from loguru import logger

from .schemas import Element, OcrResponse, PageMeta, PageResult


# --------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------- #
def ocr_bytes(data: bytes, filename: str = "upload.pdf") -> OcrResponse:
    """
    OCR a PDF or image bytes via PyMuPDF4LLM.

    For PDFs: uses pymupdf4llm.to_markdown() which preserves tables,
    headings, lists, and document structure as clean markdown.

    For images: falls back to PyMuPDF's text extraction (OCR not included —
    for scanned images, swap in `rapidocr-onnxruntime` or `easyocr`).
    """
    t0 = time.perf_counter()
    warnings: list[str] = []

    name = (filename or "").lower()
    is_pdf = name.endswith(".pdf") or not name  # default to PDF
    is_image = any(ext in name for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"))

    if is_image:
        # Image path: wrap as 1-page PDF then process
        return _ocr_image(data, filename, t0)

    # PDF path — write to temp file, run pymupdf4llm
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        logger.info("Running pymupdf4llm on {} ({} bytes)", filename, len(data))
        import pymupdf4llm
        import pymupdf as fitz

        # Get page count + dimensions first
        doc = fitz.open(tmp_path)
        total_pages = doc.page_count
        page_dims = []
        for i in range(total_pages):
            page = doc.load_page(i)
            page_dims.append((int(page.rect.width), int(page.rect.height)))
        doc.close()

        # Run PyMuPDF4LLM — returns markdown with page breaks
        # Use page_chunks=True to get per-page chunks. NOTE: each chunk is a
        # dict (keys include "text", "metadata", ...), not a plain string —
        # pull out the "text" field.
        page_chunks = pymupdf4llm.to_markdown(tmp_path, page_chunks=True)

        # Build per-page results
        pages_out: list[PageResult] = []
        for i, chunk in enumerate(page_chunks):
            if i >= total_pages:
                break
            page_md = chunk["text"] if isinstance(chunk, dict) else chunk
            w, h = page_dims[i]
            meta = PageMeta(
                page_index=i,
                width_px=w,
                height_px=h,
                dpi=72,
            )
            elements = _extract_elements(page_md, i)
            pages_out.append(
                PageResult(page=meta, markdown=page_md, elements=elements)
            )

        full_md = "\n\n---\n\n".join(p.markdown for p in pages_out)

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info("OCR done | {} pages | {} ms", len(pages_out), elapsed_ms)

    return OcrResponse(
        markdown=full_md,
        pages=pages_out,
        total_pages=len(pages_out),
        elapsed_ms=elapsed_ms,
        model="pymupdf4llm",
        warnings=warnings,
    )


# --------------------------------------------------------------------- #
# Image path — wrap as PDF then run PyMuPDF4LLM
# --------------------------------------------------------------------- #
def _ocr_image(data: bytes, filename: str, t0: float) -> OcrResponse:
    """OCR a single image by wrapping it as a 1-page PDF."""
    import pymupdf as fitz
    import pymupdf4llm
    from PIL import Image

    # Open image to get dimensions
    img = Image.open(io.BytesIO(data))
    w_px, h_px = img.size

    # Wrap as 1-page PDF
    img_pdf_bytes = fitz.open()
    img_page = img_pdf_bytes.new_page(width=w_px, height=h_px)
    img_page.insert_image(fitz.Rect(0, 0, w_px, h_px), stream=data)
    img_pdf_bytes_bytes = img_pdf_bytes.tobytes()
    img_pdf_bytes.close()

    # Write to temp and process
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(img_pdf_bytes_bytes)
        tmp_path = tmp.name

    try:
        page_chunks = pymupdf4llm.to_markdown(tmp_path, page_chunks=True)
        chunk = page_chunks[0] if page_chunks else ""
        page_md = chunk["text"] if isinstance(chunk, dict) else chunk
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    meta = PageMeta(page_index=0, width_px=w_px, height_px=h_px, dpi=72)
    elements = _extract_elements(page_md, 0)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return OcrResponse(
        markdown=page_md,
        pages=[PageResult(page=meta, markdown=page_md, elements=elements)],
        total_pages=1,
        elapsed_ms=elapsed_ms,
        model="pymupdf4llm",
        warnings=["Image was wrapped as 1-page PDF. For scanned images, text extraction may be limited — consider adding rapidocr-onnxruntime for OCR fallback."],
    )


# --------------------------------------------------------------------- #
# Element extraction from page markdown
# --------------------------------------------------------------------- #
_TABLE_RE = re.compile(
    r"((?:^\|[^\n]+\|\s*\n)(?:^\|[\s:|-]+\|\s*\n)?(?:^\|[^\n]+\|\s*\n)+)",
    re.MULTILINE,
)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_FIGURE_RE = re.compile(
    r"(?:^|\n)\s*(?:\*\*)?(?:Figure|Fig\.?|Diagram|Chart|Image|Plate)\s*[:\.\-]\s*([^\n]+)",
    re.IGNORECASE,
)


def _extract_elements(page_md: str, page_index: int) -> list[Element]:
    """Walk page markdown and return typed Element records."""
    elements: list[Element] = []
    consumed: list[tuple[int, int]] = []

    def overlaps(s, e):
        return any(not (e <= cs or s >= ce) for cs, ce in consumed)

    # Tables
    for m in _TABLE_RE.finditer(page_md):
        if overlaps(m.start(), m.end()):
            continue
        elements.append(Element(
            type="table",
            content=m.group(0).strip(),
            page_index=page_index,
        ))
        consumed.append((m.start(), m.end()))

    # Headings + figures + text (line by line)
    lines = page_md.splitlines()
    cursor = 0
    text_buf: list[str] = []

    def flush_text():
        if text_buf:
            text = "\n".join(text_buf).strip()
            if text:
                elements.append(Element(
                    type="text",
                    content=text,
                    page_index=page_index,
                ))
            text_buf.clear()

    for line in lines:
        start = cursor
        end = cursor + len(line) + 1
        if overlaps(start, end):
            cursor = end
            continue

        h = _HEADING_RE.match(line)
        fig = _FIGURE_RE.match(line)

        if h:
            flush_text()
            elements.append(Element(
                type="heading",
                content=h.group(2).strip(),
                page_index=page_index,
                extra={"level": len(h.group(1))},
            ))
            consumed.append((start, end))
        elif fig:
            flush_text()
            elements.append(Element(
                type="figure",
                content=fig.group(1).strip(),
                page_index=page_index,
            ))
            consumed.append((start, end))
        else:
            text_buf.append(line)
            consumed.append((start, end))
        cursor = end

    flush_text()
    return elements
