"""
PDF processing: render pages to PIL images via PyMuPDF.

The Unlimited-OCR model expects image FILES (paths), so we render to a
temp directory and pass paths to model.infer_multi().
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import fitz  # PyMuPDF
from PIL import Image
from loguru import logger

from .config import Settings


@dataclass
class RenderedPage:
    image: Image.Image
    page_index: int
    width_px: int
    height_px: int
    dpi: int


def render_pages(pdf_bytes: bytes, settings: Settings) -> list[RenderedPage]:
    """Render every page of the PDF to a PIL Image at the configured DPI."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count == 0:
        raise ValueError("PDF has no pages")

    if doc.page_count > settings.pdf_max_pages:
        logger.warning(
            "PDF has {} pages but max={} — truncating",
            doc.page_count, settings.pdf_max_pages,
        )
        page_range = range(settings.pdf_max_pages)
    else:
        page_range = range(doc.page_count)

    zoom = settings.pdf_dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    pages: list[RenderedPage] = []
    for i in page_range:
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        pages.append(
            RenderedPage(
                image=img,
                page_index=i,
                width_px=img.width,
                height_px=img.height,
                dpi=settings.pdf_dpi,
            )
        )
    doc.close()
    return pages
