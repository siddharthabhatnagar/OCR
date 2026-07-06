"""
Core OCR pipeline using Docling.

Docling is IBM's document AI library:
- Pure CPU (no GPU needed)
- Excellent table extraction (TableFormer model)
- Outputs clean markdown with structure preserved
- ~5 sec cold start, ~2-5 sec per page inference

API reference: https://github.com/DS4SD/docling
"""
from __future__ import annotations

import io
import time
from typing import Any

from loguru import logger

from .schemas import BoundingBox, Element, OcrResponse, PageMeta, PageResult


# Module-level singleton — Docling's DocumentConverter is expensive to
# construct (loads ~200MB of models). We build it once and reuse.
_converter: Any = None
_load_error: str | None = None


def get_converter():
    """Lazily build the Docling DocumentConverter. Idempotent."""
    global _converter, _load_error
    if _converter is not None:
        return _converter
    try:
        from docling.document_converter import DocumentConverter

        logger.info("Building Docling DocumentConverter (loads models, ~5 sec)")
        _converter = DocumentConverter()
        _load_error = None
        logger.info("Docling converter ready")
        return _converter
    except Exception as e:
        _load_error = repr(e)
        logger.exception("Failed to build Docling converter")
        raise


def is_loaded() -> bool:
    return _converter is not None


def get_load_error() -> str | None:
    return _load_error


# --------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------- #
def ocr_bytes(data: bytes, filename: str = "upload.pdf") -> OcrResponse:
    """
    OCR a PDF or image bytes via Docling.

    Docling's DocumentConverter handles both PDFs and images directly,
    so we just write to a temp file and pass it in.
    """
    t0 = time.perf_counter()
    warnings: list[str] = []

    converter = get_converter()

    import tempfile
    import os

    suffix = ".pdf" if filename.lower().endswith(".pdf") else "." + filename.rsplit(".", 1)[-1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        logger.info("Running Docling on {} ({} bytes)", filename, len(data))
        result = converter.convert(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    doc = result.document

    # Build full markdown (Docling's native export)
    full_md = doc.export_to_markdown()

    # Build per-page results. Docling exposes pages via `doc.pages`.
    pages_out: list[PageResult] = []

    # Group elements by page index. Each Docling element has `.page_no` (1-indexed).
    page_indices = sorted(doc.pages.keys()) if doc.pages else [1]

    for page_no in page_indices:
        page_info = doc.pages.get(page_no)
        width_px = int(page_info.size.width) if page_info and page_info.size else 0
        height_px = int(page_info.size.height) if page_info and page_info.size else 0

        meta = PageMeta(
            page_index=page_no - 1,  # 0-indexed for consistency
            width_px=width_px,
            height_px=height_px,
            dpi=72,  # Docling doesn't expose DPI; PDF default is 72
        )

        # Extract elements on this page
        page_elements: list[Element] = []
        for item in doc.iterate_items():
            # item.page_no is 1-indexed in Docling
            if item.page_no != page_no:
                continue
            el = _docling_item_to_element(item, page_no - 1)
            if el is not None:
                page_elements.append(el)

        # Page-level markdown: extract the section for this page
        # Docling doesn't natively paginate markdown, so we approximate by
        # joining the text content of items on this page.
        page_md = "\n\n".join(
            el.content for el in page_elements if el.type in {"text", "heading", "table", "equation"}
        )

        pages_out.append(
            PageResult(page=meta, markdown=page_md, elements=page_elements)
        )

    # If no pages were extracted (unusual), fall back to a single page
    if not pages_out:
        pages_out.append(
            PageResult(
                page=PageMeta(page_index=0, width_px=0, height_px=0, dpi=72),
                markdown=full_md,
                elements=[],
            )
        )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info("OCR done | {} pages | {} ms", len(pages_out), elapsed_ms)

    return OcrResponse(
        markdown=full_md,
        pages=pages_out,
        total_pages=len(pages_out),
        elapsed_ms=elapsed_ms,
        model="docling",
        warnings=warnings,
    )


# --------------------------------------------------------------------- #
# Map Docling items to our Element type
# --------------------------------------------------------------------- #
def _docling_item_to_element(item, page_index: int) -> Element | None:
    """Convert a Docling item to our Element schema."""
    try:
        from docling.document import (
            DocItemLabel,
            TextItem,
            TableItem,
            PictureItem,
            FormulaItem,
            ListItem,
            SectionHeaderItem,
            TitleItem,
            ParagraphItem,
        )
    except ImportError:
        # Fallback: just use the text representation
        text = getattr(item, "text", None) or str(item)
        return Element(type="text", content=text, page_index=page_index)

    # Section headers / titles
    if isinstance(item, (SectionHeaderItem, TitleItem)):
        return Element(
            type="heading",
            content=getattr(item, "text", "") or "",
            page_index=page_index,
            extra={"label": str(getattr(item, "label", "heading"))},
        )

    # Tables
    if isinstance(item, TableItem):
        # Docling exposes table as HTML/markdown via .export_to_markdown() or .data
        try:
            tbl_md = item.export_to_markdown()
        except Exception:
            tbl_md = getattr(item, "text", "") or ""
        return Element(type="table", content=tbl_md, page_index=page_index)

    # Pictures / figures
    if isinstance(item, PictureItem):
        # Docling doesn't OCR images by default; we return a placeholder caption
        caption = ""
        for cap in item.captions:
            caption += getattr(cap, "text", "") + " "
        return Element(
            type="figure",
            content=caption.strip() or "[figure]",
            page_index=page_index,
        )

    # Formulas / equations
    if isinstance(item, FormulaItem):
        text = getattr(item, "text", "") or ""
        # If Docling exported LaTeX, wrap it
        if text and not text.startswith("$$"):
            text = f"$${text}$$"
        return Element(type="equation", content=text, page_index=page_index)

    # Lists
    if isinstance(item, ListItem):
        return Element(
            type="list",
            content=getattr(item, "text", "") or "",
            page_index=page_index,
        )

    # Generic text / paragraphs
    if isinstance(item, (TextItem, ParagraphItem)):
        text = getattr(item, "text", "") or ""
        if not text.strip():
            return None
        return Element(type="text", content=text, page_index=page_index)

    # Catch-all: try to extract text
    text = getattr(item, "text", None)
    if text:
        return Element(type="text", content=text, page_index=page_index)

    return None
