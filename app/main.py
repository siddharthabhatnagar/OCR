"""FastAPI service for PaddleOCR table extraction."""
import os
import time
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .ocr import extract_text, extract_tables, decode_image
from .table_parser import parse_html_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/bmp",
    "image/tiff",
}

# CORS — allow your Vercel frontend to call this API.
# Set CORS_ORIGINS env var on Render to restrict (comma-separated).
# Default "*" allows any origin (fine for prototyping).
_cors = os.environ.get("CORS_ORIGINS", "*")
_cors_list = [o.strip() for o in _cors.split(",")] if _cors != "*" else ["*"]

app = FastAPI(
    title="PaddleOCR Table API",
    description="Table OCR service using PaddleOCR PP-StructureV2",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    service: str
    models_loaded: bool
    language: str


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — Render uses this for uptime monitoring."""
    from .ocr import _table_instance, LANG
    return HealthResponse(
        status="ok",
        service="paddle-ocr-table",
        models_loaded=_table_instance is not None,
        language=LANG,
    )


@app.get("/")
async def root():
    return {
        "service": "PaddleOCR Table API",
        "version": "1.0.0",
        "endpoints": {
            "POST /ocr": "Plain text OCR — returns text + bounding boxes",
            "POST /table": "Full table OCR — returns HTML + structure per table",
            "POST /api/extract-table": "Vercel-compatible drop-in endpoint",
            "GET /health": "Health check",
            "GET /docs": "Interactive Swagger UI",
        },
    }


async def _read_and_validate(file: UploadFile) -> bytes:
    if not file.content_type or file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            400,
            f"Unsupported file type: {file.content_type}. "
            f"Allowed: {', '.join(sorted(ALLOWED_TYPES))}",
        )
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            413,
            f"File too large ({len(content)} bytes). Max: {MAX_FILE_SIZE} bytes.",
        )
    if not content:
        raise HTTPException(400, "Empty file")
    return content


@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    """Plain text OCR. Returns list of text items with confidence + bbox."""
    content = await _read_and_validate(file)
    img = decode_image(content)
    start = time.time()
    items = extract_text(img)
    elapsed = int((time.time() - start) * 1000)
    return {
        "items": items,
        "count": len(items),
        "elapsed_ms": elapsed,
    }


@app.post("/table")
async def table_endpoint(file: UploadFile = File(...)):
    """Full table OCR. Returns all tables found in the image."""
    content = await _read_and_validate(file)
    img = decode_image(content)
    start = time.time()
    tables = extract_tables(img)
    elapsed = int((time.time() - start) * 1000)
    return {
        "tables": tables,
        "count": len(tables),
        "elapsed_ms": elapsed,
    }


@app.post("/api/extract-table")
async def extract_table_compat(file: UploadFile = File(...)):
    """
    Drop-in replacement for the Vercel Next.js frontend's /api/extract-table.

    Returns the same response shape:
        { rows, cols, cells, html, elapsed_ms, table_bbox? }

    Point your Vercel frontend's fetch() at this URL and it just works.
    """
    content = await _read_and_validate(file)
    img = decode_image(content)
    start = time.time()
    tables = extract_tables(img)
    elapsed = int((time.time() - start) * 1000)

    if not tables:
        raise HTTPException(
            422,
            "No table detected in the image. Try a clearer image with a "
            "visible table structure (rows and columns).",
        )

    # Use the first (usually largest) table found
    table = tables[0]
    parsed = parse_html_table(table["html"])

    table_bbox = None
    bbox = table.get("bbox")
    if bbox and len(bbox) >= 4:
        try:
            table_bbox = {
                "xmin": float(bbox[0]),
                "ymin": float(bbox[1]),
                "xmax": float(bbox[2]),
                "ymax": float(bbox[3]),
            }
        except (TypeError, ValueError):
            table_bbox = None

    return {
        "rows": parsed["rows"],
        "cols": parsed["cols"],
        "cells": parsed["cells"],
        "html": parsed["html"],
        "elapsed_ms": elapsed,
        "table_bbox": table_bbox,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        log_level="info",
    )
